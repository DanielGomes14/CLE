#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "common.h"

#include "cmd/processCommandLine.cuh"
#include "utils/utils.cuh"


/**
 * @brief Host processing logic, column by column.
 *
 * @param order order of the matrices
 * @param amount amount of matrices
 * @param matrixArray array with matrices
 * @param results array to store matrices determinants
 */
void hostCC(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Device processing logic, column by column.
 *
 * @param d_matrixArray pointer to matrices' array.
 * @param amount amount of matrices
 * @param order order of matrices
 * @param results pointer to store matrices determinants
 * @return __global__
 */
__global__ void deviceCC(double *d_matrixArray, double *d_results);
// void deviceCC(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Main logic of the program.
 * Makes gaussian elimination on the host and the device.
 * Compares the obtained results at the end.
 *
 * @param argc amount of arguments in the command line
 * @param argv array with the arguments from the command line
 * @return int return execution status of operation
 */
int main(int argc, char **argv)
{
    // process command line information to obtain file names
    int fileAmount = 0;
    char **fileNames;
    if (processInput(argc, argv, &fileAmount, &fileNames))
    {
        perror("error processing input");
        exit(EXIT_FAILURE);
    }

    // device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // process files
    double *h_matrixArray = NULL;
    int order = 0, amount = 0;
    for (int i = 0; i < fileAmount; i++)
    {
        // read data from file
        readData(*(fileNames + i), &h_matrixArray, &order, &amount);

        // structure to save results
        double *retrieved_results = (double *)malloc(sizeof(double) * amount);

        // allocate memory on device
        double *d_matrixArray;
        double *d_results;
        CHECK(cudaMalloc((void **)&d_matrixArray, (sizeof(double) * order * order * amount)));
        CHECK(cudaMalloc((void **)&d_results, sizeof(double) * amount));

        // copy data to device memory
        CHECK(cudaMemcpy(d_matrixArray, h_matrixArray, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // create grid and block
        dim3 grid(amount, 1, 1);
        dim3 block(order, 1, 1);

        // DEVICE PROCESSING
        double d_start = seconds();

        deviceCC<<<grid, block>>>(d_matrixArray, d_results);

        CHECK(cudaDeviceSynchronize());
        double drr = seconds() - d_start;
        printf("Device processing took <%.5f> seconds.\n", drr);

        CHECK(cudaGetLastError());                                                                        // check kernel errors
        CHECK(cudaMemcpy(retrieved_results, d_results, sizeof(double) * amount, cudaMemcpyDeviceToHost)); // return obtained results
        CHECK(cudaFree(d_matrixArray));                                                                   // free device memory

        // HOST PROCESSING
        double h_results[amount];
        double start = seconds();
        hostCC(order, amount, &h_matrixArray, h_results);
        double hrr = seconds() - start;
        printf("Host processing took <%.5f> seconds.\n", hrr);

        printf("\nRESULTS\n");
        for (int i = 0; i < amount; i++)
        {
            printf("HOST: <%+5.3e>\t DEVICE: <%+5.3e>\n", h_results[i], retrieved_results[i]);
        }
    }

    return 0;
}

/**
 * @brief Calculates determinant column by column on the host
 *
 * @param matrix pointer to matrix
 * @param order order of matrix
 * @return int determinant of matrix
 */
void hostCC(int order, int amount, double **matrixArray, double *results)
{
    for (int i = 0; i < amount; i++)
    {
        *(results + i) = column_by_column_determinant(order, ((*matrixArray) + (i * order * order)));
        // printf("%+5.3e\n", *(results + i));
    }
}

/**
 * @brief Calculates the determinant column by column on the GPU
 *
 * @param d_matrixArray the array with all matrices
 * @param d_results the array to store the results
 * @return __global__
 */
__global__ void deviceCC(double *d_matrixArray, double *d_results)
{
    
    int order = blockDim.x; /** The order of the Matrix is the size of the block since a block is a matrix **/ 
    int matrixIdx = blockIdx.x * order * order; /** jump to the current matrix**/
    int tColumn = threadIdx.x + matrixIdx; /** The column for which each thread is responsible**/
    int pivotColumn; /**  Auxiliar variable to select the index column of the pivot **/
   
    for (int currElem = 0; currElem < order; currElem++)
    {

        if (threadIdx.x < currElem)
            return;

        int iterColumn = currElem + matrixIdx; /** jump to the column of the matrix of the current iteration **/

        double pivot = d_matrixArray[iterColumn + currElem * order];  /** The Pivot will be initially the diagonal element on the iterColumn**/
        pivotColumn = iterColumn; 

        // Only one thread should do the Partial Pivoting, and update the determinant value of the Matrix
        if (threadIdx.x == currElem)
        {
            // iterate through the remaining columns of the same row
            for (int col = iterColumn + 1; col < ( matrixIdx + order); ++col)
            {   

                // if there's a bigger value on the row than the current pivot, the choosen pivot will be updated
                if (fabs(d_matrixArray[(currElem * order) + col]) > fabs(pivot))
                {

                    // update the value of the pivot and pivot col index
                    pivot = d_matrixArray[(currElem * order) + col];
                    pivotColumn = col;
                   
                }
            }
            
            if (currElem == 0)
                // initialize the results index of the matrix on the first iteration 
                d_results[blockIdx.x] = 1;

            // if the elected pivot is different from the initial one, than we perform a swap on the columns
            if (pivotColumn != iterColumn)
            {
               
                for (int k = 0; k < order; k++)
                {
                    double temp;
                    temp = d_matrixArray[(k * order) + iterColumn];
                    d_matrixArray[(k * order) + iterColumn] = d_matrixArray[(k * order) + pivotColumn];
                    d_matrixArray[(k * order) + pivotColumn] = temp;
                }
                d_results[blockIdx.x] *= -1.0; // signal the row swapping
            }

            d_results[blockIdx.x] *= pivot;
            return;
            //continue;
        }
        // syncronize all threads in the current block https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
        __syncthreads();
        iterColumn = currElem + matrixIdx;
        pivot = d_matrixArray[iterColumn + currElem * order];
        pivotColumn = iterColumn;

        // perform the reduction of the base matrix
        double const_val = d_matrixArray[tColumn + order * currElem] / pivot;
        for (int row = currElem + 1; row < order; row++)
        {

            d_matrixArray[tColumn + order * row] -= d_matrixArray[pivotColumn + order * row] * const_val;
        }
        __syncthreads();
    }
}

