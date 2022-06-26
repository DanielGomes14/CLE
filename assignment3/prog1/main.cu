#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "common.h"

#include "cmd/processCommandLine.cuh"
#include "utils/utils.cuh"

/**
 * @brief Host processing logic, row by row.
 */
void hostRR(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Device processing logic, row by row.
 */
__global__ void deviceRR(double *d_matrixArray, double *d_results);

/**
 * @brief Main logic of the program. 
 * Makes gaussian elimination on the host and the device.
 * 
 * ROW BY ROW version.
 * 
 * @param argc amount of arguments in the command line
 * @param argv array with the arguments from the command line
 * @return int return execution status of operation
 */
int main(int argc, char **argv)
{
    // process command line information to obtain file names
    int fileAmount = 0;
    char ** fileNames;
    if(processInput(argc, argv, &fileAmount, &fileNames))
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
    
    for(int i = 0; i < fileAmount; i++)
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
        deviceRR<<<grid, block>>>(d_matrixArray, d_results);
        CHECK (cudaDeviceSynchronize ());
        double drr = seconds() - d_start;

        CHECK(cudaGetLastError ());         // check kernel errors
        CHECK(cudaMemcpy(retrieved_results, d_results, sizeof(double) * amount, cudaMemcpyDeviceToHost));   // return obtained results
        CHECK(cudaFree (d_matrixArray));    // free device memory

        // HOST PROCESSING
        double h_results[amount];
        double start = seconds();
        hostRR(order, amount, &h_matrixArray, h_results);
        double hrr = seconds() - start;

        printf("\nRESULTS\n");
        for(int i = 0; i < amount; i++)
        {
            printf("MATRIX: <%d>\tHOST: <%+5.3e>\t DEVICE: <%+5.3e>\n", i + 1, h_results[i], retrieved_results[i]);
        }

        printf("\nEXECUTION TIMES\n");
        printf("Host processing took <%.5f> seconds.\n", hrr);
        printf("Device processing took <%.5f> seconds.\n", drr);
    }

    return 0;
}

/**
 * @brief Calculates determinant row by row.
 * 
 * @param matrix pointer to matrix
 * @param order order of matrix
 */
void hostRR(int order, int amount, double **matrixArray, double *results)
{
    for(int i = 0; i < amount; i++)
        *(results + i) = row_by_row_determinant(order, ((*matrixArray) + (i * order * order)));
}

/**
 * @brief Device kernel to calculate gaussian elimination, row by row.
 * 
 * @param d_matrixArray pointer to array of matrices
 * @param d_results pointer to array of results 
 */
__global__ void deviceRR(double *d_matrixArray, double *d_results)
{
    int order = blockDim.x;
    int matrixIdx = blockIdx.x * order * order;
    int tRow = threadIdx.x * order + matrixIdx;
    int pivotRow;

    for(int currElem = 0; currElem < order; currElem++)
    {
        if(threadIdx.x < currElem)
            return;

        int iterRow = matrixIdx + currElem * order;
        double pivot = d_matrixArray[iterRow + currElem];
        pivotRow = iterRow;

        // only one thread is able to do partial pivoting and update the determinant value of the matrix
        if(threadIdx.x == currElem)
        {
            // iterate through the remaining rows of the same column
            for(int row = iterRow + order; row < (matrixIdx + (order * order)); row+=order)
            {
                // if there's a bigger value on the column than the current pivot, choosen pivot will be updated
                if(fabs(d_matrixArray[row + currElem]) > fabs(pivot))
                {
                    // update the value of the pivot and pivot row index
                    pivot = d_matrixArray[row + currElem];
                    pivotRow = row;
                }
	
	
            }

            // initialize the results undex of the matrix on the first iteration
            if(currElem == 0)
                d_results[blockIdx.x] = 1;

            // if the elected pivot is different from the initial one, than we perform a swap on the rows
            if(pivotRow != iterRow)
            {
                for(int k = 0; k < order; k++)
                {
                    double temp;
                    temp = d_matrixArray[iterRow + k];
                    d_matrixArray[(iterRow) + k] = d_matrixArray[pivotRow + k];
                    d_matrixArray[(pivotRow) + k] = temp;
                }
                d_results[blockIdx.x] *= -1.0;
            }

            d_results[blockIdx.x] *= pivot;
            return;
            // continue;
        }

        // synchronize all threads in the current block
        __syncthreads();

        iterRow = matrixIdx + currElem * order;
        pivot = d_matrixArray[(iterRow) + currElem];
        pivotRow = iterRow;

        // perform the reduction of the base matrix
        double constVal = d_matrixArray[(tRow) + currElem] / pivot;

        for(int col = currElem + 1; col < order; col++)
            d_matrixArray[(tRow) + col] -= d_matrixArray[(pivotRow) + col] * constVal;
    
        __syncthreads();
    }
}
