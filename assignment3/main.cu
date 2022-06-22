#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "common.h"

#include "cmd/processCommandLine.cuh"
#include "utils/utils.cuh"

/**
 *   program configuration
 */
#ifndef SECTOR_SIZE
# define SECTOR_SIZE  512
#endif
#ifndef N_SECTORS
# define N_SECTORS    (1 << 21)                            // it can go as high as (1 << 21)
#endif

/**
 * @brief Host processing logic, row by row.
 * 
 * @param order order of the matrices
 * @param amount amount of matrices
 * @param matrixArray array with matrices
 * @param results array to store matrices determinants
 */
void hostRR(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Device processing logic, row by row.
 * 
 * @param d_matrixArray pointer to matrices' array.
 * @param amount amount of matrices
 * @param order order of matrices
 * @param results pointer to store matrices determinants
 * @return __global__ 
 */
__global__ void deviceRR(double **d_matrixArray, int amount, int order, double *results);
// void deviceRR(int order, int amount, double **matrixArray, double *results);

/**
 * @brief Main logic of the program. 
 * Makes gaussian elimination on the host and the device.
 * Compares thre obtained results at the end. 
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
        double h_results[amount];
        double d_results[amount];
        double start, hrr, drr;

        // allocate memory on device
        double *d_matrixArray;
        CHECK(cudaMalloc((void **)&d_matrixArray, (sizeof(double) * order * order * amount)));

        // copy data to device memory
        CHECK(cudaMemcpy(d_matrixArray, h_matrixArray, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // // create grid and block
        dim3 grid(order, 1, 1); 
        dim3 block(amount, 1, 1);

        // // device processing
        start = seconds();
        // deviceRR<<<grid, block>>>(&d_matrixArray, amount, order, d_results);
        drr = seconds() - start;
        printf("Device processing took <%.5f> seconds.\n", drr);

        // wait for kernel to finish
        CHECK (cudaDeviceSynchronize ());
        
        // check for kernel errors
        CHECK (cudaGetLastError ());

        // free device memory
        CHECK(cudaFree (d_matrixArray));

        // // host processing
        start = seconds();
        hostRR(order, amount, &h_matrixArray, h_results);
        hrr = seconds() - start;
        printf("Host processing took <%.5f> seconds.\n", hrr);
    }

    return 0;
}

/**
 * @brief Calculates determinant row by row
 * 
 * @param matrix pointer to matrix
 * @param order order of matrix
 * @return int determinant of matrix
 */
void hostRR(int order, int amount, double **matrixArray, double *results)
{
    for(int i = 0; i < amount; i++)
    {
        *(results + i) = row_by_row_determinant(order, ((*matrixArray) + (i * order * order)));
        // printf("%+5.3e\n", *(results + i));
    }
}


__global__ void deviceRR(double **d_matrixArray, int amount, int order, double *results)
{
    int matrixIdx = blockIdx.x * blockDim.x * blockDim.x;
    int rowIdx = 0;
    double det = 0;
    double pivotElement = 0.0;

    /*
    rest of logic here
    */

   for(int i = 0; i < order; i++)
   {
        pivotElement = *(*d_matrixArray + (blockIdx.x * blockDim.x * blockDim.x) + (threadIdx.x * threadIdx.x));
   }
   *(results + matrixIdx) = det; 
}

// /**
//  * @brief Calculates determinant column by column
//  * 
//  * @param matrix pointer to matrix
//  * @param order order of matrix
//  * @return int determinant of matrix
//  */
// void deviceRR(int order, int amount, double **matrixArray, double *results)
// {
//     // allocate device memory

//     // copy matrices to device memory

//     // process matrices on device

//     // return results from device
// }
