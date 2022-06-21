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

    // process filesS
    double **matrixArray;
    int order = 0, amount = 0;
    for(int i = 0; i < fileAmount; i++)
    {
        // read data from file
        readData(*(fileNames + i), &matrixArray, &order, &amount);

        // structure to save results
        int hostRRResults[amount], hostCCResults[amount];
        int deviceRRResults[amount], deviceCCResults[amount];
        double start, hrr, hcc, drr, dcc;

        // host processing
        start = seconds();
        hostRR(order, amount, &matrixArray, hostRRResults);
        hrr = seconds() - start;

        start = seconds();
        hostCC(order, amount, &matrixArray, hostCCResults);
        hcc = seconds() - start;


        // device processing
        for(int j = 0; j < amount; j++)
        {
            double *ptr;
            CHECK(cudaMalloc((void **)&ptr, (sizeof(double) * order * order)));
            CHECK(cudaMemcpy(ptr, (*(matrixArray + j)), (sizeof(double) * order * order), cudaMemcpyHostToDevice));
        }
        // allocate memory on device
        
        // copy memory to device
        // find right dimensions to grid and block
        // create grid and block
        // save results


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
void hostRR(int order, int amount, double ***matrixArray, int *results)
{

}

/**
 * @brief Calculates determinant column by column
 * 
 * @param matrix pointer to matrix
 * @param order order of matrix
 * @return int determinant of matrix
 */
void hostCC(int order, int amount, double ***matrixArray, int *results)
{

}
