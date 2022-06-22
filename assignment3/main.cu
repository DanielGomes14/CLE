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
 * @param order order of the matrixes
 * @param amount amount of matrixes
 * @param matrixArray array with matrixes
 * @param results array to store matrixes determinants
 */
void hostRR(int order, int amount, double **matrixArray, double *results);

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
    double *matrixArray = NULL;
    int order = 0, amount = 0;
    for(int i = 0; i < fileAmount; i++)
    {
        // read data from file
        readData(*(fileNames + i), &matrixArray, &order, &amount);

        // structure to save results
        double h_results[amount];
        double d_results[amount];
        double start, hrr, drr;

        // copy data to device memory
        // allocate memory on device
        double *d_matrixArray;
        CHECK(cudaMalloc((void **)&d_matrixArray, (sizeof(double) * order * order * amount)));


        // // host processing
        start = seconds();
        hostRR(order, amount, &matrixArray, h_results);
        hrr = seconds() - start;
        printf("Host processing took <%.3f> seconds.\n", hrr);

        
        // // copy memory to device
        // CHECK(cudaMemcpy(d_matrixArray, matrixArray, (sizeof(double) * order * order * amount), cudaMemcpyHostToDevice));

        // // create grid and block
        // dim3 grid(order, 1, 1); 
        // dim3 block(amount, 1, 1);

        // // device processing
        // start = seconds();
        // column_by_column_determinant_gpu<<<grid, block>>>();
        // drr = seconds() - start;

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
