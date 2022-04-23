#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include "./cmd/processCommandLine.h"
#include "./shared/shared.h"
#include "./utils/utils.h"

/** \brief consumer threads return status array */
int *statusWorkers;
/** \brief  Array containing the results where workers will store **/ 
double **results;
/** \brief  Array containing the amount of matrices **/ 
int *matrixAmount;
/** \brief flag used to check if there are chunks still being produced */
int stillProcessing;
/** \brief worker life cycle routine */
static void *work(void *id);
/** \brief function used to produce chunks, and then store them in shared region**/
void produceChunks(char ***fileNames, int fileAmount);

void printResults(int fileAmount);

/**
 * @brief Main function
 *
 * Process command line information.
 * Produce chunks and save them in shared region.
 * Create worker threads to process chunks in shared region.
 * Gather and consolidate gathered info of each chunk.
 * Prints results.
 *
 * @param argc number of words from the command line
 * @param argv array with the words from the command line
 * @return int (status of operation)
 */
int main(int argc, char *argv[])
{
    struct timespec start, finish; /* time limits */
    int threadAmount = 0;          // number of threads;
    int fileAmount = 0;            // amount of files
    int *status_p;                 // pointer to execution status
    char **fileNames;              // array with the names of the files

    stillProcessing = 1;

    // no inputs where given
    if (argc == 1)
    {
        perror("No arguments were provided.");
        exit(-1);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &start); /* begin of measurement */

    // processes command line information
    if (processInput(argc, argv, &threadAmount, &fileAmount, &fileNames))
        exit(-1);

    pthread_t tIdWorker[threadAmount];  //  workers internal thread id array
    unsigned int workers[threadAmount]; // workers application defined thread id array

    // initialise workers array
    for (int t_ind = 0; t_ind < threadAmount; t_ind++)
    {
        workers[t_ind] = t_ind;
    }

    statusWorkers = malloc(sizeof(int) * threadAmount);
    for (int t = 0; t < threadAmount; t++)
    {
        if (pthread_create(&tIdWorker[t], NULL, work, &workers[t]) != 0) /* thread consumer */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
    }
    // start producing chunks
    produceChunks(&fileNames, fileAmount);
    // signal workers that are waiting for more chunks while the Shared Region's FIFO is empty,
    // since there are no more chunks
    awakeWorkers();
    for (int t = 0; t < threadAmount; t++)
    {
        if (pthread_join(tIdWorker[t], (void *)&status_p) != 0) /* thread consumer */
        {
            perror("error on waiting for thread worker");
            exit(EXIT_FAILURE);
        }

        printf("thread worker, with id %d, has terminated: ", t);
        printf("its status was %d\n", *status_p);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */
    printResults(fileAmount);
    printf("\nElapsed tim = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

    return 0;
}

/**
 * @brief function used to produce chunks, and then store them in shared region 
 * @param fileNames an array containing the names of the files
 * @param fileAmount the amount of files used to produce the chunks
 */
void produceChunks(char ***fileNames, int fileAmount)
{

    char **file_names = (*fileNames);
    int amount, order = 0;
    int matrixId;
    results = malloc(fileAmount * sizeof(double *));
    matrixAmount = malloc(fileAmount * sizeof(int));
    for (int i = 0; i < fileAmount; i++)
    {
        char *file_name = file_names[i];
        matrixId = 0;
        FILE *f = fopen(file_name, "r");
        if (f == NULL)
        {
            printf("Could not open file\n");
            exit(-1);
        }
        amount = 0;
        // reads amount of matrices
        if (!fread(&amount, sizeof(int), 1, f))
        {
            printf("Error reading amount. Exiting...\n");
            exit(-1);
        }

        results[i] = malloc(amount * sizeof(double));
        matrixAmount[i] = amount;
        // reads order of matrices
        order = 0;
        if (!fread(&order, sizeof(int), 1, f))
        {
            printf("Error reading order. Exiting...");
            exit(-1);
        }
        // starting reading matrix data and store it on the chunk Struct
        for (int j = 0; j < matrixAmount[i]; j++)
        {
            double *matrix = (double *)malloc(sizeof(double) * order * order);
            if (!fread(matrix, 8, order * order, f))
                break;
            chunkInfo chunk;
            chunk.matrixPtr = matrix;
            chunk.fileId = i;
            chunk.isLastChunk = 0;
            chunk.order = order;
            chunk.matrixId = matrixId;
            matrixId++;
            storeChunk(chunk);
        }

        fclose(f);
    }
}

/**
 * @brief Worker Function
 *
 * Does the worker tasks
 * @param par pointer to application defined worker identification
 */
static void *work(void *par)
{

    unsigned int id = *((unsigned int *)par); // worker id //
    double determinant_result;
    chunkInfo chunk;
    while (1)
    {

        // Gets The chunk from the shared region

        chunk = getChunk(id);

        // no more chunks to be processed, so the thread may end
        if (chunk.isLastChunk == 1)
        {
            statusWorkers[id] = EXIT_SUCCESS;
            pthread_exit(&statusWorkers[id]);
        }
        //calculate the determinant
        determinant_result = determinant(chunk.order, chunk.matrixPtr);
        // stores in the results data structure the result, using the fileId and the matrixId
        storePartialResults(id, chunk.fileId, chunk.matrixId, determinant_result);
    }

    // end work
    statusWorkers[id] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[id]);
}

/**
 * @brief Method invoked by Main to Print the final Results gathered from all worker threads
 * 
 * @param fileAmount the amount of files used to create and process chunks
 * @param matrixAmount the amount of matrices
 */
void printResults(int fileAmount)
{
    for (int i = 0; i < fileAmount; i++)
    {
        printf("File nº: <%d>\n", i + 1);
        for (int j = 0; j < matrixAmount[i]; j++)
        {
            printf(" Matrix nº: <%d>. The determinant is %+5.3e \t\n", j + 1, results[i][j]);
        }
    }
    free(results);
    free(matrixAmount);
}
