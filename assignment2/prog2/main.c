#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "./cmd/processCommandLine.h"
#include "worker.h"
#include "dispatcher.h"

/* General definitions */
#define WORKTODO 1
#define NOMOREWORK 0

/** \brief number of workers */
int nWorkers;
/** \brief  Array containing the amount of matrices **/
int *matrixAmount;
/** \brief dispatcher life cycle routine */
void dispatcher(char ***fileNames, int fileAmount);
/** \brief worker life cycle routine */
void work(int rank);

int main(int argc, char *argv[])
{
    struct timespec start, finish; // time limits
    int fileAmount = 0;            // amount of files
    char **fileNames;              // array of pointers, each pointer points to a string literal present in CMD

    int rank, // process rank
        size; // amout of processes
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    nWorkers = size - 1;
    if (nWorkers <1){
        printf("Number of Workers Invalid, should be greater than one.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill EVERY living process (root included)
    }
    if (rank == 0) // root process (dispatcher)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start); // start counting time

        // process command line input
        if (processInput(argc, argv, &fileAmount, &fileNames) == EXIT_FAILURE)
        {
            free(fileNames);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill EVERY living process (root included)
        }
        else
            dispatcher(&fileNames, fileAmount); // dispatcher logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish); // end counting time

        // calculate execution time
        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nRoot elapsed time = %.6f s\n", executionTime); // print execution time
    }
    else
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start); // start counting time

        work(rank); // worker logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish); // end counting time

        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nWorker <%d> elapsed time = %.6f s\n", rank, executionTime); // print execution time
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}

void dispatcher(char ***fileNames, int fileAmount)
{
    int workerId;
    unsigned int whatToDo = WORKTODO;
    ; /* command */
    double **results = malloc(fileAmount * sizeof(double *));
    matrixAmount = malloc(fileAmount * sizeof(int));
    char **file_names = (*fileNames);
    int amount = 0, order = 0, lastChunkSize = 0;
    int matrixId = 0, fileId = 0, chunkId = 0;
    int chunksToSend;

    FILE *f = NULL;

    while (whatToDo)
    {
        /* If there are no more files to process*/
        if (fileId == fileAmount)
        {
            printf("No more work, sending message to workers to end..\n");
            whatToDo = NOMOREWORK;

            for (int i = 1; i <= nWorkers; i++)
                MPI_Send(&whatToDo, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD); /* tell workers there is no more work to be done */
            break;
        }
        /* Otherwise tell workers that there is work to do..*/
        for (int i = 1; i <= nWorkers; i++)
            MPI_Send(&whatToDo, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        /* If the pointer is Null open the File*/

        if (f == NULL)
        {
            f = fopen(file_names[fileId], "r");

            if (f == NULL)
            {
                printf("Could not open file\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill EVERY living process (root included)
            }
            amount = 0;
            // reads amount of matrices
            if (!fread(&amount, sizeof(int), 1, f))
            {
                printf("Error reading amount. Exiting...\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill EVERY living process (root included)
            }
            results[fileId] = malloc(amount * sizeof(double));
            matrixAmount[fileId] = amount;
            // reads order of matrices
            order = 0;
            if (!fread(&order, sizeof(int), 1, f))
            {
                printf("Error reading order. Exiting...");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // kill EVERY living process (root included)
            }
            //chunkQuantity = (int)(matrixAmount[fileId] / (nWorkers));
            lastChunkSize = matrixAmount[fileId] % (nWorkers);
        }
        chunksToSend = chunkId == (matrixAmount[fileId] - lastChunkSize) ? lastChunkSize : (nWorkers);
        for (int j = 1; j <= nWorkers; j++)
        {
            chunkInfo chunk;
            // if there's no more chunks, tell worker's that they should not wait for work
            if(matrixAmount[fileId] == chunkId){
                chunk.isLastChunk=1;
                MPI_Send(&chunk, sizeof(chunkInfo), MPI_BYTE, j, 0, MPI_COMM_WORLD);
                continue;
            }
            else{
                chunk.isLastChunk = 0;
                chunk.fileId = fileId;
                chunk.order = order;
                chunk.matrixId = matrixId;
                matrixId++;

            }
            // send matrix information
            MPI_Send(&chunk, sizeof(chunkInfo), MPI_BYTE, j, 0, MPI_COMM_WORLD);
            double *matrix = (double *)malloc(sizeof(double) * order * order);
            if (!fread(matrix, 8, order * order, f))
                break;
            // send matrix
            MPI_Send(matrix, order * order, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
            chunkId++;
            free(matrix);
        }

        // wait for partial results
        for (workerId = 1; workerId <= chunksToSend; workerId++)
        {
            double partialResultData[3]; /* received partial info computed by workers */
            MPI_Recv(partialResultData, 3, MPI_DOUBLE, workerId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            storePartialResult(results, partialResultData[1], partialResultData[2], partialResultData[0]); /* store partial results*/
        }
        // if there are no more matrices in this file, close it
        if (chunkId == matrixAmount[fileId])
        {
            fclose(f);
            f = NULL;
            fileId++;
        }
    }
    printResults(results, fileAmount);
} 

void work(int rank)
{
    unsigned int whatToDo; /* command */
    double determinant_result;
    int order;
    double *matrix;
    chunkInfo chunk;

    while (true)
    {
        MPI_Recv(&whatToDo, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (whatToDo == NOMOREWORK) /* no more work to be done by workers */
        {
            printf("Worker with rank %d terminated...\n", rank);
            return;
        }

        MPI_Recv(&chunk, sizeof(chunkInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(chunk.isLastChunk){ /* avoid staying blocked wiating for more work*/
            continue;
        }
        order = chunk.order;
        matrix = (double *)malloc(sizeof(double) * order * order);
        MPI_Recv(matrix, order * order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        determinant_result = determinant(order, matrix);
        double partialResultData[3];
        partialResultData[0] = determinant_result;
        partialResultData[1] = chunk.fileId;
        partialResultData[2] = chunk.matrixId;
        MPI_Send(partialResultData, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); /* send partial info computed to dispatcher */
        free(matrix);
    }
}

/**
 * @brief Method invoked by Dispatcher to Print the final Results gathered from all workers
 *
 * @param fileAmount the amount of files used to create and process chunks
 * @param matrixAmount the amount of matrices
 */
void printResults(double **results, int fileAmount)
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
