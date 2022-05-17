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
#include <./dispatcher.h>
/** \brief number of workers */
int nWorkers;

/** \brief  Array containing the amount of matrices **/
int *matrixAmount;

/* General definitions */

#define WORKTODO 1
#define NOMOREWORK 0

int main(int argc, char *argv[])
{
    struct timespec start, finish; /* time limits */
    int threadAmount = 0;          // number of threads;
    int fileAmount = 0;            // amount of files
    int *status_p;                 // pointer to execution status
    char **fileNames;              // array with the names of the files

    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    nWorkers = size - 1; /* number of workers */

    // no inputs where given
    if (argc == 1)
    {
        perror("No arguments were provided.");
        exit(-1);
    }

    if (rank == 0) /* dispatcher */
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start); /* begin of measurement */

        // processes command line information
        if (processInput(argc, argv, &threadAmount, &fileAmount, &fileNames))
            exit(-1);

        dispatcher(&fileNames, fileAmount);

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish); /* end of measurement */
        printf("\nElapsed tim = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);
    }
    else
    {
        worker(rank); /* worker, worker life cycle */
    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}

void dispatcher(char ***fileNames, int fileAmount)
{
    int workerId;
    unsigned int whatToDo = WORKTODO;; /* command */
    double **results=malloc(fileAmount * sizeof(double *));
    matrixAmount = malloc(fileAmount * sizeof(int));
    char **file_names = (*fileNames);
    char *file_name;
    int amount, order, chunkQuantity,lastChunkSize = 0;
    int matrixId, fileId, chunkId = 0;
    int chunksToSend;

    FILE *f;

    while (whatToDo)
    {
        /* If there are no more files to process*/
        if (fileId == fileAmount && f == NULL)
        {
            printf("No more work, sending message to workers to end..\n");
            whatToDo = NOMOREWORK;
            for (int i = 1; i <= nWorkers; i++)
                MPI_Send(&whatToDo, 1, MPI_C_BOOL, i, 0, MPI_COMM_WORLD); /* tell workers there is no more work to be done */
            break;
        }
        if (f == NULL)
        {
            f = fopen(file_name, "r");
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

            results[fileId] = malloc(amount * sizeof(double));
            matrixAmount[fileId] = amount;
            // reads order of matrices
            order = 0;
            if (!fread(&order, sizeof(int), 1, f))
            {
                printf("Error reading order. Exiting...");
                exit(-1);
            }
            chunkQuantity = (int) (matrixAmount[fileId] / (nWorkers-1));
            lastChunkSize = matrixAmount[fileId] % (nWorkers-1);
        }
        chunksToSend = chunkId == matrixAmount[fileId] ? chunkQuantity : lastChunkSize;
        for (int j = 0; j < chunksToSend; j++)
        {
            double *matrix = (double *)malloc(sizeof(double) * order * order);
            if (!fread(matrix, 8, order * order, f))
                break;
            chunkInfo chunk;
            chunk.matrixPtr = matrix;
            chunk.fileId = fileId;
            chunk.isLastChunk = 0;
            chunk.order = order;
            chunk.matrixId = matrixId;
            matrixId++;
            MPI_Send(&chunk, sizeof(chunkInfo), MPI_DOUBLE, workerId, 0, MPI_COMM_WORLD);
            chunkId++;
        }
        
        for (workerId = 1; workerId <= nWorkers; workerId++)
        {
            double partialResultData[3]; /* received partial info computed by workers */
            MPI_Recv(partialResultData, 3, MPI_DOUBLE, workerId, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            storePartialResult(results, partialResultData[1], partialResultData[2], partialResultData[0]);
        }
        if(chunksToSend == lastChunkSize)
            fclose(f);
        fileId++;
    }
    printResults(results,fileAmount);
}

void work(int rank)
{
    unsigned int whatToDo; /* command */
    double determinant_result;
    chunkInfo chunk;

    while (true)
    {
        MPI_Recv(&whatToDo, 1, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (whatToDo == NOMOREWORK) /* no more work to be done by workers */
        {
            printf("Worker with rank %d terminated...\n", rank);
            return;
        }
        MPI_Recv(&chunk, sizeof(chunkInfo), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        determinant_result = determinant(chunk.order, chunk.matrixPtr);
        double partialResultData[3];
        partialResultData[0] = determinant_result;
        partialResultData[1] = chunk.fileId;
        partialResultData[2] = chunk.matrixId;
        MPI_Send(partialResultData, 3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); /* send partial info computed to dispatcher */
    }
}

/**
 * @brief Method invoked by Dispatcher to Print the final Results gathered from all workers
 * 
 * @param fileAmount the amount of files used to create and process chunks
 * @param matrixAmount the amount of matrices
 */
void printResults(double **results,int fileAmount)
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
