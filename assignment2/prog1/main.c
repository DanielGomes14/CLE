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

#include "dispatcher/dispatcher.h"
// #include "worker/worker.h"
#include "cmd/processCommandLine.h"

void dispatcher(char ***fileNames, int fileAmount, int* size);

void worker(int rank);


/* General definitions */

#define WORKTODO 1
#define NOMOREWORK 0

int main(int argc, char *argv[])
{
    struct timespec start, finish;  // time limits
    int fileAmount = 0;             // amount of files
    char **fileNames;               // array of pointers, each pointer points to a string literal present in CMD

    int rank,                       // process rank
        size,                       // amout of processes
        workerAmount;               // amount of worker processes

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    workerAmount = size - 1;

    if (rank == 0)                                      // root process (dispatcher)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);     // start counting time

        // process command line input
        if(processInput (argc, argv, &fileAmount, &fileNames) == EXIT_FAILURE)
        {
            free(fileNames);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);    // kill EVERY living process (root included)
        }
        else
            dispatcher(&fileNames, fileAmount, &size);         // dispatcher logic


        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);    // end counting time

        // calculate execution time
        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nRoot elapsed time = %.6f s\n", executionTime);  // print execution time
    }
    else
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);     // start counting time

        worker(rank);                                   // worker logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);    // end counting time

        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nWorker <%d> elapsed time = %.6f s\n", rank, executionTime);  // print execution time
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}


void dispatcher(char ***fileNames, int fileAmount, int* size)
{
    //TODO: do stuff here

    // char** names = *fileNames;
    // FILE* f = NULL;
    // int filesToProcess = 1, actualFileID = 0, chunkSize = 0;
    // int *chunk = NULL;

    /*

    3 códigos de execução
        0 - nada
        1 - preparar pra receber e processar chunk
        2 - devolder resultados parciais

    percorre todos os ficheiros
        para cada ficheiro vai lendo e enviando chunks para cada worker
            se final ficheiro
                 

    for file in files

        chunk2process = true

        while(chunks2process)
            for worker in workers
                if chunk2process
                    chunk2process = read chunk
                    send 1
                    send size
                    send chunk
                else
                    send 0

            if !chunks2process
                for worker in workers
                    send 2
                    receive results

    */

    char *names = *fileNames;

    for(int i = 0; i < fileAmount; i++)
    {
        FILE* f = fopen(names[i], "r");
        if(!f)
        {
            perror("error openining file\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int chunkToProcess = 0;
        int *chunk = NULL, chunkSize = 0, ;
        while(chunkToProcess)
        {
            for(int j = 1; j < size; j++)
            {
                if(chunkToProcess)
                {
                    chunk = readChunk(f, &chunkSize, &chunkToProcess);
                }
            }
        }

    }

    while(filesToProcess)
    {
        // send chunk cycle
        for(int i = 1; i < size; i++)
        {
            //TODO: implement readChunk
            chunk = readChunk(fileNames, &actualFileID, f, &filesToProcess, &chunkSize);

            if(filesToProcess)
            {
                // send boolean
                MPI_Send(&filesToProcess, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // send chunk size
                MPI_Send(&chunkSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // send chunk
                MPI_Send(chunk, chunkSize, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                filesToProcess = 0;
                fclose(f);
                break;
            }
        }

        // no more files to read
        if(!filesToProcess)
            for(int i = 1; i < size; i++)
            {
                // send boolean to every worker process
                MPI_Send(&filesToProcess, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }  
    }

    return;
}

void worker(int rank)
{
    //TODO: do stuff here
    int filesToProcess = 1, chunkSize = 0;
    int* chunk = NULL;


    while(filesToProcess)
    {
        // receive boolean
        MPI_Recv(&filesToProcess, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(!filesToProcess)
            break;

        // receive chunk size
        MPI_Recv(&chunkSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // receive chunk
        MPI_Recv(chunk, chunkSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);



    }

    return;
}