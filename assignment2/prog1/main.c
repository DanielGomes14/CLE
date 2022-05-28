#include <sys/types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "cmd/processCommandLine.h"
#include "dispatcher/dispatcher.h"
#include "worker/worker.h"

void dispatcher(char ***fileNames, int fileAmount, int size, int* results);

void worker(int rank);

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

        int results[fileAmount][3];                     // [][][]
        dispatcher(&fileNames, fileAmount, size, (int*)results);         // dispatcher logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);    // end counting time

        // calculate execution time
        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("\nRoot elapsed time = %.6f s\n", executionTime);  // print execution time

        // printf("FINAL RESULTS\n");
        // for(int i = 0; i < fileAmount; i++)
        // {
        //     printf("File: <%s>\n", fileNames[i]);
        //     printf("VOWELS: <%d>\n", results[i][0]);
        //     printf("CONSONANTS: <%d>\n", results[i][1]);
        //     printf("WORDS: <%d>\n", results[i][2]);
        // }
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


// void dispatcher(char ***fileNames, int fileAmount, int* size, int* results)
// {
//     /*

//     3 códigos de execução
//         0 - nada
//         1 - preparar pra receber e processar chunk
//         2 - devolder resultados parciais
//         3 - finish everything                 

//     for file in files

//         chunk2process = true

//         while(chunks2process)
//             for worker in workers
//                 if chunk2process
//                     chunk2process = read chunk
//                     send 1
//                     send size
//                     send chunk
//                 else
//                     send 0

//             if !chunks2process
//                 for worker in workers
//                     send 2
//                     receive results

//     */

//     char *names = *fileNames;
//     int idleCode = 0, processCode = 1, returnCode = 2;

//     for(int i = 0; i < fileAmount; i++)
//     {
//         FILE* f = fopen(names[i], "r");
//         if(!f)
//         {
//             perror("error openining file\n");
//             MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//         }

//         int chunkToProcess = 1;
//         int *chunk = NULL, chunkSize = 0;
//         while(chunkToProcess)
//         {
//             for(int j = 1; j < size; j++)       // cycle to send chunks of a file
//             {
//                 if(chunkToProcess)              // chunks to process
//                 {
//                     chunk = readChunk(f, &chunkSize, &chunkToProcess);
//                     MPI_Send(&processCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
//                     MPI_Send(&chunkSize, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
//                     MPI_Send(chunk, chunkSize, MPI_INT, j, 0, MPI_COMM_WORLD);
//                 }
//                 else                            // no more chunks, but is still iteraating through workers
//                 {
//                     MPI_Send(&idleCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
//                 }
//             }

//             if(!chunkToProcess)                 // no more chunks to process, send msg to workers to return partial values
//             {
//                 for(int j = 1; j < size; j++)
//                 {
//                     int* partialResults = NULL;
//                     MPI_Send(&returnCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
//                     MPI_Recv(partialResults, 3, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//                     printf("FILE <%s> RESULTS:\n", names[i]);

//                     *((results + i*3) + 0) = *(partialResults + 0);
//                     printf("Total words: <%d>\n", *(partialResults + 0));

//                     *((results + i*3) + 1) = *(partialResults + 1);
//                     printf("Vowels: <%d>\n", *(partialResults + 0));

//                     *((results + i*3) + 2) = *(partialResults + 2);
//                     printf("Consonants: <%d>\n", *(partialResults + 0));

//                     free(partialResults);
//                 }
//             }

//         }

//     }

//     int endCode = 3;
//     for(int i = 1; i < size; i++)
//         MPI_Send(&endCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

//     return;
// }

// void worker(int rank)
// {
//     /*
//         while(1)
//             recv code
//             if  0
//                     continue
//             if  1
//                     recv chunk size
//                     recv chunk
//                     process chunk
//                     save partial results
//             if  2
//                     send partial results
//                     reset counters
//             if  3
//                     break
//     */

//     int *chunk = NULL, *counters = NULL;
//     int executionCode = 1, chunkSize = 0;

//     counters = malloc(sizeof(int) * 3); // words, vowels, consonants
//     if(!counters)
//     {
//         perror("error allocating memory for counters");
//         MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
//     }

//     while(1)
//     {
//         // receive boolean
//         MPI_Recv(&executionCode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         if(executionCode == 0)          // idle
//         {
//             continue;
//         }
//         else if(executionCode == 1)     // recv chunk size and chunk and process chunk
//         {
//             // recv chunk size
//             MPI_Recv(&chunkSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//             // recv chunk
//             MPI_Recv(chunk, chunkSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//             // process chunk
//             chunkProcessing(chunk, chunkSize, counters, counters + 1, counters + 2);

//             // free processed chunk
//             // CHECK IF IT IS NECESSARY TO FREE MEMORY
//             free(chunk);
//         }
//         else if(executionCode == 2)     // send counters
//         {
//             // send counters
//             MPI_Send(counters, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
//             // reset counters
//             *(counters + 0) = 0;
//             *(counters + 1) = 0;
//             *(counters + 2) = 0;
//         }
//         else if(executionCode == 3)     // end worker
//         {
//             // CHECK IF IT IS NECESSARY TO FREE MEMORY
//             free(counters);
//             break;
//         }
//     }

//     return;
// }