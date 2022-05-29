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

/**
 * @brief Dispatcher process main logic 
 */
void dispatcher(char ***fileNames, int fileAmount, int size, int *results);

/**
 * @brief Worker process main logic 
 */
void worker(int rank);

/**
 * @brief Main function
 * 
 * Main logic is dependent on the rank of the process (0 for DIspatcher, rest for Worker)
 * 
 * @param argc number of words fromt eh command line
 * @param argv array with the words fromt he command line
 * @return int (return status of the operation)
 */
int main(int argc, char *argv[])
{
    struct timespec start, finish;  // time limits
    int fileAmount = 0;             // amount of files
    char **fileNames;               // array of pointers, each pointer points to a string literal present in CMD

    int rank,                       // process rank
        size;                       // amout of processes

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)                                      // root process (dispatcher)
    {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);     // start counting time

        // process command line input
        if(processInput (argc, argv, &fileAmount, &fileNames) == EXIT_FAILURE)
        {
            free(fileNames);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);    // kill EVERY living process (root included)
        }

        // allocate memory for results
        int* results = malloc(sizeof(int) * fileAmount * 3);
        if(!results)
        {
            perror("error alocating memory for results");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for(int i = 0; i < fileAmount; i++)
            for(int j = 0; j < 3; j++)
                *((results + i * 3) + j) = 0;

        dispatcher(&fileNames, fileAmount, size, results);         // dispatcher logic

        clock_gettime(CLOCK_MONOTONIC_RAW, &finish);    // end counting time

        // calculate execution time
        float executionTime = (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

        printf("FINAL RESULTS\n");
        for(int i = 0; i < fileAmount; i++)
        {
            printf("\nFile: <%s>\n", fileNames[i]);
            printf("CONSONANTS: <%d>\n", *((results + i * 3) + 0));
            printf("VOWELS: <%d>\n", *((results + i * 3) + 1));
            printf("WORDS: <%d>\n", *((results + i * 3) + 2));
        }

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
