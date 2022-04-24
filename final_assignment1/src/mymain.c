#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>

#include "./cmd/processCommandLine.h"
#include "./shared/shared.h"
#include "./utils/utils.h"


/** \brief pointer to status of workers*/
int *statusWorkers;

/** \brief producer threads return status array */
int statusProd[1];

/** \brief worker life cycle routine */
static void *work(void *id);

/** \brief processes chunks*/
void processChunks(unsigned int workerId, int* results);

/** \brief Show results*/
void printResults(int threadAmount, int results[threadAmount][3], char*** fileNames);

int threadAmount = 0;   // number of threads;
int fileAmount = 0;     // amount of files
char **fileNames;       // pointer to name of files

/** \brief function used to produce chunks */
void produceChunks(char ***fileNames, int fileAmount, int ***results);

/**
 * \brief Main function
 *
 * Process command line information.
 * Produce chunks and save them in shared region.
 * Create worker threads to process chunks in shared region.
 * Gather and consolidate gathered info of each chunk.
 * Print results.
 *
 * \param argc number of words from the command line
 * \param argv array with the words from the command line
 * \return int (status of operation)
 */
int main(int argc, char *argv[])
{
    struct timespec start, finish;  // time limits

    // processes command line information
    if (processInput(argc, argv, &threadAmount, &fileAmount, &fileNames))
        exit(-1);

    // get starting time
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    pthread_t tIdWorker[threadAmount];  //  workers internal thread id array

    // pass reference to the shared structure
    int results[fileAmount][3];     // 3 counters (consonant, vowel and word)
    memset(results, 0, sizeof(results[fileAmount][3]) * 3 * fileAmount);

    statusWorkers = malloc(sizeof(int) * threadAmount);

    // create workers
    for (int t = 0; t < threadAmount; t++)
    {
        workerData data;
        data.threadId = t;
        data.results = (int*)results;

        // create(t)
        if (pthread_create(&tIdWorker[t], NULL, work, (void *)&data) != 0) /* thread consumer */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
    }

    // await for workers
    int* status_p;
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

    // get finishing time
    clock_gettime(CLOCK_MONOTONIC_RAW, &finish);

    // show final results
    printResults(fileAmount, results, &fileNames);

    printf("Elapsed time = %.6f s\n", (finish.tv_sec - start.tv_sec) / 1.0 + (finish.tv_nsec - start.tv_nsec) / 1000000000.0);

    return 0;
}

/**
 * \brief worker thread main logic
 * 
 * Obtains data regarding the worker thread, then proceds to process chunks.
 * 
 * \param par pointer to worker thread data (id and pointer to results matrix)
 */
static void *work(void *par)
{
   // get worker data(data and pointer to results matrix)
   workerData* data = (workerData*) par;
   unsigned int id = data->threadId;
   int* results = data->results;

    // starts processing chunks
    processChunks(id, results);

    // end work
    statusWorkers[id] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[id]);
}

/**
 * \brief Shows final results obtained from processing files.
 * 
 * \param fileAmount Total amount of files
 * \param results Matrix where results are stored
 * \param fileNames Pointer to array with the name of processed files
 */
void printResults(int fileAmount, int results[fileAmount][3], char*** fileNames){
    char** names = *fileNames;
    printf("\n---------------------RESULTS---------------------\n");
    for(int i = 0; i < fileAmount; i++)
    {
        printf("File: <%s>\n", (*(names + i)));
        printf("Words: <%d>\n", results[i][2]);
        printf("Vowel: <%d>\n", results[i][0]);
        printf("Consonant: <%d>\n", results[i][1]);
        printf("\n");
        
    }
}