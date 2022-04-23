#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>

#include "./cmd/processCommandLine.h"
#include "./shared/shared.h"
#include "./utils/utils.h"
#define CHUNK_SIZE 200

/** \brief pointer to status of workers*/
int *statusWorkers;

/** \brief producer threads return status array */
int statusProd[1];

/** \brief worker life cycle routine */
static void *work(void *id);

/** \brief shows program usage*/
static void printUsage(char* cmdName);

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
 * @brief Main function
 *
 * Process command line information.
 * Produce chunks and save them in shared region.
 * Create worker threads to process chunks in shared region.
 * Gather and consolidate gathered info of each chunk.
 * Print results.
 *
 * @param argc number of words from the command line
 * @param argv array with the words from the command line
 * @return int (status of operation)
 */
int main(int argc, char *argv[])
{
    // processes command line information
    if (processInput(argc, argv, &threadAmount, &fileAmount, &fileNames))
        exit(-1);

    pthread_t tIdWorker[threadAmount];  //  workers internal thread id array
    unsigned int workers[threadAmount]; // workers application defined thread id array

    // pass reference to the shared structure
    int results[fileAmount][3];     // 3 counters (consonant, vowel and word)
    memset(results, 0, sizeof(results[fileAmount][3]) * 3 * fileAmount);

    // initialise workers array
    for (int t_ind = 0; t_ind < threadAmount; t_ind++)
        workers[t_ind] = t_ind;

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

    int* status_p;
    // await for workers
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

    printResults(fileAmount, results, &fileNames);

    return 0;
}


/**
 * @brief Worker Function
 *
 * Does the worker tasks
 * @param par pointer to application defined worker identification
 */
static void *work(void *par)
{
   // get worker id
   workerData* data = (workerData*) par;
   unsigned int id = data->threadId;
   int* results = data->results;

    // starts processing chunks
    processChunks(id, results);

    // end work
    statusWorkers[id] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[id]);
}


void printResults(int fileAmount, int results[fileAmount][3], char*** fileNames){
    char** names = *fileNames;
    printf("\n---------------------RESULTS---------------------\n");
    for(int i = 0; i < fileAmount; i++)
    {
        printf("File: <%s>\n", (*(names + i)));
        printf("Consonant: <%d>\n", results[i][0]);
        printf("Vowel: <%d>\n", results[i][1]);
        printf("Words: <%d>\n", results[i][2]);
        printf("\n");
        
    }
}