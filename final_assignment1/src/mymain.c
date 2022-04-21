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

int *statusWorkers;

/** \brief producer threads return status array */
int statusProd[1];

int stillProcessing;

/** \brief producer life cycle routine */
static void *producer(void *id);
/** \brief worker life cycle routine */
static void *work(void *id);

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

    int threadAmount = 0; // number of threads;
    int fileAmount = 0;   // amount of files
    int *status_p;        // pointer to execution status
    char **fileNames;     // array with the names of the files

    stillProcessing = 1;

    // no inputs where given
    if (argc == 1)
    {
        perror("No arguments were provided.");
        exit(-1);
    }

    // processes command line information
    if (processInput(argc, argv, &threadAmount, &fileAmount, &fileNames))
        exit(-1);

    pthread_t tIdProd[1];               /* producers internal thread id array */
    pthread_t tIdWorker[threadAmount];  //  workers internal thread id array
    unsigned int prod[1];               /* producers application defined thread id array */
    unsigned int workers[threadAmount]; // workers application defined thread id array

    // pass reference to the shared structure
    // 3 counters (consonant, vowel and word)
    // TODO: Pass 3 to Const variable
    int results[fileAmount][3]; // structure to save the results of each file
    memset(results, 0, sizeof(results[3][fileAmount]) * 3 * fileAmount);

    // initialise workers array
    for (int t_ind = 0; t_ind < threadAmount; t_ind++)
        workers[t_ind] = t_ind;

    statusWorkers = malloc(sizeof(int) * threadAmount);

    // initialize producer;
    prod[0] = 0;

    // arguments to be passed to producer starting method
    threadData args;
    args.thread_id = prod[0];
    args.fileAmount = fileAmount;
    args.fileNames = &fileNames;
    args.results = &results;

    if (pthread_create(&tIdProd[0], NULL, producer, (void *)&args) != 0) /* thread producer */
    {
        perror("error on creating thread producer");
        exit(EXIT_FAILURE);
    }

    for (int t = 0; t < threadAmount; t++)
    {
        // create(t)
        if (pthread_create(&tIdWorker[t], NULL, work, (void *)&workers[t]) != 0) /* thread consumer */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
    }

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

    if (pthread_join(tIdProd[0], (void *)&status_p) != 0) /* thread producer */
    {
        perror("error on waiting for thread producer");
        exit(EXIT_FAILURE);
    }
    printf("thread producer, with id %u, has terminated: ", 0);
    printf("its status was %d\n", *status_p);

    return 0;
}

// main
//     process cmd
//     create workers in wait mode
//     for file
//         create chunk
//         signal workers(set flag to true)
//     signal workers to end(signal flag to false)
//     espera por workers
//     end

void produceChunks(char ***fileNames, int fileAmount, int ***results)
{
    /*
    for file
        open file
        for chunk in file
            dup file pointer
            create chunk
            put chunk in shared region
    */
    char **file_names = (*fileNames);
    int fileSize = 0;
    for (int i = 0; i < fileAmount; i++)
    {
        char *file_name = file_names[i];

        FILE *f = fopen(file_name, "r");
        if (!f)
        {
            perror("Error opening file.");
            exit(-1);
        }

        // calculate quantity of chunks
        // move original pointer
        fseek(f, 0L, SEEK_END); // move to end
        fileSize = ftell(f);    // get file size
        rewind(f);              // move back to start

        int chunkQuantity = (fileSize / CHUNK_SIZE) + 1;
        int lastChunkSize = fileSize % CHUNK_SIZE;

        // create chunks
        for (int i = 0; i < chunkQuantity; i++)
        {
            // chunk initialization
            chunkInfo chunk;
            chunk.f = fdopen(dup(fileno(f)), "r"); // duplicate file pointer
            chunk.fileId = i;
            chunk.matrixPtr = *results;
            // check if it is the last Chunk
            chunk.bufferSize = i == chunkQuantity - 1 ? lastChunkSize : CHUNK_SIZE;

            // put chunk on shared region
            storeChunk(chunk);

            // if there is still chunks for processing
            if (i != chunkQuantity - 1)
                fseek(f, 0L, SEEK_SET + ((i + 1) * CHUNK_SIZE));
        }
    }
}

static void *producer(void *par)
{
    unsigned int id = *((unsigned int *)par); /* producer id */
    printf("\nteste");
    //  produceChunks(&fileNames, fileAmount, &results);

    // stillProcessing = 0; // update flag
    statusProd[id] = EXIT_SUCCESS;
    pthread_exit(&statusProd[id]);
}

/**
 * @brief Worker Function
 *
 * Does the worker tasks
 * @param par pointer to application defined worker identification
 */
static void *work(void *par)
{
    /*
    get worker id
    get pointer to result matrix
    while(files exist to be processed)
        while(fifoEmpty)
            await
        get chunkInfo
        process chunkInfo
        start mutex
            save results in result matrix
        end mutex
    */

    // TODO: call worker functions
    // TODO: make initial verification to check if main is still creating chunks

    // get thread id
    unsigned int id = *((unsigned int *)par); // worker id //

    while (1)
    {
        // get chunk
        chunkInfo chunk = getChunk(id);

        // no more chunks to be processed
        if (chunk.bufferSize == -1)
        {
            statusWorkers[id] = EXIT_SUCCESS;
            pthread_exit(&statusWorkers[id]);
        }

        // count words
        processChunk(chunk);
    }

    // end work
    statusWorkers[id] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[id]);
}
