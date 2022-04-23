#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>

/** \brief flag used to check if there are chunks still being produced */
extern int stillProcessing;

/** \brief  Worker threads return status array*/
extern int *statusWorkers;

/** \brief Amount of worker threads*/
extern int threadAmount;

/** \brief Amount of files*/
extern int fileAmount;

/** \brief Data Structure where results will be stored **/
extern double **results;
/** \brief storage region*/
static chunkInfo mem[FIFO_SIZE];

/** \brief insertion pointer and retrieval pointer*/
static unsigned int ii, ri;

/** \brief flag to check if fifo is full*/
static bool full;

/** \brief Conditions to synchronize data transfer when full/empty*/
static pthread_cond_t fifoFull, fifoEmpty;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/**
 * @brief Initialise the data transfer region.
 *
 */
static void initialization()
{
    ii = ri = 0;  // initialize input and retieval indexes at 0
    full = false; // fifo is not full

    // iniitialize synchronization points of main(producer) and workers
    pthread_cond_init(&fifoFull, NULL);
    pthread_cond_init(&fifoEmpty, NULL);
}

/**
 * @brief Method to used to store a Chunk in the Shared Region's Fifo
 * 
 * @param info an object containing the information of the Chunk to be stored
 */
void storeChunk(chunkInfo info)
{

    if ((pthread_mutex_lock(&accessCR)) != 0) /* enter monitor */
    {
        perror("error on entering monitor(CF)");
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
    pthread_once(&init, initialization); /* internal data initialization */

    while (full) /* wait if the data transfer region is full */
    {

        if ((pthread_cond_wait(&fifoFull, &accessCR)) != 0)
        {
            perror("error on waiting in fifoFull");
            int status = EXIT_FAILURE;
            pthread_exit(&status);
        }
    }

    mem[ii] = info;
    ii = (ii + 1) % FIFO_SIZE;
    full = (ii == ri);
    if ((pthread_cond_signal(&fifoEmpty)) != 0) /* let a consumer know that a value has been stored */
    {

        perror("error on signaling in fifoEmpty");
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }

    if ((pthread_mutex_unlock(&accessCR)) != 0) /* exit monitor */
    {
        perror("error on exiting monitor(CF)");
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
}
/**
 * @brief Get the Chunk object from the Data Transfer Region
 * 
 * @param workerId the Id of the worker thread perfoming this action
 * @return chunkInfo 
 */
chunkInfo getChunk(unsigned int workerId)
{

    chunkInfo info;
    if ((statusWorkers[workerId] = pthread_mutex_lock(&accessCR)) != 0) /* enter monitor */
    {
        errno = statusWorkers[workerId]; /* save error in errno */
        perror("error on entering monitor(CF)");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorkers[workerId]);
    }
    pthread_once(&init, initialization); /* internal data initialization */

    while ((ii == ri) && !full && stillProcessing) /* wait if the data transfer region is empty  and there are chunks to be stored*/
    {
        if ((statusWorkers[workerId] = pthread_cond_wait(&fifoEmpty, &accessCR)) != 0)
        {
            errno = statusWorkers[workerId]; /* save error in errno */
            perror("error on waiting in fifoEmpty");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorkers[workerId]);
        }
    }
    /* If the data tranfer region is empty and there are no more chunks to be processed*/
    if ((ii == ri) && !full && !stillProcessing)
    {
        
        info.isLastChunk = 1; /* there are no more chunks to process, thus this flag is put to 1*/
        statusWorkers[workerId] = pthread_mutex_unlock(&accessCR);
        return info;

    }
    info = mem[ri];
    ri = (ri + 1) % FIFO_SIZE;
    full = false;
    if ((statusWorkers[workerId] = pthread_cond_signal(&fifoFull)) != 0) /* let a producer know that a value has been                                                                                              retrieved */
    {
        errno = statusWorkers[workerId]; /* save error in errno */
        perror("error on signaling in fifoFull");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorkers[workerId]);
    }

    if ((statusWorkers[workerId] = pthread_mutex_unlock(&accessCR)) != 0) /* exit monitor */
    {
        errno = statusWorkers[workerId]; /* save error in errno */
        perror("error on exiting monitor(CF)");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit(&statusWorkers[workerId]);
    }

    return info;
}

/**
 * @brief Method used to signal worker waiting on the @fifoEmpty condition
 * This method is useful to signal the workers waiting where there are no more chunks
 *  being produces and can end their lifecycle
 */
void awakeWorkers()
{
    if (pthread_mutex_lock(&accessCR) != 0)
    {                                            /* enter monitor */
        perror("error on entering monitor(CF)"); /* save error in errno */
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
    stillProcessing = 0; // the producer(main) has stopped producing chunks, so this flag must be updated
    if (pthread_cond_broadcast(&fifoEmpty) != 0) {
        
        perror("error on broadcast in fifoEmpty");
        exit(1);
    }
    if (pthread_mutex_unlock(&accessCR) != 0)
    {                                           /* exit */
        perror("error on exiting monitor(CF)"); /* save error in errno */
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
}

void storePartialResults(unsigned int workerId, int fileId, int matrixId, double determinant)
{
   
    results[fileId][matrixId] = determinant; /* store value */

}
