#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>

/** \brief flag */
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
static chunkInfo mem[5];

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
 **/
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
    ii = (ii + 1) % 5;
    full = (ii == ri);
    if ((pthread_cond_signal(&fifoEmpty)) != 0) /* let a consumer know that a value has been
                                                                                                         stored */
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

    while ((ii == ri) && !full && stillProcessing) /* wait if the data transfer region is empty */
    {
        if ((statusWorkers[workerId] = pthread_cond_wait(&fifoEmpty, &accessCR)) != 0)
        {
            errno = statusWorkers[workerId]; /* save error in errno */
            perror("error on waiting in fifoEmpty");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit(&statusWorkers[workerId]);
        }
    }
    if ((ii == ri) && !full && !stillProcessing)
    {
        //info = malloc(sizeof(chunkInfo));
        info.isLastChunk = 1;
        statusWorkers[workerId] = pthread_mutex_unlock(&accessCR);
        return info;

    }
    info = mem[ri];
    ri = (ri + 1) % 5;
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

void awakeWorkers()
{
    if (pthread_mutex_lock(&accessCR) != 0)
    {                                            /* enter monitor */
        perror("error on entering monitor(CF)"); /* save error in errno */
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
    printf("broadcasting...\n");
    stillProcessing = 0; // update flag
    if (pthread_cond_broadcast(&fifoEmpty) != 0) {
        /* let a consumer know that a value has been stored */
        perror("error on broadcast in fifoEmpty");
        exit(1);
    }
    printf("lixo\n");
    if (pthread_mutex_unlock(&accessCR) != 0)
    {                                           /* exit */
        perror("error on exiting monitor(CF)"); /* save error in errno */
        int status = EXIT_FAILURE;
        pthread_exit(&status);
    }
}

void storePartialResults(unsigned int workerId, int fileId, int matrixId, double determinant)
{
    // if ((statusWorkers[workerId] = pthread_mutex_lock(&accessCR)) != 0)
    // {                                            /* enter monitor */
    //     perror("error on entering monitor(CF)"); /* save error in errno */
    //     int status = EXIT_FAILURE;
    //     pthread_exit(&status);
    // }
    results[fileId][matrixId] = determinant; /* store value */

    // if ((statusWorkers[workerId] = pthread_mutex_unlock(&accessCR)) != 0)
    // {                                           /* exit */
    //     perror("error on exiting monitor(CF)"); /* save error in errno */
    //     int status = EXIT_FAILURE;
    //     pthread_exit(&status);
    // }
}
