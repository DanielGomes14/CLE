#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <errno.h>

// https://stackoverflow.com/questions/50083744/how-to-create-an-array-without-declaring-the-size-in-c

/** \brief flag */
extern int stillProcessing;

/** \brief status of main(The Producer)*/
extern int statusProd;

/** \brief  Worker threads return status array*/
extern int *statusWorkers;

/** \brief Amount of worker threads*/
extern int threadAmount;

/** \brief Amount of files*/
extern int fileAmount;

/** \brief pointer to array of pointers, each pointer points to the name of a file*/
extern char **fileNames;

/** \brief storage region*/
static chunkInfo mem[2];

/** \brief insertion pointer and retrieval pointer*/
static unsigned int ii, ri;

/** \brief flag to check if fifo is full*/
static bool full;

/** \brief Conditions to synchronize data transfer when full/empty*/
static pthread_cond_t fifoFull, fifoEmpty;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief locking flag to warrant mutual exclusion when writing results*/
static pthread_mutex_t accessWR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/**
 * @brief Initialise the data transfer region.
 *
 */
static void initialization() {
    ii = ri = 0;  // initialize input and retieval indexes at 0
    full = false;  // fifo is not full

    // iniitialize synchronization points of main(producer) and workers
    pthread_cond_init(&fifoFull, NULL);
    pthread_cond_init(&fifoEmpty, NULL);
}

/**
 * @brief Stores chunk in shared region
 * The information is saved in a FIFO so it can be obtained in order by the worker threads.
 * @param info chunk info
 */
void storeChunk(chunkInfo info)
{

    if ((statusProd = pthread_mutex_lock(&accessCR)) != 0) /* enter monitor */
    {
        errno = statusProd; /* save error in errno */
        perror("error on entering monitor(CF)");
        statusProd = EXIT_FAILURE;
        pthread_exit(&statusProd);
    }
    pthread_once(&init, initialization); /* internal data initialization */

    while (full) /* wait if the data transfer region is full */
    {
        if ((statusProd = pthread_cond_wait(&fifoFull, &accessCR)) != 0)
        {
            errno = statusProd; /* save error in errno */
            perror("error on waiting in fifoFull");
            statusProd = EXIT_FAILURE;
            pthread_exit(&statusProd);
        }
    }

    mem[ii] = info;
    ii = (ii + 1) % 2;
    full = (ii == ri);

    if ((statusProd = pthread_cond_signal (&fifoEmpty)) != 0)      /* let a consumer know that a value has been
                                                                                                               stored */
     { errno = statusProd;                                                             /* save error in errno */
       perror ("error on signaling in fifoEmpty");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }

  if ((statusProd = pthread_mutex_unlock (&accessCR)) != 0)                                  /* exit monitor */
     { errno = statusProd;                                                            /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusProd = EXIT_FAILURE;
       pthread_exit (&statusProd);
     }

}

/**
 * @brief Get the Chunk object
 * Get the next chunk from the FIFO.
 * @param workerId Id of worker thread
 * @return chunkInfo 
 */
chunkInfo getChunk(unsigned int workerId) {

    chunkInfo info;

    if ((statusWorkers[workerId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
        { errno = statusWorkers[workerId];                                                            /* save error in errno */
        perror ("error on entering monitor(CF)");
        statusWorkers[workerId] = EXIT_FAILURE;
        pthread_exit (&statusWorkers[workerId]);
    }
    pthread_once (&init, initialization);                                              /* internal data initialization */

    if(!full && !stillProcessing){
        statusWorkers[workerId] = pthread_mutex_unlock(&accessCR);
        info.bufferSize = -1;
        return info;
    }

    while ((ii == ri) && !full)                                           /* wait if the data transfer region is empty */
    { 
        if ((statusWorkers[workerId] = pthread_cond_wait (&fifoEmpty, &accessCR)) != 0)
        { 
            errno = statusWorkers[workerId];                                                          /* save error in errno */
            perror ("error on waiting in fifoEmpty");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        if(!full && !stillProcessing){
            statusWorkers[workerId] = pthread_mutex_unlock(&accessCR);
            info.bufferSize = -1;
            return info;
        }

    }

    info = mem[ri];
    ri = (ri + 1) % 2;
    full = false;

    if ((statusWorkers[workerId] = pthread_cond_signal (&fifoFull)) != 0)       /* let a producer know that a value has been                                                                                              retrieved */
     { errno = statusWorkers[workerId];                                                             /* save error in errno */
       perror ("error on signaling in fifoFull");
       statusWorkers[workerId] = EXIT_FAILURE;
       pthread_exit (&statusWorkers[workerId]);
     }

  if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
     { errno = statusWorkers[workerId];                                                             /* save error in errno */
       perror ("error on exiting monitor(CF)");
       statusWorkers[workerId] = EXIT_FAILURE;
       pthread_exit (&statusWorkers[workerId]);
     }

    return info;
}


// void showFileNames(){
//     printf("NOMES DOS FICHEIROS:\n");
//     for(int i = 0; i < fileAmount; i++){
//         printf("\t%s\n", fileNames[i]);
//     }
// }

int currentFileId = 0;
FILE* f = NULL;

void processChunks(unsigned int workerId, int*** results){
    /*
    get worker id
    get pointer to result matrix
    enter read mutex
        if FILE* == NULL
            open file
        get chunk from next file(store it on a char[])
        if EOF
            update nextFileId(++)
    exit read mutex
    process readden chunk
    enter write results mutex
        save results on result matrix
    exit write results mutex
    */
   while(currentFileId < fileAmount){
    int chunk[200] = {0};    // struct to save chunk
    int aux;                 // aux variable
    int chunkLenght = 0;
    int chunkFileId = 0;

        // enter read mutex
    if ((statusWorkers[workerId] = pthread_mutex_lock (&accessCR)) != 0)                                   /* enter monitor */
        { 
            errno = statusWorkers[workerId];                                                            /* save error in errno */
            perror ("error on entering monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }
        
        // open file(or next file)
        if(!f)
        {
            f = fopen(fileNames[currentFileId], "r");
            chunkFileId = currentFileId;
            if(!f)
            {
                perror("Error opening file.");
                statusWorkers[workerId] = EXIT_FAILURE;
                pthread_exit (&statusWorkers[workerId]);
            }
        }

        // read chunk from file
        while(1)
        {
            aux = fgetc(f);

            if(chunkLenght == 200)      // max chunk lenght
                break;
            else if(aux == EOF)  // end of file
            {
                fclose(f);
                currentFileId++;
                break;
            }

            chunk[chunkLenght++] = aux;
        }

        // exit read mutex
        if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessCR)) != 0)                                   /* exit monitor */
        { 
            errno = statusWorkers[workerId];                                                             /* save error in errno */
            perror ("error on exiting monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        int partialVowel = 0, partialConsonat = 0, partialWords = 0;
        processChunk(chunk, chunkLenght, &partialVowel, &partialConsonat, &partialWords);

        // FIXME: n sei se o mutex aqui é necessário
        // enter write results mutex
        if((statusWorkers[workerId] = pthread_mutex_lock(&accessWR)) != 0){
            errno = statusWorkers[workerId];                                                            /* save error in errno */
            perror ("Error on entering mutex to write.");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        // write result
        *(results + (3 * chunkFileId) + 0) += partialVowel;
        *(results + (3 * chunkFileId) + 1) += partialConsonat;
        *(results + (3 * chunkFileId) + 2) += partialWords;

        // exit write results mutex
        if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessWR)) != 0)                                   /* exit monitor */
        { 
            errno = statusWorkers[workerId];                                                             /* save error in errno */
            perror ("error on exiting monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

   }

    statusWorkers[workerId] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[workerId]);
}