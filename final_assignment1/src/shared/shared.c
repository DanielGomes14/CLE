#include "shared.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

extern int vowelCounter = 0;
extern int consonantCounter = 0;
extern int wordCount = 0;
//https://stackoverflow.com/questions/50083744/how-to-create-an-array-without-declaring-the-size-in-c

/** \brief  Worker threads return status array*/
extern int *statusWorkers;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the data transfer region is initialized exactly once */
static pthread_once_t init = PTHREAD_ONCE_INIT;

/**
 * @brief Initialise the data transfer region.
 * 
 */
static void initialisation(){

}

chunkInfo getChunk(){
    pChunkInfo info;

    // gets chunk info

    return *info;
}

void processChunkInfo(chunkInfo info){

    // do stuff here

}