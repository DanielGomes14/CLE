#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "shared.h"
#include "../utils/utils.h"

// https://stackoverflow.com/questions/50083744/how-to-create-an-array-without-declaring-the-size-in-c

/** \brief  Worker threads return status array*/
extern int *statusWorkers;

/** \brief Amount of worker threads*/
extern int threadAmount;

/** \brief Amount of files*/
extern int fileAmount;

/** \brief pointer to array of pointers, each pointer points to the name of a file*/
extern char** fileNames;

/** \brief pointer to array of pointers, each pointer points to the results of a file*/
extern int** resultados;

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief locking flag to warrant mutual exclusion when writing results*/
static pthread_mutex_t accessWR = PTHREAD_MUTEX_INITIALIZER;

/** \brief Current ID of file that is opened*/
int currentFileId = 0;

/** \brief FILE pointer to the current openend file*/
FILE* f = NULL;

/**
 * \brief Process chunks thread.
 * 
 * Thread that processes files, chunk by chunk.
 * 
 * \param workerId thread id of the worker
 */
void processChunks(unsigned int workerId){

    while(currentFileId < fileAmount)
    {
        int chunkLenght = 0;
        int chunkFileId = 0;
        int aux = 0;
        int* chunk;
        chunk = malloc(sizeof(int));
        if(!chunk)
        {
            errno = statusWorkers[workerId];                                                            
            perror ("error on allocating memory");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        // no more files to read
        if(currentFileId == fileAmount)
        {
            statusWorkers[workerId] = EXIT_SUCCESS;
            pthread_exit(&statusWorkers[workerId]);
        }

        // enter read mutex
        if ((statusWorkers[workerId] = pthread_mutex_lock (&accessCR)) != 0)                                   
        { 
            errno = statusWorkers[workerId];                                                            
            perror ("error on entering monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }
        
        // no more files to read
        if(currentFileId == fileAmount)
        {
            if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessCR)) != 0)                                  
            { 
                errno = statusWorkers[workerId];                                                            
                perror ("error on exiting monitor(CF)");
                statusWorkers[workerId] = EXIT_FAILURE;
                pthread_exit (&statusWorkers[workerId]);
            }
            statusWorkers[workerId] = EXIT_SUCCESS;
            pthread_exit(&statusWorkers[workerId]);
        }

        // open file(or next file)
        if(!f)
        {
            char* path = strcat(strdup("../datafiles/countWords/"), fileNames[currentFileId]);   
     
            f = fopen(path, "r");

            chunkFileId = currentFileId;
            if(!f)
            {
                perror("error opening file");
                statusWorkers[workerId] = EXIT_FAILURE;
                if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessCR)) != 0)                                   
                { 
                    errno = statusWorkers[workerId];                                                            
                    perror ("error on exiting monitor(CF)");
                    statusWorkers[workerId] = EXIT_FAILURE;
                    pthread_exit (&statusWorkers[workerId]);
                }
                pthread_exit (&statusWorkers[workerId]);
            }
        }

        // read chunk from file
        while(1)
        {
            if(chunkLenght >= MINIMUM_CHUNK_SIZE && isDelimiterChar(aux))       // max chunk lenght
                break;
            else if(aux == EOF)                                 // end of chunk
            {
                fclose(f);
                f = NULL;
                currentFileId++;
                break;
            }

            int readdenBytes = 0;
            aux = getchar_wrapper(f, &readdenBytes);  // get bytes of a single character and return their code
            chunkFileId = currentFileId;
            chunkLenght++;
            chunk = realloc(chunk, sizeof(int) * chunkLenght);
            *(chunk + chunkLenght - 1) = aux;
        }

        // exit read mutex
        if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessCR)) != 0)                                 
        { 
            errno = statusWorkers[workerId];                                                            
            perror ("error on exiting monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        // processes chunk
        int partialVowel = 0, partialConsonant = 0, partialWords = 0;
        chunkProcessing((int*)chunk, chunkLenght, &partialVowel, &partialConsonant, &partialWords);

        // enter write results mutex
        if((statusWorkers[workerId] = pthread_mutex_lock(&accessWR)) != 0){
            errno = statusWorkers[workerId];                                                            
            perror ("error on entering mutex to write");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        // write results
        *(*(resultados + chunkFileId) + 0) += partialVowel;
        *(*(resultados + chunkFileId) + 1) += partialConsonant;
        *(*(resultados + chunkFileId) + 2) += partialWords;

        // free memory
        free(chunk);

        // exit write results mutex
        if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessWR)) != 0)                                  
        { 
            errno = statusWorkers[workerId];                                                            
            perror ("error on exiting monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

   }

    statusWorkers[workerId] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[workerId]);
}