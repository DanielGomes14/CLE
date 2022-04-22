#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include "shared.h"
#include "../utils/utils.h"

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

/** \brief locking flag which warrants mutual exclusion inside the monitor */
static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief locking flag to warrant mutual exclusion when writing results*/
static pthread_mutex_t accessWR = PTHREAD_MUTEX_INITIALIZER;


void showFileNames(){
    printf("NOMES DOS FICHEIROS:\n");
    for(int i = 0; i < fileAmount; i++){
        printf("\t%s\n", fileNames[i]);
    }
}

int currentFileId = 0;
FILE* f = NULL;
/**
 * @brief Process chunks thread.
 * 
 * Thread that processes files, chunk by chunk.
 * 
 * @param workerId thread id of the worker
 * @param results pointer to results matrix
 */
void processChunks(unsigned int workerId, int* results){
    /*
    while(nextFileId != fileAmount)
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
   //TODO: change chunk dynamic array, size of chunk = [200 + 1] -> 1 = '\0'
   // \0 ajuda depois a obter informação do chunk pra processamento, como se fosse EOF
    while(currentFileId < fileAmount){
        int chunk[200] = {0};    // struct to save chunk
        int aux = 0;                 // aux variable
        int chunkLenght = 0;
        int chunkFileId = 0;

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
                pthread_exit (&statusWorkers[workerId]);
            }
        }

        // read chunk from file
        while(1)
        {

            if(chunkLenght >= 200 && aux ==)      // max chunk lenght
                break;
            else if(aux == EOF)         // end of file
            {
                fclose(f);
                f = NULL;
                currentFileId++;
                break;
            }

            aux = fgetc(f);
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

        // processes chunk
        int partialVowel = 0, partialConsonat = 0, partialWords = 0;
        processChunk2((int*)chunk, chunkLenght, &partialVowel, &partialConsonat, &partialWords);

        // enter write results mutex
        if((statusWorkers[workerId] = pthread_mutex_lock(&accessWR)) != 0){
            errno = statusWorkers[workerId];                                                            /* save error in errno */
            perror ("error on entering mutex to write");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

        // write and show results
        printf("CHUNK FILE ID: <%d>\n", chunkFileId);
        printf("WORKER ID: <%d>\n", statusWorkers[workerId]);

        printf("VOWEL: <%d>", *(results + (3 * chunkFileId) + 0));
        *(results + (3 * chunkFileId) + 0) += 1;
        printf("\t<%d>\n", *(results + (3 * chunkFileId) + 0));

        printf("CONSONANT: <%d>", *(results + (3 * chunkFileId) + 1));
        *(results + (3 * chunkFileId) + 1) += 1;
        printf("\t<%d>\n", *(results + (3 * chunkFileId) + 1));

        printf("WORDS: <%d>", *(results + (3 * chunkFileId) + 2));
        *(results + (3 * chunkFileId) + 2) += 1;
        printf("\t<%d>\n", *(results + (3 * chunkFileId) + 2));

        // exit write results mutex
        if ((statusWorkers[workerId] = pthread_mutex_unlock (&accessWR)) != 0)                                   /* exit monitor */
        { 
            printf("here...\n");
            errno = statusWorkers[workerId];                                                             /* save error in errno */
            perror ("error on exiting monitor(CF)");
            statusWorkers[workerId] = EXIT_FAILURE;
            pthread_exit (&statusWorkers[workerId]);
        }

   }

    statusWorkers[workerId] = EXIT_SUCCESS;
    pthread_exit(&statusWorkers[workerId]);
}