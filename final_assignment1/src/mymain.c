#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>
#include <unistd.h>
#include <limits.h>

#include "./cmd/processCommandLine.h"

int *statusWorker;

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

    int thread_amount = 0; // number of threads;
    int file_amount = 0;
    int *status_p;         // pointer to execution status
    char ** file_names; // array with the names of the files

    // no inputs where given
    if (argc == 1)
    {
        perror("No arguments were provided.");
        exit(-1);
    }

    // processes command line information
    if (processInput(argc, argv, &thread_amount, &file_amount, &file_names))
        exit(-1);

    printf("THREAD AMOUNT: %d\n", thread_amount);
    printf("FILE AMOUNT: %d\n", file_amount);
    
    // printf("%s\n", file_names[0]);
    // for(int i = 0; i < file_amount; i++){
    //     printf("FILE NAME: %s\n", (file_names[i]));
    // }

    
    pthread_t tIdWorker[thread_amount];  //  workers internal thread id array
    unsigned int workers[thread_amount]; // workers application defined thread id array

    // initialise workers array
    for (int t_ind = 0; t_ind < thread_amount; t_ind++)
        workers[t_ind] = t_ind;

    statusWorker = malloc(sizeof(int) * thread_amount);

    for (int t = 0; t < thread_amount; t++)
    {
        //create(t)
        if (pthread_create(&tIdWorker[t], NULL, work, workers[t]) != 0) /* thread producer */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
    }
    for (int t = 0; t < thread_amount; t++)
    {
        if (pthread_join(tIdWorker[t], (void *)&status_p) != 0) /* thread producer */
        {
            perror("error on waiting for thread worker");
            exit(EXIT_FAILURE);
        }

        printf("thread worker, with id %d, has terminated: ", t);
        printf("its status was %d\n", *status_p);
    }

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

void produce(){
    /*
    
    */

}

/**
 * @brief Worker Function
 * 
 * Does the worker tasks
 * @param par pointer to application defined worker identification
 */
void *work(void * par){
    // TODO: call worker functions

    unsigned int id = *((unsigned int *) par), // worker id //
               val;  

    // logic stuff

    //mutex start
    // save value on matrix
    // mutex end

    //end work
    statusWorker[id] = EXIT_SUCCESS;
    pthread_exit (&statusWorker[id]);
}