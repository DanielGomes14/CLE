#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>

#include <unistd.h>
#include <limits.h>

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
int main(int argc, char *argv[]){

    // process command line information
    if(argc == 1){
        perror("No arguments were provided.");
        exit(-1);
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