#include "worker.h"

/**
 * @brief Function used to process a chunk.
 * 
 * @param chunk pointer to the chunk to be processed
 * @param chunkLenght size of the chunk
 * @param vowel pointer to vowels counter
 * @param consonant pointer to consonants counter
 * @param words pointer to words counter
 */
void chunkProcessing(int* chunk, int chunkLenght, int* vowel, int* consonant, int* words){

    int current = 0, lastCharConsonant;
    int inWord = 0;

    int i = 0;
    while(i < chunkLenght)
    {
        // gets current char
        current = *(chunk + i++);

        if(current == 0 || current == EOF)
            break;

        if(!inWord)
        {
            if(isDelimiterChar(current) || isApostrophe(current))
                continue;

            if(isAlphaNumeric(current) || current == 95)
            {
                inWord = 1;
                (*words)++;
                if(isVowel(current))
                    (*vowel)++;
                lastCharConsonant = isConsonant(current);
            }
        }

        if(inWord)
        {
            if(isAlphaNumeric(current) || current == 95 || isApostrophe(current))
            {
                lastCharConsonant = isConsonant(current);
                continue;
            }
            else if(isDelimiterChar(current))
            {
                inWord = 0;
                if(lastCharConsonant)
                    (*consonant)++;
            }
        }
    }
}

/**
 * @brief Worker logic
 * 
 * It waits for a specified "execution code" from the Dispatcher.
 * According to the respective code, it follows a certain behavior:
 *  0 - idle
 *  1 - process chunk
 *  2 - return results
 *  3 - end process
 * 
 * @param rank rank of the worker process
 */
void worker(int rank)
{
    /*
        while(1)
            recv code
            if  0
                    continue
            if  1
                    recv chunk size
                    recv chunk
                    process chunk
                    save partial results
            if  2
                    send partial results
                    reset counters
            if  3
                    break
    */

    int *chunk = NULL, *counters = NULL;
    int executionCode = 1, chunkSize = 0;

    counters = malloc(sizeof(int) * 3); // words, vowels, consonants
    if(!counters)
    {
        perror("error allocating memory for counters");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for(int i = 0; i < 3; i++)
        *(counters + i) = 0;

    while(1)
    {
        // receive boolean
        MPI_Recv(&executionCode, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(executionCode == 0)          // idle
        {
            executionCode = -1;
            continue;
        }
        else if(executionCode == 1)     // recv chunk size and chunk and process chunk
        {
            // recv chunk size
            MPI_Recv(&chunkSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // recv chunk
            chunk = malloc(sizeof(int) * chunkSize);
            for(int i = 0; i < chunkSize; i++)
                *(chunk + i) = '\0';
            MPI_Recv(chunk, chunkSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // process chunk
            chunkProcessing(chunk, chunkSize, counters, counters + 1, counters + 2);

            // free processed chunk
            free(chunk);

            executionCode = -1;
        }
        else if(executionCode == 2)     // send counters
        {
            // send counters
            MPI_Send(counters, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            // reset counters
            *(counters + 0) = 0;
            *(counters + 1) = 0;
            *(counters + 2) = 0;

            executionCode = -1;
        }
        else if(executionCode == 3)     // end worker
        {
            free(counters);
            break;
        }
    }

    return;
}