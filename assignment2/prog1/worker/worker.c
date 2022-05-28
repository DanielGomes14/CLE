#include "worker.h"

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

            printf("\nVOWELS: <%d>\n", *counters);
            printf("CONSONANTS: <%d>\n", *(counters + 1));
            printf("WORDS: <%d>\n", *(counters + 2));

            // free processed chunk
            free(chunk);
        }
        else if(executionCode == 2)     // send counters
        {
            // send counters
            MPI_Send(counters, 3, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            // reset counters
            *(counters + 0) = 0;
            *(counters + 1) = 0;
            *(counters + 2) = 0;
        }
        else if(executionCode == 3)     // end worker
        {
            // CHECK IF IT IS NECESSARY TO FREE MEMORY
            free(counters);
            break;
        }
    }

    return;
}