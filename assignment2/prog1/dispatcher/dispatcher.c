#include "dispatcher.h"

/**
 * @brief Function used to read chunks from a file.
 * 
 * The file MUST be opened beforehand.
 * 
 * @param fPtr pointer to openend file
 * @param chunkSize size of the readden chunk 
 * @param chunkToProcess flag to know if there are more chunks to process
 * @return int* (returns a pointer to the chunk)
 */
int *readChunk(FILE *fPtr, int *chunkSize, int *chunkToProcess)
{
    int readdenBytes = 0, nextChar = 0;
    int *chunk = NULL;

    while(1)
    {
        if((*chunkSize) >= MIN_CHUNK_SIZE && isDelimiterChar(nextChar))
            break;
        else if(nextChar == EOF)
        {
            fclose(fPtr);
            *chunkToProcess = 0;
            break;        
        }

        nextChar = getchar_wrapper(fPtr, &readdenBytes);

        (*chunkSize) += 1;

        if(!chunk)
        {
            chunk = malloc(sizeof(int) * (*chunkSize));
            if(!chunk)
            {
                perror("error allocating memory for chunk");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            *(chunk + ((*chunkSize) - 1)) = nextChar;
        }
        else
        {
            chunk = realloc(chunk, sizeof(int) * (*chunkSize));
            if(!chunk)
            {
                perror("error allocating memory for chunk");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            *(chunk + ((*chunkSize) - 1)) = nextChar;
        }
    }

    return chunk;
}

/**
 * @brief Dispatcher logic
 * 
 * Iterates through each file, where each iteration reads a chunk and sends it 
 * to a worker process to be processed.
 * 
 * 4 "execution codes" are used to control the behavior of the Worker, so it can
 * wait for specified data that the Dispatcher sends:
 *  0 - idle
 *  1 - process chunk
 *  2 - return results
 *  3 - end process
 * 
 * At the end of each file, the results are requested to the worker processes
 * and summed to be printed.
 * 
 * @param fileNames Pointer to the array with the names of the files
 * @param fileAmount Amount of files
 * @param size Amount of processes
 * @return int* (pointer to results matrix)
 */
void dispatcher(char ***fileNames, int fileAmount, int size, int *results)
{
    char **names = *fileNames;
    int idleCode = 0, processCode = 1, returnCode = 2, endCode = 3;

    for(int i = 0; i < fileAmount; i++)
    {
        char* fileName = strdup(names[i]);
        char* dir = strdup("./files/");
        FILE* f = fopen(strcat(dir, fileName), "r");
        if(!f)
        {
            perror("error openining file\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int chunkToProcess = 1;
        int *chunk = NULL, chunkSize = 0;
        while(chunkToProcess)
        {
            for(int j = 1; j < size; j++)       // cycle to send chunks of a file
            {
                if(chunkToProcess)              // chunks to process
                {
                    chunk = readChunk(f, &chunkSize, &chunkToProcess);

                    MPI_Send(&processCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    MPI_Send(&chunkSize, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    MPI_Send(chunk, chunkSize, MPI_INT, j, 0, MPI_COMM_WORLD);
                    free(chunk);
                    chunkSize = 0;
                }
                else                            // no more chunks, but is still iterating through workers
                {
                    MPI_Send(&idleCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                }
            }

            if(!chunkToProcess)                 // no more chunks to process, send msg to workers to return partial values
            {
                for(int j = 1; j < size; j++)
                {
                    int* partialResults = malloc(sizeof(int) * 3);
                    for(int k = 0; k < 3; k++)
                        *(partialResults + k) = 0;


                    MPI_Send(&returnCode, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                    MPI_Recv(partialResults, 3, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    *((results + i * 3) + 0) += *(partialResults + 0);

                    *((results + i * 3) + 1) += *(partialResults + 1);

                    *((results + i * 3) + 2) += *(partialResults + 2);

                    free(partialResults);
                }
            }
        }
    }

    for(int i = 1; i < size; i++)
        MPI_Send(&endCode, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

    return;
}
