#ifndef DISPATCHER_H
#define DISPATCHER_H

#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <stdio.h>
#include <stdio.h>
#include <mpi.h>
#include "../utils/utils.h"

#define MIN_CHUNK_SIZE 200

int *readChunk(FILE **f, int *chunkSize, int *chunkToProcess);

void dispatcher(char ***fileNames, int fileAmount, int size, int* results);

#endif