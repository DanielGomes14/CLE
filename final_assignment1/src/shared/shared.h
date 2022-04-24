#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

#define MINIMUM_CHUNK_SIZE 50

typedef struct workerData{
    unsigned int threadId;
    int* results;
} workerData, pworkerData;

void processChunks(unsigned int workerId, int* results);

#endif