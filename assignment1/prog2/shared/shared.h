#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

#define FIFO_SIZE 5

typedef struct chunkInfo{
    int fileId;
    int matrixId;
    double* matrixPtr; 
    int order;
    int isLastChunk;
} chunkInfo, *pChunkInfo;

void storeChunk(chunkInfo info);

chunkInfo getChunk(unsigned int workerId);

void storePartialResults(int fileId, int matrixId, double determinant);

void awakeWorkers();
#endif