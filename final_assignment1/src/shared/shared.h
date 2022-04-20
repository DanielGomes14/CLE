#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

#define CHUNK_SIZE 200

typedef struct chunkInfo{
    FILE* f;
    int bufferSize;
    int fileId;
    int fileAmount;
    int** matrixPtr;   // [fileAmount][3]
} chunkInfo, *pChunkInfo;

void storeChunk(chunkInfo info);

chunkInfo getChunk(unsigned int workerId);

#endif