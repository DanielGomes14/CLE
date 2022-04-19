#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

#define CHUNK_SIZE 200

typedef struct chunkInfo{
    FILE* f;
    int bufferSize;
    int fileId;
    int** matrixPtr;   // [3][fileAmount]
} chunkInfo, *pChunkInfo;

#endif