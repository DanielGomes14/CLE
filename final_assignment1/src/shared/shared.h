#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

#define CHUNK_SIZE 200


typedef struct threadData{
   unsigned int  thread_id;
   int fileAmount;
   char ***fileNames;
   int **results;
} threadData, *pthreadData;

typedef struct workerData{
    unsigned int threadId;
    int* results;
} workerData, pworkerData;

typedef struct chunkInfo{
    FILE* f;
    int bufferSize;
    int fileId;
    int fileAmount;
    int** matrixPtr;   // [fileAmount][3]
} chunkInfo, *pChunkInfo;

void storeChunk(chunkInfo info);

chunkInfo getChunk(unsigned int workerId);

void processChunks(unsigned int workerId, int* results);

#endif