#include <stdio.h>

#ifndef DISPATCHER_H
#define DISPATCHER_H

typedef struct chunkInfo{
    int fileId;
    int matrixId;
    double* matrixPtr; 
    int order;
    int isLastChunk;
} chunkInfo, *pChunkInfo;

void storePartialResult(double **results, int fileId, int matrixId, double determinant);


void printResults(double **results,int fileAmount);

#endif