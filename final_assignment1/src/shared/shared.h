#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

typedef struct chunkInfo{
    FILE* f;
    int buffer_size;
} chunkInfo, *pChunkInfo;

#endif