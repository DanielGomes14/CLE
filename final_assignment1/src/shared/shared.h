#ifndef SHARED_H
#define SHARED_H

#include <stdio.h>

typedef struct chunkInfo{
    FILE* f;
    int start_offset;
    int end_offset;
} chunkInfo, *pChunkInfo;

#endif