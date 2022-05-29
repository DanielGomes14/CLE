#ifndef WORKER_H
#define WORKER_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "../utils/utils.h"

void chunkProcessing(int* chunk, int chunkLenght, int* vowel, int* consonant, int* words);

void worker(int rank);

#endif