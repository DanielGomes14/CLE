#ifndef PROCESSCOMMANDLINE_CUH
#define PROCESSCOMMANDLINE_CUH

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <stdio.h>

int processInput(int argc, char* argv[], int* fileAmount, char*** fileNames);

#endif