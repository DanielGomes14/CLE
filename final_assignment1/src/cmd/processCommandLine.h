#ifndef PROCESSCOMMANDLINE_H
#define PROCESSCOMMANDLINE_H

static void printUsage(char* cmdName);

int processInput(int agrc, char* argv[], int* thread_amount, int* real_file_amount, char*** file_names);

#endif