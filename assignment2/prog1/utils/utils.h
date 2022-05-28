#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>

int isVowel(int ch);

int isDelimiterChar(int ch);

int isConsonant(int ch);

int isApostrophe(int ch);

int isAlphaNumeric(int ch);

int getchar_wrapper(FILE *fp, int* totalBytesRead);

#endif