#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>

int isVowel(int ch);

int isDelimiterChar(int ch);

int isConsonant(int ch);

int isApostrophe(int ch);

int isAlphaNumeric(int ch);

void printStats(int vowels_counter, int consonant_counter, int total_words);

void processChunk(chunkInfo chunk);

int getchar_wrapper(FILE *fp);


#endif