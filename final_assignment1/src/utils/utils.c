#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>

#include <unistd.h>
#include <limits.h>

#include "../shared/shared.h"

/**
 * @brief check if char is a vowel, considering the Portuguese Alphabet cases
 *        Example: á, À ....
 *
 * \param ch: a int representation of a  character
 * \return 1 if the char is a valid Vowel or 0 if not.
 */
int isVowel(int ch)
{
    // A Case
    if ((ch == 0x61 || ch == 0x41) || (ch >= 0xC0 && ch <= 0xC6) || (ch >= 0xE0 && ch <= 0xE6))
    {
        return 1;
    }
    // E Case

    if ((ch == 0x65 || ch == 0x45) || (ch >= 0xc8 && ch <= 0xCB) || (ch >= 0xE8 && ch <= 0xEB))
    {
        return 1;
    }
    // I Case

    if ((ch == 0x69 || ch == 0x49) || (ch >= 0XCC && ch <= 0xCF) || (ch >= 0xEC && ch <= 0xEF))
    {
        return 1;
    }
    // O Case

    if ((ch == 0x6F || ch == 0x4F) || (ch >= 0xD2 && ch <= 0xD6) || (ch >= 0xF2 && ch <= 0xF6))
    {
        return 1;
    }
    // U Case

    if ((ch == 0x75 || ch == 0x55) || (ch >= 0xD9 && ch <= 0xDC) || (ch >= 0xF9 && ch <= 0xFC))
    {
        return 1;
    }
    return 0;
}

/**
 * @brief useful to know where a new word, line, etc occurs
 *
 * \param ch: a character
 * \return 1 if the char is a valid Delimiter or 0 if not.
 */

int isDelimiterChar(int ch)
{

    int arr[] = {0x20, 0x09, 0x0A, 0x0D, 0x2D, 0x22, 0x5B, 0x5D, 0x28, 0x29,
                 0x2E, 0x2C, 0x3B, 0x3B, 0x21, 0x3F, 0xAB, 0xBB, 0x3A, 8230, 8211, 8212, 8220, 8221};

    for (int i = 0; i < 24; i++)
    {

        if (arr[i] == ch)
            return 1;
    }
    return 0;
}

/**
 * @brief check if char is a consonant, considering the Portuguese Alphabet cases
 *     .. Example: Ç, ç
 *
 * \param ch: a int representation of a character
 * \return 1 if the char is a valid Vowel or 0 if not.
 */
int isConsonant(int ch)
{

    if ((ch >= 0x41 && ch <= 0x5A) || (ch >= 0x61 && ch <= 0x7A)) // check if char is between a and z or A and Z //
    {

        if (!isVowel(ch)) // then, if it isn't a vowel it will be a consonant //
            return 1;
    }

    else if (ch == 0xE7 || ch == 0xC7) // Ç or ç case, which can be considered the consonant c //
        return 1;
    return 0;
}

/**
 * @brief verifies if the character is a an apostrophe or single quotation marks
 *
 * @param ch: the integer representation of the character
 * @return int
 */
int isApostrophe(int ch)
{
    if (ch == 0x27 || ch == 8216 || ch == 8217)
        return 1;
    return 0;
}

/**
 * @brief Verifies if the character is alphanumeric
 *
 * @param ch the integer representation of the character
 * @return int
 */
int isAlphaNumeric(int ch)
{
    return ((ch >= 48 && ch <= 57) || isVowel(ch) || isConsonant(ch));
}

/**
 * @brief Prints to stdout the gathered information.
 *
 * @param vowels_counter
 * @param consonant_counter
 * @param total_words
 */
void printStats(int vowels_counter, int consonant_counter, int total_words)
{
    printf("\nWords Starting with Vowel %d", vowels_counter);
    printf("\nWords ending with consonant %d", consonant_counter);
    printf("\nTotal Words %d\n", total_words);
}

/**
 * @brief Get the unicode integer value of a character
 * Firstly, it's checked how many bytes does the character codepoint requires.
 * According to the number of bytes needed, more character may be needed to be
 * read from the text to compose the character. Example: "é" requires two bytes.
 *
 * @param fp a pointer to the file
 * @return int
 */
int getchar_wrapper(FILE *fp, int *totalBytesRead)
{
    int c = fgetc(fp);

    if (c == EOF)
        return EOF;
    if ((c & 0x80) == 0)
    {
        // 1 byte :  0xxx xxxx
        *totalBytesRead = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0)
    {
        // 2 bytes : 110x xxxx
        *totalBytesRead = 2;
        return ((c & 0x1F) << 6) | (fgetc(fp) & 0x3F);
    }
    if ((c & 0xF0) == 0xE0)
    {
        // 3 bytes : 1110 xxxx
        *totalBytesRead = 3;
        return ((c & 0x0F) << 12) | ((fgetc(fp) & 0x3F) << 6) | (fgetc(fp) & 0x3F);
    }
    if ((c & 0xF8) == 0xF0)
    {
        // 4 bytes : 1111 0xxx
        *totalBytesRead = 4;
        return ((c & 0x07) << 18) | ((fgetc(fp) & 0x3F) << 12) | ((fgetc(fp) & 0x3F) << 6) | (fgetc(fp) & 0x3F);
    }
    return 0;
}

int getchar_wrapper2(int* ptr, int *readdenBytes)
{
    int c = *(ptr + (*readdenBytes));

    if (c == EOF)
    {
        // printf("\tEOF\n");
        *readdenBytes += 1;
        return EOF;
    }

    if ((c & 0x80) == 0)        // 1 byte : 0xxx xxxx
    {
        *readdenBytes += 1;
        return c;
    }

    if (((c & 0xE0) == 0xC0))   // 2 bytes : 110x xxxx
    {
        *readdenBytes += 2;
        return ((c & 0x1F) << 6) | (*(ptr + 1) & 0x3F);
    }

    if ((c & 0xF0) == 0xE0)     // 3 bytes : 1110 xxxx
    {
        *readdenBytes += 3;
        return ((c & 0x0F) << 12) | ((*(ptr + 1) & 0x3F) << 6) | (*(ptr + 2) & 0x3F);
    }

    if ((c & 0xF8) == 0xF0)     // 4 bytes : 1111 0xxx
    {
        *readdenBytes += 4;
        return ((c & 0x07) << 18) | ((*(ptr + 1) & 0x3F) << 12) | ((*(ptr + 2) & 0x3F) << 6) | (*(ptr + 3) & 0x3F);
    }

    return 0;
}

void processChunk1(chunkInfo chunk)
{
    int totalRead = 0, in_word = 0;
    int vowels_counter = 0, consonant_counter = 0, total_words = 0;
    int lastCharConsonant = 0;
    int current;
    int totalBytesRead = 0;    // num of bytes read for a char.
    int shouldProcessMore = 0; // flag used to check if should read more bytes than the specified threshold
    while (1)
    {
        current = getchar_wrapper(chunk.f, &totalBytesRead);
        if (isDelimiterChar(current))
        {
            break;
        }
    }

    while (totalRead < chunk.bufferSize || shouldProcessMore)
    {
        current = getchar_wrapper(chunk.f, &totalBytesRead);

        if (current == EOF)
        {
            break;
        }
        if (totalRead + totalBytesRead >= chunk.bufferSize)
        {
            if (isAlphaNumeric(current))
            {
                shouldProcessMore = 1;
            }
            else if (isDelimiterChar(current))
            {
                shouldProcessMore = 0;
                break;
            }
        }
        totalRead+=totalBytesRead;

        if (!in_word)
        {
            if (isDelimiterChar(current) || isApostrophe(current))
            {
                continue;
            }

            if (isAlphaNumeric(current) || current == 95)
            {

                in_word = 1;
                total_words++;
                if (isVowel(current))
                {
                    vowels_counter++;
                }
                lastCharConsonant = isConsonant(current);
            }
        }

        if (in_word)
        {
            if (isAlphaNumeric(current) || current == 95 || isApostrophe(current))
            {
                lastCharConsonant = isConsonant(current);
                if (lastCharConsonant)
                {
                    // printf("\nlast consonant %d", current);
                }
                continue;
            }
            else if (isDelimiterChar(current))
            {
                in_word = 0;
                if (lastCharConsonant)
                {
                    consonant_counter++;
                }
            }
        }
    }

    // writes results in struct, row by row
    chunk.matrixPtr[chunk.fileId][0] += consonant_counter;
    chunk.matrixPtr[chunk.fileId][1] += vowels_counter;
    chunk.matrixPtr[chunk.fileId][2] += total_words;
}

// TODO: fazer lógica de processamento do chunk
void processChunk2(int* chunk, int chunkLenght, int* vowel, int* consonant, int* words){

    int current = 0, lastCharConsonant;
    int inWord = 0, indx = 0, readdenBytes = 0;

    // printf("length: <%d>\n", chunkLenght);

    int i = 0;
    int aux = current;
    while(i <= chunkLenght)
    {
        // gets current char
        // current = getchar_wrapper2(chunk, &readdenBytes);
        aux = current;
        current = *(chunk + i++);

        // printf("CURRENT: <%c><%d>\n", current, current);

        if(current == 0 || current == EOF)
        {
            // printf("\tREADDEN BYTES: <%d>\n", readdenBytes);
            break;
        }

        if(!inWord)
        {
            if(isDelimiterChar(current) || isApostrophe(current))
            {
                continue;
            }
            if(isAlphaNumeric(current) || current == 95)
            {
                inWord = 1;
                (*words)++;
                if(isVowel(current))
                    (*vowel)++;
                lastCharConsonant = isConsonant(current);
            }
        }

        if(inWord)
        {
            if(isAlphaNumeric(current) || current == 95 || isApostrophe(current))
            {
                lastCharConsonant = isConsonant(current);
                // printf("last char <%c><%d>\n", current, current);
                continue;
            }
            else if(isDelimiterChar(current))
            {
                inWord = 0;
                if(lastCharConsonant)
                {
                    // printf("new consonant\n");
                    printf("AUX <%c><%d>\n", aux, aux);
                    printf("AUX <%c><%d>\tCONSONANT last char <%c><%d>\n", aux, aux, current, current);
                    (*consonant)++;
                }
            }
        }
        

    }

}