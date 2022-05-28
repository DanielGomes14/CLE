#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>

#include <unistd.h>
#include <limits.h>

/**
 * \brief check if char is a vowel, considering the Portuguese Alphabet cases
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
 * \brief useful to know where a new word, line, etc occurs
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
 * \brief check if char is a consonant, considering the Portuguese Alphabet cases
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
 * \brief verifies if the character is a an apostrophe or single quotation marks
 *
 * \param ch: the integer representation of the character
 * \return int
 */
int isApostrophe(int ch)
{
    if (ch == 0x27 || ch == 8216 || ch == 8217)
        return 1;
    return 0;
}

/**
 * \brief Verifies if the character is alphanumeric
 *
 * \param ch the integer representation of the character
 * \return int
 */
int isAlphaNumeric(int ch)
{
    return ((ch >= 48 && ch <= 57) || isVowel(ch) || isConsonant(ch));
}

/**
 * \brief Get the unicode integer value of a character
 * Firstly, it's checked how many bytes does the character codepoint requires.
 * According to the number of bytes needed, more character may be needed to be
 * read from the text to compose the character. Example: "é" requires two bytes.
 *
 * \param fp a pointer to the file
 * \return int - amount of readden bytes
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
