#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <locale.h>
#include <pthread.h>

#include <unistd.h>
#include <limits.h>

/**
 * @brief check if char is a vowel, considering the Portuguese Alphabet cases
 *        Example: á, À ....
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
 * \param ch: a int representation of a character
 * \return 1 if the char is a valid Vowel or 0 if not.
 */
int isConsonant(int ch)
{
    // check if char is between a and z or A and Z
    if ((ch >= 0x41 && ch <= 0x5A) || (ch >= 0x61 && ch <= 0x7A))
    {
        // then, if it isn't a vowel it will be a consonant
        if (!isVowel(ch))
            return 1;
    }
    // Ç or ç case, which can be considered the consonant c
    else if (ch == 0xE7 || ch == 0xC7)
        return 1;
    return 0;
}

int isApostrophe(int ch)
{
    if (ch == 0x27 || ch == 0x60 || ch == 8216 || ch == 8217)
        return 1;
    return 0;
}

void printStats(int vowels_counter, int consonant_counter, int total_words)
{
    printf("\nWords Starting with Vowel %d", vowels_counter);
    printf("\nWords ending with consonant %d", consonant_counter);
    printf("\nTotal Words %d\n", total_words);
}

int getchar_wrapper(FILE *fp)
{
    int c = fgetc(fp);
    if (c == EOF)
        return EOF;
    if (!(c & 0x80))
    {
        // 1 byte :  0xxx xxxx
        return c;
    }
    if ((c & 0xC0))
    {
        // 2 bytes : 110x xxxx
        return ((c & 0x1F) << 6) + (fgetc(fp) & 0x3F);
    }
    if ((c & 0xE0))
    {
        // 3 bytes : 1110 xxxx
        return ((c & 0x0F) << 12) + ((fgetc(fp) & 0x3F) << 6) + (fgetc(fp) & 0x3F);
    }
    if ((c & 0xF0))
    {
        // 4 bytes : 1111 0xxx
        return ((c & 0x07) << 18) + ((fgetc(fp) & 0x3F) << 12) + ((fgetc(fp) & 0x3F) << 6) + (fgetc(fp) & 0x3F);
    }
    return 0;
}

void *read_file_thread(void *vargp)
{
    char *filename = (char *)vargp;
    int current;
    int vowels_counter = 0, consonant_counter = 0, total_words = 0;
    int in_word = 0;
    int lastCharConsonant = 0;
    int c;
    printf("<%s>\n", filename);
    FILE *ptr = fopen(filename, "r");

    if (!ptr)
    {
        printf("Error opening file.\nExiting...\n");
        return NULL;
    }

    while (1)
    {
        current = getchar_wrapper(ptr);
        // printf("\ncurrent %lc", current);

        if (current == EOF)
        {
            break;
        }
        if (!in_word)
        {
            if (isDelimiterChar(current) || isApostrophe(current))
            {
                continue;
            }
            //TODO: Change to only increment counter if it is alphanumeric
            in_word = 1;
            total_words++;
            // printf("\nnew word");
            if (isVowel(current))
            {
                vowels_counter++;
            }
            lastCharConsonant = isConsonant(current);
        }

        if (in_word)
        {
            if (isVowel(current) || isConsonant(current))
            {
                lastCharConsonant = isConsonant(current);
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

    printStats(vowels_counter, consonant_counter, total_words);

    pthread_exit(0);
}

int main(int argc, char *argv[])
{

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL)
    {
        printf("Current working dir: %s\n", cwd);
    }
    else
    {
        perror("getcwd() error");
        return 1;
    }

    int counter;
    pthread_t tid[argc - 1];

    if (argc == 1)
    {
        printf("No arguments provided..");
    }

    for (counter = 1; counter < argc; counter++)
    {
        // read_file(argv[counter]);
        pthread_create(&tid[counter - 1], NULL, read_file_thread, (void *)argv[counter]);
    }

    for (counter = 1; counter < argc; counter++)
    {
        pthread_join(tid[counter - 1], NULL);
    }
}
