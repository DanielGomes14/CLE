#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <wchar.h>
#include <locale.h>
#include <pthread.h>

#include <unistd.h>
#include <limits.h>

/*
 Check if character is a vowel
 \param ch: a character
*/
int isVowel(wint_t ch)
{
    // A Case
    if ((ch == 0x61 || ch == 0x41) || (ch >= 0xC0 && ch <= 0xC6) || (ch >= 0xE0 && ch <= 0xE6))
    {
        return 1;
    }
    // E Case

    if ( (ch == 0x65 || ch == 0x45 )||(ch >= 0xc8 && ch <= 0xCB) || (ch >= 0xE8 && ch <= 0xEB))
    {
        return 1;
    }
    // I Case

    if ((ch == 0x69 || ch == 0x49)|| (ch >= 0XCC && ch <= 0xCF) || (ch >= 0xEC && ch <= 0xEF))
    {
        return 1;
    }
    // O Case

    if ((ch ==0x6F || ch == 0x4F ) || (ch >= 0xD2 && ch <= 0xD6) || (ch >= 0xF2 && ch <= 0xF6))
    {
        return 1;
    }
    // U Case

    if ( (ch == 0x75 || ch == 0x55) || (ch >= 0xD9 && ch <= 0xDC) || (ch >= 0xF9 && ch <= 0xFC))
    {
        return 1;
    }
    return 0;
}

/*
 Check if character is a Delimiter,
 useful to know where a new word, line, etc occurs
 \param ch: a character
*/
int isDelimiterChar(wint_t ch)
{
    
    wint_t arr[] = {0x20, 0x09, 0x0A, 0x0D, 0x2D, 0x22, 0x5B, 0x5D, 0x28, 0x29,
                  0x2E, 0x2C, 0x3B,0x3B, 0x21, 0x3F,0xAB,0xBB,0x3A,0xE2809C, 0xE2809D, 0xE28093,
                   0xE280A6, 0x27, WEOF};
        
    for (int i = 0; i < 25; i++)
    {
        if (arr[i] == ch)
            return 1;
    }
    return 0;
}
/*
 Check if character is a Consonant
 \param ch: a character
*/
int isConsonant(wint_t ch){
    // check if char is between a and z or A and Z
    if( (ch >= 0x41 && ch <= 0x5A) || (ch >= 0x61 && ch <= 0x7A)){
        // then, if it isn't a vowel it will be a consonant
        if (!isVowel(ch)) return 1;
    }
    // Ç or ç case, which can be considered the consonant c
    else if(ch == 0xE7 || ch == 0xC7) return 1;
    return 0;
}



void printStats(int vowels_counter, int consonant_counter, int total_words)
{
    printf("\nWords Starting with Vowel %d", vowels_counter);
    printf("\nWords ending with consonant %d", consonant_counter);
    printf("\nTotal Words %d\n", total_words);
}
void read_file(char *filename)
{
    FILE *ptr;
    ptr = fopen(filename, "r");
    wint_t last = 0;
    wint_t current = 0;
    int vowels_counter = 0;
    int consonant_counter = 0;
    int total_words = 0;
    int in_word = 0;
    while (1)
    {
        current = fgetwc(ptr);
        //printf("\ncurrent %lc, last %lc", current,last);
       
        if(current == '\'' && !in_word){
            last = current;
            in_word = 1;
            continue;
        }

        // if(current == '‘'){
        //     while((current = fgetwc(ptr)) != '’'){
        //         last = current;
        //     }
        //     //test if consonant
        //     //increment counter
        //     continue;
        // }

        if (isDelimiterChar(current) && isConsonant(last) ){
            //printf("\n consonant");
            consonant_counter++;
            in_word = 0;
        }
        
        if(isDelimiterChar(current) && !isDelimiterChar(last) && !in_word){
            total_words++;
        }
        if(isDelimiterChar(last) && isVowel(current) ){
            //printf("\n vowel");
            vowels_counter++;
        }
        if(!isDelimiterChar(current) && !isVowel(current) && !isConsonant(current)){
            printf("\ndelimiter not found %d",current);
        }
        
        if(last == WEOF && current == EOF) break;
        last = current;
    }
    
    printStats(vowels_counter,consonant_counter,total_words);
}

void* read_file_thread(void* vargp){
    char* filename = (char*)vargp;
    printf("<%s>\n", filename);
    FILE* ptr = fopen(filename, "r");
    if(!ptr){
        printf("Error opening file.\nExiting...\n");
        return NULL;
    }

    wint_t last, current;
    int vowels_counter = 0, consonant_counter = 0, total_words = 0;
    int in_word = 0;

    while (1)
    {
        current = fgetwc(ptr);
        printf("\ncurrent %lc, last %lc", current,last);
        
        if(current == '\'' && !in_word){
            last = current;
            in_word = 1;
            continue;
        }

        if(current == '\'' && in_word){
            last = current;
            in_word = 0;
        }
        if (isDelimiterChar(current) && isConsonant(last) && !in_word){
            printf("\n consonant");
            in_word = 0;
            consonant_counter++;
        }
        
        if(isDelimiterChar(current) && !isDelimiterChar(last) && !in_word){
            printf("\n new word");
            total_words++;
        }
        if(isDelimiterChar(last) && isVowel(current) ){
            printf("\n vowel");
            vowels_counter++;
        }
        if(!isDelimiterChar(current) && !isVowel(current) && !isConsonant(current)){
            printf("\ndelimiter not found %d",current);
        }
        
        if(last == WEOF && current == EOF) break;
        last = current;
    }
    
    printStats(vowels_counter,consonant_counter,total_words);
    
}

int main(int argc, char *argv[])
{
    setlocale(LC_ALL, ""); 

    char cwd[PATH_MAX];
   if (getcwd(cwd, sizeof(cwd)) != NULL) {
       printf("Current working dir: %s\n", cwd);
   } else {
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
        pthread_create(&tid[counter - 1], NULL, read_file_thread, (void*)argv[counter]);
    }

    for(counter = 1; counter < argc; counter++){
        pthread_join(tid[counter-1], NULL);
    }

}
