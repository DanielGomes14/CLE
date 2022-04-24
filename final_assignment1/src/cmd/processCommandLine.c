#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>

static void printUsage(char *cmdName);

/**
 * \brief Process command line input
 * 
 * Iterates through argv to find and store thread amount, file amount and file names.
 * 
 * \param argc Argument quantity in the command line
 * \param argv Array with arguments fromt he command line
 * \param threadAmount Pointer to thread quantity
 * \param fileAmount Pointer to file amount
 * \param fileNames Pointer to pointer array where file names are stored
 * \return int Return value of command line processing
 */
int processInput (int argc, char *argv[], int* threadAmount, int* fileAmount, char*** fileNames)
{

  char** auxFileNames = NULL; 
  int opt;    // selected option

  if(argc <= 3){
    perror("No/few arguments were provided.");
    printUsage(basename("PROGRAM"));
    return EXIT_FAILURE;
  }

  opterr = 0;
  do
  { 
    switch ((opt = getopt (argc, argv, "f:n:h")))
    { 
      case 'f': /* file name */
    	        if (optarg[0] == '-')
                { 
                  fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                  printUsage(basename (argv[0]));
                  return EXIT_FAILURE;
                }

                int index = optind - 1;
                char* next = NULL;

                while(index < argc)
                {
                  next = argv[index++];     // get next element in argv

                  if(next[0] != '-')        // if element isn't an option, then its a file name
                  {    
                    if(*fileAmount == 0)   // first file name
                    {  
                      auxFileNames = malloc(sizeof(char*) * ++(*fileAmount));
                      if(!auxFileNames)   // error reallocating memory
                      {
                        fprintf(stderr, "error allocating memory for file name\n");
                        return EXIT_FAILURE;
                      }
                      *(auxFileNames + (*fileAmount) - 1) = next;
                    }
                    else                    // following file names
                    {  
                      (*fileAmount)++;
                      auxFileNames = realloc(auxFileNames, sizeof(char*) * (*fileAmount));
                      if(!auxFileNames)   // error reallocating memory
                      {
                        fprintf(stderr, "error reallocating memory for file name\n");
                        return EXIT_FAILURE;
                      }
                      *(auxFileNames + (*fileAmount) -1) = next;
                    }
                  }
                  else                      // element is something else
                    break;
                }
                break;

      case 'n': /* numeric argument */
                if (atoi(optarg) <= 0)
                   { 
                     fprintf(stderr, "%s: non positive number\n", basename (argv[0]));
                     printUsage(basename (argv[0]));
                     return EXIT_FAILURE;
                   }
                *threadAmount = (int)atoi(optarg);
                break;

      case 'h': /* help mode */
                printUsage (basename (argv[0]));
                return EXIT_SUCCESS;

      case '?': /* invalid option */
                fprintf(stderr, "%s: invalid option\n", basename (argv[0]));
  	            printUsage(basename (argv[0]));
                return EXIT_FAILURE;

      case -1:  break;
    }
  } while (opt != -1);

  // show processed information
  // file names
  printf("File names:\n");
  for(int i = 0; i < (*fileAmount); i++)
  {
    char* nome = *(auxFileNames + i);
    printf("\tfile: <%s>\n", nome);
  }

  // thread amount
  printf("Numeric value = %d\n", *threadAmount);

  // copy auxiliar pointer to fileNames pointer
  *fileNames = auxFileNames;

  return EXIT_SUCCESS;

}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 * 
 *  \param cmdName string with the name of the command
 */
static void printUsage(char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s OPTIONS [filename / positive number]\n"
           "  OPTIONS:\n"
           "  -h      --- print this help\n"
           "  -f      --- filename\n"
           "  -n      --- positive number\n", cmdName);
}
