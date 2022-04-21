#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>

/* allusion to internal functions */

static void printUsage (char *cmdName);

/**
 *  \brief Main function.
 *
 *  \param argc number of words of the command line
 * 
 *  \param argv list of words of the command line
 *
 *  \return status of operation
 */

int processInput (int argc, char *argv[], int* thread_amount, int* file_amount, char*** file_names)
{
  /* process command line options */

  int opt;                                       /* selected option */
  
  // int file_amount = 0;
  char** aux_file_names = NULL; 

  opterr = 0;
  do
  { switch ((opt = getopt (argc, argv, "f:n:h")))
    { case 'f': /* file name */
    	        if (optarg[0] == '-')
                { fprintf (stderr, "%s: file name is missing\n", basename (argv[0]));
                  printUsage (basename (argv[0]));
                  return EXIT_FAILURE;
                }

                int index = optind - 1;
                char* next = NULL;

                while(index < argc){ 
                  next = argv[index++];     // get next element in argv

                  if(next[0] != '-'){  // if element isn't an option, then its a file name
                    
                    if(*file_amount == 0){  // first file name
                      aux_file_names = malloc(sizeof(char*) * ++(*file_amount));
                      *(aux_file_names + (*file_amount) - 1) = next;
                    }
                    else{  // following file names
                      (*file_amount)++;
                      aux_file_names = realloc(aux_file_names, sizeof(char*) * (*file_amount));
                      *(aux_file_names + (*file_amount) -1) = next;
                    }

                  }
                  else  // element is an option
                    break;

                }
                break;
      case 'n': /* numeric argument */
                if (atoi (optarg) <= 0)
                   { fprintf (stderr, "%s: non positive number\n", basename (argv[0]));
                     printUsage (basename (argv[0]));
                     return EXIT_FAILURE;
                   }
                *thread_amount = (int) atoi (optarg);
                break;
      case 'h': /* help mode */
                printUsage (basename (argv[0]));
                return EXIT_SUCCESS;
      case '?': /* invalid option */
                fprintf (stderr, "%s: invalid option\n", basename (argv[0]));
  	            printUsage (basename (argv[0]));
                return EXIT_FAILURE;
      case -1:  break;
    }
  } while (opt != -1);
  if (argc == 1)
     { fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
       printUsage (basename (argv[0]));
       return EXIT_FAILURE;
     }

  int o;                                          /* counting variable */

  // printf ("File name = %s\n", fName);
  printf("File names:\n");
  for(int i = 0; i < (*file_amount); i++){
    char* nome = *(aux_file_names + i);
    printf("\tfile: <%s>\n", nome);
  }
  printf ("Numeric value = %d\n", *thread_amount);
  for (o = 0; o < argc; o++)
    printf ("Word %d = %s\n", o, argv[o]);

  /* that's all */

  *file_names = aux_file_names;

  return EXIT_SUCCESS;

}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */

static void printUsage (char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s OPTIONS [filename / positive number]\n"
           "  OPTIONS:\n"
           "  -h      --- print this help\n"
           "  -f      --- filename\n"
           "  -n      --- positive number\n", cmdName);
}
