// #include <stdio.h>
// #include <stdlib.h>
// #include <libgen.h>
// #include <unistd.h>
// #include <string.h>

// /**
//  *  \brief Main function.
//  *
//  *  \param argc number of words of the command line
//  *  \param argv list of words of the command line
//  *  \param filename pointer tot he name of the file
//  *  \param thread_amount pointer to the quantity of threads
//  *  \return status of operation
//  */
// int processInput(int argc, char* argv[], int* thread_amount)
// {

//   // processes command line arguments
//   int opt;
//   char* file_names[argc];
//   int file_amount = 0;

//   do{

//     switch ((opt = getopt (argc, argv, "f:n:h"))) { 
//       case 'f': /* file name */
//     	        if (optarg[0] == '-'){ 
//                   fprintf (stderr, "%s: file name is missing\n", basename (argv[0]));
//                   printUsage(basename (argv[0]));
//                   return EXIT_FAILURE;
//                 }

//                 strcpy(file_names[file_amount++], optarg);
//                 break;

//       case 'n': /* numeric argument */
//                 if (atoi (optarg) <= 0){ 
//                     fprintf (stderr, "%s: non positive number\n", basename (argv[0]));
//                      printUsage (basename (argv[0]));
//                      return EXIT_FAILURE;
//                    }

//                 *thread_amount = (int)atoi(optarg);
//                 break;

//       case 'h': /* help mode */
//                 printUsage (basename (argv[0]));
//                 return EXIT_SUCCESS;

//       case '?': /* invalid option */
//                 fprintf (stderr, "%s: invalid option\n", basename (argv[0]));
//   	            printUsage (basename (argv[0]));
//                 return EXIT_FAILURE;

//       case -1:  
//         break;

//     }

//   } while (opt != -1);

//   // only has name of executable
//   if (argc == 1){ 
//       fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
//       printUsage (basename (argv[0]));
//       return EXIT_FAILURE;
//      }

//   // print arguments
//   printf ("File name(s):");
//   for(int i = 0; i < file_amount; i++)
//     printf("\t%s", file_names[i]);
//   printf("\n");
//   printf ("Numeric value = %d\n", *thread_amount);

//   for (int o = 0; o < argc; o++)
//     printf ("Word %d = %s\n", o, argv[o]);


//   return EXIT_SUCCESS;
// }


// /**
//  *  \brief Print command usage.
//  *
//  *  A message specifying how the program should be called is printed.
//  *
//  *  \param cmdName string with the name of the command
//  */
// void printUsage (char *cmdName)
// {
//   fprintf (stderr, "\nSynopsis: %s OPTIONS [filename / positive number]\n"
//            "  OPTIONS:\n"
//            "  -h      --- print this help\n"
//            "  -f      --- filename\n"
//            "  -n      --- positive number\n", cmdName);
// }


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

int processInput (int argc, char *argv[], int* thread_amount)
{
  /* process command line options */

  int opt;                                       /* selected option */
  
  int file_amount = 0;
  char** file_names = NULL; 

  opterr = 0;
  do
  { switch ((opt = getopt (argc, argv, "f:n:h")))
    { case 'f': /* file name */
    	        if (optarg[0] == '-')
                { fprintf (stderr, "%s: file name is missing\n", basename (argv[0]));
                  printUsage (basename (argv[0]));
                  return EXIT_FAILURE;
                }

                // fName = optarg;
                
                if(file_amount == 0){
                  file_names = malloc(sizeof(char*) * ++file_amount);
                  *(file_names + file_amount - 1) = optarg;
                }
                else{
                  file_amount++;
                  file_names = realloc(file_names, sizeof(char*) * file_amount);
                  *(file_names + file_amount -1) = optarg;
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
  for(int i = 0; i < file_amount; i++){
    char* nome = *(file_names + i);
    printf("\tfile: <%s>\n", nome);
  }
  printf ("Numeric value = %d\n", *thread_amount);
  for (o = 0; o < argc; o++)
    printf ("Word %d = %s\n", o, argv[o]);

  /* that's all */

  return EXIT_SUCCESS;

} /* end of main */

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
