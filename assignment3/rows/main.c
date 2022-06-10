#include <stdlib.h>
#include <stdio.h>

#include "gauss.h"

/**
 * @brief Main method of gaussian elimination
 * 
 * The gaussian elimination is reached through a transformation of the matrix to a
 * base triangular matrix.
 * 
 * @param argc amount of arguments from command line
 * @param argv array witht he arguments from the command line
 */
void main(int argc, char** argv)
{
    FILE* f = NULL;
    f = fopen("mat128_32.bin", "rb");

    int order = 0;
    int amount = 0;
    double * matrix = NULL;

    fread(&amount, sizeof(int), 1, f);
    printf("AMOUNT: <%d>\n", order);

    fread(&order, sizeof(int), 1, f);
    printf("ORDER: <%d>\n", order);

    int l = 0;

    while(1)
    {
        // allocate memory
        matrix = malloc(sizeof(double) * order * order);
        if(!matrix)
        {
            perror("error alocating memory");
            exit(EXIT_FAILURE);
        }

        while(fread(matrix, (sizeof(double) * order * order), 1, f))
        {
            // calculate determinant
            double det = determinant(order, matrix);

            // print determinant
            printf("L: <%d>\tDeterminant: <%f>\n", ++l, det);
        }
        

    }

}