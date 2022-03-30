#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PATH = "/datasets/"

/**
 * @brief Calculates the determinant of a square matrix using Gaussian Elimination
 * 
 * @param order the order of a determinant
 * @param matrix the matrix with size of "order"
 * @return double 
 */
double determinant(int order, double matrix[order][order])
{

    double det = 1;
    double pivotElement;
    int pivotRow;
    for (int i = 0; i < order; ++i)
    {

        pivotElement = matrix[i][i]; // current diagonal element
        pivotRow = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int row = i + 1; row < order; ++row)
        {
            if (fabs(matrix[row][i]) > fabs(pivotElement))
            {
                // update the value of the pivot and pivot row index
                pivotElement = matrix[row][i];
                pivotRow = row;
            }
        }
        //if the diagonal element is zero then the determinant will be zeero
        if (pivotElement == 0.0)
        {
            return 0.0;
        }

        if (pivotRow != i) // if the pivotELement is not in the current row, then we perform a swap in the columns
        {
            for (int k = 0; k < order; k++)
            {
                double temp;
                temp = matrix[i][k];
                matrix[i][k] = matrix[pivotRow][k];
                matrix[pivotRow][k] = temp;
            }

            det *= -1.0; //signal the row swapping
        }

        det *= pivotElement; //update the determinant with the the diagonal value of the current row

        for (int row = i + 1; row < order; ++row) /* reduce the matrix to a  Upper Triangle Matrix */
        // as the current row and column "i" will no longer be used, we may start reducing on the next
        // row/column (i+1)
        {
            for (int col = i + 1; col < order; ++col)
            {
               
                matrix[row][col] -= matrix[row][i] * matrix[i][col] / pivotElement;  //reduce the value
            }
        }
    }

    return det;
}

int main(int argc, char *argv[])
{
    int j = 0;
    double t0, t1, t2; /* time limits */
    int amount;        // amount of matrices to read
    int order;         // order of the matrices

    if (argc == 1)
    {
        printf("No files were given to read.\n");
        exit(0);
    }
    else
    {
        printf("Files to read: <%d>\n", argc - 1);
    }

    t2 = 0.0;
    for (int i = 1; i < argc; i++)
    {
        FILE *file;
        file = fopen(argv[i], "rb");
        if (!file)
        {
            printf("Error opening file to read.\n Ending...");
            exit(1);
        }

        amount = 0;
        if (!fread(&amount, sizeof(int), 1, file))
        {
            printf("Error reading amount. Exiting...");
            exit(-1);
        }

        // reads order os matrices
        order = 0;
        if (!fread(&order, sizeof(int), 1, file))
        {
            printf("Error reding order. Exiting...");
            exit(-1);
        }

        // reads values of matrix
        double matrix[order][order];

        printf("---------------------File <%s>---------------------\n", argv[i]);
        printf("Amount of matrixes: <%d>\tOrder of matrixes: <%d>\n", amount, order);
        while (fread(&matrix, sizeof(matrix), 1, file))
        {
            t0 = ((double)clock()) / CLOCKS_PER_SEC;
            // reads matrix from .bin file
            // calculates determinant, gaussian elimination is applied inside
            printf("\t..Matrix nº: <%d>\tDeterminant: <%.3f>\n", j + 1, determinant(order, matrix));
            j++;
            t1 = ((double)clock()) / CLOCKS_PER_SEC;
            t2 += t1 - t0;
        }
        printf("\nElapsed time = %.6f s\n", t2);
    }

    return 0;
}
