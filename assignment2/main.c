#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PATH = "/datasets/"


double determinant( int order, double matrix[order][order]) {
   
    double det = 1;

    for (int i = 0; i < order; ++i) {

        double pivotElement = matrix[i][i];
        int pivotRow = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int row = i + 1; row < order; ++row) {
            if (fabs(matrix[row][i]) > fabs(pivotElement)) {
                // update the value of the pivot and pivot row index
                pivotElement = matrix[row][i];
                pivotRow = row;
            }
        }
        //if the diagonal element is zero then the determinant will be zeero
        if (pivotElement == 0.0) {
            return 0.0;
        }
        //if the pivotELement is not in the current row, then we perform a swap in the columns
        if (pivotRow != i) {
            for (int k = 0; k < order; k++){
                double temp;
                temp = matrix[i][k];
                matrix[i][k] = matrix[pivotRow][k];
                matrix[pivotRow][k] = temp;
            }
            //signal the row swapping
            det *= -1.0;
        }
        //update the determinant with the the diagonal value of the current row
        det *= pivotElement;
        // reduce the matrix to a  Upper Triangle Matrix
        for (int row = i + 1; row < order; ++row) {
            for (int col = i + 1; col < order; ++col) {
                matrix[row][col] -= matrix[row][i] * matrix[i][col] / pivotElement;
            }
        }
    }

    return det;
}


int main(int argc, char* argv[]) {

    if(argc == 1){
        printf("No files were given to read.\n");
        exit(0);
    }
    else{
        printf("Files to read: <%d>\n", argc-1);
    }

    for(int i = 1; i < argc; i++){
        FILE* file;
        file = fopen(argv[i], "rb");
        if(!file){
            printf("Error opening file to read.\n Ending...");
            exit(1);
        }

        //  reads amount of matrices to read
        int amount = 0;
        fread(&amount, sizeof(int), 1, file);

        // reads order os matrices
        int order = 0;
        fread(&order, sizeof(int), 1, file);

        // reads values of matrix
        double matrix[order][order];
        int j = 0;

        printf("---------------------File <%s>---------------------\n", argv[i]);
        printf("Amount of matrixes: <%d>\tOrder of matrixes: <%d>\n", amount, order);
        while(fread(&matrix, sizeof(matrix), 1, file)){  // reads matrix from .bin file
            // calculates determinant, gaussian elimination is applied inside
            printf("\tMatrix nº: <%d>\tDeterminant: <%f>\n", j + 1, determinant(order, matrix));
            j++;
        }
        printf("\n\n");

    }

    return 0;
}
