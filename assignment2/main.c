#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PATH = "/datasets/"

int gaussian_elimination(int order, double matrix[order][order]){
    // applies gaussian elimination to the given matrix

    int swap_count = 0;

    // for(int i = 0; i < order; i++){
    //     for(int j = 0; j < order; j++){
    //         printf("%f\t", matrix[i][j]);
    //     }
    //     printf("\n");
    // }
    // getchar();

    // upper triangular transformation
    for(int i = 0; i < order; i++){
        for(int j = i + 1; j < order; j++){
            if(fabs(matrix[i][i]) < fabs(matrix[j][i])){
                swap_count++;
                for(int k = 0; k < order; k++){
                    double temp;
                    temp = matrix[i][k];
                    matrix[i][k] = matrix[j][k];
                    matrix[j][k] = temp;
                }
            }
        }

        // gauss elimination
        for(int j = 0; j < order; j++){
            double term = matrix[j][i] / matrix[i][i];
            if(matrix[i][i] == 0)
            printf("ZEROOOOOOOOOOOOOOOOOOOOOOOOO\n");
            for(int k = 0; k < order; k++){
                matrix[j][k] = matrix[j][k] - (term * matrix[i][k]);
            }
        }

        // for(int i = 0; i < order; i++){
        //     for(int j = 0; j < order; j++){
        //         printf("%f\t", matrix[i][j]);
        //     }
        //     printf("\n");
        // }
        // getchar();

    }

    return swap_count;
}

int determinant(int order, double matrix[order][order]){
    // calculates determinant of triangular matrix

    double det = 1;
    int swap_count = gaussian_elimination(order, matrix);

    for(int i = 0; i < order; i++)
        det = det * matrix[i][i];

    return det * pow(-1, swap_count);
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
            printf("\tMatrix nº: <%d>\tDeterminant: <%d>\n", j + 1, determinant(order, matrix));
            j++;
        }
        printf("\n\n");

    }

    return 0;
}
