#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATH = "/datasets/"

void gaussian_elimination(int order, double matrix[order][order]){
    //transform normal matrix to equivalent triangular matrix

    float c, sum;
    float res[order];

    // loop for generation of upper triangular matrix
    for(int i = 0; i < order; i++){
        for(int j = 0; j < order; j++){
            if(j > i){
                c = matrix[i][j] / matrix[i][i];
                for(int k = 0; k < order; k++){
                    matrix[k][j] = (c * matrix[i][k]);
                }
            }
        }
    }

    // something???
    res[order] = matrix[order][order + 1] / matrix[order][order];

    // loop for backward substituition
    for(int i = order - 1; i >= 1; i--){
        sum = 0;
        
        for(int j = i + 1; j <= order; j++)
            sum = sum + matrix[i][j] * x[j];
        
        res[i] = (matrix[i][order + 1] - sum) / matrix[i][i];
    }


    float A[20][20],x[10],sum=0.0;
    double c;

    for(int j = 1; j <= order; j++) /* loop for the generation of upper triangular matrix*/
    {
        for(int i = 1; i <= order; i++)
        {
            if(i > j)
            {
                c = *(matrix + j * order + i) / *(matrix + j * order + j * order);
                for(int k = 1; k <= order + 1; k++)
                {
                    matrix[(i* k) - 1] = matrix[(i * k) -1] - (c * matrix[(j * k) -1]);
                    *(matrix + i * k - 1) = *(matrix + i * k - 1) - (c * (*(matrix + j * order * k)))
                }
            }
        }
    }

    x[n]=A[n][n+1]/A[n][n];

    /* this loop is for backward substitution*/
    for(int i = order - 1; i >= 1; i--)
    {
        sum=0;
        for(int j = i + 1; j <=n ; j++)
        {
            sum=sum+A[i][j]*x[j];
        }
        x[i]=(matrix[i][n+1]-sum)/matrix[i][i];
    }

    printf("\nThe solution is: \n");
    for(i=1; i<=n; i++)
    {
        printf("\nx%d=%f\t",i,x[i]); /* x1, x2, x3 are the required solutions*/
    }

}

int calcDeterminant(float* matrix){
    // calculates determinant of triangular matrix
    return 0;
}

int main(int argc, char* argv[]) {

    if(argc == 1){
        printf("No files were given to read.\n");
        exit(0);
    }
    else{
        printf("Files to read: <%d>", argc-1);
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
        while(fread(&matrix, sizeof(matrix), 1, file)){
            printf("\n\n");
            for(int i = 0; i < order; i++){
                for(int j = 0; j < order; j++){
                    printf("%f\t", matrix[i][j]);
                }
                printf("\n");
            }
        }

        // transform normal matrix in equivalent triangular matrix
        gaussian_elimination(order, matrix);

        // calculate determinant
        // int res = calcDeterminant(matrix);

    }

    return 0;
}
