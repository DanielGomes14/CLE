#include "utils.cuh"

/**
 * @brief Reads all the matrixes from a give file.
 * 
 * @param fileName name of the file
 * @param matrixArray pointer to array of matrixes
 * @param order order of the matrixes
 * @param amount total amount of matrixes
 */
void readData(char *fileName, double **matrixArray, int *order, int *amount)
{
    FILE *f = fopen(strcat(strdup("./files/"), fileName), "rb");
    if(!f)
    {
        perror("error opening file");
        exit(EXIT_FAILURE);
    }

    if(!fread(amount, sizeof(int), 1, f))
    {
        perror("error reading amount of matrixes");
        exit(EXIT_FAILURE);
    }

    if(!fread(order, sizeof(int), 1, f))
    {
        perror("error reading order of matrixes");
        exit(EXIT_FAILURE);
    }

    (*matrixArray) = (double*)malloc(sizeof(double) * (*amount) * (*order) * (*order));
    if(!(*matrixArray))
    {
        perror("error allocating memory for matrixes");
        exit(EXIT_FAILURE);
    }

    if(!fread((*matrixArray), sizeof(double), (*amount) * (*order) * (*order), f))
    {
        perror("error reading all the matrixes");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Determinant calculation of the matrix
 * 
 * This determination transforms the matrix to a base triangular matrix, column by column. 
 * 
 * @param order order of the matrix
 * @param matrix pointer to the matrix to be processed of size "order" * "order"
 * @return double (calculated determinant)
 */
double column_by_column_determinant(int order,  double *matrix)
{
    double det = 1;
    double pivotElement;
    int pivotCol;
    for (int i = 0; i < order; ++i)
    {

        pivotElement = matrix[ (i * order) + i]; // current diagonal element
        pivotCol = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int col = i + 1; col < order; ++col)
        {
            if (fabs(matrix[(i * order) + col]) > fabs(pivotElement))
            {
                // update the value of the pivot and pivot col index
                pivotElement = matrix[(i * order) + col];
                pivotCol = col;
            }
        }

        //if the diagonal element is zero then the determinant will be zeero
        if (pivotElement == 0.0)
            return 0.0;

        if (pivotCol != i) // if the pivotELement is not in the current col, then we perform a swap in the rows
        {
            for (int k = 0; k < order; k++)
            {
                double temp;
                temp = matrix[(k * order) + i];
                matrix[(k * order) + i] = matrix[(k * order) + pivotCol];
                matrix[(k * order) + pivotCol] = temp;
            }

            det *= -1.0; //signal the row swapping
        }

        det *= pivotElement; //update the determinant with the the diagonal value of the current row

        for (int col = i + 1; col < order; ++col) /* reduce the matrix to a  Base Triangle Matrix */
        // as the current row and column "i" will no longer be used, we may start reducing on the next
        // column/row (i+1)
        {
            for (int row = i + 1; row < order; ++row)
            {                   
                matrix[(row * order) + col] -= matrix[(row * order) + i] * matrix[(i * order) + col] / pivotElement;  //reduce the value
            }
        }
    }

    return det;
}

/**
 * @brief Calculates the determinant of a square matrix using Gaussian Elimination
 * 
 * This determination transforms the matrix to a base triangular matrix, row by row. 
 * 
 * @param order the order of a determinant
 * @param matrix the matrix of 1 Dimension with the length of "order" * "order"
 * @return double value of the determinant
 */
double row_by_row_determinant(int order,  double *matrix)
{
    double det = 1;
    double pivotElement;
    int pivotRow;
    for (int i = 0; i < order; ++i)
    {

        pivotElement = matrix[ (i * order) + i]; // current diagonal element
        pivotRow = i;
        // partial pivoting, which should select
        // the entry with largest absolute value from the column of the matrix
        // that is currently being considered as the pivot element
        for (int row = i + 1; row < order; ++row)
        {
            if (fabs(matrix[(row * order) + i]) > fabs(pivotElement))
            {
                // update the value of the pivot and pivot row index
                pivotElement = matrix[(row * order) + i];
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
                temp = matrix[(i * order) + k];
                matrix[(i * order) + k] = matrix[(pivotRow * order) + k];
                matrix[(pivotRow * order) + k] = temp;
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
                matrix[(row * order) + col] -= matrix[(row * order) + i] * matrix[(i * order) + col] / pivotElement;  //reduce the value
            }
        }
    }

    return det;
}

__global__ void column_by_column_determinant_gpu(double *d_matrixArray, int order, int *results)
{
    unsigned int matrixIdx = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    unsigned int columnIdx = threadIdx.x + blockDim.x *threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;;

    // unsigned int matIdx = gridDim.x * blockIdx.x

    double pivotElement = *((d_matrixArray + matrixIdx) + (columnIdx * columnIdx));

    // put values under pivot element to zero
    for(int i = 0; i < order; i++)
    {
        if(i >= columnIdx)
        {
            *((d_matrixArray + matrixIdx) + (i * order) + i) = 0;
        }
    }



}