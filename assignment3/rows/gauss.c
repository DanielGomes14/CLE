#include "gauss.h"

/**
 * @brief Determinant calculation of the matrix
 * 
 * This determination transforms the matrix to a base triangular matrix, row by row. 
 * 
 * @param order order of the matrix
 * @param matrix pointer to the matrix to be processed of size "order" * "order"
 * @return double (calculated determinant)
 */
double determinant(int order,  double * matrix)
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


double determinant2(int order, double *matrix)
{
    int pivotElement, pivotCol;
    double det = 1;
    for(int i = 0; i < order; ++i)
    {
        // elemento diagonal da matriz
        pivotElement = matrix[(i * order) + i];
        // coluna do elemento atual da diagonal
        pivotCol = i;

        for(int col = i + 1; col < order; ++col)
        {
            if(fabs(matrix[(i * order) + col]) > fabs(pivotElement))
            {
                pivotElement = matrix[(i * order) + col];
                pivotCol = col;
            }
        }

        if(pivotElement == 0.0)
            return 0.0;

        if(pivotCol != i)
        {
            for(int k = 0; k < order; k++)
            {
                double temp;
                temp = matrix[(k * order) + i];
                matrix[(k * order) + i] = matrix[(k * order) + pivotCol];
                matrix[(k * order) + pivotCol] = temp;
            }
            det *= -1.0;
        }

        det *= pivotElement;

        for(int col = i + 1; col < order; ++col)
        {
            for(int row = i + 1; row < order; ++row)
            {
                matrix[(row * order) + col] -= matrix[(i * order) + col] * matrix[(row * order) + i] / pivotElement;
            }
        }
    }
}