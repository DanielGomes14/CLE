#ifndef UTILS_CUH
#define UTILS_CUH

#include <stdio.h>
#include <math.h>

void readData(char *fileName, double **matrixArray, int *order, int *amount);

double column_by_column_determinant(int order,  double *matrix);

double row_by_row_determinant(int order, double *matrix);

#endif