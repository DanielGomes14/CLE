
#include "dispatcher.h"

/**
 * @brief Store the partial results on the Results Data Structure
 * 
 * @param fileId the Identifier of the File where the results will be put
 * @param matrixId  the Identifier of the Matrix for which the Determinant was calculated
 * @param determinant the determinant value obtained to be stored
 */
void storePartialResult(double **results, int fileId, int matrixId, double determinant)
{
   
    results[fileId][matrixId] = determinant; /* store value */

}
