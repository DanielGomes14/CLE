

/** \brief method used to Load Information from Files **/
void loadFilesInfo();
/** \brief method used to Store the Partials Results retrieved from workers **/
void storePartialResult(double **results, int fileId, int matrixId, double determinant);
/** \brief method used to print the Final Results **/
void printResults(double **results,int fileAmount);


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
