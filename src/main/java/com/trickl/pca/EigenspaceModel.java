package com.trickl.pca;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public interface EigenspaceModel
{   
   DoubleMatrix1D getEigenvalues();
   DoubleMatrix2D getEigenvectors();
   DoubleMatrix2D getMean();
   double getWeight();
}
