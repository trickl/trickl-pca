package com.trickl.pca;

import cern.colt.matrix.DoubleMatrix1D;

public interface IncrementalPCA extends EigenspaceModel
{
   void addInput(DoubleMatrix1D input, final double inputWeight, final DoubleMatrix1D spatialWeights);
   void merge(EigenspaceModel pca);
}
