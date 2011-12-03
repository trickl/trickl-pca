package com.trickl.pca;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

public class WeightedPCAResult implements EigenspaceModel {

   private DoubleMatrix1D covarianceEigenvalues;
   private DoubleMatrix2D covarianceEigenvectors;
   private DoubleMatrix2D mean;
   private double weight;

   public WeightedPCAResult(DoubleMatrix1D covarianceEigenvalues,
                            DoubleMatrix2D covarianceEigenvectors,
                            DoubleMatrix2D mean,
                            double weight) {
      this.covarianceEigenvalues = covarianceEigenvalues;
      this.covarianceEigenvectors = covarianceEigenvectors;
      this.mean = mean;
      this.weight = weight;
   }

   @Override
   public DoubleMatrix1D getEigenvalues() {
      return covarianceEigenvalues;
   }

   @Override
   public DoubleMatrix2D getEigenvectors() {
      return covarianceEigenvectors;
   }

   @Override
   public DoubleMatrix2D getMean() {
      return mean;
   }

   @Override
   public double getWeight() {
      return weight;
   }
}
