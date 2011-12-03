package com.trickl.pca;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.SingularValueDecomposition;
import cern.jet.math.Functions;
import java.util.Arrays;

/**
 *
 * @author tgee
 *
 * Simple SVD based PCA
 */
public class SVDPCA implements EigenspaceModel {

   private DoubleMatrix2D covarianceEigenvectors;

   private DoubleMatrix1D covarianceEigenvalues;

   private DoubleMatrix2D mean;

   private int observationCount;

   private int rank;

   /**
    *
    * @param input A rectangular matrix where #rows > #columns
    */
   public SVDPCA(DoubleMatrix2D input)
   {
      // If there are no rows in the input, the mean should have zero size
      // (as a non-zero size would imply a zero mean which is untrue).
      observationCount = input.rows();
      mean = input.like(input.rows() > 0 ? 1 : 0, input.columns());

      if (input.rows() > 0) {
         final double weight = 1. /  input.rows();
         input.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int first, int second, double value) {
               mean.setQuick(0, second, mean.getQuick(0, second) + value * weight);
               return value;
            }
         });

         // Subtract the mean from the input
         final DoubleMatrix2D centeredInput = input.like();
         input.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int first, int second, double value) {
               centeredInput.setQuick(first, second, value - mean.getQuick(0, second));
               return value;
            }
         });

         // Perform SVD on the centered input
         // See http://public.lanl.gov/mewall/kluwer2002.html for the relationship
         // between PCA and SVD
         boolean transpose = centeredInput.rows() > centeredInput.columns();
         SingularValueDecomposition svd = new SingularValueDecomposition(
                 transpose ? centeredInput : centeredInput.viewDice());
         rank = svd.rank();
         covarianceEigenvalues = new DenseDoubleMatrix1D(rank);
         covarianceEigenvalues.assign(Arrays.copyOfRange(svd.getSingularValues(), 0, rank));
         covarianceEigenvalues.assign(Functions.chain(Functions.div(input.rows()), Functions.square));
         covarianceEigenvectors = transpose ? svd.getV().viewPart(0, 0, input.columns(), rank).viewDice()
                                            : svd.getU().viewPart(0, 0, input.columns(), rank).viewDice();
      }
      else {
         covarianceEigenvalues = new DenseDoubleMatrix1D(0);
         covarianceEigenvectors = input.like(rank, input.columns());
      }
   }

   @Override
   public DoubleMatrix2D getMean()
   {
      return mean;
   }

   @Override
   public DoubleMatrix2D getEigenvectors()
   {
      return covarianceEigenvectors;
   }

   @Override
   public DoubleMatrix1D getEigenvalues()
   {
      return covarianceEigenvalues;
   }

   @Override
   public double getWeight() {
      return observationCount;
   }
}
