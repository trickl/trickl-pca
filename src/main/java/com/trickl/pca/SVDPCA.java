/*
 * This file is part of the Trickl Open Source Libraries.
 *
 * Trickl Open Source Libraries - http://open.trickl.com/
 *
 * Copyright (C) 2011 Tim Gee.
 *
 * Trickl Open Source Libraries are free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Trickl Open Source Libraries are distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this project.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.trickl.pca;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
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

   private final DoubleMatrix2D covarianceEigenvectors;

   private final DoubleMatrix1D covarianceEigenvalues;
   
   private final double[] variableWeights;

   private final DoubleMatrix2D mean;
   
   private final DoubleMatrix2D variances;

   private int observationCount;

   private int rank;
   
   /**    
    * @param input A rectangular matrix where #rows > #columns
    */
   public SVDPCA(DoubleMatrix2D input)
   {
      this(input, null);
   }

   /**    
    * @param input A rectangular matrix where #rows > #columns
    * @param variableWeights optional, when specified each variable is normalised (scaled
    * for equal variance) and then scaled according to the weight. Apply weights of one to each 
    * variable would standardise the data.
    */
   public SVDPCA(DoubleMatrix2D input, double[] variableWeights)
   {
      // If there are no rows in the input, the mean should have zero size
      // (as a non-zero size would imply a zero mean which is untrue).
      observationCount = input.rows();
      mean = input.like(input.rows() > 0 ? 1 : 0, input.columns());
      variances = input.like(input.rows() > 0 ? 1 : 0, input.columns());
      this.variableWeights = variableWeights;

      if (input.rows() > 0) {
          
         // Calculate the mean
         final double weight = 1. /  input.rows();
         input.forEachNonZero((int first, int second, double value) -> {
             mean.setQuick(0, second, mean.getQuick(0, second) + value * weight);
             return value;
         });
                  
         // Subtract the mean from the input
         DoubleMatrix2D centeredInput = getCenteredInput(input);
         
         // Calculate the variances
         centeredInput.forEachNonZero((int first, int second, double value) -> {
              variances.setQuick(0, second, variances.getQuick(0, second) + value * value);
              return value;
         });
         
         // Apply scaling if required
         if (this.variableWeights != null) {                          
             centeredInput = getScaledInput(centeredInput);
         }

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
   
   private DoubleMatrix2D getCenteredInput(DoubleMatrix2D input) {      
      DoubleMatrix2D centeredInput = input.copy();
      for (int i = 0; i < input.rows(); ++i) {
         centeredInput.viewRow(i).assign(mean.viewRow(0), Functions.minus);
      }
      return centeredInput;
   }
   
   private DoubleMatrix2D getScaledInput(DoubleMatrix2D centeredInput) {      
      DoubleMatrix2D standardisedInput = centeredInput.copy();     
      centeredInput.forEachNonZero((int first, int second, double value) -> {
         standardisedInput.setQuick(first, second,
                 (value * variableWeights[second]) / (Math.sqrt(variances.getQuick(0, second))));
         return value;
      });
      
      return standardisedInput;
   }
   
   public DoubleMatrix2D getEigenbasisCoordinates(DoubleMatrix2D input) {
      DoubleMatrix2D centeredInput = getCenteredInput(input);      
      if (this.variableWeights != null) {
          centeredInput = getScaledInput(centeredInput);
      }
      return getPrecenteredEigenbasisCoordinates(centeredInput);
   }
   
   private DoubleMatrix2D getPrecenteredEigenbasisCoordinates(DoubleMatrix2D input) {
       DenseDoubleMatrix2D coefficients = new DenseDoubleMatrix2D(input.rows(), covarianceEigenvalues.size());
       input.zMult(covarianceEigenvectors, coefficients, 1, 0, false, true);

       // Reconstruct the data from the lower dimensional information
       return coefficients;
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
