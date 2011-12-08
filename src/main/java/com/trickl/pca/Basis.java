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

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.jet.math.Functions;
import com.trickl.math.ChainPermutator;
import com.trickl.math.IntArrayPermutator;
import com.trickl.math.Permutator;
import com.trickl.math.StandardPermutator;
import com.trickl.matrix.ModifiedGramSchmidt;
import com.trickl.matrix.SparseUtils;
import com.trickl.sort.QuickSort;

public class Basis {
   
   public static DoubleMatrix2D getEigenbasisCoordinates(EigenspaceModel pca, final DoubleMatrix2D input) {
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      final DoubleMatrix2D mean = pca.getMean();      
      final DoubleMatrix2D centeredInput = input.like(1, input.columns());
      centeredInput.assign(0);

      input.forEachNonZero(new IntIntDoubleFunction() {

         @Override
         public double apply(int first, int second, double value) {
            centeredInput.setQuick(first, second, value - mean.getQuick(0, second));
            return value;
         }
      });

      final DoubleMatrix2D coefficients = new DenseDoubleMatrix2D(centeredInput.rows(), eigenvectors.rows());
      SparseUtils.zMult(centeredInput, eigenvectors, coefficients, false, true);

      return coefficients;
   }

   public static EigenspaceModel merge(EigenspaceModel lhs, EigenspaceModel rhs) {
      final double combinedWeight = lhs.getWeight() + rhs.getWeight();
      DoubleMatrix1D lhsEigenvalues = lhs.getEigenvalues();
      DoubleMatrix2D lhsEigenvectors = lhs.getEigenvectors();
      DoubleMatrix1D rhsEigenvalues = rhs.getEigenvalues();
      DoubleMatrix2D rhsEigenvectors = rhs.getEigenvectors();
      final DoubleMatrix2D lhsMean = lhs.getMean();
      DoubleMatrix2D rhsMean = rhs.getMean();
      final double rhsWeight = rhs.getWeight();
      int p = lhsEigenvalues.size();
      int q = rhsEigenvalues.size();

      final DoubleMatrix2D meanDifference = lhsMean.like((lhsMean.size() != 0 && rhsMean.size() != 0) ? 1 : 0,
              Math.max(lhsMean.columns(), rhsMean.columns()));
      final DoubleMatrix2D combinedMean = lhsMean.like((lhsMean.size() != 0 || rhsMean.size() != 0) ? 1 : 0,
              Math.max(lhsMean.columns(), rhsMean.columns()));
      
      if (meanDifference.size() != 0) {
         // Assume the rhs to have lhs mean values where it doesn't have values
         rhsMean.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int first, int second, double value) {
               meanDifference.setQuick(first, second, lhsMean.getQuick(first, second) - value);
               return value;
            }
         });
      }

      if (lhsMean.size() != 0) {
         combinedMean.assign(lhsMean, Functions.plusMult(lhs.getWeight() / combinedWeight));
      }
      rhsMean.forEachNonZero(new IntIntDoubleFunction() {
         @Override
         public double apply(int first, int second, double value) {
            combinedMean.setQuick(first, second, combinedMean.getQuick(first, second) + value * rhsWeight / combinedWeight);
            return value;
         }
      });

      // Construct an orthonormal basis set than spans both eigenspace models
      // and the difference in means

      // Calculate the residues of each of the eigenvectors in rhs w.r.t the eigenspace of the lhs.
      // Also calculate the residue of the difference in the mean
      DoubleMatrix2D rhsCoefficients = new DenseDoubleMatrix2D(q + meanDifference.rows(), p);
      for (int i = 0; i < p; ++i) {
         for (int j = 0; j < meanDifference.rows(); ++j) {
            rhsCoefficients.setQuick(j, i, lhsEigenvectors.viewRow(i).zDotProduct(meanDifference.viewRow(j)));
         }

         for (int j = 0; j < q; ++j) {
            rhsCoefficients.setQuick(j + meanDifference.rows(), i, lhsEigenvectors.viewRow(i).zDotProduct(rhsEigenvectors.viewRow(j)));
         }
      }

      DoubleMatrix2D rhsResiduals = rhsEigenvectors.like(rhsCoefficients.rows(), rhsEigenvectors.columns());
      rhsResiduals.viewPart(0, 0, meanDifference.rows(), meanDifference.columns()).assign(meanDifference);
      rhsResiduals.viewPart(meanDifference.rows(), 0, rhsEigenvectors.rows(), rhsEigenvectors.columns()).assign(rhsEigenvectors);
      lhsEigenvectors.zMult(rhsCoefficients, rhsResiduals.viewDice(), -1, 1, true, true);

      // Create an orthonormal basis from all the residuals,
      // Note zero vectors are removed, so that t <= q + 1      
      ModifiedGramSchmidt orthogonalDecomposition = new ModifiedGramSchmidt(rhsResiduals);
      DoubleMatrix2D orthogonalBasis = orthogonalDecomposition.getS();
      int t = orthogonalBasis.rows();

      DoubleMatrix2D extendedEigenbasis = orthogonalBasis.like(lhsEigenvectors.rows() + orthogonalBasis.rows(), lhsEigenvectors.columns());
      extendedEigenbasis.viewPart(0, 0, lhsEigenvectors.rows(), lhsEigenvectors.columns()).assign(lhsEigenvectors);
      extendedEigenbasis.viewPart(lhsEigenvectors.rows(), 0, orthogonalBasis.rows(), lhsEigenvectors.columns()).assign(orthogonalBasis);

      // Solve the new eigenproblem to get the required rotation and new eigenvalues
      // Set the last term      
      DoubleMatrix2D eigenvectors = extendedEigenbasis.like();
      DoubleMatrix1D eigenvalues = extendedEigenbasis.like1D(extendedEigenbasis.rows());
      if (p + t > 0) {
         DoubleMatrix2D intermediate = new DenseDoubleMatrix2D(p + t, p + t);
         double crossCovarianceWeight = (lhs.getWeight() * rhs.getWeight())
                 / (combinedWeight * combinedWeight);
         DoubleMatrix2D g = rhsCoefficients.viewPart(0, 0, meanDifference.rows(), rhsCoefficients.columns());
         DoubleMatrix2D gamma = new DenseDoubleMatrix2D(t, meanDifference.rows());
         orthogonalBasis.zMult(meanDifference, gamma, 1, 0, false, true);

         g.zMult(g, intermediate.viewPart(0, 0, p, p), crossCovarianceWeight, 0, true, false);
         g.zMult(gamma, intermediate.viewPart(0, p, p, t), crossCovarianceWeight, 1, true, true);
         gamma.zMult(g, intermediate.viewPart(p, 0, t, p), crossCovarianceWeight, 0, false, false);
         gamma.zMult(gamma, intermediate.viewPart(p, p, t, t), crossCovarianceWeight, 0, false, true);
         //DoubleMatrix2D thirdTerm = intermediate.copy();

         // Set the first term
         //DoubleMatrix2D firstTerm = new DenseDoubleMatrix2D(p + t, p + t);
         double lhsCovarianceWeight = lhs.getWeight() / combinedWeight;
         for (int i = 0; i < p; ++i) {
            intermediate.setQuick(i, i, intermediate.getQuick(i, i)
                    + (lhsCovarianceWeight * lhsEigenvalues.getQuick(i)));
            //firstTerm.setQuick(i, i, lhsCovarianceWeight * lhsEigenvalues.getQuick(i));
         }

         // Set the middle term         
         double rhsCovarianceWeight = rhs.getWeight() / combinedWeight;
         DoubleMatrix2D orthoCoefficients = new DenseDoubleMatrix2D(t, q);
         orthogonalBasis.zMult(rhsEigenvectors, orthoCoefficients, 1, 0, false, true);
         for (int i = 0; i < p; ++i) {
            for (int j = 0; j < p; ++j) {
               double sum = 0;
               for (int k = 0; k < q; ++k) {
                  sum += rhsCoefficients.getQuick(k + meanDifference.rows(), i) * rhsEigenvalues.getQuick(k) * rhsCoefficients.getQuick(k + meanDifference.rows(), j);
               }
               intermediate.setQuick(i, j, intermediate.getQuick(i, j) + rhsCovarianceWeight * sum);               
            }

            for (int j = 0; j < t; ++j) {
               double sum = 0;
               for (int k = 0; k < q; ++k) {
                  sum += rhsCoefficients.getQuick(k + meanDifference.rows(), i) * rhsEigenvalues.getQuick(k) * orthoCoefficients.getQuick(j, k);
               }
               intermediate.setQuick(i, p + j, intermediate.getQuick(i, p + j) + rhsCovarianceWeight * sum);               
            }
         }
         for (int i = 0; i < t; ++i) {
            for (int j = 0; j < p; ++j) {
               double sum = 0;
               for (int k = 0; k < q; ++k) {
                  sum += orthoCoefficients.getQuick(i, k) * rhsEigenvalues.getQuick(k) * rhsCoefficients.getQuick(k + meanDifference.rows(), j);
               }
               intermediate.setQuick(p + i, j, intermediate.getQuick(p + i, j) + rhsCovarianceWeight * sum);               
            }

            for (int j = 0; j < t; ++j) {
               double sum = 0;
               for (int k = 0; k < q; ++k) {
                  sum += orthoCoefficients.getQuick(i, k) * rhsEigenvalues.getQuick(k) * orthoCoefficients.getQuick(j, k);
               }
               intermediate.setQuick(p + i, p + j, intermediate.getQuick(p + i, p + j) + rhsCovarianceWeight * sum);               
            }
         }

         // Solve for the updated covariance matrix O((p+t)^3) worst case
         EigenvalueDecomposition eigensolver = new EigenvalueDecomposition(intermediate);

         // The new eigenvalues are immediately available, although not sorted
         DoubleMatrix2D eigenspaceRotationWorkspace = new DenseDoubleMatrix2D(p + t, p + t);
         DenseDoubleMatrix2D rotationWorkspace = new DenseDoubleMatrix2D(p + t, p + t);
         DenseDoubleMatrix2D eigenspaceRotation = new DenseDoubleMatrix2D(p + t, p + t);
         // Set the initial rotation to the identity matrix
         for (int i = 0; i < p + t; ++i) {
            eigenspaceRotation.setQuick(i, i, 1);
         }

         int[] eigenvalueSortOrder = new int[eigensolver.getRealEigenvalues().size()];
         getSortOrder(eigensolver.getRealEigenvalues(), eigenvalueSortOrder);
         eigenvalues = eigensolver.getRealEigenvalues().viewSelection(eigenvalueSortOrder).viewFlip();
         rotationWorkspace.assign(eigensolver.getV().viewSelection(null, eigenvalueSortOrder).viewColumnFlip());
         eigenspaceRotation.zMult(rotationWorkspace, eigenspaceRotationWorkspace);
         eigenspaceRotation.assign(eigenspaceRotationWorkspace);

         // Rotate the eigenspace
         extendedEigenbasis.zMult(eigenspaceRotation, eigenvectors.viewDice(), 1, 0, true, false);
      }

      EigenspaceModel result = new WeightedPCAResult(eigenvalues, eigenvectors, combinedMean, combinedWeight);
      return result;
   }

   private static void getSortOrder(DoubleMatrix1D column, int[] sortorder) {
      QuickSort sorter = new QuickSort();
      for (int i = 0, end = column.size(); i < end; ++i) {
         sortorder[i] = i;
      }
      Permutator permutator = new ChainPermutator(new IntArrayPermutator(sortorder), new StandardPermutator());
      sorter.setPermutator(permutator);
      sorter.sort(column.toArray(), 0, column.size());
   }
}
