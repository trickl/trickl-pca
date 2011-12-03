package com.trickl.pca;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.colt.matrix.linalg.LUDecomposition;
import cern.jet.math.Functions;
import com.trickl.math.PairedPermutator;
import com.trickl.math.Permutator;
import com.trickl.matrix.GramSchmidtCoefficients;
import com.trickl.matrix.SparseMult2DFunction;
import com.trickl.matrix.SparseUtils;
import com.trickl.sort.QuickSort;
import java.util.Arrays;
import org.apache.commons.lang.ArrayUtils;

/**
 * An incremental eigensolver
 *
 * @author tgee
 * Based on the following paper:
 * Incremental Eigenanalysis for Classification
 * Peter M. Hall, David Marshall, Ralph R. Martin 
 * 1998
 * See Also:
 * Merging and Splitting Eigenspace Models
 * Peter M. Hall, David Marshall, Ralph R. Martin
 * 2000
 * Department of Computer Science
 * University of Wales, Cardiff
 */
public class HallMarshallMartin implements EigenspaceModel {

   private final double tolerance = 1e-8;

   // For comparison to zero   
   private QuickSort sorter;
   private DoubleMatrix2D covarianceEigenvectorsUnrot;
   private DoubleMatrix1D covarianceEigenvalues;
   private DoubleMatrix2D meanSum;
   private DoubleMatrix2D eigenspaceRotation;
   private int s;
   private double weight;

   /**
    *
    * @param s The number of dimensions to embed
    * @param n The observation dimensionality
    */
   public HallMarshallMartin(int s, int n) {

      covarianceEigenvalues = new DenseDoubleMatrix1D(0);
      covarianceEigenvectorsUnrot = new SparseDoubleMatrix2D(s + 1, n);
      eigenspaceRotation = DoubleFactory2D.dense.identity(s + 1);
      meanSum = new SparseDoubleMatrix2D(0, n);
      this.s = s;
      this.weight = 0;

      sorter = new QuickSort();
   }

   @Override
   public DoubleMatrix2D getMean() {
      final DoubleMatrix2D meanVector = meanSum.like();
      meanSum.forEachNonZero(new IntIntDoubleFunction() {

         @Override
         public double apply(int first, int second, double value) {
            meanVector.setQuick(first, second, value / weight);
            return value;
         }
      });

      return meanVector;
   }

   @Override
   public DoubleMatrix2D getEigenvectors() {
      // Return eigenvectors as row vectors for consistency with other algorithms         
      DoubleMatrix2D covarianceEigenvectors = covarianceEigenvectorsUnrot.like();
      if (covarianceEigenvectorsUnrot instanceof SparseDoubleMatrix2D) {
         covarianceEigenvectorsUnrot.forEachNonZero(new SparseMult2DFunction(eigenspaceRotation, covarianceEigenvectors.viewDice(), true, false));
      } else {
         
         covarianceEigenvectorsUnrot.zMult(eigenspaceRotation, covarianceEigenvectors.viewDice(), 1, 0, true, false);
      }

      if (covarianceEigenvectors.rows() != covarianceEigenvalues.size()) {
         // Copy data into a new matrix (avoid returning a view on a sparse matrix as it invalidates most performance improvements).
         final DoubleMatrix2D covarianceEigenvectorsCopy = covarianceEigenvectors.like(covarianceEigenvalues.size(), covarianceEigenvectors.columns());
         covarianceEigenvectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               if (first < covarianceEigenvectorsCopy.rows()) {
                  covarianceEigenvectorsCopy.setQuick(first, second, value);
               }
               return value;
            }
         });
         covarianceEigenvectors = covarianceEigenvectorsCopy;
      }

      return covarianceEigenvectors;
   }

   @Override
   public DoubleMatrix1D getEigenvalues() {
      return covarianceEigenvalues;
   }

   private int[] getEigenvalueSortOrder(DoubleMatrix1D eigenvalues) {
      int[] sortorder = new int[eigenvalues.size()];
      for (int i = 0, end = eigenvalues.size(); i < end; ++i) {
         sortorder[i] = i;
      }
      Permutator permutator = new PairedPermutator(sortorder);
      sorter.setPermutator(permutator);
      sorter.sort(eigenvalues.toArray(), 0, eigenvalues.size());
      ArrayUtils.reverse(sortorder);
      return sortorder;
   }

   private DoubleMatrix2D getMeanDifference(DoubleMatrix2D rhsMean) {
      final DoubleMatrix2D meanDifference = rhsMean.like((meanSum.size() != 0 && rhsMean.size() != 0) ? 1 : 0,
              Math.max(meanSum.columns(), rhsMean.columns()));
      meanDifference.assign(0);
      if (meanDifference.size() != 0) {
         if (weight > 0) {
            rhsMean.forEachNonZero(new IntIntDoubleFunction() {

               @Override
               public double apply(int first, int second, double value) {
                  meanDifference.setQuick(first, second, (meanSum.getQuick(first, second) / weight) - value);
                  return value;
               }
            });
         } else {
            rhsMean.forEachNonZero(new IntIntDoubleFunction() {

               @Override
               public double apply(int first, int second, double value) {
                  meanDifference.setQuick(first, second, -value);
                  return value;
               }
            });
         }
      }

      return meanDifference;
   }

   public DoubleMatrix2D getEigenbasisCoordinates(DoubleMatrix2D input) {
      DoubleMatrix2D centeredInput = getMeanDifference(input);
      return getPrecenteredEigenbasisCoordinates(centeredInput);
   }

   // Optimized for performance, avoids recomputation of the eigenvectors
   // TODO: Use DoubleMatrix1D for input
   private DoubleMatrix2D getPrecenteredEigenbasisCoordinates(DoubleMatrix2D centeredInput) {

      // Note inputs are treated as column vectors. O(s)
      final DoubleMatrix2D coefficientsUnrot = new DenseDoubleMatrix2D(centeredInput.rows(), covarianceEigenvectorsUnrot.rows());
      if (centeredInput instanceof SparseDoubleMatrix2D) {
         SparseUtils.zMult(centeredInput, covarianceEigenvectorsUnrot, coefficientsUnrot, false, true);
      } else {
         centeredInput.zMult(covarianceEigenvectorsUnrot, coefficientsUnrot, 1, 0, false, true);
      }

      DoubleMatrix2D coefficients = new DenseDoubleMatrix2D(centeredInput.rows(), covarianceEigenvalues.size());
      coefficientsUnrot.zMult(eigenspaceRotation.viewPart(0, 0, coefficientsUnrot.columns(), coefficients.columns()), coefficients, 1, 0, false, false);

      return coefficients;
   }

   public void addInput(DoubleMatrix2D input, final double rhsWeight, final DoubleMatrix1D spatialWeights) {
      // TODO Handle multiple columns (currently can only handle one at a time
      if (input.rows() > 1) {
         throw new IllegalArgumentException("Supplied number of rows must be one.");
      }

      if (meanSum.rows() == 0) {
         meanSum = input.like(input.rows() > 0 ? 1 : 0, input.columns());
      } else {

         // Project the new input into the current eigenspace
         // The missing (zero) values are assumed to have MEAN value (not zero).
         // This reduces the complexity to the size of the sparse input
         // and has the added advantage of usually being
         // more statistically accurate if the missing values are simply unknown rather than zero.
         // O(s) where s is the number of non-zero elements in the input
         final DoubleMatrix2D meanDifference = getMeanDifference(input);

         final int p = Math.min(s, covarianceEigenvalues.size());

         // Calculate the new orthogonal basis, in terms of coefficients of the eigenvectors and inputs         
         GramSchmidtCoefficients gramSchmidtCoefficients = new GramSchmidtCoefficients(meanDifference, covarianceEigenvectorsUnrot, eigenspaceRotation.viewDice());
         final DoubleMatrix2D rhsCoefficients = gramSchmidtCoefficients.getVCoefficients();
         final DoubleMatrix2D inputCoefficients = gramSchmidtCoefficients.getACoefficients();

         if (p + gramSchmidtCoefficients.getRank() > 0) {
            final int t = rhsCoefficients.rows();
            DoubleMatrix2D intermediate = new DenseDoubleMatrix2D(p + t, p + t);
            DoubleMatrix1D rhsEigenvalues = new DenseDoubleMatrix1D(0);
            DoubleMatrix2D rhsEigenvectors = new SparseDoubleMatrix2D(0, input.columns());
            calculateIntermediate(intermediate, weight, rhsWeight, rhsCoefficients, inputCoefficients, rhsEigenvalues, rhsEigenvectors, meanDifference, meanDifference);

            solveSubsidiaryEigenproblem(intermediate, rhsCoefficients, inputCoefficients, meanDifference);
         }
      }

      // Calculate the new mean vector O(s)
      meanSum.assign(input, Functions.plusMult(rhsWeight));
      weight += rhsWeight;
   }

   public void merge(final EigenspaceModel rhs) {
      DoubleMatrix1D rhsEigenvalues = rhs.getEigenvalues();
      DoubleMatrix2D rhsEigenvectors = rhs.getEigenvectors();
      DoubleMatrix2D rhsMean = rhs.getMean();

      double rhsWeight = rhs.getWeight();
      double lhsWeight = weight;

      if (meanSum.size() > 0 || rhs.getEigenvalues().size() > 0) {
         final DoubleMatrix2D meanDifference = getMeanDifference(rhsMean);

         final int p = Math.min(s, covarianceEigenvalues.size());

         // Consolidate the meanDifference and rhs eigenvectors into a single sparse matrix
         final DoubleMatrix2D meanDifferenceAndRhsEigenvectors =
                 rhsEigenvectors.like(meanDifference.rows() + rhsEigenvectors.rows(), meanDifference.columns());
         meanDifference.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               meanDifferenceAndRhsEigenvectors.setQuick(first, second, value);
               return value;
            }
         });
         rhsEigenvectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               meanDifferenceAndRhsEigenvectors.setQuick(first + meanDifference.rows(), second, value);
               return value;
            }
         });

         // Construct an orthonormal basis set than spans both eigenspace models
         // and the difference in means
         // Calculate the new orthogonal basis, in terms of coefficients of the eigenvectors and mean difference
         GramSchmidtCoefficients gramSchmidtCoefficients = new GramSchmidtCoefficients(meanDifferenceAndRhsEigenvectors, covarianceEigenvectorsUnrot, eigenspaceRotation.viewDice(), s + 1 - p);
         final DoubleMatrix2D rhsCoefficients = gramSchmidtCoefficients.getVCoefficients();
         final DoubleMatrix2D inputCoefficients = gramSchmidtCoefficients.getACoefficients();

         if (p + gramSchmidtCoefficients.getRank() > 0) {
            // Solve the new eigenproblem to get the required rotation and new eigenvalues
            final int t = rhsCoefficients.rows();
            DoubleMatrix2D intermediate = new DenseDoubleMatrix2D(p + t, p + t);
            calculateIntermediate(intermediate, lhsWeight, rhsWeight, rhsCoefficients, inputCoefficients, rhsEigenvalues, rhsEigenvectors, meanDifference, meanDifferenceAndRhsEigenvectors);

            // Solve for the updated covariance matrix O((p+t)^3) worst case
            solveSubsidiaryEigenproblem(intermediate, rhsCoefficients, inputCoefficients, meanDifferenceAndRhsEigenvectors);
         }
      }

      if (meanSum.rows() == 0) {
         meanSum = rhs.getMean().like();
      }
      meanSum.assign(rhsMean, Functions.plusMult(rhsWeight));
      weight += rhs.getWeight();
   }

   private void calculateIntermediate(DoubleMatrix2D intermediate, double lhsWeight, double rhsWeight, final DoubleMatrix2D rhsCoefficients,
           final DoubleMatrix2D inputCoefficients, DoubleMatrix1D rhsEigenvalues, DoubleMatrix2D rhsEigenvectors,
           final DoubleMatrix2D meanDifference, final DoubleMatrix2D meanDifferenceAndRhsEigenvectors) {

      final int t = rhsCoefficients.rows();
      final int p = Math.min(s, covarianceEigenvalues.size());
      int q = rhsEigenvalues.size();

      // Set the third term
      final DoubleMatrix2D g = new DenseDoubleMatrix2D(meanDifference.rows(), p);
      final DoubleMatrix2D gUnrot = new DenseDoubleMatrix2D(meanDifference.rows(), covarianceEigenvectorsUnrot.rows());
      SparseUtils.zMult(meanDifference, covarianceEigenvectorsUnrot, gUnrot, false, true);
      gUnrot.zMult(eigenspaceRotation.viewPart(0, 0, gUnrot.columns(), g.columns()), g, 1, 0, false, false);

      final DoubleMatrix2D gamma = new DenseDoubleMatrix2D(meanDifference.rows(), t);
      if (meanDifference.size() > 0) {
         meanDifferenceAndRhsEigenvectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               for (int hRow = 0; hRow < inputCoefficients.rows(); ++hRow) {
                  double orthogonalNorm = inputCoefficients.getQuick(hRow, first) * value;
                  gamma.setQuick(0, hRow, gamma.getQuick(0, hRow) + orthogonalNorm * meanDifference.getQuick(0, second));
               }

               return value;
            }
         });

         meanDifference.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               for (int hRow = 0; hRow < rhsCoefficients.rows(); ++hRow) {
                  double orthogonalNorm = 0;
                  for (int i = 0; i < rhsCoefficients.columns(); ++i) {
                     double covarianceEigenvectorValue = 0;
                     for (int j = 0; j < eigenspaceRotation.rows(); ++j) {
                        covarianceEigenvectorValue += eigenspaceRotation.getQuick(j, i) * covarianceEigenvectorsUnrot.getQuick(j, second);
                     }
                     orthogonalNorm += rhsCoefficients.getQuick(hRow, i) * covarianceEigenvectorValue;
                  }
                  gamma.setQuick(0, hRow, gamma.getQuick(0, hRow) + orthogonalNorm * value);
               }

               return value;
            }
         });
      }

      double crossCovarianceWeight = (lhsWeight * rhsWeight)
              / ((lhsWeight + rhsWeight) * (lhsWeight + rhsWeight));
      g.zMult(g, intermediate.viewPart(0, 0, p, p), crossCovarianceWeight, 0, true, false);
      if (t > 0) {
         g.zMult(gamma, intermediate.viewPart(0, p, p, t), crossCovarianceWeight, 0, true, false);
         gamma.zMult(g, intermediate.viewPart(p, 0, t, p), crossCovarianceWeight, 0, true, false);
         gamma.zMult(gamma, intermediate.viewPart(p, p, t, t), crossCovarianceWeight, 0, true, false);
      }

      // Set the first term
      double lhsCovarianceWeight = lhsWeight / (lhsWeight + rhsWeight);
      for (int i = 0; i < p; ++i) {
         intermediate.setQuick(i, i, intermediate.getQuick(i, i)
                 + (lhsCovarianceWeight * covarianceEigenvalues.getQuick(i)));
      }

      // Set the middle term
      double rhsCovarianceWeight = rhsWeight / (lhsWeight + rhsWeight);
      final DoubleMatrix2D orthoCoefficients = new DenseDoubleMatrix2D(t, q);
      rhsEigenvectors.forEachNonZero(new IntIntDoubleFunction() {

         @Override
         public double apply(int first, int second, double value) {
            for (int hRow = 0; hRow < inputCoefficients.rows(); ++hRow) {
               double orthogonalNorm = 0;
               for (int i = 0; i < rhsCoefficients.columns(); ++i) {
                  for (int j = 0; j < eigenspaceRotation.rows(); ++j) {
                     orthogonalNorm += rhsCoefficients.getQuick(hRow, i) * eigenspaceRotation.getQuick(j, i) * covarianceEigenvectorsUnrot.getQuick(j, second);
                  }
               }
               for (int i = 0; i < inputCoefficients.columns(); ++i) {
                  orthogonalNorm += inputCoefficients.getQuick(hRow, i) * meanDifferenceAndRhsEigenvectors.getQuick(i, second);
               }
               orthoCoefficients.setQuick(hRow, first, orthoCoefficients.getQuick(hRow, first) + orthogonalNorm * value);
            }

            return value;
         }
      });

      final DoubleMatrix2D eigenCoefficients = new DenseDoubleMatrix2D(p, q);
      final DoubleMatrix2D eigenCoefficientsUnrot = new DenseDoubleMatrix2D(q, covarianceEigenvectorsUnrot.rows());
      SparseUtils.zMult(rhsEigenvectors, covarianceEigenvectorsUnrot, eigenCoefficientsUnrot, false, true);
      eigenCoefficientsUnrot.zMult(eigenspaceRotation.viewPart(0, 0, eigenCoefficientsUnrot.columns(), eigenCoefficients.rows()), eigenCoefficients.viewDice(), 1, 0, false, false);
      
      for (int i = 0; i < p; ++i) {
         for (int j = 0; j < p; ++j) {
            double sum = 0;
            for (int k = 0; k < q; ++k) {               
               sum += eigenCoefficients.getQuick(i, k) * rhsEigenvalues.getQuick(k) * eigenCoefficients.getQuick(j, k);
            }
            intermediate.setQuick(i, j, intermediate.getQuick(i, j) + rhsCovarianceWeight * sum);            
         }

         for (int j = 0; j < t; ++j) {
            double sum = 0;
            for (int k = 0; k < q; ++k) {               
               sum += eigenCoefficients.getQuick(i, k) * rhsEigenvalues.getQuick(k) * orthoCoefficients.getQuick(j, k);
            }
            intermediate.setQuick(i, p + j, intermediate.getQuick(i, p + j) + rhsCovarianceWeight * sum);            
         }
      }
      for (int i = 0; i < t; ++i) {
         for (int j = 0; j < p; ++j) {
            double sum = 0;
            for (int k = 0; k < q; ++k) {               
               sum += orthoCoefficients.getQuick(i, k) * rhsEigenvalues.getQuick(k) * eigenCoefficients.getQuick(j, k);
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
   }

   private void solveSubsidiaryEigenproblem(DoubleMatrix2D intermediate, DoubleMatrix2D rhsCoefficients, final DoubleMatrix2D inputCoefficients, DoubleMatrix2D meanDifferenceAndRhsEigenvectors) {
      final int p = Math.min(s, covarianceEigenvalues.size());
      final int t = inputCoefficients.rows();

      // Solve for the updated covariance matrix O((p+t)^3) worst case
      EigenvalueDecomposition eigensolver = new EigenvalueDecomposition(intermediate);

      // The new eigenvalues are immediately available, although not sorted
      // Note that 'intermediate' is symmetric so will not have complex eigenvalues
      int[] eigenvalueSortOrder = getEigenvalueSortOrder(eigensolver.getRealEigenvalues());
      int[] truncatedEigenvalueSortOrder = Arrays.copyOf(eigenvalueSortOrder, Math.min(s, eigenvalueSortOrder.length));
      covarianceEigenvalues = eigensolver.getRealEigenvalues().viewSelection(truncatedEigenvalueSortOrder);

      // Update the eigenspace rotation to have a component in the direction of the smallest eigenvector (which is truncated)
      // This ensures it has full-rank without having an effect on the final result
      DoubleMatrix2D rotationWorkspace = eigenspaceRotation.like();
      rotationWorkspace.viewPart(0, 0, p + t, Math.min(s, p + t)).assign(eigensolver.getV().viewSelection(null, truncatedEigenvalueSortOrder));
      for (int i = p + t; i < s; ++i) {
         rotationWorkspace.setQuick(i, i, 1);
      }

      final DoubleMatrix2D eigenspaceRotationShift = eigenspaceRotation.like();
      rhsCoefficients.forEachNonZero(new IntIntDoubleFunction() {

         @Override
         public double apply(int first, int second, double value) {
            for (int j = 0; j < eigenspaceRotation.rows(); ++j) {
               eigenspaceRotationShift.setQuick(j, p + first, eigenspaceRotationShift.getQuick(j, p + first) + eigenspaceRotation.getQuick(j, second) * value);
            }

            return value;
         }
      });
      eigenspaceRotation.assign(eigenspaceRotationShift, Functions.plus);

      if (p + t > s) {
         DoubleMatrix1D finalBasis = eigensolver.getV().viewPart(0, 0, Math.min(s + 1, p + t), p + t).viewColumn(eigenvalueSortOrder[s]);
         eigenspaceRotation.viewRow(s).assign(finalBasis, Functions.plus);
      } else {
         rotationWorkspace.setQuick(s, s, 1);
      }

      LUDecomposition lu = new LUDecomposition(eigenspaceRotation.copy());

      if (Math.abs(lu.det()) > tolerance) {
         // Append the residual as a new basis vector replacing the smallest eigenvalue
         // Replace the eigenvectors with the smallest eigenvalues with the new eigenvectors
         final DoubleMatrix2D centeredInputRot = covarianceEigenvectorsUnrot.like();
         final DoubleMatrix2D eigenspaceRotationInverse = lu.solve(DoubleFactory2D.dense.identity(eigenspaceRotation.rows()));

         // Add in all the inputs
         meanDifferenceAndRhsEigenvectors.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               for (int hRow = 0; hRow < inputCoefficients.rows(); ++hRow) {
                  centeredInputRot.viewPart(0, second, centeredInputRot.rows(), 1).assign(
                          eigenspaceRotationInverse.viewPart(hRow + p, 0, 1, centeredInputRot.rows()).viewDice(), Functions.plusMult(inputCoefficients.get(hRow, first) * value));
               }
               return value;
            }
         });

         covarianceEigenvectorsUnrot.assign(centeredInputRot, Functions.plus);

         // Update the eigenspace rotation O(p^2)
         DoubleMatrix2D eigenspaceRotationWorkspace = eigenspaceRotation.like();
         eigenspaceRotation.zMult(rotationWorkspace, eigenspaceRotationWorkspace);
         eigenspaceRotation.assign(eigenspaceRotationWorkspace);
      } else {
         // We must handle this case, where the inverse cannot be represented (due to orthogonality)
         // by multiplying out the eigenvectors explicitly. This is costly in performance, but fortunately
         // this special case happens rarely and usually early on, where the eigenvectors are still very
         // sparse.
         final DoubleMatrix2D covarianceEigenvectors = covarianceEigenvectorsUnrot.like();
         if (covarianceEigenvectorsUnrot instanceof SparseDoubleMatrix2D) {
            covarianceEigenvectorsUnrot.forEachNonZero(new SparseMult2DFunction(eigenspaceRotation, covarianceEigenvectors.viewDice(), true, false));
         } else {
            covarianceEigenvectorsUnrot.zMult(eigenspaceRotation, covarianceEigenvectors.viewDice(), 1, 0, true, false);
         }

         final DoubleMatrix2D inputs = meanDifferenceAndRhsEigenvectors.like(inputCoefficients.rows(), meanDifferenceAndRhsEigenvectors.columns());
         final DoubleMatrix2D inputsOffset = covarianceEigenvectorsUnrot.like();
         meanDifferenceAndRhsEigenvectors.forEachNonZero(new SparseMult2DFunction(inputCoefficients, inputs.viewDice(), true, true));
         
         inputs.forEachNonZero(new IntIntDoubleFunction() {

            @Override
            public double apply(int first, int second, double value) {
               inputsOffset.setQuick(first + p, second, value);
               return value;
            }
         });

         covarianceEigenvectors.assign(inputsOffset, Functions.plus);
         covarianceEigenvectorsUnrot = covarianceEigenvectors;
         eigenspaceRotation = rotationWorkspace;
      }
   }

   @Override
   public double getWeight() {
      return weight;
   }
}
