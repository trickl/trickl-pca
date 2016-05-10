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

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import com.trickl.matrixunit.MatrixAssert;
import java.util.Arrays;
import org.junit.Assert;

public final class EigenspaceModelAssert {

   private EigenspaceModelAssert() {

   }

   public static void assertEigenvectorsEquals(DoubleMatrix2D expected, DoubleMatrix2D actual, double tolerance) {
      Assert.assertEquals(expected.rows(), actual.rows());
      Assert.assertEquals(expected.columns(), actual.columns());
      IntArrayList expectedIndices = new IntArrayList(expected.rows());
      DoubleArrayList expectedValues = new DoubleArrayList(expected.rows());
      IntArrayList actualIndices = new IntArrayList(actual.rows());
      DoubleArrayList actualValues = new DoubleArrayList(actual.rows());
      for (int i = 0; i < expected.rows(); ++i) {
         expected.viewRow(i).getNonZeros(expectedIndices, expectedValues);
         actual.viewRow(i).getNonZeros(actualIndices, actualValues);
         Assert.assertArrayEquals(Arrays.copyOfRange(expectedIndices.elements(), 0, expectedIndices.size()),
                                  Arrays.copyOfRange(actualIndices.elements(), 0, actualIndices.size()));
         // Flip the direction of the actual eigenvector to match if necessary
         if (!expectedIndices.isEmpty()) {
            if ((expectedValues.get(expectedIndices.get(0)) > 0 &&
                actualValues.get(actualIndices.get(0)) < 0) ||
                (expectedValues.get(expectedIndices.get(0)) < 0 &&
                actualValues.get(actualIndices.get(0)) > 0)) {
               for (int j = 0; j < actualValues.size(); ++j) {
                  actualValues.setQuick(j, -actualValues.getQuick(j));
               }
            }
         }
         Assert.assertArrayEquals(Arrays.copyOfRange(expectedValues.elements(), 0, expectedValues.size()),
                                  Arrays.copyOfRange(actualValues.elements(), 0, actualValues.size()), tolerance);
      }
   }

   public static void assertEquals(EigenspaceModel expected, EigenspaceModel actual, double tolerance) {
      DoubleMatrix2D expectedMean = expected.getMean();
      DoubleMatrix1D expectedEigenvalues = expected.getEigenvalues();
      DoubleMatrix2D expectedEigenvectors = expected.getEigenvectors();
      double expectedWeight = expected.getWeight();

      DoubleMatrix2D actualMean = actual.getMean();
      DoubleMatrix1D actualEigenvalues = actual.getEigenvalues();
      DoubleMatrix2D actualEigenvectors = actual.getEigenvectors();
      double actualWeight = actual.getWeight();

      Assert.assertEquals(expectedWeight, actualWeight, tolerance);
      MatrixAssert.assertEquals(expectedMean, actualMean, tolerance);
      MatrixAssert.assertEquals(expectedEigenvalues, actualEigenvalues, tolerance);
      assertEigenvectorsEquals(expectedEigenvectors, actualEigenvectors, tolerance);
   }
}
