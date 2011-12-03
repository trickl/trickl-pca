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
