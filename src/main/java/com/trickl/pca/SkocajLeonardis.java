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

import cern.colt.function.IntDoubleFunction;
import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Functions;

/**
 * An incremental eigensolver which allows for temporal weighting
 *
 * @author tgee
 * Based on the following paper:
 * Weighted and Robust Incremental Method for Subspace Learning
 * Danijel Skocaj and Ales Leonardis
 * Faculty of Computer and Information Science
 * University of Ljubjana, Slovenia
 */
public class SkocajLeonardis implements EigenspaceModel {

    private DoubleMatrix2D eigenvectors;
    private DoubleMatrix1D eigenvalues;
    private DoubleMatrix2D mean;
    private DoubleMatrix2D coefficients;
    private int k;
    private int observationCount;

    public SkocajLeonardis(int k, int d, int capacity) {
        this.observationCount = 0;
        this.k = k;
        eigenvalues = new DenseDoubleMatrix1D(k);
        eigenvectors = new SparseDoubleMatrix2D(d, k + 1);
        coefficients = new SparseDoubleMatrix2D(k + 1, capacity);
    }

    public void addInput(DoubleMatrix2D input, IntDoubleFunction temporalWeighting, DoubleMatrix1D spatialWeights) {
        // TODO Preprocess the image using the spatial weights
        // 1. Calculate the coefficients using the spatial weights
        // 2. Reconstruct the image using these coefficients
        // 3. Blend both the reconstructed image and the historic image using the weights
        // 4. Feed to the normal algorithm

        addInput(input, temporalWeighting);
    }

    public void addInput(DoubleMatrix2D input, IntDoubleFunction temporalWeighting) {
        // Project the new input into the current eigenspace
         final DoubleMatrix2D centeredInput = input.like();
         input.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int first, int second, double value) {
               centeredInput.setQuick(0, second, value - mean.getQuick(0, second));
               return value;
            }
         });

        DoubleMatrix2D eigenVectorsView = eigenvectors.viewPart(0, 0, eigenvectors.rows(), k);
        DoubleMatrix2D inputCoeffs = new DenseDoubleMatrix2D(centeredInput.columns(), k);
        eigenVectorsView.zMult(centeredInput, inputCoeffs, 1, 0, true, false);

        // Reconstruct the new input
        DoubleMatrix2D reconstruction = mean.copy();
        eigenVectorsView.zMult(inputCoeffs, reconstruction, 1, 1, false, false);

        // Compute the residual vector
        DoubleMatrix2D residual = input.copy();
        residual.assign(reconstruction, Functions.minus);

        // Normalize the residual
        double residualSize = residual.aggregate(Functions.plus, Functions.square);
        residual.assign(Functions.div(residualSize));

        // Append the residual as a new basis vector
        eigenvectors.viewColumn(k).assign(residual.viewRow(0));

        // Update the coefficents matrix
        DoubleMatrix2D coefficientsView = coefficients.viewPart(0, 0, k + 1, observationCount);
        coefficientsView.viewColumn(observationCount).viewPart(0, k).assign(inputCoeffs.viewRow(0));
        coefficientsView.viewRow(k).assign(0);
        coefficientsView.setQuick(k, observationCount, residualSize);

        // TODO apply the temporal weighting at this stage to each set of coefficients
        // Perform PCA on the coefficients, the limiting performance factor here
        // will be the number of coefficients (n)
        SVDPCA pca = new SVDPCA(coefficients.viewPart(0, 0, k, observationCount));

        // Project the coefficients to the new basis
        DoubleMatrix2D centeredCoeffs = coefficientsView.copy();
        for (int i = 0; i < centeredCoeffs.columns(); ++i) {
            centeredCoeffs.viewColumn(i).assign(pca.getMean().viewRow(0), Functions.minus);
        }
        eigenvectors.zMult(centeredCoeffs, coefficientsView, 1, 0, true, false);

        // Update the mean
        eigenvectors.zMult(pca.getMean().viewRow(0), mean.viewRow(0), 1, 1, false);

        eigenvalues = pca.getEigenvalues();

        // Rotate the subspace
        DoubleMatrix2D eigenvectorsWorkspace = eigenvectors.copy();
        eigenvectors.zMult(pca.getEigenvectors(), eigenvectorsWorkspace);
        eigenvectors = eigenvectorsWorkspace;

        ++observationCount;
    }

    @Override
    public DoubleMatrix2D getMean() {
        return mean;
    }

    @Override
    public DoubleMatrix2D getEigenvectors() {
        return eigenvectors;
    }

    @Override
    public DoubleMatrix1D getEigenvalues() {
        return eigenvalues;
    }

   @Override
   public double getWeight() {
      return observationCount;
   }
}
