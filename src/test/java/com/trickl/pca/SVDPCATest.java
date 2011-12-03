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

import com.trickl.pca.SVDPCA;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.jet.math.Functions;
import cern.jet.random.engine.MersenneTwister;
import com.trickl.dataset.InclinedPlane3D;
import java.awt.Rectangle;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SVDPCATest {

   public SVDPCATest() {
   }

   @Before
   public void setUp() {
   }

   @After
   public void tearDown() {
   }

   @Test
   public void testInclinedPlane() throws IOException {      
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});
      
      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomEngine(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(10);

      SVDPCA pca = new SVDPCA(data);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      System.out.println("Eigenvectors:");
      System.out.println(pca.getEigenvectors());

      System.out.println("Meanvector:");
      System.out.println(pca.getMean());

      // Recalculate the input from a truncated SVD, first calculate the mean
      DoubleMatrix1D mean = new SparseDoubleMatrix1D(3);      
      for (int i = 0; i < data.rows(); ++i) {
         mean.assign(data.viewRow(i), Functions.plus);
      }
      mean.assign(Functions.div(data.rows()));

      // Truncate the SVD and calculate the coefficient matrix
      DenseDoubleMatrix2D coefficients = new DenseDoubleMatrix2D(data.rows(), 2);
      DoubleMatrix2D centeredInput = data.copy();
      for (int i = 0; i < data.rows(); ++i) {
         centeredInput.viewRow(i).assign(mean, Functions.minus);
      }
      centeredInput.zMult(pca.getEigenvectors().viewPart(0, 0, 2, 3), coefficients, 1, 0, false, true);

      // Reconstruct the data from the lower dimensional information
      DoubleMatrix2D reconstruction = data.copy();
      for (int i = 0; i < reconstruction.rows(); ++i) {
         reconstruction.viewRow(i).assign(mean);
      }
      coefficients.zMult(pca.getEigenvectors().viewPart(0, 0, 2, 3), reconstruction, 1, 1, false, false);

      // Output to file (can be read by GNU Plot)
      String fileName = "inclined-plane-svd-pca.dat";
      String packagePath = this.getClass().getPackage().getName().replaceAll("\\.", "/");
      File outputFile = new File("src/test/resources/"
              + packagePath
              + "/" + fileName);
      PrintWriter writer = new PrintWriter(outputFile);
      writer.write(data.toString());
      writer.close();
   }
}
