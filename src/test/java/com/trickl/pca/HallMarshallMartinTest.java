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

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.jet.math.Functions;
import com.trickl.dataset.InclinedPlane3D;
import com.trickl.dataset.SwissRoll3D;
import com.trickl.matrix.SparseUtils;
import com.trickl.matrixunit.MatrixAssert;
import java.awt.Rectangle;
import java.io.*;
import java.net.URL;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.math3.random.MersenneTwister;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class HallMarshallMartinTest {

   private static double tolerance = 1e-6;

   public HallMarshallMartinTest() {
   }

   @Before
   public void setUp() {
   }

   @After
   public void tearDown() {
   }

   @Test
   public void testTrivialCase() throws IOException {

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      DoubleMatrix2D data = new DenseDoubleMatrix2D(0, 3);
      data.assign(new double[][] {});      
      SVDPCA expectedPCA = new SVDPCA(data);
      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testSingleInputIncrement() throws IOException {

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      DoubleMatrix2D data = new DenseDoubleMatrix2D(1, 3);
      data.assign(new double[][] {{3, 4, -5}});
      pca.addInput(data, 1.0, null);
      SVDPCA expectedPCA = new SVDPCA(data);
      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testSingleInputMerge() throws IOException {

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      DoubleMatrix2D data = new DenseDoubleMatrix2D(1, 3);
      data.assign(new double[][] {{3, 4, -5}});
      SVDPCA rhsPCA = new SVDPCA(data);
      pca.merge(rhsPCA);      
      SVDPCA expectedPCA = rhsPCA;
      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testCurvedPlaneIncrementalReducedDimension() throws IOException {
      System.out.println("testCurvedPlaneIncrementalReducedDimension");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      // Use the swiss roll generator to create a noisy curved plane
      SwissRoll3D swissRoll = new SwissRoll3D();
      swissRoll.setRandomGenerator(new MersenneTwister(123456789));
      swissRoll.setNormal(normal);
      swissRoll.setRevolutions(0.3);
      DoubleMatrix2D data = swissRoll.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      for (int i = 0; i < data.rows(); ++i) {
         pca.addInput((DenseDoubleMatrix2D) data.viewPart(i, 0, 1, data.columns()), 1, null);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, tolerance);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      System.out.println("Eigenvectors:");
      System.out.println(pca.getEigenvectors());

      System.out.println("Meanvector:");
      System.out.println(pca.getMean());

      outputEigenspaceToFile("curved-plane-hmm.dat", data, pca);
   }

   @Test
   public void testCurvedPlaneMergeReducedDimension() throws IOException {
      System.out.println("testCurvedPlaneMergeReducedDimension");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      // Use the swiss roll generator to create a noisy curved plane
      SwissRoll3D swissRoll = new SwissRoll3D();
      swissRoll.setRandomGenerator(new MersenneTwister(123456789));
      swissRoll.setNormal(normal);
      swissRoll.setRevolutions(0.3);
      DoubleMatrix2D data = swissRoll.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      int batchSize = 5;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         SVDPCA svdPca = new SVDPCA(data.viewPart(i * batchSize, 0,
                 Math.min(batchSize, data.rows() - (i * batchSize)), 3));
         pca.merge(svdPca);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, 1e-2);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      System.out.println("Eigenvectors:");
      System.out.println(pca.getEigenvectors());

      System.out.println("Meanvector:");
      System.out.println(pca.getMean());

      outputEigenspaceToFile("curved-plane-hmm-merge.dat", data, pca);
   }

   @Test
   public void testInclinedPlaneIncrementalExact() throws IOException {
      System.out.println("testInclinedPlaneIncrementalExact");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomGenerator(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(3, 3);
      for (int i = 0; i < data.rows(); ++i) {
         pca.addInput((DenseDoubleMatrix2D) data.viewPart(i, 0, 1, data.columns()), 1, null);
      }

      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testInclinedPlaneMergeBatchExact() throws IOException {
      System.out.println("testInclinedPlaneMergeBatchExact");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomGenerator(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(3, 3);
      int batchSize = 5;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         SVDPCA svdPca = new SVDPCA(data.viewPart(i * batchSize, 0,
                 Math.min(batchSize, data.rows() - (i * batchSize)), 3));
         pca.merge(svdPca);
      }

      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testInclinedPlaneMergeIncrementalExact() throws IOException {
      System.out.println("testInclinedPlaneMergeIncrementalExact");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomGenerator(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(3, 3);
      int batchSize = 1;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         SVDPCA svdPca = new SVDPCA(data.viewPart(i * batchSize, 0,
                 Math.min(batchSize, data.rows() - (i * batchSize)), 3));
         pca.merge(svdPca);
      }

      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModelAssert.assertEquals(expectedPCA, pca, tolerance);
   }

   @Test
   public void testInclinedPlaneIncrementalReducedDimension() throws IOException {
      System.out.println("testInclinedPlaneIncrementalReducedDimension");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomGenerator(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      for (int i = 0; i < data.rows(); ++i) {
         pca.addInput((DenseDoubleMatrix2D) data.viewPart(i, 0, 1, data.columns()), 1, null);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, tolerance);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      System.out.println("Eigenvectors:");
      System.out.println(pca.getEigenvectors());

      System.out.println("Meanvector:");
      System.out.println(pca.getMean());

      outputEigenspaceToFile("inclined-plane-hmm.dat", data, pca);
   }

   @Test
   public void testInclinedPlaneMergeReducedDimension() throws IOException {
      System.out.println("testInclinedPlaneMergeReducedDimension");
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});

      InclinedPlane3D inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomGenerator(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
      DoubleMatrix2D data = inclinedPlane.generate(100);

      HallMarshallMartin pca = new HallMarshallMartin(2, 3);
      int batchSize = 5;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         SVDPCA svdPca = new SVDPCA(data.viewPart(i * batchSize, 0,
                 Math.min(batchSize, data.rows() - (i * batchSize)), 3));
         pca.merge(svdPca);
      }
      
      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, 1e-2);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      System.out.println("Eigenvectors:");
      System.out.println(pca.getEigenvectors());

      System.out.println("Meanvector:");
      System.out.println(pca.getMean());

      outputEigenspaceToFile("inclined-plane-hmm-merge.dat", data, pca);      
   }

   @Test
   public void testFilmDataIncrementalPCA() throws IOException {

      int totalFilms = 25;
      int totalFields = 100000;
      Map<Integer, String> filmTitles = new HashMap<>();
      DoubleMatrix2D spatialWeights = new SparseDoubleMatrix2D(totalFilms, totalFields);
      DoubleMatrix2D data = new SparseDoubleMatrix2D(totalFilms, totalFields);
      List<Integer> filmIds = loadFilmData(totalFilms, data, spatialWeights, filmTitles);
      
      // Segment the data into individual matrices for performance
      Map<Integer, DoubleMatrix2D> dataRows = segmentData(filmIds, data);

      // Perform the PCA
      HallMarshallMartin pca = new HallMarshallMartin(2, totalFields);
      for (int i = 0; i < data.rows(); ++i) {
         //pca.addInput(dataRows.get(i), 1, spatialWeights.viewRow(i));
         pca.addInput(dataRows.get(filmIds.get(i)), 1, null);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, tolerance);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      // Output coefficients to file (can be read by GNU Plot)      
      outputFilmEigenspaceToFile("film-data-hmm-incremental.dat", data, pca, filmIds, filmTitles);
   }

   @Test
   public void testFilmDataMergeIncrementalPCA() throws IOException {

      int totalFilms = 1000;
      int totalFields = 100000;
      Map<Integer, String> filmTitles = new HashMap<>();
      DoubleMatrix2D spatialWeights = new SparseDoubleMatrix2D(totalFilms, totalFields);
      SparseDoubleMatrix2D data = new SparseDoubleMatrix2D(totalFilms, totalFields);
      List<Integer> filmIds = loadFilmData(totalFilms, data, spatialWeights, filmTitles);

      // Segment the data into individual matrices for performance
      Map<Integer, DoubleMatrix2D> dataRows = segmentData(filmIds, data);

      // Perform the PCA
      HallMarshallMartin pca = new HallMarshallMartin(2, totalFields);
      int batchSize = 1;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         HallMarshallMartin subPca = new HallMarshallMartin(2, totalFields);
         for (int j = 0; j < Math.min(batchSize, data.rows() - (i * batchSize)); ++j) {
            int row = i * batchSize + j;
            subPca.addInput(dataRows.get(filmIds.get(row)), 1, null);
         }
         pca.merge(subPca);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, tolerance);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());
   }

   @Test
   public void testFilmDataMergeBatchedPCA() throws IOException {

      int totalFilms = 25;
      int totalFields = 100000;
      Map<Integer, String> filmTitles = new HashMap<>();
      DoubleMatrix2D spatialWeights = new SparseDoubleMatrix2D(totalFilms, totalFields);
      SparseDoubleMatrix2D data = new SparseDoubleMatrix2D(totalFilms, totalFields);
      List<Integer> filmIds = loadFilmData(totalFilms, data, spatialWeights, filmTitles);

      // Segment the data into individual matrices for performance
      Map<Integer, DoubleMatrix2D> dataRows = segmentData(filmIds, data);

      // Perform the PCA
      HallMarshallMartin pca = new HallMarshallMartin(2, totalFields);
      int batchSize = 5;
      int totalBatches = (data.rows() + batchSize - 1) / batchSize;
      for (int i = 0; i < totalBatches; ++i) {
         HallMarshallMartin subPca = new HallMarshallMartin(2, totalFields);
         for (int j = 0; j < Math.min(batchSize, data.rows() - (i * batchSize)); ++j) {
            int row = i * batchSize + j;
            subPca.addInput(dataRows.get(filmIds.get(row)), 1, null);
         }
         pca.merge(subPca);
      }

      // Ensure that the eigenvectors are orthogonal
      DoubleMatrix2D eigenvectors = pca.getEigenvectors();
      DoubleMatrix2D eigenvectorsCrossProduct = eigenvectors.like(eigenvectors.rows(), eigenvectors.rows());
      SparseUtils.zMult((SparseDoubleMatrix2D) eigenvectors, eigenvectors, eigenvectorsCrossProduct, false, true);
      MatrixAssert.assertEquals(DoubleFactory2D.dense.identity(eigenvectors.rows()), eigenvectorsCrossProduct, tolerance);

      System.out.println("Eigenvalues:");
      System.out.println(pca.getEigenvalues());

      outputFilmEigenspaceToFile("film-data-hmm-merge.dat", data, pca, filmIds, filmTitles);
   }

    private List<Integer> loadFilmData(int totalFilms, DoubleMatrix2D data, DoubleMatrix2D spatialWeights, Map<Integer, String> itemIdToLabel) throws IOException {

      HashMap<Integer, Integer> itemIdToIndex = new HashMap<>();
      List<Integer> ids = new LinkedList<>();      
      URL labelFile = this.getClass().getResource("item-titles.dat");

       try ( // Read the label file
               LineNumberReader labelReader = new LineNumberReader(new InputStreamReader(labelFile.openStream()))) {
           Pattern labelLinePattern = Pattern.compile("^(\\d+)(\\s+)(.*)$");
           for (String dataLine = labelReader.readLine();
                   dataLine != null && itemIdToLabel.size() < totalFilms;
                   dataLine = labelReader.readLine()) {
               Matcher matcher = labelLinePattern.matcher(dataLine);
               if (matcher.matches()) {
                   try {
                       int itemId = Integer.parseInt(matcher.group(1));
                       String label = matcher.group(3);
                       itemIdToLabel.put(itemId, label);
                   } catch (NumberFormatException ex) {
                       System.err.println("Line " + labelReader.getLineNumber() + " could not be parsed in file " + labelFile);
                   }
               }
           }
           totalFilms = labelReader.getLineNumber() - 1;
       }

      spatialWeights.ensureCapacity(totalFilms * 100);
      URL dataFile = this.getClass().getResource("item-vector.dat");
      LineNumberReader reader = new LineNumberReader(new InputStreamReader(dataFile.openStream()));
      Pattern dataLinePattern = Pattern.compile("^(\\S+)(\\s+)(\\S+)(\\s+)(\\S+)(\\s+)(\\S+)");
      for (String dataLine = reader.readLine(); dataLine != null; dataLine = reader.readLine()) {
         Matcher matcher = dataLinePattern.matcher(dataLine);
         if (matcher.matches()) {
            try {
               int itemId = Integer.parseInt(matcher.group(1));
               int fieldId = Integer.parseInt(matcher.group(3));
               double value = Double.parseDouble(matcher.group(5));
               double weight = Double.parseDouble(matcher.group(7));

               if (fieldId == 80043) {
                  continue;
               }

               Integer itemIndex = itemIdToIndex.get(itemId);
               if (itemIndex == null) {
                  if (itemIdToIndex.size() == totalFilms) {
                     break;
                  }
                  itemIndex = ids.size();
                  itemIdToIndex.put(itemId, itemIndex);
                  ids.add(itemId);
               }

               data.setQuick(itemIndex, fieldId, fieldId != 80043 ? value : (value - 13));
               spatialWeights.setQuick(itemIndex, fieldId, fieldId != 80043 ? weight : 0.4);
            } catch (NumberFormatException ex) {
               System.err.println("Line " + reader.getLineNumber() + " could not be parsed.");
            }
         }
      }

      return ids;
   }

   private Map<Integer, DoubleMatrix2D> segmentData(final List<Integer> ids, DoubleMatrix2D data) {
      final Map<Integer, DoubleMatrix2D> rows = new HashMap<>();
      ids.stream().forEach((id) -> {
          rows.put(id, data.like(1, data.columns()));
       });
      data.forEachNonZero((int first, int second, double value) -> {
          DoubleMatrix2D row = rows.get(ids.get(first));
          row.setQuick(0, second, value);
          return value;
      });

      return rows;
   }

   private void outputEigenspaceToFile(String fileName, DoubleMatrix2D data, EigenspaceModel pca) throws FileNotFoundException {
      // Truncate the SVD and calculate the coefficient matrix
      DenseDoubleMatrix2D coefficients = new DenseDoubleMatrix2D(data.rows(), 2);
      DoubleMatrix2D centeredInput = data.copy();
      for (int i = 0; i < data.rows(); ++i) {
         centeredInput.viewRow(i).assign(pca.getMean().viewRow(0), Functions.minus);
      }
      centeredInput.zMult(pca.getEigenvectors().viewPart(0, 0, 2, 3), coefficients, 1, 0, false, true);

      // Reconstruct the data from the lower dimensional information
      DoubleMatrix2D reconstruction = data.copy();
      for (int i = 0; i < reconstruction.rows(); ++i) {
         reconstruction.viewRow(i).assign(pca.getMean().viewRow(0));
      }
      coefficients.zMult(pca.getEigenvectors().viewPart(0, 0, 2, 3), reconstruction, 1, 1, false, false);

      // Output coefficients to file (can be read by GNU Plot)
      String packagePath = this.getClass().getPackage().getName().replaceAll("\\.", "/");
      File outputFile = new File("src/test/resources/"
              + packagePath
              + "/" + fileName);
       try (PrintWriter writer = new PrintWriter(outputFile)) {
           for (int i = 0; i < reconstruction.rows(); ++i) {
               StringBuffer outputLine = new StringBuffer();
               outputLine.append(reconstruction.getQuick(i, 0));
               outputLine.append(' ');
               outputLine.append(reconstruction.getQuick(i, 1));
               outputLine.append(' ');
               outputLine.append(reconstruction.getQuick(i, 2));
               outputLine.append(" 0"); // Color index
               writer.println(outputLine);
           }
           for (int i = 0; i < data.rows(); ++i) {
               StringBuffer outputLine = new StringBuffer();
               outputLine.append(data.getQuick(i, 0));
               outputLine.append(' ');
               outputLine.append(data.getQuick(i, 1));
               outputLine.append(' ');
               outputLine.append(data.getQuick(i, 2));
               outputLine.append(" 10"); // Color index
               writer.println(outputLine);
           }}
   }

   private void outputFilmEigenspaceToFile(String fileName, DoubleMatrix2D data, HallMarshallMartin pca, List<Integer> filmIds, Map<Integer, String> filmTitles) throws FileNotFoundException {
      // Output coefficients to file (can be read by GNU Plot)
      String packagePath = this.getClass().getPackage().getName().replaceAll("\\.", "/");
      File outputFile = new File("src/test/resources/"
              + packagePath
              + "/" + fileName);
       try (PrintWriter writer = new PrintWriter(outputFile)) {
           Map<Integer, DoubleMatrix2D> dataRows = segmentData(filmIds, data);
           filmIds.stream().map((filmId) -> {
               DoubleMatrix2D coefficients = pca.getEigenbasisCoordinates(dataRows.get(filmId));
               String itemLabel = filmTitles.get(filmId);
               StringBuffer outputLine = new StringBuffer();
               outputLine.append(coefficients.getQuick(0, 0));
               outputLine.append(' ');
               outputLine.append(coefficients.getQuick(0, 1));
               if (itemLabel != null) {
                   outputLine.append(' ');
                   outputLine.append(itemLabel);
               }
              return outputLine;
          }).forEach((outputLine) -> {
              writer.println(outputLine);
          });
}
   }
}
