package com.trickl.pca;

import com.trickl.pca.SVDPCA;
import com.trickl.pca.EigenspaceModel;
import com.trickl.pca.Basis;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.random.engine.MersenneTwister;
import com.trickl.dataset.InclinedPlane3D;
import java.awt.Rectangle;
import java.io.IOException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class MergeTest {

   private static double tolerance = 1e-7;

   private InclinedPlane3D inclinedPlane;

   public MergeTest() {
   }

   @Before
   public void setUp() {
      DoubleMatrix1D normal = new DenseDoubleMatrix1D(3);
      normal.assign(new double[]{.0, .0, 1.0});
      inclinedPlane = new InclinedPlane3D();
      inclinedPlane.setRandomEngine(new MersenneTwister(123456789));
      inclinedPlane.setNormal(normal);
      inclinedPlane.setBounds(new Rectangle(-5, -5, 10, 10));
      inclinedPlane.setNoiseStd(0.5);
   }

   @After
   public void tearDown() {
   }

   @Test
   public void testInclinedPlaneNoNewDimensions() throws IOException {

      int lhsObservationCount = 4;
      int rhsObservationCount = 6;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneNoNewDimensionsReverse() throws IOException {

      int lhsObservationCount = 4;
      int rhsObservationCount = 6;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.viewPart(0, 0, lhsObservationCount, 3).assign(inclinedPlane.generate(lhsObservationCount));
      data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3).assign(inclinedPlane.generate(rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(rhsPca, lhsPca);
      
      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneNoNewDimensionsLarge() throws IOException {

      int lhsObservationCount = 5;
      int rhsObservationCount = 5;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneTwoNewDimension() throws IOException {

      int lhsObservationCount = 2;
      int rhsObservationCount = 2;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      // CHECK
      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneIdentityLhs() throws IOException {

      int lhsObservationCount = 0;
      int rhsObservationCount = 2;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneIdentityRhs() throws IOException {

      int lhsObservationCount = 2;
      int rhsObservationCount = 0;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneMinimal() throws IOException {

      int lhsObservationCount = 1;
      int rhsObservationCount = 1;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }

   @Test
   public void testInclinedPlaneTrivial() throws IOException {

      int lhsObservationCount = 0;
      int rhsObservationCount = 0;
      DoubleMatrix2D data = new DenseDoubleMatrix2D(lhsObservationCount + rhsObservationCount, 3);
      data.assign(inclinedPlane.generate(lhsObservationCount + rhsObservationCount));

      SVDPCA lhsPca = new SVDPCA(data.viewPart(0, 0, lhsObservationCount, 3));
      SVDPCA rhsPca = new SVDPCA(data.viewPart(lhsObservationCount, 0, rhsObservationCount, 3));
      SVDPCA expectedPCA = new SVDPCA(data);

      EigenspaceModel mergedPCA = Basis.merge(lhsPca, rhsPca);

      EigenspaceModelAssert.assertEquals(expectedPCA, mergedPCA, tolerance);
   }
}