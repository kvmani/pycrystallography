# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals
from pymatgen.util.testing import PymatgenTest
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.quaternion  import Quaternion
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane


from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
from pycrystallography.core.orientationRelation  import OrientationRelation as OriReln


import numpy as np
from math import pi
from pycrystallography.utilities.pymathutilityfunctions import integerize

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class OrientationRelationCreationTestCases(PymatgenTest):
    def setUp(self):

        self.orientation = CrysOri(orientation=Orientation(euler=[0.,0.,0.]),lattice = olt.cubic(1)) 
        self.cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
        self.betaCif = self.cifPathName+'ZrFeBcc.cif'
        self.alphaCif = self.cifPathName+'Alpha-TiP63mmc.cif' 
        self.omegaZrCif = self.cifPathName+'Zr-omega.cif'
        self.cubeLat = olt.cubic(2)
        self.orthoLat = olt.orthorhombic(2, 1, 1)
        self.hexLat = olt.hexagonal(1, 1.63)
        self.hexLat2 = olt.hexagonal(3.2, 1.59*3.2)
        
        structures = [self.betaCif,self.alphaCif]
        st,latBeta = OriReln.getStructureFromCif(self.betaCif)
        st,latAlpha = OriReln.getStructureFromCif(self.alphaCif)
        
        planes = [MillerPlane(hkl =[1,1,0],lattice=latBeta),MillerPlane(hkl=[0,0,0,1],lattice=latAlpha) ]
        directions = [MillerDirection(vector =[1,-1,1],lattice=latBeta), MillerDirection(vector=[2,-1,-1,0],lattice=latAlpha) ]        
        self.zrBetaToAlphaOR=OriReln(structures=structures,planes=planes,directions=directions)        

        #self.zrBetaToAlphaOrBeta000 =  OriReln(structures=structures,orienations=[[60,53,187],[29,10,216]])
        self.zrBetaToAlphaOrBeta000 =  OriReln(structures=structures,orienations=[[0.,0,0],[0,45.,0]])
        stBeta, latBeta = OriReln.getStructureFromCif(self.betaCif)
        stOmega, latOmegaZr = OriReln.getStructureFromCif(self.omegaZrCif)
        planes = [MillerPlane(hkl=[0, 0, 0, 1], lattice=latOmegaZr), MillerPlane(hkl=[1, 1, 1], lattice=latBeta), ]
        directions = [MillerDirection(vector=[2, -1, -1, 0], lattice=latOmegaZr),
                      MillerDirection(vector=[1, 0, -1], lattice=latBeta),
                      ]
        structures = [self.betaCif, self.omegaZrCif]
        self.zrBetaToOmega_OR = OriReln(names=["BetaZr","Omega"], structures=structures, planes=planes, directions=directions)
     

                      

    def test_init(self):
        structures = [self.betaCif,self.alphaCif]
        st,latBeta = OriReln.getStructureFromCif(self.betaCif)
        st,latAlpha = OriReln.getStructureFromCif(self.alphaCif)
        planes = [MillerPlane(hkl =[1,1,0],lattice=latBeta),MillerPlane(hkl=[0,0,0,1],lattice=latAlpha) ]
        directions = [MillerDirection(vector =[1,-1,1],lattice=latBeta), MillerDirection(vector=[2,-1,-1,0],lattice=latAlpha) ]        
        orOperator=OriReln(structures=structures,planes=planes,directions=directions,initiateVariants=True)        
        self.assertIsNotNone(orOperator, "Initialization of Orientation Rln based on Miller Plane and Dir Comb"
                                         "failed")
        print(orOperator) 
        
        orientations = [[0,0,0],[315.03,90.0,24.6822]]
        orOperator=OriReln(structures=structures,orienations=orientations,initiateVariants=True) 
        print(orOperator)       
        self.assertIsNotNone(orOperator, "Initialization of Orientation Rln based on Orientations failed")
                      

        
    def test_fromAxisAnglePair(self):
        structures = [self.betaCif,self.alphaCif]
        st,latBeta = OriReln.getStructureFromCif(self.betaCif)
        st,latAlpha = OriReln.getStructureFromCif(self.alphaCif)
        axisAngles = [(-45,[-2,1,1,0])] 
        orOperator=OriReln.fromAxisAnglePair(structures=structures,lattices = [latBeta, latAlpha],axisAnglePairs=axisAngles) 
        transData = orOperator.calcualteVariants()
        print(orOperator)
      
        self.assertIsNotNone(orOperator, "Initialization from axis Angles failed")              

       
     
        
class CrystalOrientationArithmenticCases(OrientationRelationCreationTestCases):
    def test_findVariants(self):
        orOperator = self.zrBetaToAlphaOR
        transData = orOperator.calcualteVariants()
        print(orOperator)
        
    def test_findParllelPlaneInProduct(self):
        orOperator = self.zrBetaToAlphaOR
        plane = MillerPlane(hkl=[0,0,1],lattice =orOperator.parentLattice)
        planeList = orOperator.findParallelPlaneInProduct(plane, considerAllVariants=True)        
        betaPlane1 = MillerPlane(hkl=[0,0,2],lattice=orOperator.parentLattice)
        AlphaPlanes = [MillerPlane(hkl=[1,-2,1,0],lattice=orOperator.productLattice[0])]
        AlphaPlanes.append(MillerPlane(hkl=[1,0,-1,2],lattice=orOperator.productLattice[0]))
        AlphaPlanes.append(MillerPlane(hkl=[0.5,0,-0.5,1],lattice=orOperator.productLattice[0]))
       
        print(betaPlane1, betaPlane1.dspacing)
        for plane in AlphaPlanes:
            print(f"Alpha data {plane:2d} {plane.dspacing:.3f}" )
        
        for i in planeList:
            for j in i:
                print("{:2f} {:2d} {:.3f}".format(j,j,j.dspacing))
                
    def test_initializeDiffraction(self):
        orOperator = self.zrBetaToAlphaOR
        plane = MillerPlane(hkl=[0,0,1],lattice =orOperator.parentLattice)
        planeList = orOperator.initializeDiffraction()
        print(orOperator)
    
    def test_calcualteCompositeSAED(self):

        orOperator = self.zrBetaToOmega_OR
        orOperator.calcualteVariants()
        parentZoneAxis = MillerDirection(vector=[1, 1, 0], lattice=orOperator.parentLattice)
        parallelZones = orOperator.findParlallelDirectionsInProduct(direction=parentZoneAxis)
        print("parallel Zones Are : ",parallelZones, )

        print("variant set", len(orOperator._transformationData["variantSet"][0]))
        exit(-100)

        for i in range(2):
            saedData = orOperator.calculateCompositeSAED(parentZoneAxis= parentZoneAxis, productId=0, variantId=i)
            orOperator.plotSaed(saedData=saedData)

        exit(-100)
        orOperator = self.zrBetaToAlphaOrBeta000
        plane = MillerPlane(hkl=[0,0,1],lattice =orOperator.parentLattice)   
        orOperator.calcualteVariants()
        print("variant set", len(orOperator._transformationData["variantSet"][0]))
        parentZoneAxis = MillerDirection(vector=[1,1,0],lattice=orOperator.parentLattice)
        for i in range(6):
            saedData = orOperator.calculateCompositeSAED(productId=0, variantId=i, alphaTilt=45, betaTilt=0, inPlaneRotation=0., pc=[0., 0],
                                                         sf=1., Tol=1)
            orOperator.plotSaed(saedData=saedData)
    
        
        
    
        
if __name__ == '__main__':
    import unittest
    unittest.main()
