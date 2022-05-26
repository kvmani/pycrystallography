#!/usr/bin python
# -*- coding: utf-8 -*-
"""
This file is part of the pycrystallography python module

Author:         Mani Krishna K V
Website:        
Documentation:  

Version:         1.0.0
License:         The MIT License (MIT)

Copyright (c) 2017 Mani Krishna 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

test_plane.py - Unit test for plane module

"""

# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import collections
from pymatgen.util.testing import PymatgenTest
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation

import numpy as np
import copy
from math import pi, sqrt
#from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class MillerPlaneCreationTestCases(PymatgenTest):
    def setUp(self):

        lat = olt.cubic(2)
        
        self.plane1 = MillerPlane(lattice = lat, hkl = [1,1,1]) 
        self.cubeLat = olt.cubic(2)
        self.cubeLat2   = olt.cubic(1)
        self.cubeLat3   = olt.cubic(1,orientation=Orientation(axis=[0,1,0],degrees=90))
        self.orthoLat = olt.orthorhombic(2, 1, 1)
        self.hexLat = olt.hexagonal(1, 1.63)
        self.hexLat2 = olt.hexagonal(3.2, 1.59*3.2) ## Zr 
        #self.monoClinicLat = olt.from_parameters(4,6,5,120,90,120)
       # self.genLat = olt.from_lengths_and_angles([3,4,6],[90,90,120])
                 

    def test_init(self):
        hexLat = olt.hexagonal(1, 1.63)                
        hkil = [2.,-1.,-1.,0.]
        plane1 =MillerPlane(lattice=hexLat, hkl = hkil, MillerConv='Bravais') 
        self.assertIsNotNone(plane1, "Initialization of MillerPlane object using MillerDirection object failed")
                       
        hkil = [0,0,1.]
        plane1 =MillerPlane(lattice=self.cubeLat, hkl = hkil, isCartesian=True,recLattice=self.cubeLat.reciprocal_lattice_crystallographic) 
        print("{:2d}".format(plane1))
        self.assertIsNotNone(plane1, "Initialization of MillerPlane object using MillerDirection object failed")
        
        
        hkil = [0,1,1.]
        plane1 =MillerPlane(lattice=self.cubeLat, hkl = hkil, isCartesian=True,recLattice=self.cubeLat.reciprocal_lattice_crystallographic) 
        print("{:2d}".format(plane1))
        self.assertIsNotNone(plane1, "Initialization of MillerPlane object using MillerDirection object failed")
    
        hexLat = olt.hexagonal(1, 1.63)                
        hkil = [1,0,0]
        plane1 =MillerPlane(hkl = hkil,isCartesian=True,recLattice=self.hexLat.reciprocal_lattice_crystallographic) 
        print("{:2d}".format(plane1)) 
        self.assertIsNotNone(plane1, "Initialization of MillerPlane object using MillerDirection object failed")
     
        hkil = [0,-1,0]
        plane1 =MillerPlane(hkl = hkil,isCartesian=True,recLattice=self.hexLat.reciprocal_lattice_crystallographic) 
        print("{:2d}".format(plane1)) 
        self.assertIsNotNone(plane1, "Initialization of MillerPlane object using MillerDirection object failed")
     
    
    def test_fromNormalAndPoint(self):
#         plane1 = MillerPlane.fromNormalAndPoint(normal=[1,1,0], point=[1,0,0],lattice=self.cubeLat)
#         self.assertIsNotNone(plane1,"Problem in creation with normal and point")
        plane1 = MillerPlane.fromNormalAndPoint(normal=[0,0,1], point=[0,0,0],lattice=self.cubeLat)
        print(plane1)
        self.assertIsNotNone(plane1,"Problem in creation with normal and point")
        
        
        
    
class MillerPlaneOutputCases(MillerPlaneCreationTestCases):
    def test_getLatexString(self):
        hkl = [1,0,0.]
        plane1 = MillerPlane(lattice=self.orthoLat,hkl=hkl)
        
        latexString = plane1.getLatexString()
        self.assertEqual(r"$\{ 1 0 0 \}$", latexString, 'Latex string conversion problem!!!')
        
        
        plane2 = MillerPlane(lattice=self.hexLat, hkl = [2,-1,-1,0],MillerConv='MillerBravais' )
        #print("The plane hkl is ", plane2.gethkl())
        latexString = plane2.getLatexString()
        
        self.assertEqual(r"$\{ 2 \bar{1} \bar{1} 0 \}$", latexString, 'Latex string conversion problem!!!')
    def test_format(self):
        hkl = [1,0,0.]
        plane1 = MillerPlane(lattice=self.cubeLat,hkl=hkl)
        strFormat = "{:d}".format(plane1)
        #print("The format string is ", strFormat)
        self.assertEqual('hkl = ( 1  0  0)', strFormat,'Error in __format__ function')
        
        hkl = [1,0,2.]
        plane1 = MillerPlane(lattice=self.hexLat2,hkl=hkl)
        strFormat = "{:d}".format(plane1)
        #print("The format string is ", strFormat)
        self.assertEqual('hkl = ( 1  0 -1  2)', strFormat,'Error in __format__ function')
  
    def test_repr(self):
        hkl = [1,0,0.]
        plane1 = MillerPlane(lattice=self.cubeLat,hkl=hkl)
        strFormat = repr(plane1)
        #print("The repr string is ", strFormat)
        self.assertEqual('hkl = (1.00 0.00 0.00)  lattice = cubic', strFormat,'Error in __format__ function')
        
        hkl = [1,-1.0,0.,2.]
        plane1 = MillerPlane(lattice=self.hexLat,hkl=hkl)
        strFormat = repr(plane1)
        #print("The repr string is ", strFormat)
        self.assertEqual(u'hkl = (1.00 -1.00 0.00 2.00)  lattice = hexagonal', strFormat,'Error in __format__ function')


        
        
        
    
class MillerPlaneArithmeticCases(MillerPlaneCreationTestCases):
    
    def test_angle(self):
        hkl = [1,0,0.]
        plane1 = MillerPlane(lattice=self.orthoLat,hkl=hkl)        
        plane2 = MillerPlane(lattice=self.orthoLat,hkl = [1,1,0])        
        
        ang1 = plane1.angle(plane2,units='Deg')
        ang2 = plane2.angle(plane1,units='Deg')        
        self.assertAlmostEqual(ang1, ang2)             
                
        plane3 = MillerPlane(lattice=self.hexLat, hkl = [2,-1,-1,0]) 
        plane4 = MillerPlane(lattice=self.hexLat, hkl = [0,1,0]) 
                   
        ang3 = plane3.angle(plane4, units="degree") 
        

        self.assertAlmostEqual(ang3, 90., msg="Problem in angle calcualtion")
        
#         plane5 = MillerPlane(lattice = self.monoClinicLat, hkl = [1,0,1])
#         plane6 = MillerPlane(lattice = self.monoClinicLat, hkl = [-2,0,1])        
#         ang4 = plane5.angle(plane6,units='Deg')
#         self.assertAlmostEqual(ang4, 130.25,1,msg="Problem in angle calcualtion")
#         
        plane3 = MillerPlane(lattice=self.hexLat, hkl = [1,0,-1,0]) 
        plane4 = MillerPlane(lattice=self.hexLat, hkl = [2,-1,-1,2]) 
        ang3 = plane3.angle(plane4, units="degree") 
        print("angle",ang3)
   
    def test_dspacing(self):
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [1,1,0])
        self.assertAlmostEqual(plane1.dspacing,1.414,3, msg='Problem in dpsacing')
        
        plane2 = MillerPlane(lattice=self.hexLat,hkl = [0,0,0,1])
        self.assertAlmostEqual(plane2.dspacing,1.63,3, msg='Problem in dpsacing')
        
    def test_getPointInPlane(self):
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [1,1,0]) 
        point = plane1.getPointInPlane()
        
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [2,2,-4,0])
        point2 = plane2.getPointInPlane()
        print("The point is ", point2)
        
        plane3 = MillerPlane(lattice = self.cubeLat,hkl = [1,1,1])
        point3 = plane3.getPointInPlane()
        print("The point is ", point3)
        
        
    def test_Integerize(self):
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [0.99, 1.01,  3], MillerConv='Bravais' )
        [hkl, error] = plane1.integerize(AngleDeviationUnits='Deg') 
        self.assertArrayAlmostEqual(hkl, [1, 1, -2 ,3], decimal=1, err_msg='probelm in integerize')  
        
        cubicLat = olt.cubic(1)
        hkl = [-0.25,-0.5,0.75]
        plane2 = MillerDirection(lattice=cubicLat, vector = hkl)
        [plane, AngleError] = MillerPlane.integerize(plane2)
        self.assertArrayAlmostEqual(plane, [-1.0, -2.0, 3.0]) 
        
    def test_getPlaneNormal(self):
        plane1 = MillerPlane(lattice = self.hexLat2,hkl = [0,-1,1,1], MillerConv='Bravais' )
        planeNormal = plane1.getPlaneNormal()
        testPlane = MillerDirection(lattice=self.hexLat2,vector=[0,-1,1,0.591098])
        self.assertArrayAlmostEqual(planeNormal.getCartesianVec(), testPlane.getUnitVector(), 3, 'Problem in planeNormal conversion', '')
    
    def test_structureFactor(self):
        ## below is Zr hcp data
        atomData = [(40, np.array([0.,0.,0.])),
            (40, np.array([1./3., 2./3.,1./2.]))]
        
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [0,0,0,1], MillerConv='Bravais' )
        [real, imag] = plane1.structureFactor(atomData=atomData)
        self.assertArrayAlmostEqual([real, imag], [0., 0.],7,'Problem in Structure factor calcualtion')
       
       
   
    def test_diffractionIntensity(self):
        cubicAtomData = [(1, np.array([0.,0.,0.])),
                         (1, np.array([1./2,1./2,1./2.]))] ### bcc imaginary data
        
        cubicLat = olt.cubic(1)
        hkl = [1,0,0]
        plane2 = MillerPlane(lattice=cubicLat, hkl = hkl)
        Int = plane2.diffractionIntensity(atomData=cubicAtomData)
        self.assertAlmostEqual(Int, 0.,7,'Problem in diffraction intensity calcualtion')
        
        
        hcpAtomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))] ####typical Zr data
        
        hexLat = olt.hexagonal(1, 1.63)
        hkl = [1,0,-1,0]
        plane = MillerPlane(lattice=hexLat, hkl = hkl)
        Int = plane.diffractionIntensity(atomData=hcpAtomData)
        self.assertAlmostEqual(Int, 0.,7,'Problem in diffraction intensity calcualtion')
       
       
        
 
            
        
    def test_kikuchiLinePoints(self):
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [2,-1,-1,0], MillerConv='Bravais' )
        Xaxis=[1,0,0]
        Yaxis=[0,1,0]
        scalingFactor = 1000*(558.94/578.31)
        patterCenterX = 877
        patterCenterY = 935
        
        kikuchPoints = plane1.kikuchiLinePoints(Xaxis,Yaxis,scalingFactor,
                                 patterCenterX,patterCenterY,
                                 detectorScaleFactor=10000)
        
        
        
    def test_symmetricSet(self):
        plane1 = MillerPlane(lattice=self.cubeLat, hkl = [1,2,3], MillerConv="Miller")
        symSet = plane1.symmetricSet()
        self.assertEqual(len(symSet),24)
        
        plane1 = MillerPlane(lattice=self.cubeLat, hkl = [1,1,-1], MillerConv="Miller")
        symSet = plane1.symmetricSet()
        self.assertEqual(len(symSet),8)
                
        plane1 = MillerPlane(lattice=self.cubeLat, hkl = [0,1,0], MillerConv="Miller")
        symSet = plane1.symmetricSet()
        for  i in symSet:
            print(str(i))
        self.assertEqual(len(symSet),6)
        
        vec = [1,1,-2,3]
        plane1 = MillerPlane(lattice=self.hexLat2, hkl = vec)
        symSet = plane1.symmetricSet()
        self.assertEqual(len(symSet),12)
        
        orthoLat = olt.orthorhombic(1, 2, 3, orientation=Orientation(euler=[0.,0,0]))
        vec = [1,1,1]
        plane1 = MillerPlane(lattice=orthoLat, hkl = vec)
        symSet = plane1.symmetricSet()
        self.assertEqual(len(symSet),4)
        
        orthoLat = olt.orthorhombic(1, 2, 3, orientation=Orientation(euler=[0.,0,0]))
        vec = [1,0,0]
        plane1 = MillerPlane(lattice=orthoLat, hkl = vec)
        symSet = plane1.symmetricSet()
        self.assertEqual(len(symSet),2)
   
    def test_isSymmetric(self):    
        plane1 = MillerPlane(lattice=self.cubeLat, hkl=[1,1,1])
        plane2 = MillerPlane(lattice=self.cubeLat, hkl=[1,1,-1])
        sym=plane1.isSymmetric(plane2)
        self.assertEqual(sym,True,'Error in isSymmetric method')
        
        plane3 = MillerPlane(lattice=self.cubeLat, hkl=[1,0,-1])
        sym=plane1.isSymmetric(plane3)
        self.assertEqual(sym,False,'Error in isSymmetric method')
        
        plane1 = MillerPlane(lattice=self.hexLat, hkl=[2,-1,-1,0])
        plane2 = MillerPlane(lattice=self.hexLat, hkl=[-1,2,-1,0])
        sym=plane1.isSymmetric(plane2)
        self.assertEqual(sym,True,'Error in isSymmetric method')
        
        plane3 = MillerPlane(lattice=self.hexLat, hkl=[0,0,0,1])
        sym=plane1.isSymmetric(plane3)
        self.assertEqual(sym,False,'Error in isSymmetric method')
        
        
    def test_getZoneAxis(self):
        
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [2,0,-2,1], MillerConv='Bravais' )
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [-2,1,1,-2], MillerConv='Bravais' )        
        zoneAxis = MillerPlane.getZoneAxis(plane1,plane2)
        print("The zoneAxis is ", zoneAxis)
        print(zoneAxis.getUVW())
        [axis, error] = zoneAxis.integerize()
        self.assertArrayAlmostEqual(axis,[-4,  5, -1,  6],2,'Problem in Zone Axis calcualtions','')
        zoneAxis = MillerPlane.getZoneAxis(plane1,plane2,returnIntegerZoneAxis=True)
        self.assertArrayAlmostEqual(zoneAxis.getUVW(),[-4,  5, -1,  6],2,'Problem in Zone Axis calcualtions','')
        
        
        
           
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [0,-1,1,0], MillerConv='Bravais' )
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [1,-1,0,1], MillerConv='Bravais' )        
        zoneAxis = MillerPlane.getZoneAxis(plane1,plane2)
        print("The zoneAxis is ", zoneAxis)
        print(zoneAxis.getUVW())
        [axis, error] = zoneAxis.integerize()
        self.assertArrayAlmostEqual(axis,[-2,  1,  1,  3],2,'Problem in Zone Axis calcualtions','')
        zoneAxis = MillerPlane.getZoneAxis(plane1,plane2,returnIntegerZoneAxis=True)
        self.assertArrayAlmostEqual(zoneAxis.getUVW(),[-2,  1,  1,  3],2,'Problem in Zone Axis calcualtions','')
           
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [1,0,-1,0], MillerConv='Bravais' )
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [0,1,-1,0], MillerConv='Bravais' )        
        zoneAxis = MillerPlane.getZoneAxis(plane1,plane2)
        print("The zoneAxis of pattern is is ", zoneAxis)
        print(zoneAxis.getUVW())
        [axis, error] = zoneAxis.integerize()
        self.assertArrayAlmostEqual(axis,[0,0,0,1],2,'Problem in Zone Axis calcualtions','')
         
    def test_uniqueList(self):
        plane1 = MillerPlane(lattice = self.hexLat,hkl = [2,-1,-1,0], MillerConv='Bravais' )
        totalSet = plane1.symmetricSet()
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [0,1,-1,0], MillerConv='Bravais' ) 
        plane2Set = plane2.symmetricSet()
        for plane in plane2Set :
            totalSet.append(plane)
        totalSet.append(plane1)
        totalSet.append(plane2)        
        
        uniqueSet = MillerPlane.uniqueListFast(totalSet, considerSymmetry=True)
        self.assertEqual(len(uniqueSet), 2)
        
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [1,1,1] )
        totalSet = plane1.symmetricSet()
        totalSet[4:]=[]
        plane2 = MillerPlane(lattice = self.cubeLat,hkl = [0,1,0] ) 
        plane2Set = plane2.symmetricSet()
        for plane in plane2Set :
            totalSet.append(plane)
        totalSet.append(MillerPlane(lattice = self.cubeLat,hkl = [1,2,3]))        
        uniqueSet = MillerPlane.uniqueList(totalSet, considerSymmetry=True)
        self.assertEqual(len(uniqueSet), 3)
        
    def test_generatePlaneList(self):
        
        hexPlanesList = MillerPlane.generatePlaneList(6, self.cubeLat)
        print(hexPlanesList)
        self.assertEqual(len(hexPlanesList), 5, 'random error ignore it for the time being')
        

    def test_getKikuchiLine(self):
        plane1 = MillerPlane(lattice=self.cubeLat, hkl=[1,1,1])
        
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])##
        detectorCoordinates[:,2]=-0.1
        crystalOri=Orientation(euler=[0.,0.,0.])
        points = plane1.getKikuchiLine(crystalOri,detectorCoordinates)
        
        self.assertArrayAlmostEqual(points[0].evalf(), [1.,-0.9000,-0.1], 3, "kikuchi points not correct")
        plane2 = MillerPlane(lattice=self.cubeLat, hkl=[0,0,1])
        points = plane2.getKikuchiLine(crystalOri,detectorCoordinates)
        
        self.assertEqual(len(points), 0, "More than 0 points are retuned")
        
        plane1 = MillerPlane(lattice=self.cubeLat, hkl=[1,1,1])
        
        crystalOri=Orientation(euler=[45*np.pi/180.,0.,0.])
        points = plane1.getKikuchiLine(crystalOri,detectorCoordinates)
        
        self.assertArrayAlmostEqual(points[0].evalf(), [0.07071, -1.000, -0.10], 3, "kikuchi points not correct")
        plane2 = MillerPlane(lattice=self.cubeLat, hkl=[0,0,1])
        points = plane2.getKikuchiLine(crystalOri,detectorCoordinates)
        
        self.assertEqual(len(points), 0, "More than 0 points are retuned")
       
    def test_rotate(self):
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [2,0,0]) 
        ori = Orientation(axis=[0,0,1],degrees=90)
        plane1.rotate(ori.inverse)
        self.assertArrayAlmostEqual(plane1.gethkl(),[0,-2,0],5,"Probelm in rotation")
        
        plane2 = MillerPlane(lattice = self.hexLat,hkl = [2,-1,-1,0])
        ori2 = Orientation(axis=[0,0,1],degrees=120)
        plane2.rotate(ori2.inverse)
        plane2.__str__()
        self.assertArrayAlmostEqual(plane2.gethkl(),[-1,-1,2,0],5,"Probelm in rotation")
        
        plane3 = MillerPlane(lattice = self.hexLat,hkl = [2,-1,-1,0])
        ori2 = Orientation(axis=[0,0,1],degrees=30)
        plane3.rotate(ori2.inverse)
        
        self.assertArrayAlmostEqual(plane3.gethkl(),[1.73205,-1.73205,0,0],3,"Probelm in rotation")
    
    def test_steriographiProjection(self):        
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [2,0,0]) 
        ori = Orientation(axis=[0,1,0],degrees=90)
        [point, polar, isNorth] = plane1.steriographiProjection(ori)
        self.assertArrayAlmostEqual(point, [0,0],5,"Probelem in sterio")
        
        plane2 = MillerPlane(lattice = self.cubeLat,hkl = [2,0,0])        
        [point, polar, isNorth] = plane2.steriographiProjection()
        self.assertArrayAlmostEqual(point, [1,0],5,"Probelem in sterio")

        plane3 = MillerPlane(lattice = self.hexLat,hkl = [2,-1,-1,0])        
        [point, polar, isNorth] = plane3.steriographiProjection(ori)
        self.assertArrayAlmostEqual(point, [0,0],5,"Probelem in sterio")
    
        
    def test_getPlaneInOtherLatticeFrame(self):        
        plane1 = MillerPlane(lattice = self.cubeLat,hkl = [2,0,0]) 
        plane2 = plane1.getPlaneInOtherLatticeFrame(self.cubeLat2)
        self.assertArrayAlmostEqual([1,0,0], plane2.gethkl(), 5, 'Porblem in lattice conversion')
        plane3 = plane1.getPlaneInOtherLatticeFrame(self.cubeLat3)
        print(plane3)
        
        
             
        
        

if __name__ == '__main__':
    import unittest
    unittest.main()