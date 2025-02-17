# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals
from pymatgen.util.testing import PymatgenTest

# from pycrystallography.core.quaternion  import Quaternion
# from pycrystallography.core.millerDirection  import MillerDirection
# from pymatgen.core.lattice import Lattice

import logging
import sys
import os
import random
import math
# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
try:
    from pycrystallography.core.orientation import Orientation
except ImportError:
    logger.warning("Unable to find the pycrystallography package. Adjusting system path.")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname('..'))
    sys.path.insert(0, os.path.dirname('../../pycrystallography'))
    sys.path.insert(0, os.path.dirname('../../..'))
    for path in sys.path:
        logger.debug(f"Updated Path: {path}")

from pycrystallography.core.orientation import Orientation
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri

import numpy as np
from math import pi
from pycrystallography.utilities.pymathutilityfunctions import integerize

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class CrystalOrientationCreationTestCases(PymatgenTest):
    def setUp(self):

        self.orientation = CrysOri(orientation=Orientation(euler=[0.,0.,0.]),lattice = olt.cubic(1))
        
        self.cubeLat = olt.cubic(2)
        self.orthoLat = olt.orthorhombic(2, 1, 1)
        self.hexLat = olt.hexagonal(1, 1.63)
        self.hexLat2 = olt.hexagonal(3.2, 1.59*3.2)              

    def test_init(self):
        a = np.random.random(12).reshape(3,4)
        cubicOri1 =  CrysOri(orientation=Orientation(euler=[90*np.pi/180.,0.,0.]),lattice = olt.cubic(1))   
        #print(orientation)
        #print("the angle is   ", orientation.angle)
        self.assertIsNotNone(cubicOri1, "Initialization of Orientation fro cubic failed")  
        
        hcpOri1 = CrysOri(orientation=Orientation(axis=[1.,0.,0.], angle = 90*pi/180),lattice=self.hexLat2)
        self.assertIsNotNone(hcpOri1, "Initialization of Orientation fro hexagonal failed")              


class CrystalOrientationOutputCases(CrystalOrientationCreationTestCases):
    def test_format(self):
        cubicOri1 =  CrysOri(orientation=Orientation(euler=[90*np.pi/180.,0.,0.]),lattice = olt.cubic(1))
        strFormat = "{:2f}".format(cubicOri1)
        #print("The format string is ", strFormat)
        #self.assertEqual('hkl = ( 1  0  0)', strFormat,'Error in __format__ function')
        
        
    def test_repr(self):
        hcpOri1 = CrysOri(orientation=Orientation(axis=[1.,0.,0.], angle = 90*pi/180),lattice=self.hexLat2)
        strRepr = repr(hcpOri1)
        #print("The repr string is ", strFormat)
        #self.assertEqual('hkl = (1.00 0.00 0.00)  lattice = {0.5, 0.5, 0.5, 90.0, 90.0, 90.0}', strRepr,'Error in __format__ function')
        
class CrystalOrientationArithmenticCases(CrystalOrientationCreationTestCases):
    def test_uniqueList(self):
        degree = np.pi/180
        cubicOri1 = CrysOri(orientation=Orientation(euler=[0., 0., 0.]), lattice=olt.cubic(1))
        cubicOri2 = CrysOri(orientation=Orientation(euler=[90.*degree, 0., 0.]), lattice=olt.cubic(1))
        cubicOri3 = CrysOri(orientation=Orientation(euler=[20.*degree, 40.*degree, 20.*degree]), lattice=olt.cubic(1))
        orilList = [cubicOri1,cubicOri2,cubicOri3]
        uniqueList = CrysOri.uniqueList(orilList)
        self.assertEqual(len(uniqueList), 2, "Probelm in UniqueList method")



    def test_symmetricSet(self):
        cubicOri1 =  CrysOri(orientation=Orientation(euler=[0.,0.,0.]),lattice = olt.cubic(1))
        oriList = cubicOri1.symmetricSet()
        for i in oriList:
            print(i)
        self.assertEqual(len(oriList), 24, "Probelm in symmetric Oris")    
        
        print("hcp data")
        hcpOri1 = CrysOri(orientation=Orientation(axis=[1.,0.,0.], angle = 0*pi/180),lattice=self.hexLat2)
        oriList = hcpOri1.symmetricSet()
        for i in oriList:
            print(i)
        self.assertEqual(len(oriList), 12, "Probelm in symmetric Oris")    
    def test_projectTofundamentalZone(self):
        cubicOri1 = CrysOri(orientation=Orientation(euler=[np.pi/2., 0., 0.]), lattice=olt.cubic(1))
        fundOri, ind = cubicOri1.projectToFundamentalZone()
        self.assertArrayAlmostEqual(fundOri.getEulerAngles(units='deg',applyModulo=True), [0, 0, 0], 5, 'Problem in Fundamental Zone')
        hcpOri1 = CrysOri(orientation=Orientation(euler=[np.pi/3,0,0]),lattice=self.hexLat2)
        fundOri,ind = hcpOri1.projectToFundamentalZone()
        self.assertArrayAlmostEqual(fundOri.getEulerAngles(units='deg',applyModulo=True), [0,0,0], 5, 'Problem in Fundamental Zone')
        print("Done with the project to fundamenta zone testing")
              
    def test_isSymmetric(self):
        hcpOri1 = CrysOri(orientation=Orientation(euler=[np.pi/3,0,0]),lattice=self.hexLat2)
        hcpOri2 = CrysOri(orientation=Orientation(euler=[0.,0,0]),lattice=self.hexLat2)
        hcpOri3 = CrysOri(orientation=Orientation(euler=[np.pi/6.,0,0]),lattice=self.hexLat2)
        
        result1 = hcpOri1.isSymmetric(hcpOri2)
        result2 = hcpOri1.isSymmetric(hcpOri3)        
        
        self.assertTrue(result1,"Problem in isSymmetric")
        self.assertFalse(result2,"Problem in isSymmetric")
        
        cubicOri1 =  CrysOri(orientation=Orientation(euler=[0.,0.,0.]),lattice = olt.cubic(1))
        oriList = cubicOri1.symmetricSet()
        for i in oriList:
            self.assertTrue(cubicOri1.isSymmetric(CrysOri(i,olt.cubic(1))),"Problem in isSymmetric")
    
    def testFundamentalZones():
        # random_angles_radians = [random.uniform(0, 2 * math.pi) for _ in range(3)]
        angles_degree = [180,180,180]
        radianAngles = [i*degree for i in angles_degree]
        cubicOri1 =  CrysOri(orientation=Orientation(euler=radianAngles),lattice = olt.cubic(1))
        # cubicOri1.angle
        # oriList = cubicOri1.symmetricSet()
        print(angles_degree)
        print(cubicOri1.projectToFundamentalZone()[0])
       
if __name__ == '__main__':
    # import unittest
    # unittest.main()
    CrystalOrientationArithmenticCases.testFundamentalZones()