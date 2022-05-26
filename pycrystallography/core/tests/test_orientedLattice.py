'''
Created on 07-Dec-2017

@author: Admin
'''
from __future__ import division, unicode_literals
import collections
from pymatgen.util.testing import PymatgenTest
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.orientedLattice  import OrientedLattice


import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from copy import deepcopy
import unittest
from pycrystallography.core import orientation


class Test(PymatgenTest):


    def setUp(self):
        self.Orientation000 = Orientation(euler=[0,0,0])
        self.Orientation90_0_0 = Orientation(euler=[90*np.pi/180,0,0])
        self.Orientation0_90_0 = Orientation(euler=[0., 90*np.pi/180,0])        
        self.Orientation30_0_0 = Orientation(euler=[30*np.pi/180,0,0])
        self.cubeMatrix = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
        self.betaCif = self.cifPathName+'ZrFeBcc.cif'
        self.alphaCif = self.cifPathName+'Alpha-TiP63mmc.cif' 
    


    def tearDown(self):
        pass


    def test_init(self):
        cubeLat = OrientedLattice(matrix = self.cubeMatrix,
                                 orientation=self.Orientation90_0_0)               
         
        self.assertIsNotNone(cubeLat, "Initialization of MillerPlane object using MillerDirection object failed")
        self.assertArrayAlmostEqual(cubeLat.matrix,
                                     [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]], 4, 'base vectors not properly rotated')
        
   
    def test_cubic(self):
        cubeLat = OrientedLattice.cubic(1,orientation=self.Orientation000)
        self.assertIsNotNone(cubeLat, "Initialization of Oriented Lattice object failed for cubic lattice")
        self.assertArrayAlmostEqual(cubeLat.matrix,
                                     [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], 4, 'base vectors not properly rotated')
        print(cubeLat)
    def test_hexagonal(self):
        hexLat = OrientedLattice.hexagonal(1,1.63, orientation=self.Orientation000)
        self.assertIsNotNone(hexLat, "Initialization of Oriented Lattice object failed for hexagonal lattice")
        self.assertArrayAlmostEqual(hexLat.matrix,
                                     [[1.,0.,0.],[-0.5,0.8660,0.],[0.,0.,1.63]], 2, 'base vectors not properly rotated')
        
        hexLat = OrientedLattice.hexagonal(1,1.63, orientation=self.Orientation30_0_0)
        self.assertIsNotNone(hexLat, "Initialization of Oriented Lattice object failed for hexagonal lattice")
        self.assertArrayAlmostEqual(hexLat.matrix,
                                     [[0.866,-0.5,0.],[0., 1., 0.],[0.,0.,1.63]], 2, 'base vectors not properly rotated')
        
        hexLat = OrientedLattice.hexagonal(1,1.63, orientation=self.Orientation0_90_0)
        self.assertIsNotNone(hexLat, "Initialization of Oriented Lattice object failed for hexagonal lattice")
        self.assertArrayAlmostEqual(hexLat.matrix,
                                     [[1,0.,0.],[ -0.5, 0., -0.866],[0.,1.63,0.]], 2, 'base vectors not properly rotated')
         
    def test_fromCif(self):
        FeLat = OrientedLattice.fromCif(self.alphaCif)
        print(FeLat)
           
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()