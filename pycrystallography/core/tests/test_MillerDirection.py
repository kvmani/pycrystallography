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
from pycrystallography.core.orientation  import Orientation

import numpy as np
import copy
from math import pi, sqrt, sin, cos, radians
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class MillerDirectionCreationTestCases(PymatgenTest):
    def setUp(self):

        lat = olt.cubic(2)
        self.dir = MillerDirection(lattice=lat, vector = [1.,2.,3.], isCartesian=False)        
        lat = olt.cubic(2)
        self.dir1 = MillerDirection(lattice = lat, vector = [1,1,1]) 
        self.cubeLat = olt.cubic(2)
        self.orthoLat = olt.orthorhombic(2, 1, 1)
        self.hexLat = olt.hexagonal(1, 1.63)
        self.hexLat2 = olt.hexagonal(3.2, 1.59*3.2) ## Zr 
        self.monoClinicLat = olt.from_lengths_and_angles([4,6,5], [90,120,90])
        self.genLat = olt.from_lengths_and_angles([3,4,6],[90,90,120])            

    def test_init(self):
        hexLat = olt.hexagonal(1, 1.63,orientation=Orientation(euler=[30,0,0], units='Deg'))
                
        vec = [2.,-1.,-1.,0.]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv='MillerBravis')
        self.assertIsNotNone(dir1, "Initialization of MillerVector object list of x,y,z components failed")
                
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[0,90,0], units='Deg'))
        vec = [0,0,0,1.]
        dir1 = MillerDirection(lattice=hexLat, vector = vec)
        self.assertIsNotNone(dir1, "Initialization of MillerVector object list of x,y,z components failed")
                
        cubeLat = olt.cubic(1, orientation=Orientation(euler=[45,90,0], units='Deg'))
        vec = [2.,-1.,-0.]
        dir2 = MillerDirection(lattice=cubeLat, vector = vec)
        self.assertIsNotNone(dir2, "Initialization of MillerVector object list of x,y,z components failed")
        
        orthoLat = olt.orthorhombic(1, 2, 3)
        vec = [1.,1.,1.]
        dir3 = MillerDirection(lattice=orthoLat,vector=vec)
        self.assertIsNotNone(dir3, "Initialization of MillerVector object list of x,y,z components failed")
        
        hexLat = olt.hexagonal(1, 1.63)
        cartVec = [0,0,1.63]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=True)
        self.assertIsNotNone(dir1, "Initialization of MillerVector object list of x,y,z components failed")
     
class MillerDirectionOutputCases(MillerDirectionCreationTestCases):
    def test_getLatexString(self):
        uvw = [1,0,0.]
        dir1 = MillerDirection(lattice=self.orthoLat,vector=uvw)
        
        latexString = dir1.getLatexString()
        self.assertEqual(r"$[ 1 0 0 ]$", latexString, 'Latex string conversion problem!!!')
        
        
        dir1 = MillerDirection(lattice=self.hexLat, vector = [2,-1,-1,0],MillerConv='MillerBravais' )
        #print("The plane uvw is ", plane2.gethkl())
        latexString = dir1.getLatexString()
        
        self.assertEqual(r"$[ 2 \bar{1} \bar{1} 0 ]$", latexString, 'Latex string conversion problem!!!')
    def test_format(self):
        
        uvw = [1,-1.0,0.,2.]
        dir1 = MillerDirection(lattice=self.hexLat2,vector=uvw)
        dir2 = vector = dir1.getUnitVector(returnCartesian=False)
        strFormat = "{:.3f}".format(dir1)
        str2Format = "{:int}".format(dir2)
        #print("The format string is ", strFormat)
        self.assertEqual(r'uvw = [ 1 -1  0  2 ]', str2Format,'Error in __format__ function')
        
        self.assertEqual(r'uvw = [1.000 -1.000 -0.000 2.000 ] lattice={3.2, 3.2, 5.1, 90.0, 90.0, 120.0}', strFormat,'Error in __format__ function')
        
        
        uvw = [1,0,0.]
        dir1 = MillerDirection(lattice=self.cubeLat,vector=uvw)
        strFormat = format(dir1,'')
        #print("The format string is ", strFormat)
        self.assertEqual(r'uvw = [1.00 0.00 0.00 ] lattice={2.0, 2.0, 2.0, 90.0, 90.0, 90.0}', strFormat,'Error in __format__ function')
        
    def test_str(self):
        
        uvw = [1,-1.0,0.,2.]
        dir1 = MillerDirection(lattice=self.hexLat2,vector=uvw)
        dir2 = vector = dir1.getUnitVector(returnCartesian=False)
        strFormat = str(dir1)
        str2Format = str(dir2)
        #print("The format string is ", strFormat)
        self.assertEqual(r'[  1   -1   0   2 ]', str2Format,'Error in __str__ function')
        
        self.assertEqual(r'[  1   -1   0   2 ]', strFormat,'2nd case Error in __str__ function')
        
        
#         uvw = [1,0,0.]
#         dir1 = MillerDirection(lattice=self.cubeLat,vector=uvw)
#         strFormat = format(dir1,'')
#         #print("The format string is ", strFormat)
#         self.assertEqual(r'uvw = [1.00 0.00 0.00 ] lattice={2.0, 2.0, 2.0, 90.0, 90.0, 90.0}', strFormat,'Error in __format__ function')
   
  
    def test_repr(self):
        uvw = [1,0,0.]
        dir1 = MillerDirection(lattice=self.cubeLat,vector=uvw)
        strFormat = repr(dir1)
        #print("The repr string is ", strFormat)
        self.assertEqual(r'uvw = [1.000 0.000 0.000 ] lattice={2.0, 2.0, 2.0, 90.0, 90.0, 90.0}', strFormat,'Error in __repr__ function')
        
        uvw = [1,-1.0,0.,2.]
        dir1 = MillerDirection(lattice=self.hexLat,vector=uvw)
        strFormat = repr(dir1)
        #print("The repr string is ", strFormat)
        self.assertEqual(r'uvw = [1.000 -1.000 -0.000 2.000 ] lattice={1.0, 1.0, 1.6, 90.0, 90.0, 120.0}', strFormat,'Error in __repr__ function')






class MillerDirectionConversionTestCases(MillerDirectionCreationTestCases):
    
    def test_fromMillerToBravais(self):
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[30,0,0], units='Deg'))
        vec = [2.,-1.,0.] #here we dropped 't' of [uvtw] but still we specifcally mention that this is the Bravais convention by using key work MillerConv=MillerBravis
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        v = dir1.fromMillerToBravais()
        self.assertArrayAlmostEqual(v, [vec[0], vec[1], -(vec[0]+vec[1]), vec[2]])
                
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[0,0,0], units='Deg'))
        cartVec = [3,0,1.63]
        dir1 = MillerDirection(lattice=hexLat, vector = cartVec, isCartesian=True)
        v = dir1.fromMillerToBravais()
        self.assertArrayAlmostEqual([2,-1,-1,1], v)
        
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[30,90,0], units='Deg'))
        vec = [1,1,-2,3]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Miller")
        uvtw = dir1.fromMillerToBravais()
        self.assertArrayAlmostEqual(uvtw, [1,1,-2,3], err_msg='Failed for uvw to uvtw')   
        
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[0,0,0], units='Deg'))
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Miller")
        uvtw = dir1.fromMillerToBravais(integerize=True)
        self.assertArrayAlmostEqual(uvtw, [1,1,-2,3], err_msg='Failed for uvw to uvtw')   
  
             
    
    def test_getCartesianVec(self):
        
        hexLat = olt.hexagonal(1.0, 1.59*1.0, orientation=Orientation(euler=[60.*np.pi/180,0.,0.])) 
        vec = [1.,-1.,-0.,0.]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="MillerBravis")
        vec = dir1.getCartesianVec()
        self.assertArrayAlmostEqual(vec, [0, -1.7320508,0], err_msg="Error in conversion to cartesan vector")
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        v = dir1.getCartesianVec()
        self.assertArrayAlmostEqual(v, [0,0,1.63])
        
        #####nachiket 27-11-17#####
        cubLat = olt.cubic(3.54)
        cartVec = [3.54, 3.54, 3.54]
        dir1 = MillerDirection(lattice=cubLat, vector = cartVec, isCartesian=True)
        self.assertArrayAlmostEqual([1,1,1], dir1.vector) 
        #####nachiket 27-11-17#####  

        hexLat = olt.hexagonal(1, 1.63)
        vec = [-1/3,-1/3,2/3,0]
        vec1 = [-0.50,-0.86,0.0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Miller")
        vec2 = dir1.getCartesianVec()
        self.assertArrayAlmostEqual(vec2, vec1, decimal=2, err_msg='Failed for uvtw (lattice) to uvtw (cartesian)')
        
        #####nachiket 27-11-17#####
        hexLat = olt.hexagonal(1, 1.63)
        vec = [-1,-1,0]
        vec1 = [-0.50,-0.86,0.0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec1, isCartesian=True, MillerConv="Miller")
        vec2 = dir2.vector
        self.assertArrayAlmostEqual(vec, vec2, decimal=2, err_msg='Failed for uvtw (cartesian) to uvtw (lattice)')
        #####nachiket 27-11-17#####
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [1/3,1/3,-2/3,1]
        vec1 = [hexLat.a/sqrt(hexLat.a**2 + hexLat.c**2), (sqrt(3)*hexLat.a)/sqrt(hexLat.a**2 + hexLat.c**2), (2*hexLat.c)/sqrt(hexLat.a**2 + hexLat.c**2)]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Miller")
        vec2 = dir1.getCartesianVec()
        self.assertArrayAlmostEqual(vec2, vec1, decimal=1, err_msg='Failed for uvtw (lattice) to uvtw (cartesian)')
        
        monoclinicLat = olt.monoclinic(4.0,2.0,3.0,30.0)
        latVec = np.array([10.0,3.0,29.0])
        cartVec = np.array([(monoclinicLat.a*sin(radians(monoclinicLat.beta))*latVec[0]),monoclinicLat.b*latVec[1],(monoclinicLat.c * latVec[2] + (monoclinicLat.a*cos(radians(monoclinicLat.beta)))*latVec[0])])
        dir1 = MillerDirection(lattice=monoclinicLat, vector = latVec, isCartesian=False)
        vec1 = dir1.getCartesianVec()
        self.assertArrayAlmostEqual(cartVec, vec1, err_msg='Failed for uvw (lattice) to uvw (cartesian) in monoclinic')

        
    def test_getUVW(self):
        hexLat = olt.hexagonal(1, 1.63)
        vec = [2, -1, -1, 1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        uvtw = dir1.getUVW()
        self.assertArrayAlmostEqual(uvtw, vec, err_msg='Failed in gettting hexagonal uvtw' )  
        
        cubLat = olt.cubic(3.54)
        cartVec = [3.54, 3.54, 3.54]
        dir1 = MillerDirection(lattice=cubLat, vector = cartVec, isCartesian=True)
        uvw = dir1.getUVW()
        self.assertArrayAlmostEqual(uvw, [1,1,1], err_msg='Failed for uvw of cubic')

       
    def test_getDirectionInOtherLatticeFrame(self):
        hexLat = olt.hexagonal(1, 1.63)
        vec = [2,-1,-1,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec,  MillerConv="Bravais")
                
        cubLat = olt.cubic(3.0)
        Vec = [1,0,0]
        dir2 = MillerDirection(lattice=cubLat, vector = Vec, isCartesian=False)        
        dir3 = dir1.getDirectionInOtherLatticeFrame(cubLat)        
        self.assertEqual(dir3, dir2, 'Failed conversion to other lattice frame')
        
 #   def test_hexToRhombohedral(self):
 #       hexLat = olt.hexagonal(1, 1.63)
 #       vec = [0,0,1]
 #       dir1 = MillerDirection(lattice=hexLat, vector = vec,  MillerConv="Miller")
 #       
 #       dir2 = dir1.hexToRhombohedral()
 #       self.assertArrayAlmostEqual(dir2.getUVW(),[3,3,0],4,'Failed in hexToRhombohedral conversion')
 #    
    def test_getPerpendicularDirection(self):
        cubLat = olt.cubic(1.0)
        Vec = [1,1,1]
        dir1 = MillerDirection(lattice=cubLat, vector = Vec, isCartesian=False)                     
        dir1.getPerpendicularDirection(returnCartesian=True)
        #print(dir1)
        
        
        ### nachiket 27-11-17
        cubLat = olt.cubic(1)
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        
        perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
                      
        dot1 = dir1.dot(perp1)
              
        self.assertAlmostEqual(dot1,0.)
        #np.allclose(cross2.getUVW(force3Index=True),dir2.getUVW(force3Index=True))
        
        ### nachiket 27-11-17
        
        ### nachiket 27-11-17
        hexLat = olt.hexagonal(1,1.63)
        vec = [2,-1,-1,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        
        perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
        perp2 = dir1.getPerpendicularDirection() # the default for returnCartesian=False         
        dot1 = dir1.dot(perp1)
        dot2 = dir1.dot(perp2)
              
        self.assertArrayAlmostEqual([dot1,dot2],[0.,0.])
        
        
        ### nachiket 27-11-17
        
        ### nachiket 27-11-17
        cubLat = olt.cubic(1)
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        
        perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
        perp2 = dir1.getPerpendicularDirection(returnCartesian=False)
                      
        cross1 = perp1.cross(perp1)
        cross2 = dir1.cross(cross1)
              
        self.assertAlmostEqual(cross2.getMag(),0.)
        #np.allclose(cross2.getUVW(force3Index=True),dir2.getUVW(force3Index=True))
        
        ### nachiket 27-11-17
        ### nachiket 4-12-17
    def test_getOrthoset(self):
        cubLat = olt.cubic(1)
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
        cross1 = perp1.cross(dir1)
        dot1 = perp1.dot(dir1)
        dot2 = cross1.dot(dir1)
        self.assertArrayAlmostEqual([dot1,dot2],[0.,0.]) 
        
    def test_symmetricSet(self):
        cubLat = olt.cubic(1)
        vec = [1,2,3]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),24)
        
        cubLat = olt.cubic(1)
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),8)
        
        cubLat = olt.cubic(1)
        vec = [1,0,0]
        dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),6)
        
        hexLat = olt.hexagonal(1,1.63, orientation=Orientation(euler=[90*np.pi/180, 0,0]))
        vec = [1,1,-2,3]
        dir1 = MillerDirection(lattice=hexLat, vector = vec)
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),12)
        
        orthoLat = olt.orthorhombic(1, 2, 3, orientation=Orientation(euler=[0.,0,0]))
        vec = [1,1,1]
        dir1 = MillerDirection(lattice=orthoLat, vector = vec)
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),4)
        
        orthoLat = olt.orthorhombic(1, 2, 3, orientation=Orientation(euler=[0.,0,0]))
        vec = [1,0,0]
        dir1 = MillerDirection(lattice=orthoLat, vector = vec)
        symSet = dir1.symmetricSet()
        self.assertEquals(len(symSet),2)
        
    def test_isSymmetric(self):    
        dir1 = MillerDirection(lattice=self.cubeLat, vector=[1,1,1])
        dir2 = MillerDirection(lattice=self.cubeLat, vector=[1,1,-1])
        sym=dir1.isSymmetric(dir2)
        self.assertEqual(sym,True,'Error in isSymmetric method')
        
        dir3 = MillerDirection(lattice=self.cubeLat, vector=[1,0,-1])
        sym=dir1.isSymmetric(dir3)
        self.assertEqual(sym,False,'Error in isSymmetric method')
        
        dir1 = MillerDirection(lattice=self.hexLat, vector=[2,-1,-1,0])
        dir2 = MillerDirection(lattice=self.hexLat, vector=[-1,2,-1,0])
        sym=dir1.isSymmetric(dir2)
        self.assertEqual(sym,True,'Error in isSymmetric method')
        
        dir3 = MillerDirection(lattice=self.hexLat, vector=[0,0,0,1])
        sym=dir1.isSymmetric(dir3)
        self.assertEqual(sym,False,'Error in isSymmetric method')
        


        
        
        
        
  
        
        
        
class MillerDirectionArithmatic(MillerDirectionCreationTestCases):  
     
    def test_getMag(self):
        hexLat = olt.hexagonal(1, 1.63)
        vec = [0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        mag = dir1.getMag()
        self.assertAlmostEqual(mag, 1.63)
        
        orthoLat = olt.orthorhombic(1, 2, 3)
        vec = [1.,1.,1.]
        dir2 = MillerDirection(lattice=orthoLat,vector=vec)
        mag = dir2.getMag()
        self.assertAlmostEqual(mag, sqrt(14))
        
        cubicLat = olt.cubic(1)
        vec = [1.,1.,1.]
        dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        mag = dir3.getMag()
        self.assertAlmostEqual(mag, sqrt(3))
        self.assertAlmostEqual(dir3.mag, sqrt(3))
        
        
    def test_getUnitVector(self):
        orthoLat = olt.orthorhombic(1, 2, 3)
        vec = [1.,1.,1.]
        dir1 = MillerDirection(lattice=orthoLat,vector=vec)
        Unitdir = dir1.getUnitVector()
        self.assertArrayAlmostEqual(Unitdir, np.array([1,2,3])/sqrt(14))
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [2,-1,-1,0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        Unitdir = dir2.getUnitVector()
        self.assertArrayAlmostEqual(Unitdir, [1,0,0])
        
        cubicLat = olt.cubic(1)
        vec = [1.,1.,1.]
        dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        Unitdir = dir3.getUnitVector()
        self.assertArrayAlmostEqual(Unitdir, np.array([1,1,1])/sqrt(3))        
        
    def test_generateDirectionList(self):
        hexLat = olt.hexagonal(1,1.59)
        dirList = MillerDirection.generateDirectionList(uvwMax=1, lattice=hexLat, includeSymEquals=True)
        for i, item in enumerate(dirList):
            print("{:2d} {:int}".format(i, item))
        #print(dirList)
    
    def test_angle(self):
        orthoLat = olt.orthorhombic(1, 1, 1)
        vec = [1.,1.,1.]
        dir1 = MillerDirection(lattice=orthoLat,vector=vec)
        
        hexLat = olt.hexagonal(1*3.230, 1.59*3.230)
        vec = [2,-1,-1,0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        ang1 = dir1.angle(dir2,units='Deg')
        ang2 = dir2.angle(dir1,units='Deg')
        
        self.assertAlmostEqual(ang1, ang2)
        self.assertAlmostEqual(ang1, 54.735610317245346)        
        
        dir3 = MillerDirection(lattice=hexLat, vector = [1,0,-1,0])        
        ang3 = dir3.angle(dir2, units="degree")
        self.assertAlmostEqual(ang3, 30, msg="Problem in angle calcualtion")
        del dir1, dir2, dir3
        
        cubicLat = olt.cubic(1)
        vec = [1.,0.,0.]
        dir1 = MillerDirection(lattice=cubicLat,vector=vec)
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [0,0,0,1]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        ang1 = dir1.angle(dir2,units='Deg')
        ang2 = dir2.angle(dir1,units='Deg')
        
        self.assertAlmostEqual(ang1, ang2)
        self.assertAlmostEqual(ang1, 90.0) 
        del dir1, dir2       

        cubicLat = olt.cubic(1)
        vec = [1.,0.,0.]
        dir1 = MillerDirection(lattice=cubicLat,vector=vec)
        
        cubicLat = olt.cubic(1)
        vec = [1.,1.,1.]
        dir2 = MillerDirection(lattice=cubicLat,vector=vec)
       
        vec = [0,0,-1.]
        dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        
        
        ang1 = dir1.angle(dir2,units='Deg')
        ang2 = dir2.angle(dir1,units='Deg')
        ang3 = dir1.angle(dir3,considerSymmetry=True)
        
        self.assertAlmostEqual(ang1, ang2)
        self.assertAlmostEqual(ang1, 54.735, 2) 
        self.assertAlmostEqual(ang3, 0,4, "Probelm in angle calcualtion")
        del dir1, dir2   
        
        orthoLat = olt.orthorhombic(1, 1, 1)
        vec = [1.,1.,1.]
        dir1 = MillerDirection(lattice=orthoLat,vector=vec)
        
        hexLat = olt.hexagonal(1*3.230, 1.59*3.230)
        vec = [2,-1,-1,0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        ang1 = dir1.angle(dir2,units='Deg')
        ang2 = dir2.angle(dir1,units='Deg')
        
        self.assertAlmostEqual(ang1, ang2)
        self.assertAlmostEqual(ang1, 54.735610317245346)        
        
        dir3 = MillerDirection(lattice=hexLat, vector = [1,0,-1,0])        
        ang3 = dir3.angle(dir2, units="degree")
        self.assertAlmostEqual(ang3, 30, msg="Problem in angle calcualtion")
        
        dir4 = MillerDirection(lattice=hexLat, vector = [0,0,0,1]) 
        dir5 = MillerDirection(lattice=hexLat, vector = [0,0,0,2])
        dir6 = MillerDirection(lattice=hexLat, vector = [-1,0,1,0])       
        ang4 = dir3.angle(dir4, units="degree")
        ang5 = dir4.angle(dir5, units="degree")
        ang6 = dir2.angle(dir6, units="degree")
        self.assertAlmostEqual(ang4, 90, msg="Problem in angle calcualtion")
        self.assertAlmostEqual(ang5, 0, msg="Problem in angle calcualtion")
        self.assertAlmostEqual(ang6, 150, msg="Problem in angle calcualtion")        
              
    
        
    def test_rotate(self):
        hexLat = olt.hexagonal(1, 1.63)
        vec = [2,-1,-1,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False)
        dir2 = copy.copy(dir1)
        ori = Orientation(euler=[30.*np.pi/180,0.,0.])
        dir1.rotate(ori)
        ang = dir1.angle(dir2, units='Deg')
        self.assertAlmostEqual(ang, 30., msg='Failed in rotaiton')
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [2,-1,-1,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        dir2 = copy.copy(dir1)
        ori = Orientation(axis=[0.,1.,0.], degrees=90)
        dir1.rotate(ori)
        #print(dir1)
        ang = dir1.angle(dir2, units='Deg')
        self.assertAlmostEqual(ang, 90., msg='Failed in rotaiton') 
        
        cubicLat = olt.cubic(1)
        vec = [0.,0.,1.]
        dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        dir2 = copy.copy(dir1)
        ori = Orientation(axis=[0.,1.,0.], degrees=-45)
        dir1.rotate(ori)
        #print(dir1.getUVW(force3Index=True))
        self.assertArrayAlmostEqual(dir1.getUVW(force3Index=True), np.array([-1,0,1])/sqrt(2),3)
        
    def test_dot(self):
        
        orthoLat = olt.orthorhombic(1, 1, 1)
        vec = [1.,1.,1.]
        dir1 = MillerDirection(lattice=orthoLat,vector=vec)
                
        hexLat = olt.hexagonal(1, 1.63)
        vec = [0,0,0,1]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        
        vec = [2,-1,-1,0]
        dir3 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        dot1 = dir1.dot(dir2)
        dot2 = dir2.dot(dir1)
        
        dot3 = dir3.dot(dir2)
        
        self.assertAlmostEqual(dot1,dot2)
        self.assertAlmostEqual(dot3,0)
            
    def test_cross(self):
        
        orthoLat = olt.orthorhombic(1, 1, 1)
        vec = [0.,1.,0.]
        dir1 = MillerDirection(lattice=orthoLat,vector=vec)
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [0,0,0,1]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
                
        vec = [2,-1,-1,0]
        dir3 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
                
        cross1 = dir1.cross(dir2,'other')
        cross2 = dir2.cross(dir1)
        cross3 = dir2.cross(dir3)
        cross4 = cross1.cross(dir1)
        
        self.assertAlmostEqual(cross1,-cross2)
        self.assertArrayAlmostEqual([cross3.angle(dir2), cross3.angle(dir3)],[pi/2,pi/2])
        self.assertArrayAlmostEqual(cross4.getUVW(force3Index=True),dir2.getUVW(force3Index=True))
        
        cubicLat = olt.cubic(1)
        vec = [0.,1.,0.]
        dir1 = MillerDirection(lattice=cubicLat,vector=vec)
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [1,1,-2,0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
                
        cross1 = dir1.cross(dir2,'other')
        cross2 = dir2.cross(dir1)
        
        self.assertAlmostEqual(cross1,-cross2, msg='They are opposite to each other')
        self.assertArrayAlmostEqual([cross1.angle(-cross2), cross1.angle(cross2)],[0,pi])

        cubicLat = olt.cubic(1)
        vec1 = [1.,0.,0.]
        dir1 = MillerDirection(lattice=cubicLat,vector=vec1)
        
        vec2 = [0.,1.,0.]
        dir2 = MillerDirection(lattice=cubicLat,vector=vec2)
        
        cross1 = dir1.cross(dir2)
        self.assertArrayAlmostEqual(cross1.getUVW(force3Index=True),[0.0,0.0,1.0])
        
    def test_add(self):
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[30,0,0], units='Deg'))
        vec = [0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[30,0,0], units='Deg'))
        vec = [2,-1,-1,0]
        dir2 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")     
        
        orthoLat = olt.orthorhombic(1, 1, 1, orientation=Orientation(euler=[60,90,0], units='Deg'))
        vec = [1.,1.,1.]
        dir3 = MillerDirection(lattice=orthoLat,vector=vec)
        
        dir4 = dir1+dir2
        self.assertArrayAlmostEqual(dir4.getUVW(), [2,-1,-1,1], 4, 'MillerDirection addtion failed')
    
    def test_integerize(self):

        cubicLat = olt.cubic(1)
        vec = [-0.25,-0.5,0.75]
        dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        [vec, AngleError] = MillerDirection.integerize(dir1)
        self.assertArrayAlmostEqual(vec, [-1.0, -2.0, 3.0])

        hexLat = olt.hexagonal(1, 1.63, orientation=Orientation(euler=[0,0,0], units='Deg'))
        vec = [1.98,-0.98,-1,0]
        dir3 = MillerDirection(lattice=hexLat, vector = vec)
        [vec, AngleError] = MillerDirection.integerize(dir3,AngleDeviationUnits='Deg')
        self.assertArrayAlmostEqual(vec, [2, -1, -1, 0])
        
    def test_uvw(self): #nachiket 5-12-17
        
        hexLat = olt.hexagonal(1, 1.63)
        vec = [1.,1.,1.]
        dir3 = MillerDirection(lattice=hexLat,vector=vec)
        self.assertArrayAlmostEqual(dir3.uvw,[1.,1.,1.])
    
    
    
    
    def test_steriographiProjection(self):        
        cubicLat = olt.cubic(1)
        ori = Orientation(euler=np.radians(np.array([0., 23, 45.])))
        dir1 = MillerDirection(vector=[-1,0,0],lattice=cubicLat)
        [point, polarPoint, isNorth, alpha,beta ] = dir1.steriographiProjection(orientation=ori)
        print(point, polarPoint, isNorth, alpha,beta )
        self.assertArrayAlmostEqual(point, [1/np.sqrt(2),1/np.sqrt(2)],5,"Probelem in sterio")
        self.assertTrue(isNorth,"Projection is in wrong pole")
        
        

        
        
        
        vec = [1,1, 0]
        rot = Orientation(axis=[1,0,0],degrees=45.)
        z=[0,0,1]
        z_1 = rot.rotate(z)
        #print(z,z_1,rot, rot.rotation_matrix  )
        dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        [point, polarPoint, isNorth, alpha,beta ] = dir1.steriographiProjection()
        print(point, polarPoint, isNorth, alpha,beta )
        self.assertArrayAlmostEqual(point, [1/np.sqrt(2),1/np.sqrt(2)],5,"Probelem in sterio")
        self.assertTrue(isNorth,"Projection is in wrong pole")
        
        
        

        dir2 = MillerDirection(lattice=cubicLat, vector = [0,1,0])
        [point, polarPoint, isNorth,alpha,beta] = dir2.steriographiProjection()
        self.assertArrayAlmostEqual(point, [0,1],5,"Probelem in sterio")
        print(point, polarPoint, isNorth, alpha,beta )

        hexLat = olt.hexagonal(1, 1.63)
        vec = [2,-1,-1,0]
        dir3 = MillerDirection(lattice=hexLat,vector=vec)
        [point, polarPoint, isNorth, alpha,beta] = dir3.steriographiProjection()
        self.assertArrayAlmostEqual(point, [1,0],5,"Probelem in sterio")
        print(point, polarPoint, isNorth, alpha,beta )
        
    def test_applyHolderTiltsToDirection(self):
        cubicLat = olt.cubic(1)
        vec = [0,0,1 ]
        vec = [ 5, -7,  5 ]
        dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        alphaTilt = -45.
        betaTilt = -45.
        options={"fixedAxes":True,"alphaRotationFirst":True}
        rotatatedDir1 = MillerDirection.applyHolderTiltsToDirection(dir1,alphaTilt=alphaTilt,betaTilt=betaTilt,alphaAxis=[1.,0.,0],betaAxis=[0.,1.,0.],options=options)
        print("alpha = {:.2f} beta ={:.2f},  Original dir = {:int}, and final {:int}".format(alphaTilt, betaTilt, dir1, rotatatedDir1))

    def test_rotationMatrixForHolderTilts(self):
        alphaTilt = 45
        betaTilt = 0.
        alpha = alphaTilt*np.pi/180.
        beta = betaTilt*np.pi/180.
        testDir = np.array([0.,0.,1])
        mat = np.array([[np.cos(beta), 0., np.sin(beta)],
                        [np.sin(alpha)*np.sin(beta), np.cos(alpha), -np.sin(alpha)*np.cos(beta)],
                        [-np.cos(alpha)*np.sin(beta), np.sin(alpha), np.cos(alpha)*np.cos(beta)]]
                       )
        
        options={"fixedAxes":True,"alphaRotationFirst":False}
        rotation = MillerDirection.rotationMatrixForHolderTilts(alphaTilt,betaTilt,alphaAxis=[1.,0,0.],betaAxis=[0.,1.,0],options=options)
        mDir1 = MillerDirection(vector=testDir, lattice=self.cubeLat)
        mDir1Original = copy.copy(mDir1)
        
        testDirRotated1 = rotation.rotate(testDir)
        testDirRotated2 = np.matmul(mat, testDir.T)
        mDir1.rotate(rotation)
        mRotatedDir = mDir1.getCartesianVec()
        print(rotation)
        print(rotation.rotation_matrix, testDirRotated1,testDirRotated2,mDir1.getCartesianVec(), mDir1Original.getCartesianVec())
        
        self.assertArrayAlmostEqual(testDirRotated1, testDirRotated2,5,"Probelem in rotation matrix from holders")
        self.assertArrayAlmostEqual(mat, rotation.rotation_matrix,5,"Probelem in rotation matrix from holders")
        
    
if __name__ == '__main__':
    import unittest
    unittest.main()