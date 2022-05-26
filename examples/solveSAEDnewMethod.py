'''
Created on 04-Dec-2017

@author: K V Mani Krishna
'''

from __future__ import division, unicode_literals

import sys
import os
from itertools import permutations

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane


from pycrystallography.core.orientation  import Orientation
import pycrystallography.utilities.graphicUtilities as gu

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy
from tabulate import tabulate
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import time
import copy
import cProfile
from sympy.geometry import *
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
 

lat = olt.cubic(2)
plane1 = MillerPlane(lattice = lat, hkl = [1,1,1]) 
cubeLat = olt.cubic(2)
orthoLat = olt.orthorhombic(2, 1, 1)
hexLat = olt.hexagonal(1, 1.63)
hexLat2 = olt.hexagonal(3.2, 1.59*3.2) ## Zr 
 
fccAtomData = [(1, np.array([0.,0.,0.])),
         (1, np.array([.5,.5,0.])),
         (1, np.array([.5,0.,.5])),
         (1, np.array([0,.5,.5]))]
 
zrAtomData = atomData = [(40., np.array([0.,0.,0.])),
         (40., np.array([1./3., 2./3.,1./2.]))]


sa = SaedAnalyzer(lattice=cubeLat,hklMax = 4,atomData=fccAtomData)
spotData = sa.extractSpotsFromExpPattern(imageFileNameFcc111, displayImage=False)
measureData = sa.slectSpotsForIndexing(spotIds=[1,2])
print(measureData)
solution = sa.solveSetOfSpots(spotIds=[1,2], dTol=2.,angleTol=2)
for i in solution:
    print(i)
solID = 0                                            
zoneAxis = solution[solID]["zoneAxis"]
calib2 = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=[],
                                  spot1Plane=spot1,pc=[100.,50.])
   

 
saedData = sa.calcualteSAEDpattern(zoneAxis=zoneAxis, atomData=self.fccAtomData)
print("Yes the saed pattern calualtion is done", saedData)
for i in saedData["SpotData"]:
    print(i["Plane"],i['XY'])
print("Done")
self.assertTrue(np.abs(120-measureData["Angle"])<1,"Problem in spot slection")
          