# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os
#from examples.solveSAED import lattice

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation  import Orientation

import numpy as np
import scipy
import math
import copy
from math import pi, sqrt
#from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
from copy import deepcopy
from tabulate import tabulate
import itertools
import time

start = time.clock()
# hklMax = 4
# atomData = [(1, np.array([0.,0.,0.])),
#                 (1, np.array([.5,.5,0.])),
#                 (1, np.array([.5,0.,.5])),
#                 (1, np.array([0,.5,.5]))
#                 ] ### fcc imaginary data
# 
# atomData = [(40., np.array([0.,0.,0.])),
#                 (40., np.array([1./3., 2./3.,1./2.]))]
# lattice = olt.hexagonal(3.2,3.2*1.59)
#lattice = olt.
lattice = olt.cubic(5)
# plane1 = MillerPlane(hkl = [-1,1.,-1],lattice=lattice)
# plane2 = MillerPlane(hkl = [1,1.,-1],lattice=lattice)
# print(plane1.angle(plane2,"Deg"))

sa = SaedAnalyzer(hklMax=3)
sa.loadStructureFromCif(r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Vanadium.cif')

print("### plane   d Spacing  1/dspacing   Twotheta (degrees) intensity")
planeTable = sa.generateDspacingTable()
for i, item in enumerate(planeTable):
    print (item["plane"], "   ", item["dSpacing"], "   ", np.around(item["xRayPeakPosition"],3), item["intensity"] )
    
# assert 5==2, 'failed'# assert 5==5,"Success'"
print ("Done !!!") 
print ("The elapsed time is ", time.clock() - start, "seconds")     

        
                

