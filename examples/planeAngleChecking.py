# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os

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
from pymatgen.core.lattice import Lattice
from copy import deepcopy
from tabulate import tabulate
import itertools

 



#plane_set1=np.array(([0,0,0,1],[1,0,-1,0],[0,0,0,2],[0,1,-1,-1],[1,1,-2,1],[1,0,-1,3],[1,-2,1,2],[2,-1,-1,0],[0,0,0,4],[2,0,-2, 2],[1, 0, -1, 4],[2, 0, -2, 3],[1, 1, -2, 4],[2, 1, -3, 2],[1, 0, -1, 5],[2, 0, -2, 4],[3, 0, -3, 0],[2, 1, -3, 3],[0, 0, 0, 6],[2, 0, -2, 5],[1, 0, -1, 6]))
#plane_set2=np.array(([0,0,0,1],[1,0,-1,0],[0,0,0,2],[0,1,-1,-1],[1,1,-2,1],[1,0,-1,3],[1,-2,1,2],[2,-1,-1,0],[0,0,0,4],[2,0,-2, 2],[1, 0, -1, 4],[2, 0, -2, 3],[1, 1, -2, 4],[2, 1, -3, 2],[1, 0, -1, 5],[2, 0, -2, 4],[3, 0, -3, 0],[2, 1, -3, 3],[0, 0, 0, 6],[2, 0, -2, 5],[1, 0, -1, 6]))




hklMax = 3
a = range(hklMax+1)
hklList = []
for combination in itertools.product(a, a, a):
    hklList.append(combination)
    
hklList.remove((0,0,0));
      
hexLat = Lattice.hexagonal(3.23, 1.59*3.23)


plane1 = MillerPlane(lattice=hexLat,hkl=[1,-1,0,0], MillerConv="Bravais")

plane2 =  MillerPlane(lattice=hexLat,hkl=[1,0,-1,0], MillerConv="Bravais")
plane2 =  MillerPlane(lattice=hexLat,hkl=[0,1,-1,0], MillerConv="Bravais")
plane2 =  MillerPlane(lattice=hexLat,hkl=[1,-1,0,0], MillerConv="Bravais")
plane2 =  MillerPlane(lattice=hexLat,hkl=[-1,0,1,0], MillerConv="Bravais")
plane2 =  MillerPlane(lattice=hexLat,hkl=[0,-1,1,0], MillerConv="Bravais")
plane2 =  MillerPlane(lattice=hexLat,hkl=[1,0,-1,0], MillerConv="Bravais")






print("The angle is ", plane1.angle(plane2,units='Deg'))


