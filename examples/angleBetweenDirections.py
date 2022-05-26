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

from pycrystallography.core.millerDirection import MillerDirection
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation import Orientation
from pycrystallography.core.orientedLattice import OrientedLattice

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from copy import deepcopy
from tabulate import tabulate
import itertools

# plane_set1=np.array(([0,0,0,1],[1,0,-1,0],[0,0,0,2],[0,1,-1,-1],[1,1,-2,1],[1,0,-1,3],[1,-2,1,2],[2,-1,-1,0],[0,0,0,4],[2,0,-2, 2],[1, 0, -1, 4],[2, 0, -2, 3],[1, 1, -2, 4],[2, 1, -3, 2],[1, 0, -1, 5],[2, 0, -2, 4],[3, 0, -3, 0],[2, 1, -3, 3],[0, 0, 0, 6],[2, 0, -2, 5],[1, 0, -1, 6]))
# plane_set2=np.array(([0,0,0,1],[1,0,-1,0],[0,0,0,2],[0,1,-1,-1],[1,1,-2,1],[1,0,-1,3],[1,-2,1,2],[2,-1,-1,0],[0,0,0,4],[2,0,-2, 2],[1, 0, -1, 4],[2, 0, -2, 3],[1, 1, -2, 4],[2, 1, -3, 2],[1, 0, -1, 5],[2, 0, -2, 4],[3, 0, -3, 0],[2, 1, -3, 3],[0, 0, 0, 6],[2, 0, -2, 5],[1, 0, -1, 6]))

hklMax = 3
a = range(-hklMax, hklMax + 1)
uvwList = []
for combination in itertools.product(a, a, a):
    uvwList.append(combination)

uvwList.remove((0, 0, 0));

# vec = [1.,1.,1.]
# dir1 = MillerDirection(lattice=orthoLat,vector=vec)
#
cifFile = r'../data/structureData/Fe3C.cif'
lattice = OrientedLattice.fromCif(cifFile)
# hexLat = OrientedLattice.hexagonal(a=3.23, c=1.59*3.23, orientation=Orientation(0,0,0),)
# Lattice.hexagonal(3.23, 1.59*3.23)

# vec = [2,-1,-1,0]
# dir2 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravis")
# ang1 = dir1.angle(dir2,units='Deg')
AngleData = np.zeros((len(uvwList), len(uvwList)))
DirNameList = []
for i in range(len(uvwList)):
    dir1 = MillerDirection(lattice=lattice, vector=uvwList[i], MillerConv="Bravis")
    DirNameList.append(str(dir1))
    for j in range(len(uvwList)):
        dir2 = MillerDirection(lattice=lattice, vector=uvwList[j], MillerConv="Bravis")
        ang = dir1.angle(dir2, units="degree")
        AngleData[i, j] = ang

B = tabulate(AngleData, headers=DirNameList, tablefmt="plane")
print(B)

np.savetxt('AngleBetweenDirections.txt', AngleData, delimiter='  ', fmt='%10.2f', )

# for combination in itertools.product(xrange(10), repeat=4):
#    print(join(map(str, combination)))








