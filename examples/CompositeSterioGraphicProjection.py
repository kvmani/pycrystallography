# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os
import cProfile

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
from pycrystallography.core.crystalOrientation  import CrystalOrientation as CryOri,\
    CrystalOrientation


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

fccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))]
hcpAtomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))]
bccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,.5])),
               ]

bccLattice = olt.cubic(3.53,name='Beta',symbol="$\beta$")
hcpLattice = olt.hexagonal(3.23,1.59*3.23,name='Alpha',symbol='$\alpha$')
fccLattice = olt.cubic(1.0, name='FCC',symbol='$\alpha$')


inputData = {
    "BurgersOr":{
            "nameParent":'Beta',
            "latticeParent":bccLattice,
            "atomDataParent":bccAtomData,
            
            "nameProduct":'Alpha',
            "latticeProduct":hcpLattice,
            "atomDataProduct":hcpAtomData,
            
            
            "angleAxisOR":(45,[2,-1,-1,0],),
            "ParllelPlanes":[[1,1,0],[0,0,0,1]],
            "ParllelDirections" :[ [-1,1,-1],[2,-1,-1,0]]
             
            }
                                
    }

choice="BurgersOr"

plaeListParent = [[0,0,1],
                  [1,1,0],
                  ]

latticeParent = inputData[choice]["latticeParent"]
latticeProduct = inputData[choice]["latticeProduct"]

atomData1 = inputData[choice]["atomDataParent"]
atomData2 = inputData[choice]["atomDataProduct"]


planeParent = inputData[choice]["ParllelPlanes"][0]
planeParent = MillerPlane(hkl = planeParent, lattice =latticeParent )

planeProduct = inputData[choice]["ParllelPlanes"][1]
planeProduct = MillerPlane(hkl = planeProduct, lattice =latticeProduct)

dirParent = inputData[choice]["ParllelDirections"][0]
dirParent = MillerDirection(vector = dirParent, lattice =latticeParent)

dirProduct = inputData[choice]["ParllelDirections"][1]
dirProduct = MillerDirection(vector = dirProduct, lattice =latticeProduct )


ori = Orientation(euler=[0.5,0,0.2])
or2 = Orientation(ori)
oriParent = CrystalOrientation.fromPlaneAndDirection(plane=planeParent,direction=dirParent)
oriProduct = CrystalOrientation.fromPlaneAndDirection(plane=planeProduct,direction=dirProduct)


misOriAB = oriParent.misorientation(oriProduct) 
misOriBA = oriProduct.misorientation(oriParent) 

print("{:planeDir} {:planeDir} \n{:axisAngle}{:axisAngle}\n{:euler}{:euler}".format(misOriAB,misOriBA,
                                                                                         misOriAB,misOriBA,misOriAB,misOriBA))


misoriSetAB = oriParent.symmetricMisorientations(oriProduct) 
misoriSetBA = oriProduct.symmetricMisorientations(oriParent) 

misoriSetAB = CrystalOrientation.uniqueList(misoriSetAB)                                             
misoriSetBA = CrystalOrientation.uniqueList(misoriSetBA)                                             

print("{:planeDir} \n{:planeDir} \nMosOri = {:planeDir} \n{:axisAngle}\n{:euler}".format(oriParent,oriProduct,misOriAB,misOriAB,oriParent))

for item in zip(misoriSetAB,misoriSetBA):
    
        print(" {:planeDir} {:planeDir}{:euler}".format(item[0], item[1],item[0]))

