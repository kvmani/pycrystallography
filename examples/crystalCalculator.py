# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os
from pycrystallography.core.millerPlane import MillerPlane

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.orientedLattice import OrientedLattice as olt

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from copy import deepcopy
from tabulate import tabulate
import itertools
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 


print("Simple calcualtor for the crystallographic computations")
wantDTable=False
calculateDirections=False
calculatePlanes=True
cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
betaCif =cifPathName+'U2Mo.cif'
#alphaCif =cifPathName+'Alpha-TiP63mmc.cif' 
    

cubeLat = olt.cubic(2)
orthoLat = olt.orthorhombic(2, 1, 1)
cubeLat = olt.fromCif(r'D:\CurrentProjects\colloborations\jyoti\zircon\zrSiO4tetragonal.cif')
hexLat = olt.hexagonal(1, 1.63)
hexLat2 = olt.hexagonal(3.2, 1.59*3.2)

inputData={
            "DeltaHydride":{"lattice":olt.cubic(4.77)},
            "ZrAlpha":{"lattice": olt.hexagonal(3.232, 1.59*3.232)},
            "ZrBeta":{"lattice": olt.cubic(3.54)},
        }

choice="ZrBeta"
lattice=inputData[choice]["lattice"]

# direction1 = MillerDirection(lattice=lattice,vector=[1,0,-1,0])
# direction2 = MillerDirection(lattice=lattice,vector=[1,1,-2,6])



plane1 = MillerPlane(lattice=lattice, hkl=[1,1,0],)
print("d= ",plane1.getInterplanarSpacing(),"Angstrom,  1/d = ",1/plane1.dspacing,"invAngstrom")
exit()
plane2 = MillerPlane(lattice=lattice, hkl=[0,1,1],)


hklMax=3
# sa = SaedAnalyzer(hklMax=3)
# sa.loadStructureFromCif(r'D:\CurrentProjects\colloborations\jyoti\zircon\zrSiO4tetragonal.cif')
# lattice = sa._lattice
# atomData = sa._atomData

hklList = MillerPlane.generatePlaneList(hklMax=hklMax,lattice=lattice,includeSymEquals=False)
plane = MillerPlane(hkl=[1,0,0],lattice=lattice)
print("the multiplicity for plane",str(plane), "is == " , plane.multiplicity())

d_spacingData = np.zeros((len(hklList)))
PlaneList = []
Braggd_spacing=[]
print (" Plane   d_spacing   TwoTheta  Intensity ")

calibValue = (665.8491841250539+ 635.2661695383725)/2.0
print(calibValue)

if wantDTable:

    print("### plane   d Spacing  1/dspacing   expected Length On pattern")
    for i, plane in enumerate(hklList):
        #PlaneList.append(str(plane))
        d_spacingData[i]=plane.dspacing
        #twoTheta = plane.get2theta(waveLength=lamda)
        #intensity = plane.diffractionIntensity(atomData=atomData)
        #if intensity>1e-2:
        print (plane, "   ", d_spacingData[i], "   ", 1./d_spacingData[i], (1./d_spacingData[i])*calibValue )


####
if calculateDirections:
    print("The magnitudes of the vectors are : dir1=",direction1.getMag()," dir2 = ",direction2.getMag())
    print("The cartesian vectors of the directions are : dir1=",direction1.getCartesianVec()," dir2 = ",direction2.getCartesianVec())
    
    print("The Angle between the vectors is : ",np.around(direction1.angle(direction2,units='deg'),3))
    
    direction1Sym=direction1.symmetricSet()
    direction2Sym=direction2.symmetricSet()
    print("Here is the Symmetric Set for the direction1")
    for item in direction1Sym:
        print(item,end=' ')
    
    print("\nHere is the Symmetric Set for the direction2")
    for item in direction2Sym:
        print(item, end=' ')
    print("\n")
    angleSetDirs = []
    for dir1 in direction1Sym:
        for dir2 in direction2Sym:
            ang = np.around(dir1.angle(dir2,units='deg'),3)
            angleSetDirs.append([dir1, dir2, ang,])
            print(str(dir1), str(dir2),ang)

exit()            
            
if calculatePlanes:
    d1 = plane1.getInterplanarSpacing()
    d2 = plane2.getInterplanarSpacing()
    print("The dspacing of the planes  : Plane1=",d1," Plane2 = ",d2)
    
    print("The d ratio is : d1/d2=",np.around(d1/d2,3), "d2/d1=",np.around(d2/d1,3))
    print("The planeNormal  : Plane1=",plane1.getPlaneNormal()," palne2 = ",plane2.getPlaneNormal())
    
    print("The Angle between the planes is : ",np.around(plane1.angle(plane2,units='deg'),3))
    
    plane1Sym=plane1.symmetricSet()
    plane2Sym=plane2.symmetricSet()
    print("Here is the Symmetric Set for the plane1")
    for item in plane1Sym:
        print(item,end=' ')
    
    print("\nHere is the Symmetric Set for the plane2")
    for item in plane2Sym:
        print(item, end=' ')
    print("\n")
    angleSetDirs = []
    for pln1 in plane1Sym:
        for pln2 in plane2Sym:
            ang = np.around(pln1.angle(pln2,units='deg'),3)
            angleSetDirs.append([dir1, dir2, ang,])
            zoneAixs=MillerPlane.getZoneAxis(pln1,pln2)
            print(str(pln1), str(pln2),ang,zoneAixs)
    
         
    

