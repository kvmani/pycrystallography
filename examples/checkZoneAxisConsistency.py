# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os
from pycrystallography.core.millerPlane import MillerPlane
from operator import itemgetter

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

ignoreSymEquivalnets=True
cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
cifName =cifPathName+'pnma-Cr7C3.cif'
cifName =cifPathName+'FeFcc.cif'

inputData = {"pushpaTwinsSaeds":
             {"cifName":"Alpha-ZrP63mmc.cif",
              "zoneData":[{"zoneID":"zone2", "zoneAxis":[2, 2, -4, 3], "alphaBeta":[-4, -17],},
                          {"zoneID":"zone3", "zoneAxis":[-1, -1, 2, 1], "alphaBeta":[-10, -31],},
                          {"zoneID":"zone4", "zoneAxis":[-1,-1,2,3], "alphaBeta":[-13, -25],},
                          {"zoneID":"zone5", "zoneAxis":[0,1,-1,0], "alphaBeta":[20,5],},
                           ]
              },
             "Ti2O3_paper":
                {
                "cifName":"o3 ti2Rhobohedral.cif",
                "zoneData":[#{"zoneID":"zone1", "zoneAxis":[1,5,-1], "alphaBeta":[-14.9, 27.2],},
#                           {"zoneID":"zone2", "zoneAxis":[4,8,-1], "alphaBeta":[-16.5, 5.2],},
#                           {"zoneID":"zone3", "zoneAxis":[10,14,-1], "alphaBeta":[-16.9, -11.4],},
#                           {"zoneID":"zone4", "zoneAxis":[1,1,0], "alphaBeta":[-16.1, -31.0],},
#                           {"zoneID":"zone5", "zoneAxis":[10,-8,1], "alphaBeta":[4.2, -25.9],},
                          {"zoneID":"zone6", "zoneAxis":[7, 5, -1], "alphaBeta":[12.7, -25.5],},
                          {"zoneID":"zone7", "zoneAxis":[1, 2, -1], "alphaBeta":[16.3, 24.3],},
                           ]
                 }
             }

choice = "Ti2O3_paper"
cifName =cifPathName+inputData[choice]["cifName"]

lattice = olt.fromCif(cifName)
print(lattice)

zoneData = inputData[choice]["zoneData"] 

for zone in zoneData:
    zone["zoneAxisMiller"] = MillerDirection(vector=zone["zoneAxis"],lattice=lattice)
firstZone = zoneData[0]
degree = np.pi/180
data=[]
for i in range(len(zoneData)-1):
    angleBetweenZonesMeasured = np.around((np.linalg.norm(np.array(zoneData[i+1]["alphaBeta"])-np.array(zoneData[i]["alphaBeta"]))),2)
    expAngle2 =  np.cos((zoneData[i+1]["alphaBeta"][1]-zoneData[i]["alphaBeta"][1])*degree)*np.cos(zoneData[i]["alphaBeta"][0]*degree)*np.cos(zoneData[i+1]["alphaBeta"][0]*degree)+ \
                      np.sin(zoneData[i]["alphaBeta"][0]*degree)*np.sin(zoneData[i+1]["alphaBeta"][0]*degree) ### usaing fromula from ref TEM Specimen Tilting Holder By Dr. X.Z. “Jim” Li, NCMN Specialist
    
    expAngle2 = np.around(np.arccos(expAngle2)/degree,2)
    zone1Miller = zoneData[i]["zoneAxisMiller"]
    zone2Miller = zoneData[i+1]["zoneAxisMiller"]
    if ignoreSymEquivalnets:
        zone2MillerSym = [zone2Miller]
        zone1MillerSym = [zone1Miller]
    else:
        zone2MillerSym = zone2Miller.symmetricSet()
        zone1MillerSym = zone1Miller.symmetricSet()
    
    
    for j, item in enumerate(zone2MillerSym):
        for k, item1 in enumerate(zone1MillerSym):
            angle = np.around(item.angle(item1, units="Deg", considerSymmetry=False),2)
            deviation = np.around((abs(angleBetweenZonesMeasured-angle)),2)
            if (deviation<100.):
                zoneAxesCombination = zoneData[i+1]["zoneID"]+" "+zoneData[i]["zoneID"] 
                alphaBeta1 = np.around(np.array(zoneData[i]["alphaBeta"]),2)
                alphaBeta2 = np.around(np.array(zoneData[i+1]["alphaBeta"]),2)
                
                data.append({"ZoneAxes": zoneAxesCombination, "dir1":"{:int}".format(item), "dir2":"{:int}".format(item1), "angle":angle , "ExpAngle":angleBetweenZonesMeasured,
                         "expAngle2":expAngle2, "zone1AlphaBeta":alphaBeta1, "zone2AlphaBeta":alphaBeta2, "deviation":deviation})
            
    
data = sorted(data,key=itemgetter('deviation', "angle"))
for item in data:
    print(item)


