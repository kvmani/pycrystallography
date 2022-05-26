# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Admin
"""
from __future__ import division, unicode_literals

import sys
import os

# from examples.solveSAED import lattice

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))

from pycrystallography.core.millerDirection import MillerDirection
from pycrystallography.core.millerPlane import MillerPlane
from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation import Orientation

import numpy as np
import scipy
import math
import copy
from math import pi, sqrt
# from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.saedAnalyzer import SaedAnalyzer
from copy import deepcopy
from tabulate import tabulate
import itertools
import time

start = time.clock()

listOfTargetZoneAxis = [
    # [1,1,0],
    [1, 0, -1, -1],
    # #                          [-2,4,-2,3],
    #                           [1,1,-2,0],
    #
    #                         [0,1,-1,-2],
    #                         [-1,2,-1,-3],
]

misOri = Orientation(axis=[1, 1, 1], degrees=60)

oriCrystal1 = Orientation(euler=[0., 0., 0])
oriCrystal2 = Orientation(euler=[45. * np.pi / 180, 0., 0])
oriCrystal2 = oriCrystal1 * misOri
crystal1Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'
crystal2Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'

oriCrystal1 = Orientation(euler=[130.4 * np.pi / 180, 14.6 * np.pi / 180, 81.7 * np.pi / 180])
oriCrystal2 = Orientation(euler=[220.5 * np.pi / 180, 45.3 * np.pi / 180., 46.6 * np.pi / 180])

oriCrystal1 = Orientation(euler=[0 * np.pi / 180, 0 * np.pi / 180, 0 * np.pi / 180])
oriCrystal2 = Orientation(euler=[45 * np.pi / 180, 45 * np.pi / 180., 0 * np.pi / 180])

fccAtomData = [(1, np.array([0., 0., 0.])),
               (1, np.array([.5, .5, 0.])),
               (1, np.array([.5, 0., .5])),
               (1, np.array([0, .5, .5]))]
hcpAtomData = [(40., np.array([0., 0., 0.])),
               (40., np.array([1. / 3., 2. / 3., 1. / 2.]))]
bccAtomData = [(1, np.array([0., 0., 0.])),
               (1, np.array([.5, .5, .5])),
               ]
TargetZoneAxesCubic = [  # [1,1,1],
    [1, 1, 1],
]
TargetZoneAxesHcp = [  # [1,0,-1,-1],
    #                       [1,0,-1,1],
    # #                       [-1,1,0,1],
    #                       [0,1,-1,1],
    [-2, 4, -2, 3],
    # [-1,-1,2,3],
    # [1,-1,-0,0]
    #                           [1,1,-2,0],
    #
    #                         [0,1,-1,-2],
    #                         [-1,2,-1,-3],
]
inputData = {
    "Zrzone5": {
        "CrystalOris": [[348.8, 151.4, 0.0], [281.5, 162.5, 330]],  ## euler angles in degrees
        "holderTilts": [0, 0, 0],
        # "holderTilts":[33,30,90],
        "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
        "atomData": hcpAtomData,
        "TargetZoneAxes": TargetZoneAxesHcp,
    },
    "Zrzone1123": {"CrystalOris": [[144.8, 32.2, 210], [252, 180, 0]],  ## euler angles in degrees
                   "holderTilts": [0, 0, 0],
                   "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
                   "atomData": hcpAtomData,
                   "TargetZoneAxes": TargetZoneAxesHcp,
                   },
    "Zrzone4": {"CrystalOris": [[266., 147, 330], [32, 32, 150]],  ## euler angles in degrees
                "holderTilts": [0., 0., 0],
                "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
                "atomData": hcpAtomData
                },
    "Zr85idealTwin": {"CrystalOris": [[227, 19.8, 353.3], [221, 106.1, 238.8]],  ## euler angles in degrees
                      "holderTilts": [0., 0., 0],
                      "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
                      "atomData": hcpAtomData
                      },
    "Zr85idealTwinStd": {
        "angleAxis": (85.2, [2, -1, -1, 0],),
        "CrystalOris": [[0, 0, 0], [0, 85.2, 0]],  ## euler angles in degrees
        "holderTilts": [0., 0., 0],
        "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
        "atomData": hcpAtomData,
        "TargetZoneAxes": TargetZoneAxesHcp,
    },
    "Zr64@1100Compreesion": {
        "angleAxis": (64.204, [1, -1, 0, 0],),

        "holderTilts": [0., 0., 0],
        "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
        "atomData": hcpAtomData,
        "TargetZoneAxes": TargetZoneAxesHcp,
    },
    "Zr85TwinEBSD": {"CrystalOris": [[227, 19.8, 353.3], [221, 106.1, 238.8]],  ## euler angles in degrees
                     "holderTilts": [0., 0., 0],
                     "lattice": olt.hexagonal(3.23, 3.23 * 1.59),
                     "atomData": hcpAtomData,
                     "TargetZoneAxes": TargetZoneAxesHcp,
                     },
    "NiStdTwin": {"CrystalOris": [[24, 39, 21], [296, 36.4, 60.3]],
                  ## euler angles in degrees
                  "holderTilts": [0., 0., 0],
                  "lattice": olt.cubic(3.52),
                  "atomData": fccAtomData,
                  "TargetZoneAxes": TargetZoneAxesCubic,
                  },
    "NiStdTwin000": {"CrystalOris": [[0, 0, 0], [135, 109, 45]],  ## euler angles in degrees
                     # "CrystalOris" :[[0,0,0],[0,45,0]],
                     "holderTilts": [0., 0., 0],
                     "lattice": olt.cubic(3.52),
                     "atomData": fccAtomData,
                     "TargetZoneAxes": TargetZoneAxesCubic,
                     },
    "BurgersOr": {
        "angleAxis": (45, [2, -1, -1, 0],),
        # "CrystalOris" :[[0,0,0],[135,109,45]], ## euler angles in degrees
        # "CrystalOris" :[[0,0,0],[0,45,0]],
        "holderTilts": [0., 0., 0],
        "lattice1": olt.hexagonal(3.52, 1.59 * 3.52),
        "lattice2": olt.cubic(3.52),
        "atomData1": hcpAtomData,
        "atomData2": bccAtomData,
        "TargetZoneAxes": TargetZoneAxesCubic,
    }

}

choice = "NiStdTwin000"
alphaTilt, betaTilt, inPlaneRotation = inputData[choice]["holderTilts"]
lattice = inputData[choice]["lattice"]
# lattice2 = inputData[choice]["lattice2"]

atomData = inputData[choice]["atomData"]
# atomData2 =inputData[choice]["atomData2"]
listOfTargetZoneAxis = inputData[choice]["TargetZoneAxes"]

if "angleAxis" in inputData[choice]:
    angle = inputData[choice]["angleAxis"][0]
    axis = inputData[choice]["angleAxis"][1]
    eulerAngles1 = np.array([0, 0., 0])
    axis = MillerDirection(vector=axis, lattice=lattice)
    eulerAngles2 = Orientation(axis=axis.getUnitVector(), degrees=angle).getEulerAngles()
else:
    eulerAngles1 = np.array(inputData[choice]["CrystalOris"][0]) * np.pi / 180
    eulerAngles2 = np.array(inputData[choice]["CrystalOris"][1]) * np.pi / 180

oriCrystal1 = Orientation(euler=eulerAngles1)
oriCrystal2 = Orientation(euler=eulerAngles2)


crystal1Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'
crystal2Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'

machineConditions2000FX = {"Voltage": 160e3,  ##unitsw Volts
                           "AlphaTilt": 0.,  ## Degrees
                           "BetaTilt": 0.,
                           "InPlaneRotaton": 0.,
                           "CameraLength": 1e3,  ##in mm
                           }

hklList = [[2, -1, -1, 0], [1, 0 - 1, 0], [0, 0, 0, 1], [2, -1, -1, 3]]
hklList = [[0, 0, 1], [1, -1, 0], [1, 1, -1], [1, 2, 3]]

tmp = []
for item in listOfTargetZoneAxis:
    tmp.append(MillerDirection(vector=item, lattice=lattice))
listOfTargetZoneAxis = tmp

sa1 = SaedAnalyzer(name='matrix', lattice=lattice, considerDoubleDiffraction=True, atomData=atomData, hklMax=3,
                   machineConditions=machineConditions2000FX)
planeList = []
dirList = []
for item in hklList:
    planeList.append(MillerPlane(hkl=item, lattice=lattice))
    dirList.append(MillerDirection(vector=item, lattice=lattice))


# sterioData = sa1.calculateSterioGraphicProjectionDirection(dirList=None, centredOn=None, maxUVW=1, inPlaneRotation=0.)
# print(sterioData)
# sa1.plotSterioGraphicProjection(sterioData)

sa2 = SaedAnalyzer(name='twin', lattice=lattice, atomData=atomData, considerDoubleDiffraction=False, hklMax=4,
                   machineConditions=machineConditions2000FX)
# sa2.loadStructureFromCif(crystal2Cif)
crystal1lattice = sa2._lattice

saedData1 = sa1.calcualteSAEDpatternForTiltsAndRotation(oriCrystal1, alphaTilt=alphaTilt, betaTilt=betaTilt,
                                                        inPlaneRotation=inPlaneRotation)
# ax = sa1.plotSAED(
#     saedData1, marker="*r", markerSize =12, markSpots=True, shouldBlock=False,legendPosition=(0.1,.7))
saedData2 = sa2.calcualteSAEDpatternForTiltsAndRotation(oriCrystal2, alphaTilt=alphaTilt, betaTilt=betaTilt,
                                                        inPlaneRotation=inPlaneRotation)
# sa2.plotSAED(
#     saedData2, axisHandle=ax, marker=">g", markerSize =12, markSpots=True, shouldBlock=True,legendPosition=(0.7,.7))
fig = crysFig([saedData1,saedData2])
fig.plot(plotShow=True)
# currentZoneAxis = saedData1["zoneAxis"]
# for targetZoneAxis in listOfTargetZoneAxis:
#     print("targetZoneAxis=", targetZoneAxis)
#
#     symmetricZoneAxes = targetZoneAxis.symmetricSet()
#     for zoneAxis in symmetricZoneAxes:
#         sol = SaedAnalyzer.findTiltsForTargetZoneAxis(oriCrystal1, currentZoneAxis, zoneAxis, needExactTarget=True)
#         if not sol["Error"] == 1e5:
#             alphaTilt, betaTilt, inPlaneRotation = sol["alphaTilt"], sol["betaTilt"], sol["inPlaneRotation"]
#             #    alphaTilt,betaTilt,inPlaneRotation = 33,30,0
#
#             saedData1 = sa1.calcualteSAEDpatternForTiltsAndRotation(oriCrystal1, alphaTilt=alphaTilt, betaTilt=betaTilt,
#                                                                     inPlaneRotation=inPlaneRotation,
#                                                                     SAED_ANGLE_TOLERANCE=1.)
#             saedData2 = sa2.calcualteSAEDpatternForTiltsAndRotation(oriCrystal2, alphaTilt=alphaTilt, betaTilt=betaTilt,
#                                                                     inPlaneRotation=inPlaneRotation,
#                                                                     SAED_ANGLE_TOLERANCE=5.)
#             print("Zone Axes are {:int}{:int}".format(saedData1["zoneAxis"], saedData2["zoneAxis"]))
#
#             ax = sa1.plotSAED(
#                 saedData1)  # , marker="*r", markerSize =12, markSpots=True, shouldBlock=False,legendPosition=(0.1,.7))
#             sa2.plotSAED(
#                 saedData2)  # , axisHandle=ax, marker="og", markerSize =16, markSpots=True, shouldBlock=True, legendPosition=(0.7,.7))
#
#         else:
#             print("Not possible to achieve the Zone Axis " + str(zoneAxis))
