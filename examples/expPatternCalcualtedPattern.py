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
import matplotlib.pyplot as plt
import cv2

start = time.clock()

listOfTargetZoneAxis = [ #[1,1,0],
                              [1,0,-1,-1],
# #                          [-2,4,-2,3],
#                           [1,1,-2,0],
#                         
#                         [0,1,-1,-2],
#                         [-1,2,-1,-3],
                       ]


oriCrystal1 = Orientation(euler=[0.,0.,0])
crystal1Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'

primitiveAtomData=[(1, np.array([0.,0.,0.])),
                  ]
fccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))]
hcpAtomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))]
bccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,.5])),
               ]
TargetZoneAxesCubic = [#[1,1,1],
                       [1,1,1],
                       ]
TargetZoneAxesHcp =  [  #[1,0,-1,-1],
#                       [1,0,-1,1],
# #                       [-1,1,0,1],
#                       [0,1,-1,1],
                          [-2,4,-2,3],
                       # [-1,-1,2,3],
                          #[1,-1,-0,0]
#                           [1,1,-2,0],
#                         
#                         [0,1,-1,-2],
#                         [-1,2,-1,-3],
                       ]
inputData = {
    "Zrzone5":{
    "CrystalOris" :[[348.8,151.4,0.0],[281.5,162.5,330]], ## euler angles in degrees
             "holderTilts":[0,0,0],
             #"holderTilts":[33,30,90],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData,
             "TargetZoneAxes":TargetZoneAxesHcp,
               },
    "Zrzone1123":{"CrystalOris" :[[144.8,32.2,210],[252,180,0]], ## euler angles in degrees
             "holderTilts":[0,0,0],
             "lattice":olt.hexagonal(3.632,3.632*1.5925),
             "lattice":olt.hexagonal(3.232,3.232*1.5925),
             
             "atomData":hcpAtomData,
             "TargetZoneAxes":TargetZoneAxesHcp,
             },
    "Zrzone4":{"CrystalOris" :[[266.,147,330],[32,32,150]], ## euler angles in degrees
             "holderTilts":[0.,0.,0],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData
             },
    "Zr85idealTwin":{"CrystalOris" :[[227,19.8,353.3],[221,106.1,238.8]], ## euler angles in degrees
             "holderTilts":[0.,0.,0],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData
                     },
    "Zr85idealTwinStd":{
            "angleAxis":(85.2,[2,-1,-1,0],),
             "CrystalOris" :[[0,0,0],[0,85.2,0]], ## euler angles in degrees
             "holderTilts":[0.,0.,0],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData,
             "TargetZoneAxes":TargetZoneAxesHcp,
             },
    "Zr64@1100Compreesion":{
            "angleAxis":(64.204,[1,-1,0,0],),
             
             "holderTilts":[0.,0.,0],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData,
             "TargetZoneAxes":TargetZoneAxesHcp,
             },             
    "Zr85TwinEBSD":{"CrystalOris" :[[227,19.8,353.3],[221,106.1,238.8]], ## euler angles in degrees
             "holderTilts":[0.,0.,0],
             "lattice":olt.hexagonal(3.23,3.23*1.59),
             "atomData":hcpAtomData,
             "TargetZoneAxes":TargetZoneAxesHcp,
             },             
    "NiStdTwin":{"CrystalOris" :[[24,39,21],[296,36.4,60.3]],
                ## euler angles in degrees
             "holderTilts":[0.,0.,0],
             "lattice":olt.cubic(3.52),
             "atomData": fccAtomData,
             "TargetZoneAxes":TargetZoneAxesCubic,
                 }, 
    "NiStdTwin000":{"CrystalOris" :[[0,0,0],[135,109,45]], ## euler angles in degrees
             #"CrystalOris" :[[0,0,0],[0,45,0]],
             "holderTilts":[0.,0.,0],
             "lattice":olt.cubic(3.52),
             "atomData": fccAtomData,
             "TargetZoneAxes":TargetZoneAxesCubic,
                 },
    "BurgersOr":{
            "angleAxis":(45,[2,-1,-1,0],),
             #"CrystalOris" :[[0,0,0],[135,109,45]], ## euler angles in degrees
             #"CrystalOris" :[[0,0,0],[0,45,0]],
             "holderTilts":[0.,0.,0],
             "lattice1":olt.hexagonal(3.52, 1.59*3.52),
             "lattice2":olt.cubic(3.52),             
             "atomData1": hcpAtomData,
             "atomData2":bccAtomData,
             "TargetZoneAxes":TargetZoneAxesCubic,
            },
             
    "DeltaHydride":{ 
             "lattice":olt.cubic(4.47),
             "atomData": fccAtomData,
        },
    "ZrSiO4":{
             "angleAxis":(45,[2,-1,-1,0],),
             "CrystalOris" :[[0,0,0],[135,109,45]], ## euler angles in degrees 
             "lattice":olt.fromCif(r'D:\CurrentProjects\colloborations\jyoti\zircon\zrSiO4tetragonal.cif'),                        
             "atomData": primitiveAtomData,

        },
             
    "beta":{
             "lattice":olt.cubic(3.54),
             
             "atomData":bccAtomData,
            # "TargetZoneAxes":TargetZoneAxesBcc,
             },
                                
    }

patternDetails={
    "Zone1":{    
              "expPatternPath": r"D:\CurrentProjects\colloborations\jyoti\zircon\SAD-Z1-N14_5-P5-40cm.jpg",
              "patternOrigin" :[1989.32,1212.415],
              "scalingFactor" :661.08, ### corresponds to 40cm CameraLength.
              "patterRotation":30, ## degrees
              "holderTilts":[0.,0.,23.],
              "TargetZoneAxes":[
                                [3,1,5],
                               ],
            
              },
    "Zone2":{    
              "expPatternPath": r"D:\CurrentProjects\colloborations\jyoti\zircon\SAD-Z2-P9_5-N0_5-40cm.jpg",              
              "patternOrigin" :[1995.8375,1216.39],
              "scalingFactor" :661.08, ### corresponds to 40cm CameraLength.
              "scalingFactor" :661.08, ### corresponds to 40cm CameraLength.
              "patterRotation":30, ## degrees
              "holderTilts":[0.,0.,44.],
              "TargetZoneAxes":[
                                [1,6,10],
                               ],
            
              },
                
                
    "Zone3":{    
              "expPatternPath": r"D:\CurrentProjects\colloborations\jyoti\zircon\SAD-Z3-P27_5-N3_5-40cm.jpg",
              "patternOrigin" :[1982.8,1190.47],
              "scalingFactor" :661.08, ### corresponds to 40cm CameraLength.
              "patterRotation":30, ## degrees
              "holderTilts":[0.,0.,92],
              "TargetZoneAxes":[
                               [0,-2,1],
                               
                               ],
            
              },

  
    "Zone4":{    
              "expPatternPath": r"D:\CurrentProjects\colloborations\jyoti\zircon\SAD-Z4-P27-N13_5-40cm.jpg",
              "patternOrigin" :[2034.43,1275.13],
              "scalingFactor" :661.08, ### corresponds to 40cm CameraLength.
              "patterRotation":30, ## degrees
              "holderTilts":[0.,0.,-34.15],
              "TargetZoneAxes":[[9,10,5],
                               ],
            
              },

    "alphaZone1":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrix 1213 a 12.1 b 2.9.tif",
              "patternOrigin" :[645.27,733.07],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,57.],
              "TargetZoneAxes":[
                               [1,-2,1,-3],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
       
           "hydrideZone1":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrix 1216 a 1.5 b 18.7.tif",
              "patternOrigin" :[710.27,629.07],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,55],
              "TargetZoneAxes":[
                               [1,1,1],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
   
              "alphaZon2":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrix 1216 a 1.5 b 18.7.tif",
              "patternOrigin" :[710.27,629.07],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,58],
              "TargetZoneAxes":[
                               [1,1,-2,6],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
                         
   
              "matrixAwayFrmHydride":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrixAwayFromHydride.tif",
              "patternOrigin" :[666.0,696.25],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,-40],
              "TargetZoneAxes":[
                               [0,0,0,1],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
                         
              "matrixAwayFrmHydride2":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrixAwayFromHydride2_375mm.tif",
              "patternOrigin" :[666.0,711.25],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,-40],
              "TargetZoneAxes":[
                               [0,0,0,1],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
  
              "possibleBeta":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrix 1216 a 1.5 b 18.7.tif",
              "patternOrigin" :[709.0,628.25],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,-40],
              "TargetZoneAxes":[
                               [1,1,1],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
  
            "1216Final":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrix 1216 a 1.5 b 18.7_375mm_final.tif",
              "patternOrigin" :[712.27,584.07],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,58],
              "TargetZoneAxes":[
                               [1,1,-2,6],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
                         
                          
              "matrixAwayFrmHydride2Final":{
              "expPatternPath": r"D:\CurrentProjects\hydride_TEM\matrixAwayFromHydride2_375mm_0001Final.tif",
              "patternOrigin" :[666.0,711.25],
              "scalingFactor" :670.08, ### corresponds to 40cm CameraLength.
              "holderTilts":[0.,0.,-40],
              "TargetZoneAxes":[
                               [0,0,0,1],
                              #[2,2,-4,3]
                              #[1,-2,1,1]
                               ],
   },
  
  
  }        

choice="DeltaHydride"
choice="beta"
choice="Zrzone1123"
patternChoice="1216Final"

alphaTilt,betaTilt,inPlaneRotation = patternDetails[patternChoice]["holderTilts"]
lattice = inputData[choice]["lattice"]
atomData =inputData[choice]["atomData"] 
listOfTargetZoneAxis = patternDetails[patternChoice]["TargetZoneAxes"] 
expPatternPath=patternDetails[patternChoice]["expPatternPath"]
patternOrigin=patternDetails[patternChoice]["patternOrigin"]
scalingFactor=patternDetails[patternChoice]["scalingFactor"]
#patterRotation=patternDetails[patternChoice]["patterRotation"]
machineConditions2000FX ={"Voltage":160e3, ##unitsw Volts
                                "AlphaTilt":0., ## Degrees
                                "BetaTilt":0.,
                                "InPlaneRotaton":0.,
                                "CameraLength":1e3, ##in mm
                              }
        
print(lattice)
hklList = [[2,-1,-1,0],[1,0-1,0],[0,0,0,1],[2,-1,-1,3]]
hklList = [[0,0,1],[1,-1,0],[1,1,-1],[1,2,3]]

tmp = []

### for calibrtation

p1= MillerPlane(lattice=lattice,hkl=[1,3,0])
p2 = MillerPlane(lattice=lattice,hkl=[1,2,1])

p1LengthInPattern = 320.0 #### measured from imageJ
p2LengthInPattern = 241.0 #### measured from imageJ

calib1 = (1./p1.getInterplanarSpacing())/p1LengthInPattern
calib2 = (1./p2.getInterplanarSpacing())/p2LengthInPattern

print(1/calib1, 1/calib2, 1./p1.getInterplanarSpacing(),1./p2.getInterplanarSpacing())




for item in listOfTargetZoneAxis: 
    tmp.append(MillerDirection(vector=item,lattice=lattice))
listOfTargetZoneAxis=tmp    

sa1 = SaedAnalyzer(name='matrix',lattice =lattice, considerDoubleDiffraction=True,atomData= atomData, hklMax=3,machineConditions=machineConditions2000FX)
#sa1.extractSpotsFromExpPattern(imageFileName=expPatternPath, displayImage=True, minSpotSize=20, indicateLineLengthInPlot=True, showSpotLines=True, beamStopper=True)
planeList=[]
dirList=[]

fig=plt.gcf()
axes = fig.add_subplot(111,)
im=cv2.imread(expPatternPath)
axes.imshow(im)
saedData1 = sa1.calcualteSAEDpatternForTiltsAndRotation(oriCrystal1, alphaTilt=alphaTilt, betaTilt=betaTilt, inPlaneRotation=inPlaneRotation,scalingFactor=scalingFactor,patterCenter=patternOrigin)
currentZoneAxis = saedData1["zoneAxis"] 
for targetZoneAxis in listOfTargetZoneAxis:
    print("targetZoneAxis=", targetZoneAxis)
    sol = SaedAnalyzer.findTiltsForTargetZoneAxis(oriCrystal1,currentZoneAxis,targetZoneAxis,needExactTarget=True)
    if not sol["Error"]==1e5:        
            alphaTilt,betaTilt,rotation = sol["alphaTilt"],sol["betaTilt"],sol["inPlaneRotation"]
        #    alphaTilt,betaTilt,inPlaneRotation = 33,30,0
            
            saedData1 = sa1.calcualteSAEDpatternForTiltsAndRotation(oriCrystal1, alphaTilt=alphaTilt, betaTilt=betaTilt, 
                                            inPlaneRotation=inPlaneRotation,SAED_ANGLE_TOLERANCE=1.,scalingFactor=scalingFactor,patterCenter=patternOrigin)            
            ax = sa1.plotSAED(saedData1,figHandle=fig,axisHandle=axes,makeTransperent=False,markSpots=True)#, marker="*r", markerSize =12, markSpots=True, shouldBlock=False,legendPosition=(0.1,.7))
            


saedData2=sa1.calcualteSAEDpatternForTiltsAndRotation(oriCrystal1, alphaTilt=alphaTilt+22, betaTilt=betaTilt+8, inPlaneRotation=inPlaneRotation,scalingFactor=scalingFactor,patterCenter=patternOrigin)
# print("See now")
# ax = sa1.plotSAED(saedData2)#, marker="*r", markerSize =12, markSpots=True, shouldBlock=False,legendPosition=(0.1,.7))
#             
# plt.show()

print("Done")



