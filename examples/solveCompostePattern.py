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
import pycrystallography.utilities.pymathutilityfunctions as pmt

from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
from copy import deepcopy
from tabulate import tabulate
import itertools
import matplotlib.pyplot as plt
import time

plt.close("all")
start = time.clock()
machineConditions2000FX ={"Voltage":160e3, ##unitsw Volts
                                "AlphaTilt":0., ## Degrees
                                "BetaTilt":0.,
                                "InPlaneRotaton":0.,
                                "CameraLength":1e3, ##in mm
                              }



cameraLengthCalibration={"20cm":1.94318,
                         "25cm":2.48864,
                         "30cm":3.23377,
                         "40cm":4.44444,
                         "50cm":5.49495,
                         "60cm":6.77273,
                         "80cm":9.28283,
                         "100cm":11.74242,
                         "120cm":14.07576,
                         "150cm":17.69697,
                         }
cameraLengthSFs={
                         "20cm":631.,
                         "25cm":None,
                         "30cm":1067.,
                         "40cm":1499.,
                         "50cm":1819.,
                         "60cm":2237.,
                         "80cm":3037.,
                         "100cm":3834.,
                         "120cm":None,
                         "150cm":None,
                         }


for i,item  in enumerate(cameraLengthSFs.keys()):
    if cameraLengthSFs[item] is not None:
        ratio = cameraLengthSFs[item]/cameraLengthCalibration[item]
    
        print("{:2d} cameraLengthSFs = {:.4f} cameraLengthCalibration = {:.1f}, Ratio = {:.3f}".format(i, 
              cameraLengthSFs[item],cameraLengthCalibration[item],ratio))
  

patternDetails = {
                    
                    "nb_20":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2034,1117], ##Origin
                                             [2224,1094], ##P1
                                             [2003,947], ## P2
                                             ]), #### NB 001 ref 20Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"20cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_20cm.tif",
                       
                            },
                    
                    "nb_30":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2056,1154], ##Origin
                                             [2376,1106], ##P1
                                             [2002,852], ## P2
                                             ]), #### NB 001 ref 30Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_30cm.tif",
                       
                            },
                        
                    "nb_40": {"imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_40cm.tif",
                            "lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                            "guessSpotsLocation":np.array([
                                             [2065,1165], ##Origin
                                             [2514 ,1096], ##P1
                                             [2441 , 676], ## P2
                                             ]), #### NB 001 ref 40Cm paaatern
                 
                            "spotsToIndex": [4,1],
                            "cameraLength":"40cm",
                            },
                    "nb_50":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2070, 1183], ##Origin
                                             [2615 ,1100], ##P1
                                             [1975 , 668], ## P2
                                             ]), #### NB 001 ref 50Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"50cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_50cm.tif",
                                  
                            },
                    "nb_60":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2143,1222], ##Origin
                                             [2813, 1118], ##P1
                                             [2038 , 587], ## P2
                                             ]), #### NB 001 ref 60Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"60cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_60cm.tif",
                       
                            },
                    "nb_80":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2277,1255], ##Origin
                                             [3187,1118], ##P1
                                             [2132,389], ## P2
                                             ]), #### NB 001 ref 80Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"80cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_80cm.tif",
                       
                            },
                    "nb_100":{"lattice": olt.cubic(3.3003),
                            "atomData" : [(1, np.array([0.,0.,0.])),
                                          (1, np.array([1./2,1./2,1./2.]))
                                          ],
                             "guessSpotsLocation":np.array([
                                             [2004,1238], ##Origin
                                             [3153,1065], ##P1
                                             [1838,145], ## P2
                                             ]), #### NB 001 ref 100Cm paaatern
                 
                            "spotsToIndex": [4,2],
                            "cameraLength":"100cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\NbCalib_100cm.tif",
                       
                            },
                  
                    "zone4MatrixOnly":{"lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1603,1451], ##Origin
                                             [1939,1259], ##P1 {0001}
                                             [1585,1125], ## P2 {01-11}
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [5,2],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone4\SAD_40cm_Alpha_-13Beta_-25_matrix_markedProcessed.tif",
                       
                            },
                    "zone4Composite":{"lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1644,1455], ##Origin
                                             [1700,1082], ##P1 {0001}
                                             [1983,1263], ## P2 {01-11}
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [5,2],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone4\SAD_40cm_Alpha_-13Beta_-25_composite.jpg",
                       
                            },
                    
                    "zr_zone5CompositeMatrix":{"lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1727,1512], ##Origin
                                             [1796,1173], ##P1 {0001}
                                             [2425,1280], ## P2 {01-11}
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [10,2],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone5\SAD_40cm_Alpha_20Beta_5_composite_matlabProcessed.tif",
                       
                            },
                                                  
                    "zr_zone5CompositeTwin":{"lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1727,1512], ##Origin
                                             [2330,1383], ##P1
                                            #[2251,1002],
                                             [1649,1125], ## P2
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [2,7],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone5\SAD_40cm_Alpha_20Beta_5_composite_matlabProcessed.tif",
                       
                            },
                  
                    "zr_zone5MatrixOnly":{"lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1700,1512], ##Origin
                                             [1624,1134], ##P1
                                            #[2251,1002],
                                             [1033,1252], ## P2
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [8,2],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone5\SAD_40cm_Alpha_20Beta_5_matrixProcessed.tif",
                       
                            },
                                    
                  
                    "zr_zone1213Matrix":{
                            "lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1635,1521], ##Origin
                                             [2002,1262], ##P1
                                            #[2251,1002],
                                             [1544,1028], ## P2
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [3,2],
                            "cameraLength":"40cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\1213\SAD_50cm_matrix_1213_procesed.tif",
                                                                             
                            },
                    "zr_zone1213CompTwin":{
                            "lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1973,1248], ##Origin
                                             [2315,1544], ##P1
                                            #[2251,1002],
                                             [1538,1389], ## P2
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [6,3],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\1213\SAD_50cm_composite_1213Processed.tif",
                                                                             
                            },                  
                      
                    "zone2_Matrix":{
                            "lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1605,1532], ##Origin
                                             [2055,1802], ##P1
                                            #[2251,1002],
                                             [1768,2014], ## P2
                                             ]), #### Zr pattern of zone 5
                 
                            "spotsToIndex": [6,4],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone2\SAD_40cm_Alpha_-4Beta_-17_composite.tif",
                                                                             
                            },
                  
                    "zone2_Composite":{
                            "lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1605,1532], ##Origin
                                             [2612,2013], ##P1
                                            #[2251,1002],
                                             [2017,1059], ## P2
                                             ]), #### Zr pattern of zone2 
                 
                            "spotsToIndex": [6,4],
                            "cameraLength":"30cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone2\SAD_40cm_Alpha_-4Beta_-17_composite.tif",
                                                                             
                            } ,
                    "zone4_Matrix":{
                            "lattice": olt.hexagonal(3.23,3.23*1.59),
                            "atomData" : [(40., np.array([0.,0.,0.])),
                                          (40., np.array([1./3., 2./3.,1./2.]))
                                         ],

                             "guessSpotsLocation":np.array([
                                             [1650,1458], ##Origin
                                             [2015,1586], ##P1
                                            #[2251,1002],
                                             [1982,1262], ## P2
                                             ]), #### Zr pattern of zone2 
                 
                            "spotsToIndex": [5,2],
                            "cameraLength":"40cm",
                            "imageFileName":r"D:\CurrentProjects\pushpa\SAD\zone4\SAD_40cm_Alpha_-13Beta_-25_composite.jpg",
                                                                             
                            }                                                             
                  
        }
        
                



crystal1Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'
crystal2Cif = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\Fe.cif'

choice="zr_zone5CompositeMatrix"

lattice = patternDetails[choice]["lattice"]
atomData =patternDetails[choice]["atomData"] 
guessSpotsLocation = patternDetails[choice]["guessSpotsLocation"]
spotsToIndex = patternDetails[choice]["spotsToIndex"]
cameraLength = patternDetails[choice]["cameraLength"]
imageFileName = patternDetails[choice]["imageFileName"]

# d1 = MillerPlane(hkl=[0,1,-1,1],lattice=lattice).dspacing
# d2 = MillerPlane(hkl=[1,0,-1,0],lattice=lattice).dspacing
# 
# print(d1/d2,d2/d1)



sa1 = SaedAnalyzer(lattice =lattice, atomData= atomData, hklMax=6,machineConditions=machineConditions2000FX,
                   considerDoubleDiffraction=False)
sa1.manuallyExtractSpotsFromExpPattern(imageFileName=imageFileName,guessSpotsLocation=guessSpotsLocation,displayImage=True,
                                        showSpotLines=False,indicateLineLengthInPlot=False)
#  

exit()
sol = sa1.solveSetOfSpots(spotIds=spotsToIndex, dTol=5,angleTol=2, maxHkl=4,cameraLength=cameraLength)

sa1.plotSolution(solId='all',markSpots=True)
    
print("Sol ID,  zoneAxis  spot1  spot2  CorreationError   Angle   AngleError  dRatio  dError noOfMatchingSpots")
for i in sol:
    print("{:2d} {:int} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:2d}".
                  format(i["solId"],i["zoneAxis"],i["spot1"],i["spot2"],
                   i["CorrelationError"],i["Angle"], i["AngleError"],
                   i["dRatio"],i["dError"],i["noOfMatchingSpots"]))
for i in sol:
            print(i["solId"],i["Calibration"])


