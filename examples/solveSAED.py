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
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 


from pycrystallography.core.orientation  import Orientation
import pycrystallography.utilities.graphicUtilities as gu

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
import pycrystallography.utilities.pymathutilityfunctions as pmt 
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


start = time.clock()
hklMax = 3
crystalChoice = 'sc'
MaxPotentialSolutions = 1
D_TOLERANCE = 2. ##this is the tolerance for comparing the d_ratios of the planes
SAED_ANGLE_TOLERANCE=2. ##in degrees for matching the interplanar angles 
#lattice = Lattice.cubic(1)
if crystalChoice=='hcp' :
    lattice = olt.hexagonal(3.23, 1.59*3.23)
    
    atomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))]
    expSpotData = [('N/L',1.587,'N^L',90),
               ('M/L',1.876, 'M^L',57.79),
               ('M/N', 1.199, 'C^A', 32.21)
               ] ### for [0,1,-1,0]
#     expSpotData = [('N/L',1.09,'N^L',90),
#                ('M/L',1.139, 'M^L',28.62),
#                ('M/N', 1.08, 'C^A', 61.38)
#                ] ### for [2,-1,-1,0]
    

elif crystalChoice=='sc':
    lattice = olt.cubic(1,pointgroup='m-3m')
    zoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice)
    atomData = [(1, np.array([0.,0.,0.])
                 ),
                 ] ### simple cubic  imaginary data
     
elif crystalChoice=='bcc':
    lattice = olt.cubic(1,pointgroup='m-3m')
    zoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice)
    atomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([1./2,1./2,1./2.]))] ### bcc imaginary data

    expSpotData = [('A/B',1.0,'A^B',11.78), #(-6-2-2), (-6-1-3)
                ('B/C',1.658, 'M^L',90),# (-6-2-2),(01-1)
                  ]### for [-2,3,3]
    
    expSpotData = [('A/B',1.0,'A^B',25.84), #(-3-10), (-30-1)
                ('B/C',2.236, 'M^L',77.08),# (-30-1),(01-1)
                  ]### for [-1,3,3]
     
elif crystalChoice=='fcc':
    lattice = olt.cubic(1,pointgroup='m-3m')  
    zoneAxis = MillerDirection(vector=[1,2,3],lattice=lattice)   
    atomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))
                ] ### fcc imaginary data
    expSpotData = [('A/B',1.155,'A^B',54.74),
               ('B/C',1.000, 'M^L',70.52),
               ] ### for [0 1 1]
   
    expSpotData = [('A/B',1.0,'A^B',60.),
               ('B/C',1.000, 'M^L',60.),
               ] ### for [1 1 1]
    expSpotData = [('A/B',1.0,'A^B',70.52), #(11-1), (-11-1)
                ('B/C',1.0, 'M^L',109.48), #(-1,1-1),(-1,-1,1)                
                ] ### for [0 1 1] ## another set of spots for [011]
     
#     expSpotData = [('A/B',1.0,'A^B',35.10), #(13-1), (-13-1)
#                 ('B/C',1.658, 'M^L',72.45),# (-13-1),(-2,0,0)
#                 
#                 ]### for [013] 
#     expSpotData = [('A/B',1.722,'A^B',31.48), #(13-1), (220)
#                 ('B/C',1.633, 'M^L',90),# (220),(1-11)
#                  
#                 ]### for [-112] 
#     expSpotData = [('A/B',1.0291,'A^B',13.63), #(2-82), (08-2)
#                 ('B/C',4.123, 'M^L',90),# (200),(082)                
#                 ]#### for [014] 

#
elif crystalChoice=='ZrSiO4':
    
        rootDir = r'D:\CurrentProjects\colloborations\jyoti\zircon\\'
        cifName=rootDir+'zrSiO4tetragonal.cif'
        lattice = olt.fromCif(cifName)  
        zoneAxis = MillerDirection(vector=[1,2,3],lattice=lattice)   
        atomData = [(1, np.array([0.,0.,0.])),                    
                    ] ### 
        
        expSpotData = [('A/B',1.155,'A^B',54.74),
                   ('B/C',1.000, 'M^L',70.52),
                   ] ### for [0 1 1]
       
        expSpotData = [('A/B',1.0,'A^B',60.),
                   ('B/C',1.000, 'M^L',60.),
                   ] ### for [1 1 1]
        expSpotData = [('A/B',1.0,'A^B',70.52), #(11-1), (-11-1)
                ('B/C',1.0, 'M^L',109.48), #(-1,1-1),(-1,-1,1)                
                ] ### for [0 1 1] ## another set of spots for [011]
        
        #expPatternPath=rootDir+'SAD-Z1-N14_5-P5-40cm.jpg'
        #expPatternPath=rootDir+'SAD-Z3-P27_5-N3_5-40cm.jpg'
        expPatternPath=rootDir+'SAD-Z4-P27-N13_5-40cm.jpg'
elif crystalChoice=="Hydride":
        rootDir = r'D:\CurrentProjects\colloborations\jyoti\zircon\\'
        
        lattice = olt.cubic(4.77)  
        zoneAxis = MillerDirection(vector=[1,2,3],lattice=lattice)   
        atomData =atomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))] ,        
        
else :
    raise ValueError('Un known crstla system !!!!')         


#fileName = "D:\CurrentProjects\python_trials\pushpa_pycrystallography\images\zone_5_matrix_tmp.png"
# pathName = r"D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\imageData\\"
# fileName=pathName+"fcc_[  1    1   1  ] .tif"
# #fileName=pathName+"fcc_[  0    1   4  ] .png"
# pathName = r"D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\imageData\realData\13-2-19\\"
# fileName=pathName+'processed.tif'
#fileName=pathName+"hcp_[  0    0   0   1 ] .png" 


# expSpotData = gu.extractSpotDatafromSAEDpattern(imageFile=expPatternPath)
# imageData = expSpotData["imageData"]
# print("pc = ", imageData["patternCenter"])
# expSpotData = expSpotData["spotData"]
# exit()
expSpotData = [{'LineIds': '1&2', 'LineData': [Segment(Point2D(7.86, 2.74), Point2D(8.38, 1.55)), Segment(Point2D(9.71,3.5), Point2D(7.86, 2.74))], 
               'd_ratio': 1.5581, 'Angle': 88.5}]

expSpotData=[#{'LineIds': '1&2', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-172, 170))], 'd_ratio': 1.4826585755591513, 'Angle': 41.46738300754003},
# {'LineIds': '1&3', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-16, -319))], 'd_ratio': 1.1225958024417082, 'Angle': 90.32619448347297},
{'LineIds': '1&4', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-373, -300))], 'd_ratio': 1.3349971599244381, 'Angle': 42.00691421560999},
# {'LineIds': '1&5', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(173, -170))], 'd_ratio': 1.4783046909272097, 'Angle': 138.69867426273015},
# {'LineIds': '1&6', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-546, -129))], 'd_ratio': 1.5646889321933877, 'Angle': 16.490706987934672},
]

expSpotData=[#{'LineIds': '1&2', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-172, 170))], 'd_ratio': 1.4826585755591513, 'Angle': 41.46738300754003},
# {'LineIds': '1&3', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-16, -319))], 'd_ratio': 1.1225958024417082, 'Angle': 90.32619448347297},
{'LineIds': '1&4', 'LineData': [Segment2D(Point2D(0, 0), Point2D(10.0, 0.)), Segment2D(Point2D(0, 0), Point2D(0.,10.09))], "spotXyData":[[0,0],[10,0],[0,10.09]]},
# {'LineIds': '1&5', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(173, -170))], 'd_ratio': 1.4783046909272097, 'Angle': 138.69867426273015},
# {'LineIds': '1&6', 'LineData': [Segment2D(Point2D(0, 0), Point2D(-358, 20)), Segment2D(Point2D(0, 0), Point2D(-546, -129))], 'd_ratio': 1.5646889321933877, 'Angle': 16.490706987934672},
]



for i in expSpotData:
    print("The line data = ",i)

planeList = MillerPlane.generatePlaneList(hklMax=hklMax,lattice=lattice)
allowedList=[] 
planeNameList=[] 
 
for i in planeList:
    print(i, i.dspacing)
print("N=", len(planeList))

for plane in planeList :
    intensity = plane.diffractionIntensity(atomData=atomData)
    if intensity>1e-5 :
        allowedList.append(plane)
        planeNameList.append(plane.getLatexString())
d_ratio=[]  
set1=[]
set2 = [] 
plane1Plane2=[]
print ("Now starting the actual one  elapsed time is ", time.clock() - start, "seconds")
lookUpTable1 = []
for p1p2 in itertools.product(allowedList,repeat=2):
    lookUpTable1.append((p1p2[0],p1p2[1],max(p1p2[0].dspacing/p1p2[1].dspacing, 1/p1p2[0].dspacing/p1p2[1].dspacing)))

dArray = np.asarray([i[2]  for i in lookUpTable1])
solSets=[]

for i in range(len(expSpotData)):
    xyData = expSpotData[i]["spotXyData"]    
    line1 = np.array([xyData[0],xyData[1]])
    line2 = np.array([xyData[0],xyData[2]])
    angleBetweenSpots = pmt.angleBetween2Lines(line1, line2,units="Deg")    
    x1 = xyData[1][0]
    y1 = xyData[1][1]                                
    x2 = xyData[2][0]
    y2 = xyData[2][1]   
    angleBetweenSpots = np.abs(np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)*180./np.pi)    
    spot1Length = np.sqrt((xyData[1][0]-xyData[0][0])**2+(xyData[1][1]-xyData[0][1])**2) 
    spot2Length = np.sqrt((xyData[2][0]-xyData[0][0])**2+(xyData[2][1]-xyData[0][1])**2)                              
    dRatioMeasured = max(spot1Length/spot2Length, spot2Length/spot1Length)  
    expSpotData[i]["d_ratio"] = dRatioMeasured
    expSpotData[i]["Angle"]=angleBetweenSpots
    
for i in range(len(expSpotData)):
    lowLimit = expSpotData[i]["d_ratio"]-expSpotData[i]["d_ratio"]*D_TOLERANCE/100.
    upLimit =  expSpotData[i]["d_ratio"]+expSpotData[i]["d_ratio"]*D_TOLERANCE/100.
    solSets.append(np.where((dArray >= lowLimit) & (dArray <=upLimit)))
   
sa = SaedAnalyzer(lattice=lattice)
diffPatterns=[]
for i,set in enumerate(solSets):
    xyData=expSpotData[i]["spotXyData"]
    for item in set:
        spot1,spot2Tmp = lookUpTable1[item[0]][0],lookUpTable1[item[0]][1]
        spot0 = MillerPlane(hkl=[0,0,0],lattice=lattice)
        spot2Sym = spot2Tmp.symmetricSet()
        for spot2 in spot2Sym: 
            tmp = sa.calcualtePatternFrom3Spots(spotData=[spot0,spot1,spot2],xyData=xyData,allowedAngleDeviation=2,allowedDRatioDevitationPercent=10)
            if tmp is not None:
                diffPatterns.append(tmp)

if len(diffPatterns)>0:
    for i in diffPatterns:
        print(i["zoneAxis"])
        sa.plotSAED(saedData=i, plotShow=True,  markSpots=True)

print("Done")
