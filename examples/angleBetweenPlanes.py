# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:55:59 2017

@author: Admin
"""

from __future__ import division, unicode_literals

import sys
import os
from pycrystallography.core import orientation

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.saedAnalyzer import SaedAnalyzer
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation  import Orientation

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from copy import deepcopy
from tabulate import tabulate
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np

linecolors = np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1],
                      [1,1,0],
                      [0,1,1],
                      [1,0,1],
                      [0.5,1,1],
                      [1,0.5,1],
                      [1,1,0.5],
                      [0.3,0.3,1],
                      [1,0.3,0.3],
                      [0.3,1,0.3],
                      [0.6,0.6,0.],
                      [0.,0.6,0.6],
                      [0.6,0.6,0.],
                      ])
linecolorsComplement = 1.-linecolors
linecolors = np.append(linecolors,linecolorsComplement,axis=0)
#print(linecolors.shape)
#exit()


hklMax = 2
a = range(-hklMax,hklMax+1)
hklList = []
for combination in itertools.product(a, a, a):
    hklList.append(combination)    
hklList.remove((0,0,0));
lattice = Lattice.hexagonal(3.23, 1.59*3.23)
lattice = olt.cubic(1)
atomData = [(40., np.array([0.,0.,0.])),
            (40., np.array([1./3., 2./3.,1./2.]))]

# atomData = [(1, np.array([0.,0.,0.])),
#             ] ### simple cubic  imaginary data
# 

# atomData = [(1, np.array([0.,0.,0.])),
#             (1, np.array([1./2,1./2,1./2.]))] ### bcc imaginary data
# 
# atomData = [(1, np.array([0.,0.,0.])),
#             (1, np.array([.5,.5,0.])),
#             (1, np.array([.5,0.,.5])),
#             (1, np.array([0,.5,.5]))
#             ] ### fcc imaginary data

        
# lattice = Lattice.cubic(1)
zoneAxis1 = MillerDirection(vector=[1,1,1],lattice=lattice)
zoneAxis2 = MillerDirection(vector=[1,3,1],lattice=lattice)
symzoneaxis2=zoneAxis2.symmetricSet()
for dir in symzoneaxis2:
    angle=zoneAxis1.angle(dir, units='deg')
    print ('zoneAxis1  zoneAxis2 angle',zoneAxis1, dir, angle )
    

sa = SaedAnalyzer(lattice=lattice,hklMax=2)
solution = sa.findTiltsForTargetZoneAxis(Orientation(0,0,0),zoneAxis1, zoneAxis2,needExactTarget=True)
print(solution)
exit()



plotKikuchi = False
lineWidthScalingFactor = 0.1
#zoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice)

# atomData = [(40, np.array([0.,0.,0.])),
#             (40, np.array([1./3., 2./3.,1./2.]))]
hideSystematicAbsentReflection = False

patternXAxis = MillerDirection(vector=[1.,0.,0.],isCartesian=True,lattice=lattice)
patternYAxis = MillerDirection(vector=[0.,1.,0.],isCartesian=True,lattice=lattice)
patternZAxis = MillerDirection(vector=[0.,0.,1.],isCartesian=True,lattice=lattice)
if zoneAxis.angle(patternZAxis)>1e-3 :
    initialOri = Orientation.mapVector(patternZAxis.getUnitVector(),zoneAxis.getUnitVector())
    patternXAxis.rotate(initialOri)
    patternYAxis.rotate(initialOri)
 

patternRotation = Orientation(axis=zoneAxis.getUnitVector(),degrees=90.)
scalingFactor = 1000*(558.94/578.31)
patterCenterX = 877
patterCenterY = 935
detectorScaleFactor = 5000

supeImposeExpPattern = True
patternXAxis.rotate(patternRotation)
patternYAxis.rotate(patternRotation)
Xaxis = patternXAxis.getUnitVector()
Yaxis = patternYAxis.getUnitVector()
    

planeList = []
for i in range(len(hklList)):
    temp = MillerPlane(lattice=lattice,hkl=hklList[i],MillerConv="Bravis")
    planeList.append(MillerPlane(lattice=lattice,hkl=hklList[i],MillerConv="Bravis"))
    
    
z = zoneAxis.getUVW()
allowedList=[]
planeNameList=[]
Intensities = []
kikuchiLinedata = []
for i in range(len(planeList)) :
    localPlane = planeList[i]
    plane = localPlane.gethkl()
    if np.abs(np.dot(plane,z))<1e-6 :         
        allowedList.append(localPlane)
        planeNameList.append(localPlane.getLatexString(forceInteger=False))
        Intensities.append(localPlane.diffractionIntensity(atomData=atomData))
        kikuchiLinedata.append(localPlane.kikuchiLinePoints(
            Xaxis,Yaxis,scalingFactor,
            patterCenterX,patterCenterY,
            detectorScaleFactor=detectorScaleFactor, lineWidthScalingFactor=lineWidthScalingFactor)) 
              
     
   
x = np.zeros((len(allowedList)+1,))
y = np.zeros((len(allowedList)+1,))

for i in range(len(allowedList)):
    localSpot = allowedList[i]
    p = localSpot.getCartesianVec()
    x[i] = np.dot(p,Xaxis)*scalingFactor+patterCenterX
    y[i] = np.dot(p,Yaxis)*scalingFactor+patterCenterY
    print (i, localSpot, p, Xaxis, Yaxis, x[i],y[i])
    
    
x[len(allowedList)] = 0.+patterCenterX
y[len(allowedList)] = 0.+patterCenterY
planeNameList.append("[0,0,0]")

if supeImposeExpPattern :   
    img=mpimg.imread('D:\CurrentProjects\python_trials\pushpa_pycrystallography\images\zone_5_matrix_twin.png')
    imSize = img.shape
    print("before = ", img[0,0])
    img[:,0:10]=1
    img[:,imSize[1]-11:imSize[1]-1]=1

    img[0:10,:]=1
    img[imSize[0]-11:imSize[0]-1,:]=1

    print("after = ", img[0,0])
    print(img.shape)
    print(np.max(np.max(img)))
    implot = plt.imshow(img,clim=(0.4, 0.9))
    plt.tight_layout() 
    plt.set_cmap('Greys')


### Generating useful info of SAD pattern
    
Intensities = np.array(Intensities)
Intensities = np.round(100*(Intensities/np.max(Intensities)))
Intensities = np.append(Intensities,[100.]) ## correspong to [0,0,0] spot
print(Intensities.shape)
plt.scatter(x, y, Intensities)
plt.axes().set_aspect('equal', 'datalim')
plt.title("Zone axis = "+(str(zoneAxis)))


numOfColors = linecolors.shape
numOfColors = numOfColors[0]
#### plot kikuchis lines
if plotKikuchi :
    for i in range (len(kikuchiLinedata)) :
        
            upperLine = kikuchiLinedata[i][1]
            lowerLine = kikuchiLinedata[i][2]
            textLocation = kikuchiLinedata[i][3]
            textAngle = kikuchiLinedata[i][4]
            print (planeNameList[i], textLocation,textAngle )
            plt.plot(upperLine[:,0], upperLine[:,1],marker='o',color=linecolors[i%numOfColors])
            plt.plot(lowerLine[:,0], lowerLine[:,1],marker='*',color=linecolors[i%numOfColors])
            plt.annotate(
                        planeNameList[i],xy=textLocation, xytext=(0, 5),
                        rotation = textAngle, textcoords='offset points', ha='center', va='center',
                        color=linecolors[i%numOfColors])
            
        

#plt.title(r'$\alpha > \beta$')
labels = [i for i in planeNameList]
print (len(labels))
for label, x, y, Intensities in zip(labels, x, y, Intensities):
    if hideSystematicAbsentReflection :
        if Intensities >0. :
            plt.annotate(
                label,xy=(x, y), xytext=(0, 5),
                textcoords='offset points', ha='center', va='center',
                )
        
    else:
        plt.annotate(
            label,xy=(x, y), xytext=(0, 5),
                textcoords='offset points', ha='center', va='center',
                ) 

plt.tight_layout()  
plt.show() 
figName = zoneAxis.getUVW()
figName = str(figName.astype(int))+'.png'
plt.savefig(figName)
 
print(Intensities) 
print("Done !!!!")       
#print (planeNameList)



 
