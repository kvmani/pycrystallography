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


from pycrystallography.core.orientation  import Orientation
import pycrystallography.utilities.graphicUtilities as gu

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
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


start = time.clock()
hklMax = 4
crystalChoice = 'hcp'
MaxPotentialSolutions = 1
D_TOLERANCE = 5. ##this is the tolerance for comparing the d_ratios of the planes
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
    expSpotData = [('N/L',1.5583,'N^L',88),
                 ('M/L',1.1791, 'M^L',32.3),
              #   ('M/N', 1.08, 'C^A', 61.38)
                ] ### for [2,-1,-1,0]
  

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
     
    expSpotData = [('A/B',1.0,'A^B',35.10), #(13-1), (-13-1)
                ('B/C',1.658, 'M^L',72.45),# (-13-1),(-2,0,0)
                
                ]### for [013] 
    expSpotData = [('A/B',1.722,'A^B',31.48), #(13-1), (220)
                ('B/C',1.633, 'M^L',90),# (220),(1-11)
                 
                ]### for [-112] 
#     expSpotData = [('A/B',1.0291,'A^B',13.63), #(2-82), (08-2)
#                 ('B/C',4.123, 'M^L',90),# (200),(082)                
#                 ]#### for [014] 

#
else :
    raise ValueError('Un known crstla system !!!!')         


# p1 = MillerPlane(hkl=[0,8,-2],lattice=lattice)
# p2 = MillerPlane(hkl=[2,8,-2],lattice=lattice)
# print(p1.angle(p2,'Deg'),"p1D = ",p1.dspacing,"  ", p2.dspacing)
# exit()


# plane1 = MillerPlane(lattice = lattice,hkl = [1,1,0], MillerConv='Bravais' )
# 
# totalSet = plane1.symmetricSet()
# plane2 = MillerPlane(lattice = lattice,hkl = [0,1,1], MillerConv='Bravais' ) 
# plane2Set = plane2.symmetricSet()
# for plane in plane2Set :
#     totalSet.append(plane)
# totalSet.append(plane1)
# totalSet.append(plane2)        

# uniqueSet = MillerPlane.uniqueList(totalSet, considerSymmetry=True)
# for i in uniqueSet :
#     print(i)
# exit()

planeList = MillerPlane.generatePlaneList(hklMax=hklMax,lattice=lattice)
allowedList=[] 
planeNameList=[] 
 
for i in planeList:
    print(i, i.dspacing)
print("N=", len(planeList))
print ("The elapsed time is ", time.clock() - start, "seconds")  

# p1 = MillerPlane(hkl=[1,1,1],lattice=lattice)
# p2 = MillerPlane(hkl=[1,1,1],lattice=lattice)
# for i in range(1000):   
#     p1.isSymmetric(p2, checkInputDataSanity=False)

cProfile.run('MillerPlane.generatePlaneList(hklMax=hklMax,lattice=lattice)', 'restats')     
print("Profiling done")

 
for plane in planeList :
    intensity = plane.diffractionIntensity(atomData=atomData)
    if intensity>1e-5 :
        allowedList.append(plane)
        planeNameList.append(plane.getLatexString())
d_ratio=[]  
set1=[]
set2 = [] 
plane1Plane2=[]  
for p1p2 in itertools.product(allowedList,repeat=2):
    dratio = p1p2[0].dspacing/p1p2[1].dspacing
    #print(dratio)
    if (np.abs(dratio-expSpotData[0][1])/expSpotData[0][1])*100<D_TOLERANCE :
        set1.append(p1p2)
    if (np.abs(dratio-expSpotData[1][1])/expSpotData[1][1])*100<D_TOLERANCE :
        set1.append(p1p2)
set11=[]  
set22=[]
for i,planes in enumerate(set1):
    #print(planes[0],planes[1])
    sol = MillerPlane.solvePlaneforIndexing(planes[0],planes[1],expSpotData[0][1],expSpotData[0][3],
                                          dspaceTolerance=D_TOLERANCE,angleTolerance=SAED_ANGLE_TOLERANCE)
    if sol is not None:
        #print("sol=", sol["zoneAxis"]," ",sol["spot1"]," ", sol["spot2"], " angError = ", sol["angError"])
        set11.append(sol)
    sol = MillerPlane.solvePlaneforIndexing(planes[0],planes[1],expSpotData[1][1],expSpotData[1][3],
                                          dspaceTolerance=D_TOLERANCE,angleTolerance=SAED_ANGLE_TOLERANCE)
    if sol is not None:
        #print("sol=", sol["zoneAxis"]," ",sol["spot1"]," ", sol["spot2"], " angError = ", sol["angError"])
        set22.append(sol)
        
print("after clean up")
set11 = sorted(set11 , key=lambda elem: "%f %f" % (elem['angError'], elem['dError']))
dirList = [i["zoneAxis"] for i in set11]
tmp, Indices = MillerDirection.uniqueList(dirList, returnInds=True)
set11 = [set11[i] for i in Indices]
print("after clean up")
set22 = sorted(set22 , key=lambda elem: "%f %f" % (elem['angError'], elem['dError']))
dirList = [i["zoneAxis"] for i in set22]
tmp, Indices = MillerDirection.uniqueList(dirList, returnInds=True)
set22 = [set22[i] for i in Indices]

print("First set sols")
#set11 = sorted(set11 , key=lambda elem: "%f %f" % (elem['angError'], elem['dError']))
for i in set11:
    print("ZoneAxis", i["zoneAxis"]," Spot1 =",i["spot1"]," Spot2= ", i["spot2"], 
          " angError = {:.2f}".format(i["angError"]), " % dError = {:.2f}".format(i["dError"]))
        
#set22 = sorted(set22 , key=lambda elem: "%f %f" % (elem['angError'], elem['dError']))
print("Second set solutions")
for i in set22:
    print("ZoneAxis", i["zoneAxis"]," Spot1 =",i["spot1"]," Spot2= ", i["spot2"], 
          " angError = {:.2f}".format(i["angError"]), " % dError = {:.2f}".format(i["dError"]))

print ("Done !!!") 
print ("The elapsed time is ", time.clock() - start, "seconds")     
       