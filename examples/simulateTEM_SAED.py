# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:55:59 2017

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
import sympy.geometry as gm

plt.rcParams['axes.facecolor'] = 'black'


def calcualteSAEDpattern(planeList,zoneAxis,atomData,patternxAxis=None,
                         SAED_ANGLE_TOLERANCE=1.,patternCalibration=None):
    """
    Unitlity fuinction to calcualting the SAED pattern given the planeList and zoneAxis.
     
    """
    if patterCalibration is None:
        patterCenter=np.array([0.,0.])
        scalingFactor = 1.
    else:
        patterCenter  = patterCalibration["patterCenter"]
        scalingFactor = patterCalibration["scalingFactor"]
        
        
        
    saedData=[]
    if not isinstance(patternxAxis,MillerDirection):
        patternxAxis = MillerDirection(vector=patternxAxis,isCartesian=True,lattice=planeList[0].realLattice)
    if abs((zoneAxis.angle(patternxAxis)-np.pi/2))<1e-5:
        xAxis = patternxAxis.getUnitVector()  
        yAxis = zoneAxis.cross(patternxAxis).getUnitVector()
        zAxis = zoneAxis.getUnitVector()
         
        for plane in planeList: 
            ang = np.arccos(np.clip(np.dot(plane.getUnitVector(),zAxis),-1,1))*180/np.pi
            if np.abs(90.-ang)<SAED_ANGLE_TOLERANCE :
                intensity = plane.diffractionIntensity(atomData=atomData) 
                if  intensity >1e-5:          
                    p = plane.getCartesianVec()
                    x = np.dot(p,xAxis)*scalingFactor+patterCenter[0]
                    y = np.dot(p,yAxis)*scalingFactor+patterCenter[1]                   
                    print ("plane = ",plane)
                    saedData.append({"Plane":plane,"XY":[x,y],"Intensity":intensity})    
        return saedData
             
    else:
        raise ValueError ("The supplied X axis is not  perpendicualr to Zone Axis !!!!")
              
    
def claculateSaedRotationNScaling(zoneAxis,currentXAxis, identifiedSpot,spotVectorInPattern):
    """
    This for calcualting the in plane roation required so that the solutuion (from idexing programe)
    can be plotted back for user confirmation.
    spotVectorInPattern = should be a numpy array of 000 spot coordinates (in m=image frame from which the spot was obtained)
    and the hkl spot for eg if the 000 spot occurs at pixel 100,100 and 222 spot at 150,170
    spotVectorInPattern = np.array([[100,100],[150,170]])
    """
    
    
    print("spot is ", identifiedSpot)
    if abs((zoneAxis.angle(currentXAxis)-np.pi/2))<1e-5: 
        xAxis = currentXAxis.getUnitVector()  
        yAxis = zoneAxis.cross(currentXAxis).getUnitVector()
        zAxis = zoneAxis.getUnitVector()
        x = np.dot(identifiedSpot.getCartesianVec(),xAxis)
        y = np.dot(identifiedSpot.getCartesianVec(),yAxis) 
        
        sourceLine = gm.Segment(gm.Point(0.,0.,0.), gm.Point(x,y,0.))
        x1 = spotVectorInPattern[0][0]
        y1 = spotVectorInPattern[0][1]
        x2 = spotVectorInPattern[1][0]
        y2 = spotVectorInPattern[1][1]
        targetLine = gm.Segment(gm.Point(x1,y1,0.), gm.Point(x2,y2,0.))
        patternRotationAngle = float(targetLine.angle_between(sourceLine))*180./np.pi
        patternScalingFactor = float(targetLine.length/sourceLine.length)
        result = {"patternRotationAngle":patternRotationAngle,"patternScalingFactor":patternScalingFactor}
        print(result)
        return result         
        
        
    else:
        raise ValueError("The supplied pattern X axis is not lying in the required Zone Axis!!!!")



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
#linecolors[:,:] =0.;
 
hklMax = 4
crystalChoice = 'hcp'
hideSystematicAbsentReflection = True
plotKikuchi = False
supeImposeExpPattern = False
applyAlphaBetaTilts=True
alphaFirst = True
tiltFirst = True
SAED_ANGLE_TOLERANCE=1 ##in degrees
stageAlpha = 0. #### in degrees
stageBeta = -0. #### in degrees
inPlaneRotation=0. ### in degree
finalInplaneRotation = 0. # in degree

hexLat = olt.hexagonal(1, 1.63)
vec = [3,0,0]
if supeImposeExpPattern :
    scalingFactor = 1000*(558.94/578.31)
    patterCenterX = 877
    patterCenterY = 935
    detectorScaleFactor = 5000
    lineWidthScalingFactor = 0.1
else:
    scalingFactor = 1.
    patterCenterX = 0
    patterCenterY = 0.
    detectorScaleFactor = 5.
    lineWidthScalingFactor = 0.05
#lattice = Lattice.cubic(1)
if crystalChoice=='hcp' :
    lattice = olt.hexagonal(3.23, 1.59*3.23)
    zoneAxis = MillerDirection(vector=[0,0,0,1],lattice=lattice)
    #zoneAxis = MillerDirection(vector=[0,-1,1,0],lattice=lattice)
    #zoneAxis = MillerDirection(vector=[1,1,-2,3],lattice=lattice)
    #zoneAxis = MillerDirection(vector=[1,-1,0,3],lattice=lattice)
    zoneAxis = MillerDirection(vector=[-6,5,1,3],lattice=lattice)
    zoneAxis = MillerDirection(vector=[1,-1,0,0],lattice=lattice)
    
    
    atomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))]

elif crystalChoice=='sc':
    lattice = olt.cubic(1)
    zoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice)
    atomData = [(1, np.array([0.,0.,0.])),
                 ] ### simple cubic  imaginary data
     
elif crystalChoice=='bcc':
    lattice = olt.cubic(3.3)
    #zoneAxis = MillerDirection(vector=[1,1,3],lattice=lattice)
    atomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([1./2,1./2,1./2.]))] ### bcc imaginary data
    zoneAxis = MillerDirection(vector=[1,0,0],lattice=lattice) 
    zoneAxis = MillerDirection(vector=[1,1,1],lattice=lattice) 
#      zoneAxis = MillerDirection(vector=[1,2,3],lattice=lattice)
#     zoneAxis = MillerDirection(vector=[0,1,4],lattice=lattice)
#   

elif crystalChoice=='fcc':
    lattice = olt.cubic(1)  
    zoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice) 
    zoneAxis = MillerDirection(vector=[-1,1,-1],lattice=lattice) 
# #     zoneAxis = MillerDirection(vector=[1,2,3],lattice=lattice)
#     zoneAxis = MillerDirection(vector=[4,-1,0],lattice=lattice)
#     zoneAxis = MillerDirection(vector=[3,2,-2],lattice=lattice)
#     
    
#      
      
    atomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))
                ] ### fcc imaginary data
else :
    raise ValueError('Un known crstla system !!!!')         


p1 = MillerPlane(hkl=[2,0,-2,1],lattice=lattice)
p2 = MillerPlane(hkl=[1,0,-1,0],lattice=lattice)

p1 = MillerPlane(hkl=[2,0,-2,1],lattice=lattice)
p2 = MillerPlane(hkl=[1,0,-1,0],lattice=lattice)


zoneAxis = MillerPlane.getZoneAxis(p1, p2)

print(zoneAxis, zoneAxis.integerize(AngleDeviationUnits="deg"))

exit()


a = range(-hklMax,hklMax+1)
hklList = []
for combination in itertools.product(a, a, a):
    hklList.append(combination)    
hklList.remove((0,0,0));

currentZoneAxis = MillerDirection(vector=[0,0,1],lattice=lattice)
targetZoneAxis = MillerDirection(vector=[2,-1,0],lattice=lattice)

o = Orientation.mapVector(sourceVector=currentZoneAxis.getUnitVector(),targetVector=targetZoneAxis.getUnitVector())
print(o.axis, o.angle*180/np.pi, o.rotation_matrix)
tilts = np.array([45,10])*np.pi/180
err = Orientation.objectiveFunctionFindTilts(
    tilts,currentZoneAxis.getUnitVector(),targetZoneAxis.getUnitVector())
sol = optimize.minimize(Orientation.objectiveFunctionFindTilts,[0., 0.],
                  args=(currentZoneAxis.getUnitVector(),targetZoneAxis.getUnitVector(),) ,
                  bounds=((-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4)), method="Nelder-Mead")
                                             
print(err, sol)
if (sol.fun<2):
    print (sol)
    print("the alpha and beta tilts are ", np.round(sol.x*180/np.pi,2))
    alphaRotation = Orientation(axis=[1,0,0],radians=sol.x[0])
    betaRotation = Orientation(axis=[0,1,0],radians=sol.x[1])
    totalRotation  = alphaRotation*betaRotation
    tempAxis = totalRotation.rotate(currentZoneAxis.getUnitVector())
    achievedZoneAxis = MillerDirection(vector = tempAxis, isCartesian=True, lattice=lattice)
    print("The actuval zone Axis will be",achievedZoneAxis )
    
else:
    print ('No satisafactory solutiuon found check your vectors !!!!')
#exit()

requiredAlphaTilt = currentZoneAxis

patternXAxis = MillerDirection(vector=[1.,0.,0.],isCartesian=True,lattice=lattice)
patternYAxis = MillerDirection(vector=[0.,1.,0.],isCartesian=True,lattice=lattice)
patternZAxis = MillerDirection(vector=[0.,0.,1.],isCartesian=True,lattice=lattice)


if zoneAxis.angle(patternZAxis)>1e-3 :
    initialOri = Orientation.mapVector(patternZAxis.getUnitVector(),zoneAxis.getUnitVector())
    patternXAxis.rotate(initialOri)
    patternYAxis.rotate(initialOri)
    patternZAxis.rotate(initialOri)
    
patternRotation = Orientation(axis=patternZAxis.getUnitVector(),degrees=inPlaneRotation)

if applyAlphaBetaTilts :
    alphaTilt = Orientation(axis=[1,0,0],degrees=stageAlpha)
    betaTilt  = Orientation(axis=[0,1,0],degrees=stageBeta)
    if alphaFirst :
        tiltOperation = alphaTilt*betaTilt
    else:
        tiltOperation = betaTilt*alphaTilt
    if tiltFirst :
        patternTotalRotation = tiltOperation*patternRotation
    else:
        patternTotalRotation = patternRotation*tiltOperation
        
    
#### following roation is to align the pattern rotation for new zone axis ideally should be 0 
finalPatternRotation = Orientation(axis=patternZAxis.getUnitVector(),
                                       degrees=finalInplaneRotation)

patternRotation = patternTotalRotation*finalPatternRotation   
patternXAxis.rotate(patternRotation)
patternYAxis.rotate(patternRotation)
patternZAxis.rotate(patternRotation)

zoneAxis.rotate(patternRotation)
print(" The modified zone axis =  and the total rotation is = ", zoneAxis, patternRotation)

Xaxis = patternXAxis.getUnitVector()
Yaxis = patternYAxis.getUnitVector()
Zaxis = patternZAxis.getUnitVector()

if not np.allclose([patternXAxis.dot(patternYAxis),patternXAxis.dot(patternZAxis), 
                patternYAxis.dot(patternZAxis)],[0.,0.,0]):
    print(patternXAxis,patternYAxis,patternZAxis)
    raise ValueError('The pattern reference frame got messed up!!!!')

# planeList = []
# for i in range(len(hklList)):
#     if lattice.is_hexagonal():
#         planeList.append(MillerPlane(lattice=lattice,hkl=hklList[i],MillerConv="Bravais"))
#     else :
#         planeList.append(MillerPlane(lattice=lattice,hkl=hklList[i]))
#     
planeList = MillerPlane.generatePlaneList(hklMax=hklMax,lattice=lattice)   
z = zoneAxis.getUVW()
allowedList=[]

Intensities = []
kikuchiLinedata = []
for i in range(len(planeList)) :
    localPlanes = planeList[i].symmetricSet()
    for localPlane in localPlanes:
        allowedList.append(localPlane)
        
        #print("the dot is ", np.abs(np.dot(plane,z)))
#         ang = np.arccos(np.clip(np.dot(localPlane.getUnitVector(),zoneAxis.getUnitVector()),-1,1))*180/np.pi
#         #if np.abs(np.dot(plane,z))<1e-6 :
#         if np.abs(90-ang)<SAED_ANGLE_TOLERANCE :
#             intensity = localPlane.diffractionIntensity(atomData=atomData)
#             if intensity>1e-5 :
#                 allowedList.append(localPlane)
#                 planeNameList.append(localPlane.getLatexString())
#                 Intensities.append(intensity)
#                 kikuchiLinedata.append(localPlane.kikuchiLinePoints(
#                     Xaxis,Yaxis,scalingFactor,
#                     patterCenterX,patterCenterY,
#                     detectorScaleFactor=detectorScaleFactor, lineWidthScalingFactor=lineWidthScalingFactor)) 
  
x = np.zeros((len(allowedList)+1,))
y = np.zeros((len(allowedList)+1,))

dspacings = np.array([1/i.dspacing for i in allowedList])
sortedDspacings = np.argsort(dspacings)


reflectionForMarking = [(i, allowedList[i]) for i in sortedDspacings]

indentifiedSpot = MillerPlane(hkl=[2,2,0],lattice=lattice)##[1,1,1] fcc
indentifiedSpot = MillerPlane(hkl=[0,0,2],lattice=lattice)##[-4,1,0] fcc
indentifiedSpot = MillerPlane(hkl=[0,2,2],lattice=lattice)##[-4,1,0] fcc


# o = Orientation.mapVector(sourceVector=indentifiedSpot,targetVector=np.array([1.,0.,0.]))
# patternxAxis = MillerDirection(vector=[1.,0.,0.],lattice=lattice)
# patternxAxis.rotate(o)

pc=np.array([333.,237.]) ##[1,1,1] fcc
pc=np.array([206.,194.]) ##[-4,1,0] fcc

Linedata = [gm.Segment(gm.Point2D(-91, 25)+pc, gm.Point2D(0, 0)+pc), 
           gm.Segment(gm.Point2D(-67, -67)+pc, gm.Point2D(0, 0)+pc)]##[1,1,1] fcc


Linedata = [gm.Segment(gm.Point2D(33, -14)+pc, gm.Point2D(0, 0)+pc), 
           gm.Segment(gm.Point2D(-58,-133)+pc, gm.Point2D(0, 0)+pc)]##[-4,1,0] fcc



spotVectorInPattern = np.array([[Linedata[0].points[0].x,Linedata[0].points[0].y],
                                [Linedata[0].points[1].x,Linedata[0].points[1].y]])

patternCalibration = claculateSaedRotationNScaling(zoneAxis=zoneAxis,currentXAxis=patternXAxis , 
                                                   identifiedSpot=indentifiedSpot,
                                                   spotVectorInPattern=spotVectorInPattern)


o = Orientation(axis=zoneAxis.getUnitVector(),angle=-patternCalibration["patternRotationAngle"]*np.pi/180.)

o = Orientation(axis=zoneAxis.getUnitVector(),angle=-37.66*np.pi/180.)

patternXAxis.rotate(o)
patterCalibration = {"patterCenter":pc,"scalingFactor":patternCalibration["patternScalingFactor"]}
patterCalibration = {"patterCenter":[7.86,2.74],"scalingFactor":3.2}

print("patterCalibration=", patterCalibration)
saedData = calcualteSAEDpattern(allowedList,zoneAxis,atomData=atomData,
                               patternxAxis=patternXAxis,patternCalibration = patternCalibration)
#input("press anykey")

# print(Xaxis, Yaxis, Zaxis)
# saedData = []
# for i in range(len(allowedList)):    
#     localSpot = allowedList[i]
#     p = localSpot.getCartesianVec()
#     x[i] = np.dot(p,Xaxis)*scalingFactor+patterCenterX
#     y[i] = np.dot(p,Yaxis)*scalingFactor+patterCenterY
#     print (i, localSpot, p, localSpot.dspacing, x[i],y[i])
#     saedData.append((i, localSpot, p, localSpot.dspacing, x[i],y[i]))    

x = [i["XY"][0] for i in saedData]
y = [i["XY"][1] for i in saedData]
planeNameList = [i["Plane"].getLatexString(forceInteger=False) for i in saedData]
x.append(0.+pc[0])
y.append(0.+pc[1])
#y[len(allowedList)] = 0.+patterCenterY
planeNameList.append("[0,0,0]")

if supeImposeExpPattern :   
    #img=mpimg.imread('D:\CurrentProjects\python_trials\pushpa_pycrystallography\images\zone_5_matrix_twin.png')
    pathName = r"D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\imageData\\"
    fileName=pathName+"fcc_[  1    1   1  ] .tif"
    fileName=pathName+"fcc_[  0    1   4  ] .png"
    
    
    img = mpimg.imread(fileName)

    
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

print("The list pf planes is ", allowedList)
### Generating useful info of SAD pattern
    
Intensities = np.array([i["Intensity"] for i in saedData])
Intensities = np.round(200*(Intensities/np.max(Intensities)))
Intensities = np.append(Intensities,[100.]) ## correspong to [0,0,0] spot
print(Intensities.shape)
plt.scatter(x, y, Intensities,color=[1,1,1])
plt.axes().set_aspect('equal', 'datalim')
plt.title("Zone axis = "+(str(zoneAxis)))


numOfColors = linecolors.shape
numOfColors = numOfColors[0]
#### plot kikuchis lines
if plotKikuchi :
    for i in range (len(kikuchiLinedata)) :
        #if i==18 :
            upperLine = kikuchiLinedata[i][1]
            lowerLine = kikuchiLinedata[i][2]
            textLocation = kikuchiLinedata[i][3]
            textAngle = kikuchiLinedata[i][4]
            #print (planeNameList[i], textLocation,textAngle, upperLine )
            upperPoint1 = upperLine[0,:]
            upperPoint2 = upperLine[1,:]
            lowerPoint1 = lowerLine[0,:]
            lowerPoint2 = lowerLine[1,:]             
            if hideSystematicAbsentReflection  :
                if  Intensities[i]>1e-5 :               
                    
                    
                    pass
                    #gu.broken2DLineWithText(upperPoint1,upperPoint2,planeNameList[i],lineFraction=-1,ls='dashed')
                    #gu.broken2DLineWithText(lowerPoint1,lowerPoint2,'',lineFraction=-1,ls='dashed')
                                        
            else :
                    #gu.broken2DLineWithText(upperPoint1,upperPoint2,planeNameList[i],lineFraction=-1,ls='dashed')
                    pass
                    #gu.broken2DLineWithText(lowerPoint1,lowerPoint2,'',lineFraction=-1,ls='dashed')
                        
# labels = [i for i in planeNameList]
# print (len(labels))
#for i in range(len(Intensities)):
# for label, x, y, Intensities in zip(planeNameList, x, y, Intensities):    
#         plt.annotate(
#                 label,xy=(x, y), xytext=(0, 10),
#                 textcoords='offset points', ha='center', va='center',
#                 bbox=dict(facecolor=[1,1,1], edgecolor='white', pad=0.0)
#                 )
plt.tight_layout() 

# Lables = ['A','B','C','D','E','F']
# ratioLabels = ['A/B= ','C/D= ','E/F= ']
# ratioLabels2 = [' A^B= ',' C^D= ',' E^F= ']
# 
# spotDataLabels = []
# point1 = np.array([0,0])
# count=0
# for i in range(len(reflectionForMarking)): 
#     point2 = np.array(saedData[reflectionForMarking[i][0]][4:6])
#     spot1 = saedData[reflectionForMarking[i][0]][1]
#     if i<len(reflectionForMarking)-1:
#         spot2 = saedData[reflectionForMarking[i+1][0]][1]
#         point22 = np.array(saedData[reflectionForMarking[i+1][0]][4:6])
#     else:
#         spot2 = saedData[reflectionForMarking[0][0]][1]
#         point22 = np.array(saedData[reflectionForMarking[0][0]][4:6])
#     d_ratio = np.round(spot2.dspacing/spot1.dspacing,3)
#     angle = np.round(spot2.angle(spot1,units='Deg'),1) 
#     if np.abs(angle-180)>10 and angle>10:
#         print(i,str(spot1),str(spot2),angle,d_ratio,count)
# #         gu.broken2DLineWithText(point1,point2,Lables[count*2],lineFraction=-1,lc=linecolors[count])
# #         gu.broken2DLineWithText(point1,point22,Lables[count*2+1],lineFraction=-1,lc=linecolors[count])
#         spotDataLabels.append([r'$d_{'+Lables [count*2]+r'}$ ='+str(np.round(1/spot1.dspacing,2))])
#         spotDataLabels.append([r'$d_{'+Lables [count*2+1]+r'}$ ='+str(np.round(1/spot1.dspacing,2))])
#         
#         spotDataLabels.append([ratioLabels [count]+' '+str(d_ratio)+ratioLabels2[count]+str(angle)])
#         count+=1
#         if count==3 :
#             break 
# 
# spotDataLabels.append('X-->'+patternXAxis.getLatexString())
# spotDataLabels.append('Y-->'+patternYAxis.getLatexString())
# spotDataLabels.append('Z-->'+patternZAxis.getLatexString())
# spotDataLabels.append(r'stage $\alpha$ = '+str(stageAlpha)+r'$^{o}$')
# spotDataLabels.append(r'stage $\beta$ = '+str(stageBeta)+r'$^{o}$')
# spotDataLabels.append(r'inplane rot \'calib\' = '+str(inPlaneRotation)+r'$^{o}$')
# spotDataLabels.append(r'inplane rot \'Final\' = '+str(finalInplaneRotation)+r'$^{o}$')
# # infoText = [str(i)+'\n' for i in spotDataLabels ]
# plt.annotate(infoText,xy=(0,0))

#plt.legend(spotDataLabels)
 
figName = '..\\data\\ImageData\\'+crystalChoice+'_'+(str(zoneAxis))
print(figName)
figName = str(figName)+'.png'
plt.savefig(figName)
plt.show()
 
#print(Intensities) 
print("Done !!!!")
#print(spotDataLabels)       
#print (planeNameList)



 
