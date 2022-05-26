from __future__ import division, unicode_literals
import math
import itertools
import warnings

from six.moves import map, zip

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians
from pycrystallography.core.quaternion  import Quaternion
from pycrystallography.core.orientedLattice import  OrientedLattice as olt  
#import pycrystallography.utilities.pytransformations as pt
#import pycrystallography.utilities.pytransformations as pt
#import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pymathutilityfunctions as pmt
#import pycrystallography.utilities.pymathutilityfunctions as pmt 
from pymatgen.core.lattice import Lattice as lt
from pycrystallography.utilities import pyCrystUtilities as pyCrysUt
from monty.json import MSONable
from monty.dev import deprecated

#from vtk import vtkVector3d
from math import sqrt
from numpy import pi, dot, transpose, radians
from pymatgen.core import Lattice
from pycrystallography.core.orientation import Orientation
from pycrystallography.core.millerDirection import MillerDirection
from pycrystallography.core.millerPlane import MillerPlane
import pycrystallography.utilities.graphicUtilities as gu
import pycrystallography.utilities.pymathutilityfunctions as pm
from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
import warnings

import cv2
import imutils
from skimage import measure
from imutils import contours
from sympy.geometry import *
import matplotlib.pyplot as plt
import copy
#from copy import  deepcopy
import sympy.geometry as gm
from operator import itemgetter
import time
from scipy import optimize
import pymatgen as mg
import os
import json
from sympy.codegen.cnodes import static
from pycrystallography.core import orientation

__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"


options = {"GAUSSIAN_BLUR_SIZE": 5,
           "THRESHOLD_LOW":140,
           "THRESHOLD_HIGH":250,
           "NUMBER_OF_SPOTS_TO_DETECT":3,
           "MIN_ANGLE_OF_DETECTED_SPOTS": 10, # This is the min angle in deg between two detected spots (planes) 
                                              # which need to exceed for the 
                                              # two spots to be considered as measure data
           
           "HKL_MAX":8}




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


with open(os.path.join(os.path.dirname(__file__),
                       "atomic_scattering_params.json")) as f:
    ATOMIC_SCATTERING_PARAMS = json.load(f)


class SaedAnalyzer(MSONable,object):
    '''
    classdocs
    '''


    def __init__(self, name='',symbol=r"$\alpha$", lattice=None,hklMax=4, atomData=None,
                 considerDoubleDiffraction=True,machineConditions=None):
        '''
        Constructor
        '''
        self._lattice = lattice
        self._hklMax=hklMax

        if atomData is not None:
            self._atomData = atomData
        else:
            self._atomData = [(1, np.array([0.,0.,0.]))]
        
        if not lattice is None:
            self._millerPlaneSet = MillerPlane.generatePlaneList(hklMax,lattice,includeSymEquals=True)
            self._millerPlaneSymUniqueSet =MillerPlane.generatePlaneList(hklMax,lattice,includeSymEquals=False) 
            self._millerPlaneFullSet=copy.copy(self._millerPlaneSet)
        if  considerDoubleDiffraction :
            self._considerDoubleDiffraction=True
            print("considerDoubleDiffraction")
            
        else:
            print("Here")
            self._considerDoubleDiffraction=False
            self._removeSystematicAbsences()
        
        self._lookUpTable = None
        if hklMax is not None :
            hklMax = hklMax
        else:
            hklMax = options["HKL_MAX"]
        self._expSpotData= None
        self._expImageData = None
        self.expPatternCalibration=None
        self._lookUpTable = None

        
        self._minNumberOfSpotsToMatch=5 ### This is the number of Soultion spots that must be matched with Exp pattern for it to be valid solutuon
        
        if machineConditions is None:
            self._machineConditions={"Voltage":200e3, ##unitsw Volts
                                     "AlphaTilt":0., ## Degrees
                                     "BetaTilt":0.,
                                     "InPlaneRotaton":0.,
                                     "CameraLength":1e3, ##in mm
                                     }
        else:
            self._machineConditions=machineConditions
        self._waveLength=None
        self._waveLength=self.getWaveLength()
        self._cameraConstant = self._waveLength*self._machineConditions["CameraLength"] 
        self._name = name
        self._symbol = symbol
        self._planeTable=None 
        
    @property
    def lattice(self):
        return self._lattice
     
    def getWaveLength(self):
        """
        returns wave lentgh in Angstrom 
        """
        if self._waveLength is None:
            m = 9.109e-31 ##kg
            h = 6.626e-34 ## joules
            e = 1.6e-19 ## coulumbs
            _lambda = 1e10*h/np.sqrt(2*m*e*self._machineConditions["Voltage"])
            print("lambda = ",_lambda,"  Angstrom")
            self._waveLength = _lambda
        return   self._waveLength   
    
    def extractSpotsFromExpPattern(self, imageFileName=None,displayImage=True,minSpotSize=200,
                                   indicateLineLengthInPlot=True,showSpotLines=True,
                                   beamStopper=False):
        """
        function to build the exp data. 
        
        Input:
        -----
            imageFileName : path name of the Image file of the SAED pattern
        Returns
        -------
        expSpotData : A dicitonary with feilds
                        "TransmittedSpot" np array of size (2,) indicating the XY coordinates of the Transmiited spot in Image frame
                        "DiffractionSpots" np array of size(n,2) where n is the number of spots detected, XY cordiantes of the spots
            Dpending on if it hexagonal or not.
        """
#         if imageFileName is not None:            
#             expSpotData = gu.extractSpotDatafromSAEDpattern(imageFileName)
#         else:
#             raise ValueError ("A valid image path must be supplied to this fucntion for reading the image from disk")
#         
            
        image = cv2.imread(imageFileName,cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("The image could not be succesfully read. Possibly the image path is wrong or its wrong format image")
            
        blurred = cv2.GaussianBlur(image, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
        blurred = cv2.GaussianBlur(blurred, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
        thresh = cv2.threshold(blurred, options["THRESHOLD_LOW"], options["THRESHOLD_HIGH"], cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        self._expImageData=image
        plt.imshow(thresh)
        plt.gray()
        plt.show()
        # loop over the unique components
        sizes=[]
        uniquelabels = np.unique(labels)
        print(uniquelabels.shape, type(uniquelabels))
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue 
        
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            #print(numPixels)
            
            if numPixels > minSpotSize:
                mask = cv2.add(mask, labelMask)
                sizes.append(numPixels)
                #print("processig, ",numPixels)
                 
                
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = contours.sort_contours(cnts)[0]
        imageCentre = np.asarray([int(image.shape[1]/2), int(image.shape[0]/2)])
        print("imageCentre = ", imageCentre)
        #print("Number of spots detected is :",len(cnts))
          

        
        if (len(cnts)<2) :
            print("Number of spots detected is less than 2 May be image quality is not good or some options have to be tweaked!!!")
            return None
        
        
        
        spotPositions = np.zeros((len(cnts),2),dtype = np.int16)
        spotSizes = spotPositions.copy()[:,0]
        spotVectors = np.zeros((len(cnts)-1,2), dtype = np.float64 )
        distFromCentre = np.zeros((len(cnts),1),dtype = np.float64)
        #print("ctns=",cnts)
        for (i, c) in enumerate(cnts):
            # draw the bright spot on the image
            (x, y, w, h) = cv2.boundingRect(c)    
            M = cv2.moments(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(mask, (int(cX), int(cY)), int(radius),
                (255), 3)
            
            cXc, cYc = cX,cY
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("the centres", cXc, cYc, cX,cY,x+w/2,y+h/2)
            spotPositions[i,:]=np.array([cX,cY])
            distFromCentre[i] = np.linalg.norm(spotPositions[i]-imageCentre)
            spotSizes[i] = sizes[i]  
        
        longestSpotFromCentre = np.max(distFromCentre)
        shortestSpotFromCentre = np.min(distFromCentre)
        lagestSpotInd = np.argmax(spotSizes)
        
        
        indxs = np.argsort(distFromCentre,axis=0).tolist()        
        orderedSpotPositions = np.zeros(spotPositions.shape,dtype=np.int16) 
        orderedSpotSizes =  np.zeros((len(orderedSpotPositions),),dtype=np.float)
        for i in range(0,len(indxs)) :
            orderedSpotPositions[i,:] = spotPositions[indxs[i],:] 
            orderedSpotSizes[i] = spotSizes[indxs[i]]
        spotData={"TransmittedSpot":orderedSpotPositions[0].astype(float),
                  "TransmittedSpotSize":orderedSpotSizes[0].astype(float),
                  "DiffractionSpots":orderedSpotPositions[1:],
                  "SpotSizes":orderedSpotSizes[1:],
                  }
        
        print(spotData["DiffractionSpots"], spotData["SpotSizes"])
        self._expSpotData = spotData
        if displayImage:
            self.displyExpPattern(showSpotLines=showSpotLines,indicateLineLengthInPlot=indicateLineLengthInPlot,
                                  blockStatus=True)
        return spotData
                
    
    def slectSpotsForIndexing(self,spotIds=None,desiredAngle=90,desiredDratio=1.,showSelection=True):
        """
        A utility function for chosing the suitable spots for indexing.
        
        Input:
        -----
            spotIds : if None, autoamtically the best spots based on certain criteria are chosen
                      if specified as a list or array like, those spots onlshall be used for Indexing
        
            desiredAngle: this is the desired angle between the chosen spots. Of all the availble exp spots
                            the ones closest to this value are chosen.
            desireDratio: this is the desired dRatio between the chosen spots. Of all the availble exp spots
                            the ones closest to this value are chosen.Note that desiredAngle takes precedence always
        
        
        Returns:
        -------
            Dictionary of data of spots needed for Indexing along with the d_ratios,
            angle between them in the form a a dictionary.
            Example: output
            {'SpotsIds': [5, 13], 'Lines': [Segment(Point2D(0, 0), Point2D(125, 79)), Segment(Point2D(-248, 79), Point2D(0, 0))], 'd_ratio': 1.76, 'Angle': 130.0}
            in case of failure to find the spots matching the given criterion,
            "None" is returned
        
      
       
        """
        if self._expSpotData is None :
            raise ValueError ("Exp data is not yet availble fitst build it and then use this function")
        else:           
        
            if "SpotPairData" in self._expSpotData : ### already the data is built just we ned to chose the ones that satisfy our requirement
                print("Skipping the building the spotPair data as it is already bulit previously !!!")
            
            else:
                self._buildSpotPairData()
     
            result = None
            if spotIds is not None: ### case of specidfic spotIds being mandated by the calling function
                for i,pairData in enumerate(self._expSpotData["SpotPairData"]):
                    #print(pairData["SpotsIds"])
                    if spotIds == pairData["SpotsIds"] :
                        if(pairData["Angle"]<5. or pairData["Angle"]>175.):
                            print ("The chosen spots are almost paralle hence not suitable. The angle between them is ", pairData["Angle"])
                            return None 
                        result= pairData
                if result is None:
                    print("The asked spotIds could not be located Please cross check")
                    return None
                    
            else : ## we need to figure out the best spots ourselves based on the criteria
                pairData = self._expSpotData["SpotPairData"]
                sortedPairData = sorted(pairData , key=lambda elem: "%f %f" % 
                                (np.abs(desiredAngle-elem['Angle']), np.abs(desiredDratio- elem['d_ratio'])))
                result = sortedPairData[0]
            if showSelection :
                #print(pairData)                            
                p1= pairData["SpotAbsXY"][0]
                p2 =pairData["SpotAbsXY"][1] 
                plt.scatter([p1[0],p2[0]], [p1[1],p2[1]], s=120, facecolors='none', edgecolors='w')
                self.displyExpPattern()
            return result
    
    

    def _buildSpotPairData(self):
        """
        An internal method to  build the master table of the detected spots, their 
        d_ratios, angles etc so that spot selection method can focus only on selction from this list.
        
        """
        spotPositions = self._expSpotData["DiffractionSpots"]
        spotVectors = np.zeros(spotPositions.shape)
        transmittedSpot = self._expSpotData["TransmittedSpot"]
        lineData=[]
        for i,spot in enumerate(spotPositions) :
            spotVectors[i,:] = spot-transmittedSpot ## always the spotPositions[indxs[0]] is the central spot
            lineData.append({"SpotIds":i+1,
                             "Lines":Segment(Point(0,0),Point(spotVectors[i,0],spotVectors[i,1])),
                             "SpotAbsXY":spot})
        pairData=[]
        for pair in itertools.combinations(lineData,2):                   
            #print("The pair", pair, pair[0].length, pair[1].length)
                line1 = pair[0]["Lines"]
                line2 = pair[1]["Lines"]
                
                spot1 = pair[0]["SpotIds"]
                spot2 = pair[1]["SpotIds"]
                spot1XY = pair[0]["SpotAbsXY"]
                spot2XY = pair[1]["SpotAbsXY"]
                
                x1 = float(line1.p1[0])
                y1 = float(line1.p1[1])
                if (np.allclose([x1,y1],[0.,0.])) :
                    x1 = float(line1.p2[0])
                    y1 = float(line1.p2[1])                            
                x2 = float(line2.p1[0])
                y2 = float(line2.p1[1])                        
                if (np.allclose([x2,y2],[0.,0.])) :
                    x2 = float(line2.p2[0])
                    y2 = float(line2.p2[1])                       
                
                ang1 = np.abs(np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)*180./np.pi)
                
                #lineIds = pair[0]["LineId"]+"&"+pair[1]["LineId"]                    
                d_ratio = float(line1.length/line2.length)
                #print("d_ratio", d_ratio)
                pairData.append({"SpotsIds":[spot1,spot2],
                                 "Lines":[line1,line2],"d_ratio":1/d_ratio,"Angle":ang1,
                                 "SpotAbsXY":[spot1XY,spot2XY]})
                pairData.append({"SpotsIds":[spot2,spot1],
                                 "Lines":[line2,line1],"d_ratio":d_ratio,"Angle":ang1,
                                 "SpotAbsXY":[spot2XY,spot1XY]})
                ### the second appending is to enusre that both spot pair combinnations such as 1,2 and 2,1 are seperately added
        self._expSpotData["SpotPairData"] = pairData       

    
    def computeKikuchiForSaed(self,saedData,ignoreDistantSpots=False,KIKUCHI_CUT_OFF_FACTOR=1.0,maxHkl=3):
        """
        boundingBox is the region in which the kikuchi lines must be constrined they indicate starting and ending corners of rectangle
        """
        
        spotData = saedData["SpotData"]
        pc = np.array(saedData["patterCenter"])
        spotBounds=saedData['patternSpotBounds']
        if spotBounds is not None:
            maxLengthOfline = min(abs(0.5*(spotBounds[0]-spotBounds[1])), abs(0.5*(spotBounds[2]-spotBounds[3])))
        else:
            spotBounds = 1.0 #### ust some value to escape kikuchi calcualtion
        kikuchiData=[]
        print("Entered kikuchi")
        for item in spotData:
            #print("Entered kikuchi loop")
            
            if item['Intensity']>1e-3:
                if item["Plane"].maxHkl<=maxHkl:
                    spotPoint=np.array(item["XY"])
                    pointDistance=np.linalg.norm(spotPoint)
                    distFraction=pointDistance/maxLengthOfline
                    linePassesThrough= 0.5*(pc+spotPoint)
                    reciprocalVec= (linePassesThrough-pc)
                    reqLineVec= np.array([-reciprocalVec[1], reciprocalVec[0]])
                    reqLineVec = reqLineVec/np.linalg.norm(reqLineVec)
                    point1 = linePassesThrough+(reqLineVec*maxLengthOfline)
                    point2 = linePassesThrough-(reqLineVec*maxLengthOfline)
                    if distFraction>KIKUCHI_CUT_OFF_FACTOR and ignoreDistantSpots:
                        #print("Igonring current kikuch line as the spot is far away from centre !!!!")
                        continue
                    #print("foud the kikuchi line")
                    kikuchiData.append({"kikuchiPlane":item["Plane"], "pointInLine":linePassesThrough, "point1":point1,"point2":point2,"lineVector":reqLineVec,
                                            "perpVector":reciprocalVec})
                    
               
                
                
        print(" {:2d} number of kikuchis are calcualted !!!".format(len(kikuchiData)))
        if len(kikuchiData)<4:
            warnings.warn("Too few kikuchis : only {:2d} number of kikuchis are calcualted !!!".format(len(kikuchiData)))
        saedData["kikuchiData"]= kikuchiData
        return  saedData      
    
    
    
    
    def calcualteSAEDpatternForZoneAxis(self, zoneAxis=None,atomData=None,desiredPatternxAxis=None, inPlaneRotation=0,
                         patterCenter=[0.,0.],scalingFactor=1.,SAED_ANGLE_TOLERANCE=2.,expPatternCalibration=None,
                                                  patternBounds=None,patternInfoText='',holderTiltData=None, primaryThresholdIntensity=1e-2):
        """
        Utility fuinction for calcualting the SAED pattern given the planeList and zoneAxis.
        patternBounds can be specified as a list of coordiantes (definign a  rectangle) or as a single value specifying the radius of the 
        circle within which pattern spots must lie
        
        holderTiltData is a dictionary indicating the alphaTilt betaTilt and diffractionRotation (usually fixed for specific cmaeraLength) for estimating the 
        crystal Orientation.
        example :
        holderTiltData={"alphaTilt":22.,"betaTilt":-18.,"diffractionRotationAngle":12., 
                        "options":{"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":False}}
         
        """
        result = None
        checkRadius=False
        
        if patternBounds is  None:
            patternBounds = [-1e10,1e10,-1e10,1e10]
        
        if not isinstance(patternBounds,list) :## case of specifying the maximum distance in reciprocal space
            checkRadius = True
            patternRadius = patternBounds 
        if (isinstance(patternBounds,list) and  len(patternBounds)==1) :
            checkRadius = True
            patternRadius = patternBounds[0]

        
        if expPatternCalibration is None:
            if desiredPatternxAxis is None and inPlaneRotation <1e-2:
                patternXAxis, patternYAxis,patternZAxis, ori = self.makeSAEDRefFrame(zoneAxis)
            if abs(inPlaneRotation)>1e-2 and desiredPatternxAxis is None:
                patternXAxis, patternYAxis,patternZAxis, ori = self.makeSAEDRefFrame(zoneAxis,inPlaneRotation=inPlaneRotation)
            else:
                patternXAxis, patternYAxis,patternZAxis, ori = self.makeSAEDRefFrame(zoneAxis,desiredPatternxAxis=desiredPatternxAxis)
                print("Yes using the desired Pattern X axis : and is {:int}!!!!".format(patternXAxis))
                        
       
        else:
            patterCenter  =expPatternCalibration["patterCenter"]
            scalingFactor =expPatternCalibration["patternScalingFactor"]
            patternXAxis = expPatternCalibration["patternXAxis"]
            patternYAxis = expPatternCalibration["patternYAxis"]
            patternZAxis = expPatternCalibration["patternZAxis"]
        
        if  zoneAxis is None:
            zoneAxis = MillerDirection(lattice=self._lattice, vector=[0.,0.,1.])
        if atomData is None :
            atomData = self._atomData 
            
        planeList = self._millerPlaneSet
        
        saedData=[]
        spot1=[]
        spot2=[]
        #print("patterCenter", patterCenter)
        if abs((zoneAxis.angle(patternXAxis)-np.pi/2))<1e-5:
            xAxis = patternXAxis.getUnitVector()  
            yAxis = patternYAxis.getUnitVector()
            zAxis = patternZAxis.getUnitVector()
            xyData=[] 
            count=0
            for plane in planeList: 
                ang = np.arccos(np.clip(np.dot(plane.getUnitVector(),zAxis),-1,1))*180/np.pi
                if np.abs(90.-ang)<SAED_ANGLE_TOLERANCE :
                    isPrimarySpot=False
                    intensity = plane.diffractionIntensity(atomData=atomData) 
                    p = plane.getCartesianVec() 
                    x = np.dot(p,xAxis)*scalingFactor+patterCenter[0]
                    y = np.dot(p,yAxis)*scalingFactor+patterCenter[1]
                    xyData.append([x,y])
                    if  count==0 and intensity >primaryThresholdIntensity:
                        count=count+1
                        spot1 = {"Plane":plane,"XY":[x,y],"Intensity":intensity, }
                        isPrimarySpot= True
                       
                    if count==1 and intensity >primaryThresholdIntensity:
                        spotAngleWithPrimarySpot = plane.angle(spot1["Plane"], units="deg", considerSymmetry=False)
                        if spotAngleWithPrimarySpot<130.0 and spotAngleWithPrimarySpot>10:
                            spot2= {"Plane":plane,"XY":[x,y],"Intensity":intensity, "isPrimarySpot":True}
                            count=count+1
                            isPrimarySpot=True

                    
                    if checkRadius: ## checking to see if the spot falls within the circle defined by pc and patternBound as radius
                        dist = np.sqrt((x-patterCenter[0])*(x-patterCenter[0])+(y-patterCenter[1])*(y-patterCenter[1]))
                        if dist <patternRadius:
                            saedData.append({"Plane":plane,"XY":[x,y],"Intensity":intensity,"isPrimarySpot":isPrimarySpot})
                    else: ### checking by rectangular bounds
                        if x>=patternBounds[0] and x<=patternBounds[1] and y>=patternBounds[2] and y<=patternBounds[3] :
                            saedData.append({"Plane":plane,"XY":[x,y],"Intensity":intensity,"isPrimarySpot":isPrimarySpot})
            
            xyData=np.array(xyData)
            if xyData.size>=1:
                patternSpotBounds = [xyData[:,0].min(), xyData[:,0].max(), xyData[:,1].min(), xyData[:,1].max()]
            else:
                patternSpotBounds=None
            
            if holderTiltData is None:
                holderTiltData={"alphaTilt":0.,"betaTilt":0.,"diffractionRotationAngle":0., "options":{}}
                
            ori1,ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis,patternXAxis,alphaTilt=holderTiltData["alphaTilt"],
                                                           betaTilt=holderTiltData["betaTilt"],diffractionRotationAngle=holderTiltData["diffractionRotationAngle"],options={})
            
            zoneAxisInt,err = zoneAxis.integerize()
            patternXaxisInt,err = patternXAxis.integerize()
            patternYaxisInt,err = patternYAxis.integerize()
            patternInfoText+=r"phase = {:s}".format(self._symbol)
            patternInfoText += "\n"
            patternInfoText+=r"zoneAxis = {:int}".format(zoneAxis)
            patternInfoText+="\n"
            patternInfoText+=r"X-Axis = "+str(patternXaxisInt)
            patternInfoText+="\n"
            patternInfoText+=r"Y-Axis = "+str(patternYaxisInt)
            patternInfoText+="\n"
            patternInfoText+=r"ScalingFact = "+str(np.around(scalingFactor,1))
            patternInfoText+="\n"
            patternInfoText+=r"Ori1 = "+str(np.around(ori1.getEulerAngles(units='degrees'),1))
            patternInfoText+="\n"
            patternInfoText+=r"Ori2 = "+str(np.around(ori2.getEulerAngles(units='degrees'),1))
            patternInfoText+="\n"

            patternInfoTextDict = { 'zoneAxis': "{:int}".format(zoneAxis),
                                   'X-Axis': str(patternXaxisInt), 'Y-Axis': str(patternYaxisInt),
                                   'ScalingFact': str(np.around(scalingFactor,1)),
                                   'Ori1': str(np.around(ori1.getEulerAngles(units='degrees'), 1)),
                                   'Ori2': str(np.around(ori2.getEulerAngles(units='degrees'), 1))
                                   }
            
            saedDF = pd.json_normalize(saedData)

            result = {"SpotData":saedData,"zoneAxis":zoneAxis, "spot1":spot1, "spot2":spot2, "xAxis":patternXAxis,"yAxis":patternYAxis,
                      "patterCenter": patterCenter,"scalingFactor":scalingFactor,
                      "patternInfoText":patternInfoText,'patternSpotBounds':patternSpotBounds, "Ori1":ori1,"Ori2":ori2,
                      "SaedLabel":self._name, "SaedSymbol":self._symbol, "saedDF":saedDF, 'patternInfoTextDict':patternInfoTextDict}

            return result
        else:
            print("zoneAxis = ", zoneAxis,"and patternXAxis = " , patternXAxis)
            raise ValueError ("The supplied X axis is not  perpendicualr to Zone Axis !!!!")
      


    def calcualteCalibration(self, zoneAxis,spotXY,spot1Plane, pc=[0,0],alphaTilt=0.,betaTilt=0.,inPlaneRotation=0.,
                             mmForPixcel=0.00329*0.9010582473595571,cameraLength="40cm",checkForKnownCalibration=False):
        """
        Given the two spots from Image interms of their XY coordinates 
        and the respective MillerPlanes, 
        finds the scaling and rotation of the SAED pattern to be applied to 
        so that pattern can begenrated to match with the experimental pattern
        
        """       
        if isinstance(spot1Plane,MillerPlane):
            spotList = [spot1Plane]
        

        calibParameters = np.zeros_like(spotXY) #### first column is for rot angle 2nd colum is for scaling factors for each spot
        
        if isinstance(spot1Plane,list):
            spotList = spot1Plane
            print(spotList)
            if isinstance(spotXY, list):
                spotXY = np.array(spotXY)
                
            
            assert len(spotList)==spotXY.shape[0], "Probelm number of XY points not matching with the numbber of spots"
            for i, spot in enumerate(spotList):
                    calib = self.__calibrateForSingleSpot(zoneAxis=zoneAxis,spotXY=spotXY[i],spot1Plane=spot, pc=pc)
                    calibParameters[i,0] = calib["patternRotationAngle"]
                    calibParameters[i,1] = calib["patternScalingFactor"]
        
            print("Here is the Calibs", calibParameters)
            
            patternRotationAngle = np.mean(calibParameters[:,0])
            patternScalingFactor = np.mean(calibParameters[:,1])
            devaitions = 100*np.abs(calibParameters-np.array([patternRotationAngle,patternScalingFactor]))/np.array([patternRotationAngle,patternScalingFactor])
            
            spot1Plane=spotList[0]
            spotXY = spotXY[0] #### making just one spaot for rest of the calcualtion
            print("Means = ",patternRotationAngle, patternScalingFactor)
            if(np.all(np.abs(calibParameters[:,0]-patternRotationAngle)>5)):
                
                print("Severe Problem in rotations !!!!!!!"+ str(calibParameters[:,0]) )
                patternRotationAngle= calibParameters[0][0]
        
            #assert np.all(devaitions[:,1]<10) , "Problem in calibration"+ str(devaitions)                                                                                                   
            #a = input("press continue") 
            if  np.any(devaitions[:,1]>1):
                print("Sevfere Warning !!!! Sclaing factors arlso not good and hecne ignoring all other than first one!!!)")
                
                print(calibParameters)
                patternScalingFactor =calibParameters[0][1]
                #a = input("press continue")                                                                                                
        
        else:
            calib = self.__calibrateForSingleSpot(zoneAxis=zoneAxis,spotXY=spotXY,spot1Plane=spot1Plane, pc=pc)
                    
            patternRotationAngle = calib["patternRotationAngle"]
            patternScalingFactor = calib["patternScalingFactor"]
        
        patternXAxis, patternYAxis,patternZAxis = calib["patternXAxis"],calib["patternYAxis"],calib["patternZAxis"]
        x = np.dot(spot1Plane.getCartesianVec(),patternXAxis.getUnitVector())
        y = np.dot(spot1Plane.getCartesianVec(),patternYAxis.getUnitVector()) 
#  
        x1 = pc[0]
        y1 = pc[1]
        x2 = float(spotXY[0])
        y2 = float(spotXY[1])
        targetLine = np.array([[x1,y1],[x2,y2]])
        l1 = pmt.lineLength(targetLine)
        measuredLengthinMM = l1*mmForPixcel
        if cameraLength is None:
            measuredDhkl = l1
        else:
            measuredDhkl =cameraLengthCalibration[cameraLength]/measuredLengthinMM
        
        actualDspacing = spot1Plane.dspacing  
        obtainedCalibRatio=None
        if checkForKnownCalibration: ### if the calibration is known 
            print("mulfactor = ", measuredDhkl/actualDspacing,"mmForPixcel=",mmForPixcel)
            errorIndHkl  = (measuredDhkl-actualDspacing)/actualDspacing*100
            obtainedCalibRatio = patternScalingFactor/cameraLengthCalibration[cameraLength]
            IdealCalibRatio=330
            errorIncalibration = 100*abs(obtainedCalibRatio-IdealCalibRatio)/IdealCalibRatio
            print("plane = {:2d} measuredDhkl =  {:.2f} actualDspacing = {:.2f} Error ={:2f} = calibRatio={:.2f}  SF = {:.2f}".format(spot1Plane,  measuredDhkl,actualDspacing, errorIncalibration, obtainedCalibRatio, patternScalingFactor))
               
            if errorIncalibration>20:
                print("Error exceeded hence ignoring Sorry for that :: plane = {:2d} measuredDhkl =  {:.2f} actualDspacing = {:.2f} Error ={:2f} = calibRatio={:.2f}  SF = {:.2f}".format(spot1Plane,  measuredDhkl,actualDspacing, errorIncalibration, obtainedCalibRatio, patternScalingFactor))
                print("Currently camera length being used is ", cameraLength, "Check if that is correct")

                return None

        
        crystalOrientation = Orientation(matrix=np.array([patternXAxis.getUnitVector(),
                                       patternYAxis.getUnitVector(),
                                       patternZAxis.getUnitVector()]).T)
        crystalOrientation2 = Orientation(matrix=np.array([-1*patternXAxis.getUnitVector(),
                                       -1*patternYAxis.getUnitVector(),
                                       patternZAxis.getUnitVector()]).T)
        
        crystalOrientation = self.compnsateCrystalOriForSampleTilts(crystalOrientation,alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)
        crystalOrientation2 = self.compnsateCrystalOriForSampleTilts(crystalOrientation2,alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)
       
        
        
        result = {"patternRotationAngle":patternRotationAngle,"patternScalingFactor":patternScalingFactor,
                  "patternXAxis":patternXAxis,"patternYAxis":patternYAxis,
                  "patternZAxis":patternZAxis, "patterCenter":pc[0:2],
                  "crystalOrientation":crystalOrientation,
                  "crystalOrientation2":crystalOrientation2,
                  "obtainedCalibRatio":obtainedCalibRatio}

          
        return result    
       

    def solveSetOfSpots(self, spotIds=None,dTol=10,angleTol=2,maxHkl=10,onePixcel=0.00847,cameraLength=None):
        """
        Given a set of two experimental spots, finds the most likely solution of their Miller Indices
        dTol Allowed Percentage of devaition in D ratio 
        angleTol Allowed deviation in Angle in degree (not percentage)
        maxHkl is the value of the maxHkl od planes to be tried for solving. this is useful to eliminate unncesary trials with 
        set of parllel refelctions (because their d_ratio and angles will match,)
        to be more clear suppose correct solution of spot1 & spot2 is [100] and [011]
        also the [200] and [022] would satisfy this 
        by putting the max hkl as 1 we can eliminate the second set from consideration
        
        onePixcel in mm default is 0.0847
        """
        if self._lookUpTable is None:
            self.buildLookUpTable()
        
        
        if spotIds is None:
            measureData = self.slectSpotsForIndexing(spotIds=None,desiredAngle=90,desiredDratio=1.)
        else:
            measureData = self.slectSpotsForIndexing(spotIds)
        
        expD_ratio = measureData["d_ratio"]
        expAngle = measureData["Angle"]
        print("The exp data being tried is ",expD_ratio, expAngle )
        lowLimit = expD_ratio-expD_ratio*dTol/100.
        upLimit =  expD_ratio+expD_ratio*dTol/100. 
        lookUpTable = self._lookUpTable
        sol=[]
        count=0
        for i,item in enumerate(lookUpTable):            
            if item["dRatio"]>lowLimit and item["dRatio"]<=upLimit :
                    plane1 = item["spot1"]
                    plane2 = item["spot2"]
                    if np.all(plane1.gethkl()<maxHkl) and np.all(plane2.gethkl()<maxHkl) :
                        plane2Sym = plane2.symmetricSet()
                       
                        for symPlane in plane2Sym:  
                            ang = plane1.angle(symPlane,units="Deg") 
                            angLowLimt = expAngle - angleTol
                            angUpLimt =  expAngle + angleTol
                             
                            if (ang>=angLowLimt) and (ang<=angUpLimt):                          
                                zoneAxis = MillerPlane.getZoneAxis(plane1, symPlane)
                                angError = abs(ang-expAngle)
                                dratio = plane1.dspacing/symPlane.dspacing
                                #dratio = max(dratio,1/dratio)  
                                dError =100.*(abs(dratio-expD_ratio)/expD_ratio)     
                                count+=1
                                tmp = {"solId":count,"zoneAxis":zoneAxis, 
                                       "spot1":plane1,"spot1XY":measureData["SpotAbsXY"][0],
                                       "spot2":symPlane,"spot2XY":measureData["SpotAbsXY"][1],
                                            "dRatio":dratio, "dError":dError,
                                            "Angle":ang, "AngleError":angError,}
                                noOfMatchingSpots, CI, error, saedData,calibration= self.findCorrelationWithExptPattern(tmp,cameraLength=cameraLength)
                                if noOfMatchingSpots>self._minNumberOfSpotsToMatch:
                                    tmp["CorrelationError"]=error
                                    tmp["saedData"] = saedData
                                    tmp["Calibration"] = calibration
                                    tmp["noOfMatchingSpots"] = noOfMatchingSpots
                                    tmp["CI"] = CI                                
                                    sol.append(tmp)              
            

        
        if len(sol) ==0:
            print("Problem no solutiuon found !!!!")
            return None
        
        set11 = sorted(sol , key=lambda elem:  (elem["CI"], elem ["noOfMatchingSpots"], elem['CorrelationError']), reverse=True)
        sol = set11
        
        ### Now remving the duplicate solutiuons
        solArray=[]
        for i, item in enumerate(sol):
            solArray.append([item["CI"],item["noOfMatchingSpots"],item["CorrelationError"],item["Calibration"]["patternScalingFactor"]])
        
        if len(sol)>1:
            solArray = np.round(np.array(solArray),4)
            tmp, Ind = np.unique(solArray,return_index=True,axis=0) 
            tmpSol=[]
            for i in np.flip(Ind,axis=0) :
                tmpSol.append(sol[i])
            sol = tmpSol
        
        self._solSet = sol
        self._bestSolution = sol[0]
        return sol
    
    def buildLookUpTable(self):
        """
        Builds the master table of the planes and angles from which the solution is selcted for the spot pattern
        """
        if self._lookUpTable is None:
            planeList = self._millerPlaneSymUniqueSet
            lookUpTable=[]
            tmp = []
            print("Started Building the lookUp table ")
            for i,comb in enumerate(itertools.combinations_with_replacement(planeList,2)):                    
                    plane1, plane2, = comb[0], comb[1]
                    d_ratio = plane1.dspacing/plane2.dspacing
                    #print("{:d}{:d} {:.2f}".format(plane1,plane2,max(d_ratio,1/d_ratio)))
                    #d_ratio = d_ratio,1/d_ratio
                                                       
                    tmp.append({"spot1":plane1,"spot2":plane2,"dRatio":d_ratio}) 
#                    
                    tmp.append({"spot1":plane2,"spot2":plane1,"dRatio":1/d_ratio})                                   
#                
            self._dRatioAndAngleLookUpArray = np.array([i["dRatio"] for i in tmp])
            self._lookUpTable = tmp
            print("Bulding the lookUp table complete!!!!")
            return lookUpTable
              
            
        else:
            return self._lookUpTable
        
    
    
    
    def makeSAEDRefFrame(self, zoneAxis =None, desiredPatternxAxis=None,inPlaneRotation=None):
        """
        Helper funciton to genrate suitable reference framce i.e. XYZ axies of the 
        SAED patterns given the zone Axis
        """
        
        if  zoneAxis is None:
            zoneAxis = MillerDirection(lattice=self._lattice, vector=[0.,0.,1.])
        
        if  (desiredPatternxAxis is not None) and (not isinstance(desiredPatternxAxis,MillerDirection)):
            print("desiredPatternxAxis", type(desiredPatternxAxis))
            desiredPatternxAxis = MillerDirection(vector=desiredPatternxAxis,isCartesian=True,lattice=self._lattice)
         
        patternXAxis,patternYAxis,patternZAxis = self._getIdealRefFrame()

        angleWithZoneAxis = zoneAxis.angle(patternZAxis,units="Deg",considerSymmetry=False)
        if angleWithZoneAxis>1e-3 :
            #print(zoneAxis.getUnitVector(),patternZAxis.getUnitVector(),zoneAxis.getMag(), patternZAxis.getMag())
            initialOri = Orientation.mapVector(patternZAxis.getUnitVector(),zoneAxis.getUnitVector())
            patternXAxis.rotate(initialOri)
            patternYAxis.rotate(initialOri)
            patternZAxis.rotate(initialOri)
        
        crossCheckAngle = patternZAxis.angle(zoneAxis,units="Deg",considerSymmetry=False)
        assert crossCheckAngle<1e-3, "desired zone axis was not achieved!!!!"
            
        if abs(180.-angleWithZoneAxis)<1e-3:
            patternXAxis = MillerDirection(vector=[1.,0.,0.],isCartesian=True,lattice=self._lattice)
            patternYAxis = MillerDirection(vector=[0.,-1.,0.],isCartesian=True,lattice=self._lattice)
            patternZAxis = MillerDirection(vector=[0.,0.,-1.],isCartesian=True,lattice=self._lattice)
            print("Got this case as the ZoneAxis is ",zoneAxis )

        tmp1patternXAxis = copy.deepcopy(patternXAxis)
        tmp2patternXAxis = copy.deepcopy(patternXAxis)
        
        if desiredPatternxAxis is not None or inPlaneRotation is not None: ### if any of the two is not none then do the following processing 
        
            if desiredPatternxAxis is None:
                desiredPatternxAxis = copy.deepcopy(patternXAxis)           
                
            #### now we align the obained X axis of patern to the desired one supplied by user i.e. 
            #print("The angle is ", zoneAxis.angle(desiredPatternxAxis,units="Deg"))
            if abs((zoneAxis.angle(desiredPatternxAxis,considerSymmetry=False)-np.pi/2))<1.*np.pi/180:
                if inPlaneRotation is not None: ### if angle of rotation is specified then use it 
                    inPlaneRotationAngle = inPlaneRotation
                    inPlaneRotation = Orientation(axis=patternZAxis.getUnitVector(),degrees=inPlaneRotationAngle)
                    patternXAxis.rotate(inPlaneRotation.inverse)
                    patternYAxis.rotate(inPlaneRotation.inverse)
                
                else: ### case of specifying the desired X vector
                    inPlaneRotationAngle = patternXAxis.angle(desiredPatternxAxis,units="Deg") ### in case Xaxis is pecified use that to compute the require rotation
                #print("inPlaneRotationAngle = ", inPlaneRotationAngle)
                    inPlaneRotation1 = Orientation(axis=patternZAxis.getUnitVector(),degrees=inPlaneRotationAngle)
                    inPlaneRotation2 = Orientation(axis=patternZAxis.getUnitVector(),degrees=-inPlaneRotationAngle)
                    
                    ##### it should be checked if the inPlaneRotationAngle is positive or negatice
                    ###### in case  the calibrated pattern does not match with the exp pattern this must be checked
                    
                    tmp1patternXAxis.rotate(inPlaneRotation1)
                    tmp2patternXAxis.rotate(inPlaneRotation2)
                    tolerance = 0.5
                    
                    ### cross checking of the operation is succesful
                    achievedAngle1 = abs(tmp1patternXAxis.angle(desiredPatternxAxis,considerSymmetry=False, units="Deg"))
                    if achievedAngle1<tolerance :
                        #print("First one")
                        patternXAxis.rotate(inPlaneRotation1)
                        patternYAxis.rotate(inPlaneRotation1)
                    else:
                        achievedAngle2 = abs(tmp2patternXAxis.angle(desiredPatternxAxis,considerSymmetry=False,units="Deg"))
                        if achievedAngle2<tolerance :
                            #print("Second  one")
                            patternXAxis.rotate(inPlaneRotation2)
                            patternYAxis.rotate(inPlaneRotation2)
                            
                        else:
                            print("Problem both ways of rotations failed to achieve the target The angle error is : ", zoneAxis.angle(desiredPatternxAxis,considerSymmetry=False,units="deg"),zoneAxis,desiredPatternxAxis ) 
                            raise ValueError ("Probelm in achiving the desired target  X axis")       
        #             , "Probelm in achiving the desired target  X axis. The angle is "+str(achievedAngle)+"X axis = " \
        #             +str(patternXAxis)+"But target was"+str(desiredPatternxAxis)
                
            else:
                print("Warning!! The supplied X vector of the SAED pattern did not lie in the \
                zone of the zoneAxis and hence Igonring!!!! Now I am using the X asis as ", patternXAxis.getUnitVector())
                raise ValueError("patternX vector does not lie in the Zone Axis")
            
        ##### cross checking the orientation
        xVector = patternXAxis.getUnitVector()
        yVector = patternYAxis.getUnitVector()
        zVector = patternZAxis.getUnitVector()
        
        ori = Orientation(matrix=np.array([xVector,yVector,zVector]).T)
        
        assert np.allclose(ori.rotate([1,0,0]), xVector), "Problem in recovering the X axis"+str(ori.rotate(xVector))
        assert np.allclose(ori.rotate([0,1,0]), yVector), "Problem in recovering the X axis"+str(ori.rotate(yVector))
        assert np.allclose(ori.rotate([0,0,1]), zVector), "Problem in recovering the X axis"+str(ori.rotate(zVector))
        
        
        
        return patternXAxis, patternYAxis,patternZAxis, ori

    def _removeSystematicAbsences(self, intTolerance=1e-5):
        """
        Removes the planes from the list which are not allowed by structor factor calcualtions
        """
        
        planeSet = self._millerPlaneSet
        newSet = []
        for i in planeSet:
            if i.diffractionIntensity(atomData=self._atomData)>intTolerance :
                newSet.append(i)
                #print("removing the plane as int is low {}".format(i) )
        
        self._millerPlaneSet = newSet
        planeSet = self._millerPlaneSymUniqueSet
        newSet = []
        for i in planeSet:
            if i.diffractionIntensity(atomData=self._atomData)>intTolerance :
                newSet.append(i)        
        
        self._millerPlaneSymUniqueSet=newSet

    
    def plotSuperImposedExpAndSimulatedPattern(self,simulatedData,expImage,markSpots=True,
                                               ):
        """
        Helper function that overlays the calcualted solution and 
        Original SAED pattern
        """        
        if isinstance(expImage,str):
            imageFileName = expImage
            expImageData = cv2.imread(imageFileName,cv2.IMREAD_GRAYSCALE)
        else:
            expImageData = self._expImageData
            
        self.plotSAED(saedData=simulatedData,plotShow=False)
        plt.imshow(expImageData)
        plt.gray()
        plt.show()
        
    def findCorrelationWithExptPattern(self,sol,cameraLength="40cm"):
        """
        Computes the best solution among the ones obtained
        """
        
        expSpotArray = self._expSpotData["DiffractionSpots"]
        nExpSpots = expSpotArray.shape[0]           
        errorData=[]
        
        zoneAxis = sol["zoneAxis"]
        spotXYList = [sol["spot1XY"],sol["spot2XY"]]
        spots =[sol["spot1"],sol["spot2"]]
        
        pc = self._expSpotData["TransmittedSpot"]
        if isinstance(pc,np.ndarray):
            pc = pc.tolist()
        if len(pc)>2:
            del pc[2]
        assert len(pc) ==2, "issue in pattern ceter"+str(pc)
        longestExpSpot = np.max(np.sqrt(np.sum((expSpotArray-pc)*(expSpotArray-pc),axis=1)))
        shortestExpSpot = np.min(np.sqrt(np.sum((expSpotArray-pc)*(expSpotArray-pc),axis=1)))
        patternBounds =longestExpSpot+0.05*longestExpSpot
        if cameraLength is None:
            checkForKnownCalibration=False
        else:
            checkForKnownCalibration=True
            
        calib = self.calcualteCalibration(zoneAxis =zoneAxis, spotXY=spotXYList,
                                      spot1Plane=spots,pc=pc,cameraLength=cameraLength,checkForKnownCalibration=checkForKnownCalibration)
        if calib is None: ## case of camera length not matching the set scaling factor and hecne ignoring
            return (0, 0,1e5, None, None)
            
        saedData = self.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self._atomData,
                                       expPatternCalibration=calib,patternBounds=patternBounds,SAED_ANGLE_TOLERANCE=1.)
        
        solutionSpotsArray = np.array([i["XY"] for i in saedData["SpotData"]])
        nSolSpots =solutionSpotsArray.shape[0] 
        
        pc = pc[0:2]
        solSpotLengths = np.sqrt(np.sum((solutionSpotsArray-pc)*(solutionSpotsArray-pc),axis=1))
        shortestSolSpot = np.min(solSpotLengths)
        deviationPercentage = abs((shortestSolSpot-shortestExpSpot)/shortestExpSpot)*100
        
        spotCount = float(len(np.where(solSpotLengths<longestExpSpot)[0]))
        
        spotCountRatio = np.round(spotCount*1.0/float(nExpSpots),3)
       
        allowedDeviation = shortestExpSpot*0.2 ### two spots shall be compared if and only if their sepration is less than this value
        count=0
        CI=0.
        if deviationPercentage>20 :
            meanError = 1e10
            #print("The shortest vectros are not matching or too many solution spots hence solutin does not seem to be good at all")
            
            return (count, CI, meanError, saedData,calib)
        if spotCountRatio >=2:
            meanError = 2e10
            #print("too many solution spots hence solutin does not seem to be good at all")
            return (count, CI, meanError, saedData,calib)
        
        if spotCountRatio <0.5:
            meanError = 3e10
            #print("too few solution spots hence solutin does not seem to be good at all")
            return (count, CI, meanError, saedData,calib)
        error=0.
        if nSolSpots>self._minNumberOfSpotsToMatch:
            for i,spot in enumerate(solutionSpotsArray):
                nearestPoint = self.__findNearestSpot(spot, expSpotArray)
                err = np.sqrt(np.sum((spot-nearestPoint)*(spot-nearestPoint)))
                if err < allowedDeviation :               
                    error += err
                    count+=1
            if count>=self._minNumberOfSpotsToMatch :
                meanError = error/count
                CI1 = float(count)/nSolSpots
                CI2 = float(count)/nExpSpots
                CI = min([CI1,CI2])
            else:
                meanError=4e10
                
            return (count, CI,meanError, saedData, calib)
        else:
            meanError = 4e10
            print("Warning if you think this is a good solution then increase the MaxHkl for the initiiation of the Solver")
            print("Note that less than Solution Spos are available for comparision")
            return (count, CI, meanError, saedData, calib)        


    @staticmethod
    def __findNearestSpot(XY, array):
        """
        Helper function that finds the point closest to the point XY in array
        """
               
        d = np.sum((array-XY) *(array-XY),axis=1)
        indx = np.argmin(d)
        return(array[indx])
            
    def correctExpDetectedSpots(self,beamStopper=False,ignoreSpotList=None,
                                spotsForLocatingTransmittedBeam=None):   
        """
        helper method to let igone certain spots and also to adjust the central spot location in case 
        there is a beam stopper
        ignoreSpotList should be list of ids of spots to removed from consideration
        e.g. = [3,4,5,6]
        spotsForLocatingTransmittedBeam should be list of list of two spots each on either side of the 
        central spot (used when the central spot is masked because of beam stopper)
        eg. = [[4,9],[1,7]] here 4,9 and 1,7 make two sets of spots on eithr side of cetral spots.
        """
  
        spotData = self._expSpotData
        diffractionSpotsPositions=spotData["DiffractionSpots"]
        transmittedSpotPosition = spotData["TransmittedSpot"]
        
        if beamStopper:
            if spotsForLocatingTransmittedBeam is None :
                
                    print("Enter the spot IDs to be used for the determnation of the TransmittedSpt location ")
                    data = input("example format = [[1,2],[13,16]] \n in case you are satisfied  type done :")
        #             data = [[4,9],[1,7],[8,10]]
        #             data = [[4,9]]                        
                    print(data, type(data),isinstance(data,str)  )
                    if isinstance(data,list):
                        flatList = [item for sublist in data for item in sublist]
                        flatList = np.array(flatList)
                        transmittedSpotPosition = np.mean(diffractionSpotsPositions[flatList-1],axis=0)
                        plt.scatter(transmittedSpotPosition[0], transmittedSpotPosition[1], s=120, facecolors=None, edgecolors='w')
                        
                        print("here is the data", diffractionSpotsPositions, flatList-1, diffractionSpotsPositions[flatList-1],transmittedSpotPosition)                       
                        print("Now ovewrwriting !!!")                        
                        data='done'
                        plt.show()

            else:
                data = spotsForLocatingTransmittedBeam
                flatList = [item for sublist in data for item in sublist]
                flatList = np.array(flatList)
                transmittedSpotPosition = np.mean(diffractionSpotsPositions[flatList-1],axis=0)
                plt.scatter(transmittedSpotPosition[0], transmittedSpotPosition[1], s=120, facecolors='w', edgecolors='w')
                print("Now ovewrwriting !!!")
                
                data='done'
            self._expSpotData["TransmittedSpot"] = transmittedSpotPosition 
        
        if ignoreSpotList is not None:
            ignoreSpotList = np.array(ignoreSpotList)-1
            self._expSpotData["DiffractionSpots"]=np.delete(diffractionSpotsPositions ,ignoreSpotList,axis=0)
            self._expSpotData["SpotSizes"] = np.delete(self._expSpotData["SpotSizes"],ignoreSpotList)
        
        self.displyExpPattern(showSpotLines=False,blockStatus=True)
             
        
    def displyExpPattern(self, showSpotLines=False, indicateLineLengthInPlot=False,
                         blockStatus=False):
        """
        Helper function to show the exp pattern and also to overaly 
        the detected spots etc
        """
        
        image = self._expImageData
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)         
        ax1.imshow(image)
        
        
        
        spotData = self._expSpotData
        diffractionSpotsPositions=spotData["DiffractionSpots"]
        transmittedSpotPosition = spotData["TransmittedSpot"]
        
        plt.scatter(transmittedSpotPosition[0], transmittedSpotPosition[1], s=120, facecolors='none', edgecolors='w')
        
        for i , spot in enumerate(diffractionSpotsPositions,1) :
            plt.annotate(
                        str(i),xy=(spot[0], spot[1]), xytext=(0, 10),
                        textcoords='offset points', ha='center', va='center',color=[1,1,1],size=16,
                        bbox=dict(facecolor=[0,0,0], edgecolor='white', pad=0.0)
                        )
            x = np.asarray([float(spot[0]),float(transmittedSpotPosition[0])])
            y = np.asarray([float(spot[1]),float(transmittedSpotPosition[1])])
            ax1.scatter(x, y, s=120, facecolors='none', edgecolors='black')

            if showSpotLines :
                point1 = np.array([x[0],y[0]]) 
                point2 = np.array([x[1],y[1]])
                lineLength =np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2)
                if indicateLineLengthInPlot:
                    lineLengthText = '{:d}'.format(int(np.round(lineLength,0)))
                else:
                    lineLengthText = ''
                #print("The length = {:.2f}".format(lineLength))
                gu.broken2DLineWithText(point1,point2,text=lineLengthText,lineFraction=0.5,lc='w')
                    
        plt.gray()
        plt.show(block=blockStatus)
    
    
    def _getIdealRefFrame(self):
        patternXAxis = MillerDirection(vector=[1.,0.,0.],isCartesian=True,lattice=self._lattice)
        patternYAxis = MillerDirection(vector=[0.,1.,0.],isCartesian=True,lattice=self._lattice)
        patternZAxis = MillerDirection(vector=[0.,0.,1.],isCartesian=True,lattice=self._lattice)
        return (patternXAxis,patternYAxis,patternZAxis)

    
    def transformSaedPattern(self,saedData, shift=[0.,0.], scaling=1.0, alphaTilt=0., betaTilt=0.,alphaAxis=[1.,0.,0], betaAxis=[0.,1.,0.], 
                             inPlaneRotation=0.,SAED_ANGLE_TOLERANCE=1.0,options={}):
        """
        gneraric method to apply affine transformations on the saed pattern. applies tilt, in plane rotation, scaling, shifting etc.
        all the rotation angles are in degrees
        """
        transformeddData = copy.deepcopy(saedData)
        pc = (np.array(transformeddData['patterCenter'])+np.array(shift)).tolist()
        scalingFactor=transformeddData['scalingFactor']*scaling
        patternXAxis=transformeddData['xAxis']
        patternYAxis=transformeddData['yAxis']
        patternZAxis=transformeddData['zoneAxis']
        #options={"fixedAxes":fixedAxes,"alphaRotationFirst":alphaRotationFirst, "activeRotation":True}
        patternXAxis = MillerDirection.applyHolderTiltsToDirection(patternXAxis,alphaTilt,betaTilt,alphaAxis,betaAxis,options=options)
        patternYAxis = MillerDirection.applyHolderTiltsToDirection(patternYAxis,alphaTilt,betaTilt,alphaAxis,betaAxis,options=options)
        patternZAxis = MillerDirection.applyHolderTiltsToDirection(patternZAxis,alphaTilt,betaTilt,alphaAxis,betaAxis,options=options)
        
        if abs(inPlaneRotation)>1e-3: #### rotations less than 0.001 are ignored
            inPlane = Orientation(axis= patternZAxis.getUnitVector(),degrees=inPlaneRotation)
            patternXAxis.rotate(inPlane)
            patternYAxis.rotate(inPlane)
        
        assert self._sanityOfPatternAxes(patternXAxis,patternYAxis,patternZAxis), "Problem in saed pattern axes"
        transformeddData= self.calcualteSAEDpatternForZoneAxis(zoneAxis=patternZAxis,desiredPatternxAxis = patternXAxis,
                                             atomData=self._atomData, patterCenter=pc,
                                   scalingFactor=scalingFactor, SAED_ANGLE_TOLERANCE=SAED_ANGLE_TOLERANCE,
                                   )
        
        transformeddData["AlphaTilt"]=alphaTilt
        transformeddData["BetaTilt"]=betaTilt
        transformeddData["InPlaneRotation"]=inPlaneRotation
        patternInfoText = r"$\alpha$= {:.1f} $\beta$= {:.1f} $\theta$= {:.1f} ".format(alphaTilt,betaTilt,inPlaneRotation)
        patternInfoText+="\n"
        transformeddData["patternInfoText"]=patternInfoText+transformeddData["patternInfoText"]
        return transformeddData
        
          
       
    @staticmethod    
    def _sanityOfPatternAxes(patternXAxis,patternYAxis,patternZAxis):
        """
        returns True if the patten axes are orthogonal and XcrossY=Z else returns false
        """
        angleXY = patternXAxis.angle(patternYAxis,considerSymmetry=False,units="deg")
        angleYZ = patternYAxis.angle(patternZAxis,considerSymmetry=False,units="deg")
        angleZX = patternZAxis.angle(patternXAxis,considerSymmetry=False,units="deg")
        
        angleTest = np.allclose([90.0,90.0,90.0],[angleXY,angleYZ,angleZX])
        tmpZ = patternXAxis.cross(patternYAxis)
        if np.allclose(tmpZ.getUnitVector(),patternZAxis.getUnitVector()) and angleTest:
            return True
        else:
            return False
        
            
    
    
    
    def calcualteSAEDpatternForTiltsAndRotation(self,crystalOri=None,atomData=None,alphaTilt=0.,
                         betaTilt=0.,inPlaneRotation=0.,patterCenter=[0.,0.], scalingFactor=1.,SAED_ANGLE_TOLERANCE=1.,expPatternCalibration=None,
                                                  patternBounds=None):
        """
        Function for clacualting the SAED pattern for a given crystal Orienation,
        alphaTilt and betaTilt (in degrees)
        """
        if crystalOri is None:
            crystalOri = Orientation(euler=[0.,0.,0.])
            print("Warning Crystal Orienation is not specified. Hence Assuming the standard 000 orieantion")
        if  not isinstance(crystalOri, Orientation) :
            raise ValueError("crystalOri must be object of Type Orientation but what is supplied is "+type(crystalOri))

        patternXAxis,patternYAxis,patternZAxis = self.applyCrystalOriandTilts(crystalOri, alphaTilt, betaTilt, inPlaneRotation)
        
        print("The zone Axis is ", patternZAxis, patternZAxis.getUnitVector())
        
        assert np.allclose([patternXAxis.angle(patternYAxis),patternYAxis.angle(patternZAxis),patternXAxis.angle(patternZAxis)],np.array([1,1,1])*np.pi/2) , "Frame orthogonality lost"
        
        saedData = self.calcualteSAEDpatternForZoneAxis(zoneAxis=patternZAxis,desiredPatternxAxis = patternXAxis,
                                             atomData=atomData, patterCenter=patterCenter,
                                   scalingFactor=scalingFactor, SAED_ANGLE_TOLERANCE=SAED_ANGLE_TOLERANCE,
                                   )
        saedData["AlphaTilt"]=alphaTilt
        saedData["BetaTilt"]=betaTilt
        saedData["InPlaneRotation"]=inPlaneRotation
        saedData["CrystalOri"]=crystalOri
        patternInfoText = r"$\alpha$= {:.1f} $\beta$= {:.1f} $\theta$= {:.1f} ".format(alphaTilt,betaTilt,inPlaneRotation)
        patternInfoText+="\n"
        patternInfoText+="CrystalOri = {:.1f}".format(crystalOri)
        patternInfoText+="\n"
        saedData["patternInfoText"]=patternInfoText+saedData["patternInfoText"]
        
        return saedData
        
    def animateSAEDSforTilts(self, crystalOri=None,atomData=None,
                             startAlphaTilt=0.,startBetaTilt=0.,startInPlaneRotation=0.,
                             targetAlphaTilt=45.,targetBetaTilt=45.,targetInPlaneRotation=0.,
                             alphaStep=5.,betaStep=5.,inPlaneStep=5.,                             
                             patterCenter=[0.,0.], scalingFactor=1.,SAED_ANGLE_TOLERANCE=1.,markSpots=False,
                             plotKikuchi=False,detectorCoordinates=None, kikuchiWidthFactor=1.0):
        
        
        
        raise NotImplementedError("Yet to be implemented properly")
#         alphaArray = np.arange(startAlphaTilt,targetAlphaTilt+alphaStep,alphaStep)
#         betaArray = np.arange(startBetaTilt,targetBetaTilt+betaStep,betaStep)
#         inPlaneArray = np.arange(startInPlaneRotation,targetInPlaneRotation+inPlaneStep,inPlaneStep)
#         
#         plt.ion()
#         
#         annotationHandle=None
#         fig_size=[12.,9.]
#         plt.rcParams["figure.figsize"] = fig_size
#         
#         for i, item in enumerate(itertools.product(alphaArray,betaArray,inPlaneArray)):
#             plt.clf()
#             title = r"$\alpha$ = {:.2f}   $\beta$ = {:.2f}   $\theta$ {:.2f}= ".format(item[0],item[1],item[2])
#             saedData=self.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,atomData=atomData,
#                                                          alphaTilt=item[0],betaTilt=item[1],inPlaneRotation=item[2],
#                                                          patterCenter=patterCenter, scalingFactor=scalingFactor,SAED_ANGLE_TOLERANCE=1.,
#                                                          )
#             self.plotSAED(saedData)
#             
#             
#             plt.title(title)
#             plt.pause(1)
#         ##here plotting the final one one more time just to ensure that we see the pattern without disappearing    
#         saedData = self.plotSAED(saedData, markSpots=markSpots, shouldBlock=True)
  
        
         
    def plotKikuchi(self,crystalOri=None,alphaTilt=0.,betaTilt=0.,inPlaneRotation=0.,
                    patternCenter=[0.0],scalingFactor=1.0,detectorCoordinates=None,kikuchiWidthFactor=1.):
        """
        plots kikuchi lines on current axes
        """
        
        if crystalOri is None:
            crystalOri = Orientation(euler=[0.,0.,0.])
            print("Warning. Crystal Orientation was not supplied hence using [000]")
            
        
            
            combinedTilts = SaedAnalyzer.orientationForTilts(alphaTilt,betaTilt,inPlaneRotation)
            crystalOri = crystalOri*combinedTilts  
            
        planeList = self._millerPlaneSet
        kikuchiMaxHkl=2
        #print("Yes in kikuchi")
        
        for i, plane in enumerate(planeList):
            
            if all(abs(plane.hkl)<kikuchiMaxHkl):
                #print("here", plane.hkl)
                kikuchiPoints = plane.getKikuchiLine(crystalOri=crystalOri, detectorCoordinates=detectorCoordinates)
                if len(kikuchiPoints)==2:
                    kikuchiWidth = kikuchiWidthFactor*plane.dspacing
                    print("d=",plane.dspacing, kikuchiWidthFactor)
                    point1 = np.array(kikuchiPoints[0].evalf())[0:2].astype(float)*scalingFactor+patternCenter
                    point2 = np.array(kikuchiPoints[1].evalf())[0:2].astype(float)*scalingFactor+patternCenter
                    #print(point1,point2)
                    text = plane.getLatexString()                    
                    gu.plotKikuchiLinesFromPoints(point1,point2,lineWidth=kikuchiWidth,text=text,lineFraction=-1,ls='--',lc='',plotBand=False)
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()    
            
        
    def plotSAED(self,saedData,plotShow=True,figHandle=None,axisHandle=None,makeTransperent=False,markSpots=True,showAbsentSpots=True,plotKikuchi=False,
                 markKikuchi=False):
        """
        The plotting function for generating the SAEDs
        the required data for the generation of patterns must have been computed by other functions and supplied in
        required format for this function. This function only perfomrs the plotting job nothing else.
        For plotting coosite patterns supply the parameter saedData as a list of individial sade data objects
        """
        
        fig = crysFig(saedData,figHandle=figHandle,axisHandle=axisHandle)
        fig.plot(plotShow=plotShow,figHandle=figHandle,axisHandle=axisHandle,makeTransperent=makeTransperent,markSpots=markSpots,
                 showAbsentSpots=showAbsentSpots,plotKikuchi=plotKikuchi,markKikuchi=markKikuchi)
        #return ax
  
        
    def plotSolution(self,solId=None, markSpots = True,legendPosition=(0.01,0.7)):      
        
        solSet = self._solSet
        if  solId is None:
            solId = solSet[0]["solId"]
                    
        if isinstance (solId,str) and 'all' in solId.lower():
            plotAll=True
            print("alll plots will be shown now")
        else:
            plotAll = False                 
                    
        Nfigs = len(plt.get_fignums()) ####shows how many number of figures are already present
        for i, item in enumerate(solSet):
            
            if plotAll:
                solId=item["solId"]
            
            if item["solId"] == solId:
                print("Genrating plot No "+str(i+Nfigs+1))
                plt.figure(i+Nfigs+1)
                plt.subplot(111)
                self.plotSAED(saedData=item["saedData"] )
                plt.imshow(self._expImageData)
                for k in self._expSpotData["DiffractionSpots"]:            
                    plt.plot(k[0],k[1],'b<') 
                plt.axes().set_aspect('equal', 'datalim')
                title= "CI = {:.2f}".format(item["CI"])+" N of Matching spots ="+str(item["noOfMatchingSpots"])+ "Error ="+"{:.2f}".format(item["CorrelationError"])
                
                patternInfoText =r"Angle Eror (Deg) = {:.2f}".format(item["AngleError"])
                patternInfoText+="\n"
                patternInfoText+=r"%dError = {:.2f}".format(item["dError"])
                patternInfoText+="\n"
                patternInfoText+=r"ScalingFactor = {:.2f}".format(item["Calibration"]["patternScalingFactor"])
                patternInfoText+="\n"
                patternInfoText+=r"Orientation = {:.1f}".format(item["Calibration"]["crystalOrientation"])
                patternInfoText+="\n"
                patternInfoText+=r"Orientation = {:.1f}".format(item["Calibration"]["crystalOrientation2"])
                patternInfoText+="\n"                
                patternInfoText+=r"CalibRatio = {:.1f}".format(item["Calibration"]["obtainedCalibRatio"])
                
                
                annotationHandle=plt.annotate(patternInfoText, xy=legendPosition, xytext=(5, 0), 
                     xycoords='axes fraction', textcoords='offset points',
                     bbox=dict(facecolor=[1,1,1], edgecolor='white', pad=0.0))
        #print(annotationHandle,type(annotationHandle))
               
                
                plt.title(title)     
                plt.gray()
        plt.show()  
           
    @staticmethod
    def generate2Dlattice(origin=[0.,0.,],vec1=[1.0,0],vec2=[0.,1.],maxIndices= 10,latticeBounds=None,plotOn=False):
        
        checkRadius=False
        if latticeBounds is  None:
            latticeBounds = [-1e10,1e10,-1e10,1e10]
        if not isinstance(latticeBounds,list) :## case of specifying the maximum distance in reciprocal space
            checkRadius = True
            latticeRadius = latticeBounds 
        if (isinstance(latticeBounds,list) and  len(latticeBounds)==1) :
            checkRadius = True
            latticeRadius = latticeBounds[0]
            
        if isinstance(origin,list):
            origin = np.array(origin)            
        if isinstance(vec1,list):
            vec1 = np.array(vec1)
        if isinstance(vec2,list):
            vec2 = np.array(vec2)
        
        a = range(-(maxIndices),maxIndices+1)
        latticePoints=[]
        indices=[]
        vec1 = vec1-origin
        vec2 = vec2-origin
        
        
        for combination in itertools.product(a, a):
                        
            p = combination[0]*vec1+combination[1]*vec2+origin
            #print(combination,p)
            if checkRadius: ## checking to see if th espot falls within the circle defined by pc and patternBound as radius
                    dist = np.sqrt(np.sum((p-origin)*(p-origin)))
                    if dist <latticeRadius:
                            latticePoints.append(p) 
            else: ### checking by rectangular bounds
                if p[0]>=latticeBounds[0] and p[0]<=latticeBounds[1] and p[1]>=latticeBounds[2] and p[1]<=latticeBounds[3] :
                            latticePoints.append(p) 
                            indices.append(combination)

        if plotOn:
            latticePoints=np.array(latticePoints)
            plt.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')
            plt.title("2dLattice")
            plt.axes().set_aspect('equal', 'datalim')
            plt.show()
        return  latticePoints, indices

    
    def compnsateCrystalOriForSampleTilts(self,crystalOri,alphaTilt=0.,betaTilt=0.,inPlaneRotation=0.):
        """
        Corrects the crysatal orienation determined by considering the Sample holder tilts data
        """
        
        (patternXAxis,patternYAxis,patternZAxis) = self._getIdealRefFrame()
        
        patternXAxis.rotate(crystalOri)
        patternYAxis.rotate(crystalOri) 
        patternZAxis.rotate(crystalOri)
        
        inPlane = Orientation(axis= patternZAxis.getUnitVector(),degrees=inPlaneRotation).inverse
        ##in plane copnesation
        patternXAxis.rotate(inPlane)
        patternYAxis.rotate(inPlane)
        
        ## now beta tilt compnesation
        betaRotation = Orientation(axis= patternYAxis.getUnitVector(),degrees=betaTilt).inverse
        patternZAxis.rotate(betaRotation)
        patternXAxis.rotate(betaRotation)
       
        
        ## now alpha tilt compnesation
        alphaRotation = Orientation(axis= patternXAxis.getUnitVector(),degrees=alphaTilt).inverse
        patternYAxis.rotate(alphaRotation)
        patternZAxis.rotate(alphaRotation)
       
        crystalOri = Orientation(matrix=np.array([patternXAxis.getUnitVector(),
                                       patternYAxis.getUnitVector(),
                                       patternZAxis.getUnitVector()]).T)
       
        return  crystalOri
 
    @staticmethod
    def orientationForTilts(alphaTilt,betaTilt,inPlaneRotation,units="Degree"):
        """
        Helper method in which all the titls are to be specified in degrees
        returns the Orienation Object representinthe tilts
        """
        
        if "deg" not in units.lower():
            alphaTilt*=180./np.pi
            betaTilt*=180./np.pi
            inPlaneRotation*=180./np.pi
            
            
        alphaRotation = Orientation(axis=[1.,0,0],degrees=alphaTilt)
        betaRotation = Orientation(axis=[0,1.,0],degrees=betaTilt)
        betaAlpha = alphaRotation*betaRotation        
        rotatedZAxis = betaAlpha.rotate([0.,0.,1.])
        inPlaneRotation = Orientation(axis = rotatedZAxis,degrees = inPlaneRotation)
        ori = alphaRotation*betaRotation*inPlaneRotation
        #axis = (betaRotation*alphaRotation).axis
#         if not np.allclose(axis,[0,0,0]):
#             assert np.allclose(rotatedZAxis,axis) , "Probelm in Z vector in plane rotation "+str(rotatedZAxis)+" "+str(axis)
        return ori.inverse 
        
            
    
    @staticmethod
    def findTiltsForTargetZoneAxis(currentZoneAxis,targetZoneAxis,alphaAxis,betaAxis, needExactTarget=False,alphaTiltLimits=(-180.,180.,),
                                   betaTiltLimits=(-180.,180.,),fixedAxes=True):
        """
        method to find the appropraite tilt and rotation to be applied
        to achieve the required zone axis
        if needExactTarget = Falsae, tilt angles corresponding to Exact target Vector (not symmetrucally equivalent)
        shall be returned
        
        """ 
            
        alphaTiltLimits = (alphaTiltLimits[0],alphaTiltLimits[1])
        betaTiltLimits = (betaTiltLimits[0],betaTiltLimits[1])
        o = Orientation.mapVector(sourceVector=currentZoneAxis.getUnitVector(),targetVector=targetZoneAxis.getUnitVector())
        tilts = np.array([0,30.,])
        err = MillerDirection.objectiveFunctionFindTilts(tilts,currentZoneAxis,targetZoneAxis,alphaAxis, betaAxis, needExactTarget,fixedAxes=fixedAxes)
        print("Start error = :",err)
        options={'gtol': 1e-6, 'disp': False, "factor":1e7}
        i=0
        while (True):
            i=i+1
            startPoint = (np.random.randint(180,size=(2,))).tolist()
            #startPoint.append(0)
            if i ==1:
                startPoint = [0.,0.,]
            
            sol = optimize.minimize(MillerDirection.objectiveFunctionFindTilts,startPoint,
                              args=(currentZoneAxis,targetZoneAxis,alphaAxis, betaAxis, needExactTarget,fixedAxes) ,
                              bounds=(alphaTiltLimits, betaTiltLimits,), 
                              method="L-BFGS-B", options=options)
            tilts = sol.x
            err = MillerDirection.objectiveFunctionFindTilts(tilts,currentZoneAxis,targetZoneAxis,alphaAxis, betaAxis, needExactTarget,fixedAxes)
            print(i, startPoint)
            if err<2 :
                break
            else:
                print("Error is :", err, "Hence continuing", currentZoneAxis,targetZoneAxis )
            if i>10:
                print("Severe Warning !!!!!!!!!! Exceeded the max iteraqtion limit for the solver Hence Existing !!!!")
                break                                         
        print("Cross checking", err, currentZoneAxis, targetZoneAxis, sol)
        
        if (sol.fun<3):
            print (sol)
            print("the alpha and beta tilts are ", np.round(sol.x,2))
        
            alphaTilt = sol.x[0]
            betaTilt = sol.x[1]
            solution = {"alphaTilt":alphaTilt,"betaTilt":betaTilt,
                         "Error":err,
                        }
            return solution
        else:
            
            print ('No satisafactory solutiuon found check your vectors !!!!',sol)
            return {"Error":1e5}

    def loadStructureFromCif(self,cifFileName):
        """
        """
        
        
        if  isinstance(cifFileName, mg.Structure):
            structure = cifFileName
        else:
            structure = mg.Structure.from_file(cifFileName)
        lattice = structure.lattice
        
        self._structure=structure
        self._lattice=olt(matrix=lattice.matrix,orientation=Orientation(euler=[0.,0.,0.]))
        print("Loaded the structure succesfully, the lattice is \n {:.3f}".format(self._lattice))
        self._millerPlaneSet = MillerPlane.generatePlaneList(self._hklMax,self._lattice,includeSymEquals=True)
        self._millerPlaneSymUniqueSet =MillerPlane.generatePlaneList(self._hklMax,self._lattice,includeSymEquals=False) 
    
        zs = []
        coeffs = []
        fcoords = []
        occus = []
        dwfactors = []
        atomData= pyCrysUt.getAtomDataFromStructure(structure)
        self._atomData=atomData
        if self._considerDoubleDiffraction:
            self._considerDoubleDiffraction=True
        else:
            self._removeSystematicAbsences()
            print("Removing the systematic absences")
    
        
    def manuallyExtractSpotsFromExpPattern(self,imageFileName,guessSpotsLocation = None, displayImage=False,
                                           showSpotLines=False,indicateLineLengthInPlot=False):
        """
        Extracts the difraction spot info by manuals selection
        """
        
        print("Loading the image", imageFileName)
        image = cv2.imread(imageFileName,cv2.IMREAD_GRAYSCALE)
        
        self._expImageData=image
        imageWidth = image.shape[0]
        imageHeight = image.shape[1]
        
        imageCentre = np.asarray([int(image.shape[1]/2), int(image.shape[0]/2)])
        
        v1 = [imageCentre[0], imageWidth*0.2]
        v2 = [imageCentre[1], imageHeight*0.2]
        
        latticeBounds = min(imageHeight,imageWidth)/2
        
        origin = imageCentre
        print(origin,v1,v2)
                    
        if not guessSpotsLocation is None:
            spots = guessSpotsLocation
        else:
        
            spots = np.array([
                             [1725.59926137, 1529.34495589],
                             [2423.39563698 ,1291.15383991],
                             [1656.4470019 , 1860.56977554],
                             ])
        v1 = spots[1].tolist()
        v2 = spots[2].tolist()
        origin =spots[0].tolist()
        lat2d = np.array(self.generate2Dlattice(origin=origin,vec1=v1,vec2=v2,maxIndices= 3,latticeBounds=latticeBounds))
        
        spotList=[]
        spots = np.array(lat2d)
        for i in spots :
            if np.allclose(abs(i-origin),[0.,0.]):
                print("Found the orgin !!!!")
                
            else:
                spotList.append(i.tolist())
        spots = np.array(spotList) 
        
        distFromOrign = np.linalg.norm(spots-origin,axis=1)
        indx = np.argsort(distFromOrign,axis=0).tolist()
        print(indx,type(indx),distFromOrign,distFromOrign.shape)
        spots = spots[indx[:]] 
        spotData={"TransmittedSpot":origin,
                  "TransmittedSpotSize":100,
                  "DiffractionSpots":spots,
                  "SpotSizes":np.zeros((len(spots),))+100,
                  }
        
        print(spotData["DiffractionSpots"], spotData["SpotSizes"])
        self._expSpotData = spotData
        if displayImage:
            self.displyExpPattern(blockStatus=True,showSpotLines=showSpotLines, indicateLineLengthInPlot=showSpotLines)
        return spotData
       
             
    def __calibrateForSingleSpot(self,zoneAxis,spotXY,spot1Plane, pc=[0,0]):
        
        
        if len(pc)==2:
            pc.append(0) ### just adding the z value of 0 to the pattern center
        patternXAxis, patternYAxis,patternZAxis, ori = self.makeSAEDRefFrame(zoneAxis, desiredPatternxAxis=None)
        assert patternZAxis.angle(zoneAxis)<1e-3 , "Required zone axis was not achieved wanted "+str(zoneAxis)+"but achieved"+str(patternZAxis)
        x = np.dot(spot1Plane.getCartesianVec(),patternXAxis.getUnitVector())
        y = np.dot(spot1Plane.getCartesianVec(),patternYAxis.getUnitVector()) 

        x1 = pc[0]
        y1 = pc[1]
        x2 = float(spotXY[0])
        y2 = float(spotXY[1])
        sourceLine = np.array([[0.,0.],[x,y]])
        targetLine = np.array([[x1,y1],[x2,y2]])
        patternRotationAngle = pmt.angleBetween2Lines(sourceLine, targetLine,units="Deg")
        
        l1 = pmt.lineLength(targetLine)
        l2 = pmt.lineLength(sourceLine)
        patternScalingFactor = l1/l2
        origin = np.array(pc)
        sourceLine = np.array([x,y,0.])
        sourceLine = sourceLine/np.linalg.norm(sourceLine)
        targetLine = np.array([x2,y2,0.])-origin
        targetLine = targetLine/np.linalg.norm(targetLine)
        
        ori = Orientation.mapVector(sourceLine, targetLine).inverse
         
        print(sourceLine,targetLine)
        print("Rotation is {:.2f} {:.2f} {:}".format(ori,ori.degrees,ori.axis))
        patternRotationAngle = ori.degrees
        assert np.allclose(ori.rotate(sourceLine),targetLine,rtol = 1e-4,atol=1e-3), "Problem in rotation"+"rotated source & target = {:} {:}".format(ori.rotate(sourceLine),targetLine)
        
        if patternRotationAngle<0:
            print("Case of negative rotation")
            patternRotationAngle+=360
        


        Ori = Orientation(angle=-patternRotationAngle*np.pi/180,axis=patternZAxis.getUnitVector())
        oldXaxis = copy.deepcopy(patternXAxis)
        patternXAxis.rotate(Ori)
        patternYAxis.rotate(Ori)
        
        assert abs(oldXaxis.angle(patternXAxis,units="Deg")-patternRotationAngle)<1e-3 , "Problem in pattern rotation "+ \
                str(oldXaxis.angle(patternXAxis))+"  "+str(patternRotationAngle)+"  "+str(Ori)
        
        result = {"patternRotationAngle":patternRotationAngle,"patternScalingFactor":patternScalingFactor,
                  "patternXAxis":patternXAxis,"patternYAxis":patternYAxis,
                  "patternZAxis":patternZAxis,
                  }
          
        return result                
        
    
    def applyCrystalOriandTilts(self,crystalOri=Orientation.stdOri(),alphaTilt=0.,betaTilt=0.,inPlaneRotation=0.):
        """
        Helper function to apply Crystal Ori and Sample titls to calcualte the 
        Ref frame of the SAED pattern
        All tilts are in Degrees
        """
        
        (patternXAxis,patternYAxis,patternZAxis) = self._getIdealRefFrame()
               
        patternXAxis.rotate(crystalOri.inverse)
        patternYAxis.rotate(crystalOri.inverse) 
        patternZAxis.rotate(crystalOri.inverse)
        
        ## now alpha tilt
        alphaRotation = Orientation(axis= patternXAxis.getUnitVector(),degrees=alphaTilt)
        patternYAxis.rotate(alphaRotation)
        patternZAxis.rotate(alphaRotation)
        
        betaRotation = Orientation(axis= patternYAxis.getUnitVector(),degrees=betaTilt)
        patternZAxis.rotate(betaRotation)
        patternXAxis.rotate(betaRotation)
       
        inPlane = Orientation(axis= patternZAxis.getUnitVector(),degrees=inPlaneRotation)
        
        patternXAxis.rotate(inPlane)
        patternYAxis.rotate(inPlane)
          
        return patternXAxis, patternYAxis, patternZAxis
        
        
    
    
    def calculateSterioGraphicProjectionPlanes(self,planeList=None,crystalOri=None, maxHkl=4,centredOn=None,inPlaneRotation=None,desiredPatternxAxis=None):
        """
        calcualtes the steriographic projection data for the defined set of planes
        """
        if planeList is None:
            tmpList = self._millerPlaneFullSet
            planeList = []
            for i in tmpList:
                if any(i.gethkl(force3Index=True)>maxHkl):
                    continue
                else:
                    planeList.append(i)
        
        rotationNeeded, centredOn = self.__getTheOrientationForSterioProjection(crystalOri=crystalOri,centredOn=centredOn,inPlaneRotation=inPlaneRotation,desiredPatternxAxis=desiredPatternxAxis)
        
        sterioData = []
        patternInfoText = "Centred on"+str(centredOn)
        patternInfoText += "\n"
        patternInfoText +="CrystalName ="+self._name
        patternInfoText += "\n" 
        patternInfoText += r"CrystalOrienation = {:.1f}".format(rotationNeeded)
         
        planeList = sorted (planeList,reverse=True)
        
        pointXY = np.zeros((len(planeList),2))+[1e5,1e5] ## here 1e5 just some tto big number just for inititlization
        keys=['item', 'pointXY','polarPoint','isNorthPole','string']
                           
        for i, plane in enumerate(planeList):
            sterioPoint, polarPoint, isNorthPole,alphaTilt, betaTilt = plane.steriographiProjection(rotationNeeded)
            found=False
            for j in range(i+1):
                if np.allclose(polarPoint,pointXY[j]):
                    found=True
            
            if found:## already such point exists
                continue
            else:
                pointXY[i]=polarPoint
                sterioData.append({"item":plane,"pointXY":cartesianPoint,"polarPoint":polarPoint,"isNorthPole":isNorth, "string":plane.getLatexString()})
        result={}
        for key in keys:
            result[key] = [data[key] for data in sterioData]
            
        result["Orienation"]=rotationNeeded
        result["patternInfoText"]=patternInfoText
        
        return   result 
    
    
    def __getTheOrientationForSterioProjection(self,crystalOri=None,centredOn=None,inPlaneRotation=None,desiredPatternxAxis=None):
        
        if crystalOri is None:        
            if centredOn is None:
                centredOn = MillerPlane(hkl = [0,0,1], lattice=self._lattice)
        
            zoneAxis=MillerDirection(vector=centredOn.getUnitVector(),isCartesian=True,lattice=self._lattice)                
            xAxis,yAxis,zAxis,rotationNeeded = self.makeSAEDRefFrame(zoneAxis =zoneAxis, desiredPatternxAxis=desiredPatternxAxis,inPlaneRotation=inPlaneRotation)
        else:
            vec = crystalOri.inverse.rotate([0.,0.,1])
            #xAxis =  MillerDirection(vector = rotationMatrix[0],lattice=self._lattice)
            #zoneVectorInt = pmt.integerize(rotationMatrix[2])
            centredOn = MillerDirection(vector = vec, isCartesian=True, lattice=self._lattice)
            rotationNeeded=crystalOri
            
        return rotationNeeded, centredOn

        
    
    
    def findTiltsForTargetZoneAxisFromSterioGram(self,currentZoneAxis, targetZoneAxis,sterioCentredOn, sterioXaxis, alphaAxis, betaAxis,needExactTarget=False,
                                                 alphaTiltLimits=[-90,90], betaTiltLimits=[-90,90],keepTiltAxesFixed=True):
        """
        method to compute the distance that need to be travelled from the current xobe axis to target zobe axis based on the respective positions of the zone Axes on the 
        sterioGram
        """
        isSuccess=False
        xAxis,yAxis,zAxis,rotationNeeded = self.makeSAEDRefFrame(zoneAxis =sterioCentredOn,desiredPatternxAxis=sterioXaxis)
        cartesianPointCurrentZoneAxis,polarPointCurrentZoneAxis,isNorthCurrentZoneAxis, alpha, beta = currentZoneAxis.steriographiProjection(rotationNeeded)
        
        if not needExactTarget:
            targetZoneAxisList = targetZoneAxis.symmetricSet()
        else:
            targetZoneAxisList=[targetZoneAxis]
        
        startPoint = np.array(cartesianPointCurrentZoneAxis)
        targetZoneAxisSterioPositions=[]
        for item in targetZoneAxisList:
            cartesianPointTargetZoneAxis,polarPointTargetZoneAxis,isNorth,alpha,beta = item.steriographiProjection(rotationNeeded)
            if isNorthCurrentZoneAxis==isNorth:
                addDistance=0.
            else:
                addDistance=2.## diameter of sterio circle
            dist = np.linalg.norm(np.array(cartesianPointTargetZoneAxis)-startPoint)+addDistance ### this is the angualr distance in steriographic space
            absSum = np.abs(np.array(cartesianPointTargetZoneAxis)-startPoint).sum()
            targetZoneAxisSterioPositions.append({"targetZoneAxis":item,"polarPoint":polarPointTargetZoneAxis, "absSum":absSum, 
                                                  "alphaTilt":alpha, "betaTilt":beta, "distance":dist, "cartesianPoint":cartesianPointTargetZoneAxis})
        sortedList = sorted(targetZoneAxisSterioPositions, key=itemgetter('distance','absSum')) 
        chosenTargetZoneAxis = sortedList[0]["targetZoneAxis"]
        sol = self.findTiltsForTargetZoneAxis(currentZoneAxis, chosenTargetZoneAxis, alphaAxis, betaAxis, needExactTarget=True, 
                                              alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimits,fixedAxes=True)
      
        err = sol["Error"]
        
        if err<2:
            isSuccess=True
            alphaTilt, betaTilt, err = sol['alphaTilt'], sol['betaTilt'], sol['Error']
            totlaAngualarDistance = np.sqrt(alphaTilt*alphaTilt+betaTilt*betaTilt)
            return totlaAngualarDistance, alphaTilt, betaTilt, err, isSuccess, sortedList
        else:
            warnings.warn("The desired zone Axis can't be reached with the current holder tilt limit:")
            #warnings.warn("totoal angular distance to be covered to reach the required Zone Axis : {:.2f} and additional Alpha and Beta Tilts Neededed are :{:.2f} , {:.2f}".format(
            #totlaAngualarDistance,alphaTilt,  betaTilt))
            #totlaAngualarDistance=
            return None, None, None, err, isSuccess, None
        
        
    
    
    
    def calculateSterioGraphicProjectionDirection(self,dirList = None, maxUVW=2,crystalOri=None, centredOn=None,desiredPatternxAxis=None,inPlaneRotation=None):
        """
        calcualtes the steriographic projection data for the defined set of Directions
        """
        
        if dirList is None:
            print("Yes I got the None and hence generating the list on my own")
            dirList = MillerDirection.generateDirectionList(maxUVW, self._lattice, includeSymEquals=True)
        else:
            dirList = sorted(dirList , key=lambda x: x._mag) 
        
        rotationNeeded, centredOn = self.__getTheOrientationForSterioProjection(crystalOri=crystalOri,centredOn=centredOn,inPlaneRotation=inPlaneRotation,desiredPatternxAxis=desiredPatternxAxis)
        
        sterioData = []
        patternInfoText = "Centred on"+str(centredOn)
        patternInfoText += "\n"
        patternInfoText +="CrystalName ="+self._name
        patternInfoText += "\n" 
        patternInfoText += r"CrystalOrienation = {:.1f}".format(rotationNeeded)
         
        #dirList = sorted (dirList)  
             
        pointXY = [] ## here 1e5 just some tto big number just for inititlization
        
        keys=['item', 'pointXY','polarPoint','isNorthPole','string', 'alphaBeta']
                           
        for i, direction in enumerate(dirList):
            cartesianPoint,polarPoint,isNorth,alpha,beta = direction.steriographiProjection(rotationNeeded)
            
            found=False
            if i>0:
                for j in range(len(pointXY)):
                    if np.allclose(cartesianPoint,pointXY[j][0]):
                        if isNorth==pointXY[j][1]:
                            found=True
                            break
            
            if found:## already such point exists
                continue
            else:
                pointXY.append([cartesianPoint, isNorth])
                sterioData.append({"item":direction,"pointXY":cartesianPoint,"polarPoint":polarPoint, "isNorthPole":isNorth, 
                                   "alphaBeta":[alpha,beta], "string":direction.getLatexString()})
        result={}
        for key in keys:
            result[key] = [data[key] for data in sterioData]
            
        result["Orienation"]=rotationNeeded
        result["patternInfoText"]=patternInfoText
        #print (result)
        
        return   result    
    
    @staticmethod
    def extractAngleAndDvaluesFromExpSpotsData(xyData,cameraConstant=None,imagePixcelSize=None):
        
        if len(xyData) !=3:
                raise ValueError("A list of coordiantes corresponding to 3 planes must be input")
        
        
        line1 = np.array([xyData[0],xyData[1]])
        line2 = np.array([xyData[0],xyData[2]])
        angleBetweenSpots = abs(pmt.angleBetween2Lines(line1, line2,units="Deg"))
        
        if angleBetweenSpots>180:
            angleBetweenSpots=angleBetweenSpots-180.
        
        x1 = xyData[1][0]
        y1 = xyData[1][1]
                                    
        x2 = xyData[2][0]
        y2 = xyData[2][1]                       
        
        #angleBetweenSpots = np.abs(np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)*180./np.pi)
        
        spot1Length = np.sqrt((xyData[1][0]-xyData[0][0])**2+(xyData[1][1]-xyData[0][1])**2) 
        spot2Length = np.sqrt((xyData[2][0]-xyData[0][0])**2+(xyData[2][1]-xyData[0][1])**2)
        
        dRatioMeasured = max(spot1Length/spot2Length, spot2Length/spot1Length)
        result =  {"angle":angleBetweenSpots, "dRatio":dRatioMeasured, "spot1Length":spot1Length,"spot2Length":spot2Length}                               

        
        if cameraConstant is not None:
            spot1dSpacing=cameraConstant/spot1Length
            spot2dSpacing=cameraConstant/spot2Length
            result["spot1dSpacing"]=spot1dSpacing
            result["spot2dSpacing"]=spot2dSpacing
            result["spot1ReciprocalLength"]=10./spot1dSpacing
            result["spot2ReciprocalLength"]=10./spot2dSpacing
            result["imagePixcelSize"]=result["spot1ReciprocalLength"]/spot1Length
            
            return result 
        if imagePixcelSize is not None:
            result["imagePixcelSize"]=imagePixcelSize
            spot1ReciprocalLength =  spot1Length*imagePixcelSize
            spot2ReciprocalLength =  spot2Length*imagePixcelSize
            result["spot1ReciprocalLength"]=spot1ReciprocalLength
            result["spot2ReciprocalLength"]=spot2ReciprocalLength
            result["spot1dSpacing"]=10./spot1ReciprocalLength
            result["spot2dSpacing"]=10./spot2ReciprocalLength
            result["cameraConstant"] = spot1Length*result["spot1dSpacing"]
            return result
            
            
       
            
        
    
    def calcualtePatternFrom3Spots(self, spotData,xyData,allowedAngleDeviation=2,allowedDRatioDevitationPercent=10,holderData=None):
        """
        Constructs the SAED pattern based on 3 spots by doing linear combination of the respective vectors 
        here SpotData must be a list of 3 MillerPlane objects and xyData is their respective xy coordiantes (again a list)
        example data : [MillerPlane(0,0,0),MillerPlane(1,0,0),MillerPlane(0,1,0)],
        xyData= [[0,0],[10,0],[0,10]]
        Note that for the SpotData always first spot is [000]
        
        """
        if len(spotData)!=3:
            raise ValueError("A list of 3 planes must be input")
        if len(xyData) !=3:
            raise ValueError("A list of coordiantes corresponding to 3 planes must be input")
        
        xyData = np.asarray(xyData, dtype=np.float64)
        zoneAxis = MillerPlane.getZoneAxis(spotData[1], spotData[2],returnIntegerZoneAxis=True)
        if zoneAxis is None:
            return None
        line1 = np.array([xyData[0],xyData[1]])
        line2 = np.array([xyData[0],xyData[2]])
        angleBetweenSpots = abs(pmt.angleBetween2Lines(line1, line2,units="Deg"))
        
        if angleBetweenSpots>180:
            angleBetweenSpots=angleBetweenSpots-180.
        
        x1 = xyData[1][0]
        y1 = xyData[1][1]
                                    
        x2 = xyData[2][0]
        y2 = xyData[2][1]                       
        
        #angleBetweenSpots = np.abs(np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)*180./np.pi)
        
        spot1Length =np.linalg.norm(xyData[1]-xyData[0]) #np.sqrt((xyData[1][0]-xyData[0][0])**2+(xyData[1][1]-xyData[0][1])**2) 
        spot2Length =np.linalg.norm(xyData[2]-xyData[0]) # np.sqrt((xyData[2][0]-xyData[0][0])**2+(xyData[2][1]-xyData[0][1])**2)                              
        
        if spot1Length>spot2Length: #### ensuring that spot is always the nearest (to 000) spot to pattern centre
            print("Exchanging the spots !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            spot1Length,spot2Length = spot2Length, spot1Length
            xyData[1],xyData[2] = xyData[2],xyData[1]
         
        if spotData[1].dspacing<spotData[2].dspacing:
            spotData[1],spotData[2] = spotData[2],spotData[1]
            print("Exchanging the planes !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     
        dRatioMeasured = max(spot1Length/spot2Length, spot2Length/spot1Length)        
        expetedAngle   = spotData[1].angle(spotData[2],units="Deg")
        expectedDRatio = spotData[1].dspacing/spotData[2].dspacing
        expectedDRatio = max(expectedDRatio, 1./expectedDRatio) ### this is to enure always we have d ratio >= 1
        
        deviationAngle = np.around(abs(expetedAngle-angleBetweenSpots),2) #### error in degreess
        deviationDratioPercent = np.around(abs((expectedDRatio-dRatioMeasured)/expectedDRatio)*100,1)#### percentage
        
        scalingFactor1 = spot1Length*spotData[1].dspacing ###
        scalingFactor2 = spot2Length*spotData[2].dspacing
        
        meanScalingFactor = 0.5*(scalingFactor1+scalingFactor2)
        
        dev1 = abs(meanScalingFactor-scalingFactor1)/meanScalingFactor*100
        dev2 = abs(meanScalingFactor-scalingFactor2)/meanScalingFactor*100
        
        result = None
        
        if deviationAngle>allowedAngleDeviation or deviationDratioPercent>allowedDRatioDevitationPercent :
            if __debug__:
                warnings.warn("The Angle is not according to what is expected as the expetedAngle is "+str(expetedAngle)+"but given angle is "+str(angleBetweenSpots))
                warnings.warn("The measured  d ratios is  : "+str(dRatioMeasured)+" but the d expected: "+str(expectedDRatio))
            return result
        if (dev1>5. or dev2>5.):
            if __debug__:
                print("Some thng serious scaling factors are not close !! they are ",scalingFactor1, scalingFactor2 )
                print("The spot data is ",spotData, zoneAxis)
                print("lenght and dspacings of spot1 and 2 : ", spot1Length, spotData[1].dspacing, spot2Length, spotData[2].dspacing) 
            return result

        scalingFactor=scalingFactor1
        patterCenter = xyData[0]
        angleOfRotation=self.findPatternARotationToMatchExpSpots(xyData=xyData,spotData=spotData,zoneAxis=zoneAxis,scalingFactor=scalingFactor)
        patternInfoText='inPlaneRotation = '+str(np.around(angleOfRotation))     
        patternInfoText+="\n"

        saedData=self.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self._atomData,
                                           patterCenter=patterCenter, scalingFactor=scalingFactor, patternInfoText=patternInfoText,holderTiltData=holderData)        
        
        finalSaedData = SaedAnalyzer.rotateSAED(copy.deepcopy(saedData),angleOfRotation, holderData=holderData)

        saedSpotData = finalSaedData["SpotData"]  
        itemFound=False        
        isMatchingSuccessful=False
        for item in saedSpotData:
            if np.allclose(item['Plane'].gethkl(),spotData[1].gethkl()):
                itemFound=True
                if (np.allclose(item['XY'],xyData[1])):
                    isMatchingSuccessful=True
                    break
        if not itemFound:
            warnings.warn("The spot was not found in the calcualted pattern!! Some thing is seriously wrong, looking for spot : "+str(spotData[1])+'and Zone Axis is '+str(zoneAxis))
            #raise ValueError("The spot was not found in the calcualted pattern!! Some thing is seriously wrong, looking for spot : "+str(spotData[1])+'and Zone Axis is '+str(zoneAxis))
        if not isMatchingSuccessful:
            print("Initial matchiing was not succesful and hecne trying the reverse rotation!!!!")
            finalSaedData = SaedAnalyzer.rotateSAED(copy.deepcopy(saedData),-angleOfRotation, holderData=holderData)
        
        finalSaedData["angleError"]=deviationAngle
        finalSaedData["dError"]= deviationDratioPercent
        finalSaedData["spot1"]= spotData[1]
        finalSaedData["spot2"]= spotData[2]
        
        return finalSaedData
     
    
    
    def findPatternARotationToMatchExpSpots(self,xyData,spotData,zoneAxis,scalingFactor=1.0):
        """
        method to find the rotation to be applied on the given SED pattern data so that it can match with the exp spots data. Basically in plane rotaiton 
        to align the pattern with the given exp pattern
        here SpotData must be a list of 3 MillerPlane objects and xyData is their respective xy coordiantes (again a list)
        example data : [MillerPlane(0,0,0),MillerPlane(1,0,0),MillerPlane(0,1,0)],
        
        spotData = is list of planes (central spot and spot1 and spot2 respectively)
        example data : [MillerPlane(0,0,0),MillerPlane(1,0,0),MillerPlane(0,1,0)],
        xyData= [[0,0],[10,0],[0,10]]
        Note that for the SpotData always first spot is [000]
        
        returns angle of rotation in degrees such that one can use rthe self.rotateSAED method using this angle as input
        """
        tmp = zoneAxis.getOrthoset()
        patternXAxis=tmp[1]
        patternYAxis=tmp[2]
        patterCenter = xyData[0]       
        
        patternXAxis, patternYAxis,patternZAxis, ori = self.makeSAEDRefFrame(zoneAxis)
                  
        xAxis= patternXAxis.getUnitVector(returnCartesian=True)
        yAxis= patternYAxis.getUnitVector(returnCartesian=True)            
     
        p = spotData[1].getCartesianVec() 
        x = np.dot(p,xAxis)*scalingFactor+patterCenter[0]
        y = np.dot(p,yAxis)*scalingFactor+patterCenter[1]  
        
        line1 = np.array([xyData[0],xyData[1]])
        line2 = np.array([xyData[0],[x,y]]) ### this is the expected line made by automated pattern calcualtor with out proper rotation
        angleOfRotation = 360.-abs(pmt.angleBetween2Lines(line1, line2,units="Deg"))
        return angleOfRotation
    
    def plotSterioGraphicProjection(self,sterioData,figHandle=None,axisHandle=None, plotShow=True, makeTransperent=False, 
                                    marker='o', markPoints=True,projectToSameHemisphere=False):
        """
        Plots the steriographic projection of the list of the planes if provided or uses the exisitng list in the SAED anlyzzer object.
        The plane onto which centering be done can be specified. It must be a MillerPlane Object
        """
          
        fig = crysFig(sterioData,figHandle=figHandle,axisHandle=axisHandle,figType="SterioGraphicProjection",projectToSameHemisphere=projectToSameHemisphere)
        fig.plotSteriographic(plotShow=plotShow,makeTransperent=makeTransperent,markPoints=markPoints,
                 )

    @staticmethod    
    def tiltZoneAxisByAlphaAndBeta(zoneAxis,alphaTilt,betaTilt,alphaAxis,betaAxis,keepTiltAxesFixed=True):
        """
        convientinet menthod to apply holder tilts to the given zone Axis to know resultant zoneAxis as a result of the tilts.
        """
        tmpZoneAxis = copy.deepcopy(zoneAxis)
        alphaRotation = Orientation(axis= alphaAxis,degrees=alphaTilt)
        if keepTiltAxesFixed:
            betaRotation = Orientation(axis= betaAxis,degrees=betaTilt)
        else:
            tmpBetaAxis = alphaRotation.rotate(copy.deepcopy(betaAxis))
            betaRotation = Orientation(axis= tmpBetaAxis,degrees=betaTilt)
        
        totalRotation= betaRotation*alphaRotation
        tmpZoneAxis.rotate(totalRotation)
        #tmpZoneAxis.rotate(betaRotation)
        return tmpZoneAxis
     
    
    def generateDspacingTable(self,waveLength=1.5409):
        """
        returns a list of dictionaries in the form of Plane, d spacing and intensity
        """
        
        if self._planeTable is None:       
            planeList = self._millerPlaneSymUniqueSet
            planeTable=[]
            for plane in planeList :
                intensity = plane.diffractionIntensity(atomData=self._atomData)
                twoTheta = plane.get2theta(waveLength=waveLength, units='deg')
                planeTable.append({"plane":plane,"dSpacing":np.around(plane.dspacing,2),"intensity":np.around(intensity,1),
                                   "xRayPeakPosition":twoTheta})
            self._planeTable=planeTable
        
        return self._planeTable
            
  
    
    
    def solvePatternFrom3PointsCalibrated(self,expSpotData, hklMax=5,D_TOLERANCE=10, allowedAngleDeviation=2,
                                                      calibration={"cameraConstant":2252.1,"cameraLength":"60cm",
                                                                   "machine":"2000Fx"},holderData=None, spot1dRange=[],spot2dRange=[],imageData=None):
        """
        method to solve pattern from a calibrated pattern
        """
        
        result = None
        if len(spot1dRange)==0 :
            spot1dRange = [1e-5, 100] 
        if len(spot2dRange)==0 :
            spot2dRange = [1e-5, 100]
            
        xyData = expSpotData["spotXyData"]    
        line1 = np.array([xyData[0],xyData[1]])
        line2 = np.array([xyData[0],xyData[2]])
        #angleBetweenSpots = abs(pmt.angleBetween2Lines(line1, line2,units="Deg")) 
        
        spot1Length = np.sqrt((xyData[1][0]-xyData[0][0])**2+(xyData[1][1]-xyData[0][1])**2) 
        spot2Length = np.sqrt((xyData[2][0]-xyData[0][0])**2+(xyData[2][1]-xyData[0][1])**2)                              
        
        measuredDspot1 = calibration["cameraConstant"]/spot1Length
        measuredDspot2 = calibration["cameraConstant"]/spot2Length
        
        
        lowDLimitSpot1 = measuredDspot1-measuredDspot1*D_TOLERANCE/100 
        upperDLimitSpot1 = measuredDspot1+measuredDspot1*D_TOLERANCE/100
        
        lowDLimitSpot2 = measuredDspot2-measuredDspot2*D_TOLERANCE/100 
        upperDLimitSpot2 = measuredDspot2+measuredDspot2*D_TOLERANCE/100
        
        
        spot1Candiates =  self.planesWithinDrange(dMin=lowDLimitSpot1,dMax=upperDLimitSpot1)
        spot2Candiates =  self.planesWithinDrange(dMin=lowDLimitSpot2,dMax=upperDLimitSpot2)
          
        
        lookUpTable1=[]
        for p1p2 in itertools.product(spot1Candiates,spot2Candiates):            
            lookUpTable1.append((p1p2[0],p1p2[1],max(p1p2[0].dspacing/p1p2[1].dspacing, 1/(p1p2[0].dspacing/p1p2[1].dspacing))))        
        
        if len(lookUpTable1)>=1:
            result=self._solveSpotsFromLoolUpTable(lookUpTable1,expSpotData,allowedAngleDeviation,allowedDRatioDevitationPercent=10.,holderData=holderData)
        else:
            print("No solutiuion exisits!!! try different crystal system or increase tolerance in d_ratios or higher hkl indeices!!!!")
                
        if imageData is not None:
            result =self.findCorrelationWithExpPattern(result,imageData,MIN_NUMBER_OF_SPOTS_FOR_CORRELATION=5)              
                    
                
        print("Done")
        return result   
    
    def planesWithinDrange(self,dMin=0,dMax=10):
        """
        returns the planes that have dspacing (d) that satisfy dMin<=d<=dMax
        
        """
        planeList = self._millerPlaneSymUniqueSet
        planeNameList=[]
        dArray= np.array([i.dspacing for i in planeList])
        candiatesIndexes=np.where((dArray >= dMin) & (dArray <=dMax))
        candiatesIndexes =  candiatesIndexes[0].tolist() 
        allowedList = [planeList[i] for i in candiatesIndexes]
        if len(allowedList)==0:
            warnings.warn("No planes lying between d spacings {:.3f} and {:.3f} were found. ".format(dMin,dMax))
        return allowedList
        
    
    def solvePatternFrom3Points(self, expSpotData, hklMax=5,D_TOLERANCE=10, allowedAngleDeviation=2,scalingFactor=None,
                                spot1dRange=[],spot2dRange=[],imageData=None):
        """
        method to solve the zsone axis provided 3 spots are given with their XY coordiantes. Assumes first spot is central spot.
        expSpotData is a dictionary example
        expSpotData={"spotXyData":[[0,0],[10,0],[0,10.09]]}
        spot1dRange = is the low and high limits of d spacing to be considered for the spot1  e.g. [0.5 3.0], [0.3 2.0] units Angstroms
        spot2dRange = is the low and high limits of d spacing to be considered for the spot2
        
        """
        
        result=None
        
        if scalingFactor is not None:
            print("Entering the calibrated mode!!!!")
            print("Not yet implemented!!")
            return None
        
        
    
        d_ratio=[]  
        set1=[]
        set2 = [] 
        plane1Plane2=[]        
        lookUpTable1 = []

        if len(spot1dRange)==0 :
            spot1dRange = [1e-5, 100] 
        if len(spot2dRange)==0 :
            spot2dRange = [1e-5, 100]
 
        
        spot1Candiates =  self.planesWithinDrange(dMin=spot1dRange[0],dMax=spot1dRange[1])
        spot2Candiates =  self.planesWithinDrange(dMin=spot2dRange[0],dMax=spot2dRange[1])
        
        for p1p2 in itertools.product(spot1Candiates,spot2Candiates):            
            lookUpTable1.append((p1p2[0],p1p2[1],max(p1p2[0].dspacing/p1p2[1].dspacing, 1/(p1p2[0].dspacing/p1p2[1].dspacing))))        
        
        dArray = np.asarray([i[2]  for i in lookUpTable1])
                
        xyData = expSpotData["spotXyData"]    
        line1 = np.array([xyData[0],xyData[1]])
        line2 = np.array([xyData[0],xyData[2]])
        angleBetweenSpots = abs(pmt.angleBetween2Lines(line1, line2,units="Deg"))    
        x1 = xyData[1][0]
        y1 = xyData[1][1]                                
        x2 = xyData[2][0]
        y2 = xyData[2][1]   
        #angleBetweenSpots = np.abs(np.arctan2(x1*y2-y1*x2,x1*x2+y1*y2)*180./np.pi)    
        spot1Length = np.sqrt((xyData[1][0]-xyData[0][0])**2+(xyData[1][1]-xyData[0][1])**2) 
        spot2Length = np.sqrt((xyData[2][0]-xyData[0][0])**2+(xyData[2][1]-xyData[0][1])**2)                              
        dRatioMeasured = max(spot1Length/spot2Length, spot2Length/spot1Length)  
        expSpotData["d_ratio"] = dRatioMeasured
        expSpotData["Angle"]=angleBetweenSpots        
        
        lowLimit = expSpotData["d_ratio"]-expSpotData["d_ratio"]*D_TOLERANCE/100.
        upLimit =  expSpotData["d_ratio"]+expSpotData["d_ratio"]*D_TOLERANCE/100.
        solSets=np.where((dArray >= lowLimit) & (dArray <=upLimit))
        solSets =  solSets[0].tolist()  ####np.where is returning a tuple this line is to convert that into a list so that rest of the processing can be done easily
        lookUpTable1 = [lookUpTable1[item] for item in solSets]
        if len(lookUpTable1)>1:
           result=self._solveSpotsFromLoolUpTable(lookUpTable1,expSpotData,allowedAngleDeviation,allowedDRatioDevitationPercent=10)
        else:
            print("No solutiuion exisits!!! try different crystal system or increase tolerance in d_ratios or higher hkl indeices!!!!")    
        if imageData is not None:
            result =self.findCorrelationWithExpPattern(result,imageData,MIN_NUMBER_OF_SPOTS_FOR_CORRELATION=5)              
                    
                
        print("Done")
        return result   
    
    

    def _solveSpotsFromLoolUpTable(self,lookUpTable1,expSpotData,allowedAngleDeviation,allowedDRatioDevitationPercent,holderData=None):
            
            diffPatterns=[]              
            xyData=expSpotData["spotXyData"]
            solId=0
            shouldBreak=False
            for i, item in enumerate(lookUpTable1):
                    spot1Tmp,spot2Tmp = item[0],item[1]
                    spot0 = MillerPlane(hkl=[0,0,0],lattice=self._lattice)                    
                    spot1Sym = spot1Tmp.symmetricSet()
                    spot2Sym = spot2Tmp.symmetricSet()
                    for spot1 in spot1Sym:
                        for spot2 in spot2Sym: 
                            tmp  = self.calcualtePatternFrom3Spots(spotData=[spot0,spot1,spot2],xyData=xyData,allowedAngleDeviation=allowedAngleDeviation,
                                                                   allowedDRatioDevitationPercent=allowedDRatioDevitationPercent,holderData=holderData)
                            if tmp is not None:
                                appendSolution=True
                                if solId>0: ## if true we see if such zone axis already exists
                                    for pattern in diffPatterns:
                                        zoneAxis1 = pattern["pattern"]["zoneAxis"]
                                        zoneAxis2 = tmp["zoneAxis"]
                                        angle = abs(zoneAxis1.angle(zoneAxis2,considerSymmetry=True, units="Deg"))
                                        if (angle)<0.1 or angle>179. :
                                            ### case of almost parllel vectors'
                                            appendSolution=False
                                            break
                                        
                                    if appendSolution:                               
                                            tmp = self.computeKikuchiForSaed(tmp,maxHkl=2)
                                            diffPatterns.append({"solId":solId,"zoneAxis":tmp["zoneAxis"] , "pattern":tmp, "angleError":tmp["angleError"],
                                                                 "dError":tmp["dError"],"spot1":tmp["spot1"],"spot2":tmp["spot2"],"Correlation":0})
                                            
                                            solId=solId+1
                                            shouldBreak=True
                                            break
                                            
                                else: #### case of very first solution need not be compared with any other zoneAxies
                                    tmp = self.computeKikuchiForSaed(tmp,maxHkl=2)
                                    diffPatterns.append({"solId":solId,"zoneAxis":tmp["zoneAxis"], "pattern":tmp, "angleError":tmp["angleError"],
                                                         "dError":tmp["dError"],"spot1":tmp["spot1"],"spot2":tmp["spot2"],"Correlation":0})
                                    solId=solId+1
                                    shouldBreak=True
                                    break
                                    print("Found Solution")
                        if shouldBreak:
                            break                
            
            if len(diffPatterns)>0:
                result=diffPatterns                
                result = sorted(result, key=itemgetter('angleError', "dError")) 
                return result
            else:
                print("No solutiuion exisits!!! try different crystal system or increase tolerance in angles or higher hkl indeices!!!!")    
                    

    @staticmethod
    def findCorrelationWithExpPattern(solvedPatterns,imageData,MIN_NUMBER_OF_SPOTS_FOR_CORRELATION=5):
            result=solvedPatterns
            #### image data is supplied as np 2D array hence we can now find the coorelation of the solutions with exp pattern
            imageDimensions = imageData.shape
            if len(imageDimensions)==3:
                ### case of RGB image
                imData = cv2.cvtColor(imageData, cv2.COLOR_BGR2GRAY)
            else:
                imData=imageData
  
            lx,ly = imageDimensions[0],imageDimensions[1]
            X, Y = np.ogrid[0:lx, 0:ly]
            if result is not None:
                for item in  result:
                    spotData = item["pattern"]["SpotData"]
                    numPoints = len(spotData)
                    if numPoints<MIN_NUMBER_OF_SPOTS_FOR_CORRELATION:
                        warnings.warn("Number of spots are only "+str(numPoints)+" hence correlation =0")
                        item["Correlation"]=0.
                    else:
                        r = 10
                        coorrelationData=[]
                        coorrelationData = np.zeros(numPoints, dtype=float)
                        for i,spot in enumerate(spotData):
                            c = spot["XY"]
                            mask = (X - c[0] ) ** 2 + (Y - c[1]) ** 2 <= r*r
                            highestpossibleTotalIntensity=255*np.count_nonzero(mask) ## 255 is beacuse it is highest value of gray scale of of image
                            if highestpossibleTotalIntensity==0:
                                coorrelationData[i]=0
                            else:                                
                                coorrelationData[i]=np.sum(np.sum(imageData[mask]))/highestpossibleTotalIntensity
                        item["Correlation"]= np.mean(coorrelationData)    
             
             
            return result       

    
    
    @staticmethod
    def rotateSAED(saedData,rotationAngle,holderData=None):
        """
            Method for rotating the saedData by theta degrees about current ZoneAxis
            rotationAngle is in degrees
        """
        rotatedData = copy.deepcopy(saedData)
        theta=rotationAngle*np.pi/180
        cosTheta=np.cos(theta)
        sinTheta=np.sin(theta)
#                 result = {"SpotData":saedData,"zoneAxis":zoneAxis, "xAxis":patternXAxis,"yAxis":patternYAxis,
#                       "patterCenter": patterCenter,"scalingFactor":scalingFactor,
#                       "patternInfoText":patternInfoText, "angleError":deviationAngle, "dError":deviationDratioPercent}
###     saedData.append({"Plane": spot,"XY":latticePoints[i],"Intensity":intensity })
        
        
        pc = rotatedData["patterCenter"]
        spotData = rotatedData["SpotData"]
        for item in spotData:
            xy = item["XY"]
            xy = np.asarray([xy[0]-pc[0], xy[1]-pc[1]])
            xy = [xy[0]*cosTheta-xy[1]*sinTheta+pc[0], xy[1]*cosTheta+xy[0]*sinTheta+pc[1]]
            item["XY"] = xy
        rotatedData["SpotData"] = spotData    
        zoneAxis =rotatedData["zoneAxis"] 
        rotation = Orientation(axis=zoneAxis.getCartesianVec(),degrees=rotationAngle)
        rotatedData["xAxis"].rotate(rotation)
        rotatedData["yAxis"].rotate(rotation)
        rotatedData["inPlaneRotation"]=rotationAngle
        if holderData is not None:
            alphaTilt = holderData["alphaTilt"]
            betaTilt = holderData["betaTilt"]
            diffractionRotationAngle = holderData["diffractionRotationAngle"]
        else:
            alphaTilt=0.
            betaTilt=0.
            diffractionRotationAngle=0.
            
            
        ori1, ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis,rotatedData["xAxis"],alphaTilt=alphaTilt, betaTilt=betaTilt,diffractionRotationAngle =diffractionRotationAngle)
        rotatedData["Ori1"]=ori1
        rotatedData["Ori2"]=ori2
        
        patternInfoText='rotatedBy: '+str(np.around(rotationAngle,0))
        patternInfoText+="\n"
        zoneAxisInt,err = zoneAxis.integerize()
        patternXaxisInt,err = rotatedData["xAxis"].integerize()
        patternYaxisInt,err = rotatedData["yAxis"].integerize()
        patternInfoText+=r"zoneAxis = {:int}".format(zoneAxis)
        patternInfoText+="\n"
        patternInfoText+=r"X-Axis = "+str(patternXaxisInt)
        patternInfoText+="\n"
        patternInfoText+=r"Y-Axis = "+str(patternYaxisInt)
        patternInfoText+="\n"
        patternInfoText+=r"ScalingFact = "+str(np.around(rotatedData["scalingFactor"],1))
        patternInfoText+="\n"
        patternInfoText+=r"Ori1 = "+str(np.around(ori1.getEulerAngles(units='degrees'),1))
        patternInfoText+="\n"
        patternInfoText+=r"Ori2 = "+str(np.around(ori2.getEulerAngles(units='degrees'),1))
        patternInfoText+="\n"

        patternInfoTextDict = {'rotatedBy': str(np.around(rotationAngle,0)),'zoneAxis': "{:int}".format(zoneAxis),
                               'X-Axis':str(patternXaxisInt), 'Y-Axis':str(patternYaxisInt),'ScalingFact': str(np.around(rotatedData["scalingFactor"],1)),
                               'Ori1': str(np.around(ori1.getEulerAngles(units='degrees'),1)),'Ori2':str(np.around(ori2.getEulerAngles(units='degrees'),1))
                               }

        rotatedData["patternInfoText"]=patternInfoText
        rotatedData["patternInfoTextDict"] = patternInfoTextDict

        
        return  rotatedData 
    
    
    def calcualteSAEDpatternForCrystalOri(self,crystalOri,**kwarg):
        
        if not isinstance(crystalOri,Orientation):
            raise ValueError("This method takes Crystal Ori in the form of Orientation object. But "+str(type(crystalOri))+" was supplied")
        
        rotationMatrix = (crystalOri.rotation_matrix)
        xAxis =  MillerDirection(vector = rotationMatrix[0],isCartesian=True, lattice=self._lattice)
        #zoneVectorInt = pmt.integerize(rotationMatrix[2])
        zoneAxis = MillerDirection(vector = rotationMatrix[2],isCartesian=True, lattice=self._lattice)
        kwarg["desiredPatternxAxis"] = xAxis
        saedData = self.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, **kwarg)
        return saedData
           
    
    
    @staticmethod
    def getCrystalOriFromSaed(zoneAxis,patternXaxis,alphaTilt=0.,betaTilt=0.,diffractionRotationAngle=0.,options={}):
        """
        returns two possible crystal orieantions (due to 180 degree confusion in saed indexing).
        based on formalism presented in the paper : A L P H A B E T A : a de d i c a t e d o p e n - 
        s o u r c e t o o l f o r c a l c u l a t i n g T E M
        stage tilt angles  N. CAUTAERTS ,R. DELVILLE . SCHRYVERS
        
        implemetns Eqn -5 of the paper
        R_ori = R_rot^{T}R_cal^{T}R_dR_s^{T}
        where R_ori is the crystal orientation 
        R_rot^{T} is the transpose (or inverse of ) rotation due to holder alpha and beta tilts
        R_cal^{T} is the rottion to compensate the diffraction image rotation default = 0. neds to be calibrated
        R_d is the rotation to match with the detector frame of reference
        R_s^{T} is the reference frame made by the zoneaxis (as Z) and one of the spots of the saed pattern as the X axis
        
        """
        
        choiceBasedOnPaper=False
        #options["fixedAxes"]
        
        patternYaxis = np.cross(zoneAxis.getUnitVector(),patternXaxis.getUnitVector())
        patternYaxis = patternYaxis/np.linalg.norm(patternYaxis)
        
        Z_Cartesian = zoneAxis.getUnitVector()
        R_Cartesian = patternXaxis.getUnitVector()
        N_Cartesian = patternYaxis
        if choiceBasedOnPaper:
            R_s = np.array([Z_Cartesian,R_Cartesian,N_Cartesian,]).T ## here the Z_cartensian is made Z vector unlike in the reference
        else:
            R_s = np.array([R_Cartesian,N_Cartesian,Z_Cartesian]).T ## here the Z_cartensian is made Z vector unlike in the reference
        
        det = np.linalg.det(R_s)
        if abs(1.-det)>1e-6:
            raise ValueError ("Probelm in the rotation matrix as the value ofthe det = {:.5f}".format(det))
        
        R_s = Orientation(matrix=R_s)

        
        R_d1 = Orientation(euler=[0.,0.,0.])
        R_d2 = Orientation(euler =[np.pi,0.,0.],) ### orietnation for 180 degree confusion possibility
        
        
        R_cal = Orientation(axis = [0.,0.,1.], degrees=diffractionRotationAngle)
        R_rot = MillerDirection.rotationMatrixForHolderTilts(alphaTilt, betaTilt,options=options)
        Ori1 = R_rot.inverse*R_cal.inverse*R_d1*R_s.inverse
        Ori2 = R_rot.inverse*R_cal.inverse*R_d2*R_s.inverse
        return Ori1, Ori2
        
        










        