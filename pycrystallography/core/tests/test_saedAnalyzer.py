#!/usr/bin python
# -*- coding: utf-8 -*-
"""
This file is part of the pycrystallography python module

Author:         Mani Krishna K V
Website:        
Documentation:  

Version:         1.0.0
License:         The MIT License (MIT)

Copyright (c) 2017 Mani Krishna 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

test_plane.py - Unit test for plane module

"""

# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import collections
from pymatgen.util.testing import PymatgenTest
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
import pycrystallography.utilities.graphicUtilities as gu

import numpy as np
import copy
from math import pi, sqrt
#from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class SAEDCreationTestCases(PymatgenTest):
    def setUp(self):

        lat = olt.cubic(2)
        self.plane1 = MillerPlane(lattice = lat, hkl = [1,1,1]) 
        self.cubeLat = olt.cubic(2)
        self.cubeLat1 = olt.cubic(1)
        
        self.orthoLat = olt.orthorhombic(2, 1, 1)
        self.hexLat = olt.hexagonal(1, 1.63)
        self.hexLat2 = olt.hexagonal(3.2, 1.59*3.2) ## Zr 
        self.nbLattice = olt.cubic(3.03)##Nb
        
        
        self.fccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,0.])),
                (1, np.array([.5,0.,.5])),
                (1, np.array([0,.5,.5]))]
        self.bccAtomData = [(1, np.array([0.,0.,0.])),
                (1, np.array([.5,.5,.5])),
                ]
        
        self.machineConditions2000FX ={"Voltage":160e3, ##unitsw Volts
                                     "AlphaTilt":0., ## Degrees
                                     "BetaTilt":0.,
                                     "InPlaneRotaton":0.,
                                     "CameraLength":1e3, ##in mm
                                     }
        self.machineConditionsLibra ={"Voltage":200e3, ##unitsw Volts
                                     "AlphaTilt":0., ## Degrees
                                     "BetaTilt":0.,
                                     "InPlaneRotaton":0.,
                                     "CameraLength":1e3, ##in mm
                                     }
       
        self.machineConditions3010 ={"Voltage":300e3, ##unitsw Volts
                                     "AlphaTilt":0., ## Degrees
                                     "BetaTilt":0.,
                                     "InPlaneRotaton":0.,
                                     "CameraLength":1e3, ##in mm
                                     }
       
        self.zrAtomData = atomData = [(40., np.array([0.33333,    0.66667,    0.25000])),
                #(40., np.array([1./3., 2./3.,1./2.]))
                ]
        
        self.imagePathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\imageData\\'
        self.imageFileNameFcc111 = self.imagePathName+'fcc_[ -1    1  -1  ] .png'
        self.imageFileNameFcc104 = self.imagePathName+'fcc_[  0    1   4  ] .png'
        self.imageFileNameHcp0001 = self.imagePathName+'hcp_[  0    0   0   1 ] .png'
        self.imageFileNameHcp1010 = self.imagePathName+'hcp_[  0   -1   1   0 ] .png'
        
        self.imageBeamStopperZrPattern = self.imagePathName+'beamStopperPatternZr.tif'
        self.imageZrRealPattern2 = self.imagePathName+'zone_5_matrix_tmp.png'
        
        self.cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
        self.feCif = self.cifPathName+'Fe.cif'
        
       
       
    def test_init(self):
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 3)
        self.assertIsNotNone(sa, "Initialization of MillerPlane object using MillerDirection object failed")
    
 
#     def test_extractSpotsFromExpPattern(self):
#         sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 3)
#         spotData = sa.extractSpotsFromExpPattern(self.imageFileNameFcc111, displayImage=False)
#         self.assertEqual(18, len(spotData["DiffractionSpots"]), "Problem in spot detection")
#            
#         spotData = sa.extractSpotsFromExpPattern(self.imageFileNameHcp0001, displayImage=False)
#         self.assertEqual(36, len(spotData["DiffractionSpots"]), "Problem in spot detection")
#         
#         sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 3)
#         spotData = sa.extractSpotsFromExpPattern(self.imageBeamStopperZrPattern, displayImage=True,
#                                                  minSpotSize = 500,showSpotLines=False,beamStopper=True)
#         
#         centreList = [[4,9],[1,7],[8,10]] ### these are sot id used for the location of the central beam
#         ignoreSpotList=[2,3,12]
#         spotData = sa.correctExpDetectedSpots(beamStopper=True,spotsForLocatingTransmittedBeam=centreList,
#                                               ignoreSpotList=ignoreSpotList)
#         
#        
#         self.assertEqual(18, len(spotData["DiffractionSpots"]), "Problem in spot detection")
#            
#         
#        
#     def test_slectSpotsForIndexing(self):
#         sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 3)
#         spotData = sa.extractSpotsFromExpPattern(self.imageFileNameFcc111, displayImage=False)
#         measureData = sa.slectSpotsForIndexing(spotIds=[1,2])
#         self.assertTrue(np.abs(120-measureData["Angle"])<1,"Problem in spot slection")
#            
#               
#         sa1 = SaedAnalyzer(lattice=self.hexLat2,hklMax = 3)
#         spotData = sa1.extractSpotsFromExpPattern(self.imageFileNameHcp0001, displayImage=True)
#         measureData = sa1.slectSpotsForIndexing(desiredAngle=120.)
#            
#            
#         sa2 = SaedAnalyzer(lattice=self.hexLat2,hklMax = 3)
#         spotData =sa2.extractSpotsFromExpPattern(self.imageFileNameHcp1010, displayImage=False)
#         measureData = sa2.slectSpotsForIndexing(desiredAngle=90.)
#         print(measureData)
#         self.assertTrue(np.abs(90-measureData["Angle"])<1,"Problem in spot slection")
#         measureData = sa2.slectSpotsForIndexing(spotIds=[5,13])
#         print(measureData)
# # #         
    def test_calcualteSAEDpatternForZoneAxis(self):         
        
        
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 4,atomData=self.fccAtomData)
        zoneAxis = MillerDirection(lattice=self.cubeLat1,vector=[2,2,3])    
        saedData = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.fccAtomData,
                                           patterCenter=[0.,0.], scalingFactor=1.0)
        sa.plotSAED(saedData, plotShow=True, markSpots=True, showAbsentSpots=True)
        #exit()
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4,atomData=self.zrAtomData)
        zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[0,0,0,1])
         
        scalingFactor=sa._cameraConstant
         
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])*100## all are in mm
        detectorCoordinates[:,2]=-sa._machineConditions["CameraLength"] 
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.fccAtomData,
                                           patterCenter=[0.,0.], scalingFactor=scalingFactor)
        zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[2,-1,-1,0])    
        saedData2 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.fccAtomData,
                                           patterCenter=[0.,0.], scalingFactor=scalingFactor)
          
        saedData = [saedData1,saedData2]     
        #sa.plotSAED(saedData=saedData)
            
        print("Yes the saed pattern calualtion is done", saedData)
         
        for i in saedData1["SpotData"]:
            print(i["Plane"],i['XY'])
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4)
          
#         zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[1,0,-1,2])
#          
        xAxis = [ 0.90863877, -0.05274743, -0.41423822 ]
        desiredPatternxAxis = MillerDirection(vector=xAxis,lattice=self.hexLat2,isCartesian=True)
        rotation=Orientation(axis = zoneAxis.getUnitVector(),angle=83.2*np.pi/180.)
        print("before", desiredPatternxAxis.getUnitVector())
        desiredPatternxAxis.rotate(rotation)
        print("After", desiredPatternxAxis.getUnitVector())
        saedData = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, desiredPatternxAxis = desiredPatternxAxis, atomData=self.zrAtomData,
                                           patterCenter=[2198.33, 1174.0], scalingFactor=1446.)
        print("Yes the saed pattern calualtion is done", saedData)
        for i in saedData["SpotData"]:
            print(i["Plane"],i['XY'])
    
          
        patternParameters = {'patternRotationAngle': 119.85, 'patternScalingFactor': 172.555, 
                 'patternXAxis': [-0.103722, 0.208333, -0.104610, -0.000000 ],
                 'patternZAxis': [0.,0.,0., 1.0 ] ,
                 'patterCenter': [231.0, 199.0], 
                 'crystalOrientation': np.array([119.85901616, 180. , 0.])}
          
        zoneAxis = MillerDirection(vector=patternParameters["patternZAxis"],lattice=self.hexLat2)
        patternXAxis = MillerDirection(vector=patternParameters["patternXAxis"],lattice=self.hexLat2)
          
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.zrAtomData,
                                           patterCenter=patternParameters['patterCenter'], 
                                           scalingFactor=patternParameters['patternScalingFactor'])
           
        sa.plotSuperImposedExpAndSimulatedPattern(simulatedData=saedData1,expImage=self.imageFileNameHcp0001,markSpots=True,)


    def test_plotKikuchi(self):
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4,atomData=self.zrAtomData)
          
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])*4##
        detectorCoordinates[:,2]=-4.        
        crystalOri = Orientation(euler=[0*np.pi/180.,0*np.pi/180.,0.])
        alphaTilt, betaTilt, inPlaneRotation = 0.,0.,0.
        patternCenter=[0.,0.]
        scalingFactor=1.0
        kikuchiWidthFactor=1.0
           
        sa.plotKikuchi(crystalOri, alphaTilt, betaTilt, inPlaneRotation, patternCenter, scalingFactor, detectorCoordinates, kikuchiWidthFactor)
           
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4,atomData=self.zrAtomData)
          
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])*4##
        detectorCoordinates[:,2]=-4.        
        crystalOri = Orientation(euler=[0*np.pi/180.,90*np.pi/180.,0.])
        alphaTilt, betaTilt, inPlaneRotation = 0.,0.,0.
        patternCenter=[0.,0.]
        scalingFactor=1.0
        kikuchiWidthFactor=1.0
           
        sa.plotKikuchi(crystalOri, alphaTilt, betaTilt, inPlaneRotation, patternCenter, scalingFactor, detectorCoordinates, kikuchiWidthFactor)
#  


    def test_makeSAEDRefFrame(self):
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 6,atomData=self.fccAtomData)
        v1=[  1 ,   6,   2  ]
        v2=[  -2,0,1  ]
        
        zoneAxis = MillerDirection(lattice=self.cubeLat,vector=v1)
        xAxis = MillerDirection(lattice=self.cubeLat,vector=v2)
        print("testing ",zoneAxis, xAxis)
        patternXAxis, patternYAxis,patternZAxis = sa.makeSAEDRefFrame(zoneAxis =zoneAxis, desiredPatternxAxis=xAxis)
 
        
        for i in range(500):
            vec = np.random.randint(21, size=3)
            zoneAxis = MillerDirection(lattice=self.cubeLat,vector=vec)
            xAxis = zoneAxis.getPerpendicularDirection()
            crystalOri = Orientation(axis =zoneAxis.getUnitVector(),degrees=np.random.random()*180 )
            xAxis.rotate(crystalOri)
            #print("testing ",zoneAxis, xAxis)
            patternXAxis, patternYAxis,patternZAxis = sa.makeSAEDRefFrame(zoneAxis =zoneAxis, desiredPatternxAxis=xAxis)
            
            self.assertAlmostEqual(patternXAxis.angle(xAxis),0.,2,'Problem in making ref Frame for Zone Axis'+str(zoneAxis)+"and X axis"+str(xAxis))


    def test_compnsateCrystalOriForSampleTilts(self):
        
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 6,atomData=self.fccAtomData)
        OriginalOri = Orientation(euler=[0.,0.,0.])
        alphaTilt,betaTilt, inPlaneRotation  = 30,40.,90.

       
        combinedTilts = SaedAnalyzer.orientationForTilts(alphaTilt, betaTilt, inPlaneRotation)
        
        modifiedOri = combinedTilts*OriginalOri
        ori = sa.compnsateCrystalOriForSampleTilts(modifiedOri,alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)
        misAngle = ori.misorientation(OriginalOri).degrees
        
        self.assertTrue(ori.misorientation(OriginalOri).degrees<1, "problem in ori clacualtion as angle deviation is = "+str(misAngle))
        
        for i in range(10):
            OriginalOri = Orientation.random()
            alphaTilt,betaTilt, inPlaneRotation  = np.random.random((3,))*180
            combinedTilts = SaedAnalyzer.orientationForTilts(alphaTilt, betaTilt, inPlaneRotation)
            modifiedOri = combinedTilts*OriginalOri
            ori = sa.compnsateCrystalOriForSampleTilts(modifiedOri,alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)
            misAngle = ori.misorientation(OriginalOri).degrees
            self.assertTrue(ori.misorientation(OriginalOri).degrees<1, "problem in ori clacualtion as angle deviation is = "+str(misAngle))
        
        print("Completed the compnsateCrystalOriForSampleTilts testing")
    
    
    def test_calcualteCalibration(self):
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 6,atomData=self.fccAtomData)
        zoneAxis = MillerDirection(lattice=self.cubeLat,vector=[1,1,1])
        spot1 = MillerPlane(hkl=[0,2,-2],lattice=self.cubeLat)
        spot2 = MillerPlane(hkl=[0,-2,2],lattice=self.cubeLat)
        spot3 = MillerPlane(hkl=[0,-4,4],lattice=self.cubeLat)        
               
        calib1 = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=[0.3660, 1.366],
                                           spot1Plane=spot1,pc=[0,0],checkForKnownCalibration=False)
                
        self.assertTrue(abs(calib1["patternRotationAngle"])<1 or abs(360-calib1["patternRotationAngle"])<1.,"roation angle wrong as the value is"+str(calib1["patternRotationAngle"]))
        self.assertAlmostEqual(calib1["patternScalingFactor"], 1.00, 2, "scaling factor wrong as the value is "+str(calib1["patternScalingFactor"]))
          
        crystalOri = calib1["crystalOrientation"]
        sf = calib1["patternScalingFactor"]
        pc = calib1["patterCenter"]
          
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri, alphaTilt=0., 
                    betaTilt=0., inPlaneRotation=0., patterCenter=pc, scalingFactor=sf)
          
        for item in saedData["SpotData"]:
            spot = item["Plane"]
            XY = item["XY"]
            if  spot == spot1 :
                self.assertArrayAlmostEqual(XY, [0.3660, 1.366], 2, 'Problem in calibration')
            elif spot == spot2 :
                self.assertArrayAlmostEqual(XY, [-0.3660, -1.366], 2, 'Problem in calibration')
            elif spot == spot3 :
                self.assertArrayAlmostEqual(XY, [-0.3660*2, -1.366*2], 2, 'Problem in calibration')
              
            else:
                pass
        
        #### case of Nb Calibration pattern for 40cm 
        zoneAxis = MillerDirection(vector=[1,0,0],lattice=self.nbLattice)
        spot1 = MillerPlane(hkl=[0,0,-1],lattice=self.nbLattice)
        spot2 = MillerPlane(hkl=[0,-1,-1],lattice=self.nbLattice)
        spot3 = MillerPlane(hkl=[0,-1,0],lattice=self.nbLattice)
        
        spots=[spot1,spot2,spot3]  
        spotXYList = [[2513,1087],[2447,667],[1995,737]]      
               
        calib1 = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=[2513,1087],
                                           spot1Plane=spot1,pc=[2061,1153],checkForKnownCalibration=False)
                
        self.assertAlmostEqual(calib1["patternRotationAngle"],  8.30748, 2, "roation angle wrong")
        
        calib2 = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=spotXYList,
                                           spot1Plane=spots,pc=[2061,1153],checkForKnownCalibration=False)
        
        print(abs(calib2["patternRotationAngle"]))
        self.assertTrue(abs(calib2["patternRotationAngle"]- 7.6666666)<2, "roation angle wrong")
                
        self.assertTrue(abs(calib2["patternScalingFactor"]-1384)<20, "scaling factor wrong, "+str(calib2["patternScalingFactor"]))
          
        crystalOri = calib1["crystalOrientation"]
        sf = calib1["patternScalingFactor"]
        pc = calib1["patterCenter"]
        
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax=4,considerDoubleDiffraction=True)
           
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri, alphaTilt=0., 
                    betaTilt=0., inPlaneRotation=0., patterCenter=pc, scalingFactor=sf)
           
        for item in saedData["SpotData"]:
            spot = item["Plane"]
            XY = item["XY"]
            if  spot == spot1 :
                self.assertArrayAlmostEqual(XY, [0.3660, 1.366], 2, 'Problem in calibration')
            elif spot == spot2 :
                self.assertArrayAlmostEqual(XY, [-0.3660, -1.366], 2, 'Problem in calibration')
            elif spot == spot3 :
                self.assertArrayAlmostEqual(XY, [-0.3660*2, -1.366*2], 2, 'Problem in calibration')
               
            else:
                pass

        count = 0
        for i in range(2): ### radomly 10 different orieantions are now being tried
            crystalOri = Orientation.random()
            crystalOri = Orientation(euler=np.array([90,90,0])*np.pi/180.)
            print("Trying the oreination", crystalOri)
            sf = np.random.random((1,))*100
            pc = np.random.random((2,))*100
            pc = pc.tolist()
            saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri, alphaTilt=0., 
                        betaTilt=0., inPlaneRotation=0., patterCenter=pc, scalingFactor=sf)
            zoneAxis = saedData["zoneAxis"]
            rotationAngles=[]
            for item in saedData["SpotData"]:
                spot = item["Plane"]
                XY = item["XY"]
                 
                calib = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=XY,
                                               spot1Plane=spot,pc=pc)
                
                rotationAngles.append(calib["patternRotationAngle"])
                self.assertArrayAlmostEqual(sf, calib["patternScalingFactor"], 2, 'Problem in scaling factor')
                self.assertArrayAlmostEqual(crystalOri.rotation_matrix,calib["crystalOrientation"].rotation_matrix,4,'Problem in orienation')
                count+=1
                print("passed count = ",count)
            print(rotationAngles)
            
        print("Completed the calibration testing")

#     
    def test_calcualteSAEDpatternForTiltsAndRotation(self):
        
        
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax=4,atomData=self.fccAtomData)
        
        crystalOri = Orientation(euler=np.array([34.,23,45])*np.pi/180)
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri)   
        currentZoneAxis = saedData["zoneAxis"]
        targetZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.cubeLat1)
        solution = sa.findTiltsForTargetZoneAxis(crystalOri, currentZoneAxis, targetZoneAxis)
        alphaTilt,betaTilt = solution["alphaTilt"], solution["betaTilt"]
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,
                        alphaTilt=alphaTilt,betaTilt=betaTilt)      
        obtainedZoneAxis = saedData["zoneAxis"]
        self.assertAlmostEqual(obtainedZoneAxis.angle(targetZoneAxis,considerSymmetry=True),0,5,"Problem in zone axis"+str(obtainedZoneAxis))
        
        targetZoneAxis = MillerDirection(vector=[1,1,1],lattice=self.cubeLat1) 
        for i in range(10):
            crystalOri = Orientation.random()
            saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri)   
            currentZoneAxis = saedData["zoneAxis"]
            
            solution = sa.findTiltsForTargetZoneAxis(crystalOri, currentZoneAxis, targetZoneAxis)
            alphaTilt,betaTilt = solution["alphaTilt"], solution["betaTilt"]
            saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,
                            alphaTilt=alphaTilt,betaTilt=betaTilt)      
            #sa.plotSAED(saedData,markSpots=True,shouldBlock=True)
            obtainedZoneAxis = saedData["zoneAxis"]
            obtainedAngle = abs(obtainedZoneAxis.angle(targetZoneAxis,considerSymmetry=True,units="deg"))
            self.assertTrue(obtainedAngle<2,"Problem in zone axis"+str(obtainedZoneAxis)+" the angle error is "+str(obtainedAngle))
          
        
        
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax=4,atomData=self.zrAtomData)
        crystalOri = Orientation(euler=np.array([0,0,0])*np.pi/180)
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri)   
        currentZoneAxis = saedData["zoneAxis"]
        targetZoneAxis = MillerDirection(vector=[2,-1,-1,3.],lattice=self.hexLat2)
        
        solution = sa.findTiltsForTargetZoneAxis(crystalOri, currentZoneAxis, targetZoneAxis,needExactTarget=True)
        alphaTilt,betaTilt = solution["alphaTilt"], solution["betaTilt"]
            
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,
                        alphaTilt=alphaTilt,betaTilt=betaTilt)      
            
        sa.plotSAED(saedData=saedData, markSpots=True,shouldBlock=True)
        obtainedZoneAxis = saedData["zoneAxis"]
            #expectedZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.cubeLat1)
        
        self.assertAlmostEqual(obtainedZoneAxis.angle(targetZoneAxis,considerSymmetry=True,units="deg"),0,3,"Problem in zone axis"+str(obtainedZoneAxis))
          
        
        crystalOri = Orientation(matrix =np.array([[0.577350269189626,0.577350269189626,0.577350269189626],
                                            [0.707106781186547,-0.707106781186547, 0.],
                                            [0.408248290463863,  0.408248290463863, -0.816496580927726]]))
         #### above matrix corrsponds to 111 along ND
        expectedZoneAxis = MillerDirection(vector=[1,1,-2],lattice=self.cubeLat1)
        targetZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.cubeLat1)
        solution = sa.findTiltsForTargetZoneAxis(crystalOri,expectedZoneAxis, targetZoneAxis)
        alphaTilt,betaTilt,inPlaneRotation = solution["alphaTilt"], solution["betaTilt"], solution["inPlaneRotation"],
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,
                    alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)      
        
        #sa.plotSAED(saedData=saedData, markSpots=True,shouldBlock=True)
        obtainedZoneAxis = saedData["zoneAxis"]
        print(obtainedZoneAxis)
        #expectedZoneAxis = MillerDirection(vector=[1,1,-2],lattice=self.cubeLat1)
    
        self.assertAlmostEqual(obtainedZoneAxis.angle(targetZoneAxis,considerSymmetry=True),0,5,"Problem in zone axis")
        
        
        
      
        
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax=3,atomData=self.fccAtomData,considerDoubleDiffraction=True)
        crystalOri = Orientation(euler=np.array([48.4,97.2,220.2])*np.pi/180)
        crystalOri = Orientation(euler=np.array([0,0,0])*np.pi/180)
        alphaTilt,betaTilt,inPlaneRotation = 45,30,45
        
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri, atomData=self.fccAtomData,
                    alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)      
        
        sa.plotSAED(saedData=saedData, markSpots=True,shouldBlock=True)
        
        
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4,atomData=self.zrAtomData)
        crystalOri = Orientation(matrix =np.array([[0.577350269189626,0.577350269189626,0.577350269189626],
                                            [0.707106781186547,-0.707106781186547, 0.],
                                            [0.408248290463863,  0.408248290463863, -0.816496580927726]]))
         #### above matrix corrsponds to 111 along ND
        crystalOri = Orientation(euler=[45*np.pi/180.,0*np.pi/180.,0.])
        alphaTilt,betaTilt,inPlaneRotation = 0.,0.,0. ###in degrees
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])*3.0##
             
        detectorCoordinates[:,2]=-0.1
            
        kikuchiWidthFactor=1.
               
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri, atomData=self.fccAtomData,alphaTilt=alphaTilt,
                         betaTilt=betaTilt,inPlaneRotation=inPlaneRotation,patterCenter=[0.,0.], scalingFactor=1.,SAED_ANGLE_TOLERANCE=2.,
                        )
               
        print(saedData)
        sa.plotSAED(saedData=saedData, markSpots=True,shouldBlock=True)
#           

                
        patternParameters = {'patternRotationAngle': 119.85, 'patternScalingFactor': 172.555, 
                 'patternXAxis': [-0.103722, 0.208333, -0.104610, -0.000000 ],
                 'patternZAxis': [0.,0.,0., 1.0 ] ,
                 'patterCenter': [231.0, 199.0], 
                 'crystalOrientation': np.array([119.85901616, 180. , 0.])*np.pi/180}
         
        zoneAxis = MillerDirection(vector=patternParameters["patternZAxis"],lattice=self.hexLat2)
        patternXAxis = MillerDirection(vector=patternParameters["patternXAxis"],lattice=self.hexLat2)
         
        ori = Orientation(euler = patternParameters['crystalOrientation'])
        saedData1 = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=ori, atomData=self.zrAtomData,
                                           patterCenter=patternParameters['patterCenter'], 
                                           scalingFactor=patternParameters['patternScalingFactor'])
          
        sa.plotSuperImposedExpAndSimulatedPattern(simulatedData=saedData1,expImage=self.imageFileNameHcp0001,markSpots=True,)
   
        
    
    def test_solveSetOfSpots(self):
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 6,atomData=self.fccAtomData,
                          considerDoubleDiffraction=False)
        spotData = sa.extractSpotsFromExpPattern(self.imageFileNameFcc111, displayImage=False,
                                                minSpotSize = 200,showSpotLines=False,beamStopper=False)
        solution = sa.solveSetOfSpots(spotIds=[1,3], dTol=2.,angleTol=2,maxHkl=4)
        print("solutuon over Done")
        for i in solution:
            print("{:2d} {:int} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:2d}".
                  format(i["solId"],i["zoneAxis"],i["spot1"],i["spot2"],
                   i["CorrelationError"],i["Angle"], i["AngleError"],
                   i["dRatio"],i["dError"],i["noOfMatchingSpots"]))
        for i in solution:
             print(i["solId"],i["Calibration"])
        #sa.plotSolution(solId='all')
        zoneAxis = MillerDirection(vector=[1,1,1],lattice=self.cubeLat).getUnitVector(returnCartesian=False)
        
        self.assertTrue(solution[0]["zoneAxis"].isSymmetric(zoneAxis),"Could not solve it properly")

        
        
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 6,atomData=self.zrAtomData,
                          considerDoubleDiffraction=False)
        spotData = sa.extractSpotsFromExpPattern(self.imageFileNameHcp0001, displayImage=False)
        solution = sa.solveSetOfSpots(spotIds=[1,3], dTol=2.,angleTol=2,maxHkl=4)
        print("solutuon over Done")
        for i in solution:
            print("{:2d} {:int} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:2d}".
                  format(i["solId"],i["zoneAxis"],i["spot1"],i["spot2"],
                   i["CorrelationError"],i["Angle"], i["AngleError"],
                   i["dRatio"],i["dError"],i["noOfMatchingSpots"]))
        for i in solution:
            print(i["solId"],i["Calibration"])
         
        sa.plotSolution(solId=2)
        patternParameters = {'patternRotationAngle': 119.85, 'patternScalingFactor': 172.555, 
                 'patternXAxis': [-0.103722, 0.208333, -0.104610, -0.000000 ],
                 'patternZAxis': [0.,0.,0., 1.0 ] ,
                 'patterCenter': [231.0, 199.0], 
                 'crystalOrientation': np.array([119.85901616, 180. , 0.])}
#          
# #     
        zoneAxis = MillerDirection(vector=[0,0,0,1],lattice=self.hexLat2).getUnitVector(returnCartesian=False)
        
        self.assertTrue(solution[0]["zoneAxis"].isSymmetric(zoneAxis),"Could not solve it properly")

        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 9,atomData=self.fccAtomData)
        spotData = sa.extractSpotsFromExpPattern(self.imageFileNameFcc104, displayImage=False)
        measureData = sa.slectSpotsForIndexing(spotIds=[1,7])
        print(measureData)
        solution = sa.solveSetOfSpots(spotIds=[2,7], dTol=5.,angleTol=2,maxHkl=9)
        for i in solution:
            print("{:2d} {:int} {:int} {:int} {:.2f} {:.2f}".format(i["solId"],i["zoneAxis"],i["spot1"],i["spot2"],i["AngleError"],
                   i["CorrelationError"]))
     
        sa.plotSolution(solId=22, markSpots=True)
        zoneAxis = MillerDirection(vector=[4,1,0],lattice=self.cubeLat).getUnitVector(returnCartesian=False)
        
        self.assertTrue(solution[0]["zoneAxis"].isSymmetric(zoneAxis),"Could not solve it properly")
        
#         sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 4,atomData=self.fccAtomData)
#         spotData = sa.extractSpotsFromExpPattern(self.imageFileNameFcc111, displayImage=False)
         
        
#   
# 
       
#         

#                                                    
#         zoneAxis = solution[Indx]["zoneAxis"]
#          
#         spotXY = measureData["SpotAbsXY"][0]
#         spot1 =solution[Indx]["spot1"] 
#         print("Trying ", zoneAxis, spot1)
#         pc = spotData["TransmittedSpot"].tolist()[0]
#         print("spotXY and pc ",spotXY,pc)
#         calib2 = sa.calcualteCalibration(zoneAxis =zoneAxis, spotXY=spotXY,
#                                           spot1Plane=spot1,pc=pc)
# #          
# #       
#           
#         saedData = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.zrAtomData,
#                                            expPatternCalibration=calib2,plotOn=False)
#         print("Yes the saed pattern calualtion is done", saedData)
#         for i in saedData["SpotData"]:
#             print(i["Plane"],i['XY'])
#         
#         
#         print("Done")
#        
    def test_generate2Dlattice(self):
        #sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 3,atomData=self.fccAtomData)
        points = SaedAnalyzer.generate2Dlattice(origin=[10,15.], vec1=[12.,15.], vec2=[12.0,17.0], maxIndices=10, latticeBounds=5., plotOn=True)
        print("2dLattice") 
        spots = np.array( [[2452.11878824, 1273.12158461],
             [1712.80745697, 1527.91505976],
             [1640.00932121, 1922.31179733],
             [1712.80745697 ,1527.91505976]])
        v1 = spots[0].tolist()
        v2 = spots[2].tolist()
        origin =spots[1].tolist()
        latticeBounds = [0,4000,0,2500]
        #while True:

        print(v1,v2,origin)
        lat2d = np.array(SaedAnalyzer.generate2Dlattice(origin=origin,vec1=v1,vec2=v2,latticeBounds=latticeBounds,plotOn=True))
        

    def test_findTiltsForTargetZoneAxis(self):
        currentZoneAxis = MillerDirection(vector=[0,0,1.],lattice=self.cubeLat)
        targetZoneAxis = MillerDirection(vector=[1,1,1],lattice=self.cubeLat)
        alphaAxis,betaAxis=[1.,0.,0], [0.,1.,0.]
        alphaTiltLimits,betaTiltLimits,inPlaneRotationLimits=(-90.,90.,),(-90.,90,),(0.,0.,)
        sol = SaedAnalyzer.findTiltsForTargetZoneAxis(currentZoneAxis=currentZoneAxis, targetZoneAxis=targetZoneAxis,
                                                 alphaAxis=alphaAxis, betaAxis=betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimits)
       
        print(sol)
        
        self.assertAlmostEqual(sol["Error"],0,2,"Problem in Optimization")
        
        currentZoneAxis = MillerDirection(vector=[1,1,1],lattice=self.cubeLat)
        targetZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.cubeLat)
        alphaAxis,betaAxis=[1.,0.,0], [0.,1.,0.]
        alphaTiltLimits,betaTiltLimits,inPlaneRotationLimits=(-90.,90.,),(-90.,90,),(0.,0.,)
        sol = SaedAnalyzer.findTiltsForTargetZoneAxis(currentZoneAxis=currentZoneAxis, targetZoneAxis=targetZoneAxis,
                                                 alphaAxis=alphaAxis, betaAxis=betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimits)
       
        print("The reverse solutin is ", sol)
        
        self.assertAlmostEqual(sol["Error"],0,2,"Problem in Optimization")

        currentZoneAxis = MillerDirection(vector=[0,0,0,1.],lattice=self.hexLat2)
        targetZoneAxis = MillerDirection(vector=[-2,1,1,0],lattice=self.hexLat2)
        alphaTiltLimits,betaTiltLimts,inPlaneRotationLimits=(-360.,360.,),(-360.,360.,),(0.,0.,)
        alphaAxis,betaAxis=[1.,0.,0], [0.,1.,0.]
        alphaTiltLimits,betaTiltLimits,inPlaneRotationLimits=(-90.,90.,),(-90.,90,),(0.,0.,)
        sol = SaedAnalyzer.findTiltsForTargetZoneAxis(currentZoneAxis=currentZoneAxis, targetZoneAxis=targetZoneAxis,
                                                 alphaAxis=alphaAxis, betaAxis=betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimits)
       
        self.assertAlmostEqual(sol["Error"],0,2,"Problem in Optimization")
        print(sol)
        #self.assertEqual(1, 2, "as u wish")
    def test_applyCrystalOriandTilts(self):
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax=4,atomData=self.zrAtomData)
        crystalOri = Orientation(euler=np.array([0,0,0])*np.pi/180)        
        alphaTilt,betaTilt , inPlaneRotation = 45,0,0
        patternXAxis, patternYAxis, patternZAxis = sa.applyCrystalOriandTilts(crystalOri, alphaTilt, betaTilt, inPlaneRotation)
        print(patternXAxis, patternYAxis, patternZAxis) 
        
        crystalOri = Orientation(euler=np.array([0,0,0])*np.pi/180)
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri)   
        currentZoneAxis = saedData["zoneAxis"]
               
        targetZoneAxis = MillerDirection(vector=[1,1,-2,0],lattice=self.hexLat)
        solution = sa.findTiltsForTargetZoneAxis(crystalOri, currentZoneAxis, targetZoneAxis,needExactTarget=True)
        alphaTilt,betaTilt,inPlaneRotation = solution["alphaTilt"], solution["betaTilt"], solution["inPlaneRotation"]
                    
            #inPlaneRotation = np.random.randint(360)
        saedData = sa.calcualteSAEDpatternForTiltsAndRotation(crystalOri=crystalOri,
                        alphaTilt=alphaTilt,betaTilt=betaTilt,inPlaneRotation=inPlaneRotation)      
              
            #sa.plotSAED(saedData=saedData, markSpots=True,shouldBlock=True)
        obtainedZoneAxis = saedData["zoneAxis"]
            #expectedZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.cubeLat1)
        patternXAxis, patternYAxis, patternZAxis = sa.applyCrystalOriandTilts(crystalOri, alphaTilt, betaTilt, inPlaneRotation)
        self.assertAlmostEqual(patternZAxis.angle(obtainedZoneAxis,units="Deg"),0,3, "Issue is here" )
        self.assertAlmostEqual(obtainedZoneAxis.angle(targetZoneAxis,considerSymmetry=True,units="Deg"),0,5,"Problem in zone axis"+str(obtainedZoneAxis))
         
        for i in np.arange(5,120,10):
            inPlaneRotation=i
            print(i)
            patternXAxis, patternYAxis, patternNewZAxis = sa.applyCrystalOriandTilts(crystalOri, alphaTilt, betaTilt, inPlaneRotation)
            angle = patternZAxis.angle(patternNewZAxis,units="deg")
            self.assertTrue(angle<1,'Inplane problem'+str(patternZAxis)+" "+str(patternNewZAxis)+"angle ="+str(angle))
        
        
    def test_loadStructureFromCif(self):
        cifFileName = self.feCif
        sa = SaedAnalyzer(hklMax=5)
        sa.loadStructureFromCif(cifFileName=cifFileName)
        
    def test_calcualtePatternFrom3Spots(self):
        
        expSpotData={"spotXyData":[[1105,1632],[1035,1251],[1703,1510]]} 
        #expSpotData={"spotXyData":[[1965,1593],[1941,1259],[2292,1399]]} 
        
        sa=SaedAnalyzer(lattice=self.hexLat2, hklMax=3)
        sa.loadStructureFromCif(r'../../../data/structureData/Alpha-ZrP63mmc.cif')
        lat = olt.fromCif(r'../../../data/structureData/Alpha-ZrP63mmc.cif')
        spotData = [MillerPlane(lattice=lat, hkl=[0,0,0,0]),MillerPlane(lattice=lat, hkl=[0,0,0,2]),MillerPlane(lattice=lat, hkl=[2,-1,-1,0])]
        #spotData = [MillerPlane(lattice=lat, hkl=[0,0,0,0]),MillerPlane(lattice=lat, hkl=[-1,1,0,0]),MillerPlane(lattice=lat, hkl=[0,1,-1,-1])]
        
        origin=expSpotData['spotXyData'][0]
        vec1=expSpotData['spotXyData'][1]
        vec2=expSpotData['spotXyData'][2]
        saedData = sa.calcualtePatternFrom3Spots(spotData, expSpotData['spotXyData'])
        tmpXyData = [origin, vec1, vec2]
        latticePoints = gu.generate2Dlattice(origin, vec1, vec2, maxIndices=2, plotOn=False)
        
        fig=plt.gcf()
        axes = fig.add_subplot(111,)
        latticePoints=np.array(latticePoints)
        axes.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')
        #rotattedData = sa.rotateSAED(copy.deepcopy(saedData), foundAngle)
        sa.plotSAED(saedData,figHandle=fig, axisHandle=axes,markSpots=False, showAbsentSpots=False) 
        plt.show()
        exit()
        
        plane1 = MillerPlane(lattice = self.cubeLat, hkl = [0,0,0])
        plane2 = MillerPlane(lattice = self.cubeLat, hkl = [1,0,0])
        plane3 = MillerPlane(lattice = self.cubeLat, hkl = [0,1,0])
        spotData=[plane1,plane2,plane3]
        xyData = [[0,0],[10,0],[0,10]]
        sa = SaedAnalyzer(hklMax=2,lattice=self.cubeLat)
        saedData = sa.calcualtePatternFrom3Spots(spotData, xyData)
        sa.plotSAED(saedData=saedData, plotShow=True, markSpots=True)
        ### case of giving wrong ordering of plane1 and plane2 even then the programme should correct the ordering and plot it 
        plane1 = MillerPlane(lattice = self.cubeLat, hkl = [0,0,0])
        plane2 = MillerPlane(lattice = self.cubeLat, hkl = [1,1,0])
        plane3 = MillerPlane(lattice = self.cubeLat, hkl = [0,1,0])
        spotData=[plane1,plane2,plane3]
        xyData = [[0,0],[0,10],[10,10]]
        sa = SaedAnalyzer(hklMax=2,lattice=self.cubeLat)
        saedData = sa.calcualtePatternFrom3Spots(spotData, xyData)
        sa.plotSAED(saedData=saedData, plotShow=True, markSpots=True)
 
        
        
        
        exit()
#          
#         ##Test case for in pane 45 degree rotataed Pattern
#         plane1 = MillerPlane(lattice = self.cubeLat, hkl = [0,0,0])
#         plane2 = MillerPlane(lattice = self.cubeLat, hkl = [1,0,0])
#         plane3 = MillerPlane(lattice = self.cubeLat, hkl = [0,1,0])
#         spotData=[plane1,plane2,plane3]
#         xyData = [[0,0],[7.07,7.07],[-7.07,7.07]]
#         sa = SaedAnalyzer(hklMax=2,lattice=self.cubeLat)
#         saedData = sa.calcualtePatternFrom3Spots(spotData, xyData)
#         sa.plotSAED(saedData=saedData, plotShow=True, markSpots=True)  
#         
#         plane1 = MillerPlane(lattice = self.hexLat2, hkl = [0,0,0,0])
#         plane2 = MillerPlane(lattice = self.hexLat2, hkl = [0,1,-1,0])
#         plane3 = MillerPlane(lattice = self.hexLat2, hkl = [0,0,0,2])
#         spotData=[plane1,plane2,plane3]
#         xyData = [[0,0],[10,0],[0,10.09]]
#         sa = SaedAnalyzer(hklMax=2, lattice=self.hexLat2)
#         saedData = sa.calcualtePatternFrom3Spots(spotData, xyData)
#         sa.plotSAED(saedData=saedData, plotShow=True, markSpots=True)
#         
        plane1 = MillerPlane(lattice = self.hexLat2, hkl = [0,0,0,0])
        plane2 = MillerPlane(lattice = self.hexLat2, hkl = [1,0,-1,0])
        plane3 = MillerPlane(lattice = self.hexLat2, hkl = [1,-2,1,1])
        spotData=[plane1,plane2,plane3]
        xyData = [[714,584],[706,424],[1016,574]]
        sa = SaedAnalyzer(hklMax=2, lattice=self.hexLat2)
        saedData = sa.calcualtePatternFrom3Spots(spotData, xyData)
        sa.plotSAED(saedData=saedData, plotShow=True, markSpots=True)
              
    def test_solvePatternFrom3Points(self):
        ## cube 100 pattern
        expSpotData={"spotXyData":[[0,0],[10,0],[0,10.09]]} 
        hklMax=3
        D_TOLERANCE=2 
        allowedAngleDeviation=1
        sa=SaedAnalyzer(lattice=self.cubeLat, hklMax=2)
        result = sa.solvePatternFrom3Points(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation)
        print(r"Zone Axis  : AngleError DError%")
        if result is not None:
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
              
        ## hcp 1216 pattern
        expSpotData={"spotXyData":[[714,584],[706,424],[1016,574]]} 
        hklMax=3
        D_TOLERANCE=5. 
        allowedAngleDeviation=2.
        structureFile=r"D:/CurrentProjects/python_trials/work_pycrystallography/pycrystallography/data/structureData/Zr-Alpha.cif"
        lattice = olt.fromCif(structureFile)
        #lattice = self.hexLat2
        sa = SaedAnalyzer(lattice=lattice,hklMax=4)
        sa.loadStructureFromCif(structureFile)
        #sa=SaedAnalyzer(lattice=self.hexLat2, hklMax=4)
        print("lattice=",self.hexLat2)
        imName=r'D:/CurrentProjects/hydride_TEM/matrix 1216 a 1.5 b 18.7_375mm_final.tif'
        im=cv2.imread(imName)
        im.shape
        result = sa.solvePatternFrom3Points(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,spot1dRange=[1.0, 4],
                                           spot2dRange=[1.2, 4],imageData=im )
        print(r"Zone Axis  : AngleError DError%")
        if result is not None:
            for item in result:
                print(item["pattern"]["zoneAxis"], item["pattern"]["scalingFactor"], item["angleError"], item["dError"], item["Correlation"])
   

    def test_transformSaedPattern(self):
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3)
        zoneAxis = MillerDirection(lattice=self.cubeLat1,vector=[0,0,1])         
 
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
                                           patterCenter=[0.,0.], scalingFactor=100.)
        saedData1=sa.computeKikuchiForSaed(saedData1,ignoreDistantSpots=True,)
        
        fig=plt.gcf()
        axes = fig.add_subplot(111,)       
        options={"fixedAxes":True,"alphaRotationFirst":True, "activeRotation":True}
        alphaTilt,betaTilt=45,45.
        alphaAxis, betaAxis = [1.,0.,0], [0.,1.,0]
        alphaAxis, betaAxis = saedData1["xAxis"], saedData1["yAxis"]
        transformedSaed_1 = sa.transformSaedPattern(saedData1, shift=[0.,0.], alphaTilt=alphaTilt, betaTilt=betaTilt, alphaAxis=alphaAxis, betaAxis=betaAxis,
                                                  inPlaneRotation=0.,options=options)
        transformedSaed_1=sa.computeKikuchiForSaed(transformedSaed_1,ignoreDistantSpots=True,)
        alphaAxis, betaAxis = transformedSaed_1["xAxis"], transformedSaed_1["yAxis"]
        options={"fixedAxes":True,"alphaRotationFirst":True, "activeRotation":True}
        transformedSaed_2 = sa.transformSaedPattern(transformedSaed_1, shift=[0.,0.], alphaTilt=alphaTilt, betaTilt=betaTilt, alphaAxis=alphaAxis, betaAxis=betaAxis,
                                                  inPlaneRotation=0.,options=options)
        transformedSaed_2=sa.computeKikuchiForSaed(transformedSaed_2,ignoreDistantSpots=True,)
        
        options={"fixedAxes":True,"alphaRotationFirst":False, "activeRotation":True} 
        alphaAxis, betaAxis = transformedSaed_2["xAxis"], transformedSaed_2["yAxis"]
        transformedSaed_3 = sa.transformSaedPattern(transformedSaed_2, shift=[0.,0.], alphaTilt=-2.*alphaTilt, betaTilt=-2.*betaTilt, alphaAxis=alphaAxis, betaAxis=betaAxis,
                                                  inPlaneRotation=0.,options=options)
        transformedSaed_3=sa.computeKikuchiForSaed(transformedSaed_3,ignoreDistantSpots=True,)
        
        sa.plotSAED([saedData1,transformedSaed_1,transformedSaed_3 ],plotShow=True,figHandle=None,axisHandle=axes,makeTransperent=False,markSpots=True,showAbsentSpots=True,plotKikuchi=False,markKikuchi=True)
       
        
        print("Testing of transform SAED function done!!")
        
    
    
    def test_findPatternARotationToMatchExpSpots(self):
        #### case of 01-10 of Zr pattern
        expSpotData={"spotXyData":[[1105,1632],[1035,1251],[1703,1510]]} 
        expSpotData={"spotXyData":[[1965,1593],[1941,1259],[2292,1399]]} 
        
        sa=SaedAnalyzer(lattice=self.hexLat2, hklMax=3)
        sa.loadStructureFromCif(r'../../../data/structureData/Alpha-ZrP63mmc.cif')
        lat = olt.fromCif(r'../../../data/structureData/Alpha-ZrP63mmc.cif')
        zoneAxis=MillerDirection(lattice=lat,vector=[0,1,-1,0])
        zoneAxis=MillerDirection(lattice=lat,vector=[-1,-1,2,-3])
        
        
        origin=expSpotData['spotXyData'][0]
        vec1=expSpotData['spotXyData'][1]
        vec2=expSpotData['spotXyData'][2]
        saedData = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,patterCenter=origin, scalingFactor=936.0)
        spotData = [MillerPlane(lattice=lat, hkl=[0,0,0,0]),MillerPlane(lattice=lat, hkl=[0,0,0,2]),MillerPlane(lattice=lat, hkl=[2,-1,-1,0])]
        spotData = [MillerPlane(lattice=lat, hkl=[0,0,0,0]),MillerPlane(lattice=lat, hkl=[-1,1,0,0]),MillerPlane(lattice=lat, hkl=[0,1,-1,-1])]
        
        
        tmpXyData = [origin, vec1, vec2]
        foundAngle= sa.findPatternARotationToMatchExpSpots(xyData=tmpXyData,spotData=spotData,zoneAxis=zoneAxis,scalingFactor=1.0)
        print(foundAngle)
        latticePoints = gu.generate2Dlattice(origin, vec1, vec2, maxIndices=2, plotOn=False)
        fig=plt.gcf()
        axes = fig.add_subplot(111,)
        latticePoints=np.array(latticePoints)
        axes.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')
        rotattedData = sa.rotateSAED(copy.deepcopy(saedData), foundAngle)
        sa.plotSAED([saedData, rotattedData],figHandle=fig, axisHandle=axes,markSpots=False, showAbsentSpots=False) 
        plt.show()
 
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3,atomData=self.fccAtomData)
        zoneAxis = MillerDirection(lattice=self.cubeLat1,vector=[1,0,0])    
        saedData = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.fccAtomData,
                                           patterCenter=[0.,0.], scalingFactor=1.0)        
        #sa.plotSAED(saedData, plotShow=True, markSpots=True, showAbsentSpots=True)
        xyData= [[0,0],[1.0,0.0],[0,1.0]]
        spotData = [MillerPlane(lattice=self.cubeLat1, hkl=[0,0,0]),MillerPlane(lattice=self.cubeLat1, hkl=[0,0,-1]),MillerPlane(lattice=self.cubeLat1, hkl=[0,1,0])]
        saedDataOriginal = copy.deepcopy(saedData)
        data=[]
        for i in range(5):
            ang = np.random.random()*360
            #ang=10
            theta=ang*np.pi/180
            rotMat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta) ]])
            p1 = np.array(xyData[1])
            p2 = np.array(xyData[2])
            p1 = np.matmul(rotMat,p1)
            p2 = np.matmul(rotMat,p2)
            tmpXyData = [xyData[0], [p1[0],p1[1]], [p2[0],p2[1]]]
            foundAngle= sa.findPatternARotationToMatchExpSpots(xyData=tmpXyData,spotData=spotData,zoneAxis=zoneAxis,scalingFactor=1.0)
            print(ang, foundAngle)
            data.append([ang, foundAngle, ang+foundAngle])
            saedDataOriginal=sa.rotateSAED(copy.deepcopy(saedData),ang)
            rotattedData = sa.rotateSAED(copy.deepcopy(saedData), foundAngle)   
            sa.plotSAED(saedData=[saedDataOriginal, rotattedData]) 
        
        print(data)
        #self.assertAlmostEqual(foundAngle,ang,6,"Problem in pattern rotation angle fdinding")
        
        
    
    def test_planesWithinDrange(self):
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax=4)
        planeList = sa.planesWithinDrange(dMin=1.2, dMax=4.0)
        planeList = sa.planesWithinDrange(dMin=3.1, dMax=5.5)
          
        print(planeList) 
        self.assertArrayAlmostEqual(planeList[0].gethkl(),[0,0,0,-1],6,"Problem in MillerPlane within Drange")     
   
    def test_solvePatternFrom3PointsCalibrated(self):
        #### case of 01-10 of Zr pattern
        expSpotData={"spotXyData":[[1105,1632],[1035,1251],[1703,1510]]} 
        hklMax=3
        D_TOLERANCE=50 
        allowedAngleDeviation=4
        sa=SaedAnalyzer(lattice=olt.hexagonal(3.23,3.23*1.59), hklMax=3)
        spot1dRange = [1.5,4.0]
        spot2dRange = [1.5,4.0]        

        calibration={"cameraConstant":1000,"cameraLength":"40cm","machine":"2000Fx"}
        result = sa.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
                                                      spot1dRange=spot1dRange, spot2dRange=spot2dRange, calibration=calibration)
        
        print(r"Zone Axis  : AngleError DError%")
        if result is not None:
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
 
        ## bcc beta 110 real pattern 
        expSpotData={"spotXyData":[ [2126,1276],[1731,682],[1693,1238]]}
        hklMax=3
        D_TOLERANCE=50 
        allowedAngleDeviation=4
        sa=SaedAnalyzer(lattice=olt.cubic(3.60900), hklMax=3)
        calibration={"cameraConstant":1427,"cameraLength":"40cm","machine":"2000Fx"}
        result = sa.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
                                                      calibration=calibration)
        
        print(r"Zone Axis  : AngleError DError%")
        if result is not None:
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
              
    
        exit()
        
        
        ###Nb 60cm  0001 calibration pattern
        expSpotData={"spotXyData":[[2143,1222],[2811,1124],[2039,590]]} 
        hklMax=3
        D_TOLERANCE=10 
        allowedAngleDeviation=2
        sa=SaedAnalyzer(lattice=olt.cubic(3.3375), hklMax=4)
        calibration={"cameraConstant":2252.1,"cameraLength":"60cm","machine":"2000Fx"}
        result = sa.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
                                                      calibration=calibration)
        print(r"Zone Axis  : AngleError DError%")
        if result is not None:
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
              

    def test_computeKikuchiForSaed(self):
#         hcpAtomData = [(40., np.array([0.,0.,0.])),
#             (40., np.array([1./3., 2./3.,1./2.]))] ####typical Zr data
# 
#         sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 2,atomData=hcpAtomData)
#         zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[0,0,0,1])         
#  
#         saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
#                                            patterCenter=[0.,0.], scalingFactor=100.)
#         saedData1=sa.computeKikuchiForSaed(saedData1,ignoreDistantSpots=True,)
#         
#         fig=plt.gcf()
#         axes = fig.add_subplot(111,)       
#         
#         sa.plotSAED(saedData1,plotShow=True,figHandle=None,axisHandle=axes,makeTransperent=False,markSpots=True,showAbsentSpots=True,plotKikuchi=True,markKikuchi=True)
#         print("Testing of plotSAED function done!!")
#         hcpAtomData = [(40., np.array([0.,0.,0.])),
#             (40., np.array([1./3., 2./3.,1./2.]))] ####typical Zr data

        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3)
        zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[0,0,1])         
 
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
                                           patterCenter=[0.,0.], scalingFactor=100.)
        saedData1=sa.computeKikuchiForSaed(saedData1,ignoreDistantSpots=True,)
        
        fig=plt.gcf()
        axes = fig.add_subplot(111,)       
        
        sa.plotSAED(saedData1,plotShow=True,figHandle=None,axisHandle=axes,makeTransperent=False,markSpots=True,showAbsentSpots=True,plotKikuchi=True,markKikuchi=True)
        print("Testing of plotSAED function done!!")
        
    
    def test_plotSAED(self):
        
        hcpAtomData = [(40., np.array([0.,0.,0.])),
                (40., np.array([1./3., 2./3.,1./2.]))] ####typical Zr data

        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 4,atomData=hcpAtomData)
        zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[0,0,0,1])
        scalingFactor=sa._cameraConstant         
        detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])*100## all are in mm
        detectorCoordinates[:,2]=-sa._machineConditions["CameraLength"] 
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
                                           patterCenter=[0.,0.], scalingFactor=scalingFactor)
        fig=plt.gcf()
        axes = fig.add_subplot(111,)       
        
        sa.plotSAED(saedData1,plotShow=True,figHandle=None,axisHandle=axes,makeTransperent=False,markSpots=False,showAbsentSpots=False)
        print("Testing of plotSAED function done!!")
            
    
    def test_rotateSAED(self):
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 3,atomData=self.zrAtomData)
        zoneAxis = MillerDirection(lattice=self.hexLat2,vector=[0,0,0,1])         
        scalingFactor=200
        pc = [34.,4.5]         
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis, atomData=self.fccAtomData,
                                           patterCenter=pc, scalingFactor=scalingFactor)
        fig=plt.gcf()
        axes = fig.add_subplot(111,)       
        sa.plotSAED(saedData1,plotShow=True,figHandle=None,axisHandle=axes,makeTransperent=False,markSpots=True)
        rotatedData = SaedAnalyzer.rotateSAED(saedData1, rotationAngle=90)
        sa.plotSAED(rotatedData,plotShow=True,makeTransperent=False,markSpots=True)
        print("Testing of patteern Rotation function done!!")
    
    
    def test_calcualteSAEDpatternForCrystalOri(self):
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3)
        crystalOri= Orientation(euler=np.radians(np.array([15, 0, 0])),units="deg")
        saedData1 = sa.calcualteSAEDpatternForCrystalOri(crystalOri=crystalOri,patterCenter=[100,100])
        sa.plotSAED(saedData1, plotShow=True, markSpots=True)
        
        holderTiltData = {"alphaTilt":0,"betaTilt":0.,"diffractionRotationAngle":0., 
                        "options":{"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":True}}  
        patternXaxis = saedData1["xAxis"]
        zoneAxis=saedData1["zoneAxis"]
        ori1, ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis,patternXaxis,alphaTilt=0., betaTilt=0.,diffractionRotationAngle =0.)
                                                        
        print(ori1, ori2, saedData1["zoneAxis"])
    
    def test_getCrystalOriFromSaed(self):
#         sa = SaedAnalyzer(hklMax = 3)
#         cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
#         cifName =cifPathName+'Alpha-ZrP63mmc.cif'
#         cifName =cifPathName+'SiC-beta.cif'
#         sa.loadStructureFromCif(cifName)
#         lattice = olt.fromCif(cifName)
#         zoneAxis = MillerDirection(lattice=lattice,vector=[0,0,0,1])
#         holderTiltData = {"alphaTilt":9,"betaTilt":12.4,"diffractionRotationAngle":0., 
#                         "options":{"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":True}}  
#         inPlaneRotation=0.0
#         saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
#                                            patterCenter=[0.,0.], scalingFactor=100.,inPlaneRotation=inPlaneRotation, holderTiltData=holderTiltData)
#         
#         alphaTilt,betaTilt = holderTiltData["alphaTilt"], holderTiltData["betaTilt"]
#         options = {"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":False}
#         patternXaxis = saedData1["xAxis"]
#         zoneAxis=saedData1["zoneAxis"]
#         ori1, ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis,patternXaxis,alphaTilt=alphaTilt, betaTilt=betaTilt,diffractionRotationAngle =inPlaneRotation,options=options)
#                                                         
#         print(ori1, ori2, saedData1["zoneAxis"])
#         
#         exit()
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3)
        crystalOri = Orientation(euler=np.radians(np.array([90, 52, 22,])))
        saedData1 = sa.calcualteSAEDpatternForCrystalOri(crystalOri=crystalOri)
        ori1, ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis=saedData1["zoneAxis"],patternXaxis=saedData1["xAxis"],alphaTilt=0, betaTilt=0,diffractionRotationAngle =0.)
        print(crystalOri, ori1,ori2) 
        exit()
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3)
        zoneAxis = MillerDirection(lattice=self.cubeLat1,vector=[0,0,1])
        holderTiltData = {"alphaTilt":0,"betaTilt":0.,"diffractionRotationAngle":0., 
                        "options":{"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":True}}  
        inPlaneRotation=15.0
        saedData1 = sa.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxis,
                                           patterCenter=[0.,0.], scalingFactor=100.,inPlaneRotation=inPlaneRotation, holderTiltData=holderTiltData)
        
        alphaTilt,betaTilt = holderTiltData["alphaTilt"], holderTiltData["betaTilt"]
#         options = {"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":True}
#         saedData2 = sa.transformSaedPattern(saedData1, alphaTilt=alphaTilt, betaTilt=betaTilt,options=options)
        options = {"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":False}
        patternXaxis = saedData1["xAxis"]
        zoneAxis=saedData1["zoneAxis"]
        ori1, ori2 = SaedAnalyzer.getCrystalOriFromSaed(zoneAxis,patternXaxis,alphaTilt=alphaTilt, betaTilt=betaTilt,diffractionRotationAngle =0.,options=options)
                                                        
        print(ori1, ori2, saedData1["zoneAxis"])
        
    
    def test_getIdealRefFrame(self):
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 3,atomData=self.zrAtomData)        
        (patternXAxis,patternYAxis,patternZAxis) = sa._getIdealRefFrame() 
        print(patternXAxis,patternYAxis,patternZAxis )   
        
    
    def test_findTiltsForTargetZoneAxisFromSterioGrapm(self):
        sa = SaedAnalyzer(lattice=self.cubeLat,hklMax = 2,atomData=self.bccAtomData) 
        currentZoneAxis = MillerDirection(vector=[1,1,1],lattice=self.cubeLat)
        targetZoneAxis = MillerDirection(vector=[1,0,0],lattice=self.cubeLat)
        saed1 = sa.calcualteSAEDpatternForZoneAxis(currentZoneAxis,)
        sterioXaxis=saed1["xAxis"]
        alphaAxis = saed1["xAxis"].getUnitVector()
        betaAxis = saed1["yAxis"].getUnitVector()
        alphaTiltLimits,betaTiltLimts,inPlaneRotationLimits=(-90.,90.,),(-90.,90,),(0.,0.,)
        totlaAngualarDistance, alphaTilt, betaTilt,err, isSuccess, solList = sa.findTiltsForTargetZoneAxisFromSterioGram(currentZoneAxis=currentZoneAxis, targetZoneAxis=targetZoneAxis,
                                                 sterioCentredOn=currentZoneAxis, sterioXaxis=sterioXaxis,
                                                 alphaAxis = alphaAxis, betaAxis=betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimts)
        
        print(totlaAngualarDistance, alphaTilt, betaTilt,isSuccess)
        print(solList)
        
        
        
        
        sa = SaedAnalyzer(lattice=self.hexLat2,hklMax = 2,atomData=self.zrAtomData) 
        sterioCentredOn =MillerDirection(vector=[2,-1,-1,3],lattice=self.hexLat2) 
        currentZoneAxis = sterioCentredOn
        targetZoneAxis = MillerDirection(vector=[0,0,0,1],lattice=self.hexLat2)
        #sterioCentre = MillerDirection(vector=[0,0,0,1],lattice=self.hexLat2)
        saed1 = sa.calcualteSAEDpatternForZoneAxis(currentZoneAxis,inPlaneRotation=0.)
        sterioXaxis=saed1["xAxis"].getUnitVector()
        alphaAxis = saed1["xAxis"].getUnitVector()
        betaAxis = saed1["yAxis"].getUnitVector()
        alphaTiltLimits,betaTiltLimts,inPlaneRotationLimits=(-90.,90.,),(-90.,90,),(0.,0.,)
        totlaAngualarDistance, alphaTilt, betaTilt,err, isSuccess, solList = sa.findTiltsForTargetZoneAxisFromSterioGram(currentZoneAxis=currentZoneAxis, targetZoneAxis=targetZoneAxis,
                                                 sterioCentredOn=sterioCentredOn, sterioXaxis=sterioXaxis,
                                                 alphaAxis = alphaAxis, betaAxis=betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimts)
       
        print(totlaAngualarDistance, alphaTilt, betaTilt,isSuccess)
        print(solList)
    
    def test_plotSterioGraphicProjection(self):
        sa = SaedAnalyzer(lattice=self.cubeLat1,hklMax = 3,atomData=self.bccAtomData)
        zoneAxis = MillerDirection(lattice=self.cubeLat1,vector=[0,0,1])
        crystalOri = Orientation.stdOri()
        crystalOri1 = Orientation(euler=np.radians(np.array([0,0,0])))
        crystalOri2 = Orientation(euler=np.radians(np.array([90, 45, 44,])))
        
        
        dirList = [#MillerDirection(lattice=self.cubeLat1,vector=[1,5,6]),
                 MillerDirection(lattice=self.cubeLat1,vector=[1,1,0]),
                 MillerDirection(lattice=self.cubeLat1,vector=[1,1,1]),
                 MillerDirection(lattice=self.cubeLat1,vector=[1,0,0]),
                     
                    ]
        allDirList = []
        for item in dirList:
            allDirList.extend(item.symmetricSet())
        #allDirList=dirList
        dirSterioData = sa.calculateSterioGraphicProjectionDirection(dirList = allDirList, crystalOri=crystalOri1, centredOn=zoneAxis,inPlaneRotation=0.)
        dirSterioData2 =sa.calculateSterioGraphicProjectionDirection(dirList = allDirList, maxUVW=1,crystalOri=crystalOri2, centredOn=zoneAxis,inPlaneRotation=0) 
        #sa.plotSterioGraphicProjectionDirections(dirSterioData)
        fig=plt.gcf()
        axes = fig.add_subplot(111,)
        sa.plotSterioGraphicProjection([dirSterioData,dirSterioData2],  axisHandle=axes,
                                     projectToSameHemisphere=True,markPoints=[True,True])
        exit()
        sa = SaedAnalyzer(hklMax = 3)
        cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
        cifName =cifPathName+'Alpha-ZrP63mmc.cif'
        cifName =cifPathName+'SiC-beta.cif'
        crystalOri =Orientation(euler=np.radians(np.array([234.5679683,   15.28035804, 126.41173602])))
        crystalOri =Orientation(euler=np.radians(np.array([0,0,0])))
        
        sa.loadStructureFromCif(cifName)
        lattice = olt.fromCif(cifName)
        dirList = [MillerDirection(lattice=lattice,vector=[0,0,0,1]),
                        MillerDirection(lattice=lattice,vector=[0,-1,1,3]),
                   MillerDirection(lattice=lattice,vector=[0,2,-2,3]),
                   MillerDirection(lattice=lattice,vector=[-1,1,0,3]),
                   MillerDirection(lattice=lattice,vector=[-2,0,2,3]),
                    MillerDirection(lattice=lattice,vector=[1,-2,1,6]),
                    MillerDirection(lattice=lattice,vector=[-1,-1,2,3]), 
                   ]
        allDirList = []
        for item in dirList:
            allDirList.extend(item.symmetricSet())
        #allDirList=dirList
        zoneAxis = MillerDirection(lattice=lattice,vector=[2,2,-4,3])
        dirSterioData = sa.calculateSterioGraphicProjectionDirection(dirList = allDirList, crystalOri=crystalOri)
        dirSterioData2 =sa.calculateSterioGraphicProjectionDirection(dirList = None, maxUVW=1,crystalOri=crystalOri)
        fig=plt.gcf()
        axes = fig.add_subplot(111)
        sa.plotSterioGraphicProjection([dirSterioData,dirSterioData2], axisHandle=axes,
                                    projectToSameHemisphere=True,markPoints=[True,True])
         
         
        exit()
        sa = SaedAnalyzer(hklMax = 3)
        cifPathName = r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\'
        cifName =cifPathName+'SiC-beta.cif'
        
        
        sa.loadStructureFromCif(cifName)
        lattice = olt.fromCif(cifName)
        dirList = [MillerDirection(lattice=lattice,vector=[1,3,0]),
                   MillerDirection(lattice=lattice,vector=[2,1,0]),
                   MillerDirection(lattice=lattice,vector=[3,2,0]),
                   MillerDirection(lattice=lattice,vector=[3,2,1]),
                   MillerDirection(lattice=lattice,vector=[0,1,0]),
                   MillerDirection(lattice=lattice,vector=[1,1,2]),
                  
                   
                   ]
        allDirList = []
        for item in dirList:
            allDirList.extend(item.symmetricSet())
        #allDirList=dirList
        zoneAxis = MillerDirection(lattice=lattice,vector=[1,1,0])
        dirSterioData = sa.calculateSterioGraphicProjectionDirection(dirList = allDirList, crystalOri=crystalOri)
        dirSterioData2 =sa.calculateSterioGraphicProjectionDirection(dirList = None, maxUVW=1,crystalOri=crystalOri)
        fig=plt.gcf()
        axes = fig.add_subplot(111,projection='polar')
        sa.plotSterioGraphicProjection([dirSterioData,dirSterioData2], axisHandle=axes,
                                    projectToSameHemisphere=True,markPoints=[True,True])
        
        
                        
if __name__ == '__main__':
    import unittest
    unittest.main()