# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

"""
+++++++++++++++++++++++++++++++++++
Module Name: **orientationRelation.py**
+++++++++++++++++++++++++++++++++++

This module :mod:`orientation` defines the classes relating to Orientation. Essentially it wrpas the underlying Quaternion class to
make it suitable for handling the Euler angles as used in the feild of materials science.

This module contains  classes and functions listed below.

**Data classes:**
  * :py:class:`Orientation`
    
    *list of functions in this class*
   * :py:func:`Orientation.copy`
   * :py:func:`Orientation.getEulerAngles`
   * :py:func:`Orientation.get_axis`
   * :py:func:`Orientation.axis`
   * :py:func:`Orientation.random`
   * :py:func:`Orientation.misorientation`
   * :py:func:`Orientation.misorientationAngle`
   * :py:func:`Orientation.misorientationAxis`
   * :py:func:`Orientation.mean`
"""

from __future__ import division, unicode_literals


import math
import itertools
import warnings

from six.moves import map, zip

import numpy as np
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians
from pycrystallography.core.quaternion  import Quaternion
from pycrystallography.core.orientation  import Orientation  
from pycrystallography.core.orientedLattice  import OrientedLattice as Olt 
from pycrystallography.core.millerDirection  import MillerDirection 
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
from pycrystallography.core.saedAnalyzer import SaedAnalyzer as SeadAnlz
from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
import webbrowser
 
import copy

import pycrystallography.utilities.pytransformations as pt 
import pycrystallography.utilities.pymathutilityfunctions as pmt 
import pymatgen.core as mg
import logging
from tabulate import tabulate
import pandas as pd
import os, pathlib


from monty.json import MSONable
from monty.dev import deprecated



__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"

defaultPhaseData=[{"Name":'Beta', 'phaseSymbol':"$\beta$","Id":1, },
                  {"Name":'Alpha', 'phaseSymbol':"$\alpha$","Id":0, },
                  {"Name":'Gamma', 'phaseSymbol':"$\gamma$","Id":2, },
                  {"Name":'Delta', 'phaseSymbol':"$\delta$","Id":3, },
                ]



class OrientationRelation(MSONable):
    """
    A class for creating the Orienation Relation between two structures.
    """
    
    def __init__(self, names=None,symbols=None, structures=None,lattices=None,planes=None,directions=None,orienations=None,initiateVariants=False):
        """Create an crystal orientation relationship from ::
        
        """
        self._productData={}
        self._parentData={}
        
        structureData,latticeData= OrientationRelation._extractStructureAndLattice(structures,lattices)

        
        if planes is not None and directions is not None:
            if len(planes) == len(directions) and len(planes)>1:
                if isinstance(planes[0],MillerPlane) and isinstance(directions[0],MillerDirection):
                    oris=[]
                    for i, item in enumerate(zip(planes,directions)):
                        oris.append(CrysOri.fromPlaneAndDirection(plane=item[0],direction=item[1]))
                    self._productData["Orientations"]=oris[1:]
                    self._parentData["Orientations"]=oris[0]
                    
                else:
                    raise TypeError("Only Miller Planes and Directions Objects must be specified")
            else:
                raise ValueError("Unequal number of Planes and Directions are supplied. Ensure that same number of the Planes and Directions are input")
                                               
        elif orienations is not None :
            
            if len(orienations)>1 and isinstance(orienations[0],CrysOri):
                self._parentData["Orientations"]   = orienations[0]                
                self._productData["Orientations"] = orienations[1:]
            elif len(orienations)>1 and isinstance(orienations[0],Orientation):
                self._parentData["Orientations"] =CrysOri(orientation=orienations[0],lattice=latticeData[0])
                self._productData["Orientations"] =[]
                for i, item in enumerate(orienations[1:]):
                    self._productData["Orientations"].append(CrysOri(orientation=orienations[0],lattice=latticeData[i+1]))
            else: ### case of orientation being specified in terms of euler angles
                ori = Orientation(euler=np.array(orienations[0])*np.pi/180)
                self._parentData["Orientations"] =CrysOri(orientation=ori,lattice=latticeData[0])
                self._productData["Orientations"] =[]
                for i, item in enumerate(orienations[1:]):
                    ori = Orientation(euler=np.array(item)*np.pi/180)
                    self._productData["Orientations"].append(CrysOri(orientation=ori,lattice=latticeData[i+1]))
        else:
            raise ValueError("Either the orientations or the Plane/Direction combinatiosn must be specified for creating the Oreiantion Relation Object !!!!")
        
            
        self._diffractionInitialized=False       
        self._parentData["Structure"]= structureData[0]
        self._productData["Structure"]= structureData[1:]
        self._parentData["Lattice"]= latticeData[0]
        self._productData["Lattice"]= latticeData[1:]
        self._transformationData={}
        self._transOpertorsConstructed=False
        if initiateVariants:
            self.calcualteVariants()       
        
        self._numberOfProducts = len(structureData[1:])
        if self._numberOfProducts>2:
            logging.info(f"Case of more than 2 products from the single parent detected")
            self._moreThan1Products=True
        else:
            self._moreThan1Products=False

        
        if names is None:
            self._parentData["Name"]=defaultPhaseData[0]["Name"]
            self._parentData["Symbol"]=defaultPhaseData[0]["phaseSymbol"]
            self._parentData["Id"]=defaultPhaseData[0]["Id"]
            
            for key in defaultPhaseData[0]:
                tmpList = [item[key] for item in defaultPhaseData[1:]]
                self._productData[key]=tmpList
            
        else:
            self._parentData["Name"] = names[0]
            self._productData["Name"] = names[1:]
            #self._productData["Name"] = names[1]
            #warnings.warn("At the moment the name of the product is assgined as the 2nd element of the names list. Probelm wil come when more than 2 products are there!!!!")


        if symbols is None:
            self._parentData["Symbol"] = defaultPhaseData[0]["phaseSymbol"]
            for key in defaultPhaseData[0]:
                tmpList = [item[key] for item in defaultPhaseData[1:]]
                self._productData[key] = tmpList

        else:
            self._parentData["Symbol"] = symbols[0]
            self._productData["Symbol"] = symbols[1:]
            warnings.warn(
                "At the moment the name of the product is assgined as the 2nd element of the names list. Probelm wil come when more than 2 products are there!!!!")

    def __str__(self):
        
        transData=self._transformationData
        str1=''
        str1+=f"Parent : {self._parentData['Name']}--> Product : {self._productData['Name']}\n============================================\n"
        str1+="Parent Info:\n"+str(self._parentData["Structure"])
        str1+="\n Products Info\n============\n\n"
        for i in range(self._numberOfProducts):
            str1+="\nProductId = "+str(i)+"\n"+str(self._productData["Structure"])

        str1+="\nData of Transformation Operators"
        for key in transData:
            str1+="\nkey= " +str(key)
            if hasattr(transData[key],'__iter__'):
                for item in transData[key] :
                    if isinstance(item,list):
                        for ii,i in enumerate(item) :
                            if isinstance(i,CrysOri):
                                str1+="\n Variant ID: {:2d} :  {:planeDir} {:euler} {:axisAngle}".format(ii,i,i,i)
                    elif isinstance(item,CrysOri):
                        #print("here you go !!!! \n{:planeDir} {:euler} {:axisAngle}".format(item,item,item))
                        str1+="\n{:planeDir} {:euler} {:axisAngle}".format(item,item,item)
                    else:
                        #print("Hereis the non Crys Ori type", item, type(item))
                        str1+=str(item)
            else:
                str1+=str(transData[key])

        return str1

    @property
    def parentStructure(self):
        return self._parentData["Structure"]

    def productStructure(self):
        return self._productData["Structure"]
    
    @property
    def parentLattice(self):
        """
        lattice of the parent in the OR object
        """
        return self._parentData["Lattice"]
    
    @property
    def productLattice(self):
        """
        lattice of the parent in the OR object
        """
        return self._productData["Lattice"] 
    
    def initializeDiffraction(self):
        """
        method to initialize the diffraction esp (SAED patterns) realted stuff.
        """
        transformationData = self.calcualteVariants()
        parData = self._parentData
        prodData = self._productData
        self._parentData["Saed"] =SeadAnlz(name=self._parentData["Name"],symbol=self._parentData["Symbol"],
                                           considerDoubleDiffraction=True)
        self._parentData["Saed"].loadStructureFromCif(parData["Structure"])
        
        self._productData["Saed"]=[]
        for i in range(self._numberOfProducts):
            self._productData["Saed"].append(SeadAnlz(name=self._productData["Name"][i],
                                                      symbol=self._productData["Symbol"][i],considerDoubleDiffraction=True))
            self._productData["Saed"][i].loadStructureFromCif(prodData["Structure"][i])
        self._diffractionInitialized=True
        
        print ("Completed the initialization of SAED for parent and All products")
    
    def makeReport(self,format="html",savetoFile=False):
        if not self._transOpertorsConstructed:
            self.calcualteVariants()

        productVariantsList = self._transformationData["variantSet"]
        variantInfoList = []
        for i, product in enumerate(productVariantsList):
            for j, varaint in enumerate(product):
                print("Variant {:2d} Orientation (plane)[dir] :  {:simplePlaneDir} Axis@angle : {:axisAngle} EulerAngles : {:euler}".format(
                        j + 1, varaint, varaint, varaint))
                variantInfoList.append({"VarinatId":j+1, "VariantLabel":self._productData["Symbol"][i]+f"$_{j+1}$",
                                        "Ori_planeDir":"{:simplePlaneDir}".format(varaint),
                                        "Ori_AxisAngle":"{:axisAngle}".format(varaint),
                                        "Ori_Euler":"{:euler}".format(varaint),
                                        })
        variantsInfoDf = pd.json_normalize(variantInfoList)
        if "html" in format:
            text= variantsInfoDf.to_html()
            fileformat = ".html"
        elif "texttable" in format.lower():
            text = tabulate(variantsInfoDf, headers='keys', tablefmt='plain',showindex=False)
            fileformat = ".txt"

        else:
            raise ValueError (f"Unknown format only [html, texttable] are supported as of now bu was provided {format}")

        if savetoFile:
            outFile = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'tmp',
                                              'variantsInfo'+fileformat)
            with open(outFile, 'w') as f:
                f.write(text)
        return text








    def makeORString(self, format="string"):
        """
        :return: outStr --> string repre of OR for nice printing!!!
        """
        p,d = self._parentData['Orientations'].getMillerNotation()

        header = [item for item in self._productData["Symbol"] ]
        header.insert(0, self._parentData["Symbol"])
        orList=[]
        #orList.append(header)
        orList.append(["{:d}".format(p), "{:int}".format(d)])
        for item in self._productData["Orientations"]:
            p,d = item.getMillerNotation()
            orList.append(["{:d}".format(p), "{:int}".format(d)])

        orListTransposed = list(map(list, itertools.zip_longest(*orList, fillvalue=None)))
        outStr = tabulate(orListTransposed,header)
        latexStr = ""
        for i, item in enumerate(orListTransposed):
            tmp = ""
            for j, item2 in enumerate(item):
                tmp+=item2+"_"+header[j]+"||"
            tmp = tmp[:-2]+"\n" ### to remove last || sign

            latexStr+= tmp
        latexStr = latexStr.replace("$","").replace("\n", ":")
        latexStr = "$"+latexStr[:-1]+"$"  ## putting only two dollor signs and also removing the last \n

        if "string" in format:
            return outStr
        elif "latex" in format:
            return latexStr
        else:
            raise ValueError ("Orientation Relation Error: Unknown format : only string and latex are supported")
            return None

    def calculateCompositeSAEDBasedOnProductZoneAxis(self, productZoneAxis=None, *args,**kwargs):
        """

        :param productZoneAxis: desired Zobe Axis object of the product phase
        :param args:
        :param kwargs:
        :return: list of patterns (SAED patterns) list object suitable for plotting.
        """
        #productZoneAxis = MillerDirection(vector=zoneAxisProduct, lattice=latProduct)
        if not isinstance(productZoneAxis, MillerDirection):
            raise TypeError (f"Zone axis must be a milller direction object but {type(productZoneAxis)} was supplied ")
        productLattices = self.productLattice
        productId = None
        for i, lattice in enumerate(productLattices):
            if lattice==productZoneAxis.lattice:
                productId = i
        if productId is None:
            raise ValueError (f"no product lattice is matching with the lattice of the supplied product zoneAxis "
                              f"for saed creation: zoneAxis supplied is : {productZoneAxis}")


        parentZoneAxis = self.findParlallelDirectionsInParent(productZoneAxis,productId=productId)[0] ### only first one is sufficient
        patterns = self.calculateCompositeSAED(parentZoneAxis=parentZoneAxis,*args,**kwargs)
        return patterns


    def calculateCompositeSAED(self, parentZoneAxis= None, productId=0, alphaTilt=0, betaTilt=0, inPlaneRotation=0., pc=[0., 0],
                               sf=1., Tol=5):
        """
        calculate the SAED patterns for the parent and specific Product Variants
        """
        patterns=[]
        if not self._diffractionInitialized:
            self.initializeDiffraction()
        transformationData=self.calcualteVariants()
        #parentOri = self._parentData["Orientations"]

        saed1 = self._parentData["Saed"]
        saed2 = self._productData["Saed"][productId]
        if parentZoneAxis is None:
            parentZoneAxis = MillerDirection(vector=[0,0,1],lattice=self.parentLattice)


        parrentCurrentZone = MillerDirection(vector=[0,0,1],lattice=self.parentLattice)
        parentTargetZone = parentZoneAxis

        rotationRequired = Orientation.mapVector(sourceVector=parrentCurrentZone.getCartesianVec(),
                                                 targetVector=parentTargetZone.getCartesianVec())

        parentFinalOri = rotationRequired.inverse

        productSaeds=[]

        for variantId in range(len(transformationData["variantSet"][productId])):
            productOri = transformationData["variantSet"][productId][variantId]
            productFinalOri = rotationRequired.inverse*productOri.inverse
            saedData2=saed2.calcualteSAEDpatternForCrystalOri(crystalOri=productFinalOri,patterCenter=pc,
                                                              scalingFactor=sf,SAED_ANGLE_TOLERANCE=Tol)
            if np.abs(inPlaneRotation)>0.0:
                saedData2 = SeadAnlz.rotateSAED(saedData2,inPlaneRotation)

            saedData2["SaedSymbol"] = saedData2["SaedSymbol"][:-1]+f"_{{{variantId+1}}}$"
            saedData2["SaedLabel"] = saedData2["SaedSymbol"]
            productSaeds.append(saedData2)

        patterns.extend(productSaeds)

        saedData1 = saed1.calcualteSAEDpatternForCrystalOri(crystalOri=parentFinalOri,patterCenter=pc,
                                                            scalingFactor=sf,SAED_ANGLE_TOLERANCE=Tol)
        if np.abs(inPlaneRotation) > 0.0:
            saedData1 = SeadAnlz.rotateSAED(saedData1, inPlaneRotation)
        patterns.insert(0,saedData1)
        print(f"A total of {len(patterns)} were computed the Parent pattern id is : {len(patterns)-1}")




        return patterns
        
    
    def plotSaed(self, saedData,*args,**kwargs):
        """
        saed Data should be a list of several saeds i.e parent and product variants
        """
        if isinstance(saedData,list):
            figtitle = self.makeORString(format="latex")
            fig = crysFig(saedData,title=figtitle, *args,**kwargs)
            fig.plot(plotShow=True)
        else:
            raise ValueError(f"Plotting Error in OrLn class: List is expected for the plotting. But {type(saedData)} is provided!!!")
        
        
    
    def calcualteVariants(self):
        """
        construts the trasformation operator (Orientation) which transforms the Orientation of the parent into the 
        product
        """
        if not self._transOpertorsConstructed:
            transOperators=[]
            fundamentalTransOperators=[]
            variantSet=[]
            
            inverseTransOperator=[]
            fundamentalInverseTransOperator=[]
            parentVariantSet =[]
            
            print("Constructing the trnasformation operators : Please wait:::")
            for product in self._productData["Orientations"]:
                misOri = self._parentData["Orientations"].misorientation(product)
                misOriFundamental = misOri.projectTofundamentalZone()             
                
                inverseMisOri=product.misorientation(self._parentData["Orientations"]) 
                inverseMisOriFundamental =  inverseMisOri.projectTofundamentalZone()              
                
                parentSymmetricSet = self._parentData["Orientations"].symmetricSet()
                tmp=[]
                for item in parentSymmetricSet:
                    tmp.append(item.misorientation(product))                    
                    
                misOriSet = tmp
                tmp=[]
                productSymmetricSet = product.symmetricSet()
                
                for item in productSymmetricSet:
                    tmp.append(item.misorientation(self._parentData["Orientations"]))                   
                    
                inverseMisOriSet = tmp
                
                #inverseMisOriSet = product.symmetricMisorientations(self._parentData["Orientations"])                 
                transOperators.append(misOri)
                fundamentalTransOperators.append(misOriFundamental)
                variantSet.append(CrysOri.uniqueList(misOriSet,tol=1*np.pi/180))
                parentVariantSet.append(CrysOri.uniqueList(inverseMisOriSet,tol = 1*np.pi/180))
                inverseTransOperator.append(inverseMisOri)
                fundamentalInverseTransOperator.append(inverseMisOriFundamental)
                
            print("Completed the construction of trnasformation operators!!!!")
            self._transformationData["transOperator"]=transOperators
            self._transformationData["variantSet"]=variantSet
            #self._transformationData["numberOfVaiants"] = len(variantSet)
            self._transformationData["inverseTransOperator"]=inverseTransOperator
            self._transformationData["parentVariantSet"] = parentVariantSet
            self._transformationData["fundamentalTransOperators"]=fundamentalTransOperators
            self._transformationData["fundamentalInverseTransOperator"]=fundamentalInverseTransOperator
            self._transOpertorsConstructed=True
            
        
        return self._transformationData

    def getVariants(self):
        if not self._transOpertorsConstructed:
            self.calcualteVariants()
        return self._transformationData["variantSet"]


    def findParlallelDirectionsInParent(self,direction,productId = 0, considerAllVariants=True):
        """

        :param direction: Miller direction
        :param productId : integer specifying the productId (defaults to 0). In case of single parent with more
                than 1 product phases, it needs to be specified which phase is being considered fo the calculation
        :param considerAllVariants: defaults True
        :return: list of all the parllel directions
        """

        if isinstance(direction, MillerDirection):
            directionLocal = copy.copy(direction)
            if not self._transOpertorsConstructed:
                self.calcualteVariants()
            dirList = []
            #productLattice = self.productLattice[productId] ### currently we are assigning the productId (assuming more than 1 product exists)
            parentLattice = self.parentLattice ### currently we are assigning the productId (assuming more than 1 product exists)
            warnings.warn(f"Currently returns the parallel directions in parent for only one product : {productId}."
                          f"Phase name : {self._parentData['Name'][productId]} "
                          f"Call this method again with anothr ProductId if you want for another product")
            if considerAllVariants:
                # tmpList=[]
                misOriSet = self._transformationData["parentVariantSet"][productId]
                for ori in misOriSet:
                    directionLocal = copy.copy(direction)
                    tmp = directionLocal.getDirectionInOtherLatticeFrame(parentLattice)
                    # print("Before rotation {:2d}",tmp)
                    tmp.rotate(ori)
                    # print("After rotation {:2d}",tmp)
                    # tmpList.append(tmp)
                    dirList.append(tmp)

            else:  ### case of single variant
                misOri = self._transformationData["parentVariantSet"][0][0]

                tmp = directionLocal.getDirectionInOtherLatticeFrame(parentLattice)
                tmp.rotate(misOri)  #### here inverse is to ensure  passive rotation
                dirList.append(tmp)
            return dirList
        else:
            raise TypeError("Only Miller Direction Object can be sent to this method")



    def findParlallelDirectionsInProduct(self,direction,considerAllVariants=True):
        """

        :param direction: Miller direction
        :param considerAllVariants: defaults True
        :return: list of all the parllel directions
        """

        if isinstance(direction, MillerDirection):
            directionLocal = copy.copy(direction)
            if not self._transOpertorsConstructed:
                self.calcualteVariants()
            dirList = []
            for i in range(self._numberOfProducts):
                productLattice = self.productLattice[i]
                if considerAllVariants:
                    # tmpList=[]
                    misOriSet = self._transformationData["variantSet"][i]
                    for ori in misOriSet:
                        directionLocal = copy.copy(direction)
                        tmp = directionLocal.getDirectionInOtherLatticeFrame(productLattice)
                        # print("Before rotation {:2d}",tmp)
                        tmp.rotate(ori)
                        # print("After rotation {:2d}",tmp)
                        # tmpList.append(tmp)
                        dirList.append(tmp)

                else:  ### case of single variant
                    misOri = self._transformationData["variantSet"][0][0]

                    tmp = directionLocal.getDirectionInOtherLatticeFrame(productLattice)
                    tmp.rotate(misOri)  #### here inverse is to ensure  passive rotation
                    dirList.append(tmp)
            return dirList
        else:
            raise TypeError("Only Miller Direction Object can be sent to this method")

    def findParallelPlaneInProduct(self,plane,considerAllVariants=True):
        """
        find the planes of product phase that are parllel to a given parent plane
        """
        if isinstance(plane,MillerPlane):

            planeLocal = copy.copy(plane)
            if not self._transOpertorsConstructed:
                self.calcualteVariants()
            planeList = []
            for i in range(self._numberOfProducts):
                productLattice = self.productLattice[i]
                if considerAllVariants:
                    #tmpList=[]
                    misOriSet = self._transformationData["variantSet"][i]
                    for ori in misOriSet:
                        planeLocal = copy.copy(plane)
                        tmp =planeLocal.getPlaneInOtherLatticeFrame(productLattice)
                        #print("Before rotation {:2d}",tmp)
                        tmp.rotate(ori)
                        #print("After rotation {:2d}",tmp)
                        #tmpList.append(tmp)
                        planeList.append(tmp)

                else: ### case of single variant
                    misOri = self._transformationData["variantSet"][0][0]

                    tmp = planeLocal.getPlaneInOtherLatticeFrame(productLattice)
                    tmp.rotate(misOri)#### here inverse is to ensure  passive rotation
                    planeList.append(tmp)
            return planeList
        else:
            raise TypeError("Only Miller Plane Object can be sent to this method")
        
     
    @staticmethod 
    def getStructureFromCif(cifFileName):
        """
        """        
        structure = mg.Structure.from_file(cifFileName)
        lattice = Olt(matrix = structure.lattice.matrix)
        
        return structure,lattice
    @staticmethod
    def fromAxisAnglePair(structures,lattices=None,axisAnglePairs=None):
        """
        creates the OrienationRelation Object from the data of axisAngle Pair notation
        Note that len of structures, lattices is N+1 where N is the number of products
        +1 is because of the Parent.
        Hence len(axisAnglePairs) is N.
        axisAnglePairs = tuple of axis and angle pair such as (45, [0,0,1])
        axis must be specified in the ref frame of product
        """
        
        structureData, latticeData =OrientationRelation._extractStructureAndLattice(structures,lattices)
        
        operators = []
        for k in range(2):
            if k==0:
                flag = 1 ## flag is just to flip the mis orienation angle to negitive value
            else:
                flag = -1
        
            p1 = MillerPlane(hkl=[0,0,1],isCartesian=True, recLattice=latticeData[0].reciprocal_lattice_crystallographic,lattice=latticeData[0])
            planes=[p1]
            d1 = MillerDirection(vector=[1,0,0],isCartesian=True, lattice=latticeData[0])
            directions=[d1]
            ori = CrysOri.fromPlaneAndDirection(planes[0], directions[0])
            
            print("here is the Ori info of parent : {:planeDir}".format(ori))
            if axisAnglePairs is None:
                axisAnglePairs = [(0,[0,0,1])]*(len(structureData)-1) 
                print(axisAnglePairs)
            
            for i, item in enumerate(axisAnglePairs):
                    print("flag=", flag)
                    axis = MillerDirection(vector = item[1], lattice =latticeData[i+1]).getUnitVector()            
                    ori = Orientation(axis=axis,degrees=flag*item[0])
                    misOri = CrysOri(orientation=ori, lattice=latticeData[i+1])
                    p,d = misOri.planeAndDirection
                    planes.append(p)
                    directions.append(d)
                
            operators.append(OrientationRelation(structures=structureData,lattices =latticeData, 
                                                     planes=planes,directions=directions,initiateVariants=True))        
            
        assert len(operators)==2, "Some thing is wrong 2 sets of operators should have been created but "+str(len(operators))+" were created"
        
        for i in range(operators[0]._numberOfProducts):
            print("first set",operators[0]._transformationData["variantSet"][i])
            print("second set", operators[1]._transformationData["variantSet"][i])            
            operators[0]._transformationData["variantSet"][i]+=operators[1]._transformationData["variantSet"][i]
            operators[0]._transformationData["parentVariantSet"][i]+=operators[1]._transformationData["parentVariantSet"][i]
        print ("merging of the data is complete !!!")    
        
        return operators[0]
        
    
    @staticmethod
    def _extractStructureAndLattice(structures,lattices):
        """
        helper method to extract the lattices and Structures from either cif files or structure objects
        """
        
        addLatticeFromStructure=True
        structureData=[]
        latticeData=[]
        if lattices is not None and len(lattices)>1:
            if isinstance(lattices[0],Olt): 
                latticeData= lattices
                addLatticeFromStructure=False        

        if isinstance(structures,list):
            for item in structures:              
                if isinstance(item,str):  
                    if '.cif' in item.lower():
                        st,lattice = OrientationRelation.getStructureFromCif(item)
                        structureData.append(st)
                        if addLatticeFromStructure:
                            latticeData.append(lattice)
                elif isinstance(item,mg.Structure):
                    structureData.append(item)
                    if addLatticeFromStructure:
                        latticeData.append(Olt(matrix = item.lattice.matrix))
         
        return structureData, latticeData 
        