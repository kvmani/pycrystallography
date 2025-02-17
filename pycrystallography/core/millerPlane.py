# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

"""
+++++++++++++++++++++++++++++++++++
Module Name: **millerPlane.py**
+++++++++++++++++++++++++++++++++++

This module :mod:`millerPlane` defines the class Miller Plane. 
This is derived from :mod:`MillerDirection` class

This module contains classes.

**Data classes:**
 * :py:class:`MillerPlane`

    *List of functions correponding to this class are*
     * :py:func:`MillerPlane.gethkl`
     * :py:func:`MillerPlane.getPointInPlane`
     * :py:func:`MillerPlane.dspacing`
     * :py:func:`MillerPlane.getInterplanarSpacing`
     * :py:func:`MillerPlane.rotate`
     * :py:func:`MillerPlane.getPlaneInOtherLatticeFrame`
     * :py:func:`MillerPlane.getOrthoset`
     * :py:func:`MillerPlane.getPerpendicualarPlane`
     * :py:func:`MillerPlane.fromMillerToBravais`
     * :py:func:`MillerPlane.getPlaneNormal`
     * :py:func:`MillerPlane.getZoneAxis`
     * :py:func:`MillerPlane.getLatexString`
     * :py:func:`MillerPlane.structureFactor`
     * :py:func:`MillerPlane.diffractionIntensity`
     * :py:func:`MillerPlane.kikuchiLinePoints`
     
     
"""

from __future__ import division, unicode_literals
import math
import itertools
import warnings

from six.moves import map, zip
import copy

import numpy as np
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians
from pycrystallography.core.quaternion  import Quaternion  
#import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pymathutilityfunctions as pmt
#from pymatgen.core.lattice import Lattice as lt
from pycrystallography.core.orientedLattice import OrientedLattice as olt

from monty.json import MSONable
from monty.dev import deprecated

#from vtk import vtkVector3d
from math import sqrt
from numpy import pi, dot, transpose, radians
from pymatgen.core import Lattice
from pycrystallography.core.orientation import Orientation
from pycrystallography.core.millerDirection import MillerDirection
from sympy import Point, Point3D, Line, Line3D, Plane
# from sklearn.tests.test_random_projection import n_nonzeros
# from dask.array.ufunc import imag
#from sympy import *
from sympy.geometry import *

__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"

class MillerPlane(MillerDirection, MSONable):
    """
    Class for representing the crystal planes
    Primarily the plane is represented as MillerDirection which represents the normal to the plane and the point 
    through which the crystal plane passes
    This Class inherits all the properties of class MillerDirection
    
    Parameters
    ----------
    lattice: Define a lattice frame
    vector: Input the miller direction vector 
    isCartesian: [optional] Mention weather vector is in cartesian frame (default: false)
    MillerConv: Mention the direction is Miller or Bravais notation
    
    Returns
    -------
    out: object of class MillerPlane
    
    Examples
    --------
    >>> hexLat = Lattice.hexagonal(1, 1.63)
    >>> vec = [2.,-1.,-1.,0.]
    >>> dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv='MillerBravais')
    >>> plane1 = MillerPlane(lattice = dir1.lattice,hkl=dir1.vector,MillerConv='MillerBravais')
    >>> isinstance(plane1, MillerPlane)
    True
    """
    def __init__(self, lattice=olt.cubic(1), hkl = [0,0,0],isCartesian=False, MillerConv = 'Miller',
                 recLattice=None,point=None):
        
        
        self._madeFromCartesian=False
        if  len(hkl) == 4 and pt.isHexagonal(lattice):
            vector = pt.validate_number_sequence(hkl, 4)
            if np.abs(vector[0]+vector[1]+vector[2])>1e-10 :
                raise ValueError('h+k=-i not satisfied be careful in inputting values')
            h = vector[0]
            k = vector[1]
            l = vector[3]
            hkl = np.array([h,k,l])
            
              
        elif len(hkl)==3 and "Bravais" in MillerConv:
            vector = pt.validate_number_sequence(hkl, 3)            
            h = vector[0]
            k = vector[1]
            l = vector[2]
            hkl = np.array([h,k,l])
                        
            
        elif len(hkl) == 3:        
            vector = pt.validate_number_sequence(hkl, 3)
            hkl = np.array(vector)
            
        else:
            raise ValueError("In correct specification of the MillerPlane if hexagonal specify 4 indices using key word Bravais")
                        
        self.realLattice= lattice
        
        if isCartesian and recLattice is not None:  #### Note that when creating the plane using this option i.e. isCarteisn = True, 
            ##### it is assumed that the reciprocal lattice vector in cartesian frame is being sent as the hkl, further one must
            ##### input the recLattice specifically for this option.
            tempLattice=olt(recLattice.matrix)
            #hkl = tempLattice.get_fractional_coords(hkl)
            self.realLattice= recLattice.reciprocal_lattice_crystallographic
            MillerDirection.__init__(self, vector=hkl, isCartesian=True,  lattice=tempLattice)
            self._madeFromCartesian=True
            #raise ValueError("Please use the staticmethod fromNormalAndPoint !!!!")
        #self.hkl = hkl
        if recLattice is None:
            MillerDirection.__init__(self, vector=hkl, lattice=olt(lattice.reciprocal_lattice_crystallographic.matrix,
                                                               pointgroup=lattice._pointgroup))
        else:
            MillerDirection.__init__(self, vector=hkl, lattice=olt(recLattice.matrix, pointgroup=lattice._pointgroup))
            
           
            
        self._isReciprocalVector=True
        self._symmetricSet = None ## inititialized lazily
        self._symVecsCartesian = None
        #self.vector = hkl
        #self.lattice = olt(lattice.reciprocal_lattice_crystallographic.matrix)        
         
            
    
        
       
    def getPointInPlane(self):
        """
        Returns
        -------
        out : A numpy array of size (3,)
        
        Examples
        --------
        >>> plane2 = MillerPlane(lattice=Lattice.cubic(1), hkl = [1,1,1],isCartesian=True, MillerConv = 'Miller')
        >>> np.allclose(np.dot(plane2.hkl,plane2.getPointInPlane),0.00,1.0e-3)
        True
        """        
        
        hkl=np.round(self.vector,4)
        indices = np.where(np.abs(hkl)>1e-4)
        point = np.zeros((3,))
        point[indices[0][0]]=1/hkl[indices[0][0]]
#         assert np.abs(np.dot(point,self.vector)-1)<1e-4, "Point does not lie in Plane, Point is "+str(point)+"Plane is "+str(self.vector)+  \
#                 "and the dot product is "+str(np.dot(point,self.vector))
        return point        

#     def __deepcopy__(self):
#         return self.__class__(self)
    
    
    def __hash__(self):
        return hash(repr(self))
    
    # Representation
    def __str__(self):
        """An informal, nicely printable string representation of the MillerPlane object.
        """ 

        vector = self.gethkl()
        vector = np.round(np.array(vector),2).astype(int)
        #print(vector)
        #vector = self.getUVW()
        #AngleError = 0
        Marker=''   
        #print(vector)         
        if self.lattice.is_hexagonal() :           
            return "({:3d}  {:3d} {:3d} {:3d} ){:1.1}".format(vector[0], vector[1], vector[2],vector[3],Marker)
            #return "[{:15.10f}  {:15.10f} {:15.10f} {:15.10f} ]".format(vector[0], vector[1], vector[2],vector[3])  
        
        return "({:3d}  {:3d} {:3d}  ){:1.1}".format(vector[0], vector[1], vector[2],Marker)
    

    def __repr__(self):
        """The 'official' string representation of the MillerPlane object.  

        This is a string representation of a valid Python expression that could be used
        to recreate an object with the same value (given an appropriate environment)
        """
        str = self.__format__(formatstr='.2f')
        #return "MillerPlane([  ,   ,  ] ,  )".format(s), repr(self.vector[1]), repr(self.vector[2]), repr(self.lattice))
        return str

    def __format__(self, formatstr=''):
        """Inserts a customisable, nicely printable string representation of the MillerPlane object

        The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types. 
        Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
        """
        inputFormatStr=formatstr
        
        vec = self.gethkl()
        if formatstr.strip() == '' : 
            # Defualt behaviour mirrors self.__str__()
            if self._madeFromCartesian:
                vec= pmt.integerize(self.gethkl())
                formatstr = '2d' 
            else:            
                formatstr = '2d' #### assuming the MillerIndices of plane are integers
                vec = np.round(self.vector,0).astype(int)
        
        elif len(formatstr)>0 and (not "d"  in formatstr.lower()):     
            formatstr = '.2f'
            #vec = self.vector
        elif "d"  in formatstr.lower():
            formatstr = '2d' #### assuming the MillerIndices of plane are integers
            
            if self._madeFromCartesian:
                vec = pmt.integerize(self.gethkl()) 
            else:
                vec = np.round(vec).astype(int)
        else:
            raise ValueError("None of the known options to the format function are sent")
            
            
        if self.lattice.is_hexagonal() and len(vec)==4:
            string = \
                "({:" + formatstr +"} "  + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"})"
#             vector = np.array(self.fromMillerToBravais()).astype(int)
#             if formatstr.strip() == '': 
#                 vector = np.round(vector,0).astype(int)            
            str1 = string.format(vec[0], vec[1], vec[2], vec[3])
        else :     

            string = \
                "({:" + formatstr +"} "  + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"})"       
            str1 = string.format(vec[0], vec[1], vec[2])                   
            
        if inputFormatStr == '' or 'd' in inputFormatStr : 
            return str1 ## if no format string is specified we retun only miller indices       
        
        #str2 = format(self.realLattice,'.1fp')        
        str2=self.lattice._latticeType
        return str1+"  lattice = "+str2
                                        
    
    def __lt__(self,other):
        return self.dspacing<other.dspacing
    
    def gethkl(self,force3Index=False):
        """   
        Method to get the hkl of MillerPlane in Lattice Frame. 
        In most of dialy usage this is the form we employ.
        if the lattice to which this Plane belongs is
        of hexagonal returns a numpy array of (h k i l) 
        
        Returns
        -------
        res: A numpy array of size (4,) or (3,) dpending on if it hexagonal or not
                  
        """
        if not force3Index and self.lattice.is_hexagonal() :        
            hkl = self.fromMillerToBravais()        
            vector = np.array(hkl)
            return vector
        else:
            return self.vector
        
    @property
    def maxHkl(self):
        """
        Return the Max abs Hkl of the plane. for a plane {-2 1 0} returns :2
        for a plane {1 1 1} retuns 1
        
        """
        return np.abs(self.gethkl()).max()
        
    @property
    def dspacing(self):
        """
        Returns the magnitude or d- spacing of the of the MillerPlane
        
        Examples
        --------
        >>> cubeLat = Lattice.cubic(2)
        >>> plane1 = MillerPlane(lattice = cubeLat,hkl = [1,1,0])
        >>> np.allclose(plane1.dspacing,1.414,1.0e-3)
        True
        
        >>> hexLat = Lattice.hexagonal(1, 1.59)
        >>> plane1 = MillerPlane(lattice = hexLat,hkl = [0,0,0,1])
        >>> np.allclose(plane1.dspacing,1.59,1.0e-3)
        True
        """
        with np.errstate(divide='ignore'):
            return (1/np.linalg.norm(self.getCartesianVec()))
    
    @property
    def hkl(self):
        """
        Returns the magnitude of the MillerPlane
        """
        return self.vector
    
    def getInterplanarSpacing(self):
        """
        Returns the magnitude of the MillerPlane Object
        """
        return self.dspacing
     
    def rotate(self, rotation):
        """
        Rotate the MillerPlane.
        
        Parameters
        ----------
        rotation : An :py:class:`~orientation` object.
             Also it can refer to a function :py:func:`~angle`
              
        """
        if isinstance(rotation, Orientation):
            rotatedNormal = super().rotate(rotation)
            newPoint = self.getPointInPlane()
            self.vector = self.getUVW(force3Index=True)
            self.point = newPoint          
            
            
        else:
            raise TypeError("MillerPlane can be rotated only if an Orienation object is provided")
      
    
    def getPlaneInOtherLatticeFrame(self,otherLattice):
        """
        retuturns equivalent MillerPlane in lattice frame of other Lattice
        """ 
        tmp =self.getCartesianVec()
        
        return MillerPlane(hkl = tmp, isCartesian=True, 
                                      lattice=otherLattice, recLattice=otherLattice.reciprocal_lattice_crystallographic)
                    
        
        
                
    def getOrthoset(self):
        """
        return a list of 3 MillerPlanes mutually perpendicualar, making the current one as the first of the three
        """ 
        raise NotImplementedError()
               
    def getPerpendicualarPlane(self):
        """
        returns an arbitary  perendicular MillerPlane to current MillerPlane object.
        
        Parameters
        ----------
        returnCartesian : [optional] if true the returned vector is numpy array of cartesian vector else 
        MillerPlane object is returned.
        
        """ 
        raise NotImplementedError()
    
    def fromMillerToBravais(self):
        """
        Converts hexagonal 3 index direction vector (Miller Notation) to 4 index (Miller Bravais) notation
        If the lattice is not of hexagonal one same Miller direction (3 index) is returned).
        
        Returns
        -------
        out : tuple of 4 numbers 
            representing the Miller bravais directions (u v t w) for hexagonal lattice or tuple of 3 numbers , 
            essentially same vector as the input one.
        
        """
        if self.lattice.is_hexagonal() :
            vector = self.vector
            h = vector[0]
            k = vector[1]
            i = -(h+k)
            l= vector[2]
            return (h,k,i,l) 
        else:
            raise ValueError("Lattice is not hexagonal and hence Bravais conversion makes no sense !!")     
       
    def getPlaneNormal(self):
        """
        Returns the MillerDirection object representing the plane Normal of the self
        """
        uvw = self.getUnitVector()        
        return MillerDirection(vector=uvw, lattice=self.realLattice,isCartesian=True)
        
    @staticmethod          
    def getZoneAxis(plane1,plane2, returnIntegerZoneAxis=False):  
        """
        Given two MilerPlane objects retuns their zone axis as a MillerDirection
        """
        angle = plane1.angle(plane2,units="Deg")
        
        if abs(angle)>0.01 and abs(angle)<179.9:
        
            dir1 = plane1.getPlaneNormal()
            dir2 = plane2.getPlaneNormal()
            tmp = dir1.cross(dir2).getUnitVector()
            zoneAxis = MillerDirection(vector=tmp,isCartesian=True,lattice=plane1.realLattice)
            if returnIntegerZoneAxis:
                tmp, err = zoneAxis.integerize()
                if tmp.size==4: ### case of hexagonal 4 indices system
                    tmp[2]=-(tmp[0]+tmp[1]) ### this is required to avoid error dued to violoation of u_v = -t conidtion as integerizing the idrection might lead to such anomaoly above
                zoneAxis=MillerDirection(vector=tmp,lattice=plane1.realLattice)
        
            return zoneAxis
        
        else:
            return None 
    
    def getLatexString(self,forceInteger=True):
        """
        returns nicely formatted string for use in graphics.
        """
        if forceInteger :
            [v, err] = self.integerize()
        else:
            v = self.gethkl()
            v = np.round(v).astype(int) #### be careful it can ruin the original values in case the value are not really integers shoul be used only for the display purpose
        s=r'$\{ '    
        for i in range(len(v)) :
            if v[i]<0. :
                s+=(r'\bar{'+str(np.abs(v[i]))+r'} ')
            else:
                s+=(str(v[i])+' ')
        s+=r'\}$'
        return (s)        
            
    def getSimplifiedString(self):
        """
        smplfied string representation for easy text outputs
        :return:
        """



    def structureFactor(self,atomData):
        """
        Given the atom data in the form of a list of floowing format computes the structure factor
        data format : a tuple for each atom comtaining Atomic number/scatterFactor, fractional coordinates of atom in lattice units
        example for Zr (hcp) following is the data structure
        
        atomData = [(40, np.array([0.,0.,0.])),
            (40, np.array([1./3., 2./3.,1./2.]))]
        Retuturns:
        ----------
        structure Factor as real and imag components
        """ 
        real = 0.
        imag = 0. 
        pi2 = 2.*np.pi 
        hkl = self.hkl     
        for i in atomData :
            arg = pi2*(i[1][0]*hkl[0]+i[1][1]*hkl[1]+i[1][2]*hkl[2])
            if len(i)==2 : ### case where atomic occupancy is not defined!!
                real += i[0] * np.cos(arg) ### we assume occupance is 1.00
                imag += i[0] * np.sin(arg)
            else:
                real+=i[0]*i[2]*np.cos(arg)
                imag+=i[0]*i[2]*np.sin(arg)
            
        return real, imag              
            
    def diffractionIntensity(self,atomData=None):   
        """
        Essentially calls the structure factor method and converts the result into Intensity
        I = F*compl(F)
        
        """  
        [real, imag] = self.structureFactor(atomData)
        Intensity = real*real+imag*imag
        return  Intensity
    
    def kikuchiLinePoints(self,Xaxis=[1,0,0],Yaxis=[0,1,0],
                          scalingFactor=1.,patterCenterX=0.,patterCenterY=0.,
                          detectorScaleFactor=1.,lineWidthScalingFactor=1.):
        """
        to compute the kiuchi line points that can be later used to draw the kikuch lines
        """
        p = Polygon((patterCenterX,patterCenterY), detectorScaleFactor, n=4)
        p = p.rotate(np.pi/4)
        spot = self.getCartesianVec()
        lineWidth = lineWidthScalingFactor*0.5*(self.dspacing)
        
        Xaxis = Xaxis/np.linalg.norm(Xaxis)
        Yaxis = Yaxis/np.linalg.norm(Yaxis)
        
        spotX = np.dot(spot,Xaxis)*scalingFactor+patterCenterX
        spotY = np.dot(spot,Yaxis)*scalingFactor+patterCenterY
        
        unitVector = np.array([-spotY,spotX,]) ## perpendicular to kikuchi line
        unitVector = unitVector/np.linalg.norm(unitVector)
        
        point1 = Point(spotX,spotY)
        
        line1 = Line(Point(patterCenterX,patterCenterY),
                     Point(spotX,spotY))
        yAxis = Line(Point(0.,0.), Point(1.,0))
 
        #kikuchiLine = line1.perpendicular_line(point1)
        kikuchiLine = line1
        line1 = line1.perpendicular_line(point1)
        
        textAngle =float(Line.angle_between(kikuchiLine, yAxis))*180/np.pi
        kikuchPoints = p.intersect(kikuchiLine)
        if len(kikuchPoints)!=2 :
            print("the number of intersections of kikuchi lines are not proper", len(kikuchPoints))
            raise ValueError("Kikuchi points not found properly")
        else :
            
            pointList = []
            upperLine = np.zeros((len(kikuchPoints),2))
            lowerLine = np.zeros((len(kikuchPoints),2))
            for i in (kikuchPoints) :
                pointList.append([float(i.x),float(i.y)])
             
            #print(type(pointList[0][0]))
                                    
            for i in  range(len(pointList)):
                upperLine[i][0] = pointList[i][0]+unitVector[0]*lineWidth
                upperLine[i][1] = pointList[i][1]+unitVector[1]*lineWidth
                lowerLine[i][0] = pointList[i][0]-unitVector[0]*lineWidth
                lowerLine[i][1] = pointList[i][1]-unitVector[1]*lineWidth
                
            textVector = [pointList[1][0]-pointList[0][0], pointList[1][1]-pointList[1][0]]
            textVector = textVector/np.linalg.norm(textVector)
            textLocation = [pointList[0][0]+0.1*textVector[0], 
                            pointList[0][1]+0.1*textVector[1]]
            
            
            kikuchiLinedata = (pointList, upperLine,lowerLine, textLocation, textAngle)
            return kikuchiLinedata
    def get2theta(self,waveLength=1.5406, units="deg"):
        """
        Method to clacualte the 2theta angle for the given miller Plane
        """
        if "deg" in units.lower() :
            return(2*np.arcsin(waveLength/(2.*self.dspacing)))*180./np.pi
        else :
            return(2*np.arcsin(waveLength/(2*self.dspacing)))
    
    
    def symmetricSet(self, returnCartesianVecs=False):
        """
        method to get  the symmetric set considering the Symmetry of the lattice
        
        """
        if self._symmetricSet==None :
            originalVec = self._cartesianVector ### this is alrerady reciprocal vector
            newVecs = np.zeros((self.lattice._NumberOfSymmetryElements,3), dtype=np.float)
            
            for i, symElement in enumerate(self.lattice._SymmetryElements):
                newVecs[i] = symElement.rotate(originalVec)
            symVecs = pmt.uniqueRows(newVecs, thresh=1e-3) 
            symSet=[]
            
            for i, vec in enumerate(symVecs):
                tmpVec = self.lattice.get_fractional_coords(vec) ## converting into lattice frame
                symSet.append(MillerPlane(hkl=tmpVec, lattice=self.realLattice,recLattice=self.lattice)) 
            self._symmetricSet = symSet
            self._multiplicity = len(symSet)
            self._symVecsCartesian = symVecs
        if returnCartesianVecs :
            return self._symVecsCartesian
                     
        return self._symmetricSet 
    
    
    def multiplicity(self):
        """
        Returns the multiplicity of the object i.e. number of crystallographically equivalent ones
        for e.g. multiplicity of [100] in cubic is 6 and [1,1,1] is 8 etc.       
        """
        if self._multiplicity == None :            
            self._multiplicity = len(self.symmetricSet())
        return self._multiplicity
    
    
    def isSymmetric(self, other, tol=1e-3,checkInputDataSanity=True, ):
        """ Test of obj1 and obj2 are symmetrically equivalent
        
        """
        if checkInputDataSanity :
            if not isinstance(other, MillerPlane):
                raise ValueError("the other object must be of same type i.e. millerPlane or MillerDirection")
            if not self.lattice==other.lattice :
                raise ValueError ("Only objects of same lattice can be compared for symmetric equivalance")
        
        if np.abs(self._mag-other._mag)>1e-5:
                return False 
        sourceSet = self.symmetricSet(returnCartesianVecs=True)
        otherVec = other._cartesianVector
        for i in sourceSet :
            if np.allclose(i, otherVec, atol=tol, rtol=1e-3):
                return True
        return False
            
    @staticmethod
    def uniqueListFast(planeList, considerSymmetry=True, modifyOriginalList=False):
        """
        Given a plane list, returns the uniqe MillerPlane list using Fast algorithm
        Input
        -----
        considerSymmetry : deafult True, if False, does not consider symmetrically equivalence in detrmining the uniquenss of planes  
        """
        
        dArray= np.array([i.dspacing for i in planeList])*1e6
        
        dArray = dArray.astype(np.int64)
        u = np.unique(dArray)
        uiqueList=[]
        uniqueIndexList=[]
        for item in u.tolist():
            tmpIndices=np.where(dArray==item)[0]
            #tmpArray = dArray[np.where(dArray==item)]
            if tmpIndices.size==1:
                uiqueList.append(planeList[tmpIndices[0]])
                uniqueIndexList.append(tmpIndices[0])
            else:
                
                tmpList=[planeList[i] for i in tmpIndices.tolist()]
                uList = MillerPlane.uniqueList(planeList=tmpList, considerSymmetry=True,modifyOriginalList=True)
                uiqueList.extend(uList)
        
        uiqueList= sorted(uiqueList , key=lambda x: x.dspacing, reverse=True)
        return uiqueList ## we need reverse so that list is sorted such that max d spacing planes occur first
    
    
    @staticmethod
    def uniqueList(planeList, considerSymmetry=True, modifyOriginalList=False):
        """
        Given a plane list, returns the uniqe MillerPlane list 
        Input
        -----
        considerSymmetry : deafult True, if False, does not consider symmetrically equivalence in detrmining the uniquenss of planes  
        """
        
        if modifyOriginalList :
            tmpPlaneList = planeList
            
        else:
            tmpPlaneList = [i for i in planeList]
            
        if isinstance(tmpPlaneList, list):
            uniqueList=[]
            for i, plane in enumerate(tmpPlaneList):
                uniqueList.append(plane)
                delList = []
                for j in range(i+1,len(tmpPlaneList)):
                 
                        if plane.isSymmetric(tmpPlaneList[j]):                        
                            delList.append(j)
                            
                for k in reversed(delList):
                    del tmpPlaneList[k] 
                
            return uniqueList
        else:
            raise ValueError("only a List of planes must be input !!! ")
        
    @staticmethod
    def generatePlaneList(hklMax, lattice,includeSymEquals=False):
        """
        Method to genrate a list ofhkl planes which are symmetrically unique and sorted by 
        highest dspacing plane folowed by planes with lower d-spacings
        Input:
        -----
            hklMax : an integer specifying the max index of hkl
            lattice : :py:class:`~OrientedLattice` object specifying the lattice of the MillerPlanes to be genrated 
        """
        a = range(-(hklMax),hklMax+1)
        recLattice = olt(lattice.reciprocal_lattice_crystallographic.matrix,
                                                               pointgroup=lattice._pointgroup)
        hklList = []
        if includeSymEquals:
            a = range(-(hklMax),hklMax+1)
            for combination in itertools.product(a, a, a):            
                hklList.append(combination)
            hklList.remove((0,0,0))
            hklPlanelist=[]
            for i, hkl in enumerate(hklList):
                plane=MillerPlane(lattice=lattice,hkl=hkl,recLattice=recLattice)
            #print("Here is the issue")
                hklPlanelist.append(plane) 
            
            sortedList = sorted(hklPlanelist , key=lambda x: x.dspacing, reverse=True)
            return sortedList
        
        for combination in itertools.product(a, a, a):            
            hklList.append(combination)    
        del hklList[0]; ## this is (0,0,0) plane and hence being removed 
        hklPlanelist=[]
        dspacingsArray=np.zeros((len(hklList),1),dtype=float)
        for i, hkl in enumerate(hklList):
            plane=MillerPlane(lattice=lattice,hkl=hkl,recLattice=recLattice)
            #print("Here is the issue")
            hklPlanelist.append(plane)                
        sortedList = sorted(hklPlanelist , key=lambda x: x.dspacing, reverse=True)
        result = MillerPlane.uniqueListFast(sortedList)
        return result    
    
    @staticmethod
    def solvePlaneforIndexing(plane1,plane2,measuredDratio,measuredAngle,
                              dspaceTolerance=10,angleTolerance=2):
        """
        Given two MillerPlanes, finds the possible zone axis if the two spots are satisfying the 
        given angle and dratio tolerances
        dspaceTolerance = 0.10 i.e. 10%
        angleTolerance = 2 degree
        
        """
        result = None
        if (isinstance(plane1,MillerPlane) and isinstance(plane2, MillerPlane)):
            d_ratio = plane1.dspacing/plane2.dspacing
            if ((np.abs((d_ratio-measuredDratio)/measuredDratio))*100<dspaceTolerance or \
                (np.abs((1/d_ratio-measuredDratio)/measuredDratio))*100<dspaceTolerance):                
                symPlanes2 = plane2.symmetricSet()
                for k in symPlanes2 :                    
                    ang = plane1.angle(k,units='Deg')
                    if (np.abs(ang-measuredAngle))<angleTolerance :
                        zoneAxis = MillerPlane.getZoneAxis(plane1, k)
                        dError = min([np.abs(d_ratio-measuredDratio),
                                      np.abs(1/d_ratio-measuredDratio)])*100./measuredDratio
                        angError = (np.abs(ang-measuredAngle))
                        return  {"zoneAxis" : zoneAxis, "spot1": plane1, "spot2":k, 
                                 "dError" : dError, "angError" : angError, "areSpotsSymmetric":False }
            else:                
                return result        
                
            return result        
            
               
        else:
            raise ValueError('Only MillerPlanes can be input to this method !!!')
            
    
    
    @staticmethod
    def fromNormalAndPoint(normal,lattice,point=[0,0,0]): 
        """
        create a MillerPlane Object from given plane normal and the point in plane
        """      
    
        
        plane = Plane(Point3D(point),normal_vector=normal)
        
        origin = [0,0,0]
           
        origin = lattice.get_cartesian_coords(origin)
        origin = Point3D(origin)
        aAxis = Line3D(origin,Point3D(lattice.matrix[0]))
        bAxis = Line3D(origin,Point3D(lattice.matrix[1]))
        cAxis = Line3D(origin,Point3D(lattice.matrix[2]))
        
        
        angleWithX, angleWithY, angleWithZ = float(plane.angle_between(aAxis)), float(plane.angle_between(bAxis)), float(plane.angle_between(cAxis))
        print(angleWithX, angleWithY, angleWithZ)
        
        
        h = aAxis.intersection(plane)
        if len(h)==1: ### found the intersection
            p=h[0]
            h = np.dot(np.array([p.x,p.y,p.z]),lattice.matrix[0])
            if np.abs(h)<1e-5:
                h = np.dot(np.array([p.x,p.y,p.z]),-lattice.matrix[0])
                h=-1/h
            else:
                h=1/h
        else:
            h=0
            
        k = bAxis.intersection(plane)
        if len(k)==1: ### found the intersection
            p=k[0]
            k = np.dot(np.array([p.x,p.y,p.z]),lattice.matrix[1])
            if np.abs(h)<1e-5:
                k = np.dot(np.array([p.x,p.y,p.z]),-lattice.matrix[1])
                k=-1/k
            else:
                k=1/k
        else:
            k=0
   
        l = cAxis.intersection(plane)
        if len(l)==1: ### found the intersection
            p=l[0]
            l = np.dot(np.array([p.x,p.y,p.z]),lattice.matrix[2])
            if np.abs(h)<1e-5:
                l = np.dot(np.array([p.x,p.y,p.z]),-lattice.matrix[2])
                l=-1/l
            else:
                l=1/l
        else:
            l=0
   
        hkl = pmt.integerize(np.array([float(h),float(k),float(l)]))
        print(hkl)
        return MillerPlane(hkl = hkl,lattice=lattice)
        
        
       

    
    def getKikuchiLine(self,crystalOri,detectorCoordinates=None):
        """
        returns the line , representing the trace of the plane by finding the intesection with a supplied polygon
        the supplied polygon must be of 4 coordinates representing a quadrelateral 
        """
        if not isinstance(crystalOri,Orientation):
            raise ValueError("The supplied object is not Orienation !!!!")
        if detectorCoordinates is None:
            detectorCoordinates=np.array([[-1.,-1.,-1.],
                                          [+1.,-1.,-1.],
                                          [+1.,+1.,-1.],
                                          [-1.,+1.,-1.],
                                         ])### defines a square detector at Z=-1.
            
        
        p1 = Point3D(detectorCoordinates[0])
        p2 = Point3D(detectorCoordinates[1])
        p3 = Point3D(detectorCoordinates[2])
        p4 = Point3D(detectorCoordinates[3])
        detectorPlane = Plane(p3, p2,p1)
        p = self.getPointInPlane()
        normal = self.getCartesianVec()
        
        p = crystalOri.rotate(p)
        normal = crystalOri.rotate(normal)
                
        selfPlane = Plane(Point3D(0.,0.,0.),normal_vector=normal)
        trace = selfPlane.intersection(detectorPlane)
        #print("result", trace, type(trace),detectorPlane,selfPlane)
        if len(trace)==1:
            allPoints=[]
            lineAB = Segment3D(p1,p2)
            lineBC = Segment3D(p2,p3)
            lineCD = Segment3D(p3,p4)
            lineDA = Segment3D(p4,p1)
            trace=trace[0]
            pAB = lineAB.intersection(trace)
            if len(pAB)==1:
                if isinstance(pAB[0],Segment3D):
                    print("got segemtnt")
                    pAB=[p1,p2]
                allPoints.append(pAB)
                
            pBC = lineBC.intersection(trace)
            if len(pBC)==1:
                if isinstance(pBC[0],Segment3D):
                    print("got segemtnt")
                    pBC=[p2,p3] 
                allPoints.append(pBC)
            pCD = lineCD.intersection(trace)
            if len(pCD)==1:
                if isinstance(pCD[0],Segment3D):
                    print("got segemtnt")
                    pCD=[p3,p4] 
                allPoints.append(pCD)
            pDA = lineDA.intersection(trace)
            if len(pDA)==1:
                if isinstance(pDA[0],Segment3D):
                    print("got segemtnt")
                    pDA=[p4,p1] 
                allPoints.append(pDA)
            ### now flattening the list
            allPoints = [item for sublist in allPoints for item in sublist]
            
            if len(allPoints)>2:
                allPoints=pmt.uniqueList(allPoints)
                if (len(allPoints)>2):
                    print("Warning : Probelm as more than 3 points are found and they are "+str(allPoints))
                    allPoints = allPoints[0:2] 
            return allPoints
            
        else:
            #print("This plane does not appear on the geometry of the detector !!!!")
            return []
            
            


        
     
if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All tests are done")    
        
                
        
        
        
        
        
        
                                                                                                   