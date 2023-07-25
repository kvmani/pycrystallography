# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

"""
+++++++++++++++++++++++++++++++++++
Module Name: **orientation.py**
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
from pycrystallography.core.orientedLattice  import OrientedLattice as olt 
from pycrystallography.core.millerDirection  import MillerDirection 
from pycrystallography.core.millerPlane  import MillerPlane 


import pycrystallography.utilities.pytransformations as pt 
import pycrystallography.utilities.pymathutilityfunctions as pmt 


from monty.json import MSONable
from monty.dev import deprecated



__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"



class CrystalOrientation(Orientation, MSONable):
    """
    A wrapper class for quartenion package for easy mnaipulation of quaternions in terms of euler angles as used in the feild of materials science. Essentially a 3 element euler angle matrix. In general it is of Bunge notation only. All angles are in radians only. In case degree is to be used it must be specified. Other notations are also possible by explicit mentioning the same.
    """
    
    def __init__(self, *args, **kwargs):
        """Create an crystal orientation from oreiantion and lattice object::
        
        """
        
        self._lattice=None
        s = len(args)
        if s == 0:
            # No positional arguments supplied
            if len(kwargs) > 0:
                # Keyword arguments provided
                
                if ("lattice" in kwargs):
                    lattice = kwargs.get("lattice")  
                    self._lattice = lattice                
                
                if ("orientation" in kwargs):
                    ori = kwargs.get("orientation", Orientation.stdOri())                                        
                    #super().__init__(self, ori.q)
                    super().__init__(ori)
                else :
                    ori = Quaternion(*args, **kwargs)
                    Quaternion.__init__(self, ori)
                    #print("Lattice is not initialized !!!!")
               
        elif s==1: ## case of another Crystal Orientation specified
            if isinstance(args[0],CrystalOrientation):
                super().__init__(args[0].q)
                self._lattice = args[0]._lattice
            else:
                raise TypeError("Crystal Orienation Object cannot be initialised from " + str(type(args[0])))
            
        elif s==2:
            if isinstance(args[0], Orientation) and isinstance(args[1], olt):
                #Quaternion.__init__(self, args[0].q)
                super().__init__(args[0])
                self._lattice = args[1]
                
                
        else:
            raise ValueError("Improper calling of the intiiation of the CrystalOrieantion Object.")
                
        self._oriInfundamentalZone = None
        self._symmetricSet=None
        self._millerForm = None        
                    
#     def __deepcopy__(self):
#         return self.__class__(self)
    

    def __repr__(self):        
        eulerAngles = self.getEulerAngles(units='deg')
        str1 = eulerAngles.__repr__()
        str2 = "\n"+self._lattice.__str__()
        #str = str + '\n An oirenation object'
        return str1+str2
   
    def __str__(self, **kwargs):
        
        return self.__format__('.2f')
        
        
    def __format__(self, formatstr):
        
        if formatstr=='planeDir':
            p, d = self.getMillerNotation()
            hkl,err = p.integerize()
            str1 = "CrystlOri : ("+str(hkl)+")"
            uvw,err = d.integerize()
            err = np.abs(np.dot(hkl,uvw))
            #assert err<1e-5, "Problem in millerRepresentation of the "+str(err)
            str2 = " {:int}".format(d)
            str3 = " latticeType = "+self._lattice._latticeType
            return str1+str2+str3
        elif formatstr=='simplePlaneDir':
            p, d = self.getMillerNotation()
            hkl, err = p.integerize()
            str1 = f"({str(hkl)}) "
            str1=str1.replace("[","") ## removing [ and ]
            str1=str1.replace("]","")
            uvw, err = d.integerize()
            err = np.abs(np.dot(hkl, uvw))
            #assert err<1e-5, "Problem in millerRepresentation of the "+str(err)
            str2 = " {:int}".format(d)
            return str1 + str2

        elif formatstr=='axisAngle':
            ang,axis = self.axisAngle() 
            return "{:.2f}@{:int}".format(ang,axis) 
        
        elif formatstr=='euler':
            return super().__format__(formatstr='.1f')
        
        else: 
            formatstr = '.1f'

        string = \
            "[{:" + formatstr +"} "  + \
            "{:" + formatstr +"} " + \
            "{:" + formatstr +"}] " 
        eul = Orientation.getEulerAngles(self,units='deg')
        str1 =  string.format(eul[0], eul[1], eul[2])
        if self._lattice is None:
            return str1+ "lattice=None"
        else:
            str2 = " lattice = "+'{:.2f}'.format(self._lattice)
        return str1+str2
       
    
    def getMillerNotation(self):
        if self._millerForm is None:
            plane = MillerPlane(hkl = [0,0,1], isCartesian=True, recLattice=self._lattice.reciprocal_lattice_crystallographic,lattice=self._lattice)
            direction = MillerDirection(vector = [1,0,0], isCartesian=True, lattice=self._lattice)         
            plane.rotate(self)
            direction.rotate(self)
            self._millerForm=(plane, direction)
        return  self._millerForm
        
    
    
    @property
    def planeAndDirection(self):
        """
        returns the tuple of MillerPlane and Miller Direction (plane normal along Z axis and Direction along the X axis representing the crystal Orientation
    
        """
        if self._millerForm is None:
            return self.getMillerNotation()
        return self._millerForm
   
    
    def symmetricSet(self):
        if self._symmetricSet==None :  
            tmp = Orientation(Quaternion(self.q))          
            symSet = []
            for i, symElement in enumerate(self._lattice._SymmetryElements):
                symSet.append(CrystalOrientation(orientation=symElement*tmp,lattice=self._lattice))
            self._symmetricSet=symSet
        return self._symmetricSet

    def projectTofundamentalZone(self):
        """
        returns the Orientation projection to fundamental Euler space of the crystal system
        """
        tol = 1e-3
        limits = self._lattice._EulerLimits+tol
        eligibleCandidates=[]
        eligibleIndices=[]
        index=0
        if self._oriInfundamentalZone is None:
            symmetricOriSet = self.symmetricSet()
            eulerAnglesSet = [i.getEulerAngles(applyModulo=True).tolist()
                              for i in symmetricOriSet]
            mags =np.linalg.norm(np.array(eulerAnglesSet), axis=1) ### remainder being applied to bring everything back to 360 180 360 space
            index = np.argmin(mags)
            self._oriInfundamentalZone = symmetricOriSet[index]
            euler = eulerAnglesSet[index]
            self._oriInfundamentalZone = CrystalOrientation(orientation=Orientation(euler=euler),lattice=self._lattice)

            #     if all(euler<limits):
            # found = False
            # for i,item in enumerate(eulerSet):
            #     euler =np.abs(item.getEulerAngles())
            #     if all(euler<limits):
            #         self._oriInfundamentalZone = CrystalOrientation(orientation=Orientation(euler=euler),lattice=self._lattice)
            #         found = True
            #         eligibleCandidates.append(euler*180.0/np.pi)
            #         eligibleIndices.append(i)
            #


            assert all(euler<limits), f"Could not find the fundamental Ori Something is Wrong !!!! {euler=} : {limits=}"
        return  self._oriInfundamentalZone,index
    
    
    def disoreintation(self, other):
        """
        returns the minimum miso betwween two crystal orientation considering their respective symmetries
        Parameters
        ----------
        other : ``Crystal orientation`` object 
            The Orienation object with the misorieantion is desired
          
        Returns
        -------
        out : ``Crystal orientation`` object 
            An Orientation object, representing the disorientaton (i.e. lowest misorieantion angle one) between the slef, and other
            : minAngle (disorienation angle in radians)

          
   
        """
        
        misoriSet = self.symmetricMisorientations(other)
        minAngle=1e5
        for misori in misoriSet:
            if abs(misori.angle)<minAngle:
                disOri = misori
                minAngle = abs(misori.angle)

        return disOri,minAngle
            
            
    
    
    def misorientation(self,other) :
        """
        Retun the Misorienation with the other Crystal Orieantion.
          
        Parameters
        ----------
        other : ``Crystal orientation`` object 
            The Orienation object with the misorieantion is desired
          
        Returns
        -------
        out : ``Crystal orientation`` object 
            An Orientation object, representing the misorieantion between the slef, and other
          
        
        """
        if isinstance(other, CrystalOrientation):
            misor = Orientation(other*self.inverse)
            #misor = Orientation(other.inverse*self)
            
            return CrystalOrientation(orientation=misor,lattice=other._lattice)
        else:
            raise TypeError("Only Crystal Orienation Object can be passed for the misoreiantion function")

    
    def symmetricMisorientations(self,other):
        """
        returns List of the symmetric misorientations
        """
        if  isinstance(other, CrystalOrientation):
            misOrilist=[]
            for item_i in self.symmetricSet():
                for item_j in other.symmetricSet():
                    misOrilist.append(item_i.misorientation(item_j))
            return misOrilist
        else:
            raise TypeError("Only Crystal Orienation Object can be passed for the misoreiantion function ")
    
    
    def isSymmetric(self,other,tol=1e-3):     
        """
        true if the other Oreiantion is symmetric with respect to the self
        """
        
        if isinstance(other, CrystalOrientation):
            if other._lattice==self._lattice:
                symSet = self.symmetricSet()
                for item in symSet:
                    ang = np.abs(item.misorientationAngle(Orientation(euler=other.getEulerAngles())))
                    #print("ang=",ang)
                    if ang<tol:
                        return True
                else:
                    return False
            return False
        else:
            raise ValueError("Only CrystalOreiantion object must be tested for symmetric equivalence")
        
    def axisAngle(self):
        ang = self.degrees
        axis = MillerDirection(vector = self.axis, isCartesian=True,lattice=self._lattice)
        return ang,axis
        
   
    @staticmethod
    def uniqueList(oriList, considerSymmetry=True, returnInds = False, tol = 0.01):
        """
        Given an orientation list, returns the uniqe Orienation list considering the symmetry
        Input
        -----
        considerSymmetry : deafult True, if False, does not consider symmetrically equivalence in detrmining the uniquenss of dirs  
        """
        tmpOriList = [(i,Ori) for i, Ori in enumerate(oriList)]
        ids = []
        numOfDeletions=0
        
        if isinstance(tmpOriList, list):
            uniqueList=[]
            #numOfPlanes=len(tmpPlaneList)
            for i, crystalOri in enumerate(tmpOriList):
                uniqueList.append(crystalOri[1])
                if returnInds :
                    ids.append(crystalOri[0]) 
                delList = []
                for j in range(i+1,len(tmpOriList)):
                    #if considerSymmetry:                    
                        if crystalOri[1].isSymmetric(tmpOriList[j][1],tol = tol):                        
                            delList.append(j)
                        
                numOfDeletions+=len(delList)           
                for k in reversed(delList):
                    del tmpOriList[k]
                
                
            if returnInds:    
                return uniqueList, ids
            else:
                return uniqueList
        else:
            raise ValueError("only a List of dirs must be input !!! ")
         
        
    @staticmethod
    def fromPlaneAndDirection(plane,direction):
        if isinstance(plane,MillerPlane) and isinstance(direction,MillerDirection):
            if (plane.realLattice==direction.lattice):
                zVector = plane.getUnitVector()
                xVector = direction.getUnitVector()
                ori = Orientation.fromTwoOrthoDirections(xVector, zVector, 'xz')
                return CrystalOrientation(orientation=ori,lattice=direction.lattice)
            else:
                raise ValueError("Plane and Direction belong to different lattices.")
        else:
            raise TypeError("Only MillerPlane and MillerDirection Objects can be used")
            
    def ipfColor(self,ipfDir=[0,0,1]):
        symSet = self.symmetricSet()
        rfac = 1e6
        fudge = 1e-4
        pow=0.8
        localEps = 1e-16*1e3
        colGrid = np.array([[0,0,1.,],[1, 0, 1], [1, 1, 1]])
        for i in range(3):
            colGrid[i,:] = colGrid[i,:]/np.linalg.norm(colGrid[i,:])
        ipfDirs = np.zeros((len(symSet),3))
        
        for i, item in enumerate(symSet):
            ipfDirs[i,:] = item.rotate(ipfDir)
        
        ipfDirs[2,ipfDirs[2,:]==0]=localEps
        n2=np.where(np.round(ipfDirs[0,:]*rfac)>=-localEps & round(ipfDirs[0,:]*rfac)<=round(abs(ipfDirs[2,:])*rfac) & np.round(ipfDirs[0,:]*rfac)>=round(ipfDirs[1,:]*rfac))
          
        
        
    
        
        
if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All tests are done")