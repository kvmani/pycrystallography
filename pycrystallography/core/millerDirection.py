# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.
"""
+++++++++++++++++++++++++++++++++++
Module Name: **millerDirection.py**
+++++++++++++++++++++++++++++++++++

This module :mod:`millerDirection` defines the class Miller Directions. Essentially it wrpas
the underlying Vectro3d class of vtk library and makes it suitable for the use 
in crystallographic computations.

This module contains classes.

**Data classes:**
 * :py:class:`MillerDirection`

    *List of functions correponding to this class are*
     * :py:func:`MillerDirection.getUVW`
     * :py:func:`MillerDirection.fromMillerToBravais`
     * :py:func:`MillerDirection.getCartesianVec`
     * :py:func:`MillerDirection.mag`
     * :py:func:`MillerDirection.uvw`
     * :py:func:`MillerDirection.getMag`
     * :py:func:`MillerDirection.getUnitVector`
     * :py:func:`MillerDirection.angle`
     * :py:func:`MillerDirection.rotate`
     * :py:func:`MillerDirection.dot`
     * :py:func:`MillerDirection.cross`
     * :py:func:`MillerDirection.getDirectionInOtherLatticeFrame`
     * :py:func:`MillerDirection.hexToRhombohedral`
     * :py:func:`MillerDirection.hexToOrthorhombic`
     * :py:func:`MillerDirection.getOrthoset`
     * :py:func:`MillerDirection.getPerpendicularDirection` 
     * :py:func:`MillerDirection.integerize`
     * :py:func:`MillerDirection.getLatexString`
     
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
from pycrystallography.core.orientedLattice import  OrientedLattice as olt  

#import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pymathutilityfunctions as pmt 
from pymatgen.core.lattice import Lattice as lt

from monty.json import MSONable
from monty.dev import deprecated

#from vtk import vtkVector3d
from math import sqrt
from numpy import pi, dot, transpose, radians
from pymatgen.core import Lattice
from pycrystallography.core.orientation import Orientation
import copy

"""
This module defines the class Miller Directions. Essentially it wraps
the underlying Vector3d class of vtk library and makes it suitable for the use 
in crystallographic computations.
"""


__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"

class MillerDirection(MSONable):
    
    """
    Class for representing the crystal directions
    """
    def __init__(self, lattice=olt.cubic(1), vector = [0,0,0],isCartesian=False, MillerConv = 'Miller'):
        
        if not isinstance(lattice, olt):
            raise ValueError ('Lattice supplied must be of OrientedLattice class but was procided type : '+str(type(lattice)))
        if  len(vector) == 4 and lattice.is_hexagonal():
            if (np.abs(vector[0]+vector[1]+vector[2])>1e-5) :
                raise ValueError (f'u+v=-t is not satisfied as the suppled uvtw is  : {vector}')
            vector = pt.validate_number_sequence(vector, 4)
            u =(vector[0]-vector[2])
            v = (vector[1]-vector[2])
            w = vector[3]
            self.vector = np.array([u,v,w])
              
        elif len(vector)==3 and "Bravais" in MillerConv:
            vector = pt.validate_number_sequence(vector, 3)
            vv = [vector[0], vector[1], -(vector[0]+vector[1]), vector[2]]
            u =(vv[0]-vv[2])
            v = (vv[1]-vv[2])
            w = vv[3]
            self.vector = np.array([u,v,w])
            
        elif len(vector) == 3:        
            vector = pt.validate_number_sequence(vector, 3)
            self.vector = np.array(vector)
        else:
            raise ValueError("Incorrect specification of the MillerDirection; if hexagonal, specify 4 indices using key word Bravais")
                        
        if isCartesian :
            ## we have to convert the given lattice vector into Cartensian vector as internally only cartesin avectros are stored
            vector = lattice.get_fractional_coords(vector)
            self.vector = np.array(vector)
               
        self.lattice = lattice
        self._cartesianVector = lattice.get_cartesian_coords(self.vector)
        self._mag = self.getMag()
        self._isReciprocalVector=False
        self._multiplicity = None
        self._symmetricSet = None ## inititialized lazily
        self._symVecsCartesian = None
        

#     def __deepcopy__(self):
#         return self.__class__(self)
    
    
    # Representation
    def __str__(self):
        
        """
        An informal, nicely printable string representation of the MillerPlane object.
        """ 
        [vector, AngleError] = self.integerize(AngleDeviationUnits='Deg')
        if AngleError>2:
            Marker='*'
        else:
            Marker=''
            
        if self.lattice.is_hexagonal() :
           
            return "[{:3d}  {:3d} {:3d} {:3d} ]{:s}".format(vector[0], vector[1], vector[2],vector[3],Marker)
            #return "[{:5.1f}  {:5.1f} {:5.1f} {:5.1f} ]".format(vector[0], vector[1], vector[2],vector[3])  
            
          
        
        
        return "[{:3d}  {:3d} {:3d}  ]{:s}".format(vector[0], vector[1], vector[2],Marker)
    

    def __repr__(self):
        
        """
        The 'official' string representation of the MillerDirection object.  
        This is a string representation of a valid Python expression that could be used
        to recreate an object with the same value (given an appropriate environment)
        """
        str = self.__format__(".3f")
        return str 
    
    def __format__(self, formatstr):
        
        """
        Inserts a customisable, nicely printable string representation of the Quaternion object
        The syntax for `format_spec` mirrors that of the built in format specifiers for floating point types. 
        Check out the official Python [format specification mini-language](https://docs.python.org/3.4/library/string.html#formatspec) for details.
        """
        integerOutPut = False
        if formatstr.strip() == '': 
            # Defualt behaviour mirrors self.__str__()
            formatstr = '.2f'
        if "int" in formatstr:
            formatstr='2d'
            integerOutPut=True
       
        if self.lattice.is_hexagonal() :
            string = \
                "[{:" + formatstr +"} "  + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"} ]"
            vector = self.fromMillerToBravais()
            if integerOutPut:
                vector, tmp = self.integerize(AngleDeviationUnits="Deg")
                str1 = string.format(vector[0], vector[1], vector[2], vector[3])
                return str1               
             
            str1 = string.format(vector[0], vector[1], vector[2], vector[3])
        else :     

            string = \
                "[{:" + formatstr +"} "  + \
                "{:" + formatstr +"} " + \
                "{:" + formatstr +"} ]" 
            vector = self.vector
            if integerOutPut:
                vector, tmp = self.integerize(AngleDeviationUnits="Deg")
                str1 = string.format(vector[0], vector[1], vector[2])
                return str1
                         
            
            str1 = string.format(vector[0], vector[1], vector[2])                   
            
               
        str2 = format(self.lattice,'.1fp')
        
        return str1+" lattice="+str2
                                        
    def getUVW(self,force3Index=False):
        
        """   
        Method to get the MillerDirection in Lattice Frame. In regular usage, this is the form we employ.
        If the lattice to which this direction belongs is of hexagonal type, it returns a numpy array [u v t w] 
        
        Parameters
        ----------
        force3Index : [optional] 
            If set to True returns 3 index notation even for hexagonal.
        
        Returns
        -------
        out : A numpy array of size (4,) or (3,)
            Dpending on if it hexagonal or not.
        
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [2, -1, -1, 1]
        >>> dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        >>> uvtw = dir1.getUVW()
        >>> np.allclose (uvtw, vec)
        True
        
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [2, -1, -1, 0]
        >>> dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        >>> uvtw = dir1.getUVW(force3Index = True)
        >>> np.allclose (uvtw, [3., 0., 0.])
        True
                          
        """
        if not force3Index and self.lattice.is_hexagonal() :        
            vector = self.fromMillerToBravais()        
            vector = np.array(vector)
            return vector
        else:
            return self.vector
    
    def fromMillerToBravais(self, integerize=False):
        
        """
        Converts hexagonal 3 index direction vector (Miller Notation) to 4 index (Miller Bravais) notation
        If the lattice is not of hexagonal type, the same Miller direction (3 index) is returned).
        
        Input
        -----
        integerize [optional] default is False , if True returns integerized version of Bravais indices
        Returns
        -------
        out : tuple of 4 numbers 
            representing the Miller bravais directions (u v t w) for hexagonal lattice or tuple of 3 numbers , 
            essentially same vector as the input one.
        
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [3,0,0]
        >>> a = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        >>> mill2Brav = a.fromMillerToBravais()
        >>> uvtw = a.getUVW(force3Index = False)
        >>> np.allclose (uvtw, mill2Brav)
        True
        
        """
        if self.lattice.is_hexagonal() :
            vector = self.vector
            u = (2*vector[0]-vector[1])/3.0
            v = (2*vector[1]-vector[0])/3.0
            t = -(u+v)
            w = vector[2]
            bravais = np.asarray([u,v,t,w])
            if integerize:
                return pmt.integerize(bravais)
            else:
                return bravais
              
        else:
            raise ValueError("Lattice is not hexagonal and hence Bravais conversion makes no sense !!")     
    
    def getCartesianVec(self):
        
        """
        Get the equivalent 3d vector in cartesian frame of reference
        
        Returns
        -------
        out : numpy array of size (3,) 
            Equivalent to the cartesian vector defined by the MillerDirection object. 
            
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [2,-1,-1,0]
        >>> a = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        >>> cartVec = a.getCartesianVec()
        >>> np.allclose (cartVec, [3,0,0])
        True
        """
        return(self._cartesianVector)
    
    @property
    def mag(self):
        
        """
        Returns the magnitude of the MillerDirection
        
        Examples
        --------
        >>> cubicLat = Lattice.cubic(1)
        >>> vec = [1.,1.,1.]
        >>> dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        >>> np.allclose(dir3.mag, sqrt(3))
        True
        
        """
        return self.getMag()
    
    @property
    def uvw(self):  # nachiket 4-12-17
        
        """
        Returns the MillerDirection in Lattice frame.
        Essentially considers hexagonal 3-index input vector as forced 3-index.
        If vector input is Hexagonal and in Miller-Bravais notation (4 indices), 
        output is Miller-Bravais in forced 3-index notation
        
        Returns
        -------
        out : numpy array of size (3,) 
                    
        Examples
        --------
        >>> cubicLat = Lattice.cubic(1)
        >>> vec = [1,1,1]
        >>> dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        >>> np.allclose(dir3.uvw,[1,1,1])
        True
        """
        return self.vector
    
    def getMag(self): 
        
        """
        Returns the magnitude of the MillerDirection Object.
        
        Examples
        --------
        >>> cubicLat = Lattice.cubic(1)
        >>> vec = [1.,1.,1.]
        >>> dir3 = MillerDirection(lattice=cubicLat,vector=vec)
        >>> np.allclose(dir3.getMag(), sqrt(3))
        True        
        
        """
        return np.linalg.norm(self.getCartesianVec())
    
    def getUnitVector(self,returnCartesian=True):
        
        """
        returns a unit vector parallel to the MillerDirection object
        if returnCartesian=False returns another Millerdirection Object witrh Unit magnitude
        
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [2,-1,-1,0]
        >>> a = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        >>> uniVec = a.getUnitVector()
        >>> np.allclose(uniVec,[1,0,0])
        True
        """
        v = self.getCartesianVec()
        v= v/np.linalg.norm(v)
        if not returnCartesian:
            return MillerDirection(vector=v,isCartesian=True,lattice=self.lattice)
        else:
            return v
    
    def angle(self,other, units='rad',considerSymmetry=False):
        
        """
        Returns the angle between self and another MillerDirection Object
        
        Parameters
        ----------
        other : MillerDirection object
            Another MillerDirection object.
        units : [optional] deg or Deg or Degree
            If specified, returned angle is in degrees.
            
        considerSymmetry : [optional] default False. if True returns the min angle with one of the 
                            symmetric vectors of the other object
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> cubLat = Lattice.cubic(1)
        >>> vec1 = [2,-1,-1,0]
        >>> vec2 = [0,0,1]
        >>> a = MillerDirection(lattice=hexLat, vector = vec1, isCartesian=False, MillerConv="Bravais")
        >>> b = MillerDirection(lattice=cubLat, vector = vec2, isCartesian=False, MillerConv="Miller")
        >>> angBetween = a.angle(b,'deg')
        >>> np.allclose(angBetween,90)
        True
            
        """
        if isinstance(other, MillerDirection):
            if considerSymmetry:
                v2 = other.symmetricSet()
                ang = 1e10
                for item in v2:
                    tmpAng = self.angle(item)
                    if tmpAng<ang:
                        ang = tmpAng
            
            else:
                v1 = self.getCartesianVec()
                v2 = other.getCartesianVec()            
                ang = math.acos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1.0,1.0))        
            
            if "deg" in units.lower():
                ang = ang*180/np.pi
            return ang
        else :
            raise ValueError("angle : function was supplied with non MillerDirection Object")
        
    def rotate(self, rotation):
        
        """
        Rotate the MillerDirection object with a given orientation object. Essentially it is a rotation operation on a given vector
        to orient in a required orientation. 
        
        Parameters
        ----------
        rotation : An :py:class:`~orientation` object.
             Also it can refer to a function :py:func:`~angle`.


        Examples
        --------
        >>> cubicLat = Lattice.cubic(1)
        >>> vec = [0.,0.,1.]
        >>> dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        >>> dir2 = copy.deepcopy(dir1)
        >>> ori = Orientation(axis=[0.,1.,0.], degrees=-45)
        >>> dir1.rotate(ori)
        >>> np.allclose(dir1.getUVW(),[0.707, 0.0, 0.707],3)  
        True
        
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [2,-1,-1,0]
        >>> dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        >>> dir2 = copy.deepcopy(dir1)
        >>> ori = Orientation(euler=[30.*np.pi/180,0.,0.])
        >>> dir1.rotate(ori)
        >>> np.allclose(dir1.getUVW(force3Index=True),[1.73, -1.73, 0],3)  
        True          
        """
        if isinstance(rotation, Orientation):
            v = self.getCartesianVec()
            v = rotation.rotate(v)
            vector = self.lattice.get_fractional_coords(v)
            self.vector = np.array(vector)
            self._cartesianVector = v
            self._symmetricSet=None ## these are set to none as the original cvector was rotated and old
            self._symVecsCartesian=None ## values of these two varaible are no longer valid
            
            
        else:
            raise TypeError("MillerDirection can be rotated only if an Orienation object is provided but was provided an object of type"+type(rotation))
        
            
    def __add__(self, other):
        
        if isinstance(other, MillerDirection):
            if other.lattice==self.lattice :
                return self.__class__(vector=self.vector + other.vector, lattice=self.lattice)
            else:        
                raise ValueError("Two MillerDirections of different lattices can't be added")

    # Negation
    def __neg__(self):
        return self.__class__(vector= -self.vector, lattice = self.lattice)


    def __iadd__(self, other):
        return self + other

    def __radd__(self, other):
        return self + other

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __eq__(self, other):
        
        """
        Returns true if the following is true for each element of the MillerDirections
        absolute(a - b) <= (atol + rtol * absolute(b))`        
        and also the self and the other must be of same lattice 
        """
        if isinstance(other, MillerDirection):
            r_tol = 1.0e-13
            a_tol = 1.0e-14
            try:
                isEqual_1 = np.allclose(self.vector, other.vector, rtol=r_tol, atol=a_tol)
                isEqual_2 = self.lattice==other.lattice
            except AttributeError:
                raise AttributeError("Error in internal MillerDirection representation means it cannot be compared like a numpy array.")
            return (isEqual_1 & isEqual_2)
        return self.__eq__(self.__class__(other))   
    
    
    
    def __lt__(self, other):
        return self.mag<other.mag
    
    def dot(self, other):  
        
        """
        Dot product of the given MillerDirection (any lattice type) with the current one
        
        Parameters
        ----------
        other : Another MillerDirection Object with which the dot product is sought.
        
        Examples 
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> cubLat = Lattice.cubic(1)
        >>> vec1 = [2,-1,-1,0]
        >>> vec2 = [0,0,1]
        >>> a = MillerDirection(lattice=hexLat, vector = vec1, isCartesian=False, MillerConv="Bravais")
        >>> b = MillerDirection(lattice=cubLat, vector = vec2, isCartesian=False, MillerConv="Miller")
        >>> dotProd = a.dot(b)
        >>> np.allclose(dotProd,0.0)
        True   
        """
        if isinstance(other, MillerDirection):
            v1 = self.getCartesianVec()
            v2 = other.getCartesianVec()            
            return dot(v1,v2)
            
        else :
            raise TypeError("dot : function was supplied with non MillerDirection Object")
        
    def cross(self, other, latChoice='self'): 
        
        """
        To find the cross product of two :py:class:`~MillerDirection` objects. Please find the example given below for usage.
        
        Parameters
        ----------
        other : :py:class:`~MillerDirection`  object 
            :py:class:`~MillerDirection` object which needs to be crossed with self.
        latChoice : [optional] 
            Sefault is 'self' if specified as 'other' the returned :py:class:`~MillerDirection` shall be in 
            lattice frame of 'other' :py:class:`~MillerDirection`  object.
        
        Returns
        -------
        out : :py:class:`~MillerDirection` object.
            :py:class:`~MillerDirection` object which is orthogonal to both input vectors.
        
        Examples
        --------
        >>> cubicLat = Lattice.cubic(1)
        >>> orthoLat = Lattice.orthorhombic(1,1,1)
        >>> vec1 = [1.,0.,0.]
        >>> dir1 = MillerDirection(lattice=orthoLat,vector=vec1)
        >>> vec2 = [0.,1.,0.]
        >>> dir2 = MillerDirection(lattice=orthoLat,vector=vec2)
        >>> cross1 = dir1.cross(dir2)
        >>> np.allclose(cross1.getUVW(force3Index=True),[0.0, 0.0, 1.0])
        True
        
        >>> orthoLat = Lattice.orthorhombic(1, 1, 1)
        >>> vec = [0.,1.,0.]
        >>> dir1 = MillerDirection(lattice=orthoLat,vector=vec)
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [0,0,0,1]
        >>> dir2 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        >>> cross1 = dir1.cross(dir2,'other')
        >>> cross2 = dir1.cross(cross1)
        >>> np.allclose(cross1.dot(cross2),0.)
        True
        """
        if isinstance(other, MillerDirection):
            v1 = self.getCartesianVec()
            v2 = other.getCartesianVec()            
            v3 = np.cross(v1,v2)
            if 'other' in latChoice :
                return MillerDirection(lattice=other.lattice, vector=v3,isCartesian=True)
            else:
                return MillerDirection(lattice=self.lattice, vector=v3,isCartesian=True)                
            
        else :
            raise TypeError("dot : function was supplied with non MillerDirection Object")
      
    def getDirectionInOtherLatticeFrame(self,otherLattice):
        
        """
        Returns equivalent MillerDirection in lattice frame of another Lattice
        
        Examples
        --------
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> cubLat = Lattice.cubic(1)
        >>> vec1 = [2,-1,-1,0] 
        >>> a = MillerDirection(lattice=hexLat, vector = vec1, isCartesian=False, MillerConv="Bravais")  
        >>> c = a.getDirectionInOtherLatticeFrame(cubLat)
        >>> np.allclose(c,[3.,0.,0.],3)
        True                                                              
        """ 
        
        if isinstance(otherLattice, Lattice):
            if self.lattice == otherLattice :
                return self
            else:
                v = self.getCartesianVec()
                return MillerDirection(lattice=otherLattice, vector=v,isCartesian=True)
        else:
            raise TypeError("getDirectionInOtherLatticeFrame : lattice object needs to be specified")
                
    def hexToRhombohedral(self):
        print ("Warning method still not perfect !!!! soon shall be perfected !!!!")
        if self.lattice.is_hexagonal():
            a = self.lattice.a
            c = self.lattice.c
            
            a1 = self.lattice.get_cartesian_coords([2./3,1./3,1./3])
            a2 = self.lattice.get_cartesian_coords([1./3,2./3,2./3])
            a3 = self.lattice.get_cartesian_coords([-2./3,-1./3,2./3])
            
            a_rhombo = 1./3*sqrt(3*a*a+c*c)
            alpha = 2*math.asin(3/(2*sqrt(3+(c/a)*(c/a))))*180/math.pi
            lat = lt([[a1],[a2],[a3]])
            return self.getDirectionInOtherLatticeFrame(lat)
        else:
            raise ValueError("Direction from Non hexagonal lattice was asked tobe converted to rhobmohedral ")
            
    def hexToOrthorhombic(self):
        """
        Retunr the Equivalent MillerDirection object in Orthorhombic frame for a hexagonal direction
        """
        print ("Warning method still not perfect !!!! soon shall be perfected !!!!")
        if self.lattice.is_hexagonal():
            raise NotImplementedError()                      
            
            return None
        else:
            raise ValueError("Direction from Non hexagonal lattice was asked tobe converted to rhobmohedral ")
            
    def getOrthoset(self,returnCartesian=False): #nachiket 4-12-17
        
        """
        returns a list of 3  mutually perpendicular MillerDirections, making the current one as the first of the three
        
        Examples
        --------
        >>> cubLat = Lattice.cubic(1)
        >>> vec = [1,1,1]
        >>> dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        >>> perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
        >>> cross1 = perp1.cross(dir1)
        >>> dot1 = perp1.dot(dir1)
        >>> dot2 = cross1.dot(dir1)
        >>> np.allclose([dot1, dot2],[0.,0.])
        True
        """ 
        v1 = self.getUnitVector()
        v2 = self.getPerpendicularDirection(returnCartesian=True)
        v3 = np.cross(v1, v2)
        if returnCartesian:
            return [v1,v2,v3]
         
        return [MillerDirection(lattice = self.lattice, vector=v1,isCartesian=True),
                MillerDirection(lattice = self.lattice, vector=v2,isCartesian=True),
                MillerDirection(lattice = self.lattice, vector=v3,isCartesian=True)]
               
    def getPerpendicularDirection(self,returnCartesian=False): #nachiket 27/11/17 
        
        """
        An arbitrary perendicular MillerDirection object to current MillerDirection object will be returned.
        
        Parameters
        ----------
        returnCartesian : [optional] boolean
            If true, the returned vector is a numpy array of cartesian vector; else, MillerDirection object is returned.
        
        Examples
        --------
        >>> cubLat = Lattice.cubic(1)
        >>> vec = [1,1,1]
        >>> dir1 = MillerDirection(lattice=cubLat, vector = vec, MillerConv="Miller")
        >>> perp1 = dir1.getPerpendicularDirection(returnCartesian=False)
        >>> perp2 = dir1.getPerpendicularDirection(returnCartesian=False)
        >>> cross1 = perp1.cross(perp1)
        >>> cross2 = dir1.cross(cross1)
        >>> np.allclose(cross2.getMag(),0.)
        True
        """ 
        v = self.getCartesianVec()
        v = v/np.linalg.norm(v)
        vOrtho = np.random.random(3)
        vOrtho -= vOrtho.dot(v)*v
        vOrtho /=np.linalg.norm(vOrtho)
        if returnCartesian :
            return vOrtho
        return MillerDirection(lattice = self.lattice, vector=vOrtho,isCartesian=True)
              
    def integerize(self,AngleDeviationUnits='rad'):
        
        """
        To integerize the floating point miller direction arrays. Integerizing the direction will help in easy visualization of directions. 
        Additionally, it also provides the angular deviation between the integerized array and input array.
        
        Parameters
        ----------
        AngleDeviationUnits : [string], default is in radians
            It provides the option to get the output angular deviation in either degrees 'deg' or radians 'rad'. 
        
        Return
        ------
        integerizedDir : [int numpy array] size of array equal to input array size.
            Nearest integer equivalent of floating point miller directions.
        angleDeviation : [float] 
            Value of angular deviation between the integerized miller direction and input direction.
            
        Examples
        ------
        >>> cubicLat = Lattice.cubic(1)
        >>> vec = [-0.25,-0.5,0.75]
        >>> dir1 = MillerDirection(lattice=cubicLat, vector = vec)
        >>> [vec, AngleError] = MillerDirection.integerize(dir1)
        >>> np.allclose(vec, [-1.0, -2.0, 3.0])
        True   
        
        >>> hexLat = Lattice.hexagonal(1, 1.63)
        >>> vec = [1.98,-1.05,-1,0]
        >>> dir3 = MillerDirection(lattice=hexLat, vector = vec)
        >>> [vec, AngleError] = MillerDirection.integerize(dir3,AngleDeviationUnits='Deg')
        >>> np.allclose(vec, [2, -1, -1, 0])  
        True
        """
        intDirection = pmt.integerize(self.getUVW())
        newDirection = MillerDirection(self.lattice,pmt.integerize(self.getUVW(force3Index=True)))     
           
        angleDeviation = self.angle(newDirection)
        if 'deg' in AngleDeviationUnits.lower():
            angleDeviation *=180.0/math.pi
        return intDirection , angleDeviation
    
    def getLatexString(self,forceInteger=True):
        """
        Returns nicely formatted string for use in graphics.
        """
        if forceInteger :
            [v, err] = self.integerize()
        else:
            v = self.gethkl()
            v = v.astype(int) #### be careful it can ruin the original values in case the value are not really integers shoul be used only for the display purpose
        s=r'$[ '    
        for i in range(len(v)) :
            if v[i]<0. :
                s+=(r'\bar{'+str(np.abs(v[i]))+r'} ')
            else:
                s+=(str(v[i])+' ')
        s+=r']$'
        return (s)
        
    def symmetricSet(self, returnCartesianVecs=False):
        """
        method to get  the symmetric set considering the Symmetry of the lattice
        Input :
        -------
        returnCartesianVecs (optinal) : defualt = Flase , if True returns the numpy array of symmetric cartesian vectors 
        
        """
        if self._symmetricSet==None :
            originalVec = self._cartesianVector
            newVecs = np.zeros((self.lattice._NumberOfSymmetryElements,3), dtype=np.float)
            
            for i, symElement in enumerate(self.lattice._SymmetryElements):
                newVecs[i] = symElement.rotate(originalVec)
            symVecs = pmt.uniqueRows(newVecs, thresh=1e-3) 
            self._symVecsCartesian=symVecs
            symSet=[]
            
            for i, vec in enumerate(symVecs):
                symSet.append(MillerDirection(vector=vec, isCartesian=True, lattice=self.lattice)) 
            self._symmetricSet = symSet
            self._multiplicity = len(symSet)
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
    
    def isSymmetric(self, other, tol=1e-3):
        """ Test of obj1 and obj2 are symmetrically equivalent
        
        """
        if not isinstance(other, MillerDirection):
            raise ValueError("the other object must be of same type i.e. millerPlane or MillerDirection")
        if not self.lattice==other.lattice :
            raise ValueError ("Only objects of same lattice can be compared for symmetric equivalance")
        
        source = self._cartesianVector
        otherSet = other.symmetricSet(returnCartesianVecs=True)
        for i in otherSet :
            if np.allclose(source, i, atol=tol, rtol=1e-3):
                return True
        return False
            
    @staticmethod
    def uniqueList(dirList, considerSymmetry=True, returnInds = False):
        """
        Given a dir list, returns the uniqe MillerDirection list 
        Input
        -----
        considerSymmetry : deafult True, if False, does not consider symmetrically equivalence in detrmining the uniquenss of dirs  
        """
        tmpDirList = [(i,Dir) for i, Dir in enumerate(dirList)]
        ids = []
        numOfDeletions=0
        
        if isinstance(tmpDirList, list):
            uniqueList=[]
            #numOfPlanes=len(tmpPlaneList)
            for i, millerDir in enumerate(tmpDirList):
                uniqueList.append(millerDir[1])
                if returnInds :
                    ids.append(millerDir[0]) 
                delList = []
                for j in range(i+1,len(tmpDirList)):
                    #if considerSymmetry:                    
                        if millerDir[1].isSymmetric(tmpDirList[j][1]):                        
                            delList.append(j)
                        
                numOfDeletions+=len(delList)           
                for k in reversed(delList):
                    del tmpDirList[k]
                
                
            if returnInds:    
                return uniqueList, ids
            else:
                return uniqueList
        else:
            raise ValueError("only a List of dirs must be input !!! ")
    
    
    @staticmethod
    def objectiveFunctionFindTilts(tilts,currentZoneAxis, targetZoneAxis,alphaAxis=[1.,0.,0.], betaAxis=[0.,1.,0.],needExactTarget=False,fixedAxes=True,alphaRotationFirst=True):
        alphaTilt = tilts[0]
        betaTilt = tilts[1]
        tmpSourceVector=copy.deepcopy(currentZoneAxis)
        sourceZ = MillerDirection(vector=tmpSourceVector.getUnitVector(returnCartesian=True),isCartesian=True, lattice=currentZoneAxis.lattice)
        options = {"fixedAxes":fixedAxes,"alphaRotationFirst":alphaRotationFirst}
        sourceZ = MillerDirection.applyHolderTiltsToDirection(sourceZ,alphaTilt,betaTilt,alphaAxis,betaAxis,options=options)
        if needExactTarget:
            err = sourceZ.angle(targetZoneAxis,considerSymmetry=False,units="Deg")
        else:
            err = sourceZ.angle(targetZoneAxis,considerSymmetry=True,units="Deg")
        return err

    @staticmethod
    def rotationMatrixForHolderTilts(alphaTilt,betaTilt,alphaAxis=[1.,0,0.],betaAxis=[0.,1.,0],options={}):
        """
        returns the equivalnet rotation (orientation to be applied) to account for the holder tilts
        """
        defaultOptions = {"fixedAxes":True,"alphaRotationFirst":False, "activeRotation":True}
        
        if isinstance(alphaAxis,MillerDirection):
            alphaAxis = alphaAxis.getUnitVector()
        if isinstance(betaAxis,MillerDirection):
            betaAxis = betaAxis.getUnitVector()
        
        if "fixedAxes" in options:
            fixedAxes=options["fixedAxes"]
        else:
            fixedAxes = defaultOptions["fixedAxes"]
        if "activeRotation" in options:
            activeRotation = options["activeRotation"]
        else:
            activeRotation  = defaultOptions["activeRotation"]
        
        if "alphaRotationFirst" in options:
            alphaRotationFirst=options["alphaRotationFirst"]
        else:
            alphaRotationFirst = defaultOptions["alphaRotationFirst"]
       
        if fixedAxes:
            alphaRotation = Orientation(axis = alphaAxis,degrees= alphaTilt)
            betaRotation = Orientation(axis =betaAxis,degrees= betaTilt)
            if alphaRotationFirst:                
                totalRotation =betaRotation*alphaRotation
            else:
                totalRotation =alphaRotation*betaRotation
        else:
            raise ValueError ("At the moment only fixed axes rotation is implemented.")
        
        if activeRotation:
            return totalRotation
        else:
            return totalRotation.inverse
    
    
    @staticmethod
    def applyHolderTiltsToDirection(direction,alphaTilt,betaTilt,alphaAxis=[1.,0,0.],betaAxis=[0.,1.0],options={}):
        """
        general method to apply holder tilts to any direction 
        direection a numpy array of (3,) size or a MillerDirection object or a list of 3 numbers indicating the vector
        alphaTilt, betaTilt are in degrees
        alphaAxis,betaAxis are the axes about which the rotatio must be performed.
        optons is a dictionary
        
        """
            
        totalRotation=MillerDirection.rotationMatrixForHolderTilts(alphaTilt,betaTilt,alphaAxis,betaAxis,options)
        
        if isinstance(direction, (MillerDirection,)):
            tmp = copy.copy(direction)
            tmp.rotate(totalRotation)            
            return tmp
        elif isinstance(direction, (np.ndarray,)):
            if direction.size==3:
                return totalRotation.rotate(direction) 
            else:
                raise ValueError ("numpy arrayu was supplied as direction but has size not equal to 3")
        else:
            raise ValueError ("Supplied directio was neither MillerDirection nor np array representing a 3d vector !!!!")
            

    
    
    
    def steriographiProjection(self,orientation=Orientation.stdOri()):  
        """
        Returns the steriographic Projection coordinates
        Input: CrystalOrienation default: Orientation(0,0,0)
        """ 
        
        vec = orientation.rotate(self.getUnitVector())
        vec = vec/np.linalg.norm(vec)
        
        sterioPoint = np.array([0.,0.])
        isNorthPole=True
        #line1 = [[0.,0.,0],[0.,0.,1]] ### z axis
        if np.abs(1+vec[2])>1e-5:
            if vec[2]>=0:
                sterioPoint = vec[0:2]/(1+vec[2]) 
                               
            else:
                sterioPoint = vec[0:2]/(1-vec[2])
                #line1 = [[0.,0.,0],[0.,0.,-1]] ### z axis
                isNorthPole=False
        
        #line2 = [[0.,0.,0],vec.tolist()]
        ##ref : MarcDegraf book pg 30
        r=90*np.sqrt(sterioPoint[0]*sterioPoint[0]+sterioPoint[1]*sterioPoint[1])
        phi = np.arctan2(vec[1],vec[0])*180./np.pi
        
        betaTilt = np.arcsin(np.clip(vec[0],-1,1))
        if np.allclose(np.cos(betaTilt),0.):
            warnings.warn("The case of beta tilt being 90 the vector is :"+str(vec)+" and sterio point is :"+str(sterioPoint) +"beta Tilt="+str(betaTilt*180/np.pi))
            alphaTilt=0.
        else:
            tmp = -(vec[1]/np.cos(betaTilt))
            alphaTilt = 180./np.pi*np.arcsin(np.clip(tmp,-1,1))
        betaTilt = 180./np.pi*betaTilt
        
        polarPoint = [phi,r]
        return  sterioPoint, polarPoint, isNorthPole,alphaTilt, betaTilt   
        
    @staticmethod
    def generateDirectionList(uvwMax, lattice,includeSymEquals=False):
        """
        Method to genrate a list ofhkl planes which are symmetrically unique and sorted by 
        highest dspacing plane folowed by planes with lower d-spacings
        Input:
        -----
            hklMax : an integer specifying the max index of hkl
            lattice : :py:class:`~OrientedLattice` object specifying the lattice of the MillerPlanes to be genrated 
        """
        a = range(uvwMax+1)
        uvwList = []
        isHexagonal = lattice.is_hexagonal()
        if includeSymEquals:
            a = range(-(uvwMax),uvwMax+1)
            for combination in itertools.product(a, a, a):            
                uvwList.append(combination)
            uvwList.remove((0,0,0))
            uvwDirlist=[]
            for i, uvw in enumerate(uvwList):
                if isHexagonal:
                    direction=MillerDirection(lattice=lattice,vector=[uvw[0],uvw[1],-(uvw[0]+uvw[1]), uvw[2]])
                else:
                    direction=MillerDirection(lattice=lattice,vector=uvw)
            
                uvwDirlist.append(direction) 
            sortedList = sorted(uvwDirlist , key=lambda x: x._mag)
            return sortedList
        
        for combination in itertools.product(a, a, a):            
            uvwList.append(combination)    
        del uvwList[0]; ## this is (0,0,0) plane and hence being removed 
        uvwDirlist=[]
        
        for i, uvw in enumerate(uvwList):
            direction=MillerDirection(lattice=lattice,vector=uvw)
            #print("Here is the issue")
            uvwDirlist.append(direction)                
        sortedList = sorted(uvwDirlist , key=lambda x: x._mag)
        result = MillerDirection.uniqueList(sortedList)
        return result   
    
   
        
if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All tests are done")    

            
            
                                                                                                   