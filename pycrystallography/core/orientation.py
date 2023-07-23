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



class Orientation(Quaternion, MSONable):
    """
    A wrapper class for quartenion package for easy mnaipulation of quaternions in terms of euler angles as used in the feild of materials science. Essentially a 3 element euler angle matrix. In general it is of Bunge notation only. All angles are in radians only. In case degree is to be used it must be specified. Other notations are also possible by explicit mentioning the same.
    """
    
    def __init__(self, *args, **kwargs):
        """Create an orientation from various possible inputs. These include::
         
            1]. euler angles: (in the form of numpy array of (3,) size, or list of 3 elements, or a tuple of size 3)
            2]. axis-angle pairs
            3]. Another Orienation Object
            4]. Another Quaternion Object         
    

        Parameters
        ----------            
        Euler_Angles : [optional] sequence of 3 numbers 
            It can be numpy array of (3,) size, or list of 3 elements, or a tuple of size 3.
        Angle: [optonal] Angle of rotation
            Angle in radians, if degrees is desired use keyword "degrees="
        Axis : [optonal] Axis of rotation if Angle of rotaion is specified
            Axis a tuple, or list or np array of 3 elements (axis need not be normalized one).
        Orientation : 
            an exisitng Orienation object.
        Quaternion : 
            an exisitng Quaternion object.
    
        Returns
        -------
        out :  An Orientation object
        The result is represented as a `Orientation` object.
        
        Examples
        --------                    
        >>> ori = Orientation(euler=[0,0,0])
        >>> isinstance(ori,Orientation)
        True
        
        >>> 
        >>> ori = Orientation(axis=[1,0,0],angle=math.pi/2)
        >>> isinstance(ori,Orientation)
        True
        >>> 
        
        >>> ori1 = Orientation(euler=[pi/2,0,0])
        >>> ori2 = Orientation(ori1)
        >>> ori1==ori2
        True
        >>> 
        >>> q1 = Quaternion(axis=[1,1,1],degrees=90)
        >>> ori1 = Orientation(q1)
        >>> isinstance(ori1,Orientation)
        True
        >>> 

        
        """
        #self._isRotationFlipped=False
        s = len(args)
        if s == 0:
            # No positional arguments supplied
            if len(kwargs) > 0:
                # Keyword arguments provided
                if ("euler" in kwargs) or ("Euler" in kwargs):
                    euler = kwargs.get("euler", [0.0,0.0,0.0])
                    if euler is None:
                        euler = [0.0, 0.0, 0.0]
                    eulerAngles = np.array(euler) # converting into np array so that even lists can be passesdasargumentsincreatingthi    object.
                    
                    
                    rotMat=pt.rotMatrix_from_euler_ebsd(eulerAngles[0],eulerAngles[1],eulerAngles[2])
                    q = Quaternion(matrix=rotMat)                    
                    Quaternion.__init__(self, q)

#                 elif ("axis" in kwargs) or ("radians" in kwargs) or ("degrees" in kwargs) or ("angle" in kwargs):
#                     try:
#                         axis = self._validate_number_sequence(kwargs["axis"], 3)
#                     except KeyError:
#                         raise ValueError(
#                             "A valid rotation 'axis' parameter must be provided to describe a meaningful rotation."
#                         )
#                     angle = kwargs.get('radians') or self.to_radians(kwargs.get('degrees')) or kwargs.get('angle') or 0.0
#                     Quaternion.__init__(self, angle=angle,axis=axis)
#                     #The above statment creates Quat object with angle in negative sense. Thuis was done to ensure 
#                     #consistency with EBSD conventions where roations are in negative sense to those auumed in Quaternion package.
#                     
                else :
                    q = Quaternion(*args, **kwargs)
                    Quaternion.__init__(self, q)
        elif s == 1:
            # Single positional argument supplied
            if isinstance(args[0], Orientation):
                #self.q = args[0].q
                Quaternion.__init__(self, args[0].q)
                return
            elif isinstance(args[0], Quaternion):
                #self.q = args[0].q
                Quaternion.__init__(self, args[0])
                return

            elif (args[0] is None) or (type(args[0]) is str):
                raise TypeError("Object cannot be initialised from " + str(type(args[0])))
            elif type(args[0]) is list or type(args[0]) is tuple :
                euler = self._validate_number_sequence(args, 3)
                #self.q = self._validate_number_sequence(args, 3)
                rotMat=pt.rotMatrix_from_euler_ebsd(euler[0],euler[1],euler[2])
                q = Quaternion(matrix=rotMat)                    
                Quaternion.__init__(self, q)
            else :
                raise ValueError("Single positional argument is provided which is neither an Orientation, Quaternion, or a string that can be converted to meaningful sequence of Euler angles or a list/ tuple of euler angles")

            return
        
        else: 
            # More than one positional argument supplied
            euler = self._validate_number_sequence(args, 3)
            #self.q = self._validate_number_sequence(args, 3)
            rotMat=pt.rotMatrix_from_euler_ebsd(euler[0],euler[1],euler[2])
            q = Quaternion(matrix=rotMat)                    
            Quaternion.__init__(self, q)
            return   
    
       
    def _validate_number_sequence(self, seq, n):
        """Validate a sequence to be of a certain length and ensure it's a numpy array of floats.

        Raises:
            ValueError: Invalid length or non-numeric value
        """
        if seq is None:
            return np.zeros(n)
        if len(seq) is n:
            try:
                l = [float(e) for e in seq]
            except ValueError:
                raise ValueError("One or more elements in sequence <" + repr(seq) + "> cannot be interpreted as a real number")
            else:
                return np.asarray(l)
        elif len(seq) == 0:
            return np.zeros(n)
        else:
            raise ValueError("Unexpected number of elements in sequence. Got: " + str(len(seq)) + ", Expected: " + str(n) + ".")
    
    @staticmethod
    def mapVector(sourceVector,targetVector):
        """
        method to create an orieantion object which maps the given Source vector onto a Target vector
        Parameters
        ----------
        sourceVector : np array (1X3)  
        targetVector : np array (1X3) to which sourceVetor must be mapped 
        
        Returns
        --------
        ori : An orientation object such that  targetVector = ori*sourceVetor 
            
        """
        s = sourceVector/np.linalg.norm(sourceVector)
        t = targetVector/np.linalg.norm(targetVector)
        theta = np.arccos(np.clip(np.dot(s,t),-1,1))
        vec = np.cross(s,t)
        if np.linalg.norm(vec) <1e-6:
            if np.abs(np.dot(s,t)+1.0)<1e-6: ####case of antiparllel vectors
                axis = pmt.perpendicularVector(s)
#                 print("Warning!!! Function: MapVector. Case of anti parllel vectors. Assuming the rotation \
#                 about 001 is enough. In case of unexpected results probelem could be from this part of the code.\
#                  Correct way would be to find a vector perpendicualr to the one of the source or target \
#                 vectors and use that as the axis and rotrate 180 about that")
                #print("output form map vectors !!!",s,t,axis)
                return Orientation(axis = axis,angle = np.pi)
            return Orientation(euler=[0.,0.,0])
        return (Orientation(axis=vec,angle=theta))
         
                
    def copy(self):
        """Deep copy of self."""
        return self.__class__(self.data.copy())

#     def __deepcopy__(self):
#         return self.__class__(self)
       
  
    def __repr__(self, **kwargs):
        
        eulerAngles = Orientation.getEulerAngles(self,units='deg')
        str = eulerAngles.__repr__()
        #str = str + '\n An oirenation object'
        return str
   
    def __str__(self, **kwargs):
        
        eulerAngles = np.around(Orientation.getEulerAngles(self,units='deg'),2)
        str = eulerAngles.__str__()
        return str   
    
    def __format__(self, formatstr):
        if formatstr.strip() == '': # Defualt behaviour mirrors self.__str__()
            formatstr = '.2f'

        string = \
            "[{:" + formatstr +"} "  + \
            "{:" + formatstr +"} " + \
            "{:" + formatstr +"}] " 
        eul = Orientation.getEulerAngles(self,units='deg')
        return string.format(eul[0], eul[1], eul[2])
       

#     def get_axis(self, undefined=np.zeros(3)):
#         """Get the axis or vector about which the quaternion rotation occurs
# 
#         For a null rotation (a purely real quaternion), the rotation angle will 
#         always be `0`, but the rotation axis is undefined. 
#         It is by default assumed to be `[0, 0, 0]`.
# 
#         Params:
#             undefined: [optional] specify the axis vector that should define a null rotation. 
#                 This is geometrically meaningless, and could be any of an infinite set of vectors, 
#                 but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.
# 
#         Returns:
#             A Numpy unit 3-vector describing the Quaternion object's axis of rotation.
# 
#         Note:
#             This feature only makes sense when referring to a unit quaternion. 
#             Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
#         """
#         self.angle
#         tolerance = 1e-17
#         self._normalise()
#         norm = np.linalg.norm(self.vector)
#         if norm < tolerance:
#             # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
#             return undefined 
#         else:
#             if self._isRotationFlipped :
#                 return -self.vector / norm
#             else:
#                 return self.vector / norm
# 
#     @property
#     def axis(self):
#         return self.get_axis()
# 
#     @property
#     def angle(self):
#         """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis. 
# 
#         This is guaranteed to be within the range (-pi:pi) with the direction of 
#         rotation indicated by the sign.
# 
#         When a particular rotation describes a 180 degree rotation about an arbitrary 
#         axis vector `v`, the conversion to axis / angle representation may jump 
#         discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`, 
#         each being geometrically equivalent (see Note in documentation).
# 
#         Returns
#         -------
#         res : A real number in the range `(-pi:pi)` describing the angle of rotation 
#             in radians about a Quaternion object's axis of rotation. 
# 
#         Note
#         ----
#         This feature only makes sense when referring to a unit quaternion. 
#         Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
#         """
#         self._normalise()
#         norm = np.linalg.norm(self.vector)
#         ang= self._wrap_angle(2.0 * np.arctan2(norm,self.scalar))
#         if ang<0:
#             self._isRotationFlipped=True
#             return abs(ang)
#         else:
#             return ang
#     
#     
    
    
    
    
    
    
    def getEulerAngles(self, units='radians',applyModulo=False):
        """ Return the Euler angles in Radinas as 1X3 numpy array. 
            
        Parameters
        ----------
        units : string, optional  
             if Deg or Degrees, retuned angles will be in Degrees.
        applyModulo : if True will return the modulo of [360 180 360] to ensure that 360 is treated as 0.

        Returns
        --------
        out : A Numpy array 
            Euler angles.

        Examples
        --------
        >>> ori1 = Orientation(axis=[1,0,0],degrees=90)
        >>> eulerAngs = ori1.getEulerAngles(units='degrees')
        >>> np.allclose(eulerAngs,np.array([0., pi/2,0.]))
        False
        >>> np.allclose(eulerAngs,np.array([0., 90.,0.]))
        True
        >>> 

        """
        eulerAngles = pt.eulerAngles_from_rotMat_ebsd(self.rotation_matrix)
        if applyModulo:
            eulerAngles = ((eulerAngles*180.0/np.pi).round(5)%[360.0,180.0,360.0])*np.pi/180 ### to make 360 as 0 and 180 as 0

        if units.lower() == 'degree' or units.lower() =='deg' or units.lower() == 'degrees':
            eulerAngles *= 180.0/pi  # converting to degree
        
        return eulerAngles
        
       
#     def get_axis(self, undefined=np.zeros(3)):
#         """Get the axis or vector about which the rotation occurs
#         
#         For a null rotation (a purely real quaternion or euler = [0,0,0]), the rotation angle will 
#         always be `0`, but the rotation axis is undefined. 
#         It is by default assumed to be `[0, 0, 0]`.
# 
#         Parameters
#         ----------
#         undefined: [optional]
#             specify the axis vector that should define a null rotation. 
#             This is geometrically meaningless, and could be any of an infinite set of vectors, 
#             but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.
# 
#         Returns
#         -------
#         out : A numpy array
#             A unit vector describing the Quaternion object's axis of rotation.
#         
# 
#         Note
#         -----
#         This method overrides the one from the base calss and returns a negative axis of the axis 
#         returned by underlying Quaternion object. This was done to ensure consistency with EBSD 
#         conventions where roations are in negative sense to those asumed in Quaternion package.
#             
#         """
#         tolerance = 1e-17
#         self._normalise()
#         norm = np.linalg.norm(self.vector)
#         if norm < tolerance:
#             # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
#             return undefined 
#         else:
#             return -(self.vector / norm) ## here - ve sign is added to make it consitenet with EBSD orienation notations
# 
#     @property
#     def axis(self):
#         """To access the axis of an object.
#         """
#         return self.get_axis()


    @classmethod
    def random(cls,n=1):
        """Generate  random Orientation.
         
        Parameters
        ----------
        n : An integer value.
            Number of random Orientations desired. If n>2, returns a list of random Orientation Objects.

        Returns
        -------
        out : ``orientation`` object(s). 
            List of 'n' Orientation objects. If n=1, a single Orientation object.

        Examples
        --------
        >>> a = Orientation.random(10)  
        >>> len(a)==10
        True
        >>>  
        >>> a = Orientation.random()
        >>> isinstance(a,Orientation)
        True
        >>>

        """
        pi_2 = 2*pi
        if n>2 :
            oriList = [None]*n
            for i in range(n) :
                Phi1, Phi, Phi2 = np.random.random(3)
                Phi1*=pi_2
                Phi*=pi
                Phi2*=pi_2
                oriList[i] = cls(Phi1,Phi,Phi2)
            return oriList


        Phi1, Phi, Phi2 = np.random.random(3)
        Phi1*=pi_2
        Phi*=pi
        Phi2*=pi_2
        return cls(Phi1,Phi,Phi2)

    def misorientation(self,other) :
        """
        Retun the Misorienation with the other orieantion.
          
        Parameters
        ----------
        other : ``orientation`` object 
            The Orienation object with the misorieantion is desired
          
        Returns
        -------
        out : ``orientation`` object 
            An Orientation object, representing the misorieantion between the slef, and other
          
        Examples
        --------
        >>> o2 = Orientation(axis=[0,0,1],angle=math.pi/2)
        >>> o1 = Orientation(axis=[0,0,1],angle=0)
        >>> o3 = o1.misorientation(o2)
        >>> o3.angle == math.pi/2
        True
        >>> 

        """
        if isinstance(other, Orientation):
            # mis of a,b i.e. delta (a,b) = inv(b)*a
#             misOri = np.dot(other.rotation_matrix.T,self.rotation_matrix)
#             return Orientation(matrix=misOri)
            return Orientation(other*self.inverse).inverse

    def misorientationAngle(self,other,units='radians') :
        """
          Retuns the Misorienation Angle with other orieantion in radians .

          Parameters
          ----------
            other : Orienation object
                The Orienation object with the misorieantion angle is desired
            units : [optional] 
                if 'degrees' of 'deg' the returned value will be in degree
          
          Returns
          -------
            angle : Floating value.
                Misorieantion angle.
          
          Examples
          --------
          >>> o2 = Orientation(axis=[0,0,1],angle=math.pi/2)
          >>> o1 = Orientation(axis=[0,0,1],angle=0)
          >>> ang = o1.misorientationAngle(o2,units='deg')
          >>> np.allclose(ang,90)
          True
          >>> 
          
        """
        misOri = Orientation.misorientation(self,other)
        angle = misOri.angle
        if units.lower()=="deg" or units.lower()=="degrees" :
            angle =misOri.degrees
        return angle
    
#     def rotate(self,vector,isPassiveRotation=True):
#         """
#         wraps the underlying Quaternion rotaiton function to make the rotation passive by default.
#          
#         """
#         if isinstance(vector, Quaternion):
#             return self._rotate_quaternion(vector)
#         q = Quaternion(vector=vector)
#         a = self._rotate_quaternion(q).vector
#         if isinstance(vector, list):
#             l = [x for x in a]
#             return l
#         elif isinstance(vector, tuple):
#             l = [x for x in a]
#             return tuple(l)
#         else:
#             return a
#        
        
 

    def misorientationAxis(self,other) :
        """
        Retuns the Misorienation Axis (a numpy array of shape (3,)) with other orieantion.
          
        Parameters
        ----------
        other : ``orientation`` object. 
            The Orienation object with which the misorieantion angle is desired.
                      
        Returns
        -------
        out : A numpy array of size(3,).
            A common axis about which misorientation of two orinetations exist. 
          
        Examples
        --------
        >>> o2 = Orientation(axis=[0,0,1],angle=math.pi/2)
        >>> o1 = Orientation(axis=[0,0,1],angle=0)
        >>> ang = o1.misorientationAngle(o2,units='deg')
        >>> np.allclose(ang,90)
        True
        >>> 
        """
        misOri = Orientation.misorientation(self,other)
        return  misOri.axis


    @staticmethod
    def mean(*args, **kwargs):
        """
        Returns the mean orieantion of the given set of orienations and weights (if given). otherwise, all Orientations are given
        same weight. 
        
        
        Parameters
        ----------
          listOfOrientations : list of N ``orientation`` objects.
              List of N number of orienations for which mean is sought.
          weights: [optional ] list of N floating values.
              weights for each of the orieantion, taken as 1/N if not given.
              
          or

          o1 :``orientation`` object
              first orieantion
          o2 :``orientation`` object 
              second Orienation
                    
        Returns
        -------
          mean : ``orientation`` object
              Mean Orientation of two or more Orientation objects.  

        Note 
        ----
        A better and faster method shall have to be implemented for performance reasons as the current one employs iterative
        and slow proceess of arriving at mean of the iontermediate orieantions.
        
        Examples
        --------
        >>> o1 = Orientation(axis=[1, 0, 0], angle=0.0)
        >>> o2 = Orientation(axis=[1, 0, 0], angle=2*pi/3)
        >>> num_intermediates = 2
        >>> list1 = list(Orientation.intermediates(o1, o2,num_intermediates, include_endpoints=True))
        >>> weight = 0.5
        >>> weightsList = [weight for i in range(len(list1))]
        >>> mean_o = Orientation.mean(o1,o2)
        >>> mean_o==Orientation(axis=[1, 0, 0], angle=pi/3)
        True
        >>> mean_list1 = Orientation.mean(list1)
        >>> mean_list2 = Orientation.mean(list1, weightsList)
        >>> np.allclose(mean_list1.getEulerAngles(), mean_list2.getEulerAngles())
        True        
        
        """
        s= len(args)
        if s==1 : #one position arg is given, most likey a list of Orientation objects, then we assume equal weights for all
            if isinstance(args[0],list) : 
                weightList = np.ones((len(args[0]),),dtype=float)*(1./len(args[0])) # creating equal weights
                meanOrieantion = Orientation._list_mean(args[0],weightList)
            else :
                raise ValueError("Only one positional argument is provided but it is not a list of Orienations.")
        if s==2 : ## case of list Oris and weights or two Orieantions
            if isinstance(args[0],list) and len(args[0]) ==len(args[1]):
                meanOrieantion = Orientation._list_mean(args[0],args[1])
            elif isinstance(args[0],Orientation) and isinstance(args[1],Orientation):
                meanOrieantion = Orientation.slerp(args[0], args[1], 0.5)
            else :
                raise ValueError("The number of orieantions in list is not matching with the weights or the two arguments supplied are not of Orienation type.")
        if s==3 :
            # three arguments were provided , the last one must be weight
            if isinstance(args[0], Orientation) and isinstance(args[1], Orientation)  and  (args[2]<=1. and args[2]>=0) :
                meanOrieantion = Orientation.slerp(args[0], args[1], args[2])

        return meanOrieantion

    @staticmethod
    def _list_mean(ori_list,weightList):
        """
        Helper method for finding the mean of the list of orieantions as per the weights given to them.
        """
        ws_1 = np.array(weightList)
        n = len(ori_list)
        if n==1 :
            return Orientation(ori_list[0])
        if n==2 :
            return Orientation.slerp(ori_list[0],ori_list[1],weightList[1])

        list_1 = ori_list
        list_2 = [None]*(n-1)        
        if len(ws_1) ==n :
            ws_1 = ws_1/sum(ws_1) # normalizing the weights.
            ws_1 = weightList.copy()
                      
            while(n>=3):            
                list_2 = [None]*(n-1)
                ws_2 = np.zeros((n-1,),dtype=float)      
                
                for i in range(n-1):
                    ws_2[i] = ws_1[i+1]/(ws_1[i]+ws_1[i+1])
                    list_2[i] = Orientation.slerp(list_1[i],list_1[i+1],ws_2[i])
                list_1 = list_2
                ws_2 = ws_2/sum(ws_2)
                ws_1 = ws_2.copy()
                n = len(list_1)
                
                mean_q = Orientation.slerp(list_1[0],list_1[1],ws_1[1])           
           
            return Orientation(mean_q)
        else :
            raise ValueError("The No Of Orientations and Weights are not matching")

        
    @staticmethod 
    def objectiveFunctionFindTilts(tilts,sourceVector,targetVector):
        alphaTilt = tilts[0]
        betaTilt = tilts[1]
#         sourceVector = np.array([0,0,1])
#         targetVector = np.array([0,1,1])
        alphaRotation = Orientation(axis=[1,0,0],radians=alphaTilt)
        betaRotation = Orientation(axis=[0,1,0],radians=betaTilt)
        totalRotation  = alphaRotation*betaRotation
        obtainedVector = totalRotation.rotate(sourceVector)
        targetVector = targetVector/np.linalg.norm(targetVector)
        obtainedVector = obtainedVector/np.linalg.norm(obtainedVector)
        err = np.arccos(np.clip(np.dot(obtainedVector,targetVector),-1.,1.))*180/np.pi
        return err

    @staticmethod
    def stdOri():
        return Orientation(euler=[0.,0.,0])
    
    @staticmethod
    def fromTwoOrthoDirections(vec1,vec2,vectorsParallelTo):
        """
        creates an Orienation object from the specification of two mutually orthogonal directions
        (numpy arrays) which lie parall to the cartesian X,Y or Z axes.
        vectorsParallelTo: a strin specifying the what cartesian Axes the vec1 and vec2 are parllel to
        
        e.g. vectorsParallelTo='XY',or 'XZ'
        """
        vec1 = vec1/np.linalg.norm(vec1)
        vec2 = vec2/np.linalg.norm(vec2)
        
        if np.abs(np.dot(vec1,vec2))>1e-5:
            raise ValueError(f'The supplied vectors are not Orthogonal!!! Their angles are :{np.arccos(np.dot(vec1,vec2))*180/np.pi} degree')
        
        if 'xy' in vectorsParallelTo.lower():   ###vec1 =x, vec2=y, vec3=z         
            vec3 = np.cross(vec1,vec2)
            matrix = np.array([vec1,vec2,vec3])
            
        elif 'xz' in vectorsParallelTo.lower(): ###vec1 =x, vec2=z, vec3=y   
            vec3 = np.cross(vec2,vec1)
            vec3 = vec3/np.linalg.norm(vec3)
            matrix = np.array([vec1,vec3,vec2])
        else:
            raise ValueError('Unknown option. Only xy,xz are supported now')         
        return (Orientation(matrix=matrix.T))   
    
if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All tests are done")