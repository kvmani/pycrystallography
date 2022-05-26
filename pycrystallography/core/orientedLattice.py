'''
Created on 06-Dec-2017

@author: Admin
'''
from __future__ import division, unicode_literals
import math
import itertools
import warnings
import numpy as np
from numpy.linalg import inv
from numpy import pi, dot, transpose, radians
from pycrystallography.core.quaternion  import Quaternion  
#import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pytransformations as pt
import pycrystallography.utilities.pymathutilityfunctions as pmt 
from pymatgen.core.lattice import Lattice 
from pycrystallography.core.orientation  import Orientation
from monty.json import MSONable
from monty.dev import deprecated
import spglib as spg
import os
import itertools
import math
import subprocess
from monty.serialization import loadfn
import pymatgen.core as mg

module_dir = os.path.dirname(os.path.abspath(__file__))
#print("Yes I am being executed for readinf the symmetry opertors")
symOperators = loadfn(os.path.join(module_dir, "symmetryOperators.yaml"))

latticeTypeCode = {'cubic':1,
                'tetragonal':2,
                'orthorhombic':3,
                'monoclinic':4,
                'hexagonal':5,
                'rhombohedral':6,
                'triclinic':7}

class OrientedLattice(Lattice, MSONable):
    '''
    Lattice with Orieantion in euler angles. This allows lattices of arbitary Orientation to be genrated.
    It also stores all the rotational symmetry elements of the Lattice in the form of numpy array of NX3X3 
    matrix of where N is the number of rotational symmetry operators for the lattice.
    '''


    def __init__(self, matrix,name='Alpha',symbol='$\alpha$',  orientation=Orientation(euler=[0,0,0]), pointgroup=None):
        '''
        Constructor
        '''
        if isinstance(orientation, Orientation) :
            #orientation = Orientation.inverse
            self._orientation = orientation.inverse
        else:
            raise ValueError('orientation needs to be Orientation Object'+
                             'but was provided'+type(orientation))
        
        rotatedMatrix = np.zeros((3,3))
        ### this needs to be confirmed once for all wether we should 
        #### take inverse of the input orienation or not?
        for i in range(3) :
            rotatedMatrix[i] = orientation.rotate(matrix[i])
#         l = Lattice(matrix=rotatedMatrix)
        Lattice.__init__(self, matrix = rotatedMatrix)
        
        if pointgroup is None :
            spgLattice = np.array(rotatedMatrix) 
            fakeAtomPosition = [[0.,0.,0.]]
            atomNumbers= [1,]
            spgCell = (spgLattice, fakeAtomPosition, atomNumbers)
            self._symmetryData = spg.get_symmetry_dataset(spgCell, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)
            pointgroup = self._symmetryData['pointgroup']
            #print("the hall number is ", self._symmetryData['hall_number'])
        self._pointgroup = pointgroup
        self._latticeCode = None ### 1 for cubic, 2 for tetragonal etc
        temp = symOperators[pointgroup]
        self._latticeType = temp['latticeType']        
        self._NumberOfSymmetryElements = temp['NumberOfSymmetryElements'] ### set of rotational symmetry operators
        symElems = np.asarray(temp['SymmetryElements'])
        assert(np.allclose(symElems.shape, (self._NumberOfSymmetryElements,3,3)))
        #print(temp.shape, temp)
        symmetryList=[]
        for i, mat in enumerate(symElems) :
            symmetryList.append(Orientation(matrix=mat))
        
        self._EulerLimits = np.asarray(temp["EulerLimits"])*np.pi/180
        self._SymmetryElements = symmetryList
        self._name=name
        self._symbol=symbol 
        
       
#     def __deepcopy__(self):
#         return self.__class__(self)

    
    
    
    def __str__(self):
        outs= super().__str__()
        N = len(self._SymmetryElements)
        name=self._name
        pg = self._pointgroup
        List = []
        List.append(outs)
        
        s = [outs+"\n"+name+"\npoint group " +"".join(pg),
             "Lattice Type : "+"".join(self._latticeType),
             "Number of Rotational Symmetries : "+"".join("%3d" % N),
             "Lattice Orieantation : "+"".join(str(self._orientation))
            ]
        return("\n".join(s))
    
    @staticmethod
    def cubic(a,orientation=Orientation(euler=[0.,0.,0.]), pointgroup='m-3m',name='Beta',symbol='$\beta'):
        """
        Convenience constructor for a cubic lattice.

        Input:
        ------
            a (float): The *a* lattice parameter of the cubic cell.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
        --------
            Cubic lattice of dimensions a x a x a.
        """
        
        lat =  OrientedLattice([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
                               orientation=orientation, pointgroup=pointgroup,name=name,symbol=symbol)
   
        lat._latticeCode = 1
        #lat._symmetrySet = 
        return lat
    
    @staticmethod
    def tetragonal(a, c, orientation=Orientation(euler=[0.,0.,0.]), pointgroup='4/mmm',name='Alpha',symbol='$\alpha$'):
        """
        Convenience constructor for a tetragonal lattice.

        Input:
        ----
            a (float): *a* lattice parameter of the tetragonal cell.
            c (float): *c* lattice parameter of the tetragonal cell.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
        --------
            Tetragonal lattice of dimensions a x a x c.
        """
        lat = OrientedLattice.from_parameters(a, a, c, 90, 90, 90, orientation=orientation, pointgroup=pointgroup,name=name,symbol=symbol)
        lat._latticeCode = 2
        return lat

    @staticmethod
    def orthorhombic(a, b, c, orientation=Orientation(euler=[0.,0.,0.]), pointgroup='mmm',name='Alpha',symbol='$\alpha$'):
        """
        Convenience constructor for an orthorhombic lattice.

        Input:
        -----
            a (float): *a* lattice parameter of the orthorhombic cell.
            b (float): *b* lattice parameter of the orthorhombic cell.
            c (float): *c* lattice parameter of the orthorhombic cell.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
        -------
            Orthorhombic lattice of dimensions a x b x c.
        """
        lat = OrientedLattice.from_parameters(a, b, c, 90, 90, 90,orientation=orientation, pointgroup=pointgroup,name=name,symbol=symbol )
        lat._latticeCode = 3
        return lat
        

    @staticmethod
    def monoclinic(a, b, c, beta, orientation=Orientation(euler=[0.,0.,0.]), pointgroup='2/m',name='Alpha',symbol='$\alpha$'):
        """
        Convenience constructor for a monoclinic lattice.

        Input:
        ------
            a (float): *a* lattice parameter of the monoclinc cell.
            b (float): *b* lattice parameter of the monoclinc cell.
            c (float): *c* lattice parameter of the monoclinc cell.
            beta (float): *beta* angle between lattice vectors b and c in
                degrees.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
        --------
            Monoclinic lattice of dimensions a x b x c with non right-angle
            beta between lattice vectors a and c.
        """
        lat = OrientedLattice.from_parameters(a, b, c, 90, beta, 90, orientation=orientation, pointgroup=pointgroup,name=name,symbol=symbol)
        lat._latticeCode = 4
        return lat
        

    @staticmethod
    def hexagonal(a, c, orientation=Orientation(euler=[0.,0.,0.]),pointgroup='6/mmm',name='Alpha',symbol='$\alpha$'):
        """
        Convenience constructor for a hexagonal lattice.

        Args:
            a (float): *a* lattice parameter of the hexagonal cell.
            c (float): *c* lattice parameter of the hexagonal cell.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
            Hexagonal lattice of dimensions a x a x c.
        """
        lat = OrientedLattice.from_parameters(a, a, c, 90, 90, 120, orientation=orientation,name=name,symbol=symbol)
        lat._latticeCode = 5
        return lat
        
    @staticmethod
    def rhombohedral(a, alpha, orientation=Orientation(euler=[0.,0.,0.]),name='Alpha',symbol='$\alpha$'):
        """
        Convenience constructor for a rhombohedral lattice.

        Input:
        -----
            a (float): *a* lattice parameter of the rhombohedral cell.
            alpha (float): Angle for the rhombohedral lattice in degrees.
            orientation : An :mod:`Orientation` object defaults to [0,0,0] (euler angles)

        Returns:
        -------
            Rhombohedral lattice of dimensions a x a x a.
        """
        lat = OrientedLattice.from_parameters(a, a, a, alpha, alpha, alpha, orientation=orientation,name=name,symbol=symbol)
        lat._latticeCode = 6
        return lat
        

    @staticmethod
    def fromCif(cifFileName):
        """
        Create an Oriented Lattice object from a CIF file
        """        
        structure = mg.Structure.from_file(cifFileName)
        return OrientedLattice(matrix = structure.lattice.matrix)
        
        
    
    @staticmethod
    def from_parameters(a, b, c, alpha, beta, gamma, 
                        orientation=Orientation(euler=[0.,0.,0.]),pointgroup=None,name='Alpha',symbol='$\alpha$'):
        """
        Create a Lattice using unit cell lengths and angles (in degrees).

        Input:
        ------
            a (float): *a* lattice parameter.
            b (float): *b* lattice parameter.
            c (float): *c* lattice parameter.
            alpha (float): *alpha* angle in degrees.
            beta (float): *beta* angle in degrees.
            gamma (float): *gamma* angle in degrees.
            Orientation (Orientation Object) : This is the Orientatio by which lattice must be rotated

        Returns:
        --------
            Lattice with the specified lattice parameters.
        """

        alpha_r = radians(alpha)
        beta_r = radians(beta)
        gamma_r = radians(gamma)
        val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r))\
            / (np.sin(alpha_r) * np.sin(beta_r))
        # Sometimes rounding errors result in values slightly > 1.
        val = np.clip(val,-1.,1.)
        gamma_star = np.arccos(val)
        vector_a = [a * np.sin(beta_r), 0.0, a * np.cos(beta_r)]
        vector_b = [-b * np.sin(alpha_r) * np.cos(gamma_star),
                    b * np.sin(alpha_r) * np.sin(gamma_star),
                    b * np.cos(alpha_r)]
        vector_c = [0.0, 0.0, float(c)]
        
        
        return OrientedLattice([vector_a, vector_b, vector_c], orientation=orientation, pointgroup=pointgroup,
                               name=name,symbol=symbol)    
        
        
       
    
        
        