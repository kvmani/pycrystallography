# -*- coding: utf-8 -*-
# transformations.py

from __future__ import division, print_function


from pymatgen.core.lattice import Lattice
from scipy.spatial.distance import squareform, pdist
import math as mt
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
#sys.path.insert(0, os.path.dirname('../..'))

#import pycrystallography.core.orientation as Ori
#from pycrystallography.core.orientation  import Orientation

__version__ = '2017.02.17'
__docformat__ = 'restructuredtext en'
__all__ = ()

def sub2ind(array_shape, rows, cols):
    """
    :param array_shape: tuple describing the array shape of 2D numpy array
    :param rows:
    :param cols:
    :return: ind such that np.flatten(array[ind])=array[rows][cols]
    """
    ind = rows*array_shape[1] + cols
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    return int(ind)

def ind2sub(array_shape, ind):
    """

    :param array_shape: array_shape: tuple describing the array shape of 2D numpy array
    :param ind: ind corresponding to the flattened array
    :return: (rows,col)  a tuple such that array[rows][cols] = np.flatten(array[ind])
    """
    # ind[ind < 0] = -1
    # ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = int((int(ind)/ array_shape[1]))
    cols = int(int(ind) % array_shape[1])
    return (rows, cols)

def find_gcd(x, y):
    while(y):
        x, y = y, x % y
    return x
         
def gcdOfArray(Array):
    gcd=find_gcd(Array[0],Array[1])
    for i in range(2,len(Array)):
        gcd=find_gcd(gcd,Array[i])
    return(gcd)


def integerizeMatrix(matrix):
    """

    :param matrix: a numpy array of typically 3X matrix
    :return: intMatrix : integerized matrix
    """
    intMatrix = np.copy(matrix)
    for i, row in enumerate(matrix):
        intMatrix[i] = integerize(row,reduceLowestInt=False)
    return intMatrix



def integerize(Direction,reduceLowestInt=True,rtol=1e-5, atol=1e-8):
    if isinstance(Direction, np.ndarray):
        ### check if it is already in integer form
        #err = np.sum(np.abs((Direction-Direction.astype(int))))

        if np.allclose(Direction,0):## fro trivial case of all elements of the vector being Zeros
            return np.zeros(Direction.shape).astype(int)
        
        intDirection = Direction.astype(int)
        
        if np.allclose(Direction,intDirection, rtol,atol ) or not reduceLowestInt:
            return intDirection
        else:
            #print("Yes it had comehere")
            increasingOrder = np.abs(np.sort(np.abs(Direction[np.nonzero(np.abs(Direction)>1e-3)]), axis=None))
            integerArray = np.around(np.around((Direction/increasingOrder[0]),2)*10.0)
            HCF = gcdOfArray(np.abs(integerArray))
            integerizedDirection = integerArray/HCF
            return integerizedDirection.astype(int)
    else:
        print('Error! input vector should be ndarray')
        

def angleBetween2Lines(line1, line2,units=""):
    """
    angle between two lines lines must be specified in the form of numpy arrays 
    e.g. 
    line1 = np.array([[0.,0.],[1.,0]])
    line2 = np.array([[0.,0.],[1.,1.]])
    returns angle in rdains if units ="Deg" return angel in Deg
    """
    angle1 = np.arctan2(line1[0][1] - line1[1][1],
                               line1[0][0] - line1[1][0])
    angle2 = np.arctan2(line2[0][1] - line2[1][1],
                               line2[0][0] - line2[1][0])
    if "deg" in units.lower():
        return 180/np.pi*(angle1-angle2)
    return angle1-angle2


def lineLength(line):
    """
    returns the magnitude of the line length
    """
   
    return (np.sqrt(((line[1][0]-line[0][0])*(line[1][0]-line[0][0]))+ \
        ((line[1][1]-line[0][1])*(line[1][1]-line[0][1]))))
    

def removeDuplicates(values):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output



def uniqueList(seq, idfun=None): 
     # order preserving
     if idfun is None:
         def idfun(x): return x
     seen = {}
     result = []
     for item in seq:
         marker = idfun(item)
         # in old Python versions:
         # if seen.has_key(marker)
         # but in new ones:
         if marker in seen: continue
         seen[marker] = 1
         result.append(item)
     return result
      
    


def uniqueRows(arr, thresh=0.0, metric='euclidean'):
    """Returns subset of rows that are unique, in terms of Euclidean distance
    """
    distances = squareform(pdist(arr, metric=metric))
    idxset = {tuple(np.nonzero(v)[0]) for v in distances <= thresh}
    return arr[[x[0] for x in idxset]]

def perpendicularVector(v):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    tol = 1e-10
    if np.allclose(v,[0,0,0]):
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if np.abs(v[0])< tol:
        return np.array([1., 0, 0])
    if np.abs(v[1])<tol:
        return np.array([0, 1., 0])
    if np.abs(v[2])<tol:
        return  np.array([0, 0., 1.])

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array((1, 1, -1.0 * (v[0] + v[1]) / v[2]))


def projectionOntoPlane(line1,plane):
    """
    Finds the unit vector of the projection of a line onto a plane
    """
    print("Yet to be implemented")
    raise ValueError ("Sorry")


if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    #err = objectiveFunctionFindTilts()
    #print(err)
    
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All test are done")
