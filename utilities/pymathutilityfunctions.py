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


def find_gcd(x, y):
    while(y):
        x, y = y, x % y
    return x
         
def gcdOfArray(Array):
    gcd=find_gcd(Array[0],Array[1])
    for i in range(2,len(Array)):
        gcd=find_gcd(gcd,Array[i])
    return(gcd)

def integerize(Direction):
    if isinstance(Direction, np.ndarray):
        ### check if it is already in integer form
        #err = np.sum(np.abs((Direction-Direction.astype(int))))
        if np.allclose(Direction, Direction.astype(int)):
            return Direction.astype(int)
        
        
        else:
            increasingOrder = np.abs(np.sort(np.abs(Direction[np.nonzero(np.abs(Direction)>1e-10)]), axis=None))
            integerArray = np.around(np.around((Direction/increasingOrder[0]),2)*10.0)
            HCF = gcdOfArray(np.abs(integerArray))
            integerizedDirection = integerArray/HCF
            return integerizedDirection.astype(int)
    else:
        print('Error! input vector should be ndarray')
        

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



if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    np.set_printoptions(suppress=True, precision=5)
    #err = objectiveFunctionFindTilts()
    #print(err)
    
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All test are done")
