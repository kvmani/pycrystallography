# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:47:29 2017

@author: Mani Krishna
"""
from __future__ import division, unicode_literals

import sys
import os
from pycrystallography.core.millerPlane import MillerPlane
from operator import itemgetter

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
import collections
from pymatgen.util.testing import PymatgenTest

from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.crystalOrientation import CrystalOrientation

from pycrystallography.core.orientedLattice import OrientedLattice as olt

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from copy import deepcopy
from tabulate import tabulate
import itertools
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
import pycrystallography.utilities.pymathutilityfunctions as pmt 


data ={"saritaNfcData": {"dataType":"hkl,uvw", 
        "data":  [
    
#         {"deformed":{"hkl":[0,0,1], "uvw":[1,1,0]},  #### from the mail sent by sarita of NFC.
#         "nucleated":{"hkl":[1,1,0], "uvw":[1,-1,2]},        
#         },
#          
#         {"deformed":{"hkl":[1,1,2], "uvw":[1,-1,0]},
#         "nucleated":{"hkl":[1,1,0], "uvw":[1,-1,2]},        
#         },
#          
#         {"deformed":{"hkl":[1,1,1], "uvw":[1,-1,0]},
#         "nucleated":{"hkl":[1,1,0], "uvw":[1,-1,2]},        
#         },
#  
#         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
#         "nucleated":{"hkl":[1,1,0], "uvw":[1,-1,2]},        
#         },
#          
#         ##### now second rwo
#         {"deformed":{"hkl":[0,0,1], "uvw":[1,1,0]},
#         "nucleated":{"hkl":[1,1,1], "uvw":[1,1,-2]},        
#         },
#          
#         {"deformed":{"hkl":[1,1,2], "uvw":[1,-1,0]},
#         "nucleated":{"hkl":[1,1,1], "uvw":[1,1,-2]},    
#         },
#          
#         {"deformed":{"hkl":[1,1,1], "uvw":[1,-1,0]},
#         "nucleated":{"hkl":[1,1,1], "uvw":[1,1,-2]},     
#         },
#  
#         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
#         "nucleated":{"hkl":[1,1,1], "uvw":[1,1,-2]},       
#         },
#         
#         
#         {"deformed":{"hkl":[0,0,1], "uvw":[1,1,0]},
#         "nucleated":{"hkl":[0,1,1], "uvw":[1,0,0]},        
#         },
#          
#         {"deformed":{"hkl":[1,1,2], "uvw":[1,-1,0]},
#         "nucleated":{"hkl":[0,1,1], "uvw":[1,0,0]},        
#         },
#     
#         
#         {"deformed":{"hkl":[1,1,1], "uvw":[0,-1,1]},
#         "nucleated":{"hkl":[0,1,1], "uvw":[1,0,0]},        
#         },
#          
#         {"deformed":{"hkl":[0,1,1], "uvw":[0,-1,1]},
#         "nucleated":{"hkl":[0,1,1], "uvw":[1,0,0]},        
#         },
#  
#         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
#         "nucleated":{"hkl":[0,1,1], "uvw":[1,0,0]},        
#         },
#         
#         {"deformed":{"hkl":[0,1,1], "uvw":[0,-1,1]},
#         "nucleated":{"hkl":[1,1,0], "uvw":[1,-1,2]}, 
#         },
#         
#         {"deformed":{"hkl":[0,1,1], "uvw":[0,-1,1]},
#         "nucleated":{"hkl":[1,1,1], "uvw":[1,1,-2]}, 
#         },
            #### set sent on 19-09-2019      
         {"deformed":{"hkl":[0,0,1], "uvw":[1,1,0]},
                "nucleated":{"hkl":[1,1,1], "uvw":[1,-1,0]},       
                },
        
         {"deformed":{"hkl":[1,1,2], "uvw":[1,-1,0]},
                "nucleated":{"hkl":[1,1,1], "uvw":[1,-1,0]},       
                },
        
         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
                "nucleated":{"hkl":[0,0,1], "uvw":[1,1,0]},       
                },
        
         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
                "nucleated":{"hkl":[1,1,1], "uvw":[1,-1,0]},       
                },
        
         {"deformed":{"hkl":[1,1,1], "uvw":[1,1,-2]},
                "nucleated":{"hkl":[1,1,2], "uvw":[1,-1,0]},       
                },

     
        ]
       },
       
      "astroDataSet": {"dataType":"EulerAngles", 
        "data": [{"euler1":[-87.87, 127.93 ,65.32], "euler2":[-12.53, 136.90, 99.62]},
                 {"euler1":[7.15, 127.09 ,47.31], "euler2":[-4.45, 133.95, 51.69]},
                 {"euler1":[114.04, 134.35 ,97.73], "euler2":[-12.53, 136.90, 99.62]},
                 {"euler1":[-130.30,121.19,-28.37], "euler2":[-115.86,105.53,-77.91]},
                 ]
            }
       }
#114.04 134.35 97.73
# Test (1)4X4-outputs.png -87.87 127.93 65.32 15.391559
# Test (1)4X4-targets.png -12.53 136.90 99.62 8.663809
dataChoice = "saritaNfcData"
data = data[dataChoice]
dataType = data["dataType"]
data = data["data"]

lattice = olt.cubic(1)

for i,item in enumerate(data):
    if dataType == "EulerAngles":
        euler1 = item["euler1"]
        euler2 = item["euler2"]
        
        ori1 =  CrystalOrientation(orientation=Orientation(euler=np.array(euler1)*np.pi/180),lattice = olt.cubic(1)) 
        ori2 =  CrystalOrientation(orientation=Orientation(euler=np.array(euler2)*np.pi/180),lattice = olt.cubic(1)) 
        misOri = ori1.disoreintation(ori2)
        
    elif  dataType == "hkl,uvw":
        def_hkl = item["deformed"]["hkl"]
        def_uvw = item["deformed"]["uvw"]
        nucl_hkl = item["nucleated"]["hkl"]
        nucl_uvw = item["nucleated"]["uvw"]  
        plane = MillerPlane(hkl=def_hkl, lattice=lattice)
        dir = MillerDirection(vector=def_uvw,lattice=lattice)
        deformedGrainOri = CrystalOrientation.fromPlaneAndDirection(plane=plane, direction=dir)
        plane = MillerPlane(hkl=nucl_hkl, lattice=lattice)
        dir = MillerDirection(vector=nucl_uvw,lattice=lattice)
        nucleiGrainOri = CrystalOrientation.fromPlaneAndDirection(plane=plane, direction=dir)
        misOri = deformedGrainOri.disoreintation(nucleiGrainOri)

    else:
        raise ValueError("Unknown data type only euler angles and hkl & uve format are supported !!!!")  
    
    #symmetricMisorientations
    
    axis = misOri.axis
    axis = pmt.integerize(axis)
    if dataType == "hkl,uvw":
        print(i+1, "deformed grain Ori =", def_hkl, def_uvw, "nucl grain Ori =", nucl_hkl, nucl_uvw, "--> Mis Ori = ", np.abs(np.around(misOri.angle*180/np.pi,2)), "@",  axis)
    else:
        print(i+1, "Ori1 =", euler1, " Ori 2 =", euler2 , "--> Mis Ori = ", np.abs(np.around(misOri.angle*180/np.pi,2)), "@",  axis)



print("yes")