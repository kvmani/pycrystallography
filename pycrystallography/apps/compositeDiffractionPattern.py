#### started on 26-05-2021
import sys
import os, pathlib
import matplotlib
#matplotlib.use('TkAgg')
cwd = pathlib.Path(__file__).parent.parent.resolve()
pyCrystallographyDir = cwd
print(f"cwd={cwd} pyCrystallographyDir={pyCrystallographyDir}")
import matplotlib.pyplot as plt
from pymatgen.util.testing import PymatgenTest
from tabulate import tabulate
import sympy as sp
try:
    from pycrystallography.core.orientation  import Orientation
    from pycrystallography.core.quaternion  import Quaternion
    from pycrystallography.core.millerDirection  import MillerDirection
    from pycrystallography.core.millerPlane  import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
    from pycrystallography.core.orientationRelation  import OrientationRelation as OriReln
    from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    import pycrystallography.utilities.pyCrystUtilities as pyCrysUt
    import pycrystallography.utilities.pymathutilityfunctions as pmut
except:
    print("Unable to find the pycrystallography package!!! trying to now alter the system path !!")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname('..'))
    sys.path.insert(0, os.path.dirname('../../pycrystallography'))
    sys.path.insert(0, os.path.dirname('../../..'))
    for item in sys.path:
        print(f"Updated Path : {item}")
    from pycrystallography.core.orientation  import Orientation
    from pycrystallography.core.quaternion  import Quaternion
    from pycrystallography.core.millerDirection  import MillerDirection
    from pycrystallography.core.millerPlane  import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
    from pycrystallography.core.orientationRelation  import OrientationRelation as OriReln
    from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    import pycrystallography.utilities.pyCrystUtilities as pyCrysUt


import pymatgen as pm
from pymatgen.analysis.diffraction.xrd import XRDCalculator as Xrd


import numpy as np
from math import pi
from pycrystallography.utilities.pymathutilityfunctions import integerize
import os
import logging
import pandas as pd

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13
logging.basicConfig(level=logging.INFO, )
orientation = CrysOri(orientation=Orientation(euler=[0., 0., 0.]), lattice=olt.cubic(1))
cifPathName = r'../../data/structureData'

##### code for JB Singh diffraction pattern generation

mainData={


    "SameehanFe2B/Fe":{
        "parent":[
            {"PhaseName": "Alpha",  ### parent data for product alpha(g/a)
             "symbol": r"Fe-$\alpha$",
             "cifName": "Fe.cif",
             "requiredZones": [
                               # [0, 0, 1, ],
                                [1,1,3],
                               # [1,2,2],
                               # [1,1,3],
                               # [1,1,0],
                               # [1,1,2],
                               ],
             "OR_Plane": [2,1,1],
             "OR_Direction": [1,1,-3],
             },

        ],
        "products": [

            {"PhaseName": "Fe2B",  ### Product alpha
             "symbol": r"$\beta$",
             "cifName": "Fe2B_BCT.cif",  ###
             "OR_Plane": [3,3,0],
             "OR_Direction": [1,-1,3],
             # "requiredZones":[[1,0,0],
             #                ],
             },

        ]

    },

    "ZrBetaToOmega":{
        "parent":[{"PhaseName":"Beta",
                      "symbol":r"$\beta",
                     "cifName":"Zr-Beta.cif",
                     "requiredZones":[#[0,0,1],
                                       #[1,1,1],
                                       [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[1,1,1],
                      "OR_Direction":[1,-1,0],
                      }],
            "products":[{"PhaseName":"Omega", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\omega$",
                     "cifName":"Zr-omega.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,0,1],
                     "OR_Direction":[2,-1,-1,0],
                     # "requiredZones":[[0,0,0,1],
                     #
                     #                ],
                      },]
                     },

    "ZrBetaToAlpha": {
        "parent": [{"PhaseName": "Beta",
                   "symbol": r"$\beta",
                   "cifName": "Zr-Beta.cif",
                   "requiredZones": [  [1,1,0],
                       # [1,1,1],
                       [0, 0, 1],
                       # [0,1,2],
                   ],
                   "OR_Plane": [1, 1, 0],
                   "OR_Direction": [1, -1, 1],
                   },
                 ],
        "products": [{"PhaseName": "Alpha",  ###
                    "symbol": r"$\alpha$",
                    "cifName": "Alpha-ZrP63mmc.cif",  ###
                    "OR_Plane": [0, 0, 0, 1],
                    "OR_Direction": [2, -1, -1, 0],
                    "requiredZones": [[0, 0, 0, 1],

                                      ],
                    },
                    ]
    },

    "gamma-GammaPrime":{
            "parent":[{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,0,1],
                                       #[1,1,1],
                                       #[1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      }],
            "products":[{"PhaseName":"GammaPrime", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\gamma^{''}$",
                     "cifName":"gmaaDoublePrime.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],

                                    ],
                      }],
         },
    ### now gamma - sigma phase OR:
"gamma-sigma":{
            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,0,1],
                                       [1,1,1],
                                       [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[1,1,1],
                      "OR_Direction":[0,-1,1],
                      },
            "product":{"PhaseName":"Sigma",
                     "symbol":r"$\sigma$",
                     "cifName":"sigma_FeCr-JB.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,4,0],
                     "requiredZones":[[0,0,1],

                                    ],
                      },
         },
    ### gamma-mu phase OR
"gamma-mu":{
            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,0,1],
                                       [1,1,1],
                                       [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[1,1,1],
                      "OR_Direction":[0,-1,1],
                      },
            "product":{"PhaseName":"Mu",
                     "symbol":r"$\mu$",
                     "cifName":"mu_Co7W6.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,0,1],
                     "OR_Direction":[1,0,-1,0],
                    "requiredZones":[[0,0,1],

                                    ],
                      },
         },
"gamma-delta":{ ### for JB Book
            "parent":[{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[1,1,0],
                                       # [1,1,1],
                                       # [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane": [1,1,1], #[1, 1, -2],  #[1,1,1],
                      "OR_Direction":[1,-1,0],
                      }],
            "products":[{"PhaseName":"Delta", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\delta$",
                     "cifName":"o-Ni3Nb-JB.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction": [0,1,0] , #[1,0,0], ## frozen on 13-01-2022
                     "requiredZones":[[0,0,1],
                                    ],
                      }],
         },
"gamma-M23C6":{
            "parent":[{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[#[0,0,1],
                                       [1,1,1],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      }],
            "products":[{"PhaseName":"M_23C_6", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$M23C6$",
                     "cifName":"M23C6_JB.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      }],
         },
"gamma-M6C":{
            "parent":[{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,0,1],
                                       [1,1,1],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      }],
            "products":[{"PhaseName":"M_6C1", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$M_6C$",
                     "cifName":"M6C_JB2.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      }],
         },

"delta-deltaHcp":{ ### this is being tried to see how the the Hexagonal equivalent of delta phase related to gama phase.
            "parent":{"PhaseName":"delta",
                      "symbol":r"$\delta",
                     "cifName":"o-Ni3Nb-JB.cif",
                     "requiredZones":[ [1,1,0],
                                       [0,0,1],
                                       [2,1,1],
                                       [2,0,1],
                                       [13,-10,7],
                                       [7,13,-10],
                                       [-10,13,7],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,1,0],
                      },
            "product":{"PhaseName":"HexDelta", ### Hexagonal equivalent of delta
                     "symbol":r"$Hex_\delta$",
                     "cifName":"o-Ni3Nb-Hex-JB.cif", ### artificially created
                     "OR_Plane":[0,0,0,1],
                     "OR_Direction":[2,-1,-1,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      },
         },

"gamma-MC":{
            "parent":[{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,0,1],
                                      # [1,1,1],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      }],
            "products":[{"PhaseName":"MC",
                     "symbol":r"$MC$",
                     "cifName":"NbC_JB.cif",
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      }],
         },



"beta-lauve":{ #### For Tewari sir's paper on NbZrCr precipitation
            "parent":{"PhaseName":"Beta",
                      "symbol":r"$\beta",
                     "cifName":"Nb.cif",
                     "requiredZones":[ [1,1,0],
                                       [1,1,1],
                                       [0,0,1],
                                       [1,1,2],
                                    ],
                      "OR_Plane":[0,1,1],
                      "OR_Direction":[1,-1,1],
                      },
            "product":{"PhaseName":"Lave",
                     "symbol":r"$L$",
                     "cifName":  "Lave_NbCr2_MgZn2.cif", #### "Alpha-ZrP63mmc.cif", ### Lave_NbCr2_MgZn2.cif",
                     "OR_Plane":[0,0,0,1],
                     "OR_Direction":[1,1,-2,0],
                     "requiredZones":[[0,0,0,1],
                                    ],
                      },
         },

"rakesh-U-Mo-Gamma-Alpha":{

            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ #[0,0,1],
                                       [1,2,2],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[1,1,0],
                      "OR_Direction":[0,0,1],
                      },
            "product":{"PhaseName":"Alpha", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      },
},

"rakesh-U-Mo-Gamma-Alpha-ver2":{

            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [0,0,1],
                                       #[1,2,2],
                                       #[1,1,3],
                                       [1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,-1,0],
                      },
            "product":{"PhaseName":"Alpha", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif) with modified lattice parameters
                     "OR_Plane":[1,0,0],
                     "OR_Direction":[0,0,1],
                     "requiredZones":[[0,0,1],
                                    ],
                      },
},


"rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT2":{ #### case of one parent and 2 products with RT2 !!!!

            "parent":[
                {"PhaseName": "Gamma",
                 ### parent data for product Gamma prime(g/g')
                 "symbol": r"$\gamma$",
                 "cifName": "Gamma-U.cif",
                 "requiredZones": [[0, 0, 1],
                                   # [1,1,0],
                                   # [1,1,0],
                                   # [1,1,2],
                                   ],
                 "OR_Plane": [1, 1, 0],
                 "OR_Direction": [0, 0, 1],
                 },

                     {"PhaseName":"Gamma",  ### parent data for product alpha(g/a)
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [6,-6,5,],
                                       #[1,2,2],
                                       #[1,1,3],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,-1,0],
                      },

                      ],

            "products":[


                {"PhaseName": "GammaPrime",  ### one of RT2 Product GammaPrime
                 "symbol": r"$\gamma^{'}$",
                 "cifName": "U2Mo.cif",  ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                 "OR_Plane": [1,1,0],
                 "OR_Direction": [0,0,1],
                 "requiredZones": [[0, 0, 1],
                                   ],
                 },
                    {"PhaseName":"Alpha", ### one of RT2 Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif) with modified lattice parameters
                     "OR_Plane":[1,0,0],
                     "OR_Direction":[0,0,1],
                     "requiredZones":[[1,0,1],
                                    ],
                      },
                       ]
},


"rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT3":{ #### case of one parent and 2 products with RT2 !!!!

            "parent":[

                     {"PhaseName":"Gamma",  ### parent data for product alpha(g/a)
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [0,0,1,],
                                       #[1,2,2],
                                       #[1,1,3],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[1,-1,0],
                      "OR_Direction":[1,1,1],
                      },
                    {"PhaseName":"Gamma",### parent data for product Gamma prime(g/g')
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [1,1,1],
                                       #[1,1,0],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[1,1,0],
                      "OR_Direction":[0,0,1],
                      },
                      ],

            "products":[

                    {"PhaseName":"Alpha", ### one of RT2 Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif) with modified lattice parameters
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[1,0,0],
                                    ],
                      },
                {"PhaseName": "GammaPrime",  ### one of RT2 Product GammaPrime
                 "symbol": r"$\gamma^{'}$",
                 "cifName": "U2Mo.cif",  ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                 "OR_Plane": [1, 1, 0],
                 "OR_Direction": [0, 0, 1],
                 "requiredZones": [[0, 0, 1],
                                   ],
                 },
                       ]
},


"rakesh-U-Mo-Gamma-Alpha-GammaPrime-RM1":{ #### OR based on pttern 5611

            "parent":[
                {"PhaseName": "Gamma",  ### parent data for product alpha(g/a)
                 "symbol": r"$\gamma$",
                 "cifName": "Gamma-U.cif",
                 "requiredZones": [[0,0,1],
                                    # [1,2,2],
                                    # [1,1,3],
                                    # [1,1,0],
                                    # [1,1,2],
                                   ],
                 "OR_Plane": [-2,0,0],
                 "OR_Direction": [0,0,1],
                 },

                {"PhaseName":"Gamma",### parent data for product Gamma prime(g/g')
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     # "requiredZones":[ [1,1,3],
                     #                   #[1,1,0],
                     #                #[1, 2, 2],
                     #                   # [1, 1, 3],
                     #                   # [1, 1, 0],
                     #                   # [1, 1, 2],
                     #                ],
                      "OR_Plane":[1,1,0],
                      "OR_Direction":[0,0,1],
                      },


                      ],

            "products":[

                    {"PhaseName":"Alpha", ### Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver4.cif", ###
                     "OR_Plane":[-1,3,2],
                     "OR_Direction":[-2,0,-1],
                     # "requiredZones":[[1,0,0],
                     #                ],
                      },
                        {"PhaseName":"GammaPrime", ### one of RT2 Product GammaPrime
                     "symbol":r"$\gamma^{'}$",
                     "cifName":"U2Mo.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[1,1,0],
                     "OR_Direction":[0,0,1],
                     # "requiredZones":[[0,0,1],
                     #                ],
                      },
                       ]
},

"rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT1":{ #### case of one parent and 2 products with RT1 !!!!

            "parent":[
                {"PhaseName": "Gamma",  ### parent data for product alpha(g/a)
                 "symbol": r"$\gamma$",
                 "cifName": "Gamma-U.cif",
                 "requiredZones": [[0,0,1],
                                    # [1,2,2],
                                    # [1,1,3],
                                    # [1,1,0],
                                    # [1,1,2],
                                   ],
                 "OR_Plane": [1, 1, 0],
                 "OR_Direction": [0, 0, 1],
                 },

                {"PhaseName":"Gamma",### parent data for product Gamma prime(g/g')
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [1,1,3],
                                       #[1,1,0],
                                    #[1, 2, 2],
                                       # [1, 1, 3],
                                       # [1, 1, 0],
                                       # [1, 1, 2],
                                    ],
                      "OR_Plane":[1,1,0],
                      "OR_Direction":[0,0,1],
                      },


                      ],

            "products":[

                    {"PhaseName":"Alpha", ### one of RT2 Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ###
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0], #### changed to 010 from 100
                     # "requiredZones":[[1,0,0],
                     #                ],
                      },
                        {"PhaseName":"GammaPrime", ### one of RT2 Product GammaPrime
                     "symbol":r"$\gamma^{'}$",
                     "cifName":"U2Mo.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[1,1,0],
                     "OR_Direction":[0,0,1],
                     # "requiredZones":[[0,0,1],
                     #                ],
                      },
                       ]
},



"rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT1a":{ #### case of one parent and 2 products with RT1 !!!!

            "parent":[
                {"PhaseName": "Gamma",  ### parent data for product alpha(g/a)
                 "symbol": r"$\gamma$",
                 "cifName": "Gamma-U.cif",
                 "requiredZones": [[0,0,1],
                                    # [1,2,2],
                                    # [1,1,3],
                                    # [1,1,0],
                                    # [1,1,2],
                                   ],
                 "OR_Plane": [1, 1, 0],
                 "OR_Direction": [0, 0, 1],
                 },

                {"PhaseName":"Gamma",### parent data for product Gamma prime(g/g')
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [1,1,3],
                                       #[1,1,0],
                                    #[1, 2, 2],
                                       # [1, 1, 3],
                                       # [1, 1, 0],
                                       # [1, 1, 2],
                                    ],
                      "OR_Plane":[1,1,0],
                      "OR_Direction":[0,0,1],
                      },


                      ],

            "products":[

                    {"PhaseName":"Alpha", ### one of RT2 Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ###
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0], #### changed to 010 from 100
                     # "requiredZones":[[1,0,0],
                     #                ],
                      },
                        {"PhaseName":"GammaPrime", ### one of RT2 Product GammaPrime
                     "symbol":r"$\gamma^{'}$",
                     "cifName":"U2Mo.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     # "requiredZones":[[0,0,1],
                     #                ],
                      },
                       ]
},



"rakesh-U-Mo-Gamma-Alpha-GammaPrime-Suman":{ #### case of one parent and 2 products with suman !!!!

            "parent":[
                {"PhaseName": "Gamma",  ### parent data for product alpha(g/a)
                 "symbol": r"$\gamma$",
                 "cifName": "Gamma-U.cif",
                 "requiredZones": [[0, 0, 1],
                                   # [1,2,2],
                                   # [1,1,3],
                                   # [1,1,0],
                                   # [1,1,2],
                                   ],
                 "OR_Plane": [0, -1, 1],
                 "OR_Direction": [0, 1, 1],
                 },
                {"PhaseName":"Gamma",### parent data for product Gamma prime(g/g')
                      "symbol":r"$\gamma$",
                     "cifName":"Gamma-U.cif",
                     "requiredZones":[ [0,0,1],
                                       #[1,1,0],
                                       #[1,1,0],
                                       #[1,1,2],
                                    ],
                      "OR_Plane":[0,-1,1],
                      "OR_Direction":[0,1,1],
                      },


                      ],

            "products":[

                    {"PhaseName":"Alpha", ### one of RT2 Product alpha
                     "symbol":r"$\alpha$",
                     "cifName":"Alpha-U_ver2.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif) with modified lattice parameters
                     "OR_Plane":[1,1,0],
                     "OR_Direction":[0,0,1],
                     # "requiredZones":[[0,0,1],
                     #                ],
                      },
                    {"PhaseName":"GammaPrime", ### one of RT2 Product GammaPrime
                     "symbol":r"$\gamma^{'}$",
                     "cifName":"U2Mo.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[1,-1,0],
                     "OR_Direction":[1,1,0],
                     "requiredZones":[[0,0,1],
                                    ],
                      },
                       ]
},
}


# dataChoice = "gamma-M23C6"
#dataChoice = "gamma-sigma"
# dataChoice = "gamma-mu"
dataChoice = "SameehanFe2B/Fe"
#dataChoice = "gamma-GammaPrime"
# dataChoice = "gamma-M23C6"
# dataChoice = "gamma-M6C"
# dataChoice = "gamma-MC"
# dataChoice = "gamma-MC"
# dataChoice = "rakesh-U-Mo Gamma-gammaPrime"
# dataChoice = "rakesh-U-Mo-Gamma-Alpha-ver2"
# dataChoice = "gamma-mu"
# dataChoice = "gamma-mu"
#dataChoice = "rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT3"
# # dataChoice = "rakesh-U-Mo-Gamma-Alpha-GammaPrime-RM1"
# dataChoice = "rakesh-U-Mo-Gamma-Alpha-GammaPrime-Suman"
# dataChoice = "rakesh-U-Mo-Gamma-Alpha-GammaPrime-RT2"

parentsData = mainData[dataChoice]["parent"]
productsData = mainData[dataChoice]["products"]
OrParentToProduct=[]
allSadData=[]
for i, item in enumerate(zip(parentsData,productsData)):
    if i>0:
        print("Skipping i>0 case !!!!! dont forget to change it later!!!")
        continue

    parentData, productData = item[0],item[1]
    parentCif = os.path.join(cifPathName, parentData["cifName"])
    productCif = os.path.join(cifPathName, productData["cifName"])
    names = [parentData["PhaseName"], productData["PhaseName"]]
    symbols = [parentData["symbol"], productData["symbol"]]
    structures = [parentCif, productCif]
    stParent, latParent = OriReln.getStructureFromCif(parentCif)
    stProduct, latProduct = OriReln.getStructureFromCif(productCif)

    planes = [MillerPlane(hkl=parentData["OR_Plane"], lattice=latParent),
              MillerPlane(hkl=productData["OR_Plane"], lattice=latProduct)]
    directions = [MillerDirection(vector=parentData["OR_Direction"], lattice=latParent),
                  MillerDirection(vector=productData["OR_Direction"], lattice=latProduct),
                  ]

    OrParentToProduct.append(OriReln(names=names, symbols=symbols, structures=structures, planes=planes,
                                directions=directions))
    print(OrParentToProduct)
    print(OrParentToProduct[i].makeORString(format="latex"))



    pyCrysUt.makeTEMDiffractgionTable(lattice=latParent, structure=stParent,
                                      dfFileName=os.path.join(r"../../tmp", parentData["PhaseName"]+"_"+parentData["PhaseName"] + '.html'))
    pyCrysUt.makeTEMDiffractgionTable(lattice=latProduct, structure=stProduct,
                                      dfFileName=os.path.join(r"../../tmp", parentData["PhaseName"]+"_"+productData["PhaseName"] + '.html'))

    OrParentToProduct[i].makeReport(format="html", savetoFile=True)

    dirs = [[1,1,1], [1,1,0], [2,0,0],[0,2,0],[0,0,2], [2,2,0], [1,1,2]] ### for cross checking the correspondence matrix
    for dir in dirs:
        dirMiller = MillerDirection(vector=dir, lattice=latParent)
        planeMiller = MillerPlane(hkl=dir, lattice=latParent)
        print(f"vec Mag = {dirMiller} : {dirMiller.mag}  {planeMiller} : {planeMiller.dspacing}")
    #exit(-150)
    # ### code for JB book
    # variants = OrParentToProduct[i].getVariants()
    # baseMatrix = np.array([[-1, 2 / 3, 1 / 2], [1, 2 / 3, 1 / 2], [0, 2 / 3, -1]])
    # baseMatrix = np.array([[6.0,4.,3.],[-6.,4.,3.],[0.,4.,-6.0]]) ## JB version
    # baseMatrix = np.array([[3, -6, 4],[3, 6, 4],[-6, 0, 4]]) ### mani version  ## use the variantsCrossChecking.py for determining this one
    # baseMatrixInverse = np.array([[2,2,-4],[-3,3,0],[3,3,3]]) ### This is the inverse matrix for the above ## use the variantsCrossChecking.py for determining this one
    # outList = []
    # for ii, variant in enumerate(variants[0]):
    #     if ii==0:
    #         #correctionMatrix = np.matmul(variant.rotation_matrix,baseMatrix)
    #         correctionMatrix = variant.rotation_matrix.T ## inv of Rotation
    #     preMultipler = np.matmul(correctionMatrix,variant.rotation_matrix)
    #     variantJB = np.around(np.matmul(preMultipler, baseMatrix),0)
    #     variantJBMatrix = sp.Matrix(variantJB)
    #     variantJBMatrixInv = 36*variantJBMatrix.inv()
    #     #variantJBMatrixInv =
    #
    #     # variantJB[variantJB==5]=6
    #     # variantJB[variantJB==-5]=-6
    #     #variantJB[0] =variantJB[0]*6./4. ### because we are getting 4 instead of 6 for some reason
    #     #variantJB[0] =variantJB[0]*6/4 ### because we are getting 4 instead of 6 for some reason
    #     #variantJB = variantJB.T
    #     inVariantJB = np.around(np.matmul(preMultipler, baseMatrixInverse),0)
    #
    #     variantJBMatrixInv = np.array(variantJBMatrixInv).astype(int) ### converting back to numpy array
    #     #variantJBMatrixInv=variantJBMatrixInv.inv()
    #     #variantJB = variantJB.T ### just for easy copying in ut put remove later
    #     #inVariantJB =
    #     row1 = [f"{ii+1:2d}", str(variantJB[0]).replace('[','').replace(']',''),str(inVariantJB[0]).replace('[','').replace(']','')+',' ,
    #             str(variantJBMatrixInv[0,:]).replace('[','').replace(']','').replace("Matrix(","").replace(")","")+',' ,
    #             #"det="+str(np.around(float(sp.Matrix(variantJBMatrix).det()))), "inv="+str(np.around(float(sp.Matrix(variantJBMatrixInv).det())))
    #             ]
    #     row2 = ["  ", str(variantJB[1]).replace('[','').replace(']',''),str(inVariantJB[1]).replace('[','').replace(']','')+',',
    #             str(variantJBMatrixInv[1,:]).replace('[','').replace(']','').replace("Matrix(","").replace(")","")+',']
    #     row3 = ["  ", str(variantJB[2]).replace('[','').replace(']',''),str(inVariantJB[2]).replace('[','').replace(']','')+',',
    #             str(variantJBMatrixInv[2,:]).replace('[','').replace(']','').replace("Matrix(","").replace(")","")+',']
    #     tmp =  "["+str(variantJB[0]).replace('[','').replace(']','')+","+str(variantJB[1]).replace('[','').replace(']','')+","+str(variantJB[2]).replace('[','').replace(']','')+"],"
    #     if tmp[1]==' ':
    #         tmp = "["+tmp[2:]
    #     tmp = tmp.replace(" ",", ")
    #     tmp = tmp.replace(",, ",",")
    #     tmp = tmp.replace(", ,",",")
    #
    #     row = [f" ", tmp, " ", " "]
    #
    #     outList.append(row1)
    #     outList.append(row2)
    #     outList.append(row3)
    #     outList.append(["", "", ""])
    #     #outList.append(row)
    #     # outList.append(["", "", ""])
    #     #print(f"[{ii}]: {np.around(variant.rotation_matrix,3)} \n baseMatrix : {baseMatrix} \n JB variant : {variantJB.T}   \n correction : {correctionMatrix}  ")
    #
    # table = tabulate(outList,headers=('variantId','CubicToOrtho','OrthoToCubic'),tablefmt='html')
    # with open('../../tmp/variantsOut.html','w') as f:
    #     f.write(table)
    # print(table) ### this html file will generate all the transformation matrices from gamma to delta. copy the output and put it in the script variantsCrossChecking.py for genrating the plane correspondence table
    # #exit(-100)
    # ### end of code for JB book
    #
    #
    #
    #
    #
    #
    for zoneAxis in parentData["requiredZones"]:

        parentZoneAxis = MillerDirection(vector=zoneAxis,lattice=latParent)
        zoneListProduct = OrParentToProduct[i].findParlallelDirectionsInProduct(parentZoneAxis)
        parentPlane = MillerPlane(hkl=zoneAxis,lattice=latParent)
        parallelPlaneListProduct = OrParentToProduct[i].findParallelPlaneInProduct(parentPlane)
        if "requiredZones" in productData: ### cheking if the  desired product zone axis is defined
            for zoneAxisProduct in productData["requiredZones"]:
                productZoneAxis = MillerDirection(vector=zoneAxisProduct,lattice=latProduct)
                saedData = OrParentToProduct[i].calculateCompositeSAEDBasedOnProductZoneAxis(productZoneAxis=productZoneAxis, )
                #OrParentToProduct[i].plotSaed(saedData[:2])

                zoneListParent = OrParentToProduct[i].findParlallelDirectionsInParent(productZoneAxis)

                for ii, zone in enumerate(zoneListParent):
                    print(f"ref zone product : {productZoneAxis} and parallel in parent variant id {ii}  :{zone.integerize()}, cartesian : {zone.getCartesianVec()}")


        print("Now the usual earlier stuff")
        for ii, zone in enumerate(zoneListProduct):
            print(f"ref zone {parentZoneAxis} and parallel in variant id {ii}  :{zone.integerize()}, cartesian : {zone.getCartesianVec()}")

        for ii, zone in enumerate(zoneListProduct):
            print(f"ref zone {parentZoneAxis} and parallel in variant id {ii}  :{zone.integerize()}, cartesian : {zone.getCartesianVec()}")
        for ii, plane in enumerate(parallelPlaneListProduct):
            print(f"ref zone {parentPlane} and parallel in variant id {ii}  :{plane.integerize()}, cartesian : {plane.getCartesianVec()}")

    #     #exit(-200)

        # # sadData = OrParentToProduct.calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0, variantIds=[0,1,2,3,4,5,6,7,8,9,10,11],
        # #                                                   pc=[0.,0],sf=1.,Tol=1,inPlaneRotation=-15.) ## 111 zone
        #
        # sadData = OrParentToProduct.calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0, variantIds=[2,3, 4, 6, 9, 10],
        #                                                   pc=[0.,0],sf=1.,Tol=1,inPlaneRotation=0.) ## 100 zone for  delta
        sadData = OrParentToProduct[i].calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0,
                                                              pc=[0.,0], sf=1., Tol=1., inPlaneRotation=135.)
        #OrParentToProduct.plotSaed(sadData)
        # if i>0:
        #     sadData = sadData[1:] #### the 0 element is matrix only. Hence we are removing them for all compote patterns of the 2ndproduct onwards
        allSadData.extend(sadData)

# degree = np.pi/180.
# Ori1 = CrysOri(orientation=Orientation(euler=[45.0*degree, 90.*degree, 90*degree]), lattice=latProduct)
# Ori2 = CrysOri(orientation=Orientation(euler=[135.*degree, 90*degree, 90*degree]), lattice=latProduct)
# print("Misor= ", Ori1.misorientation(Ori2), "Disor= ", Ori1.disoreintation(Ori2))



    # OrParentToProduct[i].plotSaed([allSadData[0],allSadData[1],allSadData[2],])
    OrParentToProduct[i].plotSaed(allSadData[0:2])
    plt.gcf().suptitle(dataChoice, fontsize=16)