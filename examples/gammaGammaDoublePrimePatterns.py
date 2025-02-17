from pymatgen.util.testing import PymatgenTest
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
cifPathName = r'../data/structureData'


##### code for JB Singh diffraction pattern generation


mainData={"gamma-GammaPrime":{
            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[#[0,0,1],
                                       #[1,1,1],
                                       [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      },
            "product":{"PhaseName":"GammaPrime", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\gamma^{''}$",
                     "cifName":"gmaaDoublePrime.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],

                                    ],
                      },
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
"gamma-delta":{
            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[[0,1,0],
                                       [1,1,1],
                                       [1,1,0],
                                      # [0,1,2],
                                    ],
                      "OR_Plane":[1,1,1],
                      "OR_Direction":[1,-1,0],
                      },
            "product":{"PhaseName":"Delta", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$\delta^{''}$",
                     "cifName":"o-Ni3Nb-JB.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[0,1,0],
                     "requiredZones":[[0,0,1],

                                    ],
                      },
         },
"gamma-M23C6":{
            "parent":{"PhaseName":"Gamma",
                      "symbol":r"$\gamma$",
                     "cifName":"Ni.cif",
                     "requiredZones":[#[0,0,1],
                                       #[1,1,1],
                                       #[1,1,0],
                                       [1,1,2],
                                    ],
                      "OR_Plane":[0,0,1],
                      "OR_Direction":[1,0,0],
                      },
            "product":{"PhaseName":"M_23C_6", ### one of ( disorderedOmega, OrderedOmega)
                     "symbol":r"$M_{23}C_{6}$",
                     "cifName":"M23C6_JB.cif", ### one of (Ordered_UZr2.cif,Zr-omega.cif)
                     "OR_Plane":[0,0,1],
                     "OR_Direction":[1,0,0],
                     "requiredZones":[[0,0,1],

                                    ],
                      },
         },
}
dataChoice = "gamma-GammaPrime"
# dataChoice = "gamma-sigma"
# dataChoice = "gamma-mu"
dataChoice = "gamma-delta"
#dataChoice = "gamma-M23C6"
parentData = mainData[dataChoice]["parent"]
productData = mainData[dataChoice]["product"]

parentCif = os.path.join(cifPathName,parentData["cifName"])
productCif = os.path.join(cifPathName,productData["cifName"])
names = [parentData["PhaseName"],productData["PhaseName"]]
symbols = [parentData["symbol"],productData["symbol"]]
structures = [parentCif, productCif]
stParent, latParent = OriReln.getStructureFromCif(parentCif)
stProduct, latProduct = OriReln.getStructureFromCif(productCif)

planes = [MillerPlane(hkl=parentData["OR_Plane"], lattice=latParent),
          MillerPlane(hkl=productData["OR_Plane"], lattice=latProduct)]
directions = [MillerDirection(vector=parentData["OR_Direction"], lattice=latParent),
                MillerDirection(vector=productData["OR_Direction"], lattice=latProduct),
              ]

OrParentToProduct = OriReln(names=names,symbols=symbols, structures=structures, planes=planes, directions=directions)
print(OrParentToProduct)

pyCrysUt.makeTEMDiffractgionTable(lattice=latParent,structure=stParent,
                                  dfFileName=os.path.join(r"../tmp",parentData["PhaseName"]+'.html'))
pyCrysUt.makeTEMDiffractgionTable(lattice=latProduct,structure=stProduct,
                                  dfFileName=os.path.join(r"../tmp",productData["PhaseName"]+'.html'))

for zoneAxis in parentData["requiredZones"]:
    parentZoneAxis = MillerDirection(vector=zoneAxis,lattice=latParent)
    zoneListProduct = OrParentToProduct.findParlallelDirectionsInProduct(parentZoneAxis)
    for ii, zone in enumerate(zoneListProduct):
        print(f"ref zone {parentZoneAxis} and parallel in variant id {ii}  :{zone.integerize()}, cartesian : {zone.getCartesianVec()}")

    sadData = OrParentToProduct.calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0, variantIds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                                       pc=[0.,0], sf=1., Tol=1)
    # sadData = OrParentToProduct.calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0, variantIds=[0],
    #                                                   pc=[0.,0],sf=1.,Tol=1)
    OrParentToProduct.plotSaed(sadData)


