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

mainData={
            "parent":{"PhaseName":"Ni",
                     "cifName":"Ni.cif",
                     "requiredZones":[#[1,1,0],
                                       #[1,-1,1],
                                       [0,1,1],
                                      [0,1,2],
                                    ],
                      "OR_Plane":[1,-1,1],
                      "OR_Direction":[1,2,1],
                      },
            "product":{"PhaseName":"Ni",
                     "cifName":"Ni.cif",
                     "OR_Plane":[1,-1,1],
                     "OR_Direction":[-1,-2,-1],
                      },

         }

parentData = mainData["parent"]
productData = mainData["product"]

parentCif = os.path.join(cifPathName,parentData["cifName"])
productCif = os.path.join(cifPathName,productData["cifName"])
names = [mainData["parent"]["PhaseName"],mainData["product"]["PhaseName"]]
structures = [parentCif, productCif]
stParent, latParent = OriReln.getStructureFromCif(parentCif)
stProduct, latProduct = OriReln.getStructureFromCif(productCif)

planes = [MillerPlane(hkl=parentData["OR_Plane"], lattice=latParent),
          MillerPlane(hkl=productData["OR_Plane"], lattice=latProduct)]
directions = [MillerDirection(vector=parentData["OR_Direction"], lattice=latParent),
                MillerDirection(vector=productData["OR_Direction"], lattice=latProduct),
              ]

OrParentToProduct = OriReln(names=names,structures=structures, planes=planes, directions=directions)


for zoneAxis in parentData["requiredZones"]:
    parentZoneAxis = MillerDirection(vector=zoneAxis,lattice=latParent)
    zoneListProduct = OrParentToProduct.findParlallelDirectionsInProduct(parentZoneAxis)
    for ii, zone in enumerate(zoneListProduct):
        print(f"ref zone {parentZoneAxis} and parallel in variant id {ii}  :{zone.integerize()}, cartesian : {zone.getCartesianVec()}")

    sadData = OrParentToProduct.calculateCompositeSAED(parentZoneAxis=parentZoneAxis, productId=0, variantIds=[0, 1, 2, 3],
                                                       pc=[0.,0], sf=1., Tol=1)
    OrParentToProduct.plotSaed(sadData)


