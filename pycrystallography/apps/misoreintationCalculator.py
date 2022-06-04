#### started on 26-05-2021
import sys
import os, pathlib
cwd = pathlib.Path(__file__).parent.parent.resolve()
pyCrystallographyDir = cwd
print(f"cwd={cwd} pyCrystallographyDir={pyCrystallographyDir}")

from pymatgen.util.testing import PymatgenTest
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

phase1Data = {"cifName":'Cr.cif'}
phase2Data = {"cifName":"Fe.cif"}

phase1Cif = os.path.join(cifPathName, phase1Data["cifName"])
phase2Cif = os.path.join(cifPathName, phase2Data["cifName"])

degree = np.pi/180.
stphase1, latphase1 = OriReln.getStructureFromCif(phase1Cif)
stphase2, latphase2 = OriReln.getStructureFromCif(phase2Cif)

Ori1 = CrysOri(orientation=Orientation(euler=[45.0*degree, 90.*degree, 90*degree]), lattice=latphase2)
Ori2 = CrysOri(orientation=Orientation(euler=[135.*degree, 90*degree, 90*degree]), lattice=latphase2)
print("Misor= ", Ori1.misorientation(Ori2), "Disor= ", Ori1.disoreintation(Ori2))
print("MisorAngle = ", Ori1.misorientation(Ori2).angle/degree, "Disor Angle= ", Ori1.disoreintation(Ori2).angle/degree)