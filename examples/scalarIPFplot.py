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
from pycrystallography.ebsd.ebsd import Ebsd
import pymatgen as pm
import pandas as pd
import numpy as np

inputFile = r'D:\CurrentProjects\colloborations\suman_phd\ScalarIPFData.csv'
simulatedEbsdOutPutFilefilePath = r'D:\CurrentProjects\colloborations\suman_phd\ScalarIPFData.ang'

data = pd.read_csv(inputFile)
Oridata = np.array([np.array(data["Euler1(phi1)"]), np.array(data["Euler2(phi)"]), np.array(data["Euler3(phi2)"])])
Oridata = (Oridata.T).tolist() ### transpose is need to make each row an orientation.
scalarData = (1.0*data["NonSchmidFactor"]).tolist()
print(Oridata, "\n scalar data = ", scalarData)
ebsd = Ebsd()
ebsd.generateSimulatedEbsdMap(orientationList=Oridata, simulatedEbsdOutPutFilefilePath = simulatedEbsdOutPutFilefilePath,
                              headerFileName="bcc_header.txt",IqData=scalarData)
print("Done")
