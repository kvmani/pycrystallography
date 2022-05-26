from __future__ import division, unicode_literals
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.saedAnalyzer import SaedAnalyzer
import pycrystallography.utilities.graphicUtilities as gu
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
import os
import copy
from math import pi, sqrt
#from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy
import matplotlib.pyplot as plt
import cv2

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13


hcpAtomData = [(40., np.array([0., 0., 0.])),
               (40., np.array([1. / 3., 2. / 3., 1. / 2.]))]  ####typical Zr data

cifPathName = r'C:\Users\Admin\PycharmProjects\pycrystallography\data\structureData'
uAlphaCif = os.path.join(cifPathName , 'Alpha-U_ver4.cif')
lat = olt.fromCif(uAlphaCif) ## U-Alpha
# saAlpha = SaedAnalyzer(hklMax=4)
# saAlpha.loadStructureFromCif(cifFileName=uAlphaCif)
latAlpha = olt.fromCif(uAlphaCif)
saAlpha = SaedAnalyzer(symbol=r"$\alpha$", lattice=latAlpha, hklMax=3, )
saAlpha.loadStructureFromCif(uAlphaCif)
cifPathName = r'C:\Users\Admin\PycharmProjects\pycrystallography\data\structureData'
uGammaCif = os.path.join(cifPathName , 'Gamma-U.cif')
latGamma = olt.fromCif(uGammaCif) ## U-Gamma
# saGamma = SaedAnalyzer(hklMax=3)
# saGamma.loadStructureFromCif(cifFileName=uGammaCif)
saGamma = SaedAnalyzer(symbol=r"$\gamma$", lattice=latGamma, hklMax=3, )
saGamma.loadStructureFromCif(uGammaCif)


latGamma = olt.fromCif(uGammaCif)
zoneAxisGamma = MillerDirection(lattice=latGamma, vector=[0,0,1])
zoneAxisAlpha = MillerDirection(lattice=latAlpha, vector=[3, 1 ,2])

saedData2 = saAlpha.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxisAlpha,
                                               patterCenter=[0., 0.],)
saedData1 = saGamma.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxisGamma,
                                               patterCenter=[0., 0.],inPlaneRotation=45.)

#expSpotData={"spotXyData":[[1105,1632],[1035,1251],[1703,1510]], "patternImage":"5596"} ### spots 3 & 4
expSpotData={"spotXyData":[[360.62, 518.21],  [267.65, 338.05], [228.77, 474.62]], "patternImage":"5596"} ## spots 1 and 2
expSpotData={"spotXyData":[[357.91, 516.85],[393.97, 405.96],[507.22, 436.59]], "patternImage":"5596"} ## spots 3 and 4
#expSpotData={"spotXyData":[[357.9 , 518.21],[506.71, 434.49],[276.31, 372.75]], "patternImage":"5596"} ## spots 3 and 5
expSpotData={"spotXyData":[[357.9 , 518.21],[228.77, 474.62],[276.31, 372.75]], "patternImage":"5596"} ## spots 2 and 5
expSpotData={"spotXyData":[[381., 525.],[228.64, 476.59],[280.81, 577.39]], "patternImage":"5608"} ## spots 1 and 2
expSpotData={"spotXyData":[[350.16, 502.57],[380.13, 610.58],[270.31, 583.07]], "patternImage":"5618"} ## spots 1 and 2 hexagonal pattern
expSpotData={"spotXyData":[[373.25, 475.9],[180.04, 476.12],[149.82, 366.9]], "patternImage":"5611"} ## spots 4 and 5 rhombos pattern
# expSpotData={"spotXyData":[[357.9 , 518.21],[429.28, 286.44],[277.67, 374.11]], "patternImage":"5596"} ## spots 5 and 9 rhombos pattern

calibration={"cameraConstant":1.702*195.33,"cameraLength":"100cm","machine":"2000Fx"} ## camera constant = length of spotIn pixcels*dspacing in Angstroms

result = saAlpha.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=4, D_TOLERANCE=10, allowedAngleDeviation=5,
                                        calibration=calibration)

print(result)

fig = plt.gcf()
axes = fig.add_subplot(111, )
logger.info("Starting the plot")
saAlpha.plotSAED([saedData1,saedData2], plotShow=True, figHandle=None, axisHandle=axes, makeTransperent=False, markSpots=False,
            showAbsentSpots=False)

print("Testing of plotSAED function done!!")

