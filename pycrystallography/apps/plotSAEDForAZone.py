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
ALMOST_EQUAL_TOLERANCE = 1e-3


cifPathName = r'../../data/structureData'
phaseCif = os.path.join(cifPathName , 'Fe2B_BCT.cif')
lat = olt.fromCif(phaseCif) ## U-Alpha
# saAlpha = SaedAnalyzer(hklMax=4)
# saAlpha.loadStructureFromCif(cifFileName=uAlphaCif)
latAlpha = olt.fromCif(phaseCif)
saAlpha = SaedAnalyzer(symbol=r"$\alpha$", lattice=latAlpha, hklMax=3, )
saAlpha.loadStructureFromCif(phaseCif)

zoneAxisAlpha = MillerDirection(lattice=latAlpha, vector=[1,1,3])
zoneAxisAlpha2 = MillerDirection(lattice=latAlpha, vector=[0,0,1])

saedData1 = saAlpha.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxisAlpha,
                                               patterCenter=[0., 0.],)
saedData2 = saAlpha.calcualteSAEDpatternForZoneAxis(zoneAxis=zoneAxisAlpha2,
                                               patterCenter=[0., 0.],)

fig = plt.gcf()
axes = fig.add_subplot(111, )
logger.info("Starting the plot")
saAlpha.plotSAED([saedData1,saedData2], plotShow=True, figHandle=None, axisHandle=axes, makeTransperent=False, markSpots=False,
            showAbsentSpots=False)

print("Testing of plotSAED function done!!")

