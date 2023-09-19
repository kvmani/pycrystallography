from pymatgen.analysis.diffraction.xrd import XRDCalculator as Xrd
import pymatgen as pm
import sympy as sp
from tabulate import tabulate
from pymatgen.util.testing import PymatgenTest
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import matplotlib
import logging
import numpy as np
import random
from tqdm import tqdm
# matplotlib.use('TkAgg')
cwd = pathlib.Path(__file__).parent.parent.resolve()
pyCrystallographyDir = cwd
print(f"cwd={cwd} pyCrystallographyDir={pyCrystallographyDir}")
try:
    from pycrystallography.core.orientation import Orientation

except:
    print("Unable to find the pycrystallography package!!! trying to now alter the system path !!")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname('..'))
    sys.path.insert(0, os.path.dirname('../../pycrystallography'))
    sys.path.insert(0, os.path.dirname('../../..'))
    for item in sys.path:
        print(f"Updated Path : {item}")
    from pycrystallography.core.orientation import Orientation
    from pycrystallography.core.quaternion import Quaternion
    from pycrystallography.core.millerDirection import MillerDirection
    from pycrystallography.core.millerPlane import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation import CrystalOrientation as CrysOri
    from pycrystallography.core.orientationRelation import OrientationRelation as OriReln
    from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    import pycrystallography.utilities.pyCrystUtilities as pyCrysUt
    import pycrystallography.utilities.pymathutilityfunctions as pmut
    from pycrystallography.ebsd.ebsd import Ebsd

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
ebsdAngFile = r'tmp/Al-B4CModelScan.ang'
fileDir = os.path.dirname(ebsdAngFile)
fileName = os.path.basename(ebsdAngFile)

for i in tqdm(range(10)):
    ebsd = Ebsd(logger=logger)
    logging.info(f"current dir is : {os.getcwd()}")
    ebsd.fromAng(fileName=ebsdAngFile)
    rotationAngle = random.choice([0, 90, 180, 270])
    flipMode = random.choice([None, 'h', 'v'])
    maskNumber = random.randint(0, 25)
    fileSuffix = f"rot_{rotationAngle}_flip_{flipMode}_mask_{maskNumber}"
    ebsd.rotateAndFlipData(
        flipMode=flipMode, rotate=rotationAngle)  # augmentation

    ebsd.crop(start=(1, 1), dimensions=(150, 150))

    ebsd.reduceEulerAngelsToFundamentalZone()
    ebsd.writeNpyFile(pathName=os.path.join(
        fileDir, f'Target_{i}_{fileSuffix}_'+fileName[:-4]+'.npy'))
    ebsd.writeAng(pathName=os.path.join(
        fileDir, f'Target_{i}_{fileSuffix}_'+fileName))
    ebsd.writeEulerAsPng(pathName=os.path.join(
        fileDir, f'Target_{i}_{fileSuffix}_'+fileName[:-4]+'.tiff'), showMap=False)
    # maskImg = np.full((80, 50), True, dtype=bool)
    maskImg = f"data/programeData/ebsdMaskFolder/{maskNumber}.png"
    ebsd.applyMask(maskImg, displayImage=False)
    # ebsd.rotateAndFlipData(flipMode='vertical', rotate=90)
    # ebsd.reduceEulerAngelsToFundamentalZone()

    ebsd.writeNpyFile(pathName=os.path.join(
        fileDir, f'Source_{i}_{fileSuffix}_'+fileName[:-4]+'.npy'))
    ebsd.writeAng(pathName=os.path.join(
        fileDir, f'Source_{i}_{fileSuffix}_'+fileName))
    ebsd.writeEulerAsPng(pathName=os.path.join(
        fileDir, f'Source_{i}_{fileSuffix}_'+fileName[:-4]+'.tiff'), showMap=False)

    logging.info(f"done with the processing written the files")
