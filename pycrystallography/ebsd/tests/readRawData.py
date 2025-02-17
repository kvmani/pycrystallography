import os
import sys
import logging
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def configure_path():
    """Configures the system path to locate the pycrystallography package."""
    try:
        from pycrystallography.core.orientation import Orientation
    except ImportError:
        logger.warning("Unable to find the pycrystallography package. Adjusting system path.")
        sys.path.insert(0, os.path.abspath('.'))
        sys.path.insert(0, os.path.dirname('..'))
        sys.path.insert(0, os.path.dirname('../../pycrystallography'))
        sys.path.insert(0, os.path.dirname('../../..'))
        for path in sys.path:
            logger.debug(f"Updated Path: {path}")

        
def rwData(imagePath):
    from pycrystallography.ebsd.ebsd import Ebsd
    ebsd = Ebsd(logger=logger)
    ebsd.fromAng(imagePath)
    print(ebsd._data)
    ebsd.writeCtf(f"pycrystallography/ebsd/tests/xcelFile1.csv")
    ebsd.writeEulerAsPng(r"pycrystallography/ebsd/tests/rawImage.tiff")
def main():
    configure_path()
    rwData(r"data/sampleTestData/shahshank_10-400rpm-1 mm from top.ang")


main()