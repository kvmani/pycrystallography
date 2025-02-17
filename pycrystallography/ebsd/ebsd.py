'''
Created on 27-Jun-2019

@author: K V Mani Krishna
'''

import os
import sys
import logging
import re
import string
import copy
from random import shuffle
import h5py
from tqdm import tqdm
import copy
from orix.quaternion import Orientation as Ori
from orix.quaternion.symmetry import Oh
from orix.quaternion.symmetry import C4, C2


try:
    from pycrystallography.core.orientation import Orientation
    # from pycrystallography.core.quaternion import Quaternion
    # from pycrystallography.core.millerDirection import MillerDirection
    # from pycrystallography.core.millerPlane import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation import CrystalOrientation as CrysOri
    # from pycrystallography.core.orientationRelation import OrientationRelation as OriReln
    # from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    # from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    # import pycrystallography.utilities.pyCrystUtilities as pyCrysUt
    # import pycrystallography.utilities.pymathutilityfunctions as pmut
except:
    print("Unable to find the pycrystallography package!!! trying to now alter the system path !!")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname('..'))
    sys.path.insert(0, os.path.dirname('../../pycrystallography'))
    sys.path.insert(0, os.path.dirname('../../..'))

    for item in sys.path:
        print(f"Updated Path : {item}")
    from pycrystallography.core.orientation import Orientation
    # from pycrystallography.core.quaternion import Quaternion
    # from pycrystallography.core.millerDirection import MillerDirection
    # from pycrystallography.core.millerPlane import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation import CrystalOrientation as CrysOri
    # from pycrystallography.core.orientationRelation import OrientationRelation as OriReln
    # from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    # from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    # import pycrystallography.utilities.pyCrystUtilities as pyCrysUt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imsave
# from absl.logging import ABSLLogger
# from pycrystallography.utilities import pymathutilityfunctions as mtu
import pathlib

degree = np.pi/180


class Ebsd(object):
    '''
    classdocs
    '''

    def __init__(self, filePath=None, logger=None):
        
        '''
        Constructor
        '''
        self._sampleName = ''
        self._ebsdFilePath = filePath
        self._stepSize = None
        self._isSquareGrid = True
        self._phases = [None]
        self._header = ''
        self._extn = ''
        self._data = None
        self._nHeaderLines = None
        self._ebsdFormat = None
        self._logger = logger
        self._autoName = "ebsd"  # used for creating the data set name automatically
        self._isCropped = False  # useful to test if the scan data is cropped after loading
        # set to true when all angles are projected to fundamental zone
        self._isEulerAnglesReduced = False

    def fromAng(self, fileName, isReducedToFundamentalZone = False):
        """
        method for reading ang Files
        """

        count = 0
        header = []
        logging.debug("Attempting to read the file :"+fileName)
        ebsdFilebaseName = os.path.basename(fileName)
        self._ebsdFilePath = fileName
        readPriasData = False
        with open(fileName, "r") as f:
            for line in f:
                if line[0] == "#":
                    header.append(line)
                    if "# COLUMN_COUNT" in line:
                        self.nColumns = int(line.split(' ')[-1])
                    elif "# Column 11: PRIAS Bottom Strip" in line:
                        readPriasData = True

                    elif "# COLUMN_HEADERS" in line:
                        self.columnNames = line.replace(
                            "# COLUMN_HEADERS: ", "").split(" ")
                    elif "# XSTEP" in line:
                        self.xStep = float(line.split(' ')[-1])
                    elif "# YSTEP" in line:
                        self.yStep = float(line.split(' ')[-1])
                    elif "# NCOLS_ODD:" in line:
                        self.nXPixcels = int(line.split(' ')[-1])
                    elif "# NROWS:" in line:
                        self.nYPixcels = int(line.split(' ')[-1])
                    elif "# GRID:" in line:
                        self._gridType = line.split(' ')[-1]
                        if "hex" in self._gridType.lower():
                            logger.warning(
                                f"Detected the scan to have hexagonal grid. I am not sure if the things are going to work well!!!!")

                    else:
                        continue
                else:
                    break

        self._nHeaderLines = len(header)
        self._header = header
        self._ebsdFormat = "ang"
        self._isEulerAnglesReduced = isReducedToFundamentalZone
        self.columnNames = [
            element for element in self.columnNames if element != "index"]

        if readPriasData:
            dType = {"phi1": np.float16, "PHI": np.float64, "phi2": np.float64,
                     "x": np.float16, "y": np.float16,
                     "IQ": np.double, "CI": np.double, "Phase": np.uint8,
                     "SEM": np.int_, "FIT": np.float16,
                     "PRIAS_Bottom": np.float16, "PRIAS_Center": np.float16, "PRIAS_Top": np.float16}
            columnNames = ["phi1", "PHI", "phi2", "x",
                           "y", "IQ", "CI", "Phase", "SEM", "FIT", "PRIAS_Bottom", "PRIAS_Center", "PRIAS_Top"]
        else:
            dType = {"phi1": np.float16, "PHI": np.float64, "phi2": np.float64,
                     "x": np.float16, "y": np.float16,
                     "IQ": np.double, "CI": np.double, "Phase": np.uint8,
                     "SEM": np.int_, "FIT": np.float16}
            columnNames = ["phi1", "PHI", "phi2", "x",
                           "y", "IQ", "CI", "Phase", "SEM", "FIT"]

        self._data = pd.read_csv(self._ebsdFilePath, names=columnNames, dtype=dType, skiprows=self._nHeaderLines,
                                 sep="\s+|\t+|\s+\t+|\t+\s+")
        self._autoName = ebsdFilebaseName
        self.readPhaseFromAng()
        self.__makeEulerData()


    def fromHDF(self, fileName):
        """
        Method for reading HDF files and converting them to a format similar to ANG reader.
        """
        header = []
        logging.debug("Attempting to read the HDF file: " + fileName)
        ebsdFilebaseName = os.path.basename(fileName)
        self._ebsdFilePath = fileName
        readPriasData = False

        # Open the HDF5 file
        with h5py.File(fileName, "r") as h5file:
            # Identify the target dataset folder
            target_dataset_name = next(
                name for name in h5file if name not in ["Manufacturer", "Version"]
            )
            header_group = h5file[f"/{target_dataset_name}/EBSD/Header"]
            data_group = h5file[f"/{target_dataset_name}/EBSD/Data"]
            print(header_group)
            # Read Grid Properties
            self.xStep = header_group["Step X"][()]
            self.yStep = header_group["Step Y"][()]
            self.nXPixcels = int(header_group["nColumns"][()])
            self.nYPixcels = int(header_group["nRows"][()])
            grid_type_value = header_group["Grid Type"][()]
            if isinstance(grid_type_value, (np.ndarray, list)):
                self._gridType = grid_type_value[0].decode() if isinstance(grid_type_value[0], bytes) else grid_type_value[0]
            else:
                self._gridType = grid_type_value.decode() if isinstance(grid_type_value, bytes) else grid_type_value

            # Handle hexagonal grids
            if "hex" in self._gridType.lower():
                logging.warning(
                    "Detected the scan to have hexagonal grid. This might cause issues!"
                )

            # Read PRIAS data detection
            prias_columns = ["PRIAS Bottom Strip", "PRIAS Center", "PRIAS Top"]
            readPriasData = all(col in data_group for col in prias_columns)

            # Define data columns
            columnNames = [
                "phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase", "SEM", "FIT"
            ]
            dType = {
                "phi1": np.float16, "PHI": np.float64, "phi2": np.float64,
                "x": np.float16, "y": np.float16,
                "IQ": np.double, "CI": np.double, "Phase": np.uint8,
                "SEM": np.int_, "FIT": np.float16
            }

            if readPriasData:
                columnNames += ["PRIAS_Bottom", "PRIAS_Center", "PRIAS_Top"]
                dType.update({
                    "PRIAS_Bottom": np.float16,
                    "PRIAS_Center": np.float16,
                    "PRIAS_Top": np.float16
                })

            # Map HDF dataset names to column data
            column_mapping = {
                "phi1": "Phi1", "PHI": "Phi", "phi2": "Phi2",
                "x": "X Position", "y": "Y Position",
                "IQ": "IQ", "CI": "CI", "Phase": "Phase",
                "SEM": "SEM Signal", "FIT": "Fit",
                "PRIAS_Bottom": "PRIAS Bottom Strip",
                "PRIAS_Center": "PRIAS Center Square",
                "PRIAS_Top": "PRIAS Top Strip",
            }

            # Read data into a dictionary
            data_dict = {col: data_group[column_mapping[col]][()] for col in columnNames}

        # Convert to DataFrame
        self._data = pd.DataFrame(data_dict)
        self._ebsdFormat = "hdf"
        self.columnNames = columnNames
        self._autoName = ebsdFilebaseName

        logging.info("Successfully read the HDF file.")
        self.__makeEulerData()


    def fromCtf(self, fileName, isReducedToFundamentalZone = False):
        """
        method for reading the ctf files
        """
        searchString = "Phase    X    Y    Bands    Error    Euler1    Euler2    Euler3    MAD    BC    BS"
        searchString = searchString.translate(
            {ord(c): None for c in string.whitespace})

        count = 0
        header = []
        logging.debug("Attempting to read the file :"+fileName)
        self._ebsdFilePath = fileName
        with open(fileName, "r") as f:
            for line in f:
                count = count+1
                header.append(line)
                tmpLine = copy.copy(line)
                tmpLine = tmpLine.translate(
                    {ord(c): None for c in string.whitespace})

                if searchString in tmpLine:
                    break
                elif "Phases" in line:
                    self.nPhases = int(line.split('\t')[1])
                elif "XCells" in line:
                    self.nXPixcels = int(line.split('\t')[1])
                elif "YCells" in line:
                    self.nYPixcels = int(line.split('\t')[1])
                elif "XStep" in line:
                    self.xStep = float(line.split('\t')[1])
                elif "YStep" in line:
                    self.yStep = float(line.split('\t')[1])
                elif "AcqE1" in line:
                    self.AcqE1 = float(line.split('\t')[1])
                elif "AcqE2" in line:
                    self.AcqE2 = float(line.split('\t')[1])
                elif "AcqE3" in line:
                    self.AcqE3 = float(line.split('\t')[1])
                else:
                    continue
        
        self._isEulerAnglesReduced = isReducedToFundamentalZone
        self._ebsdFormat = 'ctf'
        self._isSquareGrid = True
        self._gridType = "SqrGrid"
        self._header = header
        self._nHeaderLines = count
        logging.info(
            "The header line count in the given EBSD file is :"+str(self._nHeaderLines))
        self.readPhaseFromCtfString(line=self._header[13])
        dType = {"Phase": np.uint8,   "X": np.float16, "Y": np.float16, "Bands": np.uint8, "Error": np.int8, "Euler1": np.float64,
                 "Euler2": np.float64, "Euler3": np.float64, "MAD": np.float16, "BC": np.uint16, "BS": np.uint16}
        columnNames = ["Phase",    "X",    "Y",    "Bands",    "Error",
                       "Euler1",   "Euler2",   "Euler3",   "MAD",  "BC",   "BS"]
        self._data = pd.read_csv(self._ebsdFilePath, names=columnNames,
                                 dtype=dType, skiprows=self._nHeaderLines, sep="\t")
        self.__makeEulerData()

        logging.info(
            "Completed the reading of the ebsd file : summary is : \n" + str(self._data.describe()))

    def call_makeEulerData(self):
        self.__makeEulerData()

    def __makeEulerData(self):

        eulerLimits = (-1., 1.,)
        eulerLimitsImage = (0, 255)
        if(self._isEulerAnglesReduced):
            latticeLimits = np.asarray(self._lattice._EulerLimits)
        else:
            latticeLimits = [np.pi * 2, np.pi, np.pi * 2]
        data = self._data
        xPixcels = self.nXPixcels
        yPixcels = self.nYPixcels
        degree_to_radian = np.pi / 180.0

        if "ang" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T

            # --- Detect if angles are in degrees (heuristic check) ---
            max_angle = np.max(eulerData)
            if max_angle > 2.0 * np.pi:
                # Convert to radians
                eulerData *= degree_to_radian
                # Update the DataFrame columns
                self._data["phi1"] = eulerData[:, 0]
                self._data["PHI"]  = eulerData[:, 1]
                self._data["phi2"] = eulerData[:, 2]
                logging.info("Detected degrees in .ang file; converted to radians.")

        elif "ctf" in self._ebsdFormat:
            eulerData = np.array([data["Euler1"], data["Euler2"], data["Euler3"]]).T

            max_angle = np.max(eulerData)
            if max_angle > 2.0 * np.pi:
                eulerData *= degree_to_radian
                self._data["Euler1"] = eulerData[:, 0]
                self._data["Euler2"] = eulerData[:, 1]
                self._data["Euler3"] = eulerData[:, 2]
                logging.info("Detected degrees in .ctf file; converted to radians.")

        elif "hdf" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T

            max_angle = np.max(eulerData)
            if max_angle > 2.0 * np.pi:
                eulerData *= degree_to_radian
                self._data["phi1"] = eulerData[:, 0]
                self._data["PHI"]  = eulerData[:, 1]
                self._data["phi2"] = eulerData[:, 2]
                logging.info("Detected degrees in .hdf file; converted to radians.")

        else:
            raise ValueError(
                f"Unknown format {self._ebsdFormat} only ctf, ang and hdf are supported as of now"
            )


        shape = yPixcels, xPixcels,
        eulerData = np.stack([eulerData[:, 0].reshape(shape), eulerData[:, 1].reshape(
            shape), eulerData[:, 2].reshape(shape)], axis=2)
        oriDataInt = np.zeros_like(eulerData, dtype=np.uint8,)
        self._eulerDataRaw = eulerData  # euler angle data with out nornamlizing
        oriDataInt[:, :, 0] = np.interp(eulerData[:, :, 0], (0, latticeLimits[0]), eulerLimitsImage).astype(np.uint8)
        oriDataInt[:, :, 1] = np.interp(eulerData[:, :, 1], (0, latticeLimits[1]), eulerLimitsImage).astype(np.uint8)
        oriDataInt[:, :, 2] = np.interp(eulerData[:, :, 2], (0, latticeLimits[2]), eulerLimitsImage).astype(np.uint8)

        oriDataNormalized = np.zeros_like(eulerData, dtype=np.float64,)
        oriDataNormalized[:, :, 0] = np.interp(eulerData[:, :, 0], (eulerData[:, :, 0].min(
        ), eulerData[:, :, 0].max()), eulerLimits)
        oriDataNormalized[:, :, 1] = np.interp(eulerData[:, :, 1], (eulerData[:, :, 1].min(
        ), eulerData[:, :, 1].max()), eulerLimits)
        oriDataNormalized[:, :, 2] = np.interp(eulerData[:, :, 2], (eulerData[:, :, 2].min(
        ), eulerData[:, :, 2].max()), eulerLimits)

        # data suitable for image display typically between 0-255
        self._oriDataInt = oriDataInt
        # data suitable for normalized npz format for ML work
        self._oriDataNormalized = oriDataNormalized
        self._shape = shape
        self._numPixcels = xPixcels*yPixcels
        # self.writeEulerAsPng()

    def readPhaseFromCtfString(self, line=None):
        tmpLine = copy.copy(line)
        strList = []
        count = 0
        while (1):
            count += 1
            ind = tmpLine.find("\t")
            strList.append(tmpLine[:ind])
            tmpLine = tmpLine[ind+1:]
            print(tmpLine)
            if count > 5 or ind < 0:
                break

        latticeConstants = strList[0].split(';')
        latticeConstants = np.asarray([float(i) for i in latticeConstants])
        latticeAngles = strList[1].split(';')
        latticeAngles = np.asarray([float(i) for i in latticeAngles])
        self._intCrystNumber = int(strList[4])
        self._phaseName = strList[2]
        self._nBands = int(strList[3])  # max no. of kikuch bands
        # self.crystalLattice=
        logging.debug("lattice const and angles : " +
                      str(latticeConstants)+str(latticeAngles))
        logging.warning(
            "At the moment only the cubic and hcp are implemeted !! Others will follow soon")
        if np.allclose(latticeConstants-latticeConstants[0], [0., 0., 0.]) and np.allclose(latticeAngles, [90., 90., 90.]) and self._intCrystNumber >= 195:
            self.crystalSystem = 'cubic'
            self._lattice = olt.cubic(
                a=latticeConstants[0], pointgroup='m-3m', name=self._phaseName)
        elif np.allclose(latticeConstants[:2], -latticeConstants[0], [0., 0.,]) and np.allclose(latticeAngles, [90., 90., 120.]) and self._intCrystNumber >= 168 and self._intCrystNumber <= 194:
            self.crystalSystem = 'hexagonal'
            self._lattice = olt.hexagonal(
                a=latticeConstants[0], c=latticeConstants[2], pointgroup='6/mmm', name=self._phaseName)
        else:
            logging.critical(
                "Unknown crystal system : the line being parsed is : "+line)
            raise ValueError("Unknown system !!!!" + line)
        logging.info(
            "sucesffuly parsed the phase : and it is \n"+str(self._lattice))

    def makePropertyMap(self, property=None, scaleRange=None, ifWriteImage=False):
        """
        make a map out of any property of the ebsd data file
        property : a str indicating the which field of the EBSD data one wants to plot as Image 
        """

        if property in self._data.columns:
            data = self._data[property]
            if scaleRange is not None:
                dataLimitsRequired = scaleRange
            # mapData = np.zeros_like(data,dtype=np.uint8,)
                mapData = np.interp(
                    data, (data.min(), data.max()), dataLimitsRequired).astype(np.uint8)
            else:
                mapData = data
            mapData = np.array(mapData).reshape(self._shape)
            if ifWriteImage:
                imName = self._ebsdFilePath[:-4]+"_"+property+".png"
                im = Image.fromarray(mapData)
                im.save(imName)
            return mapData

        else:
            logging.critical("NO field named "+property +
                             " exists in the ebsd data")
            raise ValueError("NO field named "+property +
                             " exists in the ebsd data")

    def makeEulerDatainImFormat(self, saveAsNpy=False, pathName=None):
        """
        returns the euler data in the form of a MXNX3 image suitable for ML work
        if path is provided, it can save the data in to respective file as npy format
        """

        if "ctf" in self._ebsdFormat:
            euler1 = self.makePropertyMap(property="Euler1")
            euler2 = self.makePropertyMap(property="Euler2")
            euler3 = self.makePropertyMap(property="Euler3")
        else:
            euler1 = self.makePropertyMap(property="phi1")
            euler2 = self.makePropertyMap(property="PHI")
            euler3 = self.makePropertyMap(property="phi2")

        eulerDataMap = np.stack([euler1, euler2, euler3], axis=-1)
        if saveAsNpy:
            if pathName is None:
                pathName = self._ebsdFilePath[:-4] + ".npy"
            logging.debug("the shape of the data is :" +
                          str(eulerDataMap.shape))
            np.save(pathName, eulerDataMap)
            logging.debug("Saved the file :" + pathName)

        return eulerDataMap

    def writeEulerAsPng(self, pathName=None, showMap=False):
        """
        utility method for saving the euler map as png image
        """

        self.__makeEulerData()
        im = Image.fromarray(self._oriDataInt)
        logging.debug("the shape of the euler map is :" +
                      str(self._oriDataInt.shape))
        if pathName is None:
            tiffName = self._ebsdFilePath[:-3]+"png"
        else:
            tiffName = pathName
        im.save(tiffName)
        logging.debug("Saved the file :"+tiffName)
        if showMap:
            plt.imshow(self._oriDataInt)
            plt.title("Euler color map")
            plt.show()

    def addNewDataColumn(self, dataColumnName="newCol", fillValues=0,):
        """
        method to add a new data colum to the exisitng ebsd data file
        """
        if dataColumnName in self._data.columns:
            logging.warning("The supplied coulm name :" + dataColumnName +
                            " already exists and hence gettig over written !!!")

        if isinstance(fillValues, np.ndarray) and fillValues.size == self._numPixcels:
            self._data[dataColumnName] = fillValues
        else:
            self._data[dataColumnName] = np.full(
                (self._numPixcels,), fillValues)
        logging.info("Succesfully added the column :" +
                     dataColumnName+" to the ebsd data set !!!")

    def writeCtf(self, pathName=None):
        """
        Writes self._data to a .ctf file with a column header line and 
        tab-separated columns in this order:
        Phase, X, Y, Bands, Error, Euler1, Euler2, Euler3, MAD, BC, BS
        using a vectorized approach for speed.
        """
        df = self._data.copy()

        # 1) Convert Euler angles from radians to degrees if needed
        df["Euler1"] = df["Euler1"] * (180.0 / np.pi)
        df["Euler2"] = df["Euler2"] * (180.0 / np.pi)
        df["Euler3"] = df["Euler3"] * (180.0 / np.pi)

        # 2) Determine output path
        if pathName is None:
            pathName = self._ebsdFilePath[:-4] + "_mod.ctf"

        # 3) If "ctf" not in format, skip
        if "ctf" not in self._ebsdFormat:
            logging.warning("Not a CTF format; skipping write.")
            return

        # 4) Open file once
        with open(pathName, "w") as f:
            # 4a) Write header lines
            for line in self._header[:-1]:
                f.write(line)

            # 4b) Write the column names line
            f.write("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n")

            # 5) Build string columns for each DataFrame row using vectorized string formatting.
            # We'll create new columns to store the formatted strings, then combine them.

            # For each column, define the format and convert to string
            # Example:
            #   Phase as int
            #   X as 4 decimals float
            #   etc.

            # Convert numeric columns to the desired string format
            df["Phase_str"]  = df["Phase"].astype(int).astype(str)
            df["X_str"]      = df["X"].apply(lambda v: f"{v:.4f}")
            df["Y_str"]      = df["Y"].apply(lambda v: f"{v:.4f}")
            df["Bands_str"]  = df["Bands"].astype(int).astype(str)
            df["Error_str"]  = df["Error"].astype(int).astype(str)
            df["E1_str"]     = df["Euler1"].apply(lambda v: f"{v:.4f}")
            df["E2_str"]     = df["Euler2"].apply(lambda v: f"{v:.4f}")
            df["E3_str"]     = df["Euler3"].apply(lambda v: f"{v:.4f}")
            df["MAD_str"]    = df["MAD"].apply(lambda v: f"{v:.4f}")
            df["BC_str"]     = df["BC"].astype(int).astype(str)
            df["BS_str"]     = df["BS"].astype(int).astype(str)

            # 6) Concatenate them with tab separators
            # We'll build a single column "line" that includes all data + "\n"
            df["line"] = (
                df["Phase_str"]  + "\t" +
                df["X_str"]      + "\t" +
                df["Y_str"]      + "\t" +
                df["Bands_str"]  + "\t" +
                df["Error_str"]  + "\t" +
                df["E1_str"]     + "\t" +
                df["E2_str"]     + "\t" +
                df["E3_str"]     + "\t" +
                df["MAD_str"]    + "\t" +
                df["BC_str"]     + "\t" +
                df["BS_str"]     + "\n"
            )

            # 7) Now we have a column of lines in df["line"]
            # We can write them all at once to file for speed

            f.writelines(df["line"].tolist())

        logging.info(f"Wrote the EBSD data file: {pathName} successfully!!!")
        
    def writeNpyFile(self, pathName=None):
        """
        Export to npy format the raw data of the image of field you want to export , example is euler angles
        """
        if pathName is None:
            pathName = self._ebsdFilePath[:-4]+".npy"

        im = self._eulerDataRaw.astype(np.float32)
        logging.debug("the shape of the data is :"+str(im.shape))
        npyName = pathName
        np.save(pathName, self._eulerDataRaw)
        logging.info("Saved the file :"+pathName)


    def computeMaskedFraction(self, maskImge, threshold, maskSize=[30,30]):
        """
        Loads the image (if maskImge is str), applies the given threshold,
        resizes to maskSize (or self._shape, as needed), and returns the fraction
        of pixels that would be True.
        """
        if isinstance(maskImge, str):
            # Load
            mask = Image.open(maskImge)
            if mask.mode != "P":
                mask = mask.convert("L")
            # Threshold
            bin_mask = mask.point(lambda x: 255 if x > threshold else 0, '1')
            # Resize if you prefer to self._shape or maskSize:
            bin_mask = bin_mask.resize(self._shape, Image.Resampling.NEAREST)
            bin_mask = np.array(bin_mask)
            bin_mask = (bin_mask != 0)
        elif isinstance(maskImge, np.ndarray):
            # If already an array, threshold might not apply if it's bool
            # But in principle, if it's not bool, apply threshold logic similarly:
            if maskImge.dtype == np.bool_:
                bin_mask = maskImge
            else:
                bin_mask = (maskImge > threshold)
            if maskSize != list(bin_mask.shape):
                bin_mask = Image.fromarray(bin_mask.astype(np.uint8)*255)
                bin_mask = bin_mask.resize(maskSize, Image.Resampling.NEAREST)
                bin_mask = np.array(bin_mask)
                bin_mask = (bin_mask != 0)
        else:
            raise ValueError("Invalid maskImge type for thresholding.")
        
        # Now compute fraction
        total_pixels = bin_mask.size
        masked_pixels = bin_mask.sum()
        fraction = masked_pixels / total_pixels
        return fraction, bin_mask

    def applyMaskWithPercentage(
        self, 
        maskImge, 
        desired_percent=0.20,   # e.g., aim to mask 20% of the image
        maskSize=[30, 30],
        maskLocation=None, 
        displayImage=False,
        max_iters=15,
        tolerance=0.05
    ):
        
        
        # Start with a middle threshold, e.g. 128
        threshold = 128
        # We'll adjust threshold in steps initially (e.g. �32),
        # and then reduce step_size each iteration to refine
        step_size = 12

        # Track the best threshold found so far
        best_threshold = threshold
        best_fraction = 0.0
        best_diff = float('inf')

        # Convert from e.g. 30% to 0.30
        desired = desired_percent

        # Load
        mask = Image.open(maskImge)
        if mask.mode != "P":
            mask = mask.convert("L")
        mask = np.array(mask)

        # Threshold

        for i in range(max_iters):
            # 1) Compute current fraction for the current threshold
            current_fraction, bin_mask = self.computeMaskedFraction(
                mask, threshold, maskSize=maskSize
            )

            # 2) How close are we to the desired fraction?
            diff = abs(current_fraction - desired)

            # 3) Update "best threshold" if we improved
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
                best_fraction = current_fraction

            # 4) Check if we're within absolute tolerance
            #    e.g., if desired=0.30 and tolerance=0.05 => we accept [0.25 .. 0.35]
            if diff <= tolerance:
                print(
                    f"Early stopping at iteration {i+1}, "
                    f"fraction={current_fraction*100:.2f}%, threshold={threshold}"
                )
                break

            # 5) Adjust threshold for the next iteration
            #    If current fraction < desired => we want more pixels => lower threshold
            #    else => we want fewer pixels => raise threshold
            if current_fraction < desired:
                threshold -= step_size
            else:
                threshold += step_size

            # Keep threshold in [0, 255]
            # threshold = max(0, min(255, threshold))

            # 6) Reduce step size for finer adjustments
            # step_size = max(1, step_size // 2)

        # After finishing the loop or early-stopping, we have our best threshold
        final_fraction, bin_mask = self.computeMaskedFraction(maskImge, best_threshold, maskSize=maskSize)
        print(
            f"Best threshold={best_threshold}, "
            f"Masked fraction ~ {final_fraction*100:.2f}% (desired {desired*100:.2f}%)"
        )

        # Finally, call applyMask with the best threshold
        self.applyMask(
            maskImge,
            maskSize=maskSize,
            maskLocation=maskLocation,
            displayImage=displayImage,
            threshold=best_threshold
        )


    def applyMask(self, maskImge, maskSize=[30, 30], maskLocation=None, threshold=128, displayImage=False):
        """
        maskImage : is a path to binary image or numpy boolean array.
        maskSize mXn of mask to which the given input mask is scaled. Ignored if the input mask is np.ndarray of bool type.
        maskLocation = i,j (int) of the pixcels about which the mask must be placed in the EBSD data.
        """

        self.addNewDataColumn("isModified", fillValues=False)

        if type(maskImge) == np.ndarray:
            mask = maskImge
            if maskSize != mask.shape:
                maskImg = Image.fromarray(mask)
                mask = maskImg.resize(maskSize, Image.Resampling.NEAREST)
                mask = np.array(mask)
            mask = mask.astype(bool)  # Ensure binary

        elif type(maskImge) != None:
            try:
                if(type(maskImge)) == str:
                    mask = Image.open(maskImge)
                else:
                    mask = maskImge
                # Check the mode and convert to binary-compatible if necessary
                
                assert isinstance(mask, Image.Image), "mask is not a PIL Image object, apply mask function expects image with PIL only"
                if mask.mode != "P":  # If not palette mode
                    mask = mask.convert("L")  # Convert to grayscale
                    mask = mask.point(lambda x: 255 if x > threshold else 0, '1')  # Binarize using a threshold
                mask = mask.resize(self._shape, Image.Resampling.NEAREST)
                # mask.show()
                mask = np.array(mask)
                mask = mask != 0
            except Exception as e:
                logging.fatal(e)


        else:
            raise ValueError(
                f"{type(maskImge)}: The supplied mask is neither a file path nor a valid boolen numpy array.")
        
        # assert mask.shape == self._shape, f"Mask shape {mask.shape} does not match EBSD data shape {self._shape}!"

        mainMask = np.full((self._shape), False, )

        # if maskLocation is None:
        #     maskMargin = max(maskSize)
        #     ebsdDataImageDimSmall = min(self._shape)
        #     maskLocation = np.random.randint(low = 0, high = ebsdDataImageDimSmall, size=(2,) )

        # assert mask.dtype == bool, f"The supplied mask is not of boolen type !!!! as the type is : {type()}"
#         ebsdShape = self._shape
#         startInd = min(maskLocation[0]-int(maskSize[0]/2),0), min(maskLocation[1]-int(maskSize[1]/2),0)
#         endInd = min(maskLocation[0]+int(maskSize[0]/2), ebsdShape[0]), min(maskLocation[1]+int(maskSize[1]/2),ebsdShape[1])
# #         tmp1 = np.arange(startInd[0],endInd[0])
# #         tmp2= np.arange(startInd[1],endInd[1])
        # mainMask[startInd[0]:endInd[0], startInd[1]:endInd[1]]= mask
        mask = np.array(mask)
        indx = np.where(mask.T.reshape(-1))
        indx = indx[0].tolist()
        indx_length = len(indx)
        if(indx_length == 0):
            logging.warning(f"empty mask looks like threshold is not good. skipping mask")
            return
        logging.debug(f"number of point being changes are {len(indx)}")
        logging.info("projecting to fundamental zone before applying the mask")
        # self.reduceEulerAngelsToFundamentalZone()
        # print(indx[100])
        # exit()
        max_index = np.prod(self._shape) - 1
        logging.debug(f"Max index in mask: {max(indx)}, Max allowed index: {max_index}")
        logging.debug(f"Number of indices: {indx_length}")
        assert max(indx) <= max_index, "Out of bounds index detected!"

        if "ctf" in self._ebsdFormat:
            self._data["MAD"][indx] = 5.0
            self._data["Euler1"][indx] = 360.0*degree
            self._data["Euler2"][indx] = 180.0*degree
            self._data["Euler3"][indx] = 360.0*degree
            self._data["isModified"][indx] = True
            self._data["Bands"][indx] = 0
        elif "ang" in self._ebsdFormat:
            self._data["CI"][indx] = -1.0
            self._data["phi1"][indx] = 360.0*degree
            self._data["PHI"][indx] = 180.0*degree
            self._data["phi2"][indx] = 360.0*degree
            self._data["isModified"][indx] = True
        elif "hdf" in self._ebsdFormat:
            self._data.loc[indx, "CI"] = -1.0
            self._data.loc[indx, "phi1"] = 360.0*degree
            self._data.loc[indx, "PHI"] = 180.0*degree
            self._data.loc[indx, "phi2"] = 360.0*degree
            self._data.loc[indx, "isModified"] = True
        self.__makeEulerData()
        # self.writeEulerAsPng(showMap=displayImage)

        logging.debug("Done with masking")


   



    def generateSimulatedEbsdMap(self, orientationList, simulatedEbsdOutPutFilefilePath, sizeOfGrain=2,
                                 headerFileName="hcp_header.txt", IqData=None):
        """

        :param orientationList: list of euler angles tuples in degrees example list :[[10,20,100],[0,0,0]]
        :param sizeOfGrain: (int) number of pixcels in a grain; eg: 10
        :param simulatedEbsdOutPutFilefilePath: file where you wanat to write the simulated EBSD file:
        :param headerFilePath: header that determines the symmetry and phase info etc Note only the name of the header
               need to be given the path of the header is in the ../../data/programedata/.
               possible values are : hcp_header.txt for hcp Zr
                                   : bcc_header.txt for bcc Zr
        :param IqData : default None, If specified as a list of values (float), they will be assigned to the each grain. Length of this list must match with the number of grains!
        :return: simulated EBSD map in ang format (actually does not return but writes the required ang file.
        """

        # InputIQData = rand(size(InputGrainOriData), 1) * 10
        XStep = 1.0
        YStep = 1.0
        degree = np.pi/180
        NoOfGrains = len(orientationList)
        NoOfGrains = int(np.square(np.ceil(np.sqrt(NoOfGrains))))
        TotalPixels = int(
            np.square((np.ceil(np.sqrt(NoOfGrains)) * sizeOfGrain)))
        NoOfXPixcels = int(np.sqrt(TotalPixels))
        NoOfYPixcels = int(np.sqrt(TotalPixels))
        # Format is ph1  phi phi2 in third dimension for each X Y
        OriData = np.zeros((NoOfXPixcels, NoOfYPixcels)) - 0.0
        # Format is ph1  phi phi2 in third dimension for each X Y
        OriData = np.dstack((OriData, OriData, OriData))
        IQData = np.zeros((NoOfXPixcels, NoOfYPixcels)) + 0.
        CIData = np.zeros((NoOfXPixcels, NoOfYPixcels)) - 1.0
        NumberOfGrainsPerRow = NoOfXPixcels / sizeOfGrain
        NumberOfGrainsPerColumn = NoOfYPixcels / sizeOfGrain
        if IqData is None:
            InputIQData = np.round(np.random.random(
                (len(orientationList), 1)) * 10, 2)
        else:
            if not isinstance(IqData, list):
                raise TypeError(
                    f"The IQ data must of of type 'list' but was suplied with type {type(IqData)}")

            assert len(orientationList) == len(
                IqData), f"The length of the IQ data supplied is {len(IqData)} while number of Orienations are {len(orientationList)}"

            InputIQData = np.array(IqData)

        for i in range(len(orientationList)):
            (row, col) = mtu.ind2sub(
                (NumberOfGrainsPerRow, NumberOfGrainsPerColumn), i)
            Grain_XPixcels_start = row * sizeOfGrain
            Grain_XPixcels_end = Grain_XPixcels_start + sizeOfGrain
            Grain_YPixcels_start = col * sizeOfGrain
            Grain_YPixcels_end = Grain_YPixcels_start + sizeOfGrain
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,
                    Grain_YPixcels_start: Grain_YPixcels_end, 0] = orientationList[i][0]*degree
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,
                    Grain_YPixcels_start: Grain_YPixcels_end, 1] = orientationList[i][1]*degree
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,
                    Grain_YPixcels_start: Grain_YPixcels_end, 2] = orientationList[i][2]*degree
            IQData[Grain_XPixcels_start: Grain_XPixcels_end,
                   Grain_YPixcels_start: Grain_YPixcels_end] = InputIQData[i]
            CIData[Grain_XPixcels_start: Grain_XPixcels_end,
                   Grain_YPixcels_start: Grain_YPixcels_end] = 1.0
            # IQData [Grain_XPixcels_start: Grain_XPixcels_end][Grain_YPixcels_start: Grain_YPixcels_end] = InputIQData[i]
            # CIData [Grain_XPixcels_start: Grain_XPixcels_end][Grain_YPixcels_start: Grain_YPixcels_end] = 1.0

        x = np.linspace(1, XStep*NoOfXPixcels, NoOfXPixcels)-1
        y = np.linspace(1, YStep*NoOfYPixcels, NoOfYPixcels)-1
        X, Y = np.meshgrid(x, y)

        OriData = np.reshape(OriData, (NoOfXPixcels * NoOfYPixcels, 3))
        IQData = np.reshape(IQData, (NoOfXPixcels * NoOfYPixcels, 1))
        CIData = np.reshape(CIData, (NoOfXPixcels * NoOfYPixcels, 1))
        FinalData = np.zeros((OriData.shape[0], 10))
        FinalData[:, 6] = 1
        FinalData[:, 3] = X.flatten()
        FinalData[:, 4] = Y.flatten()
        FinalData[:, 0:3] = OriData
        # Ind = np.where(FinalData[:, 1] < 0); ### pixcels  not assigned ori
        FinalData[:, 5] = IQData.flatten()
        FinalData[:, 6] = CIData.flatten()  # '##-1; ## Assgning   CI - 1
        # note this is made to plot any given property  value to  be  sent  to  IQ data
        FinalData[:, 8] = 50.0
        FinalData[:, 9] = 1.5
        dType = {"phi1": np.float64, "PHI": np.float64, "phi2": np.float64, "x": np.float16, "y": np.float16,
                 "IQ": np.float64, "CI": np.float64, "Phase": np.uint8,
                 "SEM": np.float64, "FIT": np.float64, }  # sem means detecor signal
        columnNames = ["phi1", "PHI", "phi2", "x",
                       "y", "IQ", "CI", "Phase", "SEM", "FIT"]

        self._data = pd.DataFrame(FinalData, columns=columnNames)
        self._oriData = OriData
        self.nXPixcels = NoOfXPixcels
        self.nYPixcels = NoOfYPixcels
        self._shape = (NoOfXPixcels, NoOfYPixcels)
        for columnName in self._data:
            self._data[columnName] = self._data[columnName].astype(
                dType[columnName])

       # headerFileName = os.path.join("../../data/programeData", headerFileName)
        headerFileName = os.path.join(pathlib.Path(
            __file__).parent.parent.parent, 'data', "programeData", headerFileName)
        with open(headerFileName, "rb") as f:
            self._header = []
            for line in f:
                self._header.append(line)
        self._ebsdFilePath = simulatedEbsdOutPutFilefilePath
        self._ebsdFormat = "ang"
        self._isSquareGrid = True
        self._gridType = "SqrGrid"
        self._nHeaderLines = len(self._header)
        self.writeAng(pathName=simulatedEbsdOutPutFilefilePath)

    @staticmethod
    def __replace_all_except_last__(input_string, old_substring, new_substring):
        occurrences = input_string.count(old_substring)
        if occurrences <= 1:
            return input_string

        parts = input_string.rsplit(old_substring, occurrences - 1)
        modified_parts = [new_substring.join(parts[:-1])] + parts[-1:]
        return old_substring.join(modified_parts)

    def writeAng(self, pathName=None):
        """
        Writes self._data to an .ang file using a vectorized approach for speed.
        Expected columns (based on your previous example):
            phi1, PHI, phi2, x, y, IQ, CI, Phase, SEM, FIT
        With typical formatting like:
            phi1, PHI, phi2 => 5 decimals
            x, y => 4 decimals
            IQ => round(2) then 2 decimals
            CI => 2 decimals
            Phase => int (or 2-digit field)
            SEM => .3f
            FIT => .3f
        """
        # 1) Copy DataFrame (so we don't modify self._data in place)
        df = self._data.copy()

        # 2) Determine the output path
        if pathName is None:
            # Example naming logic (adjust as needed):
            basename = self.__replace_all_except_last__(
                self._autoName + "_mod.ang", ".ang", "_"
            )
            pathName = os.path.join(
                os.path.dirname(self._ebsdFilePath), basename
            )

        # 3) Check if "ang" in format
        if "ang" not in self._ebsdFormat:
            logging.warning("Not an ANG format; skipping write.")
            return

        # 4) Prepare lines
        lines_list = []

        # 4a) Convert header lines to strings
        #     The last header line might be excluded if you prefer, e.g. self._header[:-1]
        for line in self._header[:-1]:
            if isinstance(line, str):
                lines_list.append(line)
            else:
                # if line is already bytes, convert to string (utf-8) if needed
                lines_list.append(line.decode("utf-8", errors="ignore"))

        # 4b) We'll insert a newline if your headers don't end with one
        # You can skip this if each line in self._header already ends with '\n'
        if not lines_list or not lines_list[-1].endswith("\n"):
            lines_list.append("\n")

        # 5) Build vectorized string columns for each column in df
        #    Adjust decimal places to match your desired format.

        # phi1, PHI, phi2 => 5 decimals
        df["phi1_str"] = df["phi1"].apply(lambda v: f"{v:.5f}")
        df["PHI_str"]  = df["PHI"].apply(lambda v: f"{v:.5f}")
        df["phi2_str"] = df["phi2"].apply(lambda v: f"{v:.5f}")

        # x, y => 4 decimals
        df["x_str"]    = df["x"].apply(lambda v: f"{v:.4f}")
        df["y_str"]    = df["y"].apply(lambda v: f"{v:.4f}")

        # IQ => e.g. round(2) then 2 decimals (like your sample line used np.around)
        df["IQ_str"]   = df["IQ"].round(2).apply(lambda v: f"{v:.2f}")
        # CI => 2 decimals
        df["CI_str"]   = df["CI"].apply(lambda v: f"{v:.2f}")

        # Phase => int or 2-digit field
        df["Phase_str"] = df["Phase"].astype(int).astype(str)

        # SEM => .3f
        df["SEM_str"]  = df["SEM"].apply(lambda v: f"{v:.3f}")
        # FIT => .3f
        df["FIT_str"]  = df["FIT"].apply(lambda v: f"{v:.3f}")

        # 6) Combine columns into a single row string with spaces
        #    Example line: "phi1_str PHI_str phi2_str x_str y_str IQ_str CI_str Phase_str SEM_str FIT_str\n"
        df["line_str"] = (
            df["phi1_str"] + " " +
            df["PHI_str"]  + " " +
            df["phi2_str"] + " " +
            df["x_str"]    + " " +
            df["y_str"]    + " " +
            df["IQ_str"]   + " " +
            df["CI_str"]   + " " +
            df["Phase_str"]+ "  " +   # double space if you want alignment
            df["SEM_str"]  + " " +
            df["FIT_str"]  + "\n"
        )

        # 7) Convert lines to a single string (or list of strings)
        data_lines = df["line_str"].tolist()

        # 8) Concatenate header + data into final text
        # Convert your header lines list into a single string
        header_str = "".join(lines_list)
        data_str   = "".join(data_lines)
        final_output = header_str + data_str

        # 9) Write everything at once
        with open(pathName, "w") as f:
            f.write(final_output)

        logging.info("Wrote the EBSD data file: %s successfully!!!", pathName)


    @staticmethod
    def makeEbsdScanFromNpzFile(npzFile):
        """
        Thi is a method to convert an euler angle data file (npz file used in ML work)
        to equivalent ebsd scan assuming it is of cubic material.

        """
        data = np.load(npzFile)

    
    def in_euler_fundamental_region(self, orientation) -> np.ndarray:
        """
        Returns Euler angles (in radians) for each orientation, projected into the
        fundamental Euler region of the proper subgroup using a fully vectorized approach.
        
        For each input orientation, among all symmetric equivalents that lie inside the 
        fundamental region (defined by pg.euler_fundamental_region, e.g., [360,90,90] in degrees),
        we choose the one with the maximum value of (alpha + beta + gamma). In case of a tie,
        the one with the larger gamma (phi2) is preferred. If none are valid, we fallback to the
        0th equivalent.
        
        Returns
        -------
        euler_in_region : np.ndarray of shape (N, 3)
            One Euler triplet (in radians) per input orientation.
        """
        pg = orientation.symmetry.proper_subgroup

        # Generate all symmetric equivalents.
        # In many orix implementations, O.to_euler() returns either:
        #  - For one orientation: an array of shape (N_sym, 3)
        #  - For multiple orientations: an array of shape (N_sym, N, 3)
        O = pg._special_rotation.outer(orientation)
        angles = O.to_euler()  # may be shape (N_sym, 3) or (N_sym, N, 3)
        # angles = self.equivalent().to_euler()
        # Get fundamental region limits (in radians) and add a small tolerance.
        tol = 1e-3
        max_alpha_deg, max_beta_deg, max_gamma_deg = pg.euler_fundamental_region
        max_alpha = np.radians(max_alpha_deg) + tol
        max_beta  = np.radians(max_beta_deg)  + tol
        max_gamma = np.radians(max_gamma_deg) + tol

        BIG = 1e5  # A large multiplier for scoring
        # print(angles)
        # print(max_alpha, max_beta, max_gamma)
        # CASE 1: Single input orientation (angles.ndim == 2: shape (N_sym, 3))
        if angles.ndim == 2:
            # Extract individual angles; shape: (N_sym,)
            alpha = angles[:, 0]
            beta  = angles[:, 1]
            gamma = angles[:, 2]
            # Adjust gamma using the subgroup's primary axis order.
            # gamma = np.mod(gamma, 2 * np.pi / pg._primary_axis_order)
            # print("alpha\n", alpha, "\nbeta\n", beta, "\ngamma\n",gamma)
            # Determine which symmetric equivalents are inside the fundamental region.
            is_inside = (alpha <= max_alpha) & (beta <= max_beta) & (gamma <= max_gamma)
            
            # Compute a score: primary score = (alpha + beta + gamma), tie-break by gamma.
            score = (alpha + beta + gamma) * BIG + gamma  # shape (N_sym,)
            # Mask those not inside.
            score[~is_inside] = -np.inf
            
            # Select the best equivalent (fallback to 0th if all are invalid).
            best_idx = np.argmax(score)
            
            # Return as a (1,3) array.
            euler_in_region = angles[best_idx, :].reshape(1, 3)
        
        # CASE 2: Multiple input orientations (angles.ndim == 3: shape (N_sym, N, 3))
        else:
            # Separate the angles: each is shape (N_sym, N)
            alpha = angles[:, :, 0]
            beta  = angles[:, :, 1]
            gamma = angles[:, :, 2]
            # gamma = np.mod(gamma, 2 * np.pi / pg._primary_axis_order)
            # print("alpha\n", alpha, "\nbeta\n", beta, "\ngamma\n",gamma)
            # is_inside has shape (N_sym, N)
            is_inside = (alpha <= max_alpha) & (beta <= max_beta) & (gamma <= max_gamma)
            
            # Compute score with tie-breaker.
            score = (alpha + beta + gamma) * BIG + gamma  # shape (N_sym, N)
            score[~is_inside] = -np.inf
            
            # For each of the N orientations, choose the index (over the N_sym dimension) with the maximum score.
            best_idx = np.argmax(score, axis=0)  # shape (N,)
            
            # For any orientation where all equivalents were out-of-bound, np.argmax returns 0.
            N = alpha.shape[1]
            chosen_alpha = alpha[best_idx, np.arange(N)]
            chosen_beta  = beta[best_idx, np.arange(N)]
            chosen_gamma = gamma[best_idx, np.arange(N)]
            euler_in_region = np.column_stack((chosen_alpha, chosen_beta, chosen_gamma))
        
        return euler_in_region

    def reduceEulerAngelsToFundamentalZone_vectorized(self):
        self.__makeEulerData()
        if not self._isEulerAnglesReduced:
            if "ang" in self._ebsdFormat:
                self.readPhaseFromAng()
                eulerData = self._data[["phi1", "PHI", "phi2"]].to_numpy()
            elif "ctf" in self._ebsdFormat:
                self.readPhaseFromCtfString(line=self._header[13])
                eulerData = self._data[["Euler1", "Euler2", "Euler3"]].to_numpy()
            else:
                raise ValueError(
                    f"Unknown ebsd format : {self._ebsdFormat} only ang and ctf are supported")
            
            logging.info(f"performing euler angel reduction to fundamental zone!!")

            ori = Ori.from_euler(eulerData, symmetry=Oh, direction="lab2crystal", degrees=False)
            reduced_angles = self.in_euler_fundamental_region(ori)
            if "ang" in self._ebsdFormat:
                self._data["phi1"] = reduced_angles[:, 0]
                self._data["PHI"] = reduced_angles[:, 1]
                self._data["phi2"] = reduced_angles[:, 2]

            elif "ctf" in self._ebsdFormat:
                self._data["Euler1"] = reduced_angles[:, 0]
                self._data["Euler2"] = reduced_angles[:, 1]
                self._data["Euler3"] = reduced_angles[:, 2]
            else:
                raise ValueError(
                    f"Unknown ebsd format : {self._ebsdFormat} only ang and ctf are supported")
            
            self.__makeEulerData()
            self._isEulerAnglesReduced = True
            logging.info(f"Converted the euler angeles into fundamental zone")
            
        else:
            logging.info(f"Euler angles are already reduced to their fundamental zone")


    def reduceEulerAngelsToFundamentalZone(self):
        self.__makeEulerData()
        if not self._isEulerAnglesReduced:
            if "ang" in self._ebsdFormat:
                self.readPhaseFromAng()
                eulerData = self._data[["phi1", "PHI", "phi2"]].to_numpy()
            elif "ctf" in self._ebsdFormat:
                self.readPhaseFromCtfString(line=self._header[13])
                eulerData = self._data[["Euler1", "Euler2", "Euler3"]].to_numpy()
            else:
                pass
            logging.info(
                f"performing euler angel reduction to fundamental zone!!")
            
            nPoints = self.nXPixcels*self.nYPixcels

            for i in tqdm(np.arange(nPoints)):
                euler = eulerData[i]
                ori = CrysOri(orientation=Orientation(euler=euler.tolist()),
                              lattice=self._lattice)
                ori, index = ori.projectToFundamentalZone()
                tmp = ori.getEulerAngles(applyModulo=True)
                # logging.debug(f"euler angeles before : {euler}--> {tmp}")
                eulerData[i] = tmp

            if "ang" in self._ebsdFormat:
                self._data["phi1"] = eulerData[:, 0]
                self._data["PHI"] = eulerData[:, 1]
                self._data["phi2"] = eulerData[:, 2]

            elif "ctf" in self._ebsdFormat:
                self._data["Euler1"] = eulerData[:, 0]
                self._data["Euler2"] = eulerData[:, 1]
                self._data["Euler3"] = eulerData[:, 2]
            else:
                raise ValueError(
                    f"Unknown ebsd format : {self._ebsdFormat} only ang and ctf are supported")

            self.__makeEulerData()
            self._isEulerAnglesReduced = True
            logging.info(f"Converted the euler angeles into fundamental zone")

        else:
            logging.info(
                f"Euler angles are already reduced to fundamental zone and hence skipping this call")

    @staticmethod
    def __replace_numberFromHeader(text, new_integer):
        """
        Replaces the integer at the end of a header line with new_integer,
        then appends a newline. 

        This is suitable for lines like:
            "# NCOLS_ODD: 835"
            "XCells\t835"

        It looks for the last integer in the string and replaces it.
        If it doesn't find an integer, it just returns the original text.
        """
        text = text.rstrip("\n")  # remove trailing newline if any
        pattern = r"(\D*)(\d+)$"
        match = re.match(pattern, text)
        if match:
            prefix = match.group(1)  # everything except the trailing digits
            updated_text = prefix + str(new_integer) + "\n"
            return updated_text
        else:
            return text + "\n"  # fallback, or handle differently if needed

    def crop(self, crop_size=(256, 256), overlap=0.3):
        """
        Create multiple cropped EBSD objects from the current EBSD data based on the
        specified crop_size and overlap. Each cropped region is returned as a separate
        EBSD object in a list.

        Parameters
        ----------
        crop_size : tuple of int, optional
            The (height, width) in pixels for each sub-crop. Defaults to (256, 256).
        overlap : float, optional
            The fraction of overlap between consecutive crops, between 0 and 1.
            Defaults to 0.3 (i.e., 30% overlap).

        Returns
        -------
        List[Ebsd]
            A list of Ebsd objects, each containing one cropped region of the original data.
            If no valid sub-crop can be formed, returns an empty list.
        """
        import logging
        nRows, nCols = self._shape
        
        crop_height, crop_width = crop_size

        # Basic parameter checks
        if not (0 <= overlap < 1):
            raise ValueError("Overlap must be in [0, 1).")
        if crop_height <= 0 or crop_width <= 0:
            raise ValueError("Crop dimensions must be positive integers.")
        if crop_height > nRows or crop_width > nCols:
            logging.info("Requested crop_size is larger than EBSD data. No crops possible.")
            return []

        # Determine stepping
        step_x = int(crop_height * (1 - overlap))
        step_y = int(crop_width  * (1 - overlap))
        step_x = max(step_x, 1)
        step_y = max(step_y, 1)

        logging.info(f"Generating sub-crops with crop_size={crop_size}, overlap={overlap}, "
                     f"step_x={step_x}, step_y={step_y}.")

        subcrops = []

        for row_start in range(0, nRows - crop_height + 1, step_x):
            for col_start in range(0, nCols - crop_width + 1, step_y):
                new_ebsd = copy.deepcopy(self)
                shape = new_ebsd._shape  # (nRows, nCols)

                row_end = row_start + crop_height
                col_end = col_start + crop_width

                # Build mask
                mask = np.zeros(shape=shape, dtype=bool)
                mask[row_start:row_end, col_start:col_end] = True

                # Flatten to find outside-crop indices
                indx = np.where(mask.reshape(-1) == False)[0].tolist()
                if len(indx) == len(new_ebsd._data):
                    continue

                logging.info(
                    f"Removing {len(indx)} data points for sub-cropping "
                    f"({row_start}:{row_end}, {col_start}:{col_end})"
                )

                # Identify which columns hold X, Y
                if "ang" in new_ebsd._ebsdFormat:
                    xcol, ycol = "x", "y"
                elif "ctf" in new_ebsd._ebsdFormat:
                    xcol, ycol = "X", "Y"
                elif "hdf" in new_ebsd._ebsdFormat:
                    xcol, ycol = "x", "y"
                else:
                    raise ValueError(
                        f"Unknown ebsd format {new_ebsd._ebsdFormat}. "
                        "Only '.ang', '.ctf' and '.hdf' are supported as of now."
                    )

                # Calculate new origin from top-left pixel of the crop
                # ind = mtu.sub2ind(shape, row_start, col_start)
                # new_origin_x = new_ebsd._data.iloc[ind, new_ebsd._data.columns.get_loc(xcol)]
                # new_origin_y = new_ebsd._data.iloc[ind, new_ebsd._data.columns.get_loc(ycol)]

                # Shift X, Y so top-left corner is (0, 0)
                # new_ebsd._data[xcol] = new_ebsd._data[xcol] - new_origin_x # 0 to 100
                # new_ebsd._data[ycol] = new_ebsd._data[ycol] - new_origin_y

                npXcells, xStep = crop_width, self.xStep
                npYcells, yStep = crop_height, self.yStep
                x = np.arange(0, npXcells * xStep, xStep)
                y = np.arange(0, npYcells * yStep, yStep)

                # x = np.linspace(0., npXcells * xStep, npXcells, endpoint=True)
                # y = np.linspace(0., npYcells * yStep, npYcells, endpoint=True)



                # Drop points outside the crop
                new_ebsd._data = new_ebsd._data.drop(index=indx).reset_index(drop=True)
                X, Y = np.meshgrid(x, y, indexing='xy')
                X, Y = X.ravel(), Y.ravel()
                new_ebsd._data[xcol] = X
                new_ebsd._data[ycol] = Y

                # Update shape
                new_ebsd._shape = (crop_height, crop_width)

                ##################################################################
                # Update the header lines depending on the file format
                ##################################################################
                numberOfchangedValues = 0

                if "ang" in new_ebsd._ebsdFormat:
                    # Expect 3 lines to change (# NCOLS_ODD, # NCOLS_EVEN, # NROWS)
                    for i, line in enumerate(new_ebsd._header):
                        if "# NCOLS_ODD:" in line:
                            new_ebsd._header[i] = new_ebsd.__replace_numberFromHeader(
                                line, crop_width
                            )
                            numberOfchangedValues += 1
                        elif "# NCOLS_EVEN:" in line:
                            new_ebsd._header[i] = new_ebsd.__replace_numberFromHeader(
                                line, crop_width
                            )
                            numberOfchangedValues += 1
                        elif "# NROWS:" in line:
                            new_ebsd._header[i] = new_ebsd.__replace_numberFromHeader(
                                line, crop_height
                            )
                            numberOfchangedValues += 1

                    # We expect 3 changes for ANG
                    assert numberOfchangedValues == 3, (
                        f"Expected to change 3 values in ANG header, but changed {numberOfchangedValues}. "
                        f"Header is now: {new_ebsd._header}"
                    )

                elif "ctf" in new_ebsd._ebsdFormat:
                    # For CTF, typically we only have XCells, YCells lines to update
                    for i, line in enumerate(new_ebsd._header):
                        # e.g. "XCells\t835\n", "YCells\t575\n"
                        if line.startswith("XCells\t"):
                            new_ebsd._header[i] = new_ebsd.__replace_numberFromHeader(
                                line, crop_width
                            )
                            numberOfchangedValues += 1
                        elif line.startswith("YCells\t"):
                            new_ebsd._header[i] = new_ebsd.__replace_numberFromHeader(
                                line, crop_height
                            )
                            numberOfchangedValues += 1

                    # We expect 2 changes for CTF
                    assert numberOfchangedValues == 2, (
                        f"Expected to change 2 values in CTF header, but changed {numberOfchangedValues}. "
                        f"Header is now: {new_ebsd._header}"
                    )

                elif "hdf" in new_ebsd._ebsdFormat:
                    # Might not store shape in text lines, or might need a different approach
                    # If you do store them, handle similarly
                    pass

                else:
                    raise ValueError(
                        f"Unknown ebsd format {new_ebsd._ebsdFormat}. Only ANG/CTF/HDF are supported."
                    )

                # Mark as cropped, fix shape attributes
                new_ebsd._isCropped = True
                new_ebsd.nXPixcels, new_ebsd.nYPixcels = new_ebsd._shape[1], new_ebsd._shape[0]
                new_ebsd._lattice = self._lattice
                new_ebsd.__makeEulerData()

                # Adjust naming
                new_ebsd._autoName = (
                    new_ebsd._autoName
                    + new_ebsd.__replace_characters__(f"_cropped_{row_start}_{col_start}_{crop_size}")
                )

                subcrops.append(new_ebsd)

        if not subcrops:
            logging.info("No valid sub-crops were generated.")
        else:
            logging.info(f"Generated {len(subcrops)} sub-crops.")

        return subcrops


    def rotateAndFlipData(self, flipMode=None, rotate=0):
        """
        flip the ebsd data, typically useful for augmentation in machine learning
        flipMode: one of None, vertical, horizontal
        rotate one of 90,180,270 : Note that it is physcial rotation not orienation rotation
        """
        operationString = ""
        shape = self._shape
        data = self._data
        if flipMode is not None and "h" in flipMode:  # horizontal flip
            axis = 0
            operationString += f'flipped_hor'
        elif flipMode is not None and "v" in flipMode:
            axis = 1  # vertical flip
            operationString += f'flipped_vert'

        if "ang" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T
        elif "ctf" in self._ebsdFormat:
            eulerData = np.array(
                [data["Euler1"], data["Euler2"], data["Euler3"]]).T
        elif "hdf" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T
        else:
            raise ValueError(
                f"Unknown format {self._ebsdFormat} only ctf, ang and hdf are supported as of now")

        eulerData = np.stack(
            [eulerData[:, 0].reshape(shape), eulerData[:, 1].reshape(shape), eulerData[:, 2].reshape(shape)], axis=2)
                
        rotatedData = eulerData
        if rotate > 0:
            if shape[0] == shape[1]:
                logging.debug(f"attempting the ebsd data physcial rotation")

                if rotate == 90:
                    rotatedData = np.rot90(rotatedData, k=1, axes=(0, 1))
                elif rotate == 180:
                    rotatedData = np.rot90(rotatedData, k=2, axes=(0, 1))
                elif rotate == 270:
                    rotatedData = np.rot90(rotatedData, k=3, axes=(0, 1))
                else:
                    raise ValueError(
                        f"rotation provided is {rotate} but only one of [90,180,270] degree are supported")
                operationString += f'_rotation_{rotate}'

            else:
                logging.warning(f"skipping the rotation as the ebsd data is not of square shape. "
                                f"ebsd shape :{shape}")

        if flipMode is not None:
            flippedData = np.flip(rotatedData, axis=axis)
        else:
            flippedData = rotatedData

        if "ang" in self._ebsdFormat:
            self._data["phi1"] = flippedData[:, :, 0].reshape(-1)
            self._data["PHI"] = flippedData[:, :, 1].reshape(-1)
            self._data["phi2"] = flippedData[:, :, 2].reshape(-1)
        elif "ctf" in self._ebsdFormat:
            self._data["Euler1"] = flippedData[:, :, 0].reshape(-1)
            self._data["Euler2"] = flippedData[:, :, 1].reshape(-1)
            self._data["Euler3"] = flippedData[:, :, 2].reshape(-1)
        elif "hdf" in self._ebsdFormat:
            self._data["phi1"] = flippedData[:, :, 0].reshape(-1)
            self._data["PHI"] = flippedData[:, :, 1].reshape(-1)
            self._data["phi2"] = flippedData[:, :, 2].reshape(-1)
        else:
            raise ValueError(
                f"Unknown format {self._ebsdFormat} only ctf, ang and hdf are supported as of now")

        self.__makeEulerData()
        self._autoName = self._autoName+operationString
        logging.info(f"completd the flipping of orienation data.")
        logging.warning(
            f"Only the orientation data is flipped rest of data such as IQ, Fit etc are not as of now!!!")

    @staticmethod
    def __replace_characters__(input_string):
        characters_to_replace = ['(', ')', '[', ']', "'", ',']
        replacement_char = '_'

        for char in characters_to_replace:
            input_string = input_string.replace(char, replacement_char)

        return input_string

    def readPhaseFromAng(self):
        if "ang" in self._ebsdFormat:
            for line in self._header:
                if "LatticeConstants" in line:
                    tmp = line.replace("# LatticeConstants", "")
                    tmp = np.fromstring(tmp, dtype=np.double, sep=" ")
                    a, b, c, alpha, beta, gamma = tmp  # in angstroms and degrees
                    latticeConstants = tmp[0:3]
                    latticeAngles = tmp[3:6]
                    latticeParameterLine = line
                if "# MaterialName" in line:
                    self._phaseName = line.split("\t")[1].split(" ")[0]

            logging.warning(f"Currently checking the lattice parameters only for determing the crystal sytems assuming only cubic and hexagonal are proceessed. Modify this part for "
                            f"other crystal systems !!!")

            if np.allclose(latticeConstants - latticeConstants[0], [0., 0., 0.]) and np.allclose(latticeAngles,
                                                                                                 [90., 90.,
                                                                                                  90.]):
                self.crystalSystem = 'cubic'
                self._lattice = olt.cubic(a=latticeConstants[0], pointgroup='m-3m',
                                          name=self._phaseName)
                logging.debug(f"Found the {self._phaseName} to be cubic!!!")
            elif np.allclose(latticeConstants[:2], -latticeConstants[0], [0., 0., ]) and np.allclose(latticeAngles,
                                                                                                     [90., 90.,
                                                                                                      120.]):
                self.crystalSystem = 'hexagonal'
                self._lattice = olt.hexagonal(a=latticeConstants[0], c=latticeConstants[2],
                                              pointgroup='6/mmm', name=self._phaseName)
                logging.debug(
                    f"Found the {self._phaseName} to be hexagonal!!!")

            else:
                logging.critical(
                    "Unknown crystal system : the line being parsed is : " + latticeParameterLine)
                raise ValueError("Unknown system !!!!" + latticeParameterLine)

            logging.info(
                "Succesfully parse ang header and populated the crystal structure info!!!")
            return (0)


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    fileName = r'..\data\ebsdData\SuperNi-Ni-Fcc.ctf'
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # Get the current working directory of the Python interpreter
    current_working_directory = os.getcwd()
    # Compare the two paths
    if current_file_directory == current_working_directory:
        logging.info(
            "Current working directory matches the directory of the running Python file.")
    else:
        logging.info(
            "Current working directory does not match the directory of the running Python file.")
        try:
            # Change the current working directory
            os.chdir(current_file_directory)
            print("Current working directory changed to:", os.getcwd())
        except OSError as e:
            print("Error:", e)

    testFromAng = False
    if testFromAng:
        fileName = r'../../tmp/Al-B4CModelScan.ang'
        # fileName = r'C:\Users\vk0237\OneDrive - UNT System\Desktop\Model_256X256_EbsdData.ang'
        ebsd = Ebsd(logger=logger)
        logging.info(f"current dir is : {os.getcwd()}")
        ebsd.fromAng(fileName=fileName)
        # maskImg = np.full((80, 50), True, dtype=bool)
        maskImg = r"../../data/programeData/ebsdMaskFolder/3.png"
        ebsd.applyMask(maskImg, displayImage=True)
        ebsd.crop(start=(1, 1), dimensions=(150, 150))
        # ebsd.rotateAndFlipData(flipMode='vertical', rotate=90)
        # ebsd.reduceEulerAngelsToFundamentalZone()
        ebsd.writeAng()
        ebsd.writeNpyFile()
        ebsd.writeEulerAsPng(showMap=True)
        exit(-100)

    fileName = r'D:\CurrentProjects\python_trials\machineLearning\EnhanceMicroStructure\kikuchiProject\ebsdCleanUp\EBSD maps\def_map_1_sw_cleaned.ctf'
    line = '3.570;3.570;3.570    90.000;90.000;90.000    Ni-superalloy    11    225            Generic superalloy'
    ebsd = Ebsd(logger=logger)
    loadFromFile = False
    if loadFromFile:
        inputFileName = r'C:\Users\Admin\PycharmProjects\pycrystallography\tmp\uniformGridOut.txt'
        oriData = np.genfromtxt(inputFileName, delimiter=",")
        orientationList = oriData[:, 0:3]
        iQData = oriData[:, 3]
        ebsd.generateSimulatedEbsdMap(orientationList=orientationList,
                                      simulatedEbsdOutPutFilefilePath=inputFileName+'.ang', sizeOfGrain=5, IqData=iQData.tolist())
        print(oriData)
        exit(-1)

    testGenerateSimulatedEbsdMap = True
    if testGenerateSimulatedEbsdMap:
        deg = np.pi/180.0
        cubicOri1 = CrysOri(orientation=Orientation(
            euler=[0.*deg, 45.*deg, 0.*deg]), lattice=olt.cubic(1))
        cubicOri2 = CrysOri(orientation=Orientation(
            euler=[0.*deg, 30.*deg, 0.*deg]), lattice=olt.cubic(1))
        oriList = cubicOri1.symmetricSet()
        oriList = [i.projectToFundamentalZone()[0].getEulerAngles(
            units="degree", applyModulo=True).tolist() for i in oriList]
        oriList2 = cubicOri2.symmetricSet()
        projectToFundamentalZOne = False
        if projectToFundamentalZOne:
            oriList2 = [i.projectToFundamentalZone()[0].getEulerAngles(units="degree", applyModulo=True).tolist() for i in
                        oriList2]
        else:
            oriList2 = [i.getEulerAngles(units="degree").tolist()
                        for i in oriList2]

        # oriList2= [i.getEulerAngles(units="degree").tolist() for i in oriList2]
        oriList.extend(oriList2)
        shuffle(oriList)

        ebsd.generateSimulatedEbsdMap(orientationList=oriList,
                                      headerFileName="bcc_header.txt",
                                      simulatedEbsdOutPutFilefilePath=r"../../tmp/simulatedEbsdMeenakshi.ang", sizeOfGrain=5,)
        ebsd.makeEulerDatainImFormat(saveAsNpy=True)
        # ebsd.writeNpyFile(pathName=None)
        # ebsd.readPhaseFromCtfString(ebsd._header[13])
        print("Done")
        exit(-300)

        # ebsd.fromCtf(fileName)
    # ebsd.applyMask(maskImge=r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\ebsdData\maskFolder\mask (1).png',maskSize=[50,50])
    # ebsd.applyMask(maskImge=r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\ebsdData\maskFolder\mask (4).png',maskSize=[10,10])
    #
    # ebsd.writeEulerAsPng(showMap=True)
    # ebsd.makePropertyMap(property="MAD")
    # ebsd.addNewDataColumn(dataColumnName="isMod", )
    # ebsd.writeCtf(pathName=None)
