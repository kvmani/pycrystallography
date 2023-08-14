'''
Created on 27-Jun-2019

@author: K V Mani Krishna
'''

import os
import logging
import re
import string
import copy
from random import shuffle

from tqdm import tqdm

from pycrystallography.core.orientedLattice  import OrientedLattice
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.crystalOrientation import CrystalOrientation as CrysOri, CrystalOrientation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imsave
#from absl.logging import ABSLLogger
from pycrystallography.utilities import pymathutilityfunctions as mtu
import pathlib


degree=np.pi/180

class Ebsd(object):
    '''
    classdocs
    '''

    def __init__(self, filePath=None, logger=None):
        '''
        Constructor
        '''
        self._sampleName=''
        self._ebsdFilePath=filePath
        self._stepSize=None
        self._isSquareGrid=True
        self._phases=[None]
        self._header=''
        self._extn=''
        self._data=None
        self._nHeaderLines=None
        self._ebsdFormat=None
        self._logger=logger
        self._autoName="ebsd" ## used for creating the data set name automatically
        self._isCropped=False ## useful to test if the scan data is cropped after loading
        
    def fromAng(self, fileName):
        """
        method for reading ang Files
        """

        count=0
        header=[]
        logging.debug("Attempting to read the file :"+fileName)
        ebsdFilebaseName= os.path.basename(fileName)
        self._ebsdFilePath=fileName
        with open(fileName, "r") as f:
            for line in f:
                if line[0]=="#":
                    header.append(line)
                    if "# COLUMN_COUNT" in line:
                        self.nColumns = int(line.split(' ')[-1])
                    elif "# COLUMN_HEADERS" in line:
                        self.columnNames = line.replace("# COLUMN_HEADERS: ","").split(" ")
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
                            logger.warning(f"Detected the scan to have hexagonal grid. I am not sure if the things are going to work well!!!!")


                    else:
                        continue
                else:
                    break;

        self._nHeaderLines = len(header)
        self._header=header
        self._ebsdFormat="ang"
        self.columnNames= [element for element in self.columnNames if element != "index"]
        dType = {"phi1": np.float16, "PHI": np.float64, "phi2": np.float64,
                 "x": np.float16, "y": np.float16,
                 "IQ": np.double, "CI": np.double, "Phase": np.uint8,
                 "SEM":np.int, "FIT":np.float16}
        columnNames = ["phi1", "PHI", "phi2","x", "y", "IQ", "CI", "Phase", "SEM", "FIT"]
        self._data = pd.read_csv(self._ebsdFilePath, names=columnNames, dtype=dType, skiprows=self._nHeaderLines,
                                 sep="\s+|\t+|\s+\t+|\t+\s+")
        self._autoName=ebsdFilebaseName
        self.__makeEulerData()

    def fromCtf(self, fileName):
        """
        method for reading the ctf files
        """
        searchString="Phase    X    Y    Bands    Error    Euler1    Euler2    Euler3    MAD    BC    BS" 
        searchString=searchString.translate({ord(c): None for c in string.whitespace}) 
        
        count=0
        header=[]
        logging.debug("Attempting to read the file :"+fileName)
        self._ebsdFilePath=fileName
        with open(fileName, "r") as f:
            for line in f:
                count=count+1
                header.append(line)
                tmpLine = copy.copy(line)
                tmpLine=tmpLine.translate({ord(c): None for c in string.whitespace}) 
                
                if searchString in tmpLine:
                    break
                elif "Phases" in line:
                    self.nPhases = int(line.split('\t')[1])
                elif "XCells" in line:
                    self.nXPixcels= int(line.split('\t')[1])
                elif "YCells" in line:
                    self.nYPixcels= int(line.split('\t')[1])
                elif "XStep" in line:
                    self.xStep= float(line.split('\t')[1])
                elif "YStep" in line:
                    self.yStep= float(line.split('\t')[1])
                elif "AcqE1" in line:
                    self.AcqE1= float(line.split('\t')[1])
                elif "AcqE2" in line:
                    self.AcqE2= float(line.split('\t')[1])
                elif "AcqE3" in line:
                    self.AcqE3= float(line.split('\t')[1])
                else:
                    continue
                    
        self._ebsdFormat='ctf'
        self._isSquareGrid=True
        self._gridType="SqrGrid"
        self._header=header
        self._nHeaderLines=count
        logging.info("The header line count in the given EBSD file is :"+str(self._nHeaderLines)) 
        self.readPhaseFromCtfString(line=self._header[13])
        dType = {"Phase": np.uint8,   "X":np.float16,"Y":np.float16,"Bands":np.uint8, "Error":np.int8,"Euler1":np.float64,
                 "Euler2":np.float64,"Euler3":np.float64, "MAD":np.float16,"BC":np.uint16,"BS":np.uint16}
        columnNames=["Phase",    "X",    "Y",    "Bands",    "Error",    "Euler1",   "Euler2",   "Euler3",   "MAD",  "BC",   "BS"]
        self._data = pd.read_csv(self._ebsdFilePath, names=columnNames, dtype=dType, skiprows=self._nHeaderLines, sep="\t")
        self.__makeEulerData()
        
    
        logging.info("Completed the reading of the ebsd file : summary is : \n" +str(self._data.describe()))
        
    
    def __makeEulerData(self):
        
        eulerLimits=(0,255,)
        
        data = self._data
        xPixcels = self.nXPixcels
        yPixcels = self.nYPixcels
        if "ang" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T
        elif "ctf" in self._ebsdFormat:
            eulerData = np.array([data["Euler1"],data["Euler2"],data["Euler2"]]).T
        else:
            raise ValueError(f"Unknown format {self._ebsdFormat} only ctf and ang are supported as of now")

        shape=yPixcels,xPixcels,
        eulerData = np.stack([eulerData[:,0].reshape(shape), eulerData[:,1].reshape(shape),eulerData[:,2].reshape(shape)],axis=2)
        oriData = np.zeros_like(eulerData, dtype=np.uint8,)
        self._eulerData = eulerData
        oriData[:,:,0]= np.interp(eulerData[:,:,0], (eulerData[:,:,0].min(), eulerData[:,:,0].max()), eulerLimits).astype(np.uint8)
        oriData[:,:,1]= np.interp(eulerData[:,:,1], (eulerData[:,:,1].min(), eulerData[:,:,1].max()), eulerLimits).astype(np.uint8)
        oriData[:,:,2]= np.interp(eulerData[:,:,2], (eulerData[:,:,2].min(), eulerData[:,:,2].max()), eulerLimits).astype(np.uint8)
        
        self._oriData=oriData ### data suitable for image display
        self._shape=shape
        self._numPixcels = xPixcels*yPixcels
        #self.writeEulerAsPng()
        


        
    def readPhaseFromCtfString(self,line=None):
        tmpLine = copy.copy(line)
        strList=[]
        count=0
        while(1):
            count+=1
            ind = tmpLine.find("\t")
            strList.append(tmpLine[:ind])
            tmpLine=tmpLine[ind+1:]
            print(tmpLine)
            if count>5 or ind<0:
                break
            
        latticeConstants = strList[0].split(';')
        latticeConstants = np.asarray([float(i) for i in latticeConstants])
        latticeAngles = strList[1].split(';')
        latticeAngles = np.asarray([float(i) for i in latticeAngles])
        self._intCrystNumber = int(strList[4])
        self._phaseName = strList[2]
        self._nBands = int(strList[3]) ### max no. of kikuch bands
        #self.crystalLattice=
        logging.debug("lattice const and angles : "+str(latticeConstants)+str(latticeAngles))
        logging.warning("At the moment only the cubic and hcp are implemeted !! Others will follow soon")
        if np.allclose(latticeConstants-latticeConstants[0], [0.,0.,0.]) and np.allclose(latticeAngles, [90.,90.,90.]) and self._intCrystNumber>=195:
            self.crystalSystem='cubic'
            self._lattice = OrientedLattice.cubic(a=latticeConstants[0], pointgroup='m-3m', name=self._phaseName)
        elif np.allclose(latticeConstants[:2], -latticeConstants[0], [0.,0.,]) and np.allclose(latticeAngles, [90.,90.,120.]) and self._intCrystNumber>=168 and self._intCrystNumber<=194:
            self.crystalSystem='hexagonal'
            self._lattice = OrientedLattice.hexagonal(a=latticeConstants[0],c=latticeConstants[2], pointgroup='6/mmm', name=self._phaseName)
        else:
            logging.critical("Unknown crystal system : the line being parsed is : "+line)  
            raise ValueError("Unknown system !!!!" +line )  
        logging.info("sucesffuly parsed the phase : and it is \n"+str(self._lattice))             
                    
    
    def makePropertyMap(self, property=None,scaleRange=None,ifWriteImage=False):
        """
        make a map out of any property of the ebsd data file
        property : a str indicating the which field of the EBSD data one wants to plot as Image 
        """
        
        if property in self._data.columns:
            data = self._data[property]
            if scaleRange is not None:
                dataLimitsRequired=scaleRange
            #mapData = np.zeros_like(data,dtype=np.uint8,)
                mapData= np.interp(data, (data.min(), data.max()), dataLimitsRequired).astype(np.uint8)
            else:
                mapData = data
            mapData=np.array(mapData).reshape(self._shape)
            if ifWriteImage:
                imName = self._ebsdFilePath[:-4]+"_"+property+".png"
                im = Image.fromarray(mapData)
                im.save(imName)
            return mapData

        else:
            logging.critical("NO field named "+property+" exists in the ebsd data")
            raise ValueError("NO field named "+property+" exists in the ebsd data")


    
    def makeEulerDatainImFormat(self,saveAsNpy=False,pathName=None):
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

        eulerDataMap = np.stack([euler1,euler2,euler3], axis=-1)
        if saveAsNpy:
            if pathName is None:
                pathName = self._ebsdFilePath[:-4] + ".npy"
            logging.debug("the shape of the data is :" + str(eulerDataMap.shape))
            np.save(pathName, eulerDataMap)
            logging.debug("Saved the file :" + pathName)


        return eulerDataMap


    def writeEulerAsPng(self,showMap=False):
        """
        utility method for saving the euler map as png image
        """

        self.__makeEulerData()
        im = Image.fromarray(self._oriData)
        logging.debug("the shape of the euler map is :"+str(self._oriData.shape))
        tiffName = self._ebsdFilePath[:-3]+"png"
        im.save(tiffName)
        logging.debug("Saved the file :"+tiffName)
        if showMap:
            plt.imshow(self._oriData[:,:,0])
            plt.title("Euler color map")
            plt.show()
        
    
    def addNewDataColumn(self,dataColumnName="newCol", fillValues=0,):
        """
        method to add a new data colum to the exisitng ebsd data file
        """
        if dataColumnName in self._data.columns:
            logging.warning("The supplied coulm name :"+ dataColumnName+" already exists and hence gettig over written !!!")
            
            
        if isinstance(fillValues, np.ndarray) and fillValues.size==self._numPixcels:
            self._data[dataColumnName]=fillValues
        else:
            self._data[dataColumnName]=np.full((self._numPixcels,),fillValues)
        logging.info("Succesfully added the column :"+dataColumnName+" to the ebsd data set !!!")
          
        
    def writeCtf(self,pathName=None):
        if pathName is None:
            pathName = self._ebsdFilePath[:-4]+"_mod.ctf"
        
        with open(pathName, "w") as f:
            for line in self._header[:-1]:
                f.write(line)
        ##float_format="%%.4f"
        self._data.to_csv(pathName, index=False, sep="\t",  mode="a")
        logging.info("Wrote the EBSD data file :"+pathName +" Succesfully !!!")
        
        
    def writeNpyFile(self, pathName):
        """
        Export to npy format the raw data of the image of field you want to export , example is euler angles
        """
        if pathName is None:
            pathName = self._ebsdFilePath[:-4]+".npy"
        
        im = self._oriData.astype(np.float32)
        logging.debug("the shape of the data is :"+str(im.shape))
        npyName = pathName
        np.save(pathName,self._oriData)
        logging.debug("Saved the file :"+pathName)
        
        
        
    
    def applyMask(self,maskImge, maskSize=[30,30],maskLocation=None, displayImage=False):
        """
        maskImage : is a path to binary image or numpy boolean array.
        maskSize mXn of mask to which the given input mask is scaled. Ignored if the input mask is np.ndarray of bool type.
        maskLocation = i,j (int) of the picels about which the mask must be placed in the EBSD data.
        """ 
        
        self.addNewDataColumn("isModified", fillValues=False)

        if type(maskImge)==str: 
            try:
                mask = Image.open(maskImge)
                mask=mask.resize(self._shape, Image.ANTIALIAS)
                #mask.show()
                mask = np.array(mask)
                mask = mask==0
            except Exception as e:
                logging.fatal(e)

        elif type(maskImge)==np.ndarray:
            mask = maskImge
            if not maskSize==mask.shape:
                maskImg = Image.fromarray(mask)
                maskImg.resize(maskSize)
                mask = np.array(maskImg)
            maskSize=mask.shape
        else:
            raise ValueError("The supplied mask is neither a file path nor a valid boolen numpy array.")
        
        mainMask = np.full((self._shape),False, )

        # if maskLocation is None:
        #     maskMargin = max(maskSize)
        #     ebsdDataImageDimSmall = min(self._shape)
        #     maskLocation = np.random.randint(low = 0, high = ebsdDataImageDimSmall, size=(2,) )
            
        assert mask.dtype==bool, f"The supplied mask is not of boolen type !!!! as the type is : {type()}"
#         ebsdShape = self._shape
#         startInd = min(maskLocation[0]-int(maskSize[0]/2),0), min(maskLocation[1]-int(maskSize[1]/2),0)
#         endInd = min(maskLocation[0]+int(maskSize[0]/2), ebsdShape[0]), min(maskLocation[1]+int(maskSize[1]/2),ebsdShape[1])
# #         tmp1 = np.arange(startInd[0],endInd[0])
# #         tmp2= np.arange(startInd[1],endInd[1])
        #mainMask[startInd[0]:endInd[0], startInd[1]:endInd[1]]= mask
        indx = np.where(mask.T.reshape(-1))
        indx = indx[0].tolist()
        logging.debug(f"number of point being changes are {len(indx)}")
        if "ctf" in self._ebsdFormat:
            self._data["MAD"][indx] = 5.0
            self._data["Euler1"][indx] = 0.0
            self._data["Euler2"][indx] = 0.0
            self._data["Euler3"][indx] = 0.0
            self._data["isModified"][indx]=True
        elif "ang" in self._ebsdFormat:
            self._data["CI"][indx] = -1.0
            self._data["phi1"][indx] = -10.0*degree
            self._data["PHI"][indx] = -10.0*degree
            self._data["phi2"][indx] = -10.0*degree
            self._data["isModified"][indx] = True
        self.__makeEulerData()       
        self.writeEulerAsPng(showMap=displayImage)

        logging.debug("Done with masking")             
        

    def generateSimulatedEbsdMap(self, orientationList, simulatedEbsdOutPutFilefilePath,sizeOfGrain=2,
                                 headerFileName="hcp_header.txt",IqData=None):
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

        ##InputIQData = rand(size(InputGrainOriData), 1) * 10
        XStep = 1.0
        YStep = 1.0
        degree = np.pi/180
        NoOfGrains = len(orientationList)
        NoOfGrains = int(np.square(np.ceil(np.sqrt(NoOfGrains))))
        TotalPixels = int(np.square((np.ceil(np.sqrt(NoOfGrains)) * sizeOfGrain)))
        NoOfXPixcels = int(np.sqrt(TotalPixels))
        NoOfYPixcels = int(np.sqrt(TotalPixels))
        OriData = np.zeros(( NoOfXPixcels, NoOfYPixcels)) - 0.0 #### Format is ph1  phi phi2 in third dimension for each X Y
        OriData = np.dstack(( OriData, OriData, OriData))  #### Format is ph1  phi phi2 in third dimension for each X Y
        IQData = np.zeros((NoOfXPixcels, NoOfYPixcels)) + 0.
        CIData = np.zeros((NoOfXPixcels, NoOfYPixcels)) - 1.0
        NumberOfGrainsPerRow = NoOfXPixcels / sizeOfGrain
        NumberOfGrainsPerColumn = NoOfYPixcels / sizeOfGrain
        if IqData is None:
            InputIQData = np.round(np.random.random((len(orientationList),1)) * 10,2)
        else:
            if not isinstance(IqData,list):
                raise TypeError (f"The IQ data must of of type 'list' but was suplied with type {type(IqData)}")

            assert len(orientationList)==len(IqData), f"The length of the IQ data supplied is {len(IqData)} while number of Orienations are {len(orientationList)}"

            InputIQData = np.array(IqData)

        for i in range(len(orientationList)):
            (row , col) = mtu.ind2sub((NumberOfGrainsPerRow, NumberOfGrainsPerColumn), i)
            Grain_XPixcels_start = row * sizeOfGrain
            Grain_XPixcels_end = Grain_XPixcels_start + sizeOfGrain
            Grain_YPixcels_start = col * sizeOfGrain
            Grain_YPixcels_end = Grain_YPixcels_start + sizeOfGrain
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,Grain_YPixcels_start: Grain_YPixcels_end,0] = orientationList[i][0]*degree
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,Grain_YPixcels_start: Grain_YPixcels_end,1] = orientationList[i][1]*degree
            OriData[Grain_XPixcels_start: Grain_XPixcels_end,Grain_YPixcels_start: Grain_YPixcels_end,2] = orientationList[i][2]*degree
            IQData[Grain_XPixcels_start: Grain_XPixcels_end,Grain_YPixcels_start: Grain_YPixcels_end] = InputIQData[i]
            CIData[Grain_XPixcels_start: Grain_XPixcels_end,Grain_YPixcels_start: Grain_YPixcels_end] = 1.0
            #IQData [Grain_XPixcels_start: Grain_XPixcels_end][Grain_YPixcels_start: Grain_YPixcels_end] = InputIQData[i]
            #CIData [Grain_XPixcels_start: Grain_XPixcels_end][Grain_YPixcels_start: Grain_YPixcels_end] = 1.0

        x = np.linspace(1,XStep*NoOfXPixcels,NoOfXPixcels)-1
        y = np.linspace(1,YStep*NoOfYPixcels,NoOfYPixcels)-1
        X ,Y = np.meshgrid(x,y)

        OriData = np.reshape(OriData, (NoOfXPixcels * NoOfYPixcels, 3))
        IQData = np.reshape(IQData, (NoOfXPixcels * NoOfYPixcels, 1));
        CIData = np.reshape(CIData, (NoOfXPixcels * NoOfYPixcels, 1));
        FinalData = np.zeros((OriData.shape[0], 10))
        FinalData[:, 6] = 1
        FinalData[:, 3] = X.flatten()
        FinalData[:, 4] = Y.flatten()
        FinalData[:, 0:3] = OriData
        #Ind = np.where(FinalData[:, 1] < 0); ### pixcels  not assigned ori
        FinalData[:, 6] = CIData.flatten() # '##-1; ## Assgning   CI - 1
        FinalData[:, 5] = IQData.flatten() ### note this is made to plot any given property  value to  be  sent  to  IQ data
        FinalData[:, 8] = 50;
        FinalData[:, 9] = 1.5;
        dType = {"phi1": np.float64,"PHI": np.float64, "phi2": np.float64,"x": np.float16, "y": np.float16,
                   "IQ": np.float64, "CI": np.float64,
                  "FIT": np.uint16, "Phase": np.uint8, "SEM": np.float64} ### sem means detecor signal
        columnNames = ["phi1", "PHI", "phi2", "x", "y", "IQ","CI", "FIT", "Phase", "SEM"]

        self._data = pd.DataFrame(FinalData,columns=columnNames)
        self._oriData = OriData
        self.nXPixcels=NoOfXPixcels
        self.nYPixcels=NoOfYPixcels
        self._shape=(NoOfXPixcels,NoOfYPixcels)
        for columnName in self._data:
            self._data[columnName] = self._data[columnName].astype(dType[columnName])

       # headerFileName = os.path.join("../../data/programeData", headerFileName)
        headerFileName = os.path.join(pathlib.Path(__file__).parent.parent.parent,'data',"programeData", headerFileName)
        with open(headerFileName,"rb") as f:
            self._header=[]
            for line in f:
                self._header.append(line)
        self._ebsdFilePath = simulatedEbsdOutPutFilefilePath
        self._ebsdFormat="ang"
        self._isSquareGrid=True
        self._gridType="SqrGrid"
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

    def writeAng(self,pathName=None):
        if pathName is None:
            basename = self.__replace_all_except_last__(self._autoName+"_mod.ang",".ang","_")
            pathName =os.path.join(os.path.dirname(self._ebsdFilePath),basename)

        with open(pathName, "wb") as f:
            for line in self._header[:-1]:
                if isinstance(line,str): ### happens when header is string data
                    f.write(line.encode())
                else:
                    f.write(line) ## when header is read as binary byte data (from standard ang file)

        df = self._data
        with open(pathName, "ab") as f:
            for ind in self._data.index:
                if "ang" in self._ebsdFormat:
                    line = f"{df['phi1'][ind]:.5f} {df['PHI'][ind]:.5f} {df['phi2'][ind]:.5f} {df['x'][ind]:.4f} {df['y'][ind]:.4f} " \
                       f"{np.around(df['IQ'][ind],2):.2f} {df['CI'][ind]:.2f} {df['Phase'][ind]:2d}  {df['SEM'][ind]:8d} {df['FIT'][ind]:.3f}\n"
                elif "ctf" in self._ebsdFormat:
                    line = f"{df['Euler1'][ind]:.5f} {df['Euler2'][ind]:.5f} {df['Euler3'][ind]:.5f} {df['X'][ind]:.2f} {df['Y'][ind]:.2f} " \
                           f"{np.around(df['IQ'][ind], 2):.2f} {df['CI'][ind]:.2f} {df['Phase'][ind]:2d} {df['Fit'][ind]:3d} {df['sem'][ind]:.4f}\n"

                f.write(line.encode())

        logging.info("Wrote the EBSD data file :"+pathName +" Succesfully !!!")

    def reduceEulerAngelsToFundamentalZone(self):
        self.readPhaseFromAng()
        logging.info(f"Done with euler angel reduction to fundamental zone!!")
        eulerData =np.array(self._data.iloc[:,0:3])
        nPoints = self.nXPixcels*self.nYPixcels

        for i in tqdm(np.arange(nPoints)):
            euler = eulerData[i]
            ori = CrystalOrientation(orientation=Orientation(euler=euler.tolist()),
                                     lattice=self._lattice)
            ori, index = ori.projectToFundamentalZone()
            tmp = ori.getEulerAngles(applyModulo=True)
            logging.debug(f"euler angeles before : {euler}--> {tmp}")
            eulerData[i]=tmp

        if "ang" in self._ebsdFormat:
            self._data["phi1"] = eulerData[:,0]
            self._data["PHI"] = eulerData[:,1]
            self._data["phi2"] = eulerData[:,2]

        elif "ctf" in self._ebsdFormat:
            self._data["Euler1"] = eulerData[:, 0]
            self._data["Euler2"] = eulerData[:, 1]
            self._data["Euler2"] = eulerData[:, 2]
        else:
            raise ValueError(f"Unknown ebsd format : {self._ebsdFormat} only ang and ctf are supported")

        self.__makeEulerData()
        logging.info(f"Converted the euler angeles into fundamental zone")

    @staticmethod
    def __replace_numberFromHeader(text, new_integer):
        """
        used for replacing the integer at the end of the header line.
        Also puts line ending character at the end
        """
        pattern = r"(\D*)(\d+)$"
        match = re.match(pattern, text)

        if match:
            prefix = match.group(1)
            updated_text = prefix + str(new_integer)+"\n"
            return updated_text
        else:
            return text

    def crop(self, start=(0,0),dimensions=(10,10)):
        """
        crop the ebsd data to have dimensions=(l,w) from start point (x,y)
        l = length of crop (in pixcels)
        w = width of crop in pixcels
        (x,y) = start point of crop.
        all dimensions are to be specified in the form of pixcels (integers)
        """
        shape = self._shape
        cropParameters = start[0],start[0]+dimensions[0],start[1],start[1]+dimensions[1],

        if start[0]+dimensions[0]>shape[0] or start[1]+dimensions[1]>shape[1]:
            raise ValueError(f"the ebsd data has shape {shape} but cropping parameters are :{cropParameters}")

        mask = np.zeros(shape=shape, dtype=bool)
        mask[cropParameters[0]:cropParameters[1], cropParameters[2]:cropParameters[3]] = True
        indx = np.where(mask.reshape(-1)==False)
        indx = indx[0].tolist()

        logging.info(f"removing {len(indx)} number of data points for croppiing the ebsd data")
        if "ang" in self._ebsdFormat:
            x, y = "x", "y"
        elif "ctf" in self._ebsdFormat:
            x, y = "X", "Y"
        else:
            raise ValueError(f"Uknown ebsd format {self._ebsdFormat} only '.ang' and '.ctf' are supported as of now")

        ind = mtu.sub2ind(self._shape,start[0],start[1])
        newOriginInXYsystem = (self._data.iloc[ind, self._data.columns.get_loc(x)],
                               self._data.iloc[ind, self._data.columns.get_loc(y)])

        self._data[x] = self._data[x] - self._data.iloc[ind, self._data.columns.get_loc(x)]
        self._data[y] = self._data[y] - self._data.iloc[ind, self._data.columns.get_loc(y)]

        self._data = self._data.drop(index = indx)
        ### now adjusting the first point X Y to become 0,0
        self._shape = (dimensions[0],dimensions[1])
        numberOfchangedValues=0
        for i,line in enumerate(self._header):
            if "# NCOLS_ODD:" in line:
                self._header[i] = self.__replace_numberFromHeader(line, dimensions[1])
                numberOfchangedValues+=1
            elif "# NROWS:" in line:
                self._header[i] = self.__replace_numberFromHeader(line, dimensions[0])
                numberOfchangedValues += 1
            elif "# NCOLS_EVEN:" in line:
                self._header[i] = self.__replace_numberFromHeader(line, dimensions[1])
                numberOfchangedValues += 1
            else:
                continue
        assert numberOfchangedValues ==3 , f"Suposed to have changed 3 values but could change only {numberOfchangedValues}. " \
                                           f"The modified header is {self._header}"
        logging.debug(f"modifd headeris : {self._header}")
        self._isCropped=True
        self.nXPixcels, self.nYPixcels = self._shape[1],self._shape[0]
        self.__makeEulerData()
        self._autoName = self._autoName + self.__replace_characters__(f"cropped_{dimensions}")
        logging.info(f"cropped the ebsd data points now the cropped size of the data is : {self._shape}")

    def rotateAndFlipData(self, flipMode=None, rotate=0):
        """
        flip the ebsd data, typically useful for augmentation in machine learning
        flipMode: one of None, vertical, horizontal
        rotate one of 90,180,270 : Note that it is physcial rotation not orienation rotation
        """
        operationString=""
        shape = self._shape
        data = self._data
        if flipMode is not None and "h" in flipMode: ##horizontal flip
            axis = 0
            operationString += f'flipped_hor'
        elif flipMode is not None:
            axis = 1 ##vertical flip
            operationString+=f'flipped_vert'

        if "ang" in self._ebsdFormat:
            eulerData = np.array([data["phi1"], data["PHI"], data["phi2"]]).T
        elif "ctf" in self._ebsdFormat:
            eulerData = np.array([data["Euler1"],data["Euler2"],data["Euler2"]]).T
        else:
            raise ValueError(f"Unknown format {self._ebsdFormat} only ctf and ang are supported as of now")

        eulerData = np.stack(
            [eulerData[:, 0].reshape(shape), eulerData[:, 1].reshape(shape), eulerData[:, 2].reshape(shape)], axis=2)


        rotatedData = eulerData
        if rotate > 0:
            if shape[0] == shape[1]:
                logging.debug(f"attempting the ebsd data physcial rotation")
                if rotate ==90:
                    rotatedData = np.rot90(rotatedData, k=1,axes = (0,1))
                elif rotate ==180:
                    rotatedData = np.rot90(rotatedData, k=2, axes=(0, 1))
                elif rotate ==270:
                    rotatedData = np.rot90(rotatedData, k=2, axes=(0, 1))
                else:
                    raise ValueError (f"rotation provided is {rotate} but only one of [90,180,270] degree are supported")
                operationString+=f'_rotation_{rotate}'

            else:
                logging.warning(f"skipping the rotation as the ebsd data is not of square shape. "
                                f"ebsd shape :{shape}")

        if flipMode is not None:
            flippedData = np.flip(rotatedData, axis=axis)
        if "ang" in self._ebsdFormat:
            self._data["phi1"]=flippedData[:,:,0].reshape(-1)
            self._data["PHI"]=flippedData[:,:,0].reshape(-1)
            self._data["phi2"]=flippedData[:,:,0].reshape(-1)
        elif "ctf" in self._ebsdFormat:
            self._data["Euler1"] = flippedData[:, :, 0].reshape(-1)
            self._data["Euler2"] = flippedData[:, :, 0].reshape(-1)
            self._data["Euler3"] = flippedData[:, :, 0].reshape(-1)
        else:
            raise ValueError(f"Unknown format {self._ebsdFormat} only ctf and ang are supported as of now")

        self.__makeEulerData()
        self._autoName = self._autoName+operationString
        logging.info(f"completd the flipping of orienation data.")
        logging.warning(f"Only the orientation data is flipped rest of data such as IQ, Fit etc are not as of now!!!")

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
                    tmp = np.fromstring(tmp, dtype=np.double, sep = " ")
                    a,b,c , alpha,beta,gamma= tmp ### in angstroms and degrees
                    latticeConstants = tmp[0:3]
                    latticeAngles = tmp[3:6]
                    latticeParameterLine = line
                if "# MaterialName" in line:
                    self._phaseName = line.split("\t")[1].split(" ")[0]

            logging.warning(f"Currently checking the lattice parameters only for determing the crystal sytems assuming only cubic and hexagonal are proceessed. Modify this part for "
                            f"other crystal systems !!!")



            if np.allclose(latticeConstants - latticeConstants[0], [0., 0., 0.]) and np.allclose(latticeAngles,
                                                                                                 [90., 90.,
                                                                                                  90.]) :
                self.crystalSystem = 'cubic'
                self._lattice = OrientedLattice.cubic(a=latticeConstants[0], pointgroup='m-3m',
                                                      name=self._phaseName)
                logging.debug(f"Found the {self._phaseName} to be cubic!!!")
            elif np.allclose(latticeConstants[:2], -latticeConstants[0], [0., 0., ]) and np.allclose(latticeAngles,
                                                                                                     [90., 90.,
                                                                                                      120.]) :
                self.crystalSystem = 'hexagonal'
                self._lattice = OrientedLattice.hexagonal(a=latticeConstants[0], c=latticeConstants[2],
                                                          pointgroup='6/mmm', name=self._phaseName)
                logging.debug(f"Found the {self._phaseName} to be hexagonal!!!")

            else:
                logging.critical("Unknown crystal system : the line being parsed is : " + latticeParameterLine)
                raise ValueError("Unknown system !!!!" + latticeParameterLine)

            logging.info("Succesfully parse ang header and populated the crystal structure info!!!")
            return(0)


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    fileName=r'..\data\ebsdData\SuperNi-Ni-Fcc.ctf'

    testFromAng=True
    if testFromAng:
        fileName = r'../../tmp/Al-B4CModelScan.ang'
        ebsd = Ebsd(logger=logger)
        ebsd.fromAng(fileName=fileName)
        # maskImg = np.full((80, 50), True, dtype=bool)
        #maskImg = r"../../data/programeData/ebsdMaskFolder/3.png"
        #ebsd.applyMask(maskImg,displayImage=True)
        ebsd.crop(start = (10,10), dimensions=(150,150))
        ebsd.rotateAndFlipData(flipMode='vertical', rotate=180)
        #ebsd.reduceEulerAngelsToFundamentalZone()
        ebsd.writeAng()
        exit(-100)

    fileName =r'D:\CurrentProjects\python_trials\machineLearning\EnhanceMicroStructure\kikuchiProject\ebsdCleanUp\EBSD maps\def_map_1_sw_cleaned.ctf'
    line='3.570;3.570;3.570    90.000;90.000;90.000    Ni-superalloy    11    225            Generic superalloy' 
    ebsd = Ebsd(logger=logger)
    loadFromFile=False
    if loadFromFile:
        inputFileName = r'C:\Users\Admin\PycharmProjects\pycrystallography\tmp\uniformGridOut.txt'
        oriData = np.genfromtxt(inputFileName, delimiter=",")
        orientationList = oriData[:,0:3]
        iQData = oriData[:,3]
        ebsd.generateSimulatedEbsdMap(orientationList=orientationList,
                                      simulatedEbsdOutPutFilefilePath=inputFileName+'.ang', sizeOfGrain=5, IqData=iQData.tolist())
        print(oriData)
        exit(-1)

    testGenerateSimulatedEbsdMap=False
    if testGenerateSimulatedEbsdMap:
        deg=np.pi/180.0
        cubicOri1 = CrysOri(orientation=Orientation(euler=[0.*deg, 0.*deg, 0.*deg]), lattice=olt.cubic(1))
        cubicOri2 = CrysOri(orientation=Orientation(euler=[45.*deg, 30.*deg, 0.*deg]), lattice=olt.cubic(1))
        oriList = cubicOri1.symmetricSet()
        oriList = [i.projectToFundamentalZone()[0].getEulerAngles(units="degree", applyModulo=True).tolist() for i in oriList]
        oriList2 = cubicOri2.symmetricSet()
        projectToFundamentalZOne=False
        if projectToFundamentalZOne:
            oriList2 = [i.projectToFundamentalZone()[0].getEulerAngles(units="degree", applyModulo=True).tolist() for i in
                        oriList2]
        else:
            oriList2 = [i.getEulerAngles(units="degree").tolist() for i in oriList2]

        #oriList2= [i.getEulerAngles(units="degree").tolist() for i in oriList2]
        oriList.extend(oriList2)
        shuffle(oriList)

        ebsd.generateSimulatedEbsdMap(orientationList=oriList,
                                      headerFileName="bcc_header.txt",
                                      simulatedEbsdOutPutFilefilePath=r"../../tmp/simulatedEbsd.ang",sizeOfGrain=5,)
        ebsd.makeEulerDatainImFormat(saveAsNpy=True)
        # ebsd.writeNpyFile(pathName=None)
        # ebsd.readPhaseFromCtfString(ebsd._header[13])
        print("Done")
        exit(-300)

        # ebsd.fromCtf(fileName)
    # ebsd.applyMask(maskImge=r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\ebsdData\maskFolder\mask (1).png',maskSize=[50,50])
    # ebsd.applyMask(maskImge=r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\ebsdData\maskFolder\mask (4).png',maskSize=[10,10])
    #
    #ebsd.writeEulerAsPng(showMap=True)
    #ebsd.makePropertyMap(property="MAD")
    #ebsd.addNewDataColumn(dataColumnName="isMod", )
    #ebsd.writeCtf(pathName=None)

        
        
        
        