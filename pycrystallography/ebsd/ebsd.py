'''
Created on 27-Jun-2019

@author: K V Mani Krishna
'''

import os
import logging
import string
import copy
from random import shuffle

from pycrystallography.core.orientedLattice  import OrientedLattice
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imsave
#from absl.logging import ABSLLogger
from pycrystallography.utilities import pymathutilityfunctions as mtu
import pathlib



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
        
    def fromAng(self, fileName):
        """
        method for reading ang Files
        """

        count=0
        header=[]
        logging.debug("Attempting to read the file :"+fileName)
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
            raise ValueError("Unknown format only ctf and ang are supported as of now")

        shape=yPixcels,xPixcels,
        eulerData = np.stack([eulerData[:,0].reshape(shape), eulerData[:,1].reshape(shape),eulerData[:,2].reshape(shape)],axis=2)
        oriData = np.zeros_like(eulerData, dtype=np.uint8,)
        oriData[:,:,0]= np.interp(eulerData[:,:,0], (eulerData[:,:,0].min(), eulerData[:,:,0].max()), eulerLimits).astype(np.uint8)
        oriData[:,:,1]= np.interp(eulerData[:,:,1], (eulerData[:,:,1].min(), eulerData[:,:,1].max()), eulerLimits).astype(np.uint8)
        oriData[:,:,2]= np.interp(eulerData[:,:,2], (eulerData[:,:,2].min(), eulerData[:,:,2].max()), eulerLimits).astype(np.uint8)
        
        self._oriData=oriData
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
        
        
        
    
    def applyMask(self,maskImge, maskSize=[30,30],maskLocation=None):
        """
        maskImage : is a path to binary image or numpy boolean array.
        maskSize mXn of mask to which the given input mask is scaled. Ignored if the input mask is np.ndarray of bool type.
        maskLocation = i,j (int) of the picels about which the mask must be placed in the EBSD data.
        """ 
        
        self.addNewDataColumn("isModified", fillValues=False)
        if type(maskImge)==str: 
            try:
                mask = Image.open(maskImge)
                
                mask=mask.resize(maskSize, Image.ANTIALIAS)
                #mask.show()
                mask = np.array(mask)
                mask = mask==0
            except Exception as e:
                logging.fatal(e)

        elif type(maskImge)==np.ndarray:
            mask = maskImge
            maskSize=mask.shape
        else:
            raise ValueError("The supplied mask is neither a file path nor a valid boolen numpy array.")
        
        mainMask = np.full((self._shape),False, )
        
        
        if maskLocation is None:
            maskMargin = max(maskSize)
            ebsdDataImageDimSmall = min(self._shape)
            maskLocation = np.random.randint(low = maskMargin+5, high = ebsdDataImageDimSmall-maskMargin-5, size=(2,) )
            
        assert mask.dtype==bool, f"The supplied mask is not of boolen type !!!! as the type is : {type()}"
        
        startInd = maskLocation[0]-int(maskSize[0]/2), maskLocation[1]-int(maskSize[1]/2)
        endInd = maskLocation[0]+int(maskSize[0]/2), maskLocation[1]+int(maskSize[1]/2)
#         tmp1 = np.arange(startInd[0],endInd[0])
#         tmp2= np.arange(startInd[1],endInd[1])
        mainMask[startInd[0]:endInd[0], startInd[1]:endInd[1]]= mask
        indx = np.where(mainMask.reshape(-1))
        indx = indx[0].tolist()
        if "ctf" in self._ebsdFormat:
            self._data["MAD"][indx] = 0.0
            self._data["Euler1"][indx] = 0.0
            self._data["Euler2"][indx] = 0.0
            self._data["Euler3"][indx] = 0.0
            self._data["isModified"][indx]=True
        elif "ang" in self._ebsdFormat:
            self._data["CI"][indx] = -1.0
            self._data["phi1"][indx] = 0.0
            self._data["PHI"][indx] = 0.0
            self._data["phi2"][indx] = 0.0
            self._data["isModified"][indx] = True

        
        self.__makeEulerData()       
        self.writeEulerAsPng()
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

    def writeAng(self,pathName=None):
        if pathName is None:
            pathName =self._ebsdFilePath[:-4]+"_mod.ang"

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


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    fileName=r'..\data\ebsdData\SuperNi-Ni-Fcc.ctf'

    testFromAng=True
    if testFromAng:
        fileName = r'D:\mani\Al-B4C-Composites\Al-B4CModelScan.ang'
        ebsd = Ebsd(logger=logger)
        ebsd.fromAng(fileName=fileName)
        maskImg = np.full((80, 50), True, dtype=bool)
        ebsd.applyMask(maskImg)
        ebsd.writeAng(pathName=r"..\..\tmp\simulatedEbsd_OIM_recreated.ang")
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
        oriList = [i.projectTofundamentalZone()[0].getEulerAngles(units="degree",applyModulo=True).tolist() for i in oriList]
        oriList2 = cubicOri2.symmetricSet()
        projectToFundamentalZOne=False
        if projectToFundamentalZOne:
            oriList2 = [i.projectTofundamentalZone()[0].getEulerAngles(units="degree", applyModulo=True).tolist() for i in
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

        
        
        
        