import sys
from PyQt4 import QtCore, QtGui, uic 
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import random
import pycrystallography.utilities.graphicUtilities as gu
from pycrystallography.core.orientedLattice import OrientedLattice as olt
import json
import os
from pycrystallography.core.saedAnalyzer import SaedAnalyzer
import logging
import tifffile as tif
from PyQt4.QtCore import QThread
from matplotlib.widgets import Slider, Button, RadioButtons

qtCreatorFile = r"D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\examples\saedSolverGUI.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class loadCrystalThread(QThread):
    def __init__(self, cifFileList,structureNameList,maxHkl=3):
        QThread.__init__(self)
        self.cifFileList = cifFileList
        self.structureNameList=structureNameList
        self.maxHkl=maxHkl

    def __del__(self):
        self.wait()

    def _loadCrystal(self, cifFile,structureName):
        #structureFiles=cifFile
        #crystals=[]
        lattice = olt.fromCif(cifFile)
        saedAnalyzer = SaedAnalyzer(lattice=lattice,hklMax=self.maxHkl)
        saedAnalyzer.loadStructureFromCif(cifFile)
        logging.info("Just created the SAED analyzer object for crystal  :")
        return {"structureName":structureName,"lattice":lattice, "saedAnalyzer":saedAnalyzer}
          

    def run(self):
        for i,cifFile in enumerate(self.cifFileList):
            result = self._loadCrystal(cifFile,self.structureNameList[i])
            print("loaded lattice: ", result["lattice"])
            self.emit(SIGNAL('addCrystalFromThread(PyQt_PyObject)'), result)



class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=10, height=4.8, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
#         pos1 = self.axes.get_position()        
#         self.spotsScatterPlot = fig.add_axes(pos1) ### this is where scatter plot showing the diffracted spots from 2d lattice functionis plotted
#         self.spotsScatterPlot.set_alpha(0.2)
        
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtGui.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)    

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)    



class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        #self.navi_toolbar = NavigationToolbar(self.canvas, self)
 
        self.compute_initial_figure()        

    def compute_initial_figure(self,data=None):

            self.axes.text(0.2,0.5, "Diffraction Pattern not yet loaded")
            
    def update_figure(self,data,imageName='',latticePoints=None,title='',spcialPointsForMarking=None):        
        self.axes.cla()
        self.axes.imshow(data, cmap='gray')
        axcolor = 'lightgoldenrodyellow'

        if latticePoints is not None:
            latticePoints=np.array(latticePoints)
            self.axes.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')     
        
        fontSpecialPoints = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 12,
        }
        
        if spcialPointsForMarking is not None:
            for i,item in enumerate(spcialPointsForMarking):
                self.axes.text(item[0],item[1], str(i),fontdict=fontSpecialPoints)
        
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        
        self.axes.text(0,-10, imageName,fontdict=font )
        self.axes.set_title(title)
        #self.axes.autoscale()
        self.axes.set_aspect('equal', 'datalim')
        self.axes.axis('equal')
        self.draw()
    def plotSolution(self,data,imageName,latticePoints,saedSolution,saedAnalyzer,title="",markSpots=True,showAbsentSpots=False): 
        self.axes.cla()
        self.axes.imshow(data)
        bounds=data.shape
        print("pattern bounds are ", bounds)
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
 
        if latticePoints is not None:
            latticePoints=np.array(latticePoints)
            self.axes.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')     

        saedData = saedSolution["pattern"]
        angleError = str(saedSolution["angleError"])
        dError = str(saedSolution["dError"])
        saedAnalyzer.plotSAED(saedData,plotShow=True,figHandle=None,axisHandle=self.axes,makeTransperent=False,markSpots=markSpots,showAbsentSpots=showAbsentSpots)
        #self.axes.set_title(title+" AngleError= "+angleError+r" %dError= "+dError)
        self.axes.text(0,-20,imageName,fontdict=font)
        #self.axes.autoscale()
        
        #self.axes.axis('equal')
        self.axes.set_xlim(right=bounds[0],left=-50) ## put in the 
        self.axes.set_ylim(top=-50,bottom=bounds[1]) ## put in the 
        #self.axes.set_aspect('equal',)
        self.draw()

        print("Plotted the solution")
           
        
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)     
        self.setupUi(self)
        self.main_widget = QtGui.QWidget(self.plotArea)
        l = QtGui.QVBoxLayout(self.main_widget)         
        dc = MyStaticMplCanvas(self.main_widget, width=8, height=8, dpi=100)  
        dc.compute_initial_figure()      
        l.addWidget(dc)
        logTextBox = QPlainTextEditLogger(self)
        # You can format what is printed to text box
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.INFO)        
        ll= QtGui.QVBoxLayout(self.textOutPutWindow)
        ll.addWidget(logTextBox.widget) 
        
               
        self.addToolBar(NavigationToolbar(dc, self))
        #l.addWidget()
        self.AppData=  {}
        self.AppData["patternName"]=''
        self.AppData["jsonFileName"]=''
        self.AppData["SAEDSolutuions"]=None
        self.AppData["solverOptions"]={"maxHkl":3, "dMin":0, "dMax":10, "allowedAngleError":5., "allowedDError":10.,"cameraConstant":1079}
        self.AppData['solutionPlotOptions']={'markSpots':False, 'hideSystamticAbsences':True}
        self.AppData["cifFiles"]=[]
        self.AppData["filePathData"]={'lastUsedDirForPattern':{'path':'','default':''},
                                      'lastUsedDirForConfigFile':{'path':'','default':''},
                                      'dirForProgramTmpData':{'path':r'../../data/programeData/'},
                                      'pathStorageFile':r'../data/programeData/pathStorageFile.jason'}
                                      
        jsonName=self.AppData["filePathData"]['pathStorageFile']
        with open(jsonName, 'r') as f:
            tmpData = json.load(f)
        for key in tmpData:
            self.AppData['filePathData'][key]=tmpData[key]
            logging.debug("Loaded default paths successfully")
        
        #AppData["solverOptions"]["cameraConstant"]
        #self.setLayout(l)
        
        self.tableWidgetSolutions.setRowCount(1)
        self.tableWidgetSolutions.setColumnCount(8)
        self.tableWidgetSolutions.setHorizontalHeaderLabels(["S.No", "CrystalName" , "   ZoneAxis  ", "  spot1 ","  spot2 " , "AngleError", r"dError(%)", "Correlation"])
        self.tableWidgetSolutions.resizeColumnsToContents()
        self.tableWidgetSolutions.resizeRowsToContents()
        self.pushButtonSolve.setEnabled(False)
        
        self.tableWidgetDSpacing.setRowCount(1)
        self.tableWidgetDSpacing.setColumnCount(4)
        self.tableWidgetDSpacing.setHorizontalHeaderLabels(["  Plane  ", "dSpacing" , r"  1/d   ", "Int(I)"])
        self.plainTextEditDmin.setPlainText ("1.0")
        self.plainTextEditDmax.setPlainText ("5.0")
        self.plainTextEditDeltaAngle.setPlainText("5.0")
        self.plainTextEditDeltaDSpacing.setPlainText("10.0")
        
        
        #self.loadDefautls()
        
        self.labelProgressIndicator.hide()
        self.AppData["Origin"] = [self.OriginX.value(),self.OriginY.value()]
        self.AppData["Spot1"] = [self.Spot1X.value(),self.Spot1Y.value()]
        self.AppData["Spot2"] = [self.Spot2X.value(),self.Spot2Y.value()]
        #self.AppData["markSpots"] = self.radioButtonMarkSpots.isChecked()        
        self.pushButtonLoadPattern.clicked.connect(lambda : self.selectFile())        
        self.OriginX.valueChanged.connect(self.updateOriginX)
        self.OriginY.valueChanged.connect(self.updateOriginY)
        self.Spot1X.valueChanged.connect(self.updateSpot1X)
        self.Spot1Y.valueChanged.connect(self.updateSpot1Y)
        self.Spot2X.valueChanged.connect(self.updateSpot2X)
        self.Spot2Y.valueChanged.connect(self.updateSpot2Y)
        self.pushButtonLoadConfiguration.clicked.connect(lambda :self.loadConfiguration())
        
        self.pushButtonLoadCrystals.clicked.connect(lambda :self.loadCrystals())
        self.pushButtonSaveConfiguration.clicked.connect(lambda :self.saveConfiguration())
        self.comboBoxCrystalNames.currentIndexChanged.connect(self.choseDifferentCrystal)
        self.comboBoxChoseMachine.currentIndexChanged.connect(self.choseDifferentMachine)
        self.comboBoxChoseCameraLength.currentIndexChanged.connect(self.choseDifferentCameraLength)
        
        self.pushButtonSolve.clicked.connect(lambda : self.startSolver())
        self.pushButtonClearSolutions.clicked.connect(lambda : self.clearSolutions())
        self.pushButtonGenerateDTable.clicked.connect(lambda : self.generateDTable()) 
        self.tableWidgetSolutions.cellClicked.connect(self.cell_was_clicked)
        self.spinBoxMaxHkl.valueChanged.connect(self.updateMaxHkl)
        self.plainTextEditDmin.textChanged.connect(self.DminValuechanged)
        self.plainTextEditDmax.textChanged.connect(self.DmaxValuechanged)
        self.plainTextEditDeltaAngle.textChanged.connect(self.angleErrorValuechanged)
        self.plainTextEditDeltaDSpacing.textChanged.connect(self.dSpacingErrorValuechanged)  
        self.plainTextEditCameraConst.textChanged.connect(self.cameraConstChanged) 
        self.markSpotsButton.toggled.connect(lambda:self.markSpotsOptionChanged(self.markSpotsButton))
        self.showSystematicAbsncesButton.toggled.connect(lambda:self.showSystematicAbsncesOptionChanged(self.showSystematicAbsncesButton))
                    
        self.AppData["plotHandle"]=dc 
        defaultConfig=r"..\data\imageData\pushpa_beta_omeag_SAED_60cm.json"
        #self.loadConfiguration(fileName=defaultConfig,updatePlot=False)
        widgetStates= [{'handle':self.pushButtonSolve, 'setState':'hide'},
                        {'handle':self.pushButtonGenerateDTable, 'setState':'hide'},
                        {'handle':self.markSpotsButton, 'setState':'hide'},
                        {'handle':self.pushButtonGenerateDTable, 'setState':'hide'},
                        {'handle':self.comboBoxCrystalNames, 'setState':'hide'},
                        {'handle':self.pushButtonClearSolutions, 'setState':'hide'},]
        self.setWidgetStates(widgetStates)
        self.pushButtonSaveAll.clicked.connect(lambda : self.saveAll())
            
    
    def saveAll(self):
        self.savePathInfo()
        self.saveConfiguration()
        logging.info("saved the important stuff !!!!")
        
    
    
    def savePathInfo(self):
        fileName = self.AppData['filePathData']['pathStorageFile']
        dumpData=self.AppData['filePathData']
        with open(fileName, 'w') as fp:
            json.dump(dumpData, fp, sort_keys=True, indent=4)
        logging.info("Saved the path storage file:")
    
    def clearSolutions(self):
        widgetStates= [{'handle':self.pushButtonSolve, 'setState':'show'},
                        {'handle':self.markSpotsButton, 'setState':'hide'},
                        {'handle':self.pushButtonClearSolutions, 'setState':'hide'},
                        {'handle':self.pushButtonSolve, 'setState':'show'},
                        ]
        self.tableWidgetSolutions.setRowCount(0)
        self.setWidgetStates(widgetStates)
        
    
    def showSystematicAbsncesOptionChanged(self,b):
        logging.info("Now the option of showing the absent spots is being changed!!!")
        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['hideSystamticAbsences']=True
        else:
            self.AppData['solutionPlotOptions']['hideSystamticAbsences']=False
     
    
    def markSpotsOptionChanged(self,b):
        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['markSpots']=True
        else:
            self.AppData['solutionPlotOptions']['markSpots']=False
        
    
    def generateDTable(self):
        
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]        
        planeTable = sa.generateDspacingTable()
        if len(planeTable)>1:
            self.tableWidgetDSpacing.setRowCount(len(planeTable))
            for i,item in enumerate(planeTable,start=0):
                if i==0:
                    continue                
                self.tableWidgetDSpacing.setItem(i-1,0, QTableWidgetItem(str(item["plane"])))
                self.tableWidgetDSpacing.setItem(i-1,1, QTableWidgetItem(str(item["dSpacing"])))
                self.tableWidgetDSpacing.setItem(i-1,2, QTableWidgetItem(str(np.around((1./item["dSpacing"]),2))))                
                self.tableWidgetDSpacing.setItem(i-1,3, QTableWidgetItem(str(item["intensity"])))
                #twoTheta = 
                #print(item["plane"],item["dSpacing"],item["intensity"], )
                
    
    def cameraConstChanged(self):
        txt =self.plainTextEditCameraConst.toPlainText() 
        if len(txt)>0:
            cameraConstant=float(txt)
            self.AppData["solverOptions"]["cameraConstant"]=cameraConstant
        self.updateMeasuredValues()
        self.pushButtonSolve.setEnabled(True)    
    
    def angleErrorValuechanged(self):
        txt =self.plainTextEditDeltaAngle.toPlainText() 
        if len(txt)>0:
            allowedAngleError=float(txt)
            self.AppData["solverOptions"]["allowedAngleError"]=allowedAngleError
        self.pushButtonSolve.setEnabled(True)
    
    def dSpacingErrorValuechanged(self):
        txt =self.plainTextEditDeltaDSpacing.toPlainText() 
        if len(txt)>0:
            allowedDError=float(txt)
            self.AppData["solverOptions"]["allowedDError"]=allowedDError
        self.pushButtonSolve.setEnabled(True)
    
    
    def DminValuechanged(self):
        txt =self.plainTextEditDmin.toPlainText() 
        if len(txt)>0:
            dmin=float(txt)
            print("Yes",dmin)
            self.AppData["solverOptions"]["dMin"]=dmin
            logging.info("D min of Solver changed to "+str(dmin))
        
        self.pushButtonSolve.setEnabled(True)
    
    def DmaxValuechanged(self):
        txt =self.plainTextEditDmax.toPlainText() 
        if len(txt)>0:
            dmax=float(txt)
            self.AppData["solverOptions"]["dMax"]=dmax
            logging.info("D max of Solver changed to "+str(dmax))
        self.pushButtonSolve.setEnabled(True)  
    
    def updateMaxHkl(self):
        maxHkl=self.spinBoxMaxHkl.value()
        if maxHkl>4:
            #QtGui.QMessageBox.information(self, "Done!", "Done loadig the crystals Now Enjoy!")
            QtGui.QMessageBox.information(self,"Warning!!!", "maxHkl chosen is {:2d} It can take considerable time to load. Please wait.".format(maxHkl))
            
        self.AppData["solverOptions"]["maxHkl"]=maxHkl
        self.loadCifs()
        logging.info("Max Hkl of sover changed to "+str(maxHkl)+'and the crystals are relaoded with changed settings')
        self.pushButtonSolve.setEnabled(True)
    
    def cell_was_clicked(self, row, column):
        print("Row %d and Column %d was clicked" % (row, column))
        solution=self.AppData["SAEDSolutuions"][row]
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]
        imageName=self.AppData["patternName"]
        
        self.AppData["plotHandle"].plotSolution(data=self.AppData["patternImageData"],imageName=imageName,latticePoints=self.AppData["latticePoints"],saedSolution=solution,
                                                    saedAnalyzer=sa,title=None,markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'])
    
    
    def startSolver(self):
        self.labelProgressIndicator.show()
        self.labelProgressIndicator.setText("Solver is working ! Wait few seconds !!!")
        self.solvePattern()
        
    
    def solvePattern(self):
        
      
        expSpotData={"spotXyData":[self.AppData["Origin"],self.AppData["Spot1"], self.AppData["Spot2"]]} 
        hklMax=self.AppData["solverOptions"]["maxHkl"]
        D_TOLERANCE=self.AppData["solverOptions"]["allowedDError"]
        allowedAngleDeviation=self.AppData["solverOptions"]["allowedAngleError"]
        spot1dRange = [self.AppData["solverOptions"]["dMin"], self.AppData["solverOptions"]["dMax"]]
        spot2dRange = [self.AppData["solverOptions"]["dMin"], self.AppData["solverOptions"]["dMax"]]        
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]
        lattice=crystal["lattice"]
        logging.info("Started the solver !!!!")
        self.generateDTable()
        calibration= {"cameraConstant":self.AppData["solverOptions"]["cameraConstant"]}
        logging.info("The current D tolerance is : {:.1f}".format(D_TOLERANCE))
        result=sa.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=6,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
                                                      calibration=calibration,imageData=self.AppData["patternImageData"])
        
        
#         result = sa.solvePatternFrom3Points(expSpotData, hklMax=hklMax,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
#                                             spot1dRange=spot1dRange,
#                                            spot2dRange=spot2dRange,imageData=self.AppData["patternImageData"])
#          
        self.labelProgressIndicator.setText("Solver is Done !!!")
        
        if result is not None:
            print("Congratulations Some solutions were found!!!!")
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
            self.AppData["SAEDSolutuions"]=result
            self.tableWidgetSolutions.setRowCount(len(result))
            self.pushButtonClearSolutions.setEnabled(True)
            for i,item in enumerate(result):
                self.tableWidgetSolutions.setItem(i,0, QTableWidgetItem(str(i+1)))
                self.tableWidgetSolutions.setItem(i,1, QTableWidgetItem(self.AppData["currentCrystalName"]))
                zoneAxisName = "{:int}".format(item["pattern"]["zoneAxis"])
                self.tableWidgetSolutions.setItem(i,2, QTableWidgetItem(zoneAxisName[5:]))
                self.tableWidgetSolutions.setItem(i,3, QTableWidgetItem(str(item["spot1"])))
                self.tableWidgetSolutions.setItem(i,4, QTableWidgetItem(str(item["spot2"])))                
                self.tableWidgetSolutions.setItem(i,5, QTableWidgetItem(str(item["angleError"])))
                self.tableWidgetSolutions.setItem(i,6, QTableWidgetItem(str(item["dError"])))
                self.tableWidgetSolutions.setItem(i,7, QTableWidgetItem(str(np.around(item["Correlation"],3))))
                
            
            self.tableWidgetSolutions.show()
            widgetStates= [{'handle':self.pushButtonSolve, 'setState':'hide'},
                        {'handle':self.markSpotsButton, 'setState':'show'},
                        {'handle':self.pushButtonGenerateDTable, 'setState':'hide'},
                        {'handle':self.pushButtonClearSolutions, 'setState':'show'},
                        ]
        
        
            self.setWidgetStates(widgetStates)  
            
            self.AppData["plotHandle"].plotSolution(data=self.AppData["patternImageData"],imageName=self.AppData["patternName"],latticePoints=self.AppData["latticePoints"],saedSolution=result[0],
                                                    saedAnalyzer=sa,title=None,markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'])
            logging.info("A total of {:2d} solutions were found. Hope one of them works for you".format(len(result)))
        else:
            print("No solutions found !! try changing the solutiuon parameters!!!")
            
            logging.info("No solutions were found !! Please try changing the paramters of the solver!!!")
            QtGui.QMessageBox.information(self,"Warning!!!", "No solutions have been found!! Consider Changing the solver parameters and Try Again ")
            
    
    
    def setWidgetStates(self,widgetList):
        """
        helper function to set the state to enable, disable, visible or invisible.
        must be a list of dicts of the form:
        [{'handle':self.tableWidgetSolutions, 'setState':'show'},
         {'handle':self.pushButtonClearSolutions, 'setState':'hide'},
        ]
        """
        
        for item in widgetList:
            if item['setState']=='enable':
                item['handle'].enabled=True
            if item['setState']=='disable':    
                item['handle'].enabled=False
            if item['setState']=='hide':
                item['handle'].hide()
            if item['setState']=='show':
                item['handle'].show()
        
    
    def loadCrystals(self):
        cifFiles = QtGui.QFileDialog.getOpenFileNames(self, 'Chose the cif File', 
         r'../data/structureData',"cif files (*.cif )")
        if len(cifFiles)>0:
            self.AppData["structureFiles"]=cifFiles
            self.AppData["structureNames"]=[]
            for item in cifFiles:
                self.AppData["structureNames"].append(os.path.basename(item)[:-4]) 
            
            self.loadCifs()
            logging.info("Succesfully loaded the crystal")
        else:
            logging.error("No cif files were choosen !! please chose properly !!!")
        

    def choseDifferentCameraLength(self,i):
        
        machine=self.AppData["machine"]
        self.AppData["machine"]["cameraLength"]=machine[i]["cameraLength"]
        logging.info("Updated the Camera Const ", machine["machineName"])
        logging.info("Default camera Length of {:s} is applied ".format(self.AppData["machine"]["cameraLength"]))
 
    
    
    def choseDifferentMachine(self,i):
            
        machines=self.AppData["microscopes"]
        machine=machines[i]
        self.AppData["machine"]=machine
        self.AppData["machine"]["cameraLength"]=machine[0]["cameraLength"]
        for item in machine:            
            self.comboBoxChoseCameraLength.addItem(item["cameraLength"])
        #self.AppData[]
        logging.info("Updated the new machine succesfully !!! and the new machine is " +str(machine["machineName"]))
        logging.info("Default camera Length of {:s} is applied ".format(self.AppData["machine"]["cameraLength"]))
    
    def choseDifferentCrystal(self,i):
        
        print("Just entered the before change : ", self.AppData["crystals"][0]["structureName"])
        for k in range(self.comboBoxCrystalNames.count()):
            print ("The index is ", k , "the item is : ",self.comboBoxCrystalNames.itemData(k))
      
        crystals=self.AppData["crystals"][i]
        self.AppData["currentCrystalName"]=crystals["structureName"]
        self.AppData["currentCrystal"]=crystals
        
        print("Updated the new crystal succesfully !!! and the new crystal is ",crystals["structureName"])
        
    def addCrystalFromThread(self,result):
        
        self.AppData["crystals"].append({"structureName":result['structureName'], "lattice":result['lattice'], "saedAnalyzer":result['saedAnalyzer']})
        self.AppData["currentCrystal"]=self.AppData["crystals"][0]
        self.AppData["currentCrystalName"]=self.AppData["currentCrystal"]["structureName"] 
        self.comboBoxCrystalNames.clear()  
        crystalList=[item["structureName"] for item in  self.AppData["crystals"]]            
        self.comboBoxCrystalNames.addItems(crystalList)
        self.progressBar.setValue(self.progressBar.value()+1)
        
    def doneCrystalDone(self):
        QtGui.QMessageBox.information(self, "Done!", "Done loadig the crystals Now Enjoy!")
        crystalList= [i["structureName"] for i in self.AppData["crystals"]]
        if len(self.AppData["crystals"])>0:
            self.AppData["currentCrystal"]=self.AppData["crystals"][0]
            self.AppData["currentCrystalName"]=self.AppData["crystals"][0]["structureName"] 
            self.comboBoxCrystalNames.clear()               
            self.comboBoxCrystalNames.addItems(crystalList)
            widgetStates= [{'handle':self.pushButtonSolve, 'setState':'show'},
                        {'handle':self.pushButtonGenerateDTable, 'setState':'show'},
                        {'handle':self.pushButtonGenerateDTable, 'setState':'show'},
                        {'handle':self.comboBoxCrystalNames, 'setState':'show'}
                        ]
        self.setWidgetStates(widgetStates)
        
        logging.info("Crystals were initialized succesfully")

 
    def loadCifs(self):
        structureFiles=self.AppData["structureFiles"]
        print("the file that will be processed are ", structureFiles)
        crystals=[]
        self.AppData["crystals"]=crystals
        totalNumberOfStructures=len(structureFiles)
        self.progressBar.setMaximum(totalNumberOfStructures)
        self.progressBar.setValue(0)
        self.cifLoadThread = loadCrystalThread(cifFileList=structureFiles,structureNameList=self.AppData["structureNames"],maxHkl=self.AppData["solverOptions"]['maxHkl'])
        self.connect(self.cifLoadThread, SIGNAL("addCrystalFromThread(PyQt_PyObject)"), self.addCrystalFromThread)
        self.connect(self.cifLoadThread, SIGNAL("finished()"), self.doneCrystalDone)
        self.cifLoadThread.start()
        
        crystalList= [i["structureName"] for i in self.AppData["crystals"]]
        
    
    
    def loadMachine(self,machineName,cameraLength):
        machines=self.AppData["microscopes"]
        for name in machines:
            if name==machineName:
                for item in machines[name]:
                    if item["cameraLength"]==cameraLength:
                        logging.info("Yes found the calibrtation parametrs")
                        self.AppData["calibration"] = item
                        self.AppData["calibration"]["machineName"]=name
                        logging.info("loaded Machine is : "+str(self.AppData["calibration"]))
                        break
                
    
    def loadDefautls(self):
        jsonName=r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\defaults\defaultsForGUI.json'
        self.AppData["jsonFileName"]=jsonName        
        with open(jsonName, 'r') as f:
            tmpData = json.load(f)
        for key in tmpData:
           self.AppData[key]=tmpData[key]
        
        machines=self.AppData["microscopes"]
        #self.AppData["machine"]["cameraLength"]=machines[0][0]["cameraLength"]
        self.AppData["NumberOfMachines"]=len(machines)
        for item in machines:            
            self.comboBoxChoseMachine.addItem(item)
        
        #self.loadMachine(machineName="2000Fx",cameraLength="60cm")
        print(tmpData)
             
        self.loadSelectedFile(self.AppData["imageFileName"],autoSpotPositions=False) 
        self.OriginX.setValue(self.AppData["Origin"][0])
        self.OriginY.setValue(self.AppData["Origin"][1])        
        self.Spot1X.setValue(self.AppData["Spot1"][0])
        self.Spot1Y.setValue(self.AppData["Spot1"][1])        
        self.Spot2X.setValue(self.AppData["Spot2"][0])        
        self.Spot2Y.setValue(self.AppData["Spot2"][1])
        #self.loadCifs()
        #print(self.AppData["Spot2"][0], type(self.AppData["Spot2"][0]))
        #self.updatePlot()
        print("Loaded the defaults succesfully !!!")
        
    
    def loadConfiguration(self, fileName=None,updatePlot=True):
        """
        setting updatePlot=False ensures no display of the pattern. Useful for the initialization of the GUI where we dont want to show default pattern
        """
        if fileName is None:
            defaultFilePath=self.AppData["filePathData"]["lastUsedDirForConfigFile"]['path']
            if not os.path.isfile(defaultFilePath):
               defaultFilePath =self.AppData["filePathData"]["lastUsedDirForConfigFile"]['default']
            jsonName = QtGui.QFileDialog.getOpenFileName(self, 'Chose the json File ', 
                            defaultFilePath,"Json files (*.json )", QtGui.QFileDialog.DontUseNativeDialog)
            self.AppData["jsonFileName"]=jsonName
            self.AppData["filePathData"]["lastUsedDirForConfigFile"]['path']=jsonName       
        else:
            jsonName=fileName
        with open(jsonName, 'r') as f:
            tmpData = json.load(f)
        for key in tmpData:
            self.AppData[key]=tmpData[key]
            
        logging.info("Just read the JSON file : {:s} successfully".format(jsonName) )
        if updatePlot:
            self.loadSelectedFile(self.AppData["imageFileName"],autoSpotPositions=False) 
        self.OriginX.setValue(self.AppData["Origin"][0])
        self.OriginY.setValue(self.AppData["Origin"][1])        
        self.Spot1X.setValue(self.AppData["Spot1"][0])
        self.Spot1Y.setValue(self.AppData["Spot1"][1])        
        self.Spot2X.setValue(self.AppData["Spot2"][0])        
        self.Spot2Y.setValue(self.AppData["Spot2"][1])
        self.spinBoxMaxHkl.setValue(self.AppData["solverOptions"]["maxHkl"])
        self.plainTextEditDmax.setPlainText(str(self.AppData["solverOptions"]["dMax"]))
        self.plainTextEditDmin.setPlainText(str(self.AppData["solverOptions"]["dMin"]))
        self.plainTextEditDeltaAngle.setPlainText(str(self.AppData["solverOptions"]["allowedAngleError"]))
        self.plainTextEditDeltaDSpacing.setPlainText(str(self.AppData["solverOptions"]["allowedDError"]))
        self.plainTextEditCameraConst.setPlainText(str(self.AppData["solverOptions"]["cameraConstant"]))
        if updatePlot:
            self.updatePlot()
        self.pushButtonSolve.setEnabled(True)
             
    
    def selectFile(self):
        imDir=self.AppData["filePathData"]["lastUsedDirForPattern"]['path']
        if not os.path.isdir(imDir):
               defaultFilePath =self.AppData["filePathData"]["lastUsedDirForPattern"]['default']
        
        imName = QtGui.QFileDialog.getOpenFileName(self, 'Chose the Diffraction Pattern', 
         defaultFilePath,"Image files (*.jpg *.tif *.png)", QtGui.QFileDialog.DontUseNativeDialog)
        self.AppData["imageFileName"]=imName
        self.loadSelectedFile(imName) 
        
    def loadSelectedFile(self,imName="",autoSpotPositions=True):
        
        if imName=="":
            self.AppData["workingDirectory"]=""
            self.AppData["patternName"]=""
        else:
            #imName = r"D:/CurrentProjects/python_trials/work_pycrystallography/pycrystallography/data/SADCalib/60cm8bit.tif"    
            if '..' in imName or './' in imName or '\.' in imName: ## relative path is provided
                fileName = os.path.basename(imName)
                dirName = os.path.dirname(os.path.realpath(imName))
                imName = os.path.join(dirName,fileName)
            if imName[-4:]=='.tif':
                im = tif.imread(imName)
                
            else:
                im=cv2.imread(imName,cv2.IMREAD_ANYDEPTH)
            
            if im is None:
                    logging.error("Could not read the image data from : {:s} Could be due to improper format".format(imName) )
                
            if im is None:
                logging.error("Unable to load the open the image {:s} check the path.".format(imName))
            
            if im.max()>255: ###more than 8 bit now we convert that to 8 bit
                logging.info("Image is more than 8 bit. Hence converting to 8 bit automatically.")
                im = (im/im.max()*255).astype(np.uint8)
                assert im.max()>250 , "Problem in converting to 8bit"
                
            #im = cv2.cv.LoadImage(imName,cv2.cv.CV_LOAD_GRAYSCALE)
            #self.axes.imshow(im)
            self.AppData["patternImageData"]=im
            print(im.shape)        
            self.AppData["patternBounds"] = [0,0,im.shape[1],im.shape[0]]
            self.AppData["patternName"]=os.path.basename(imName)
            if "plotHandle" in self.AppData:
                self.AppData["plotHandle"].update_figure(im,self.AppData["patternName"], latticePoints=None,title='')
            self.AppData["workingDirectory"]=os.path.dirname(imName)
        
        if autoSpotPositions:
            self.AppData["Origin"] =   [int(im.shape[0]/2),int(im.shape[1]/2)]
            self.OriginX.setValue(self.AppData["Origin"][0])
            self.OriginY.setValue(self.AppData["Origin"][1])
            
            self.Spot1X.setValue(self.AppData["Origin"][0]+200)
            self.Spot1Y.setValue(self.AppData["Origin"][1]+0)
            
            self.Spot2X.setValue(self.AppData["Origin"][0]+0)
            self.Spot2Y.setValue(self.AppData["Origin"][1]+200)
     
            self.AppData["Spot1"][0]=self.OriginX.value()
            self.AppData["Spot1"][1]=self.OriginX.value()
            self.AppData["Spot2"][0]=self.OriginX.value()
            self.AppData["Spot2"][1]=self.OriginX.value()
            
        
        print(self.AppData["patternName"])
        
        
    def updateMeasuredValues(self):
        xyData=[self.AppData["Origin"],self.AppData["Spot1"], self.AppData["Spot2"]]
        cameraConstant= self.AppData["solverOptions"]["cameraConstant"]
        measuredValues = SaedAnalyzer.extractAngleAndDvaluesFromExpSpotsData(xyData=xyData, cameraConstant=cameraConstant)
        #result =  {"angle":angleBetweenSpots, "dRatio":dRatioMeasured, "spot1Length":spot1Length,"spot1Length":spot2Length}                               

        text = "measuredAngle & Dratio : {:.1f} , {:.3f}\n lenghts of spot1&2 : {:.2f}, {:.2f}\n ".format(measuredValues["angle"],
                                                                                                                                measuredValues["dRatio"],
                                                                                                                                measuredValues["spot1Length"],
                                                                                                                                measuredValues["spot2Length"],
                                                                                                                                )
    
        text=text+"D spacings measured based on CameraConst \n spot1 and spot2 : {:.3f} , {:.3f}Ang\n".format(measuredValues["spot1dSpacing"],measuredValues["spot2dSpacing"])
        self.plainTextEditMeasuredValues.setPlainText(text)   
            
    def updateOriginX(self):
        self.AppData["Origin"][0]=self.OriginX.value()
        print(self.AppData["Origin"])
        self.updateMeasuredValues()
        self.updatePlot()        
    def updateOriginY(self):
        self.AppData["Origin"][1]=self.OriginY.value()
        print(self.AppData["Origin"])
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot1X(self):
        self.AppData["Spot1"][0]=self.Spot1X.value()
        print(self.AppData["Spot1"])
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot1Y(self):
        self.AppData["Spot1"][1]=self.Spot1Y.value()
        print(self.AppData["Spot1"])
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot2X(self):
        self.AppData["Spot2"][0]=self.Spot2X.value()
        print(self.AppData["Spot2"])
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot2Y(self):
        self.AppData["Spot2"][1]=self.Spot2Y.value()
        print(self.AppData["Spot2"])
        self.updateMeasuredValues()
        self.updatePlot()

    def updatePlot(self):
        origin = self.AppData["Origin"]
        vec1 = self.AppData["Spot1"]
        vec2 = self.AppData["Spot2"]
        spcialPointsForMarking=[origin,vec1,vec2]
        maxIndices=3  
        latticeBounds=self.AppData["patternBounds"]
        latticeBounds = [latticeBounds[0],latticeBounds[2],latticeBounds[1],latticeBounds[3]]      
        latticePoints = gu.generate2Dlattice(origin, vec1, vec2, maxIndices=5,latticeBounds=latticeBounds, plotOn=False)
        self.AppData["latticePoints"]=latticePoints      
        data = self.AppData["patternImageData"]         
        self.AppData["plotHandle"].update_figure(data,self.AppData["patternName"],latticePoints,title='',spcialPointsForMarking=spcialPointsForMarking)
        
    def saveConfiguration(self):            
        
        imageName = self.AppData["imageFileName"]
        if len(imageName)>0:
            suggestedJsonName = imageName[:-4]+".json"
        else:
            suggestedJsonName = os.path.join(self.AppData["workingDirectory"],'unttiled.json')
        
        name = QtGui.QFileDialog.getSaveFileName(self,  'Save File', suggestedJsonName, "Json Files (*.json)") 
        
        keysToStore = ['Origin','Spot1','Spot2','imageFileName','markSpots','patternBounds','workingDirectory', 'structureFiles','structureNames',
                       'solverOptions']
        tmpDict = {}
         
        for item in  keysToStore:
            tmpDict[item]=self.AppData[item]      
        with open(name, 'w') as fp:
            json.dump(tmpDict, fp, sort_keys=True, indent=4)
        
        
         
        

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())