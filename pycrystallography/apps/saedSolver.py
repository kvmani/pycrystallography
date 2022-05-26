import os, sys
import numpy as np

os.environ['QT_API'] = 'pyqt'
from qtpy import QT_VERSION
QT5 = QT_VERSION[0] == '5'
del QT_VERSION


if QT5:
    from qtpy import QtWidgets
    from qtpy import QtCore
    from qtpy.QtCore import Qt
    from qtpy import QtGui
    from qtpy import QtWidgets
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
    from qtpy.QtCore import QThread
else:
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    QtGui.QtWidgets = QtGui.QWidget
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
    from PyQt4.QtCore import QThread
    

from pycrystallography.visualization.plotter import MyStaticMplCanvas
import logging
import json
from pycrystallography.core.saedAnalyzer import SaedAnalyzer
import tifffile as tif

import cv2
import pycrystallography.utilities.graphicUtilities as gu
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.apps.saedSimulator import SaedSimulatorWindow
from pycrystallography.apps.saedSpotsPicker import DraggablePoints
from pylatexenc.latex2text import LatexNodes2Text
from pycrystallography.io.dm3Reader import DM3

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


class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtGui.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)    

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)    


class MainWindow(QtGui.QWidget):
    #class MainWindow(QtWidgets.QMainWindow, WindowMixin):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        
        self.setGeometry(0,0, 800,1020)
        self.setWindowTitle("saedSolver")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        #self.resize(500,650)
        self.setMinimumSize(500,650)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.center()
        
        # --- Menu --- #
        open = QtGui.QAction("Exit", self)
        save = QtGui.QAction("Save", self)
        build = QtGui.QAction("Build", self)
        exit = QtGui.QAction("Quit", self)
        
        menu_bar = QtGui.QMenuBar()
        file = menu_bar.addMenu("&File")
        help = menu_bar.addMenu("&Help")
        
        file.addAction(open)
        file.addAction(save)
        file.addAction(build)
        file.addAction(exit)
        
        self.vBoxLoadFiles =QtGui.QGroupBox("Loaiding & Saving Operations") 
        self.loadPatternButton=QtGui.QPushButton("Load pattern")
        self.loadConfigurationButton=QtGui.QPushButton("Load Configuration")
        self.loadCrystalsButton=QtGui.QPushButton("Load Crystals")
        self.comboBoxCrystalNames=QtGui.QComboBox()
        self.saveConfigurationButton=QtGui.QPushButton("Save Configuration")
        self.generateReportButton=QtGui.QPushButton("Generate Report ")
        self.saveAllButton=QtGui.QPushButton("Save All")
        
        
        self.fileLoadLayout = QtGui.QGridLayout()
        self.fileLoadLayout.addWidget(self.loadPatternButton,0,0,1,1)
        self.fileLoadLayout.addWidget(self.loadConfigurationButton,0,1,1,1)
        self.fileLoadLayout.addWidget(self.loadCrystalsButton,0,2,1,1)
        self.fileLoadLayout.addWidget(self.comboBoxCrystalNames,0,3,1,2)
        self.fileLoadLayout.addWidget(self.saveConfigurationButton,1,0,1,1)
        self.fileLoadLayout.addWidget(self.saveAllButton,1,2,1,1)
        self.fileLoadLayout.addWidget(self.generateReportButton,1,3,1,2)
        
#         spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
#         self.fileLoadLayout.addItem(spacerItem)      
        
        
        self.vBoxLoadFiles.setLayout(self.fileLoadLayout)
        
        groupBoxAdjustSpots =QtGui.QGroupBox("Spot Adjustments")
        gridLayOut = QtGui.QGridLayout()
        adjustSpotsLable=QtGui.QLabel("adjustSpots")
        spotsXLable=QtGui.QLabel("X")
        spotsYLable=QtGui.QLabel("Y")
        spotsOriginLable=QtGui.QLabel("Origin")
        spotsSpot1Lable=QtGui.QLabel("spot1")
        spotsSpot2Lable=QtGui.QLabel("spot2")
        self.spotsOriginEditX=QtGui.QLineEdit("100")
        self.spotsOriginEditY=QtGui.QLineEdit("100")
        self.spotsSpot1EditX=QtGui.QLineEdit("150")
        self.spotsSpot1EditY=QtGui.QLineEdit("100")
        self.spotsSpot2EditX=QtGui.QLineEdit("100")
        self.spotsSpot2EditY=QtGui.QLineEdit("150")
        
        gridLayOut.addWidget(adjustSpotsLable,0,0)
        gridLayOut.addWidget(spotsXLable,0,1)
        gridLayOut.addWidget(spotsYLable,0,2)
        
        gridLayOut.addWidget(spotsOriginLable,1,0)
        gridLayOut.addWidget(self.spotsOriginEditX,1,1)
        gridLayOut.addWidget(self.spotsOriginEditY,1,2)
        
        gridLayOut.addWidget(spotsSpot1Lable,2,0)
        gridLayOut.addWidget(self.spotsSpot1EditX,2,1)
        gridLayOut.addWidget(self.spotsSpot1EditY,2,2)
        
        gridLayOut.addWidget(spotsSpot2Lable,3,0)
        gridLayOut.addWidget(self.spotsSpot2EditX,3,1)
        gridLayOut.addWidget(self.spotsSpot2EditY,3,2)
        groupBoxAdjustSpots.setLayout(gridLayOut)
        self.plotOptionsBox = QtGui.QGroupBox("Plot Options")
        plotOptionsLayout = QtGui.QHBoxLayout()
        self.plotOptionsMarkSpotsHnadle=QtGui.QCheckBox()
        self.plotOptionsMarkSpotsHnadle.setText("markSpots")
        self.plotOptionsPlotKikuchiHandle=QtGui.QCheckBox()
        self.plotOptionsPlotKikuchiHandle.setText("plotKikuchi")
        
        self.plotOptionsMarkKikuchi=QtGui.QCheckBox()
        self.plotOptionsMarkKikuchi.setText("mark Kikuchi")
        self.plotOptionsShowSystematicAbsncentSpots=QtGui.QCheckBox()
        self.plotOptionsShowSystematicAbsncentSpots.setText("Show Systematic Absent Reflections")
        
        #plotOptionsLayout.addWidget(QtGui.QLabel("markSpots:"))
        plotOptionsLayout.addWidget(self.plotOptionsMarkSpotsHnadle)
        #plotOptionsLayout.addWidget(QtGui.QLabel())
        plotOptionsLayout.addWidget(self.plotOptionsPlotKikuchiHandle)
        #plotOptionsLayout.addWidget(QtGui.QLabel(""))
        plotOptionsLayout.addWidget(self.plotOptionsMarkKikuchi)
        plotOptionsLayout.addWidget(self.plotOptionsShowSystematicAbsncentSpots)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        plotOptionsLayout.addItem(spacerItem)
        
        self.plotOptionsBox.setLayout(plotOptionsLayout)
        
        
        self.hBoxTables =QtGui.QGroupBox("Results") 
        self.dTableHandle=QtGui.QTableWidget()
        self.solutionsTableHandle=QtGui.QTableWidget()
         
        self.hBoxTablesLayout = QtGui.QHBoxLayout()
        self.hBoxTablesLayout.addWidget(self.dTableHandle,35)
        self.hBoxTablesLayout.addWidget(self.solutionsTableHandle,65)
        self.hBoxTables.setLayout(self.hBoxTablesLayout)
        
        
        self.plotAreaHandle=QtGui.QGraphicsView()
#         self.setupUi(self)
#         self.main_widget = QtGui.QWidget(self.plotArea)
        plotLayout = QtGui.QVBoxLayout(self.plotAreaHandle)         
        self.mainPlot = MyStaticMplCanvas(self.plotAreaHandle, width=8, height=8, dpi=100)  
        self.mainPlot.compute_initial_figure()      
        plotLayout.addWidget(NavigationToolbar(self.mainPlot, self))
        plotLayout.addWidget(self.mainPlot)
        
        
        #self.plotAreaHandle.setSceneRect( 0, 0, 800, 800 )
        self.solverOptionsBox = QtGui.QGroupBox("Solver Options")
        solverOptionsLayout = QtGui.QFormLayout()
        self.solverOptionsMaxHklHnadle=QtGui.QLineEdit("3")
        self.solverOptionsCameraConstantHandle=QtGui.QLineEdit("1009.0")
        self.solverOptionsAllowedDeltaAngleHandle=QtGui.QLineEdit("5.0")
        self.solverOptionsAllowedDeltaD_RatioHandle=QtGui.QLineEdit("10.0")
        
        solverOptionsLayout.addRow(QtGui.QLabel("maxHkl:"), self.solverOptionsMaxHklHnadle)
        solverOptionsLayout.addRow(QtGui.QLabel("Camera Constant:"), self.solverOptionsCameraConstantHandle)
        solverOptionsLayout.addRow(QtGui.QLabel("Allowed Error in Angle (deg):"), self.solverOptionsAllowedDeltaAngleHandle)
        solverOptionsLayout.addRow(QtGui.QLabel("Allowed Error in d Ratio:"), self.solverOptionsAllowedDeltaD_RatioHandle)
        
        self.solverOptionsBox.setLayout(solverOptionsLayout)
        
        self.measurementConditionsBox = QtGui.QGroupBox("Measurment Conditions")
        measurementConditionsLayout = QtGui.QFormLayout()
        self.measurementConditionsVoltage=QtGui.QLineEdit("200kV")
        self.measurementConditionsCameraLengthHandle=QtGui.QLineEdit("100cm")
        self.measurementConditionsAlphaTilt=QtGui.QLineEdit("0.")
        self.measurementConditionsBetaTilt=QtGui.QLineEdit("0.")
        self.measurementConditionsdiffractionRotationAngle=QtGui.QLineEdit("0.")
        
        measurementConditionsLayout.addRow(QtGui.QLabel("TEM voltage:"), self.measurementConditionsVoltage)
        measurementConditionsLayout.addRow(QtGui.QLabel("Camera Length:"), self.measurementConditionsCameraLengthHandle)
        measurementConditionsLayout.addRow(QtGui.QLabel(u" \u03b1 Tilt (\xb0)"), self.measurementConditionsAlphaTilt)
        measurementConditionsLayout.addRow(QtGui.QLabel(u" \u03b2 Tilt (\xb0)"), self.measurementConditionsBetaTilt)
        measurementConditionsLayout.addRow(QtGui.QLabel(u"diffractionRotationAngle(\xb0)"), self.measurementConditionsdiffractionRotationAngle)
        self.measurementConditionsBox.setLayout(measurementConditionsLayout)
        
        
        self.vBoxSolverButtons =QtGui.QGroupBox() 
        self.generateDTableButton=QtGui.QPushButton("Generate d table")
        self.solveButton=QtGui.QPushButton("Solve")
        self.clearSolutionsButton=QtGui.QPushButton("clear Solutions")
        
        self.buttonLayout1 = QtGui.QHBoxLayout()
        self.buttonLayout1.addWidget(self.generateDTableButton)
        self.buttonLayout1.addWidget(self.solveButton)
        self.buttonLayout1.addWidget(self.clearSolutionsButton)
        self.vBoxSolverButtons.setLayout(self.buttonLayout1)
        
        vboxMesages = QtGui.QGroupBox()
        vboxMesagesLayout= QtGui.QHBoxLayout()
        self.meseuredValuesLable=QtGui.QLabel("measured \n Values:")
        self.meseuredValuesTextBox=QtGui.QPlainTextEdit("measured Values Will be shown here !!!")
        
        vboxMesagesLayout.addWidget(self.meseuredValuesLable,10)
        vboxMesagesLayout.addWidget(self.meseuredValuesTextBox,90)
        vboxMesages.setLayout(vboxMesagesLayout)

        vboxStatusDisplay = QtGui.QGroupBox()
        vboxStatusDisplayLaout= QtGui.QHBoxLayout()
        self.solverStatusMessage=QtGui.QLabel("solver is not started Yet!!!")
        self.progressBar=QtGui.QProgressBar()
        
        
        vboxStatusDisplayLaout.addWidget(self.solverStatusMessage,60)
        vboxStatusDisplayLaout.addWidget(self.progressBar,40)
        vboxStatusDisplay.setLayout(vboxStatusDisplayLaout)

        
        vbox2 = QtGui.QGroupBox()
        vBox2Layout = QtGui.QGridLayout()
        vBox2Layout.addWidget(self.vBoxLoadFiles,0,0,1,4)
        vBox2Layout.addWidget(self.solverOptionsBox,1,0,1,2)
        vBox2Layout.addWidget(self.measurementConditionsBox,1,2,1,2)
        #vBox2Layout.addWidget(self.solverOptionsBox,1,1,1,1)
        vBox2Layout.addWidget(groupBoxAdjustSpots,2,2,1,2)
        vBox2Layout.addWidget(vboxMesages,2,0,1,2)
        
        vBox2Layout.addWidget(self.vBoxSolverButtons,5,0,1,4)
        vBox2Layout.addWidget(self.hBoxTables,6,0,2,4)
        vBox2Layout.addWidget(vboxStatusDisplay,4,0,1,4)
        vbox2.setLayout(vBox2Layout)
        self.adjustBrightness=QtGui.QLineEdit("1.0")
        adjustBrightnessLable=QtGui.QLabel("Adjust Brightness")
        adjustBrightnessLable.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.adjustContrast=QtGui.QLineEdit("1.0")
        adjustContrastLabel=QtGui.QLabel("Adjust Contrast")
        adjustContrastLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.histogramEqualizeEnable=QtGui.QCheckBox()
        histogramEqualizeEnableLabel=QtGui.QLabel("Enable Histogram Equalization")
        histogramEqualizeEnableLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.imageThresholdLow=QtGui.QLineEdit("25")
        self.imageThresholdHigh=QtGui.QLineEdit("200")
        imageThresholdLowLabel=QtGui.QLabel("Image Threshold Lower Cutoff")
        imageThresholdHighLabel=QtGui.QLabel("Image Threshold higher Cutoff")
        self.acceptSolutiuoiForSimulatorButton=QtGui.QPushButton("Use this solution for Simulator")
        self.acceptSolutiuoiForSimulatorButton.setEnabled(True)
        self.adjustSpotsInteractively = QtGui.QPushButton("Adjust Spots")
        
        vbox3 = QtGui.QGroupBox("Image Display Properties")
        vBox3Layout = QtGui.QGridLayout()
        vBox3Layout.addWidget(self.plotOptionsBox,  0,0,1,10)
        vBox3Layout.addWidget(adjustBrightnessLable,1,0,1,1)
        vBox3Layout.addWidget(self.adjustBrightness,1,1,1,1)
        vBox3Layout.addWidget(adjustContrastLabel,  1,2,1,1)
        vBox3Layout.addWidget(self.adjustContrast,  1,3,1,1)
        vBox3Layout.addWidget(histogramEqualizeEnableLabel,1,4,1,1)
        
        vBox3Layout.addWidget(self.histogramEqualizeEnable,1,5,1,1)
        vBox3Layout.addWidget(imageThresholdLowLabel,1,6,1,1)
        vBox3Layout.addWidget(self.imageThresholdLow,1,7,1,1)
        vBox3Layout.addWidget(imageThresholdHighLabel,1,8,1,1)
        vBox3Layout.addWidget(self.imageThresholdHigh,1,9,1,1)
        vBox3Layout.addWidget(self.plotAreaHandle, 2,0,10,10)
        vBox3Layout.addWidget(self.acceptSolutiuoiForSimulatorButton,2,11,1,1)
        vBox3Layout.addWidget(self.adjustSpotsInteractively,3,11,1,1)
        
#knakndlk
        vbox3.setLayout(vBox3Layout)
 
        self.optionsAndPlotAreaBox=QtGui.QGroupBox()
        self.optionsAndPlotAreaBoxLayout = QtGui.QHBoxLayout()
 
        self.optionsAndPlotAreaBoxLayout.addWidget(vbox2,25)
        self.optionsAndPlotAreaBoxLayout.addWidget(vbox3,50)
        
        self.optionsAndPlotAreaBox.setLayout(self.optionsAndPlotAreaBoxLayout)
        
                
        tab_widget = QtGui.QTabWidget()
        self.tab1 = QtGui.QWidget()
        self.tab2 = QtGui.QWidget()
        
        p1_vertical = QtGui.QVBoxLayout(self.tab1)
        p2_vertical = QtGui.QVBoxLayout(self.tab2)
        
        tab_widget.addTab(self.tab1, "MainWindow")
        tab_widget.addTab(self.tab2, "SimulatedSaed")
        
        self.messageWindow = QtGui.QTextBrowser()
        p1_vertical.addWidget(self.optionsAndPlotAreaBox,80)
        p1_vertical.addWidget(self.messageWindow,20)
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(menu_bar)
        vbox.addWidget(tab_widget)
        
        self.setWindowIcon(QtGui.QIcon(r'..\..\data\programeData\mainIcon.png'))
        #self.setWindowIcon
        self.setLayout(vbox)
        logTextBox = QPlainTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        # You can control the logging level
        logging.getLogger().setLevel(logging.INFO)
        ll= QtGui.QVBoxLayout(self.messageWindow)
        ll.addWidget(logTextBox.widget)  
        self.initilizeAppData()
        self.AppData["AllowSimulator"]=True
        self.loadPatternButton.clicked.connect(lambda : self.selectFile())
        self.spotsOriginEditX.editingFinished.connect(self.updateOriginX)
        self.spotsOriginEditY.editingFinished.connect(self.updateOriginY)
        self.spotsSpot1EditX.editingFinished.connect(self.updateSpot1X)
        self.spotsSpot1EditY.editingFinished.connect(self.updateSpot1Y)
        self.spotsSpot2EditX.editingFinished.connect(self.updateSpot2X)
        self.spotsSpot2EditY.editingFinished.connect(self.updateSpot2Y)
        self.solverOptionsCameraConstantHandle.textChanged.connect(self.cameraConstChanged)
        self.solverOptionsMaxHklHnadle.textChanged.connect(self.updateMaxHkl)
        self.solverOptionsAllowedDeltaAngleHandle.textChanged.connect(self.angleErrorValuechanged)
        self.solverOptionsAllowedDeltaD_RatioHandle.textChanged.connect(self.dSpacingErrorValuechanged)
        self.loadConfigurationButton.clicked.connect(lambda :self.loadConfiguration())
        self.loadCrystalsButton.clicked.connect(lambda :self.loadCrystals())
        self.generateDTableButton.clicked.connect(lambda : self.generateDTable())
        self.solveButton.clicked.connect(lambda : self.startSolver())
        self.solutionsTableHandle.cellClicked.connect(self.cell_was_clicked)
        self.clearSolutionsButton.clicked.connect(lambda : self.clearSolutions())
        self.saveConfigurationButton.clicked.connect(lambda : self.saveConfiguration())
        self.adjustBrightness.editingFinished.connect(self.updateBrightness)
        self.adjustContrast.editingFinished.connect(self.updateContrast)
        self.histogramEqualizeEnable.toggled.connect(lambda:self.histrogramEqulizationOptionChanged(self.histogramEqualizeEnable))
        self.plotOptionsMarkSpotsHnadle.toggled.connect(lambda:self.markSpotsOptionChanged(self.plotOptionsMarkSpotsHnadle))
        self.plotOptionsPlotKikuchiHandle.toggled.connect(lambda : self.showKikuchLinesOptionChanged(self.plotOptionsPlotKikuchiHandle))
        self.plotOptionsMarkKikuchi.toggled.connect(lambda : self.markKikuchLinesOptionChanged(self.plotOptionsMarkKikuchi))
        self.plotOptionsShowSystematicAbsncentSpots.toggled.connect(lambda : self.hideSystamticAbsencesOptionChanged(self.plotOptionsShowSystematicAbsncentSpots))
        self.saveAllButton.clicked.connect(lambda : self.saveAll())
        self.acceptSolutiuoiForSimulatorButton.clicked.connect(self.initilizeSimulator)
        self.adjustSpotsInteractively.clicked.connect(self.adjustSpots)
        self.measurementConditionsVoltage.editingFinished.connect(self.updateMeasurementConditions)
        self.measurementConditionsCameraLengthHandle.editingFinished.connect(self.updateMeasurementConditions)
        self.measurementConditionsAlphaTilt.editingFinished.connect(self.updateMeasurementConditions)
        self.measurementConditionsBetaTilt.editingFinished.connect(self.updateMeasurementConditions)
        self.measurementConditionsdiffractionRotationAngle.editingFinished.connect(self.updateMeasurementConditions)
    
    def updateMeasurementConditions(self):
        self.AppData["measurementConditions"]["voltage"]= self.measurementConditionsVoltage.text()
        self.AppData["measurementConditions"]["cameraLength"]= self.measurementConditionsCameraLengthHandle.text()
        self.AppData["measurementConditions"]["alphaTilt"]=float(self.measurementConditionsAlphaTilt.text())
        self.AppData["measurementConditions"]["betaTilt"]=float(self.measurementConditionsBetaTilt.text())
        self.AppData["measurementConditions"]["diffractionRotationAngle"] = float(self.measurementConditionsdiffractionRotationAngle.text())
        logging.info("Updated the measuremnt conditions succesfully")
    
    def setMeasurementConditions(self):    
        self.measurementConditionsVoltage.setText(self.AppData["measurementConditions"]["voltage"])
        self.measurementConditionsCameraLengthHandle.setText(self.AppData["measurementConditions"]["cameraLength"])
        self.measurementConditionsAlphaTilt.setText(str(np.around(self.AppData["measurementConditions"]["alphaTilt"],2)))
        self.measurementConditionsBetaTilt.setText(str(np.around(self.AppData["measurementConditions"]["betaTilt"],2)))
        self.measurementConditionsdiffractionRotationAngle.setText(str(np.around(self.AppData["measurementConditions"]["diffractionRotationAngle"],2)))
    
    def adjustSpots(self):
        initialPoints=[self.AppData["Origin"],self.AppData["Spot1"],self.AppData["Spot2"] ]
        dp = DraggablePoints(initialPoints=initialPoints, imageData=self.AppData["patternImageData"],axisHandle=None,showPlot=True)
        spotList = dp.returnSpotPositions()
        spotList = np.array(spotList)
        self.AppData["Origin"]=spotList[0]
        self.AppData["Spot1"]=spotList[1]
        self.AppData["Spot2"]=spotList[2]
        self.updatePlot()
        self.updateMeasuredValues()
        logging.info("Updfated the spot measure values please check if they make sense!!!")
        
    
    def initilizeSimulator(self):
        
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]
        saedData = self._currentSolution["pattern"]
        simulatorFrame = SaedSimulatorWindow(currentSaed=saedData, currentSaedAnalyzer=sa)
        self.tab2=simulatorFrame
        self.tab2.show()
    
    def dSpacingErrorValuechanged(self):
        txt =self.solverOptionsAllowedDeltaD_RatioHandle.text() 
        if len(txt)>0:
            allowedDError=float(txt)
            self.AppData["solverOptions"]["allowedDError"]=allowedDError
        self.pushButtonSolve.setEnabled(True)
    
    def angleErrorValuechanged(self):
        txt =self.solverOptionsAllowedDeltaAngleHandle.text() 
        if len(txt)>0:
            allowedAngleError=float(txt)
            self.AppData["solverOptions"]["allowedAngleError"]=allowedAngleError
        self.pushButtonSolve.setEnabled(True)

    
    def updateMaxHkl(self):
        maxHkl=int(self.solverOptionsMaxHklHnadle.text())
        if maxHkl>4:
            #QtGui.QMessageBox.information(self, "Done!", "Done loadig the crystals Now Enjoy!")
            QtGui.QMessageBox.information(self,"Warning!!!", "maxHkl chosen is {:2d} It can take considerable time to load. Please wait.".format(maxHkl))
            
        self.AppData["solverOptions"]["maxHkl"]=maxHkl
        self.loadCifs()
        logging.info("Max Hkl of sover changed to "+str(maxHkl)+'and the crystals are relaoded with changed settings')
        self.pushButtonSolve.setEnabled(True)
    
    
    def savePathInfo(self):
        fileName = self.AppData['filePathData']['pathStorageFile']
        dumpData=self.AppData['filePathData']
        print(self.AppData["filePathData"])
        with open(fileName, 'w') as fp:
            json.dump(dumpData, fp, sort_keys=True, indent=4)
        logging.info("Saved the path storage file:")
    
    def saveAll(self):
        self.savePathInfo()
        self.saveConfiguration()
        logging.info("saved the important stuff !!!!")
    
    
    def markSpotsOptionChanged(self,b):

        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['markSpots']=True
        else:
            self.AppData['solutionPlotOptions']['markSpots']=False
        
        crystal = self.AppData["currentCrystal"]
        sa  = crystal["saedAnalyzer"]
        saed= self._currentSolution
        
        self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
    
     
    
    def hideSystamticAbsencesOptionChanged(self,b):

        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['hideSystamticAbsences']=True
        else:
            self.AppData['solutionPlotOptions']['hideSystamticAbsences']=False
        
        crystal = self.AppData["currentCrystal"]
        sa  = crystal["saedAnalyzer"]
        saed= self._currentSolution
        
        self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
    
     
    
    def showKikuchLinesOptionChanged(self,b):
        
        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['plotKikuchi']=True
        else:
            self.AppData['solutionPlotOptions']['plotKikuchi']=False
        
        crystal = self.AppData["currentCrystal"]
        sa  = crystal["saedAnalyzer"]
        saed= self._currentSolution
        logging.info("Changed the kikuch option")
        self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
    
    
    def markKikuchLinesOptionChanged(self,b):
        
        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['markKikuchi']=True
        else:
            self.AppData['solutionPlotOptions']['markKikuchi']=False
        
        crystal = self.AppData["currentCrystal"]
        sa  = crystal["saedAnalyzer"]
        saed= self._currentSolution
        self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
    
    
    
    def updateBrightness(self):
        brightness = float(self.adjustBrightness.text())
        self.AppData['solutionPlotOptions']["brightness"]=brightness
        logging.info("Adjusting the Brighness : {:.2f}".format(brightness))
        self.updatePlot()
    
    def updateContrast(self):
        contrast = float(self.adjustContrast.text())
        self.AppData['solutionPlotOptions']["contrast"]=contrast
        logging.info("Adjusting the contrast {:.2f}".format(contrast))
        self.updatePlot()
    
    def histogramEqualize(self):
        contrast = float(self.adjustContrast.text())
        self.AppData['solutionPlotOptions']["contrast"]=contrast
        self.updatePlot()



    def histrogramEqulizationOptionChanged(self,b):
        if b.isChecked()==True:
            self.AppData['solutionPlotOptions']['histogramEqualize']=True
        else:
            self.AppData['solutionPlotOptions']['histogramEqualize']=False

        self.updatePlot()

    
    def saveConfiguration(self):            
        
        imageName = self.AppData["imageFileName"]
        if len(imageName)>0:
            suggestedJsonName = imageName[:-4]+".json"
        else:
            suggestedJsonName = os.path.join(self.AppData["workingDirectory"],'unttiled.json')
        
        name = QtGui.QFileDialog.getSaveFileName(self,  'Save File', suggestedJsonName, "Json Files (*.json)") 
        lastUsedPath=os.path.dirname(name) 
        self.AppData["filePathData"]["lastUsedDirForConfigFile"]["path"]=lastUsedPath
        keysToStore = ['Origin','Spot1','Spot2','imageFileName','patternBounds','workingDirectory', 'structureFiles','structureNames',
                       'solverOptions','solutionPlotOptions','measurementConditions']
        tmpDict = {}
         
        for item in  keysToStore:
            if item in self.AppData:
                if isinstance(self.AppData[item],(np.ndarray,)):
                    tmpDict[item]=self.AppData[item].tolist()
                else:
                    tmpDict[item]=self.AppData[item]      
        with open(name, 'w') as fp:
            json.dump(tmpDict, fp, sort_keys=True, indent=4)
        
        self.savePathInfo()
    
    def initilizeAppData(self):
        self.AppData=  {}
        self.AppData["patternName"]=''
        self.AppData["imageFileName"]=''
        self.AppData["jsonFileName"]=''
        self.AppData["workingDirectory"]=""
        self.AppData["SAEDSolutuions"]=None
        self.AppData["solverOptions"]={"maxHkl":3, "dMin":0, "dMax":10, "allowedAngleError":5., "allowedDError":10.,"cameraConstant":1079.}
        self.AppData['solutionPlotOptions']={'markSpots':False, 'hideSystamticAbsences':True, "histogramEqualize":False,"brightness":1.0, "contrast":1.0,
                                             'plotKikuchi':False, 'markKikuchi':False }
        self.AppData["cifFiles"]=[]
        self.AppData["filePathData"]={'lastUsedDirForPattern':{'path':'','default':''},
                                      'lastUsedDirForConfigFile':{'path':'','default':''},
                                      'dirForProgramTmpData':{'path': r'../../data/programeData/'},
                                      'pathStorageFile': r'../../data/programeData/pathStorageFile.jason'}
                                      
        logging.info("The path file is located in : "+self.AppData["filePathData"]['pathStorageFile'])
        self.AppData["Origin"]=[100,100]
        self.AppData["Spot1"]=[150,100]
        self.AppData["Spot2"]=[100,150]
        self.AppData["patternBounds"]=[0,0,4008,2671]
        self.AppData["currentCrystal"]=None
        self.AppData["measurementConditions"]={"voltage":"200kV", "cameraLength":"40cm", "alphaTilt":0.,"betaTilt":0., "diffractionRotationAngle":0.,        
                                               "options":{"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":False}}
        self.AppData["imagePixcelSize"]=None
        self.AppData["imagePixcelUnits"]=None
                        
        
        jsonName=self.AppData["filePathData"]['pathStorageFile']
        print(jsonName)
        with open(jsonName, 'r') as f:
            tmpData = json.load(f)
        for key in tmpData:
            self.AppData['filePathData'][key]=tmpData[key]
            logging.debug("Loaded default paths successfully")
            
        self.solutionsTableHandle.setRowCount(1)
        self.solutionsTableHandle.setColumnCount(8)
        self.solutionsTableHandle.setHorizontalHeaderLabels(["S.No", "CrystalName" , "   ZoneAxis  ", "  spot1 ","  spot2 " , "AngleError", r"dError(%)", "Correlation"])
        self.solutionsTableHandle.resizeColumnsToContents()
        self.solutionsTableHandle.resizeRowsToContents()
        
        self.dTableHandle.setRowCount(1)
        self.dTableHandle.setColumnCount(4)
        self.dTableHandle.setHorizontalHeaderLabels(["  Plane  ", "dSpacing" , r"  1/d (inv nm)", "Int(I)"])
        self.setMeasurementConditions()
        #self.solveButton.hide()
        logging.info("Updated the appData with defaults!!")
    
    def updateMeasuredValues(self):
        xyData=[self.AppData["Origin"],self.AppData["Spot1"], self.AppData["Spot2"]]
        print("Current xyData:",xyData)
        if self.AppData["imagePixcelSize"] is not None:
            imagePixcelSize=self.AppData["imagePixcelSize"]
            measuredValues = SaedAnalyzer.extractAngleAndDvaluesFromExpSpotsData(xyData=xyData, imagePixcelSize=imagePixcelSize) 
            self.AppData["solverOptions"]["cameraConstant"] = measuredValues["cameraConstant"]
            self.solverOptionsCameraConstantHandle.setText(str(np.around(measuredValues["cameraConstant"],3)))
            logging.info("Update the camera constant from Dm3 file.")
        else:
            cameraConstant= self.AppData["solverOptions"]["cameraConstant"]
            measuredValues = SaedAnalyzer.extractAngleAndDvaluesFromExpSpotsData(xyData=xyData, cameraConstant=cameraConstant)
        
        print(measuredValues)
        text = "measuredAngle & Dratio : {:.1f} , {:.3f}\n lenghts of spot1&2 : {:.2f}, {:.2f} (in Pixcels)\n ".format(measuredValues["angle"],
                                                                                                                                measuredValues["dRatio"],
                                                                                                                                measuredValues["spot1Length"],
                                                                                                                                measuredValues["spot2Length"],
                                                                                                                                )
    
        text=text+r"\nD spacings measured based on CameraConst \n spot1 and spot2 : {:.3f} , {:.3f} $\AA$\n".format(measuredValues["spot1dSpacing"],measuredValues["spot2dSpacing"])
        
        text = text+ r"\n Reiprocal Lengths of spot1 and spot2 : {:.3f}, {:.3f} ".format(measuredValues["spot1ReciprocalLength"],measuredValues["spot2ReciprocalLength"])
        text = text+r"$ nm^{-1}$"
        text = text+" The Image picxcel size is : {:.3f} (inv) nm".format(measuredValues["imagePixcelSize"])
        text = LatexNodes2Text().latex_to_text(text)
        self.meseuredValuesTextBox.setPlainText(text)   

    def choseDifferentCrystal(self,i):
        
        print("Just entered the before change : ", self.AppData["crystals"][0]["structureName"])
        for k in range(self.comboBoxCrystalNames.count()):
            print ("The index is ", k , "the item is : ",self.comboBoxCrystalNames.itemData(k))
      
        crystals=self.AppData["crystals"][i]
        self.AppData["currentCrystalName"]=crystals["structureName"]
        self.AppData["currentCrystal"]=crystals
        logging.info("Updated the new crystal succesfully !!! and the new crystal is ",crystals["structureName"])

    def loadCrystals(self):
        cifFiles = QtGui.QFileDialog.getOpenFileNames(self, 'Chose the cif File', 
         r'../../data/structureData',"cif files (*.cif )")
        if len(cifFiles)>0:
            self.AppData["structureFiles"]=cifFiles
            self.AppData["structureNames"]=[]
            for item in cifFiles:
                self.AppData["structureNames"].append(os.path.basename(item)[:-4]) 
            self.loadCifs()
            logging.info("Succesfully loaded the crystal")
        else:
            logging.error("No cif files were choosen !! please chose properly !!!")
    
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
        
    
    def loadSelectedFile(self,imName="",spotPositions=None,autoSpotPositions=True):
        
        im=None
        if imName=="":
            self.AppData["workingDirectory"]=""
            self.AppData["patternName"]=""
        else:
            #imName = r"D:/CurrentProjects/python_trials/work_pycrystallography/pycrystallography/data/SADCalib/60cm8bit.tif"    
            if '..' in imName or r'./' in imName or r'\.' in imName: ## relative path is provided
                fileName = os.path.basename(imName)
                dirName = os.path.dirname(os.path.realpath(imName))
                imName = os.path.join(dirName,fileName)
            if imName[-4:]=='.tif':
                im = tif.imread(imName)
                
            if imName[-4:]==".dm3":
                
                dm3f = DM3(imName)
                infoText = dm3f.info
                im = dm3f.imagedata
                self.AppData["imagePixcelSize"],self.AppData["imagePixcelUnits"]=dm3f.pxsize
                logging.info("Succefully loaded the DM3 file and the info of the image is = "+ str(infoText))
                logging.info("Image pixcel size is set to be... "+str(self.AppData["imagePixcelSize"]))
                
                   
            else:
                im=cv2.imread(imName,cv2.IMREAD_ANYDEPTH)
            
            if im is None:
                logging.error("Unable to load the open the image {:s} check the path.".format(imName))
            
            if len(im.shape)==3:
                ## case of RGB image
                im=im[:,:,0] ### taking only red part of the image
            
            if im.max()>255: ###more than 8 bit now we convert that to 8 bit
                logging.info("Image is more than 8 bit. Hence converting to 8 bit automatically.")
                im = (im/im.max()*255).astype(np.uint8)
                assert im.max()>250 , "Problem in converting to 8bit"
                
            #im = cv2.cv.LoadImage(imName,cv2.cv.CV_LOAD_GRAYSCALE)
            #self.axes.imshow(im)
            self.AppData["patternImageData"]=im
            #print(im.shape)        
            self.AppData["patternBounds"] = [0,0,im.shape[1],im.shape[0]]
            self.AppData["patternName"]=os.path.basename(imName)
            self.updatePlot()
            ax = self.mainPlot.axes
            if spotPositions is None:
                initialPoints=[self.AppData["Origin"],self.AppData["Spot1"],self.AppData["Spot2"] ]
                dp = DraggablePoints(initialPoints=None, imageData=self.AppData["patternImageData"],axisHandle=None,showPlot=True)
                spotList = dp.returnSpotPositions()
            else:
                spotList = spotPositions
                
            spotList = np.array(spotList)
            self.AppData["Origin"]=spotList[0]
            self.AppData["Spot1"]=spotList[1]
            self.AppData["Spot2"]=spotList[2]
            logging.info("Updated the spot positions as per requirement : they are :"+str(spotList))
            self.updatePlot()
            self.updateMeasuredValues()
            logging.info("Updfated the spot measure values please check if they make sense!!!")
            #self.mainPlot.update_figure()
            self.AppData["workingDirectory"]=os.path.dirname(imName)
            autoSpotPositions=False
        
        if autoSpotPositions:
            centre=np.array([int(im.shape[0]/2),int(im.shape[1]/2)])
            length=int(centre.min()*.05)
            self.AppData["Origin"] =   [centre[0],centre[1]]
            self.AppData["Spot1"]= [centre[0]+length,centre[1]]
            self.AppData["Spot2"]= [centre[0],centre[1]+length]
            
            
        self.spotsOriginEditX.setText("{:.2f}".format(self.AppData["Origin"][0]))
        self.spotsOriginEditY.setText("{:.2f}".format(self.AppData["Origin"][1]))
        
        self.spotsSpot1EditX.setText("{:.2f}".format(self.AppData["Spot1"][0]))
        self.spotsSpot1EditY.setText("{:.2f}".format(self.AppData["Spot1"][1]))
        
        self.spotsSpot2EditX.setText("{:.2f}".format(self.AppData["Spot2"][0]))
        self.spotsSpot2EditY.setText("{:.2f}".format(self.AppData["Spot2"][1]))

        self.solveButton.show()    
        
   
    def updateOriginX(self):
        self.AppData["Origin"][0]=float(self.spotsOriginEditX.text())
        logging.info("the origin is updated to be : "+str(self.AppData["Origin"]))
        self.updateMeasuredValues()
        self.updatePlot()        
    def updateOriginY(self):
        self.AppData["Origin"][1]=float(self.spotsOriginEditY.text())
        logging.info("the origin is updated to be : "+str(self.AppData["Origin"]))
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot1X(self):
        self.AppData["Spot1"][0]=float(self.spotsSpot1EditX.text())
        logging.info("the spot1 is updated to be : "+str(self.AppData["Spot1"]))
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot1Y(self):
        self.AppData["Spot1"][1]=float(self.spotsSpot1EditY.text())
        logging.info("the spot1 is updated to be : "+str(self.AppData["Spot1"]))
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot2X(self):
        self.AppData["Spot2"][0]=float(self.spotsSpot2EditX.text())
        logging.info("the spot2 is updated to be : "+str(self.AppData["Spot2"]))
        self.updateMeasuredValues()
        self.updatePlot()
    def updateSpot2Y(self):
        self.AppData["Spot2"][1]=float(self.spotsSpot2EditY.text())
        logging.info("the spot2 is updated to be : "+str(self.AppData["Spot2"]))
        self.updateMeasuredValues()
        self.updatePlot()

    def updatePlot(self):
        origin = self.AppData["Origin"]
        vec1 = self.AppData["Spot1"]
        vec2 = self.AppData["Spot2"]
        spcialPointsForMarking=[origin,vec1,vec2]
        latticeBounds=self.AppData["patternBounds"]
        latticeBounds = [latticeBounds[0],latticeBounds[2],latticeBounds[1],latticeBounds[3]]      
        latticePoints = gu.generate2Dlattice(origin, vec1, vec2, maxIndices=5,latticeBounds=latticeBounds, plotOn=False)
        self.AppData["latticePoints"]=latticePoints      
        data = self.AppData["patternImageData"]         
        self.mainPlot.updateFigureData(imData=data, imageName=self.AppData["patternName"], spcialPointsForMarking=spcialPointsForMarking, latticePoints=latticePoints)
        self.mainPlot.update_figure(brightness=self.AppData['solutionPlotOptions']["brightness"],contrast=self.AppData['solutionPlotOptions']["contrast"],
                                    histogramEqualize=self.AppData['solutionPlotOptions']["histogramEqualize"])

    

    def startSolver(self):
        #self.p.show()
        self.solverStatusMessage.setText("Solver is working ! Wait few seconds !!!")
        self.solvePattern()
    
    def generateDTable(self):
        
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]        
        planeTable = sa.generateDspacingTable()
        if len(planeTable)>1:
            self.dTableHandle.setRowCount(len(planeTable))
            for i,item in enumerate(planeTable,start=0):
                if i==0:
                    continue                
                self.dTableHandle.setItem(i-1,0, QTableWidgetItem(str(item["plane"])))
                self.dTableHandle.setItem(i-1,1, QTableWidgetItem(str(item["dSpacing"])))
                self.dTableHandle.setItem(i-1,2, QTableWidgetItem(str(np.around((10./item["dSpacing"]),2))))                
                self.dTableHandle.setItem(i-1,3, QTableWidgetItem(str(item["intensity"])))
    
    def solvePattern(self):
        self._currentSolution=None
        expSpotData={"spotXyData":[self.AppData["Origin"],self.AppData["Spot1"], self.AppData["Spot2"]]} 
        hklMax=self.AppData["solverOptions"]["maxHkl"]
        D_TOLERANCE=self.AppData["solverOptions"]["allowedDError"]
        allowedAngleDeviation=self.AppData["solverOptions"]["allowedAngleError"]
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]
        lattice=crystal["lattice"]
        logging.info("Started the solver !!!!")
        self.generateDTable()
        calibration= {"cameraConstant":self.AppData["solverOptions"]["cameraConstant"]}
        logging.info("The current D tolerance is : {:.1f}".format(D_TOLERANCE))
        result=sa.solvePatternFrom3PointsCalibrated(expSpotData, hklMax=6,D_TOLERANCE=D_TOLERANCE, allowedAngleDeviation=allowedAngleDeviation,
                                                      calibration=calibration,imageData=self.AppData["patternImageData"],holderData=self.AppData["measurementConditions"])
        
        self.solverStatusMessage.setText("Solver is Done !!!")
        
        if result is not None:
            print("Congratulations Some solutions were found!!!!")
            for item in result:
                print(item["pattern"]["zoneAxis"], item["angleError"], item["dError"])
            self.AppData["SAEDSolutuions"]=result
            self.solutionsTableHandle.setRowCount(len(result))
            self.clearSolutionsButton.setEnabled(True)
            for i,item in enumerate(result):
                self.solutionsTableHandle.setItem(i,0, QTableWidgetItem(str(i+1)))
                self.solutionsTableHandle.setItem(i,1, QTableWidgetItem(self.AppData["currentCrystalName"]))
                zoneAxisName = "{:int}".format(item["pattern"]["zoneAxis"])
                self.solutionsTableHandle.setItem(i,2, QTableWidgetItem(zoneAxisName[5:]))
                self.solutionsTableHandle.setItem(i,3, QTableWidgetItem(str(item["spot1"])))
                self.solutionsTableHandle.setItem(i,4, QTableWidgetItem(str(item["spot2"])))                
                self.solutionsTableHandle.setItem(i,5, QTableWidgetItem(str(item["angleError"])))
                self.solutionsTableHandle.setItem(i,6, QTableWidgetItem(str(item["dError"])))
                self.solutionsTableHandle.setItem(i,7, QTableWidgetItem(str(np.around(item["Correlation"],3))))
            
            self.solutionsTableHandle.show()
            self._currentSolution=result[0]
            self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
    
            self.acceptSolutiuoiForSimulatorButton.setEnabled(True)
            logging.info("A total of {:2d} solutions were found. Hope one of them works for you".format(len(result)))
        else:
            print("No solutions found !! try changing the solutiuon parameters!!!")
            logging.info("No solutions were found !! Please try changing the paramters of the solver!!!")
            QtGui.QMessageBox.information(self,"Warning!!!", "No solutions have been found!! Consider Changing the solver parameters and Try Again ")
 
        
   
    
    def clearSolutions(self):
        self.solutionsTableHandle.setRowCount(0)
    
    def cell_was_clicked(self, row, column):
        print("Row %d and Column %d was clicked" % (row, column))
        #solution=self.AppData["SAEDSolutuions"][row]
        self._currentSolution=self.AppData["SAEDSolutuions"][row]
        crystal = self.AppData["currentCrystal"]
        sa = crystal["saedAnalyzer"]
        imageName=self.AppData["patternName"]
        
        self.mainPlot.plotSolution(saedSolution=self._currentSolution, saedAnalyzer=sa, title=None,
                                    markSpots=self.AppData['solutionPlotOptions']['markSpots'],showAbsentSpots=self.AppData['solutionPlotOptions']['hideSystamticAbsences'],
                                    plotShow=False,plotKikuchi=self.AppData['solutionPlotOptions']['plotKikuchi'],markKikuchi=self.AppData['solutionPlotOptions']['markKikuchi'])
        self.acceptSolutiuoiForSimulatorButton.setEnabled(True)
    
    def cameraConstChanged(self):
        txt =self.solverOptionsCameraConstantHandle.text() 
        cameraConstant=float(txt)
        self.AppData["solverOptions"]["cameraConstant"]=cameraConstant
        self.updateMeasuredValues()
        self.solveButton.setEnabled(True)    
 
    
    
    
    def selectFile(self):
        imDir=self.AppData["filePathData"]["lastUsedDirForPattern"]['path']
        if not os.path.isdir(imDir):
               defaultFilePath =self.AppData["filePathData"]["lastUsedDirForPattern"]['default']
        else:
            defaultFilePath=imDir
        
        imName = QtGui.QFileDialog.getOpenFileName(self, 'Chose the Diffraction Pattern', 
        defaultFilePath,"Image files (*.jpg *.tif *.png *.dm3)")
        self.AppData["imageFileName"]=imName
        self.loadSelectedFile(imName)
        lastUsedPath=os.path.dirname(imName) 
        self.AppData["filePathData"]["lastUsedDirForPattern"]["path"]=lastUsedPath
        self.savePathInfo()
            
    
    def loadConfiguration(self, fileName=None,updatePlot=True):
        """
        setting updatePlot=False ensures no display of the pattern. Useful for the initialization of the GUI where we dont want to show default pattern
        """
        if fileName is None:
            defaultFilePath=self.AppData["filePathData"]["lastUsedDirForConfigFile"]['path']
            if not os.path.isfile(defaultFilePath):
               defaultFilePath =self.AppData["filePathData"]["lastUsedDirForConfigFile"]['default']
            jsonName = QtGui.QFileDialog.getOpenFileName(self, 'Chose the json File ', 
                            defaultFilePath,"Json files (*.json )", )
            self.AppData["jsonFileName"]=jsonName
            self.AppData["filePathData"]["lastUsedDirForConfigFile"]['path']=jsonName       
        else:
            jsonName=fileName
        with open(jsonName, 'r') as f:
            tmpData = json.load(f)
        for key in tmpData:
            self.AppData[key]=tmpData[key]
            
        logging.info("Just read the JSON file : {:s} successfully".format(jsonName) )
        self.savePathInfo()
        if updatePlot:
            spotPositions = [self.AppData["Origin"], self.AppData["Spot1"], self.AppData["Spot2"]]
            self.loadSelectedFile(self.AppData["imageFileName"],spotPositions=spotPositions , autoSpotPositions=True) 
        
        self.setMeasurementConditions()
        self.spotsOriginEditX.setText("{:.2f}".format(self.AppData["Origin"][0]))
        self.spotsOriginEditY.setText("{:.2f}".format(self.AppData["Origin"][1]))        
        self.spotsSpot1EditX.setText("{:.2f}".format(self.AppData["Spot1"][0]))
        self.spotsSpot1EditY.setText("{:.2f}".format(self.AppData["Spot1"][1]))    
        self.spotsSpot2EditX.setText("{:.2f}".format(self.AppData["Spot2"][0]))      
        self.spotsSpot2EditY.setText("{:.2f}".format(self.AppData["Spot2"][1]))
        self.solverOptionsMaxHklHnadle.setText(str(self.AppData["solverOptions"]["maxHkl"]))
        self.solverOptionsAllowedDeltaAngleHandle.setText(str(self.AppData["solverOptions"]["allowedAngleError"]))
        self.solverOptionsAllowedDeltaD_RatioHandle.setText(str(self.AppData["solverOptions"]["allowedDError"]))
        self.solverOptionsCameraConstantHandle.setText(str(self.AppData["solverOptions"]["cameraConstant"]))
        
        if self.AppData["currentCrystal"] is None:
            self.loadCifs()
            
        if updatePlot:
            self.mainPlot.update_figure()
        self.updateMeasuredValues()
        self.solveButton.setEnabled(True)
   
    
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
        

app = QtGui.QApplication(sys.argv)
frame = MainWindow()
frame.show()
sys.exit(app.exec_())  