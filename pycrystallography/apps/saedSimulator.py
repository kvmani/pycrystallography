import os,sys
import numpy as np
os.environ['QT_API'] = 'PyQt5'
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
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
import copy
import itertools
import time
import matplotlib.pyplot as plt
from pylatexenc.latex2text import LatexNodes2Text
from operator import itemgetter
sp = QSizePolicy()


class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QtGui.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)    

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)    
class SaedSimulatorWindow(QtGui.QWidget):
    def __init__(self,currentSaed=None, currentSaedAnalyzer=None):
        QtGui.QWidget.__init__(self)
        self.setGeometry(0,0, 800,1020)
        self.setWindowTitle("saedSimulator")
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.center()
        open = QtGui.QAction("Exit", self)
        save = QtGui.QAction("Save", self)
        build = QtGui.QAction("Build", self)
        exit = QtGui.QAction("Quit", self)
        menu_bar = QtGui.QMenuBar()
        file = menu_bar.addMenu("&File")
        help = menu_bar.addMenu("&Help")
        self.AppData={}
        self.AppData["intiatedFromSolver"]=False
        self.AppData['alphaTilt']=0.
        self.AppData['betaTilt']=0.
        self.AppData['InitialAlphaTilt']=0.
        self.AppData['InitialBetaTilt']=0.
        self.AppData['InitialinPlaneRotation']=0.
        self.AppData['inPlaneRotation']=0.
        self.AppData["tiltStepSize"]=10.0
        
        self.AppData['solutionPlotOptions']={'markSpots':False, 'hideSystamticAbsences':True, 
                                             'plotKikuchi':False, 'markKikuchi':False }
        self.AppData['sterioPlotOptions']={'markSpots':False,  }
        
        self.AppData["TEMHolderLimits"]={'alpha':[-45,45], 'beta':[-45,45],}
        self.AppData["animationOptions"]={"timeForFrame":0.01, 'tiltStepSize':5}
        self.AppData["patternModified"]=False
        self.AppData["maxHkl"]=3
        size=[50,50]
        barcPic  = QtGui.QPushButton("")
        #barcPic.setGeometry(0,0,5,5)
        pic = QtGui.QIcon(r'../../data/programeData/barcLogo.png')
        barcPic.resize(size[0]+6, size[1]+6)
        barcPic.setIcon(pic)
        barcPic.setIconSize(QSize(size[0],size[1]))
        barcPic.setSizePolicy(sp)
        file.addAction(open)
        file.addAction(save)
        file.addAction(build)
        file.addAction(exit)
        
        self.plotAllSymmetricDirs=False
        
        self.mainBox =QtGui.QGroupBox("Initial Conditions") 
        self.loadCrystalButton=QtGui.QPushButton("loadCrystal")
        crystalNameLabel=QtGui.QLabel("Crystal")
        self.crystalNameLineEdit=QtGui.QLineEdit("Si")
        self.crystalNameLineEdit.setReadOnly(True)
        self.alphaTilt=QtGui.QLineEdit("0")
        self.betaTilt=QtGui.QLineEdit("0")
        self.inPlaneRotation=QtGui.QLineEdit("0")
        alphaTiltLabel=QtGui.QLabel('Initial AlphaTilt')
        betaTiltLabel=QtGui.QLabel('Initial BetaTilt')
        inPlaneRotationLabel=QtGui.QLabel('Initial inPlaneRotation')
        self.radioButtonSterioBasedOnZoneAxis = QRadioButton("BasedOnZoneAxis")
        self.radioButtonSterioBasedOnCrystalOri = QRadioButton("BasedOnCrystalOri")
        self.radioButtonSterioBasedOnCrystalOri.setChecked(True)
        self.startingZoneAxisLabel=QtGui.QLabel('Initial Zone Axis')
        self.startingZoneAxis=QtGui.QLineEdit("0 0 0 1")
        self.crystalOriLabel=QtGui.QLabel('Crystal Orientation (Euler Angles)')
        self.crystalOri=QtGui.QLineEdit("0 0 0 ")
        targetZoneAxisLabel=QtGui.QLabel('Target Zone Axis')
        self.targetZoneAxis=QtGui.QLineEdit("1 1 0")
        self.errorInAchivingTargetZoneAxis=QtGui.QLineEdit("")
        errorInAchivingTargetZoneAxisLabel=QtGui.QLabel("Error in Achiving Target Zone")
       
        self.currentTiltsBox =  QtGui.QGroupBox("Current Holder Tilts")
        self.currentAlphaTiltHandle=QtGui.QLineEdit("0")
        self.currentAlphaTiltHandle.setReadOnly(True)
        self.currentBetaTiltHandle=QtGui.QLineEdit("0")
        self.currentBetaTiltHandle.setReadOnly(True)
        currentAlphaTilLabel=QtGui.QLabel('Current AlphaTilt')
        currentBetaTilLabel=QtGui.QLabel('Current BetaTilt')
        self.currentTiltsBoxLayout = QtGui.QHBoxLayout()
        self.currentTiltsBoxLayout.addWidget(currentAlphaTilLabel)
        self.currentTiltsBoxLayout.addWidget(self.currentAlphaTiltHandle)
        self.currentTiltsBoxLayout.addWidget(currentBetaTilLabel)
        self.currentTiltsBoxLayout.addWidget(self.currentBetaTiltHandle)
        self.currentTiltsBox.setLayout(self.currentTiltsBoxLayout)
        
        self.holderTiltLimitsBox =  QtGui.QGroupBox("Holder Tilt Limits")
        self.alphaTiltLimitHandle=QtGui.QLineEdit("45")
        self.betaTiltLimitHandle=QtGui.QLineEdit("45")
        alphaTiltLimitLabel=QtGui.QLabel(u'\u03b1 TiltLimit \u00B1 (\xb0))')
        betaTiltLimitLabel=QtGui.QLabel(u'\u03b2 TiltLimit \u00B1 (\xb0))')
        self.holderTiltLimitsLayout = QtGui.QHBoxLayout()
        self.holderTiltLimitsLayout.addWidget(alphaTiltLimitLabel)
        self.holderTiltLimitsLayout.addWidget(self.alphaTiltLimitHandle)
        self.holderTiltLimitsLayout.addWidget(betaTiltLimitLabel)
        self.holderTiltLimitsLayout.addWidget(self.betaTiltLimitHandle)
        self.holderTiltLimitsBox.setLayout(self.holderTiltLimitsLayout)
 
        
        
        
        
        self.ApplyTiltsBox =  QtGui.QGroupBox("Tilts")
        self.ApplyPositiveAlphaTilt=QtGui.QPushButton(u" +\u03b1")
        self.ApplyNegativeAlphaTilt=QtGui.QPushButton("- \u03b1")
        self.ApplyPositiveBetaTilt=QtGui.QPushButton(u" +\u03b2")
        self.ApplyNegativeBetaTilt=QtGui.QPushButton(u" -\u03b2")
        self.setTiltStpeSize=QtGui.QLineEdit("10")
        self.gotoAlphaTilt=QtGui.QLineEdit("0")
        gotoAlphaTiltLabel=QtGui.QLabel("Goto AlphaTilt")
        self.gotoBetaTilt=QtGui.QLineEdit("0")
        gotoBetaTiltLabel=QtGui.QLabel("Goto BetaTilt")
        self.gotoSpecificAlphaBetaTilts=QtGui.QPushButton("Go")

        self.ApplyTiltsBoxLayout = QtGui.QGridLayout()
        self.ApplyTiltsBoxLayout.addWidget(self.ApplyPositiveAlphaTilt,0,1,1,1)
        self.ApplyTiltsBoxLayout.addWidget(self.ApplyNegativeAlphaTilt, 2,1,1,1)
        self.ApplyTiltsBoxLayout.addWidget(self.ApplyPositiveBetaTilt, 1,0,1,1)
        self.ApplyTiltsBoxLayout.addWidget(self.ApplyNegativeBetaTilt, 1,2,1,1)
        self.ApplyTiltsBoxLayout.addWidget(self.setTiltStpeSize, 1,1,1,1,)
        self.ApplyTiltsBoxLayout.addWidget(gotoAlphaTiltLabel, 3,0,1,1,)
        self.ApplyTiltsBoxLayout.addWidget(self.gotoAlphaTilt, 3,1,1,1,)
        self.ApplyTiltsBoxLayout.addWidget(gotoBetaTiltLabel, 4,0,1,1,)
        self.ApplyTiltsBoxLayout.addWidget(self.gotoBetaTilt, 4,1,1,1,)
        self.ApplyTiltsBoxLayout.addWidget(self.gotoSpecificAlphaBetaTilts,3,2,1,1)
        self.ApplyTiltsBox.setLayout(self.ApplyTiltsBoxLayout)

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
        plotOptionsLayout.addWidget(self.plotOptionsMarkSpotsHnadle)
        plotOptionsLayout.addWidget(self.plotOptionsPlotKikuchiHandle)
        plotOptionsLayout.addWidget(self.plotOptionsMarkKikuchi)
        plotOptionsLayout.addWidget(self.plotOptionsShowSystematicAbsncentSpots)
        self.plotOptionsBox.setLayout(plotOptionsLayout)
        
        self.animateToTargetZoneBox=QtGui.QGroupBox("Animate To Target Zone Options")
        animateToTargetZoneLayout = QtGui.QHBoxLayout()
        self.timeForFrame=QtGui.QLineEdit("1")
        self.stepSizeForTilts=QtGui.QLineEdit("5")
        steSizeForTiltsLabel=QtGui.QLabel("stepSize For Tilts")
        
        self.holderTiltsForDesiredZoneDisplayBox=QtGui.QGroupBox("Holder tilts needed to get to the desired Zone")
        holderTiltsForDesiredZoneDisplayLayout = QtGui.QHBoxLayout()
        finalAlphaTiltNeededLabel = QtGui.QLabel(u" \u03b1 Tilt (\xb0))")
        finalBetaTiltNeededLabel = QtGui.QLabel(u" \u03b2 Tilt (\xb0))")
        self.finalAlphaTiltNeeded=QtGui.QLineEdit("0")
        newFont = QtGui.QFont("Times", 16, QtGui.QFont.Bold)
        self.finalAlphaTiltNeeded.setReadOnly(True)
        self.finalAlphaTiltNeeded.setFont(newFont)
        self.finalBetaTiltNeeded=QtGui.QLineEdit("0")
        self.finalBetaTiltNeeded.setReadOnly(True)
        self.finalBetaTiltNeeded.setFont(newFont)
        holderTiltsForDesiredZoneDisplayLayout.addWidget(finalAlphaTiltNeededLabel)
        holderTiltsForDesiredZoneDisplayLayout.addWidget(self.finalAlphaTiltNeeded)
        holderTiltsForDesiredZoneDisplayLayout.addWidget(finalBetaTiltNeededLabel)
        holderTiltsForDesiredZoneDisplayLayout.addWidget(self.finalBetaTiltNeeded)
        
        
        self.holderTiltsForDesiredZoneDisplayBox.setLayout(holderTiltsForDesiredZoneDisplayLayout)
        
        
        self.goToTargetZoneOptionsBox=QtGui.QGroupBox("Options for going to target Zone")
        goToTargetZoneOptionsLayout = QtGui.QHBoxLayout()
        self.angleBetweenCurrentAndTargetZone=QtGui.QLineEdit("to be computed")
        self.angleBetweenCurrentAndTargetZone.setReadOnly(True)
        angleBetweenCurrentAndTargetZoneLabel=QtGui.QLabel("Angle btwn Current and Target Zone")
        
        self.allowAnySymmetricEquivalentCheckBox=QtGui.QCheckBox()
        self.allowAnySymmetricEquivalentCheckBox.setText("Allow going to Sym Equi Zone Axis")
        self.allowAnySymmetricEquivalentCheckBox.setChecked(True)
        
        goToTargetZoneOptionsLayout.addWidget(angleBetweenCurrentAndTargetZoneLabel)
        goToTargetZoneOptionsLayout.addWidget(self.angleBetweenCurrentAndTargetZone)
        goToTargetZoneOptionsLayout.addWidget(self.allowAnySymmetricEquivalentCheckBox)
        self.goToTargetZoneOptionsBox.setLayout(goToTargetZoneOptionsLayout)
        
        currentZoneAxisLabel=QtGui.QLabel('Current Zone Axis')
        newFont = QtGui.QFont("Times", 14, QtGui.QFont.Bold)
#         font = QFont()
#         font.setBold(True)
#         font.setWeight(24) 
        currentZoneAxisLabel.setFont(newFont)
        self.currentZoneAxisHandle=QtGui.QLineEdit("0  0 1")
        self.currentZoneAxisHandle.setFont(newFont)
        self.currentZoneAxisHandle.setReadOnly(True)

        timeForFrameLabel = QtGui.QLabel("timeForFrame(sec)")
        self.animateButton=QtGui.QPushButton("AnimateNow")
        animateToTargetZoneLayout.addWidget(timeForFrameLabel)
        animateToTargetZoneLayout.addWidget(self.timeForFrame)
        
        animateToTargetZoneLayout.addWidget(steSizeForTiltsLabel)
        animateToTargetZoneLayout.addWidget(self.stepSizeForTilts)
        animateToTargetZoneLayout.addWidget(self.animateButton)

        self.animateToTargetZoneBox.setLayout(animateToTargetZoneLayout)
        self.logWindow = QtGui.QTextBrowser()
        
        self.alphaBetaTable = QtGui.QTableWidget()
        self.alphaBetaTable.setRowCount(1)
        self.alphaBetaTable.setColumnCount(6)
        self.alphaBetaTable.setHorizontalHeaderLabels([ "    ZoneAxis    " , u"        \u03b1 (\u00B0)     ", u"       \u03b2 (\u00B0)      ", "isNorthPole",    "AngluarDistance", "Remarks    "])
        self.alphaBetaTable.resizeColumnsToContents()
        self.alphaBetaTable.resizeRowsToContents()
        
        self.plotAreaHandle=QtGui.QGraphicsView()
        self.mainPlot = MyStaticMplCanvas(self.plotAreaHandle, width=8, height=8, dpi=100)  
        self.mainPlot.compute_initial_figure()
        plotLayout = QtGui.QVBoxLayout(self.plotAreaHandle)
        plotLayout.addWidget(NavigationToolbar(self.mainPlot, self))
        plotLayout.addWidget(self.mainPlot)
        
        self.plotAreaBox =QtGui.QGroupBox("Plot Area") 
        logTextBox = QPlainTextEditLogger(self.logWindow)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)
        self.plotAreaBox.setLayout(plotLayout)
        
        
        self.zoneAxesListForPlottingHandle=QtGui.QLineEdit("0 0 1")
        zoneAxesListForPlottingLabel = QtGui.QLabel("List of directions for Sterio Plot")
        self.plotAllSymmetricSetCheckBox = QtGui.QCheckBox("Plot ALl Symmetric Set")
        self.plotAllSymmetricSetCheckBox.setChecked(False)
        
        self.sterioPlotAreaHandle=QtGui.QGraphicsView()
        self.sterioPlot = MyStaticMplCanvas(self.plotAreaHandle, width=8, height=8, dpi=100)  
        self.sterioPlot.compute_initial_figure()
        sterioPlotLayout = QtGui.QGridLayout(self.sterioPlotAreaHandle)
        sterioPlotLayout.addWidget(NavigationToolbar(self.sterioPlot, self),0,0,1,6)
        sterioPlotLayout.addWidget(self.sterioPlot,1,0,10,6)
        sterioPlotLayout.addWidget(zoneAxesListForPlottingLabel,11,0,1,1)
        sterioPlotLayout.addWidget(self.zoneAxesListForPlottingHandle,11,1,1,4)
        sterioPlotLayout.addWidget(self.plotAllSymmetricSetCheckBox,11,4,1,1)
        
        
        self.sterioPlotAreaBox =QtGui.QGroupBox("Sterio Plot Area") 
        self.sterioPlotAreaBox.setLayout(sterioPlotLayout)        
        self.mainLayout = QtGui.QGridLayout()
        self.mainLayout.addWidget(self.plotAreaBox,4,2,10,8)
        self.mainLayout.addWidget(self.ApplyTiltsBox,6,0,2,2)
        self.mainLayout.addWidget(self.sterioPlotAreaBox, 4,13,10,7)
        self.mainLayout.addWidget(self.currentTiltsBox, 0,17,1,3)
        self.mainLayout.addWidget(self.holderTiltLimitsBox, 1,17,1,3)
        self.mainLayout.addWidget(currentZoneAxisLabel, 4,10,1,2)
        self.mainLayout.addWidget(self.currentZoneAxisHandle, 5,10,1,2)
        self.mainLayout.addWidget(self.alphaBetaTable,6,10,7,3)
        self.mainLayout.addWidget(self.loadCrystalButton,0,0,1,1)
        self.mainLayout.addWidget(crystalNameLabel,0,1,1,1)
        self.mainLayout.addWidget(self.crystalNameLineEdit,0,2,1,1)
        self.mainLayout.addWidget(alphaTiltLabel,0,3,1,1)
        self.mainLayout.addWidget(self.alphaTilt,0,4,1,1)
        self.mainLayout.addWidget(betaTiltLabel,0,5,1,1)
        self.mainLayout.addWidget(self.betaTilt,0,6,1,1)
        self.mainLayout.addWidget(inPlaneRotationLabel,0,7,1,1)
        self.mainLayout.addWidget(self.inPlaneRotation,0,8,1,1)
        self.mainLayout.addWidget(self.radioButtonSterioBasedOnCrystalOri,0,9,1,1)
        self.mainLayout.addWidget(self.radioButtonSterioBasedOnZoneAxis,0,10,1,1)
        self.mainLayout.addWidget(self.crystalOriLabel,0,11,1,1)
        self.mainLayout.addWidget(self.crystalOri,0,12,1,1)

        self.mainLayout.addWidget(self.startingZoneAxisLabel,0,11,1,1)
        self.mainLayout.addWidget(self.startingZoneAxis,0,12,1,1)
        
        self.mainLayout.addWidget(targetZoneAxisLabel,0,13,1,1)
        self.mainLayout.addWidget(self.targetZoneAxis,0,14,1,1)
        self.mainLayout.addWidget(errorInAchivingTargetZoneAxisLabel,0,15,1,1)
        self.mainLayout.addWidget(self.errorInAchivingTargetZoneAxis,0,16,1,1)
        self.mainLayout.addWidget(self.plotOptionsBox,1,0,1,4)
        self.mainLayout.addWidget(self.goToTargetZoneOptionsBox,1,4,1,4)
        self.mainLayout.addWidget(self.animateToTargetZoneBox,1,8,1,4)
        self.mainLayout.addWidget(self.holderTiltsForDesiredZoneDisplayBox,1,13,1,3)
        self.mainLayout.addWidget(logTextBox.widget,14,0,3,18)
        logging.info("Just started. And here is the messaging window..............")
        self.mainBox.setLayout(self.mainLayout)
        
        menuAndIcon = QtGui.QGroupBox()
        menuAndIconLayout = QtGui.QHBoxLayout()
        menuAndIconLayout.addWidget(menu_bar)
        menuAndIconLayout.addWidget(barcPic)
        menuAndIcon.setLayout(menuAndIconLayout)
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(menuAndIcon,5)
        vbox.addWidget(self.mainBox,95)
        

        self.setWindowIcon(QtGui.QIcon(r'..\..\data\programeData\mainIcon.png'))
        self.setLayout(vbox)
        if currentSaedAnalyzer is None:
            self.loadCrystal(cifName='../../data/structureData/Si.cif')
            self._startingZoneAxis = MillerDirection(lattice=self.sa.lattice,vector=[0,0,1],MillerConv="Miller")
        else:
            self.sa = currentSaedAnalyzer
            
        self._updateGuiBasedOnsterioZoneAxisBasedOrOrientationBased() 
        
        if currentSaed is not None:
            print(currentSaed)
            self._startingZoneAxis = currentSaed["zoneAxis"]
            self._targetZoneAxis = copy.deepcopy(self._startingZoneAxis)
            txt ="{:int}".format(self._targetZoneAxis)
            txt = txt[5:]
            self.targetZoneAxis.setText(txt)
            #self.AppData['InitialinPlaneRotation'] = currentSaed["InPlaneRotation"]
            logging.info("Setting the pattern in the simulator to the chosen solution from solver")
            logging.info("Taken the pattern rotation to be :{:.2f}".format(self.AppData['InitialinPlaneRotation']))
            #self.inPlaneRotation.setText()
            self.saedData1=copy.deepcopy(currentSaed)
            txt ="{:int}".format(self._startingZoneAxis)
            txt = txt[5:]
            self.startingZoneAxis.setText(txt)
            self.AppData["intiatedFromSolver"]=True
        
        self.lattice=self.sa.lattice
        self.intializePlots()
        self.alphaTilt.editingFinished.connect(self.holderTiltValuesChanged)
        self.betaTilt.editingFinished.connect(self.holderTiltValuesChanged)
        self.inPlaneRotation.editingFinished.connect(self.holderTiltValuesChanged)
        self.animateButton.clicked.connect(lambda : self.animateToTargetZoneAxis())
        self.plotOptionsMarkKikuchi.toggled.connect(lambda:self.saedPlotOptionChanged())
        self.plotOptionsPlotKikuchiHandle.toggled.connect(lambda:self.saedPlotOptionChanged())
        self.plotOptionsMarkSpotsHnadle.toggled.connect(lambda:self.saedPlotOptionChanged())
        self.plotOptionsShowSystematicAbsncentSpots.toggled.connect(lambda:self.saedPlotOptionChanged())
        self.startingZoneAxis.editingFinished.connect(self.constructInitialZoneAxis)
        self.ApplyPositiveAlphaTilt.clicked.connect(lambda : self.alphaPositiveTiltsRequested())
        self.ApplyPositiveBetaTilt.clicked.connect(lambda : self.betaPositiveTiltsRequested())
        self.ApplyNegativeAlphaTilt.clicked.connect(lambda : self.alphaNegativeTiltsRequested())
        self.ApplyNegativeBetaTilt.clicked.connect(lambda : self.betaNegativeTiltsRequested())
        self.setTiltStpeSize.editingFinished.connect(self.changeTiltStepSize)
        self.gotoSpecificAlphaBetaTilts.clicked.connect(self.gotoSpecificTiltPositions)
        self.zoneAxesListForPlottingHandle.editingFinished.connect(self.zoneAxesDisplyListUpdated)
        self.radioButtonSterioBasedOnZoneAxis.toggled.connect(self._updateGuiBasedOnsterioZoneAxisBasedOrOrientationBased)
        self.radioButtonSterioBasedOnCrystalOri.toggled.connect(self._updateGuiBasedOnsterioZoneAxisBasedOrOrientationBased)
        self.crystalOri.editingFinished.connect(self.constructInitialOrientation)
        self.plotAllSymmetricSetCheckBox.toggled.connect(lambda : self.plotAllSymmetricSetOptionChanged())
        self.targetZoneAxis.editingFinished.connect(self.constructTargetZoneAxis)
        self.loadCrystalButton.clicked.connect(self.loadNewCrystal)
        self.alphaTiltLimitHandle.editingFinished.connect(self.updateTiltLimits)
        self.betaTiltLimitHandle.editingFinished.connect(self.updateTiltLimits)
        
    def updateTiltLimits(self):
        holderLimits = self.AppData["TEMHolderLimits"]
        alphaLimits = abs(float(self.alphaTiltLimitHandle.text()))   
        holderLimits["alpha"] = [-alphaLimits, alphaLimits]
        betaLimits = abs(float(self.betaTiltLimitHandle.text()))
        holderLimits["beta"] = [-betaLimits, betaLimits]
        logging.info("Updated the holder tilt limits")
        
    
    
    def loadNewCrystal(self):
        self.loadCrystal()
        self.intializePlots()
    
    def plotAllSymmetricSetOptionChanged(self):
        logging.debug("The symmetric vectors plotting option is "+str(self.plotAllSymmetricSetCheckBox.isChecked()))
        self.plotAllSymmetricDirs = self.plotAllSymmetricSetCheckBox.isChecked()
        #self.getZoneAxesListForSterioPlot()
        self.intializePlots()
        logging.info("The symmetric vectors plotting option changed now it is "+str(self.plotAllSymmetricDirs))
    
    def zoneAxesDisplyListUpdated(self):
        self.getZoneAxesListForSterioPlot()
        self.modifyAndUpdatePattern(alphaTilt=0., betaTilt=0.)
    
    
    def getZoneAxesListForSterioPlot(self):
        logging.debug("The symmetric vectors plotting option is "+str(self.plotAllSymmetricSetCheckBox.isChecked()))
        txt = self.zoneAxesListForPlottingHandle.text()
        zoneList = txt.split(',')
        self.zoneListFordisplayInSterioPlot = []
        if len(zoneList)>0:
            for item in zoneList:
                tmp = self.__constructMillerDirFromTxt(text=item)
                if self.plotAllSymmetricDirs:
                    tmp = tmp.symmetricSet()    
                else:
                    tmp = [tmp]
                self.zoneListFordisplayInSterioPlot.extend(tmp)
            logging.info("Added the zoneLists to the steriogram as requested The number of dir is :"+str(len(self.zoneListFordisplayInSterioPlot)))
            
    
    
    def loadCrystal(self,cifName=None): 
        if cifName is None:
            cifName = QtGui.QFileDialog.getOpenFileName(self, 'Chose the cif File', 
             r'../../data/structureData',"cif file (*.cif )")
        try: 
            lattice = olt.fromCif(cifName)
            saedAnalyzer = SaedAnalyzer(lattice=lattice,hklMax=self.AppData["maxHkl"])
            saedAnalyzer.loadStructureFromCif(cifName)
            logging.info("Just created the SAED analyzer object for crystal  :")
            self.sa=saedAnalyzer
            baseName = os.path.basename(cifName)
            self.crystalNameLineEdit.setText(baseName[:-4])
        except:
            logging.error("Failed loading the crystal. Most likely the cif file is not correct : the cifFile is "+cifName)
            
    
    def gotoSpecificTiltPositions(self):
        gotoAlphaTilt = float(self.gotoAlphaTilt.text())
        gotoBetaTilt = float(self.gotoBetaTilt.text())
        additionalAlphaTilt = gotoAlphaTilt- float(self.currentAlphaTiltHandle.text())
        additionalBetaTilt = gotoBetaTilt-float(self.currentBetaTiltHandle.text())
        logging.info("Now going to the alpha beta position of : {:2f} {:.2f}. For ding this I am performing following additional delta tilts  for alpha and beta {:2f} {:.2f}".format(gotoAlphaTilt,gotoBetaTilt,
                                                                                                                                                                                       additionalAlphaTilt, additionalBetaTilt))
        self.modifyAndUpdatePattern(alphaTilt=additionalAlphaTilt, betaTilt=additionalBetaTilt)
         
    
    def changeTiltStepSize(self):
        self.AppData["tiltStepSize"] = np.abs(float(self.setTiltStpeSize.text()))
        logging.info("Tilt step Size is now updated to {:.2f}".format(self.AppData["tiltStepSize"]))
    
    def alphaPositiveTiltsRequested(self):
        currentHolderAlpha=float(self.currentAlphaTiltHandle.text())
        currentHolderAlpha=currentHolderAlpha+self.AppData["tiltStepSize"]
        #self.currentAlphaTiltHandle.setText("{:.2f}".format(currentHolderAlpha))
        logging.info("Tilting by alpha axis")
        self.modifyAndUpdatePattern(alphaTilt=self.AppData["tiltStepSize"],betaTilt=0.)
    
    def betaPositiveTiltsRequested(self):
        currentHolderBeta=float(self.currentBetaTiltHandle.text())
        currentHolderBeta=currentHolderBeta+self.AppData["tiltStepSize"]
        #self.currentBetaTiltHandle.setText("{:.2f}".format(currentHolderBeta))
        logging.info("Tilting by beta axis")
        self.modifyAndUpdatePattern(alphaTilt=0.,betaTilt=self.AppData["tiltStepSize"])
        
    def alphaNegativeTiltsRequested(self):
        currentHolderAlpha=float(self.currentAlphaTiltHandle.text())
        currentHolderAlpha=currentHolderAlpha-self.AppData["tiltStepSize"]
        #self.currentAlphaTiltHandle.setText("{:.2f}".format(currentHolderAlpha))
        logging.info("Tilting by alpha axis")
        self.modifyAndUpdatePattern(alphaTilt=-self.AppData["tiltStepSize"],betaTilt=0.)
    
    def betaNegativeTiltsRequested(self):
        currentHolderBeta=float(self.currentBetaTiltHandle.text())
        currentHolderBeta=currentHolderBeta-self.AppData["tiltStepSize"]
        #self.currentBetaTiltHandle.setText("{:.2f}".format(currentHolderBeta))
        logging.info("Tilting by beta axis")
        self.modifyAndUpdatePattern(alphaTilt=0.,betaTilt=-self.AppData["tiltStepSize"])
        
    def intializePlots(self):
        
        self.desiredDirectionList=[#MillerDirection(lattice=self.lattice,vector=[1,1,1]),
                             ]
        
        self.allDesiredDirList = []
        for item in self.desiredDirectionList:
            self.allDesiredDirList.extend(item.symmetricSet())
        
        if not self.AppData["intiatedFromSolver"]:
            if self.AppData["sterioBasedOnZoneAxis"]:
                
                self.saedData1 = self.sa.calcualteSAEDpatternForZoneAxis(zoneAxis=self._startingZoneAxis,
                                           patterCenter=[0.,0.], scalingFactor=100., inPlaneRotation=self.AppData['InitialinPlaneRotation'])
                logging.debug("Yes entered the portion of recalcualting the zone axis based on modifed zoneAxis and inPlane rotation values")
                self.crystalOrientation = self.saedData1["Ori1"]
            else:
                
                self.saedData1 = self.sa.calcualteSAEDpatternForCrystalOri(crystalOri=self.crystalOrientation,patterCenter=[0.,0.], scalingFactor=100.)
                self._startingZoneAxis = self.saedData1["zoneAxis"]
                logging.info("Computed the saed Pattern from Crystal Orientation")
                
        self._sterioCentredOn = copy.deepcopy(self._startingZoneAxis)        
        self.saedData1=self.sa.computeKikuchiForSaed(self.saedData1,ignoreDistantSpots=True,maxHkl=2)
        self._currentZoneAxis=self.saedData1["zoneAxis"]
        self._sterioXAxis = copy.deepcopy(self.saedData1["xAxis"])
        self._alphaAxis =copy.deepcopy(self.saedData1["xAxis"].getUnitVector())
        self._betaAxis = copy.deepcopy(self.saedData1["yAxis"].getUnitVector())

        self.dirSterioData = self.sa.calculateSterioGraphicProjectionDirection(dirList = None, maxUVW=1 , crystalOri=self.crystalOrientation,)
        self.specialSterioPoints=self.sa.calculateSterioGraphicProjectionDirection(dirList = self.allDesiredDirList,crystalOri=self.crystalOrientation)
        self.getZoneAxesListForSterioPlot()
        logging.debug("From initilization rotation = {:.1f} pattern  xAxis: {:int} sterio xAxis: {:int}".format(self.AppData['InitialinPlaneRotation'], self.saedData1["xAxis"], self._sterioXAxis))
        self.AppData["patternModified"]=False
        self.transformedSaed = copy.deepcopy(self.saedData1)
        #self.transformedSaed=self.sa.computeKikuchiForSaed(self.transformedSaed,ignoreDistantSpots=True,)
        self.mainPlot.updateFigureData()
        self.sterioPlot.updateFigureData()
        self.mainPlot.plotSaed(saedData=self.saedData1, saedAnalyzer=self.sa, markSpots=False,showAbsentSpots=False,plotKikuchi=False,markKikuchi=False,autoPatternBounds=True)
        #self.sterioPlot.plotSteriographicData(sterioData=[self.dirSterioData,self.specialSterioPoints],saedAnalyzer=self.sa,markPoints=[True,False])
        self.steriDataOfzoneListFordisplayInSterioPlot =self.sa.calculateSterioGraphicProjectionDirection(dirList=self.zoneListFordisplayInSterioPlot, crystalOri=self.crystalOrientation)
        self.sterioPlot.plotSteriographicData(sterioData=[self.dirSterioData,self.specialSterioPoints,self.steriDataOfzoneListFordisplayInSterioPlot ],saedAnalyzer=self.sa,markPoints=[True,False,True])
        
        self.currentZoneAxisHandle.setText(str(self.transformedSaed['zoneAxis']))
        self.AppData["alphaTilt"] = copy.deepcopy(self.AppData['InitialAlphaTilt'] ) 
        self.AppData["betaTilt"] = copy.deepcopy(self.AppData['InitialBetaTilt'])
        self.currentAlphaTiltHandle.setText("{:.2f}".format(self.AppData["alphaTilt"]))
        self.currentBetaTiltHandle.setText("{:.2f}".format(self.AppData["betaTilt"]))
        self.buildAlphaBetaTable()
  
    def buildAlphaBetaTable(self):
        data = self.steriDataOfzoneListFordisplayInSterioPlot
        self.alphaBetaTable.setRowCount(len(data["item"]))
        alphaBetaList=[]
        #data["distanceFromCentre"]=[0]*len(data["item"]) ## holds data of alpha beta distacnes for sorintg the stuff
        for i,item in enumerate(data["item"]):
            alpha,beta = data["alphaBeta"][i]
            isNorthPole = data["isNorthPole"][i]
            if isNorthPole:
                additionalAngle = 0
            else:
                additionalAngle  =180.
            alphaBetaList.append({"index":i, "dist":additionalAngle+np.sqrt(alpha*alpha+beta*beta)})
        alphaBetaList = sorted(alphaBetaList, key=itemgetter('dist', "index")) 
        TEMHolderLimits =  self.AppData["TEMHolderLimits"]
        alphaLimits = TEMHolderLimits["alpha"] 
        betaLimits = TEMHolderLimits["beta"] 
        font = QFont()
        font.setBold(True)  
        font.setWeight(12)
        
        for i, item in enumerate(alphaBetaList):
            index = item["index"]
            zoneAxis = data["string"][index]
            zoneAxis = LatexNodes2Text().latex_to_text(zoneAxis) ## converting to unicode text
            alpha,beta = data["alphaBeta"][index]
            isNorthPole = data["isNorthPole"][index]
            dist = np.around(alphaBetaList[i]["dist"],2)
            if alpha>alphaLimits[0] and  alpha<alphaLimits[1] and beta>betaLimits[0] and  beta<betaLimits[1] and isNorthPole:
                remarks="Can Reach"
                color = QtGui.QColor(0, 200,0) ## green
            else:
                remarks="Sorry !! . Can't reach"
                color = QtGui.QColor(255, 255,153)  ## red
            
            alpha = "{:.2f}".format(alpha)
            beta = "{:.2f}".format(beta)
            self.alphaBetaTable.setItem(i,0, QTableWidgetItem(zoneAxis))
            self.alphaBetaTable.setItem(i,1, QTableWidgetItem(alpha))
            self.alphaBetaTable.setItem(i,2, QTableWidgetItem(beta))
            self.alphaBetaTable.setItem(i,3, QTableWidgetItem(str(isNorthPole)))
            self.alphaBetaTable.setItem(i,4, QTableWidgetItem(str(dist)))
            self.alphaBetaTable.setItem(i,5, QTableWidgetItem(remarks))
            for j in range(self.alphaBetaTable.columnCount()):
                self.alphaBetaTable.item(i,j).setBackground(color)
                self.alphaBetaTable.item(i,j).setFont(font)
                
            
            
        self.alphaBetaTable.show() 
        logging.info("Updated the zone Axes alpha beta table")           
    
    def saedPlotOptionChanged(self):
        """
        """
        
        self.AppData['solutionPlotOptions']['plotKikuchi']=self.plotOptionsPlotKikuchiHandle.isChecked()
        self.AppData['solutionPlotOptions']['markKikuchi']=self.plotOptionsMarkKikuchi.isChecked()
        self.AppData['solutionPlotOptions']['markSpots']=self.plotOptionsMarkSpotsHnadle.isChecked()
        self.AppData['solutionPlotOptions']['hideSystamticAbsences']=self.plotOptionsShowSystematicAbsncentSpots.isChecked()
        self.modifyAndUpdatePattern()
        logging.info("Options changed and hence updating the plot.")
        
    
    def animateToTargetZoneAxis(self):
        
        self.constructTargetZoneAxis()
        
        currentAlphaTilt = self.AppData["alphaTilt"]
        currentBetaTilt = self.AppData["betaTilt"]
        neededAlphaTilt, neededBetaTilt, error = self.findTitlsForTargetZone()
        alphaArray = np.arange(currentAlphaTilt,neededAlphaTilt,self.AppData["animationOptions"]["tiltStepSize"]).tolist()
        
        angleBetweenCurrentAndTargetZoneAxes=np.sqrt(neededAlphaTilt*neededAlphaTilt+neededBetaTilt*neededBetaTilt)
        
        self.angleBetweenCurrentAndTargetZone.setText("{:.2f}".format(angleBetweenCurrentAndTargetZoneAxes))
        
        logging.info("The current Zone Axis is : {:int} and Targetr Zone Axis is : {:int} and angle to be traversed is {:.2f}".format(self._currentZoneAxis,
                    self._targetZoneAxis, angleBetweenCurrentAndTargetZoneAxes))
        if angleBetweenCurrentAndTargetZoneAxes<1:
            QtGui.QMessageBox.information(self, "You already are close to one of the similar zone Axes nothing much to do", "Press Ok")
            return None
            
        
        logging.info("Starting the animation Be ready!!!!")
        if len(alphaArray)<=1:
            alphaArray = np.arange(currentAlphaTilt,neededAlphaTilt,-self.AppData["animationOptions"]["tiltStepSize"]).tolist()
        
        alphaArray.append(neededAlphaTilt)
        
        betaArray = np.arange(currentBetaTilt,neededBetaTilt,self.AppData["animationOptions"]["tiltStepSize"]).tolist()
        
        if len(betaArray)<=1:
            betaArray = np.arange(currentBetaTilt,neededBetaTilt,-self.AppData["animationOptions"]["tiltStepSize"]).tolist()
        
        betaArray.append(neededBetaTilt)
        
        logging.info("Alpha Array and Beta Array Are : "+str(alphaArray)+" : "+str(betaArray))
        logging.info("Titls needed are : {:.2f} {:.2f}".format(neededAlphaTilt, neededBetaTilt))
        
        alphaRange = neededAlphaTilt-currentAlphaTilt
        betaRange = neededBetaTilt-currentBetaTilt
        nAlphaSteps = np.ceil(alphaRange/self.AppData["animationOptions"]["tiltStepSize"]).astype(np.int16)
        nBetaSteps = np.ceil(betaRange/self.AppData["animationOptions"]["tiltStepSize"]).astype(np.int16)
        alphaStepSize = alphaRange/10.
        betaStepSize = betaRange/10.
        nAlphaSteps=10
        nBetaSteps=10
        
        beta=betaArray[0]
        for i in np.arange(nAlphaSteps): 
            self.modifyAndUpdatePattern(alphaTilt=alphaStepSize, betaTilt=0.)
            #logging.info("Currently tilting over Alpha axis : alpha tilt and beta tilt are : {:.2f}  {:.2f}".format(alpha,beta))
            fig = plt.gcf()
            fig.set_size_inches(0.01,0.01)
            plt.pause(self.AppData["animationOptions"]["timeForFrame"])
            plt.close()
            self.modifyAndUpdatePattern(alphaTilt=0., betaTilt=betaStepSize)
            #logging.info("Currently tilting over beta axis : alpha tilt and beta tilt are : {:.2f}  {:.2f}".format(alpha,beta))
            fig = plt.gcf()
            fig.set_size_inches(0.01,0.01)
            plt.pause(self.AppData["animationOptions"]["timeForFrame"])
            plt.close()
        
        QtGui.QMessageBox.information(self, "Hope to have reached required zone axis by now", "Press Ok to continue")
        logging.info("Upadated the tilts and current Zone axes")
        return 1
    
    
        
    def _updateGuiBasedOnsterioZoneAxisBasedOrOrientationBased(self):
        
        if self.radioButtonSterioBasedOnZoneAxis.isChecked():
            self.crystalOriLabel.hide()
            self.crystalOri.hide()
            self.startingZoneAxisLabel.show()
            self.startingZoneAxis.show()
            self.AppData["sterioBasedOnZoneAxis"]=True
            self.constructInitialZoneAxis()
            logging.info("The option for sterio plot has been changed to use Zone Axis as  basis")
        else:
            self.crystalOriLabel.show()
            self.crystalOri.show()
            self.startingZoneAxisLabel.hide()
            self.startingZoneAxis.hide()
            self.AppData["sterioBasedOnZoneAxis"]=False
            self.constructInitialOrientation()
            logging.info("The option for sterio plot has been changed to use Crystal Orientation as basis")
    
    
    def constructInitialOrientation(self):
        text = self.crystalOri.text()
        vector = np.fromstring(text, dtype=float, sep=' ')
        if vector.size==3:
            self.crystalOrientation =Orientation(euler=np.radians(np.array(vector)))
            logging.info("Updated the crystal Orienation to be : {:.2f}".format(self.crystalOrientation))
            self.intializePlots()
        else:
            logging.error("Wrong number of values for euler angles. Please enter 3 numbers (seprated by space) and the vector is "+str(vector))
            raise ValueError ("Wrong number of values for euler angles. Please enter 3 numbers (seprated by space) and the vector is "+str(vector))
            
            
    
    def constructInitialZoneAxis(self):
        
        text = self.startingZoneAxis.text()
        self._startingZoneAxis = self.__constructMillerDirFromTxt(text)
        logging.info("Updated the current Zone Axis to be : {:int}".format(self._startingZoneAxis))
        self.intializePlots()
    
    
    def __constructMillerDirFromTxt(self, text):
        vector = np.fromstring(text, dtype=int, sep=' ')
        logging.debug(str(vector)+"size of vector = {:2d}".format(vector.size))
        try:
            if vector.size==4:
                millerDir = MillerDirection(lattice=self.sa.lattice, vector=vector,MillerConv='Bravais')
            else:
                millerDir = MillerDirection(lattice=self.sa.lattice, vector=vector)
            return millerDir
        except ValueError as err:
            logging.exception("Miller direction creation failed")
 
    
    def constructTargetZoneAxis(self):
        text = self.targetZoneAxis.text()
        self._targetZoneAxis = self.__constructMillerDirFromTxt(text)
        neededAlphaTilt, neededBetaTilt, error = self.findTitlsForTargetZone()
        if error<2.0:
            logging.info("Updated the current Zone Axis to be : {:int}".format(self._targetZoneAxis))
        else:
            logging.error("OOps target zone Axis can't be reached under current tilt constraints.")
            
        
        
        
    
    def findTitlsForTargetZone(self):
        #self.constructTargetZoneAxis()
        alphaTiltLimits=self.AppData["TEMHolderLimits"]["alpha"]
        betaTiltLimts=self.AppData["TEMHolderLimits"]["beta"]
        logging.info("current zone Axis ="+str(self._currentZoneAxis)+" Target Zone Axis ="+str(self._targetZoneAxis))
        
        if not self.AppData["patternModified"]:
        
            alphaAxis =self.saedData1["xAxis"].getUnitVector()
            betaAxis = self.saedData1["yAxis"].getUnitVector()
        else:
            alphaAxis =self.transformedSaed["xAxis"].getUnitVector()
            betaAxis = self.transformedSaed["yAxis"].getUnitVector()

        totlaAngualarDistance, alphaTilt, betaTilt,err, isSuccess, solList = self.sa.findTiltsForTargetZoneAxisFromSterioGram(currentZoneAxis=self._currentZoneAxis, targetZoneAxis=self._targetZoneAxis,
                                                 sterioCentredOn=self._sterioCentredOn, sterioXaxis=self._sterioXAxis,
                                                 alphaAxis = self._alphaAxis, betaAxis=self._betaAxis, alphaTiltLimits=alphaTiltLimits, betaTiltLimits=betaTiltLimts,
                                                 keepTiltAxesFixed=True)
        
        if isSuccess:
            alphaTilt=alphaTilt
            betaTilt=betaTilt
            error=err
            self.finalAlphaTiltNeeded.setText(str(np.around(alphaTilt,2)))
            self.finalBetaTiltNeeded.setText(str(np.around(betaTilt,2)))
            angleBetweenCurrentAndTargetZoneAxes=np.sqrt(alphaTilt*alphaTilt+betaTilt*betaTilt)
            self.angleBetweenCurrentAndTargetZone.setText("{:.2f}".format(angleBetweenCurrentAndTargetZoneAxes))
            logging.info("totoal angular distance to be covered to reach the required Zone Axis : {:.2f} and additional Alpha and Beta Tilts Neededed are :{:.2f} , {:.2f}".format(
            totlaAngualarDistance,alphaTilt,  betaTilt))
            return alphaTilt,betaTilt,error
        else:
            #ret = QtGui.QMessageBox.critical(self, "Unable To Find Tilts", "It seems requested Zone Axis is not possible under given holder limits !!!!",
            logging.error(r"OOps target zone Axis can't be reached under current tilt constraints. Try some equivalent vectors instead ") 
            return None, None, 1e5
    
    def gotoTargetZone(self):
        neededAlphaTilt, neededBetaTilt, error = self.findTitlsForTargetZone()
        if error<2.0:
            self.transformedSaed = self.sa.transformSaedPattern(copy.deepcopy(self.saedData1), alphaTilt=neededAlphaTilt, betaTilt=neededBetaTilt)
            self.alphaTilt.setText("{:.2f}".format(neededAlphaTilt))
            self.betaTilt.setText("{:.2f}".format(neededBetaTilt))
            #QtGui.QMessageBox.information(self,"Information!!!","Got the required alpha and beta tilts they are \n : {:.2f}, {:.2f}".format(sol["alphaTilt"],sol["betaTilt"] ))
            self.errorInAchivingTargetZoneAxis.setText("{:.1f}".format(error))
            self.AppData['alphaTilt']=neededAlphaTilt
            self.AppData['betaTilt']=neededBetaTilt
            self.modifyAndUpdatePattern()
        else:
            logging.error("OOps target zone Axis can't be reached under current tilt constraints.")
       
    
    def holderTiltValuesChanged(self):
        self.AppData['InitialAlphaTilt'] = float(self.alphaTilt.text())
        self.AppData['InitialBetaTilt'] = float(self.betaTilt.text())
        self.AppData['InitialinPlaneRotation']= float(self.inPlaneRotation.text())
        
        self.AppData['alphaTilt']=copy.deepcopy(self.AppData['InitialAlphaTilt'])
        self.AppData['betaTilt']=copy.deepcopy(self.AppData['InitialBetaTilt'])
        self.constructInitialZoneAxis()
        #self.intializePlots()
    
    def modifyAndUpdatePattern(self,alphaTilt=None,betaTilt=None):
        if alphaTilt is None:
            alphaTilt=self.AppData['alphaTilt']
        if betaTilt is None:
            betaTilt=self.AppData['betaTilt']

        alphaAxis =self.transformedSaed["xAxis"].getUnitVector()
        betaAxis = self.transformedSaed["yAxis"].getUnitVector()
        self._currentZoneAxis=copy.deepcopy(self.transformedSaed["zoneAxis"])
        options={"fixedAxes":True,"alphaRotationFirst":True,"activeRotation":True}
        self.transformedSaed = self.sa.transformSaedPattern(copy.deepcopy(self.transformedSaed), shift=[0.,0.], alphaTilt=alphaTilt,
                                                             betaTilt=betaTilt,alphaAxis=self._alphaAxis,betaAxis=self._betaAxis,options=options)
        self.transformedSaed = self.sa.computeKikuchiForSaed(self.transformedSaed,maxHkl=2)
        tmpZoneAxis =copy.copy(self.transformedSaed["zoneAxis"])
        #logging.info("current zone axis is {:int} and target zone axis is : {:int}  and the tm zone axis is : {:int}".format(self._currentZoneAxis,self._targetZoneAxis, tmpZoneAxis))
        logging.info(r"patternModified = "+str(self.AppData["patternModified"]))
        self.allDesiredDirList.append(tmpZoneAxis)
        #self.dirSterioData = self.sa.calculateSterioGraphicProjectionDirection(dirList = None, maxUVW=2,centredOn=self.sterioCentredOn,desiredPatternxAxis=self._sterioXAxis)
        self.specialSterioPoints=self.sa.calculateSterioGraphicProjectionDirection(dirList = self.allDesiredDirList,crystalOri=self.crystalOrientation)
        self.steriDataOfzoneListFordisplayInSterioPlot =self.sa.calculateSterioGraphicProjectionDirection(dirList=self.zoneListFordisplayInSterioPlot,crystalOri=self.crystalOrientation)
        logging.debug("Zone Axis before trnasformation : {:int} alpha beta being applied are : {:.2f} {:.2f}".format(self.transformedSaed["zoneAxis"],alphaTilt,betaTilt))
        logging.info("alpha and beta are "+ str(alphaTilt)+ " "+str(betaTilt)+str(tmpZoneAxis)+"the number of elemetns in list :"+str(len(self.allDesiredDirList)))
        
        self.AppData["patternModified"]=True
        tmp = self.AppData['solutionPlotOptions']
        saedOptions = {'markSpots':tmp['markSpots'],'showAbsentSpots':tmp['hideSystamticAbsences'], 'plotKikuchi':tmp['plotKikuchi'],'markKikuchi':tmp['markKikuchi'],
                       'autoPatternBounds':False}
        self.mainPlot.plotSaed(saedData=self.transformedSaed, saedAnalyzer=self.sa, **saedOptions)
        self.sterioPlot.plotSteriographicData(sterioData=[self.dirSterioData,self.specialSterioPoints,self.steriDataOfzoneListFordisplayInSterioPlot ],saedAnalyzer=self.sa,markPoints=[True,False,True])
        self.currentZoneAxisHandle.setText(str(self.transformedSaed['zoneAxis']))
        self._currentZoneAxis = copy.deepcopy(self.transformedSaed['zoneAxis'])
        self.currentAlphaTiltHandle.setText("{:.2f}".format(alphaTilt+float(self.currentAlphaTiltHandle.text())))
        self.currentBetaTiltHandle.setText("{:.2f}".format(betaTilt+float(self.currentBetaTiltHandle.text())))
        self.buildAlphaBetaTable()
    
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    frame = SaedSimulatorWindow()
    frame.show()
    sys.exit(app.exec_()) 