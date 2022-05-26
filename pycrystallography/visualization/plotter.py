import sys,os
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
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from qtpy.QtCore import QThread
else:
    from PyQt4 import QtGui, QtCore
    from PyQt4.QtCore import *
    from PyQt4.QtGui import *
    QtGui.QtWidgets = QtGui.QWidget
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt4.QtCore import QThread


import copy
from PIL import Image, ImageOps, ImageEnhance

import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import arange, sin, pi


from matplotlib.figure import Figure
from Augmentor.Operations import HistogramEqualisation
from pycrystallography.core.saedAnalyzer import SaedAnalyzer


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=10, height=4.8, dpi=100,projection=''):
        fig = Figure(figsize=(width, height), dpi=dpi)
        if projection.lower()=='polar':
            self.axes = fig.add_subplot(111,projection='polar')
        else:
            self.axes = fig.add_subplot(111,)
            
        self.axes = fig.add_subplot(111)
       
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""
    
    def __init__(self, imData=None, imageName='', spcialPointsForMarking=None, latticePoints=None, *args, **kwargs):
        
        
        self._imData=imData
        #self._imDataForDisplay=Image.fromarray(imData)
        
        self._spcialPointsForMarking=spcialPointsForMarking
        self._latticePoints=latticePoints
        self._imageName=imageName
        self._imDataForDisplay=None
        
        MyMplCanvas.__init__(self, *args, **kwargs)
        self.compute_initial_figure()        

    def compute_initial_figure(self,message=""):

            self.axes.text(0.2,0.5, message)
            plt.gca().set_axis_off()
            
    def update_figure(self,title='',brightness=1.0,contrast=1.0,histogramEqualize=False):        
        self.axes.cla()
        if self._imData is not None:
            self.adjustImageAppearance(brightness=brightness,contrast=contrast,histogramEqualize=histogramEqualize)
        axcolor = 'lightgoldenrodyellow'
        if self._latticePoints is not None:
            latticePoints=np.array(self._latticePoints)
            self.axes.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')     
        
        fontSpecialPoints = {'family': 'serif',
        'color':  'white',
        'weight': 'normal',
        'size': 16,
        }
        arrowProps = {'width':3,'head_width':5,'length_includes_head':True}
        
        if self._spcialPointsForMarking is not None:
            for i,item in enumerate(self._spcialPointsForMarking):
                self.axes.text(item[0],item[1], str(i),fontdict=fontSpecialPoints)
            arrow1_base = np.array(self._spcialPointsForMarking[0]) 
            arrow1_head = np.array(self._spcialPointsForMarking[1])
            arrow1_head = arrow1_head-arrow1_base
            self.axes.arrow(arrow1_base[0],arrow1_base[1],arrow1_head[0],arrow1_head[1],**arrowProps)
            arrow2_base = np.array(self._spcialPointsForMarking[0]) 
            arrow2_head = np.array(self._spcialPointsForMarking[2])
            arrow2_head = arrow2_head-arrow2_base
            self.axes.arrow(arrow2_base[0],arrow2_base[1],arrow2_head[0],arrow2_head[1],**arrowProps)
        
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        
        self.axes.text(0,-10, self._imageName,fontdict=font )
        self.axes.set_title(title)
        #self.axes.autoscale()
        #self.axes.set_aspect('equal', 'datalim')
        self.axes.axis('equal')
        plt.axis('off')
        self.draw()
        
    
    def adjustImageAppearance(self, brightness=1.0,contrast=1.0,histogramEqualize=False,):
        if histogramEqualize:
            self._imDataForDisplay=ImageOps.equalize(self._imDataForDisplay)
        if brightness !=1.0:
            brightnessEnhancer=ImageEnhance.Brightness(self._imDataForDisplay)
            self._imDataForDisplay=brightnessEnhancer.enhance(brightness)
        if contrast !=1.0:
            contrastEnhancer=ImageEnhance.Contrast(self._imDataForDisplay)
            self._imDataForDisplay=contrastEnhancer.enhance(contrast)
        self._updateImageAppearanceInPlot()    
            
    def _updateImageAppearanceInPlot(self):
        self.axes.cla()
        if self._imDataForDisplay is not None:
            self.axes.imshow(self._imDataForDisplay,cmap='gray')
            self.draw()
    
    def plotSolution(self,saedSolution,saedAnalyzer,title="",markSpots=True,showAbsentSpots=False,markKikuchi=False,plotKikuchi=False,plotShow=False,autoPatternBounds=True): 
        
        if saedSolution is not None:
            self._updateImageAppearanceInPlot()
            if self._imData is not None:
                bounds=self._imData.shape
            else:
                bounds = [1e10, 1e10]
            print("pattern bounds are ", bounds)
            self.update_figure()
            setBounds=False
            if not autoPatternBounds:
                ylims = self.axes.get_ylim()
                xlims = self.axes.get_xlim()
                setBounds=True
                 
            saedData = saedSolution["pattern"]
            angleError = str(saedSolution["angleError"])
            dError = str(saedSolution["dError"])
            #print("here is the kikuchi data\n", saedData['kikuchiData'])
            saedAnalyzer.plotSAED(saedData,plotShow=plotShow,figHandle=None,axisHandle=self.axes,makeTransperent=False,markSpots=markSpots,showAbsentSpots=showAbsentSpots,
                                  plotKikuchi=plotKikuchi,markKikuchi=markKikuchi)
            
            
            if setBounds:
                self.axes.set_ylim(ylims)
                self.axes.set_xlim(xlims)
            plt.axis('off')
            self.draw()

    def plotSaed(self,saedData,saedAnalyzer,plotShow=False,makeTransperent=False,markSpots=False,showAbsentSpots=False,
                                  plotKikuchi=False,markKikuchi=False, autoPatternBounds=True):
        setBounds=False
        if not autoPatternBounds:
            ylims = self.axes.get_ylim()
            xlims = self.axes.get_xlim()
            setBounds=True
        self.axes.cla()
        
        saedAnalyzer.plotSAED(saedData,plotShow=plotShow,figHandle=None,axisHandle=self.axes,makeTransperent=False,markSpots=markSpots,showAbsentSpots=showAbsentSpots,
                                  plotKikuchi=plotKikuchi,markKikuchi=markKikuchi)
        
            
        
        plt.axis('off')
        #self.axes('off')
        #self.axes.autoscale()
        if setBounds:
            self.axes.set_ylim(ylims)
            self.axes.set_xlim(xlims)
            
        self.draw()
    
    
    def plotSteriographicData(self,sterioData, saedAnalyzer,plotShow=False,makeTransperent=False,markPoints=False,showAbsentSpots=False,
                                 ):
        self.axes.cla()
        saedAnalyzer.plotSterioGraphicProjection(sterioData,plotShow=plotShow,axisHandle=self.axes, makeTransperent=makeTransperent,markPoints=markPoints,
                                  )

        self.draw()

        
    
    
    def updateFigureData(self,imData=None, imageName='', spcialPointsForMarking=None, latticePoints=None,):
        if imData is not None:
            self._imData=imData
            self._imDataForDisplay=Image.fromarray(imData)
        else:
            self._imData=None
            self._imDataForDisplay=None
            
                
        if len(imageName)>0:
            self._imageName=imageName
        if spcialPointsForMarking is not None:
            self._spcialPointsForMarking=spcialPointsForMarking
        if latticePoints is not None:
            self._latticePoints=latticePoints
        