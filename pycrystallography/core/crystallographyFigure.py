# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

"""
+++++++++++++++++++++++++++++++++++
Module Name: **crystallographyFigure.py**
+++++++++++++++++++++++++++++++++++

This module :mod:`crystallographyFigure` defines the classes relating to plotting the crystallographic Data. 

This module contains  classes and functions listed below.


"""

from __future__ import division, unicode_literals

from six.moves import map, zip
import sys
import numpy as np
from numpy.linalg import inv
from matplotlib.patches import Circle
from matplotlib import colors
from matplotlib.lines import Line2D
import os, pathlib
import logging
from tabulate import tabulate
import shutil, warnings

import matplotlib.pyplot as plt
import pycrystallography.utilities.graphicUtilities as gu
from pycrystallography.core.orientation import Orientation
import pycrystallography.utilities.pymathutilityfunctions as pmt
import pandas as pd
import webbrowser
from matplotlib.widgets import CheckButtons, TextBox



__author__ = "K V Mani Krishna"
__copyright__ = ""
__version__ = "1.0"
__maintainer__ = "K V Mani Krishna"
__email__ = "kvmani@barc.gov.in"
__status__ = "Alpha"
__date__ = "July 14 2017"

plot000Reflection=False
defaultOptions = {
                   "Figure":{"plotShow":True},
                   "SAED":{"makeDataFrame":True,"NumberOfMaxSpotsForMarking":2, "markSpots":False, "makeTransperent":False,
                           "markPrimarySpots":False,
                           "pixcelOffsetMagnitude":30,### offset for the labels to diffraction spots!!!
                           "writeSpotGroupId":False,
                           "markerSizes":           [100,100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], ### first one is of parent
                           "markerStyles":          ["o","h", "*", "X", "D", "P", "*", ">", "<", "v", "^", "s", "d", "p"],
                            "markerStylesForLegend":["ok", "rh","go","b+","ro","*y","*m",">k", "<k","vg","^b","Dr","Xy","okm"],
                           "markerColors":          ["k", 'r', 'g', 'b', 'r', 'y',  'm', 'k', 'k', 'g',  'b', 'r', 'y',  'k'],
                           "shouldFillMarker":      [True, False, False, False, False, False, False, False, False, False, False, False, False,True],
                           #"markerColor":['tab:blue', 'tab:red','tab:black', 'tab:green','tab:yellow'],
                           "spotLabelXyOffset":[(0,10),(0,-10),(10,0),(-10,0),(-5,5),(5,-5)],
                          "spotslabelFaceColor":[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
                           "includeTextInfo":False,
                           "InfoTextPosition":[(0.7,0.6),(0.3,0.6),(0.7,0.1),(0.7,0.3),(0.1,0.3),(0.1,0.5)],
                           "saedLabel":["Alpha","Beta","Gamma","delta","alpha_11","beta_11"],
                           "SaedSymbol":[r"$\alpha$",r"$\beta$",r"$\gamma$",r"$\delta$",r"$\alpha_{11}$",r"$\beta_{11}$"],
                          "showAbsentSpots":True,'plotKikuchi':False,'markKikuchi':False},
                    "SteriographicPlot":{'projectToSameHemisphere':True,"markerSizes":[16,16,48],"markerStyles":["o","x",">"],"markPoints":[True,False,False],
                                         "InfoTextPosition":[(0.7,0.9),(0.1,0.9), (0.7,0.1)],"markerEdgeColors":['b','r','k'],}
                  
                 }


class CrystallographyFigure(object):
    """
    A class for creating the Crystallographic Figures such as sterofgrams, TEm patterns etc.
    """
    
    def __init__(self,data,figHandle=None,axisHandle=None,figType="Saed",projectToSameHemisphere=True,title=None):
        """Create an crystal orientation relationship from ::
        note that you need to input both fig handle and axis handle in case the figure is to be sent to specific plot area
        """
        self._isPolar=False
        self._staticAnnotationsData=[]
        self._staticAnnotationsCount=0
        self._activeAnnotationId=0
        self._annotationActivePosition=(0,0)
        self._staticAnnotationList=[]
        self._hideStaticAnnotations=False
        self._displayHelp=False
        self._df=None
        self._patternVisibility=None
        self._rotationValue = 5 ## degree
        if figType=="Saed":
            self._figureType = "Saed" ##other options are SterioGraphic Projection etc
            self._data = data 
            if isinstance (data, list):
                self._isCompositeFigure=True
                self._numberOfCompositePlots = len(data)  
            else:
                self._isCompositeFigure=False       
                self._numberOfCompositePlots = 1    
            
            
            if figHandle is None and axisHandle is not None:
                self._fig = axisHandle.figure
                self._ax = axisHandle
            elif figHandle is None and axisHandle is  None:
                self._fig = plt.figure()
                self._ax = self._fig.add_subplot(111,)
            
            else:
                self._fig = figHandle
                self._ax = axisHandle



            self._fig.canvas.mpl_connect("motion_notify_event", self.hover)
            self._fig.canvas.mpl_connect("motion_notify_event", self.hover_botonAnnotation)
            self._fig.canvas.mpl_connect('button_press_event', self.onclick )
            self._fig.canvas.mpl_connect('key_press_event', self.keyPress)

        elif figType=='SterioGraphicProjection':
            self._figureType="SterioGraphic"
            self._data = data 
            if isinstance (data, list):
                self._isCompositeFigure=True
                self._numberOfCompositePlots = len(data)  
            else:
                self._isCompositeFigure=False       
                self._numberOfCompositePlots = 1    
            
            if figHandle is None and axisHandle is not None:
                self._fig = axisHandle.figure
                self._axNorth = axisHandle
                #self._axNorth.set_projectionn()
                self._plotBothHemiSpheres=False
            elif figHandle is None and axisHandle is  None and projectToSameHemisphere:
                self._fig = plt.figure()
                self._axNorth = self._fig.add_subplot(111,)
                self._axNorth.margins(0)
                self._axNorth.set_title("NorthPole")
                self._plotBothHemiSpheres=False
                
            elif figHandle is None and axisHandle is  None and not projectToSameHemisphere:
                self._fig = plt.figure()
                self._axNorth = self._fig.add_subplot(121, projection='polar')
                self._axNorth.margins(0)
                self._axNorth.set_title("NorthPole")
                self._axSouth = self._fig.add_subplot(122, projection='polar')
                self._axSouth.margins(0)
                self._axSouth.set_title("SouthPole")
                self._plotBothHemiSpheres=True
                self._isPolar=True
            else:
                self._fig = figHandle
                self._ax = axisHandle
            self._fig.canvas.mpl_connect("motion_notify_event", self.hover)
            self._fig.canvas.mpl_connect("motion_notify_event", self.hover_botonAnnotation)

        else:
            raise ValueError("Unknown Figure type!! Currently only 2 types of Figuress are possible : Saed, SterioGraphicProjection but was supplied "+figType)

        if title is not None:
            self._figTitle = title
        else:
            self._figTitle = None
        self._isVisibilityCheckButtonPresent=False
        self._createVisibilityCheckButtons()
        self._createRotationTextBox()

        print("Data Loaded")


    def update_annot(self,ind,annotationId):
        cmap = plt.cm.RdYlGn
        norm = plt.Normalize(1, 4)
        c = np.random.randint(1, 5, size=15)
        pos = self._scatterHandles[annotationId].get_offsets()[ind["ind"][0]]
        self._annot.xy = pos
        self._annotationActivePosition=pos
        text = self._annotationList[annotationId]["label"]
        self._annot.set_text(text)
        self._annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        self._annot.get_bbox_patch().set_alpha(0.8)

    def onclick(self,event):
        self._staticAnnotationsCount+=1
        text = self._staticAnnotationList[self._activeAnnotationId]


        tmpDict = {"id": self._staticAnnotationsCount, "XY":(event.xdata, event.ydata), "spotPosition":self._annotationList[self._activeAnnotationId]["XY"],
                                           "markerXY":(), "text":text, "Spot":self._annotationList[self._activeAnnotationId]["Spot"],
                   "Label":self._annotationList[self._activeAnnotationId]["Label"]}

        pos=self._annotationActivePosition
        radius=40
        angle = 2*np.pi*np.random.random()
        labelOffset = [np.around(radius * np.cos(angle)), np.around(radius * np.sin(angle))]
        connectionstyle = "angle,angleA=-90,angleB=180,rad=0"
        tmpDict["annotationhandle"] = self._ax.annotate(text, xy=pos, xytext=labelOffset, textcoords="offset points",
                                                        fontsize=14, alpha=0.8,
                            bbox=dict(fc="white", ec="grey", pad=0.0, ),
                            arrowprops=dict(arrowstyle="-|>", connectionstyle=connectionstyle))
        tmpDict["annotationhandle"].set_visible(True)
        tmpDict["annotationhandle"].draggable()
        self._staticAnnotationsData.append(tmpDict)
        self._fig.canvas.draw()  # redraw the figure

    def keyPress(self,event):
        print(f"Pressed key: {event.key}")
        sys.stdout.flush()
        if event.key == 'h' or event.key=='H': ### hide or un hide exisiting static annotations
            self._hideStaticAnnotations = not self._hideStaticAnnotations  ## toggling state
            for item in self._staticAnnotationsData:
                    item["annotationhandle"].set_visible(self._hideStaticAnnotations)
            self._fig.canvas.draw_idle()
            logging.debug("hide state of the static annotations toggled!!!")

        if event.key == 'd' or event.key=='d': ### delete last added annotation
            if self._staticAnnotationsCount>0:
                logging.warning(f"Deleteting the last added annotation as the {event.key} is pressed")
                self._staticAnnotationsData[-1]["annotationhandle"].remove()
                del self._staticAnnotationsData[-1]
                del self._staticAnnotationList[-1]
                self._staticAnnotationsCount-=1
                self._fig.canvas.draw_idle()
        if event.key == 'i' or event.key=='I': ### display help info on how to use crystallogrpicFig interactively
            self._displayHelp = not self._displayHelp ## toggling state
            self._helpAnnotationHandle.set_visible(self._displayHelp)
            self._fig.canvas.draw_idle()


    def hover(self,event):
        #print("hai")
        vis = self._annot.get_visible()
        if event.inaxes == self._ax:
            for annotationId, scatterHandles in enumerate(self._scatterHandles):
                cont, ind = scatterHandles.contains(event)
                if cont:
                    self._activeAnnotationId=annotationId
                    self.update_annot(ind,annotationId)
                    self._annot.set_visible(True)
                    self._fig.canvas.draw_idle()
                else:
                    if vis:
                        self._annot.set_visible(False)
                        self._fig.canvas.draw_idle()
        plt.show()

    def hover_botonAnnotation(self, event):
        #vis = self._bottomAnnotation.get_visible()
        if event.inaxes == self._ax:
            x, y = event.xdata, event.ydata
            dSpacing = np.around(self._figureDspacingScalingFactor/ np.sqrt(x ** 2 + y ** 2), 3)
            text = f"$|g^{{*}}|={np.around(1/dSpacing,3)}\AA^{{-1}}; d={dSpacing} \AA$"
            text2=""
            if self._staticAnnotationsCount>1:
                spot1Index, spot2Index= self._staticAnnotationsCount-1, self._staticAnnotationsCount-2
                spot1pos,spot2pos = self._staticAnnotationsData[spot1Index]["spotPosition"], self._staticAnnotationsData[spot2Index]["spotPosition"]
                spot1,spot2 = self._staticAnnotationsData[spot1Index]["Spot"], self._staticAnnotationsData[spot2Index]["Spot"]
                label1,label2 = self._staticAnnotationsData[spot1Index]["Label"], self._staticAnnotationsData[spot2Index]["Label"]
                line1 = np.array([[0,0], spot1pos])
                line2 = np.array([[0,0], spot2pos])
                angleBetweenSpots = abs(pmt.angleBetween2Lines(line1, line2, units="Deg"))
                if angleBetweenSpots>180.:
                    angleBetweenSpots-=180.
                angleBetweenSpots = np.around(angleBetweenSpots,2)
                dRatio = spot1.dspacing/spot2.dspacing
                if dRatio < 1.0:
                    dRatio =1/dRatio
                text2 = f"\n{label1} {label2}: {angleBetweenSpots}$\degree$; dRatio={np.around(dRatio,3)}"

            text=text+text2
            ymin, ymax = self._ax.get_ylim()
            xmin, xmax = self._ax.get_xlim()
            pos = xmax-0.4*xmax, ymin-0.1*ymin
            self._bottomAnnotation.xy = pos
            self._bottomAnnotation.set_text(text)
            self._bottomAnnotation.set_visible(True)
            self._fig.canvas.draw_idle()

        plt.show()

    def _setPatternVisibility(self):
        print("Yes working on check buttons!!!!")
        self._patternVisibility = self._checkButtons.get_status()
        self._ax.clear()
        self.buildDataFrame()
        self.plot()

    def buildDataFrame(self, rotationAngle=0, *rgs,**kwargs):
        """

        :param patternVisibility: list of bulen values (for composite patterns) indicating which of them need to be included True means include for showings,
        e.g. if tmatrix, var1, var2 are threre in saed_data, and we want only matrix and var1 to appear in plot, use patternVisibility = [True, True, false]

        :return:
        """
        Data = self._data

        self._staticAnnotationsData = []
        self._staticAnnotationsCount = 0
        self._activeAnnotationId = 0
        self._annotationActivePosition = (0, 0)
        self._staticAnnotationList = []


        if not isinstance(Data,list):
            Data = [Data]
        if self._patternVisibility is None: ## happens when plot is called first.
                self._patternVisibility = [True for i in Data]


        df = pd.DataFrame(
            columns=["SpotHkl", "includePlot", "SpotGroupId", "SpotSubId", "NumberOfOverLappingSpots", "XY", "Label",
                     "markerStyle", "markerColor",
                     "markerSize", "showMarker", "absentSpot", "PhaseName", "dSpacing", "markSpot",
                     "intensity", "LabelPosition", "LableAngle", "shouldFillMarker", "isPrimarySpot", "patternInfoText",
                     "OverLappingSpotIds"])

        mainLegend = []
        htmlOutFileName = ""
        patternInfoTextDictsList = []
        self._annotationList = []
        self._scatterHandles = []  #### list of all scatter plot artists for dynamic annotation display

        if np.abs(rotationAngle) > 0:
            rotateOn=True
            theta = rotationAngle * np.pi / 180
            cosTheta = np.cos(theta)
            sinTheta = np.sin(theta)
            print(f"Rotation is on value : {rotationAngle}")
        else:
            rotateOn=False

        for i, data in enumerate(Data):
            if self._patternVisibility[i]:  ### this will decide if this is included in the pattern (acts like a mask)

                if data is not None:
                    spotDict = {}
                    patternCenter = data["patterCenter"]
                    patternInfoTextDictsList.append({"Pattern": data["SaedLabel"], "data": data["patternInfoTextDict"]})
                    scalingFactor = data["scalingFactor"]
                    spotData = data["SpotData"]
                    if "SaedSymbol" in data:
                        SaedSymbol = data["SaedSymbol"]
                    else:
                        SaedSymbol = self._extractOption("SaedSymbol", "SAED", **kwargs)
                    if "SaedLabel" in data:
                        saedLabel = data["SaedLabel"]
                    else:
                        saedLabel = self._extractOption("saedLabel", "SAED", **kwargs)

                    htmlOutFileName += saedLabel + "{:int} ".format(data["zoneAxis"])
                    htmlOutFileName = htmlOutFileName.replace("uvw =", " ")

                    #ax.plot(patternCenter[0], patternCenter[1], 'ko', markersize=16)
                    markerStyle = self._extractOption("markerStyles", "SAED", Ind=i, **kwargs)
                    markerColor = self._extractOption("markerColors", "SAED", Ind=i, **kwargs)
                    markerSize = self._extractOption("markerSizes", "SAED", Ind=i, **kwargs)
                    plotKikuchi = self._extractOption("plotKikuchi", "SAED", Ind=i, **kwargs)
                    showAbsentSpots = self._extractOption("showAbsentSpots", "SAED", Ind=i, **kwargs)
                    makeTransperent = self._extractOption("makeTransperent", "SAED", Ind=i, **kwargs)
                    writeSpotGroupId = self._extractOption("writeSpotGroupId", "SAED", Ind=i, **kwargs)
                    shouldFillMarker = self._extractOption("shouldFillMarker", "SAED", Ind=i, **kwargs)
                    markerStylesForLegend = markerColor + markerStyle
                    if makeTransperent:
                        alpha = 0.5
                    else:
                        alpha = 1.0
                    markSpots = self._extractOption("markSpots", "SAED", Ind=i, **kwargs, alpha=alpha)
                    markPrimarySpots = self._extractOption("markPrimarySpots", "SAED", Ind=i, **kwargs, alpha=alpha)

                    for ii, item in enumerate(spotData):
                        spotDict["SpotHkl"] = item["Plane"]
                        #spotDict["includePlot"] = patternVisibility[ii] ### this will decide if this is included in the pattern (acts like a mask)
                        if rotateOn:
                            pc=[0,0]
                            xy = item["XY"]
                            xy = np.asarray([xy[0] - pc[0], xy[1] - pc[1]])
                            xy = [xy[0] * cosTheta - xy[1] * sinTheta + pc[0],
                                  xy[1] * cosTheta + xy[0] * sinTheta + pc[1]]
                            item["XY"] = xy

                        spotDict["XY"] = np.around(item["XY"], 3)
                        spotDict["Label"] = item["Plane"].getLatexString()[:-1] + r"_{" + SaedSymbol.replace("$",
                                                                                                             "") + r"}$"
                        spotDict["markerStyle"] = markerStyle
                        spotDict["markerColor"] = markerColor
                        spotDict["markerSize"] = markerSize
                        spotDict["absentSpot"] = False if item["Intensity"] > 1e-3 else True
                        spotDict["showMarker"] = True if item["Intensity"] > 1e-3 else False
                        if showAbsentSpots:
                            spotDict["showMarker"] = True

                        if item["isPrimarySpot"] and markPrimarySpots:
                            spotDict["markSpot"] = True
                        else:
                            spotDict["markSpot"] = markSpots

                        spotDict["PhaseName"] = saedLabel
                        spotDict["dSpacing"] = item["Plane"].dspacing
                        spotDict["intensity"] = item["Intensity"]
                        spotDict["LabelPosition"] = [None, None]
                        spotDict["shouldFillMarker"] = shouldFillMarker

                        df = df.append(spotDict, ignore_index=True)

                    if not shouldFillMarker:
                        markerFaceColor = 'none'
                    else:
                        markerFaceColor = markerColor

                    mainLegend.append((Line2D([0], [0], marker=markerStyle, lw=0, color=markerColor, label=saedLabel,
                                                  markerfacecolor=markerFaceColor, markersize=15)))


        self._df = self.__groupDiffractionSpots(df, tolerance=1e-3)
        self._mainLegend= mainLegend
        self._patternInfoTextDictsList=patternInfoTextDictsList
        self._htmlOutFileName=htmlOutFileName


    def _createRotationTextBox(self):
        graphBox = self._fig.add_axes([0.2,0.9,0.05,0.07 ])
        self._RotationTexBoxHandle = TextBox(graphBox, "Rotate By (Deg): ")
        self._RotationTexBoxHandle.set_val(str(self._rotationValue))
        self._RotationTexBoxHandle.on_submit(lambda e : self._rotatePlot())

    def _rotatePlot(self):
        """
        method to rotate the SAED in plane by an angle
        :return:
        """
        rotationAngle = float(self._RotationTexBoxHandle.text)/2.0 ### this is a work around the bug that is causing pattern to rotate twice instead of once hence we are going to
                                                                   ### divinde the required roation by 2.0 so that after 2 succesive rotations we get what we want.
        logging.warning("Rotating by half to achieve the required rotation bug needs to be corrected later!!!")
        warnings.warn("Rotating by half to achieve the required rotation bug needs to be corrected later!!!")
        self._ax.clear()
        self._fig.canvas.draw()  # redraw the figure
        self.buildDataFrame(rotationAngle=rotationAngle)
        self._rotationValue=0. ### setting rotation value back to 0
        self.plot()

    def _createVisibilityCheckButtons(self):
        if self._numberOfCompositePlots>1:
            self._checkBoxexAxes = plt.axes([0.05, 0.4, 0.1, 0.15])
            labels=[]
            for i, item in enumerate(self._data):
                if "SaedSymbol" in item:
                    labels.append(item["SaedSymbol"])
                else:
                    labels.append(str(i))

            visibility = [True for i in labels]
            self._checkButtons = CheckButtons(self._checkBoxexAxes, labels, visibility)
            self._isVisibilityCheckButtonPresent=True
            self._checkButtons.on_clicked( lambda e : self._setPatternVisibility())
        else:
            self._isVisibilityCheckButtonPresent=False




    def plot(self,*args,**kwargs):
        """

        :param patternVisibility: list of bulen values (for composite patterns) indicating which of them need to be included True means include for showings,
        e.g. if tmatrix, var1, var2 are threre in saed_data, and we want only matrix and var1 to appear in plot, use patternVisibility = [True, True, false]
        :param args:
        :param kwargs:
        :return:
        """

        ax = self._ax
        ax.clear()


        ax.plot(0, 0, 'ok', picker=True, markersize=16, )
        plt.sca(ax)

        if "saed" in self._figureType.lower():
            if self._df is None:
                self.buildDataFrame(*args,**kwargs)
            df = self._df
            fc = colors.to_rgba('lightgrey')
            ec = 'none' ### colors.to_rgba('black')
            fc = fc[:-1]+(0.5,)
            pixcelOffsetMagnitude = 30
            for index, row in df.iterrows():
                if not row["absentSpot"]:
                    x, y = row["XY"]
                    if row["showMarker"]:
                        markerStyle, markerColor, markerSize, shouldFillMarker = row["markerStyle"], row["markerColor"], row["markerSize"], row["shouldFillMarker"]
                        if not shouldFillMarker:
                            scatterHandle = ax.scatter(x, y, marker=markerStyle, s=markerSize,facecolors='none', edgecolors=markerColor, linewidths=2)
                        else:
                            scatterHandle = ax.scatter(x, y, marker=markerStyle, s=markerSize, facecolors=markerColor, edgecolors=markerColor,linewidths=2)
                        self._scatterHandles.append(scatterHandle)

                if not row["absentSpot"]:
                    label, LabelPosition, angle, dSpacing = row["Label"], \
                                                  pixcelOffsetMagnitude * (np.array(row["LabelPosition"])), \
                                                  row["LableAngle"], np.around(row["dSpacing"],3)

                    if row["NumberOfOverLappingSpots"] < 2:
                        LabelPosition, rotation = [0, 15], 0

                    if len(row["OverLappingSpotIds"])>0:
                        localLabel = ""
                        localLabel2 = ""
                        for overLappingSpot in row["OverLappingSpotIds"]:
                            if not df.loc[overLappingSpot, 'absentSpot']: ### add only those overalapping spots which are allowed by structure factor
                                localLabel+=f"{df.loc[overLappingSpot, 'Label']}, $d={np.around(row['dSpacing'],3)} \AA$\n"
                                localLabel2+=f"{df.loc[overLappingSpot, 'Label']}\n"
                        localLabel=localLabel[:-1]
                        localLabel2=localLabel2[:-1]
                    else:
                        localLabel = f"{label}, $d={dSpacing}\AA$"
                        localLabel2= f"{label}"

                    self._annotationList.append({"Spot":row["SpotHkl"], "label":localLabel, "XY":(x,y), "Label":row["Label"]})
                    self._staticAnnotationList.append(localLabel2)



            self._annot = ax.annotate("", xy=(0, 0), xytext=LabelPosition, textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),family='monospace',
                                      arrowprops=dict(arrowstyle="->"))
            self._annot.set_visible(False)

            ymin, ymax = self._ax.get_ylim()
            xmin, xmax = self._ax.get_xlim()
            pos = (xmax, ymin)
            self._bottomAnnotation = ax.annotate("", xy=pos, xytext=LabelPosition, textcoords="offset points",
                                                 bbox=dict(boxstyle="round", fc="w"),
                                                 #arrowprops=dict(arrowstyle="->")
                                                 )
            self._bottomAnnotation.set_visible(False)
            x,y = df.loc[0, "XY"]
            dSpacing = df.loc[0, "dSpacing"]
            self._figureDspacingScalingFactor = dSpacing/(1/np.sqrt(x*x+y*y))

            df = df.sort_values(["SpotGroupId","SpotSubId"])
            dfTmp=df.copy()
            dfTmp=dfTmp[dfTmp["intensity"] > 1e-3]
            # df.round({'dogs': 1, 'cats': 0})
            dfTmp=dfTmp.round({"XY": 3, "dSpacing": 3})
            # dfTmp["XY"] = dfTmp["XY"]
            patternInfoTextTable = pd.json_normalize(self._patternInfoTextDictsList)
            htmlHeaderFileName = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'tmp',
                                          'htmlHeader.html')
            patternInfoTextTable.to_html(htmlHeaderFileName)
            patternInfoTextTable = patternInfoTextTable.rename(columns={"data.zoneAxis": "zoneAxis", "data.X-Axis": "X-Axis", "data.Ori1":"Ori1"})
            patternInfoTextTableText = tabulate(patternInfoTextTable[["Pattern", "zoneAxis", "X-Axis", "Ori1"]],
                                                headers='keys', tablefmt='plain',showindex=False)


            patternInfoTextTableText = patternInfoTextTableText.replace("uvw =", "")
            #### following lines makes the pattern info to appear at the top corner of the pattern upon mouse hover!!!
            infoPosition = xmin,ymax-(ymax-ymin)/5
            scatterHandle = ax.scatter(infoPosition[0],infoPosition[1], marker="*", s=10, facecolors=markerColor, edgecolors=markerColor, )

            ax.annotate("Info", xy=infoPosition )

            self._scatterHandles.append(scatterHandle)

            self._annotationList.append({"Spot":None, "label":patternInfoTextTableText, "XY":infoPosition})
            try:
                table=dfTmp.pivot(index="SpotGroupId", columns="PhaseName",
                                  values=["SpotHkl", "XY", "dSpacing", "intensity", ])

                df.to_csv("df_difraftionPatternFigure.csv")

                if len(self._htmlOutFileName)>200:
                    logging.warning(f"The length of file name exceeding 200 characters. It is actually {len(self._htmlOutFileName)}")
                    htmlOutFileName=self._htmlOutFileName[:30]+"____"

                if self._figTitle is None:
                    self._figTitle=self._htmlOutFileName

                htmlOutFileName = self._htmlOutFileName+".html"
                htmlOutFileName=htmlOutFileName.replace(":","_")
                #htmlFile = os.path.join(r"../../tmp", htmlOutFileName)
                htmlFile = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'tmp',
                                              htmlOutFileName)
                mergedHtmlName = os.path.join(pathlib.Path(__file__).parent.parent.parent, 'tmp',
                                              'merged'+htmlOutFileName)

                table.to_html(htmlFile)


                #### merge header and html file here
                with open(mergedHtmlName, 'wb') as wfd:
                    for f in [htmlHeaderFileName, htmlFile, ]:
                        with open(f, 'rb') as fd:
                            shutil.copyfileobj(fd, wfd)

                #####
                fileName = f'file:///'+mergedHtmlName
                webbrowser.open_new_tab(fileName)
            except:
                print("Could not create pivot table")

            text = "\n".join(["h|H--> toggle diaply of annotations",
                            "i|I--> toggle display of this help info",
                            "d|D--> delete the latest annotation",
                            ])
            self._helpAnnotationHandle = self._ax.annotate(text, xy=(0.1, 0.9), xycoords="figure fraction", xytext=(0, 0),
                                                           textcoords="offset points",
                                                           bbox=dict(boxstyle="round", fc="w"))
            self._helpAnnotationHandle.set_visible(self._displayHelp)
            ax.autoscale()
            ax.legend()
            ax.legend(handles=self._mainLegend, loc='right')
            ax.set_aspect('equal')
            ax.axis('equal')
            ax.text(0.35,0.01,"Developed by Mani Krishna, BARC",transform=ax.transAxes,
                    horizontalalignment='center',fontsize=14)
            self._fig.suptitle(self._figTitle, fontsize=16)
            # mng = plt.get_current_fig_manager()
            # mng.full_screen_toggle()
            plotShow = self._extractOption("plotShow","Figure", **kwargs)
            ##print(df)
            if plotShow:
                plt.show()

    def plotSteriographic(self,*args,**kwargs ):
        Data = self._data
        fig = self._fig,
        
        if self._plotBothHemiSpheres:
            axNorth = self._axNorth
            axSouth = self._axSouth
            projectToSameHemisphere=False
        else:
            axNorth = self._axNorth
            projectToSameHemisphere=True
            
        #if "steriographicrojection" in self._figureType.lower():
        scaleTo90=True
        if scaleTo90:
                scalingFactor=90
        else:
                scalingFactor = 1.0
                
        if  self._numberOfCompositePlots ==1: 
                Data = [Data]
                Data.append(None) 
        for j, data in enumerate(Data):
                if data is not None:
                    for i,item in enumerate(data["item"]):
                        if self._isPolar:
                            position = data["polarPoint"][i]
                        else:
                            #position = np.array(data["alphaBeta"][i])
                            position = np.array(data["polarPoint"][i])
                            tmpX = position[1]*np.cos(position[0]*np.pi/180)
                            tmpY = position[1]*np.sin(position[0]*np.pi/180)
                            position[0], position[1] =tmpX, tmpY
                            
                        
                        text = data["string"][i]
                        isNorthPole = data["isNorthPole"][i]
                          
                        
                        if projectToSameHemisphere:
                            ax=axNorth
                        else:
                            if isNorthPole:
                                ax = axNorth
                            else:
                                ax = axSouth
                                #print("Yes plotting the point on the south "+text)
                        radian= np.pi/180         
                        markerStyles= self._extractOption("markerStyles","SteriographicPlot", Ind=j, **kwargs)
                        markerSize = self._extractOption("markerSizes","SteriographicPlot", Ind=j, **kwargs)
                        #markerFaceColor=self._extractOption("markerFaceColors","SteriographicPlot", Ind=j, **kwargs)
                        markerEdgeColor=self._extractOption("markerEdgeColors","SteriographicPlot", Ind=j, **kwargs)
                        if isNorthPole:
                            markerFaceColor =markerEdgeColor
                        else:
                            markerFaceColor = 'none' 
                        
                        #print("Yes here is the plotting stuff")
                        if self._isPolar:
                            ax.scatter(position[0]*radian,position[1], s=markerSize, marker=markerStyles, facecolors=markerFaceColor,edgecolors=markerEdgeColor)
                        else:
                            ax.scatter(position[0],position[1], s=markerSize, marker=markerStyles, facecolors=markerFaceColor,edgecolors=markerEdgeColor)
                        
                            
                        markPoints=self._extractOption("markPoints","SteriographicPlot",Ind=j, **kwargs)
                        if markPoints:
                            if self._isPolar:
                                ax.annotate(text,xy=(position[0]*radian,position[1]), xytext=(0, 10),
                                                 textcoords='offset points', ha='center', va='center',fontsize=10,
                                                 bbox=dict(facecolor=[1,1,1], edgecolor='white', pad=0.0)
                                                 )
                            else:
                                ax.annotate(text,xy=(position[0],position[1]), xytext=(0, 10),
                                                 textcoords='offset points', ha='center', va='center',fontsize=10,
                                                 bbox=dict(facecolor=[1,1,1], edgecolor='white', pad=0.0)
                                                 )
                                
                         
                       
                
                    InfoTextPosition = self._extractOption("InfoTextPosition","SteriographicPlot", Ind=j, **kwargs)
                    #ax=axNorth
                    ax.annotate(data["patternInfoText"], xy=InfoTextPosition, xytext=(5, 0), 
                             xycoords='figure fraction', textcoords='offset points',fontsize=10,
                             bbox=dict(facecolor=[1,1,1], edgecolor='white', pad=0.0))
    
                
        if self._isPolar:
            rMax=90.
            axNorth.set_ylim(0,90.)
            axNorth.grid(True)
            
            
            if not projectToSameHemisphere:
                axSouth.set_ylim(0.,90.)
                axSouth.grid(True)
        else:
            zVec = np.array([0,0,1.])
            angs = np.linspace(10.,90.,num=9)
            sterioPoint = np.array([0.,0.])
            radii=[]
            for ang in angs:
                rot = Orientation(axis=[1,0,0],degrees=ang)
                vec = rot.rotate(zVec)
                sterioPoint = vec[0:2]/(1+vec[2]) 
                radii.append(np.linalg.norm(sterioPoint)*90)
            
            pc = [0.,0]
            for i, item in enumerate(radii):
                circle = Circle((pc[0], pc[1]), radius=item, fc='k',fill=False)
                ax.add_patch(circle)
            xaxis = np.array([[-1,0.],[1.0,0]])*scalingFactor
            yaxis = np.array([[0,-1], [0,1]])*scalingFactor
            ax.plot(xaxis[:,0],xaxis[:,1])
            ax.plot(yaxis[:,0],yaxis[:,1])
            ax.set_aspect('equal','box')
            
            ax.axis('off')
        
        plotShow = self._extractOption("plotShow","Figure", **kwargs)
        if plotShow:
                plt.show()


    
    @staticmethod    
    def _extractOption(optionName,optionBelongsTo,Ind=0, **kwargs,):
        """
        helper method to extract the suitable options
        optionName is the string indicating Which Option you want to extract
        optionBelongsTo is the string indicating the key of the defaultOptions dictionary from which the extraction has to be made
        Ind is the Index of the multi valued option (for example markerSizes etc) useful for Composite Plots in which for each individual
        plot I, we use the default options Index I
        """ 
        if optionName in kwargs:
            tmp = kwargs.pop(optionName)
            if isinstance(tmp,(list,tuple)):
                #kwargs.pop(optionName)
                return tmp[Ind]
            else:
                return tmp
        else:
            if optionName in defaultOptions[optionBelongsTo]:
                tmp = defaultOptions[optionBelongsTo][optionName]
                if isinstance(tmp,(list,tuple)):
                    if Ind>=len(tmp):
                        warnings.warn("index exceeded the limit and hence assinging last value")
                        return tmp[-1]
                    else:
                        return tmp[Ind]
                else:
                    return tmp
            else:
                raise ValueError(optionName+" Does not exist in the default option List of "+optionBelongsTo)
        

    @staticmethod
    def __groupDiffractionSpots(df,tolerance=1e-3):
        """
        :param df: data frame object containing the diffraction data
        :return:
        """

        radius = 1.
        x, y = [item[0] for item in df["XY"].tolist()], [item[1] for item in df["XY"].tolist()]
        markedList = []
        groupList = []
        for i, point in enumerate(zip(x, y)):
            if i not in markedList:
                currentList = [i]
            else:
                currentList = []
            markedList.append(i)
            for j in range(i, len(x)):
                if j not in markedList:
                    dist = (point[0] - x[j]) ** 2 + (point[1] - y[j]) ** 2
                    if dist < tolerance:
                        currentList.append(j)
                        markedList.append(j)
            if len(currentList) > 0:
                groupList.append(currentList)

        df['OverLappingSpotIds'] = df['OverLappingSpotIds'].astype(
            'object')  ### this will allow us to store arbitary list values into this column
        for i,group in enumerate(groupList):
            nSpots = len(group)
            angles = 2*np.pi/nSpots*np.arange(nSpots)
            count= 0
            for j,spot in enumerate(group): ### for counting how many allowed spots are present at this location for helping in labelling
                if not df["absentSpot"][spot]:
                    count+=1

            for j,spot in enumerate(group):
                labelOffset = [radius*np.cos(angles[j]),radius*np.sin(angles[j])]
                df.at[spot, "LabelPosition"]=labelOffset
                df.at[spot, "LableAngle"]=angles[j]*180/np.pi
                df.at[spot, "SpotGroupId"]= i
                df.at[spot, "SpotSubId"] = j
                df.at[spot, "NumberOfOverLappingSpots"]=count
                df.at[spot, "OverLappingSpotIds"] = group ### 0 elment is the same spot itself and hence we store from 1 onwards

        return df


