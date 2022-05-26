import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycrystallography.utilities.graphicUtilities as gu
import numpy as np
import pycrystallography.utilities.pymathutilityfunctions as pmt
import warnings


markers = ['o','*','s','d',]
edgecolors = ['w','w','w','r']
annotationPositions = [(0.9,0.5),(0.9,0.1),(0.1,0.1),(0.1,0.9)]
class DraggablePoint:
    lock = None #only one can be animated at a time
    def __init__(self, point):
        self.point = point
        self.press = None
        self.background = None
        self._parent=None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event',  self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event',    self.on_motion)

    def setParent(self,parent):
        #print("The parent is being set", parent)
        self._parent=parent

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        #print(f"dx={dx} {dy} {xpress} {ypress} {event.xdata} {event.ydata} {self.point.center}")

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)
        if self._parent is not None:
            self._parent.plot2dLattice()
        else:
            print("None object")

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        if self._parent is not None:
            self._parent.plot2dLattice()
            self._parent._updateSpotPositions()
        else:
            print("None object")
        self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)



class DraggablePoints:
    def __init__(self,imageData,initialPoints=None,axisHandle=None,showPlot=False,
                 ):
        imSize = imageData.shape
        self._imData=imageData
        length = min(imSize) * .01
        if initialPoints is None:
            centre = np.array([1.0 * imSize[1] / 2, 1.0 * imSize[0] / 2])
            spot1 = centre + np.array([0.0, length]) * (0.5 + np.random.random())
            spot2 = centre + np.array([length, 0.]) * (0.5 + np.random.random())
            initialPoints = [centre,spot1,spot2 ]

        if axisHandle is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', 'box')
            self._axisHandle=ax
        else:
            self._axisHandle=axisHandle

        drs=[]
        for ii in range(len(initialPoints)):
            if initialPoints is None:
                centre = np.array([1.0*imSize[1]/2, 1.0*imSize[0]/2])
                spot1 = centre+np.array([0.0,length])*(0.5+np.random.random())
                spot2 = centre+np.array([length,0.])*(0.5+np.random.random())
            else:
                initialPoint=initialPoints[ii]
                centre= initialPoint[0]
                spot1 = initialPoint[1]
                spot2 = initialPoint[2]

            spotData = [centre,spot1,spot2]
            colorData = ['r','g','b']
            circleRadius = 10*length/5.

            for i, spot in enumerate(spotData):
                circ = patches.Circle((spot[0],spot[1]),circleRadius,
                        fc = colorData[i],alpha=0.9)
                self._axisHandle.add_patch(circ)
                dr = DraggablePoint(circ)
                dr.setParent(self)
                dr.connect()
                drs.append(dr)
        self.points = drs
        self._updateSpotPositions()
        self.plot2dLattice()
        if showPlot:
            plt.show()


    def plot2dLattice(self):
        self._updateSpotPositions()

        self._axisHandle.clear()
        for ii in range(int(len(self.xyList)/3)):
            origin,vec1,vec2 = self.xyList[ii*3+0], self.xyList[ii*3+1],self.xyList[ii*3+2]
            origin = np.array(origin)
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            self._axisHandle.annotate(f"{ii}", xy=(origin), xytext=(0, 0),
                        textcoords='offset points', ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor=[1, 1, 1], edgecolor='white', pad=0.0))
            self._axisHandle.annotate(f"{ii}", xy=(vec1), xytext=(0, 0),
                        textcoords='offset points', ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor=[1, 1, 1], edgecolor='white', pad=0.0))
            self._axisHandle.annotate(f"{ii}", xy=(vec2), xytext=(0, 0),
                        textcoords='offset points', ha='center', va='center', fontsize=10,
                        bbox=dict(facecolor=[1, 1, 1], edgecolor='white', pad=0.0))


            latticePoints = gu.generate2Dlattice(origin, vec1, vec2,maxIndices=3)
            latticePoints= np.array(latticePoints)
            marker, edgecolor = markers[ii], edgecolors[ii]
            self._axisHandle.scatter(latticePoints[:,0],latticePoints[:,1],s=200, marker = marker,
                                     facecolors='none', lw=1,
                                     edgecolors=edgecolor,label=f"phase{ii}")

            #print(f"Scatter plot :{ii}, {marker} {edgecolor}")
        self._axisHandle.imshow(self._imData,cmap="gray", origin='upper')
        warnings.warn("Assuming the image is a gray scale one without any color channels. Plot limits are set based on the image dimensions  assuming a 2d array!!!")
        xlim = 0, self._imData.shape[1]
        ylim = 0, self._imData.shape[0]

        self._axisHandle.set_xlim(xlim)
        self._axisHandle.set_ylim(ylim)
        # self._axisHandle.invert_xaxis()
        self._axisHandle.invert_yaxis()

        self._axisHandle.legend()
        self.addArrows()
        for point in self.points:
            self._axisHandle.add_patch(point.point)

    def addArrows(self,showAngle=True):

        for ii in range(int(len(self.xyList)/3)):
            arrowProps = {'width':3,'length_includes_head':True,
                          'color':'white' }
            p1,p2,p3 = self.xyList[ii*3+0], self.xyList[ii*3+1], self.xyList[ii*3+2]
            arrow1_base = np.array(p1)
            arrow1_head = np.array(p2)
            arrow1_head = arrow1_head-arrow1_base
            self._axisHandle.arrow(arrow1_base[0],arrow1_base[1],arrow1_head[0],arrow1_head[1],**arrowProps,)
            arrow2_base = np.array(p1)
            arrow2_head = np.array(p3)
            arrow2_head = arrow2_head-arrow2_base
            self._axisHandle.arrow(arrow2_base[0],arrow2_base[1],arrow2_head[0],arrow2_head[1],**arrowProps)
            if showAngle:
                line1 = np.array([p1,p2])
                line2 = np.array([p1,p3])
            angleBetweenSpots = abs(pmt.angleBetween2Lines(line1,line2,units="Deg"))
            #if angleBetweenSpots>180:
            #    angleBetweenSpots=angleBetweenSpots-180.

            arcCentre =  self.xyList[0]
            fractionOfLength=0.75
            line1Length=np.linalg.norm(line1[0]-line1[1])
            line2Length=np.linalg.norm(line2[0]-line2[1])
            lengthRatio = max(line1Length/line2Length, line2Length/line1Length)

            arcWidth=line1Length*fractionOfLength
            arcHeight=np.linalg.norm(line2[0]-line2[1])*fractionOfLength
            horizontlaLine = np.array([self.xyList[0],[self.xyList[0][0],self.xyList[0][1]+10.]])
            arcAngle = abs(pmt.angleBetween2Lines(horizontlaLine,line1,units="Deg"))
            arc=patches.Arc(self.xyList[0], width=arcWidth, height=arcHeight, angle=arcAngle,theta1=0.,theta2=angleBetweenSpots)
            self._axisHandle.add_patch(arc)
            text = f" phase {ii} : \nAngle = {angleBetweenSpots:.2f}\n D Ratio={lengthRatio:.2f} \n pointSet : " \
                   f"{np.around(p1,2)},\n {np.around(p2,2)}, \n {np.around(p3,2)} "
            annotationPosition = annotationPositions[ii]
            ann = self._axisHandle.annotate(text , xy = annotationPosition, xytext=(5,0),xycoords='axes fraction',textcoords='offset points', ha='center', va='center', bbox=dict(facecolor=[1,1,1], edgecolor='black', pad=0.0))
            ann.draggable()

    def returnSpotPositions(self):
        """
        returns the coordiantes of the 3 spots in ther form of a list
        """
        self._updateSpotPositions()
        pointsList = []

        for ii in range(int(len(self.xyList)/3)):
            pointSet=np.around(np.array([self.xyList[ii * 3 + 0], self.xyList[ii * 3 + 1], \
                       self.xyList[ii * 3 + 2]]),2)
            pointsList.append([pointSet])
        return pointsList

    def _updateSpotPositions(self):
        xyList=[]
        for point in self.points:
            #point.setParent(self)
            xyList.append(point.point.center)
        self.xyList = xyList


if __name__ == '__main__':
    import os
    from PIL import Image
    import numpy as np
    electronEnergy = 160e3 ### units eV
    lamda = 1.23e3/(np.sqrt(electronEnergy*(1+1.978e-7*electronEnergy))) #### ref:https://www.jeol.co.jp/en/words/emterms/search_result.html?keyword=wavelength%20of%20electron
    lamda = 2.8510e-12 ## units meters at 160 kV
    cameraConstant = lamda*1
    rMeasured = 21
    d = cameraConstant/rMeasured
    print(lamda, d)

    vec1 = np.array([4,4,4])
    vec2 = np.array([3,3,-6])
    cross = np.cross(vec1,vec2)
    print(vec1, vec2, "cross is = ", cross)

    exit(-10)

    dataChoice = "5596"
    patternData = {

                   "5608": {"imName": "5608.jpg",
                            "initialPoints":
                                [
                                    #[[381., 525.],[228.64, 476.59],[280.81, 577.39]], ## spot 1 & 2
                                    [[373.25, 475.9],[180.04, 476.12],[149.82, 366.9]] ## spots 4 and 5
                                    # [[381, 525.], [515., 565], [486, 476.]],
                                    # [[381, 525.], [248, 485.], [282, 580.]],
                                    # [[381, 525.], [248, 500.], [150, 400.]]
                                ]
                            },
                   "5611": {"imName": "5611.jpg",
                            "initialPoints":
                                [
                                    [[381, 525.], [432, 626], [332, 677.]],
                                    # [[381, 525.], [515., 565], [486, 476.]],
                                    # [[381, 525.], [248, 485.], [282, 580.]],
                                    # [[381, 525.], [248, 500.], [150, 400.]]
                                ]
                            },
                   "simulatedAlpha100": {"imName": "simulatedAlpha100.tif",
                            "initialPoints":
                                [
                                    [[381, 525.], [432, 626], [332, 677.]],
                                    [[381, 525.], [515., 565], [486, 476.]],
                                    [[381, 525.], [248, 485.], [282, 580.]],
                                    [[381, 525.], [248, 500.], [150, 400.]]
                                ]
                            },
                   "5596": {"imName": "5596.jpg",
                                         "initialPoints":
                                             [
                                                 #[[360.62, 518.21],  [267.65, 338.05], [228.77, 474.62]], ## spots 1, 2
                                                 # [[357.91, 516.85],[393.97, 405.96],[507.22, 436.59]], ## spots 3,4
                                                 # [[359.27, 515.49],[321.98, 312.24], [265.44, 345.58]], ## spot 2,7
                                                  #[[357.9 , 518.21],[506.71, 434.49],[276.31, 372.75]], ## spot 5,3
                                                  #[[357.9 , 518.21],[267.65, 338.05],[276.31, 372.75]], ## spot 2, 5
                                                  [[357.9 , 518.21],[267.65, 338.05],[276.31, 372.75]], ## spot 5,9


                                             ]
                                         },

                   "5607": {"imName": "5607.jpg",
                                         "initialPoints":
                                             [
                                                 [[361.98, 518.21],[266.29, 343.48],[190.74, 614.52]], ### spot 1,2
                                                 # [[357.91, 516.85],[393.97, 405.96],[507.22, 436.59]], ## spot 3,4
                                                 #[[359.27, 515.49],[321.98, 312.24], [265.44, 345.58]], ## spot 2,7
                                                  #[[381, 525.], [248, 500.], [150, 400.]]
                                             ]
                                         },

                   "5618": {"imName": "5618-1.jpg",
                            "initialPoints":
                                [
                                    [[381, 525.], [432, 626], [332, 677.]],
                                    # [[381, 525.], [515., 565], [486, 476.]],
                                    # [[381, 525.], [248, 485.], [282, 580.]],
                                    # [[381, 525.], [248, 500.], [150, 400.]]
                                ]
                            },

                   }
    inData = patternData[dataChoice]
    imPath = r'D:\CurrentProjects\colloborations\rakesh_MFD\TEM_U_9Mo_500C_48h'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    # imageData = (np.random.random((200,400))*255).T.astype(np.uint8)
    imName = os.path.join(imPath,inData["imName"])
    imageData = np.array(Image.open(imName))
    initialPoints = np.array(inData["initialPoints"])
    pp1 = DraggablePoints(imageData=imageData,initialPoints=initialPoints,axisHandle=ax,showPlot=True,
                          )
    pointSet = pp1.returnSpotPositions()
    print("Here is the data", pp1.returnSpotPositions())
    for point in pointSet:
        print(point)

