'''
Created on 29-Nov-2016

@author: K V Mani Krishna
'''
from pymatgen.core.lattice import Lattice
# from pycrystallography.core.millerPlane import MillerPlane
# from pycrystallography.core.millerDirection import MillerDirection
import math as mt
import numpy as np
import matplotlib.pyplot as plt

from imutils import contours
#from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import itertools
import numpy
#from sympy import *
from sympy.geometry import *
#from pycrystallography.core.millerDirection import MillerDirection




def broken2DLineWithText(point1,point2,axisHandle=None,text='',lineFraction=0.5,ls='-',lc=''):
    """
    A utility function to construct a line annotation object for nice display of the lines 
    point1 and point2 are np arrays of 1X2 size indicating starting and end point so the line to be drwan
    0<lineFraction<1 if specified, fractional position of the text along the line
    if lineFraction<-0., random position is asigned along the line 
    text = string to be printed 
    """
#     point2=np.random.random(2)
#     point1=np.random.random(2)
#     point2 = np.array([1,0.5])
#     point1 = np.array([0,0])
    v = np.array(point2-point1)
    lineLength = np.linalg.norm(v)
    
    if lineLength>1e-5:
        v_dir = v/lineLength
        if  lineFraction < 0. :
            lineFraction = np.random.random()
            
        textPosition=lineFraction*lineLength
            
    #     ax = plt.gca()
    #     ax.set_xlim(-4,4)
    #     ax.set_ylim(-4,4)
        x = [point1[0],point2[0]]
        y = [point1[1],point2[1]]
        
        textPosition = point1+v_dir*textPosition
        #textPosition = np.array([point1[0]+v_dir[0]*textPosition, point1[1]+v_dir[1]*textPosition])
        textAngle = np.arctan2(v[1], v[0])*180./np.pi
        
        #print(point1, point2, v,v_dir,textAngle,textPosition)
        if len(lc)==0:
            lc=[0,0,0]
            
        if axisHandle is None:
            plt.plot(x,y,ls=ls,color=lc,picker=True)
        else:
            axisHandle.plot(x,y,ls=ls,color=lc,picker=True)
        #print(textPosition)
        if not len(text)==0:
            if axisHandle is None:
                plt.annotate(
                        text,xy=(textPosition[0], textPosition[1]), xytext=(5, 5),
                        textcoords='offset points', ha='center', va='center',
                        rotation=textAngle,size=12,color=lc)  
            else:
                axisHandle.annotate(
                        text,xy=(textPosition[0], textPosition[1]), xytext=(5, 5),
                        textcoords='offset points', ha='center', va='center',
                        rotation=textAngle,size=12,color=lc)  
    else:
        print("Warning !!! two points are too close and hence ignoring !!!! no line will be produced")
      

def plotKikuchiLinesFromPoints(point1,point2,axisHandle=None,plotBand=False, lineWidth=1.0,text='',lineFraction=0.5,ls='--',lc=''):
    """
    given two kikuchi points, draws them on the screen by calcualting the double lines
    """
    
    if not isinstance(point1,np.ndarray):
        point1 = np.array(point1)
    if not isinstance(point2,np.ndarray):
        point2 = np.array(point2)
        
    if plotBand:    ### plots two parllel lines representing the kikuchi band
        unitVector = point1-point2
        #print(unitVector,point1,point2,type(unitVector))
        mag = np.linalg.norm(unitVector)
        if mag>1e-5:
            unitVector= unitVector/mag
            
            perpVector = np.array([-unitVector[1],unitVector[0]])
            
            p1X = point1[0]+perpVector[0]*lineWidth
            p1Y = point1[1]+perpVector[1]*lineWidth
            p2X = point2[0]+perpVector[0]*lineWidth
            p2Y = point2[1]+perpVector[1]*lineWidth
            point1 = np.array([p1X,p1Y])
            point2 = np.array([p2X,p2Y])
            
            broken2DLineWithText( point1,point2,axisHandle=axisHandle,text=text,lineFraction=lineFraction,ls=ls,lc=lc)
            
            p1X = point1[0]-perpVector[0]*lineWidth
            p1Y = point1[1]-perpVector[1]*lineWidth
            p2X = point2[0]-perpVector[0]*lineWidth
            p2Y = point2[1]-perpVector[1]*lineWidth
            point1 = np.array([p1X,p1Y])
            point2 = np.array([p2X,p2Y])
            
            broken2DLineWithText( point1,point2,axisHandle=axisHandle,text='',ls=ls,lc=lc)
            
        else:
            print("Warning !!! two points are too close and hence ignoring !!!! no line will be produced")
      
    else:
        broken2DLineWithText( point1,point2,axisHandle=axisHandle,text=text,lineFraction=lineFraction,ls=ls,lc=lc)
        
             





def extractSpotDatafromSAEDpattern(imageFile):
    """
    Function to get extract the SAD spots info for enabling the indexing.
    The returned results is in the form of a list of dictionary with each elment of list corresponding 
    to the Spot info
    Folowing are the feilds of the Dictionary
    "LineIds" : str identifying the spots for e.g 1&2
    "d_ratio" : the ratio of the lenghths of the reciprocal vectors representig the two spots
    "Angle" : The angle between the two spots in degrees
    """

    options = {"GAUSSIAN_BLUR_SIZE": 11,
           "THRESHOLD_LOW":50,
           "THRESHOLD_HIGH": 200,
           "NUMBER_OF_SPOTS_TO_DETECT":3,
           "MIN_ANGLE_OF_DETECTED_SPOTS": 10, # This is the min angle in deg between two detected spots (planes) 
                                              # which need to exceed for the 
                                              # two spots to be considered as measure data
           }

# construct the argument parse and parse the arguments

    print(imageFile)
    image = cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
    blurred = cv2.GaussianBlur(blurred, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
    thresh = cv2.threshold(blurred, options["THRESHOLD_LOW"], options["THRESHOLD_HIGH"], cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue 
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        print("detected!!!!, numPixels=", numPixels)
    
        if numPixels > 200:
            mask = cv2.add(mask, labelMask)   
            
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
    imageCentre = np.asarray([int(image.shape[1]/2), int(image.shape[0]/2)])
     
    # loop over the contours
    spotPositions = np.zeros((len(cnts),2),dtype = np.int16)
    spotVectors = np.zeros((len(cnts)-1,2), dtype = np.float64 )
    distFromCentre = np.zeros((len(cnts),1),dtype = np.float64)
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)    
        M = cv2.moments(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(mask, (int(cX), int(cY)), int(radius),
            (255), 3)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        spotPositions[i,:]=np.array([cX,cY])
        distFromCentre[i] = np.linalg.norm(spotPositions[i]-imageCentre)  
    
    indxs = np.argsort(distFromCentre,axis=0).tolist()
    
     
   # indxs = indxs[0:options["NUMBER_OF_SPOTS_TO_DETECT"]+1]
    plt.imshow(image)
    print("The spot id sare ", indxs)
    lineData=[]
    for i in range(0,len(indxs)-1) :
        spotVectors[i,:] = spotPositions[indxs[i+1]]-spotPositions[indxs[0]] ## always the spotPositions[indxs[0]] is the central spot
        lineData.append({"Lines":Segment(Point(0,0),Point(spotVectors[i,0],spotVectors[i,1])),
                         "LineId":str(i+1)})
        plt.annotate(
                    str(i+1),xy=(spotPositions[indxs[i+1],0], spotPositions[indxs[i+1],1]), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='center',color=[1,1,1],size=16,
                    bbox=dict(facecolor=[0,0,0], edgecolor='white', pad=0.0)
                    )
        x = np.asarray([float(spotPositions[indxs[i+1],0]),float(spotPositions[indxs[0],0])])
        y = np.asarray([float(spotPositions[indxs[i+1],1]),float(spotPositions[indxs[0],1])])
        #plt.plot(x,y,'w')
        point1 = np.array([x[0],y[0]]) 
        point2 = np.array([x[1],y[1]])
        lineLength =float(Segment(Point(0,0),Point(spotVectors[i,0],spotVectors[i,1])).length) 
        print("The length = {:.2f}".format(lineLength))
        broken2DLineWithText(point1,point2,"L= {:.2f}".format(lineLength),lineFraction=0.5,lc='w')
                        
    #spotData = {"pc":spotPositions[indxs[0]], "spots":spotPositions[indxs[1:]]}        
    imageData={"patternCenter":spotPositions[indxs[0]],"imageData":image}
    spotData=[]
    #spotList = list(indxs[1:len(indxs)])
    print(lineData)
    count=0
    for pair in itertools.combinations(lineData,2): 
          
        #print("The pair", pair, pair[0].length, pair[1].length)
        line1 = pair[0]["Lines"]
        line2 = pair[1]["Lines"]
        lineIds = pair[0]["LineId"]+"&"+pair[1]["LineId"]
        
        d_ratio = float(max(line1.length/line2.length, line2.length/line1.length))
        #print("d_ratio", d_ratio)
        ang = float(line2.angle_between(line1))*180./np.pi
        print("angle=", ang)       
#         if ang>90.:
#             ang = 180-ang
        
        if (ang>options["MIN_ANGLE_OF_DETECTED_SPOTS"]):
            spotData.append({"LineIds":lineIds,"LineData":[line1,line2],
                                     "d_ratio":d_ratio,"Angle": ang})
            count+=1
        if count>(options["NUMBER_OF_SPOTS_TO_DETECT"]+1) :
            break
    
    spotDataLabels= ["The Spot IDs, Lenght Ratios, Angle"]
    
    for i in spotData:
        print(i)
        strLine = "{:s} {:.2f} {:.1f}".format(i["LineIds"],i["d_ratio"],i["Angle"])
        #print(strLine)
        spotDataLabels.append(strLine)
        
    plt.legend(spotDataLabels)
    plt.gray()
    plt.show()
    print("done")
    
    if len(spotData)==0:
        raise ValueError ("Not spots cpuld be identified Try changing the options !!!")
    
    else:        
        return {"imageData":imageData, "spotData":spotData}
        
        
#     plt.imshow(blurred, cmap = 'gray', interpolation = 'bicubic')
#     #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.show()

def onclick(event):
    if event.inaxes is not None:
        pass
    else:
        print ('Clicked ouside axes bounds but inside plot window')    
        plt.close()

def calcClosestDatapoint(X, event):
    distances = [np.linalg.norm(np.array([event.xdata,event.ydata])-X[i]) for i in range(X.shape[0])]
    id = np.argmin(distances)
    return (id,distances[id])

def callGvectors(fig1,ax,origin = (0.0,0.0),vec1 = (1.0,0.0),vec2=(0.0,1.0)): 
    
    
    xc,yc = origin[0],origin[1]
    arrowTip1 =vec1 
    arrowTip2 =vec2
   
    
    bbox_args = dict(boxstyle="circle", pad=0.0, facecolor="None", edgecolor='red')   
    arrow_args = dict(width=2, shrink=0.00,facecolor=[1,1,1], edgecolor='black')
    
    an11 = ax.annotate('', xy=(xc,yc), xycoords='data',bbox=bbox_args,)
    an11.draggable()
    
    an12 = ax.annotate('+', xy=(xc,yc), xycoords=an11, xytext=(arrowTip1),textcoords='data',bbox=bbox_args,arrowprops=dict(arrowstyle="simple",fc=(1,1,1)))
    an12.draggable()
    
    an13 = ax.annotate('+',size=14, xy=(xc,yc), xycoords=an11,xytext=(arrowTip1[0]+arrowTip1[0]*0.2,arrowTip1[1]+arrowTip1[1]*0.2),textcoords=an11,bbox=bbox_args,ha="center", va="center")
    an14 = ax.annotate('g1',size=14, xy=(xc,yc), xycoords=an11,xytext=(0.5,0.7),textcoords=an12,ha="center", va="center")
   #an13.draggable()

    an21 = ax.annotate('', xy=(xc,yc), xycoords='data',bbox=bbox_args,)

    
    an22 = ax.annotate('+', xy=(xc,yc), xycoords=an11, xytext=(arrowTip2),textcoords='data',bbox=bbox_args,arrowprops=dict(arrowstyle="simple",fc=(1,1,1)))
    an22.draggable()
    
    an23 = ax.annotate('+',size=14, xy=(xc,yc), xycoords=an11,xytext=(arrowTip2[0]+arrowTip2[0]*0.2,arrowTip2[1]+arrowTip2[1]*0.2),textcoords=an11,bbox=bbox_args,ha="center", va="center")
    an24 = ax.annotate('g2',size=14, xy=(xc,yc),xytext=(0.7,0.5),textcoords=an22,ha="center", va="center")
    cid = fig1.canvas.mpl_connect('button_release_event', onclick)
    
    
    plt.show()
    
    GvecCoords = np.zeros(shape=(4,2))
    GvecCoords[0,:] = an11.get_position()
    GvecCoords[1,:] = an12.get_position()
    GvecCoords[2,:] = an11.get_position()
    GvecCoords[3,:] = an22.get_position()
    
    
    
    return GvecCoords

def PlotG1G2(X, origin, vec1, vec2):
    fig1 = plt.gcf()
    ax   = plt.gca()
    line = ax.scatter(X[:,0],X[:,1],s=120, facecolors='none', edgecolors='r')
    GvecCoords = callGvectors(fig1,ax,origin,vec1,vec2)
    return GvecCoords    

def generate2Dlattice(origin=[0.,0.,],vec1=[1.0,0],vec2=[0.,1.],maxIndices= 10,latticeBounds=None,plotOn=False):
        
    checkRadius=False
    if latticeBounds is  None:
        latticeBounds = [-1e10,1e10,-1e10,1e10]
    if not isinstance(latticeBounds,list) :## case of specifying the maximum distance in reciprocal space
        checkRadius = True
        latticeRadius = latticeBounds 
    if (isinstance(latticeBounds,list) and  len(latticeBounds)==1) :
        checkRadius = True
        latticeRadius = latticeBounds[0]
        
    if isinstance(origin,list):
        origin = np.array(origin)            
    if isinstance(vec1,list):
        vec1 = np.array(vec1)
    if isinstance(vec2,list):
        vec2 = np.array(vec2)
    
    a = range(-(maxIndices),maxIndices+1)
    latticePoints=[]
    vec1 = vec1-origin
    vec2 = vec2-origin

    for combination in itertools.product(a, a):            
        p = combination[0]*vec1+combination[1]*vec2+origin
        if checkRadius: ## checking to see if th espot falls within the circle defined by pc and patternBound as radius
                dist = np.sqrt(np.sum((p-origin)*(p-origin)))
                if dist <latticeRadius:
                        latticePoints.append(p) 
        else: ### checking by rectangular bounds
            if p[0]>=latticeBounds[0] and p[0]<=latticeBounds[1] and p[1]>=latticeBounds[2] and p[1]<=latticeBounds[3] :
                        latticePoints.append(p) 

    if plotOn:
        latticePoints=np.array(latticePoints)
        plt.scatter(latticePoints[:,0],latticePoints[:,1], s=120, facecolors='none', edgecolors='r')
        plt.title("2dLattice")
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
    return  latticePoints   


if __name__ == "__main__":
    import doctest
    import random  # noqa: used in doctests
    numpy.set_printoptions(suppress=True, precision=5)
    doctest.testmod()
    #q = quaternion_from_euler(90*numpy.pi/180,0,0,'rzxz')
    print("All test are done")
