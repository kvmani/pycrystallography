'''
Created on 22-Jan-2018

@author: K V Mani Krishna
'''

from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
from matplotlib import pyplot as plt
import itertools
from sympy import *
from sympy.geometry import *
import pycrystallography.utilities.graphicUtilities as gu


options = {"GAUSSIAN_BLUR_SIZE": 11,
           "THRESHOLD_LOW":50,
           "THRESHOLD_HIGH": 200,
           "NUMBER_OF_SPOTS_TO_DETECT":5,
           "MIN_ANGLE_OF_DETECTED_SPOTS": 10, # This is the min angle in deg between two detected spots (planes) 
                                              # which need to exceed for the 
                                              # two spots to be considered as measure data
           }

# construct the argument parse and parse the arguments

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

    image = cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
    blurred = cv2.GaussianBlur(image, (options["GAUSSIAN_BLUR_SIZE"], options["GAUSSIAN_BLUR_SIZE"]), 0)
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
    
        if numPixels > 200:
            mask = cv2.add(mask, labelMask)   
            
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = contours.sort_contours(cnts)[0]
    imageCentre = np.asarray([int(image.shape[0]/2), int(image.shape[1]/2)])
     
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
    indxs = indxs[0:options["NUMBER_OF_SPOTS_TO_DETECT"]+1]
    plt.imshow(image)
    print(indxs)
    lineData=[]
    for i in range(0,len(indxs)-1) :
        spotVectors[i,:] = spotPositions[indxs[i+1]]-spotPositions[indxs[0]] ## always the spotPositions[indxs[0]] is the central spot
        lineData.append({"Lines":Segment(Point(0,0),Point(spotVectors[i,0],spotVectors[i,1])),
                         "LineId":str(i+1)})
        plt.annotate(
                    str(i+1),xy=(spotPositions[indxs[i+1],0], spotPositions[indxs[i+1],1]), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='center',color=[1,1,1]
                    )
        x = np.asarray([float(spotPositions[indxs[i+1],0]),float(spotPositions[indxs[0],0])])
        y = np.asarray([float(spotPositions[indxs[i+1],1]),float(spotPositions[indxs[0],1])])
        #plt.plot(x,y,'w')
        point1 = np.array([x[0],y[0]]) 
        point2 = np.array([x[1],y[1]])
        lineLength =float(Segment(Point(0,0),Point(spotVectors[i,0],spotVectors[i,1])).length) 
        print("The length = {:.2f}".format(lineLength))
        gu.broken2DLineWithText(point1,point2,"{:.2f}".format(lineLength),lineFraction=0.5,lc='w')
                        
            
    saedMeasuredData=[]
    #spotList = list(indxs[1:len(indxs)])
    print(lineData)
    for pair in itertools.combinations(lineData,2):   
        #print("The pair", pair, pair[0].length, pair[1].length)
        line1 = pair[0]["Lines"]
        line2 = pair[1]["Lines"]
        lineIds = pair[0]["LineId"]+"&"+pair[1]["LineId"]
        
        d_ratio = float(max(line1.length/line2.length, line2.length/line1.length))
        #print("d_ratio", d_ratio)
        ang = float(line1.angle_between(line2)*180./np.pi)
        if (ang>options["MIN_ANGLE_OF_DETECTED_SPOTS"]):
            saedMeasuredData.append({"LineIds":lineIds,"d_ratio":d_ratio,"Angle": ang})
    spotDataLabels= ["The Spot IDs, Lenght Ratios, Angle"]
    for i in saedMeasuredData:
        print(i)
        strLine = "{:s} {:.2f} {:.1f}".format(i["LineIds"],i["d_ratio"],i["Angle"])
        print(strLine)
        spotDataLabels.append(strLine)
        
    plt.legend(spotDataLabels)
    plt.gray()
    plt.show()
    print("done")
    
    if len(saedMeasuredData)==0:
        raise ValueError ("Nop spots cpuld be identified Try changing the options !!!")
    
    else:
        return saedMeasuredData
        
        
    # plt.imshow(blurred, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()




    