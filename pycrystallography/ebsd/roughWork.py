import numpy as np
from PIL import Image
from skimage.transform import (hough_line, hough_line_peaks,
                                probabilistic_hough_line, radon) 
from skimage.feature import canny 
from skimage import data, img_as_float 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from scipy import ndimage as ndi 
from skimage.feature import peak_local_max 
import pandas as pd

from sklearn import preprocessing as prep



def extents(f):
   delta = f[1] - f[0]
   return [f[0] - delta/2, f[-1] + delta/2]


#image = Image.open(r'..\..\data\ebsdData\2lines_test.tif')
image = np.zeros((100,100))
idx = np.arange(25, 75)
# image[idx[::-1], idx] = 255
# image[idx, idx] = 255
image[:,95]=255
#image[:,50] = 255



thetaStart=-9.9999
theta = np.arange(thetaStart,180,1)
h = radon(image,theta,circle=False)
h_b= radon(np.ones_like(image),theta,circle=False)

fig, axes = plt.subplots(1, 4, figsize=(15, 6)) 
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Pattern')
#ax[0].set_axis_off()

ax[1].imshow(h,cmap='gray')
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('equal')

ax[2].imshow(h/h_b,cmap='gray')
ax[2].set_title('ratio')
ax[2].set_xlabel('Angles (degrees)')
ax[2].set_ylabel('Distance (pixels)')
ax[2].axis('equal')


#finding maxima or peaks in Radon space
im = img_as_float(h/h_b)
im = np.nan_to_num(im)
image_max = ndi.maximum_filter(im, size=15, mode='constant') 
coordinates = peak_local_max(im, min_distance=5,num_peaks=1)
#num_peaks=8,threshold_rel=0.1
coordinatesOri = coordinates.copy()
coordinatesOri[:,1]=coordinatesOri[:,1]+thetaStart

print("result=", coordinatesOri)

ax[3].imshow(im, cmap='gray')
ax[3].autoscale(False)
ax[3].plot(coordinatesOri[:, 1], coordinatesOri[:, 0], 'r.')
#ax[3].axis('off')
ax[3].set_title('Peak local max')

plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 6)) 
ax = axes.ravel()
for i in range(0, coordinatesOri.shape[0]):
    rho, t=coordinatesOri[i,:]
    a, b = np.cos(t*np.pi/180), np.sin(t*np.pi/180)
    x0,y0 = a*rho,b*rho
    p1 = [round(x0+100*(-b)), round(y0+100*(a))]
    p2 = [round(x0-100*(-b)), round(y0-100*(a))]
    linePoints = np.array([p1,p2])

    ax[1].plot(linePoints[:,0], linePoints[:,1])
    
    print(p1,p2)
ax[0].imshow(image)
plt.show()


#coordinates
