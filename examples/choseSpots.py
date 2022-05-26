# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:55:59 2017

@author: Admin
"""

from __future__ import division, unicode_literals

import sys
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))


from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane



from pycrystallography.core.orientation  import Orientation
import pycrystallography.utilities.graphicUtilities as gu

import numpy as np
import copy
from math import pi, sqrt
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from copy import deepcopy
from tabulate import tabulate
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import sympy.geometry as gm




tt = np.arange(0.0, 5.0, 0.5)
t  = np.append(tt,tt)
ss = np.cos(2*np.pi*tt)
s  = np.append(ss,ss+2)
X  = np.array([t,s]).T  

xc = (X[0,0]+X[-1,0])/2 
yc = (X[0,1]+X[-1,1])/2
origin = [xc,yc]
vec1 = [0+xc,2+yc]
vec2 = [2+xc,0+yc]
 
#print (gu.PlotG1G2(X,origin, vec1, vec2))


from matplotlib.widgets import  EllipseSelector
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(-2.0, 2.0, 0.001)
s = t ** 2
initial_text = "t ** 2"
l, = plt.plot(t, s, lw=2)


def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, 'Evaluate', initial=initial_text)
text_box.on_submit(submit)

plt.show()
