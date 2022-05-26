### script for generating the uniform orientation grid
# as of now primarily indended for genrasting the deformation directios for MD simulaitons

import sys
import os, pathlib
import matplotlib
#matplotlib.use('TkAgg')
cwd = pathlib.Path(__file__).parent.parent.resolve()
pyCrystallographyDir = cwd
print(f"cwd={cwd} pyCrystallographyDir={pyCrystallographyDir}")
import matplotlib.pyplot as plt
from pymatgen.util.testing import PymatgenTest
from tabulate import tabulate
import sympy as sp
try:
    from pycrystallography.core.orientation  import Orientation
    from pycrystallography.core.quaternion  import Quaternion
    from pycrystallography.core.millerDirection  import MillerDirection
    from pycrystallography.core.millerPlane  import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
    from pycrystallography.core.orientationRelation  import OrientationRelation as OriReln
    from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    import pycrystallography.utilities.pyCrystUtilities as pyCrysUt
    import pycrystallography.utilities.pymathutilityfunctions as pmut
except:
    print("Unable to find the pycrystallography package!!! trying to now alter the system path !!")
    sys.path.insert(0, os.path.abspath('.'))
    sys.path.insert(0, os.path.dirname('..'))
    sys.path.insert(0, os.path.dirname('../../pycrystallography'))
    sys.path.insert(0, os.path.dirname('../../..'))
    for item in sys.path:
        print(f"Updated Path : {item}")
    from pycrystallography.core.orientation  import Orientation
    from pycrystallography.core.quaternion  import Quaternion
    from pycrystallography.core.millerDirection  import MillerDirection
    from pycrystallography.core.millerPlane  import MillerPlane
    from pycrystallography.core.orientedLattice import OrientedLattice as olt
    from pycrystallography.core.crystalOrientation  import CrystalOrientation as CrysOri
    from pycrystallography.core.orientationRelation  import OrientationRelation as OriReln
    from pycrystallography.core.saedAnalyzer import SaedAnalyzer as Sad
    from pycrystallography.core.crystallographyFigure import CrystallographyFigure as crysFig
    import pycrystallography.utilities.pyCrystUtilities as pyCrysUt
    import pycrystallography.utilities.pymathutilityfunctions as pmut


import pymatgen as pm
from pymatgen.analysis.diffraction.xrd import XRDCalculator as Xrd


import numpy as np
from math import pi
from pycrystallography.utilities.pymathutilityfunctions import integerize
import os
import logging
import pandas as pd

def integerisedMillerDirectionFromCartesian(uvw,lat):
    # if lat.is_hexagonal():
    #     if len(uvw)==4:
    #         uvw[2] = -(uvw[1]+uvw[2]) ## just correct any mistake in u+v=-t condition beacuse of integerization
    millerUVW = MillerDirection(vector=uvw, lattice=lat, isCartesian=True)

    millerUVWInt, error = millerUVW.integerize(AngleDeviationUnits='deg')
    if lat.is_hexagonal():
        millerUVWInt[2] = -(millerUVWInt[0]+millerUVWInt[1]) ## just correct any mistake in u+v=-t condition beacuse of integerization
    millerUVWInt = MillerDirection(vector=millerUVWInt, lattice=lat)
    return millerUVWInt,np.around(error,2)

lat = olt.hexagonal(3.23, 3.23*1.59)
degree=np.pi/180
fundamentalRegionProperties = {"azimuthRange":[0,30], "tiltRange":[0,90], 'resolution' : 5 ,## all angles are in degrees
                               "tiltAxis":[0,1,-1,0], "inPlaneRotationAxis":[0,0,0,1],
                               "ipfSampleDirection":[0,0,1],
                               "fundamentalRegionVertices":[[0,0,0,1],
                                                            [2,-1,-1,0],
                                                            [1,0,-1,0]
                                                            ]
                               }

step = fundamentalRegionProperties["resolution"]
azimuthAngles = np.arange(fundamentalRegionProperties["azimuthRange"][0],fundamentalRegionProperties["azimuthRange"][1]+step, step)
tiltAngles = np.arange(fundamentalRegionProperties["tiltRange"][0],fundamentalRegionProperties["tiltRange"][1]+step, step)
tilt_X, azimuth_Y = np.meshgrid(tiltAngles,azimuthAngles, indexing='ij')
# tilt_X = tilt_X.flatten()
# azimuth_Y = azimuth_Y.flatten()
result=[]
result2=[]
for i in range(tiltAngles.size):
    for j in range(azimuthAngles.size):
        tiltAxis = MillerDirection(vector=fundamentalRegionProperties["tiltAxis"], lattice=lat)
        inPlaneRotationAxis = MillerDirection(vector=fundamentalRegionProperties["inPlaneRotationAxis"], lattice=lat)
        inPlaneOri = Orientation(axis=inPlaneRotationAxis.getCartesianVec(),angle=azimuthAngles[j]*degree)
        tiltOri = Orientation(axis=tiltAxis.getCartesianVec(),angle=tiltAngles[i]*degree)
        totalOri = inPlaneOri*tiltOri
        rotatedVec = totalOri.rotate(fundamentalRegionProperties["ipfSampleDirection"])
        rotatedX = totalOri.rotate([1., 0, 0])
        rotatedY = totalOri.rotate([0.,1.,0.])
        rotatedZ = totalOri.rotate([0.,0.,1.])

        # rotatedX = MillerDirection(vector=rotatedX, lattice=lat,isCartesian=True)
        # rotatedXInt,errorX = rotatedX.integerize()
        # rotatedX = MillerDirection(vector=rotatedXInt, lattice=lat)
        rotatedX, errorX = integerisedMillerDirectionFromCartesian(rotatedX, lat)
        rotatedY, errorY = integerisedMillerDirectionFromCartesian(rotatedY, lat)
        rotatedZ, errorZ = integerisedMillerDirectionFromCartesian(rotatedZ, lat)

        angleXY = np.around(rotatedX.angle(rotatedY,units='deg',considerSymmetry=False),2)
        angleYZ = np.around(rotatedY.angle(rotatedZ,units='deg',considerSymmetry=False),2)
        angleZX = np.around(rotatedZ.angle(rotatedX,units='deg',considerSymmetry=False),2)

        rotatedDir = MillerDirection(vector=pmut.integerize(np.array(rotatedVec), rtol=1e-2, atol=1e-1), lattice=lat,
                                     isCartesian=True)

        xMag, yMag,zMag = np.around(rotatedX.mag,3), np.around(rotatedY.mag,3), np.around(rotatedZ.mag,3)

        result2.append({"index":f'{i}_{j}' , "Tilt": tiltAngles[i], "azimuth": azimuthAngles[j], "Orientation": totalOri, "LoadingDir": rotatedDir,
                        "crystalX":rotatedX, "crystalY":rotatedY,"crystalZ":rotatedZ,"angleXY":angleXY, "angleYZ":angleYZ, "angleZX":angleZX,
                        "xMag":xMag, "yMag":yMag, "zMag":zMag,
                        "VectDev":{"X":errorX, "Y":errorY, "Z":errorZ, }})
        result.append({ "Orientation": totalOri, "Tilt": tiltAngles[i], })

df = pd.json_normalize(result)
df2 = pd.json_normalize(result2)
print(df)
df2.to_html(r'../../tmp/uniformGridOut.html', index=False)# C:\Users\Admin\PycharmProjects\pycrystallography\tmp C:\Users\Admin\PycharmProjects\pycrystallography\pycrystallography\apps\generateUniformOrientationGrid.py
df.to_csv( r'../../tmp/uniformGridOut.txt',index=False)# C:\Users\Admin\PycharmProjects\pycrystallography\tmp C:\Users\Admin\PycharmProjects\pycrystallography\pycrystallography\apps\generateUniformOrientationGrid.py
print('done')


