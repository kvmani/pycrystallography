### script for processing the MD data (of homofgenous nucleation paper).
### initial traget is to calcualte the angle with referene to reference direction (in response to reviwer comments)
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


import pymatgen as pm
from pymatgen.analysis.diffraction.xrd import XRDCalculator as Xrd
import pandas as pd
import numpy as np
dataPath = r'D:\CurrentProjects\molecualrDynamics\nucleation_analysis' ## folder where all the stress data is compiled in the form of file_list_Compression_Zr_300K.out files
fileNames = [ 'file_list_Compression_Zr_10K.out.modified',
            # 'file_list_Compression_Zr_300K.out.modified',
            #'file_list_Tensile_Zr_10K.out.modified',
             #'file_list_Tensile_Zr_300K.out.modified',#
             #'file_list_Compression_Zr_300K.out', 'file_list_Compression_Zr_10K.out','file_list_Tensile_Zr_10K.out',
             ]
refDirections = [[0,0,0,1],
                 [2,-1,-1,0],
                 [1,0,-1,0]
                 ]
loadindgDirections = [[0,0,0,1],
                    [1,0,-1,3],
                    [1,0,-1,2],
                    [11,0,-11,13],
                    [1,0,-1,1],
                    [12,0,-12,7],
                    [5,0,-5,2],
                    [1,0,-1,0],

                      ]
zrLattice = olt.hexagonal(1,1.59)
strOut=""
for refDir in refDirections:
    for loadingDir in loadindgDirections:
        refDirMiller = MillerDirection(vector=refDir, lattice=zrLattice)
        loadingDirMiller = MillerDirection(vector=loadingDir, lattice=zrLattice)
        angle = refDirMiller.angle(loadingDirMiller, considerSymmetry=True, units='deg')
        strOut += f'{refDir}  {loadingDir} {angle} \n'

print(strOut)
#exit(-1)
colNames = ['Xu', 'Xv','Xt', 'Xw', 'Yu', 'Yv','Yt', 'Yw', 'Zu', 'Zv','Zt', 'Zw',
            'Xx','Xy','Xz' , 'Yx','Yy','Yz', 'zx','zy','zz',
            'StressX', 'StressY', 'StressZ' ,
            # 'StrainX1', 'StrainX2','StrainX3',
            # 'StrainY1', 'StrainY2','StrainY3','StrainZ1','StrainZ2','StrainZ3'
          ]

outStr = ""
outStr2 = ""
data=[]
dataDict = {}
degree=np.pi/180
mat = Orientation(euler=[30*degree, 60*degree,90*degree], ) #### for chcking the answer sheets of trainees of 2022 batch
mat2 = Orientation(euler=[45*degree, 45*degree,0*degree], )
mat2 = Orientation(euler=[90*degree, 45*degree,0*degree], )

print(np.around(mat.rotation_matrix,3), np.around(mat2.rotation_matrix,3))

for fileName in fileNames:
    count = 0
    filePath = os.path.join(dataPath, fileName)
    df = pd.read_csv(filePath, names=colNames, sep=',')
    for index, row in df.iterrows():
        count+=1
        grainId = count
        print('I am here : ', index, count)
        #print(f'prcessing row {index+1}')
        xDir = MillerDirection(vector = [row["Xu"],row["Xv"],row["Xt"],row["Xw"]], lattice=zrLattice)
        yDir = MillerDirection(vector = [row["Yu"],row["Yv"],row["Yt"],row["Yw"]], lattice=zrLattice)
        zDir = MillerDirection(vector = [row["Zu"],row["Zv"],row["Zt"],row["Zw"]],lattice=zrLattice)
        Ori1 = Orientation.fromTwoOrthoDirections(zDir.getCartesianVec(),xDir.getCartesianVec(),vectorsParallelTo='XZ' )
        Ori2 = Orientation.fromTwoOrthoDirections(xDir.getCartesianVec(),zDir.getCartesianVec(),vectorsParallelTo='XY' )
        Ori3 = Orientation.fromTwoOrthoDirections(xDir.getCartesianVec(),zDir.getCartesianVec(),vectorsParallelTo='XZ' )
        dataDict["Ori1"] = Ori1
        print(index, xDir, Ori1)
        data.append(dataDict)
        outStr += f'{grainId} {index+1} X {Ori1}  X:{str(xDir)}  Y:{str(yDir)}  Z:{str(zDir)}  {row["StressX"]}\n'
        outStr2 += f'{grainId},{Ori1},{row["StressX"]}\n'
        count += 1
        grainId = count
        outStr += f'{grainId} {index+1} Y {Ori2}  X:{str(xDir)}  Y:{str(zDir)}  Z:{str(yDir)}  {row["StressY"]}\n'
        outStr2 += f'{grainId},{Ori2},{row["StressY"]}\n'
        count += 1
        grainId = count
        outStr += f'{grainId} {index+1} Z {Ori3}  X:{str(zDir)}  Y:{str(yDir)}  Z:{str(xDir)}  {row["StressZ"]}\n'
        outStr2 += f'{grainId},{Ori3},{row["StressZ"]}\n'

    outStr2 = outStr2.translate(str.maketrans('', '', '[]'))
    outStr2=outStr2.replace(" ",",").replace(",,",",")
    print(outStr, "\n here is the 2nd str : \n", outStr2)
    with open(filePath+'.Ori', 'w') as f:
        f.write(outStr2)

print(df)

