#### sal helper code for shabana to cross check the variants etc:
import sympy as sp
import numpy as np
import pandas as pd
import sys, os
import re

from scipy.linalg import sqrtm, inv

def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))

# def convertToHtmlString(line):
#     modLine = line.split()
#     for number in modLine:
#         if number[0]=="-":
#             number = number[1:] ### remving the minus sign
#
# import pprint
oldString = "[-20 -34  89], [4 5 6], [2/3, -5/7 8/9]"
oldString2 = "this is my second{{ new_string-d }}"
# oldString = "this is my {{-1}}"
# oldString2 = "this is my second{{ -205 }}"

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))
import pycrystallography.utilities.pymathutilityfunctions as pymt
sp.init_printing()
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.crystalOrientation import CrystalOrientation as CryOrn
from pycrystallography.core.millerPlane import MillerPlane
from pycrystallography.core.millerDirection import MillerDirection

baseMatrix = np.array([[-1,2/3,1/2], [1,2/3,1/2],[0,2/3,-1]])
#baseMatrix = np.array([[-1,2/3,1/2], [0,2/3,-1],[1,2/3,1/2],]) ### modified one
lat = olt.cubic(1)

originalMatricesOfJB = [[-6,4,3,6,4,3,0,4,-6],
[-6,4,-3,0,4,6,6,4,-3],
[0,4,-6,-6,4,3,6,4,3],
[-6,-4,3,6,-4,3,0,4,6],
[0,-4,6,6,-4,-3,6,4,3],
[6,-4,3,0,-4,-6,6,4,-3],
[6,4,-3,6,-4,3,0,-4,-6],
[6,4,3,0,-4,6,6,-4,-3],
[0,4,6,-6,-4,3,6,-4,3],
[6,-4,-3,6,4,3,0,-4,6],
[0,-4,-6,6,4,-3,6,-4,3],
[6,-4,3,0,4,6,-6,-4,3],
                    ]
correctedMatricesOfJB = [
[3., 6., 4.,3., -6., 4.,-6., 0., 4.],
 [3., -6., 4.,-3., -6., -4.,-6., -0., 4.],
 [-3., -6., -4.,-3., 6., -4.,-6., -0., 4.],
 [-3., 6., -4.,3., 6., 4.,-6., 0., 4.],
 [3., -6., 4.,-6., 0., 4.,3., 6., 4.],
 [-6., -0., 4.,-3., 6., -4.,3., 6., 4.],
 [-3., 6., -4.,6., -0., -4.,3., 6., 4.],
 [6., -0., -4.,3., -6., 4.,3., 6., 4.],
 [-6., 0., 4.,3., 6., 4.,3., -6., 4.],
 [3., 6., 4.,6., -0., -4.,3., -6., 4.],
 [6., -0., -4.,-3., -6., -4.,3., -6., 4.],
 [-3., -6., -4.,-6., 0., 4.,3., -6., 4.],

]


rotationMatrices = [[-6,4,3,6,4,3,0,4,-6],
[-6,4,-3,0,4,6,6,4,-3],
[0,4,-6,-6,4,3,6,4,3],
[-6,-4,3,6,-4,3,0,4,6],
[0,-4,6,6,-4,-3,6,4,3],
[6,-4,3,0,-4,-6,6,4,-3],
[6,4,-3,6,-4,3,0,-4,-6],
[6,4,3,0,-4,6,6,-4,-3],
[0,4,6,-6,-4,3,6,-4,3],
[6,-4,-3,6,4,3,0,-4,6],
[0,-4,-6,6,4,-3,6,-4,3],
[6,-4,3,0,4,6,-6,-4,3],
                    ]
HabitPlanes = [

    [1,1,0],
    [1,-1,0],
    [0,1,1],
    [1,0,1],
    [1,0,-1],
    [0,-1,-1],
            [1,1,1  ],
            [-1,1,1 ],
            [1,-1,1 ],
            [1,1,-1 ],
    [1,1,2],
    [1,-1,2],
    [1,-1,-2],
    [1,1,-2],

            [2,0,0  ],
            [0,2,0  ],
            [0,0,2  ],
            [2,2,0  ],
            [-2,2,0 ],
            [2,0,2  ],
            [2,0,-2 ],
            [0,2,2  ],
            [0,2,-2 ],
            [4,2,2  ],
            [-4,2,2 ],
            [4,-2,2 ],
            [4,2,-2  ],
            [2,4,2  ],
            [-2,4,2 ],
            [2,-4,2 ],
            [2,4,-2 ],
            [2,2,4  ],
            [-2,2,4 ],
            [2,-2,4 ],
            [2,2,-4 ],
            [4,2,0  ],
            [-4,2,0 ],
            [2,4,0  ],
            [-2,4,0 ],
            [2,0,4  ],
            [-2,0,4 ],
            [4,0,2  ],
            [-4,0,2 ],
            [0,4,2  ],
            [0,-4,2 ],
            [0,2,4  ],
            [0,-2,4 ],

    ]

deltaHabitPlanes = [ ##[1,1,0], [1,3,3],[3,1,3], [1,1,3], [3,3,0],
                    ## [1,1,1],[1,1,2],[3,3,1],[1,3,2],
                    #[1,0,0],[1,2,0],[1,2,3],[3,2,3],
                    #[1,0,1],[1,2,1],[1,0,2],[1,2,2],[3,2,1],
                    #[2,1,0],[2,1,3],
                    ##[2,1,1],[2,3,1],
                    #[2,1,2],
                    #[0,1,0],[0,1,3],[0,1,6],[0,3,6],[4,1,0]
                    #[0,1,1],
                    #[0,1,2],
                    # [2,0,3],[2,2,3],[2,4,3],[2,6,3]
                    # [2,0,0],[2,2,0],[2,0,6],
                    # [2,2,1],[2,0,1],
                    # [2,2,2],[2,0,2],
                    # [0,0,3],[0,2,3],[4,2,3],[4,0,3],[0,4,3]
                    # [0,2,0],[0,2,6],[4,2,6],[4,0,0],[0,0,6]
                    # [0,0,1],[0,0,5],[0,1,2],
                    [0,0,2],[0,2,2],[0,0,4],[4,2,2]

                    #[3,2,0], [6,0,0], [ 1,1,0],[1,3,3],[0,2,0], [-2,2,0], [0,2,0],[1, 1, 0], [1,3,3]
                   ] ### these are for inverting delta plaes into the gamma equivalnet  ones


mat1 = sp.Matrix([[-6,4,3 ], [6,4,3],[0,4,-6],]) ### JB
mat2 = sp.Matrix([[-3,3,0],[3,3,3],[2,2,-4]]) ## JB

mat1 = sp.Matrix([[3, -6, 4],[3, 6, 4],[-6, 0, 4]]) ### mani This matrix will have to be pasted into the line no:744 for baseMatrix of compositeDiffraction.py script for delta variants transformation matrices
mat2 = sp.Matrix([[2,2,-4],[-3,3,0],[3,3,3]]) ## mani mani This matrix will have to be pasted into the line no:745 for baseMatrixInverse of compositeDiffraction.py script for delta variants inv transformation matrices

product1 = mat1*mat2
product2 = mat2*mat1
invMat1 = 6*mat1.inv()
invMat2 = 6*mat2.inv()
print(mat1,mat2, "products", product1, product2,"inverses", invMat1,invMat2, "determinants", mat1.det(), mat2.det())
#exit(-10)
outPutString = []
# outPutString.append(f'')
variantsnames = ['Plane']
for i, matrix in enumerate(rotationMatrices):
    variantsnames.append(f"Var_{i}")

df = pd.DataFrame(columns=variantsnames)
delta2gammaList=[]
for i, habitPlane in enumerate(HabitPlanes):
    newLine = f"{habitPlane}"
    if i>len(deltaHabitPlanes)-1:
        break
    habitPlaneVectorDelta = deltaHabitPlanes[i]

    localDict = {"Plane":f"{habitPlane}" }
    for j, matrix in enumerate(correctedMatricesOfJB):
        print(f"j={j}")
        matrix = [int(i) for i in matrix]
        rotMatrix = np.array(matrix).reshape((3,3))
        rotMatrix = sp.Rational(1,6)*sp.Matrix(np.array(matrix).reshape((3,3)))
        habitPlaneVector = sp.Matrix(habitPlane)
        invMatrix = rotMatrix.inv()
        ##result = np.around(np.asarray(1/6*np.matmul(habitPlaneVector.T, rotMatrix,),dtype=np.float64),3).squeeze()
        result = habitPlaneVector.T * rotMatrix
        #result = rotMatrix*habitPlaneVector
        #result = np.around(np.asarray(1./6*np.matmul(habitPlaneVector.T, rotMatrix),dtype=np.float64).squeeze(),3)
        if j==0:
            habitPlaneVectorDelta = sp.Matrix(habitPlaneVectorDelta)
            equivalnetGammaPlane =sp.Rational(1,1)*habitPlaneVectorDelta.T*rotMatrix.inv()
            crossCheck = equivalnetGammaPlane*rotMatrix
            if np.allclose(np.array(habitPlaneVectorDelta).astype(np.float64),np.array(crossCheck).astype(np.float64) ):
                print("alright")
            else:
                print("oops")
            strOut = f"({str(equivalnetGammaPlane[0])}, {str(equivalnetGammaPlane[1])}, {str(equivalnetGammaPlane[2])})"
            strOut2 = str(equivalnetGammaPlane)
            delta2gammaList.append([deltaHabitPlanes[i], strOut])

        maxDenom = max(result[0].q,result[1].q,result[2].q) ### trying to get the max denomonator of all the fractional numbeers which are in p/q form
        result = result*maxDenom
        newLine = newLine+f'   {result}'
        #pprint.pprint(result)
        strOut = "({:4d} {:4d} {:4d})".format(int(result[0]),int(result[1]),int(result[2]))
        #strOut = "("+ str(result[0]) + " "+ str(result[1])+" "+str(result[2])+")"
        first = re.sub(r"(\w/\w)", r"{\1}", strOut)
        final = re.sub(r"(-\{(.*?)\})", r"\\bar(\1)", first).replace("-", "")
        sp.print_python(result)
        localDict[f"Var_{j}"] = strOut
    df = df.append(localDict, ignore_index=True)


    outPutString.append(newLine)

# for line in outPutString:
#     print(line, end="\n")

print (df)
df.to_html('variantsIntegers.html')
for item in delta2gammaList:
    print(item)
for item in delta2gammaList:
    print(item[1])