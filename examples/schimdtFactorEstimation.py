'''
Created on 18-Jul-2019

@author: Admin
'''

from __future__ import division, unicode_literals

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))
sys.path.insert(0, os.path.dirname('../..'))

import matplotlib.pyplot as plt
from pycrystallography.core.millerDirection  import MillerDirection
from pycrystallography.core.millerPlane  import MillerPlane
from pycrystallography.core.orientation  import Orientation
import numpy as np
import copy
from math import pi, sqrt
from pycrystallography.core.orientedLattice import OrientedLattice as olt
from pycrystallography.core.saedAnalyzer import SaedAnalyzer 
from scipy import optimize
import logging


def getSchimid(loadingDir, crystalOri, plane, burgVec, lattice):
    
    loadingDir = np.array(loadingDir)
    loadingDir=loadingDir/np.linalg.norm(loadingDir)
    loadingDir = MillerDirection(vector= loadingDir, lattice=lattice, isCartesian=True)
    loadingDirInCrystalFrame = copy.copy(loadingDir)
    loadingDirInCrystalFrame.rotate(crystalOri)
    angSlipLoad = plane.angle(loadingDirInCrystalFrame, units="Deg",considerSymmetry=False)
    angBurgLoad = burgVec.angle(loadingDirInCrystalFrame, units="Deg",considerSymmetry=False)
    schmid  = np.cos(angSlipLoad*np.pi/180)*np.cos(angBurgLoad*np.pi/180)
    logging.debug("Schimid = "+str(schmid)+ " ang of loading with Slip and Burgers Vsc :"+str(np.around(np.array([angSlipLoad, angBurgLoad]),3)))
    return abs(schmid),angSlipLoad,angBurgLoad


def objectiveFunctionSchimid(loadingDir,crystalOri,slipSystems, expectedValues=[0.04,0.22,0.18]):
        calcualtedSchmids=[]
        for system in slipSystems:
            plane=system["Plane"]
            burgersVec = system["BurgersVec"]
            schimd,angSlipLoad,angBurgLoad =getSchimid(loadingDir, crystalOri, plane, burgersVec, lattice)
            calcualtedSchmids.append(schimd)
        schmidArray = np.array(calcualtedSchmids)
        err = np.sum(np.abs(schmidArray-np.array(expectedValues)))
        return err


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    inputData = {"sumanData":
                 {"CrystalOri":[136.9,19.3,215.7], ### from paper H. Wang a, b, c, C.J. Boehlert c, **, Q.D. Wang a, b, *, D.D. Yin d, W.J. Ding, IJP, 84 (2016) 255-276
                  #"CrystalOri":[0.,0.,0],
                  "loadingDir":[-1.,0.,0], ### assuming the tensile axis along the Y axis of the frame of reference},
                  "loadingDir" : [ 0.17269742, -0.04014018,  1.09610329 ],
                  "slipSystems":[{"id":1, "slipPlane":[0, 0 , 0, 1], "burgersVec":[-2, 1,1 ,0],},
                                 {"id":2, "slipPlane":[0, 0 , 0, 1], "burgersVec":[-1, 2,-1 ,0],},
                                 {"id":3, "slipPlane":[0, 0 , 0, 1], "burgersVec":[-1, -1,2 ,0]},
                                 {"id":4, "slipPlane":[0, -1,1,0], "burgersVec":[2,-1,-1,0],},                             
                                 {"id":5, "slipPlane":[1, 0,-1,1], "burgersVec":[1,-2,1,0],},  
                                 {"id":6, "slipPlane":[-1, 1,0,0], "burgersVec":[1,1,-2,0],},  
                                 {"id":7, "slipPlane":[1, 1,-2,2], "burgersVec":[-1,-1,2,3],},  
                                 {"id":8, "slipPlane":[-1, 2,-1,2], "burgersVec":[1,-2,1,3],},
                                 {"id":9, "slipPlane":[-2, 1,1,2], "burgersVec":[2,-1,-1,3],},  
                                 {"id":10, "slipPlane":[-1,-1,2,2], "burgersVec":[1,1,-2,3],},  
                                 {"id":11, "slipPlane":[1, -2,1,2], "burgersVec":[-1,2,-1,3],},  
                                 {"id":12, "slipPlane":[2, -1,-1,2], "burgersVec":[-2,1,1,3],},    
                                 ],
                  "crystalCif": r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\Mg_P63mmc.cif',
                 },
                 
                 "sumanCuData":
                 {"CrystalOri":[136.9,19.3,215.7], ### from paper H. Wang a, b, c, C.J. Boehlert c, **, Q.D. Wang a, b, *, D.D. Yin d, W.J. Ding, IJP, 84 (2016) 255-276
                  #"CrystalOri":[0.,0.,0],
                  "loadingDir":[-1.,0.,0], ### assuming the tensile axis along the Y axis of the frame of reference},
                  "slipSystems":[{"id":1, "slipPlane":[1, 1 , 1], "burgersVec":[-2, 1,1 ,0],},
                                 {"id":2, "slipPlane":[1, -1 , -1], "burgersVec":[-1, 2,-1 ,0],},
                                 {"id":3, "slipPlane":[1, -1 , 1], "burgersVec":[-1, -1,2 ,0]},
                                  {"id":4, "slipPlane":[1, 1, -1], "burgersVec":[2,-1,-1,0],},                             
                                 ],
                  "crystalCif": r'D:\CurrentProjects\python_trials\work_pycrystallography\pycrystallography\data\structureData\\Alpha-ZrP63mmc.cif',
                 },
                 
               
                }
    
    choice = "sumanData"
    CrystalOri = inputData[choice]["CrystalOri"]
    crystalOri = Orientation(euler=np.array(CrystalOri)*np.pi/180)
    crystalCif = inputData[choice]["crystalCif"]
    slipSystems = inputData[choice]["slipSystems"]
    lattice = olt.fromCif(crystalCif)
    for key in inputData[choice]:
        logging.info(str(key) + ": "+str(inputData[choice][key]))
    result = []
    offset = [0.01,0.01]
    legendText=[]
    
    wantToRunOptimization=False
    
    for system in slipSystems:
        loadingDir = inputData[choice]["loadingDir"]
        plane = MillerPlane(hkl = system["slipPlane"], lattice=lattice)
        burgDir = MillerDirection(vector = system["burgersVec"], lattice=lattice)
        #planeDirAngle = plane.angle(dir,considerSymmetry=False, units="deg")
        ang = np.arccos(np.clip(np.dot(plane.getUnitVector(),burgDir.getUnitVector()),-1,1))*180/np.pi
        assert np.allclose(ang, [90.,]), "The direction dose not lie in plane "+str(ang)+" "+str(plane)+ " "+str(burgDir) +" id = "+str(system["id"])
        schmid,angSlipLoad,angBurgLoad = getSchimid(loadingDir,crystalOri, plane, burgDir, lattice)
        rotatedPlane = copy.copy(plane)
        rotatedPlane.rotate(crystalOri.inverse)
        trace = rotatedPlane.getUnitVector(returnCartesian=True)
        
        trace = np.around(np.array([-trace[1],trace[0]]),3)
        system["Plane"] = plane
        system["BurgersVec"]=burgDir
        system["angSlipLoad"]=np.around(angSlipLoad,1)
        system["angBurgLoad"]=np.around(angBurgLoad,1)
        system["PalneCartesian"]=np.around(plane.getCartesianVec(),3)
        system["burgVecCaretisan"]=np.around(burgDir.getCartesianVec(),3)
        system["schmid"]=schmid
        system["trace"]=trace
        result.append(system)
        plt.plot([0,trace[0]], [0,trace[1]])
        legendText.append(str(system["id"]))
        plt.text(x=trace[0]+offset[0], y=trace[1]+offset[1], s=str(system["id"]))

    for item in result:
        logging.info(str(item))
        
    plt.legend(legendText)
    plt.axis("equal")


if wantToRunOptimization:
    startPoint=[1,1,1] 
    expectedValues =  [0.04,0.22,0.18]
    options={'gtol': 1e-6, 'disp': False, "factor":1e7} 
    
    slipSystems = slipSystems[:3]      
    sol = optimize.minimize(objectiveFunctionSchimid,startPoint,
                                  args=(crystalOri,slipSystems,expectedValues, ) ,
                                  bounds=([-1.1,1.1], [-1.1,1.1],[-1.1,1.1]), 
                                  method="L-BFGS-B", options=options)
    dir = sol.x
    print(sol, dir)
    
    err = objectiveFunctionSchimid(loadingDir=dir, crystalOri=crystalOri,slipSystems=slipSystems, expectedValues=expectedValues )
    print(err)
    
else:
    plt.show()
print("done")

