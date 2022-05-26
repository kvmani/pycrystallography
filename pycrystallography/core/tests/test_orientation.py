# coding: utf-8
# Copyright (c) Pycrystallography Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals

import collections
from pymatgen.util.testing import PymatgenTest
from pycrystallography.core.orientation  import Orientation
from pycrystallography.core.quaternion  import Quaternion
from pycrystallography.core.millerDirection  import MillerDirection
from pymatgen.core.lattice import Lattice
from pycrystallography.core.orientedLattice import OrientedLattice as olt

import numpy as np
from math import pi
from pycrystallography.utilities.pymathutilityfunctions import integerize

degree = pi/180
ALMOST_EQUAL_TOLERANCE = 13

class OrientationCreationTestCases(PymatgenTest):
    def setUp(self):

        self.orientation = Orientation(euler=[0.,0.,0.],units='deg')              

    def test_init(self):
        a = np.random.random(12).reshape(3,4)
        orientation = Orientation(euler=[0.,0.,0.])
        #print(orientation)
        #print("the angle is   ", orientation.angle)
        self.assertIsNotNone(orientation, "Initialization of Orientation object by euler angles failed")  
        
        orientation = Orientation(axis=[1.,0.,0.], angle = 90*pi/180)
        self.assertIsNotNone(orientation, "Initialization of Orientation object by axis angle pair  failed")              
        
        orientation = Orientation(matrix=np.array([[0.,1.,0.],[-1.,0,0.],[0,0,1]]))
        self.assertIsNotNone(orientation, "Initialization of Orientation object by rotMatrxr  failed")              
                      
   
        orientation = Orientation(axis=[1.,1.0,0.], angle = 25*pi/180)
        self.assertIsNotNone(orientation, "Initialization of Orientation object by axis angle pair  failed") 
        
        q1 = Quaternion.random()
        q2 = Orientation(q1)
        self.assertIsInstance(q2, Orientation)
        self.assertIsNotNone(q2, "Initializing with Quat object failed")

        Ori = Orientation.random()
        self.assertIsNotNone(Ori, "Initializing with Random method failed")

        Ori1 = Orientation(20*degree, 30*degree,120*degree)
        self.assertIsNotNone(Ori1, "Initializing with sequnce of Euler angles (with out key word Euler) failed")
              
        #Ori2 = Orientation(euler = [20.,30.,120.], units='degrees')
        #self.assertIsNotNone(Ori2, "Initializing with sequnce of Euler angles in degrees failed")
               
        #self.assertAlmostEqual(Ori1,Ori2,'Problem in converting degrees into radians')
        
        with self.assertRaises(ValueError):
            Ori = Orientation(10)

                     
 

    def test_init_copy(self):
        q1 = Orientation(euler=[20.,30.,90.], units='deg')        
        q2 = Orientation(q1)

        
        #q2 = Quaternion(q1)

        self.assertIsInstance(q2, Orientation)
        self.assertEqual(q2, q1)


        with self.assertRaises(TypeError):
            q3 = Orientation(None)
        with self.assertRaises(TypeError):
            q4 = Orientation("String")
        
        
    def test_rotationMatrix(self):
        ori1 = Orientation(euler=[np.radians(90.),0.,0.])
        mat1 = ori1.rotation_matrix
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0,0,1]], mat1)
        
        ori1 = Orientation(euler=np.radians([18.43,5.12,355.4]))
        mat1 = ori1.rotation_matrix
        print(mat1,'here2')
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0,0,1]], mat1)                        
        exit()
        
        ori1 = Orientation(euler=np.radians([208.8,27.8,49.1]))
        mat1 = ori1.rotation_matrix
        print(mat1)
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0,0,1]], mat1)
        
        
        
        
        ####ref MarcDegraef paper on Rotation conventions 
        mat = np.array([[0.,1.,0.],[-1.,0.,0],[0.,0.,1.]])
        eulerAngles = np.array([pi/2,0.,0.])
        ori = Orientation(euler=eulerAngles)
        self.assertArrayAlmostEqual(ori.rotation_matrix, mat)
        print(ori.elements)
        self.assertArrayAlmostEqual(ori.elements,[1/np.sqrt(2),0.,0.,-1/np.sqrt(2)])
        self.assertArrayAlmostEqual(ori.axis,[0.,0.,-1.])
        
        oriFromMatrix = Orientation(matrix = mat)
        print("Axis and Angle are ",oriFromMatrix.degrees,oriFromMatrix.axis)
        self.assertEqual(oriFromMatrix,ori1,'Unable to make Ori formt the rotation matrix')

        ori1 = Orientation(axis=[0., 0., -1], angle=90*pi/180)
        mat1 = ori1.rotation_matrix
        self.assertArrayAlmostEqual(ori1.angle, 90*pi/180,6,'Test failed for angle checking ')
        self.assertArrayAlmostEqual(ori1.axis, [0.,0.,-1],6,'Test failed for axis checking ')
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0.,0.,1.]], mat1,6,'Test failed when Orientation is initiated by axis angle pair')
        self.assertArrayAlmostEqual(ori1.q, [1/np.sqrt(2),0,0,-1/np.sqrt(2)], 5, "Prob in Quat ")

        ori1 = Orientation(matrix=np.array([[0.,1.,0.],[-1.,0,0.],[0,0,1]]))
        mat1 = ori1.rotation_matrix
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0,0,1]], mat1,6,'Test failed when Orientation is initiated by rotation matrix')
        

        euler = np.radians(np.array([0,90,0]))        
        ori2 =  Orientation(euler=[euler[0],euler[1],euler[2]])
        mat2 = ori2.rotation_matrix
        self.assertArrayAlmostEqual([[1.,0.,0.],[0.,0.,1.],[0.0,-1., 0.,]], mat2)

        ori1 = Orientation(axis=[1., 0., 0], angle=90*pi/180)
        mat1 = ori1.inverse.rotation_matrix
        self.assertArrayAlmostEqual(ori1.angle, 90*pi/180,6,'Test failed for angle checking ')
        self.assertArrayAlmostEqual(ori1.inverse.axis, [-1.,0.,0.],6,'Test failed for axis checking ')
        self.assertArrayAlmostEqual([[1.,0.,0.],[0.,0.,1.],[0.0,-1., 0.,]], mat1,6,'Test failed when Orientation is initiated by axis angle pair')
        
        ori1 = Orientation(matrix=np.array([[0.,1.,0.],[-1.,0,0.],[0,0,1]]))
        mat1 = ori1.rotation_matrix
        self.assertArrayAlmostEqual([[0.,1.,0.],[-1.,0,0.],[0,0,1]], mat1,6,'Test failed when Orientation is initiated by rotation matrix')
        
        
        
        euler = np.radians(np.array([90,90,0]))        
        ori3 =  Orientation(euler=[euler[0],euler[1],euler[2]])
        mat3 = ori3.rotation_matrix
        self.assertArrayAlmostEqual( [[0,  1,  0],[-0,  0,  1],[1, -0,  0]], mat3)

        euler = np.radians(np.array([10,20,40]))        
        ori4 =  Orientation(euler=[euler[0],euler[1],euler[2]])
        mat4 = ori4.rotation_matrix
        self.assertArrayAlmostEqual([[0.649519,0.727869,0.219846],[-0.758022,0.597291,0.262003],[0.059391,-0.336824,0.939693]], mat4, 4,'Failed rotation matrix test to 4 decimels for random euler set')
        
        
        
        
        hexLat = olt.hexagonal(1, 1.59)
        vec =[1,-1,0,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, MillerConv="Bravais")
        ori1 = Orientation(axis=dir1.getCartesianVec(), angle=17.827*pi/180)
        print (ori1.rotation_matrix,"rotation matrix")
        
        vec1 =[ 0.98799631, -0.020791,    0.15307198]
        vec2 =[-0.020791,    0.96398892,  0.26512844]
        vec3 =[-0.15307198, -0.26512844,  0.95198523]
        dir1=MillerDirection(lattice=hexLat, vector = vec1, isCartesian="True")
        dir2=MillerDirection(lattice=hexLat, vector = vec2, isCartesian="True")
        dir3=MillerDirection(lattice=hexLat, vector = vec3, isCartesian="True")
        print(dir1)
        print(dir2)
        print(dir3)
        
        
        ori2 = Quaternion(matrix=np.array(ori1.rotation_matrix))
        print("misorientation axis", ori2.get_axis())
        print("misorientation angle", ori2.angle*(180/np.pi))

        hexLat = olt.hexagonal(1, 1.59)
        vec    = ori2.get_axis()
        inhex  = MillerDirection(lattice=hexLat, vector = vec,  isCartesian=True)
        print(inhex,"from miller to bravis")
        
        
       
    def test_getEulerAngles(self):

        euler = np.radians(np.array([90,90,0]))        
        ori1 =  Orientation(euler=euler)
        mat1 = ori1.rotation_matrix
        eulerCalculated = ori1.getEulerAngles()
        self.assertArrayAlmostEqual(euler, eulerCalculated, 7,'Failed rotation matrix test to 7 decimels for random euler set')
        

        euler = np.radians(np.array([0,0,0]))        
        ori2 =  Orientation(euler=euler)
        mat2 = ori1.rotation_matrix
        eulerCalculated = ori2.getEulerAngles()
        self.assertArrayAlmostEqual(euler*180/pi, eulerCalculated*180/pi, 7,'Failed rotation matrix test to 7 decimels for random euler set')    
        print ("here is the beginning")
        
        ori3 = Orientation(matrix = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1]])) 
        
        cal = ori3.getEulerAngles() 
        self.assertArrayAlmostEqual([0, 180., 0], cal*180/pi, 1,'Failed getEulerAngles test to 2 decimels for random euler set')    
             
        #exit()   
        numOfOrisToTest = 100
        euler_limits = np.array([360,180,360])*pi/180
        for i in range (numOfOrisToTest) :
            a = (np.random.rand(3))            
            a*=euler_limits
            oris = Orientation(euler=a)  
            eulerCalculated = oris.getEulerAngles()        
            self.assertArrayAlmostEqual(a*180/pi, eulerCalculated*180/pi, 1,'Failed getEulerAngles test to 2 decimels for random euler set')    
            
    def test_random(self) :
        o1 = Orientation.random()
        self.assertIsInstance(o1, Orientation, 'Creating the single random orieantion failed' ) 
        oList = Orientation.random(10)
        print(oList)

        for i in oList :
            self.assertIsInstance(i, Orientation, 'Creating the list of random orieantions failed')
            
                    

class OrientationAithmeticTestCases(PymatgenTest):
    def test_misorientation(self):

        euler = np.radians(np.array([90.,0.,0.]))        
        ori1 =  Orientation(euler=euler)
        
        euler = np.radians(np.array([0.,0.,0.]))        
        ori2 =  Orientation(euler=euler)
        
        #print("The quat is \n", ori1.q)
        misori = ori1.misorientation(ori2)
        #self.assertEqual(ori1, misori,'Failed misorieantion test')
        self.assertAlmostEqual(misori.angle,90*degree,5,'Calcualted Misorienation Angle is wrong')

        euler = np.radians(np.array([5., 2.,23.]))        
        ori1 =  Orientation(euler=euler)
        
        euler = np.radians(np.array([43., 10., 8.]))        
        ori2 =  Orientation(euler=euler)
        
        #print("The quat is \n", ori1.q)
        misori = ori1.misorientation(ori2)
        #self.assertAlmostEqual(ori1, misori,'Failed misorieantion test')
        self.assertAlmostEqual(misori.angle/degree,24.404,3,'Calcualted Misorienation Angle is wrong')
                
        ori1 = Orientation(matrix=np.array([[-0.26583529,  0.46044023,  0.84695123],[-0.95451993,  0.,         -0.29814711],[-0.13740733, -0.88852018,  0.43777975]]))
        ori2 = Orientation(matrix=np.array([[ 0.,  0.,  1.],[  1.28197512e-16,   1.00000000e+00,   1.06057524e-16],[ -1.00000000e+00,   0.00000000e+00,  -6.12323400e-17]]))
        
        
        misori = ori1.misorientation(ori2)
        
        euler = np.radians(np.array([5,8,10])) 
        ori1 = Orientation(euler=euler)
                
        euler = np.radians(np.array([3,20,14])) 
        ori2 = Orientation(euler=euler)
        misori = ori1.misorientation(ori2)
        axis = integerize(misori.axis)
        ori3 = misori*ori1
        
        print("the value of misori = ",ori3, misori,misori.angle*180./np.pi,axis)
        
        self.assertEqual(ori2,ori3)
        
    
    def test_misorientationAngle(self):

        euler = np.radians(np.array([0.,90.,0.]))        
        ori1 =  Orientation(euler=euler)
       
        
        euler = np.radians(np.array([0,0,0]))        
        ori2 =  Orientation(euler=euler)
        
        #print("The quat is \n", ori1.q)
        misoriAngle = ori1.misorientationAngle(ori2)
        self.assertAlmostEqual(misoriAngle, 90*degree,'Failed misorieantion test')
        hexLat = olt.hexagonal(1, 1.593)
        vec = [-1,2,-1,3]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[1,1,-2,-1]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori1 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        
        vec = [0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[0,1,-1,2]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori2 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        
#         vec = [2,-1,-1,-6]
#         dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
#         vec=[0,0,0,1]
#         dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
#         dir2=dir1.cross(dir)
#         dir3=dir1.cross(dir2)
#         ori2 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
#         ori2 = Orientation(matrix=np.array([[  1.28197512e-16,   1.00000000e+00,   1.06057524e-16],[-0.95451993,  0.,         -0.29814711],[-0.29946837,  0.,          0.95410623]]))
#         ori1 = Orientation(matrix=np.array([[1,0,0],[0,1,0],[0,0,1]]))
#         ori2 = Orientation(matrix=np.array([[ 1,0,0],[ 0,1,0],[ 0,0,1]]))
        misoriAngle = ori1.misorientationAngle(ori2,units="Deg")
        print("misorientation angle", misoriAngle)
    
    def test_misorientationAxis(self):

        euler = np.radians(np.array([0.,90.,0.]))        
        ori1 =  Orientation(euler=euler)
        
        euler = np.radians(np.array([0,0,0]))        
        ori2 =  Orientation(euler=euler)
        
        #print("The quat is \n", ori1.q)
        misoriAxis = ori1.misorientationAxis(ori2)
        self.assertArrayAlmostEqual(misoriAxis, [1., 0., 0.], 7,'Failed misorieantion test')
        
        hexLat = olt.hexagonal(1, 1.593)
        vec = [0,1,-1,0]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[-1,2,-1,3]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori1 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        
        vec = [0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[0,1,-1,2]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori2 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        
        
        #orintation 1
        vec = [-1,2,-1,3]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[0,1,-1,0]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        
        print("1st set",dir1.angle(dir,units="deg"))
        
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori1 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        #orientation 2
        
        vec=[0,0,0,1]
        dir1 = MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        vec=[1,-2,1,6]
        dir= MillerDirection(lattice=hexLat, vector = vec, isCartesian=False, MillerConv="Bravais")
        print("2nd set" , dir1.angle(dir,units="deg"))
        dir2=dir1.cross(dir)
        dir3=dir1.cross(dir2)
        ori2 = Orientation(matrix=np.array([dir1.getUnitVector(),dir2.getUnitVector(),dir3.getUnitVector()]))
        misoriAxis = ori1.misorientationAxis(ori2)
        print("misorientation axis", misoriAxis)
        print("misorientation integerized axis", integerize(misoriAxis))
        v=MillerDirection(lattice=hexLat, vector = misoriAxis, isCartesian=True)
        
        dir1 = MillerDirection(lattice=hexLat, vector = misoriAxis, isCartesian=True)
        print("lattice frame misorentation axis",dir1)
        v = dir1.fromMillerToBravais()
        print("millerbravais indices",v)
        
        
              

    def test_slrep(self):
        o1 = Orientation(axis=[1, 0, 0], angle=0.0)
        o2 = Orientation(axis=[1, 0, 0], angle=pi/2)
        o3 = Orientation.slerp(o1, o2, 0.5)
        self.assertEqual(o3, Orientation(axis=[1,0,0], angle=pi/4))
        self.assertIsInstance(o3,Orientation,'The created object is not of orientation type')

    def test_interpolate(self):
        o1 = Orientation(axis=[1, 0, 0], angle=0.0)
        o2 = Orientation(axis=[1, 0, 0], angle=2*pi/3)
        num_intermediates = 3
        base = pi/6
        list1 = list(Orientation.intermediates(o1, o2, num_intermediates, include_endpoints=False))
        list2 = list(Orientation.intermediates(o1, o2, num_intermediates, include_endpoints=True))
        self.assertEqual(len(list1), num_intermediates)
        self.assertEqual(len(list2), num_intermediates+2)
        self.assertEqual(list1[0], list2[1])
        self.assertEqual(list1[1], list2[2])
        self.assertEqual(list1[2], list2[3])
    
        self.assertEqual(list2[0], o1)
        self.assertEqual(list2[1], Orientation(axis=[1, 0, 0], angle=base))
        self.assertEqual(list2[2], Orientation(axis=[1, 0, 0], angle=2*base))
        self.assertEqual(list2[3], Orientation(axis=[1, 0, 0], angle=3*base))
        self.assertEqual(list2[4], o2)

    def test_mean(self) :

        o1 = Orientation(axis=[1, 0, 0], angle=0.0)
        o2 = Orientation(axis=[1, 0, 0], angle=2*pi/3)
        num_intermediates = 2
        
        list1 = list(Orientation.intermediates(o1, o2, num_intermediates, include_endpoints=True))
        
        weight = 0.5
        weightsList = [weight for i in range(len(list1))]   

        mean_o = Orientation.mean(o1,o2)
        self.assertEqual(mean_o,Orientation(axis=[1, 0, 0], angle=pi/3))
        mean_list1 = Orientation.mean(list1)
        mean_list2 = Orientation.mean(list1, weightsList)

        self.assertEqual(mean_list1, mean_list2)
        self.assertEqual(mean_list1,mean_o)
        
        o1 = Orientation(axis=[0.,0.,1], angle=-171*np.pi/180.)
        o2 = Orientation(axis=[0,0,-1.], angle=9)
        meanOri = Orientation.mean(o1,o2)
        print("The angle and axis are", meanOri.degrees,meanOri.axis)
    
    def test_mapVector(self):
        sourceVc = [1,0,0]
        targetVec = [1,1,1]
        ori = Orientation.mapVector(sourceVector=sourceVc, targetVector=targetVec)
        targetCheck = ori.rotate(sourceVc)
        self.assertArrayAlmostEqual(targetCheck/np.linalg.norm(targetCheck), targetVec/np.linalg.norm(targetVec), 5, "Failed the Map Vector")
        
            

        

    def test_rotation(self):
        #### from marc de graef paper:
        ori = Orientation(axis=[1,1,1],degrees=120.)
        r = [0,0,1]
        print("the axis n ange r",ori.axis,ori.degrees)
        rPrime = ori.inverse.rotate(r)
        self.assertArrayAlmostEqual(rPrime, [0,1.0,0], 7, "Problem in rotation")
        
        ## testing the combined rotation
        alpha = Orientation(axis = [1,1,1],degrees=120.)
        beta = Orientation(axis = [1,1,0],degrees=180.)
        gamma = alpha.inverse*beta.inverse
        
        mat = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        self.assertArrayAlmostEqual(gamma.rotation_matrix, mat, 7, "Problem in combined rotation")
        rDoublePrime = gamma.rotate(r)
        self.assertArrayAlmostEqual(rDoublePrime, [0,-1.0,0], 7, "Problem in rotation")
        
        
        
        o  = Orientation(axis=[1,1,1], angle=-2*pi/3)
        o2 = Orientation(axis=[1, 0, 0], angle=-pi)
        o3 = Orientation(axis=[1, 0, 0], angle=pi)        
        vec = o3.rotate([0,0,1])        
        precision = ALMOST_EQUAL_TOLERANCE
#         for r in [1, 3.8976, -69.7, -0.000001]:
#             # use np.testing.assert_almost_equal() to compare float sequences
#             np.testing.assert_almost_equal(o.rotate((r, 0, 0)), (0, r, 0), decimal=ALMOST_EQUAL_TOLERANCE)
#             np.testing.assert_almost_equal(o.rotate([0, r, 0]), [0, 0, r], decimal=ALMOST_EQUAL_TOLERANCE)
#             np.testing.assert_almost_equal(o.rotate(np.array([0, 0, r])), np.array([r, 0, 0]), decimal=ALMOST_EQUAL_TOLERANCE)
#             self.assertEqual(o.rotate(Orientation(vector=[-r, 0, 0])), Orientation(vector=[0, -r, 0]))
#             np.testing.assert_almost_equal(o.rotate([0, -r, 0]), [0, 0, -r], decimal=ALMOST_EQUAL_TOLERANCE)
#             self.assertEqual(o.rotate(Orientation(vector=[0, 0, -r])), Orientation(vector=[-r, 0, 0]))
#         
#             np.testing.assert_almost_equal(o2.rotate((r, 0, 0)), o3.rotate((r, 0, 0)), decimal=ALMOST_EQUAL_TOLERANCE)
#             np.testing.assert_almost_equal(o2.rotate((0, r, 0)), o3.rotate((0, r, 0)), decimal=ALMOST_EQUAL_TOLERANCE)
#             np.testing.assert_almost_equal(o2.rotate((0, 0, r)), o3.rotate((0, 0, r)), decimal=ALMOST_EQUAL_TOLERANCE)
        

if __name__ == '__main__':
    import unittest
    unittest.main()
