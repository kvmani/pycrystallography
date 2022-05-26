'''
Created on 18-Dec-2017

@author: Admin
'''
import numpy as np
import spglib as spg
from pycrystallography.core.orientation import Orientation 
import pymatgen as mg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


lattice = np.array([[1., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.]]) * 1.0
positions = [[0., 0., 0.],
             #[0.5, 0.5, 0.5]
             ]
numbers= [1,] * 2
numbers= [1,]


lattice = mg.Lattice.cubic(4.2)
structure = mg.Structure(lattice, ["Cs", "Cl"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])

# You can create a Structure using spacegroup symmetry as well.
li2o = mg.Structure.from_spacegroup("Fm-3m", mg.Lattice.cubic(3),
                                        ["Li", "O"],
                                        [[0.25, 0.25, 0.25], [0, 0, 0]])

finder = SpacegroupAnalyzer(structure)
print(finder.get_space_group_symbol())
print("done")



        