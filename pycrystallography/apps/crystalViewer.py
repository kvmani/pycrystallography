from pymatgen.vis.structure_vtk import StructureVis
from pymatgen import Lattice, Structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_lattice(lat,ax,color='k',isreciprocal=False,markBasisVectors=True):

    basisVectors = [(0,1), (0,3), (0,4)]
    lineStyle="-"
    basisVectorsNames = [r'$\vec{a}$',r'$\vec{b}$',r'$\vec{c}$']
    reciprocalBasisVectorsNames = [r'$\vec{a^{*}}$',r'$\vec{b^{*}}$',r'$\vec{c^{*}}$']
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    edges = [(0, 1), (0, 3), (1, 2), (2, 3), (1, 5), (0, 4), (5, 4), (5, 6), (6, 7), (5, 4), (2, 6), (3, 7), (4, 7)]
    if isreciprocal:
        basisVectorsNames = reciprocalBasisVectorsNames
        lineStyle = "--"
    for edge in edges:
        p1, p2 = vertices[edge[0]], vertices[edge[1]]
        p1Coord = lat.get_cartesian_coords(p1)
        p2Coord = lat.get_cartesian_coords(p2)
        ax.plot(*zip(p1Coord, p2Coord),color,linestyle= lineStyle)


    for i, basisVector in enumerate(basisVectors):
        p1, p2 = vertices[basisVector[0]], vertices[basisVector[1]]
        p1Coord = lat.get_cartesian_coords(p1)*0.45
        p2Coord = lat.get_cartesian_coords(p2)*0.45
        x,y,z = 0.5*(p1Coord+p2Coord)
        #ax.plot(*zip(p1Coord, p2Coord), color)
        a = Arrow3D(*zip(p1Coord, p2Coord), mutation_scale=20,
                    lw=4, arrowstyle="-|>", color=color)
        ax.add_artist(a)
        zdir = (x,y,z)
        if markBasisVectors:
            ax.text(x,y,z, basisVectorsNames[i], zdir,fontsize=14,)
    ax.axis('off')



    ax.grid(visible=False)


structure = Structure.from_file('../../data/structureData/Alpha-TiP63mmc.cif')
# structure = Structure.from_file('../../data/structureData/Alpha-U_ver2.cif')
# structure = Structure.from_file('../../data/structureData/o3 ti2Rhobohedral.cif')
# structure = Structure.from_file('../../data/structureData/zrSiO4tetragonal.cif')
#vis = StructureVis(structure)
lat = structure.lattice
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
plot_lattice(lat,ax,color='k')
recipLat = lat.reciprocal_lattice
plot_lattice(recipLat,ax,color='r',isreciprocal=True)
plt.grid(b=None)
plt.show()
