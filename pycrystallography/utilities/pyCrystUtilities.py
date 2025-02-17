# -*- coding: utf-8 -*-
from __future__ import division, print_function


from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pycrystallography.core.millerPlane import MillerPlane
from pycrystallography.core.millerDirection import MillerDirection
from scipy.spatial.distance import squareform, pdist
import math as mt
import numpy as np
import pandas as pd
from math import cos,sin

import sys
import os
import pymatgen as pm
from pymatgen.analysis.diffraction.xrd import XRDCalculator as Xrd

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.dirname('..'))
sys.path.insert(0, os.path.dirname('../pycrystallography'))

def getAtomDataFromStructure(structure):
    """

    :param structure: pymatgen.core.Structure object (typically created from cif file)
    :return: Atom Data in the form of list of tuple of atom data (Z, [x,y,z], occupancy) Note x,y,z are fractional coordinates suitable for diffraction computation
    """
    atomData=[]
    if isinstance(structure,Structure):
        for site in structure:
            for sp, occu in site.species.items():
                atomData.append([sp.Z, site.frac_coords, occu, ])

    else:
        raise ValueError(f"Only pymatgen Structure object can be sent. But {type(structure)} type was sent to the method :getAtomDataFromStructure")
    return atomData

def makeDiffractgionTable(lattice,structure, hklMax=3, dfFileName=r'../tmp/xrdOutput.html'):
    """

    :param lattice: Lattice object
    :param structure: Structure object (typically madfe from cif file)
    :param hklMax: default 3
    :param dfFileName: path of the output file to which data is written
    :return: df : Data frame having ther computed Xrd table
    """

    xrd = Xrd()
    xrayPattern = xrd.get_pattern(structure=structure)

    df = pd.DataFrame({"Plane:Multiplicty": xrayPattern.hkls, "dSpacing": xrayPattern.d_hkls,
                        "2Theta": xrayPattern.x,
                       "xrdIntensity": xrayPattern.y, })
    df.to_html(dfFileName, )
    df.to_csv(dfFileName[:-4]+'.txt')
    return df

def makeTEMDiffractgionTable(lattice,structure, hklMax=3, dfFileName=r'../tmp/xrdOutput.html'):
    """

    :param lattice: lattice of the unit cell
    :param structure: Crystal structure of the cell
    :param hklMax: default 3
    :param dfFileName: file name to which the output data is written
    :return: data frame
    """
    hklList = MillerPlane.generatePlaneList(hklMax=hklMax, lattice=lattice, includeSymEquals=False)
    hklNames, dSpacings, structureFactor, multiplicity, intensity, twoTheta, spotPresent = [], [], [], [], [], [], []
    atomData = getAtomDataFromStructure(structure)

    for i, plane in enumerate(hklList):
        if i > 0:  ##i==0 will be 000 spot
            intTmp = np.around(plane.diffractionIntensity(atomData=atomData), 3)
            if intTmp > 1e-2:
                spotPresent.append(True)
            else:
                spotPresent.append(False)
            intensity.append(intTmp)
            hklNames.append(str(plane))
            dSpacings.append(plane.dspacing)
            structureFactor.append(np.around(plane.structureFactor(atomData=atomData), 3))
            multiplicity.append(plane.multiplicity())
            twoTheta.append(np.around(plane.get2theta(), 3))
    df = pd.DataFrame({"Plane": hklNames, "dSpacing": dSpacings,
                       "multiplicity": multiplicity, "2Theta": twoTheta,
                       "intensity": intensity, "relIntenisty": np.array(intensity) / max(intensity),
                       "spotPresent": spotPresent})
    # df = df.drop(df.index[0])
    df.to_html(dfFileName)
    return df
