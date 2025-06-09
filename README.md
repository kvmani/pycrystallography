PyCrystallography
=================

PyCrystallography is a Python toolkit for crystallographic analysis and visualization. It provides
object-oriented utilities for common tasks such as handling Miller indices, computing angles between
directions or planes and generating diffraction patterns. The package builds on well known
libraries such as `numpy` and `pymatgen` and is aimed at materials scientists working with
orientation relationships and phase transformations.

Key features
------------
* Tools to create and manipulate crystallographic directions and planes.
* Orientation utilities with quaternion support for easy rotation handling.
* Functions to calculate stereographic projections, diffraction patterns and interplanar spacings.
* Example scripts demonstrating TEM SAED simulation and angle calculations.

The `examples` directory contains runnable scripts. After installing the requirements you can test the
installation by running::

    python examples/angleBetweenDirections.py

which prints a table of angles between different directions using data from ``data/structureData``.

For installation and detailed documentation see the ``docs`` folder or the project website.
