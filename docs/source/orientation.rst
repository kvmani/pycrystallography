Orientation relationships
=========================

Orientation relationships (ORs) describe how a product phase aligns with a
parent lattice across an interface. In pycrystallography an OR is defined by a
set of direction or plane pairs that must be coincident between phases. These
pairs are used to construct an orientation matrix via :mod:`orix`, after which
all symmetry-equivalent variants can be enumerated.

Defining an OR
--------------

ORs are declared in the configuration file alongside the phases that
participate in the relationship. Each entry lists the parent/child phase names
and at least two matching direction pairs:

.. code-block:: yaml

   phases:
     - name: beta-zr
       structure: zr_beta
       space_group: 229
     - name: alpha-zr
       structure: zr_alpha
       space_group: 194

   orientation_relations:
     - name: burgers-zr
       parent_phase: beta-zr
       child_phase: alpha-zr
       parent_directions:
         - [1, 1, 0]
         - [1, -1, 1]
       child_directions:
         - [0, 0, 1]
         - [1, 0, 0]

Directions can be provided either in three-index (cubic) or four-index (hexagonal)
notation. Hexagonal indices are converted to Cartesian vectors prior to calling
``orix``.

Computing the orientation matrix
--------------------------------

The :class:`pycrystallography.adapters.orix_adapter.OrientationFactory` takes the
direction pairs, normalises them, and calculates the orientation matrix that maps
parent vectors onto the child lattice. Under the hood the factory uses
``orix.quaternion.orientation.Orientation`` objects to guarantee numerically stable
results while respecting lattice symmetries.

Once the fundamental orientation has been obtained, the
:class:`pycrystallography.adapters.orix_adapter.VariantGenerator` enumerates all
symmetry-equivalent variants by combining the parent and child point groups. The
resulting :class:`pycrystallography.core.models.Variant` objects keep track of the
parent/child phases and carry a deterministic ``label`` for downstream tooling.

From ORs to diffraction
-----------------------

Variants generated from an OR are passed to diffraction calculators. The
``pcg composite tem`` command rotates each variant structure and evaluates a
composite SAED pattern. The new ``pcg powder xrd`` command complements this by
computing powder diffraction peak lists for single phases, making it easy to
compare, for example, the α and β zirconium structures defined above.

To experiment interactively, load the example configuration and run::

   pcg composite tem --relation burgers-zr --config ../examples/burgers_zr.yaml
   pcg powder xrd --phase alpha-zr --config ../examples/burgers_zr.yaml

Both commands accept ``--dry-run`` to preview the calculation without emitting
files. Consult :mod:`pycrystallography.core.models` for the Python APIs backing
these workflows.