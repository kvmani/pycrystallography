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
         - direction: [1, 1, 0]
         - direction: [1, -1, 1]
         - plane: [1, 1, 0]
       child_directions:
         - direction: [0, 0, 1]
         - direction: [1, 0, 0]
         - plane: [0, 0, 0, 1]

Each entry may describe either a direction or a plane normal. Directions and
planes can be expressed in three-index notation or in the four-index
Miller–Bravais form (``[u v t w]``/``(h k i l)``). Four-index values are
validated—``u + v + t = 0`` for directions, ``h + k + i = 0`` for planes—and
converted to Cartesian vectors before being passed to :mod:`orix`.

Computing the orientation matrix
--------------------------------

The :class:`pycrystallography.adapters.orix_adapter.OrientationFactory` now uses
all supplied direction/plane pairs simultaneously. A singular value
decomposition determines the least-squares rotation that maps each child vector
onto its parent counterpart while rejecting colinear inputs with a descriptive
``ValueError``. The resulting
:class:`orix.quaternion.orientation.Orientation` is numerically stable and
respects lattice symmetry.

Once the fundamental orientation has been obtained, the
:class:`pycrystallography.adapters.orix_adapter.VariantGenerator` enumerates all
symmetry-equivalent variants by combining the parent and child point groups. The
resulting :class:`pycrystallography.core.models.Variant` objects keep track of the
parent/child phases and carry a deterministic ``label`` for downstream tooling.

From ORs to diffraction
-----------------------

Variants generated from an OR are passed to diffraction calculators. The
``pcg composite tem`` command rotates each variant structure and evaluates a
composite SAED pattern. The ``pcg powder xrd`` command complements this by
computing powder diffraction peak lists for single phases, making it easy to
compare, for example, the α and β zirconium structures defined above.

Mapping parent features to child variants
-----------------------------------------

The new ``pcg or map`` command projects parent directions and planes into the
child lattice for every variant. If no features are specified on the command
line the tool reuses the configuration inputs. For example::

   pcg or map --relation burgers-zr --config ../examples/burgers_zr.yaml

produces output similar to::

   Variant burgers-zr-v01
     Direction [1 1 0] -> [0 0 1] (four-index [0 0 0 1])
     Direction [1 -1 1] -> [1 0 0] (four-index [2 -1 -1 0])
     Plane     (1 1 0) -> (0 0 1) (four-index (0 0 0 1))

This mapping functionality is also available via
:func:`pycrystallography.analysis.orientation_mapping.map_parent_features_to_child_variants`
for programmatic workflows.

To experiment interactively, load the example configuration and run::

   pcg composite tem --relation burgers-zr --config ../examples/burgers_zr.yaml
   pcg powder xrd --phase alpha-zr --config ../examples/burgers_zr.yaml

Both commands accept ``--dry-run`` to preview the calculation without emitting
files. Consult :mod:`pycrystallography.core.models` for the Python APIs backing
these workflows.