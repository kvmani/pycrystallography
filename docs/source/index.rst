##.. image:: _static/pycrystallography.png
##   :width: 300 px
##   :alt: pycrystallography
##   :align: center

Getting started
----------------

The ``pcg`` command line interface provides ready-to-run examples that exercise the
configuration system, orientation relationship tooling, and diffraction calculators.

Validate the bundled Burgers orientation relationship configuration::

   pcg config validate --config ../examples/burgers_zr.yaml --print

Generate a composite SAED pattern (dry run)::

   pcg composite tem --relation burgers-zr --config ../examples/burgers_zr.yaml --dry-run

Compute a powder XRD peak list for the α-Zr phase (``--dry-run`` skips file writes)::

   pcg powder xrd --phase alpha-zr --config ../examples/burgers_zr.yaml --dry-run

Drop the ``--dry-run`` option to emit both a ``CSV`` table and publication-ready
plot. Replace ``alpha-zr`` with ``beta-zr`` to study the body-centred cubic phase.

Orientation relationships
-------------------------

Read :doc:`orientation` for an in-depth explanation of how direction/plane pairs are
converted into orientation matrices via :mod:`orix`, how variants are enumerated, and
how these relations feed diffraction calculations.

API documentation
-----------------

For detailed documentation of all modules and classes, please refer to the
:doc:`API docs </modules>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

