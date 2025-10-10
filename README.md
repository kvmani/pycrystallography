PyCrystallography
=================

PyCrystallography is a modern orchestration layer for composite diffraction analysis. The library
delegates crystallographic maths to `orix <https://orix.readthedocs.io>`_ and structure handling to
`pymatgen <https://pymatgen.org>`_, focusing on configuration management, plug-in extensibility and
user experience.

Highlights
----------

* Strict configuration schema powered by Pydantic with support for environment and CLI overrides.
* Adapters that bridge pycrystallography's domain models to orix (orientations, variants) and
  pymatgen (structure loading, TEM diffraction calculators).
* Plug-in architecture for diffraction calculators discoverable through entry points. A composite
  TEM/SAED implementation is included by default.
* Typer-based CLI (``pcg``) with dry-run mode, logging controls and multi-format outputs.

Quick start
-----------

1. Install the project in a virtual environment::

       pip install -e .[dev]

2. Validate a configuration::

       pcg config validate --config examples/burgers_zr.yaml --print

3. Generate a composite Burgers OR pattern for Zr (dry run)::

       pcg composite tem --relation burgers-zr --config examples/burgers_zr.yaml --dry-run

Outputs (CSV + PNG) are written to ``./output`` by default. Use ``--out`` to change the destination.
See ``docs/`` for tutorials on extending the calculator registry and building plug-ins.
