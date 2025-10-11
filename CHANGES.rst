Unreleased
----------
* Restored composite diffraction UX from ``main`` with CIF-backed structures, OR YAML
  schema validation, interactive plotting improvements (variant toggles, annotation
  avoidance) and HTML reporting driven by ``orix``/``pymatgen`` integrations.
* Rebuilt the package around a modular architecture with configuration, adapters, plug-in
  diffraction calculators and a Typer-based CLI.
* Added configuration schema with environment overrides and new tests covering loaders,
  variant generation and CLI dry-run behaviour.
* Added a powder XRD workflow (CLI + plotting) leveraging pymatgen's peak calculator.
* Documented orientation relationships, the Burgers OR example, and powder workflows in
  the Sphinx docs and README. Added a tutorial notebook comparing α/β zirconium phases.
* Validated Miller–Bravais indices, added plane-normal handling in the orientation builder,
  and refactored ``OrientationFactory`` to fit rotations from all supplied pairs while
  rejecting colinear inputs.
* Preserved TEM camera-length precision in ``DiffractionData`` and introduced parent→child
  feature mapping utilities with a ``pcg or map`` CLI command and comprehensive tests.

v1.0.0
------
* Alpha of Pycrystallography
