# Developer guide

## Architecture overview

```
pycrystallography/
├── adapters/              # Thin wrappers around third-party libraries
├── analysis/              # Calculators, reports, orientation mapping
├── cli/                   # Typer-based entry points
├── config/                # Pydantic models and loaders
├── core/                  # Domain models and variant management
├── io/                    # CIF and OR YAML ingestion
├── plotting/              # Interactive figure primitives
└── tests/                 # Unit, property, UI, and performance checks
```

Key modules:

* `io.cif.StructureCache`: caches resolved CIF paths and verifies lattice
  metadata.
* `io.or_yaml.OrientationLibrary`: validates OR YAML files, resolving CIF
  references and constructing `OrientationRelation` objects with variant lists.
* `core.variant_manager.VariantManager`: maintains visibility/state and marker
  assignments across variants.
* `plotting.crystallographic_figure.CrystallographicFigure`: Matplotlib-based
  interactive composite diffraction plot with tooltips, annotations, and widget
  integration.
* `analysis.reports.html.HtmlReportBuilder`: produces standalone HTML reports
  bundling plots, variant metadata, and configuration snapshots.

## Extending the pipeline

1. **Add a new orientation document**
   * Place the YAML under `examples/` or a project-specific directory.
   * Reference the CIF (relative or absolute path) and update the
     configuration (`orientation_relations[*].or_document`).
   * Validate with `pcg config validate`.
2. **Custom plotting styles**
   * Copy `configs/plot_settings.yaml` and adjust `markers`, `annotations`,
     `tooltip`, and `widgets` as desired.
   * Pass `PlotSettings.from_yaml(...)` into `CrystallographicFigure` or place
     the file next to your configuration so the CLI picks it up automatically.
3. **Reports**
   * Subclass `HtmlReportBuilder` or provide an alternate builder in
     `analysis/reports` if additional sections or templating engines are
     required. The CLI can be extended by injecting a different report factory.

## Performance considerations

* Variant enumeration uses `orix` symmetry operators and is typically
  sub-millisecond for < 32 variants. Property tests in `tests/test_variant_manager.py`
  ensure deterministic ordering.
* `CrystallographicFigure` caches scatter data and only toggles visibility on
  existing artists, avoiding re-computation when variants are toggled.
* Annotation placement uses a greedy polar sweep with cached bounding boxes;
  the radius/steps can be tuned via `plot_settings.yaml`.
* The HTML report builder only renders once per CLI invocation and reuses the
  same figure save call as the PNG export.

## Coding standards

* Python ≥3.10 with type hints enforced by `mypy` (configured in `pyproject.toml`).
* `ruff` handles linting/formatting. Run `ruff check --fix .` before committing.
* Unit tests live under `tests/` with the following categories:
  * Configuration and IO validation (`test_config_loader.py`, `test_or_yaml.py`).
  * Adapters (`test_pymatgen_adapter.py`, `test_variant_generator.py`).
  * Plotting/reporting (`test_crystallographic_figure.py`, `test_html_report.py`).
  * Documentation invariants (`test_docs_feature_table.py`).
* Determinism is paramount; avoid reliance on random state. Marker palettes and
  annotation layouts use deterministic iteration order.

## CLI tips

* `pcg composite tem --help` lists all composite options, including
  `--calculator` overrides and `--dry-run` mode.
* Use `pcg report phase` (planned) or instantiate `HtmlReportBuilder` directly
  to regenerate reports after tweaking plot settings.
* Set `MPLBACKEND=Agg` in CI or headless environments when running scripts that
  construct `CrystallographicFigure` instances interactively.
