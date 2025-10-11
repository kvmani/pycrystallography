# User guide

## Orientation-relation workflow

1. **Prepare structures and OR metadata**
   * CIFs live under `data/structureData/` and are loaded via
     `pymatgen.Structure.from_file`.
   * Orientation relationships are encoded in YAML files that follow
     `schemas/or_schema.yaml`. See `examples/or_zr.yaml` for a concrete
     Burgers relation referencing the α-Zr CIF.
2. **Validate the configuration**
   ```bash
   pcg config validate --config examples/quickstart.yaml --print
   ```
   This command ensures that the OR YAML resolves, CIF metadata is consistent
   with the optional lattice parameters, and all phases referenced in the
   configuration are available.
3. **Generate a composite pattern**
   ```bash
   pcg composite tem --relation burgers-zr --config examples/quickstart.yaml
   ```
   The CLI will:
   * Enumerate symmetry-equivalent variants with `orix` using the space-group
     numbers derived from the CIF and YAML.
   * Compute a TEM composite pattern via the registered calculator.
   * Instantiate `CrystallographicFigure` with `configs/plot_settings.yaml`
     and write `PNG`, `CSV`, and `HTML` artefacts under `./output` (or the
     path passed via `--out`).

## Interactive figure features

`CrystallographicFigure` powers the interactive plot saved by the CLI and can
be scripted from Python for custom workflows:

```python
from pathlib import Path

from pycrystallography.plotting import CrystallographicFigure, PlotSettings
from pycrystallography.config import load_config
from pycrystallography.cli.composite import build_orientation_relation, build_phases
from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.core.variant_manager import MarkerPalette, VariantManager

cfg = load_config(Path("examples/quickstart.yaml"))
loader = StructureLoader.from_yaml(Path("data/registry.yaml"))
phases = build_phases(cfg, loader)
relation, variants, spec = build_orientation_relation(cfg, phases, "burgers-zr")

# Compute a pattern using the registered calculator (omitted here for brevity)
pattern = ...
settings = PlotSettings.from_yaml("configs/plot_settings.yaml")
palette = MarkerPalette(
    shapes=settings.markers["shapes"],
    colors=settings.markers["colors"],
)
manager = VariantManager(variants, palette=palette, default_visibility=spec.plot.visibility if spec and spec.plot else None)
figure = CrystallographicFigure(pattern, manager, settings=settings, backend="Agg")
figure.save("output/custom.png")
```

### Variant toggles

A `matplotlib.widgets.CheckButtons` panel is automatically embedded in the
figure. Toggling a variant updates the scatter artist, fades the corresponding
legend entry, and preserves tooltip/annotation state for the remaining variants.

### Hover tooltips

Moving the cursor over a reflection displays a tooltip with configurable
fields (default: variant label, reciprocal spacing `g`, `d` spacing, intensity,
and `(hkl)`). The tooltip content is driven by `configs/plot_settings.yaml` and
can be customised per project.

### Click-to-annotate

Left-clicking a reflection creates a styled `(hkl)` label that is automatically
placed to avoid overlapping existing annotations. The layout engine performs a
polar sweep around the point, testing candidate positions until a collision-free
placement is found.

### HTML reports

The CLI saves an offline HTML report per relation. Each report contains:

* Phase summary with CIF metadata validation warnings.
* Orientation relation expressed in `(hkl)[uvw]` notation with optional Euler
  angles.
* Variant catalogue showing marker/colour assignments and visibility defaults.
* Embedded composite pattern image (PNG) and a JSON configuration snapshot for
  reproducibility.

Reports are written beside the plot/CSV artefacts. They are static HTML files
that can be shared without additional dependencies.
