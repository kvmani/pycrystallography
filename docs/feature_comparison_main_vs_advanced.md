# Feature comparison: `main` vs. `advanced`

The following matrix captures the interactive diffraction and orientation-relation tooling
available in the legacy `main` branch (tagged from `origin/master`) versus the modernised
`advanced` branch (this worktree). File references use repository-relative paths with
Python symbol anchors.

| # | User-facing capability | `main` implementation | `advanced` status (pre-migration) | Gap summary |
| - | ---------------------- | --------------------- | --------------------------------- | ----------- |
| 1 | Composite diffraction panels with per-variant include/exclude toggles | `pycrystallography/core/crystallographyFigure.py::CrystallographyFigure._init_composite_controls` (CheckButtons at L314–L365) | Absent; `pycrystallography/plotting/composite.py::plot_tem_pattern` renders static scatter with no interactivity | Reintroduce variant visibility manager and interactive widgets | 
| 2 | Hover tooltips exposing d-spacing, reciprocal spacing and variant metadata | `crystallographyFigure.py::CrystallographyFigure.hover` (L188–L277) | Missing; no hover callbacks registered in `advanced` plotting modules | Implement consolidated tooltip pipeline with configurable fields | 
| 3 | Spot hover/click showing indexed (hkl) metadata | `crystallographyFigure.py::CrystallographyFigure.hover_botonAnnotation` and `onclick` (L279–L420) | Missing; pattern scatter lacks annotations | Add click-to-annotate flow producing non-overlapping labels |
| 4 | Unique, configurable markers per variant, propagated to legends/reports | Marker settings defined in `defaultOptions['SAED']` (L47–L74) and consumed in `_plotSelectedPatterns` | Partial; markers default to matplotlib cycle and collide between variants | Create deterministic marker/style allocator honouring YAML config |
| 5 | Automated spot label layout avoiding overlap when multiple reflections share coordinates | `_updateStaticAnnotationPositions` (L422–L511) performs greedy offset search | Missing; no annotation support | Provide layout engine with bounding-box collision avoidance |
| 6 | HTML phase report bundling OR metadata, Euler angles, configuration snapshot and generated figures | `crystallographyFigure.py::CrystallographyFigure.makeReport` (L620–L820) | Missing in `advanced`; only image export logic exists | Build new report builder leveraging CIF+YAML metadata |

## Gap closure checklist

- [ ] Variant visibility state machine with persistence
- [ ] Tooltip manager integrating configurable fields (d-spacing, g, intensity, variant label, (hkl))
- [ ] Annotation engine with overlap avoidance and batch updates for coincident reflections
- [ ] Marker palette allocator driven by `configs/plot_settings.yaml`
- [ ] HTML report generator per phase with assets saved alongside metadata snapshot
- [ ] CIF-backed metadata ingestion and validation for OR YAML documents

## Evidence capture

Screenshots and GIFs demonstrating the restored interactive experience will be attached to
the pull request once the migration work is complete.
