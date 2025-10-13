from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import numpy as np

from pycrystallography.adapters.pymatgen_adapter import StructureLoader
from pycrystallography.cli.composite import build_orientation_relation, build_phases
from pycrystallography.config import load_config
from pycrystallography.core.models import CompositePattern
from pycrystallography.core.variant_manager import MarkerPalette, VariantManager
from pycrystallography.plotting import CrystallographicFigure, PlotSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_TEMPLATE = textwrap.dedent(
    """
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
        or_document: {or_doc}
        or_name: burgers-zr
    tem:
      zone_axis: [1, 1, 0]
      voltage: 200.0
      camera_length: 160.0
      intensity_threshold: 0.001
    """
)


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    or_doc = Path(__file__).resolve().parents[1] / "examples" / "or_zr.yaml"
    config_yaml = CONFIG_TEMPLATE.format(or_doc=or_doc.as_posix())
    path = tmp_path / "config.yaml"
    path.write_text(config_yaml)
    return path


@pytest.fixture()
def orientation_bundle(sample_config: Path):
    cfg = load_config(sample_config)
    registry = Path(__file__).resolve().parents[1] / "data" / "registry.yaml"
    loader = StructureLoader.from_yaml(registry)
    phases = build_phases(cfg, loader)
    relation, variants, spec = build_orientation_relation(cfg, phases, "burgers-zr")
    q_values = []
    intensities = []
    labels = []
    hkls = []
    for index, variant in enumerate(variants):
        base_q = 0.9 + 0.12 * index
        q_values.extend([base_q, base_q + 0.02])
        intensities.extend([0.3 + 0.1 * index, 0.32 + 0.1 * index])
        labels.extend([variant.label, variant.label])
        hkls.extend([(1, 0, 0), (1, 1, 0)])
    q_array = np.array(q_values, dtype=float)
    intensity_array = np.array(intensities, dtype=float)
    label_array = np.array(labels, dtype="U32")
    rotation_vectors = np.column_stack((q_array, np.linspace(0.05, 0.05 * len(q_array), len(q_array))))
    metadata = {
        "zone_axis": cfg.tem.zone_axis,
        "rotation_vectors": rotation_vectors,
    }
    pattern = CompositePattern(
        identifier="test-pattern",
        variants=tuple(variants),
        q_values=q_array,
        intensities=intensity_array,
        variant_labels=label_array,
        hkls=tuple(hkls),
        metadata=metadata,
    )
    settings = PlotSettings.from_mapping({})
    palette = MarkerPalette(
        shapes=settings.markers.get("shapes", ("o",)),
        colors=settings.markers.get("colors", ("#1f77b4",)),
    )
    manager = VariantManager(variants, palette=palette)
    figure = CrystallographicFigure(pattern, manager, settings=settings, backend="Agg")
    return cfg, relation, variants, spec, manager, pattern, figure
