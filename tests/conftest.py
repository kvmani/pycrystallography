from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_YAML = textwrap.dedent(
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
        parent_directions:
          - direction: [1, 1, 0]
          - direction: [1, -1, 1]
          - plane: [1, 1, 0]
        child_directions:
          - direction: [0, 0, 1]
          - direction: [1, 0, 0]
          - plane: [0, 0, 0, 1]
    tem:
      zone_axis: [1, 1, 0]
      voltage: 200.0
      camera_length: 160.0
      intensity_threshold: 0.001
    """
)


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML)
    return path
