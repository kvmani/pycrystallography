"""HTML report generation for crystallographic analyses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from ...core.models import OrientationRelation, Phase
from ...core.variant_manager import VariantState
from ...io.or_yaml import OrientationRelationSpec
from ...plotting.crystallographic_figure import CrystallographicFigure


@dataclass(slots=True)
class ReportArtifacts:
    html_path: Path
    image_path: Path


class HtmlReportBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

    def build(
        self,
        *,
        phase: Phase,
        relation: OrientationRelation,
        variants: Sequence[VariantState],
        orientation_spec: OrientationRelationSpec,
        figure: CrystallographicFigure,
        configuration: Mapping[str, object],
    ) -> ReportArtifacts:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        image_path = self.output_dir / "figures" / f"{relation.name}.png"
        figure.save(image_path, dpi=int(figure.settings.figure.get("dpi", 160)))
        variant_rows = []
        for state in variants:
            variant_rows.append(
                {
                    "label": state.variant.label,
                    "visible": state.visible,
                    "marker": state.style.marker,
                    "color": state.style.color,
                }
            )
        warnings = list(phase.metadata.get("warnings", [])) if isinstance(phase.metadata, Mapping) else []
        html = self._render_html(
            phase=phase,
            relation=relation,
            variants=variant_rows,
            orientation_spec=orientation_spec,
            image_path=image_path.relative_to(self.output_dir),
            configuration=configuration,
            timestamp=timestamp,
            warnings=warnings,
        )
        html_path = self.output_dir / f"{relation.name}.html"
        html_path.write_text(html, encoding="utf-8")
        return ReportArtifacts(html_path=html_path, image_path=image_path)

    def _render_html(
        self,
        *,
        phase: Phase,
        relation: OrientationRelation,
        variants: Sequence[Mapping[str, object]],
        orientation_spec: OrientationRelationSpec,
        image_path: Path,
        configuration: Mapping[str, object],
        timestamp: str,
        warnings: Sequence[str],
    ) -> str:
        def _fmt_indices(values: Sequence[int]) -> str:
            return "(" + " ".join(str(v) for v in values) + ")"

        variants_html = "\n".join(
            f"<tr><td>{row['label']}</td><td>{'Yes' if row['visible'] else 'No'}</td><td>{row['marker']}</td><td><span style=\"color:{row['color']}\">{row['color']}</span></td></tr>"
            for row in variants
        )
        warnings_html = "\n".join(f"<li>{message}</li>" for message in warnings) or "<li>None</li>"
        euler_parent = orientation_spec.Euler_parent
        euler_child = orientation_spec.Euler_product
        euler_html = ""
        if euler_parent or euler_child:
            parent_str = (
                ", ".join(f"{value:.2f}°" for value in euler_parent)
                if euler_parent
                else "Not provided"
            )
            child_str = (
                ", ".join(f"{value:.2f}°" for value in euler_child)
                if euler_child
                else "Not provided"
            )
            euler_html = f"<p><strong>Euler angles</strong> (Bunge ZXZ, degrees)<br>Parent: {parent_str}<br>Product: {child_str}</p>"
        config_json = json.dumps(configuration, indent=2, default=str)
        return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>{phase.name} – {relation.name} report</title>
    <style>
      body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2rem; color: #1a202c; }}
      h1 {{ font-size: 1.8rem; margin-bottom: 0.2rem; }}
      h2 {{ margin-top: 1.6rem; font-size: 1.3rem; }}
      table {{ border-collapse: collapse; width: 100%; margin-top: 0.8rem; }}
      th, td {{ border: 1px solid #cbd5e0; padding: 0.4rem 0.6rem; text-align: left; }}
      th {{ background: #edf2f7; }}
      pre {{ background: #f7fafc; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }}
      .meta {{ color: #4a5568; font-size: 0.9rem; }}
    </style>
  </head>
  <body>
    <h1>{phase.name} – {relation.name}</h1>
    <p class=\"meta\">Generated {timestamp}</p>
    <section>
      <h2>Orientation relationship</h2>
      <p><strong>Parent phase:</strong> {relation.parent_phase.name}<br>
         <strong>Product phase:</strong> {relation.child_phase.name}</p>
      <p><code>(hkl)_P { _fmt_indices(orientation_spec.hkl_parent) } [uvw]_P { _fmt_indices(orientation_spec.uvw_parent) }</code></p>
      <p><code>(hkl)_C { _fmt_indices(orientation_spec.hkl_product) } [uvw]_C { _fmt_indices(orientation_spec.uvw_product) }</code></p>
      {euler_html}
    </section>
    <section>
      <h2>Variants</h2>
      <table>
        <thead>
          <tr><th>Label</th><th>Visible</th><th>Marker</th><th>Colour</th></tr>
        </thead>
        <tbody>
          {variants_html}
        </tbody>
      </table>
    </section>
    <section>
      <h2>Warnings</h2>
      <ul>
        {warnings_html}
      </ul>
    </section>
    <section>
      <h2>Composite pattern</h2>
      <img src=\"{image_path.as_posix()}\" alt=\"Composite diffraction pattern\" style=\"max-width:100%;height:auto;border:1px solid #cbd5e0;border-radius:0.5rem;\">
    </section>
    <section>
      <h2>Configuration snapshot</h2>
      <pre>{config_json}</pre>
    </section>
  </body>
</html>
"""


__all__ = ["HtmlReportBuilder", "ReportArtifacts"]
