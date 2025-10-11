from __future__ import annotations

from pycrystallography.analysis.reports.html import HtmlReportBuilder


def test_html_report_generation(tmp_path, orientation_bundle):
    _cfg, relation, variants, spec, manager, pattern, figure = orientation_bundle
    builder = HtmlReportBuilder(tmp_path)
    report = builder.build(
        phase=relation.child_phase,
        relation=relation,
        variants=list(manager),
        orientation_spec=spec,
        figure=figure,
        configuration={"identifier": pattern.identifier},
    )
    assert report.html_path.exists()
    assert report.image_path.exists()
    content = report.html_path.read_text()
    assert relation.name in content
    assert pattern.identifier in content
