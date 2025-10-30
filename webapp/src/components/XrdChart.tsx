import Plot from 'react-plotly.js';

import { useDiffractionStore } from '../hooks/useDiffractionStore';

export function XrdChart() {
  const xrdPattern = useDiffractionStore((state) => state.xrdPattern);
  const structure = useDiffractionStore((state) => state.structure);

  if (!xrdPattern || !structure) {
    return (
      <div
        style={{
          height: '360px',
          borderRadius: '16px',
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(148, 163, 184, 0.15)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'rgba(148, 163, 184, 0.8)'
        }}
      >
        Generate the diffraction pattern to preview the plot.
      </div>
    );
  }

  const x = xrdPattern.peaks.map((peak) => peak.two_theta);
  const y = xrdPattern.peaks.map((peak) => peak.intensity);
  const texts = xrdPattern.peaks.map((peak) => `(${peak.hkl.join(' ')})`);

  return (
    <div style={{ borderRadius: '16px', overflow: 'hidden', border: '1px solid rgba(148, 163, 184, 0.15)' }}>
      <Plot
        data={[
          {
            x,
            y,
            type: 'bar',
            marker: { color: '#6366f1' },
            text: texts,
            hovertemplate: '2θ=%{x:.2f}°<br>I=%{y:.2f}<br>hkl=%{text}<extra></extra>'
          }
        ]}
        layout={{
          title: `${structure.name ?? 'Phase'} powder XRD`,
          paper_bgcolor: 'rgba(15, 23, 42, 0.9)',
          plot_bgcolor: 'rgba(15, 23, 42, 0.9)',
          font: { color: '#e2e8f0' },
          margin: { l: 60, r: 10, t: 50, b: 60 },
          xaxis: { title: '2θ (°)', color: '#cbd5f5' },
          yaxis: { title: 'Relative intensity', color: '#cbd5f5' },
          height: 360
        }}
        config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
