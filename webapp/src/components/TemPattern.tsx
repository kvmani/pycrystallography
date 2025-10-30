import Plot from 'react-plotly.js';

import { useDiffractionStore } from '../hooks/useDiffractionStore';

export function TemPattern() {
  const temPattern = useDiffractionStore((state) => state.temPattern);
  const temSettings = useDiffractionStore((state) => state.temSettings);

  if (!temPattern) {
    return (
      <div
        data-testid="tem-pattern"
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
        Select a zone axis and generate the TEM pattern.
      </div>
    );
  }

  const xs = temPattern.reflections.map((reflection) => reflection.position[0]);
  const ys = temPattern.reflections.map((reflection) => reflection.position[1]);
  const intensities = temPattern.reflections.map((reflection) => reflection.intensity);
  const hkls = temPattern.reflections.map((reflection) => `(${reflection.hkl.join(' ')})`);

  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const margin = 0.1 * Math.max(maxX - minX, maxY - minY, 1);

  const markerSizes = intensities.map((intensity) => 6 + intensity * 40);

  return (
    <div
      data-testid="tem-pattern"
      style={{ borderRadius: '16px', overflow: 'hidden', border: '1px solid rgba(148, 163, 184, 0.15)' }}
    >
      <Plot
        data={[
          {
            x: xs,
            y: ys,
            mode: 'markers+text',
            type: 'scatter',
            text: hkls,
            textposition: 'top center',
            textfont: { color: '#cbd5f5', size: 10 },
            marker: {
              size: markerSizes,
              color: intensities,
              colorscale: 'Viridis',
              showscale: true,
              colorbar: {
                title: 'Relative intensity',
                tickcolor: '#e2e8f0',
                titlefont: { color: '#e2e8f0' },
                tickfont: { color: '#e2e8f0' }
              }
            },
            hovertemplate:
              'x=%{x:.2f} Å<br>y=%{y:.2f} Å<br>I=%{marker.color:.2f}<br>hkl=%{text}<extra></extra>'
          }
        ]}
        layout={{
          title: `TEM diffraction (zone axis [${temSettings.zone_axis.join(' ')}])`,
          paper_bgcolor: 'rgba(15, 23, 42, 0.9)',
          plot_bgcolor: 'rgba(15, 23, 42, 0.9)',
          font: { color: '#e2e8f0' },
          margin: { l: 50, r: 20, t: 50, b: 50 },
          xaxis: {
            title: 'Detector X (Å)',
            color: '#cbd5f5',
            range: [minX - margin, maxX + margin],
            zeroline: true,
            zerolinecolor: '#475569'
          },
          yaxis: {
            title: 'Detector Y (Å)',
            color: '#cbd5f5',
            range: [minY - margin, maxY + margin],
            zeroline: true,
            zerolinecolor: '#475569',
            scaleanchor: 'x',
            scaleratio: 1
          },
          height: 360
        }}
        config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
