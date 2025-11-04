import { useMemo, useState } from 'react';

import { useDiffractionStore } from '../hooks/useDiffractionStore';
import type { PlaneOverlay } from '../types/structure';

import { PlaneSettingsModal } from './PlaneSettingsModal';

type VectorTriple = [number, number, number];

const numberInputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 10px',
  borderRadius: '10px',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  background: 'rgba(15, 23, 42, 0.6)',
  color: '#f8fafc'
};

const palette = ['#38bdf8', '#f97316', '#a855f7', '#10b981', '#ef4444'];
const arrowPalette = ['#facc15', '#f472b6', '#c084fc', '#22d3ee', '#fb7185'];

function parseTriple(values: [string, string, string]): VectorTriple {
  return values.map((value) => Number(value) || 0) as VectorTriple;
}

function formatMiller([h, k, l]: VectorTriple) {
  return `(${h} ${k} ${l})`;
}

function formatDirection([u, v, w]: VectorTriple) {
  return `[${u} ${v} ${w}]`;
}

function formatLabel(overlay: PlaneOverlay) {
  return `${formatMiller(overlay.hkl)}${formatDirection(overlay.uvw)}`;
}

export function PlaneOverlayControls() {
  const overlays = useDiffractionStore((state) => state.planeOverlays);
  const addPlane = useDiffractionStore((state) => state.addPlaneOverlay);
  const removePlane = useDiffractionStore((state) => state.removePlaneOverlay);
  const setPlaneVisibility = useDiffractionStore((state) => state.setPlaneVisibility);
  const updatePlane = useDiffractionStore((state) => state.updatePlaneOverlay);
  const clearPlanes = useDiffractionStore((state) => state.clearPlaneOverlays);

  const [hklInputs, setHklInputs] = useState<[string, string, string]>(['1', '1', '0']);
  const [uvwInputs, setUvwInputs] = useState<[string, string, string]>(['1', '0', '0']);
  const [error, setError] = useState<string | null>(null);
  const [activeOverlayId, setActiveOverlayId] = useState<string | null>(null);

  const activeOverlay = useMemo(
    () => overlays.find((item) => item.id === activeOverlayId) ?? null,
    [activeOverlayId, overlays]
  );

  const handleAddPlane = () => {
    const hkl = parseTriple(hklInputs);
    const uvw = parseTriple(uvwInputs);
    const hklLength = Math.hypot(hkl[0], hkl[1], hkl[2]);
    if (hklLength === 0) {
      setError('Miller indices (hkl) must not be all zeros.');
      return;
    }
    const colorIndex = overlays.length % palette.length;
    const uvwLength = Math.hypot(uvw[0], uvw[1], uvw[2]);
    const direction = uvwLength === 0 ? ([0, 1, 0] as VectorTriple) : uvw;
    addPlane({
      hkl,
      uvw: direction,
      color: palette[colorIndex],
      arrowColor: arrowPalette[colorIndex % arrowPalette.length],
      opacity: 0.45,
      offset: 0,
      arrowLength: 2.5,
      visible: true,
      label: formatMiller(hkl) + formatDirection(direction)
    });
    setError(null);
    setHklInputs(['1', '1', '0']);
    setUvwInputs(['1', '0', '0']);
  };

  return (
    <section
      style={{
        display: 'grid',
        gap: '16px',
        padding: '20px',
        borderRadius: '16px',
        background: 'rgba(15, 23, 42, 0.7)',
        border: '1px solid rgba(148, 163, 184, 0.15)'
      }}
    >
      <header>
        <h3 style={{ margin: 0 }}>Crystallographic overlays</h3>
        <p style={{ margin: '4px 0 0', color: 'rgba(148, 163, 184, 0.85)', fontSize: '14px' }}>
          Add planes and directions in crystal coordinates. Each entry renders a semi-transparent mesh and
          in-plane arrow directly in the viewer.
        </p>
      </header>

      <div style={{ display: 'grid', gap: '12px' }}>
        <div style={{ display: 'flex', gap: '12px' }}>
          {hklInputs.map((value, index) => (
            <input
              key={`add-hkl-${index}`}
              type="number"
              value={value}
              onChange={(event) => {
                const next = [...hklInputs] as [string, string, string];
                next[index] = event.target.value;
                setHklInputs(next);
              }}
              style={numberInputStyle}
              placeholder={['h', 'k', 'l'][index]}
            />
          ))}
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          {uvwInputs.map((value, index) => (
            <input
              key={`add-uvw-${index}`}
              type="number"
              value={value}
              onChange={(event) => {
                const next = [...uvwInputs] as [string, string, string];
                next[index] = event.target.value;
                setUvwInputs(next);
              }}
              style={numberInputStyle}
              placeholder={['u', 'v', 'w'][index]}
            />
          ))}
        </div>
        {error ? (
          <div style={{ color: '#fda4af', fontSize: '13px' }} data-testid="plane-overlay-error">
            {error}
          </div>
        ) : null}
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end' }}>
          <button
            type="button"
            onClick={() => {
              clearPlanes();
              setError(null);
            }}
            style={{
              padding: '10px 16px',
              borderRadius: '12px',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              background: 'transparent',
              color: '#e2e8f0',
              cursor: 'pointer'
            }}
          >
            Clear overlays
          </button>
          <button
            type="button"
            onClick={handleAddPlane}
            style={{
              padding: '10px 18px',
              borderRadius: '12px',
              border: 'none',
              background: '#22d3ee',
              color: '#0f172a',
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            ➕ Add plane
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gap: '10px' }}>
        <h4 style={{ margin: '8px 0 0' }}>Active overlays</h4>
        {overlays.length === 0 ? (
          <p style={{ margin: 0, color: 'rgba(148, 163, 184, 0.75)' }}>
            No overlays defined. Use the form above to add (hkl)[uvw] entries.
          </p>
        ) : (
          overlays.map((overlay) => (
            <div
              key={overlay.id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px 12px',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: '12px',
                border: '1px solid rgba(148, 163, 184, 0.2)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span
                  style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '50%',
                    border: '2px solid rgba(148, 163, 184, 0.35)',
                    background: overlay.color
                  }}
                />
                <div>
                  <div style={{ fontWeight: 600 }}>{formatLabel(overlay)}</div>
                  <div style={{ fontSize: '13px', color: 'rgba(148, 163, 184, 0.75)' }}>
                    offset {overlay.offset.toFixed(2)}, opacity {overlay.opacity.toFixed(2)}
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px' }}>
                  <input
                    type="checkbox"
                    checked={overlay.visible}
                    onChange={(event) => setPlaneVisibility(overlay.id, event.target.checked)}
                  />
                  Visible
                </label>
                <button
                  type="button"
                  onClick={() => setActiveOverlayId(overlay.id)}
                  style={{
                    padding: '6px 10px',
                    borderRadius: '10px',
                    border: '1px solid rgba(148, 163, 184, 0.35)',
                    background: 'transparent',
                    color: '#e2e8f0',
                    cursor: 'pointer'
                  }}
                  aria-label={`Configure ${formatLabel(overlay)}`}
                >
                  ⚙️
                </button>
                <button
                  type="button"
                  onClick={() => removePlane(overlay.id)}
                  style={{
                    padding: '6px 10px',
                    borderRadius: '10px',
                    border: '1px solid rgba(248, 113, 113, 0.35)',
                    background: 'rgba(248, 113, 113, 0.15)',
                    color: '#fecaca',
                    cursor: 'pointer'
                  }}
                  aria-label={`Remove ${formatLabel(overlay)}`}
                >
                  ✕
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {activeOverlay ? (
        <PlaneSettingsModal
          overlay={activeOverlay}
          onClose={() => setActiveOverlayId(null)}
          onUpdate={(updates) => updatePlane(activeOverlay.id, updates)}
        />
      ) : null}
    </section>
  );
}
