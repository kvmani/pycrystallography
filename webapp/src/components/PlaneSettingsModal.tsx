import { useEffect, useState } from 'react';
import { HexColorPicker } from 'react-colorful';

import type { PlaneOverlay } from '../types/structure';

interface PlaneSettingsModalProps {
  overlay: PlaneOverlay;
  onClose: () => void;
  onUpdate: (updates: Partial<PlaneOverlay>) => void;
}

type VectorTriple = [number, number, number];

function parseNumber(value: string): number {
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return 0;
  }
  return parsed;
}

export function PlaneSettingsModal({ overlay, onClose, onUpdate }: PlaneSettingsModalProps) {
  const [localHkl, setLocalHkl] = useState<VectorTriple>(overlay.hkl);
  const [localUvw, setLocalUvw] = useState<VectorTriple>(overlay.uvw);

  useEffect(() => {
    setLocalHkl(overlay.hkl);
    setLocalUvw(overlay.uvw);
  }, [overlay]);

  const updateVectorValue = (index: number, next: number, setter: (value: VectorTriple) => void, vector: VectorTriple) => {
    const copy = [...vector] as VectorTriple;
    copy[index] = next;
    setter(copy);
  };

  const handleSave = () => {
    const hklLength = Math.hypot(localHkl[0], localHkl[1], localHkl[2]);
    if (hklLength === 0) {
      return;
    }
    const uvwLength = Math.hypot(localUvw[0], localUvw[1], localUvw[2]);
    const safeUvw: VectorTriple = uvwLength === 0 ? [0, 1, 0] : localUvw;
    onUpdate({ hkl: localHkl, uvw: safeUvw });
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15, 23, 42, 0.65)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '20px'
      }}
    >
      <div
        style={{
          width: 'min(520px, 100%)',
          maxHeight: '90vh',
          overflowY: 'auto',
          background: 'rgba(15, 23, 42, 0.95)',
          borderRadius: '20px',
          border: '1px solid rgba(148, 163, 184, 0.2)',
          padding: '24px',
          display: 'grid',
          gap: '18px'
        }}
      >
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h3 style={{ margin: 0 }}>Plane settings</h3>
            <p style={{ margin: '6px 0 0', color: 'rgba(148, 163, 184, 0.85)' }}>
              Adjust rendering attributes for the (hkl)[uvw] overlay.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              border: 'none',
              background: 'transparent',
              color: '#94a3b8',
              fontSize: '20px',
              cursor: 'pointer'
            }}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        <section style={{ display: 'grid', gap: '12px' }}>
          <h4 style={{ margin: 0 }}>Miller indices (hkl)</h4>
          <div style={{ display: 'flex', gap: '12px' }}>
            {localHkl.map((value, index) => (
              <input
                key={`hkl-${index}`}
                type="number"
                value={value}
                onChange={(event) =>
                  updateVectorValue(index, parseNumber(event.target.value), setLocalHkl, localHkl)
                }
                style={{
                  flex: 1,
                  padding: '8px 10px',
                  borderRadius: '12px',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  background: 'rgba(15, 23, 42, 0.6)',
                  color: '#f8fafc'
                }}
              />
            ))}
          </div>
        </section>

        <section style={{ display: 'grid', gap: '12px' }}>
          <h4 style={{ margin: 0 }}>[uvw] direction</h4>
          <div style={{ display: 'flex', gap: '12px' }}>
            {localUvw.map((value, index) => (
              <input
                key={`uvw-${index}`}
                type="number"
                value={value}
                onChange={(event) =>
                  updateVectorValue(index, parseNumber(event.target.value), setLocalUvw, localUvw)
                }
                style={{
                  flex: 1,
                  padding: '8px 10px',
                  borderRadius: '12px',
                  border: '1px solid rgba(148, 163, 184, 0.3)',
                  background: 'rgba(15, 23, 42, 0.6)',
                  color: '#f8fafc'
                }}
              />
            ))}
          </div>
          <p style={{ margin: 0, fontSize: '13px', color: 'rgba(148, 163, 184, 0.75)' }}>
            The direction will be projected onto the plane automatically if needed.
          </p>
        </section>

        <section style={{ display: 'grid', gap: '14px' }}>
          <div style={{ display: 'grid', gap: '8px' }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Plane opacity</span>
              <span style={{ color: 'rgba(148, 163, 184, 0.85)' }}>{overlay.opacity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={overlay.opacity}
              onChange={(event) => onUpdate({ opacity: Number(event.target.value) })}
            />
          </div>

          <div style={{ display: 'grid', gap: '8px' }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Offset along normal</span>
              <span style={{ color: 'rgba(148, 163, 184, 0.85)' }}>{overlay.offset.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={-2}
              max={2}
              step={0.05}
              value={overlay.offset}
              onChange={(event) => onUpdate({ offset: Number(event.target.value) })}
            />
          </div>

          <div style={{ display: 'grid', gap: '8px' }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Arrow length</span>
              <span style={{ color: 'rgba(148, 163, 184, 0.85)' }}>{overlay.arrowLength.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={10}
              step={0.1}
              value={overlay.arrowLength}
              onChange={(event) => onUpdate({ arrowLength: Number(event.target.value) })}
            />
          </div>
        </section>

        <section style={{ display: 'grid', gap: '16px' }}>
          <div style={{ display: 'grid', gap: '8px' }}>
            <span>Plane color</span>
            <HexColorPicker color={overlay.color} onChange={(value) => onUpdate({ color: value })} />
          </div>
          <div style={{ display: 'grid', gap: '8px' }}>
            <span>Direction arrow color</span>
            <HexColorPicker color={overlay.arrowColor} onChange={(value) => onUpdate({ arrowColor: value })} />
          </div>
        </section>

        <footer style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '10px 16px',
              borderRadius: '12px',
              border: '1px solid rgba(148, 163, 184, 0.25)',
              background: 'transparent',
              color: '#e2e8f0',
              cursor: 'pointer'
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            style={{
              padding: '10px 18px',
              borderRadius: '12px',
              border: 'none',
              background: '#2563eb',
              color: '#f8fafc',
              cursor: 'pointer',
              fontWeight: 600
            }}
          >
            Save changes
          </button>
        </footer>
      </div>
    </div>
  );
}
