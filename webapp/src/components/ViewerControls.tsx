import { useMemo, useState } from 'react';
import { HexColorPicker } from 'react-colorful';

import { useDiffractionStore } from '../hooks/useDiffractionStore';

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 10px',
  borderRadius: '10px',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  background: 'rgba(15, 23, 42, 0.6)',
  color: '#f8fafc'
};

export function ViewerControls() {
  const structure = useDiffractionStore((state) => state.structure);
  const supercell = useDiffractionStore((state) => state.supercell);
  const setSupercell = useDiffractionStore((state) => state.setSupercell);
  const uiConfig = useDiffractionStore((state) => state.uiConfig);
  const updateUiConfig = useDiffractionStore((state) => state.updateUiConfig);
  const elementColors = useDiffractionStore((state) => state.elementColors);
  const setElementColor = useDiffractionStore((state) => state.setElementColor);

  const [backgroundPickerOpen, setBackgroundPickerOpen] = useState(false);
  const uniqueElements = useMemo(() => {
    if (!structure) return [];
    const set = new Set(structure.atom_sites.map((site) => site.element));
    return Array.from(set).sort();
  }, [structure]);

  if (!uiConfig) {
    return null;
  }

  const handleSupercellChange = (index: number) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const next = [...supercell] as [number, number, number];
    const raw = Number(event.target.value);
    const clamped = Math.max(1, Math.min(3, Number.isNaN(raw) ? 1 : Math.round(raw)));
    next[index] = clamped;
    setSupercell(next);
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
        <h3 style={{ margin: 0 }}>Viewer settings</h3>
        <p style={{ margin: '4px 0 0', color: 'rgba(148, 163, 184, 0.85)', fontSize: '14px' }}>
          Tailor the 3D scene to match your preferred styling.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px' }}>
        {(['atom_scale', 'bond_thickness'] as const).map((key) => (
          <label key={key} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {key === 'atom_scale' ? 'Atom scale' : 'Bond thickness'}
            <input
              style={inputStyle}
              type="number"
              step="0.05"
              min={0.1}
              max={5}
              value={uiConfig[key]}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (!Number.isNaN(value)) {
                  updateUiConfig({ [key]: value });
                }
              }}
            />
          </label>
        ))}
        <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          Supercell (a×b×c)
          <div style={{ display: 'flex', gap: '6px' }}>
            {supercell.map((value, index) => (
              <input
                key={index}
                style={{ ...inputStyle, width: '100%' }}
                type="number"
                min={1}
                max={3}
                value={value}
                onChange={handleSupercellChange(index)}
              />
            ))}
          </div>
        </label>
      </div>

      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
        <div style={{ flex: '0 0 auto' }}>
          <button
            type="button"
            onClick={() => setBackgroundPickerOpen((open) => !open)}
            style={{
              background: uiConfig.background_color,
              border: '1px solid rgba(148, 163, 184, 0.25)',
              borderRadius: '999px',
              padding: '10px 16px',
              color: '#0f172a',
              cursor: 'pointer'
            }}
            data-testid="viewer-background-toggle"
          >
            Background
          </button>
        </div>
        <input
          type="text"
          aria-label="Background color hex"
          data-testid="viewer-background-hex"
          value={uiConfig.background_color}
          onChange={(event) => updateUiConfig({ background_color: event.target.value })}
          style={{
            ...inputStyle,
            width: '140px'
          }}
        />
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="checkbox"
            checked={uiConfig.show_bonds}
            onChange={(event) => updateUiConfig({ show_bonds: event.target.checked })}
          />
          Show bonds
        </label>
      </div>

      {backgroundPickerOpen ? (
        <HexColorPicker
          color={uiConfig.background_color}
          onChange={(value) => updateUiConfig({ background_color: value })}
        />
      ) : null}

      <div style={{ display: 'grid', gap: '12px' }}>
        <h4 style={{ margin: '8px 0 0' }}>Element palette</h4>
        {uniqueElements.map((element) => (
          <div
            key={element}
            data-testid={`element-palette-${element.toLowerCase()}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
              padding: '10px 12px',
              background: 'rgba(15, 23, 42, 0.6)',
              borderRadius: '12px'
            }}
          >
            <span style={{ fontWeight: 600 }}>{element}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '28px',
                height: '28px',
                borderRadius: '50%',
                border: '2px solid rgba(148, 163, 184, 0.25)',
                background: elementColors[element] ?? '#94a3b8'
              }}
            />
            <input
              type="text"
              aria-label={`${element} color hex`}
              data-testid={`element-color-input-${element.toLowerCase()}`}
              value={elementColors[element] ?? '#94a3b8'}
              onChange={(event) => setElementColor(element, event.target.value)}
              style={{ ...inputStyle, width: '120px' }}
            />
            <HexColorPicker
              color={elementColors[element] ?? '#94a3b8'}
              onChange={(value) => setElementColor(element, value)}
            />
          </div>
          </div>
        ))}
      </div>
    </section>
  );
}
