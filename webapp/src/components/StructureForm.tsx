import { ChangeEvent } from 'react';

import { useDiffractionStore } from '../hooks/useDiffractionStore';

const containerStyle: React.CSSProperties = {
  display: 'grid',
  gap: '16px',
  background: 'rgba(17, 25, 40, 0.6)',
  borderRadius: '16px',
  padding: '24px',
  border: '1px solid rgba(255, 255, 255, 0.05)'
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  borderRadius: '12px',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  background: 'rgba(15, 23, 42, 0.6)',
  color: '#f8fafc'
};

const labelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '6px',
  fontSize: '14px'
};

export function StructureForm() {
  const structure = useDiffractionStore((state) => state.structure);
  const updateLattice = useDiffractionStore((state) => state.updateLattice);
  const updateAtomSite = useDiffractionStore((state) => state.updateAtomSite);
  const addAtomSite = useDiffractionStore((state) => state.addAtomSite);
  const removeAtomSite = useDiffractionStore((state) => state.removeAtomSite);
  const setStructureName = useDiffractionStore((state) => state.setStructureName);
  const setSpaceGroup = useDiffractionStore((state) => state.setSpaceGroup);

  if (!structure) {
    return null;
  }

  const handleLatticeChange = (key: keyof typeof structure.lattice) =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      if (!Number.isNaN(value)) {
        updateLattice({ [key]: value });
      }
    };

  const handleAtomChange = (index: number, key: 'element' | 'x' | 'y' | 'z' | 'occupancy' | 'label') =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const targetValue = event.target.value;
      if (key === 'element' || key === 'label') {
        updateAtomSite(index, { [key]: targetValue });
      } else {
        const value = Number(targetValue);
        if (!Number.isNaN(value)) {
          updateAtomSite(index, { [key]: value });
        }
      }
    };

  return (
    <section style={containerStyle}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px' }}>Crystal Structure</h2>
          <p style={{ margin: '4px 0 0', color: 'rgba(148, 163, 184, 0.9)', fontSize: '14px' }}>
            Adjust lattice parameters, space group, and atom positions.
          </p>
        </div>
        <button
          type="button"
          onClick={() => addAtomSite({})}
          style={{
            padding: '8px 14px',
            borderRadius: '999px',
            border: 'none',
            background: 'linear-gradient(135deg, #6366f1, #a855f7)',
            color: '#fff',
            cursor: 'pointer'
          }}
        >
          Add atom
        </button>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px' }}>
        <label style={labelStyle}>
          Phase name
          <input
            style={inputStyle}
            value={structure.name ?? ''}
            onChange={(event) => setStructureName(event.target.value || null)}
            placeholder="Optional"
          />
        </label>
        <label style={labelStyle}>
          Space group
          <input
            style={inputStyle}
            value={structure.space_group ?? ''}
            onChange={(event) => setSpaceGroup(event.target.value || null)}
            placeholder="e.g. Im-3m"
          />
        </label>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
        {(['a', 'b', 'c', 'alpha', 'beta', 'gamma'] as const).map((key) => (
          <label key={key} style={labelStyle}>
            {key.toUpperCase()}
            <input
              style={inputStyle}
              type="number"
              step="any"
              value={structure.lattice[key]}
              onChange={handleLatticeChange(key)}
            />
          </label>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>Atom sites</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '600px' }}>
            <thead>
              <tr style={{ textAlign: 'left', color: 'rgba(148, 163, 184, 0.9)', fontSize: '13px' }}>
                <th style={{ padding: '8px' }}>Element</th>
                <th style={{ padding: '8px' }}>x</th>
                <th style={{ padding: '8px' }}>y</th>
                <th style={{ padding: '8px' }}>z</th>
                <th style={{ padding: '8px' }}>Occupancy</th>
                <th style={{ padding: '8px' }}>Label</th>
                <th style={{ padding: '8px' }}></th>
              </tr>
            </thead>
            <tbody>
              {structure.atom_sites.map((site, index) => (
                <tr key={`${site.element}-${index}`} style={{ borderTop: '1px solid rgba(148, 163, 184, 0.15)' }}>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      value={site.element}
                      onChange={handleAtomChange(index, 'element')}
                    />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      type="number"
                      step="any"
                      value={site.x}
                      onChange={handleAtomChange(index, 'x')}
                    />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      type="number"
                      step="any"
                      value={site.y}
                      onChange={handleAtomChange(index, 'y')}
                    />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      type="number"
                      step="any"
                      value={site.z}
                      onChange={handleAtomChange(index, 'z')}
                    />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      type="number"
                      step="any"
                      value={site.occupancy}
                      onChange={handleAtomChange(index, 'occupancy')}
                    />
                  </td>
                  <td style={{ padding: '8px' }}>
                    <input
                      style={inputStyle}
                      value={site.label ?? ''}
                      onChange={handleAtomChange(index, 'label')}
                    />
                  </td>
                  <td style={{ padding: '8px', textAlign: 'right' }}>
                    <button
                      type="button"
                      onClick={() => removeAtomSite(index)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'rgba(239, 68, 68, 0.9)',
                        cursor: 'pointer'
                      }}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
