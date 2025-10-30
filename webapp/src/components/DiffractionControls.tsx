import { useState } from 'react';

import { useDiffractionStore } from '../hooks/useDiffractionStore';
import { computePowderPattern, computeTemPattern, generateDiffraction } from '../services/api';

const sectionStyle: React.CSSProperties = {
  display: 'grid',
  gap: '16px',
  padding: '24px',
  borderRadius: '16px',
  background: 'rgba(15, 23, 42, 0.65)',
  border: '1px solid rgba(148, 163, 184, 0.15)'
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 10px',
  borderRadius: '10px',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  background: 'rgba(17, 25, 40, 0.6)',
  color: '#f8fafc'
};

const buttonStyle: React.CSSProperties = {
  padding: '10px 16px',
  borderRadius: '12px',
  border: 'none',
  cursor: 'pointer',
  fontWeight: 600,
  color: '#fff',
  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)'
};

export function DiffractionControls() {
  const structure = useDiffractionStore((state) => state.structure);
  const summary = useDiffractionStore((state) => state.summary);
  const setSummary = useDiffractionStore((state) => state.setSummary);
  const xrdSettings = useDiffractionStore((state) => state.xrdSettings);
  const setXrdSettings = useDiffractionStore((state) => state.setXrdSettings);
  const temSettings = useDiffractionStore((state) => state.temSettings);
  const setTemSettings = useDiffractionStore((state) => state.setTemSettings);
  const setXrdPattern = useDiffractionStore((state) => state.setXrdPattern);
  const setTemPattern = useDiffractionStore((state) => state.setTemPattern);
  const setStructure = useDiffractionStore((state) => state.setStructure);

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState({ all: false, xrd: false, tem: false });

  if (!structure) {
    return null;
  }

  const handleGenerateAll = async () => {
    if (!structure) return;
    setLoading((prev) => ({ ...prev, all: true }));
    setError(null);
    try {
      const response = await generateDiffraction(structure, xrdSettings, temSettings);
      setStructure(response.structure, response.summary);
      setSummary(response.summary);
      setXrdPattern(response.xrd);
      setTemPattern(response.tem);
    } catch (err) {
      console.error(err);
      setError('Failed to generate diffraction outputs.');
    } finally {
      setLoading((prev) => ({ ...prev, all: false }));
    }
  };

  const handleXrd = async () => {
    if (!structure) return;
    setLoading((prev) => ({ ...prev, xrd: true }));
    setError(null);
    try {
      const pattern = await computePowderPattern(structure, xrdSettings);
      setXrdPattern(pattern);
    } catch (err) {
      console.error(err);
      setError('XRD calculation failed.');
    } finally {
      setLoading((prev) => ({ ...prev, xrd: false }));
    }
  };

  const handleTem = async () => {
    if (!structure) return;
    setLoading((prev) => ({ ...prev, tem: true }));
    setError(null);
    try {
      const pattern = await computeTemPattern(structure, temSettings);
      setTemPattern(pattern);
    } catch (err) {
      console.error(err);
      setError('TEM calculation failed.');
    } finally {
      setLoading((prev) => ({ ...prev, tem: false }));
    }
  };

  const updateXrdSetting = (key: keyof typeof xrdSettings) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setXrdSettings({ [key]: Number(event.target.value) });
  };

  const updateTemSetting = (key: keyof typeof temSettings) => (event: React.ChangeEvent<HTMLInputElement>) => {
    if (key === 'zone_axis') return;
    setTemSettings({ [key]: Number(event.target.value) });
  };

  const updateZoneAxis = (index: number) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const values = [...temSettings.zone_axis] as [number, number, number];
    values[index] = Number(event.target.value) || 0;
    setTemSettings({ zone_axis: values });
  };

  return (
    <section style={sectionStyle}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0 }}>Diffraction settings</h2>
          <p style={{ margin: '4px 0 0', color: 'rgba(148, 163, 184, 0.85)', fontSize: '14px' }}>
            Configure computation parameters and regenerate results instantly.
          </p>
        </div>
        <button type="button" style={buttonStyle} onClick={handleGenerateAll} disabled={loading.all}>
          {loading.all ? 'Generating…' : 'Generate all'}
        </button>
      </header>

      {summary ? (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '12px',
            padding: '16px',
            background: 'rgba(15, 23, 42, 0.6)',
            borderRadius: '12px'
          }}
        >
          <div>
            <div style={{ color: 'rgba(148, 163, 184, 0.7)', fontSize: '13px' }}>Formula</div>
            <div style={{ fontWeight: 600 }}>{summary.formula}</div>
          </div>
          <div>
            <div style={{ color: 'rgba(148, 163, 184, 0.7)', fontSize: '13px' }}>Density</div>
            <div style={{ fontWeight: 600 }}>{summary.density.toFixed(3)} g/cm³</div>
          </div>
          <div>
            <div style={{ color: 'rgba(148, 163, 184, 0.7)', fontSize: '13px' }}>Volume</div>
            <div style={{ fontWeight: 600 }}>{summary.volume.toFixed(3)} Å³</div>
          </div>
          <div>
            <div style={{ color: 'rgba(148, 163, 184, 0.7)', fontSize: '13px' }}>Space group</div>
            <div style={{ fontWeight: 600 }}>{summary.space_group ?? 'Unknown'}</div>
          </div>
        </div>
      ) : null}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
        <div style={{ display: 'grid', gap: '10px' }}>
          <h3 style={{ margin: 0 }}>Powder XRD</h3>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            Wavelength (Å)
            <input type="number" step="0.0001" style={inputStyle} value={xrdSettings.wavelength} onChange={updateXrdSetting('wavelength')} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            2θ min (°)
            <input type="number" style={inputStyle} value={xrdSettings.two_theta_min} onChange={updateXrdSetting('two_theta_min')} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            2θ max (°)
            <input type="number" style={inputStyle} value={xrdSettings.two_theta_max} onChange={updateXrdSetting('two_theta_max')} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            Min intensity
            <input type="number" step="0.1" style={inputStyle} value={xrdSettings.min_intensity} onChange={updateXrdSetting('min_intensity')} />
          </label>
          <button type="button" style={buttonStyle} onClick={handleXrd} disabled={loading.xrd}>
            {loading.xrd ? 'Calculating…' : 'Regenerate XRD'}
          </button>
        </div>

        <div style={{ display: 'grid', gap: '10px' }}>
          <h3 style={{ margin: 0 }}>TEM diffraction</h3>
          <div>
            <div style={{ marginBottom: '6px' }}>Zone axis</div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {temSettings.zone_axis.map((component, index) => (
                <input
                  key={index}
                  type="number"
                  style={{ ...inputStyle, width: '100%' }}
                  value={component}
                  onChange={updateZoneAxis(index)}
                />
              ))}
            </div>
          </div>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            Voltage (kV)
            <input type="number" style={inputStyle} value={temSettings.voltage} onChange={updateTemSetting('voltage')} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            Camera length (mm)
            <input type="number" style={inputStyle} value={temSettings.camera_length} onChange={updateTemSetting('camera_length')} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            Intensity threshold
            <input
              type="number"
              step="0.0001"
              style={inputStyle}
              value={temSettings.intensity_threshold}
              onChange={updateTemSetting('intensity_threshold')}
            />
          </label>
          <button type="button" style={buttonStyle} onClick={handleTem} disabled={loading.tem}>
            {loading.tem ? 'Calculating…' : 'Regenerate TEM'}
          </button>
        </div>
      </div>

      {error ? <div style={{ color: '#f97316', fontWeight: 600 }}>{error}</div> : null}
    </section>
  );
}
