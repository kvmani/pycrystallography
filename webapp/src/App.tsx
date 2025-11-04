import { useEffect, useState } from 'react';

import { CifUpload } from './components/CifUpload';
import { DiffractionControls } from './components/DiffractionControls';
import { StructureForm } from './components/StructureForm';
import { TemPattern } from './components/TemPattern';
import { UnitCellViewer } from './components/UnitCellViewer';
import { ViewerControls } from './components/ViewerControls';
import { PlaneOverlayControls } from './components/PlaneOverlayControls';
import { XrdChart } from './components/XrdChart';
import { useDiffractionStore } from './hooks/useDiffractionStore';
import { fetchUiConfig } from './services/api';

import './App.css';

export default function App() {
  const setUiConfig = useDiffractionStore((state) => state.setUiConfig);
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const config = await fetchUiConfig();
        setUiConfig(config);
      } catch (err) {
        console.error(err);
        setConfigError('Failed to load viewer configuration. Using defaults.');
      }
    })();
  }, [setUiConfig]);

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>PyCrystallography Diffraction Workbench</h1>
          <p>
            Explore single-phase diffraction interactively. Import CIF files or craft structures manually,
            then generate high-fidelity unit cell, XRD, and TEM visualisations powered by pymatgen and orix.
          </p>
        </div>
        <div className="hero-actions">
          <CifUpload />
        </div>
      </header>

      {configError ? <div className="warning">{configError}</div> : null}

      <main className="content-grid">
        <section className="left-panel">
          <StructureForm />
          <ViewerControls />
          <PlaneOverlayControls />
        </section>
        <section className="right-panel">
          <UnitCellViewer />
        </section>
      </main>

      <section className="diffraction-section">
        <DiffractionControls />
      </section>

      <section className="visualisation-grid">
        <XrdChart />
        <TemPattern />
      </section>
    </div>
  );
}
