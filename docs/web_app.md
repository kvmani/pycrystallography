# Web diffraction workbench

The web workbench pairs a FastAPI backend (for CIF ingestion and diffraction
calculations) with a React + three.js front-end (for interactive rendering).

## Launching the stack

1. Create a Python environment with the project installed (`pip install -e .`).
2. Start the backend:

   ```bash
   pcg web run --host 0.0.0.0 --port 8000
   ```

   Additional options:

   * `--ui-config PATH` – load alternative defaults for colours, radii, etc.
   * `--allowed-origin https://your.domain` – set specific CORS origins (repeat
     flag for multiples).

3. In another terminal start the UI:

   ```bash
   cd webapp
   npm install
   npm run dev -- --host
   ```

   Set `VITE_API_BASE_URL` if the backend is not on `http://localhost:8000`.

## Features

* **CIF upload + manual editing** – drop a CIF or use the pre-populated Fe BCC
  template. All lattice parameters, atom sites, and occupancies remain editable.
* **3D unit cell viewer** – rendered with `@react-three/fiber`. Users can toggle
  bonds, adjust atom radii/colours, pick background colours, and replicate the
  cell into supercells (e.g. `3 × 3 × 3`).
* **Powder XRD chart** – powered by pymatgen’s `XRDCalculator` backend and
  Plotly for presentation. Hkl annotations appear on hover.
* **TEM diffraction plot** – driven by pymatgen’s `TEMCalculator`. Users provide
  the zone axis (e.g. Fe `[001]`, `[110]`, Zr `[0001]`, `[11-20]`) and receive a
  recalculated scatter plot with intensity-scaled markers.
* **Configurable defaults** – stored in `configs/diffraction_web_ui.yaml`.

## Example workflow

1. Start the backend and UI using the steps above.
2. Drag `data/structureData/Fe.cif` into the upload panel. Review the populated
   sites and adjust the supercell to `3 × 3 × 3` for clarity.
3. Set the zone axis to `[0, 0, 1]` and click **Generate all** to compute the
   XRD/TEM outputs. Use **Regenerate TEM** with `[1, 1, 0]` for a second
   projection.
4. Repeat for `data/structureData/Zr-Alpha.cif`, exploring `[0, 0, 0, 1]` and
   `[11, -2, 0]` zone axes.
5. Capture screenshots (3D viewer, XRD, TEM panels) for reporting.

The app is designed to be modular so future releases can extend the pipeline to
multi-phase diffraction or orientation relation management without rewriting
front-end code.
