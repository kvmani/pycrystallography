
This workflow coordinates specialized “agents” (workstreams) to rewrite **pycrystallography** by **delegating crystallographic math and diffraction to `orix` and `pymatgen`**. Our code provides orchestration, domain modeling, configuration, CLI, plotting, and exports.

## Universal Rules (Apply to every agent/PR)
- **Delegation**: Use `orix` for orientations/symmetry/variants and `pymatgen` for structures/CIF/XRD/electron diffraction. Do not re‑implement these.
- **Quality Gates (hard)**:
  - ✅ All tests pass (unit + property + CLI; cross‑platform CI).
  - ✅ `ruff` + `mypy` pass with no new warnings.
  - ✅ Docs updated wherever behavior or public API changes.
  - ✅ Changelog updated; migration notes if applicable.
- **Unit tests required** for every new public method/class.
- **Determinism**: Reproducible outputs (ordered, fixed seeds).
- **Immutability & SoC**: No global mutable state; clear boundaries (`adapters`, `core`, `analysis`, `cli`, `config`, `plotting`, `io`).

## Agent Roster & Deliverables

### 1) `foundation-agent`
**Goal**: Project scaffolding and CI.
- Create `pyproject.toml` (PEP 621) with `hatchling`, dependencies: `orix`, `pymatgen`, `numpy`, `pydantic`, `typer`, `matplotlib`, `pytest`, `hypothesis`, `ruff`, `mypy`, `mkdocs` (or sphinx), `pre-commit`.
- Add `pcg` console script.
- Pre‑commit hooks: ruff format+lint, mypy, trailing whitespace, end‑of‑file fixer.
- GitHub Actions CI: lint, type‑check, tests, docs build on ubuntu/macos/windows.
- Seed `docs/` with mkdocs (or sphinx+mkdocstrings) and theme.
- Create repo layout: `core/`, `adapters/`, `analysis/`, `cli/`, `config/`, `plotting/`, `io/`, `data/`, `tests/`.

### 2) `adapters-agent`
**Goal**: Thin, typed adapters for third‑party libs.
- `adapters/orix_adapter.py`: orientation <-> symmetry utilities, OR/variant enumeration, symmetry‑aware uniqueness, vectorised paths.
- `adapters/pymatgen_adapter.py`: CIF → structure/space group, XRD/TEM calculators wrapper, safe defaults, deterministic ordering.
- Tests with tiny curated CIFs; no network I/O. Document caveats and version pins.

### 3) `config-agent`
**Goal**: Strict configuration schema.
- Pydantic models for project/output/physics/phases/filters/plotting.
- Validation CLI: `pcg config validate --config .. --print` with helpful errors.
- Tests: precedence (YAML/JSON file < env < CLI), missing keys, type errors.

### 4) `core-agent`
**Goal**: Domain model with composition.
- `Phase`, `Orientation`, `OR`, `Variant`, `CompositePattern` as small classes.
- All heavy math calls routed to adapters.
- Property tests for symmetry‑aware uniqueness (e.g., variant dedup).

### 5) `calculators-agent`
**Goal**: Composite diffraction (XRD and TEM/SAED) parity.
- Implement `CompositeXRDCalculator` and `CompositeTEMCalculator` using adapters.
- Inputs: phases list, OR/variants, ranges (2θ, wavelength; TEM kV, camera length, zone axis), filters/windows.
- Outputs: peak tables (CSV/JSON), arrays (NPZ), plots (PNG/PDF). Deterministic ordering and file layout.
- Reproduce legacy `apps/compositeDiffractionPattern.py` example with curated CIFs.
- Image‑regression tests for plots; numerical tolerances centralized.

### 6) `cli-agent`
**Goal**: CLI ergonomics.
- Commands: `pcg composite xrd|tem`, `pcg or variants`, `pcg config validate`.
- Options: `--config`, `--out`, `--style`, `--normalize`, `--log-level`, `--dry-run`.
- Tests using `typer.testing` or subprocess; ensure deterministic output trees.

### 7) `plotting-agent`
**Goal**: Publication‑quality figures.
- Matplotlib utilities with clear defaults (size, dpi, legends, LaTeX labels optional).
- Image‑regression tests (pytest‑mpl or equivalent).

### 8) `data-agent`
**Goal**: Example data & registry.
- All required data files must stay in dat folder (it already has lot of cif files etc. use only them dont add any extras!)
- Curate tiny CIFs; `data/registry.yaml` mapping friendly names → CIF paths.
- Integrity tests; no external fetch during tests/CI.

### 9) `docs-agent`
**Goal**: Developer & user docs.
- Quickstart, configuration guide, CLI guide, API overview, design decisions, migration notes (legacy → new), troubleshooting.
- Auto‑API docs from docstrings via mkdocstrings (or sphinx + autosummary).

### 10) `bench-agent` (optional)
**Goal**: Baselines & performance hygiene.
- Simple `timeit` baselines for variant enumeration and diffraction table generation.
- Document vectorisation gains from `orix`.

## Collaboration & PR Process
1. Branch naming: `feat/*`, `fix/*`, `docs/*`, `refactor/*`.
2. Conventional commits preferred.
3. Open PR with checklist auto‑generated (tests/docs/ruff/mypy/ci passing).
4. Reviewers confirm **delegation** (no reinvention), **tests added**, **docs updated**.
5. Merge only when **all quality gates** are green.

### Commit policy
- Never include any binary cotent in the commit
- ensure all tests pass for every change.
-include screenshots, where aproprate in message summary chatbox, after you completed the task, for easy preview or review (dont include in the comit code, however) to jusdge the success of the task

---
**One‑line ethos**: *Leverage `orix` + `pymatgen` for the science; excel at orchestration, testing, and UX.*
