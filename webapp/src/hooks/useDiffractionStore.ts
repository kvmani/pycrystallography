import { create } from 'zustand';

import type {
  AtomSite,
  LatticeParameters,
  PlaneOverlay,
  PowderPatternResponse,
  StructureModel,
  StructureSummary,
  TemPatternResponse,
  TemSettings,
  UiConfig,
  XrdSettings
} from '../types/structure';

type DiffractionState = {
  structure: StructureModel | null;
  summary: StructureSummary | null;
  xrdSettings: XrdSettings;
  temSettings: TemSettings;
  xrdPattern: PowderPatternResponse | null;
  temPattern: TemPatternResponse | null;
  uiConfig: UiConfig | null;
  supercell: [number, number, number];
  elementColors: Record<string, string>;
  planeOverlays: PlaneOverlay[];
  setStructure: (structure: StructureModel, summary?: StructureSummary | null) => void;
  updateLattice: (updates: Partial<LatticeParameters>) => void;
  updateAtomSite: (index: number, updates: Partial<AtomSite>) => void;
  addAtomSite: (site?: Partial<AtomSite>) => void;
  removeAtomSite: (index: number) => void;
  setXrdSettings: (settings: Partial<XrdSettings>) => void;
  setTemSettings: (settings: Partial<TemSettings>) => void;
  setXrdPattern: (pattern: PowderPatternResponse | null) => void;
  setTemPattern: (pattern: TemPatternResponse | null) => void;
  setSummary: (summary: StructureSummary | null) => void;
  setUiConfig: (config: UiConfig) => void;
  updateUiConfig: (updates: Partial<UiConfig>) => void;
  setSupercell: (supercell: [number, number, number]) => void;
  setElementColor: (element: string, color: string) => void;
  setStructureName: (name: string | null) => void;
  setSpaceGroup: (spaceGroup: string | null) => void;
  addPlaneOverlay: (overlay: Omit<PlaneOverlay, 'id'> & { id?: string }) => void;
  removePlaneOverlay: (id: string) => void;
  updatePlaneOverlay: (id: string, updates: Partial<PlaneOverlay>) => void;
  setPlaneVisibility: (id: string, visible: boolean) => void;
  clearPlaneOverlays: () => void;
};

function clampSupercell([a, b, c]: [number, number, number]): [number, number, number] {
  const clamp = (value: number) => Math.max(1, Math.min(3, Math.round(value)));
  return [clamp(a), clamp(b), clamp(c)];
}

const defaultXrd: XrdSettings = {
  wavelength: 1.5406,
  two_theta_min: 10,
  two_theta_max: 90,
  min_intensity: 0
};

const defaultTem: TemSettings = {
  zone_axis: [0, 0, 1],
  voltage: 200,
  camera_length: 160,
  intensity_threshold: 1e-3
};

const defaultSupercell: [number, number, number] = [1, 1, 1];

const defaultPlaneOverlay: PlaneOverlay = {
  id: 'default-plane-100-001',
  hkl: [1, 0, 0],
  uvw: [0, 0, 1],
  offset: 0,
  color: '#38bdf8',
  arrowColor: '#f97316',
  opacity: 0.45,
  arrowLength: 2.5,
  visible: true,
  label: '(100)[001]'
};

function createPlaneId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `plane-${Math.random().toString(36).slice(2, 9)}`;
}

const defaultStructure: StructureModel = {
  name: 'Fe (bcc)',
  space_group: 'Im-3m',
  lattice: {
    a: 2.8665,
    b: 2.8665,
    c: 2.8665,
    alpha: 90,
    beta: 90,
    gamma: 90
  },
  atom_sites: [
    { element: 'Fe', x: 0, y: 0, z: 0, occupancy: 1, label: 'Fe1' },
    { element: 'Fe', x: 0.5, y: 0.5, z: 0.5, occupancy: 1, label: 'Fe2' }
  ]
};

export const useDiffractionStore = create<DiffractionState>((set) => ({
  structure: defaultStructure,
  summary: null,
  xrdSettings: defaultXrd,
  temSettings: defaultTem,
  xrdPattern: null,
  temPattern: null,
  uiConfig: null,
  supercell: clampSupercell(defaultSupercell),
  elementColors: {},
  planeOverlays: [defaultPlaneOverlay],
  setStructure: (structure, summary = null) =>
    set((state) => ({
      structure,
      summary,
      xrdPattern: null,
      temPattern: null,
      elementColors: state.elementColors
    })),
  updateLattice: (updates) =>
    set((state) =>
      state.structure
        ? {
            structure: {
              ...state.structure,
              lattice: { ...state.structure.lattice, ...updates }
            }
          }
        : {}
    ),
  updateAtomSite: (index, updates) =>
    set((state) => {
      if (!state.structure) return {};
      const atom_sites = state.structure.atom_sites.map((site, idx) =>
        idx === index ? { ...site, ...updates } : site
      );
      return { structure: { ...state.structure, atom_sites } };
    }),
  addAtomSite: (site) =>
    set((state) => {
      if (!state.structure) return {};
      const defaults: AtomSite = {
        element: 'Fe',
        x: 0,
        y: 0,
        z: 0,
        occupancy: 1,
        label: null
      };
      const atom_sites = [...state.structure.atom_sites, { ...defaults, ...site }];
      return { structure: { ...state.structure, atom_sites } };
    }),
  removeAtomSite: (index) =>
    set((state) => {
      if (!state.structure) return {};
      const atom_sites = state.structure.atom_sites.filter((_, idx) => idx !== index);
      return { structure: { ...state.structure, atom_sites } };
    }),
  setXrdSettings: (settings) =>
    set((state) => ({ xrdSettings: { ...state.xrdSettings, ...settings } })),
  setTemSettings: (settings) =>
    set((state) => ({ temSettings: { ...state.temSettings, ...settings } })),
  setXrdPattern: (pattern) => set({ xrdPattern: pattern }),
  setTemPattern: (pattern) => set({ temPattern: pattern }),
  setSummary: (summary) => set({ summary }),
  setUiConfig: (config) =>
    set(() => ({
      uiConfig: config,
      supercell: clampSupercell(config.default_supercell),
      elementColors: Object.fromEntries(config.element_colors.map((item) => [item.element, item.color]))
    })),
  updateUiConfig: (updates) =>
    set((state) =>
      state.uiConfig
        ? {
            uiConfig: { ...state.uiConfig, ...updates }
          }
        : {}
    ),
  setSupercell: (supercell) => set({ supercell: clampSupercell(supercell) }),
  setElementColor: (element, color) =>
    set((state) => {
      const nextColors = { ...state.elementColors, [element]: color };
      if (state.uiConfig) {
        return {
          elementColors: nextColors,
          uiConfig: {
            ...state.uiConfig,
            element_colors: Object.entries(nextColors).map(([el, value]) => ({ element: el, color: value }))
          }
        };
      }
      return { elementColors: nextColors };
    }),
  setStructureName: (name) =>
    set((state) =>
      state.structure
        ? {
            structure: {
              ...state.structure,
              name
            }
          }
        : {}
    ),
  setSpaceGroup: (spaceGroup) =>
    set((state) =>
      state.structure
        ? {
            structure: {
              ...state.structure,
              space_group: spaceGroup ?? undefined
            }
          }
        : {}
    ),
  addPlaneOverlay: (overlay) =>
    set((state) => {
      const id = overlay.id ?? createPlaneId();
      const next: PlaneOverlay = {
        ...defaultPlaneOverlay,
        ...overlay,
        id
      };
      return { planeOverlays: [...state.planeOverlays, next] };
    }),
  removePlaneOverlay: (id) =>
    set((state) => ({ planeOverlays: state.planeOverlays.filter((item) => item.id !== id) })),
  updatePlaneOverlay: (id, updates) =>
    set((state) => ({
      planeOverlays: state.planeOverlays.map((item) => (item.id === id ? { ...item, ...updates } : item))
    })),
  setPlaneVisibility: (id, visible) =>
    set((state) => ({
      planeOverlays: state.planeOverlays.map((item) => (item.id === id ? { ...item, visible } : item))
    })),
  clearPlaneOverlays: () => set({ planeOverlays: [] })
}));

export const selectStructure = () => useDiffractionStore.getState().structure;
