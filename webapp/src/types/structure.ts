export interface LatticeParameters {
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
}

export interface AtomSite {
  element: string;
  x: number;
  y: number;
  z: number;
  occupancy: number;
  label?: string | null;
}

export interface StructureModel {
  name?: string | null;
  lattice: LatticeParameters;
  space_group?: string | null;
  atom_sites: AtomSite[];
}

export interface StructureSummary {
  formula: string;
  density: number;
  volume: number;
  lattice_vectors: number[][];
  space_group?: string | null;
  warnings: string[];
}

export interface PowderPeak {
  two_theta: number;
  intensity: number;
  d_spacing: number;
  hkl: number[];
}

export interface PowderPatternResponse {
  peaks: PowderPeak[];
}

export interface TemReflection {
  g: number;
  intensity: number;
  hkl: number[];
  position: [number, number];
}

export interface TemPatternResponse {
  reflections: TemReflection[];
}

export interface XrdSettings {
  wavelength: number;
  two_theta_min: number;
  two_theta_max: number;
  min_intensity: number;
}

export interface TemSettings {
  zone_axis: [number, number, number];
  voltage: number;
  camera_length: number;
  intensity_threshold: number;
}

export interface GenerateResponse {
  structure: StructureModel;
  summary: StructureSummary;
  xrd: PowderPatternResponse;
  tem: TemPatternResponse;
}

export interface StructureResponse {
  structure: StructureModel;
  summary: StructureSummary;
}

export interface ElementColor {
  element: string;
  color: string;
}

export interface UiConfig {
  background_color: string;
  atom_scale: number;
  bond_thickness: number;
  show_bonds: boolean;
  default_supercell: [number, number, number];
  element_colors: ElementColor[];
}

export interface PlaneOverlay {
  id: string;
  hkl: [number, number, number];
  uvw: [number, number, number];
  offset: number;
  color: string;
  arrowColor: string;
  opacity: number;
  arrowLength: number;
  visible: boolean;
  label?: string | null;
}
