import { Suspense, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { useDiffractionStore } from '../hooks/useDiffractionStore';
import type { LatticeParameters, StructureModel } from '../types/structure';

const defaultColor = '#94a3b8';

const covalentRadii: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  Fe: 1.26,
  Zr: 1.60
};

function toRadians(angle: number) {
  return (angle * Math.PI) / 180;
}

function latticeToMatrix(lattice: LatticeParameters): THREE.Matrix3 {
  const alpha = toRadians(lattice.alpha);
  const beta = toRadians(lattice.beta);
  const gamma = toRadians(lattice.gamma);

  const a = lattice.a;
  const b = lattice.b;
  const c = lattice.c;

  const vA = new THREE.Vector3(a, 0, 0);
  const vB = new THREE.Vector3(b * Math.cos(gamma), b * Math.sin(gamma), 0);

  const cx = c * Math.cos(beta);
  const cy = c * (Math.cos(alpha) - Math.cos(beta) * Math.cos(gamma)) / Math.sin(gamma);
  const cz = Math.sqrt(Math.max(c * c - cx * cx - cy * cy, 0));
  const vC = new THREE.Vector3(cx, cy, cz);

  const matrix = new THREE.Matrix3();
  matrix.set(
    vA.x, vB.x, vC.x,
    vA.y, vB.y, vC.y,
    vA.z, vB.z, vC.z
  );
  return matrix;
}

interface AtomInstance {
  position: THREE.Vector3;
  element: string;
}

function generateSupercell(structure: StructureModel, supercell: [number, number, number]): AtomInstance[] {
  const matrix = latticeToMatrix(structure.lattice);
  const atoms: AtomInstance[] = [];
  const [nx, ny, nz] = supercell;
  structure.atom_sites.forEach((site) => {
    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) {
        for (let k = 0; k < nz; k += 1) {
          const fractional = new THREE.Vector3(site.x + i, site.y + j, site.z + k);
          const cartesian = fractional.applyMatrix3(matrix);
          atoms.push({ element: site.element, position: cartesian });
        }
      }
    }
  });
  return atoms;
}

function buildCellEdges(lattice: LatticeParameters, supercell: [number, number, number]): THREE.LineSegments {
  const matrix = latticeToMatrix(lattice);
  const [nx, ny, nz] = supercell;
  const vectors = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(nx, 0, 0),
    new THREE.Vector3(0, ny, 0),
    new THREE.Vector3(0, 0, nz),
    new THREE.Vector3(nx, ny, 0),
    new THREE.Vector3(nx, 0, nz),
    new THREE.Vector3(0, ny, nz),
    new THREE.Vector3(nx, ny, nz)
  ].map((vec) => vec.applyMatrix3(matrix.clone()));

  const edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 4],
    [1, 5],
    [2, 4],
    [2, 6],
    [3, 5],
    [3, 6],
    [4, 7],
    [5, 7],
    [6, 7]
  ];

  const geometry = new THREE.BufferGeometry();
  const positions: number[] = [];
  edges.forEach(([start, end]) => {
    const a = vectors[start];
    const b = vectors[end];
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  });
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({ color: '#e2e8f0', linewidth: 1 });
  return new THREE.LineSegments(geometry, material);
}

function computeBonds(atoms: AtomInstance[], elementColors: Record<string, string>, bondThickness: number) {
  const cylinders: JSX.Element[] = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      const a = atoms[i];
      const b = atoms[j];
      const radiusA = covalentRadii[a.element] ?? 1.0;
      const radiusB = covalentRadii[b.element] ?? 1.0;
      const cutoff = 1.2 * (radiusA + radiusB);
      const distance = a.position.distanceTo(b.position);
      if (distance <= cutoff && distance > 1e-3) {
        const midpoint = new THREE.Vector3().addVectors(a.position, b.position).multiplyScalar(0.5);
        const direction = new THREE.Vector3().subVectors(b.position, a.position);
        const bondLength = direction.length();
        const orientation = new THREE.Matrix4();
        orientation.lookAt(new THREE.Vector3(0, 0, 0), direction.normalize(), new THREE.Vector3(0, 1, 0));
        orientation.multiply(new THREE.Matrix4().makeRotationX(Math.PI / 2));
        cylinders.push(
          <mesh key={`bond-${i}-${j}`} position={midpoint} matrixAutoUpdate={false}>
            <cylinderGeometry args={[bondThickness, bondThickness, bondLength, 16]} />
            <meshStandardMaterial color={elementColors[a.element] ?? defaultColor} />
            <primitive object={orientation} attach="matrix" />
          </mesh>
        );
      }
    }
  }
  return cylinders;
}

function AtomSpheres({
  atoms,
  elementColors,
  atomScale
}: {
  atoms: AtomInstance[];
  elementColors: Record<string, string>;
  atomScale: number;
}) {
  return (
    <group>
      {atoms.map((atom, index) => {
        const radius = atomScale * (covalentRadii[atom.element] ?? 1.0) * 0.4;
        const color = elementColors[atom.element] ?? defaultColor;
        return (
          <mesh key={`${atom.element}-${index}`} position={atom.position}>
            <sphereGeometry args={[radius, 42, 42]} />
            <meshStandardMaterial color={color} roughness={0.35} metalness={0.15} />
          </mesh>
        );
      })}
    </group>
  );
}

export function UnitCellViewer({
  height = 420
}: {
  height?: number;
}) {
  const structure = useDiffractionStore((state) => state.structure);
  const supercell = useDiffractionStore((state) => state.supercell);
  const elementColors = useDiffractionStore((state) => state.elementColors);
  const uiConfig = useDiffractionStore((state) => state.uiConfig);

  const atoms = useMemo(() => (structure ? generateSupercell(structure, supercell) : []), [structure, supercell]);
  const bonds = useMemo(() => {
    if (!structure || !uiConfig?.show_bonds) return [];
    return computeBonds(atoms, elementColors, uiConfig.bond_thickness);
  }, [atoms, structure, elementColors, uiConfig]);

  const cellEdges = useMemo(() => {
    if (!structure) return null;
    return buildCellEdges(structure.lattice, supercell);
  }, [structure, supercell]);

  if (!structure) {
    return null;
  }

  const background = uiConfig?.background_color ?? '#0f172a';
  const atomScale = uiConfig?.atom_scale ?? 0.8;

  return (
    <div
      data-testid="unit-cell-viewer"
      style={{ height, borderRadius: '16px', overflow: 'hidden', border: '1px solid rgba(148, 163, 184, 0.15)' }}
    >
      <Canvas style={{ background }} shadows>
        <PerspectiveCamera makeDefault position={[6, 6, 6]} fov={45} near={0.1} far={200} />
        <ambientLight intensity={0.8} />
        <pointLight position={[15, 15, 15]} intensity={1.2} />
        <directionalLight position={[-10, -10, 10]} intensity={0.4} />
        <Suspense fallback={null}>
          <AtomSpheres atoms={atoms} elementColors={elementColors} atomScale={atomScale} />
          {uiConfig?.show_bonds ? bonds : null}
          {cellEdges ? <primitive object={cellEdges} /> : null}
        </Suspense>
        <OrbitControls enableDamping dampingFactor={0.15} minDistance={1} maxDistance={60} />
      </Canvas>
    </div>
  );
}
