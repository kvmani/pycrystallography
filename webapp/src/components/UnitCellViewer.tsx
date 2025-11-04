import { Suspense, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Html, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { useDiffractionStore } from '../hooks/useDiffractionStore';
import type { LatticeParameters, PlaneOverlay, StructureModel } from '../types/structure';

const defaultColor = '#94a3b8';
const boundaryTolerance = 1e-5;
const planeHighlightTolerance = 0.08;

const covalentRadii: Record<string, number> = {
  H: 0.31,
  C: 0.76,
  N: 0.71,
  O: 0.66,
  Fe: 1.26,
  Zr: 1.6
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
  matrix.set(vA.x, vB.x, vC.x, vA.y, vB.y, vC.y, vA.z, vB.z, vC.z);
  return matrix;
}

interface AtomInstance {
  position: THREE.Vector3;
  fractional: THREE.Vector3;
  element: string;
}

function expandCoordinate(value: number): number[] {
  const coords = new Set<number>();
  const nearZero = Math.abs(value) < boundaryTolerance;
  const nearOne = Math.abs(value - 1) < boundaryTolerance || 1 - value < boundaryTolerance;

  if (nearZero) {
    coords.add(0);
    coords.add(1);
  } else if (nearOne) {
    coords.add(1);
    coords.add(0);
  } else {
    coords.add(value);
  }

  coords.add(value);

  return Array.from(coords);
}

function generateBoundaryImages(site: StructureModel['atom_sites'][number]): [number, number, number][] {
  const xImages = expandCoordinate(site.x);
  const yImages = expandCoordinate(site.y);
  const zImages = expandCoordinate(site.z);
  const combinations: [number, number, number][] = [];
  const seen = new Set<string>();

  xImages.forEach((x) => {
    yImages.forEach((y) => {
      zImages.forEach((z) => {
        const key = `${x.toFixed(6)}|${y.toFixed(6)}|${z.toFixed(6)}`;
        if (!seen.has(key)) {
          combinations.push([x, y, z]);
          seen.add(key);
        }
      });
    });
  });

  return combinations;
}

function generateSupercell(structure: StructureModel, supercell: [number, number, number]): AtomInstance[] {
  const matrix = latticeToMatrix(structure.lattice);
  const atoms: AtomInstance[] = [];
  const [nx, ny, nz] = supercell;
  const seen = new Set<string>();
  structure.atom_sites.forEach((site) => {
    const boundaryImages = generateBoundaryImages(site);
    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) {
        for (let k = 0; k < nz; k += 1) {
          boundaryImages.forEach(([x, y, z]) => {
            const fractional = new THREE.Vector3(x + i, y + j, z + k);
            const key = `${site.element}:${fractional.x.toFixed(6)}:${fractional.y.toFixed(6)}:${fractional.z.toFixed(6)}`;
            if (seen.has(key)) {
              return;
            }
            seen.add(key);
            const cartesian = fractional.clone().applyMatrix3(matrix);
            atoms.push({ element: site.element, position: cartesian, fractional: fractional.clone() });
          });
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
  atomScale,
  highlights
}: {
  atoms: AtomInstance[];
  elementColors: Record<string, string>;
  atomScale: number;
  highlights: Map<number, string>;
}) {
  return (
    <group>
      {atoms.map((atom, index) => {
        const radius = atomScale * (covalentRadii[atom.element] ?? 1.0) * 0.4;
        const color = elementColors[atom.element] ?? defaultColor;
        const highlightColor = highlights.get(index) ?? null;
        return (
          <mesh key={`${atom.element}-${index}`} position={atom.position}>
            <sphereGeometry args={[radius, 42, 42]} />
            <meshStandardMaterial
              color={color}
              emissive={highlightColor ?? '#000000'}
              emissiveIntensity={highlightColor ? 0.65 : 0}
              roughness={0.35}
              metalness={0.15}
            />
          </mesh>
        );
      })}
    </group>
  );
}

interface PlaneData {
  center: THREE.Vector3;
  quaternion: THREE.Quaternion;
  arrowDirection: THREE.Vector3;
  labelPosition: THREE.Vector3;
  planeSize: number;
}

function formatOverlayLabel(overlay: PlaneOverlay) {
  const [h, k, l] = overlay.hkl;
  const [u, v, w] = overlay.uvw;
  return `(${h} ${k} ${l})[${u} ${v} ${w}]`;
}

function computePlaneData(overlay: PlaneOverlay, lattice: LatticeParameters, extent: number): PlaneData | null {
  const matrix = latticeToMatrix(lattice);
  const reciprocal = matrix.clone().invert().transpose();
  const normalFrac = new THREE.Vector3(...overlay.hkl);
  if (normalFrac.lengthSq() < 1e-8) {
    return null;
  }

  const normalCart = normalFrac.clone().applyMatrix3(reciprocal);
  if (normalCart.lengthSq() < 1e-8) {
    return null;
  }

  let directionFrac = new THREE.Vector3(...overlay.uvw);
  if (directionFrac.lengthSq() < 1e-8) {
    directionFrac = new THREE.Vector3(0, 1, 0);
  }

  const normalFracLengthSq = normalFrac.lengthSq();
  const projection = directionFrac
    .clone()
    .sub(normalFrac.clone().multiplyScalar(directionFrac.dot(normalFrac) / normalFracLengthSq));

  let inPlaneFrac = projection;
  if (inPlaneFrac.lengthSq() < 1e-6) {
    const candidates = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1)
    ];
    for (const candidate of candidates) {
      const projected = candidate
        .clone()
        .sub(normalFrac.clone().multiplyScalar(candidate.dot(normalFrac) / normalFracLengthSq));
      if (projected.lengthSq() > 1e-6) {
        inPlaneFrac = projected;
        break;
      }
    }
  }

  if (inPlaneFrac.lengthSq() < 1e-6) {
    return null;
  }

  const inPlaneCart = inPlaneFrac.clone().applyMatrix3(matrix);
  if (inPlaneCart.lengthSq() < 1e-6) {
    return null;
  }

  const normalCartNorm = normalCart.clone().normalize();
  const u = inPlaneCart.clone().normalize();
  const v = new THREE.Vector3().crossVectors(normalCartNorm, u).normalize();
  if (v.lengthSq() < 1e-6) {
    return null;
  }

  const planeSize = Math.max(extent * 1.3, 3.2);
  const offsetScalar = overlay.offset / normalFracLengthSq;
  const center = normalFrac.clone().multiplyScalar(offsetScalar).applyMatrix3(matrix);
  const basis = new THREE.Matrix4().makeBasis(u, v, normalCartNorm);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(basis);
  const labelPosition = center.clone().add(v.clone().multiplyScalar(planeSize * 0.55));

  return {
    center,
    quaternion,
    arrowDirection: u,
    labelPosition,
    planeSize
  };
}

function PlaneOverlayMesh({
  overlay,
  lattice,
  extent
}: {
  overlay: PlaneOverlay;
  lattice: LatticeParameters;
  extent: number;
}) {
  const data = useMemo(() => computePlaneData(overlay, lattice, extent), [overlay, lattice, extent]);

  const geometry = useMemo(() => {
    if (!data) {
      return null;
    }
    return new THREE.PlaneGeometry(data.planeSize, data.planeSize, 1, 1);
  }, [data]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!data || !geometry) {
    return null;
  }

  const arrowLength = Math.max(overlay.arrowLength, data.planeSize * 0.35);

  return (
    <group>
      <mesh position={data.center} quaternion={data.quaternion}>
        <primitive object={geometry} />
        <meshStandardMaterial
          color={overlay.color}
          transparent
          opacity={overlay.opacity}
          depthWrite={false}
          side={THREE.DoubleSide}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>
      <arrowHelper args={[data.arrowDirection, data.center, arrowLength, overlay.arrowColor]} />
      <Html
        position={data.labelPosition}
        style={{
          background: 'rgba(15, 23, 42, 0.85)',
          padding: '4px 8px',
          borderRadius: '8px',
          fontSize: '12px',
          color: '#e2e8f0',
          whiteSpace: 'nowrap',
          border: '1px solid rgba(148, 163, 184, 0.35)'
        }}
        center
      >
        {formatOverlayLabel(overlay)}
      </Html>
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
  const planeOverlays = useDiffractionStore((state) => state.planeOverlays);

  const atoms = useMemo(() => (structure ? generateSupercell(structure, supercell) : []), [structure, supercell]);
  const bonds = useMemo(() => {
    if (!structure || !uiConfig?.show_bonds) return [];
    return computeBonds(atoms, elementColors, uiConfig.bond_thickness);
  }, [atoms, structure, elementColors, uiConfig]);

  const cellEdges = useMemo(() => {
    if (!structure) return null;
    return buildCellEdges(structure.lattice, supercell);
  }, [structure, supercell]);

  const boundingExtent = useMemo(() => {
    if (atoms.length === 0) {
      return 4;
    }
    const box = new THREE.Box3();
    atoms.forEach((atom) => box.expandByPoint(atom.position));
    const size = box.getSize(new THREE.Vector3());
    const maxDimension = Math.max(size.x, size.y, size.z);
    return Math.max(maxDimension, 4);
  }, [atoms]);

  const highlightMap = useMemo(() => {
    const map = new Map<number, string>();
    const activeOverlays = planeOverlays.filter((overlay) => overlay.visible);
    if (activeOverlays.length === 0) {
      return map;
    }
    activeOverlays.forEach((overlay) => {
      const [h, k, l] = overlay.hkl;
      const offset = overlay.offset;
      atoms.forEach((atom, index) => {
        const value = h * atom.fractional.x + k * atom.fractional.y + l * atom.fractional.z;
        const diff = value - offset;
        const periodicDistance = Math.abs(diff - Math.round(diff));
        if (periodicDistance < planeHighlightTolerance) {
          map.set(index, overlay.color);
        }
      });
    });
    return map;
  }, [atoms, planeOverlays]);

  const planeMeshes = useMemo(() => {
    if (!structure) {
      return [] as JSX.Element[];
    }
    return planeOverlays
      .filter((overlay) => overlay.visible)
      .map((overlay) => (
        <PlaneOverlayMesh key={overlay.id} overlay={overlay} lattice={structure.lattice} extent={boundingExtent} />
      ));
  }, [planeOverlays, structure, boundingExtent]);

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
          <AtomSpheres atoms={atoms} elementColors={elementColors} atomScale={atomScale} highlights={highlightMap} />
          {uiConfig?.show_bonds ? bonds : null}
          {cellEdges ? <primitive object={cellEdges} /> : null}
          {planeMeshes}
        </Suspense>
        <OrbitControls enableDamping dampingFactor={0.15} minDistance={1} maxDistance={60} />
      </Canvas>
    </div>
  );
}
