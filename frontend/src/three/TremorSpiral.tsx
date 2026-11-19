import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { spiralPoints } from "../lib/spiral";

interface Props {
  tremor: number;
  color: string;
  radius?: number;
  turns?: number;
  tube?: number;
  /** Seconds to draw the line; 0 shows it complete. */
  drawSeconds?: number;
  seed?: number;
  reduced: boolean;
}

const RADIAL = 8;

/** A spiral drawn as a glowing tube, optionally "being drawn" by a pen tip. */
export function TremorSpiral({ tremor, color, radius = 3, turns = 5, tube = 0.05, drawSeconds = 0, seed = 1, reduced }: Props) {
  const segments = Math.round(260 * turns);
  const { geometry, curve } = useMemo(() => {
    const pts = spiralPoints({ turns, radius, tremor, steps: 160 * turns, seed }).map(
      ([x, y], i, all) => new THREE.Vector3(x, y, (i / all.length - 0.5) * radius * 0.18),
    );
    const c = new THREE.CatmullRomCurve3(pts);
    return { curve: c, geometry: new THREE.TubeGeometry(c, segments, tube, RADIAL, false) };
  }, [turns, radius, tremor, seed, tube, segments]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  const tip = useRef<THREE.Mesh>(null);
  const started = useRef<number | null>(null);

  useFrame(({ clock }) => {
    if (started.current === null) started.current = clock.elapsedTime;
    const raw = reduced || drawSeconds === 0 ? 1 : Math.min(1, (clock.elapsedTime - started.current) / drawSeconds);
    const p = 1 - Math.pow(1 - raw, 3);
    geometry.setDrawRange(0, Math.max(1, Math.floor(p * segments)) * RADIAL * 6);
    if (tip.current) {
      tip.current.visible = drawSeconds > 0 && raw < 1;
      tip.current.position.copy(curve.getPointAt(Math.min(0.9999, Math.max(0.0001, p))));
    }
  });

  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.75} roughness={0.35} metalness={0.2} />
      </mesh>
      <mesh ref={tip}>
        <sphereGeometry args={[tube * 2.4, 16, 16]} />
        <meshBasicMaterial color="#fff4dc" />
      </mesh>
    </group>
  );
}
