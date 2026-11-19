import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

interface Props {
  /** 0 = steady voice, 1 = strong jitter and shimmer. */
  jitter: number;
  color: string;
  width?: number;
  reduced: boolean;
}

const COLS = 260;

/**
 * A sustained "aaah" drawn as a ribbon. Jitter shifts the wave's timing cycle to cycle,
 * shimmer changes its height: the two measures the voice features are built on.
 */
export function VoiceRibbon({ jitter, color, width = 3.8, reduced }: Props) {
  const geometry = useMemo(() => new THREE.PlaneGeometry(width, 0.06, COLS, 1), [width]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const baseX = useMemo(() => {
    const pos = geometry.attributes.position as THREE.BufferAttribute;
    return Array.from({ length: COLS + 1 }, (_, c) => pos.getX(c));
  }, [geometry]);

  const shape = (t: number) => {
    const pos = geometry.attributes.position as THREE.BufferAttribute;
    for (let c = 0; c <= COLS; c++) {
      const x = baseX[c];
      const shimmer = 1 + jitter * 0.5 * Math.sin(x * 13.7 + t * 5.3) * Math.sin(x * 3.1 - t * 1.9);
      const timing = jitter * 0.85 * Math.sin(x * 4.9 + t * 2.4);
      const y = 0.42 * shimmer * Math.sin(x * 5.4 - t * 2.2 + timing);
      const z = 0.12 * Math.cos(x * 1.7 + t * 0.8);
      pos.setXYZ(c, x, y + 0.03, z);
      pos.setXYZ(c + COLS + 1, x, y - 0.03, z);
    }
    pos.needsUpdate = true;
  };

  useFrame(({ clock }) => shape(reduced ? 0 : clock.elapsedTime));

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} side={THREE.DoubleSide} roughness={0.5} />
    </mesh>
  );
}
