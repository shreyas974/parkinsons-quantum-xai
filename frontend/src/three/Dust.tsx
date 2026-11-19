import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

/** Faint particles along the camera's path, so depth reads while flying between scenes. */
export function Dust({ count, reduced }: { count: number; reduced: boolean }) {
  const geometry = useMemo(() => {
    let s = 17;
    const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (rnd() - 0.5) * 26;
      arr[i * 3 + 1] = (rnd() - 0.5) * 14;
      arr[i * 3 + 2] = 12 - rnd() * 100;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [count]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const ref = useRef<THREE.Points>(null);
  useFrame((_, dt) => {
    if (!reduced && ref.current) ref.current.rotation.z += dt * 0.004;
  });
  return (
    <points ref={ref} geometry={geometry}>
      <pointsMaterial color="#b8c4e6" size={0.04} sizeAttenuation transparent opacity={0.55} depthWrite={false} />
    </points>
  );
}
