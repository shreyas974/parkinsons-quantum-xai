import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { scrollListeners, scrollState } from "../lib/store";

/** Camera stop for each anchor section: hero, signals, qubits, screening, results. */
const STOPS: { pos: [number, number, number]; look: [number, number, number] }[] = [
  { pos: [0, 0.35, 9], look: [0, 0, 0] },
  { pos: [0, 0.0, -15.6], look: [0, -1.0, -26] },
  { pos: [0, 0.45, -45.0], look: [0, -0.35, -56] },
  { pos: [0, 3.2, -61], look: [0, -1, -74] },
  { pos: [0, 4.2, -67], look: [0, -2, -82] },
];

/** Share of each segment the camera waits before moving on: the qubit stop holds through the pinned story. */
const HOLD = [0, 0, 0.72, 0];

const smooth = (t: number) => t * t * (3 - 2 * t);

/**
 * Flies the camera through the scenes as the page scrolls. With reduced motion it jumps
 * between stops instead of gliding, and ignores the pointer.
 */
export function CameraRig({ reduced }: { reduced: boolean }) {
  const camera = useThree((s) => s.camera);
  const invalidate = useThree((s) => s.invalidate);
  const look = useRef(new THREE.Vector3(...STOPS[0].look));
  const pos = new THREE.Vector3();
  const tgt = new THREE.Vector3();

  useEffect(() => {
    // In on-demand rendering (reduced motion) a scroll must ask for a new frame.
    const l = () => invalidate();
    scrollListeners.add(l);
    return () => void scrollListeners.delete(l);
  }, [invalidate]);

  useFrame((state, dt) => {
    const a = reduced ? Math.round(scrollState.anchor) : scrollState.anchor;
    const i = Math.min(STOPS.length - 2, Math.max(0, Math.floor(a)));
    const raw = Math.min(1, Math.max(0, a - i));
    const t = smooth(Math.min(1, Math.max(0, (raw - HOLD[i]) / (1 - HOLD[i]))));
    const A = STOPS[i];
    const B = STOPS[i + 1];
    pos.set(...A.pos).lerp(tgt.set(...B.pos), t);
    const lookTarget = tgt.set(...A.look).lerp(new THREE.Vector3(...B.look), t);

    // Narrow screens see less sideways, so step back from whatever we are looking at.
    const aspect = state.size.width / state.size.height;
    if (aspect < 1.35) {
      const k = Math.min(2.4, Math.max(1, 1.35 / aspect));
      pos.sub(lookTarget).multiplyScalar(k).add(lookTarget);
    }
    if (!reduced) {
      pos.x += state.pointer.x * 0.35;
      pos.y += state.pointer.y * 0.2;
    }

    if (reduced) {
      camera.position.copy(pos);
      look.current.copy(lookTarget);
    } else {
      const k = 1 - Math.exp(-dt * 3.5);
      camera.position.lerp(pos, k);
      look.current.lerp(lookTarget, k);
    }
    camera.lookAt(look.current);
  });
  return null;
}
