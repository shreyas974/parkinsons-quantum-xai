import { Html } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { DEFAULT_QUBIT_NAMES, probZero } from "../lib/features";
import { GOLD, TEAL_BRIGHT } from "../lib/palette";
import { circuitState, scrollListeners, scrollState, useScene } from "../lib/store";

const EXAMPLE_ANGLES = [2.3, 2.05, 0.95, 1.1, 2.1, 1.35, 0.7, 1.6];
const R = 0.55;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function layout(aspect: number) {
  const wide = aspect > 1.15;
  return Array.from({ length: 8 }, (_, i) =>
    wide ? new THREE.Vector3((i - 3.5) * 1.38, 0, 0) : new THREE.Vector3(((i % 4) - 1.5) * 1.6, i < 4 ? 2.6 : -0.2, 0),
  );
}

interface QubitProps {
  position: THREE.Vector3;
  angle: number;
  measured: boolean;
  reduced: boolean;
}

/** One Bloch sphere. Reads the circuit step every frame, so scrolling never re-renders React. */
function Qubit({ position, angle, measured, reduced }: QubitProps) {
  const tilt = useRef<THREE.Group>(null);
  const spin = useRef<THREE.Group>(null);
  const shell = useRef<THREE.MeshStandardMaterial>(null);
  const arrow = useRef<THREE.MeshStandardMaterial>(null);
  const tip = useRef<THREE.MeshStandardMaterial>(null);
  const halo = useRef<THREE.Mesh>(null);
  const wire = useMemo(() => new THREE.WireframeGeometry(new THREE.SphereGeometry(R, 14, 8)), []);
  useEffect(() => () => wire.dispose(), [wire]);

  useFrame(({ clock }, dt) => {
    const s = circuitState.step;
    const encode = clamp01(s);
    const entangle = clamp01(s - 1);
    const measure = clamp01(s - 2);
    // Step 1: RY(theta) tips the arrow from |0> (up) towards |1> (down).
    const target = -angle * encode;
    if (tilt.current) tilt.current.rotation.z = reduced ? target : THREE.MathUtils.damp(tilt.current.rotation.z, target, 5, dt);
    // Step 2: entangling layers; a turn about the vertical axis stands in for the phases they add.
    if (spin.current && !reduced) spin.current.rotation.y += dt * (0.15 + 1.2 * entangle * (1 - measure * 0.7));
    // Step 3: only the measured qubit stays bright.
    const fade = measured ? 1 : 1 - 0.65 * measure;
    if (shell.current) shell.current.opacity = 0.07 + 0.08 * entangle * fade;
    if (arrow.current) arrow.current.opacity = fade;
    if (tip.current) tip.current.opacity = fade;
    if (halo.current) {
      const pulse = reduced ? 1 : 1 + 0.06 * Math.sin(clock.elapsedTime * 3);
      halo.current.scale.setScalar(measured ? (0.85 + 0.15 * measure) * pulse : 1);
      (halo.current.material as THREE.MeshBasicMaterial).opacity = measured ? 0.25 + 0.75 * measure : 0;
    }
  });

  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[R, 32, 16]} />
        <meshStandardMaterial ref={shell} color={TEAL_BRIGHT} transparent opacity={0.07} roughness={0.2} depthWrite={false} />
      </mesh>
      <lineSegments geometry={wire}>
        <lineBasicMaterial color={TEAL_BRIGHT} transparent opacity={0.18} />
      </lineSegments>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[R, 0.006, 6, 64]} />
        <meshBasicMaterial color={TEAL_BRIGHT} transparent opacity={0.5} />
      </mesh>
      <mesh ref={halo} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[R * 1.3, 0.016, 8, 72]} />
        <meshBasicMaterial color={GOLD} transparent opacity={0} toneMapped={false} />
      </mesh>
      <group ref={spin}>
        <group ref={tilt}>
          <mesh position={[0, R * 0.45, 0]}>
            <cylinderGeometry args={[0.02, 0.02, R * 0.9, 8]} />
            <meshStandardMaterial ref={arrow} color="#ffffff" emissive="#ffffff" emissiveIntensity={1.2} transparent />
          </mesh>
          <mesh position={[0, R * 0.95, 0]}>
            <coneGeometry args={[0.06, 0.14, 12]} />
            <meshStandardMaterial ref={tip} color="#ffffff" emissive="#ffffff" emissiveIntensity={1.4} transparent />
          </mesh>
        </group>
      </group>
    </group>
  );
}

/** The 8 qubits after angle encoding, linked in the ring of CNOTs each layer applies. */
export function QubitArray({ reduced }: { reduced: boolean }) {
  const { angles, names } = useScene();
  const aspect = useThree((s) => s.size.width / s.size.height);
  const positions = useMemo(() => layout(aspect), [aspect]);
  const shown = angles ?? EXAMPLE_ANGLES;
  const labels = names ?? DEFAULT_QUBIT_NAMES;

  const ring = useMemo(() => {
    const pts = [...positions, positions[0]].map((p) => p.clone().add(new THREE.Vector3(0, 0, -0.02)));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(
      g,
      new THREE.LineDashedMaterial({ color: TEAL_BRIGHT, dashSize: 0.12, gapSize: 0.1, transparent: true, opacity: 0.2, toneMapped: false }),
    );
    line.computeLineDistances();
    return line;
  }, [positions]);
  useEffect(() => () => ring.geometry.dispose(), [ring]);

  useFrame(({ clock }) => {
    const entangle = clamp01(circuitState.step - 1);
    const pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(clock.elapsedTime * 2.2);
    (ring.material as THREE.LineDashedMaterial).opacity = 0.12 + entangle * (0.45 + 0.35 * pulse);
  });

  // Labels go into a fixed layer of their own: the canvas listens on the whole page, so drei would
  // otherwise attach them to the scrolling page and they would drift off screen.
  const portal = useMemo(() => ({ current: document.getElementById("scene-labels") as HTMLElement }), []);

  // Only show the HTML labels while this section is on screen; elsewhere they would float over other text.
  const [labelsOn, setLabelsOn] = useState(false);
  useEffect(() => {
    const check = () => setLabelsOn(scrollState.anchor > 1.6 && scrollState.anchor < 2.86);
    check();
    scrollListeners.add(check);
    return () => void scrollListeners.delete(check);
  }, []);

  return (
    <group>
      <primitive object={ring} />
      {positions.map((p, i) => (
        <group key={i}>
          <Qubit position={p} angle={shown[i] ?? 0} measured={i === 0} reduced={reduced} />
          {labelsOn && (
            <Html position={[p.x, p.y - R - 0.42, p.z]} center className="qubit-label" zIndexRange={[5, 0]} portal={portal}>
              <b>{labels[i]}</b>
              <span>
                {Math.round(((shown[i] ?? 0) * 180) / Math.PI)}°
                {aspect > 1.15 && <> · P(0) {Math.round(probZero(shown[i] ?? 0) * 100)}%</>}
              </span>
            </Html>
          )}
        </group>
      ))}
    </group>
  );
}
