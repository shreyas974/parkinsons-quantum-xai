import { PerformanceMonitor } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Bloom, EffectComposer, Vignette } from "@react-three/postprocessing";
import { useRef, useState } from "react";
import * as THREE from "three";
import { GOLD, NAVY, TEAL, TEAL_BRIGHT } from "../lib/palette";
import { CameraRig } from "./CameraRig";
import { Dust } from "./Dust";
import { QubitArray } from "./QubitArray";
import { TremorSpiral } from "./TremorSpiral";
import { VoiceRibbon } from "./VoiceRibbon";

function HeroSpiral({ reduced }: { reduced: boolean }) {
  const group = useRef<THREE.Group>(null);
  const aspect = useThree((s) => s.size.width / s.size.height);
  const wide = aspect > 1.1;
  useFrame((state, dt) => {
    const g = group.current;
    if (!g || reduced) return;
    g.rotation.z += dt * 0.04;
    g.rotation.x = THREE.MathUtils.damp(g.rotation.x, -0.35 + state.pointer.y * 0.12, 3, dt);
    g.rotation.y = THREE.MathUtils.damp(g.rotation.y, state.pointer.x * 0.18, 3, dt);
  });
  return (
    <group position={wide ? [2.55, -0.1, 0] : [0, 4.3, -1]} scale={wide ? 1 : 0.95}>
      <group ref={group} rotation={[-0.35, 0, 0]}>
        <TremorSpiral tremor={0.08} color={GOLD} radius={2.6} turns={5} tube={0.042} drawSeconds={3.6} reduced={reduced} />
      </group>
    </group>
  );
}

function SignalPair({ reduced }: { reduced: boolean }) {
  const aspect = useThree((s) => s.size.width / s.size.height);
  const stacked = aspect < 0.95;
  const left: [number, number, number] = stacked ? [0, 2.4, 0] : [-2.85, 0, 0];
  const right: [number, number, number] = stacked ? [0, -2.4, 0] : [2.85, 0, 0];
  const spin = useRef<THREE.Group>(null);
  const spin2 = useRef<THREE.Group>(null);
  useFrame((_, dt) => {
    if (reduced) return;
    if (spin.current) spin.current.rotation.z += dt * 0.08;
    if (spin2.current) spin2.current.rotation.z += dt * 0.08;
  });
  return (
    <group>
      <group position={left}>
        <group ref={spin} position={[0, 1.05, 0]}>
          <TremorSpiral tremor={0} color={TEAL_BRIGHT} radius={1.05} turns={4} tube={0.028} reduced={reduced} />
        </group>
        <group position={[0, -1.25, 0]}>
          <VoiceRibbon jitter={0} color={TEAL_BRIGHT} width={3.6} reduced={reduced} />
        </group>
      </group>
      <group position={right}>
        <group ref={spin2} position={[0, 1.05, 0]}>
          <TremorSpiral tremor={0.075} color={GOLD} radius={1.0} turns={4} tube={0.028} seed={5} reduced={reduced} />
        </group>
        <group position={[0, -1.25, 0]}>
          <VoiceRibbon jitter={1} color={GOLD} width={3.6} reduced={reduced} />
        </group>
      </group>
    </group>
  );
}

/**
 * The whole 3D layer: one fixed canvas behind the page; the camera flies from scene to scene on scroll.
 * Bloom makes the emissive lines glow. It switches off on weak devices, with reduced motion, and
 * whenever the frame rate drops.
 */
export default function Experience({ reduced, light }: { reduced: boolean; light: boolean }) {
  const [dpr, setDpr] = useState(light ? 1 : 1.6);
  const [fx, setFx] = useState(!light && !reduced);
  return (
    <Canvas
      className="scene"
      aria-hidden="true"
      style={{ position: "fixed", inset: 0, zIndex: 0, opacity: "var(--scene-opacity, 1)" }}
      eventSource={document.getElementById("root")!}
      eventPrefix="client"
      dpr={[1, dpr]}
      frameloop={reduced ? "demand" : "always"}
      camera={{ fov: 42, near: 0.1, far: 140, position: [0, 0.35, 9] }}
      gl={{ antialias: !fx && !light, powerPreference: "high-performance" }}
    >
      <PerformanceMonitor
        onDecline={() => {
          setDpr(1);
          setFx(false);
        }}
        onIncline={() => setDpr(light ? 1 : 1.75)}
      />
      <color attach="background" args={[NAVY]} />
      <fog attach="fog" args={[NAVY, 9, 34]} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[4, 6, 6]} intensity={1.0} />
      <pointLight position={[2.5, 0.5, 3]} color={GOLD} intensity={8} distance={14} />
      <CameraRig reduced={reduced} />
      <HeroSpiral reduced={reduced} />
      <group position={[0, -1.6, -26]} scale={0.8}>
        <SignalPair reduced={reduced} />
        <pointLight position={[0, 1, 4]} color="#ffffff" intensity={4} distance={14} />
      </group>
      <group position={[0, -0.45, -56]}>
        <QubitArray reduced={reduced} />
        <pointLight position={[0, 2, 4]} color={TEAL} intensity={10} distance={16} />
      </group>
      <Dust count={light ? 350 : 900} reduced={reduced} />
      {fx && (
        <EffectComposer multisampling={0}>
          <Bloom mipmapBlur intensity={1.1} radius={0.45} luminanceThreshold={0.6} luminanceSmoothing={0.2} />
          <Vignette offset={0.3} darkness={0.55} />
        </EffectComposer>
      )}
    </Canvas>
  );
}
