import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../lib/env";
import { ScrollTrigger } from "../lib/motion";
import { circuitState, useScene } from "../lib/store";

const STEPS = [
  { title: "Encode", text: "RY(x) tips each arrow from |0⟩ (up) towards |1⟩ (down) by the feature’s value, 0 to 180°." },
  { title: "Entangle", text: "Trainable rotations and a ring of CNOT gates (the glowing loop) link the qubits, layer after layer." },
  { title: "Measure", text: "The gold-ringed qubit is read out. Its average, between −1 and +1, becomes the Parkinson’s score." },
];

/**
 * A pinned scroll story: while the section holds still, scrolling walks the 3D qubits through
 * the circuit. With reduced motion there is no pin and the finished state shows straight away.
 */
export function QuantumSection() {
  const { angles, demo } = useScene();
  const reduced = usePrefersReducedMotion();
  const section = useRef<HTMLElement>(null);
  const [active, setActive] = useState(reduced ? 3 : 0);
  const bar = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (reduced || !section.current) {
      circuitState.step = 3;
      setActive(3);
      return;
    }
    circuitState.step = 0;
    const st = ScrollTrigger.create({
      trigger: section.current,
      start: "top top",
      end: "+=200%",
      pin: true,
      scrub: true,
      onUpdate: (self) => {
        const step = Math.min(3, self.progress * 3.3);
        circuitState.step = step;
        setActive(Math.min(3, Math.floor(step)));
        if (bar.current) bar.current.style.transform = `scaleX(${self.progress})`;
      },
    });
    return () => {
      st.kill();
      circuitState.step = 3;
    };
  }, [reduced]);

  const yours = angles !== null;
  return (
    <section className="quantum" id="quantum" data-anchor="2" aria-labelledby="quantum-h" ref={section}>
      <div className="section-head">
        <p className="eyebrow">02 · Eight qubits</p>
        <h2 id="quantum-h">{yours ? "Your inputs, encoded as qubits" : "Each feature becomes a qubit’s angle"}</h2>
        <p className="lede">
          {yours
            ? demo
              ? "The arrows show the example output, because no trained model is loaded yet."
              : "The arrows show the real encoded angles from your last prediction."
            : "4 voice measures and 4 spiral shape patterns are loaded as rotations on 8 simulated qubits. Scroll to run the circuit."}
        </p>
      </div>
      <div className="circuit-progress" aria-hidden="true">
        <span ref={bar} style={{ transform: `scaleX(${reduced ? 1 : 0})` }} />
      </div>
      <ol className="steps-row">
        {STEPS.map((s, i) => (
          <li key={s.title} className={`glass${active >= i + 1 ? " is-on" : ""}${active === i + 1 ? " is-current" : ""}`} aria-current={active === i + 1 ? "step" : undefined}>
            <b><span className="step-n">{i + 1}</span> {s.title}</b>
            <span>{s.text}</span>
          </li>
        ))}
      </ol>
      <p className="step-detail" aria-hidden="true">{STEPS[Math.max(0, active - 1)].text}</p>
      <p className="fine">
        Labels show each angle and the chance of reading 0 before entanglement. Simulated on a normal computer; no
        quantum advantage is claimed.
      </p>
    </section>
  );
}
