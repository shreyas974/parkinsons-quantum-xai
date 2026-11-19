import { spiralPoints } from "../lib/spiral";

/** Shown while the 3D layer loads, and instead of it on devices without WebGL. */
export function StaticBackdrop() {
  const d = spiralPoints({ turns: 5, radius: 230, tremor: 6, steps: 900 })
    .map(([x, y], i) => `${i ? "L" : "M"}${(x + 260).toFixed(1)} ${(y + 260).toFixed(1)}`)
    .join("");
  return (
    <div className="backdrop" aria-hidden="true">
      <svg viewBox="0 0 520 520">
        <path d={d} fill="none" stroke="#f2b54a" strokeWidth="3" strokeLinecap="round" opacity="0.8" />
      </svg>
    </div>
  );
}
