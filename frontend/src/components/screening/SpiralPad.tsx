import { useEffect, useRef, useState } from "react";
import { spiralPoints } from "../../lib/spiral";

const SIZE = 340;

/** A drawing pad for the Archimedean spiral test. The faint guide is on its own layer and never exported. */
export function SpiralPad({ onDone }: { onDone: (blob: Blob) => void }) {
  const ink = useRef<HTMLCanvasElement>(null);
  const guide = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<[number, number] | null>(null);
  const [length, setLength] = useState(0);
  const [showGuide, setShowGuide] = useState(true);

  const scale = Math.min(2, window.devicePixelRatio || 1);

  function reset() {
    const c = ink.current!;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, SIZE, SIZE);
    setLength(0);
  }

  useEffect(() => {
    for (const c of [ink.current!, guide.current!]) {
      c.width = SIZE * scale;
      c.height = SIZE * scale;
    }
    reset();
    const g = guide.current!.getContext("2d")!;
    g.setTransform(scale, 0, 0, scale, 0, 0);
    g.strokeStyle = "rgba(70, 90, 140, 0.28)";
    g.setLineDash([3, 6]);
    g.lineWidth = 1.5;
    g.beginPath();
    spiralPoints({ turns: 4, radius: SIZE * 0.42, tremor: 0, steps: 600 }).forEach(([x, y], i) =>
      i ? g.lineTo(SIZE / 2 + x, SIZE / 2 + y) : g.moveTo(SIZE / 2 + x, SIZE / 2 + y),
    );
    g.stroke();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function point(e: React.PointerEvent): [number, number] {
    const r = ink.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * SIZE, ((e.clientY - r.top) / r.height) * SIZE];
  }

  function move(e: React.PointerEvent) {
    if (!drawing.current) return;
    const p = point(e);
    const ctx = ink.current!.getContext("2d")!;
    ctx.strokeStyle = "#1f2a44";
    ctx.lineWidth = 2.6;
    ctx.lineCap = ctx.lineJoin = "round";
    const from = last.current ?? p;
    ctx.beginPath();
    ctx.moveTo(...from);
    ctx.lineTo(...p);
    ctx.stroke();
    setLength((l) => l + Math.hypot(p[0] - from[0], p[1] - from[1]));
    last.current = p;
  }

  const enough = length > 400;

  return (
    <div className="pad">
      <div className="pad-surface">
        <canvas
          ref={ink}
          aria-label="Drawing area: draw a spiral from the centre outwards"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            drawing.current = true;
            last.current = point(e);
          }}
          onPointerMove={move}
          onPointerUp={() => {
            drawing.current = false;
            last.current = null;
          }}
        />
        <canvas ref={guide} className="pad-guide" hidden={!showGuide} aria-hidden="true" />
      </div>
      <div className="pad-tools">
        <label className="check">
          <input type="checkbox" checked={showGuide} onChange={(e) => setShowGuide(e.target.checked)} /> Show guide
        </label>
        <button type="button" className="btn btn-small btn-ghost" onClick={reset}>
          Clear
        </button>
        <button
          type="button"
          className="btn btn-small btn-primary"
          disabled={!enough}
          onClick={() => ink.current!.toBlob((b) => b && onDone(b), "image/png")}
        >
          Use this drawing
        </button>
      </div>
      <p className="fine">
        Start in the middle and draw outwards, about four turns, without lifting. Drawings on a screen differ from the
        pen-on-paper scans the model learned from, so treat the result as a demo.
      </p>
    </div>
  );
}
