/** Archimedean spiral points with an optional tremor, shared by the 3D scenes and the example drawings. */
export function spiralPoints(opts: { turns: number; radius: number; tremor: number; steps: number; seed?: number }) {
  const { turns, radius, tremor, steps, seed = 1 } = opts;
  const maxT = turns * Math.PI * 2;
  const pts: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * maxT;
    const grow = t / maxT;
    // Two incommensurate wobbles plus a slow drift: a hand with a resting tremor rarely shakes at one frequency.
    const shake = tremor * (Math.sin(t * 11.3 + seed) * 0.6 + Math.sin(t * 4.7 + seed * 2.1) * 0.4) * (0.35 + grow);
    const drift = tremor * 0.8 * Math.sin(t * 0.9 + seed) * grow;
    const r = radius * grow + shake + drift;
    pts.push([r * Math.cos(t), r * Math.sin(t)]);
  }
  return pts;
}

/** Render an example spiral to a PNG, ink on white paper, like a scanned test sheet. */
export async function exampleSpiralPng(tremor: number, size = 512): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = "#1f2a44";
  ctx.lineWidth = size / 170;
  ctx.lineCap = ctx.lineJoin = "round";
  const pts = spiralPoints({ turns: 4, radius: size * 0.42, tremor: tremor * size * 0.012, steps: 1400, seed: 3 });
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(size / 2 + x, size / 2 + y) : ctx.moveTo(size / 2 + x, size / 2 + y)));
  ctx.stroke();
  return new Promise((resolve) => canvas.toBlob((b) => resolve(b!), "image/png"));
}
