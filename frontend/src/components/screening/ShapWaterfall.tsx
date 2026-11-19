import { formatSigned, friendlyName } from "../../lib/features";
import type { Contribution } from "../../lib/types";

interface Props {
  base: number;
  items: Contribution[];
}

const ROW = 34;
const LEFT = 268;
const RIGHT = 64;
const W = 700;

/**
 * Waterfall of SHAP values: start at the average score, then each feature's push in turn.
 * Values are written on every bar, so the direction never depends on colour alone.
 */
export function ShapWaterfall({ base, items }: Props) {
  let running = base;
  const steps = items.map((c) => {
    const from = running;
    running += c.shap;
    return { ...c, from, to: running };
  });
  const values = [base, ...steps.map((s) => s.to)];
  let lo = Math.max(0, Math.min(...values) - 0.04);
  let hi = Math.min(1, Math.max(...values) + 0.04);
  if (hi - lo < 0.2) {
    const mid = (hi + lo) / 2;
    lo = Math.max(0, mid - 0.1);
    hi = Math.min(1, mid + 0.1);
  }
  const X = (v: number) => LEFT + ((v - lo) / (hi - lo)) * (W - LEFT - RIGHT);
  const H = steps.length * ROW + 64;
  const ticks: number[] = [];
  const stepT = hi - lo > 0.5 ? 0.2 : 0.1;
  for (let t = Math.ceil(lo / stepT) * stepT; t <= hi + 1e-9; t += stepT) ticks.push(Number(t.toFixed(2)));

  return (
    <figure className="waterfall">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Feature pushes from the average score ${base.toFixed(2)} to ${running.toFixed(2)}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={X(t)} x2={X(t)} y1={8} y2={H - 40} className="grid" />
            <text x={X(t)} y={H - 24} textAnchor="middle" className="tick">{t.toFixed(1)}</text>
          </g>
        ))}
        <line x1={X(base)} x2={X(base)} y1={4} y2={H - 40} className="base" />
        {steps.map((s, i) => {
          const y = 12 + i * ROW;
          const x1 = X(Math.min(s.from, s.to));
          const w = Math.max(2, Math.abs(X(s.to) - X(s.from)));
          const up = s.shap >= 0;
          return (
            <g key={s.feature}>
              <text x={LEFT - 12} y={y + 15} textAnchor="end" className="label">
                {friendlyName(s.feature)}
                <tspan className="tag" dx={6}>{s.signal}</tspan>
              </text>
              <rect x={x1} y={y} width={w} height={20} rx={4} className={up ? "up" : "down"} />
              <text x={up ? x1 + w + 6 : x1 - 6} y={y + 15} textAnchor={up ? "start" : "end"} className="val">
                {formatSigned(s.shap)}
              </text>
              {i < steps.length - 1 && <line x1={X(s.to)} x2={X(s.to)} y1={y + 20} y2={y + ROW} className="link" />}
            </g>
          );
        })}
        <text x={X(base)} y={H - 6} textAnchor="middle" className="foot">average {base.toFixed(2)}</text>
      </svg>
      <figcaption>
        Orange bars push towards Parkinson’s, blue bars towards healthy. They add up from the average score of the
        training data to this score.
      </figcaption>
    </figure>
  );
}
