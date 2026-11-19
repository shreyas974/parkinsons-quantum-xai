import type { ModelInfo } from "../../lib/types";

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
}

export function ModelPicker({ models, value, onChange }: Props) {
  return (
    <div className="model-grid" role="radiogroup" aria-label="Model">
      {models.map((m) => (
        <button key={m.id} type="button" role="radio" aria-checked={value === m.id} className={`model ${m.kind}`} onClick={() => onChange(m.id)}>
          <span className="kind">{m.kind === "quantum" ? "Quantum" : "Classical"}</span>
          <b>{m.label}</b>
          <span className="score">
            {m.balanced_accuracy === null ? "Score after experiments" : `Balanced accuracy ${m.balanced_accuracy.toFixed(2)}`}
          </span>
        </button>
      ))}
    </div>
  );
}
