import { useState } from "react";
import { exampleSpiralPng } from "../../lib/spiral";
import type { SpiralChoice } from "../../lib/types";
import { Segmented } from "./Segmented";
import { SpiralPad } from "./SpiralPad";

interface Props {
  value: SpiralChoice | null;
  onChange: (v: SpiralChoice | null) => void;
}

export function SpiralInput({ value, onChange }: Props) {
  const [mode, setMode] = useState<"draw" | "upload" | "example">("draw");
  const [error, setError] = useState("");

  function choose(blob: Blob, source: SpiralChoice["source"], label: string) {
    if (value) URL.revokeObjectURL(value.previewUrl);
    onChange({ blob, source, label, previewUrl: URL.createObjectURL(blob) });
  }

  function takeFile(file: File | undefined) {
    if (!file) return;
    setError("");
    if (!["image/png", "image/jpeg"].includes(file.type)) return setError("Use a PNG or JPG photo or scan of the spiral.");
    if (file.size > 8 * 1024 * 1024) return setError("That image is larger than 8 MB.");
    choose(file, "upload", file.name);
  }

  if (value) {
    return (
      <div className="input-block">
        <div className="chosen">
          <img src={value.previewUrl} alt={`Chosen spiral: ${value.label}`} width={96} height={96} />
          <div>
            <b>{value.label}</b>
            <span>{value.source === "drawn" ? "Drawn on screen" : value.source === "example" ? "Synthetic example, not patient data" : "Uploaded image"}</span>
          </div>
          <button type="button" className="btn btn-small btn-ghost" onClick={() => onChange(null)}>
            Change
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="input-block">
      <Segmented
        label="Spiral input"
        value={mode}
        options={[["draw", "Draw it"], ["upload", "Upload photo"], ["example", "Example"]]}
        onChange={(m) => {
          setMode(m);
          setError("");
        }}
      />
      {mode === "draw" && <SpiralPad onDone={(b) => choose(b, "drawn", "Your drawing")} />}
      {mode === "upload" && (
        <label className="dropzone">
          <input id="spiral-file" type="file" accept="image/png,image/jpeg" onChange={(e) => takeFile(e.target.files?.[0])} />
          <b>Choose a photo or scan</b>
          <span>A spiral drawn in dark pen on plain white paper, PNG or JPG, under 8 MB.</span>
        </label>
      )}
      {mode === "example" && (
        <div className="choice-grid">
          <button type="button" className="choice" onClick={async () => choose(await exampleSpiralPng(0), "example", "Smooth spiral")}>
            <b>Smooth spiral</b>
            <span>Computer-drawn, no tremor</span>
          </button>
          <button type="button" className="choice" onClick={async () => choose(await exampleSpiralPng(1), "example", "Shaky spiral")}>
            <b>Shaky spiral</b>
            <span>Computer-drawn, with a tremor-like wobble</span>
          </button>
          <p className="fine">Synthetic drawings for trying the app. They are not patient data.</p>
        </div>
      )}
      {error && <p className="field-error" role="alert">{error}</p>}
    </div>
  );
}
