import { useState } from "react";
import { missingColumns, parseFirstRow } from "../../lib/csv";
import { friendlyName, VOICE_GROUPS, voiceGroup } from "../../lib/features";
import type { VoiceChoice, VoiceSample } from "../../lib/types";
import { Segmented } from "./Segmented";

interface Props {
  samples: VoiceSample[];
  columns: string[];
  value: VoiceChoice | null;
  onChange: (v: VoiceChoice | null) => void;
}

export function VoiceInput({ samples, columns, value, onChange }: Props) {
  const [mode, setMode] = useState<"sample" | "file">("sample");
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState("");

  async function takeFile(file: File | undefined) {
    if (!file) return;
    setFileError("");
    if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") {
      setFileError("That isn’t a CSV file. Export the 22 voice features as a .csv with a header row.");
      return onChange(null);
    }
    const values = parseFirstRow(await file.text());
    const missing = missingColumns(values, columns);
    onChange({ kind: "file", file, values, missing });
  }

  const shown: Record<string, string | number> | null =
    value?.kind === "sample" ? value.sample.values : value?.kind === "file" ? value.values : null;

  return (
    <div className="input-block">
      <Segmented
        label="Voice input"
        value={mode}
        options={[["sample", "Sample recording"], ["file", "Upload CSV"]]}
        onChange={(m) => {
          setMode(m);
          onChange(null);
        }}
      />

      {mode === "sample" && (
        <div className="choice-grid" role="radiogroup" aria-label="Sample recordings">
          {samples.length === 0 && <p className="muted">Samples load from the prediction service.</p>}
          {samples.map((s) => {
            const on = value?.kind === "sample" && value.sample.id === s.id;
            return (
              <button key={s.id} type="button" role="radio" aria-checked={on} className="choice" onClick={() => onChange({ kind: "sample", sample: s })}>
                <b>{s.label}</b>
                <span>{s.description}</span>
              </button>
            );
          })}
          <p className="fine">Real rows from the UCI Parkinsons dataset (CC BY 4.0). The app never sees the true label.</p>
        </div>
      )}

      {mode === "file" && (
        <label
          className={`dropzone${dragging ? " is-over" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void takeFile(e.dataTransfer.files[0]);
          }}
        >
          <input id="voice-file" type="file" accept=".csv,text/csv" onChange={(e) => void takeFile(e.target.files?.[0])} />
          <b>{value?.kind === "file" ? value.file.name : "Drop a CSV here, or choose a file"}</b>
          <span>A header with the 22 UCI feature names and one row of values.</span>
        </label>
      )}

      {fileError && <p className="field-error" role="alert">{fileError}</p>}
      {value?.kind === "file" && value.missing.length > 0 && (
        <p className="field-error" role="alert">
          Missing or non-numeric: {value.missing.slice(0, 6).join(", ")}
          {value.missing.length > 6 ? ` and ${value.missing.length - 6} more` : ""}.
        </p>
      )}
      {value?.kind === "file" && value.missing.length === 0 && (
        <p className="field-ok">All {columns.length} voice values found.</p>
      )}

      {shown && (
        <details className="values">
          <summary>See the {columns.length} values</summary>
          <div className="values-groups">
            {Object.entries(VOICE_GROUPS).map(([g, title]) => (
              <div key={g}>
                <h4>{title}</h4>
                <dl>
                  {columns
                    .filter((c) => voiceGroup(c) === g)
                    .map((c) => (
                      <div key={c}>
                        <dt title={c}>{friendlyName(c)}</dt>
                        <dd>{shown[c] === undefined ? "—" : Number(shown[c]).toPrecision(4)}</dd>
                      </div>
                    ))}
                </dl>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
