import { useEffect, useRef, useState } from "react";
import { getHealth, getModels, getSamples, predict } from "../../lib/api";
import { sceneStore } from "../../lib/store";
import type { Health, ModelInfo, PredictResponse, SpiralChoice, VoiceChoice, VoiceSample } from "../../lib/types";
import { ModelPicker } from "./ModelPicker";
import { ResultPanel } from "./ResultPanel";
import { SpiralInput } from "./SpiralInput";
import { VoiceInput } from "./VoiceInput";

const shortName = (f: string) =>
  f.replace(/^spiral_PC/, "Spiral ").replace(/^MDVP:/, "").replace(/^(Jitter|Shimmer):/, "");

export function ScreeningSection() {
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState("");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [samples, setSamples] = useState<VoiceSample[]>([]);
  const [voice, setVoice] = useState<VoiceChoice | null>(null);
  const [spiral, setSpiral] = useState<SpiralChoice | null>(null);
  const [model, setModel] = useState("vqc");
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const timer = useRef<number>();

  useEffect(() => {
    Promise.all([getHealth(), getModels(), getSamples()])
      .then(([h, m, s]) => {
        setHealth(h);
        setModels(m);
        setSamples(s);
        if (!m.some((x) => x.id === "vqc") && m[0]) setModel(m[0].id);
      })
      .catch((e: Error) => setOffline(e.message));
  }, []);

  const columns = health?.voice_columns ?? [];
  const voiceReady = voice !== null && (voice.kind === "sample" || voice.missing.length === 0);
  const ready = voiceReady && spiral !== null && !offline && status !== "running";

  async function run() {
    if (!voice || !spiral) return;
    setStatus("running");
    setError("");
    setElapsed(0);
    const t0 = performance.now();
    timer.current = window.setInterval(() => setElapsed(Math.floor((performance.now() - t0) / 1000)), 500);
    try {
      const r = await predict(voice, spiral, model);
      setResult(r);
      setStatus("done");
      sceneStore.set({ angles: r.angles, names: r.feature_names.map(shortName), demo: r.demo });
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    } finally {
      window.clearInterval(timer.current);
    }
  }

  return (
    <section className="screen" id="screen" data-anchor="3" aria-labelledby="screen-h">
      <div className="section-head" data-reveal>
        <p className="eyebrow">03 · Try it</p>
        <h2 id="screen-h">Run a screening</h2>
        <p className="lede">Three inputs, one score, and the reasons behind it. Nothing you upload is stored.</p>
      </div>

      {offline && (
        <p className="service-banner" role="alert">
          {offline}
        </p>
      )}
      {health?.demo && !offline && (
        <p className="service-banner demo" role="status">
          <b>Demo mode.</b> {health.reason}
        </p>
      )}

      <div className="screen-grid">
        <div className="steps">
          <div className="step glass">
            <h3><span className="num">1</span> Voice features</h3>
            <VoiceInput samples={samples} columns={columns} value={voice} onChange={setVoice} />
          </div>
          <div className="step glass">
            <h3><span className="num">2</span> Spiral drawing</h3>
            <SpiralInput value={spiral} onChange={setSpiral} />
          </div>
          <div className="step glass">
            <h3><span className="num">3</span> Model</h3>
            <ModelPicker models={models} value={model} onChange={setModel} />
          </div>
          <button type="button" className="btn btn-primary btn-wide" disabled={!ready} onClick={() => void run()}>
            {status === "running" ? "Predicting…" : "Predict"}
          </button>
          {!ready && status !== "running" && !offline && (
            <p className="fine center">
              {!voiceReady ? "Choose the voice features first." : !spiral ? "Now add a spiral." : ""}
            </p>
          )}
        </div>
        <ResultPanel status={status} result={result} error={error} elapsed={elapsed} onRetry={() => void run()} />
      </div>
    </section>
  );
}
