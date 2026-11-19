import type { Health, ModelInfo, PredictResponse, ResultsSummary, SpiralChoice, VoiceChoice, VoiceSample } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

export class ApiError extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("The prediction service isn't reachable. Start it with: uvicorn backend.main:app --port 8000");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`;
    throw new ApiError(detail);
  }
  return res.json() as Promise<T>;
}

export const getHealth = () => call<Health>("/api/health");
export const getModels = () => call<ModelInfo[]>("/api/models");
export const getSamples = () => call<VoiceSample[]>("/api/samples");
export const getResults = () => call<ResultsSummary>("/api/results");

export function predict(voice: VoiceChoice, spiral: SpiralChoice, model: string): Promise<PredictResponse> {
  const form = new FormData();
  form.append("model", model);
  if (voice.kind === "sample") form.append("voice_json", JSON.stringify(voice.sample.values));
  else form.append("voice_csv", voice.file);
  const type = spiral.blob.type === "image/jpeg" ? "jpg" : "png";
  form.append("spiral", spiral.blob, `spiral.${type}`);
  return call<PredictResponse>("/api/predict", { method: "POST", body: form });
}
