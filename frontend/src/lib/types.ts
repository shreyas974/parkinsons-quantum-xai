// Mirrors backend/schemas.py. Keep the two in step.

export type Signal = "voice" | "spiral";

export interface Contribution {
  feature: string;
  signal: Signal;
  value: number;
  shap: number;
}

export interface PredictResponse {
  demo: boolean;
  model: string;
  model_label: string;
  model_kind: "quantum" | "classical";
  probability: number;
  label: "higher" | "lower";
  base_value: number;
  contributions: Contribution[];
  angles: number[];
  feature_names: string[];
  voice_share: number;
  spiral_share: number;
  elapsed_ms: number;
  note: string;
}

export interface ModelInfo {
  id: string;
  label: string;
  kind: "quantum" | "classical";
  balanced_accuracy: number | null;
}

export interface Health {
  status: string;
  demo: boolean;
  reason: string;
  models: string[];
  voice_columns: string[];
}

export interface VoiceSample {
  id: string;
  label: string;
  description: string;
  values: Record<string, number>;
}

export interface ResultRow {
  input: string;
  model: string;
  label: string;
  kind: "quantum" | "classical";
  folds: number;
  seeds: number | null;
  balanced_accuracy: number | null;
  balanced_accuracy_std: number | null;
  sensitivity: number | null;
  specificity: number | null;
  f1: number | null;
  auc: number | null;
  train_time_s: number | null;
}

export interface ResultsSummary {
  available: boolean;
  note: string;
  rows: ResultRow[];
}

/** What the user has chosen for the voice input. */
export type VoiceChoice =
  | { kind: "sample"; sample: VoiceSample }
  | { kind: "file"; file: File; values: Record<string, string>; missing: string[] };

/** What the user has chosen for the spiral input. */
export interface SpiralChoice {
  blob: Blob;
  previewUrl: string;
  source: "drawn" | "upload" | "example";
  label: string;
}
