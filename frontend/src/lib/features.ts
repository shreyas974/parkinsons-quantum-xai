/** Plain-language names for the UCI voice measures and the spiral components. */
const VOICE: Record<string, [string, string]> = {
  "MDVP:Fo(Hz)": ["Average pitch", "pitch"],
  "MDVP:Fhi(Hz)": ["Highest pitch", "pitch"],
  "MDVP:Flo(Hz)": ["Lowest pitch", "pitch"],
  "MDVP:Jitter(%)": ["Pitch wobble, %", "jitter"],
  "MDVP:Jitter(Abs)": ["Pitch wobble, absolute", "jitter"],
  "MDVP:RAP": ["Pitch wobble, RAP", "jitter"],
  "MDVP:PPQ": ["Pitch wobble, PPQ", "jitter"],
  "Jitter:DDP": ["Pitch wobble, DDP", "jitter"],
  "MDVP:Shimmer": ["Loudness wobble", "shimmer"],
  "MDVP:Shimmer(dB)": ["Loudness wobble, dB", "shimmer"],
  "Shimmer:APQ3": ["Loudness wobble, APQ3", "shimmer"],
  "Shimmer:APQ5": ["Loudness wobble, APQ5", "shimmer"],
  "MDVP:APQ": ["Loudness wobble, APQ", "shimmer"],
  "Shimmer:DDA": ["Loudness wobble, DDA", "shimmer"],
  NHR: ["Noise in the voice (NHR)", "noise"],
  HNR: ["Clear tone vs noise (HNR)", "noise"],
  RPDE: ["Voice irregularity (RPDE)", "nonlinear"],
  DFA: ["Signal scaling (DFA)", "nonlinear"],
  spread1: ["Pitch spread (spread1)", "nonlinear"],
  spread2: ["Pitch spread (spread2)", "nonlinear"],
  D2: ["Signal complexity (D2)", "nonlinear"],
  PPE: ["Pitch irregularity (PPE)", "nonlinear"],
};

export const VOICE_GROUPS: Record<string, string> = {
  pitch: "Pitch",
  jitter: "Pitch wobble (jitter)",
  shimmer: "Loudness wobble (shimmer)",
  noise: "Noise",
  nonlinear: "Irregularity measures",
};

export function friendlyName(feature: string): string {
  if (VOICE[feature]) return VOICE[feature][0];
  const pc = /^spiral_PC(\d+)$/.exec(feature);
  if (pc) return `Spiral shape pattern ${pc[1]}`;
  return feature;
}

export function voiceGroup(feature: string): string {
  return VOICE[feature]?.[1] ?? "other";
}

export const DEFAULT_QUBIT_NAMES = ["Voice 1", "Voice 2", "Voice 3", "Voice 4", "Spiral 1", "Spiral 2", "Spiral 3", "Spiral 4"];

/** Probability that a qubit prepared with RY(theta) from |0> is measured as 0. */
export const probZero = (theta: number) => Math.cos(theta / 2) ** 2;

export const formatPct = (p: number) => `${Math.round(p * 100)}%`;
export const formatSigned = (v: number, digits = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
