/**
 * Verified results from the research branch (shreyas/data-ml-quantum, infographic commit 2079598).
 * Checked against results/results.csv, wilcoxon.csv and shap_metrics.json. Edit here, not in the markup.
 */
export const RESEARCH = {
  commit: "2079598",
  branch: "shreyas/data-ml-quantum",
  foldResults: 810,
  summaryRows: 54,
  tests: 14,
  design: "3 seeds × 5 person-grouped folds = 15 runs per model",
  fusion: [
    { id: "mlp", label: "Small neural network (MLP)", kind: "classical", experiment: "E3 early fusion", mean: 0.775, sd: 0.169 },
    { id: "qsvm", label: "Quantum-kernel SVM", kind: "quantum", experiment: "E3 early fusion", mean: 0.755, sd: 0.157 },
    { id: "late", label: "Late fusion (MLP + RF)", kind: "classical", experiment: "E4 late fusion", mean: 0.688, sd: 0.131 },
    { id: "vqc", label: "Variational quantum classifier", kind: "quantum", experiment: "E3 early fusion", mean: 0.556, sd: 0.121 },
  ],
  wilcoxon: { quantum: "Quantum-kernel SVM", classical: "MLP", p: 0.7797, pairs: 15 },
  agreement: { a: "Random Forest", b: "VQC", spearman: 0.452, examples: 30 },
  shares: [
    { model: "Random Forest", voice: 74.1, spiral: 25.9 },
    { model: "VQC", voice: 66.4, spiral: 33.6 },
  ],
  topVoiceFeatures: ["PPE", "spread1"],
  data: { recordings: 195, subjectIds: 32, healthyIds: 8, spirals: 102 },
  image: {
    preview: "/research/research-results-preview.webp",
    full: "/research/research-results.png",
    width: 2175,
    height: 5220,
    sizeMb: 1.6,
  },
} as const;
