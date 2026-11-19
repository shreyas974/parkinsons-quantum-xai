import { describe, expect, it } from "vitest";
import { RESEARCH } from "./research";

describe("research figures", () => {
  it("modality shares add up to 100% for each model", () => {
    for (const s of RESEARCH.shares) expect(s.voice + s.spiral).toBeCloseTo(100, 5);
  });
  it("fusion results are valid balanced accuracies, best first", () => {
    const means = RESEARCH.fusion.map((f) => f.mean);
    for (const f of RESEARCH.fusion) {
      expect(f.mean).toBeGreaterThan(0);
      expect(f.mean).toBeLessThanOrEqual(1);
      expect(f.sd).toBeGreaterThan(0);
    }
    expect([...means].sort((a, b) => b - a)).toEqual(means);
  });
  it("keeps the verified headline numbers", () => {
    expect(RESEARCH.fusion.find((f) => f.id === "mlp")?.mean).toBe(0.775);
    expect(RESEARCH.fusion.find((f) => f.id === "qsvm")?.mean).toBe(0.755);
    expect(RESEARCH.wilcoxon).toMatchObject({ p: 0.7797, pairs: 15 });
    expect(RESEARCH.agreement).toMatchObject({ spearman: 0.452, examples: 30 });
    expect(RESEARCH.data).toEqual({ recordings: 195, subjectIds: 32, healthyIds: 8, spirals: 102 });
  });
});
