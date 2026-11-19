import { describe, expect, it } from "vitest";
import { missingColumns, parseFirstRow } from "./csv";
import { formatSigned, friendlyName, probZero } from "./features";
import { anchorAt } from "./scroll";
import { spiralPoints } from "./spiral";

describe("csv", () => {
  it("reads the first row, including quoted cells", () => {
    const row = parseFirstRow('name,"MDVP:Fo(Hz)",PPE\nphon_1,119.99,0.28\nphon_2,1,2\n');
    expect(row).toEqual({ name: "phon_1", "MDVP:Fo(Hz)": "119.99", PPE: "0.28" });
  });
  it("names missing or non-numeric columns", () => {
    expect(missingColumns({ PPE: "0.2", HNR: "loud" }, ["PPE", "HNR", "DFA"])).toEqual(["HNR", "DFA"]);
  });
});

describe("features", () => {
  it("gives plain names", () => {
    expect(friendlyName("PPE")).toBe("Pitch irregularity (PPE)");
    expect(friendlyName("spiral_PC2")).toBe("Spiral shape pattern 2");
  });
  it("RY(theta) measurement probability", () => {
    expect(probZero(0)).toBeCloseTo(1);
    expect(probZero(Math.PI)).toBeCloseTo(0);
    expect(probZero(Math.PI / 2)).toBeCloseTo(0.5);
  });
  it("formats signed values with a real minus sign", () => {
    expect(formatSigned(-0.04)).toBe("−0.04");
    expect(formatSigned(0.14)).toBe("+0.14");
  });
});

describe("scroll anchors", () => {
  const tops = [0, 1000, 2000, 3000];
  it("interpolates between sections", () => {
    expect(anchorAt(500, tops)).toBeCloseTo(0.5);
    expect(anchorAt(2500, tops)).toBeCloseTo(2.5);
    expect(anchorAt(9999, tops)).toBe(3);
  });
});

describe("spiral", () => {
  it("grows outwards and the tremor changes the path", () => {
    const calm = spiralPoints({ turns: 3, radius: 1, tremor: 0, steps: 300 });
    const shaky = spiralPoints({ turns: 3, radius: 1, tremor: 0.05, steps: 300 });
    expect(Math.hypot(...calm[300])).toBeCloseTo(1);
    expect(Math.hypot(...shaky[150])).not.toBeCloseTo(Math.hypot(...calm[150]), 3);
  });
});
