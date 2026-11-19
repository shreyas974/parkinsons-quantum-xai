/**
 * Read the first data row of a small CSV so the page can show a preview and name missing columns
 * before anything is sent. The server checks it again; this is only for quick feedback.
 */
export function parseFirstRow(text: string): Record<string, string> {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) return {};
  const header = splitLine(lines[0]);
  const row = splitLine(lines[1]);
  const out: Record<string, string> = {};
  header.forEach((h, i) => {
    out[h.trim()] = (row[i] ?? "").trim();
  });
  return out;
}

function splitLine(line: string): string[] {
  const cells: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      cells.push(cur);
      cur = "";
    } else cur += ch;
  }
  cells.push(cur);
  return cells;
}

export function missingColumns(values: Record<string, string>, required: string[]): string[] {
  return required.filter((c) => values[c] === undefined || values[c] === "" || !Number.isFinite(Number(values[c])));
}
