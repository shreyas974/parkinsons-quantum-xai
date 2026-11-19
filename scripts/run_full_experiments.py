"""Run the experiment matrix, then publish validated fold and summary tables."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLASSICAL = ("lr", "svm", "rf", "mlp", "xgb")
METRICS = ("balanced_accuracy", "sensitivity", "specificity", "f1", "auc", "train_time_s")
INPUT_CODES = {"voice": "E1", "spiral": "E2", "fusion": "E3"}


@dataclass(frozen=True)
class Run:
    experiment: str
    kind: str
    models: tuple[str, ...]
    seed: int
    epochs: int = 60
    fraction: float = 1.0

    @property
    def name(self):
        models = "-".join(self.models)
        return f"{self.experiment}-{self.kind}-{models}-s{self.seed}-e{self.epochs}-n{self.fraction}"

    @property
    def setting(self):
        if self.experiment in {"A2", "A3", "A4"}:
            return f"{self.epochs} epochs, {self.fraction:.0%} of training groups"
        return "standard"


def run_plan():
    for seed in range(3):
        yield Run("E3", "fusion", ("vqc",), seed)
        yield Run("A1", "fusion", ("vqc_noent",), seed)
    for seed in range(3):
        for kind, experiment in INPUT_CODES.items():
            quantum = ("qsvm",) if kind == "fusion" else ("qsvm", "vqc")
            yield Run(experiment, kind, quantum, seed)
            yield Run(experiment, kind, CLASSICAL, seed)
            if kind != "fusion":
                yield Run("A1", kind, ("vqc_noent",), seed)
                yield Run("A3", kind, ("vqc_noise",), seed, epochs=5)
            yield Run("A2", kind, ("vqc_l1", "vqc_l2", "vqc_l3", "vqc_l4"), seed, epochs=5)
            for fraction in (0.25, 0.5, 0.75):
                yield Run("A4", kind, ("vqc",), seed, epochs=5, fraction=fraction)
        yield Run("E4", "fusion", ("late_fusion",), seed)
        yield Run("E5", "voice", ("rf_full",), seed)


def validate(frame, run):
    expected = {(model, fold) for model in run.models for fold in range(5)}
    keys = ["model", "fold"]
    if (set(frame[keys].itertuples(index=False, name=None)) != expected
            or frame.duplicated(keys).any()
            or not frame["input"].eq(run.kind).all()
            or not frame["seed"].eq(run.seed).all()
            or not frame[list(METRICS)].notna().all().all()):
        raise ValueError(f"Incomplete results for {run.name}")


def execute(run, directory, env):
    out = directory / f"{run.name}.csv"
    log = directory / f"{run.name}.log"
    if not out.exists():
        partial = out.with_suffix(".partial.csv")
        partial.unlink(missing_ok=True)
        command = [
            sys.executable, "-u", "-m", "qpd.experiments", "--models", ",".join(run.models),
            "--inputs", run.kind, "--seeds", str(run.seed), "--folds", "5",
            "--epochs", str(run.epochs), "--train-fraction", str(run.fraction),
            "--loss-dir", str(RESULTS / "loss"), "--out", str(partial),
        ]
        with log.open("w", encoding="utf-8") as stream:
            print(" ".join(command), file=stream, flush=True)
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        validate(pd.read_csv(partial), run)
        partial.replace(out)
    frame = pd.read_csv(out)
    validate(frame, run)
    return frame.assign(experiment=run.experiment, setting=run.setting,
                        train_fraction=run.fraction, epochs=run.epochs)


def add_ablation_references(results):
    depth_reference = results[(results["experiment"] == "A2") & (results["model"] == "vqc_l3")]
    size_reference = depth_reference.assign(experiment="A4", model="vqc")
    noise_reference = depth_reference[depth_reference["input"] != "fusion"].assign(experiment="A3", model="vqc")
    return pd.concat([results, size_reference, noise_reference], ignore_index=True)


def test_best_fusion(results):
    fusion = results[results["experiment"] == "E3"]
    average = fusion.groupby("model").balanced_accuracy.mean()
    classical = average.loc[list(CLASSICAL)].idxmax()
    quantum = average.loc[["qsvm", "vqc"]].idxmax()
    pairs = fusion.pivot(index=["seed", "fold"], columns="model", values="balanced_accuracy")
    # Round differences so subtraction artifacts cannot break ties in the signed ranks.
    delta = (pairs[quantum] - pairs[classical]).round(12)
    if len(delta) != 15 or delta.isna().any():
        raise ValueError("Expected 15 complete paired fusion scores")
    if delta.eq(0).all():
        statistic, p_value = 0.0, 1.0
    else:
        # Wilcoxon (1945) ranks the paired differences.
        statistic, p_value = wilcoxon(delta, alternative="two-sided", method="auto")
    return pd.DataFrame([{
        "quantum": quantum, "classical": classical, "pairs": len(delta),
        "quantum_mean": average[quantum], "classical_mean": average[classical],
        "statistic": statistic, "p_value": p_value,
    }])


def publish(results):
    order = ["experiment", "input", "model", "train_fraction", "seed", "fold"]
    results = add_ablation_references(results).sort_values(order)
    if results.duplicated(order).any():
        raise ValueError("Duplicate fold results in experiment matrix")
    summary = results.groupby(["experiment", "input", "model", "setting"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    tables = {
        "results.csv": results,
        "summary.csv": summary.reset_index(),
        "wilcoxon.csv": test_best_fusion(results),
    }
    for filename, frame in tables.items():
        destination = RESULTS / filename
        temporary = destination.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
        print(f"Wrote {len(frame)} rows to {destination}", flush=True)
    print(tables["wilcoxon.csv"].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--shard-dir", type=Path, default=RESULTS / "full_run")
    args = parser.parse_args()
    args.shard_dir = args.shard_dir.resolve()
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    frames, failures = [], []
    runs = list(run_plan())
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(execute, run, args.shard_dir, env): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            try:
                frames.append(future.result())
                print(f"Completed {len(frames)}/{len(runs)}: {run.name}", flush=True)
            except (OSError, subprocess.CalledProcessError, ValueError) as err:
                failures.append(run.name)
                print(f"Failed {run.name}: {err}", flush=True)
    if failures:
        raise SystemExit(f"Check shard logs and retry: {', '.join(failures)}")
    publish(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    main()
