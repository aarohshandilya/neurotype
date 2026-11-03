"""Export held-out predictions for the website:  python export_web.py --source kaggle"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import data as D
import model as M

STRIDE = 2  # keep every second sample, enough for display


def _value(array: np.ndarray, i: int) -> float | None:
    return None if np.isnan(array[i]) else round(float(array[i]), 4)


def export(cfg: D.DataConfig, out: Path) -> dict:
    results, oof = M.load_results(cfg), M.load_oof(cfg)
    if not results or oof is None:
        raise D.DataError(f"Not evaluated yet. First run: python model.py --source {cfg.source}")
    d = D.load(cfg)
    picks = [d.ch_names.index(c) for c in D.DISPLAY_CHANNELS]
    models = results["models"]
    trials = []
    for i in range(len(d.y)):
        trial = {"s": int(d.subjects[i]), "r": int(d.runs[i]), "y": int(d.y[i]),
                 "w": np.round(d.X[i, picks][:, ::STRIDE], 1).tolist(),
                 "p": {m: _value(oof[f"within_{m}"], i) for m in models}}
        if results["loso"]:
            trial["l"] = {m: _value(oof[f"loso_{m}"], i) for m in models if f"loso_{m}" in oof}
        trials.append(trial)
    train = {m: {s: results[s][m]["train_balanced_accuracy"] for s in ("within", "loso") if m in results[s]}
             for m in models}
    payload = {"format": "neurotype-replay-1", "source": cfg.source, "laneNames": list(D.DISPLAY_CHANNELS),
               "lanesFs": d.sfreq / STRIDE, "models": models, "train": train, "trials": trials}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NeuroType results for the website.")
    parser.add_argument("--source", choices=("synthetic",) + D.REAL_SOURCES, default="kaggle")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--runs", type=int, nargs="+", default=list(D.IMAGERY_RUNS))
    parser.add_argument("--out", type=Path, default=D.ROOT / "website" / "replay.json")
    args = parser.parse_args()
    default = D.DEFAULT_SUBJECTS if args.source in D.REAL_SOURCES else (1, 2, 3)
    config = D.DataConfig(args.source, tuple(args.subjects or default), tuple(args.runs))
    try:
        payload = export(config, args.out)
    except D.DataError as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"Wrote {args.out} with {len(payload['trials'])} trials. Drop it onto the website.")
