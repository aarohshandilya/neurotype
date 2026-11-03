"""Training, leakage-safe evaluation and attribution for NeuroType.

Run from a terminal:  python model.py --source synthetic
                      python model.py --source kaggle --subjects 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data as D

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    torch, nn = None, None
    HAS_TORCH = False

SEED = 2026
ARTIFACT_DIR = D.ROOT / "artifacts"
MODEL_NAMES = {"csp": "CSP + logistic regression", "eegnet": "EEGNet"}
THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
Progress = Callable[[float, str], None]


# Baseline ------------------------------------------------------------------------------

def fit_csp(X: np.ndarray, y: np.ndarray):
    from mne.decoding import CSP
    import mne
    mne.set_log_level("ERROR")
    pipe = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    return pipe.fit(X.astype(np.float64), y)


def predict_csp(pipe, X: np.ndarray) -> np.ndarray:
    return pipe.predict_proba(X.astype(np.float64))[:, 1]


def saliency_csp(pipe, x: np.ndarray) -> np.ndarray:
    """Exact gradient of the logistic score with respect to the input trial."""
    csp, scaler, clf = (step for _, step in pipe.steps)
    F = csp.filters_[: csp.n_components]
    sources = F @ x.astype(np.float64)
    power = (sources ** 2).mean(axis=1)
    weights = clf.coef_[0] / scaler.scale_ / power * 2 / x.shape[-1]
    return np.einsum("k,kc,kt->ct", weights, F, sources)


# Deep model ----------------------------------------------------------------------------

class EEGNet(nn.Module if HAS_TORCH else object):
    """Compact temporal/spatial CNN after Lawhern et al. (2018), EEGNet-8,2."""

    def __init__(self, n_channels: int, n_samples: int, F1: int = 8, D_mult: int = 2, dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D_mult
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, 81), padding="same", bias=False),  # 0.5 s temporal filters
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D_mult, (n_channels, 1), groups=F1, bias=False),  # spatial filters
            nn.BatchNorm2d(F1 * D_mult), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(F2, F2, (1, 17), padding="same", groups=F2, bias=False),  # separable conv
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(F2 * (n_samples // 32), 2)

    def forward(self, x):
        return self.classifier(self.features(x))


def _prepare(X: np.ndarray):
    # Each trial is scaled by itself, so no statistics pass between train and test. One scale
    # per trial, not per channel: motor imagery is a power difference between channels.
    X = X - X.mean(axis=-1, keepdims=True)
    X = X / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    return torch.as_tensor(X[:, None], dtype=torch.float32)


def fit_eegnet(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 60) -> "EEGNet":
    # The epoch count is fixed in advance: early stopping on held-out data would leak.
    torch.manual_seed(seed)
    model = EEGNet(X.shape[1], X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xt, yt = _prepare(X), torch.as_tensor(y, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(xt), generator=gen).split(16):
            if len(idx) < 2:  # batch norm needs at least two trials
                continue
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
    return model.eval()


def predict_eegnet(model, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model.eval()(_prepare(X)), dim=1)[:, 1].numpy()


def saliency_eegnet(model, x: np.ndarray) -> np.ndarray:
    inp = _prepare(x[None]).requires_grad_(True)
    logits = model.eval()(inp)
    (logits[0, 1] - logits[0, 0]).backward()
    return inp.grad[0, 0].numpy()


def binned_saliency(grad: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Absolute attribution, averaged into time bins and scaled to 0-1 for display."""
    bins = np.array_split(np.abs(grad), n_bins, axis=1)
    out = np.stack([b.mean(axis=1) for b in bins], axis=1)
    return out / (out.max() + 1e-12)


# Metrics -------------------------------------------------------------------------------

def wilson_interval(correct: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, d = correct / n, 1 + z ** 2 / n
    centre, half = (p + z ** 2 / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [centre - half, centre + half]


def summarise(y: np.ndarray, p_right: np.ndarray) -> dict:
    pred = (p_right >= 0.5).astype(int)
    conf = np.maximum(p_right, 1 - p_right)
    correct = pred == y
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)

    bins, ece = [], 0.0
    edges = np.linspace(0.5, 1.0, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
        if mask.any():
            bins.append({"low": float(lo), "high": float(hi), "n": int(mask.sum()),
                         "confidence": float(conf[mask].mean()), "accuracy": float(correct[mask].mean())})
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())

    selective = []
    for t in THRESHOLDS:
        keep = conf >= t
        selective.append({"threshold": t, "coverage": float(keep.mean()),
                          "accuracy": float(correct[keep].mean()) if keep.any() else None,
                          "n": int(keep.sum())})
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": wilson_interval(int(correct.sum()), len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": {name: {"precision": float(prec[i]), "recall": float(rec[i]),
                             "f1": float(f1[i]), "support": int(sup[i])}
                      for i, name in enumerate(D.LABELS)},
        "ece": float(ece),
        "reliability": bins,
        "selective": selective,
    }


# Evaluation ----------------------------------------------------------------------------

def artifact_dir(cfg: D.DataConfig) -> Path:
    return ARTIFACT_DIR / cfg.key


def load_results(cfg: D.DataConfig) -> dict | None:
    path = artifact_dir(cfg) / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_oof(cfg: D.DataConfig) -> dict[str, np.ndarray] | None:
    path = artifact_dir(cfg) / "oof.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _fold_path(cfg: D.DataConfig, model: str, subject: int, run: int) -> Path:
    suffix = "pt" if model == "eegnet" else "joblib"
    return artifact_dir(cfg) / "folds" / f"{model}_s{subject}_r{run}.{suffix}"


def fold_saliency(cfg: D.DataConfig, model: str, subject: int, run: int, x: np.ndarray) -> np.ndarray | None:
    """Attribution from the fold model that did not train on this run."""
    path = _fold_path(cfg, model, subject, run)
    if not path.exists() or (model == "eegnet" and not HAS_TORCH):
        return None
    if model == "csp":
        return binned_saliency(saliency_csp(joblib.load(path), x))
    net = EEGNet(x.shape[0], x.shape[1])
    net.load_state_dict(torch.load(path, weights_only=True))
    return binned_saliency(saliency_eegnet(net, x))


def _make_folds(d: D.EEGData) -> dict[str, list[tuple[str, dict, np.ndarray, np.ndarray]]]:
    """Whole runs or whole participants are held out; trials are never split at random."""
    within, loso = [], []
    for s in np.unique(d.subjects):
        mine = d.subjects == s
        for r in np.unique(d.runs[mine]):
            test = mine & (d.runs == r)
            within.append((f"P{s} run {r}", {"subject": int(s), "run": int(r)}, mine & ~test, test))
    if len(np.unique(d.subjects)) >= 2:
        for s in np.unique(d.subjects):
            test = d.subjects == s
            loso.append((f"P{s}", {"subject": int(s)}, ~test, test))
    return {"within": within, "loso": loso}


def _assert_no_overlap(train: np.ndarray, test: np.ndarray, d: D.EEGData, scheme: str) -> None:
    assert not (train & test).any()
    if scheme == "within":
        pairs = set(zip(d.subjects[train], d.runs[train]))
        assert not pairs & set(zip(d.subjects[test], d.runs[test]))
    else:
        assert not set(d.subjects[train]) & set(d.subjects[test])


def _score_scheme(d: D.EEGData, folds: list, scheme: str, models: list[str], cfg: D.DataConfig,
                  tick: Callable[[str], None]) -> tuple[dict, dict]:
    summaries, oof = {}, {}
    for model in models:
        p = np.full(len(d.y), np.nan)
        fold_scores, train_scores = [], []
        for i, (name, meta, train, test) in enumerate(folds):
            tick(f"{MODEL_NAMES[model]}: {name} held out")
            _assert_no_overlap(train, test, d, scheme)
            if len(np.unique(d.y[train])) < 2 or len(np.unique(d.y[test])) < 2:
                continue
            if model == "csp":
                fitted = fit_csp(d.X[train], d.y[train])
                predict = lambda X, f=fitted: predict_csp(f, X)
            else:
                fitted = fit_eegnet(d.X[train], d.y[train], seed=SEED + i)
                predict = lambda X, f=fitted: predict_eegnet(f, X)
            p[test] = predict(d.X[test])
            fold_scores.append(balanced_accuracy_score(d.y[test], p[test] >= 0.5))
            train_scores.append(balanced_accuracy_score(d.y[train], predict(d.X[train]) >= 0.5))
            if scheme == "within":
                path = _fold_path(cfg, model, meta["subject"], meta["run"])
                path.parent.mkdir(parents=True, exist_ok=True)
                if model == "csp":
                    joblib.dump(fitted, path)
                else:
                    torch.save(fitted.state_dict(), path)
        scored = ~np.isnan(p)
        if not scored.any():
            continue
        result = summarise(d.y[scored], p[scored])
        result["fold_mean"], result["fold_std"] = float(np.mean(fold_scores)), float(np.std(fold_scores))
        result["n_folds"] = len(fold_scores)
        result["train_balanced_accuracy"] = float(np.mean(train_scores))
        result["per_participant"] = {
            int(s): float(balanced_accuracy_score(d.y[scored & (d.subjects == s)],
                                                  p[scored & (d.subjects == s)] >= 0.5))
            for s in np.unique(d.subjects[scored])}
        summaries[model], oof[f"{scheme}_{model}"] = result, p
    return summaries, oof


def run_evaluation(cfg: D.DataConfig, deep: bool = True, progress: Progress | None = None) -> dict:
    """Train every fold, write metrics and out-of-fold predictions, keep fold models for replay."""
    progress = progress or (lambda *_: None)
    d = D.load(cfg, lambda f, m: progress(0.15 * f, m))
    models = ["csp"] + (["eegnet"] if deep and HAS_TORCH else [])
    folds = _make_folds(d)
    if not folds["within"]:
        raise D.DataError("No folds could be built from this selection.")

    out = artifact_dir(cfg)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total, done = (len(folds["within"]) + len(folds["loso"])) * len(models), [0]

    def tick(message: str) -> None:
        progress(0.15 + 0.85 * done[0] / total, message)
        done[0] += 1

    started = time.time()
    within, oof = _score_scheme(d, folds["within"], "within", models, cfg, tick)
    loso, oof_loso = _score_scheme(d, folds["loso"], "loso", models, cfg, tick)
    oof.update(oof_loso)

    n_subjects = len(cfg.subjects)
    limit = None
    if n_subjects < 2:
        limit = "Not run: this needs at least two participants."
    elif n_subjects < 10:
        limit = (f"Based on {n_subjects} participants, so this is a rough indication only, "
                 "not evidence about the wider population.")
    results = {
        "source": cfg.source,
        "subjects": list(cfg.subjects),
        "runs": list(cfg.runs),
        "n_trials": int(len(d.y)),
        "models": models,
        "seed": SEED,
        "seconds": round(time.time() - started, 1),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "uses_executed_runs": cfg.uses_executed_runs,
        "annotations": d.annotation_report,
        "within": within,
        "loso": loso,
        "loso_note": limit,
    }
    np.savez_compressed(out / "oof.npz", **oof)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    progress(1.0, "Evaluation finished")
    return results


def _print_summary(results: dict) -> None:
    print(f"\n{results['source']}  participants {results['subjects']}  runs {results['runs']}  "
          f"{results['n_trials']} trials  {results['seconds']} s")
    for scheme, title in (("within", "Held-out run"), ("loso", "Held-out participant")):
        for model, r in results[scheme].items():
            print(f"{title:21s} {model:7s} balanced acc {r['balanced_accuracy']:.3f}  "
                  f"macro F1 {r['macro_f1']:.3f}  train {r['train_balanced_accuracy']:.3f}  "
                  f"ECE {r['ece']:.3f}")
    if results["loso_note"]:
        print("Note:", results["loso_note"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate NeuroType models.")
    parser.add_argument("--source", choices=("synthetic",) + D.REAL_SOURCES, default="synthetic")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--runs", type=int, nargs="+", default=list(D.IMAGERY_RUNS))
    parser.add_argument("--no-deep", action="store_true", help="Skip EEGNet")
    args = parser.parse_args()
    default = D.DEFAULT_SUBJECTS if args.source in D.REAL_SOURCES else (1, 2, 3)
    config = D.DataConfig(args.source, tuple(args.subjects or default), tuple(args.runs))
    try:
        _print_summary(run_evaluation(config, deep=not args.no_deep,
                                      progress=lambda f, m: print(f"[{f:4.0%}] {m}", flush=True)))
    except D.DataError as exc:
        raise SystemExit(f"Error: {exc}")
