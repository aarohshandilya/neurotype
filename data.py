"""EEG loading and preprocessing for NeuroType.

Two sources share one output format (`EEGData`):
  * kaggle: real recordings from a Kaggle copy of the EEG Motor Movement/Imagery dataset.
  * physionet: the same recordings, fetched from PhysioNet through MNE.
  * synthetic: simulated signals for the offline demo. Never real EEG.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PIPELINE_VERSION = 1

SFREQ = 160.0
BAND = (7.0, 30.0)  # mu and beta rhythms, where motor imagery shows up
EPOCH = (0.5, 3.5)  # seconds after the cue; skips the visual response to the cue itself

# T1/T2 mean different things in different runs. In the runs below T1 = left, T2 = right.
# In runs 5, 6, 9, 10, 13, 14 they mean fists/feet, so those runs are rejected.
IMAGERY_RUNS = (4, 8, 12)
EXECUTED_RUNS = (3, 7, 11)
LABELS = ("Left", "Right")
ANNOTATION_TO_CLASS = {"T1": 0, "T2": 1}

# Commonly excluded: 88, 92 and 100 use a different sampling rate or timing; 89 and 104
# have recording problems. Keeping them out avoids silent resampling surprises.
EXCLUDED_SUBJECTS = (88, 89, 92, 100, 104)
VALID_SUBJECTS = tuple(s for s in range(1, 110) if s not in EXCLUDED_SUBJECTS)
DEFAULT_SUBJECTS = (1, 2, 3, 4, 5)
MB_PER_RUN = 2.6  # approximate EDF size, used only for the download notice

# A community-uploaded copy of the PhysioNet recordings. Every file still passes the
# annotation, sampling-rate and channel checks below. The whole dataset downloads at once.
KAGGLE_DATASET = "gamalasran/physionet-eeg-motor-movement-imagery"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
KAGGLE_APPROX_GB = 2
KAGGLE_DIR = DATA_DIR / "kaggle"  # unzip a manual download here to skip kagglehub
REAL_SOURCES = ("kaggle", "physionet")

SYNTHETIC_CHANNELS = ("FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                      "CP3", "CPz", "CP4", "F3", "Fz", "F4")
DISPLAY_CHANNELS = ("C3", "Cz", "C4")


class DataError(RuntimeError):
    """A problem the interface can show to the user as plain guidance."""


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"
    subjects: tuple[int, ...] = (1, 2, 3)
    runs: tuple[int, ...] = IMAGERY_RUNS

    def __post_init__(self):
        object.__setattr__(self, "subjects", tuple(sorted(set(int(s) for s in self.subjects))))
        object.__setattr__(self, "runs", tuple(sorted(set(int(r) for r in self.runs))))

    @property
    def key(self) -> str:
        payload = json.dumps([self.source, self.subjects, self.runs, SFREQ, BAND, EPOCH,
                              PIPELINE_VERSION])
        return f"{self.source}_{hashlib.sha1(payload.encode()).hexdigest()[:10]}"

    @property
    def uses_executed_runs(self) -> bool:
        return any(r in EXECUTED_RUNS for r in self.runs)


@dataclass
class EEGData:
    X: np.ndarray  # trials x channels x samples, microvolts
    y: np.ndarray  # 0 = left, 1 = right
    subjects: np.ndarray
    runs: np.ndarray
    ch_names: list[str]
    sfreq: float
    source: str
    annotation_report: list[dict] = field(default_factory=list)

    @property
    def is_real(self) -> bool:
        return self.source in REAL_SOURCES

    @property
    def times(self) -> np.ndarray:
        return EPOCH[0] + np.arange(self.X.shape[-1]) / self.sfreq


def validate_config(cfg: DataConfig) -> None:
    if cfg.source not in ("synthetic",) + REAL_SOURCES:
        raise DataError(f"Unknown data source: {cfg.source}")
    if not cfg.subjects:
        raise DataError("Choose at least one participant.")
    if len(cfg.runs) < 2:
        raise DataError("Choose at least two runs. One run must be held out for testing.")
    if cfg.source in REAL_SOURCES:
        bad = [s for s in cfg.subjects if s not in VALID_SUBJECTS]
        if bad:
            raise DataError(f"Participants {bad} are unavailable. Use 1-109, excluding "
                            f"{list(EXCLUDED_SUBJECTS)}.")
        bad = [r for r in cfg.runs if r not in IMAGERY_RUNS + EXECUTED_RUNS]
        if bad:
            raise DataError(f"Runs {bad} are not left/right hand runs. Use {list(IMAGERY_RUNS)} "
                            f"for imagery or {list(EXECUTED_RUNS)} for executed movement.")


def _cache_path(cfg: DataConfig) -> Path:
    return DATA_DIR / "processed" / f"{cfg.key}.npz"


def is_prepared(cfg: DataConfig) -> bool:
    return cfg.source == "synthetic" or _cache_path(cfg).exists()


def _check_annotations(raw, subject: int, run: int) -> dict:
    """Confirm the loaded annotations before trusting any label."""
    counts = Counter(str(d).strip() for d in raw.annotations.description)
    if not counts["T1"] or not counts["T2"]:
        raise DataError(f"S{subject:03d} run {run} has no T1/T2 annotations "
                        f"(found {dict(counts)}), so its labels cannot be confirmed.")
    unexpected = set(counts) - {"T0", "T1", "T2"}
    if unexpected:
        raise DataError(f"S{subject:03d} run {run} has unexpected annotations: {sorted(unexpected)}.")
    return {"subject": subject, "run": run, "T0_rest": counts["T0"],
            "T1_left": counts["T1"], "T2_right": counts["T2"]}


def _physionet_fetcher(cfg: DataConfig) -> Callable[[int], list]:
    from mne.datasets import eegbci

    raw_dir = DATA_DIR / "mne"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def fetch(subject: int) -> list:
        try:
            return eegbci.load_data(subject, list(cfg.runs), path=str(raw_dir), update_path=False)
        except Exception as exc:
            raise DataError(f"Could not download participant {subject} from PhysioNet: {exc}. "
                            "Check your connection and try again; finished files stay cached.") from exc

    return fetch


def _index_edf(root: Path) -> dict[str, Path]:
    return {p.name.upper(): p for p in root.rglob("*") if p.suffix.lower() == ".edf"}


def _kaggle_fetcher(cfg: DataConfig) -> Callable[[int], list]:
    """Find S001R04.edf-style files at any depth: uploads do not share one folder layout."""
    index = _index_edf(KAGGLE_DIR) if KAGGLE_DIR.exists() else {}
    if not index:
        try:
            import kagglehub
            index = _index_edf(Path(kagglehub.dataset_download(KAGGLE_DATASET)))
        except ImportError as exc:
            raise DataError("kagglehub is not installed. Run: pip install kagglehub") from exc
        except Exception as exc:
            raise DataError(f"Could not download the dataset from Kaggle: {exc}. Sign in with "
                            "a Kaggle API token, or download it from the dataset page and unzip it "
                            f"into {KAGGLE_DIR}.") from exc
    if not index:
        raise DataError(f"No EDF files were found in the Kaggle dataset. Expected names like "
                        f"S001R04.edf. Check {KAGGLE_URL}.")

    def fetch(subject: int) -> list:
        names = [f"S{subject:03d}R{run:02d}.EDF" for run in cfg.runs]
        missing = [n for n in names if n not in index]
        if missing:
            raise DataError(f"The Kaggle files are missing {', '.join(missing[:4])}. "
                            f"Found {len(index)} EDF files in total.")
        return [index[n] for n in names]

    return fetch


def _load_real(cfg: DataConfig, progress: Callable[[float, str], None]) -> EEGData:
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    progress(0.0, "Finding the recordings. A first Kaggle download can take a while.")
    fetch = _kaggle_fetcher(cfg) if cfg.source == "kaggle" else _physionet_fetcher(cfg)
    X, y, subjects, runs, report, ch_names = [], [], [], [], [], None
    total = len(cfg.subjects) * len(cfg.runs)

    for i, subject in enumerate(cfg.subjects):
        progress(i / len(cfg.subjects), f"Fetching participant {subject}")
        paths = fetch(subject)

        for j, (run, path) in enumerate(zip(cfg.runs, paths)):
            progress((i * len(cfg.runs) + j) / total, f"Preprocessing S{subject:03d} run {run}")
            raw = mne.io.read_raw_edf(path, preload=True)
            eegbci.standardize(raw)
            if raw.info["sfreq"] != SFREQ:
                raise DataError(f"S{subject:03d} is sampled at {raw.info['sfreq']} Hz, not {SFREQ:.0f} Hz.")
            report.append(_check_annotations(raw, subject, run))

            # Filtering stays inside one continuous run, so no signal crosses a held-out boundary.
            raw.filter(*BAND, fir_design="firwin", skip_by_annotation="edge")
            events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
            epochs = mne.Epochs(raw, events, {"T1": 1, "T2": 2}, tmin=EPOCH[0],
                                tmax=EPOCH[1] - 1 / SFREQ, picks="eeg", baseline=None,
                                proj=False, preload=True)
            if ch_names is None:
                ch_names = list(epochs.ch_names)
            elif list(epochs.ch_names) != ch_names:
                raise DataError(f"S{subject:03d} run {run} has a different channel layout.")
            labels = epochs.events[:, 2] - 1  # event 1 (T1) -> 0 left, event 2 (T2) -> 1 right
            X.append(epochs.get_data(copy=False).astype(np.float32) * 1e6)
            y.append(labels)
            subjects.append(np.full(len(labels), subject))
            runs.append(np.full(len(labels), run))

    return EEGData(np.concatenate(X), np.concatenate(y).astype(int), np.concatenate(subjects),
                   np.concatenate(runs), ch_names, SFREQ, cfg.source, report)


def _load_synthetic(cfg: DataConfig) -> EEGData:
    """Simulated contralateral mu suppression. It exists to exercise the software only;
    results on it say nothing about EEG or about these models on real people."""
    rng = np.random.default_rng(2026)
    n_trials, n_samples = 15, int(round((EPOCH[1] - EPOCH[0]) * SFREQ))
    names = list(SYNTHETIC_CHANNELS)
    sos_mu = butter(4, (8, 12), "bandpass", fs=SFREQ, output="sos")
    sos_bg = butter(4, BAND, "bandpass", fs=SFREQ, output="sos")
    left_side = np.array([n in ("FC3", "C5", "C3", "C1", "CP3") for n in names])
    right_side = np.array([n in ("FC4", "C2", "C4", "C6", "CP4") for n in names])
    t = np.arange(n_samples) / SFREQ
    envelope = 1 - 0.6 * np.exp(-0.5 * ((t - 1.4) / 0.8) ** 2)  # suppression dips mid-trial

    X, y, subjects, runs = [], [], [], []
    for subject in cfg.subjects:
        gains = rng.uniform(0.7, 1.3, len(names))
        for run in cfg.runs:
            labels = rng.permutation(np.arange(n_trials) % 2)
            for label in labels:
                pad = 80  # filter transient padding, trimmed below
                mu = sosfiltfilt(sos_mu, rng.standard_normal((2, n_samples + 2 * pad)), axis=-1)[:, pad:-pad]
                bg = sosfiltfilt(sos_bg, rng.standard_normal((len(names), n_samples + 2 * pad)), axis=-1)[:, pad:-pad]
                mu = mu / mu.std(axis=-1, keepdims=True)
                signal = 4.0 * bg
                # Imagining the right hand suppresses mu over the left hemisphere, and vice versa.
                signal[left_side] += 1.5 * mu[0] * (envelope if label == 1 else 1)
                signal[right_side] += 1.5 * mu[1] * (envelope if label == 0 else 1)
                X.append(signal * gains[:, None])
                y.append(label)
                subjects.append(subject)
                runs.append(run)
    return EEGData(np.array(X, dtype=np.float32), np.array(y), np.array(subjects), np.array(runs),
                   names, SFREQ, "synthetic")


def load(cfg: DataConfig, progress: Callable[[float, str], None] | None = None) -> EEGData:
    """Load trials. Recordings download only when this is called, and are cached."""
    validate_config(cfg)
    progress = progress or (lambda *_: None)
    if cfg.source == "synthetic":
        return _load_synthetic(cfg)

    cache = _cache_path(cfg)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return EEGData(z["X"], z["y"], z["subjects"], z["runs"], z["ch_names"].tolist(),
                           float(z["sfreq"]), cfg.source, json.loads(str(z["report"])))
    data = _load_real(cfg, progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=data.X, y=data.y, subjects=data.subjects, runs=data.runs,
                        ch_names=np.array(data.ch_names), sfreq=data.sfreq,
                        report=json.dumps(data.annotation_report))
    return data
