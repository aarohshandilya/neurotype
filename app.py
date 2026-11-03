"""NeuroType: an EEG motor-imagery research demo. Run with: streamlit run app.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data as D
import model as M

INK, MUTED, COBALT, ELECTRIC, TEAL, LINE = "#0A1B3F", "#4A5F85", "#1450E6", "#2E8BFF", "#0792A8", "#CFE1FA"
BLUES = [[0, "#F2F8FF"], [0.5, "#7DB6FF"], [1, "#0D2F9E"]]
SPEEDS = {"Slow": 3.6, "Normal": 2.4, "Fast": 1.2}
PANEL = (Path(__file__).parent / "panel.html").read_text()
CHART_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="NeuroType", layout="wide", initial_sidebar_state="auto")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.stApp, .stMarkdown, .stApp button, .stApp input { font-family:"Instrument Sans",system-ui,sans-serif; }
.stApp { background:radial-gradient(70% 40% at 72% 0%, #D9EAFF 0%, rgba(217,234,255,0) 100%), #F2F8FF; }
#MainMenu, footer, [data-testid="stDeployButton"] { display:none; }
.block-container { padding-top:2.2rem; max-width:1240px; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #CFE1FA; }
h2, h3, h4 { font-family:Sora,sans-serif !important; color:#0A1B3F; letter-spacing:-.01em; }
.nt-head { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; flex-wrap:wrap; margin-bottom:14px; }
.nt-word { font:700 clamp(38px,5.4vw,60px)/1 Sora,sans-serif; letter-spacing:-.045em; margin:0; background:linear-gradient(100deg,#0A1B3F 8%,#1450E6 62%,#2E8BFF); -webkit-background-clip:text; background-clip:text; color:transparent; }
.nt-sub { margin:8px 0 0; color:#0A1B3F; font-size:17px; }
.nt-by { margin:2px 0 0; color:#4A5F85; font-size:14px; }
.nt-badge { font:600 13px/1 Sora,sans-serif; padding:11px 15px; border-radius:999px; background:#fff; color:#0D2F9E; border:1.5px solid #1450E6; box-shadow:0 0 0 4px rgba(46,139,255,.14); white-space:nowrap; }
.nt-badge.synth { color:#6B4600; border-color:#B98500; box-shadow:0 0 0 4px rgba(185,133,0,.13); }
.nt-note { background:#fff; border:1px solid #CFE1FA; border-left:4px solid #1450E6; border-radius:12px; padding:14px 16px; margin:6px 0 14px; }
.nt-note h4 { margin:0 0 4px; font-size:17px; padding:0; }
.nt-note p { margin:0; color:#33486E; }
.stButton button { border-radius:10px; font-weight:600; border-color:#CFE1FA; }
.stButton button[kind="primary"] { background:#1450E6; border-color:#1450E6; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:#CFE1FA !important; border-radius:16px; }
</style>""", unsafe_allow_html=True)

ss = st.session_state
for key, value in {"idx": 0, "playing": False, "started": False, "revealed": False,
                   "replay_key": None, "train_requested": False}.items():
    ss.setdefault(key, value)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_data(cfg: D.DataConfig) -> D.EEGData:
    return D.load(cfg)


@st.cache_data(show_spinner=False, max_entries=3)
def get_oof(cfg: D.DataConfig, stamp: float):
    return M.load_oof(cfg)


@st.cache_data(show_spinner=False, max_entries=64)
def get_saliency(cfg: D.DataConfig, stamp: float, model: str, subject: int, run: int, trial: int):
    return M.fold_saliency(cfg, model, subject, run, get_data(cfg).X[trial])


def go_to(idx: int) -> None:
    ss.update(idx=idx, playing=False, revealed=True)


def toggle_play(n: int) -> None:
    if ss.playing:
        ss.update(playing=False, revealed=True)
    else:
        if ss.idx >= n - 1:
            ss.idx = 0
        ss.update(playing=True, started=False, revealed=True)


def reset() -> None:
    ss.update(idx=0, playing=False, revealed=False)


def note(title: str, body: str) -> None:
    st.markdown(f'<div class="nt-note"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Instrument Sans, sans-serif", color=INK, size=13),
                      showlegend=False, hoverlabel=dict(bgcolor="#fff", bordercolor=LINE), **kw)
    return fig


# Control rail --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    source = st.radio("Data source", ["synthetic", "kaggle", "physionet"],
                      format_func=lambda s: {"synthetic": "Synthetic demo (no download)",
                                             "kaggle": "Recorded EEG from Kaggle",
                                             "physionet": "Recorded EEG from PhysioNet"}[s])
    if source in D.REAL_SOURCES:
        subjects = st.multiselect("Dataset participants", D.VALID_SUBJECTS, default=list(D.DEFAULT_SUBJECTS))
        runs = st.multiselect("Dataset runs", list(D.IMAGERY_RUNS + D.EXECUTED_RUNS), default=list(D.IMAGERY_RUNS),
                              help="4, 8, 12 are imagined left/right hand. 3, 7, 11 are executed movement.")
    else:
        subjects, runs = [1, 2, 3], list(D.IMAGERY_RUNS)
    cfg = D.DataConfig(source, tuple(subjects), tuple(runs))
    try:
        D.validate_config(cfg)
        config_error = None
    except D.DataError as exc:
        config_error = str(exc)

results = None if config_error else M.load_results(cfg)
prepared = not config_error and D.is_prepared(cfg)
stamp = (M.artifact_dir(cfg) / "results.json").stat().st_mtime if results else 0.0

# Header --------------------------------------------------------------------------------
real = source in D.REAL_SOURCES
badge = ('<span class="nt-badge">RECORDED EEG — NOT LIVE</span>' if real else
         '<span class="nt-badge synth">SYNTHETIC SIGNAL — NOT EEG, NOT LIVE</span>')
st.markdown(f"""<div class="nt-head"><div><h1 class="nt-word">NeuroType</h1>
<p class="nt-sub">EEG motor-imagery research demo</p><p class="nt-by">Aaroh Shandilya · 2026</p></div>{badge}</div>""",
            unsafe_allow_html=True)

if config_error:
    with st.sidebar:
        st.error(config_error)
    note("Check the data selection", config_error)
    st.stop()

# Heavy actions run here, in the main column, so progress is visible -------------------
if ss.pop("prepare_requested", False):
    bar = st.progress(0.0, "Starting download")
    try:
        D.load(cfg, lambda f, m: bar.progress(min(f, 1.0), m))
        st.rerun()
    except D.DataError as exc:
        st.error(str(exc))
        st.stop()

if ss.train_requested:
    ss.train_requested = False
    bar = st.progress(0.0, "Starting")
    try:
        M.run_evaluation(cfg, deep=ss.get("use_deep", True), progress=lambda f, m: bar.progress(min(f, 1.0), m))
        st.cache_data.clear()
        reset()
        st.rerun()
    except D.DataError as exc:
        st.error(str(exc))
        st.stop()

if not prepared:
    n_files = len(cfg.subjects) * len(cfg.runs)
    if source == "kaggle":
        note("Download the recordings first",
             f'The first time, this fetches the whole <a href="{D.KAGGLE_URL}">Kaggle dataset</a>, roughly '
             f"{D.KAGGLE_APPROX_GB} GB, then reuses it. A Kaggle API token may be needed. You can also "
             f"download it yourself and unzip it into <code>data/kaggle</code>.")
    else:
        note("Download the recordings first",
             f"This selection needs {n_files} files, about {n_files * D.MB_PER_RUN:.0f} MB, from PhysioNet. "
             "They are stored in the project's data folder and reused next time.")
    st.button("Download and prepare data", type="primary", on_click=lambda: ss.update(prepare_requested=True))
    st.stop()

try:
    with st.spinner("Loading trials"):
        dataset = get_data(cfg)
except D.DataError as exc:
    st.error(str(exc))
    st.stop()

available = results["models"] if results else []
oof = get_oof(cfg, stamp) if results else None

with st.sidebar:
    prefix = "Simulated participant" if not real else "Participant"
    subject = st.selectbox("Replay participant", cfg.subjects, format_func=lambda s: f"{prefix} {s}")
    run = st.selectbox("Replay run", cfg.runs, format_func=lambda r: f"Run {r}")
    if available:
        model_key = st.radio("Model", available, format_func=M.MODEL_NAMES.get)
    else:
        model_key = None
        st.caption("Model: none trained yet")
    threshold = st.slider("Confidence threshold", 0.50, 0.95, 0.60, 0.01,
                          help="Below this confidence the classifier abstains and no command is sent.")
    speed = st.select_slider("Replay speed", list(SPEEDS), value="Normal")

    mask = (dataset.subjects == subject) & (dataset.runs == run)
    trials = np.flatnonzero(mask)
    n = len(trials)
    replay_key = (cfg.key, subject, run, stamp)  # switching model keeps the trial, for comparison
    if ss.replay_key != replay_key:
        ss.replay_key = replay_key
        reset()
    if n == 0:
        st.warning("This run has no trials.")
        st.stop()
    ss.idx = min(ss.idx, n - 1)

    st.button("Pause" if ss.playing else "Start", type="primary", width="stretch",
              on_click=toggle_play, args=(n,))
    a, b, c = st.columns(3)
    a.button("Back", width="stretch", disabled=ss.idx == 0, on_click=go_to, args=(ss.idx - 1,))
    b.button("Next", width="stretch", disabled=ss.idx >= n - 1, on_click=go_to, args=(ss.idx + 1,))
    c.button("Reset", width="stretch", on_click=reset)

    st.divider()
    st.checkbox("Include EEGNet", key="use_deep", value=M.HAS_TORCH, disabled=not M.HAS_TORCH,
                help=None if M.HAS_TORCH else "Install torch to enable EEGNet.")
    st.button("Train and evaluate again" if results else "Train and evaluate", width="stretch",
              on_click=lambda: ss.update(train_requested=True))

if not results:
    estimate = "about a minute" if not real else "around 8 minutes on a two-core CPU for 5 participants"
    note("Not evaluated yet",
         f"The signal replay works now, but predictions need a trained model. Training runs every "
         f"held-out fold and takes {estimate}.")
    st.button("Train and evaluate", type="primary", key="train_main",
              on_click=lambda: ss.update(train_requested=True))
if results and results["uses_executed_runs"]:
    st.warning("This selection includes executed-movement runs, so it is not imagery alone.")

key = f"within_{model_key}"
p_right = oof[key][trials] if oof is not None and key in oof else None


def decision_for(i: int) -> dict | None:
    if p_right is None or np.isnan(p_right[i]):
        return None
    confidence = float(max(p_right[i], 1 - p_right[i]))
    return {"side": D.LABELS[int(p_right[i] >= 0.5)], "confidence": confidence, "accepted": confidence >= threshold}


def cursor_path() -> list[int]:
    pos, path = 0, [0]
    for i in range(n):
        d = decision_for(i)
        if d and d["accepted"]:
            pos = int(np.clip(pos + (1 if d["side"] == "Right" else -1), -5, 5))
        path.append(pos)
    return path


def render_panel(i: int, mode: str, sweep: float) -> None:
    trial = trials[i]
    picks = [dataset.ch_names.index(c) for c in D.DISPLAY_CHANNELS if c in dataset.ch_names]
    wave = dataset.X[trial, picks][:, ::2]
    wave = np.clip(wave / (3 * wave.std() + 1e-9), -1, 1)
    path = cursor_path()
    payload = {
        "trial": i + 1, "total": n, "where": f"{prefix} {subject}, run {run}", "mode": mode,
        "durationMs": int(sweep * 1000), "threshold": threshold,
        "signals": np.round(wave, 3).tolist(), "channels": [dataset.ch_names[c] for c in picks],
        "decision": decision_for(i), "truth": D.LABELS[int(dataset.y[trial])],
        "cursorBefore": path[i], "cursorAfter": path[i + 1],
        "idle": "Signal only. Train a model to see predictions.",
    }
    st.iframe(PANEL.replace("/*PAYLOAD*/", json.dumps(payload)), height="content")


def replay_score(upto: int) -> str | None:
    decisions = [decision_for(i) for i in range(upto + 1)]
    if not decisions or decisions[0] is None:
        return None
    accepted = [i for i, d in enumerate(decisions) if d["accepted"]]
    right = sum(decisions[i]["side"] == D.LABELS[int(dataset.y[trials[i]])] for i in accepted)
    text = f"So far: {len(accepted)} accepted, {len(decisions) - len(accepted)} abstained"
    return text + (f", {right} of {len(accepted)} accepted commands match the true label." if accepted else ".")


def waveform_chart(trial: int) -> go.Figure:
    fig = go.Figure()
    picks = [c for c in D.DISPLAY_CHANNELS if c in dataset.ch_names]
    step = 2.2 * max(float(np.abs(dataset.X[trial, dataset.ch_names.index(c)]).max()) for c in picks)
    for lane, name in enumerate(picks):
        y = dataset.X[trial, dataset.ch_names.index(name)]
        fig.add_scatter(x=dataset.times, y=y - lane * step, mode="lines", line=dict(color=COBALT, width=1.3),
                        customdata=y, name=name, hovertemplate=name + ": %{customdata:.1f} µV at %{x:.2f} s<extra></extra>")
    fig.update_xaxes(title="Seconds after cue", showgrid=False, linecolor=LINE)
    fig.update_yaxes(tickvals=[-i * step for i in range(len(picks))], ticktext=picks, showgrid=False, zeroline=False)
    return base_layout(fig, 250)


def probability_chart(i: int) -> go.Figure:
    p = float(p_right[i])
    fig = go.Figure(go.Bar(x=[1 - p, p], y=[f"Left {1 - p:.0%}", f"Right {p:.0%}"], orientation="h",
                           marker=dict(color=[COBALT, TEAL], cornerradius=4), width=0.5,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.add_vline(x=threshold, line=dict(color=INK, width=1.5, dash="dot"),
                  annotation_text=f"Threshold {threshold:.0%}", annotation_position="top left", annotation_font_size=12)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor="#E7F0FC", zeroline=False, title="Probability")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return base_layout(fig, 250)


def saliency_chart(sal: np.ndarray) -> go.Figure:
    top = np.argsort(sal.mean(axis=1))[::-1][:14]
    top = top[np.argsort(sal[top].mean(axis=1))]
    edges = np.linspace(D.EPOCH[0], D.EPOCH[1], sal.shape[1] + 1)
    fig = go.Figure(go.Heatmap(z=sal[top], x=(edges[:-1] + edges[1:]) / 2, y=[dataset.ch_names[c] for c in top],
                               colorscale=BLUES, zmin=0, zmax=1, xgap=1, ygap=1,
                               colorbar=dict(tickvals=[0, 1], ticktext=["Low", "High"], thickness=10, outlinewidth=0, len=.8),
                               hovertemplate="%{y} at %{x:.2f} s: %{z:.2f}<extra></extra>"))
    fig.update_xaxes(title="Seconds after cue", showgrid=False)
    fig.update_yaxes(showgrid=False, type="category", tickfont=dict(size=11))
    return base_layout(fig, 300)


# The replay advances inside a fragment, so only this part of the page redraws.
sweep_seconds = SPEEDS[speed]


@st.fragment(run_every=(sweep_seconds + 1.0) if ss.playing else None)
def live() -> None:
    if ss.playing:
        if ss.started:
            if ss.idx + 1 < n:
                ss.idx += 1
            else:
                ss.update(playing=False, revealed=True)
                st.rerun(scope="app")
        ss.started = True

    i = ss.idx
    animate = ss.playing
    render_panel(i, "animate" if animate else ("show" if ss.revealed else "ready"), sweep_seconds)

    shown = i - 1 if animate else i  # the analysis must not give away the trial still being read
    if not ss.revealed or shown < 0:
        st.caption("Analysis appears once a trial has been read.")
        return
    score = replay_score(shown)
    if score:
        st.caption(score)

    trial = int(trials[shown])
    st.markdown(f"### Trial {shown + 1} analysis")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**EEG waveform**" if real else "**Synthetic waveform**")
        st.plotly_chart(waveform_chart(trial), width="stretch", config=CHART_CONFIG, key="wave")
    with right, st.container(border=True):
        st.markdown("**Class probability**")
        if p_right is None:
            st.caption("Not available until a model is trained.")
        else:
            st.plotly_chart(probability_chart(shown), width="stretch", config=CHART_CONFIG, key="prob")

    with st.container(border=True):
        st.markdown("**Channel and time attribution**")
        sal = get_saliency(cfg, stamp, model_key, int(subject), int(run), trial) if model_key else None
        if sal is None:
            st.caption("Not available until a model is trained.")
        else:
            st.plotly_chart(saliency_chart(sal), width="stretch", config=CHART_CONFIG, key="sal")
            st.caption("Gradient attribution for the 14 highest-scoring channels: where the model's score was "
                       "most sensitive. It describes the model, not what caused the brain signal.")


live()

# Evaluation ----------------------------------------------------------------------------
st.markdown("### Evaluation")
if not results:
    st.info("Not evaluated yet.")
else:
    if not real:
        st.warning("These scores come from simulated signals. They test the software and say nothing "
                   "about performance on real EEG.")
    st.caption(f"{results['n_trials']} trials from {len(results['subjects'])} participants, runs "
               f"{', '.join(map(str, results['runs']))}. Evaluated {results['created']}. Chance is 50%.")
    if results["annotations"]:
        with st.expander("Annotation check for each recording"):
            st.caption("Counts read from the loaded files. T1 is left and T2 is right in these runs.")
            st.dataframe(pd.DataFrame(results["annotations"]), hide_index=True, width="stretch")
    tab_within, tab_loso, tab_cal = st.tabs(["Held-out run", "Held-out participant", "Uncertainty"])

    def comparison(scheme: dict) -> None:
        rows = [{"Model": M.MODEL_NAMES[m], "Balanced accuracy": r["balanced_accuracy"], "Macro F1": r["macro_f1"],
                 "Accuracy 95% interval": f"{r['accuracy_ci95'][0]:.0%} to {r['accuracy_ci95'][1]:.0%}",
                 "Across folds": f"{r['fold_mean']:.2f} ± {r['fold_std']:.2f}",
                 "Training score (not held out)": r["train_balanced_accuracy"]} for m, r in scheme.items()]
        st.dataframe(pd.DataFrame(rows).style.format(precision=3, subset=["Balanced accuracy", "Macro F1", "Training score (not held out)"]),
                     hide_index=True, width="stretch")
        names = list(scheme)
        pick = st.radio("Show details for", names, format_func=M.MODEL_NAMES.get, horizontal=True,
                        key=f"detail_{scheme is results['loso']}")
        r = scheme[pick]
        a, b, c = st.columns([1, 1.15, 1])
        with a:
            st.markdown("**Confusion matrix**")
            z = np.array(r["confusion"])
            fig = go.Figure(go.Heatmap(z=z, x=[f"Predicted {l.lower()}" for l in D.LABELS], y=[f"True {l.lower()}" for l in D.LABELS],
                                       colorscale=BLUES, showscale=False, text=z, xgap=2, ygap=2,
                                       hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
            for yi in range(2):
                for xi in range(2):
                    fig.add_annotation(x=xi, y=yi, text=str(z[yi, xi]), showarrow=False,
                                       font=dict(size=20, color="#fff" if z[yi, xi] > z.max() * .6 else INK))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(base_layout(fig, 240), width="stretch", config=CHART_CONFIG, key=f"cm_{scheme is results['loso']}_{pick}")
        with b:
            st.markdown("**Per class**")
            st.dataframe(pd.DataFrame(r["per_class"]).T.rename(columns=str.capitalize).style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Support": "{:.0f}"}), width="stretch")
        with c:
            st.markdown("**Per participant**")
            st.dataframe(pd.DataFrame({"Participant": [f"P{k}" for k in r["per_participant"]],
                                       "Balanced accuracy": list(r["per_participant"].values())}).style.format(precision=3, subset=["Balanced accuracy"]),
                         hide_index=True, width="stretch")

    with tab_within:
        st.caption("Each model trains on one participant's other runs and is tested on the run it never saw. "
                   "Replay uses exactly these held-out predictions.")
        comparison(results["within"])
    with tab_loso:
        if results["loso"]:
            st.caption("Each model trains on the other participants and is tested on someone it never saw.")
            if results["loso_note"]:
                st.warning(results["loso_note"])
            comparison(results["loso"])
        else:
            st.info(results["loso_note"])
    with tab_cal:
        st.caption("Held-out-run predictions only. Raising the threshold sends fewer commands; a useful "
                   "confidence score makes the commands that remain more accurate.")
        if model_key:
            r = results["within"][model_key]
            st.markdown(f"**{M.MODEL_NAMES[model_key]}**: expected calibration error {r['ece']:.3f}")
            rows = [s for s in r["selective"] if s["accuracy"] is not None]
            x = [s["threshold"] for s in rows]
            fig = go.Figure()
            fig.add_scatter(x=x, y=[s["accuracy"] for s in rows], mode="lines+markers", name="Accuracy of accepted commands",
                            line=dict(color=COBALT, width=2), marker=dict(size=8))
            fig.add_scatter(x=x, y=[s["coverage"] for s in rows], mode="lines+markers", name="Share of trials accepted",
                            line=dict(color=TEAL, width=2, dash="dash"), marker=dict(size=8, symbol="square"))
            fig.add_vline(x=threshold, line=dict(color=INK, width=1.5, dash="dot"))
            fig.update_xaxes(title="Confidence threshold", tickformat=".0%", showgrid=False)
            fig.update_yaxes(range=[0, 1.05], tickformat=".0%", gridcolor="#E7F0FC", zeroline=False)
            base_layout(fig, 280)
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14, x=0))
            st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, key="selective")

# About ---------------------------------------------------------------------------------
st.markdown("### About and limitations")
with st.container(border=True):
    a, b = st.columns(2)
    with a:
        st.markdown("""
**What this shows.** Recorded trials, or simulated ones in demo mode, are replayed, classified as imagined left or right hand movement,
and turned into an on-screen cursor command. No headset is connected, nothing is read from you,
and no real device is controlled.

**Models.** A baseline combines common spatial patterns with logistic regression. EEGNet is a compact
convolutional network with temporal and spatial filters. Both use 7-30 Hz signals from 0.5 to 3.5 s after the cue.

**Evaluation.** Whole runs or whole participants are held out, never random trials. Replayed trials
are always scored by a model that did not train on that run.
""")
    with b:
        st.markdown("""
**Limitations.** This is a research demo, not a medical device or a diagnostic tool. Each run has only
about 15 trials, so scores on small subsets are noisy. Performance varies widely between people.
Attributions describe the model and are not causal explanations.

**Data.** Schalk G, et al. BCI2000: a general-purpose brain-computer interface system.
*IEEE Trans Biomed Eng* 51(6), 2004. Distributed by PhysioNet: Goldberger AL, et al. *Circulation* 101(23), 2000. The Kaggle option uses a community copy of these recordings.

**Methods and software.** Lawhern VJ, et al. EEGNet. *J Neural Eng* 15(5), 2018.
Gramfort A, et al. MNE-Python. *Front Neurosci* 7, 2013.
""")
