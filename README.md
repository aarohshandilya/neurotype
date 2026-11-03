# NeuroType: Control a Computer with Brain Signals

Aaroh Shandilya, 2026. Academic machine-learning demo.

NeuroType replays recorded EEG trials, classifies each one as imagined left-hand or right-hand
movement, and turns confident predictions into simple on-screen cursor commands.

It is a research-data replay. No EEG headset is connected, it does not read anyone's thoughts, it
diagnoses nothing, and it controls no real device.

## Setup and run

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in synthetic demo mode with no download. Choose **Train and evaluate** (about a
minute) to get predictions and metrics.

### Using real EEG

The default real source is a Kaggle copy of the PhysioNet recordings:
[gamalasran/physionet-eeg-motor-movement-imagery](https://www.kaggle.com/datasets/gamalasran/physionet-eeg-motor-movement-imagery).
PhysioNet remains available as a second option.

1. In the app, set **Data source** to **Recorded EEG from Kaggle**.
2. Choose **Download and prepare data**. Kaggle delivers the whole dataset, roughly 2 GB, once.
   If Kaggle asks you to sign in, create an API token in your Kaggle settings, or download the
   dataset yourself and unzip it into `data/kaggle/` (any folder depth works).
3. Choose **Train and evaluate**. This took about 8 minutes on a two-core CPU with 64 channels.

Or from a terminal:

```bash
python model.py --source kaggle --subjects 1 2 3 4 5 --runs 4 8 12
```

The Kaggle upload is a community copy, not published by PhysioNet. Each file must pass the same
checks as a direct download: T0/T1/T2 annotations, 160 Hz sampling and a matching channel layout.
Check its Kaggle page for the licence, and cite the original authors below.

Without torch installed the app still runs with the CSP baseline; add `--no-deep` to the command
above to skip EEGNet on purpose. Downloads go to `data/`, trained models and metrics to
`artifacts/`. Both are git-ignored.

## Website

`website/index.html` is an interactive version that needs no install: open it in a browser.
It simulates signals and trains a small CSP classifier in the browser, so held-out scores are
computed live. To show real results, run the commands below, then drop the file onto the page.

```bash
python model.py --source kaggle
python export_web.py --source kaggle     # writes website/replay.json
```

After editing `website/template.html` or `engine.js`, rebuild with `python website/build.py`.

## How it works

**Data.** Runs 4, 8 and 12 are imagined left/right hand movement. Each file's annotations are
checked on load: T0 is rest, T1 is left, T2 is right, and a run without T1 and T2 is rejected.
The app stores the per-run annotation counts with the results. T1 and T2 mean fists and feet in
other runs, so those runs are refused. Runs 3, 7 and 11 (executed movement) are accepted with a
warning. Participants 88, 89, 92, 100 and 104 are excluded for recording irregularities.

**Preprocessing.** 7-30 Hz band-pass within each continuous run, then epochs from 0.5 to 3.5 s
after the cue, 160 Hz, no resampling.

**Models.**
- Baseline: common spatial patterns (6 components, Ledoit-Wolf regularisation), standardised log
  power, logistic regression.
- EEGNet: compact temporal and spatial CNN, 60 fixed epochs, seeded. No LSTM was added: with so
  few trials it would add fragility without clear benefit.

**Evaluation.** Nothing is split at random.
- Held-out run: for each participant, train on their other runs and test on the unseen run.
- Held-out participant: train on everyone else and test on the unseen person. With fewer than 10
  participants the app and results say this is only a rough indication.
- Reported: balanced accuracy, macro F1, confusion matrix, per-class precision, recall and F1,
  per-participant scores, and a 95% Wilson interval. Training scores appear in their own column.
- Each fold asserts that train and test share no run or participant. Scaling is per trial or
  fitted on training data only, and the epoch count is fixed, so test data cannot steer training.

**Replay.** Every replayed prediction and attribution comes from the fold model that never saw
that run. The true label is displayed separately and never affects the command. Below the
confidence threshold the model abstains and no command is sent. The whole 3 s window is
classified at once, so the decision appears when the sweep ends.

**Uncertainty.** The Uncertainty tab shows expected calibration error, and accuracy against
coverage across thresholds, using held-out-run predictions only.

**Attribution.** The heatmap shows the size of the gradient of the model's score with respect to
each channel and time point: exact for the baseline, backpropagated for EEGNet. These are model
attributions, not causal explanations.

## Limitations

- This is not a medical device and is for research and teaching only.
- Each run has about 15 trials, so scores on small subsets are noisy, and performance varies
  widely between people. A few participants say little about the wider population.
- Replay is offline. A working brain-computer interface also has to cope with live artefacts,
  feedback, fatigue and timing.
- Synthetic mode is simulated. Its scores test the software and say nothing about real EEG.
- Neural-network probabilities are often poorly calibrated. No recalibration step is applied.
- BCI Competition IV 2a is not implemented. It needs a manual download and is a future extension.

## Project structure

```
app.py               Streamlit interface
panel.html           Replay panel: command cards, cursor, signal sweep
export_web.py        Exports held-out results for the website
website/             Interactive website: page, browser engine, build script
data.py              Kaggle and PhysioNet loading, annotation checks, preprocessing, synthetic signals
model.py             CSP, EEGNet, evaluation, metrics, attribution, command-line training
requirements.txt     Python dependencies
.streamlit/          Theme
```

## Data and credits

Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: a general-purpose
brain-computer interface (BCI) system. IEEE Trans Biomed Eng. 2004;51(6):1034-1043.

Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research
resource for complex physiologic signals. Circulation. 2000;101(23):e215-e220.

Lawhern VJ, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer
interfaces. J Neural Eng. 2018;15(5):056013.

Gramfort A, et al. MEG and EEG data analysis with MNE-Python. Front Neurosci. 2013;7:267.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
