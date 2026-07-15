# ASDMotion — SMM Detection (1D-CNN + Live Demo)

Detection of stereotypical motor movements (SMMs) in children with ASD from body-pose
motion. A lightweight 1D-CNN classifies short windows of per-joint speed features
extracted from pose keypoints. The repo covers the full path from the annotated dataset
to a live webcam application: **data prep → training → temporal-detection test → live app**.

The live app estimates pose with MoveNet, feeds the same speed features the model was
trained on into the 1D-CNN, and reports SMM probability, a rolling alert rate, and a
per-session summary.

---

## What's in the repo

| File | Purpose |
|---|---|
| `asdmotion_prep.ipynb` | Turns the annotated pickle into windowed feature arrays for training. |
| `asdmotion_train_v1_tensorflow.ipynb` | Trains the model in three stages (initial → improved → tuned), evaluates, and saves the winner. |
| `Temporal_Detection_test.ipynb` | Evaluates the trained model on continuous/temporal data rather than isolated windows. |
| `smm_backend.py` | The detection pipeline: pose, features, model, alerter, episode logic, session logging. No UI. |
| `smm_app.py` | Desktop front end (PySide6). Start/stop live detection and view a session report. |

### Not in the repo (generated or external)

Everything below is `.gitignore`d because it's either raw input or produced by the code
above. A fresh clone **builds** but is not **runnable** until these exist:

| Path | Where it comes from |
|---|---|
| `dataset.pkl` | The ASDPose annotated dataset (external — [download](https://drive.google.com/file/d/13t1tO4ZxTKmQG-w6fTy3hHQy8gX1bopl/view?usp=sharing), see step 1). |
| `prepared/` | Written by the prep and training notebooks (feature arrays, `v1_model.keras`, `scaler.json`, `v1_meta.json`). |
| `prepared/*.tflite` | Converted from the trained model + downloaded MoveNet (see setup). |
| `kt_dir/` | Keras-tuner search state, regenerated when training runs. |
| `sessions/` | Created by the app each time a session is saved. |

---

## Requirements

- Python 3.10+
- A GPU is recommended for training but not required. The live app runs fine on CPU.
- Linux notes: the app is a Qt application. On Wayland run it with the Wayland platform
  plugin (see [Run the live app](#4-run-the-live-app)).

### Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Core
pip install numpy opencv-python

# Training / notebooks
pip install "tensorflow[and-cuda]==2.20"   # CPU-only: pip install tensorflow==2.20
pip install keras-tuner scikit-learn matplotlib jupyter

# Live app
pip install PySide6

# TFLite runtime for the app (one of these; tensorflow already provides tf.lite)
pip install ai-edge-litert                 # or rely on the installed tensorflow
```

> **Python 3.13 note.** The notebooks run on 3.13, but TensorFlow's model-export path has
> a `wrapt` incompatibility there that surfaces only at the TFLite conversion step. The
> conversion recipe in [step 3](#3-produce-the-files-the-live-app-needs) is written to
> avoid it; if you hit it elsewhere, `pip install "wrapt==1.14.1"` is the known-good pin.

> **GPU note.** `tensorflow[and-cuda]==2.20` bundles the CUDA libraries via pip — no
> system CUDA install needed. Version 2.21 has a known GPU-detection regression; pin to
> 2.20. Verify with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

---

## Setup and run

The pipeline is sequential. To run the live app you must first produce the model
artifacts, because the notebooks are committed but their outputs are not.

### 1. Prepare the dataset

This project trains on the **ASDPose annotated dataset** from the Dinstein-Lab ASDMotion
project ([github.com/Dinstein-Lab/ASDMotion](https://github.com/Dinstein-Lab/ASDMotion)) —
skeletal (pose) representations of children with ASD, with 7,352 manually annotated SMM
segments. Download the **[Annotated Dataset for Training](https://drive.google.com/file/d/13t1tO4ZxTKmQG-w6fTy3hHQy8gX1bopl/view?usp=sharing)**
(a `.pkl` file) from that repo's *Downloads* section, save it as `dataset.pkl` in the repo
root, then run `asdmotion_prep.ipynb` top to bottom.

Key prep settings (top of the notebook):

- `WINDOW = 90` — frames per sample (3 s at 30 fps)
- `STRIDE_TRAIN = 30` — overlapping train windows (acts as augmentation)
- `STRIDE_TEST = 90` — non-overlapping test windows (honest evaluation)
- `MAX_GAP = 4` — interpolate missing-keypoint runs up to this length

This writes to `prepared/`: `X_{split}.npy`, `y_{split}.npy`, `cat_{split}.npy`,
`scaler.json`, and split metadata.

> **The scaler is the contract between offline and live.** `scaler.json` holds the
> speed median/IQR/cap computed here. The live app applies the exact same normalization.
> If you re-run prep, the scaler changes and the model must be retrained to match.

### 2. Train the model

Run `asdmotion_train_v1_tensorflow.ipynb`. It trains three stages and keeps the best:

- **initial** — a plain CNN, shown overfitting (no regularization).
- **improved** — adds BatchNorm, dropout, AdamW weight decay, label smoothing, augmentation.
- **tuned** — a Keras-tuner Bayesian search over kernel size, dropout, learning rate,
  weight decay, and augmentation strength (filter widths fixed).

The winner (best validation PR-AUC) is saved as `prepared/v1_model.keras`, and the
decision threshold that meets the target recall is written to `prepared/v1_meta.json`.
A multi-seed cell reports mean ± std so results aren't a single lucky run.

### 3. Produce the files the live app needs

The app loads **TFLite** models, not `.keras`. Two artifacts must be added to `prepared/`.

- **Convert the trained model to TFLite** → `prepared/v1_model.tflite`. The training
  notebook does this in its final cells. On TensorFlow 2.20 with Python 3.13, the usual
  `from_keras_model(...)` and `.export()` paths both fail (see
  [Troubleshooting](#troubleshooting)); the reliable route is a **concrete-function**
  conversion, which sidesteps both bugs:

  ```python
  import tensorflow as tf

  @tf.function(input_signature=[tf.TensorSpec([1, L, C], tf.float32)])  # L=90, C=20
  def serve(x):
      return final_model(x, training=False)

  concrete = serve.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], final_model)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
      tf.lite.OpsSet.SELECT_TF_OPS,   # fallback for any unconverted op
  ]
  tflite = converter.convert()
  open("prepared/v1_model.tflite", "wb").write(tflite)
  ```

  Then verify the TFLite model matches Keras — the notebook's last cell prints
  `max abs diff`, which should be well under `1e-3` (a value around `1e-4` is normal
  float32 drift and is fine; the live app will behave identically to evaluation).

- **Download MoveNet SinglePose Lightning (TFLite)** → `prepared/movenet.tflite`
  (the app defaults to this path; Thunder also works and is less jittery).

After this, `prepared/` should contain at least:

```
prepared/
├── v1_model.tflite     # your trained SMM classifier
├── movenet.tflite      # pose estimator
├── scaler.json         # speed normalization (from prep)
└── v1_meta.json        # decision threshold + model metadata (from training)
```

### 4. Run the live app

```bash
python smm_app.py
# On Wayland (e.g. niri, GNOME Wayland):
QT_QPA_PLATFORM=wayland python smm_app.py
```

The start screen shows an artifact checklist — the **Start** button stays disabled until
all four files above are found, so a green checklist means you're ready.

---

## Using the app

**Start screen.** Pick a camera and set the detection parameters, then start:

- **Decision threshold** — probability above which a window counts as an SMM. Pre-filled
  from `v1_meta.json`. **Set this per person before starting** (see calibration note below).
- **Keypoint confidence** — pose keypoints below this score are treated as missing.
- **Alert level (enter)** / **Confirm level** — control the *rolling rate* alarm, not the
  per-window decision. `enter` is the fraction of recent windows that must be flagged to
  raise an alert; `confirm` is the level above which an alert is trusted rather than
  treated as a transient.

**Running screen.** Live webcam with a pose bounding box (green = calm, red = alert),
the current SMM probability, a rolling-rate meter with the enter/confirm levels marked,
a 60-second score plot, and duration/episode counters. Press **Stop** (or `Esc`) to end.

**Report screen.** Session summary — duration, episodes, SMMs per minute, % time in SMM,
median episode length, peak rate, time alerting. **Save session** appends one row to
`sessions/log.csv` and writes the full score trace to `sessions/<id>_trace.json`.
The CSV is append-only; past sessions are never overwritten.

> **Calibrate the threshold per setup.** The threshold saved during training was chosen
> on the evaluation data (OpenPose keypoints). The live app uses MoveNet, whose keypoint
> jitter differs, and normalized speed is inflated at greater camera distance. A still
> person can therefore score above a threshold tuned offline. Practical fix: have the
> person stand/sit still briefly, watch the probability, and set the decision threshold
> above their still-baseline before starting. Different people (distance, body size) may
> need different thresholds.

---

## Architecture

The code splits along one line: **anything that would still make sense with no screen
attached lives in the backend.**

```
                 ┌──────────────────── smm_app.py (UI only) ────────────────────┐
   webcam ──▶ InferenceWorker (QThread) ──▶ widgets: video, meters, plots, report
                 └───────────────┬───────────────────────────────────────────────┘
                                 │ calls
                 ┌───────────────▼──────────── smm_backend.py (no UI) ───────────┐
   frame ──▶ SmmEngine.process() ──▶ Result(prob, rate, alert, state, keypoints)
              │  MoveNet → StreamingFeatureProcessor (90×20 window) → 1D-CNN → Alerter
              └─ SessionRecorder: episodes, summary stats, log.csv + trace JSON
```

- **`SmmEngine`** — one frame in, one `Result` out. Camera-agnostic: a webcam, a video
  file, or synthetic frames all work, which is what makes the backend testable.
- **`SessionRecorder`** — accumulates scored windows and derives the session summary. The
  metrics (SMMs/min, % time with SMMs, median episode length) mirror the ones reported
  per child in the ASDMotion paper, so a live session is comparable to offline evaluation.
- **Feature parity is critical.** `StreamingFeatureProcessor` in the backend reproduces
  the offline prep exactly (shoulder-width scale, short-gap hold, speed, robust scaling,
  clipped confidence). Do not change it without re-running prep and retraining.

Quick headless check of the pipeline (no camera, no UI):

```bash
python smm_backend.py --selftest
```

---

## Troubleshooting

- **Start button disabled.** One of the four artifacts is missing; the checklist shows
  which. Complete steps 1–3.
- **`list_physical_devices('GPU')` returns `[]`.** Use `tensorflow[and-cuda]==2.20`
  (not 2.21), and make sure Jupyter runs from the same environment you installed into
  (`import sys; sys.executable`).
- **TFLite conversion: `from_keras_model(...)` raises `'NoneType' object is not callable`.**
  TF 2.20 ships Keras 3, whose internals the TFLite Keras path no longer supports. Use the
  concrete-function conversion in [step 3](#3-produce-the-files-the-live-app-needs) instead.
- **TFLite conversion: `.export()` / `saved_model.save` raises
  `this __dict__ descriptor does not support '_DictWrapper' objects`.** A `wrapt` /
  Python 3.13 incompatibility in TensorFlow's SavedModel tracking. Either pin
  `wrapt==1.14.1`, or (recommended) use the concrete-function conversion, which avoids the
  SavedModel path entirely.
- **`tf.lite.Interpreter is deprecated` warning.** Harmless. The runtime moved to
  `ai_edge_litert`; the app already prefers it, and the warning doesn't affect output.
- **Window won't go fullscreen / app won't launch on Wayland.** Run with
  `QT_QPA_PLATFORM=wayland`. A tiling compositor may manage window size itself.
- **A still person reads as SMM.** Expected domain shift (MoveNet vs the training pose
  estimator, and distance-dependent speed inflation). Raise the decision threshold for
  that person, raise keypoint confidence, or use MoveNet Thunder. See the calibration note.
- **Training loss keeps dropping but validation plateaus.** That's the expected, healthy
  picture after regularization — select the early-stopping epoch and don't over-tune.
---

## Dataset & attribution

The training data (`dataset.pkl`) and the SMM detection task come from the **ASDMotion /
ASDPose** project by the Dinstein Lab:

- Repository: <https://github.com/Dinstein-Lab/ASDMotion> (MIT license)
- Dataset download: [Annotated Dataset for Training](https://drive.google.com/file/d/13t1tO4ZxTKmQG-w6fTy3hHQy8gX1bopl/view?usp=sharing)
- Paper: Barami T., Manelis-Baram L., Kaiser H., Ilan M., et al. *Automated identification
  and quantification of stereotypical movements from video recordings of children with
  ASD.* 2024.

This project is an **independent reimplementation** with a different, lighter approach: it
trains a 1D-CNN on per-joint **speed** features rather than the paper's PoseConv3D model on
3D pose heatmaps, and runs live inference with MoveNet instead of OpenPose. It is not
affiliated with or endorsed by the original authors. If you use the dataset, honor the
original project's license and terms and cite the paper:

```bibtex
@article{Barami2024,
  author  = {Tal Barami and Liora Manelis-Baram and Hadas Kaiser and Michal Ilan and others},
  title   = {Automated identification and quantification of stereotypical movements
             from video recordings of children with ASD},
  year    = {2024},
  doi     = {10.1101/2024.03.02.582828},
  journal = {bioRxiv}
}
```

The dataset contains clinical data from children with ASD; use it for research under the
terms set by the original authors.