#!/usr/bin/env python3
"""
smm_backend.py — SMM detection pipeline. No UI.

Everything here would still make sense with no screen attached: pose estimation,
streaming features, the CNN, the rolling-rate alerter, episode detection, session
statistics, and persistence. Nothing in this module imports Qt or draws anything.

Public surface:
    Config              — paths and tunables, one place to change them
    SmmEngine           — feed it frames, it returns Result objects
    Result              — one frame's worth of output
    SessionRecorder     — accumulates Results, computes stats, saves to disk
    find_episodes       — contiguous flagged windows -> discrete episodes
    check_artifacts     — which model files are present

Self-test (no camera, synthetic keypoints — exercises the pipeline end to end):
    python smm_backend.py --selftest

Deps: numpy, opencv-python, and tflite-runtime OR tensorflow
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

import numpy as np

# tflite interpreter: tflite_runtime if present, else ai_edge_litert, else full TF
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except Exception:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


# ── configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    movenet: str = os.path.join("prepared", "movenet.tflite")
    cnn: str = os.path.join("prepared", "v1_model.tflite")
    scaler: str = os.path.join("prepared", "scaler.json")
    meta: str = os.path.join("prepared", "v1_meta.json")
    sessions_dir: str = "sessions"

    cam: int = 0
    conf_thresh: float = 0.20      # keypoint score below this is treated as missing
    window: int = 90               # frames per model input (3 s @ 30 fps)
    max_gap: int = 4               # frames a keypoint may be held through

    threshold: float | None = None  # None -> read from meta
    roll_seconds: float = 20.0
    enter: float = 0.50
    confirm: float = 0.65
    hold_seconds: float = 1.0

    # episode detection: contiguous flagged windows -> one SMM event
    min_windows_open: int = 2      # consecutive flagged windows needed to open
    max_gap_close: int = 1         # gaps this small do not split an episode

    @property
    def log_csv(self) -> str:
        return os.path.join(self.sessions_dir, "log.csv")

    def artifacts(self) -> dict[str, str]:
        return {"MoveNet": self.movenet, "SMM model": self.cnn,
                "Scaler": self.scaler, "Model metadata": self.meta}


def check_artifacts(cfg: Config) -> list[tuple[str, str, bool]]:
    """[(label, path, exists)] — lets a caller show a readiness checklist."""
    return [(k, v, os.path.exists(v)) for k, v in cfg.artifacts().items()]


def load_threshold(cfg: Config, default: float = 0.118) -> float:
    if cfg.threshold is not None:
        return cfg.threshold
    try:
        return float(json.load(open(cfg.meta))["threshold"])
    except Exception:
        return default


# ── pose + features ──────────────────────────────────────────────────────────
# COCO-17 indices kept by the model (ears, shoulders, elbows, wrists, hips)
KEEP = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LSH, RSH = 2, 3   # left/right shoulder positions WITHIN the kept array


class TFLiteModel:
    def __init__(self, path):
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def __call__(self, x):
        self.it.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"])


class MoveNet:
    """SinglePose Lightning TFLite. Letterboxes to 192x192, returns (17,3)
    [y, x, score] in [0,1] of the *original* frame."""

    def __init__(self, path, size=192):
        self.m = TFLiteModel(path)
        self.size = size

    def _letterbox(self, img):
        import cv2
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        rz = cv2.resize(img, (nw, nh))
        canvas = np.zeros((self.size, self.size, 3), np.uint8)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[py:py + nh, px:px + nw] = rz
        return canvas, s, px, py

    def __call__(self, frame_rgb):
        canvas, s, px, py = self._letterbox(frame_rgb)
        x = canvas[None].astype(self.m.inp["dtype"])
        out = self.m(x).reshape(17, 3)
        h, w = frame_rgb.shape[:2]
        kp = out.copy()
        kp[:, 0] = (out[:, 0] * self.size - py) / s / h
        kp[:, 1] = (out[:, 1] * self.size - px) / s / w
        return kp


class StreamingFeatureProcessor:
    """Per-frame keypoints -> the (90, 20) window the CNN expects.

    Mirrors the offline prep exactly: rolling shoulder-width scale, short-gap
    hold, timestamped frame-to-frame speed, clip at p99.5 then (x - median) / IQR,
    clipped confidence. Do not change this without re-running prep.
    """

    def __init__(self, scaler, window=90, conf_thresh=0.2, max_gap=4, sw_buf=30):
        self.MED = np.array(scaler["speed_median"], np.float32)
        self.IQR = np.array(scaler["speed_iqr"], np.float32)
        self.CAP = np.array(scaler["speed_cap"], np.float32)
        self.window, self.conf_thresh, self.max_gap = window, conf_thresh, max_gap
        self.ring = deque(maxlen=window)
        self.sw = deque(maxlen=sw_buf)
        self.prev_pos = None
        self.prev_valid = None
        self.prev_t = None
        self.last_pos = np.zeros((10, 2), np.float32)
        self.age = np.full(10, 999)

    @property
    def warming(self) -> bool:
        return len(self.ring) < self.window

    @property
    def fill(self) -> float:
        return len(self.ring) / self.window

    def update(self, kps, t):
        """kps: (17,3) [y, x, score] in [0,1]. t: timestamp (s).
        Returns the (90,20) window once the ring is full, else None."""
        pos = kps[KEEP, :2].astype(np.float32).copy()
        score = kps[KEEP, 2].astype(np.float32)
        valid = score >= self.conf_thresh

        for k in range(10):                       # streaming short-gap hold
            if valid[k]:
                self.last_pos[k] = pos[k]
                self.age[k] = 0
            elif self.age[k] < self.max_gap:
                pos[k] = self.last_pos[k]
                valid[k] = True
                self.age[k] += 1
            else:
                self.age[k] += 1

        if valid[LSH] and valid[RSH]:             # rolling shoulder-width scale
            self.sw.append(float(np.linalg.norm(pos[LSH] - pos[RSH])))
        scale = float(np.median(self.sw)) if self.sw else None

        if scale and scale > 1e-6:
            Pn = pos / scale
            speed = np.zeros(10, np.float32)
            if self.prev_pos is not None and self.prev_t is not None:
                dt = max(t - self.prev_t, 1e-3)
                both = valid & self.prev_valid
                disp = np.linalg.norm(Pn - self.prev_pos / scale, axis=1)
                speed = np.where(both, disp / dt, 0.0).astype(np.float32)
            sp_scaled = (np.minimum(speed, self.CAP) - self.MED) / self.IQR
            conf = np.clip(score, 0.0, 1.0)
            self.ring.append(np.concatenate([sp_scaled, conf]).astype(np.float32))

        self.prev_pos = pos.copy()
        self.prev_valid = valid.copy()
        self.prev_t = t

        if len(self.ring) == self.window:
            return np.stack(self.ring).astype(np.float32)
        return None


# ── rolling-rate alerter ─────────────────────────────────────────────────────
class Alerter:
    """Rolling SMM-rate with adaptive decay to clear lingering transient crossings.

    raw_rate = fraction of recent windows (over roll_seconds) with prob >= threshold.
    The effective rate follows raw_rate, except:
      - a crossing of `enter` that does NOT reach `confirm` within `hold_seconds`
        is a transient -> stepped below `enter` and suppressed until raw_rate
        itself falls back under `enter`;
      - a crossing that reaches `confirm` is trusted -> natural (slow) decay.
    """

    CALM, TENTATIVE, CONFIRMED, SUPPRESSED = "calm", "tentative", "confirmed", "suppressed"

    def __init__(self, threshold, roll_seconds=20, enter=0.5,
                 confirm=0.65, hold_seconds=1.0, margin=0.01):
        self.threshold, self.roll = threshold, roll_seconds
        self.enter, self.confirm = enter, confirm
        self.hold, self.margin = hold_seconds, margin
        self.hist = deque()
        self.state = self.CALM
        self.t_enter = None
        self.eff = 0.0

    def update(self, prob, t):
        self.hist.append((t, 1.0 if prob >= self.threshold else 0.0))
        while self.hist and t - self.hist[0][0] > self.roll:
            self.hist.popleft()
        raw = float(np.mean([f for _, f in self.hist])) if self.hist else 0.0

        s = self.state
        if s == self.CALM:
            self.eff = raw
            if raw >= self.enter:
                self.state = self.TENTATIVE
                self.t_enter = t
        elif s == self.TENTATIVE:
            if raw >= self.confirm:
                self.state = self.CONFIRMED
                self.eff = raw
            elif raw < self.enter:
                self.state = self.CALM
                self.eff = raw
            elif t - self.t_enter >= self.hold:
                self.state = self.SUPPRESSED
                self.eff = min(raw, self.enter - self.margin)
            else:
                self.eff = raw
        elif s == self.CONFIRMED:
            self.eff = raw
            if raw < self.enter:
                self.state = self.CALM
        elif s == self.SUPPRESSED:
            if raw >= self.confirm:
                self.state = self.CONFIRMED
                self.eff = raw
            elif raw < self.enter:
                self.state = self.CALM
                self.eff = raw
            else:
                self.eff = min(raw, self.enter - self.margin)
        return self.eff, self.eff >= self.enter


# ── engine: one frame in, one result out ─────────────────────────────────────
@dataclass
class Result:
    """Everything the caller needs to know about one processed frame."""
    t: float                 # seconds since the engine started
    keypoints: np.ndarray    # (17,3) [y, x, score], normalized to the frame
    prob: float              # SMM probability for the current window
    rate: float              # effective rolling rate
    alert: bool              # rate >= enter
    state: str               # calm | tentative | confirmed | suppressed
    warming: bool            # ring buffer not yet full
    fill: float              # 0..1 ring buffer fill
    scored: bool             # a new window was scored on this frame
    latency_ms: float


class SmmEngine:
    """Loads the models and turns frames into Results. Camera-agnostic:
    hand it RGB arrays from a webcam, a video file, or a test harness."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.threshold = load_threshold(cfg)
        scaler = json.load(open(cfg.scaler))
        self.cnn = TFLiteModel(cfg.cnn)
        self.movenet = MoveNet(cfg.movenet)
        self.proc = StreamingFeatureProcessor(
            scaler, window=cfg.window, conf_thresh=cfg.conf_thresh, max_gap=cfg.max_gap)
        self.alerter = Alerter(self.threshold, cfg.roll_seconds,
                               cfg.enter, cfg.confirm, cfg.hold_seconds)
        self.t0 = None
        self.prob = 0.0
        self.rate = 0.0
        self.alert = False

    def process(self, frame_rgb: np.ndarray, t: float | None = None) -> Result:
        t = time.time() if t is None else t
        if self.t0 is None:
            self.t0 = t
        t_start = time.perf_counter()

        kps = self.movenet(frame_rgb)
        win = self.proc.update(kps, t)
        scored = win is not None
        if scored:
            self.prob = float(self.cnn(win[None])[0, 0])
            self.rate, self.alert = self.alerter.update(self.prob, t)

        return Result(
            t=t - self.t0,
            keypoints=kps,
            prob=self.prob,
            rate=self.rate,
            alert=self.alert,
            state=self.alerter.state,
            warming=self.proc.warming,
            fill=self.proc.fill,
            scored=scored,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )


# ── episodes ─────────────────────────────────────────────────────────────────
def find_episodes(trace, min_open=2, max_gap=1):
    """Merge contiguous flagged windows into discrete episodes.

    trace: [(t, prob, flagged), ...]. An episode needs `min_open` consecutive
    flagged windows to open, and a gap of `max_gap` or fewer does not split it.
    This is the live analogue of concatenating neighbouring SMM frames into one
    movement, as the ASDMotion paper does offline.
    """
    episodes = []
    start = None
    run = 0
    gap = 0
    last_t = 0.0
    for t, _p, flagged in trace:
        if flagged:
            if start is None:
                start, run = t, 1
            else:
                run += 1
            gap = 0
            last_t = t
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if run >= min_open:
                    episodes.append((start, last_t))
                start, run, gap = None, 0, 0
    if start is not None and run >= min_open:
        episodes.append((start, trace[-1][0]))
    return episodes


# ── session recording ────────────────────────────────────────────────────────
@dataclass
class SessionRecorder:
    """Accumulates scored windows, derives session statistics, saves to disk.

    The summary metrics deliberately mirror the ones the ASDMotion paper reports
    per child (SMMs per minute, percentage of time with SMMs, median episode
    length) so a live session is directly comparable to the offline evaluation.
    """
    cfg: Config
    threshold: float
    trace: list = field(default_factory=list)        # (t, prob, flagged)
    rate_trace: list = field(default_factory=list)   # (t, rate, alert)
    fps_hist: list = field(default_factory=list)

    def add(self, r: Result, fps: float | None = None):
        if r.scored:
            self.trace.append((r.t, r.prob, bool(r.prob >= self.threshold)))
            self.rate_trace.append((r.t, r.rate, bool(r.alert)))
        if fps is not None:
            self.fps_hist.append(fps)

    def episodes(self):
        return find_episodes(self.trace, self.cfg.min_windows_open, self.cfg.max_gap_close)

    def summary(self) -> dict:
        eps = self.episodes()
        dur = self.trace[-1][0] if self.trace else 0.0
        smm_time = sum(e - s for s, e in eps)
        lengths = sorted(e - s for s, e in eps)
        median = lengths[len(lengths) // 2] if lengths else 0.0
        n_alert = sum(1 for _t, _r, a in self.rate_trace if a)
        return {
            "session_id": uuid.uuid4().hex[:10],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(dur, 1),
            "threshold": self.threshold,
            "enter_level": self.cfg.enter,
            "confirm_level": self.cfg.confirm,
            "n_windows": len(self.trace),
            "n_flagged": sum(1 for _t, _p, f in self.trace if f),
            "n_episodes": len(eps),
            "episode_time_s": round(smm_time, 1),
            "normal_time_s": round(max(dur - smm_time, 0.0), 1),
            "smms_per_min": round(len(eps) / max(dur / 60, 1e-6), 3),
            "pct_time_smm": round(100 * smm_time / max(dur, 1e-6), 2),
            "median_episode_s": round(median, 2),
            "pct_time_alerting": round(100 * n_alert / max(len(self.rate_trace), 1), 2),
            "peak_rate": round(max((r for _t, r, _a in self.rate_trace), default=0.0), 3),
            "mean_fps": round(float(np.mean(self.fps_hist)), 1) if self.fps_hist else 0.0,
        }

    def save(self, summary: dict, notes: str = "") -> tuple[str, str]:
        """Append one row to log.csv and dump the full trace. Returns both paths.
        The CSV is opened in append mode — earlier sessions are never overwritten.
        """
        os.makedirs(self.cfg.sessions_dir, exist_ok=True)
        row = dict(summary)
        row["notes"] = notes.replace("\n", " ").strip()

        first = not os.path.exists(self.cfg.log_csv)
        with open(self.cfg.log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if first:
                w.writeheader()
            w.writerow(row)


def update_notes(cfg: Config, session_id: str, notes: str) -> bool:
    """Rewrite the notes cell of one already-saved session row in place.

    Returns True if the row was found and updated. Used when the operator types
    notes on the report screen after the session has already been auto-saved.
    """
    if not os.path.exists(cfg.log_csv):
        return False
    with open(cfg.log_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "notes" not in fields:
        return False

    hit = False
    for row in rows:
        if row.get("session_id") == session_id:
            row["notes"] = notes.replace("\n", " ").strip()
            hit = True
    if hit:
        with open(cfg.log_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return hit


def read_sessions(cfg: Config) -> list[dict]:
    """Read every saved session back from log.csv, newest first.

    Reading the log is backend work — the dashboard is only a view over this.
    Returns [] when no sessions have been saved yet. Numeric columns are cast
    back to float/int so the caller doesn't have to reparse strings.
    """
    if not os.path.exists(cfg.log_csv):
        return []

    int_cols = {"n_windows", "n_flagged", "n_episodes"}
    rows = []
    with open(cfg.log_csv, newline="") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            for k, v in list(row.items()):
                if k in ("session_id", "timestamp", "notes"):
                    continue
                try:
                    row[k] = int(v) if k in int_cols else float(v)
                except (TypeError, ValueError):
                    pass  # leave unparseable values as-is
            rows.append(row)
    rows.reverse()  # newest first
    return rows


# Time ranges the dashboard can filter by. "all" means no filtering.
RANGES = ("today", "week", "month", "all")


def filter_sessions_by_range(rows: list[dict], range_key: str,
                             now: datetime | None = None) -> list[dict]:
    """Keep only sessions whose timestamp falls in the given window.

    range_key: "today" (since local midnight), "week" (last 7 days),
    "month" (last 30 days), or "all". Rows with an unparseable timestamp are
    dropped for any range other than "all".
    """
    if range_key == "all":
        return rows
    now = now or datetime.now()
    if range_key == "today":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_key == "week":
        cutoff = now - timedelta(days=7)
    elif range_key == "month":
        cutoff = now - timedelta(days=30)
    else:
        return rows

    out = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(str(r.get("timestamp", "")))
        except (TypeError, ValueError):
            continue
        if ts >= cutoff:
            out.append(r)
    return out


def episode_vs_normal_time(rows: list[dict]) -> tuple[float, float]:
    """Sum episode time and normal time across sessions (seconds).

    Falls back to deriving episode time from pct_time_smm x duration for older
    rows saved before episode_time_s existed, so a mixed log still totals right.
    """
    ep = norm = 0.0
    for r in rows:
        dur = float(r.get("duration_s", 0) or 0)
        if r.get("episode_time_s") not in (None, ""):
            e = float(r["episode_time_s"])
        else:  # legacy row: reconstruct from percentage
            e = dur * float(r.get("pct_time_smm", 0) or 0) / 100.0
        ep += e
        norm += max(dur - e, 0.0)
    return round(ep, 1), round(norm, 1)


# ── self-test ────────────────────────────────────────────────────────────────
def selftest(cfg: Config):
    """Drives the pipeline with synthetic keypoints — no camera, no models needed
    for the feature half; the CNN is exercised if the artifact is present."""
    scaler = json.load(open(cfg.scaler))
    proc = StreamingFeatureProcessor(scaler, conf_thresh=cfg.conf_thresh)
    threshold = load_threshold(cfg)
    alerter = Alerter(threshold, cfg.roll_seconds, cfg.enter, cfg.confirm, cfg.hold_seconds)
    cnn = TFLiteModel(cfg.cnn)

    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.7, (17, 2)).astype(np.float32)
    rec = SessionRecorder(cfg, threshold)

    t = 0.0
    n = 0
    for i in range(300):
        t += 1 / 30
        jitter = rng.normal(0, 0.05 if i > 100 else 0.005, (17, 2))   # calm, then active
        kps = np.concatenate([np.clip(base + jitter, 0, 1),
                              np.full((17, 1), 0.9, np.float32)], axis=1)
        win = proc.update(kps, t)
        if win is not None:
            assert win.shape == (cfg.window, 20), win.shape
            prob = float(cnn(win[None])[0, 0])
            rate, on = alerter.update(prob, t)
            rec.trace.append((t, prob, prob >= threshold))
            rec.rate_trace.append((t, rate, on))
            n += 1

    s = rec.summary()
    print(f"selftest OK — {n} windows scored, shapes valid")
    print(f"  threshold {threshold:.3f}   episodes {s['n_episodes']}   "
          f"peak rate {s['peak_rate']:.2f}")
    print(f"  window range [{win.min():.2f}, {win.max():.2f}]")


def main():
    ap = argparse.ArgumentParser(description="SMM backend — pipeline only, no UI")
    ap.add_argument("--movenet", default=Config.movenet)
    ap.add_argument("--cnn", default=Config.cnn)
    ap.add_argument("--scaler", default=Config.scaler)
    ap.add_argument("--meta", default=Config.meta)
    ap.add_argument("--conf", type=float, default=Config.conf_thresh)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    cfg = Config(movenet=a.movenet, cnn=a.cnn, scaler=a.scaler,
                 meta=a.meta, conf_thresh=a.conf)
    if a.selftest:
        selftest(cfg)
    else:
        print("This module is the backend. Run the app with:  python smm_app.py")
        for label, path, ok in check_artifacts(cfg):
            print(f"  {'OK  ' if ok else 'MISS'}  {label:<16} {path}")


if __name__ == "__main__":
    main()
