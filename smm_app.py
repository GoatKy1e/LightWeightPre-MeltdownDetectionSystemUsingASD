#!/usr/bin/env python3
"""
smm_app.py — desktop front end for the SMM detector.

UI only. Every model, feature, and statistic lives in smm_backend; this module
owns widgets, painting, threading, and the camera overlay. If a line here would
still be needed with no screen attached, it belongs in the backend instead.

Run:
    QT_QPA_PLATFORM=xcb python smm_app.py

Deps: PySide6, opencv-python  (plus the backend's own)
"""
import sys
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QRectF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QDoubleSpinBox, QStackedWidget,
    QFrame, QSizePolicy, QTextEdit, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)

from smm_backend import (
    Config, SmmEngine, SessionRecorder, Result, check_artifacts, load_threshold,
    read_sessions, update_notes, filter_sessions_by_range, episode_vs_normal_time,
    RANGES,
)

# ── palette ──────────────────────────────────────────────────────────────────
BG, PANEL, LINE = "#12141a", "#1a1d26", "#2a2f3d"
TEXT, MUTED = "#e6e8ee", "#8b93a7"
CALM, ALERT, ACCENT = "#3ecf8e", "#ff6b5e", "#6c8cff"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT};
           font-family: 'Inter','Segoe UI','DejaVu Sans',sans-serif; font-size: 14px; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; }}
QLabel#h1 {{ font-size: 30px; font-weight: 600; }}
QLabel#h2 {{ font-size: 18px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; font-size: 13px; }}
QLabel#stat {{ font-size: 26px; font-weight: 600; }}
QLabel#statlabel {{ color: {MUTED}; font-size: 12px; }}
QPushButton {{ background: {ACCENT}; color: #ffffff; border: none; border-radius: 10px;
               padding: 12px 22px; font-size: 15px; font-weight: 600; }}
QPushButton:hover {{ background: #7f9bff; }}
QPushButton:disabled {{ background: #2a2f3d; color: {MUTED}; }}
QPushButton#danger {{ background: {ALERT}; }}
QPushButton#danger:hover {{ background: #ff8177; }}
QPushButton#ghost {{ background: transparent; color: {TEXT}; border: 1px solid {LINE}; }}
QPushButton#ghost:hover {{ border-color: {MUTED}; }}
QComboBox, QDoubleSpinBox {{ background: {PANEL}; border: 1px solid {LINE};
                             border-radius: 8px; padding: 8px 10px; }}
QTextEdit {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 8px; }}
"""


def fmt_hms(seconds):
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def panel():
    f = QFrame()
    f.setObjectName("panel")
    return f


def stat_block(value, label):
    box = QVBoxLayout()
    v = QLabel(value)
    v.setObjectName("stat")
    l = QLabel(label)
    l.setObjectName("statlabel")
    box.addWidget(v)
    box.addWidget(l)
    box.setSpacing(2)
    w = QWidget()
    w.setLayout(box)
    return w, v


def draw_overlay(rgb, r: Result, conf_thresh):
    """Bounding box + probability label. Presentation, so it lives here."""
    h, w = rgb.shape[:2]
    vis = r.keypoints[r.keypoints[:, 2] >= conf_thresh]
    if not len(vis):
        return
    ys, xs = vis[:, 0] * h, vis[:, 1] * w
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    pad = 20
    color = (255, 107, 94) if r.alert else (62, 207, 142)
    cv2.rectangle(rgb, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), color, 2)
    tag = f"SMM {r.prob:.2f}" + ("  warming up" if r.warming else "")
    cv2.rectangle(rgb, (x0 - pad, y0 - pad - 28), (x0 - pad + 240, y0 - pad), color, -1)
    cv2.putText(rgb, tag, (x0 - pad + 8, y0 - pad - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (18, 20, 26), 2)


# ── capture thread ───────────────────────────────────────────────────────────
class InferenceWorker(QThread):
    """Owns the camera and drives the backend engine. All heavy work happens here;
    the UI thread only paints what this emits."""

    frame_ready = Signal(object, object)   # rgb frame, Result
    fps_ready = Signal(float)
    failed = Signal(str)

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._running = False
        self.recorder = None

    def stop(self):
        self._running = False

    def run(self):
        try:
            engine = SmmEngine(self.cfg)
        except Exception as e:
            self.failed.emit(f"Could not load models: {e}")
            return

        self.recorder = SessionRecorder(self.cfg, engine.threshold)

        cap = cv2.VideoCapture(self.cfg.cam)
        if not cap.isOpened():
            self.failed.emit(f"Could not open camera {self.cfg.cam}.")
            return

        self._running = True
        fps_hist = deque(maxlen=30)
        last = time.time()

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.failed.emit("Camera stopped delivering frames.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r = engine.process(rgb)

            now = time.time()
            fps_hist.append(1.0 / max(now - last, 1e-6))
            last = now
            mean_fps = float(np.mean(fps_hist))

            self.recorder.add(r, mean_fps)
            draw_overlay(rgb, r, self.cfg.conf_thresh)

            self.frame_ready.emit(rgb, r)
            self.fps_ready.emit(mean_fps)

        cap.release()


# ── custom widgets ───────────────────────────────────────────────────────────
class RateMeter(QWidget):
    """Rolling-rate bar with the enter and confirm levels marked."""

    def __init__(self, enter=0.5, confirm=0.65, parent=None):
        super().__init__(parent)
        self.enter, self.confirm = enter, confirm
        self.rate, self.on = 0.0, False
        self.setFixedHeight(26)

    def set(self, rate, on):
        self.rate, self.on = rate, on
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(LINE))
        p.drawRoundedRect(0, 0, w, h, 6, 6)
        p.setBrush(QColor(ALERT if self.on else CALM))
        p.drawRoundedRect(0, 0, int(max(0.0, min(self.rate, 1.0)) * w), h, 6, 6)
        for lvl, style in ((self.enter, Qt.SolidLine), (self.confirm, Qt.DashLine)):
            x = int(lvl * w)
            p.setPen(QPen(QColor(TEXT), 1, style))
            p.drawLine(x, 0, x, h)


class ScorePlot(QWidget):
    """Last N seconds of SMM probability, threshold drawn as a dashed line."""

    def __init__(self, threshold, seconds=60, parent=None):
        super().__init__(parent)
        self.threshold, self.seconds = threshold, seconds
        self.pts = deque(maxlen=1800)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def add(self, t, prob):
        self.pts.append((t, prob))
        self.update()

    def reset(self):
        self.pts.clear()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))

        ty = h - self.threshold * h
        p.setPen(QPen(QColor(MUTED), 1, Qt.DashLine))
        p.drawLine(0, int(ty), w, int(ty))
        p.setFont(QFont("", 9))
        p.drawText(6, int(ty) - 4, f"threshold {self.threshold:.3f}")

        if len(self.pts) < 2:
            return
        t0 = self.pts[-1][0] - self.seconds
        vis = [(t, v) for t, v in self.pts if t >= t0]
        if len(vis) < 2:
            return

        def xy(t, v):
            return (t - t0) / self.seconds * w, h - v * h

        p.setPen(QPen(QColor(ACCENT), 2))
        prev = xy(*vis[0])
        for t, v in vis[1:]:
            cur = xy(t, v)
            p.drawLine(int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
            prev = cur


class Timeline(QWidget):
    """Whole-session score trace with detected episodes shaded."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trace, self.episodes, self.threshold = [], [], 0.5
        self.setMinimumHeight(120)

    def set_data(self, trace, episodes, threshold):
        self.trace, self.episodes, self.threshold = trace, episodes, threshold
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))
        if not self.trace:
            return
        dur = max(self.trace[-1][0], 1e-6)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 107, 94, 55))
        for s, e in self.episodes:
            x0, x1 = s / dur * w, e / dur * w
            p.drawRect(int(x0), 0, max(int(x1 - x0), 2), h)

        ty = h - self.threshold * h
        p.setPen(QPen(QColor(MUTED), 1, Qt.DashLine))
        p.drawLine(0, int(ty), w, int(ty))

        p.setPen(QPen(QColor(ACCENT), 2))
        prev = None
        for t, v, _f in self.trace:
            cur = (t / dur * w, h - v * h)
            if prev:
                p.drawLine(int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
            prev = cur


# rotating palette for categorical slices/bars
CHART_COLORS = ["#6c8cff", "#3ecf8e", "#ff6b5e", "#f5c451", "#a978ff", "#4fd0e0"]


class DonutChart(QWidget):
    """Donut/pie chart with a centre total and a side legend.

    data: list of (label, value). Zero-value slices are dropped. If everything
    is zero, shows an empty ring rather than dividing by zero.
    """

    def __init__(self, center_label="", parent=None):
        super().__init__(parent)
        self.data = []
        self.center_label = center_label
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, data, center_label=None):
        self.data = [(l, float(v)) for l, v in data if v]
        if center_label is not None:
            self.center_label = center_label
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))

        total = sum(v for _l, v in self.data)
        # donut occupies the left square; legend to the right
        size = min(h, w * 0.55) - 16
        cx, cy = 8 + size / 2, h / 2
        ring = size * 0.26  # ring thickness
        rect = QRectF(cx - size / 2, cy - size / 2, size, size)
        inner = QRectF(rect.x() + ring, rect.y() + ring,
                       rect.width() - 2 * ring, rect.height() - 2 * ring)

        if total <= 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(LINE))
            p.drawEllipse(rect)
            p.setBrush(QColor(PANEL))
            p.drawEllipse(inner)
            p.setPen(QColor(MUTED))
            p.setFont(QFont("", 10))
            p.drawText(rect, Qt.AlignCenter, "no data")
            return

        # slices, drawn as pie wedges then punched out to a ring
        p.setPen(Qt.NoPen)
        start = 90 * 16  # start at top, Qt angles are 1/16 degree
        for i, (_l, v) in enumerate(self.data):
            span = -int(round(v / total * 360 * 16))
            p.setBrush(QColor(CHART_COLORS[i % len(CHART_COLORS)]))
            p.drawPie(rect, start, span)
            start += span
        p.setBrush(QColor(PANEL))
        p.drawEllipse(inner)

        # centre total
        p.setPen(QColor(TEXT))
        p.setFont(QFont("", 15, QFont.Bold))
        p.drawText(inner, Qt.AlignCenter,
                   f"{int(total)}" if total == int(total) else f"{total:.0f}")
        if self.center_label:
            p.setPen(QColor(MUTED))
            p.setFont(QFont("", 8))
            lbl_rect = QRectF(inner.x(), inner.y() + inner.height() * 0.60,
                              inner.width(), inner.height() * 0.3)
            p.drawText(lbl_rect, Qt.AlignHCenter | Qt.AlignTop, self.center_label)

        # legend
        lx = 8 + size + 20
        ly = (h - len(self.data) * 24) / 2
        p.setFont(QFont("", 9))
        for i, (label, v) in enumerate(self.data):
            y = ly + i * 24
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(CHART_COLORS[i % len(CHART_COLORS)]))
            p.drawRoundedRect(QRectF(lx, y + 3, 12, 12), 3, 3)
            p.setPen(QColor(TEXT))
            pct = v / total * 100
            p.drawText(QRectF(lx + 20, y, w - lx - 24, 18),
                       Qt.AlignVCenter | Qt.AlignLeft, f"{label}   {pct:.0f}%")


class BarChart(QWidget):
    """Vertical bar chart. data: list of (label, value). Auto-scales to the max.

    Bars use the accent colour; the single tallest bar is highlighted. Labels
    sit under each bar and values above it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.suffix = ""
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, data, suffix=""):
        self.data = [(str(l), float(v)) for l, v in data]
        self.suffix = suffix
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))
        if not self.data:
            p.setPen(QColor(MUTED))
            p.setFont(QFont("", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "no data")
            return

        pad_l, pad_r, pad_t, pad_b = 12, 12, 22, 28
        plot_w = w - pad_l - pad_r
        plot_h = h - pad_t - pad_b
        vmax = max(v for _l, v in self.data) or 1.0
        imax = max(range(len(self.data)), key=lambda i: self.data[i][1])

        n = len(self.data)
        slot = plot_w / n
        bw = min(slot * 0.6, 60)

        p.setFont(QFont("", 8))
        for i, (label, v) in enumerate(self.data):
            bh = (v / vmax) * plot_h
            x = pad_l + slot * i + (slot - bw) / 2
            y = pad_t + (plot_h - bh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(ACCENT if i == imax else "#3a4360"))
            p.drawRoundedRect(QRectF(x, y, bw, bh), 4, 4)
            # value above
            p.setPen(QColor(TEXT))
            vtxt = f"{v:.0f}{self.suffix}" if v == int(v) else f"{v:.1f}{self.suffix}"
            p.drawText(QRectF(pad_l + slot * i, y - 18, slot, 16),
                       Qt.AlignHCenter | Qt.AlignBottom, vtxt)
            # label below
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(pad_l + slot * i, h - pad_b + 4, slot, 20),
                       Qt.AlignHCenter | Qt.AlignTop, label)


def fmt_duration(seconds):
    """Human-readable duration for legends: 45s, 12m 30s, 1h 05m."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


class PieChart(QWidget):
    """Two-slice donut: episode time vs normal time, with a legend and centre total.

    set_data takes (episode_seconds, normal_seconds). Handles the all-zero case
    (no sessions in range) by drawing an empty ring instead of dividing by zero.
    """

    EP_COLOR = "#ff6b5e"      # episode (SMM) time
    NORM_COLOR = "#3ecf8e"    # normal time

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ep = 0.0
        self.norm = 0.0
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, ep_seconds, normal_seconds):
        self.ep = max(float(ep_seconds), 0.0)
        self.norm = max(float(normal_seconds), 0.0)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PANEL))

        total = self.ep + self.norm
        size = min(h, w * 0.55) - 24
        cx, cy = 12 + size / 2, h / 2
        ring = size * 0.26
        rect = QRectF(cx - size / 2, cy - size / 2, size, size)
        inner = QRectF(rect.x() + ring, rect.y() + ring,
                       rect.width() - 2 * ring, rect.height() - 2 * ring)

        if total <= 0:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(LINE))
            p.drawEllipse(rect)
            p.setBrush(QColor(PANEL))
            p.drawEllipse(inner)
            p.setPen(QColor(MUTED))
            p.setFont(QFont("", 10))
            p.drawText(rect, Qt.AlignCenter, "no data")
            return

        # two wedges, drawn then punched out to a ring
        p.setPen(Qt.NoPen)
        start = 90 * 16  # top, clockwise
        for value, color in ((self.ep, self.EP_COLOR), (self.norm, self.NORM_COLOR)):
            span = -int(round(value / total * 360 * 16))
            p.setBrush(QColor(color))
            p.drawPie(rect, start, span)
            start += span
        p.setBrush(QColor(PANEL))
        p.drawEllipse(inner)

        # centre: total time
        p.setPen(QColor(TEXT))
        p.setFont(QFont("", 13, QFont.Bold))
        p.drawText(QRectF(inner.x(), inner.y() + inner.height() * 0.24,
                          inner.width(), inner.height() * 0.32),
                   Qt.AlignHCenter | Qt.AlignBottom, fmt_duration(total))
        p.setPen(QColor(MUTED))
        p.setFont(QFont("", 8))
        p.drawText(QRectF(inner.x(), inner.y() + inner.height() * 0.56,
                          inner.width(), inner.height() * 0.3),
                   Qt.AlignHCenter | Qt.AlignTop, "total time")

        # legend
        lx = 12 + size + 24
        rows = [("Episode time", self.ep, self.EP_COLOR),
                ("Normal time", self.norm, self.NORM_COLOR)]
        ly = (h - len(rows) * 40) / 2
        for i, (label, value, color) in enumerate(rows):
            y = ly + i * 40
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawRoundedRect(QRectF(lx, y + 2, 14, 14), 3, 3)
            p.setPen(QColor(TEXT))
            p.setFont(QFont("", 10, QFont.Bold))
            pct = value / total * 100
            p.drawText(QRectF(lx + 24, y, w - lx - 28, 18),
                       Qt.AlignVCenter | Qt.AlignLeft, f"{label}   {pct:.1f}%")
            p.setPen(QColor(MUTED))
            p.setFont(QFont("", 9))
            p.drawText(QRectF(lx + 24, y + 18, w - lx - 28, 18),
                       Qt.AlignVCenter | Qt.AlignLeft, fmt_duration(value))


# ── main window ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMM live detector")
        #self.resize(1240, 760)
        self.showFullScreen()
        self.showMaximized()
        self.cfg = Config()
        self.worker = None
        self.recorder = None
        self.t_start = None
        self.threshold = load_threshold(self.cfg)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.IDLE, self.RUNNING, self.REPORT, self.DASHBOARD = 0, 1, 2, 3
        self.stack.addWidget(self._build_idle())
        self.stack.addWidget(self._build_running())
        self.stack.addWidget(self._build_report())
        self.stack.addWidget(self._build_dashboard())

        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    def _pill(self, on):
        c = ALERT if on else CALM
        self.alert_pill.setText("ALERT" if on else "ok")
        self.alert_pill.setStyleSheet(
            f"background: {c}; color: {BG}; border-radius: 14px; "
            f"font-weight: 700; font-size: 13px; padding: 0 14px;")

    # ── idle ──
    def _build_idle(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(60, 50, 60, 50)
        root.setSpacing(18)

        title = QLabel("SMM live detector")
        title.setObjectName("h1")
        sub = QLabel("Stereotypical motor movement detection from webcam pose  ·  "
                     "MoveNet → 1D-CNN")
        sub.setObjectName("muted")
        root.addWidget(title)
        root.addWidget(sub)
        root.addSpacing(10)

        checks = panel()
        cl = QVBoxLayout(checks)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(6)
        hdr = QLabel("Artifacts")
        hdr.setObjectName("h2")
        cl.addWidget(hdr)
        self.ready = True
        for label, path, ok in check_artifacts(self.cfg):
            self.ready &= ok
            row = QLabel(f"{'✓' if ok else '✗'}  {label} — {path}")
            row.setStyleSheet(f"color: {CALM if ok else ALERT};")
            cl.addWidget(row)
        root.addWidget(checks)

        ctl = panel()
        gl = QGridLayout(ctl)
        gl.setContentsMargins(20, 16, 20, 16)
        gl.setHorizontalSpacing(18)

        def spin(lo, hi, step, dec, val, col, label):
            gl.addWidget(QLabel(label), 0, col)
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(dec)
            s.setValue(val)
            gl.addWidget(s, 1, col)
            return s

        gl.addWidget(QLabel("Camera"), 0, 0)
        self.cam_box = QComboBox()
        for i in range(4):
            self.cam_box.addItem(f"Camera {i}", i)
        gl.addWidget(self.cam_box, 1, 0)

        self.thr_box = spin(0.01, 0.99, 0.005, 3, self.threshold, 1, "Decision threshold")
        self.conf_box = spin(0.05, 0.90, 0.05, 2, self.cfg.conf_thresh, 2, "Keypoint confidence")
        self.enter_box = spin(0.05, 0.95, 0.05, 2, self.cfg.enter, 3, "Alert level (enter)")
        self.confirm_box = spin(0.05, 0.99, 0.05, 2, self.cfg.confirm, 4, "Confirm level")
        gl.setColumnStretch(5, 1)
        root.addWidget(ctl)

        root.addStretch(1)
        self.start_btn = QPushButton("Start live detection")
        self.start_btn.setMinimumHeight(52)
        self.start_btn.setEnabled(self.ready)
        self.start_btn.clicked.connect(self.start_session)
        root.addWidget(self.start_btn)

        if not self.ready:
            warn = QLabel("Some artifacts are missing. Check the paths above.")
            warn.setStyleSheet(f"color: {ALERT};")
            root.addWidget(warn)
        return page

    # ── running ──
    def _build_running(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        body = QHBoxLayout()
        body.setSpacing(16)

        vid_panel = panel()
        vl = QVBoxLayout(vid_panel)
        vl.setContentsMargins(10, 10, 10, 10)
        self.video = QLabel("Starting camera…")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(720, 540)
        self.video.setStyleSheet(f"color: {MUTED}; border-radius: 8px;")
        vl.addWidget(self.video)
        body.addWidget(vid_panel, 3)

        side = QVBoxLayout()
        side.setSpacing(14)

        prob_panel = panel()
        pl = QVBoxLayout(prob_panel)
        pl.setContentsMargins(20, 16, 20, 16)
        cap = QLabel("SMM probability  ·  this window")
        cap.setObjectName("statlabel")
        self.prob_lbl = QLabel("0.00")
        self.prob_lbl.setStyleSheet(f"font-size: 48px; font-weight: 700; color: {CALM};")
        self.state_lbl = QLabel("Warming up")
        self.state_lbl.setObjectName("muted")
        pl.addWidget(cap)
        pl.addWidget(self.prob_lbl)
        pl.addWidget(self.state_lbl)
        side.addWidget(prob_panel)

        rate_panel = panel()
        rl = QVBoxLayout(rate_panel)
        rl.setContentsMargins(20, 16, 20, 16)
        rl.setSpacing(8)
        rcap = QLabel("Rolling SMM rate  ·  last 20 s")
        rcap.setObjectName("statlabel")
        rrow = QHBoxLayout()
        self.rate_val = QLabel("0.00")
        self.rate_val.setStyleSheet(f"font-size: 34px; font-weight: 700; color: {CALM};")
        self.alert_pill = QLabel("ok")
        self.alert_pill.setAlignment(Qt.AlignCenter)
        self.alert_pill.setFixedHeight(28)
        self.alert_pill.setMinimumWidth(88)
        self._pill(False)
        rrow.addWidget(self.rate_val)
        rrow.addStretch(1)
        rrow.addWidget(self.alert_pill)
        self.meter = RateMeter()
        self.alert_state_lbl = QLabel("calm")
        self.alert_state_lbl.setObjectName("muted")
        rl.addWidget(rcap)
        rl.addLayout(rrow)
        rl.addWidget(self.meter)
        rl.addWidget(self.alert_state_lbl)
        side.addWidget(rate_panel)

        plot_panel = panel()
        pp = QVBoxLayout(plot_panel)
        pp.setContentsMargins(12, 12, 12, 12)
        pcap = QLabel("Last 60 seconds")
        pcap.setObjectName("statlabel")
        self.plot = ScorePlot(self.threshold)
        pp.addWidget(pcap)
        pp.addWidget(self.plot)
        side.addWidget(plot_panel, 1)

        stats_panel = panel()
        sg = QGridLayout(stats_panel)
        sg.setContentsMargins(20, 16, 20, 16)
        w1, self.dur_lbl = stat_block("00:00", "Duration")
        w2, self.epi_lbl = stat_block("0", "Episodes")
        sg.addWidget(w1, 0, 0)
        sg.addWidget(w2, 0, 1)
        side.addWidget(stats_panel)

        body.addLayout(side, 2)
        root.addLayout(body, 1)

        bar = QHBoxLayout()
        self.perf_lbl = QLabel("—")
        self.perf_lbl.setObjectName("muted")
        bar.addWidget(self.perf_lbl)
        bar.addStretch(1)
        self.stop_btn = QPushButton("Stop and open report")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setMinimumHeight(46)
        self.stop_btn.clicked.connect(self.stop_session)
        bar.addWidget(self.stop_btn)
        root.addLayout(bar)
        return page

    # ── report ──
    def _build_report(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(16)

        head = QLabel("Session report")
        head.setObjectName("h1")
        self.rep_when = QLabel("")
        self.rep_when.setObjectName("muted")
        root.addWidget(head)
        root.addWidget(self.rep_when)

        stats = panel()
        sg = QGridLayout(stats)
        sg.setContentsMargins(24, 20, 24, 20)
        sg.setHorizontalSpacing(30)
        blocks = [("Duration", "r_dur"), ("Episodes", "r_epi"), ("SMMs / min", "r_rate"),
                  ("Time in SMM", "r_pct"), ("Median episode", "r_med"),
                  ("Peak rate", "r_peak"), ("Time alerting", "r_alert"), ("Mean FPS", "r_fps")]
        for i, (label, attr) in enumerate(blocks):
            w, lbl = stat_block("—", label)
            setattr(self, attr, lbl)
            sg.addWidget(w, 0, i)
        root.addWidget(stats)

        tl_panel = panel()
        tp = QVBoxLayout(tl_panel)
        tp.setContentsMargins(14, 12, 14, 12)
        tcap = QLabel("Session timeline — shaded bands are detected episodes")
        tcap.setObjectName("statlabel")
        self.timeline = Timeline()
        tp.addWidget(tcap)
        tp.addWidget(self.timeline)
        root.addWidget(tl_panel, 1)

        nrow = QHBoxLayout()
        ncap = QLabel("Notes")
        ncap.setObjectName("statlabel")
        nrow.addWidget(ncap)
        nrow.addStretch(1)
        self.notes_btn = QPushButton("Save notes")
        self.notes_btn.setObjectName("ghost")
        self.notes_btn.clicked.connect(self.save_notes)
        nrow.addWidget(self.notes_btn)
        self.notes = QTextEdit()
        self.notes.setMaximumHeight(80)
        self.notes.setPlaceholderText("What was this session? Anything worth remembering.")
        root.addLayout(nrow)
        root.addWidget(self.notes)

        bar = QHBoxLayout()
        self.saved_lbl = QLabel("")
        self.saved_lbl.setObjectName("muted")
        bar.addWidget(self.saved_lbl)
        bar.addStretch(1)
        again = QPushButton("New session")
        again.setObjectName("ghost")
        again.clicked.connect(lambda: self.stack.setCurrentIndex(self.IDLE))
        self.reports_btn = QPushButton("View all reports")
        self.reports_btn.clicked.connect(self.open_dashboard)
        bar.addWidget(again)
        bar.addWidget(self.reports_btn)
        root.addLayout(bar)
        return page

    # ── dashboard ──
    def _build_dashboard(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 32, 40, 32)
        root.setSpacing(16)

        top = QHBoxLayout()
        head = QLabel("All session reports")
        head.setObjectName("h1")
        top.addWidget(head)
        top.addStretch(1)
        back = QPushButton("New session")
        back.setObjectName("ghost")
        back.clicked.connect(lambda: self.stack.setCurrentIndex(self.IDLE))
        top.addWidget(back)
        root.addLayout(top)

        self.dash_sub = QLabel("")
        self.dash_sub.setObjectName("muted")
        root.addWidget(self.dash_sub)

        # time-range selector — filters the whole dashboard
        sel = QHBoxLayout()
        sel.setSpacing(8)
        rlbl = QLabel("Show:")
        rlbl.setObjectName("muted")
        sel.addWidget(rlbl)
        self.range_btns = {}
        for key, text in [("today", "Today"), ("week", "This week"),
                          ("month", "This month"), ("all", "All time")]:
            b = QPushButton(text)
            b.setObjectName("ghost")
            b.setCheckable(True)
            b.clicked.connect(lambda _c=False, k=key: self.set_range(k))
            self.range_btns[key] = b
            sel.addWidget(b)
        sel.addStretch(1)
        root.addLayout(sel)
        self.dash_range = "all"

        # summary cards — aggregates across every saved session
        cards = panel()
        cg = QGridLayout(cards)
        cg.setContentsMargins(24, 20, 24, 20)
        cg.setHorizontalSpacing(30)
        agg = [("Sessions", "d_n"), ("Total time", "d_time"), ("Total episodes", "d_epi"),
               ("Avg SMMs / min", "d_rate"), ("Avg time in SMM", "d_pct"),
               ("Avg peak rate", "d_peak")]
        for i, (label, attr) in enumerate(agg):
            w, lbl = stat_block("—", label)
            setattr(self, attr, lbl)
            cg.addWidget(w, 0, i)
        root.addWidget(cards)

        # charts row: episode vs normal pie
        charts = QHBoxLayout()
        charts.setSpacing(16)

        pie_panel = panel()
        pie_l = QVBoxLayout(pie_panel)
        pie_l.setContentsMargins(16, 14, 16, 14)
        pcap = QLabel("Episode time vs normal time")
        pcap.setObjectName("statlabel")
        self.pie = PieChart()
        pie_l.addWidget(pcap)
        pie_l.addWidget(self.pie)
        charts.addWidget(pie_panel, 1)

        root.addLayout(charts, 1)

        # per-session table
        tbl_panel = panel()
        tpl = QVBoxLayout(tbl_panel)
        tpl.setContentsMargins(14, 12, 14, 12)
        tcap = QLabel("Every session, newest first")
        tcap.setObjectName("statlabel")
        tpl.addWidget(tcap)

        self.COLS = [
            ("When", "timestamp"), ("Duration", "duration_s"), ("Episodes", "n_episodes"),
            ("SMMs/min", "smms_per_min"), ("% SMM", "pct_time_smm"),
            ("Median ep.", "median_episode_s"), ("Peak rate", "peak_rate"),
            ("% alerting", "pct_time_alerting"), ("Threshold", "threshold"),
            ("FPS", "mean_fps"), ("Notes", "notes"),
        ]
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([h for h, _k in self.COLS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: {PANEL}; border: none; gridline-color: {LINE};
                            alternate-background-color: #171a22; }}
            QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none;
                                    border-bottom: 1px solid {LINE}; padding: 8px 10px;
                                    font-weight: 600; }}
            QTableWidget::item {{ padding: 8px 10px; border-bottom: 1px solid {LINE}; }}
            QTableWidget::item:selected {{ background: #232838; color: {TEXT}; }}
        """)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)  # Notes takes slack
        tpl.addWidget(self.table)
        root.addWidget(tbl_panel, 1)

        self.dash_empty = QLabel("No sessions in this range.")
        self.dash_empty.setObjectName("muted")
        self.dash_empty.setAlignment(Qt.AlignCenter)
        self.dash_empty.setVisible(False)
        root.addWidget(self.dash_empty)

        return page

    def set_range(self, key):
        self.dash_range = key
        for k, b in self.range_btns.items():
            b.setChecked(k == key)
        self.refresh_dashboard()

    def refresh_dashboard(self):
        # keep the toggle buttons in sync when opened directly
        for k, b in self.range_btns.items():
            b.setChecked(k == self.dash_range)

        all_rows = read_sessions(self.cfg)
        rows = filter_sessions_by_range(all_rows, self.dash_range)
        self.dash_empty.setVisible(not rows)
        self.table.setVisible(bool(rows))

        n = len(rows)
        span = {"today": "today", "week": "in the last 7 days",
                "month": "in the last 30 days", "all": "recorded"}[self.dash_range]
        self.dash_sub.setText(f"{n} session{'s' if n != 1 else ''} {span}")

        if rows:
            total_time = sum(r.get("duration_s", 0) or 0 for r in rows)
            total_epi = sum(int(r.get("n_episodes", 0) or 0) for r in rows)
            avg = lambda k: sum(r.get(k, 0) or 0 for r in rows) / n
            self.d_n.setText(str(n))
            self.d_time.setText(fmt_hms(total_time))
            self.d_epi.setText(str(total_epi))
            self.d_rate.setText(f"{avg('smms_per_min'):.2f}")
            self.d_pct.setText(f"{avg('pct_time_smm'):.1f}%")
            self.d_peak.setText(f"{avg('peak_rate'):.2f}")
        else:
            for attr in ("d_n", "d_time", "d_epi", "d_rate", "d_pct", "d_peak"):
                getattr(self, attr).setText("—")

        # pie: episode time vs normal time, summed across the filtered range
        ep, norm = episode_vs_normal_time(rows)
        self.pie.set_data(ep, norm)

        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, (_h, key) in enumerate(self.COLS):
                self.table.setItem(i, j, QTableWidgetItem(self._fmt_cell(key, r.get(key))))

    @staticmethod
    def _fmt_cell(key, val):
        if val is None or val == "":
            return "—" if key != "notes" else ""
        if key == "timestamp":
            return str(val).replace("T", "  ")
        if key == "duration_s":
            return fmt_hms(val)
        if key in ("smms_per_min", "peak_rate"):
            return f"{float(val):.2f}"
        if key in ("pct_time_smm", "pct_time_alerting"):
            return f"{float(val):.1f}%"
        if key == "median_episode_s":
            return f"{float(val):.1f}s"
        if key == "threshold":
            return f"{float(val):.3f}"
        if key == "mean_fps":
            return f"{float(val):.1f}"
        return str(val)

    # ── control ──
    def start_session(self):
        self.cfg.cam = self.cam_box.currentData()
        self.cfg.threshold = self.thr_box.value()
        self.cfg.conf_thresh = self.conf_box.value()
        self.cfg.enter = self.enter_box.value()
        self.cfg.confirm = self.confirm_box.value()

        self.threshold = self.cfg.threshold
        self.plot.threshold = self.threshold
        self.plot.reset()
        self.meter.enter = self.cfg.enter
        self.meter.confirm = self.cfg.confirm
        self.t_start = time.time()

        self.worker = InferenceWorker(self.cfg)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.fps_ready.connect(self.on_fps)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

        self.clock.start(500)
        self.stack.setCurrentIndex(1)

    def stop_session(self):
        self.clock.stop()
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
        self.build_report()
        self.stack.setCurrentIndex(2)

    @Slot(str)
    def on_failed(self, msg):
        self.clock.stop()
        QMessageBox.critical(self, "Live detection stopped", msg)
        self.stack.setCurrentIndex(0)

    @Slot(object, object)
    def on_frame(self, rgb, r):
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(img).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        c = ALERT if r.alert else CALM
        self.prob_lbl.setText(f"{r.prob:.2f}")
        self.prob_lbl.setStyleSheet(f"font-size: 48px; font-weight: 700; color: {c};")
        self.state_lbl.setText(
            f"Warming up — {int(r.fill * 100)}% of the window filled" if r.warming
            else f"above threshold {self.threshold:.3f}" if r.prob >= self.threshold
            else f"below threshold {self.threshold:.3f}")

        self.rate_val.setText(f"{r.rate:.2f}")
        self.rate_val.setStyleSheet(f"font-size: 34px; font-weight: 700; color: {c};")
        self.meter.set(r.rate, r.alert)
        self._pill(r.alert)
        self.alert_state_lbl.setText(
            f"{r.state}  ·  enter {self.cfg.enter:.2f}  ·  confirm {self.cfg.confirm:.2f}")

        if not r.warming:
            self.plot.add(r.t, r.prob)

    @Slot(float)
    def on_fps(self, fps):
        self.perf_lbl.setText(f"{fps:.1f} fps")

    def _tick(self):
        if not self.worker or not self.worker.recorder:
            return
        self.dur_lbl.setText(fmt_hms(time.time() - self.t_start))
        self.epi_lbl.setText(str(len(self.worker.recorder.episodes())))

    # ── report ──
    def build_report(self):
        self.recorder = self.worker.recorder if self.worker else None
        if not self.recorder:
            return
        s = self.recorder.summary()
        self.summary = s

        self.rep_when.setText(s["timestamp"].replace("T", "  ·  "))
        self.r_dur.setText(fmt_hms(s["duration_s"]))
        self.r_epi.setText(str(s["n_episodes"]))
        self.r_rate.setText(f"{s['smms_per_min']:.2f}")
        self.r_pct.setText(f"{s['pct_time_smm']:.1f}%")
        self.r_med.setText(f"{s['median_episode_s']:.1f}s")
        self.r_peak.setText(f"{s['peak_rate']:.2f}")
        self.r_alert.setText(f"{s['pct_time_alerting']:.0f}%")
        self.r_fps.setText(f"{s['mean_fps']:.1f}")

        self.timeline.set_data(self.recorder.trace, self.recorder.episodes(), self.threshold)
        self.notes.clear()

        # Auto-save the moment the report opens — no separate Save button.
        csv_path = self.recorder.save(self.summary, "")
        self.saved_id = self.summary["session_id"]
        self.saved_lbl.setText(f"Saved to {csv_path}  ·  add notes below and press Save notes")
        self.notes_btn.setEnabled(True)

    def save_notes(self):
        sid = getattr(self, "saved_id", None)
        if not sid:
            return
        if update_notes(self.cfg, sid, self.notes.toPlainText()):
            self.saved_lbl.setText("Notes saved.")
        else:
            self.saved_lbl.setText("Could not find the saved row to update.")

    def open_dashboard(self):
        self.refresh_dashboard()
        self.stack.setCurrentIndex(self.DASHBOARD)

    def closeEvent(self, e):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
