"""
Unit tests for the Pre Meltdown Detection System backend.

Each test corresponds to a case in the unit test plan (TC-01 .. TC-20).
Interface cases (TC-21 .. TC-30) require the Qt application and a camera and
are verified manually; they are listed in the report but not automated here.

Run:  python -m unittest UnitTesting -v
   or: python UnitTesting.py
"""

import os
import json
import csv
import tempfile
import unittest
import warnings
import numpy as np

# The backend reads small JSON files with the open()-inside-json.load() idiom,
# which emits ResourceWarning on some interpreters. It is harmless; silence it
# so the test output stays readable.
warnings.simplefilter("ignore", ResourceWarning)

import smm_backend as bk
from smm_backend import (
    Config, check_artifacts, load_threshold,
    StreamingFeatureProcessor, Alerter, find_episodes,
    SessionRecorder, update_notes, read_sessions, filter_sessions_by_range,
    KEEP,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def make_scaler(n=10):
    """A trivial identity-like scaler so feature output is easy to reason about."""
    return {
        "speed_median": [0.0] * n,
        "speed_iqr": [1.0] * n,
        "speed_cap": [1e9] * n,
    }


def kps_all_valid(y=0.5, x=0.5, score=0.9):
    """A (17,3) keypoint array with every point confidently detected."""
    k = np.zeros((17, 3), np.float32)
    k[:, 0] = y
    k[:, 1] = x
    k[:, 2] = score
    # give the two shoulders a fixed separation so a body scale can be found
    lsh, rsh = KEEP[bk.LSH], KEEP[bk.RSH]
    k[lsh, 1] = 0.4
    k[rsh, 1] = 0.6
    return k


class WriteTempCSV:
    """Context manager: a Config pointing at a fresh temp sessions dir."""
    def __enter__(self):
        self.dir = tempfile.mkdtemp()
        self.cfg = Config(sessions_dir=self.dir)
        return self.cfg

    def __exit__(self, *a):
        pass


# ── TC-01 .. TC-05 : configuration and artifacts ─────────────────────────────
class TestConfigAndArtifacts(unittest.TestCase):

    def test_TC01_log_csv_path(self):
        cfg = Config(sessions_dir="sessions")
        self.assertEqual(cfg.log_csv, os.path.join("sessions", "log.csv"))

    def test_TC02_all_artifacts_present(self):
        d = tempfile.mkdtemp()
        cfg = Config(
            movenet=os.path.join(d, "m.tflite"),
            cnn=os.path.join(d, "c.tflite"),
            scaler=os.path.join(d, "s.json"),
            meta=os.path.join(d, "meta.json"),
        )
        for p in cfg.artifacts().values():
            open(p, "w").close()
        result = check_artifacts(cfg)
        self.assertTrue(all(exists for _label, _path, exists in result))

    def test_TC03_one_artifact_missing(self):
        d = tempfile.mkdtemp()
        cfg = Config(
            movenet=os.path.join(d, "m.tflite"),
            cnn=os.path.join(d, "c.tflite"),
            scaler=os.path.join(d, "s.json"),
            meta=os.path.join(d, "meta.json"),
        )
        # create all but the scaler
        open(cfg.movenet, "w").close()
        open(cfg.cnn, "w").close()
        open(cfg.meta, "w").close()
        result = dict((label, exists) for label, _path, exists in check_artifacts(cfg))
        self.assertFalse(result["Scaler"])
        self.assertTrue(result["MoveNet"] and result["SMM model"] and result["Model metadata"])

    def test_TC04_threshold_from_meta(self):
        d = tempfile.mkdtemp()
        meta = os.path.join(d, "meta.json")
        with open(meta, "w") as f:
            json.dump({"threshold": 0.42}, f)
        cfg = Config(meta=meta)
        self.assertAlmostEqual(load_threshold(cfg), 0.42)

    def test_TC05_threshold_default_when_missing(self):
        cfg = Config(meta=os.path.join(tempfile.mkdtemp(), "nope.json"))
        self.assertAlmostEqual(load_threshold(cfg, default=0.118), 0.118)


# ── TC-06 .. TC-09 : streaming feature processor ─────────────────────────────
class TestFeatureProcessor(unittest.TestCase):

    def test_TC06_warming_before_full(self):
        p = StreamingFeatureProcessor(make_scaler(), window=90)
        out = None
        for i in range(50):
            out = p.update(kps_all_valid(), t=i / 30.0)
        self.assertIsNone(out)
        self.assertTrue(p.warming)

    def test_TC07_window_shape_when_full(self):
        p = StreamingFeatureProcessor(make_scaler(), window=90)
        out = None
        for i in range(90):
            out = p.update(kps_all_valid(), t=i / 30.0)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (90, 20))
        self.assertFalse(p.warming)

    def test_TC08_short_gap_is_held(self):
        p = StreamingFeatureProcessor(make_scaler(), window=90, max_gap=4)
        # establish one wrist as valid, then drop it for 3 frames
        wrist = 6  # index within the kept-10 array (a wrist)
        for i in range(5):
            p.update(kps_all_valid(), t=i / 30.0)
        dropped = kps_all_valid()
        dropped[KEEP[wrist], 2] = 0.0            # below conf_thresh -> missing
        # 3-frame gap, within max_gap
        for i in range(5, 8):
            p.update(dropped, t=i / 30.0)
        # the held keypoint should still be considered valid (age < max_gap)
        self.assertLessEqual(p.age[wrist], p.max_gap)

    def test_TC09_long_gap_marked_invalid(self):
        p = StreamingFeatureProcessor(make_scaler(), window=90, max_gap=4)
        wrist = 6
        for i in range(5):
            p.update(kps_all_valid(), t=i / 30.0)
        dropped = kps_all_valid()
        dropped[KEEP[wrist], 2] = 0.0
        # 6-frame gap, beyond max_gap
        for i in range(5, 11):
            p.update(dropped, t=i / 30.0)
        self.assertGreater(p.age[wrist], p.max_gap)


# ── TC-10 .. TC-11 : alerter state machine ───────────────────────────────────
class TestAlerter(unittest.TestCase):

    def test_TC10_sustained_confirms(self):
        a = Alerter(threshold=0.5, roll_seconds=20, enter=0.5,
                    confirm=0.65, hold_seconds=1.0)
        rate = alert = None
        # feed many high probabilities so the rolling rate exceeds confirm
        for i in range(40):
            rate, alert = a.update(prob=0.99, t=i * 0.5)
        self.assertEqual(a.state, Alerter.CONFIRMED)
        self.assertTrue(alert)

    def test_TC11_transient_is_suppressed(self):
        a = Alerter(threshold=0.5, roll_seconds=20, enter=0.5,
                    confirm=0.65, hold_seconds=1.0)
        # Fill the rolling window with exactly 60% flagged, so the raw rate sits
        # at 0.60 — above `enter` (0.5) but below `confirm` (0.65) — and never
        # reaches confirm. Held past hold_seconds, this must be judged a transient.
        t = 0.0
        # 40 samples span the 20 s window at 0.5 s spacing; 24 flagged = 0.60
        seq = [0.99] * 24 + [0.0] * 16
        for pr in seq:
            a.update(prob=pr, t=t)
            t += 0.5
        # keep the rate steady at 0.60 well past the 1 s hold period
        pattern = [0.99, 0.99, 0.99, 0.0, 0.0]      # 3/5 = 0.60
        for _rep in range(20):
            for pr in pattern:
                a.update(prob=pr, t=t)
                t += 0.5
        self.assertEqual(a.state, Alerter.SUPPRESSED)
        _rate, alert = a.update(prob=0.99, t=t)      # one more steady step
        self.assertFalse(alert)                       # suppressed -> no alert


# ── TC-12 .. TC-14 : episode detection ───────────────────────────────────────
class TestFindEpisodes(unittest.TestCase):

    def test_TC12_two_consecutive_open_one_episode(self):
        trace = [(0.0, 0.9, True), (1.0, 0.9, True), (2.0, 0.1, False)]
        eps = find_episodes(trace, min_open=2, max_gap=1)
        self.assertEqual(len(eps), 1)

    def test_TC13_single_flag_no_episode(self):
        trace = [(0.0, 0.1, False), (1.0, 0.9, True), (2.0, 0.1, False)]
        eps = find_episodes(trace, min_open=2, max_gap=1)
        self.assertEqual(len(eps), 0)

    def test_TC14_single_gap_does_not_split(self):
        trace = [(0.0, 0.9, True), (1.0, 0.9, True),
                 (2.0, 0.1, False),                      # 1-frame gap
                 (3.0, 0.9, True), (4.0, 0.9, True)]
        eps = find_episodes(trace, min_open=2, max_gap=1)
        self.assertEqual(len(eps), 1)


# ── TC-15 .. TC-16 : session recorder ────────────────────────────────────────
def scored_result(t, prob, rate, alert):
    """Build a minimal Result-like object the recorder will accept."""
    return bk.Result(
        t=t, keypoints=np.zeros((17, 3), np.float32), prob=prob, rate=rate,
        alert=alert, state="", warming=False, fill=1.0, scored=True, latency_ms=0.0)


class TestSessionRecorder(unittest.TestCase):

    def _recorder(self):
        cfg = Config(sessions_dir=tempfile.mkdtemp())
        return SessionRecorder(cfg, threshold=0.5)

    def test_TC15_summary_matches_trace(self):
        rec = self._recorder()
        # two flagged windows -> one episode
        rec.add(scored_result(0.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(2.0, 0.1, 0.2, False), fps=30.0)
        s = rec.summary()
        self.assertEqual(s["n_windows"], 3)
        self.assertEqual(s["n_flagged"], 2)
        self.assertEqual(s["n_episodes"], 1)

    def test_TC16_save_appends_without_overwrite(self):
        rec = self._recorder()
        rec.add(scored_result(0.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
        rec.save(rec.summary(), notes="first")
        rec.save(rec.summary(), notes="second")
        with open(rec.cfg.log_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)                   # both rows present
        self.assertEqual(rows[0]["notes"], "first")      # first not overwritten


# ── TC-17 .. TC-20 : notes, reading, filtering ───────────────────────────────
class TestPersistence(unittest.TestCase):

    def _saved_session(self):
        cfg = Config(sessions_dir=tempfile.mkdtemp())
        rec = SessionRecorder(cfg, threshold=0.5)
        rec.add(scored_result(0.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
        summary = rec.summary()
        rec.save(summary, notes="")
        return cfg, summary["session_id"]

    def test_TC17_update_notes_existing(self):
        cfg, sid = self._saved_session()
        ok = update_notes(cfg, sid, "observed hand flapping")
        self.assertTrue(ok)
        rows = read_sessions(cfg)
        self.assertEqual(rows[0]["notes"], "observed hand flapping")

    def test_TC18_update_notes_missing_id(self):
        cfg, _sid = self._saved_session()
        ok = update_notes(cfg, "does_not_exist", "x")
        self.assertFalse(ok)

    def test_TC19_read_sessions_empty(self):
        cfg = Config(sessions_dir=tempfile.mkdtemp())     # no log written
        self.assertEqual(read_sessions(cfg), [])

    def test_TC20_filter_today(self):
        from datetime import datetime, timedelta
        now = datetime(2026, 1, 15, 12, 0, 0)
        rows = [
            {"timestamp": now.replace(hour=9).isoformat()},          # today
            {"timestamp": (now - timedelta(days=2)).isoformat()},    # earlier
        ]
        kept = filter_sessions_by_range(rows, "today", now=now)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
#python -m unittest tests.UnitTesting -v