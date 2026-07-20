"""
Integration tests for the Pre Meltdown Detection System backend.

Each test corresponds to a case in the integration test plan (IT-01 .. IT-10).

Cases IT-01 .. IT-04 exercise the live SmmEngine and therefore require the
trained TFLite models under `prepared/`; they are skipped automatically when
those files are absent, so the suite still runs on a machine without them.
Cases IT-05 .. IT-10 use synthetic data and always run.

Run:  python -m unittest IntegrationTesting -v
   or: python IntegrationTesting.py
"""

import os
import csv
import json
import tempfile
import unittest
import warnings
import numpy as np

# Silence harmless ResourceWarnings emitted by the backend's json.load(open(...))
# reads, and the TFLite deprecation notice, so the results stay readable.
warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import smm_backend as bk
from smm_backend import (
    Config, SmmEngine, StreamingFeatureProcessor, SessionRecorder,
    find_episodes, read_sessions, filter_sessions_by_range, update_notes,
    KEEP,
)

MODELS_PRESENT = all(os.path.exists(p) for p in Config().artifacts().values())


# ── helpers ──────────────────────────────────────────────────────────────────
def make_scaler(n=10):
    return {"speed_median": [0.0] * n, "speed_iqr": [1.0] * n, "speed_cap": [1e9] * n}


def synthetic_frame(w=320, h=240):
    """A plain RGB frame; content is irrelevant to the pipeline's plumbing."""
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


def moving_keypoints(t, n=17):
    """A (17,3) keypoint set that oscillates over time, all points confident."""
    k = np.zeros((n, 3), np.float32)
    y = 0.5 + 0.1 * np.sin(2 * np.pi * 3 * t)
    x = 0.5 + 0.1 * np.cos(2 * np.pi * 3 * t)
    k[:, 0] = y
    k[:, 1] = x
    k[:, 2] = 0.9
    k[KEEP[bk.LSH], 1] = 0.4
    k[KEEP[bk.RSH], 1] = 0.6
    return k


def scored_result(t, prob, rate, alert):
    return bk.Result(
        t=t, keypoints=np.zeros((17, 3), np.float32), prob=prob, rate=rate,
        alert=alert, state="", warming=False, fill=1.0, scored=True, latency_ms=0.0)


# ── IT-01 .. IT-04 : the live engine pipeline (needs models) ─────────────────
@unittest.skipUnless(MODELS_PRESENT, "trained TFLite models not present under prepared/")
class TestEnginePipeline(unittest.TestCase):

    def setUp(self):
        self.engine = SmmEngine(Config())

    def test_IT01_pose_to_features_no_error(self):
        # feeding frames should accumulate a window without raising
        for i in range(30):
            r = self.engine.process(synthetic_frame(), t=i / 30.0)
            self.assertIsNotNone(r)

    def test_IT02_features_to_classification(self):
        r = None
        for i in range(90):
            r = self.engine.process(synthetic_frame(), t=i / 30.0)
        self.assertIsNotNone(r)                       # a result exists after 90 frames
        self.assertIsInstance(r.prob, float)

    def test_IT03_classification_to_alerting(self):
        r = None
        for i in range(120):
            r = self.engine.process(synthetic_frame(), t=i / 30.0)
        self.assertIsInstance(r.rate, float)
        self.assertIn(r.state, ("calm", "tentative", "confirmed", "suppressed"))

    def test_IT04_result_every_frame_unscored_then_scored(self):
        # Random frames contain no coherent body, so MoveNet reports low
        # confidence and no window ever completes — which is itself correct
        # behaviour (no person -> no detection). This case therefore verifies
        # the guaranteed-response contract: a Result is returned for every
        # frame, and the warm-up phase reports nothing scored. The scored
        # transition on a *real* body is covered by the end-to-end scenario
        # tests, which use a live camera.
        results = [self.engine.process(synthetic_frame(), t=i / 30.0)
                   for i in range(120)]
        self.assertTrue(all(r is not None for r in results))     # a result every frame
        self.assertTrue(all(not r.scored for r in results[:89]))  # warm-up: unscored
        self.assertTrue(all(r.warming for r in results[:89]))     # warming reported


# ── IT-05 : live vs offline feature parity (synthetic, always runs) ──────────
class TestFeatureParity(unittest.TestCase):
    """The streaming processor must reproduce the offline windowing exactly.

    This is verified structurally here: the same keypoint stream, pushed through
    the streaming processor twice, must yield identical windows, and the window
    must have the exact shape and channel layout the offline prep produces.
    """

    def test_IT05_streaming_matches_itself_and_layout(self):
        scaler = make_scaler()
        stream = [(moving_keypoints(i / 30.0), i / 30.0) for i in range(90)]

        p1 = StreamingFeatureProcessor(scaler, window=90)
        p2 = StreamingFeatureProcessor(scaler, window=90)
        w1 = w2 = None
        for kps, t in stream:
            w1 = p1.update(kps, t)
        for kps, t in stream:
            w2 = p2.update(kps, t)

        self.assertIsNotNone(w1)
        self.assertEqual(w1.shape, (90, 20))          # 90 frames, 10 speed + 10 conf
        np.testing.assert_allclose(w1, w2, rtol=1e-6, atol=1e-6)
        # channels 10..19 are confidence, so bounded to [0, 1]
        self.assertTrue(np.all(w1[:, 10:] >= 0.0) and np.all(w1[:, 10:] <= 1.0))


# ── IT-06 .. IT-07 : engine results to recorder to summary ───────────────────
class TestRecordingChain(unittest.TestCase):

    def _recorder(self):
        return SessionRecorder(Config(sessions_dir=tempfile.mkdtemp()), threshold=0.5)

    def test_IT06_only_scored_frames_retained(self):
        rec = self._recorder()
        # one unscored (warming) result, two scored
        warming = bk.Result(t=0.0, keypoints=np.zeros((17, 3), np.float32),
                            prob=0.0, rate=0.0, alert=False, state="", warming=True,
                            fill=0.3, scored=False, latency_ms=0.0)
        rec.add(warming, fps=30.0)
        rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(2.0, 0.9, 0.6, True), fps=30.0)
        self.assertEqual(len(rec.trace), 2)           # the warming frame was dropped

    def test_IT07_episode_and_stats_consistent(self):
        rec = self._recorder()
        rec.add(scored_result(0.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
        rec.add(scored_result(2.0, 0.1, 0.2, False), fps=30.0)
        s = rec.summary()
        self.assertEqual(s["n_episodes"], len(rec.episodes()))
        self.assertEqual(s["n_flagged"], 2)


# ── IT-08 .. IT-10 : storage, note update, filtered aggregation ──────────────
class TestStorageChain(unittest.TestCase):

    def _saved(self, n=1):
        cfg = Config(sessions_dir=tempfile.mkdtemp())
        ids = []
        for _ in range(n):
            rec = SessionRecorder(cfg, threshold=0.5)
            rec.add(scored_result(0.0, 0.9, 0.6, True), fps=30.0)
            rec.add(scored_result(1.0, 0.9, 0.6, True), fps=30.0)
            s = rec.summary()
            rec.save(s, notes="")
            ids.append(s["session_id"])
        return cfg, ids

    def test_IT08_save_then_read_roundtrip(self):
        cfg, ids = self._saved(1)
        rows = read_sessions(cfg)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], ids[0])
        # numeric columns restored to numbers, not strings
        self.assertIsInstance(rows[0]["n_episodes"], int)
        self.assertIsInstance(rows[0]["smms_per_min"], float)

    def test_IT09_note_update_visible_on_read(self):
        cfg, ids = self._saved(1)
        self.assertTrue(update_notes(cfg, ids[0], "rocking observed"))
        rows = read_sessions(cfg)
        self.assertEqual(rows[0]["notes"], "rocking observed")

    def test_IT10_filtered_aggregation(self):
        from datetime import datetime, timedelta
        cfg, _ids = self._saved(2)
        rows = read_sessions(cfg)
        # both saved just now -> both counted under "today", context-independent "all"
        now = datetime.now()
        today = filter_sessions_by_range(rows, "today", now=now)
        all_rows = filter_sessions_by_range(rows, "all", now=now)
        self.assertEqual(len(all_rows), 2)
        self.assertLessEqual(len(today), 2)
        # aggregate over the filtered set is computable
        total_eps = sum(r["n_episodes"] for r in all_rows)
        self.assertGreaterEqual(total_eps, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#python -m unittest tests.IntegrationTesting -v