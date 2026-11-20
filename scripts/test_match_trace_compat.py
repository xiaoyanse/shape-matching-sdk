"""定向验证 Python 绑定与冻结的 dist/runtime 匹配诊断 ABI。"""

from __future__ import annotations

import ctypes as ct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from shape_sdk import NativeModel, NativeShapeError, last_error_info, train_model
from shape_sdk import diagnostics
from shape_sdk.native_library import MatchTrace


class MatchTraceCompat(unittest.TestCase):
    def test_frozen_runtime_negotiates_once_and_propagates_other_errors(self):
        frame = np.zeros((320, 320, 3), np.uint8)
        frame[42:198, 42:48] = frame[42:48, 42:198] = 255
        frame[192:198, 42:198] = frame[42:198, 192:198] = 255
        frame[65:95, 65:95] = frame[80:180, 165:171] = 255
        frame[120:155, 90:155] = 200
        asset = train_model(
            frame, (32, 32, 176, 176), angle_min=0, angle_max=0, angle_step=1,
            scale_min=1, scale_max=1, scale_step=1, weak_th=20, strong_th=40, feature_count=32,
        )
        with NativeModel(asset) as model:
            hits = model.match(frame, threshold=60)
            self.assertTrue(hits)
            other_error = NativeShapeError("invalid match trace output ABI (other diagnostic)")
            with patch.object(diagnostics, "invoke", side_effect=other_error) as invoke:
                with self.assertRaises(NativeShapeError) as raised:
                    model.last_match_info()
                self.assertIs(raised.exception, other_error)
                invoke.assert_called_once()

            library = model._library
            native_trace = library.vf_shape_last_match_info
            sizes = []

            def record_size(handle, out, error, capacity):
                sizes.append(ct.cast(out, ct.POINTER(MatchTrace)).contents.struct_size)
                return native_trace(handle, out, error, capacity)

            with patch.object(library, "vf_shape_last_match_info", new=record_size):
                self.check_native_handle_error(model)
                self.assertEqual(sizes, [184, 112])
                sizes.clear()
                snapshot = model.last_match_info()
                self.assertEqual(sizes, [184, 112])
                self.assertEqual(snapshot["final_candidates"], len(hits))
                self.assertEqual(set(snapshot), {
                    "coarse_candidates", "windows_searched", "final_candidates",
                    "window_best_scores", "coarse_ms", "coarse_resize_ms", "windows_ms",
                    "finish_ms",
                })
                sizes.clear()
                self.check_native_handle_error(model)
                self.assertEqual(sizes, [112])
                prior_error = last_error_info()
                self.assertEqual(prior_error[0], diagnostics.HANDLE_INVALID)
                self.assertEqual(model.last_match_info(), snapshot)
                self.assertEqual(sizes, [112, 112])
                self.assertEqual(last_error_info(), prior_error)

    def check_native_handle_error(self, model):
        with patch.object(model, "_handle", ct.c_uint64(0xDEADBEEF)):
            with self.assertRaises(NativeShapeError) as raised:
                model.last_match_info()
        self.assertEqual(raised.exception.args, ("model handle is closed or unknown",))


if __name__ == "__main__":
    unittest.main()
