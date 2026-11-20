"""Compare the legacy decision fields for the 21 neutral/25%/30% attribution cases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_TRACE = ("coarse_candidates", "windows_searched", "final_candidates", "window_best_scores")
CAPACITY = 16


def bits(value):
    if isinstance(value, float):
        return struct.pack(">d", value).hex()
    if isinstance(value, (list, tuple)):
        return [bits(item) for item in value]
    return value


def snapshot(case_id, frame, hits, trace, matched):
    return {
        "id": case_id,
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "matched": matched,
        "count": len(hits),
        "hits": [{key: bits(value) for key, value in asdict(hit).items()} for hit in hits],
        **{key: bits(trace[key]) for key in OLD_TRACE},
    }


def case_result(model, case, generate, threshold, tolerance):
    frame, truth = generate.render_frame(
        case["seed"], occlusion_ratio=case["ratio"], occlusion_anchor=case["anchor"])
    hits = model.match(frame, threshold=threshold, max_candidates=CAPACITY)
    trace = model.last_match_info()
    best = max(hits, key=lambda hit: hit.score, default=None)
    matched = best is not None and best.score >= threshold and math.dist(
        best.center, truth["center"]) <= tolerance
    return snapshot(case["id"], frame, hits, trace, matched)


def pair_result(model, generate, threshold, tolerance, clean):
    import numpy as np

    if clean:
        left, _ = generate.render_frame(0)
        right, _ = generate.render_frame(1)
    else:
        left, _ = generate.render_frame(0, occlusion_ratio=0.25, occlusion_anchor="center")
        right, _ = generate.render_frame(0, occlusion_ratio=0.40, occlusion_anchor="center")
    x, y = map(int, generate.TARGET_CENTER)
    frame = np.full_like(left, generate.BG_BASE)
    half = 176
    centers = (480, 1440)
    for src, cx in zip((left, right), centers, strict=True):
        frame[y-half:y+half, cx-half:cx+half] = src[y-half:y+half, x-half:x+half]
    hits = model.match(frame, threshold=threshold, max_candidates=2)
    trace = model.last_match_info()
    near = [[i for i, hit in enumerate(hits) if math.dist(hit.center, (cx, y)) <= tolerance]
            for cx in centers]
    row = snapshot("clean/two" if clean else "mixed/25+40", frame, hits, trace, bool(near[0]))
    return dict(row, near=near, scores=[hit.score for hit in hits])


def rescue_trace(trace):
    return {"collect": trace.get("window_collect_best_scores"),
            "rescued": trace.get("rescued_candidates")}


def rescue_result(model, generate, threshold, tolerance):
    frame, truth = generate.render_frame(0, occlusion_ratio=0.40, occlusion_anchor="center")
    hits = model.match(frame, threshold=threshold, max_candidates=CAPACITY)
    trace = model.last_match_info()
    matched = any(hit.score >= threshold and math.dist(hit.center, truth["center"]) <= tolerance
                  for hit in hits)
    return dict(snapshot("rescue/40", frame, hits, trace, matched), **rescue_trace(trace))


def worker(train: bool, mode: str | None) -> int:
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import cv2
    from occlusion_attribution import build_matrix, generate, run_baseline, NativeModel
    from shape_sdk.native_library import runtime_directory

    cv2.setNumThreads(1)
    model_bytes = run_baseline.train_reference_model()[0] if train else sys.stdin.buffer.read()
    matrix = [case for case in build_matrix(10) if case["ratio"] <= 0.30]
    with NativeModel(model_bytes) as model:
        threshold = run_baseline.THRESHOLD
        tolerance = run_baseline.CENTER_TOLERANCE_PX
        if mode == "--mixed":
            rows = [pair_result(model, generate, threshold, tolerance, clean) for clean in (False, True)]
        elif mode == "--rescue":
            rows = [rescue_result(model, generate, threshold, tolerance)]
        else:
            rows = [case_result(model, case, generate, threshold, tolerance) for case in matrix]
    runtime = runtime_directory().resolve()
    result = {"runtime": str(runtime), "rows": rows,
              "dll_sha256": hashlib.sha256((runtime / "visionforge_shape.dll").read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(model_bytes).hexdigest()}
    if train:
        result["model"] = base64.b64encode(model_bytes).decode("ascii")
    sys.stdout.write(json.dumps(result))
    return 0


def run_worker(runtime: Path, asset: bytes | None, mode: str | None) -> dict:
    env = {**os.environ, "SHAPE_SDK_RUNTIME": str(runtime), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1"}
    command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
               "--worker", "train" if asset is None else "load"]
    if mode:
        command.append(mode)
    completed = subprocess.run(
        command, input=asset or b"", env=env, cwd=ROOT, capture_output=True, timeout=120, check=False)
    if completed.returncode:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise SystemExit(f"Worker for {runtime} failed ({completed.returncode}): {stderr}")
    result = json.loads(completed.stdout)
    if Path(result["runtime"]) != runtime:
        raise SystemExit(f"Unexpected runtime: {result['runtime']} != {runtime}")
    return result


def differences(old: dict, new: dict) -> list[str]:
    changed = [key for key in ("matched", "count", *OLD_TRACE) if old[key] != new[key]]
    for index, (a, b) in enumerate(zip(old["hits"], new["hits"])):
        for key in a:
            if a[key] != b[key]:
                values = ""
                if key == "score":
                    left, right = (struct.unpack(">d", bytes.fromhex(v))[0] for v in (a[key], b[key]))
                    values = f" ({left:.12g} -> {right:.12g})"
                changed.append(f"hits[{index}].{key}{values}")
    return changed


def compare_pair(old: dict, new: dict) -> int:
    clean = old["id"] == "clean/two"
    expected_count = 2 if clean else 1
    target_indices = sorted(i for indices in old["near"] for i in indices)
    if (old["count"] != expected_count or target_indices != list(range(expected_count))
            or (clean and any(len(indices) != 1 for indices in old["near"]))
            or (not clean and old["near"] != [[0], []])):
        raise SystemExit(f"Legacy pair fixture did not qualify: {old['id']} {old['near']}")
    changed = differences(old, new)
    sys.stdout.write(f"{old['id']}: old scores={old['scores']} new scores={new['scores']}; "
                     f"legacy differences={changed}\n")
    if not clean:
        sys.stdout.write("Additional mixed-target recovery is a separate study, not a default-path gate.\n")
    return int(bool(changed))


def compare_rescue(old: dict, new: dict) -> int:
    if old["count"] or not old["coarse_candidates"] or not old["windows_searched"]:
        raise SystemExit("Rescue fixture must be an actual legacy fine miss, not a coarse miss")
    legacy_windows_kept = all(old[key] == new[key] for key in OLD_TRACE if key != "final_candidates")
    observed = new["collect"] is not None and new["rescued"] > 0 and any(
        42.0 <= score < 70.0 for score in new["collect"])
    count_kept = new["count"] == new["final_candidates"] == 1
    sys.stdout.write(f"{old['id']}: matched={new['matched']} legacy windows={legacy_windows_kept} "
                     f"count={new['count']} collect={new['collect']} rescued={new['rescued']}\n")
    return int(not all((new["matched"], legacy_windows_kept, observed, count_kept)))


def compare(old: dict, new: dict, mode: str | None) -> int:
    pairs = list(zip(old["rows"], new["rows"], strict=True))
    expected = {"--mixed": 2, "--rescue": 1}.get(mode, 21)
    if len(pairs) != expected or len({a["id"] for a, _ in pairs}) != expected:
        raise SystemExit(f"Expected {expected} distinct cases")
    if old["model_sha256"] != new["model_sha256"]:
        raise SystemExit("Model bytes differ")
    sys.stdout.write(f"old {old['dll_sha256']} {old['runtime']}\n"
                     f"new {new['dll_sha256']} {new['runtime']}\n"
                     f"shared model {old['model_sha256']}\n")
    for a, b in pairs:
        if a["id"] != b["id"] or a["frame_sha256"] != b["frame_sha256"]:
            raise SystemExit(f"Input differs for {a['id']}")
    if mode == "--mixed":
        return max(compare_pair(a, b) for a, b in pairs)
    if mode == "--rescue":
        return compare_rescue(*pairs[0])
    failed = legacy_matches = 0
    for a, b in pairs:
        legacy_matches += a["matched"]
        changed = differences(a, b)
        if not a["matched"] or changed:
            failed += 1
            sys.stdout.write(f"{a['id']}: {', '.join(changed) if a['matched'] else 'legacy did not match'}\n")
    sys.stdout.write(f"Legacy matches {legacy_matches}/21; bitwise equivalent {21 - failed}/21\n")
    return int(failed != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-runtime", type=Path, default=ROOT / "dist" / "runtime")
    parser.add_argument("--new-runtime", type=Path)
    parser.add_argument("--worker", choices=("train", "load"), help=argparse.SUPPRESS)
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument("--mixed", action="store_true", help="compare default behavior for mixed and clean pairs")
    cases.add_argument("--rescue", action="store_true", help="check v2 separately on one genuine fine miss")
    args = parser.parse_args()
    mode = "--mixed" if args.mixed else "--rescue" if args.rescue else None
    if args.worker:
        return worker(args.worker == "train", mode)
    if args.new_runtime is None:
        parser.error("--new-runtime is required")
    old_runtime, new_runtime = args.old_runtime.resolve(), args.new_runtime.resolve()
    if old_runtime == new_runtime:
        parser.error("the two runtimes must be different directories")
    old = run_worker(old_runtime, None, mode)
    new = run_worker(new_runtime, base64.b64decode(old.pop("model"), validate=True), mode)
    if old["dll_sha256"] == new["dll_sha256"]:
        raise SystemExit("The two DLLs are identical; no differential evidence")
    return compare(old, new, mode)


if __name__ == "__main__":
    raise SystemExit(main())
