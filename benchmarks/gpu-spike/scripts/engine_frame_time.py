"""引擎全帧匹配耗时基线（生产 CPU 实现，SIMD+OMP）——GPU 尖峰报告对照用。

同数据同模型：fact 帧 1 双线性放大到 1920x1080，官方绑定 match，预热 10 取
后续 30 次中位。用法：
    SHAPE_SDK_RUNTIME=dist/runtime uv run --project bindings/python \
        python benchmarks/gpu-spike/scripts/engine_frame_time.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bindings" / "python"))
from shape_sdk import NativeModel  # noqa: E402

FACTS = REPO / "benchmarks" / "shape-facts-v1"
MODEL = REPO / "build" / "gpu-spike" / "data" / "anchor-model.bin"
THRESHOLD = 70.0


def bench(model: NativeModel, image: np.ndarray, tag: str) -> None:
    for _ in range(10):
        model.match(image, threshold=THRESHOLD)
    times = []
    for _ in range(30):
        started = time.perf_counter()
        hits = model.match(image, threshold=THRESHOLD)
        times.append((time.perf_counter() - started) * 1000)
    top = "（无命中——锚超出训练尺度属预期）"
    if hits:
        best = max(hits, key=lambda h: h.score)
        top = (f"top：分数 {best.score:.1f} 角 {best.angle:.1f}° 中心 "
               f"({best.center[0]:.1f}, {best.center[1]:.1f}) 命中 {len(hits)}")
    print(f"引擎 match {tag} 中位 {statistics.median(times):.2f} ms "
          f"（min {min(times):.2f} max {max(times):.2f}，30 次）{top}")


def main() -> int:
    frame = cv2.imread(str(FACTS / "frames" / "synthetic-f0001-positive.png"), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit("fact 帧 1 缺失")
    big = np.ascontiguousarray(
        cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR))
    with NativeModel(MODEL.read_bytes()) as model:
        bench(model, np.ascontiguousarray(frame), "1280x960 原生")
        bench(model, big, "1920x1080 放大(锚超尺度)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
