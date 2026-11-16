"""shape-facts v1 端到端基准：官方 Python 绑定 + 合成事实帧。

语义：帧 1 裁锚邻域作参考训练（±30°/5°、单尺度），对四帧匹配，
对照 spec 预期真值（锚中心/在场性）判定，并输出训练/逐帧耗时。
预期真值来自 shape-facts-v1/template-and-frame-spec-v1.json。

用法（需 numpy + opencv 的 Python 环境）：
    set SHAPE_SDK_RUNTIME=<制品目录>   # 缺省仓内 dist/runtime
    python benchmarks/run_facts.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bindings" / "python"))
from shape_sdk import NativeModel, NativeShapeError, train_model  # noqa: E402

FACTS = HERE / "shape-facts-v1"
SPEC = json.loads((FACTS / "template-and-frame-spec-v1.json").read_text(encoding="utf-8"))
THRESHOLD = float(SPEC["template_algorithm"]["score_threshold"])
CENTER_TOLERANCE_PX = 8.0  # 网格量化 4px 的两倍容差
CROP_HALF = 130
TEMPLATE_ROI = (30, 30, 200, 200)  # 裁剪邻域内居中图案（对齐 spec 200px+余量30）

FRAMES = {
    "1": "synthetic-f0001-positive.png",
    "2": "synthetic-f0002-anchor-absent-target-positive.png",
    "3": "synthetic-f0003-target-negative.png",
    "4": "synthetic-f0004-positive-new-pose.png",
}


def load_frame(frame_id: str) -> np.ndarray:
    image = cv2.imread(str(FACTS / "frames" / FRAMES[frame_id]), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"基准帧缺失：{FRAMES[frame_id]}")
    return image


def main() -> int:
    anchors = SPEC["frame_spec"]["anchor_center_and_angle"]
    reference = load_frame("1")
    center = anchors["1"][0]
    x0, y0 = center[0] - CROP_HALF, center[1] - CROP_HALF
    patch = np.ascontiguousarray(
        reference[y0 : y0 + CROP_HALF * 2, x0 : x0 + CROP_HALF * 2]
    )

    started = time.perf_counter()
    model_bytes = train_model(
        patch,
        TEMPLATE_ROI,
        angle_min=-30.0,
        angle_max=30.0,
        angle_step=5.0,
        scale_min=1.0,
        scale_max=1.0,
        scale_step=0.05,
    )
    train_ms = (time.perf_counter() - started) * 1000

    failures: list[str] = []
    print(f"训练耗时 {train_ms:.1f} ms（13 位姿），阈值 {THRESHOLD}")
    with NativeModel(model_bytes) as model:
        for frame_id, filename in FRAMES.items():
            frame = load_frame(frame_id)
            started = time.perf_counter()
            try:
                hits = model.match(frame, threshold=THRESHOLD)
            except NativeShapeError as error:
                failures.append(f"帧{frame_id} 匹配异常：{error}")
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000
            expected = anchors.get(frame_id)
            if expected is None:
                verdict = "PASS" if not hits else f"FAIL（误报 {len(hits)} 个）"
                detail = "预期无锚，实际无命中" if not hits else f"最高分 {hits[0].score:.1f}"
            else:
                if not hits:
                    verdict, detail = "FAIL（漏检）", "预期命中但无候选"
                else:
                    best = max(hits, key=lambda item: item.score)
                    offset = float(np.hypot(best.center[0] - expected[0][0], best.center[1] - expected[0][1]))
                    verdict = "PASS" if offset <= CENTER_TOLERANCE_PX else "FAIL（定位超差）"
                    detail = (
                        f"中心偏差 {offset:.2f}px（容差 {CENTER_TOLERANCE_PX}），"
                        f"分数 {best.score:.1f}，角度 {best.angle:.1f}°"
                    )
            if verdict.startswith("FAIL"):
                failures.append(f"帧{frame_id} {verdict}：{detail}")
            print(f"帧{frame_id}（{filename}）：{verdict} — {detail}，耗时 {elapsed_ms:.1f} ms")

    if failures:
        print(f"\n基准未通过：{len(failures)} 项失败")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\n基准通过：4 帧判定全部符合预期真值")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
