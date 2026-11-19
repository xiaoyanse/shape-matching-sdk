"""GPU 尖峰数据管道：fact 帧 1 训练锚模板（两级容器）并落盘最小链路数据。

与 benchmarks/run_facts.py 同 spec：帧 1 锚邻域裁剪，ROI(30,30,200,200)，
±30°/5° 单尺度，默认特征参数（128/30/60）。产出（build/gpu-spike/data/）：
  anchor-model.bin   官方 VF2S3 两级容器（C++ 侧自行解码，引擎零修改）
  spike-meta.json    编排参数：k/组界/偏置/各段位姿数（人读+工具读）
  anchor-summary.json 概要（人读校验）
另打印 OpenCV 7 抔高斯核（供 CUDA 侧硬编码对齐 vendor GaussianBlur 语义）。

用法：SHAPE_SDK_RUNTIME=dist/runtime uv run --project bindings/python \
      python benchmarks/gpu-spike/scripts/dump_model.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "bindings" / "python"))
from shape_sdk import train_model  # noqa: E402

FACTS = REPO / "benchmarks" / "shape-facts-v1"
OUT_DIR = REPO / "build" / "gpu-spike" / "data"
CROP_HALF = 130
TEMPLATE_ROI = (30, 30, 200, 200)
SINGLE_SIGNATURE = b"VFSHAPE1:3df935f8e4b8c8a8ed72984211543e1b7696452e:p3\x00"


class Reader:
    def __init__(self, blob: bytes, pos: int = 0):
        self.blob = blob
        self.pos = pos

    def u32(self) -> int:
        (value,) = struct.unpack_from("<I", self.blob, self.pos)
        self.pos += 4
        return value

    def u64(self) -> int:
        (value,) = struct.unpack_from("<Q", self.blob, self.pos)
        self.pos += 8
        return value

    def f64(self) -> float:
        (value,) = struct.unpack_from("<d", self.blob, self.pos)
        self.pos += 8
        return value


def decode_single_level(blob: bytes) -> dict:
    r = Reader(blob, len(SINGLE_SIGNATURE))
    if blob[: len(SINGLE_SIGNATURE)] != SINGLE_SIGNATURE or r.u32() != 1:
        raise SystemExit("单级段签名/schema 不符")
    width, height = r.u32(), r.u32()
    config = {"roi_xywh": (r.u32(), r.u32(), r.u32(), r.u32()),
              "angle": (r.f64(), r.f64(), r.f64()),
              "scale": (r.f64(), r.f64(), r.f64())}
    features, weak, strong = r.u32(), r.f64(), r.f64()
    poses = []
    for _ in range(r.u32()):
        pose = {"angle": r.f64(), "scale": r.f64(), "canvas": (r.u32(), r.u32()),
                "transform": [r.f64() for _ in range(6)]}
        pose["levels"] = []
        for _ in range(r.u32()):
            level = {"wh": (r.u32(), r.u32()), "tl": (r.u32(), r.u32()),
                     "pyramid_level": r.u32()}
            level["features"] = [
                {"x": r.u32(), "y": r.u32(), "label": r.u32(), "theta": r.f64()}
                for _ in range(r.u32())
            ]
            pose["levels"].append(level)
        poses.append(pose)
    if r.pos != len(blob):
        raise SystemExit("单级段尾部有残余字节")
    return {"size": (width, height), "config": config, "features": features,
            "weak": weak, "strong": strong, "poses": poses}


def decode_container(blob: bytes) -> dict:
    if blob[:5] != b"VF2S3":
        raise SystemExit("容器签名不符")
    r = Reader(blob, 5)
    n = r.u64()
    if not 0 < n <= 64:
        raise SystemExit("容器 fines 数量非法")
    lens = [r.u64() for _ in range(n + 3)]
    parts = []
    for length in lens:
        parts.append(blob[r.pos : r.pos + length])
        r.pos += length
    if r.pos != len(blob):
        raise SystemExit("容器尾部有残余字节")
    return {"fines": [decode_single_level(p) for p in parts[:n]],
            "coarse": decode_single_level(parts[n]),
            "meta_raw": parts[n + 2].decode("utf-8")}


def main() -> int:
    frame = cv2.imread(str(FACTS / "frames" / "synthetic-f0001-positive.png"), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit("fact 帧 1 缺失")
    center = (600, 420)
    x0, y0 = center[0] - CROP_HALF, center[1] - CROP_HALF
    patch = np.ascontiguousarray(frame[y0 : y0 + CROP_HALF * 2, x0 : x0 + CROP_HALF * 2])
    model_bytes = train_model(
        patch, TEMPLATE_ROI,
        angle_min=-30.0, angle_max=30.0, angle_step=5.0,
        scale_min=1.0, scale_max=1.0, scale_step=0.05,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "anchor-model.bin").write_bytes(model_bytes)
    decoded = decode_container(model_bytes)
    meta = json.loads(decoded["meta_raw"])

    def level_brief(model: dict) -> dict:
        return {"pose_count": len(model["poses"]),
                "detector": [model["features"], model["weak"], model["strong"]],
                "features_per_level_min": min(
                    len(l["features"]) for p in model["poses"] for l in p["levels"]),
                "features_per_level_max": max(
                    len(l["features"]) for p in model["poses"] for l in p["levels"]),
                "angle_range": [model["poses"][0]["angle"], model["poses"][-1]["angle"]],
                "angle_step": model["config"]["angle"][2]}

    spike_meta = {
        "k": meta["k"],
        "groups": [
            {"lo": g[0], "hi": g[1], "bias": list(b)}
            for g, b in zip(meta["groups"], meta["group_bias"])
        ],
        "coarse": level_brief(decoded["coarse"]),
        "fines": [level_brief(f) for f in decoded["fines"]],
        "gaussian7": cv2.getGaussianKernel(7, 0).ravel().tolist(),
        "model_bytes": len(model_bytes),
    }
    (OUT_DIR / "spike-meta.json").write_text(
        json.dumps(spike_meta, indent=1), encoding="utf-8")

    coarse = decoded["coarse"]
    print(f"容器 {len(model_bytes)} 字节：k={meta['k']}，"
          f"coarse {len(coarse['poses'])} 位姿（步长 {coarse['config']['angle'][2]}°），"
          f"{len(decoded['fines'])} 个 fine 组")
    for index, fine in enumerate(decoded["fines"]):
        group = spike_meta["groups"][index]
        print(f"  fine[{index}] 组界 [{group['lo']:+.0f},{group['hi']:+.0f}]° "
              f"{len(fine['poses'])} 位姿（步长 {fine['config']['angle'][2]}°，"
              f"特征/层 {spike_meta['fines'][index]['features_per_level_min']}"
              f"-{spike_meta['fines'][index]['features_per_level_max']}）")
    print(f"高斯核(7)：{spike_meta['gaussian7']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
