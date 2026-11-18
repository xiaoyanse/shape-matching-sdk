"""形状匹配 SDK Python 接入最小样例：合成图训练 → 加载 → 元数据 → 单帧匹配 → 批量匹配。

运行前提（仓库根目录）：
    cd bindings/python && uv sync
    uv run python ../../examples/python/train_match_batch.py

运行时定位：优先 SHAPE_SDK_RUNTIME 环境变量（制品目录，含 visionforge_shape.dll
与 runtime-manifest.json），缺省自动发现仓内 dist/runtime。消费方打包分发时
随制品携带 shape_sdk 包并显式设置该环境变量。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings" / "python"))

from shape_sdk import NativeModel, NativeShapeError, match_batch, train_model  # noqa: E402

METADATA_KEYS = (
    "reference_width", "reference_height", "roi_xywh",
    "angle_min", "angle_max", "angle_step", "scale_min", "scale_max", "scale_step",
    "weak_th", "strong_th", "feature_count",
)


def synth_reference() -> np.ndarray:
    """黑底白框 + 内部块的参考图（与主链回归同画法，进程内构造，无外部素材）。"""
    a = np.zeros((320, 320, 3), np.uint8)
    a[42:198, 42:48] = 255
    a[42:48, 42:198] = 255
    a[192:198, 42:198] = 255
    a[42:198, 192:198] = 255
    a[65:95, 65:95] = 255
    a[120:155, 90:155] = 200
    a[80:180, 165:171] = 255
    return np.ascontiguousarray(a)


def make_frame(reference: np.ndarray, offset: tuple[int, int], size: int = 384) -> np.ndarray:
    """把参考图整体平移贴到大画布上，作为带目标的待匹配帧。"""
    frame = np.zeros((size, size, 3), np.uint8)
    dx, dy = offset
    h, w = reference.shape[:2]
    frame[dy:dy + h, dx:dx + w] = reference
    return np.ascontiguousarray(frame)


def first_fine_segment(container: bytes) -> bytes:
    """从容器的首个 fine 段取旧单级格式资产（单级模型只来自旧格式，见头文件合同）。"""
    assert container[:5] == b"VF2S3", "容器 magic 不符"
    count = struct.unpack_from("<Q", container, 5)[0]
    start = 13 + 8 * (count + 3)
    length = struct.unpack_from("<Q", container, 13)[0]
    return container[start:start + length]


def train_defaults(reference: np.ndarray) -> bytes:
    return train_model(reference, (32, 32, 176, 176), angle_min=-20, angle_max=20,
                       angle_step=8, scale_min=0.8, scale_max=1.2, scale_step=0.2)


def train_custom(reference: np.ndarray) -> bytes:
    return train_model(reference, (32, 32, 176, 176), angle_min=0, angle_max=0, angle_step=1,
                       scale_min=1, scale_max=1, scale_step=1, weak_th=20, strong_th=40,
                       feature_count=32)


def section_train_and_load() -> NativeModel:
    reference = synth_reference()
    data = train_defaults(reference)
    print(f"[1] 训练：两级容器模型 {len(data)} 字节（ROI (32,32,176,176)，±20°/8°，0.8–1.2x）")
    model = NativeModel(data)
    meta = model.metadata
    missing = [key for key in METADATA_KEYS if key not in meta]
    assert not missing, f"metadata 缺键 {missing}"
    print(f"[2] 元数据：参考图 {meta['reference_width']}x{meta['reference_height']}，"
          f"ROI {meta['roi_xywh']}，角度 [{meta['angle_min']},{meta['angle_max']}]/"
          f"{meta['angle_step']}°，缩放 [{meta['scale_min']},{meta['scale_max']}]/"
          f"{meta['scale_step']}，特征筛选 ({meta['feature_count']},{meta['weak_th']},"
          f"{meta['strong_th']})（0=默认三参 128/30/60 的物化值）")
    return model


def section_single_match(model: NativeModel) -> None:
    offset = (36, 24)
    frame = make_frame(synth_reference(), offset)
    hits = model.match(frame, threshold=60, max_candidates=16)
    assert hits, "平移帧应命中，实际 0 个"
    best = max(hits, key=lambda c: c.score)
    expected = (119.5 + offset[0], 119.5 + offset[1])
    assert abs(best.center[0] - expected[0]) < 4 and abs(best.center[1] - expected[1]) < 4, \
        f"命中中心 {best.center} 偏离预期 {expected}"
    print(f"[3] 单帧匹配：{len(hits)} 个候选，最佳 score={best.score:.1f} "
          f"center=({best.center[0]:.1f},{best.center[1]:.1f})（预期近 "
          f"({expected[0]:.1f},{expected[1]:.1f})），angle={best.angle:.1f}° "
          f"scale={best.scale:.2f} ambiguous={best.pose_ambiguous}")

    empty = model.match(np.zeros((384, 384, 3), np.uint8), threshold=60)
    print(f"[3] 负样本帧（纯背景）：{len(empty)} 个命中（预期 0，无目标不错报）")


def section_batch(reference: np.ndarray) -> None:
    two_level = NativeModel(train_defaults(reference))
    custom = NativeModel(train_custom(reference))
    single_level = NativeModel(first_fine_segment(train_custom(reference)))
    try:
        results = match_batch(
            [(two_level, 60, 16), (custom, 70, 8), (single_level, 60, 16)],
            make_frame(reference, (36, 24)))
        assert isinstance(results, list) and len(results) == 3
        for index, result in enumerate(results):
            assert isinstance(result, list) and result, f"批量项 {index} 应命中：{result!r}"
        print(f"[4] 批量匹配（两级+两级自定义+单级，同帧）：3/3 项命中，逐项阈值 60/70/60")

        single_level.close()
        mixed = match_batch([(two_level, 60, 16), (single_level, 60, 16)],
                            make_frame(reference, (36, 24)))
        assert isinstance(mixed[0], list) and mixed[0], "成功项不应受失败项影响"
        assert isinstance(mixed[1], NativeShapeError), f"已关闭项应独立报错：{mixed[1]!r}"
        print(f"[4] 逐项隔离：已关闭模型项报错「{mixed[1]}」，成功项不受影响")
    finally:
        two_level.close()
        custom.close()
        single_level.close()


def section_lifecycle() -> None:
    try:
        NativeModel(b"bad")
        raise AssertionError("坏字节应被拒绝")
    except NativeShapeError as exc:
        print(f"[5] 生命周期：坏字节被拒（{exc}）")
    model = NativeModel(train_custom(synth_reference()))
    model.close()
    model.close()
    try:
        model.match(synth_reference())
        raise AssertionError("关闭后匹配应报错")
    except NativeShapeError as exc:
        print(f"[5] 生命周期：close 幂等，关闭后调用报错「{exc}」")


def main() -> int:
    reference = synth_reference()
    with section_train_and_load() as model:
        section_single_match(model)
    section_batch(reference)
    section_lifecycle()
    print("样例完成：训练 / 元数据 / 单帧匹配 / 批量匹配 / 生命周期 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
