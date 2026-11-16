"""官方绑定主链合同测试：训练→元数据→往返→单/批量匹配→释放，合成图全内存。

用法（在含 numpy 的 Python 环境）：
    set SHAPE_SDK_RUNTIME=<制品目录>   # 缺省自动装配 build/tD/runtime
    python scripts/test_main_chain.py
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from ctypes import byref, c_uint64, create_string_buffer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPENCV_BIN = Path(os.environ.get("SHAPE_SDK_OPENCV_BIN", "C:/opencv/build/x64/vc16/bin"))
BUILD_DLLS = (
    ROOT / "build" / "tD" / "Release" / "visionforge_shape.dll",
    ROOT / "build" / "tD" / "visionforge_shape.dll",
)
ASSEMBLED_RUNTIME = ROOT / "build" / "tD" / "runtime"
DIST_MANIFEST = ROOT / "dist" / "runtime" / "runtime-manifest.json"
METADATA_KEYS = (
    "reference_width", "reference_height", "roi_xywh",
    "angle_min", "angle_max", "angle_step", "scale_min", "scale_max", "scale_step",
    "weak_th", "strong_th", "feature_count",
)


def assemble_runtime() -> None:
    """构建 DLL 与根 dist 清单不一致时自装配 build/tD/runtime（不写根 dist）。"""
    built = next((path for path in BUILD_DLLS if path.is_file()), None)
    if built is None:
        return
    digest = hashlib.sha256(built.read_bytes()).hexdigest()
    try:
        manifest = json.loads(DIST_MANIFEST.read_text(encoding="utf-8"))
        if manifest["files"].get("visionforge_shape.dll") == digest:
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    worlds = [p for p in OPENCV_BIN.glob("opencv_world*.dll") if not p.stem.endswith("d")]
    assert len(worlds) == 1, f"需要唯一 Release opencv_world DLL，实际 {len(worlds)} 个"
    sources = {
        "visionforge_shape.dll": built,
        worlds[0].name: worlds[0],
        "shape_based_matching.LICENSE": ROOT / "vendor" / "shape_based_matching" / "LICENSE",
        "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
        "OpenCV.LICENSE": OPENCV_BIN.parents[2] / "LICENSE",
    }
    ASSEMBLED_RUNTIME.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in sources.items():
        target = ASSEMBLED_RUNTIME / name
        target.write_bytes(source.read_bytes())
        files[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    revision = anchor = ""
    try:
        revision = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        anchor = subprocess.run(["git", "describe", "--tags", "--always"], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        pass
    payload = {"abi_version": 3, "platform": "windows-x64", "sdk_revision": revision,
               "anchor": anchor, "files": files}
    (ASSEMBLED_RUNTIME / "runtime-manifest.json").write_text(
        json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.environ.setdefault("SHAPE_SDK_RUNTIME", str(ASSEMBLED_RUNTIME))


assemble_runtime()

sys.path.insert(0, str(ROOT / "bindings" / "python"))

from shape_sdk import NativeModel, NativeShapeError, invoke, match_batch, train_model  # noqa: E402


def synth_reference() -> np.ndarray:
    a = np.zeros((320, 320, 3), np.uint8)
    a[42:198, 42:48] = 255
    a[42:48, 42:198] = 255
    a[192:198, 42:198] = 255
    a[42:198, 192:198] = 255
    a[65:95, 65:95] = 255
    a[120:155, 90:155] = 200
    a[80:180, 165:171] = 255
    return np.ascontiguousarray(a)


def first_fine_segment(container: bytes) -> bytes:
    assert container[:5] == b"VF2S3", "容器 magic 不符"
    count = struct.unpack_from("<Q", container, 5)[0]
    start = 13 + 8 * (count + 3)
    length = struct.unpack_from("<Q", container, 13)[0]
    return container[start:start + length]


def segment_detector(segment: bytes) -> tuple[int, float, float]:
    offset = segment.index(b"\0") + 1 + 12 + 64
    return struct.unpack_from("<Idd", segment, offset)


def model_export(model: NativeModel) -> bytes:
    size = c_uint64()
    invoke(model._library, "export", model._handle, None, 0, byref(size))
    buffer = create_string_buffer(int(size.value))
    invoke(model._library, "export", model._handle, buffer, size.value, byref(size))
    return buffer.raw[: size.value]


def train_defaults(reference: np.ndarray) -> bytes:
    return train_model(reference, (32, 32, 176, 176), angle_min=-20, angle_max=20,
                       angle_step=8, scale_min=0.8, scale_max=1.2, scale_step=0.2)


def train_custom(reference: np.ndarray) -> bytes:
    return train_model(reference, (32, 32, 176, 176), angle_min=0, angle_max=0, angle_step=1,
                       scale_min=1, scale_max=1, scale_step=1, weak_th=20, strong_th=40,
                       feature_count=32)


def detector_tuple(meta: dict) -> tuple:
    return (meta["feature_count"], meta["weak_th"], meta["strong_th"])


def group_full_chain() -> None:
    reference = synth_reference()
    data = train_defaults(reference)
    assert isinstance(data, bytes) and 0 < len(data) <= 32 * 1024 * 1024, \
        f"模型字节非法：{len(data)}"
    with NativeModel(data) as model:
        meta = model.metadata
        missing = [key for key in METADATA_KEYS if key not in meta]
        assert not missing, f"metadata 缺少键 {missing}，实际 {sorted(meta)}"
        assert meta["roi_xywh"] == (32, 32, 176, 176), f"ROI 回显失真：{meta['roi_xywh']}"
        hits = model.match(reference, threshold=60)
        assert len(hits) >= 1, f"同图匹配命中 {len(hits)} 应 ≥1"
        best = max(hits, key=lambda candidate: candidate.score)
        assert abs(best.center[0] - 119.5) < 40 and abs(best.center[1] - 119.5) < 40, \
            f"命中中心偏移：{best.center} 应近 (119.5,119.5)"


def group_metadata_roundtrip() -> None:
    reference = synth_reference()
    with NativeModel(train_defaults(reference)) as model:
        meta = model.metadata
        for key, expected in (("angle_min", -20), ("angle_max", 20), ("angle_step", 8),
                              ("scale_min", 0.8), ("scale_max", 1.2), ("scale_step", 0.2)):
            assert meta[key] == expected, \
                f"{key} 往返漂移：实际 {meta[key]} 期望 {expected}"


def group_detector_roundtrip() -> None:
    reference = synth_reference()
    container = train_custom(reference)
    expected = (32, 20.0, 40.0)
    first = segment_detector(first_fine_segment(container))
    assert first == expected, f"首次导出 detector 参数漂移：{first} 应为 {expected}"
    single1 = NativeModel(first_fine_segment(container))
    try:
        actual = detector_tuple(single1.metadata)
        assert actual == expected, f"单级一次加载 metadata 未回填：{actual} 应为 {expected}"
        reexport1 = segment_detector(model_export(single1))
        assert reexport1 == expected, f"一次往返导出漂移：{reexport1} 应为 {expected}"
        single2 = NativeModel(model_export(single1))
        try:
            actual2 = detector_tuple(single2.metadata)
            assert actual2 == expected, f"二次加载 metadata 未回填：{actual2} 应为 {expected}"
            reexport2 = segment_detector(model_export(single2))
            assert reexport2 == expected, f"二次往返导出漂移：{reexport2} 应为 {expected}"
        finally:
            single2.close()
    finally:
        single1.close()
    with NativeModel(container) as reloaded:
        actual_container = detector_tuple(reloaded.metadata)
        assert actual_container == expected, \
            f"两级容器 metadata 三字段漂移：{actual_container} 应为 {expected}"
        recontainer = segment_detector(first_fine_segment(model_export(reloaded)))
        assert recontainer == expected, \
            f"容器重导出首段参数漂移：{recontainer} 应为 {expected}"


def group_batch_mix() -> None:
    reference = synth_reference()
    two_bytes = train_defaults(reference)
    single_default = NativeModel(first_fine_segment(two_bytes))
    single_custom = NativeModel(first_fine_segment(train_custom(reference)))
    two_level = NativeModel(two_bytes)
    try:
        results = match_batch([(two_level, 60, 16), (single_default, 60, 16),
                               (single_custom, 60, 16)], reference)
        for index, result in enumerate(results):
            assert isinstance(result, list), f"批量项 {index} 失败：{result!r}"
            assert len(result) >= 1, f"批量项 {index} 命中 {len(result)} 应 ≥1"
        two_level.close()
        mixed = match_batch([(two_level, 60, 16), (single_default, 60, 16)], reference)
        assert isinstance(mixed[0], NativeShapeError), \
            f"关闭模型项应独立报错：{mixed[0]!r}"
        assert isinstance(mixed[1], list) and len(mixed[1]) >= 1, "关闭项不应影响成功项"
    finally:
        single_default.close()
        single_custom.close()
        two_level.close()


def group_lifecycle() -> None:
    try:
        NativeModel(b"bad")
        raise AssertionError("坏字节应被拒绝")
    except NativeShapeError as exc:
        assert "invalid model size" in str(exc), f"坏字节错误文本异常：{exc}"
    model = NativeModel(train_custom(synth_reference()))
    model.close()
    model.close()
    try:
        model.match(synth_reference())
        raise AssertionError("关闭后匹配应报错")
    except NativeShapeError as exc:
        assert "已关闭" in str(exc), f"关闭后错误文本异常：{exc}"


GROUPS = [
    ("py_full_chain", group_full_chain),
    ("py_metadata_roundtrip", group_metadata_roundtrip),
    ("py_detector_roundtrip", group_detector_roundtrip),
    ("py_batch_mix", group_batch_mix),
    ("py_lifecycle", group_lifecycle),
]


def main() -> int:
    results = []
    for name, fn in GROUPS:
        try:
            fn()
            results.append((name, True))
            print(f"[PASS] {name}")
        except AssertionError as exc:
            results.append((name, False))
            print(f"[FAIL] {name}: {exc}")
    failed = [name for name, ok in results if not ok]
    print(f"主链分组：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
