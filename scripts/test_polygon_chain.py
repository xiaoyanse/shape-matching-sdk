"""官方绑定多边形合同测试：train_polygon→元数据往返→容器方言→suggest_contours。

用法（在含 numpy 的 Python 环境）：
    set SHAPE_SDK_RUNTIME=<制品目录>   # 缺省仓内 dist/runtime
    python scripts/test_polygon_chain.py
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings" / "python"))

from shape_sdk import (  # noqa: E402
    ERROR_CODES,
    NativeModel,
    NativeShapeError,
    last_error_info,
    suggest_contours,
    train_model,
    train_polygon_model,
)


def synth_reference() -> np.ndarray:
    """480×480 黑底：白色方框 (100..219)² 与内部灰块（形状与轮廓间留黑边距）。"""
    image = np.zeros((480, 480, 3), np.uint8)
    image[100:220, 100:220, :] = 255
    image[150:170, 150:170, :] = 90
    return image


def suggest_scene() -> np.ndarray:
    """400×400 黑底：两个分离白矩形（suggest 建议源）。"""
    scene = np.zeros((400, 400, 3), np.uint8)
    scene[60:161, 60:181, :] = 255
    scene[240:341, 240:361, :] = 255
    return scene


def split_container(data: bytes) -> list[bytes]:
    assert data[:5] == b"VF2S3", "容器 magic 不符"
    total = struct.unpack_from("<Q", data, 5)[0] + 3
    lengths = struct.unpack_from(f"<{total}Q", data, 13)
    cursor = 13 + 8 * total
    parts = []
    for length in lengths:
        parts.append(data[cursor:cursor + length])
        cursor += length
    return parts


def rebuild_container(parts: list[bytes]) -> bytes:
    out = bytearray(b"VF2S3")
    out += struct.pack("<Q", len(parts) - 3)
    for part in parts:
        out += struct.pack("<Q", len(part))
    for part in parts:
        out += part
    return bytes(out)


def strip_poly_key(container: bytes) -> bytes:
    parts = split_container(container)
    text = parts[-1].decode("utf-8")
    at = text.find('"poly":')
    assert at > 0 and text.endswith("}"), "meta 段缺少 poly 键"
    text = text[:at - 1] + "}"
    parts[-1] = text.encode("utf-8")
    return rebuild_container(parts)


def normalized(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """与 SDK 同式的绕向归一期望（有向面积为负时反序）。"""
    area = 0.0
    for index, (x, y) in enumerate(points):
        nx, ny = points[(index + 1) % len(points)]
        area += x * ny - nx * y
    return list(reversed(points)) if area < 0 else points


def vertices_equal(got, want) -> bool:
    return len(got) == len(want) and all(
        g[0] == w[0] and g[1] == w[1] for g, w in zip(got, want)
    )


def group_polygon_train() -> None:
    outline = [(90.0, 310.0), (310.0, 310.0), (310.0, 90.0), (90.0, 90.0)]
    data = train_polygon_model(
        synth_reference(), outline,
        angle_min=-30.0, angle_max=30.0, angle_step=15.0,
        scale_min=1.0, scale_max=1.0, scale_step=1.0,
    )
    model = NativeModel(data)
    try:
        hits = model.match(synth_reference(), threshold=70.0)
        assert len(hits) >= 1, f"多边形自匹配应命中，实际 {len(hits)}"
        assert hits[0].score >= 70.0, f"自匹配分应 ≥70，实际 {hits[0].score}"
        meta = model.polygon_metadata
        assert vertices_equal(meta["outline"], normalized(outline)), \
            f"绕向归一往返漂移：{meta['outline']}"
        assert meta["hole_count"] == 0, f"洞数应 0，实际 {meta['hole_count']}"
    finally:
        model.close()


def group_polygon_holed_and_reject() -> None:
    outline = [(90.0, 90.0), (230.0, 90.0), (230.0, 230.0), (90.0, 230.0)]
    holes = [[(140.0, 140.0), (180.0, 140.0), (180.0, 180.0), (140.0, 180.0)]]
    image = synth_reference()
    image[140:181, 140:181, :] = 0
    image[150:170, 150:170, :] = 90
    data = train_polygon_model(
        image, outline, holes,
        angle_min=-30.0, angle_max=30.0, angle_step=15.0,
        scale_min=1.0, scale_max=1.0, scale_step=1.0,
    )
    model = NativeModel(data)
    try:
        meta = model.polygon_metadata
        assert meta["hole_count"] == 1, f"洞数应 1，实际 {meta['hole_count']}"
        assert vertices_equal(meta["outline"], normalized(outline)), "带洞往返顶点漂移"
    finally:
        model.close()

    assert ERROR_CODES["INVALID_POLYGON"] == 17, "INVALID_POLYGON 码值漂移"
    bowtie = [(100.0, 150.0), (300.0, 150.0), (200.0, 100.0), (200.0, 300.0)]
    try:
        train_polygon_model(
            synth_reference(), bowtie,
            angle_min=-30.0, angle_max=30.0, angle_step=15.0,
            scale_min=1.0, scale_max=1.0, scale_step=1.0,
        )
        raise AssertionError("自交轮廓应被拒绝")
    except NativeShapeError:
        code, params = last_error_info()
        assert code == 17, f"错误码应为 17，实际 {code}"
        assert params and int(params[0]) == 4, f"细分类型应为 4，实际 {params}"
    try:
        train_polygon_model(
            synth_reference(), [(20.0, 20.0), (42.0, 20.0), (20.0, 42.0)],
            angle_min=-30.0, angle_max=30.0, angle_step=15.0,
            scale_min=1.0, scale_max=1.0, scale_step=1.0,
        )
        raise AssertionError("外接矩形<24px 应被拒绝")
    except NativeShapeError:
        code, params = last_error_info()
        assert code == 17 and params and int(params[0]) == 9, \
            f"外接矩形拒绝码/细分异常：{code} {params}"


def group_polygon_container() -> None:
    outline = [(90.0, 310.0), (310.0, 310.0), (310.0, 90.0), (90.0, 90.0)]
    holes = [[(180.0, 260.0), (220.0, 260.0), (220.0, 220.0), (180.0, 220.0)]]
    container = train_polygon_model(
        synth_reference(), outline, holes,
        angle_min=-30.0, angle_max=30.0, angle_step=15.0,
        scale_min=1.0, scale_max=1.0, scale_step=1.0,
    )
    reloaded = NativeModel(container)
    try:
        meta = reloaded.polygon_metadata
        assert vertices_equal(meta["outline"], normalized(outline)), "重载往返顶点漂移"
        assert meta["hole_count"] == 1, "重载洞数漂移"
    finally:
        reloaded.close()

    legacy = NativeModel(strip_poly_key(container))
    try:
        meta = legacy.polygon_metadata
        rect = [(90.0, 90.0), (310.0, 90.0), (310.0, 310.0), (90.0, 310.0)]
        assert vertices_equal(meta["outline"], rect), f"删poly外接矩形回退漂移：{meta['outline']}"
        assert meta["hole_count"] == 0, "删poly洞数应 0"
    finally:
        legacy.close()

    rect_model_bytes = train_model(
        synth_reference(), (42, 42, 156, 156),
        angle_min=-30.0, angle_max=30.0, angle_step=15.0,
        scale_min=1.0, scale_max=1.0, scale_step=1.0,
    )
    rect_asset = NativeModel(rect_model_bytes)
    try:
        meta = rect_asset.polygon_metadata
        roi = [(42.0, 42.0), (198.0, 42.0), (198.0, 198.0), (42.0, 198.0)]
        assert vertices_equal(meta["outline"], roi), "矩形资产 ROI 四角回退漂移"
        assert meta["hole_count"] == 0, "矩形资产洞数应 0"
    finally:
        rect_asset.close()


def group_suggest() -> None:
    scene = suggest_scene()
    first = suggest_contours(scene)
    assert len(first) >= 2, f"建议条数应 ≥2，实际 {len(first)}"
    for suggestion in first:
        assert 3 <= len(suggestion) <= 8, f"建议顶点数越 3..8：{len(suggestion)}"
    again = suggest_contours(scene)
    assert first == again, "两次 suggest 输出不一致（确定性破坏）"

    hinted = suggest_contours(scene, (30, 30, 220, 220))
    assert len(hinted) >= 1, "先验框内应有建议"
    for suggestion in hinted:
        for x, y in suggestion:
            assert 30 <= x <= 250 and 30 <= y <= 250, f"先验框建议顶点越框：({x},{y})"

    for bad in (
        dict(roi_hint=(300, 300, 200, 200)),
        dict(max_suggestions=0),
        dict(max_vertices_per=2),
    ):
        try:
            suggest_contours(scene, **bad)
            raise AssertionError(f"非法参数应拒绝：{bad}")
        except NativeShapeError:
            pass

    blank = np.zeros((200, 200, 3), np.uint8)
    assert suggest_contours(blank) == [], "空图应无建议（非错误）"


GROUPS = [
    ("py_polygon_train", group_polygon_train),
    ("py_polygon_holed_and_reject", group_polygon_holed_and_reject),
    ("py_polygon_container", group_polygon_container),
    ("py_suggest", group_suggest),
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
        except NativeShapeError as exc:
            results.append((name, False))
            print(f"[FAIL] {name}: 原生错误 {exc}")
    failed = [name for name, ok in results if not ok]
    print(f"多边形分组：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
