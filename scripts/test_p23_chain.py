"""P2/P3 合同测试：suggest_contours_ex 参数化建议 + estimate_training_ex 多边形预算。

用法（在含 numpy 的 Python 环境）：
    set SHAPE_SDK_RUNTIME=<制品目录>   # 缺省仓内 dist/runtime
    python scripts/test_p23_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings" / "python"))

from shape_sdk import (  # noqa: E402
    ERROR_CODES,
    NativeShapeError,
    estimate_training,
    estimate_training_ex,
    last_error_info,
    suggest_contours,
    suggest_contours_ex,
)


def suggest_scene() -> np.ndarray:
    """400×400 黑底：亮矩形 (60..160)² 与暗矩形 gray=20 (240..340)²。"""
    scene = np.zeros((400, 400, 3), np.uint8)
    scene[60:161, 60:181, :] = 255
    scene[240:341, 240:361, :] = 20
    return scene


def circle_scene() -> np.ndarray:
    """400×400 黑底：圆心 (200,200) 半径 60 的白色圆盘（epsilon 灵敏素材）。"""
    yy, xx = np.mgrid[0:400, 0:400]
    scene = np.zeros((400, 400, 3), np.uint8)
    scene[(yy - 200) ** 2 + (xx - 200) ** 2 <= 3600, :] = 255
    return scene


def group_suggest_ex_defaults() -> None:
    scene = suggest_scene()
    v1 = suggest_contours(scene)
    ex = suggest_contours_ex(scene)
    assert v1 == ex, "默认参数 _ex 与 v1 输出不一致"
    assert len(v1) >= 1, "亮矩形应产出建议"

    circle = circle_scene()
    assert suggest_contours(circle) == suggest_contours_ex(circle), \
        "圆盘素材默认参数输出不一致"


def group_suggest_ex_canny() -> None:
    scene = suggest_scene()
    default = suggest_contours_ex(scene)
    assert len(default) == 1, f"默认阈值应只见亮矩形，实际 {len(default)} 条"
    dim = suggest_contours_ex(scene, canny_low=8.0, canny_high=24.0)
    assert len(dim) >= 2, f"低阈值应见暗矩形（≥2 条），实际 {len(dim)} 条"
    again = suggest_contours_ex(scene, canny_low=8.0, canny_high=24.0)
    assert dim == again, "低阈值两次调用输出不一致（确定性破坏）"
    assert len(suggest_contours_ex(scene, canny_low=0.0, canny_high=0.0)) == 1, \
        "0=默认语义失效"


def group_suggest_ex_epsilon() -> None:
    circle = circle_scene()
    # max_vertices_per=32 放开顶点上限，避免逼近循环加粗 epsilon 掩盖参数效果
    default = suggest_contours_ex(circle, max_vertices_per=32)
    fine = suggest_contours_ex(circle, max_vertices_per=32, epsilon_factor=0.005)
    assert len(default) >= 1 and len(fine) >= 1, "圆盘应有建议"
    default_vertices = max(len(s) for s in default)
    fine_vertices = max(len(s) for s in fine)
    assert fine_vertices > default_vertices, \
        f"小 epsilon 应更细（{fine_vertices} 顶点 vs 默认 {default_vertices}）"


def group_suggest_ex_reject() -> None:
    scene = suggest_scene()
    code_detector = ERROR_CODES["DETECTOR_PARAMS"]
    assert code_detector == 8, "DETECTOR_PARAMS 码值漂移"
    cases = [
        dict(canny_low=-1.0),
        dict(canny_low=5.0, canny_high=2.0),
        dict(epsilon_factor=0.0005),
        dict(epsilon_factor=0.5),
    ]
    for bad in cases:
        try:
            suggest_contours_ex(scene, **bad)
            raise AssertionError(f"非法参数应拒绝：{bad}")
        except NativeShapeError:
            code, params = last_error_info()
            assert code == code_detector, f"错误码应为 8，实际 {code}（{bad}）"
            assert params, f"缺数值上下文（{bad}）"
    try:
        suggest_contours_ex(scene, canny_low=float("nan"))
        raise AssertionError("NaN 参数应拒绝")
    except NativeShapeError:
        pass


ESTIMATE_SIZE = (480, 480)
ANGLE_KW = dict(
    angle_min=-30.0, angle_max=30.0, angle_step=15.0,
    scale_min=1.0, scale_max=1.0, scale_step=1.0,
)
LEDGER_KEYS = (
    "pose_used", "pose_limit", "pose_over", "canvas_used", "canvas_limit",
    "canvas_over", "pixel_used", "pixel_limit", "pixel_over", "over_budget",
)


def group_estimate_rect_equivalence() -> None:
    rect = (90, 90, 220, 220)
    v1 = estimate_training(ESTIMATE_SIZE, rect, **ANGLE_KW)
    assert v1["formula_version"] == 1, f"v1 公式版本应 1，实际 {v1['formula_version']}"
    ex_rect = estimate_training_ex(ESTIMATE_SIZE, roi_xywh=rect, **ANGLE_KW)
    assert ex_rect == v1, "_ex 矩形路径与 v1 逐字段不一致"
    assert ex_rect["polygon_fill_ratio"] == 1.0, "矩形 fill ratio 应 1.0"

    # 矩形缺 ROI / 多边形带 ROI 的互斥校验
    for bad in (dict(), dict(outline=[(90.0, 90.0), (310.0, 90.0), (310.0, 310.0)],
                             roi_xywh=(90, 90, 220, 220))):
        try:
            estimate_training_ex(ESTIMATE_SIZE, **bad)
            raise AssertionError(f"非法入参应拒绝：{bad}")
        except NativeShapeError:
            pass


def group_estimate_polygon_ledger() -> None:
    square = [(90.0, 90.0), (310.0, 90.0), (310.0, 310.0), (90.0, 310.0)]
    square_report = estimate_training_ex(ESTIMATE_SIZE, square, **ANGLE_KW)
    assert square_report["formula_version"] == 2, "多边形路径公式版本应 2"
    rect_report = estimate_training(ESTIMATE_SIZE, (90, 90, 220, 220), **ANGLE_KW)
    for key in LEDGER_KEYS:
        assert square_report[key] == rect_report[key], f"同式同序漂移：{key}"

    triangle = [(90.0, 90.0), (290.0, 90.0), (90.0, 290.0)]
    triangle_report = estimate_training_ex(ESTIMATE_SIZE, triangle, **ANGLE_KW)
    triangle_rect = estimate_training(ESTIMATE_SIZE, (90, 90, 200, 200), **ANGLE_KW)
    for key in LEDGER_KEYS:
        assert triangle_report[key] == triangle_rect[key], f"三角形账本漂移：{key}"
    assert abs(triangle_report["polygon_fill_ratio"] - 0.5) < 1e-9, \
        f"三角形 fill ratio 应 0.5，实际 {triangle_report['polygon_fill_ratio']}"
    assert abs(square_report["polygon_fill_ratio"] - 1.0) < 1e-9, "正方形 fill ratio 应 1.0"

    holed = [(90.0, 90.0), (310.0, 90.0), (310.0, 310.0), (90.0, 310.0)]
    holes = [[(140.0, 140.0), (180.0, 140.0), (180.0, 180.0), (140.0, 180.0)]]
    holed_report = estimate_training_ex(ESTIMATE_SIZE, holed, holes, **ANGLE_KW)
    want = (48400.0 - 1600.0) / 48400.0
    assert abs(holed_report["polygon_fill_ratio"] - want) < 1e-9, \
        f"带洞 fill ratio 手算不符：{holed_report['polygon_fill_ratio']} vs {want}"
    assert holed_report["formula_version"] == 2, "带洞路径公式版本应 2"


def group_estimate_polygon_reject() -> None:
    bowtie = [(100.0, 150.0), (300.0, 150.0), (200.0, 100.0), (200.0, 300.0)]
    try:
        estimate_training_ex(ESTIMATE_SIZE, bowtie, **ANGLE_KW)
        raise AssertionError("自交轮廓应被拒绝")
    except NativeShapeError:
        code, params = last_error_info()
        assert code == ERROR_CODES["INVALID_POLYGON"], f"错误码应为 17，实际 {code}"
        assert params and int(params[0]) == 4, f"细分类型应为 4，实际 {params}"


GROUPS = [
    ("py_suggest_ex_defaults", group_suggest_ex_defaults),
    ("py_suggest_ex_canny", group_suggest_ex_canny),
    ("py_suggest_ex_epsilon", group_suggest_ex_epsilon),
    ("py_suggest_ex_reject", group_suggest_ex_reject),
    ("py_estimate_rect_equivalence", group_estimate_rect_equivalence),
    ("py_estimate_polygon_ledger", group_estimate_polygon_ledger),
    ("py_estimate_polygon_reject", group_estimate_polygon_reject),
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
    print(f"P2/P3 分组：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
