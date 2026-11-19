"""诊断三件套绑定合同测试：错误码+数值参数、预算查询边界一致、匹配诊断 trace。

运行前提：SHAPE_SDK_RUNTIME 指向含新导出的制品目录（如 build/t2a/runtime），
本脚本只读校验 manifest 哈希链，不装配不写盘：
    set SHAPE_SDK_RUNTIME=<制品目录>
    python scripts/test_diagnostics_chain.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def runtime_intact(runtime_dir: Path) -> bool:
    """目录 manifest 可读且声明文件逐项哈希吻合才算可用，与绑定加载同口径。"""
    try:
        manifest = json.loads((runtime_dir / "runtime-manifest.json").read_text(encoding="utf-8"))
        return all(hashlib.sha256((runtime_dir / name).read_bytes()).hexdigest() == digest
                   for name, digest in manifest["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def assemble_free_runtime() -> Path:
    explicit = os.environ.get("SHAPE_SDK_RUNTIME")
    if explicit and runtime_intact(Path(explicit)):
        return Path(explicit)
    raise SystemExit("须先设置 SHAPE_SDK_RUNTIME 指向含新导出的制品目录（如 build/t2a/runtime）")


RUNTIME = assemble_free_runtime()
sys.path.insert(0, str(ROOT / "bindings" / "python"))

import shape_sdk as sdk  # noqa: E402
from shape_sdk import (  # noqa: E402
    NativeModel,
    NativeShapeError,
    error_message,
    estimate_training,
    last_error_info,
    train_model,
)


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


def train_defaults(reference: np.ndarray) -> bytes:
    return train_model(reference, (32, 32, 176, 176), angle_min=-20, angle_max=20,
                       angle_step=8, scale_min=0.8, scale_max=1.2, scale_step=0.2)


def expect_coded(code: int, min_params: int, what: str) -> tuple[int, tuple[float, ...]]:
    got, params = last_error_info()
    assert got == code, f"{what} 错误码漂移：{got}（{error_message(got, params)}）应为 {code}"
    assert min_params <= len(params) <= 6, f"{what} 参数个数越界：{len(params)}"
    return got, params


def group_error_codes() -> None:
    reference = synth_reference()
    try:
        train_model(reference, (32, 32, 176, 176), angle_min=-20, angle_max=20,
                    angle_step=0.05, scale_min=1, scale_max=1, scale_step=1)
        raise AssertionError("角度步过细应被拒绝")
    except NativeShapeError as exc:
        assert "pose budget exceeded" in str(exc), f"角度步过细错误文本异常：{exc}"
    code, params = expect_coded(sdk.ERROR_CODES["INVALID_ANGLE_RANGE"], 3, "角度步过细")
    assert params[0] == 0.05, f"参数[0]应为当前步长 0.05：{params}"
    assert params[1] == 350.0, f"参数[1]应为档数上限 350：{params}"
    assert abs(params[2] - 40.0 / 349.0) < 1e-9, f"参数[2]应为建议步长：{params}"
    text = error_message(code, params)
    assert "建议步长" in text, f"人话文案应含建议步长提示：{text}"

    try:
        train_model(reference, (32, 32, 176, 176), angle_min=0, angle_max=0, angle_step=1,
                    scale_min=0.25, scale_max=2, scale_step=0.001)
        raise AssertionError("缩放步过细应被拒绝")
    except NativeShapeError as exc:
        assert "pose budget exceeded" in str(exc), f"缩放步过细错误文本异常：{exc}"
    expect_coded(sdk.ERROR_CODES["INVALID_SCALE_RANGE"], 3, "缩放步过细")

    try:
        train_model(reference, (300, 300, 100, 100), angle_min=0, angle_max=0, angle_step=1,
                    scale_min=1, scale_max=1, scale_step=1)
        raise AssertionError("ROI 越界应被拒绝")
    except NativeShapeError as exc:
        assert "invalid training ROI" in str(exc), f"ROI 错误文本异常：{exc}"
    expect_coded(sdk.ERROR_CODES["INVALID_ROI"], 2, "ROI 越界")

    model = NativeModel(train_defaults(reference))
    try:
        hits = model.match(reference, threshold=60)
        assert len(hits) >= 1, "同图匹配应命中"
        # 成功调用不清除上次失败记录（上一次失败是 ROI 越界）
        expect_coded(sdk.ERROR_CODES["INVALID_ROI"], 2, "成功匹配后记录保留")
        trace = model.last_match_info()
        assert trace["final_candidates"] == len(hits), f"trace 最终候选数不符：{trace}"
        assert trace["coarse_candidates"] >= 1 and trace["windows_searched"] >= 1, \
            f"两级 trace 计数漂移：{trace}"
    finally:
        model.close()


def group_budget_boundary() -> None:
    reference = synth_reference()
    big = np.zeros((1600, 1600, 3), np.uint8)
    big[600:756, 600:656] = 255
    big[600:656, 600:756] = 255
    big[700:756, 600:756] = 255
    big = np.ascontiguousarray(big)

    def estimate(image: np.ndarray, roi, **kw) -> dict:
        angles = {k: kw.pop(k) for k in ("angle_min", "angle_max", "angle_step")}
        scales = {k: kw.pop(k) for k in ("scale_min", "scale_max", "scale_step")}
        return estimate_training((image.shape[1], image.shape[0]), roi,
                                 **angles, **scales, **kw)

    # 姿态账本压线/超限 ↔ 训练接受/拒绝
    report = estimate(reference, (32, 32, 176, 176), angle_min=-87, angle_max=87,
                      angle_step=1, scale_min=0.5, scale_max=1.0, scale_step=0.5)
    assert report["formula_version"] == 1, f"公式版本应为 1：{report}"
    assert report["pose_used"] == 350 and report["over_budget"] == 0, f"压线报告漂移：{report}"
    model = NativeModel(train_model(reference, (32, 32, 176, 176), angle_min=-87, angle_max=87,
                                    angle_step=1, scale_min=0.5, scale_max=1.0, scale_step=0.5))
    model.close()

    report = estimate(reference, (32, 32, 176, 176), angle_min=-87.5, angle_max=87.5,
                      angle_step=1, scale_min=0.5, scale_max=1.0, scale_step=0.5)
    assert report["pose_over"] and report["over_budget"] == sdk.ERROR_CODES["POSE_BUDGET"], \
        f"超限报告漂移：{report}"
    try:
        train_model(reference, (32, 32, 176, 176), angle_min=-87.5, angle_max=87.5,
                    angle_step=1, scale_min=0.5, scale_max=1.0, scale_step=0.5)
        raise AssertionError("超限训练应被拒绝")
    except NativeShapeError as exc:
        assert "pose budget exceeded" in str(exc), f"超限错误文本异常：{exc}"
    expect_coded(sdk.ERROR_CODES["POSE_BUDGET"], 2, "姿态超限")

    # 画布账本：800² 压线通过、850² 超限（1600² 参考图，缩放 2）
    report = estimate(big, (375, 375, 800, 800), angle_min=0, angle_max=0, angle_step=1,
                      scale_min=1.0, scale_max=2.0, scale_step=1.0)
    assert not report["canvas_over"] and report["over_budget"] == 0, f"画布压线漂移：{report}"
    model = NativeModel(train_model(big, (375, 375, 800, 800), angle_min=0, angle_max=0,
                                    angle_step=1, scale_min=1.0, scale_max=2.0, scale_step=1.0))
    model.close()
    report = estimate(big, (375, 375, 850, 850), angle_min=0, angle_max=0, angle_step=1,
                      scale_min=1.0, scale_max=2.0, scale_step=1.0)
    assert report["canvas_over"] and report["over_budget"] == sdk.ERROR_CODES["CANVAS_BUDGET"], \
        f"画布超限漂移：{report}"
    try:
        train_model(big, (375, 375, 850, 850), angle_min=0, angle_max=0, angle_step=1,
                    scale_min=1.0, scale_max=2.0, scale_step=1.0)
        raise AssertionError("画布超限训练应被拒绝")
    except NativeShapeError as exc:
        assert "canvas budget exceeded" in str(exc), f"画布错误文本异常：{exc}"

    # 像素账本：21 档缩放累计超限 ↔ 拒绝；3 档同 ROI 通过 ↔ 接受
    report = estimate(big, (450, 450, 700, 700), angle_min=-8, angle_max=8, angle_step=8,
                      scale_min=1.0, scale_max=2.0, scale_step=0.05)
    assert report["pixel_over"] and report["over_budget"] == sdk.ERROR_CODES["PIXEL_BUDGET"], \
        f"像素超限漂移：{report}"
    assert not report["canvas_over"], f"画布应未超限：{report}"
    try:
        train_model(big, (450, 450, 700, 700), angle_min=-8, angle_max=8, angle_step=8,
                    scale_min=1.0, scale_max=2.0, scale_step=0.05)
        raise AssertionError("像素超限训练应被拒绝")
    except NativeShapeError as exc:
        assert "pixel budget exceeded" in str(exc), f"像素错误文本异常：{exc}"
    report = estimate(big, (450, 450, 700, 700), angle_min=-8, angle_max=8, angle_step=8,
                      scale_min=1.0, scale_max=1.2, scale_step=0.1)
    assert report["over_budget"] == 0, f"像素压线报告漂移：{report}"


def group_match_trace() -> None:
    reference = synth_reference()
    model = NativeModel(train_model(reference, (32, 32, 176, 176), angle_min=0, angle_max=0,
                                    angle_step=1, scale_min=1, scale_max=1, scale_step=1))
    try:
        frame = np.zeros((1600, 1600, 3), np.uint8)
        patch = np.ascontiguousarray(reference[32:208, 32:208])
        for x, y in ((100, 100), (800, 100), (1350, 1350)):
            frame[y:y + 176, x:x + 176] = patch
        frame = np.ascontiguousarray(frame)
        hits = model.match(frame, threshold=60)
        assert len(hits) >= 3, f"三目标召回 {len(hits)} 应 ≥3"
        trace = model.last_match_info()
        assert trace["windows_searched"] >= 2, f"三目标精修窗数 {trace['windows_searched']} 应 ≥2"
        assert trace["coarse_candidates"] >= trace["windows_searched"], f"粗候选数口径漂移：{trace}"
        assert trace["final_candidates"] == len(hits), f"最终候选数不符：{trace}"
        for n in range(min(trace["windows_searched"], 8)):
            score = trace["window_best_scores"][n]
            assert 0 < score <= 100, f"窗 {n} 最佳分越界：{score}"
        assert trace["coarse_ms"] == 0 and trace["windows_ms"] == 0, "常规构建耗时字段应为 0"
        again = model.last_match_info()
        assert again == trace, "二次 trace 查询应得同一快照"
    finally:
        model.close()


GROUPS = [
    ("diag_error_codes", group_error_codes),
    ("diag_budget_boundary", group_budget_boundary),
    ("diag_match_trace", group_match_trace),
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
    print(f"诊断分组：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
