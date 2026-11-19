"""体检与审计绑定合同测试：health 分级与自匹配、audit 分布单调与推荐区间
数学（手工算对照）、INSEPARABLE 噪声构造、NO_NEG 边界、错误码 18/19。

运行前提：SHAPE_SDK_RUNTIME 指向含新导出的制品目录（如 build/t2b/runtime），
本脚本只读校验 manifest 哈希链，不装配不写盘：
    set SHAPE_SDK_RUNTIME=<制品目录>
    python scripts/test_audit_chain.py
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
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
    raise SystemExit("须先设置 SHAPE_SDK_RUNTIME 指向含新导出的制品目录（如 build/t2b/runtime）")


RUNTIME = assemble_free_runtime()
sys.path.insert(0, str(ROOT / "bindings" / "python"))

import shape_sdk as sdk  # noqa: E402
from shape_sdk import (  # noqa: E402
    NativeModel,
    NativeShapeError,
    audit_discriminability,
    error_message,
    last_error_info,
    model_health,
    train_model,
)

ROI = (100, 100, 176, 176)


def synth_reference() -> np.ndarray:
    """480×480 黑底：白色方框 (100..219)² 与内部灰块（形状与 ROI 边界留黑边距）。"""
    image = np.zeros((480, 480, 3), np.uint8)
    image[100:220, 100:220, :] = 255
    image[150:170, 150:170, :] = 90
    return np.ascontiguousarray(image)


def fact_frame(patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """fact 语义帧：480×480 黑底，帧 1/3/4 有锚（patch 各落一位点）、帧 2 无锚。"""
    frame = np.zeros((480, 480, 3), np.uint8)
    frame[y:y + 176, x:x + 176] = patch
    return np.ascontiguousarray(frame)


def noise_frame() -> np.ndarray:
    rng = np.random.default_rng(20260317)
    return np.ascontiguousarray(rng.integers(0, 256, (320, 320, 3), dtype=np.uint8))


def train_defaults(reference: np.ndarray) -> bytes:
    return train_model(reference, ROI, angle_min=0, angle_max=0, angle_step=1,
                       scale_min=1.0, scale_max=1.0, scale_step=1.0)


def first_fine_segment(container: bytes) -> bytes:
    assert container[:5] == b"VF2S3", "容器 magic 不符"
    count = struct.unpack_from("<Q", container, 5)[0]
    start = 13 + 8 * (count + 3)
    length = struct.unpack_from("<Q", container, 13)[0]
    return container[start:start + length]


def expect_math(report: dict, what: str) -> None:
    for group in ("pos", "neg"):
        stats = [report[f"{group}_min"], report[f"{group}_p25"], report[f"{group}_median"],
                 report[f"{group}_p75"], report[f"{group}_max"]]
        if report[f"{group}_count"] == 0 and group == "neg":
            continue
        assert stats == sorted(stats), f"{what} {group} 分布应单调：{stats}"
    assert report["separation"] == report["pos_p25"] - report["neg_max"], \
        f"{what} 分离度应=pos_p25−neg_max：{report}"
    lo = max(30.0, report["neg_max"] + 5.0)
    hi = min(85.0, report["pos_p25"] - 5.0)
    assert report["recommend_lo"] == lo and report["recommend_hi"] == hi, \
        f"{what} 推荐区间数学漂移：[{report['recommend_lo']},{report['recommend_hi']}] " \
        f"应为 [{lo},{hi}]"
    inseparable = report["flag"] == sdk.AUDIT_FLAGS["INSEPARABLE"]
    assert inseparable == (lo > hi), f"{what} INSEPARABLE 判定应与 lo>hi 一致"
    low = report["pos_count"] < 5 or report["neg_count"] < 5
    assert (report["confidence"] == sdk.AUDIT_CONFIDENCE["LOW"]) == low, \
        f"{what} 置信度应与样本门槛一致：{report}"


def group_health() -> None:
    reference = synth_reference()
    model = NativeModel(train_defaults(reference))
    try:
        report = model_health(model)
        assert report["verdict"] == sdk.HEALTH_VERDICTS["HEALTHY"], \
            f"健康模板应 HEALTHY：{report['verdict_text']}"
        assert report["self_score_valid"] and report["self_score"] >= 70.0, \
            f"容器自匹配应实测且 ≥70：{report['self_score']}"
        assert report["pose_count"] >= 1, f"姿态数应 ≥1：{report}"
        assert 4 <= report["fine_min_features"] <= report["fine_max_features"] <= 512, \
            f"fine 特征数口径漂移：{report}"
        assert report["coarse_features"] >= 4, f"粗层特征数应 ≥4：{report}"
        assert report["template_width"] == 176 and report["template_height"] == 176, \
            f"模板尺寸应=训练 ROI 176×176：{report}"
        assert (report["detector_features"], report["detector_weak"],
                report["detector_strong"]) == (128, 30.0, 60.0), "默认三参快照漂移"
        assert "健康" in report["verdict_text"], f"人话文案异常：{report['verdict_text']}"

        single = NativeModel(first_fine_segment(train_defaults(reference)))
        try:
            solo = model_health(single)
            assert not solo["self_score_valid"] and solo["self_score"] == 0.0, \
                f"单级自匹配应未实测：{solo}"
            assert solo["coarse_features"] == 0, f"单级粗层特征数应 0：{solo}"
            assert solo["verdict"] == sdk.HEALTH_VERDICTS["HEALTHY"], \
                f"单级特征充足应 HEALTHY：{solo['verdict_text']}"
        finally:
            single.close()
    finally:
        model.close()


def group_audit_fact() -> None:
    reference = synth_reference()
    patch = np.ascontiguousarray(reference[100:276, 100:276])
    pos = [fact_frame(patch, 40, 40), fact_frame(patch, 260, 60), fact_frame(patch, 60, 260)]
    neg = [np.zeros((480, 480, 3), np.uint8)]
    model = NativeModel(train_defaults(reference))
    try:
        report = audit_discriminability(model, pos, neg)
        assert report["pos_count"] == 3 and report["pos_zero_hits"] == 0, \
            f"正组应 3 帧全有分：{report}"
        assert report["neg_count"] == 0 and report["neg_zero_hits"] == 1, \
            f"负组空白应零命中：{report}"
        assert report["pos_max"] >= 70.0, f"正组最高分应 ≥70：{report['pos_max']}"
        expect_math(report, "fact 正负组")
        assert report["flag"] == sdk.AUDIT_FLAGS["RECOMMENDED"], f"应 RECOMMENDED：{report}"
        assert "可推荐" in report["flag_text"], f"flag 人话异常：{report['flag_text']}"

        no_neg = audit_discriminability(model, pos, None)
        assert no_neg["flag"] == sdk.AUDIT_FLAGS["NO_NEGATIVE"], f"缺负组应 NO_NEGATIVE：{no_neg}"
        assert no_neg["neg_count"] == 0 and no_neg["recommend_lo"] == 0.0 \
            and no_neg["recommend_hi"] == 0.0, f"缺负组不应产推荐区间：{no_neg}"
        assert no_neg["pos_count"] == 3, "无负组正组分布应保留"

        noise = noise_frame()
        inseparable = audit_discriminability(model, [noise, noise], [noise, noise])
        assert inseparable["flag"] == sdk.AUDIT_FLAGS["INSEPARABLE"], \
            f"噪声双组应 INSEPARABLE：{inseparable}"

        try:
            audit_discriminability(model, [], neg)
            raise AssertionError("正组空应拒绝")
        except NativeShapeError as exc:
            assert "正组" in str(exc), f"正组空错误文本异常：{exc}"
    finally:
        model.close()


def probe_reference() -> np.ndarray:
    """粗层全败探针图：复刻 tests/abi_chain_support.h 的 320 合成图（同内容同源，
    其灰块边缘几何在 40×40 ROI 内可过 fine 训练——探测实证，勿改构造）。"""
    image = np.zeros((320, 320, 3), np.uint8)
    for x0, y0, x1, y1, value in (
        (42, 42, 47, 197, 255), (42, 42, 197, 47, 255), (42, 192, 197, 197, 255),
        (192, 42, 197, 197, 255), (65, 65, 94, 94, 255), (90, 120, 154, 154, 200),
        (165, 80, 170, 179, 255),
    ):
        image[y0:y1 + 1, x0:x1 + 1] = value
    return np.ascontiguousarray(image)


def group_coded_failures() -> None:
    assert sdk.ERROR_CODES["INSUFFICIENT_FEATURES"] == 18, "码 18 值漂移"
    assert sdk.ERROR_CODES["COARSE_TRAIN_FAILED"] == 19, "码 19 值漂移"
    flat = np.zeros((480, 480, 3), np.uint8)
    try:
        train_model(flat, ROI, angle_min=0, angle_max=0, angle_step=1,
                    scale_min=1.0, scale_max=1.0, scale_step=1.0)
        raise AssertionError("无边缘 ROI 应被拒绝")
    except NativeShapeError:
        code, params = last_error_info()
        assert code == 18, f"错误码应为 18，实际 {code}"
        assert params and params[0] == 4.0, f"参数应为特征数下限 4：{params}"
    assert "特征不足" in error_message(18, (4.0,)), "码 18 人话文案异常"

    try:
        train_model(probe_reference(), (120, 120, 40, 40), angle_min=0, angle_max=0,
                    angle_step=1, scale_min=1.0, scale_max=1.0, scale_step=1.0)
        raise AssertionError("粗层全败应被拒绝")
    except NativeShapeError:
        code, params = last_error_info()
        assert code == 19, f"错误码应为 19，实际 {code}"
        assert params and params[0] == 1.0, f"参数应为尝试的 K 档数 1：{params}"
    assert "K 档" in error_message(19, (1.0,)), "码 19 人话文案异常"


GROUPS = [
    ("py_health", group_health),
    ("py_audit_fact", group_audit_fact),
    ("py_audit_coded", group_coded_failures),
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
    print(f"体检/审计分组：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
