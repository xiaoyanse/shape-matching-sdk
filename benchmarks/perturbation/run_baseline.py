"""受控扰动基准运行器：对当前 dist/runtime 跑因子矩阵，产出 CSV 与能力曲线汇总。

用法（仓库根）：
    cd bindings/python && uv run python ../../benchmarks/perturbation/run_baseline.py --out ../../build/perturb
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "bindings" / "python"))

import cv2  # noqa: E402
import generate  # noqa: E402
from shape_sdk import NativeModel, train_model  # noqa: E402

THRESHOLD = 70.0
CENTER_TOLERANCE_PX = 8.0
CROP_HALF = 160
TEMPLATE_ROI = (32, 32, 256, 256)
TRAIN_ANGLES = (-85.0, 85.0, 5.0)


def train_reference_model() -> tuple[bytes, dict]:
    """中性基线帧裁目标邻域训练 35 位姿单尺度模板。"""
    frame, truth = generate.render_frame(0)
    cx, cy = (int(v) for v in truth["center"])
    x0, y0 = cx - CROP_HALF, cy - CROP_HALF
    patch = np.ascontiguousarray(frame[y0 : y0 + CROP_HALF * 2, x0 : x0 + CROP_HALF * 2])
    started = time.perf_counter()
    model_bytes = train_model(
        patch,
        TEMPLATE_ROI,
        angle_min=TRAIN_ANGLES[0],
        angle_max=TRAIN_ANGLES[1],
        angle_step=TRAIN_ANGLES[2],
        scale_min=1.0,
        scale_max=1.0,
        scale_step=0.05,
    )
    info = {
        "train_ms": round((time.perf_counter() - started) * 1000, 1),
        "poses": int(round((TRAIN_ANGLES[1] - TRAIN_ANGLES[0]) / TRAIN_ANGLES[2])) + 1,
        "roi_xywh": list(TEMPLATE_ROI),
    }
    return model_bytes, info


def _angle_error(best_angle: float, truth_angle: float) -> float:
    delta = abs(best_angle - truth_angle) % 360.0
    return min(delta, 360.0 - delta)


def evaluate_case(model: NativeModel, case: dict) -> dict:
    frame, truth = generate.render_frame(**generate._case_args(case))
    started = time.perf_counter()
    hits = model.match(frame, threshold=THRESHOLD)
    match_ms = (time.perf_counter() - started) * 1000
    best = max(hits, key=lambda item: item.score) if hits else None
    center = truth["center"]
    matched = any(
        item.score >= THRESHOLD
        and float(np.hypot(item.center[0] - center[0], item.center[1] - center[1]))
        <= CENTER_TOLERANCE_PX
        for item in hits
    )
    return {
        "id": case["id"],
        "factor": case["factor"],
        "level": case["level"],
        "bg": case.get("bg", "flat"),
        "occlusion_ratio": case.get("occlusion_ratio", 0.0),
        "occlusion_anchor": case.get("occlusion_anchor", "none"),
        "gain": case.get("gain", 1.0),
        "bias": case.get("bias", 0.0),
        "present": case.get("present", True),
        "seed": case["seed"],
        "best_score": round(best.score, 2) if best else 0.0,
        "n_hits": len(hits),
        "matched": matched,
        "center_err_px": round(float(np.hypot(best.center[0] - center[0], best.center[1] - center[1])), 2)
        if best
        else "",
        "angle_err_deg": round(_angle_error(best.angle, truth["angle_deg"]), 2)
        if best and truth["present"]
        else "",
        "match_ms": round(match_ms, 2),
    }


def _group_stats(rows: list[dict]) -> dict:
    scores = [row["best_score"] for row in rows]
    matches = [row for row in rows if row["matched"]]
    errors = [row["center_err_px"] for row in matches]
    times = [row["match_ms"] for row in rows]
    return {
        "n": len(rows),
        "recall": len(matches) / len(rows),
        "mean_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "mean_err": float(np.mean(errors)) if errors else float("nan"),
        "mean_ms": float(np.mean(times)),
    }


def _fmt_recall(value: float) -> str:
    return f"{value * 100:.0f}%"


def summarize(rows: list[dict], runtime_info: dict, train_info: dict) -> str:
    """每因子一条能力曲线（遮挡比/光照组合/背景 vs score 与召回），附边界识别。"""
    present_rows = [row for row in rows if row["present"]]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in present_rows:
        groups[(row["factor"], row["level"])].append(row)
    stats = {key: _group_stats(rows_) for key, rows_ in groups.items()}
    lines = [
        "# 受控扰动基线报告",
        "",
        f"- runtime：{runtime_info}",
        f"- 模板：256px 目标 / {train_info['poses']} 位姿（-85..85° 步 5°，单尺度），训练 {train_info['train_ms']} ms",
        f"- 匹配阈值 {THRESHOLD}，命中判定：score≥阈值且中心误差≤{CENTER_TOLERANCE_PX}px；每档 {stats[('baseline', 'neutral')]['n']} 种子",
        "",
        "## 遮挡曲线（面积比 vs score/召回）",
        "",
        "| 面积比 | 锚点 | 召回 | 平均分 | 分数范围 | 平均中心误差px | 平均耗时ms |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    occlusion_keys = sorted(
        (key for key in stats if key[0] == "occlusion"),
        key=lambda key: (key[1].split("@")[0], key[1]),
    )
    for key in [("baseline", "neutral")] + occlusion_keys:
        entry = stats[key]
        ratio, anchor = key[1].split("@") if key[0] == "occlusion" else ("0.0", "none")
        err = f"{entry['mean_err']:.2f}" if np.isfinite(entry["mean_err"]) else "-"
        lines.append(
            f"| {ratio} | {anchor} | {_fmt_recall(entry['recall'])} | {entry['mean_score']:.1f} "
            f"| {entry['min_score']:.0f}–{entry['max_score']:.0f} | {err} | {entry['mean_ms']:.1f} |"
        )
    fell = [
        ratio for ratio in generate.OCCLUSION_RATIOS[1:]
        if stats.get(("occlusion", f"{ratio:.1f}@center"), {}).get("recall", 1.0) < 1.0
    ]
    if fell:
        lines.append(f"\n中心遮挡召回首破档：{fell[0]:.0%}（此前档召回 100%）")
    lines += [
        "",
        "## 光照网格（增益 × 偏置 → 召回/平均分）",
        "",
        "| 增益\\偏置 | " + " | ".join(f"{bias:+d}" for bias in generate.BIAS_GRID) + " |",
        "|---:|" + "---:|" * len(generate.BIAS_GRID),
    ]
    for gain in generate.GAIN_GRID:
        cells = []
        for bias in generate.BIAS_GRID:
            key = ("lighting", f"g{gain:.3f}/b{bias:+d}")
            entry = stats.get(key)
            if entry is None:
                cells.append("（基线）")
            else:
                cells.append(f"{_fmt_recall(entry['recall'])}/{entry['mean_score']:.1f}")
        lines.append(f"| {gain:.3f} | " + " | ".join(cells) + " |")
    dark = [
        key for key in stats if key[0] == "lighting" and stats[key]["recall"] < 1.0
    ]
    if dark:
        lines.append(f"\n光照召回破档组合数：{len(dark)}（含增益/偏置：{', '.join(sorted(key[1] for key in dark))}）")
    else:
        lines.append("\n光照网格全部组合召回 100%")
    lines += ["", "## 背景", "", "| 背景 | 召回 | 平均分 | 平均耗时ms |", "|---|---:|---:|---:|"]
    for kind in generate.BG_LEVELS:
        entry = stats.get(("background", kind), stats.get(("baseline", "neutral")))
        if entry is None:
            continue
        lines.append(
            f"| {kind} | {_fmt_recall(entry['recall'])} | {entry['mean_score']:.1f} | {entry['mean_ms']:.1f} |"
        )
    negative = [row for row in rows if not row["present"]]
    if negative:
        false_hits = sum(row["n_hits"] for row in negative)
        top = max((row["best_score"] for row in negative), default=0.0)
        lines += [
            "",
            "## 负样本对照（无目标）",
            "",
            f"- {len(negative)} 帧，误报候选共 {false_hits} 个，最高分 {top:.1f}（阈值 {THRESHOLD}）",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="扰动基准运行器")
    parser.add_argument("--out", default=str(HERE.parents[2] / "build" / "perturb"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from shape_sdk.native_library import runtime_directory, sdk_version

    runtime_info = {
        "sdk_version": sdk_version(),
        "dir": str(runtime_directory()),
    }
    model_bytes, train_info = train_reference_model()
    matrix = generate.build_matrix(args.seeds)
    rows = []
    with NativeModel(model_bytes) as model:
        warmup = evaluate_case(model, matrix[0])
        rows.append(warmup)
        for case in matrix[1:]:
            rows.append(evaluate_case(model, case))
    csv_path = out / "perturb_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows, runtime_info, train_info)
    (out / "baseline_report.md").write_text(report, encoding="utf-8")
    (out / "run_config.json").write_text(
        json.dumps({
            "runtime": runtime_info,
            "train": train_info,
            "threshold": THRESHOLD,
            "center_tolerance_px": CENTER_TOLERANCE_PX,
            "seeds": args.seeds,
            "cases": len(rows),
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(report)
    print(f"逐样本 CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
