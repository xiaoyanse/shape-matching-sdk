"""遮挡失败归因实测：逐案记录两级匹配 trace，区分粗层丢失/候选未精修/精层不足。

复用 perturbation.generate 遮挡因子与锚模板 spec；纯测量不改引擎，产物归 build/attribution/。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "perturbation"))
sys.path.insert(0, str(HERE.parents[0] / "bindings" / "python"))

import generate  # noqa: E402
import run_baseline  # noqa: E402
from shape_sdk import NativeModel  # noqa: E402

OCCLUSION_RATIOS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
WINDOW_SLOTS = 8
HIT = "hit"
OVER_TH_MISLOCATED = "over_th_mislocated"
COARSE_LOSS = "coarse_loss"
NOT_REFINED = "not_refined"
FINE_INSUFFICIENT = "fine_insufficient"
STATE_LABELS = {
    HIT: "命中",
    OVER_TH_MISLOCATED: "过阈错位",
    COARSE_LOSS: "粗层丢失",
    NOT_REFINED: "候选未精修",
    FINE_INSUFFICIENT: "精层不足",
}


def build_matrix(n_variants: int) -> list[dict]:
    """每档 n_variants 个变体：前 2 个 center 居中块（形态对照），其余 edge 边缘块（位置随种子）。"""
    cases = [{"id": "baseline/neutral", "ratio": 0.0, "anchor": "none", "seed": 0}]
    for ratio in OCCLUSION_RATIOS:
        for variant in range(n_variants):
            anchor = "center" if variant < 2 else "edge"
            cases.append({
                "id": f"occl/{ratio:.2f}@{anchor}/v{variant:02d}",
                "ratio": ratio,
                "anchor": anchor,
                "seed": variant,
            })
    return cases


def evaluate_case(model: NativeModel, case: dict, threshold: float) -> dict:
    frame, truth = generate.render_frame(
        case["seed"], occlusion_ratio=case["ratio"], occlusion_anchor=case["anchor"])
    started = time.perf_counter()
    hits = model.match(frame, threshold=threshold)
    match_ms = (time.perf_counter() - started) * 1000
    trace = model.last_match_info()
    center = truth["center"]
    best = max(hits, key=lambda item: item.score) if hits else None
    center_err = (round(float(np.hypot(best.center[0] - center[0], best.center[1] - center[1])), 2)
                  if best else "")
    matched = bool(best) and center_err <= run_baseline.CENTER_TOLERANCE_PX
    best_score = round(best.score, 2) if best else 0.0
    window_best = tuple(round(float(v), 2) for v in trace["window_best_scores"])
    searched = trace["windows_searched"]
    if matched:
        state = HIT
    elif best_score >= threshold:
        state = OVER_TH_MISLOCATED
    elif trace["coarse_candidates"] == 0:
        state = COARSE_LOSS
    elif searched == 0:
        state = NOT_REFINED
    else:
        state = FINE_INSUFFICIENT
    return {
        "id": case["id"],
        "ratio": case["ratio"],
        "anchor": case["anchor"],
        "seed": case["seed"],
        "matched": matched,
        "best_score": best_score,
        "n_hits": len(hits),
        "center_err_px": center_err,
        "attribution": state,
        "coarse_candidates": trace["coarse_candidates"],
        "windows_searched": searched,
        "final_candidates": trace["final_candidates"],
        **{f"window_best_{i}": window_best[i] for i in range(WINDOW_SLOTS)},
        "max_window_best": max(window_best),
        "coarse_over_cap": trace["coarse_candidates"] > WINDOW_SLOTS,
        "match_ms": round(match_ms, 2),
    }


def sanity_problems(row: dict, threshold: float) -> list[str]:
    """测量链路自检：干净帧必须命中且 trace 两级计数有效，否则产物不可信。"""
    problems = []
    if not row["matched"]:
        problems.append(f"干净帧未命中（最高分 {row['best_score']}）")
    if row["coarse_candidates"] <= 0:
        problems.append("trace 粗层候选数为 0（非两级模型或诊断导出不可用）")
    if row["windows_searched"] <= 0:
        problems.append("trace 精修窗数为 0")
    if row["final_candidates"] <= 0:
        problems.append("trace 最终候选数为 0")
    return [f"{problem}（阈值 {threshold}）" for problem in problems]


def _pct(part: int, total: int) -> str:
    return f"{part / total * 100:.0f}%" if total else "-"


def _band_share(rows: list[dict], low: float, high: float, state: str) -> tuple[int, int]:
    fails = [row["attribution"] for row in rows
             if low < row["ratio"] <= high and row["attribution"] != HIT]
    return sum(s == state for s in fails), len(fails)


def summarize(rows: list[dict], runtime_info: dict, train_info: dict,
              threshold: float, variants: int) -> str:
    coarse_gate = max(45.0, threshold - 20.0)
    fine_gate = max(50.0, threshold * 0.85)
    by_ratio: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        by_ratio[row["ratio"]].append(row)
    ratios = sorted(r for r in by_ratio if r > 0.0)
    lines = [
        "# 遮挡失败归因报告",
        "",
        f"- runtime：{runtime_info}",
        f"- 模板：{train_info['poses']} 位姿（-85..85° 步 5°，单尺度），阈值 {threshold}，"
        f"每档 {variants} 变体（2 center 居中块 + {variants - 2} edge 边缘块·位置随种子）",
        f"- 引擎内部门槛（两级编排）：粗层收集 max(45, T−20)={coarse_gate:g}（1/k 缩图），"
        f"精层收集 max(50, 0.85T)={fine_gate:g}",
        f"- 归因口径：命中=过阈且中心误差≤{run_baseline.CENTER_TOLERANCE_PX}px；粗层丢失="
        f"coarse_candidates==0；候选未精修=粗候选>0 且 windows_searched==0；精层不足=其余失败",
        "",
        "## 三态逐档表",
        "",
        "| 面积比 | 案数 | 命中 | 粗层丢失 | 候选未精修 | 精层不足 | 过阈错位 | 平均最高分 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in sorted(by_ratio):
        group = by_ratio[ratio]
        counts = Counter(row["attribution"] for row in group)
        mean_best = float(np.mean([row["best_score"] for row in group]))
        cells = " | ".join(
            f"{counts[state]}（{_pct(counts[state], len(group))}）"
            for state in (HIT, COARSE_LOSS, NOT_REFINED, FINE_INSUFFICIENT, OVER_TH_MISLOCATED)
        )
        lines.append(f"| {ratio:.2f} | {len(group)} | {cells} | {mean_best:.1f} |")
    lines += ["", "## 锚点细分", "", "| 面积比 | center 命中 | edge 命中 |", "|---:|---:|---:|"]
    for ratio in ratios:
        cells = []
        for anchor in ("center", "edge"):
            group = [row for row in by_ratio[ratio] if row["anchor"] == anchor]
            cells.append(f"{sum(row['matched'] for row in group)}/{len(group)}")
        lines.append(f"| {ratio:.2f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 精层不足细分",
        "",
        f"window_best 为该精修窗实际拿到的最高分，仅覆盖 ≥精层门槛 {fine_gate:g} 的候选；"
        "全零=该案所有精修窗连精层门槛都没过（更低分数对 trace 不可见）。",
        "",
        "| 面积比 | 精层不足案数 | 全窗零分案数 | 非零窗最高分均值 | 非零窗最高分最大 | 粗候选均值 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in ratios:
        fine_rows = [row for row in by_ratio[ratio] if row["attribution"] == FINE_INSUFFICIENT]
        if not fine_rows:
            continue
        nonzero = [row["max_window_best"] for row in fine_rows if row["max_window_best"] > 0.0]
        mean_nz = f"{float(np.mean(nonzero)):.1f}" if nonzero else "-"
        max_nz = f"{max(nonzero):.1f}" if nonzero else "-"
        lines.append(
            f"| {ratio:.2f} | {len(fine_rows)} "
            f"| {sum(1 for row in fine_rows if row['max_window_best'] <= 0.0)} "
            f"| {mean_nz} | {max_nz} "
            f"| {float(np.mean([row['coarse_candidates'] for row in fine_rows])):.1f} |"
        )
    lines += ["", "## 粗/精主导分界", ""]
    for ratio in ratios:
        fails = [row["attribution"] for row in by_ratio[ratio] if row["attribution"] != HIT]
        if not fails:
            lines.append(f"- {ratio:.0%}：全部命中")
        else:
            dist = "、".join(f"{STATE_LABELS[state]} {n}（{_pct(n, len(fails))}）"
                             for state, n in Counter(fails).most_common())
            lines.append(f"- {ratio:.0%}：失败 {len(fails)} 案 —— {dist}")
    mid_coarse, mid_total = _band_share(rows, 0.0, 0.40, COARSE_LOSS)
    mid_fine, _ = _band_share(rows, 0.0, 0.40, FINE_INSUFFICIENT)
    high_coarse, high_total = _band_share(rows, 0.40, 1.0, COARSE_LOSS)
    high_fine, _ = _band_share(rows, 0.40, 1.0, FINE_INSUFFICIENT)
    lines += ["", "## P1 方案建议（数据驱动）", ""]
    if mid_total == 0:
        lines.append("- 25–40% 中段无失败案例，与下版 P1 目标档不交叠，建议扩梯度后再选型")
    elif mid_coarse * 2 >= mid_total:
        lines.append(
            f"- 中段（25–40%）失败 {mid_total} 案中粗层丢失 {mid_coarse}"
            f"（{_pct(mid_coarse, mid_total)}）：目标在 1/k 缩图粗层即被丢——"
            "P1 首选粗层自适应阈值/遮挡感知粗层")
    else:
        lines.append(
            f"- 中段（25–40%）失败 {mid_total} 案中精层不足 {mid_fine}"
            f"（{_pct(mid_fine, mid_total)}）：目标进了候选但分数卡在精层——"
            "P1 首选部分分比/最优子集投票")
    if high_total:
        tail = "；高段需粗层侧方案兜底" if high_coarse * 2 >= high_total else ""
        lines.append(
            f"- 高段（45–60%）失败 {high_total} 案：粗层丢失 {high_coarse}"
            f"（{_pct(high_coarse, high_total)}）、精层不足 {high_fine}"
            f"（{_pct(high_fine, high_total)}）{tail}")
    fine_all = [row for row in rows if row["attribution"] == FINE_INSUFFICIENT]
    if fine_all:
        zero = sum(1 for row in fine_all if row["max_window_best"] <= 0.0)
        lines.append(
            f"- 精层不足案中全窗零分 {zero}/{len(fine_all)}：该部分分数连精层门槛 "
            f"{fine_gate:g} 都未过（trace 不可见更低分），部分分比方案需连带评估精层收集门槛耦合")
    lines += [
        "",
        "## 诚实边界",
        "",
        f"- coarse_candidates>8（仅前 8 窗精修，trace 无法确认真目标窗是否在内）共 "
        f"{sum(1 for row in rows if row['coarse_over_cap'])} 案；本场景平坦背景单目标，风险低",
        "- 候选未精修在引擎结构上不可达（windows_searched=min(8,粗候选)，src/two_stage.cpp）；"
        f"实测 {sum(1 for row in rows if row['attribution'] == NOT_REFINED)} 例，与源码一致",
        "- 聚类去重与 B7 几何精修只作用于过阈候选，不影响三态归因",
        "- 结论限于：平坦背景、单目标、256px 目标、±30° 位姿、该模板 spec、方形灰块（128 灰）遮挡",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="遮挡失败归因实测（纯测量，不改引擎）")
    parser.add_argument("--out", default=str(HERE.parents[0] / "build" / "attribution"))
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=run_baseline.THRESHOLD)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 案（0=全量，冒烟用）")
    args = parser.parse_args()
    if not generate.repro_check():
        return 1
    from shape_sdk.native_library import runtime_directory, sdk_version

    runtime_info = {"sdk_version": sdk_version(), "dir": str(runtime_directory())}
    model_bytes, train_info = run_baseline.train_reference_model()
    matrix = build_matrix(args.variants)
    if args.limit > 0:
        matrix = matrix[:args.limit]
    rows = []
    with NativeModel(model_bytes) as model:
        for case in matrix:
            rows.append(evaluate_case(model, case, args.threshold))
    problems = sanity_problems(rows[0], args.threshold)
    if problems:
        raise SystemExit("测量链路自检失败：" + "；".join(problems))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "attribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows, runtime_info, train_info, args.threshold, args.variants)
    (out / "attribution_report.md").write_text(report, encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "runtime": runtime_info,
        "train": train_info,
        "threshold": args.threshold,
        "coarse_gate": max(45.0, args.threshold - 20.0),
        "fine_gate": max(50.0, args.threshold * 0.85),
        "variants": args.variants,
        "ratios": list(OCCLUSION_RATIOS),
        "cases": len(rows),
        "center_tolerance_px": run_baseline.CENTER_TOLERANCE_PX,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"逐案 CSV：{csv_path}（{len(rows)} 案）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
