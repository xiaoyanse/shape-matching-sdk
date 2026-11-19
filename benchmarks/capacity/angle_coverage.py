"""宽角度粗档训练的 B7 覆盖能力实测：档距×偏移×素材矩阵，CSV+结论摘要。

训练档距 S∈{3,5,8,10,15}°（±30° 范围），测试图=原图绕目标中心旋转已知 δ
（warpAffine 双线性+常数底），真值角度=基准角+δ、中心不变。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bindings" / "python"))

from shape_sdk import NativeModel, train_model  # noqa: E402
from shape_sdk.native_library import runtime_directory, sdk_version  # noqa: E402

THRESHOLD = 70.0
CENTER_TOLERANCE_PX = 8.0
ANGLE_MIN, ANGLE_MAX = -30.0, 30.0
STEPS = (3.0, 5.0, 8.0, 10.0, 15.0)
DELTA_FRACTIONS = (0.0, -0.5, -0.25, 0.25, 0.5)
DEFAULT_REAL_DIR = Path("E:/dev/01-active/082-aieng-py/data/shape-matching-real48-v1/sources")

CANVAS_PX = 540
CROP_ROI = (140, 140, 260, 260)
FACT_BG = (150, 150, 150)
FACT_FRAMES = (
    {"group": "fact-frame1",
     "png": ROOT / "benchmarks/shape-facts-v1/frames/synthetic-f0001-positive.png",
     "center": (600.0, 420.0), "base_angle": 15.0},
    {"group": "fact-frame4",
     "png": ROOT / "benchmarks/shape-facts-v1/frames/synthetic-f0004-positive-new-pose.png",
     "center": (620.0, 400.0), "base_angle": -10.0},
)
REAL_PICKS = (
    {"group": "real-img0", "res": "img0",
     "png": "0229a1dbfd704dfc9451104f63ed90d0.png", "roi": (950, 1700, 500, 500)},
    {"group": "real-img2", "res": "img2",
     "png": "02f3f94e407b731121f34aec29e43a17.png", "roi": (700, 1950, 500, 500)},
)
REAL_MAX_EDGE = 4096


def draw_anchor_pattern(img: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    """复刻 benchmarks/shape-facts-v1/source/vision_facts.cpp drawAnchorPattern。"""
    cx, cy = center
    rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    ink = (45, 45, 45)

    def warp(px: float, py: float) -> tuple[float, float]:
        return (float(rot[0, 0] * px + rot[0, 1] * py + rot[0, 2]),
                float(rot[1, 0] * px + rot[1, 1] * py + rot[1, 2]))

    box = [warp(cx + dx, cy + dy) for dx, dy in ((-80, -80), (80, -80), (80, 80), (-80, 80))]
    for i in range(4):
        cv2.line(img, tuple(int(round(v)) for v in box[i]),
                 tuple(int(round(v)) for v in box[(i + 1) % 4]), ink, 8, cv2.LINE_8)
    cv2.circle(img, tuple(int(round(v)) for v in warp(cx + 36, cy - 30)), 26, ink, -1)
    tri = [warp(cx + dx, cy + dy) for dx, dy in ((-56, 48), (10, -58), (58, 40))]
    for i in range(3):
        cv2.line(img, tuple(int(round(v)) for v in tri[i]),
                 tuple(int(round(v)) for v in tri[(i + 1) % 3]), ink, 10, cv2.LINE_8)


def canvas_reference() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """锚图案黑底画布训练参考（spec trainTemplate 路径：孤立图案、无场景杂波）。"""
    canvas = np.zeros((CANVAS_PX, CANVAS_PX, 3), np.uint8)
    draw_anchor_pattern(canvas, (CANVAS_PX / 2.0, CANVAS_PX / 2.0), 0.0)
    return canvas, CROP_ROI


def scene_reference(frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """场景图裁目标邻域训练参考（对照口径：背景真实但可能含相邻图案切片）。"""
    x, y, w, h = roi
    patch = np.ascontiguousarray(frame[y : y + h, x : x + w])
    return patch, (0, 0, w, h)


def load_real_base(real_dir: Path, manifest_dir: Path) -> list[dict]:
    """读 real48 原图并缩至 4096 限内（原配方 cv2.resize 默认插值），校验冻结 SHA。"""
    manifest = json.loads((manifest_dir / "fixture-manifest.json").read_text(encoding="utf-8"))
    entries = []
    for pick in REAL_PICKS:
        ref = next(item for item in manifest["references"] if item["res"] == pick["res"])
        png_path = real_dir / pick["png"]
        digest = hashlib.sha256(png_path.read_bytes()).hexdigest()
        if digest != manifest["files"][f"sources/{pick['png']}"]:
            raise SystemExit(f"素材指纹不符：{png_path}")
        raw = cv2.imread(str(png_path))
        height, width = raw.shape[:2]
        scale = min(1.0, REAL_MAX_EDGE / max(width, height))
        size = (int(width * scale), int(height * scale))
        base = cv2.resize(raw, size) if scale < 1.0 else raw
        corners = np.stack([base[0, 0], base[0, -1], base[-1, 0], base[-1, -1]])
        border = tuple(int(v) for v in np.median(corners, axis=0))
        entries.append({**pick, "base": base, "border": border,
                        "center": (pick["roi"][0] + pick["roi"][2] / 2.0,
                                   pick["roi"][1] + pick["roi"][3] / 2.0)})
    return entries


def angle_grid(low: float, high: float, step: float) -> list[float]:
    """复刻 src/matcher.cpp samples：自 low 步进枚举、末点不足补 high。"""
    values = [low + n * step for n in range(int(np.floor((high - low) / step + 1e-9)) + 1)]
    if high - values[-1] > 1e-7:
        values.append(high)
    return values


def angle_error(measured: float, truth: float) -> float:
    delta = abs(measured - truth) % 360.0
    return min(delta, 360.0 - delta)


def rotate_frame(frame: np.ndarray, center: tuple[float, float], delta: float,
                 border: tuple[int, int, int]) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(center, delta, 1.0)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=border)


def build_subjects(real_dir: Path) -> list[dict]:
    subjects = []
    for spec in FACT_FRAMES:
        frame = cv2.imread(str(spec["png"]))
        cx, cy = spec["center"]
        subjects.append({**spec, "kind": "fact", "border": FACT_BG, "frame": frame,
                         "train_frame": frame,
                         "train_roi": (int(cx) - 130, int(cy) - 130, 260, 260)})
    manifest_dir = ROOT / "benchmarks/shape-matching-real48-v1"
    for entry in load_real_base(real_dir, manifest_dir):
        subjects.append({**entry, "kind": "real", "frame": entry["base"],
                         "base_angle": 0.0,
                         "train_frame": entry["base"], "train_roi": entry["roi"]})
    return subjects


def run_matrix(subjects: list[dict], steps: tuple[float, ...], train_source: str) -> list[dict]:
    rows = []
    for subject in subjects:
        base = subject["frame"]
        for step in steps:
            if train_source == "canvas" and subject["kind"] == "fact":
                reference, roi = canvas_reference()
            else:
                reference, roi = scene_reference(subject["train_frame"], subject["train_roi"])
            started = time.perf_counter()
            model_bytes = train_model(
                reference, roi,
                angle_min=ANGLE_MIN, angle_max=ANGLE_MAX, angle_step=step,
                scale_min=1.0, scale_max=1.0, scale_step=0.05,
            )
            train_ms = (time.perf_counter() - started) * 1000.0
            grid = angle_grid(ANGLE_MIN, ANGLE_MAX, step)
            with NativeModel(model_bytes) as model:
                for fraction in DELTA_FRACTIONS:
                    delta = round(fraction * step, 4)
                    canvas_train = train_source == "canvas" and subject["kind"] == "fact"
                    ref_base = subject["base_angle"] if canvas_train else 0.0
                    truth_angle = ref_base + delta
                    test = rotate_frame(base, subject["center"], delta, subject["border"])
                    started = time.perf_counter()
                    hits = model.match(test, threshold=THRESHOLD)
                    match_ms = (time.perf_counter() - started) * 1000.0
                    best = max(hits, key=lambda h: h.score) if hits else None
                    coarse = min(grid, key=lambda a: angle_error(a, truth_angle))
                    center_err = float("nan") if best is None else float(np.hypot(
                        best.center[0] - subject["center"][0],
                        best.center[1] - subject["center"][1]))
                    matched = any(
                        h.score >= THRESHOLD
                        and np.hypot(h.center[0] - subject["center"][0],
                                     h.center[1] - subject["center"][1]) <= CENTER_TOLERANCE_PX
                        for h in hits)
                    rows.append({
                        "group": subject["group"], "train_source": train_source,
                        "ref_base_angle": round(ref_base, 4),
                        "step_deg": step, "delta_deg": delta,
                        "truth_angle_deg": round(truth_angle, 4),
                        "poses": len(grid),
                        "coarse_angle_err_deg": round(angle_error(coarse, truth_angle), 3),
                        "best_score": round(best.score, 2) if best else 0.0,
                        "passed": bool(best and best.score >= THRESHOLD),
                        "matched": matched,
                        "angle_err_b7_deg": round(angle_error(best.angle, truth_angle), 3)
                        if best else "",
                        "b7_gain_deg": round(angle_error(coarse, truth_angle)
                                             - angle_error(best.angle, truth_angle), 3)
                        if best else "",
                        "center_err_px": round(center_err, 2) if best else "",
                        "n_hits": len(hits),
                        "train_ms": round(train_ms, 1), "match_ms": round(match_ms, 2),
                    })
    return rows


def _fmt(value, pattern: str = ".2f") -> str:
    return format(value, pattern) if isinstance(value, float) and np.isfinite(value) else "-"


def summarize(rows: list[dict]) -> str:
    lines = [
        "# 宽角度粗档训练 B7 覆盖能力实测",
        "",
        "判定口径：档距 S 可背书 = 该档距全部(素材,偏移)点最高分过阈(≥70)"
        "且 B7 精修后角度误差 ≤ S/8 且中心偏差 ≤ 8px。",
        "",
        "## 每档距全池最坏（跨素材帧）",
        "",
        "| 档距S | 位姿数 | 点数 | 过阈 | 最坏分数 | 最坏B7角误差° | 上限S/8° | 最坏中心px | 粗档最坏角误差° | B7增益均值° | 可背书 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    verdicts = {}
    for step in sorted({row["step_deg"] for row in rows}):
        pool = [row for row in rows if row["step_deg"] == step]
        scored = [row for row in pool if row["passed"]]
        b7_errs = [row["angle_err_b7_deg"] for row in scored]
        center_errs = [row["center_err_px"] for row in pool
                       if isinstance(row["center_err_px"], float)]
        coarse_errs = [row["coarse_angle_err_deg"] for row in pool]
        gains = [row["b7_gain_deg"] for row in scored if isinstance(row["b7_gain_deg"], float)]
        worst_score = min(row["best_score"] for row in pool)
        ok = (len(scored) == len(pool)
              and b7_errs and max(b7_errs) <= step / 8.0
              and center_errs and max(center_errs) <= CENTER_TOLERANCE_PX)
        verdicts[step] = ok
        lines.append(
            f"| {step:g} | {pool[0]['poses']} | {len(pool)} | {len(scored)} | {worst_score:.1f} "
            f"| {_fmt(max(b7_errs), '.3f') if b7_errs else '-'} | {step / 8.0:.3f} "
            f"| {_fmt(max(center_errs) if center_errs else float('nan'))} "
            f"| {_fmt(max(coarse_errs), '.3f')} | {_fmt(float(np.mean(gains)) if gains else float('nan'), '.3f')} "
            f"| {'是' if ok else '否'} |")
    lines += ["", "## 分素材最坏（每档距）", "",
              "| 素材 | 档距S | 最坏分数 | 过阈/点 | 最坏B7角误差° | 最坏中心px | 可背书 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for group in sorted({row["group"] for row in rows}):
        for step in sorted({row["step_deg"] for row in rows}):
            pool = [row for row in rows if row["group"] == group and row["step_deg"] == step]
            scored = [row for row in pool if row["passed"]]
            b7_errs = [row["angle_err_b7_deg"] for row in scored]
            center_errs = [row["center_err_px"] for row in pool
                           if isinstance(row["center_err_px"], float)]
            ok = (len(scored) == len(pool)
                  and b7_errs and max(b7_errs) <= step / 8.0
                  and center_errs and max(center_errs) <= CENTER_TOLERANCE_PX)
            lines.append(
                f"| {group} | {step:g} | {min(row['best_score'] for row in pool):.1f} "
                f"| {len(scored)}/{len(pool)} | {_fmt(max(b7_errs), '.3f') if b7_errs else '-'} "
                f"| {_fmt(max(center_errs) if center_errs else float('nan'))} "
                f"| {'是' if ok else '否'} |")
    violations = [row for row in rows if not (
        row["passed"]
        and isinstance(row["angle_err_b7_deg"], float)
        and row["angle_err_b7_deg"] <= row["step_deg"] / 8.0
        and isinstance(row["center_err_px"], float)
        and row["center_err_px"] <= CENTER_TOLERANCE_PX)]
    lines += ["", "## 违例/边界组合明细", ""]
    if violations:
        lines += ["| 素材 | 档距S | 偏移δ° | 真值角° | 分数 | B7角误差° | 中心px |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in violations:
            lines.append(
                f"| {row['group']} | {row['step_deg']:g} | {row['delta_deg']:+g} "
                f"| {row['truth_angle_deg']:+.2f} | {row['best_score']:.1f} "
                f"| {_fmt(row['angle_err_b7_deg'] if isinstance(row['angle_err_b7_deg'], float) else float('nan'), '.3f')} "
                f"| {_fmt(row['center_err_px'] if isinstance(row['center_err_px'], float) else float('nan'))} |")
    else:
        lines.append("无：全部组合满足可背书判据。")
    backed = [step for step, ok in verdicts.items() if ok]
    lines += ["", "## 结论",
              "",
              f"- 可背书档距：{', '.join(f'{s:g}°' for s in backed) if backed else '无'}",
              f"- 不可背书档距：{', '.join(f'{s:g}°' for s in sorted(set(verdicts) - set(backed))) or '无'}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="宽角度粗档训练 B7 覆盖能力实测")
    parser.add_argument("--out", default=str(ROOT / "build" / "capacity"))
    parser.add_argument("--real-dir", default=str(DEFAULT_REAL_DIR))
    parser.add_argument("--steps", default=",".join(f"{s:g}" for s in STEPS))
    parser.add_argument("--train-source", choices=("canvas", "scene"), default="canvas")
    args = parser.parse_args()
    steps = tuple(float(v) for v in args.steps.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subjects = build_subjects(Path(args.real_dir))
    rows = run_matrix(subjects, steps, args.train_source)

    csv_path = out / f"angle_coverage_{args.train_source}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize(rows)
    (out / f"capacity_report_{args.train_source}.md").write_text(report, encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "runtime": {"sdk_version": sdk_version(), "dir": str(runtime_directory())},
        "train_source": args.train_source,
        "steps_deg": steps,
        "delta_fractions_of_step": DELTA_FRACTIONS,
        "angle_range_deg": [ANGLE_MIN, ANGLE_MAX],
        "threshold": THRESHOLD,
        "center_tolerance_px": CENTER_TOLERANCE_PX,
        "real_dir": args.real_dir,
        "rows": len(rows),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"逐样本 CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
