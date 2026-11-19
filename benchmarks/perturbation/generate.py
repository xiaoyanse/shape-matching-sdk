"""受控扰动基准帧生成器：富边缘目标 + 背景/遮挡/光照三因子独立扰动。

全部随机性来自 np.random.default_rng(seed)，同参数同种子输出字节一致。
"""

from __future__ import annotations

import numpy as np
import cv2

FRAME_W, FRAME_H = 1920, 1080
TARGET_SIZE = 256
TARGET_CENTER = (960.0, 540.0)
ANGLE_RANGE = (-30.0, 30.0)
BG_LEVELS = ("flat", "noise", "stripes")
OCCLUSION_RATIOS = tuple(round(0.1 * n, 1) for n in range(9))
OCCLUSION_ANCHORS = ("center", "edge")
GAIN_GRID = (0.5, 0.875, 1.25, 1.625, 2.0)
BIAS_GRID = (-60, -30, 0, 30, 60)
BG_BASE = 40
OCCLUSION_GRAY = 128


def draw_target(size: int = TARGET_SIZE) -> np.ndarray:
    """黑底线框目标图案（复刻 tests/geometric_refinement_test.cpp 的富边缘画法）。"""
    canvas = np.zeros((size, size), np.uint8)
    m = size / 256.0
    pt = lambda x, y: (int(round(x * m)), int(round(y * m)))
    cv2.rectangle(canvas, pt(36, 50), pt(212, 202), 240, 4, cv2.LINE_AA)
    cv2.circle(canvas, pt(132, 154), int(round(70 * m)), 220, 4, cv2.LINE_AA)
    cv2.line(canvas, pt(80, 224), pt(240, 90), 180, 5, cv2.LINE_AA)
    cv2.rectangle(canvas, pt(190, 228), pt(218, 250), 200, -1)
    cv2.line(canvas, pt(150, 60), pt(150, 118), 210, 3, cv2.LINE_AA)
    cv2.circle(canvas, pt(66, 100), int(round(24 * m)), 230, 3, cv2.LINE_AA)
    return canvas


def paint_background(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "flat":
        return np.full((FRAME_H, FRAME_W), BG_BASE, np.uint8)
    if kind == "noise":
        return np.clip(rng.normal(BG_BASE, 20.0, (FRAME_H, FRAME_W)), 0, 255).astype(np.uint8)
    if kind == "stripes":
        theta = np.deg2rad(rng.uniform(0.0, 180.0))
        period = rng.uniform(16.0, 32.0)
        xx, yy = np.meshgrid(np.arange(FRAME_W, dtype=np.float32),
                             np.arange(FRAME_H, dtype=np.float32))
        phase = ((xx * np.cos(theta) + yy * np.sin(theta)) / period) % 2.0
        stripes = np.where(phase < 1.0, 25.0, 65.0)
        return np.clip(stripes + rng.normal(0.0, 4.0, (FRAME_H, FRAME_W)), 0, 255).astype(np.uint8)
    raise ValueError(f"未知背景档：{kind}")


def _stamp_target(frame: np.ndarray, angle: float) -> None:
    target = draw_target()
    cx, cy = target.shape[1] / 2.0, target.shape[0] / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[0, 2] += TARGET_CENTER[0] - cx
    matrix[1, 2] += TARGET_CENTER[1] - cy
    warped = cv2.warpAffine(target, matrix, (FRAME_W, FRAME_H),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.full_like(target, 255), matrix, (FRAME_W, FRAME_H),
                          flags=cv2.INTER_LINEAR, borderValue=0)
    frame[mask > 0] = warped[mask > 0]


def _apply_occlusion(frame: np.ndarray, rng: np.random.Generator,
                     ratio: float, anchor: str) -> None:
    edge = TARGET_SIZE * float(np.sqrt(ratio))
    cx, cy = TARGET_CENTER
    if anchor == "center":
        bx, by = cx, cy
    elif anchor == "edge":
        theta = np.deg2rad(rng.uniform(0.0, 360.0))
        reach = TARGET_SIZE * 0.5 - edge * 0.5
        bx = cx + np.cos(theta) * reach
        by = cy + np.sin(theta) * reach
    else:
        raise ValueError(f"未知遮挡锚点：{anchor}")
    x0, y0 = int(round(bx - edge / 2)), int(round(by - edge / 2))
    x1, y1 = int(round(bx + edge / 2)), int(round(by + edge / 2))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, FRAME_W), min(y1, FRAME_H)
    if x1 > x0 and y1 > y0:
        frame[y0:y1, x0:x1] = OCCLUSION_GRAY


def render_frame(
    seed: int,
    *,
    bg: str = "flat",
    occlusion_ratio: float = 0.0,
    occlusion_anchor: str = "none",
    gain: float = 1.0,
    bias: float = 0.0,
    present: bool = True,
) -> tuple[np.ndarray, dict]:
    """按参数合成一帧 BGR；返回 (帧, 真值)。同参数同种子字节一致。"""
    rng = np.random.default_rng(seed)
    frame = paint_background(rng, bg)
    angle = float(rng.uniform(*ANGLE_RANGE)) if present else float("nan")
    if present:
        _stamp_target(frame, angle)
    if occlusion_ratio > 0.0:
        if not 0.0 < occlusion_ratio <= OCCLUSION_RATIOS[-1]:
            raise ValueError(f"遮挡比超出梯度范围：{occlusion_ratio}")
        _apply_occlusion(frame, rng, occlusion_ratio, occlusion_anchor)
    if gain != 1.0 or bias != 0.0:
        frame = np.clip(frame.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    truth = {
        "seed": seed,
        "bg": bg,
        "occlusion_ratio": occlusion_ratio,
        "occlusion_anchor": occlusion_anchor,
        "gain": gain,
        "bias": bias,
        "present": present,
        "center": list(TARGET_CENTER),
        "angle_deg": None if not present else round(angle, 4),
    }
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), truth


def build_matrix(n_seeds: int = 5) -> list[dict]:
    """因子扫描清单：三因子独立梯度，共享中性基点（自动去重），另含无目标负样本组。"""
    cases: list[dict] = []
    seen: set[tuple] = set()

    def add(factor: str, level: str, **params) -> None:
        for seed in range(n_seeds):
            key = (params.get("bg"), params.get("occlusion_ratio", 0.0),
                   params.get("occlusion_anchor", "none"),
                   params.get("gain", 1.0), params.get("bias", 0.0),
                   params.get("present", True), seed)
            if key in seen:
                continue
            seen.add(key)
            cases.append({
                "id": f"{factor}/{level}/s{seed}",
                "factor": factor, "level": level, "seed": seed, **params,
            })

    add("baseline", "neutral")
    for kind in BG_LEVELS:
        add("background", kind, bg=kind)
    for ratio in OCCLUSION_RATIOS[1:]:
        for anchor in OCCLUSION_ANCHORS:
            add("occlusion", f"{ratio:.1f}@{anchor}", occlusion_ratio=ratio, occlusion_anchor=anchor)
    for gain in GAIN_GRID:
        for bias in BIAS_GRID:
            add("lighting", f"g{gain:.3f}/b{bias:+d}", gain=gain, bias=bias)
    add("negative", "absent", present=False)
    return cases


def repro_check(n_cases: int = 12) -> bool:
    """同清单条目重复渲染逐字节比对，验证可复现性。"""
    matrix = build_matrix()
    step = max(1, len(matrix) // n_cases)
    picks = matrix[::step][:n_cases]
    for case in picks:
        first, _ = render_frame(**_case_args(case))
        second, _ = render_frame(**_case_args(case))
        if first.tobytes() != second.tobytes():
            print(f"不可复现：{case['id']}")
            return False
    print(f"可复现性通过：{len(picks)} 条清单条目两次渲染字节一致")
    return True


def _case_args(case: dict) -> dict:
    defaults = {
        "bg": "flat", "occlusion_ratio": 0.0, "occlusion_anchor": "none",
        "gain": 1.0, "bias": 0.0, "present": True,
    }
    return {"seed": case["seed"], **{k: case.get(k, default) for k, default in defaults.items()}}


def dump_frames(matrix: list[dict], out_dir, limit: int = 0) -> list[dict]:
    """按清单落盘 PNG + 真值 JSON 清单（产物归 build/perturb/，不进 benchmarks）。"""
    import json
    from pathlib import Path

    out = Path(out_dir)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in (matrix if limit <= 0 else matrix[:limit]):
        frame, truth = render_frame(**_case_args(case))
        name = case["id"].replace("/", "_")
        cv2.imwrite(str(frames_dir / f"{name}.png"), frame)
        rows.append({"id": case["id"], "frame": f"frames/{name}.png", **truth})
    (out / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已落盘 {len(rows)} 帧 + manifest.json → {out}")
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="受控扰动基准帧生成器")
    parser.add_argument("--out", default=None, help="落盘目录（缺省只跑自检不落盘）")
    parser.add_argument("--limit", type=int, default=0, help="只落盘前 N 条（0=全量）")
    parser.add_argument("--seeds", type=int, default=5, help="每档种子数")
    args = parser.parse_args()
    if not repro_check():
        return 1
    if args.out:
        dump_frames(build_matrix(args.seeds), args.out, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
