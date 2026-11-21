"""suggest_contours 验货门（AIENG shape-contour-pick-erase 前置）：输出质量实测。

确定性合成场景（真值已知）→ 调引擎 suggest_contours(_ex) → 逐条建议与真值
多边形 IoU 评估，覆盖：清洁基线/多目标/噪声/贴边目标/工业杂乱背景/roi_hint
聚焦/确定性复核/参数敏感性/耗时。产出逐场景表 + 给 AIENG 交互实施的结论。

用法：SHAPE_SDK_RUNTIME=<制品目录> uv run --project bindings/python \
    python benchmarks/suggest-contours/verify_pick.py
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bindings" / "python"))
from shape_sdk import suggest_contours, suggest_contours_ex  # noqa: E402

W, H = 1920, 1080
BG_GRAY = 40


def lcg(seed: int):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        yield (state >> 33) & 0xFFFFFFFF


def poly_gt(pts):
    return np.array(pts, np.int32)


def scene_multi(seed: int = 1, noise: float = 0.0, edge_targets: bool = False,
                clutter: bool = False):
    """4 类真值目标：矩形/六边形/L 形/三角形，面积已知降序；可选噪声/贴边/杂乱。"""
    img = np.full((H, W, 3), BG_GRAY, np.uint8)
    rng = lcg(seed)
    gts = []
    # 面积降序放置，留足间距
    gts.append(("rect", poly_gt([[200, 200], [700, 200], [700, 620], [200, 620]])))
    gts.append(("hex", poly_gt([[1200, 150], [1420, 280], [1420, 540], [1200, 670],
                                [980, 540], [980, 280]])))
    gts.append(("L", poly_gt([[300, 800], [640, 800], [640, 1000],
                             [500, 1000], [500, 900], [300, 900]])))
    gts.append(("tri", poly_gt([[1500, 900], [1750, 700], [1800, 950]])))
    if edge_targets:
        gts.append(("edge-r", poly_gt([[1780, 300], [2120, 300], [2120, 700], [1780, 700]])))
        gts.append(("edge-b", poly_gt([[1000, 860], [1300, 860], [1300, 1200], [1000, 1200]])))
    for name, pts in gts:
        shade = 120 + (next(rng) % 100)
        cv2.fillPoly(img, [np.clip(pts, 0, [W - 1, H - 1])], (shade, shade, shade))
    if clutter:  # 工业杂乱：条纹带 + 圆形铭牌
        for x in range(60, W - 60, 120):
            cv2.rectangle(img, (x, 40), (x + 50, 90), (90, 90, 90), -1)
        cv2.circle(img, (900, 900), 70, (160, 160, 160), -1)
        cv2.circle(img, (900, 900), 40, (60, 60, 60), -1)
    if noise > 0:
        count = int(H * W * noise)
        ys = (np.fromiter(rng, np.uint32, count) % H).astype(np.int32)
        xs = (np.fromiter(rng, np.uint32, count) % W).astype(np.int32)
        bits = (np.fromiter(rng, np.uint32, count) & 1) == 0
        img[ys, xs] = np.where(bits, 20, 220)[:, None]
    visible = []
    for name, pts in gts:
        clip = np.clip(pts, [0, 0], [W - 1, H - 1])
        visible.append((name, clip))
    return img, visible


def mask_of(pts, shape=(H, W)):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
    return m


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def evaluate(tag, suggestions, gts, max_iou_report=3):
    """逐真值找最佳建议（IoU），逐建议归类：命中(≥0.8)/部分(0.5-0.8)/伪建议(<0.5)。"""
    sug_masks = [mask_of(s) for s in suggestions]
    rows = []
    used = set()
    for name, gt in gts:
        gt_mask = mask_of(gt)
        best, best_i = 0.0, -1
        for i, sm in enumerate(sug_masks):
            v = iou(gt_mask, sm)
            if v > best:
                best, best_i = v, i
        if best_i >= 0:
            used.add(best_i)
        rows.append((name, best, best_i, len(suggestions[best_i]) if best_i >= 0 else 0))
    spurious = 0
    for i, sm in enumerate(sug_masks):
        if i in used:
            continue
        if max((iou(sm, mask_of(gt)) for _, gt in gts), default=0.0) < 0.5:
            spurious += 1
    verts = sorted(len(s) for s in suggestions)
    print(f"[{tag}] 建议 {len(suggestions)} 条（顶点 {verts}）命中 "
          f"{sum(1 for _, b, _, _ in rows if b >= 0.8)}/{len(gts)}，部分 "
          f"{sum(1 for _, b, _, _ in rows if 0.5 <= b < 0.8)}，伪建议 {spurious}")
    for name, best, idx, nv in rows:
        print(f"  {name:7s} best-IoU={best:.3f} (建议#{idx}, {nv} 顶点)")
    return rows, spurious


def main() -> int:
    print(f"场景 {W}x{H}；引擎 suggest_contours 默认参（max_sug=8/vert=8）")
    img, gts = scene_multi()
    s1 = suggest_contours(img)
    evaluate("S1 清洁多目标", s1, gts)

    img2, gts2 = scene_multi(noise=0.02)
    evaluate("S2 噪声 2%", suggest_contours(img2), gts2)

    img3, gts3 = scene_multi(noise=0.05)
    evaluate("S3 噪声 5%", suggest_contours(img3), gts3)

    img4, gts4 = scene_multi(edge_targets=True)
    evaluate("S4 贴边目标", suggest_contours(img4), gts4)

    img5, gts5 = scene_multi(clutter=True)
    evaluate("S5 工业杂乱", suggest_contours(img5), gts5)

    hint = (940, 120, 520, 580)  # 只圈六边形
    s6 = suggest_contours(img, roi_hint=hint)
    print(f"[S6 roi_hint 聚焦] 圈六边形 → {len(s6)} 条建议，各顶点数 {[len(s) for s in s6]}")
    hexa_gt = mask_of(gts[1][1])
    if s6:
        print(f"  首 IoU vs 六边形={iou(mask_of(s6[0]), hexa_gt):.3f}")

    again = suggest_contours(img)
    print(f"[S7 确定性] 两次调用逐字节一致: {again == s1}")

    print("[S8 参数敏感性] 工业杂乱场景 canny 扫描（顶点上限 16）：")
    for lo, hi in [(30, 90), (50, 150), (70, 210), (90, 270)]:
        s = suggest_contours_ex(img5, max_vertices_per=16, canny_low=float(lo),
                                canny_high=float(hi))
        rows, _ = evaluate(f"  canny {lo}/{hi}", s, gts5)
        hit = sum(1 for _, b, _, _ in rows if b >= 0.8)
        print(f"  → 命中 {hit}/{len(gts5)}")

    t0 = time.perf_counter()
    samples = []
    for _ in range(20):
        t1 = time.perf_counter()
        suggest_contours(img)
        samples.append((time.perf_counter() - t1) * 1000)
    print(f"[S9 耗时] 全图建议中位 {statistics.median(samples):.1f}ms（均值 "
          f"{(time.perf_counter() - t0) / 20 * 1000:.1f}ms/次，20 次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
