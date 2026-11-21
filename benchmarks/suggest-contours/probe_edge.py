"""贴边目标追问探针：顶点上限/roi_hint 对贴边目标建议的影响（验货门补充）。"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bindings" / "python"))
sys.path.insert(0, str(REPO / "benchmarks" / "suggest-contours"))
from verify_pick import scene_multi, mask_of, iou  # noqa: E402
from shape_sdk import suggest_contours  # noqa: E402

img, gts = scene_multi(edge_targets=True)
for cap in (8, 16, 32, 64):
    s = suggest_contours(img, max_suggestions=16, max_vertices_per=cap)
    edges = []
    for name, gt in gts:
        if not name.startswith("edge"):
            continue
        gm = mask_of(gt)
        best = max((iou(mask_of(x), gm) for x in s), default=0.0)
        edges.append(f"{name}={best:.3f}")
    print(f"cap={cap:2d} 条数={len(s)} 顶点={[len(x) for x in s][:8]} 贴边: {edges}")

hint = (1760, 260, 160, 480)
s = suggest_contours(img, roi_hint=hint, max_vertices_per=16)
best = max((iou(mask_of(x), mask_of(gts[4][1])) for x in s), default=0.0) if s else -1
print("roi_hint 圈贴边目标可见区:", len(s), "条", [len(x) for x in s], f"IoU vs 可见区={best:.3f}")
