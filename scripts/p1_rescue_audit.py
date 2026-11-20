"""P1 遮挡救分离线审计驱动：帧清单、先决单调性门、假峰采样、误报矩阵、归因复跑对拍。

只测量不改基准脚本；用 --out-root 隔离返工验证产物。
    python -B scripts/p1_rescue_audit.py <command> --out-root build/p1-rework
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cv2  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "perturbation"))
sys.path.insert(0, str(ROOT / "bindings" / "python"))

import generate  # noqa: E402
import run_baseline  # noqa: E402

OUT = ROOT / "build" / "p1-rework"
AUDIT = OUT / "audit"
BASELINE_ATTRIBUTION = ROOT / "build" / "attribution" / "attribution.csv"
RATIOS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
FP_SEEDS = 20
EQUIV_FIELDS = ("matched", "best_score", "n_hits", "center_err_px", "attribution",
                "coarse_candidates", "windows_searched", "final_candidates",
                *(f"window_best_{i}" for i in range(8)))


def _exe() -> Path:
    path = OUT / "shape_occlusion_rescue_test.exe"
    if not path.is_file():
        raise SystemExit(f"未找到 {path}，先在独立构建目录编译")
    return path


def _write_manifest(path: Path, rows: list[dict], header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"清单落盘 {len(rows)} 行 → {path}")


def cmd_prepare() -> int:
    frames = AUDIT / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    neutral, _ = generate.render_frame(0)
    ref_path = frames / "reference.png"
    cv2.imwrite(str(ref_path), neutral)

    cases = [{"id": "baseline/neutral", "ratio": 0.0, "anchor": "none", "seed": 0}]
    for ratio in RATIOS:
        for variant in range(10):
            anchor = "center" if variant < 2 else "edge"
            cases.append({"id": f"occl/{ratio:.2f}@{anchor}/v{variant:02d}", "ratio": ratio,
                          "anchor": anchor, "seed": variant})
    rows = []
    for case in cases:
        frame, truth = generate.render_frame(case["seed"], occlusion_ratio=case["ratio"],
                                             occlusion_anchor=case["anchor"])
        name = case["id"].replace("/", "_")
        png = frames / f"{name}.png"
        cv2.imwrite(str(png), frame)
        rows.append({"id": case["id"], "ratio": case["ratio"], "anchor": case["anchor"],
                     "seed": case["seed"], "png": str(png),
                     "cx": truth["center"][0], "cy": truth["center"][1]})
    _write_manifest(AUDIT / "audit_manifest.csv", rows,
                    ["id", "ratio", "anchor", "seed", "png", "cx", "cy"])

    fp_rows = []
    for bg in generate.BG_LEVELS:
        for seed in range(FP_SEEDS):
            frame, _ = generate.render_frame(seed, bg=bg, present=False)
            name = f"negative/{bg}/s{seed:02d}".replace("/", "_")
            png = frames / f"{name}.png"
            cv2.imwrite(str(png), frame)
            fp_rows.append({"id": f"negative/{bg}/s{seed:02d}", "png": str(png)})
    _write_manifest(AUDIT / "fp_manifest.csv", fp_rows, ["id", "png"])
    return 0


def _run_exe(mode: str, manifest: str) -> list[dict]:
    completed = subprocess.run(
        [str(_exe()), mode, str(AUDIT / "frames" / "reference.png"), str(AUDIT / manifest)],
        capture_output=True, text=True, timeout=1800)
    if completed.returncode != 0:
        raise SystemExit(f"{mode} 失败 rc={completed.returncode}：{completed.stderr[-2000:]}")
    lines = completed.stdout.strip().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:] if line]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def cmd_vgate() -> int:
    rows = _run_exe("--v-audit", "audit_manifest.csv")
    print(f"先决门采样 {len(rows)} 案")
    per_ratio: dict[float, list[float]] = {}
    unmeasured = 0
    all_ratio, all_vratio = [], []
    for row in rows:
        ratio = float(row["ratio"])
        v = int(row["v"])
        nf = int(row["nf"])
        if not 0 <= v <= nf or nf == 0:
            unmeasured += 1
            continue
        per_ratio.setdefault(ratio, []).append(v / nf)
        all_ratio.append(ratio)
        all_vratio.append(v / nf)
    print("ratio  mean_v/nf  min  max  n")
    means = []
    for ratio in sorted(per_ratio):
        values = per_ratio[ratio]
        means.append((ratio, sum(values) / len(values)))
        print(f"{ratio:.2f}  {means[-1][1]:.3f}  {min(values):.3f}  {max(values):.3f}  {len(values)}")
    if len(all_ratio) < 2:
        sys.stdout.write("没有足量的 v/nf 实测值\n")
        return 1
    correlation = _pearson(all_ratio, all_vratio)
    print(f"Pearson(ratio, v/nf) = {correlation:.3f}，真值位置不可测案 {unmeasured}")
    monotone = all(means[i][1] >= means[i + 1][1] - 0.02 for i in range(len(means) - 1))
    expected = {0.0: 1, **dict.fromkeys(RATIOS, 10)}
    coverage = ({ratio: len(values) for ratio, values in per_ratio.items()} == expected
                and len({row["id"] for row in rows}) == len(rows) and unmeasured == 0)
    gate = coverage and monotone and correlation <= -0.7
    print(f"单调性判定：{'通过' if gate else '不通过'}"
          f"（逐档均值非增={monotone}，相关={correlation:.3f}≤-0.7）")
    rescued_rows = [(row["id"], row["adj"]) for row in rows if row["rescued"] == "1"]
    print(f"adj≥70 预演救活 {len(rescued_rows)} 案："
          + ", ".join(f"{rid}(adj={adj})" for rid, adj in rescued_rows[:16]))
    (AUDIT / "vgate_summary.json").write_text(json.dumps({
        "coverage": coverage, "correlation": correlation, "monotone": monotone, "gate": gate,
        "means": {f"{r:.2f}": m for r, m in means}, "unmeasured": unmeasured,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if gate else 1


def cmd_fpband() -> int:
    rows = _run_exe("--fp-probe", "fp_manifest.csv")
    # 探针按编译期收集门槛（=ceil(β·70)）枚举，其下限即带下限——β 变更时
    # 带随探针自动移动，不在此处硬编码带边界
    band = [float(row["raw"]) for row in rows if float(row["raw"]) < 70.0
            and row["pose_angle"] != "error"]
    errors = sum(1 for row in rows if row["pose_angle"] == "error")
    over = [float(row["raw"]) for row in rows if float(row["raw"]) >= 70.0
            and row["pose_angle"] != "error"]
    print(f"假峰采样行 {len(rows)}（预算错误帧 {errors}）；门槛-70 分带命中 {len(band)}，"
          f"70+ 命中 {len(over)}")
    if band:
        print(f"带内分布：min={min(band):.2f} max={max(band):.2f} "
              f"p95={sorted(band)[int(len(band)*0.95)]:.2f} mean={sum(band)/len(band):.2f}")
    by_bg: dict[str, list[float]] = {}
    for row in rows:
        bg = row["id"].split("/")[1]
        raw = float(row["raw"])
        if row["pose_angle"] != "error" and raw < 70.0:
            by_bg.setdefault(bg, []).append(raw)
    for bg, values in sorted(by_bg.items()):
        print(f"  {bg}: n={len(values)} max={max(values):.2f}")
    (AUDIT / "fpband_summary.json").write_text(json.dumps({
        "band_count": len(band), "band_max": max(band) if band else 0.0,
        "over_count": len(over), "budget_errors": errors,
        "by_bg": {k: {"n": len(v), "max": max(v)} for k, v in by_bg.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def _runtime() -> Path:
    value = os.environ.get("SHAPE_SDK_RUNTIME")
    expected = OUT / "runtime"
    if not value or Path(value).resolve() != expected.resolve():
        raise SystemExit(f"需要 SHAPE_SDK_RUNTIME 指向当前输出根的运行时：{expected}")
    return expected


def cmd_enginefp() -> int:
    _runtime()
    from shape_sdk import NativeModel  # noqa: E402

    model_bytes, _ = run_baseline.train_reference_model()
    rows = []
    false_positive = 0
    with NativeModel(model_bytes) as model:
        for bg in generate.BG_LEVELS:
            for seed in range(FP_SEEDS):
                frame, _ = generate.render_frame(seed, bg=bg, present=False)
                hits = model.match(frame, threshold=70.0)
                false_positive += len(hits)
                rows.append({"kind": "negative", "bg": bg, "seed": seed, "ratio": "",
                             "anchor": "", "n_hits": len(hits),
                             "best_score": round(max((h.score for h in hits), default=0.0), 2),
                             "center_err_px": ""})
        for bg in ("noise", "stripes"):
            for ratio in (0.4, 0.5):
                for seed in range(5):
                    anchor = "center" if seed % 2 == 0 else "edge"
                    frame, _ = generate.render_frame(seed, bg=bg, occlusion_ratio=ratio,
                                                     occlusion_anchor=anchor)
                    hits = model.match(frame, threshold=70.0)
                    best = max(hits, key=lambda h: h.score) if hits else None
                    err = (round(float(np.hypot(best.center[0] - 960, best.center[1] - 540)), 2)
                           if best else "")
                    rows.append({"kind": "clutter_occluded", "bg": bg, "seed": seed,
                                 "ratio": ratio, "anchor": anchor, "n_hits": len(hits),
                                 "best_score": round(best.score, 2) if best else 0.0,
                                 "center_err_px": err})
    negatives = [r for r in rows if r["kind"] == "negative"]
    print(f"负样本 {len(negatives)} 帧误报合计 {false_positive}（要求 0）；"
          f"杂乱×遮挡记录 {len(rows) - len(negatives)} 行")
    with (OUT / "enginefp.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return 1 if false_positive else 0


def _attribution_rows(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    cases = 1 + len(RATIOS) * 10
    if len(rows) != cases or len(by_id) != cases:
        raise SystemExit(f"归因矩阵必须有 {cases} 案且无重复：{path}")
    return by_id


def _recovery_failures(old_rows: dict, new_rows: dict) -> list[str]:
    failures = []
    for ratio, minimum in ((0.35, 9), (0.40, 9), (0.45, 8)):
        group = [r for r in new_rows.values() if float(r["ratio"]) == ratio]
        passed = sum(r["matched"] == "True" for r in group)
        if len(group) != 10 or passed < minimum:
            failures.append(f"{ratio:.0%}: {passed}/{len(group)}，要求 ≥{minimum}/10")
    subset = [key for key, row in old_rows.items()
              if float(row["ratio"]) == 0.50 and row["attribution"] == "fine_insufficient"]
    passed = sum(new_rows[key]["matched"] == "True" for key in subset)
    if len(subset) != 7 or passed < 3:
        failures.append(f"50% 精层不足子集: {passed}/{len(subset)}，要求 ≥3/7")
    sys.stdout.write(f"50% 精层不足子集：{passed}/{len(subset)}\n")
    return failures


def cmd_attribution() -> int:
    runtime = _runtime()
    env = {**os.environ, "SHAPE_SDK_RUNTIME": str(runtime)}
    completed = subprocess.run(
        [sys.executable, "-B", str(ROOT / "benchmarks" / "occlusion_attribution.py"),
         "--out", str(OUT / "attribution")],
        env=env, capture_output=True, text=True, timeout=3600, cwd=str(ROOT),
        encoding="utf-8", errors="replace")
    print(completed.stdout[-3000:])
    if completed.returncode != 0:
        raise SystemExit(f"归因复跑失败 rc={completed.returncode}：{completed.stderr[-2000:]}")
    new_rows = _attribution_rows(OUT / "attribution" / "attribution.csv")
    old_rows = _attribution_rows(BASELINE_ATTRIBUTION)
    if new_rows.keys() != old_rows.keys() or any(
        any(old[field] != new_rows[key][field] for field in ("ratio", "anchor", "seed"))
        for key, old in old_rows.items()
    ):
        raise SystemExit("归因矩阵与原基线的案号或输入参数不一致")
    drift = []
    for case_id, old in old_rows.items():
        new = new_rows[case_id]
        if float(old["ratio"]) > 0.30:
            continue
        for field in EQUIV_FIELDS:
            if str(old[field]) != str(new[field]):
                drift.append(f"{case_id}.{field}: {old[field]} → {new[field]}")
    print(f"等价 CSV 摘要（0/25/30% 共 {sum(1 for r in old_rows.values() if float(r['ratio']) <= 0.30)} 案）："
          f"{'摘要字段一致，逐位仍须跑 test_p1_match_equivalence.py' if not drift else f'{len(drift)} 处漂移'}")
    for item in drift[:20]:
        print(f"  {item}")
    print("恢复矩阵（matched 旧→新）：")
    for ratio in RATIOS:
        group = [r for r in new_rows.values() if abs(float(r["ratio"]) - ratio) < 1e-9]
        old_group = [r for r in old_rows.values() if abs(float(r["ratio"]) - ratio) < 1e-9]
        if group:
            print(f"  {ratio:.0%}: {sum(r['matched'] == 'True' for r in old_group)}/"
                  f"{len(old_group)} → {sum(r['matched'] == 'True' for r in group)}/{len(group)}"
                  f"  mean_best={sum(float(r['best_score']) for r in group)/len(group):.1f}")
    match_ms_old = [float(r["match_ms"]) for r in old_rows.values() if float(r["match_ms"]) > 0]
    match_ms_new = [float(r["match_ms"]) for r in new_rows.values() if float(r["match_ms"]) > 0]
    print(f"match_ms 历史均值 {sum(match_ms_old)/len(match_ms_old):.1f} → "
          f"{sum(match_ms_new)/len(match_ms_new):.1f}；2× 门须同机同批复核，历史数据不判通过")
    failures = _recovery_failures(old_rows, new_rows)
    for failure in failures:
        sys.stdout.write(f"未达恢复门：{failure}\n")
    return int(bool(drift or failures))


def cmd_assemble_runtime() -> int:
    runtime = OUT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    dll = OUT / "visionforge_shape.dll"
    if not dll.exists():
        raise SystemExit(f"缺少主库 {dll}，先构建 build/p1")
    import re
    import shutil

    cmake_text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    version = re.search(r"VERSION\s+(\d+\.\d+\.\d+)", cmake_text).group(1)
    shutil.copyfile(dll, runtime / "visionforge_shape.dll")
    opencv_src = next(p for p in Path(r"C:\opencv\build\x64\vc16\bin").glob("opencv_world*.dll")
                      if not p.name.endswith("d.dll"))
    shutil.copyfile(opencv_src, runtime / opencv_src.name)
    files = {}
    for name in ("visionforge_shape.dll", opencv_src.name):
        files[name] = hashlib.sha256((runtime / name).read_bytes()).hexdigest()
    manifest = {"abi_version": 3, "sdk_version": version, "platform": "windows-x64",
                "files": files}
    (runtime / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"运行时已装配 → {runtime}（sdk_version {version}）")
    return 0


def main() -> int:
    global OUT, AUDIT
    commands = {"prepare": cmd_prepare, "vgate": cmd_vgate, "fpband": cmd_fpband,
                "enginefp": cmd_enginefp, "attribution": cmd_attribution,
                "assemble-runtime": cmd_assemble_runtime}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    parser.add_argument("--out-root", type=Path, default=OUT)
    args = parser.parse_args()
    build = (ROOT / "build").resolve()
    protected = [(build / name).resolve() for name in ("p1", "attribution")]
    output = args.out_root.resolve()
    if output == build or not output.is_relative_to(build) or any(
        output.is_relative_to(path) for path in protected
    ):
        parser.error("--out-root 须位于独立 build 子目录，不得覆盖 build/p1 或 build/attribution")
    for name in ("audit", "audit/frames", "runtime", "attribution"):
        if (output / name).resolve() != output / name:
            parser.error(f"输出子目录不得重定向：{output / name}")
    OUT, AUDIT = output, output / "audit"
    return commands[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
