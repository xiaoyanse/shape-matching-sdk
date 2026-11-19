"""从 vendor line2Dup.cpp 逐字提取 SIMILARITY_LUT，生成 C 头文件。

保证 GPU/CPU 两侧查表与 vendor 字节级同源（手工转写易错，脚本提取零漂移）。
用法：python benchmarks/gpu-spike/scripts/gen_lut.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "vendor" / "shape_based_matching" / "line2Dup.cpp"
OUT = REPO / "benchmarks" / "gpu-spike" / "src" / "similarity_lut.h"


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    match = re.search(r"SIMILARITY_LUT\[256\]\s*=\s*\{(.*?)\};", text, re.S)
    if not match:
        raise SystemExit("vendor SIMILARITY_LUT 未找到")
    values = [v.strip() for v in match.group(1).split(",")]
    values = [v.replace("LUT3", "3") for v in values]
    numbers = [int(v) for v in values]
    if len(numbers) != 256 or any(v not in (0, 3, 4) for v in numbers):
        raise SystemExit(f"LUT 提取异常：{len(numbers)} 项")
    lines = ["#ifndef GPU_SPIKE_SIMILARITY_LUT_H",
             "#define GPU_SPIKE_SIMILARITY_LUT_H",
             "// gen_lut.py extracts this table verbatim from vendor line2Dup.cpp.",
             "#ifdef __CUDACC__", "__constant__", "#else", "static const", "#endif",
             "unsigned char SIMILARITY_LUT[256] = {"]
    for row in range(0, 256, 16):
        lines.append("    " + ", ".join(str(v) for v in numbers[row : row + 16]) + ",")
    lines.append("};")
    lines.append("#endif")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"LUT 256 项写入 {OUT}（4 的个数={numbers.count(4)}，3 的个数={numbers.count(3)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
