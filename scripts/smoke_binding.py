"""官方 Python 绑定加载冒烟：manifest 哈希校验 + ABI 协商 + 函数签名绑定。

用法（在含 numpy 的 Python 环境）：
    set SHAPE_SDK_RUNTIME=<制品目录>   # 缺省 dist/runtime
    python scripts/smoke_binding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

from shape_sdk import load_library, runtime_directory  # noqa: E402


def main() -> int:
    library = load_library()
    print("runtime:", runtime_directory())
    print("ABI:", library.vf_shape_abi_version())
    print("冒烟通过：manifest 校验 + ABI 协商 + 函数签名绑定全部成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
