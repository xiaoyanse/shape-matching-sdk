"""PE 导入表预验（弹窗红线）：列出 exe 的 DLL 依赖并核对可解析性。

用法：PATH 前置好依赖目录后运行
    python pe_imports.py <exe>
退出码：0=全部可解析；1=有缺失（此时启动 exe 会弹"找不到 DLL"对话框）。
"""

from __future__ import annotations

import ctypes
import struct
import sys
from pathlib import Path


def rva_to_offset(data: bytes, rva: int, sections: list) -> int:
    for name, vaddr, vsize, raw, rawsize in sections:
        if vaddr <= rva < vaddr + max(vsize, rawsize):
            return raw + (rva - vaddr)
    return -1


def import_dlls(path: str) -> list[str]:
    data = Path(path).read_bytes()
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe_offset : pe_offset + 4] == b"PE\x00\x00", "非 PE 文件"
    machine, _sections, _ts, _psym, _nsym, opt_size, _flags = struct.unpack_from(
        "<HHIIIHH", data, pe_offset + 4)
    assert machine == 0x8664, "仅支持 x64 PE"
    opt = pe_offset + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    assert magic == 0x20B, "仅支持 PE32+"
    data_dirs = opt + 112  # PE32+ 数据目录
    import_rva, import_size = struct.unpack_from("<II", data, data_dirs + 8)
    section_count = _sections
    sec_table = opt + opt_size
    sections = []
    for i in range(section_count):
        entry = sec_table + i * 40
        name = data[entry : entry + 8].rstrip(b"\0").decode()
        vsize, vaddr, rawsize, raw = struct.unpack_from("<IIII", data, entry + 8)
        sections.append((name, vaddr, vsize, raw, rawsize))
    dlls = []
    if import_rva == 0:
        return dlls
    table = rva_to_offset(data, import_rva, sections)
    for i in range(200):
        entry = table + i * 20
        original, _ts, _fc, name_rva, _thunk = struct.unpack_from("<IIIII", data, entry)
        if original == 0 and name_rva == 0:
            break
        name_off = rva_to_offset(data, name_rva, sections)
        end = data.index(b"\0", name_off)
        dlls.append(data[name_off:end].decode("ascii"))
    return dlls


SEARCH_PATHS = [
    Path("C:/Windows/System32"),
    Path("C:/Windows/SysWOW64"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print("用法：python pe_imports.py <exe>")
        return 2
    dlls = import_dlls(sys.argv[1])
    missing = []
    print(f"导入 DLL 共 {len(dlls)} 个：")
    for dll in dlls:
        found = None
        if ctypes.windll.kernel32.SearchPathW(None, dll, None, 0, None, None) != 0:
            # SearchPath 用当前进程搜索序（含 PATH）
            found = "PATH"
        else:
            for base in SEARCH_PATHS:
                if (base / dll).exists():
                    found = str(base)
                    break
        status = found if found else "缺失!"
        print(f"  {dll:36s} {status}")
        if not found:
            missing.append(dll)
    if missing:
        print(f"缺失 {len(missing)} 个，禁止启动（会触发系统弹窗）")
        return 1
    print("全部可解析，可安全启动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
