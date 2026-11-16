"""旧零 cfg 容器迁移：从各 fine 段序列化三参恢复 meta.cfg，输出可被当前版本加载的容器。

用法：SHAPE_SDK_RUNTIME=<制品目录> python scripts/migrate_zero_cfg.py <in> <out>
背景：P1 编码器把默认训练三参写为 meta.cfg [0,0,0]，当前版本严格校验拒绝加载。
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"VF2S3"
DETECTOR_META_BYTES = 1 + 12 + 64
DETECTOR_TUPLE_BYTES = 20


def split_container(container: bytes) -> list[bytes]:
    if container[:5] != MAGIC:
        raise ValueError("不是两级容器格式（VF2S3）")
    count = struct.unpack_from("<Q", container, 5)[0]
    lengths = struct.unpack_from("<" + "Q" * (count + 3), container, 13)
    parts, offset = [], 13 + 8 * (count + 3)
    for length in lengths:
        end = offset + length
        if end > len(container):
            raise ValueError("段表长度超出容器，容器已损坏")
        parts.append(container[offset:end])
        offset = end
    if count < 1 or len(parts) != count + 3:
        raise ValueError("容器段数非法")
    return parts


def rebuild_container(parts: list[bytes]) -> bytes:
    return (MAGIC + struct.pack("<Q", len(parts) - 3) +
            struct.pack("<" + "Q" * len(parts), *map(len, parts)) + b"".join(parts))


def segment_detector(segment: bytes) -> tuple[int, float, float]:
    offset = segment.index(b"\0") + DETECTOR_META_BYTES
    if offset + DETECTOR_TUPLE_BYTES > len(segment):
        raise ValueError("fine 段过短，读不到 detector 三参")
    return struct.unpack_from("<Idd", segment, offset)


def fine_params(parts: list[bytes]) -> tuple[int, float, float]:
    params = [segment_detector(seg) for seg in parts[:-3]]
    head = params[0]
    divergent = [p for p in params[1:] if p != head]
    if divergent:
        raise ValueError(f"各 fine 段序列化三参不一致：{head} 与 {divergent[0]}")
    return head


def load_verify(container: bytes, expected: tuple[int, float, float]) -> tuple[bool, str]:
    sys.path.insert(0, str(ROOT / "bindings" / "python"))
    from shape_sdk import NativeModel, NativeShapeError

    try:
        with NativeModel(container) as model:
            meta = model.metadata
    except NativeShapeError as exc:
        return False, f"新库加载失败：{exc}"
    actual = (meta["feature_count"], meta["weak_th"], meta["strong_th"])
    if actual != expected:
        return False, f"metadata 三参不符：实际 {actual}，期望 {expected}"
    return True, (f"加载校验通过：metadata 三参 = ({actual[0]}, {actual[1]}, {actual[2]})；"
                  "同图匹配行为请以原始参考图在迁移前库上的命中为基准对照验证。")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法：python scripts/migrate_zero_cfg.py <in> <out>", file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    try:
        container = src.read_bytes()
    except OSError as exc:
        print(f"读取输入失败：{exc}", file=sys.stderr)
        return 2
    try:
        parts = split_container(container)
    except ValueError as exc:
        print(f"{exc}；单级旧格式不受零 cfg 收紧影响，无需迁移。", file=sys.stderr)
        return 1
    try:
        meta = json.loads(parts[-1].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        print(f"meta 段不是合法 JSON：{exc}；容器已损坏，未改动。", file=sys.stderr)
        return 1
    cfg = meta.get("cfg")
    if not isinstance(cfg, list) or len(cfg) < 13:
        print("meta.cfg 缺失或长度非法，容器结构与预期不符", file=sys.stderr)
        return 1
    current = tuple(cfg[10:13])
    if all(current):
        print(f"meta.cfg 三参 {list(current)} 已非零，无需迁移；原样输出（幂等）。"
              "若该容器当前版本加载仍被拒，请用原图重新训练。")
        dst.write_bytes(container)
        return 0
    try:
        recovered = fine_params(parts)
    except ValueError as exc:
        print(f"{exc}；无法安全恢复，原样输出，容器未改动。", file=sys.stderr)
        dst.write_bytes(container)
        return 1
    # 段内 <Idd 序为 (features, weak, strong)，meta.cfg 序为 (weak, strong, features)
    cfg[10:13] = [recovered[1], recovered[2], recovered[0]]
    migrated_parts = list(parts)
    migrated_parts[-1] = json.dumps(meta).encode("utf-8")
    migrated = rebuild_container(migrated_parts)
    print(f"meta.cfg 三参 {list(current)} → 恢复自 {len(parts) - 3} 个 fine 段的 {list(recovered)}")
    ok, report = load_verify(migrated, recovered)
    print(report)
    if not ok:
        print("加载校验未通过，未写出迁移结果。"
              "若运行库不可用，先构建装配（scripts/build_native.ps1）"
              "或设置 SHAPE_SDK_RUNTIME 指向制品目录后重试。", file=sys.stderr)
        return 1
    try:
        dst.write_bytes(migrated)
    except OSError as exc:
        print(f"写出失败：{exc}", file=sys.stderr)
        return 2
    print(f"迁移完成：{dst}（{len(migrated)} 字节）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
