"""同帧批量匹配；模型独立失败，锁按稳定顺序保护原生调用寿命。"""

from __future__ import annotations

import ctypes as ct
import math
from contextlib import ExitStack

import numpy as np

from .native import NativeCandidate, NativeModel, _candidates, _image_args
from .native_library import BatchItem, Candidate, NativeShapeError, invoke


def match_batch(
    requests: list[tuple[NativeModel, float, int]], frame_bgr: np.ndarray
) -> list[list[NativeCandidate] | NativeShapeError]:
    image_args = _image_args(frame_bgr)
    if not 1 <= len(requests) <= 16:
        raise NativeShapeError("单帧批量匹配须包含 1–16 个模型")
    results: list[list[NativeCandidate] | NativeShapeError] = [
        NativeShapeError("模板匹配未执行") for _ in requests
    ]
    with ExitStack() as locks:
        models = {id(model): model for model, _, _ in requests}
        for _, model in sorted(models.items()):
            locks.enter_context(model._lock)
        valid = []
        buffers = []
        library = None
        for index, (model, threshold, capacity) in enumerate(requests):
            try:
                model._require_open()
                if not math.isfinite(threshold) or not 0 < threshold <= 100:
                    raise NativeShapeError("相似度阈值须大于 0 且不大于 100")
                if not isinstance(capacity, int) or not 2 <= capacity <= 256:
                    raise NativeShapeError("候选数量须为 2–256")
                if library is not None and model._library is not library:
                    raise NativeShapeError("批量模型来自不同原生组件")
                library = model._library
                buffers.append((Candidate * capacity)())
                valid.append((index, model, threshold, capacity))
            except NativeShapeError as exc:
                results[index] = exc
        if not valid:
            return results
        items = (BatchItem * len(valid))(
            *[
                BatchItem(model._handle.value, threshold, output, capacity, 0, 1)
                for (_, model, threshold, capacity), output in zip(valid, buffers, strict=True)
            ]
        )
        invoke(library, "match_batch", *image_args, items, len(items), ct.sizeof(BatchItem))
        for (index, model, threshold, capacity), item, output in zip(
            valid, items, buffers, strict=True
        ):
            try:
                if (
                    item.status not in (0, 1, 2)
                    or item.count > capacity
                    or item.handle != model._handle.value
                    or item.threshold != threshold
                    or item.capacity != capacity
                    or ct.cast(item.candidates, ct.c_void_p).value != ct.addressof(output)
                ):
                    raise NativeShapeError("原生批量输出状态或数量非法")
                if item.status:
                    if item.count:
                        raise NativeShapeError("原生失败项返回了候选")
                    raise NativeShapeError(item.error.decode("utf-8", errors="replace"))
                results[index] = _candidates(output, item.count)
            except NativeShapeError as exc:
                results[index] = exc
    return results
