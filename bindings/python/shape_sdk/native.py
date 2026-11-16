"""受控句柄与真实 BGR 图像的模板训练、匹配接口。"""

from __future__ import annotations

import ctypes as ct
import math
import threading
import weakref
from dataclasses import dataclass

import numpy as np

from .native_library import (
    Candidate,
    NativeShapeError,
    TrainConfig,
    invoke,
    load_library,
)

__all__ = ["NativeCandidate", "NativeModel", "NativeShapeError", "train_model"]
MAX_MODEL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class NativeCandidate:
    score: float
    pose_id: int
    center: tuple[float, float]
    angle: float
    scale: float
    corners: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    transform: tuple[float, ...]
    pose_ambiguous: bool


def _image_args(image: np.ndarray) -> tuple:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.flags.c_contiguous
        or any(edge < 24 or edge > 4096 for edge in image.shape[:2])
    ):
        raise NativeShapeError("图像须为连续 uint8 BGR，宽高范围 24–4096")
    return (
        image.ctypes.data,
        image.nbytes,
        image.shape[1],
        image.shape[0],
        image.strides[0],
    )


def _close_unchecked(library: ct.CDLL, handle: int) -> None:
    error = ct.create_string_buffer(2048)
    library.vf_shape_close(handle, error, len(error))


def _candidates(output, count: int) -> list[NativeCandidate]:
    if count > len(output):
        raise NativeShapeError("原生候选输出数量超过缓冲区")
    candidates = []
    for item in output[:count]:
        values = (
            item.score,
            item.angle,
            item.scale,
            *item.center,
            *item.corners,
            *item.bbox,
            *item.transform,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0 <= item.score <= 100
            or item.pose_id < 0
            or item.pose_ambiguous not in (0, 1)
            or item.scale <= 0
            or item.bbox[2] < 0
            or item.bbox[3] < 0
        ):
            raise NativeShapeError("原生候选输出非法")
        candidates.append(
            NativeCandidate(
                score=item.score,
                pose_id=item.pose_id,
                center=tuple(item.center),
                angle=item.angle,
                scale=item.scale,
                corners=tuple((item.corners[n], item.corners[n + 1]) for n in range(0, 8, 2)),
                bbox=tuple(item.bbox),
                transform=tuple(item.transform),
                pose_ambiguous=bool(item.pose_ambiguous),
            )
        )
    return candidates


class NativeModel:
    def __init__(self, model_bytes: bytes):
        if not isinstance(model_bytes, bytes) or not 0 < len(model_bytes) <= MAX_MODEL_BYTES:
            raise NativeShapeError("模板模型字节非法或超过 32 MiB")
        self._lock = threading.RLock()
        self._library = load_library()
        self._handle = ct.c_uint64()
        buffer = ct.create_string_buffer(model_bytes)
        invoke(self._library, "load", buffer, len(model_bytes), ct.byref(self._handle))
        self._finalizer = weakref.finalize(
            self, _close_unchecked, self._library, self._handle.value
        )

    def _require_open(self) -> None:
        if not self._handle.value:
            raise NativeShapeError("模板模型已关闭")

    @property
    def metadata(self) -> dict:
        with self._lock:
            self._require_open()
            config = TrainConfig(struct_size=ct.sizeof(TrainConfig))
            width, height = ct.c_int32(), ct.c_int32()
            invoke(
                self._library,
                "metadata",
                self._handle,
                ct.byref(config),
                ct.byref(width),
                ct.byref(height),
            )
            return {
                "reference_width": width.value,
                "reference_height": height.value,
                "roi_xywh": (
                    config.roi_x,
                    config.roi_y,
                    config.roi_width,
                    config.roi_height,
                ),
                **{
                    key: getattr(config, key)
                    for key in (
                        "angle_min",
                        "angle_max",
                        "angle_step",
                        "scale_min",
                        "scale_max",
                        "scale_step",
                        # 两级容器加载件的 config 由原生端回填；旧资产回填落地前
                        # 此三键可能为零值，0 仍表示原生默认 128/30/60 语义
                        "weak_th",
                        "strong_th",
                        "feature_count",
                    )
                },
            }

    def match(
        self,
        frame_bgr: np.ndarray,
        *,
        threshold: float = 80.0,
        max_candidates: int = 16,
    ) -> list[NativeCandidate]:
        image_args = _image_args(frame_bgr)
        if not math.isfinite(threshold) or not 0 < threshold <= 100:
            raise NativeShapeError("相似度阈值须大于 0 且不大于 100")
        if not isinstance(max_candidates, int) or not 2 <= max_candidates <= 256:
            raise NativeShapeError("候选数量须为 2–256，不能将多目标截成唯一匹配")
        with self._lock:
            self._require_open()
            output = (Candidate * max_candidates)()
            count = ct.c_uint32()
            invoke(
                self._library,
                "match",
                self._handle,
                *image_args,
                threshold,
                output,
                max_candidates,
                ct.byref(count),
            )
            return _candidates(output, count.value)

    def close(self) -> None:
        with self._lock:
            if self._handle.value:
                invoke(self._library, "close", self._handle)
                self._handle.value = 0
                self._finalizer.detach()

    def __enter__(self) -> NativeModel:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def train_model(
    reference_bgr: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    *,
    angle_min: float,
    angle_max: float,
    angle_step: float,
    scale_min: float,
    scale_max: float,
    scale_step: float,
    weak_th: float = 0.0,
    strong_th: float = 0.0,
    feature_count: int = 0,
) -> bytes:
    """训练模板模型；weak_th/strong_th/feature_count 为特征筛选参数（0=默认 30/60/128）。

    梯度幅度迟滞阈与每层特征上限从建模期控制虚警（弱边缘杂波不进模型）。
    """
    image_args = _image_args(reference_bgr)
    if len(roi_xywh) != 4 or any(not isinstance(v, int) or abs(v) > 4096 for v in roi_xywh):
        raise NativeShapeError("训练 ROI 须为四项整数像素坐标")
    config = TrainConfig(
        ct.sizeof(TrainConfig),
        *roi_xywh,
        angle_min,
        angle_max,
        angle_step,
        scale_min,
        scale_max,
        scale_step,
        weak_th,
        strong_th,
        feature_count,
    )
    library, handle = load_library(), ct.c_uint64()
    invoke(library, "train", *image_args, ct.byref(config), ct.byref(handle))
    try:
        size = ct.c_uint64()
        invoke(library, "export", handle, None, 0, ct.byref(size))
        if not 0 < size.value <= MAX_MODEL_BYTES:
            raise NativeShapeError("原生模型输出超过预算")
        output = ct.create_string_buffer(size.value)
        invoke(library, "export", handle, output, size.value, ct.byref(size))
        return output.raw
    finally:
        _close_unchecked(library, handle.value)
