"""受控句柄与真实 BGR 图像的模板训练、匹配接口。"""

from __future__ import annotations

import ctypes as ct
import math
import threading
import weakref
from dataclasses import dataclass

import numpy as np

from .diagnostics import last_match_info as _last_match_info
from .native_library import (
    Candidate,
    NativeShapeError,
    Polygon,
    PolygonTrainParams,
    TrainConfig,
    Vertex,
    image_args as _image_args,
    invoke,
    load_library,
)

__all__ = [
    "NativeCandidate",
    "NativeModel",
    "NativeShapeError",
    "suggest_contours",
    "train_model",
    "train_polygon_model",
]
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


def _close_unchecked(library: ct.CDLL, handle: int) -> None:
    error = ct.create_string_buffer(2048)
    library.vf_shape_close(handle, error, len(error))


def _export_handle(library: ct.CDLL, handle: int) -> bytes:
    """两段式导出并释放训练句柄（即用即释，导出失败也保证关闭）。"""
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


def _require_polygon(library: ct.CDLL) -> None:
    if not getattr(library, "_polygon_exports", False):
        raise NativeShapeError(
            "模板匹配原生组件缺少多边形导出（train_polygon/suggest_contours/"
            "polygon_metadata，旧运行时），请升级制品后使用多边形接口"
        )


def _vertex_array(points) -> ct.Array:
    """序列 (x, y) → Vertex 数组；轻量校验（几何终审在原生端）。"""
    items = list(points)
    if not 3 <= len(items) <= 64:
        raise NativeShapeError("轮廓顶点数须在 3–64")
    array = (Vertex * len(items))()
    for index, point in enumerate(items):
        x, y = point
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise NativeShapeError("顶点坐标须为数值")
        array[index].x = float(x)
        array[index].y = float(y)
    return array


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

    def last_match_info(self) -> dict:
        """最近一次匹配的诊断快照（字段见 diagnostics.last_match_info）。"""
        return _last_match_info(self)

    @property
    def polygon_metadata(self) -> dict:
        """多边形元数据回读：outline 顶点与洞数；矩形资产回退 ROI 四角、洞数 0。"""
        with self._lock:
            self._require_open()
            library = self._library
            _require_polygon(library)
            count, holes = ct.c_uint32(), ct.c_uint32()
            invoke(library, "polygon_metadata", self._handle, None, 0,
                   ct.byref(count), ct.byref(holes))
            outline = (Vertex * count.value)()
            invoke(library, "polygon_metadata", self._handle, outline, count.value,
                   ct.byref(count), ct.byref(holes))
            return {
                "outline": tuple((vertex.x, vertex.y) for vertex in outline),
                "hole_count": holes.value,
            }

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
    return _export_handle(library, handle)


def train_polygon_model(
    reference_bgr: np.ndarray,
    outline,
    holes=(),
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
    """多边形模板训练并导出资产字节。

    outline/holes 为 (x, y) 顶点序列（参考图坐标系）；洞 ≤8、每洞 3–64 顶点、
    面积 ≥64px² 且完全在外环内；外环 3–64 顶点、任意绕向（SDK 归一）。
    角度/缩放/阈值语义与 train_model 同名参数一致。
    """
    image_args = _image_args(reference_bgr)
    library = load_library()
    _require_polygon(library)
    outline_array = _vertex_array(outline)
    hole_arrays = [_vertex_array(hole) for hole in holes]
    if len(hole_arrays) > 8:
        raise NativeShapeError("洞数须 ≤8")
    hole_views = (Polygon * len(hole_arrays))()
    for index, array in enumerate(hole_arrays):
        hole_views[index].vertices = ct.cast(array, ct.POINTER(Vertex))
        hole_views[index].count = len(array)
    params = PolygonTrainParams(
        ct.sizeof(PolygonTrainParams),
        Polygon(ct.cast(outline_array, ct.POINTER(Vertex)), len(outline_array)),
        hole_views,
        len(hole_arrays),
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
    handle = ct.c_uint64()
    invoke(library, "train_polygon", *image_args, ct.byref(params), ct.byref(handle))
    return _export_handle(library, handle)


def suggest_contours(
    frame_bgr: np.ndarray,
    roi_hint: tuple[int, int, int, int] | None = None,
    *,
    max_suggestions: int = 8,
    max_vertices_per: int = 8,
) -> list[list[tuple[float, float]]]:
    """自动轮廓提取建议：返回建议列表（每条为顶点 (x, y) 列表，原图坐标系）。

    确定性（同图两次调用输出一致）；roi_hint 为 xywh 先验框（须完整落在图内），
    None=全图；面积过小或无法化简到 max_vertices_per 的轮廓不输出。
    返回 [] 表示无建议（非错误）。
    """
    image_args = _image_args(frame_bgr)
    library = load_library()
    _require_polygon(library)
    if not isinstance(max_suggestions, int) or not 1 <= max_suggestions <= 256:
        raise NativeShapeError("建议条数须为 1–256")
    if not isinstance(max_vertices_per, int) or not 3 <= max_vertices_per <= 64:
        raise NativeShapeError("每条建议顶点数须为 3–64")
    hint = None
    if roi_hint is not None:
        if len(roi_hint) != 4 or any(not isinstance(v, int) for v in roi_hint):
            raise NativeShapeError("先验框须为四项整数 xywh")
        hint = (ct.c_int32 * 4)(*roi_hint)
    vertices = (Vertex * (max_suggestions * max_vertices_per))()
    counts = (ct.c_uint32 * max_suggestions)()
    error = ct.create_string_buffer(2048)
    rc = library.vf_shape_suggest_contours(*image_args, hint, max_suggestions,
                                           max_vertices_per, vertices, counts, error,
                                           len(error))
    if rc < 0:
        raise NativeShapeError(error.value.decode("utf-8", errors="replace"))
    suggestions = []
    offset = 0
    for index in range(rc):
        count = counts[index]
        suggestions.append([(vertices[offset + j].x, vertices[offset + j].y)
                            for j in range(count)])
        offset += count
    return suggestions
