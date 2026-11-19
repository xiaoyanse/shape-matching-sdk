"""原生匹配器装载与固定宽度 ABI。"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
import sys
import threading
from pathlib import Path


class NativeShapeError(RuntimeError):
    pass


class TrainConfig(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("roi_x", ct.c_int32),
        ("roi_y", ct.c_int32),
        ("roi_width", ct.c_int32),
        ("roi_height", ct.c_int32),
        ("angle_min", ct.c_double),
        ("angle_max", ct.c_double),
        ("angle_step", ct.c_double),
        ("scale_min", ct.c_double),
        ("scale_max", ct.c_double),
        ("scale_step", ct.c_double),
        # ABI v2（特征筛选参数化）：0=原生默认（128/30/60）
        ("weak_th", ct.c_double),
        ("strong_th", ct.c_double),
        ("feature_count", ct.c_int32),
    ]


class Candidate(ct.Structure):
    _fields_ = [
        ("score", ct.c_double),
        ("pose_id", ct.c_int32),
        ("pose_ambiguous", ct.c_uint32),
        ("center", ct.c_double * 2),
        ("angle", ct.c_double),
        ("scale", ct.c_double),
        ("corners", ct.c_double * 8),
        ("bbox", ct.c_double * 4),
        ("transform", ct.c_double * 6),
    ]


class BatchItem(ct.Structure):
    _fields_ = [
        ("handle", ct.c_uint64),
        ("threshold", ct.c_double),
        ("candidates", ct.POINTER(Candidate)),
        ("capacity", ct.c_uint32),
        ("count", ct.c_uint32),
        ("status", ct.c_int32),
        ("error", ct.c_char * 256),
    ]


class BudgetReport(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("formula_version", ct.c_uint32),
        ("pose_used", ct.c_uint32),
        ("pose_limit", ct.c_uint32),
        ("pose_over", ct.c_uint32),
        ("canvas_used", ct.c_uint32),
        ("canvas_limit", ct.c_uint32),
        ("canvas_over", ct.c_uint32),
        ("pixel_over", ct.c_uint32),
        ("pixel_used", ct.c_uint64),
        ("pixel_limit", ct.c_uint64),
        ("over_budget", ct.c_int32),
    ]


class MatchTrace(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("coarse_candidates", ct.c_uint32),
        ("windows_searched", ct.c_uint32),
        ("final_candidates", ct.c_uint32),
        ("window_best_scores", ct.c_double * 8),
        ("coarse_ms", ct.c_double),
        ("coarse_resize_ms", ct.c_double),
        ("windows_ms", ct.c_double),
        ("finish_ms", ct.c_double),
    ]


class Vertex(ct.Structure):
    _fields_ = [
        ("x", ct.c_double),
        ("y", ct.c_double),
    ]


class Polygon(ct.Structure):
    _fields_ = [
        ("vertices", ct.POINTER(Vertex)),
        ("count", ct.c_uint32),
    ]


class PolygonTrainParams(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_uint32),
        ("outline", Polygon),
        ("holes", ct.POINTER(Polygon)),
        ("hole_count", ct.c_uint32),
        ("angle_min", ct.c_double),
        ("angle_max", ct.c_double),
        ("angle_step", ct.c_double),
        ("scale_min", ct.c_double),
        ("scale_max", ct.c_double),
        ("scale_step", ct.c_double),
        ("weak_th", ct.c_double),
        ("strong_th", ct.c_double),
        ("feature_count", ct.c_int32),
    ]


_library: ct.CDLL | None = None
_lock = threading.Lock()
_dll_directories: list[object] = []


def runtime_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "shape_matching"
    explicit = os.environ.get("SHAPE_SDK_RUNTIME")
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[3] / "dist" / "runtime"


def sdk_version() -> str:
    manifest = json.loads(
        (runtime_directory() / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    version = manifest.get("sdk_version")
    if not isinstance(version, str) or not version:
        raise NativeShapeError("模板匹配原生清单缺少 sdk_version，请重新构建装配运行包")
    return version


def load_library() -> ct.CDLL:
    global _library
    with _lock:
        if _library is not None:
            return _library
        directory = runtime_directory()
        manifest_path = directory / "runtime-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["abi_version"] != 3:
                raise NativeShapeError("模板匹配原生 ABI 版本不兼容")
            for filename, expected in manifest["files"].items():
                if Path(filename).name != filename:
                    raise NativeShapeError("模板匹配原生清单路径非法")
                actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
                if actual != expected:
                    raise NativeShapeError(f"模板匹配原生文件校验失败：{filename}")
            filename = (
                "visionforge_shape.dll" if sys.platform == "win32" else "libvisionforge_shape.so"
            )
            if filename not in manifest["files"]:
                raise NativeShapeError("模板匹配原生清单缺少主库")
            if sys.platform == "win32":
                _dll_directories.append(os.add_dll_directory(str(directory)))
            library = ct.CDLL(str(directory / filename))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise NativeShapeError(
                f"模板匹配原生组件不可用，请构建或修复安装：{directory}"
            ) from exc
        library.vf_shape_abi_version.argtypes = []
        library.vf_shape_abi_version.restype = ct.c_uint32
        if library.vf_shape_abi_version() != 3:
            raise NativeShapeError("模板匹配原生 ABI 不兼容")
        error = [ct.c_void_p, ct.c_uint32]
        image = [ct.c_void_p, ct.c_uint64, ct.c_int32, ct.c_int32, ct.c_uint64]
        signatures = {
            "train": image + [ct.POINTER(TrainConfig), ct.POINTER(ct.c_uint64)] + error,
            "load": [ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_uint64)] + error,
            "export": [ct.c_uint64, ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_uint64)] + error,
            "metadata": [
                ct.c_uint64,
                ct.POINTER(TrainConfig),
                ct.POINTER(ct.c_int32),
                ct.POINTER(ct.c_int32),
            ]
            + error,
            "match": [ct.c_uint64]
            + image
            + [ct.c_double, ct.POINTER(Candidate), ct.c_uint32, ct.POINTER(ct.c_uint32)]
            + error,
            "close": [ct.c_uint64] + error,
            "match_batch": image + [ct.POINTER(BatchItem), ct.c_uint32, ct.c_uint32] + error,
        }
        for name, signature in signatures.items():
            try:
                function = getattr(library, f"vf_shape_{name}")
            except AttributeError as exc:
                raise NativeShapeError(
                    f"模板匹配原生组件缺少 vf_shape_{name}，请重新构建或修复安装"
                ) from exc
            function.argtypes = signature
            function.restype = ct.c_int32
        # 诊断三件套为加性导出：旧运行时可能缺失，装载不得因此失败，
        # 缺失时诊断接口在使用处报错
        optional_signatures = {
            "estimate_training": [
                ct.POINTER(TrainConfig),
                ct.c_int32,
                ct.c_int32,
                ct.POINTER(BudgetReport),
            ]
            + error,
            "last_match_info": [ct.c_uint64, ct.POINTER(MatchTrace)] + error,
            "last_error_info": [
                ct.POINTER(ct.c_int32),
                ct.POINTER(ct.c_double * 6),
                ct.POINTER(ct.c_int32),
            ],
        }
        optional = {}
        for name, signature in optional_signatures.items():
            try:
                optional[name] = getattr(library, f"vf_shape_{name}")
            except AttributeError:
                optional[name] = None
        library._diagnostics_exports = all(value is not None for value in optional.values())
        for name, function in optional.items():
            if function is None:
                continue
            function.argtypes = optional_signatures[name]
            function.restype = ct.c_int32
        if optional["last_error_info"] is not None:
            optional["last_error_info"].restype = None
        # 多边形三导出同为加性：旧运行时缺失不阻断装载，调用处报错
        polygon_signatures = {
            "train_polygon":
                image + [ct.POINTER(PolygonTrainParams), ct.POINTER(ct.c_uint64)] + error,
            "suggest_contours": image + [
                ct.POINTER(ct.c_int32),
                ct.c_uint32,
                ct.c_uint32,
                ct.POINTER(Vertex),
                ct.POINTER(ct.c_uint32),
            ] + error,
            "polygon_metadata": [
                ct.c_uint64,
                ct.POINTER(Vertex),
                ct.c_uint32,
                ct.POINTER(ct.c_uint32),
                ct.POINTER(ct.c_uint32),
            ]
            + error,
        }
        polygon = {}
        for name, signature in polygon_signatures.items():
            try:
                polygon[name] = getattr(library, f"vf_shape_{name}")
            except AttributeError:
                polygon[name] = None
        library._polygon_exports = all(value is not None for value in polygon.values())
        for name, function in polygon.items():
            if function is None:
                continue
            function.argtypes = polygon_signatures[name]
            function.restype = ct.c_int32
        _library = library
        return library


def invoke(library: ct.CDLL, name: str, *args: object) -> None:
    error = ct.create_string_buffer(2048)
    if getattr(library, f"vf_shape_{name}")(*args, error, len(error)):
        raise NativeShapeError(error.value.decode("utf-8", errors="replace"))
