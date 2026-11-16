"""形状匹配 SDK 官方 Python 绑定（ctypes，ABI v3）。

运行时定位：优先 SHAPE_SDK_RUNTIME 环境变量（制品目录，含
visionforge_shape.dll 与 runtime-manifest.json），默认仓内 dist/runtime。
消费方集成时随制品分发本包并设置环境变量，或把制品放到默认路径。
"""

from .native import NativeCandidate, NativeModel, train_model
from .native_batch import match_batch
from .native_library import (
    BatchItem,
    Candidate,
    NativeShapeError,
    TrainConfig,
    invoke,
    load_library,
    runtime_directory,
)

__all__ = [
    "BatchItem",
    "Candidate",
    "NativeCandidate",
    "NativeModel",
    "NativeShapeError",
    "TrainConfig",
    "invoke",
    "load_library",
    "match_batch",
    "runtime_directory",
    "train_model",
]
