"""形状匹配 SDK 官方 Python 绑定（ctypes，ABI v3）。

运行时定位：优先 SHAPE_SDK_RUNTIME 环境变量（制品目录，含
visionforge_shape.dll 与 runtime-manifest.json），默认仓内 dist/runtime。
消费方集成时随制品分发本包并设置环境变量，或把制品放到默认路径。
"""

from .diagnostics import (
    AUDIT_CONFIDENCE,
    AUDIT_FLAGS,
    ERROR_CODES,
    HEALTH_VERDICTS,
    audit_discriminability,
    error_message,
    estimate_training,
    last_error_info,
    last_match_info,
    model_health,
)
from .native import (
    NativeCandidate,
    NativeModel,
    suggest_contours,
    train_model,
    train_polygon_model,
)
from .native_batch import match_batch
from .native_library import (
    AuditReport,
    BatchItem,
    BudgetReport,
    Candidate,
    HealthReport,
    MatchTrace,
    NativeShapeError,
    Polygon,
    PolygonTrainParams,
    TrainConfig,
    Vertex,
    invoke,
    load_library,
    runtime_directory,
)

__all__ = [
    "AuditReport",
    "AUDIT_CONFIDENCE",
    "AUDIT_FLAGS",
    "BatchItem",
    "BudgetReport",
    "Candidate",
    "ERROR_CODES",
    "HEALTH_VERDICTS",
    "HealthReport",
    "MatchTrace",
    "NativeCandidate",
    "NativeModel",
    "NativeShapeError",
    "Polygon",
    "PolygonTrainParams",
    "TrainConfig",
    "Vertex",
    "audit_discriminability",
    "error_message",
    "estimate_training",
    "invoke",
    "last_error_info",
    "last_match_info",
    "load_library",
    "match_batch",
    "model_health",
    "runtime_directory",
    "suggest_contours",
    "train_model",
    "train_polygon_model",
]
