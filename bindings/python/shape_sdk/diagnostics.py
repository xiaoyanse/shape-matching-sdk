"""错误码人话映射、训练预算查询与最近匹配诊断查询。

错误码与原生 vf_shape_error_code 一一对应；错误文本仍由原生层给出，
本模块的 error_message 提供绑定层人话文案（码+数值参数拼接）。
"""

from __future__ import annotations

import ctypes as ct

from .native_library import (
    BudgetReport,
    MatchTrace,
    NativeShapeError,
    TrainConfig,
    invoke,
    load_library,
)

__all__ = [
    "ERROR_CODES",
    "error_message",
    "estimate_training",
    "last_error_info",
    "last_match_info",
]

OK = 0
INVALID_IMAGE = 1
INVALID_ROI = 2
INVALID_ANGLE_RANGE = 3
INVALID_SCALE_RANGE = 4
POSE_BUDGET = 5
PIXEL_BUDGET = 6
CANVAS_BUDGET = 7
DETECTOR_PARAMS = 8
THRESHOLD_INVALID = 9
CAPACITY_INVALID = 10
HANDLE_INVALID = 11
HANDLE_BUDGET = 12
MODEL_FORMAT = 13
MODEL_SIZE = 14
MODEL_TAMPER = 15
SEARCH_BUDGET = 16
INTERNAL = 99

ERROR_CODES = {
    "OK": OK,
    "INVALID_IMAGE": INVALID_IMAGE,
    "INVALID_ROI": INVALID_ROI,
    "INVALID_ANGLE_RANGE": INVALID_ANGLE_RANGE,
    "INVALID_SCALE_RANGE": INVALID_SCALE_RANGE,
    "POSE_BUDGET": POSE_BUDGET,
    "PIXEL_BUDGET": PIXEL_BUDGET,
    "CANVAS_BUDGET": CANVAS_BUDGET,
    "DETECTOR_PARAMS": DETECTOR_PARAMS,
    "THRESHOLD_INVALID": THRESHOLD_INVALID,
    "CAPACITY_INVALID": CAPACITY_INVALID,
    "HANDLE_INVALID": HANDLE_INVALID,
    "HANDLE_BUDGET": HANDLE_BUDGET,
    "MODEL_FORMAT": MODEL_FORMAT,
    "MODEL_SIZE": MODEL_SIZE,
    "MODEL_TAMPER": MODEL_TAMPER,
    "SEARCH_BUDGET": SEARCH_BUDGET,
    "INTERNAL": INTERNAL,
}

_CODE_MESSAGES = {
    OK: "成功",
    INVALID_IMAGE: "图像不合法：须为连续 uint8 BGR，宽高均在 24–4096 像素",
    INVALID_ROI: "训练 ROI 不合法：≥24px 且须完整落在参考图内",
    INVALID_ANGLE_RANGE: "角度范围或步长不合法：范围 [-180,180]、步长>0；"
    "步长过细时参数依次为当前步长、档数上限（350）、建议步长",
    INVALID_SCALE_RANGE: "缩放范围或步长不合法：范围 [0.25,2]、步长>0；"
    "步长过细时参数依次为当前步长、档数上限（350）、建议步长",
    POSE_BUDGET: "姿态总数超预算：角度×缩放档位数超上限（参数：档位数,上限）",
    PIXEL_BUDGET: "训练像素总量超预算（参数：用量,上限）",
    CANVAS_BUDGET: "训练画布超尺寸预算：缩放后模板外扩画布单边须 ≤4096（参数：画布边,上限）",
    DETECTOR_PARAMS: "特征筛选参数不合法：0<weak≤strong≤1020，特征数 16–512",
    THRESHOLD_INVALID: "相似度阈值须为 (0,100] 内的有限值",
    CAPACITY_INVALID: "候选容量须为 2–256",
    HANDLE_INVALID: "模型句柄已关闭或未知",
    HANDLE_BUDGET: "活跃句柄数超上限 128，请先释放不再使用的模型（参数：当前数,上限）",
    MODEL_FORMAT: "模型资产格式无法识别",
    MODEL_SIZE: "模型资产尺寸超预算 ≤32MiB（参数：字节数,上限）",
    MODEL_TAMPER: "模型资产校验失败：内容与容器合同不符或被篡改",
    SEARCH_BUDGET: "匹配搜索预算超限：姿态数×搜索块数过多（参数：用量,上限）",
    INTERNAL: "未分类内部错误",
}


def error_message(code: int, params: tuple[float, ...] = ()) -> str:
    """错误码人话文案；params 为 last_error_info 返回的数值上下文。"""
    text = _CODE_MESSAGES.get(code, f"未知错误码 {code}")
    if params:
        shown = ", ".join(format(value, ".6g") for value in params)
        return f"{text}（参数：{shown}）"
    return text


def _require_diagnostics(library) -> None:
    if not getattr(library, "_diagnostics_exports", False):
        raise NativeShapeError(
            "模板匹配原生组件缺少诊断导出（estimate_training/last_match_info/"
            "last_error_info，旧运行时），请升级制品后使用诊断接口"
        )


def last_error_info() -> tuple[int, tuple[float, ...]]:
    """本线程最近一次失败调用的 (错误码, 数值参数)；成功调用不清除上次记录。"""
    library = load_library()
    _require_diagnostics(library)
    code, count = ct.c_int32(), ct.c_int32()
    params = (ct.c_double * 6)()
    library.vf_shape_last_error_info(ct.byref(code), params, ct.byref(count))
    usable = max(0, min(count.value, 6))
    return code.value, tuple(params[index] for index in range(usable))


def estimate_training(
    image_size: tuple[int, int],
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
) -> dict:
    """训练预算查询（纯计算）：返回姿态/画布/像素三账本与首要超限因子。"""
    if len(image_size) != 2 or any(not isinstance(v, int) or not 24 <= v <= 4096 for v in image_size):
        raise NativeShapeError("图像尺寸须为两整数且在 24–4096")
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
    report = BudgetReport()
    report.struct_size = ct.sizeof(BudgetReport)
    library = load_library()
    _require_diagnostics(library)
    invoke(library, "estimate_training", ct.byref(config), image_size[0],
           image_size[1], ct.byref(report))
    return {
        "formula_version": report.formula_version,
        "pose_used": report.pose_used,
        "pose_limit": report.pose_limit,
        "pose_over": bool(report.pose_over),
        "canvas_used": report.canvas_used,
        "canvas_limit": report.canvas_limit,
        "canvas_over": bool(report.canvas_over),
        "pixel_used": report.pixel_used,
        "pixel_limit": report.pixel_limit,
        "pixel_over": bool(report.pixel_over),
        "over_budget": report.over_budget,
    }


def last_match_info(model) -> dict:
    """句柄最近一次 match/match_batch 的诊断快照（单级模型仅 final_candidates 有效）。"""
    with model._lock:
        model._require_open()
        _require_diagnostics(model._library)
        trace = MatchTrace()
        trace.struct_size = ct.sizeof(MatchTrace)
        invoke(model._library, "last_match_info", model._handle, ct.byref(trace))
        return {
            "coarse_candidates": trace.coarse_candidates,
            "windows_searched": trace.windows_searched,
            "final_candidates": trace.final_candidates,
            "window_best_scores": tuple(trace.window_best_scores),
            "coarse_ms": trace.coarse_ms,
            "coarse_resize_ms": trace.coarse_resize_ms,
            "windows_ms": trace.windows_ms,
            "finish_ms": trace.finish_ms,
        }
