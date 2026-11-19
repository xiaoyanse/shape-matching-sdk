"""错误码人话映射、训练预算查询与最近匹配诊断查询。

错误码与原生 vf_shape_error_code 一一对应；错误文本仍由原生层给出，
本模块的 error_message 提供绑定层人话文案（码+数值参数拼接）。
"""

from __future__ import annotations

import ctypes as ct

from .native_library import (
    AuditReport,
    BudgetReport,
    HealthReport,
    MatchTrace,
    NativeShapeError,
    TrainConfig,
    image_args as _image_args,
    invoke,
    load_library,
)

__all__ = [
    "AUDIT_CONFIDENCE",
    "AUDIT_FLAGS",
    "ERROR_CODES",
    "HEALTH_VERDICTS",
    "audit_discriminability",
    "error_message",
    "estimate_training",
    "last_error_info",
    "last_match_info",
    "model_health",
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
INVALID_POLYGON = 17
INSUFFICIENT_FEATURES = 18
COARSE_TRAIN_FAILED = 19
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
    "INVALID_POLYGON": INVALID_POLYGON,
    "INSUFFICIENT_FEATURES": INSUFFICIENT_FEATURES,
    "COARSE_TRAIN_FAILED": COARSE_TRAIN_FAILED,
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
    INVALID_POLYGON: "多边形训练参数非法：顶点 3–64 且在图域内、无自交；洞 ≤8、"
    "每洞面积 ≥64px² 且完全在外环内（参数：细分类型,主参数,次参数）",
    INSUFFICIENT_FEATURES: "参考 ROI 内边缘特征不足：训练未产生任何姿态模板，"
    "请放宽 ROI、提高对比度或降低阈值参数（参数：特征数下限）",
    COARSE_TRAIN_FAILED: "两级容器粗层训练在全部 K 档失败：ROI 过小或缩放档位"
    "无法满足粗层合同（参数：尝试的 K 档数）",
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


HEALTH_VERDICTS = {"HEALTHY": 0, "DEGRADED": 1, "WEAK": 2}
AUDIT_FLAGS = {"RECOMMENDED": 0, "INSEPARABLE": 1, "NO_NEGATIVE": 2}
AUDIT_CONFIDENCE = {"HIGH": 0, "LOW": 1}

_VERDICT_TEXT = {
    0: "健康：结构与自匹配均在正常范围",
    1: "降级：自匹配分 <90 或存在特征数 <8 的层，建议复核模板质量",
    2: "薄弱：自匹配分 <70 或特征数触 4 下限，不建议直接投入使用",
}
_FLAG_TEXT = {
    0: "可推荐：正负分数可分，推荐区间可用（建议性）",
    1: "不可分离：正负分数重叠（lo>hi），区间字段仅为计算中间值",
    2: "无负样本：仅报告正组分布，未推荐阈值",
}
_CONFIDENCE_TEXT = {0: "高（两组有效样本均 ≥5）", 1: "低（任一组有效样本 <5）"}


def _require_audit(library) -> None:
    if not getattr(library, "_audit_exports", False):
        raise NativeShapeError(
            "模板匹配原生组件缺少体检/审计导出（model_health/audit_discriminability，"
            "旧运行时），请升级制品后使用体检与审计接口"
        )


def model_health(model) -> dict:
    """模型健康体检（零输入）：结构量+容器模板图自匹配分与分级。

    verdict 0/1/2=HEALTHY/DEGRADED/WEAK（verdict_text 为人话）；单级旧资产
    无容器模板图，self_score_valid=False 且自匹配项不参与分级。
    """
    with model._lock:
        model._require_open()
        library = model._library
        _require_audit(library)
        report = HealthReport()
        report.struct_size = ct.sizeof(HealthReport)
        invoke(library, "model_health", model._handle, ct.byref(report))
        return {
            "verdict": report.verdict,
            "verdict_text": _VERDICT_TEXT.get(report.verdict, f"未知分级 {report.verdict}"),
            "pose_count": report.pose_count,
            "fine_min_features": report.fine_min_features,
            "fine_max_features": report.fine_max_features,
            "coarse_features": report.coarse_features,
            "detector_features": report.detector_features,
            "detector_weak": report.detector_weak,
            "detector_strong": report.detector_strong,
            "self_score": report.self_score,
            "self_score_valid": bool(report.self_score_valid),
            "template_width": report.template_width,
            "template_height": report.template_height,
        }


def _audit_group_args(images, label: str) -> tuple:
    """同组图像校验（同宽高）并构建指针数组：返回 (数组, 宽, 高, 步长)。"""
    width = height = 0
    pointers = (ct.c_void_p * len(images))()
    for index, image in enumerate(images):
        data_ptr, _, w, h, stride = _image_args(image)
        if index == 0:
            width, height = w, h
        elif (w, h) != (width, height):
            raise NativeShapeError(f"{label}图像须同宽高")
        pointers[index] = data_ptr
    return pointers, width, height, stride


def audit_discriminability(model, pos_images, neg_images=None) -> dict:
    """区分度审计与建议性阈值推荐（advisory）：正/负组逐帧最高分分布+推荐区间。

    pos_images 至少 1 帧；neg_images 可选（None/空时只报正组分布，
    flag=NO_NEGATIVE，不推荐区间）。同组图像须同宽高（连续 uint8 BGR）。
    flag 0/1/2=RECOMMENDED/INSEPARABLE/NO_NEGATIVE；recommend 区间为统计
    建议（lo=max(30, neg_max+5)、hi=min(85, pos_p25−5)），消费方按场景调整。
    """
    pos = list(pos_images)
    if not pos:
        raise NativeShapeError("正组图像至少 1 帧")
    neg = list(neg_images) if neg_images else []
    pos_args = _audit_group_args(pos, "正组")
    if neg:
        neg_args = _audit_group_args(neg, "负组")
    else:
        neg_args = ((ct.c_void_p * 0)(), 0, 0, 0)
    with model._lock:
        model._require_open()
        library = model._library
        _require_audit(library)
        report = AuditReport()
        report.struct_size = ct.sizeof(AuditReport)
        invoke(
            library,
            "audit_discriminability",
            model._handle,
            pos_args[0], pos_args[1], pos_args[2], pos_args[3], len(pos),
            neg_args[0], neg_args[1], neg_args[2], neg_args[3], len(neg),
            ct.byref(report),
        )
        return {
            "flag": report.flag,
            "flag_text": _FLAG_TEXT.get(report.flag, f"未知结论 {report.flag}"),
            "confidence": report.confidence,
            "confidence_text": _CONFIDENCE_TEXT.get(
                report.confidence, f"未知置信度 {report.confidence}"
            ),
            "pos_count": report.pos_count,
            "pos_zero_hits": report.pos_zero_hits,
            "pos_min": report.pos_min,
            "pos_p25": report.pos_p25,
            "pos_median": report.pos_median,
            "pos_p75": report.pos_p75,
            "pos_max": report.pos_max,
            "neg_count": report.neg_count,
            "neg_zero_hits": report.neg_zero_hits,
            "neg_min": report.neg_min,
            "neg_p25": report.neg_p25,
            "neg_median": report.neg_median,
            "neg_p75": report.neg_p75,
            "neg_max": report.neg_max,
            "separation": report.separation,
            "recommend_lo": report.recommend_lo,
            "recommend_hi": report.recommend_hi,
        }
