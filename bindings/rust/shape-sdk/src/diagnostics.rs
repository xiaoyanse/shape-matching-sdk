//! 诊断三件套安全层：训练预算查询、最近错误信息、最近匹配诊断快照。
//! ABI 结构、函数指针类型与 unsafe 调用边界在本模块闭环（布局合同见
//! include/shape_matching.h）；错误码常量与人话映射在 `error::error_code`。

use crate::error::ShapeError;
use crate::ffi::{NativeAbi, TrainConfigAbi};
use crate::library::{train_config_abi, ShapeLibrary};
use crate::model::{Model, TrainParams};

/// 训练预算账本（vf_shape_budget_report，v2 布局：尾部 polygon_fill_ratio）
#[repr(C)]
pub(crate) struct BudgetReportAbi {
    pub struct_size: u32,
    pub formula_version: u32,
    pub pose_used: u32,
    pub pose_limit: u32,
    pub pose_over: u32,
    pub canvas_used: u32,
    pub canvas_limit: u32,
    pub canvas_over: u32,
    pub pixel_over: u32,
    pub pixel_used: u64,
    pub pixel_limit: u64,
    pub over_budget: i32,
    pub polygon_fill_ratio: f64,
}

impl Default for BudgetReportAbi {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            formula_version: 0,
            pose_used: 0,
            pose_limit: 0,
            pose_over: 0,
            canvas_used: 0,
            canvas_limit: 0,
            canvas_over: 0,
            pixel_over: 0,
            pixel_used: 0,
            pixel_limit: 0,
            over_budget: 0,
            polygon_fill_ratio: 0.0,
        }
    }
}

/// 最近匹配诊断快照（vf_shape_match_trace）
#[repr(C)]
pub(crate) struct MatchTraceAbi {
    pub struct_size: u32,
    pub coarse_candidates: u32,
    pub windows_searched: u32,
    pub final_candidates: u32,
    pub window_best_scores: [f64; 8],
    pub coarse_ms: f64,
    pub coarse_resize_ms: f64,
    pub windows_ms: f64,
    pub finish_ms: f64,
}

impl Default for MatchTraceAbi {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            coarse_candidates: 0,
            windows_searched: 0,
            final_candidates: 0,
            window_best_scores: [0.0; 8],
            coarse_ms: 0.0,
            coarse_resize_ms: 0.0,
            windows_ms: 0.0,
            finish_ms: 0.0,
        }
    }
}

pub(crate) type EstimateTrainingFn = unsafe extern "C" fn(
    *const TrainConfigAbi,
    i32,
    i32,
    *mut BudgetReportAbi,
    *mut u8,
    u32,
) -> i32;
pub(crate) type EstimateTrainingExFn = unsafe extern "C" fn(
    *const TrainConfigAbi,
    i32,
    i32,
    *const crate::polygon::PolygonParamsAbi,
    *mut BudgetReportAbi,
    *mut u8,
    u32,
) -> i32;
pub(crate) type LastMatchInfoFn = unsafe extern "C" fn(u64, *mut MatchTraceAbi, *mut u8, u32) -> i32;
/// 无失败路径：void 返回，三出口指针均可为空
pub(crate) type LastErrorInfoFn = unsafe extern "C" fn(*mut i32, *mut f64, *mut i32);

const ERROR_CAPACITY: usize = 2048;

fn missing_export(name: &str) -> ShapeError {
    ShapeError::Library(format!(
        "模板匹配原生组件缺少诊断导出 {name}（旧运行时），请升级制品后使用诊断接口"
    ))
}

impl NativeAbi {
    /// 预算查询：纯计算（配置域不合法由原生端拒绝并带码落账）
    pub(crate) fn estimate_training(
        &self,
        config: &TrainConfigAbi,
        image_w: i32,
        image_h: i32,
        report: &mut BudgetReportAbi,
    ) -> Result<(), ShapeError> {
        let estimate = self.estimate_fn.ok_or_else(|| missing_export("vf_shape_estimate_training"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: config/report 指向调用方栈上对象；error 缓冲容量一致传递。
        let status = unsafe {
            (estimate)(config, image_w, image_h, report, error.as_mut_ptr(),
                       ERROR_CAPACITY as u32)
        };
        ShapeError::check(status, &error)
    }

    /// 预算查询 _ex：polygon 为空指针时与矩形路径完全一致，非空走多边形核算
    pub(crate) fn estimate_training_ex(
        &self,
        config: &TrainConfigAbi,
        image_w: i32,
        image_h: i32,
        polygon: *const crate::polygon::PolygonParamsAbi,
        report: &mut BudgetReportAbi,
    ) -> Result<(), ShapeError> {
        let estimate_ex = self
            .estimate_ex_fn
            .ok_or_else(|| missing_export("vf_shape_estimate_training_ex"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: config/polygon/report 均在调用期间借用；report struct_size 预设。
        let status = unsafe {
            (estimate_ex)(config, image_w, image_h, polygon, report, error.as_mut_ptr(),
                          ERROR_CAPACITY as u32)
        };
        ShapeError::check(status, &error)
    }

    /// 最近匹配诊断快照（句柄锁下读取，非阻塞）
    pub(crate) fn last_match_info(
        &self,
        handle: u64,
        trace: &mut MatchTraceAbi,
    ) -> Result<(), ShapeError> {
        let last_match =
            self.last_match_fn.ok_or_else(|| missing_export("vf_shape_last_match_info"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: trace 指向调用方栈上对象，struct_size 已按合同预设。
        let status =
            unsafe { (last_match)(handle, trace, error.as_mut_ptr(), ERROR_CAPACITY as u32) };
        ShapeError::check(status, &error)
    }

    /// 本线程最近一次失败调用的错误码与数值参数（无失败路径；旧运行时缺导出报错）
    pub(crate) fn last_error_info(&self) -> Result<(i32, [f64; 6], i32), ShapeError> {
        let last_error =
            self.last_error_fn.ok_or_else(|| missing_export("vf_shape_last_error_info"))?;
        let mut code = 0i32;
        let mut params = [0f64; 6];
        let mut count = 0i32;
        // SAFETY: 三个出口指针均指向本栈帧对象；原生端按合同逐项回填。
        unsafe { (last_error)(&mut code, params.as_mut_ptr(), &mut count) };
        Ok((code, params, count))
    }
}

/// 本线程最近一次失败调用的错误码与数值参数（前 `param_count` 项有效）
#[derive(Debug, Clone, PartialEq)]
pub struct LastErrorInfo {
    pub code: i32,
    pub param_count: usize,
    pub params: [f64; 6],
}

/// 训练预算账本：三账本用量/上限/超限标志 + 首要超限因子（0=未超限，
/// 否则 `error_code` 的 POSE_BUDGET/CANVAS_BUDGET/PIXEL_BUDGET）。
/// polygon_fill_ratio 仅 formula_version=2 时构成消费承诺（掩码面积/外接矩形
/// 面积比）；矩形路径恒 1.0。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetReport {
    pub formula_version: u32,
    pub pose_used: u32,
    pub pose_limit: u32,
    pub pose_over: bool,
    pub canvas_used: u32,
    pub canvas_limit: u32,
    pub canvas_over: bool,
    pub pixel_used: u64,
    pub pixel_limit: u64,
    pub pixel_over: bool,
    pub over_budget: i32,
    pub polygon_fill_ratio: f64,
}

/// 预算查询输入形态（vf_shape_estimate_training_ex 的 config/params 二选一语义）
#[derive(Debug, Clone, Copy)]
pub enum TrainingShape<'a> {
    /// 矩形 ROI 训练参数（formula_version=1，与 `estimate_training` 一致）
    Rect(&'a TrainParams),
    /// 多边形掩码训练参数（与 `train_polygon_model` 同参语义，formula_version=2）
    Polygon(&'a crate::polygon::PolygonTrainParams),
}

/// 最近一次匹配诊断快照：计数常开；毫秒仅诊断计时构建非 0（合同）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchTrace {
    pub coarse_candidates: u32,
    pub windows_searched: u32,
    pub final_candidates: u32,
    pub window_best_scores: [f64; 8],
    pub coarse_ms: f64,
    pub coarse_resize_ms: f64,
    pub windows_ms: f64,
    pub finish_ms: f64,
}

fn budget_from_abi(report: &BudgetReportAbi) -> BudgetReport {
    BudgetReport {
        formula_version: report.formula_version,
        pose_used: report.pose_used,
        pose_limit: report.pose_limit,
        pose_over: report.pose_over != 0,
        canvas_used: report.canvas_used,
        canvas_limit: report.canvas_limit,
        canvas_over: report.canvas_over != 0,
        pixel_used: report.pixel_used,
        pixel_limit: report.pixel_limit,
        pixel_over: report.pixel_over != 0,
        over_budget: report.over_budget,
        polygon_fill_ratio: report.polygon_fill_ratio,
    }
}

impl ShapeLibrary {
    /// 本线程最近一次失败调用的错误码与数值参数（成功调用不清除上次记录）
    pub fn last_error_info(&self) -> Result<LastErrorInfo, ShapeError> {
        let (code, params, count) = self.abi().last_error_info()?;
        Ok(LastErrorInfo {
            code,
            param_count: count.clamp(0, 6) as usize,
            params,
        })
    }

    /// 训练预算查询：按训练同一公式核算姿态/画布/像素三账本（纯计算，不建句柄）
    pub fn estimate_training(
        &self,
        params: &TrainParams,
        image_w: i32,
        image_h: i32,
    ) -> Result<BudgetReport, ShapeError> {
        params.validate()?;
        let (feature_count, weak_th, strong_th) = match &params.detector {
            None => (0, 0.0, 0.0),
            Some(detector) => (detector.feature_count, detector.weak_th, detector.strong_th),
        };
        let config = train_config_abi(params, feature_count, weak_th, strong_th);
        let mut report = BudgetReportAbi::default();
        self.abi()
            .estimate_training(&config, image_w, image_h, &mut report)?;
        Ok(budget_from_abi(&report))
    }

    /// 多边形训练预算查询（_ex）：矩形路径与 `estimate_training` 逐字段一致；
    /// 多边形路径与 `train_polygon_model` 同参语义，formula_version=2。
    /// 注意 polygon 路径 ABI 层仍要求 config 指针合法——本方法按合同传仅含
    /// struct_size 的壳配置（原生端多边形路径不消费其字段）。
    pub fn estimate_training_ex(
        &self,
        image_w: i32,
        image_h: i32,
        shape: TrainingShape<'_>,
    ) -> Result<BudgetReport, ShapeError> {
        // 多边形路径的壳配置：仅承担 ABI 结构校验，字段不参与核算
        let mut config = TrainConfigAbi {
            struct_size: std::mem::size_of::<TrainConfigAbi>() as u32,
            ..TrainConfigAbi::default()
        };
        let polygon_storage: (
            crate::polygon::PolygonParamsAbi,
            Vec<crate::polygon::VertexAbi>,
            Vec<Vec<crate::polygon::VertexAbi>>,
            Vec<crate::polygon::PolygonAbi>,
        );
        let polygon_ptr = match shape {
            TrainingShape::Rect(params) => {
                params.validate()?;
                let (feature_count, weak_th, strong_th) = match &params.detector {
                    None => (0, 0.0, 0.0),
                    Some(detector) => {
                        (detector.feature_count, detector.weak_th, detector.strong_th)
                    }
                };
                config = train_config_abi(params, feature_count, weak_th, strong_th);
                std::ptr::null()
            }
            TrainingShape::Polygon(params) => {
                crate::polygon::validate_polygon_params(params)?;
                polygon_storage = crate::polygon::build_params_abi(params);
                &polygon_storage.0
            }
        };
        let mut report = BudgetReportAbi::default();
        self.abi()
            .estimate_training_ex(&config, image_w, image_h, polygon_ptr, &mut report)?;
        Ok(budget_from_abi(&report))
    }
}

impl Model {
    /// 最近一次 match/match_batch 的诊断快照。单级模型仅 final_candidates
    /// 有效；句柄尚未匹配时全 0
    pub fn last_match_info(&self) -> Result<MatchTrace, ShapeError> {
        self.with_live_handle(|handle| {
            let mut trace = MatchTraceAbi::default();
            self.library().abi().last_match_info(handle, &mut trace)?;
            Ok(MatchTrace {
                coarse_candidates: trace.coarse_candidates,
                windows_searched: trace.windows_searched,
                final_candidates: trace.final_candidates,
                window_best_scores: trace.window_best_scores,
                coarse_ms: trace.coarse_ms,
                coarse_resize_ms: trace.coarse_resize_ms,
                windows_ms: trace.windows_ms,
                finish_ms: trace.finish_ms,
            })
        })?
    }
}

// 编译期布局核对（64 位平台口径，含尾部对齐填充；v2 尾部追加 fill ratio 后为 72）
const _: () = assert!(std::mem::size_of::<BudgetReportAbi>() == 72);
const _: () = assert!(std::mem::size_of::<MatchTraceAbi>() == 112);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::error_code;
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("crate 应位于仓内 bindings/rust/shape-sdk")
            .to_path_buf()
    }

    fn task_runtime() -> PathBuf {
        repo_root().join("build").join("t2a").join("runtime")
    }

    fn open_task_library() -> ShapeLibrary {
        ShapeLibrary::open_at(&task_runtime()).expect("诊断测试须先装配 build/t2a/runtime")
    }

    fn params(roi: (i32, i32, i32, i32), amin: f64, amax: f64, astep: f64,
              smin: f64, smax: f64, sstep: f64) -> TrainParams {
        TrainParams {
            roi_xywh: roi,
            angle_min: amin,
            angle_max: amax,
            angle_step: astep,
            scale_min: smin,
            scale_max: smax,
            scale_step: sstep,
            detector: None,
        }
    }

    #[test]
    fn estimate_pose_boundary() {
        let library = open_task_library();
        let ok = library
            .estimate_training(&params((32, 32, 176, 176), -87.0, 87.0, 1.0, 0.5, 1.0, 0.5),
                               320, 320)
            .expect("压线配置应产报告");
        assert_eq!(ok.formula_version, 1);
        assert_eq!(ok.pose_used, 350);
        assert!(!ok.pose_over && ok.over_budget == error_code::OK);
        let over = library
            .estimate_training(&params((32, 32, 176, 176), -87.5, 87.5, 1.0, 0.5, 1.0, 0.5),
                               320, 320)
            .expect("超限配置仍应产报告");
        assert!(over.pose_over);
        assert_eq!(over.pose_used, 352);
        assert_eq!(over.over_budget, error_code::POSE_BUDGET);
    }

    #[test]
    fn estimate_rejects_invalid_domain() {
        let library = open_task_library();
        let error = library
            .estimate_training(&params((32, 32, 176, 176), 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                               320, 320)
            .expect_err("step=0 应拒绝");
        assert_eq!(error.native_status(), Some(1));
        let info = library.last_error_info().expect("last_error_info 查询失败");
        assert_eq!(info.code, error_code::INVALID_ANGLE_RANGE);
    }

    #[test]
    fn error_message_covers_codes() {
        for code in [error_code::INVALID_ANGLE_RANGE, error_code::POSE_BUDGET] {
            let text = crate::error::error_message(code, &[0.05, 350.0]);
            assert!(text.contains("参数"), "码 {code} 文案应含参数：{text}");
        }
        assert_eq!(error_code::name(99), "INTERNAL");
        assert_eq!(error_code::name(-1), "UNKNOWN");
    }
}
