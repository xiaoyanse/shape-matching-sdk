//! 模型健康体检与区分度审计安全层：零输入体检（分级+容器模板图自匹配）与
//! 建议性阈值推荐（advisory）。ABI 结构、函数指针类型与 unsafe 调用边界在本
//! 模块闭环（布局合同见 include/shape_matching.h）；两个导出同为加性（旧运行
//! 时缺失容忍，调用处报错），语义对齐 Python 侧 diagnostics.model_health /
//! audit_discriminability。

use crate::error::ShapeError;
use crate::ffi::NativeAbi;
use crate::model::{BgrImage, Model};

/// 体检分级（vf_shape_health_verdict）：取更重——任一层特征数触下限（≤4）
/// 或实测自匹配 <70 为 WEAK；未达 WEAK 且自匹配 <90 或任一层 <8 为 DEGRADED。
/// 单级资产（self_score_valid=false）自匹配项不参与分级。
pub mod health_verdict {
    pub const HEALTHY: i32 = 0;
    pub const DEGRADED: i32 = 1;
    pub const WEAK: i32 = 2;

    /// 分级名（未知值返回 UNKNOWN）
    pub fn name(verdict: i32) -> &'static str {
        match verdict {
            HEALTHY => "HEALTHY",
            DEGRADED => "DEGRADED",
            WEAK => "WEAK",
            _ => "UNKNOWN",
        }
    }
}

/// 审计结论（vf_shape_audit_flag）：INSEPARABLE 时区间字段仅为计算中间值
/// （lo>hi），不得消费；NO_NEGATIVE 表示负组未提供、只报正组分布。
pub mod audit_flag {
    pub const RECOMMENDED: i32 = 0;
    pub const INSEPARABLE: i32 = 1;
    pub const NO_NEGATIVE: i32 = 2;

    /// 结论名（未知值返回 UNKNOWN）
    pub fn name(flag: i32) -> &'static str {
        match flag {
            RECOMMENDED => "RECOMMENDED",
            INSEPARABLE => "INSEPARABLE",
            NO_NEGATIVE => "NO_NEGATIVE",
            _ => "UNKNOWN",
        }
    }
}

/// 审计置信度（vf_shape_audit_confidence）：任一组有效分数样本 <5 时为 LOW。
pub mod audit_confidence {
    pub const HIGH: i32 = 0;
    pub const LOW: i32 = 1;

    /// 置信度名（未知值返回 UNKNOWN）
    pub fn name(confidence: i32) -> &'static str {
        match confidence {
            HIGH => "HIGH",
            LOW => "LOW",
            _ => "UNKNOWN",
        }
    }
}

/// vf_shape_health_report
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct HealthReportAbi {
    pub struct_size: u32,
    pub verdict: i32,
    pub pose_count: u32,
    pub fine_min_features: u32,
    pub fine_max_features: u32,
    pub coarse_features: u32,
    pub detector_features: i32,
    pub detector_weak: f64,
    pub detector_strong: f64,
    pub self_score: f64,
    pub self_score_valid: u32,
    pub template_width: u32,
    pub template_height: u32,
}

impl Default for HealthReportAbi {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            verdict: 0,
            pose_count: 0,
            fine_min_features: 0,
            fine_max_features: 0,
            coarse_features: 0,
            detector_features: 0,
            detector_weak: 0.0,
            detector_strong: 0.0,
            self_score: 0.0,
            self_score_valid: 0,
            template_width: 0,
            template_height: 0,
        }
    }
}

/// vf_shape_audit_report
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct AuditReportAbi {
    pub struct_size: u32,
    pub flag: i32,
    pub confidence: i32,
    pub pos_count: u32,
    pub pos_zero_hits: u32,
    pub pos_min: f64,
    pub pos_p25: f64,
    pub pos_median: f64,
    pub pos_p75: f64,
    pub pos_max: f64,
    pub neg_count: u32,
    pub neg_zero_hits: u32,
    pub neg_min: f64,
    pub neg_p25: f64,
    pub neg_median: f64,
    pub neg_p75: f64,
    pub neg_max: f64,
    pub separation: f64,
    pub recommend_lo: f64,
    pub recommend_hi: f64,
}

impl Default for AuditReportAbi {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            flag: 0,
            confidence: 0,
            pos_count: 0,
            pos_zero_hits: 0,
            pos_min: 0.0,
            pos_p25: 0.0,
            pos_median: 0.0,
            pos_p75: 0.0,
            pos_max: 0.0,
            neg_count: 0,
            neg_zero_hits: 0,
            neg_min: 0.0,
            neg_p25: 0.0,
            neg_median: 0.0,
            neg_p75: 0.0,
            neg_max: 0.0,
            separation: 0.0,
            recommend_lo: 0.0,
            recommend_hi: 0.0,
        }
    }
}

// 编译期布局核对（与 C 头 sizeof 约定一致，64 位平台口径）
const _: () = assert!(std::mem::size_of::<HealthReportAbi>() == 72);
const _: () = assert!(std::mem::size_of::<AuditReportAbi>() == 136);

pub(crate) type ModelHealthFn =
    unsafe extern "C" fn(u64, *mut HealthReportAbi, *mut u8, u32) -> i32;
pub(crate) type AuditDiscriminabilityFn = unsafe extern "C" fn(
    u64,
    *const *const u8,
    i32,
    i32,
    u64,
    u32,
    *const *const u8,
    i32,
    i32,
    u64,
    u32,
    *mut AuditReportAbi,
    *mut u8,
    u32,
) -> i32;

const ERROR_CAPACITY: usize = 2048;

fn missing_audit_export(name: &str) -> ShapeError {
    ShapeError::Library(format!(
        "模板匹配原生组件缺少体检/审计导出 {name}（旧运行时），请升级制品后使用体检与审计接口"
    ))
}

impl NativeAbi {
    /// 零输入体检（分级+自匹配）
    pub(crate) fn model_health(
        &self,
        handle: u64,
        report: &mut HealthReportAbi,
    ) -> Result<(), ShapeError> {
        let health =
            self.model_health_fn.ok_or_else(|| missing_audit_export("vf_shape_model_health"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: report 指向调用方栈上对象，struct_size 已按合同预设。
        let status = unsafe { (health)(handle, report, error.as_mut_ptr(), ERROR_CAPACITY as u32) };
        ShapeError::check(status, &error)
    }

    /// 区分度审计（图像指针数组借用于调用期间；容量一致传递）
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn audit_discriminability(
        &self,
        handle: u64,
        pos: &[*const u8],
        pos_width: i32,
        pos_height: i32,
        neg: &[*const u8],
        neg_width: i32,
        neg_height: i32,
        report: &mut AuditReportAbi,
    ) -> Result<(), ShapeError> {
        let audit = self
            .audit_fn
            .ok_or_else(|| missing_audit_export("vf_shape_audit_discriminability"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        let pos_ptr = pos.as_ptr();
        let neg_ptr = if neg.is_empty() { std::ptr::null() } else { neg.as_ptr() };
        // SAFETY: pos/neg 指针数组与指向的图像缓冲仅在调用期间借用（库不保留
        // 调用方指针）；report 为出口缓冲。
        let status = unsafe {
            (audit)(
                handle,
                pos_ptr,
                pos_width,
                pos_height,
                3 * pos_width as u64,
                pos.len() as u32,
                neg_ptr,
                neg_width,
                neg_height,
                3 * neg_width as u64,
                neg.len() as u32,
                report,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)
    }
}

/// 零输入体检报告：结构量+容器模板图自匹配分与分级（字段语义见 C 头合同）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HealthReport {
    pub verdict: i32,
    pub pose_count: u32,
    pub fine_min_features: u32,
    pub fine_max_features: u32,
    pub coarse_features: u32,
    pub detector_features: i32,
    pub detector_weak: f64,
    pub detector_strong: f64,
    pub self_score: f64,
    pub self_score_valid: bool,
    pub template_width: u32,
    pub template_height: u32,
}

/// 区分度审计报告：两组逐帧最高分分布+建议性阈值区间（advisory，非合同承诺；
/// separation=pos_p25−neg_max，lo=max(30, neg_max+5)、hi=min(85, pos_p25−5)）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditReport {
    pub flag: i32,
    pub confidence: i32,
    pub pos_count: u32,
    pub pos_zero_hits: u32,
    pub pos_min: f64,
    pub pos_p25: f64,
    pub pos_median: f64,
    pub pos_p75: f64,
    pub pos_max: f64,
    pub neg_count: u32,
    pub neg_zero_hits: u32,
    pub neg_min: f64,
    pub neg_p25: f64,
    pub neg_median: f64,
    pub neg_p75: f64,
    pub neg_max: f64,
    pub separation: f64,
    pub recommend_lo: f64,
    pub recommend_hi: f64,
}

/// 同组图像校验（同宽高）并构建指针数组
fn group_ptrs(images: &[BgrImage<'_>], label: &str) -> Result<(Vec<*const u8>, i32, i32), ShapeError> {
    let mut pointers = Vec::with_capacity(images.len());
    let mut width = 0;
    let mut height = 0;
    for (index, image) in images.iter().enumerate() {
        if index == 0 {
            width = image.width();
            height = image.height();
        } else if (image.width(), image.height()) != (width, height) {
            return Err(ShapeError::InvalidInput(format!("{label}图像须同宽高")));
        }
        pointers.push(image.data().as_ptr());
    }
    Ok((pointers, width, height))
}

impl Model {
    /// 模型健康体检（零输入）：分级与结构量；单级旧资产 self_score_valid=false
    pub fn health(&self) -> Result<HealthReport, ShapeError> {
        self.with_live_handle(|handle| {
            let mut report = HealthReportAbi::default();
            self.library().abi().model_health(handle, &mut report)?;
            Ok(HealthReport {
                verdict: report.verdict,
                pose_count: report.pose_count,
                fine_min_features: report.fine_min_features,
                fine_max_features: report.fine_max_features,
                coarse_features: report.coarse_features,
                detector_features: report.detector_features,
                detector_weak: report.detector_weak,
                detector_strong: report.detector_strong,
                self_score: report.self_score,
                self_score_valid: report.self_score_valid != 0,
                template_width: report.template_width,
                template_height: report.template_height,
            })
        })?
    }

    /// 区分度审计与建议性阈值推荐：pos 至少 1 帧，neg 可选（空=只报正组分布，
    /// flag=NO_NEGATIVE）；同组图像须同宽高
    pub fn audit_discriminability(
        &self,
        pos: &[BgrImage<'_>],
        neg: &[BgrImage<'_>],
    ) -> Result<AuditReport, ShapeError> {
        if pos.is_empty() {
            return Err(ShapeError::InvalidInput("正组图像至少 1 帧".into()));
        }
        let (pos_ptrs, pos_width, pos_height) = group_ptrs(pos, "正组")?;
        let (neg_ptrs, neg_width, neg_height) = if neg.is_empty() {
            (Vec::new(), 0, 0)
        } else {
            group_ptrs(neg, "负组")?
        };
        self.with_live_handle(|handle| {
            let mut report = AuditReportAbi::default();
            self.library().abi().audit_discriminability(
                handle,
                &pos_ptrs,
                pos_width,
                pos_height,
                &neg_ptrs,
                neg_width,
                neg_height,
                &mut report,
            )?;
            Ok(AuditReport {
                flag: report.flag,
                confidence: report.confidence,
                pos_count: report.pos_count,
                pos_zero_hits: report.pos_zero_hits,
                pos_min: report.pos_min,
                pos_p25: report.pos_p25,
                pos_median: report.pos_median,
                pos_p75: report.pos_p75,
                pos_max: report.pos_max,
                neg_count: report.neg_count,
                neg_zero_hits: report.neg_zero_hits,
                neg_min: report.neg_min,
                neg_p25: report.neg_p25,
                neg_median: report.neg_median,
                neg_p75: report.neg_p75,
                neg_max: report.neg_max,
                separation: report.separation,
                recommend_lo: report.recommend_lo,
                recommend_hi: report.recommend_hi,
            })
        })?
    }
}
