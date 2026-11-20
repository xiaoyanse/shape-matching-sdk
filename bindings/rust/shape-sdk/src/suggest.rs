//! 轮廓建议安全层：固定阈值（v1）与参数化（_ex）建议。ABI 结构、unsafe 调用
//! 与建议收集编排在本模块闭环；布局合同见 include/shape_matching.h。两个导出
//! 同为加性（旧运行时缺失容忍，调用处报错），语义对齐 Python 侧 native.py。

use crate::error::ShapeError;
use crate::ffi::NativeAbi;
use crate::library::ShapeLibrary;
use crate::model::BgrImage;
use crate::polygon::{PolygonVertex, VertexAbi};

/// 轮廓建议参数（vf_shape_suggest_params）；None 字段=原生默认（Canny 50/150、
/// epsilon 0.02×周长）。有效域：canny_low>0、canny_high≥canny_low、
/// epsilon_factor 0.001..0.2，越界由原生端按 DETECTOR_PARAMS 拒绝。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuggestParams {
    pub canny_low: Option<f64>,
    pub canny_high: Option<f64>,
    pub epsilon_factor: Option<f64>,
}

/// vf_shape_suggest_params（None→0=原生默认）
#[repr(C)]
pub(crate) struct SuggestParamsAbi {
    pub struct_size: u32,
    pub canny_low: f64,
    pub canny_high: f64,
    pub epsilon_factor: f64,
}

// 编译期布局核对（与 C 头 sizeof 约定一致）
const _: () = assert!(std::mem::size_of::<SuggestParamsAbi>() == 32);

pub(crate) type SuggestContoursFn = unsafe extern "C" fn(
    *const u8,
    u64,
    i32,
    i32,
    u64,
    *const i32,
    u32,
    u32,
    *mut VertexAbi,
    *mut u32,
    *mut u8,
    u32,
) -> i32;
pub(crate) type SuggestContoursExFn = unsafe extern "C" fn(
    *const u8,
    u64,
    i32,
    i32,
    u64,
    *const i32,
    u32,
    u32,
    *const SuggestParamsAbi,
    *mut VertexAbi,
    *mut u32,
    *mut u8,
    u32,
) -> i32;

const ERROR_CAPACITY: usize = 2048;

fn missing_suggest_export(name: &str) -> ShapeError {
    ShapeError::Library(format!(
        "模板匹配原生组件缺少轮廓建议导出 {name}（旧运行时），请升级制品后使用建议接口"
    ))
}

impl NativeAbi {
    /// 轮廓建议：返回原始返回码（≥0=建议条数，负=失败已带文本）
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn suggest_contours(
        &self,
        image: &BgrImage<'_>,
        roi_hint: *const i32,
        max_suggestions: u32,
        max_vertices_per: u32,
        vertices: &mut [VertexAbi],
        counts: &mut [u32],
    ) -> Result<i32, ShapeError> {
        let suggest = self
            .suggest_contours_fn
            .ok_or_else(|| missing_suggest_export("vf_shape_suggest_contours"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: 输入图/顶点/计数缓冲均在调用期间借用，容量一致传递。
        let status = unsafe {
            (suggest)(
                image.data().as_ptr(),
                image.data().len() as u64,
                image.width(),
                image.height(),
                3 * image.width() as u64,
                roi_hint,
                max_suggestions,
                max_vertices_per,
                vertices.as_mut_ptr(),
                counts.as_mut_ptr(),
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        if status < 0 {
            // 负值分层：-1 可报告 / -2 未分类，镜像正域 status 码
            return Err(ShapeError::native(-status, &error));
        }
        Ok(status)
    }

    /// 参数化轮廓建议：返回原始返回码（≥0=建议条数，负=失败已带文本）
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn suggest_contours_ex(
        &self,
        image: &BgrImage<'_>,
        roi_hint: *const i32,
        max_suggestions: u32,
        max_vertices_per: u32,
        params: *const SuggestParamsAbi,
        vertices: &mut [VertexAbi],
        counts: &mut [u32],
    ) -> Result<i32, ShapeError> {
        let suggest = self
            .suggest_contours_ex_fn
            .ok_or_else(|| missing_suggest_export("vf_shape_suggest_contours_ex"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: 输入图/参数/顶点/计数缓冲均在调用期间借用，容量一致传递。
        let status = unsafe {
            (suggest)(
                image.data().as_ptr(),
                image.data().len() as u64,
                image.width(),
                image.height(),
                3 * image.width() as u64,
                roi_hint,
                max_suggestions,
                max_vertices_per,
                params,
                vertices.as_mut_ptr(),
                counts.as_mut_ptr(),
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        if status < 0 {
            return Err(ShapeError::native(-status, &error));
        }
        Ok(status)
    }
}

/// 轮廓建议：返回建议集合（每条为顶点数组，原图坐标系）；空集=无建议非错误。
/// roi_hint 为 xywh 先验框（须完整落在图内），None=全图；max_vertices_per
/// 限每条顶点数 3..64，max_suggestions 1..256。
pub fn suggest_contours(
    image: BgrImage<'_>,
    roi_hint: Option<[i32; 4]>,
    max_suggestions: u32,
    max_vertices_per: u32,
) -> Result<Vec<Vec<PolygonVertex>>, ShapeError> {
    run_suggest(roi_hint, max_suggestions, max_vertices_per, |abi, roi, vertices, counts| {
        abi.suggest_contours(&image, roi, max_suggestions, max_vertices_per, vertices, counts)
    })
}

/// 参数化轮廓建议：默认参数（params=None 或全部 None 字段）下与
/// `suggest_contours` 输出逐字节一致；阈值域由原生端终审。
pub fn suggest_contours_ex(
    image: BgrImage<'_>,
    roi_hint: Option<[i32; 4]>,
    max_suggestions: u32,
    max_vertices_per: u32,
    params: Option<&SuggestParams>,
) -> Result<Vec<Vec<PolygonVertex>>, ShapeError> {
    let to_abi = |value: Option<f64>| value.unwrap_or(0.0);
    let params_abi = params.map(|p| SuggestParamsAbi {
        struct_size: std::mem::size_of::<SuggestParamsAbi>() as u32,
        canny_low: to_abi(p.canny_low),
        canny_high: to_abi(p.canny_high),
        epsilon_factor: to_abi(p.epsilon_factor),
    });
    let params_ptr = params_abi
        .as_ref()
        .map_or(std::ptr::null(), |p| p as *const SuggestParamsAbi);
    run_suggest(roi_hint, max_suggestions, max_vertices_per, |abi, roi, vertices, counts| {
        abi.suggest_contours_ex(
            &image,
            roi,
            max_suggestions,
            max_vertices_per,
            params_ptr,
            vertices,
            counts,
        )
    })
}

/// 建议公共编排：容量校验、缓冲准备与逐条收集；invoke 闭包返回建议条数
fn run_suggest(
    roi_hint: Option<[i32; 4]>,
    max_suggestions: u32,
    max_vertices_per: u32,
    invoke: impl Fn(
        &NativeAbi,
        *const i32,
        &mut [VertexAbi],
        &mut [u32],
    ) -> Result<i32, ShapeError>,
) -> Result<Vec<Vec<PolygonVertex>>, ShapeError> {
    if !(1..=256).contains(&max_suggestions) {
        return Err(ShapeError::InvalidInput("建议条数须为 1–256".into()));
    }
    if !(3..=64).contains(&max_vertices_per) {
        return Err(ShapeError::InvalidInput("每条建议顶点数须为 3–64".into()));
    }
    let mut vertices =
        vec![VertexAbi { x: 0.0, y: 0.0 }; max_suggestions as usize * max_vertices_per as usize];
    let mut counts = vec![0u32; max_suggestions as usize];
    let roi = roi_hint
        .as_ref()
        .map_or(std::ptr::null(), |rect| rect.as_ptr());
    let library = ShapeLibrary::global()?;
    let count = invoke(library.abi(), roi, &mut vertices, &mut counts)?;
    collect_suggestions(count, &vertices, &counts)
}

fn collect_suggestions(
    count: i32,
    vertices: &[VertexAbi],
    counts: &[u32],
) -> Result<Vec<Vec<PolygonVertex>>, ShapeError> {
    let mut suggestions = Vec::with_capacity(count as usize);
    let mut offset = 0usize;
    for count_per in counts.iter().take(count as usize) {
        let end = offset + *count_per as usize;
        suggestions.push(
            vertices[offset..end]
                .iter()
                .map(|v| PolygonVertex { x: v.x, y: v.y })
                .collect(),
        );
        offset = end;
    }
    Ok(suggestions)
}
