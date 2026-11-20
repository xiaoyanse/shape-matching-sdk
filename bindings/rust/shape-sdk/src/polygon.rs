//! 多边形模板能力安全层：训练参数、多边形训练、多边形元数据回读。
//! ABI 结构与 unsafe 边界在本模块闭环；布局合同见 include/shape_matching.h。
//! 导出为加性（旧运行时缺失容忍，调用处报错），语义对齐 Python 侧
//! native.py 的多边形接口。轮廓建议族在 `suggest` 模块。

use crate::error::ShapeError;
use crate::ffi::NativeAbi;
use crate::library::ShapeLibrary;
use crate::model::{BgrImage, DetectorParams, Model};

/// 多边形顶点（参考图坐标系，像素）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolygonVertex {
    pub x: f64,
    pub y: f64,
}

/// 多边形训练参数（角度/缩放/阈值语义同 `TrainParams`；轮廓为顶点集合）。
/// 外环 3..64 顶点、任意绕向（SDK 归一）；洞 ≤8、每洞 3..64 顶点、面积
/// ≥64px² 且完全在外环内；外接矩形为预算与位姿映射的记账基。
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonTrainParams {
    pub outline: Vec<PolygonVertex>,
    pub holes: Vec<Vec<PolygonVertex>>,
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_step: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    /// None 用原生默认三参（128/30/60）
    pub detector: Option<DetectorParams>,
}

/// 多边形元数据回读：多边形资产返回归一化外环顶点与洞数；矩形资产（含旧
/// 格式）回退训练 ROI 四角且洞数为 0
#[derive(Debug, Clone, PartialEq)]
pub struct PolygonMetadata {
    pub outline: Vec<PolygonVertex>,
    pub hole_count: u32,
}

// —— ABI 裸结构（与 C 头文件布局一致，64 位口径） ——

/// vf_shape_vertex
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct VertexAbi {
    pub x: f64,
    pub y: f64,
}

/// vf_shape_polygon（顶点数组借用）
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct PolygonAbi {
    pub vertices: *const VertexAbi,
    pub count: u32,
}

/// vf_shape_polygon_train_params
#[repr(C)]
pub(crate) struct PolygonParamsAbi {
    pub struct_size: u32,
    pub outline: PolygonAbi,
    pub holes: *const PolygonAbi,
    pub hole_count: u32,
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_step: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    pub weak_th: f64,
    pub strong_th: f64,
    pub feature_count: i32,
}

// 编译期布局核对（与 C 头 sizeof 约定一致）
const _: () = assert!(std::mem::size_of::<VertexAbi>() == 16);
const _: () = assert!(std::mem::size_of::<PolygonAbi>() == 16);
const _: () = assert!(std::mem::size_of::<PolygonParamsAbi>() == 112);

pub(crate) type TrainPolygonFn = unsafe extern "C" fn(
    *const u8,
    u64,
    i32,
    i32,
    u64,
    *const PolygonParamsAbi,
    *mut u64,
    *mut u8,
    u32,
) -> i32;
pub(crate) type PolygonMetadataFn =
    unsafe extern "C" fn(u64, *mut VertexAbi, u32, *mut u32, *mut u32, *mut u8, u32) -> i32;

const ERROR_CAPACITY: usize = 2048;

fn missing_polygon_export(name: &str) -> ShapeError {
    ShapeError::Library(format!(
        "模板匹配原生组件缺少多边形导出 {name}（旧运行时），请升级制品后使用多边形接口"
    ))
}

impl NativeAbi {
    /// 多边形训练并注册句柄
    pub(crate) fn train_polygon(
        &self,
        image: &BgrImage<'_>,
        params: &PolygonParamsAbi,
    ) -> Result<u64, ShapeError> {
        let train = self
            .train_polygon_fn
            .ok_or_else(|| missing_polygon_export("vf_shape_train_polygon"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        let mut handle = 0u64;
        // SAFETY: image/params 指针仅在调用期间借用；handle/error 为出口缓冲。
        let status = unsafe {
            (train)(
                image.data().as_ptr(),
                image.data().len() as u64,
                image.width(),
                image.height(),
                3 * image.width() as u64,
                params,
                &mut handle,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(handle)
    }

    /// 多边形元数据回读（两次调用模式由调用方编排）
    pub(crate) fn polygon_metadata(
        &self,
        handle: u64,
        outline: *mut VertexAbi,
        capacity: u32,
        outline_count: *mut u32,
        hole_count: *mut u32,
    ) -> Result<(), ShapeError> {
        let metadata = self
            .polygon_metadata_fn
            .ok_or_else(|| missing_polygon_export("vf_shape_polygon_metadata"))?;
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: 出口指针均指向调用方缓冲；capacity 与 outline 缓冲一致。
        let status = unsafe {
            (metadata)(
                handle,
                outline,
                capacity,
                outline_count,
                hole_count,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)
    }
}

pub(crate) fn validate_polygon_params(params: &PolygonTrainParams) -> Result<(), ShapeError> {
    let invalid = || {
        ShapeError::InvalidInput(
            "多边形轮廓须为 3–64 顶点、洞 ≤8 且每洞 3–64 顶点，坐标须为有限数值".into(),
        )
    };
    if !(3..=64).contains(&params.outline.len()) || params.holes.len() > 8 {
        return Err(invalid());
    }
    let finite = |points: &[PolygonVertex]| {
        (3..=64).contains(&points.len())
            && points.iter().all(|p| p.x.is_finite() && p.y.is_finite())
    };
    if !finite(&params.outline) || params.holes.iter().any(|hole| !finite(hole)) {
        return Err(invalid());
    }
    Ok(())
}

/// 安全参数 → ABI 结构；返回的存储必须与 ABI 同寿命（顶点数组被其借用）。
/// 洞存储先收集完整再建视图：Vec 生长会重定位，视图必须指向最终稳定缓冲。
pub(crate) fn build_params_abi(
    params: &PolygonTrainParams,
) -> (
    PolygonParamsAbi,
    Vec<VertexAbi>,
    Vec<Vec<VertexAbi>>,
    Vec<PolygonAbi>,
) {
    let to_abi = |points: &[PolygonVertex]| {
        points
            .iter()
            .map(|p| VertexAbi { x: p.x, y: p.y })
            .collect::<Vec<_>>()
    };
    let outline_abi = to_abi(&params.outline);
    let hole_storage: Vec<Vec<VertexAbi>> =
        params.holes.iter().map(|hole| to_abi(hole)).collect();
    let hole_views: Vec<PolygonAbi> = hole_storage
        .iter()
        .map(|points| PolygonAbi {
            vertices: points.as_ptr(),
            count: points.len() as u32,
        })
        .collect();
    let outline = PolygonAbi {
        vertices: outline_abi.as_ptr(),
        count: outline_abi.len() as u32,
    };
    let (feature_count, weak_th, strong_th) = match &params.detector {
        None => (0, 0.0, 0.0),
        Some(detector) => (detector.feature_count, detector.weak_th, detector.strong_th),
    };
    let abi = PolygonParamsAbi {
        struct_size: std::mem::size_of::<PolygonParamsAbi>() as u32,
        outline,
        holes: hole_views.as_ptr(),
        hole_count: hole_views.len() as u32,
        angle_min: params.angle_min,
        angle_max: params.angle_max,
        angle_step: params.angle_step,
        scale_min: params.scale_min,
        scale_max: params.scale_max,
        scale_step: params.scale_step,
        weak_th,
        strong_th,
        feature_count,
    };
    (abi, outline_abi, hole_storage, hole_views)
}

/// 多边形训练并导出资产字节（训练句柄即用即释）；走全局共享库实例
pub fn train_polygon_model(
    reference: BgrImage<'_>,
    params: &PolygonTrainParams,
) -> Result<Vec<u8>, ShapeError> {
    ShapeLibrary::global()?.train_polygon_model(reference, params)
}

impl ShapeLibrary {
    /// 多边形训练并导出资产字节（训练句柄即用即释，导出失败也保证关闭）
    pub fn train_polygon_model(
        &self,
        reference: BgrImage<'_>,
        params: &PolygonTrainParams,
    ) -> Result<Vec<u8>, ShapeError> {
        validate_polygon_params(params)?;
        // 借用存储与 ABI 同寿命：顶点数组必须活到原生调用返回
        let (abi, _outline_abi, _hole_storage, _hole_views) = build_params_abi(params);
        let handle = self.abi().train_polygon(&reference, &abi)?;
        let exported = self.export_model_bytes(handle);
        self.close_unchecked(handle);
        exported
    }
}

impl Model {
    /// 多边形元数据回读（矩形资产回退 ROI 四角，洞数 0）
    pub fn polygon_metadata(&self) -> Result<PolygonMetadata, ShapeError> {
        self.with_live_handle(|handle| -> Result<PolygonMetadata, ShapeError> {
            let abi = self.library().abi();
            let mut count = 0u32;
            let mut holes = 0u32;
            abi.polygon_metadata(handle, std::ptr::null_mut(), 0, &mut count, &mut holes)?;
            let mut outline = vec![VertexAbi { x: 0.0, y: 0.0 }; count as usize];
            let outline_ptr = if outline.is_empty() {
                std::ptr::null_mut()
            } else {
                outline.as_mut_ptr()
            };
            abi.polygon_metadata(handle, outline_ptr, count, &mut count, &mut holes)?;
            Ok(PolygonMetadata {
                outline: outline
                    .into_iter()
                    .map(|v| PolygonVertex { x: v.x, y: v.y })
                    .collect(),
                hole_count: holes,
            })
        })?
    }
}
