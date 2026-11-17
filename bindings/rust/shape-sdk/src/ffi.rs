//! C ABI v3 裸声明与调用层——全部 unsafe 边界（符号解析、指针调用、错误缓冲）
//! 收拢在本模块。结构体布局、函数指针类型、逐调用包装均在此；安全层只消费
//! `Result`，不接触裸指针。布局合同见 include/shape_matching.h。

use std::path::Path;

// Windows 侧用 os 层库类型以携带依赖搜索 flags；其余平台用安全封装
#[cfg(windows)]
use libloading::os::windows::Library;
#[cfg(not(windows))]
use libloading::Library;

use crate::error::ShapeError;

/// 原生入口错误缓冲容量（对齐 Python 绑定的 2048 字节）
const ERROR_CAPACITY: usize = 2048;

/// 训练配置（vf_shape_train_config）
#[repr(C)]
pub(crate) struct TrainConfigAbi {
    pub struct_size: u32,
    pub roi_x: i32,
    pub roi_y: i32,
    pub roi_width: i32,
    pub roi_height: i32,
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

impl Default for TrainConfigAbi {
    fn default() -> Self {
        Self {
            struct_size: 0,
            roi_x: 0,
            roi_y: 0,
            roi_width: 0,
            roi_height: 0,
            angle_min: 0.0,
            angle_max: 0.0,
            angle_step: 0.0,
            scale_min: 0.0,
            scale_max: 0.0,
            scale_step: 0.0,
            weak_th: 0.0,
            strong_th: 0.0,
            feature_count: 0,
        }
    }
}

/// 单个匹配结果（vf_shape_candidate），数值合法性由安全层校验
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CandidateAbi {
    pub score: f64,
    pub pose_id: i32,
    pub pose_ambiguous: u32,
    pub center: [f64; 2],
    pub angle: f64,
    pub scale: f64,
    pub corners: [f64; 8],
    pub bbox: [f64; 4],
    pub transform: [f64; 6],
}

impl Default for CandidateAbi {
    fn default() -> Self {
        Self {
            score: 0.0,
            pose_id: 0,
            pose_ambiguous: 0,
            center: [0.0; 2],
            angle: 0.0,
            scale: 0.0,
            corners: [0.0; 8],
            bbox: [0.0; 4],
            transform: [0.0; 6],
        }
    }
}

/// 批量条目（vf_shape_batch_item）；error 为库侧逐项诊断输出
#[repr(C)]
pub(crate) struct BatchItemAbi {
    pub handle: u64,
    pub threshold: f64,
    pub candidates: *mut CandidateAbi,
    pub capacity: u32,
    pub count: u32,
    pub status: i32,
    pub error: [u8; 256],
}

// 说明：BatchItemAbi 由本模块单线程构造后传入原生调用，无需跨线程传递。

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type TrainFn = unsafe extern "C" fn(
    *const u8,
    u64,
    i32,
    i32,
    u64,
    *const TrainConfigAbi,
    *mut u64,
    *mut u8,
    u32,
) -> i32;
type LoadFn = unsafe extern "C" fn(*const u8, u64, *mut u64, *mut u8, u32) -> i32;
type ExportFn = unsafe extern "C" fn(u64, *mut u8, u64, *mut u64, *mut u8, u32) -> i32;
type MetadataFn =
    unsafe extern "C" fn(u64, *mut TrainConfigAbi, *mut i32, *mut i32, *mut u8, u32) -> i32;
type MatchFn = unsafe extern "C" fn(
    u64,
    *const u8,
    u64,
    i32,
    i32,
    u64,
    f64,
    *mut CandidateAbi,
    u32,
    *mut u32,
    *mut u8,
    u32,
) -> i32;
type CloseFn = unsafe extern "C" fn(u64, *mut u8, u32) -> i32;
type MatchBatchFn = unsafe extern "C" fn(
    *const u8,
    u64,
    i32,
    i32,
    u64,
    *mut BatchItemAbi,
    u32,
    u32,
    *mut u8,
    u32,
) -> i32;

/// 已解析的原生 ABI：库句柄与 8 个入口的函数指针同寿命共存于本结构，
/// 调用仅经由本结构方法进行（库存活期间指针有效，不存在自引用失效问题）。
pub(crate) struct NativeAbi {
    #[allow(dead_code)] // 字段本身承载库寿命；Drop 由它负责
    library: Library,
    abi_version_fn: AbiVersionFn,
    train_fn: TrainFn,
    load_fn: LoadFn,
    export_fn: ExportFn,
    metadata_fn: MetadataFn,
    match_fn: MatchFn,
    close_fn: CloseFn,
    match_batch_fn: MatchBatchFn,
}

/// 装载动态库。Windows 侧显式搜索 DLL 自身目录（同目录 opencv 依赖）与
/// 系统目录，语义对齐 Python 侧 os.add_dll_directory 的装载前准备。
#[cfg(windows)]
fn load_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };
    // SAFETY: 路径来自运行时发现且经 manifest 哈希校验；flags 为官方推荐组合。
    unsafe {
        Library::load_with_flags(
            path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    }
}

#[cfg(not(windows))]
fn load_library(path: &Path) -> Result<Library, libloading::Error> {
    // SAFETY: 同上，dlopen 无搜索路径 flags 需求。
    unsafe { Library::new(path) }
}

impl NativeAbi {
    /// 装载动态库并解析全部 8 个符号；缺失任一即失败
    pub(crate) fn open(path: &Path) -> Result<Self, ShapeError> {
        // SAFETY: 装载外部库会执行其初始化代码；路径来自本 crate 的运行时发现
        // 且经 manifest 哈希校验。函数指针从 Symbol 解引用拷贝后仅随本结构调用，
        // 库句柄存于同结构字段，不会被先于指针使用而卸载。
        unsafe {
            let unavailable = |source: libloading::Error| {
                ShapeError::Library(format!(
                    "模板匹配原生组件不可用，请构建或修复安装：{}（{source}）",
                    path.display()
                ))
            };
            let library = load_library(path).map_err(unavailable)?;
            let missing = |name: &str| {
                ShapeError::Library(format!("模板匹配原生组件缺少 {name}，请重新构建或修复安装"))
            };
            Ok(Self {
                abi_version_fn: *library
                    .get(b"vf_shape_abi_version\0")
                    .map_err(|_| missing("vf_shape_abi_version"))?,
                train_fn: *library
                    .get(b"vf_shape_train\0")
                    .map_err(|_| missing("vf_shape_train"))?,
                load_fn: *library
                    .get(b"vf_shape_load\0")
                    .map_err(|_| missing("vf_shape_load"))?,
                export_fn: *library
                    .get(b"vf_shape_export\0")
                    .map_err(|_| missing("vf_shape_export"))?,
                metadata_fn: *library
                    .get(b"vf_shape_metadata\0")
                    .map_err(|_| missing("vf_shape_metadata"))?,
                match_fn: *library
                    .get(b"vf_shape_match\0")
                    .map_err(|_| missing("vf_shape_match"))?,
                close_fn: *library
                    .get(b"vf_shape_close\0")
                    .map_err(|_| missing("vf_shape_close"))?,
                match_batch_fn: *library
                    .get(b"vf_shape_match_batch\0")
                    .map_err(|_| missing("vf_shape_match_batch"))?,
                library,
            })
        }
    }

    /// 版本协商入口（无失败路径）
    pub(crate) fn abi_version(&self) -> u32 {
        // SAFETY: 无参纯查询入口，不接触调用方缓冲。
        unsafe { (self.abi_version_fn)() }
    }

    /// 训练并注册句柄；data 须为连续 BGR8（长度 3×width×height）
    pub(crate) fn train(
        &self,
        data: &[u8],
        width: i32,
        height: i32,
        config: &TrainConfigAbi,
    ) -> Result<u64, ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut handle = 0u64;
        // SAFETY: data/config 指针仅在调用期间借用（库不保留调用方指针）；
        // error 缓冲可写长度即容量；handle 为出口参数。
        let status = unsafe {
            (self.train_fn)(
                data.as_ptr(),
                data.len() as u64,
                width,
                height,
                3 * width as u64,
                config,
                &mut handle,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(handle)
    }

    /// 加载模型资产并注册句柄
    pub(crate) fn load(&self, bytes: &[u8]) -> Result<u64, ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut handle = 0u64;
        // SAFETY: bytes 只读借用；handle/error 为本栈帧出口缓冲。
        let status = unsafe {
            (self.load_fn)(
                bytes.as_ptr(),
                bytes.len() as u64,
                &mut handle,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(handle)
    }

    /// 两段式导出第一步：以空缓冲查询资产长度
    pub(crate) fn export_size(&self, handle: u64) -> Result<u64, ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut size = 0u64;
        // SAFETY: bytes=NULL 且 capacity=0 是合同约定的长度查询模式。
        let status = unsafe {
            (self.export_fn)(
                handle,
                std::ptr::null_mut(),
                0,
                &mut size,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(size)
    }

    /// 两段式导出第二步：取回资产字节，返回实写长度
    pub(crate) fn export_into(&self, handle: u64, bytes: &mut [u8]) -> Result<u64, ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut size = 0u64;
        // SAFETY: bytes 可写长度与 capacity 一致；调用期间独占借用。
        let status = unsafe {
            (self.export_fn)(
                handle,
                bytes.as_mut_ptr(),
                bytes.len() as u64,
                &mut size,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(size)
    }

    /// 查询训练配置回显与参考图尺寸
    pub(crate) fn metadata(&self, handle: u64) -> Result<(TrainConfigAbi, i32, i32), ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut config = TrainConfigAbi {
            struct_size: std::mem::size_of::<TrainConfigAbi>() as u32,
            ..TrainConfigAbi::default()
        };
        let mut width = 0i32;
        let mut height = 0i32;
        // SAFETY: 三个出口指针均指向本栈帧对象；struct_size 已按合同预设。
        let status = unsafe {
            (self.metadata_fn)(
                handle,
                &mut config,
                &mut width,
                &mut height,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok((config, width, height))
    }

    /// 单模型匹配；返回实得候选数（写入 candidates 前缀）
    pub(crate) fn match_one(
        &self,
        handle: u64,
        data: &[u8],
        width: i32,
        height: i32,
        threshold: f64,
        candidates: &mut [CandidateAbi],
    ) -> Result<u32, ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        let mut count = 0u32;
        // SAFETY: 输入图只读借用（合同：调用返回前调用方不得改写）；
        // candidates/count 为出口缓冲，容量一致传递。
        let status = unsafe {
            (self.match_fn)(
                handle,
                data.as_ptr(),
                data.len() as u64,
                width,
                height,
                3 * width as u64,
                threshold,
                candidates.as_mut_ptr(),
                candidates.len() as u32,
                &mut count,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)?;
        Ok(count)
    }

    /// 释放句柄（原生侧幂等：未知/已关闭句柄同样返回 0）
    pub(crate) fn close(&self, handle: u64) -> Result<(), ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: handle 为值传递；error 为栈上缓冲。
        let status = unsafe { (self.close_fn)(handle, error.as_mut_ptr(), ERROR_CAPACITY as u32) };
        ShapeError::check(status, &error)
    }

    /// 同帧批量匹配；items 的 count/status/error 为逐项出口，返回码仅表示
    /// 批量流程完成（合同：非 0 时逐项状态不构成完整结果）
    pub(crate) fn match_batch(
        &self,
        data: &[u8],
        width: i32,
        height: i32,
        items: &mut [BatchItemAbi],
    ) -> Result<(), ShapeError> {
        let mut error = [0u8; ERROR_CAPACITY];
        // SAFETY: items 数组与内部 candidates 指针均在调用期间独占借用；
        // item_size 按合同传结构体实际大小。
        let status = unsafe {
            (self.match_batch_fn)(
                data.as_ptr(),
                data.len() as u64,
                width,
                height,
                3 * width as u64,
                items.as_mut_ptr(),
                items.len() as u32,
                std::mem::size_of::<BatchItemAbi>() as u32,
                error.as_mut_ptr(),
                ERROR_CAPACITY as u32,
            )
        };
        ShapeError::check(status, &error)
    }
}

// 编译期布局核对：与 C 头文件的 sizeof 约定一致（64 位平台口径）
const _: () = assert!(std::mem::size_of::<TrainConfigAbi>() == 96);
const _: () = assert!(std::mem::size_of::<CandidateAbi>() == 192);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(std::mem::size_of::<BatchItemAbi>() == 296);
