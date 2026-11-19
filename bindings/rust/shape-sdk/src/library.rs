//! 原生库装载与校验：运行时目录发现、manifest 哈希链校验、ABI 协商、
//! 全局缓存与两段式导出帮助。语义对齐 Python 侧 native_library.py。

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::ShapeError;
use crate::ffi::{NativeAbi, TrainConfigAbi};
use crate::model::BgrImage;
use crate::model::TrainParams;

/// 期望的 ABI 版本（include/shape_matching.h 合同）
const ABI_VERSION: u32 = 3;
/// 模型资产字节上限（vf_shape_load 的 32MiB 合同）
pub(crate) const MAX_MODEL_BYTES: usize = 32 * 1024 * 1024;
/// 单帧批量模型数上限（vf_shape_match_batch 的 1..16 合同）
pub(crate) const MAX_BATCH_ITEMS: usize = 16;

/// runtime-manifest.json 模式（未知字段容忍——serde 默认忽略）
#[derive(Debug, Deserialize)]
struct Manifest {
    abi_version: u32,
    sdk_version: String,
    files: HashMap<String, String>,
}

/// 平台主库文件名（对齐 Python：win32 用 dll，其余用 so）
fn main_library_filename() -> &'static str {
    if cfg!(windows) {
        "visionforge_shape.dll"
    } else {
        "libvisionforge_shape.so"
    }
}

fn unavailable(dir: &Path) -> ShapeError {
    ShapeError::Library(format!("模板匹配原生组件不可用，请构建或修复安装：{}", dir.display()))
}

fn hex_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// 校验 manifest 全部文件条目：文件名须为纯文件名（无路径分量），
/// 逐文件比对 SHA256（小写十六进制严格相等，口径同 Python 侧）
fn verify_files(dir: &Path, files: &HashMap<String, String>) -> Result<(), ShapeError> {
    for (filename, expected) in files {
        if Path::new(filename).file_name().and_then(|f| f.to_str()) != Some(filename) {
            return Err(ShapeError::Library("模板匹配原生清单路径非法".into()));
        }
        let bytes = fs::read(dir.join(filename)).map_err(|_| unavailable(dir))?;
        if hex_digest(&bytes) != *expected {
            return Err(ShapeError::Library(format!("模板匹配原生文件校验失败：{filename}")));
        }
    }
    Ok(())
}

/// 读取并校验 manifest：解析失败/缺 abi_version/files 均归为组件不可用，
/// abi_version 非 3 为版本不兼容（对齐 Python 侧两类报错口径）
fn verify_manifest(dir: &Path) -> Result<Manifest, ShapeError> {
    let text = fs::read_to_string(dir.join("runtime-manifest.json")).map_err(|_| unavailable(dir))?;
    let manifest: Manifest =
        serde_json::from_str(&text).map_err(|_| unavailable(dir))?;
    if manifest.abi_version != ABI_VERSION {
        return Err(ShapeError::Library("模板匹配原生 ABI 版本不兼容".into()));
    }
    if !manifest.files.contains_key(main_library_filename()) {
        return Err(ShapeError::Library("模板匹配原生清单缺少主库".into()));
    }
    verify_files(dir, &manifest.files)?;
    Ok(manifest)
}

/// 已装载并通过校验协商的原生库实例
pub struct ShapeLibrary {
    runtime_dir: PathBuf,
    sdk_version: String,
    abi: NativeAbi,
}

impl ShapeLibrary {
    /// 运行时目录发现：`SHAPE_SDK_RUNTIME` 优先（空值视为未设置），
    /// 否则取仓内 dist/runtime（编译期 crate 目录上溯三级到仓根）
    pub fn discover_runtime_dir() -> Result<PathBuf, ShapeError> {
        if let Some(explicit) = std::env::var_os("SHAPE_SDK_RUNTIME")
            .filter(|value| !value.is_empty())
        {
            return Ok(PathBuf::from(explicit));
        }
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = crate_dir
            .parent()
            .and_then(Path::parent)
            .and_then(Path::parent)
            .ok_or_else(|| {
                ShapeError::Library("无法从 crate 位置定位仓根，请设置 SHAPE_SDK_RUNTIME".into())
            })?;
        Ok(repo_root.join("dist").join("runtime"))
    }

    /// 按默认发现规则装载：manifest 哈希链校验 → 装载 → 符号解析 → ABI 协商
    pub fn open() -> Result<Self, ShapeError> {
        let dir = Self::discover_runtime_dir()?;
        Self::open_at(&dir)
    }

    /// 指定制品目录装载（测试与自定义部署入口）
    pub fn open_at(dir: &Path) -> Result<Self, ShapeError> {
        let manifest = verify_manifest(dir)?;
        let abi = NativeAbi::open(&dir.join(main_library_filename()))?;
        if abi.abi_version() != ABI_VERSION {
            return Err(ShapeError::Library("模板匹配原生 ABI 不兼容".into()));
        }
        Ok(Self {
            runtime_dir: dir.to_path_buf(),
            sdk_version: manifest.sdk_version,
            abi,
        })
    }

    /// 全局共享实例：成功后缓存，失败不缓存（下次重试，对齐 Python 模块级缓存）
    pub fn global() -> Result<Arc<Self>, ShapeError> {
        static CACHE: OnceLock<Arc<ShapeLibrary>> = OnceLock::new();
        static SLOW_PATH: Mutex<()> = Mutex::new(());
        if let Some(cached) = CACHE.get() {
            return Ok(Arc::clone(cached));
        }
        let _guard = SLOW_PATH.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = CACHE.get() {
            return Ok(Arc::clone(cached));
        }
        let library = Arc::new(Self::open()?);
        let _ = CACHE.set(Arc::clone(&library));
        Ok(library)
    }

    /// 本实例的制品目录
    pub fn runtime_dir(&self) -> &Path {
        &self.runtime_dir
    }

    /// 产品版本（manifest sdk_version，与 ABI 版本分属不同命名空间）
    pub fn sdk_version(&self) -> &str {
        &self.sdk_version
    }

    pub(crate) fn abi(&self) -> &NativeAbi {
        &self.abi
    }

    /// 训练模板模型并导出资产字节；训练句柄即用即释（导出失败也保证关闭）
    pub fn train_model(
        &self,
        reference: BgrImage<'_>,
        params: &TrainParams,
    ) -> Result<Vec<u8>, ShapeError> {
        params.validate()?;
        let (feature_count, weak_th, strong_th) = match &params.detector {
            None => (0, 0.0, 0.0),
            Some(detector) => (detector.feature_count, detector.weak_th, detector.strong_th),
        };
        let config = train_config_abi(params, feature_count, weak_th, strong_th);
        let handle = self
            .abi
            .train(reference.data(), reference.width(), reference.height(), &config)?;
        let exported = self.export_model_bytes(handle);
        self.close_unchecked(handle);
        exported
    }

    /// 尽力关闭句柄（Drop 路径忽略结果，对齐 Python _close_unchecked）
    pub(crate) fn close_unchecked(&self, handle: u64) {
        let _ = self.abi.close(handle);
    }

    /// 两段式导出：先查长度再取回；超预算或长度不一致拒绝
    pub(crate) fn export_model_bytes(&self, handle: u64) -> Result<Vec<u8>, ShapeError> {
        let size = self.abi.export_size(handle)?;
        if size == 0 || size as usize > MAX_MODEL_BYTES {
            return Err(ShapeError::Output("原生模型输出超过预算".into()));
        }
        let mut bytes = vec![0u8; size as usize];
        let written = self.abi.export_into(handle, &mut bytes)?;
        if written != size {
            return Err(ShapeError::Output("原生模型输出长度不一致".into()));
        }
        Ok(bytes)
    }

}

pub(crate) fn train_config_abi(
    params: &TrainParams,
    feature_count: i32,
    weak_th: f64,
    strong_th: f64,
) -> TrainConfigAbi {
    let (roi_x, roi_y, roi_width, roi_height) = params.roi_xywh;
    TrainConfigAbi {
        struct_size: std::mem::size_of::<TrainConfigAbi>() as u32,
        roi_x,
        roi_y,
        roi_width,
        roi_height,
        angle_min: params.angle_min,
        angle_max: params.angle_max,
        angle_step: params.angle_step,
        scale_min: params.scale_min,
        scale_max: params.scale_max,
        scale_step: params.scale_step,
        weak_th,
        strong_th,
        feature_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_runtime(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("shape-sdk-ffi-test-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("创建测试临时目录失败");
        dir
    }

    fn write_manifest(dir: &Path, json: &str) {
        std::fs::write(dir.join("runtime-manifest.json"), json).expect("写测试 manifest 失败");
    }

    /// 好链：主库在册 + 哈希匹配 + abi 3 → 通过
    #[test]
    fn manifest_good_chain_passes() {
        let dir = temp_runtime("good");
        std::fs::write(dir.join(main_library_filename()), b"fake-lib").expect("写主库失败");
        std::fs::write(dir.join("note.txt"), b"side-file").expect("写附属文件失败");
        let json = format!(
            "{{\"abi_version\":3,\"sdk_version\":\"0.1.0\",\"files\":{{\"{}\":\"{}\",\"note.txt\":\"{}\"}}}}",
            main_library_filename(),
            hex_digest(b"fake-lib"),
            hex_digest(b"side-file")
        );
        write_manifest(&dir, &json);
        let manifest = verify_manifest(&dir).expect("好链应通过");
        assert_eq!(manifest.sdk_version, "0.1.0");
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }

    #[test]
    fn manifest_abi_mismatch_rejected() {
        let dir = temp_runtime("abi");
        let json = "{\"abi_version\":2,\"sdk_version\":\"0.1.0\",\"files\":{}}";
        write_manifest(&dir, json);
        let err = verify_manifest(&dir).expect_err("ABI 非 3 应拒绝");
        assert_eq!(err, ShapeError::Library("模板匹配原生 ABI 版本不兼容".into()));
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }

    #[test]
    fn manifest_path_traversal_rejected() {
        let dir = temp_runtime("traversal");
        std::fs::write(dir.join(main_library_filename()), b"fake-lib").expect("写主库失败");
        // 主库条目合法，确保哈希校验阶段走到路径分量检查
        let json = format!(
            "{{\"abi_version\":3,\"sdk_version\":\"0.1.0\",\"files\":{{\"{}\":\"{}\",\"../evil\":\"00\"}}}}",
            main_library_filename(),
            hex_digest(b"fake-lib")
        );
        write_manifest(&dir, &json);
        let err = verify_manifest(&dir).expect_err("路径分量应拒绝");
        assert_eq!(err, ShapeError::Library("模板匹配原生清单路径非法".into()));
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }

    #[test]
    fn manifest_hash_mismatch_rejected() {
        let dir = temp_runtime("hash");
        std::fs::write(dir.join(main_library_filename()), b"tampered").expect("写主库失败");
        let json = format!(
            "{{\"abi_version\":3,\"sdk_version\":\"0.1.0\",\"files\":{{\"{}\":\"{}\"}}}}",
            main_library_filename(),
            hex_digest(b"expected-content")
        );
        write_manifest(&dir, &json);
        let err = verify_manifest(&dir).expect_err("哈希不符应拒绝");
        assert!(err.message().contains("文件校验失败"), "实际：{err}");
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }

    #[test]
    fn manifest_missing_main_library_rejected() {
        let dir = temp_runtime("nomain");
        std::fs::write(dir.join("note.txt"), b"x").expect("写附属文件失败");
        let json = format!(
            "{{\"abi_version\":3,\"sdk_version\":\"0.1.0\",\"files\":{{\"note.txt\":\"{}\"}}}}",
            hex_digest(b"x")
        );
        write_manifest(&dir, &json);
        let err = verify_manifest(&dir).expect_err("缺主库应拒绝");
        assert_eq!(err, ShapeError::Library("模板匹配原生清单缺少主库".into()));
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }

    #[test]
    fn manifest_broken_json_rejected() {
        let dir = temp_runtime("broken");
        write_manifest(&dir, "not-json");
        let err = verify_manifest(&dir).expect_err("坏 JSON 应拒绝");
        assert!(matches!(err, ShapeError::Library(_)));
        std::fs::remove_dir_all(&dir).expect("清理测试临时目录失败");
    }
}
