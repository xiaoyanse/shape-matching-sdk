//! 安全层：图像视图校验、训练/加载/匹配/批量匹配的安全包装与候选数值校验。
//! 语义对齐 Python 侧 native.py 与 native_batch.py；错误一律走 ShapeError。

use std::sync::{Arc, Mutex, MutexGuard};

use crate::error::ShapeError;
use crate::ffi::{BatchItemAbi, CandidateAbi};
use crate::library::{ShapeLibrary, MAX_BATCH_ITEMS, MAX_MODEL_BYTES};

/// 连续 BGR 8bit 三通道图像视图（stride 恒为 3×width，坐标原点左上）
#[derive(Debug, Clone, Copy)]
pub struct BgrImage<'a> {
    width: i32,
    height: i32,
    data: &'a [u8],
}

impl<'a> BgrImage<'a> {
    /// 宽高须在 24..4096 像素，数据长度须等于 3×宽×高（连续性由长度合同保证）
    pub fn new(width: i32, height: i32, data: &'a [u8]) -> Result<Self, ShapeError> {
        let invalid =
            || ShapeError::InvalidInput("图像须为连续 uint8 BGR，宽高范围 24–4096".into());
        if !(24..=4096).contains(&width)
            || !(24..=4096).contains(&height)
            || data.len() != 3 * width as usize * height as usize
        {
            return Err(invalid());
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn width(&self) -> i32 {
        self.width
    }

    pub fn height(&self) -> i32 {
        self.height
    }

    pub(crate) fn data(&self) -> &[u8] {
        self.data
    }
}

/// 匹配候选（字段与 vf_shape_candidate 一一对应；几何位于输入帧坐标系）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// 0..100 的特征方向相似度（阈值过滤用，非校准概率）
    pub score: f64,
    /// 命中姿态索引（两级容器模型为 fine 组内局部索引，跨组不可比）
    pub pose_id: i32,
    /// 姿态多解标记（聚类合并过角度差 ≥15° 或缩放比 ±14% 且分差 <10 的多解）
    pub pose_ambiguous: bool,
    /// 训练 ROI 四角映射均值落点（两级 top1 为亚像素精修后中心）
    pub center: (f64, f64),
    /// 与训练角度同口径（两级 top1 为精修角，可小幅偏离离散档位）
    pub angle: f64,
    /// 相对训练 ROI 的缩放系数
    pub scale: f64,
    /// 训练 ROI 四角映射落点：左上、右上、右下、左下
    pub corners: [(f64, f64); 4],
    /// 四角外接框 (x, y, w, h)
    pub bbox: (f64, f64, f64, f64),
    /// 2×3 行优先仿射矩阵 [a b tx; c d ty]（含 ROI 偏移的整幅参考图坐标 → 帧）
    pub transform: [f64; 6],
}

/// 特征筛选三参；零值=原生默认（feature_count=128、weak_th=30、strong_th=60）
#[derive(Debug, Clone, Copy)]
pub struct DetectorParams {
    pub feature_count: i32,
    pub weak_th: f64,
    pub strong_th: f64,
}

/// 训练参数（角度单位度，范围限制 [-180,180]；缩放 [0.25,2]；step>0——由原生端终审）
#[derive(Debug, Clone, Copy)]
pub struct TrainParams {
    /// 参考图坐标系内的训练 ROI (x, y, w, h)，像素
    pub roi_xywh: (i32, i32, i32, i32),
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_step: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    /// None 用原生默认三参（128/30/60）
    pub detector: Option<DetectorParams>,
}

impl TrainParams {
    pub(crate) fn validate(&self) -> Result<(), ShapeError> {
        let (roi_x, roi_y, roi_width, roi_height) = self.roi_xywh;
        if [roi_x, roi_y, roi_width, roi_height]
            .iter()
            .any(|value| value.abs() > 4096)
        {
            return Err(ShapeError::InvalidInput(
                "训练 ROI 须为四项整数像素坐标".into(),
            ));
        }
        Ok(())
    }
}

/// 模型元数据（键名对齐 Python 侧 metadata 字典）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelMetadata {
    /// 参考图坐标系宽高（新容器模型记录原图尺寸；旧格式回退模板灰图尺寸）
    pub reference_width: i32,
    pub reference_height: i32,
    pub roi_xywh: (i32, i32, i32, i32),
    pub angle_min: f64,
    pub angle_max: f64,
    pub angle_step: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    /// 旧资产可能为零值（0 仍表示原生默认 128/30/60 语义）
    pub weak_th: f64,
    pub strong_th: f64,
    pub feature_count: i32,
}

struct ModelInner {
    /// 原生句柄；0=已关闭
    handle: u64,
}

/// 受控模型句柄：方法内部串行化（同句柄调用原生侧本就串行，此处保护
/// 关闭检查与调用的原子性）；跨线程共享安全；Drop 时尽力关闭
pub struct Model {
    library: Arc<ShapeLibrary>,
    inner: Mutex<ModelInner>,
}

impl Model {
    /// 从模型资产字节加载（两级容器格式优先，非容器按旧单级格式解码）
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ShapeError> {
        let library = ShapeLibrary::global()?;
        Self::from_library(&library, bytes)
    }

    /// 从指定库实例加载模型资产（多运行时部署/诊断入口用）
    pub fn from_library(library: &Arc<ShapeLibrary>, bytes: &[u8]) -> Result<Self, ShapeError> {
        if bytes.is_empty() || bytes.len() > MAX_MODEL_BYTES {
            return Err(ShapeError::InvalidInput(
                "模板模型字节非法或超过 32 MiB".into(),
            ));
        }
        let handle = library.abi().load(bytes)?;
        Ok(Self {
            library: Arc::clone(library),
            inner: Mutex::new(ModelInner { handle }),
        })
    }

    fn lock_inner(&self) -> MutexGuard<'_, ModelInner> {
        // 锁中毒只意味着持有者恐慌，句柄本身仍有效：恢复而非放弃
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn require_open(inner: &ModelInner) -> Result<u64, ShapeError> {
        if inner.handle == 0 {
            Err(ShapeError::InvalidInput("模板模型已关闭".into()))
        } else {
            Ok(inner.handle)
        }
    }

    /// 句柄锁下执行原生查询（诊断入口用；锁保护 open 检查与调用的原子性）
    pub(crate) fn with_live_handle<T>(&self, f: impl FnOnce(u64) -> T) -> Result<T, ShapeError> {
        let inner = self.lock_inner();
        let handle = Self::require_open(&inner)?;
        Ok(f(handle))
    }

    /// 原生 ABI 访问（诊断入口用）
    pub(crate) fn library(&self) -> &ShapeLibrary {
        &self.library
    }

    /// 查询训练配置回显与参考图尺寸
    pub fn metadata(&self) -> Result<ModelMetadata, ShapeError> {
        let inner = self.lock_inner();
        let handle = Self::require_open(&inner)?;
        let (config, width, height) = self.library.abi().metadata(handle)?;
        Ok(ModelMetadata {
            reference_width: width,
            reference_height: height,
            roi_xywh: (
                config.roi_x,
                config.roi_y,
                config.roi_width,
                config.roi_height,
            ),
            angle_min: config.angle_min,
            angle_max: config.angle_max,
            angle_step: config.angle_step,
            scale_min: config.scale_min,
            scale_max: config.scale_max,
            scale_step: config.scale_step,
            weak_th: config.weak_th,
            strong_th: config.strong_th,
            feature_count: config.feature_count,
        })
    }

    /// 单模型匹配：threshold 须为 (0,100] 内有限值；max_candidates 2..256
    pub fn match_(
        &self,
        image: BgrImage<'_>,
        threshold: f64,
        max_candidates: u32,
    ) -> Result<Vec<Candidate>, ShapeError> {
        validate_threshold(threshold)?;
        if !(2..=256).contains(&max_candidates) {
            return Err(ShapeError::InvalidInput(
                "候选数量须为 2–256，不能将多目标截成唯一匹配".into(),
            ));
        }
        let mut buffer = vec![CandidateAbi::default(); max_candidates as usize];
        let inner = self.lock_inner();
        let handle = Self::require_open(&inner)?;
        let count = self.library.abi().match_one(
            handle,
            image.data(),
            image.width(),
            image.height(),
            threshold,
            &mut buffer,
        )?;
        candidates_from_raw(&buffer, count as usize)
    }

    /// 显式关闭（幂等：已关闭为无操作）；之后的调用返回"模板模型已关闭"
    pub fn close(&self) -> Result<(), ShapeError> {
        let mut inner = self.lock_inner();
        if inner.handle != 0 {
            self.library.abi().close(inner.handle)?;
            inner.handle = 0;
        }
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let handle = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .handle;
        if handle != 0 {
            self.library.close_unchecked(handle);
        }
    }
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Model")
    }
}

/// 训练模板模型并导出资产字节（训练句柄即用即释）；走全局共享库实例
pub fn train_model(reference: BgrImage<'_>, params: &TrainParams) -> Result<Vec<u8>, ShapeError> {
    ShapeLibrary::global()?.train_model(reference, params)
}

/// 单帧批量匹配请求（threshold (0,100]、容量 2..256，非法值只影响该项）
pub struct BatchRequest<'a> {
    pub model: &'a Model,
    pub threshold: f64,
    pub max_candidates: u32,
}

/// 同一输入帧对多个模型匹配（1..=16 个），逐项隔离。
/// 外层 Err 对齐头文件合同：公共图像校验失败或批量编排级失败时整批失败，
/// 此时逐项结果不构成完整结果，仅在内层 Ok 中逐项给出。
pub fn match_batch(
    requests: &[BatchRequest<'_>],
    image: BgrImage<'_>,
) -> Result<Vec<Result<Vec<Candidate>, ShapeError>>, ShapeError> {
    if requests.is_empty() || requests.len() > MAX_BATCH_ITEMS {
        return Err(ShapeError::InvalidInput(
            "单帧批量匹配须包含 1–16 个模型".into(),
        ));
    }
    // 模型去重并按地址稳定排序加锁，防交叉死锁（对齐 Python 按 id 排序加锁）
    let mut unique: Vec<&Model> = Vec::with_capacity(requests.len());
    for request in requests {
        if !unique.iter().any(|m| std::ptr::eq(*m, request.model)) {
            unique.push(request.model);
        }
    }
    unique.sort_by_key(|m| *m as *const Model as usize);
    let guards: Vec<MutexGuard<'_, ModelInner>> =
        unique.iter().map(|m| m.lock_inner()).collect();

    let mut results: Vec<Result<Vec<Candidate>, ShapeError>> = requests
        .iter()
        .map(|_| Err(ShapeError::InvalidInput("模板匹配未执行".into())))
        .collect();
    let mut valid: Vec<ValidEntry> = Vec::new();
    let mut buffers: Vec<Vec<CandidateAbi>> = Vec::new();
    let mut library: Option<Arc<ShapeLibrary>> = None;
    for (index, request) in requests.iter().enumerate() {
        // 逐项入参校验：open 状态、阈值、容量、库一致性（失败只影响该项）
        let entry = (|| -> Result<(u64, f64, u32), ShapeError> {
            let guard = unique
                .iter()
                .position(|m| std::ptr::eq(*m, request.model))
                .map(|gi| &guards[gi])
                .ok_or_else(|| ShapeError::Output("批量模型锁缺失（内部错误）".into()))?;
            let handle = Model::require_open(guard)?;
            validate_threshold(request.threshold)?;
            if !(2..=256).contains(&request.max_candidates) {
                return Err(ShapeError::InvalidInput("候选数量须为 2–256".into()));
            }
            if let Some(first) = &library {
                if !Arc::ptr_eq(&request.model.library, first) {
                    return Err(ShapeError::InvalidInput(
                        "批量模型来自不同原生组件".into(),
                    ));
                }
            }
            Ok((handle, request.threshold, request.max_candidates))
        })();
        match entry {
            Ok((handle, threshold, capacity)) => {
                library.get_or_insert_with(|| Arc::clone(&request.model.library));
                buffers.push(vec![CandidateAbi::default(); capacity as usize]);
                valid.push(ValidEntry {
                    index,
                    handle,
                    threshold,
                    capacity,
                });
            }
            Err(error) => results[index] = Err(error),
        }
    }
    // 全部条目校验失败：不进原生，逐项 Err 返回（对齐 Python 行为）
    let Some(library) = library else {
        return Ok(results);
    };

    let mut items: Vec<BatchItemAbi> = valid
        .iter()
        .zip(buffers.iter_mut())
        .map(|(entry, buffer)| BatchItemAbi {
            handle: entry.handle,
            threshold: entry.threshold,
            candidates: buffer.as_mut_ptr(),
            capacity: entry.capacity,
            count: 0,
            status: 1,
            error: [0u8; 256],
        })
        .collect();
    // 整批调用：返回非 0（公共图像校验/编排级失败）→ 外层 Err
    library
        .abi()
        .match_batch(image.data(), image.width(), image.height(), &mut items)?;

    for ((entry, item), buffer) in valid.iter().zip(&items).zip(&buffers) {
        results[entry.index] = consume_batch_item(entry, item, buffer);
    }
    Ok(results)
}

/// 逐项消费：回显校验（句柄/阈值/容量/缓冲指针）→ 状态语义 → 候选数值校验
fn consume_batch_item(
    entry: &ValidEntry,
    item: &BatchItemAbi,
    buffer: &[CandidateAbi],
) -> Result<Vec<Candidate>, ShapeError> {
    if !matches!(item.status, 0..=2)
        || item.count > entry.capacity
        || item.handle != entry.handle
        || item.threshold != entry.threshold
        || item.capacity != entry.capacity
        || !std::ptr::eq(item.candidates, buffer.as_ptr())
    {
        return Err(ShapeError::Output(
            "原生批量输出状态或数量非法".into(),
        ));
    }
    if item.status != 0 {
        if item.count != 0 {
            return Err(ShapeError::Output("原生失败项返回了候选".into()));
        }
        return Err(ShapeError::native(item.status, &item.error));
    }
    candidates_from_raw(buffer, item.count as usize)
}

/// 通过入参校验的批量条目（index 为 requests 下标）
struct ValidEntry {
    index: usize,
    handle: u64,
    threshold: f64,
    capacity: u32,
}

fn validate_threshold(threshold: f64) -> Result<(), ShapeError> {
    if threshold.is_finite() && threshold > 0.0 && threshold <= 100.0 {
        Ok(())
    } else {
        Err(ShapeError::InvalidInput(
            "相似度阈值须大于 0 且不大于 100".into(),
        ))
    }
}

/// 候选数值合法性校验（对齐 Python _candidates）：全字段有限、score 0..100、
/// pose_id 非负、ambiguous 0/1、scale 正数、bbox 宽高非负
fn candidates_from_raw(
    buffer: &[CandidateAbi],
    count: usize,
) -> Result<Vec<Candidate>, ShapeError> {
    if count > buffer.len() {
        return Err(ShapeError::Output("原生候选输出数量超过缓冲区".into()));
    }
    let mut candidates = Vec::with_capacity(count);
    for raw in &buffer[..count] {
        let finite = raw.score.is_finite()
            && raw.angle.is_finite()
            && raw.scale.is_finite()
            && raw.center.iter().all(|v| v.is_finite())
            && raw.corners.iter().all(|v| v.is_finite())
            && raw.bbox.iter().all(|v| v.is_finite())
            && raw.transform.iter().all(|v| v.is_finite());
        if !finite
            || !(0.0..=100.0).contains(&raw.score)
            || raw.pose_id < 0
            || !matches!(raw.pose_ambiguous, 0 | 1)
            || raw.scale <= 0.0
            || raw.bbox[2] < 0.0
            || raw.bbox[3] < 0.0
        {
            return Err(ShapeError::Output("原生候选输出非法".into()));
        }
        candidates.push(Candidate {
            score: raw.score,
            pose_id: raw.pose_id,
            pose_ambiguous: raw.pose_ambiguous == 1,
            center: (raw.center[0], raw.center[1]),
            angle: raw.angle,
            scale: raw.scale,
            corners: std::array::from_fn(|i| (raw.corners[2 * i], raw.corners[2 * i + 1])),
            bbox: (raw.bbox[0], raw.bbox[1], raw.bbox[2], raw.bbox[3]),
            transform: raw.transform,
        });
    }
    Ok(candidates)
}
