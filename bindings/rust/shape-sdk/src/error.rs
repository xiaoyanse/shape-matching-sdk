//! 错误语义：绑定层校验 / 原生 status 1|2 / 原生输出非法 / 装载失败。

use std::fmt;

/// 形状匹配错误。`Native.status` 对齐 C ABI 错误分层：
/// 1=可报告失败（参数非法、资源超限、解码失败等），2=未分类原生失败。
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// 绑定层入参校验失败（图像/阈值/容量/ROI/模型字节等）
    InvalidInput(String),
    /// 原生调用返回非 0：status 码 + 库侧错误文本（尽力诊断，非程序化分类）
    Native { status: i32, message: String },
    /// 原生输出数值非法（候选越界/非有限/批量回显不符等）
    Output(String),
    /// 原生组件装载或 manifest 校验失败
    Library(String),
}

impl ShapeError {
    /// 由原生返回码 + NUL 结尾错误缓冲构造（文本按 UTF-8 有损替换解码）
    pub(crate) fn native(status: i32, buffer: &[u8]) -> Self {
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        let message = String::from_utf8_lossy(&buffer[..end]).into_owned();
        ShapeError::Native { status, message }
    }

    /// 返回码非 0 转 Err；错误文本透传库侧诊断信息
    pub(crate) fn check(status: i32, buffer: &[u8]) -> Result<(), Self> {
        if status == 0 {
            Ok(())
        } else {
            Err(Self::native(status, buffer))
        }
    }

    /// 原生 status 码（仅 `Native` 变体有，0/1/2 之外的取值视为未分类）
    pub fn native_status(&self) -> Option<i32> {
        match self {
            ShapeError::Native { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// 错误文本（绑定层文案或库侧诊断）
    pub fn message(&self) -> &str {
        match self {
            ShapeError::InvalidInput(m) | ShapeError::Output(m) | ShapeError::Library(m) => m,
            ShapeError::Native { message, .. } => message,
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for ShapeError {}

/// 稳定错误码镜像（vf_shape_error_code）：与错误文本并存，供程序化分类。
/// 数值上下文经 `ShapeLibrary::last_error_info` 按线程查询。
pub mod error_code {
    pub const OK: i32 = 0;
    pub const INVALID_IMAGE: i32 = 1;
    pub const INVALID_ROI: i32 = 2;
    pub const INVALID_ANGLE_RANGE: i32 = 3;
    pub const INVALID_SCALE_RANGE: i32 = 4;
    pub const POSE_BUDGET: i32 = 5;
    pub const PIXEL_BUDGET: i32 = 6;
    pub const CANVAS_BUDGET: i32 = 7;
    pub const DETECTOR_PARAMS: i32 = 8;
    pub const THRESHOLD_INVALID: i32 = 9;
    pub const CAPACITY_INVALID: i32 = 10;
    pub const HANDLE_INVALID: i32 = 11;
    pub const HANDLE_BUDGET: i32 = 12;
    pub const MODEL_FORMAT: i32 = 13;
    pub const MODEL_SIZE: i32 = 14;
    pub const MODEL_TAMPER: i32 = 15;
    pub const SEARCH_BUDGET: i32 = 16;
    pub const INTERNAL: i32 = 99;

    /// 码名（未知码返回 UNKNOWN）
    pub fn name(code: i32) -> &'static str {
        match code {
            OK => "OK",
            INVALID_IMAGE => "INVALID_IMAGE",
            INVALID_ROI => "INVALID_ROI",
            INVALID_ANGLE_RANGE => "INVALID_ANGLE_RANGE",
            INVALID_SCALE_RANGE => "INVALID_SCALE_RANGE",
            POSE_BUDGET => "POSE_BUDGET",
            PIXEL_BUDGET => "PIXEL_BUDGET",
            CANVAS_BUDGET => "CANVAS_BUDGET",
            DETECTOR_PARAMS => "DETECTOR_PARAMS",
            THRESHOLD_INVALID => "THRESHOLD_INVALID",
            CAPACITY_INVALID => "CAPACITY_INVALID",
            HANDLE_INVALID => "HANDLE_INVALID",
            HANDLE_BUDGET => "HANDLE_BUDGET",
            MODEL_FORMAT => "MODEL_FORMAT",
            MODEL_SIZE => "MODEL_SIZE",
            MODEL_TAMPER => "MODEL_TAMPER",
            SEARCH_BUDGET => "SEARCH_BUDGET",
            INTERNAL => "INTERNAL",
            _ => "UNKNOWN",
        }
    }
}

/// 错误码人话文案（与 Python 绑定 diagnostics.error_message 同文案口径）；
/// params 为 `last_error_info` 返回的数值上下文
pub fn error_message(code: i32, params: &[f64]) -> String {
    let mut text = match code {
        error_code::OK => "成功".to_string(),
        error_code::INVALID_IMAGE => "图像不合法：须为连续 uint8 BGR，宽高均在 24–4096 像素".into(),
        error_code::INVALID_ROI => "训练 ROI 不合法：≥24px 且须完整落在参考图内".into(),
        error_code::INVALID_ANGLE_RANGE => {
            "角度范围或步长不合法：范围 [-180,180]、步长>0；步长过细时参数依次为当前步长、档数上限（350）、建议步长".into()
        }
        error_code::INVALID_SCALE_RANGE => {
            "缩放范围或步长不合法：范围 [0.25,2]、步长>0；步长过细时参数依次为当前步长、档数上限（350）、建议步长".into()
        }
        error_code::POSE_BUDGET => "姿态总数超预算：角度×缩放档位数超上限（参数：档位数,上限）".into(),
        error_code::PIXEL_BUDGET => "训练像素总量超预算（参数：用量,上限）".into(),
        error_code::CANVAS_BUDGET => {
            "训练画布超尺寸预算：缩放后模板外扩画布单边须 ≤4096（参数：画布边,上限）".into()
        }
        error_code::DETECTOR_PARAMS => "特征筛选参数不合法：0<weak≤strong≤1020，特征数 16–512".into(),
        error_code::THRESHOLD_INVALID => "相似度阈值须为 (0,100] 内的有限值".into(),
        error_code::CAPACITY_INVALID => "候选容量须为 2–256".into(),
        error_code::HANDLE_INVALID => "模型句柄已关闭或未知".into(),
        error_code::HANDLE_BUDGET => {
            "活跃句柄数超上限 128，请先释放不再使用的模型（参数：当前数,上限）".into()
        }
        error_code::MODEL_FORMAT => "模型资产格式无法识别".into(),
        error_code::MODEL_SIZE => "模型资产尺寸超预算 ≤32MiB（参数：字节数,上限）".into(),
        error_code::MODEL_TAMPER => "模型资产校验失败：内容与容器合同不符或被篡改".into(),
        error_code::SEARCH_BUDGET => "匹配搜索预算超限：姿态数×搜索块数过多（参数：用量,上限）".into(),
        error_code::INTERNAL => "未分类内部错误".into(),
        other => return format!("未知错误码 {other}"),
    };
    if !params.is_empty() {
        let shown: Vec<String> = params.iter().map(|v| format!("{v:.6}")).collect();
        text.push_str(&format!("（参数：{}）", shown.join(", ")));
    }
    text
}
