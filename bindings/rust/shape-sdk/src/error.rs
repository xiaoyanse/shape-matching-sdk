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
