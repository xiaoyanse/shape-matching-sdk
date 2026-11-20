//! VisionForge 形状模板匹配 SDK 官方 Rust 绑定（ABI v3，libloading 运行时装载）。
//!
//! 运行时定位与 Python 绑定同契约：优先 `SHAPE_SDK_RUNTIME` 环境变量（制品目录，
//! 含原生库与 runtime-manifest.json），默认仓内 `dist/runtime`（按编译期 crate
//! 位置上溯仓根，前提是 crate 位于仓内 bindings/rust/shape-sdk；外部集成请设
//! 环境变量）。装载即校验 manifest 哈希链并协商 `vf_shape_abi_version()==3`。
//!
//! 模块划分：`error`（错误语义+错误码人话）｜`ffi`（全部 unsafe：裸声明与指针
//! 调用）｜`library`（装载/校验/协商/全局缓存）｜`model`（安全层：训练/加载/
//! 匹配/批量）｜`polygon`（多边形训练/多边形元数据）｜`suggest`（轮廓建议）｜
//! `diagnostics`（预算查询/最近错误/最近匹配诊断）｜`audit`（健康体检/区分度
//! 审计）。

mod audit;
mod diagnostics;
mod error;
mod ffi;
mod library;
mod model;
mod polygon;
mod suggest;

pub use audit::{audit_confidence, audit_flag, health_verdict, AuditReport, HealthReport};
pub use diagnostics::{BudgetReport, LastErrorInfo, MatchTrace, TrainingShape};
pub use error::{error_code, error_message, ShapeError};
pub use library::ShapeLibrary;
pub use model::{
    BatchRequest, BgrImage, Candidate, DetectorParams, Model, ModelMetadata, TrainParams,
    match_batch, train_model,
};
pub use polygon::{PolygonMetadata, PolygonTrainParams, PolygonVertex, train_polygon_model};
pub use suggest::{SuggestParams, suggest_contours, suggest_contours_ex};
