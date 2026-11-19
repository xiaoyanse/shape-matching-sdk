//! 诊断三件套绑定合同（Rust 版）：预算查询与训练拒绝边界一致、失败调用的
//! 错误码+参数落账、最近匹配 trace。运行时取本升级批独立构建 build/t2a
//! （visionforge_shape.dll + manifest，由构建脚本装配），与既有测试的
//! dist/runtime 互不干扰。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shape_sdk::{
    error_code, error_message, BgrImage, Model, ShapeLibrary, TrainParams,
};

const SIZE: usize = 320;
const BIG: usize = 1600;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate 应位于仓内 bindings/rust/shape-sdk")
        .to_path_buf()
}

/// 诊断接口只存在于本升级批制品（build/t2a）；全部调用须绑定该实例，
/// 不得走全局单例（其默认发现 dist/runtime 旧制品，无新导出）
fn task_library() -> Arc<ShapeLibrary> {
    let dir = repo_root().join("build").join("t2a").join("runtime");
    let library = ShapeLibrary::open_at(&dir).expect("诊断测试须先装配 build/t2a/runtime");
    Arc::new(library)
}

/// 纯程序构造 BGR 连续缓冲（口径同 C++ synth_reference/place_targets）
fn fill(data: &mut [u8], size: usize, x0: usize, y0: usize, x1: usize, y1: usize, value: u8) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            let base = (y * size + x) * 3;
            data[base..base + 3].fill(value);
        }
    }
}

fn synth_reference() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    fill(&mut data, SIZE, 42, 42, 47, 197, 255);
    fill(&mut data, SIZE, 42, 42, 197, 47, 255);
    fill(&mut data, SIZE, 42, 192, 197, 197, 255);
    fill(&mut data, SIZE, 192, 42, 197, 197, 255);
    fill(&mut data, SIZE, 65, 65, 94, 94, 255);
    fill(&mut data, SIZE, 90, 120, 154, 154, 200);
    fill(&mut data, SIZE, 165, 80, 170, 179, 255);
    data
}

fn three_target_frame(reference: &[u8]) -> Vec<u8> {
    let mut frame = vec![0u8; BIG * BIG * 3];
    for (x, y) in [(100usize, 100usize), (800, 100), (1350, 1350)] {
        for row in 0..176 {
            let src = ((32 + row) * SIZE + 32) * 3;
            let dst = ((y + row) * BIG + x) * 3;
            let bytes = 176 * 3;
            frame[dst..dst + bytes].copy_from_slice(&reference[src..src + bytes]);
        }
    }
    frame
}

fn view(data: &[u8], size: usize) -> BgrImage<'_> {
    BgrImage::new(size as i32, size as i32, data).expect("合成图构造失败")
}

fn params(roi: (i32, i32, i32, i32), angles: (f64, f64, f64), scales: (f64, f64, f64),
          detector: bool) -> TrainParams {
    TrainParams {
        roi_xywh: roi,
        angle_min: angles.0,
        angle_max: angles.1,
        angle_step: angles.2,
        scale_min: scales.0,
        scale_max: scales.1,
        scale_step: scales.2,
        detector: detector.then_some(shape_sdk::DetectorParams {
            feature_count: 32,
            weak_th: 20.0,
            strong_th: 40.0,
        }),
    }
}

#[test]
fn budget_boundary_matches_train() {
    let library = task_library();
    let pixels = synth_reference();
    let reference = view(&pixels, SIZE);
    let big = vec![0u8; BIG * BIG * 3];
    // 姿态账本：350 恰好压线 ↔ 训练通过；352 ↔ 训练拒绝
    let ok = library
        .estimate_training(&params((32, 32, 176, 176), (-87.0, 87.0, 1.0), (0.5, 1.0, 0.5), true),
                           SIZE as i32, SIZE as i32)
        .expect("压线配置应产报告");
    assert_eq!(ok.pose_used, 350);
    assert_eq!(ok.over_budget, error_code::OK);
    let trained = library
        .train_model(reference, &params((32, 32, 176, 176), (-87.0, 87.0, 1.0),
                                        (0.5, 1.0, 0.5), true))
        .expect("压线配置训练应通过");
    assert!(!trained.is_empty());
    let over = library
        .estimate_training(&params((32, 32, 176, 176), (-87.5, 87.5, 1.0), (0.5, 1.0, 0.5), true),
                           SIZE as i32, SIZE as i32)
        .expect("超限配置仍应产报告");
    assert_eq!(over.pose_used, 352);
    assert_eq!(over.over_budget, error_code::POSE_BUDGET);
    let error = library
        .train_model(reference, &params((32, 32, 176, 176), (-87.5, 87.5, 1.0),
                                        (0.5, 1.0, 0.5), true))
        .expect_err("超限配置训练应拒绝");
    assert_eq!(error.native_status(), Some(1));
    let info = library.last_error_info().expect("last_error_info 查询失败");
    assert_eq!(info.code, error_code::POSE_BUDGET, "产品档数超限应落 POSE_BUDGET");
    assert_eq!(info.param_count, 2);
    assert_eq!(info.params[0], 352.0, "参数[0]应为当前档位数");
    assert_eq!(info.params[1], 350.0, "参数[1]应为档数上限");
    // 像素账本：21 档缩放累计 ↔ 训练拒绝；查询不建句柄零成本
    let pixel_over = library
        .estimate_training(
            &params((450, 450, 700, 700), (-8.0, 8.0, 8.0), (1.0, 2.0, 0.05), true),
            BIG as i32, BIG as i32,
        )
        .expect("像素超限配置应产报告");
    assert!(pixel_over.pixel_over && pixel_over.over_budget == error_code::PIXEL_BUDGET);
    assert!(!pixel_over.canvas_over, "画布应未超限，超限因子唯一");
    let error = library
        .train_model(view(&big, BIG),
                     &params((450, 450, 700, 700), (-8.0, 8.0, 8.0), (1.0, 2.0, 0.05), true))
        .expect_err("像素超限训练应拒绝");
    assert!(error.message().contains("pixel budget"), "实际：{error}");
}

#[test]
fn angle_over_fine_records_code_and_params() {
    let library = task_library();
    let pixels = synth_reference();
    let reference = view(&pixels, SIZE);
    let error = library
        .train_model(reference, &params((32, 32, 176, 176), (-20.0, 20.0, 0.05),
                                        (1.0, 1.0, 1.0), true))
        .expect_err("角度步过细应拒绝");
    assert!(error.message().contains("pose budget exceeded"), "实际：{error}");
    let info = library.last_error_info().expect("last_error_info 查询失败");
    assert_eq!(info.code, error_code::INVALID_ANGLE_RANGE);
    assert_eq!(info.param_count, 3);
    assert_eq!(info.params[0], 0.05, "参数[0]应为当前步长");
    assert_eq!(info.params[1], 350.0, "参数[1]应为档数上限");
    assert!((info.params[2] - 40.0 / 349.0).abs() < 1e-9, "参数[2]应为建议步长");
    let text = error_message(info.code, &info.params[..info.param_count]);
    assert!(text.contains("建议步长"), "人话文案应含建议步长提示：{text}");
}

#[test]
fn match_trace_three_targets() {
    let library = task_library();
    let pixels = synth_reference();
    let reference = view(&pixels, SIZE);
    let bytes = library
        .train_model(
            reference,
            &params((32, 32, 176, 176), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), true),
        )
        .expect("trace 模板训练失败");
    let model = Model::from_library(&library, &bytes).expect("trace 模板加载失败");
    let frame = three_target_frame(&pixels);
    let hits = model
        .match_(view(&frame, BIG), 60.0, 16)
        .expect("三目标匹配失败");
    assert!(hits.len() >= 3, "三目标召回 {} 应 ≥3", hits.len());
    let trace = model.last_match_info().expect("trace 查询失败");
    assert!(trace.windows_searched >= 2, "三目标场景精修窗数应 ≥2");
    assert!(trace.windows_searched <= 8);
    assert!(trace.coarse_candidates >= trace.windows_searched);
    assert_eq!(trace.final_candidates as usize, hits.len());
    for n in 0..(trace.windows_searched as usize).min(8) {
        let score = trace.window_best_scores[n];
        assert!(score > 0.0 && score <= 100.0, "窗 {n} 最佳分越界：{score}");
    }
    assert_eq!(trace.coarse_ms, 0.0, "常规构建耗时字段应为 0");
    let again = model.last_match_info().expect("二次 trace 查询失败");
    assert_eq!(trace, again, "二次调用不应清除快照");
}
