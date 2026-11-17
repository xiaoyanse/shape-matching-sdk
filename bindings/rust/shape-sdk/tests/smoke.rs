//! 冒烟：合成图训练→导出→加载→元数据→单/批量匹配→生命周期与错误路径。
//! 前提：仓内 dist/runtime 存在且通过 manifest 哈希链校验（运行时目录按
//! 编译期 crate 位置上溯仓根发现）。合成图画法复刻 tests/abi_chain_test.cpp
//! 的 synth_reference（320×320 黑底 + 白框 + 内部块）。

use shape_sdk::{
    match_batch, train_model, BatchRequest, BgrImage, Model, ShapeError, TrainParams,
};

const SIZE: usize = 320;

/// 纯程序构造 BGR 连续缓冲（矩形填充含首末像素，口径同 cv::rectangle 实心）
fn fill(data: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, value: u8) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            let base = (y * SIZE + x) * 3;
            data[base..base + 3].fill(value);
        }
    }
}

fn synth_reference() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    fill(&mut data, 42, 42, 47, 197, 255); // 左框条
    fill(&mut data, 42, 42, 197, 47, 255); // 上框条
    fill(&mut data, 42, 192, 197, 197, 255); // 下框条
    fill(&mut data, 192, 42, 197, 197, 255); // 右框条
    fill(&mut data, 65, 65, 94, 94, 255); // 内部块 1（白）
    fill(&mut data, 90, 120, 154, 154, 200); // 内部块 2（灰 200）
    fill(&mut data, 165, 80, 170, 179, 255); // 内部块 3（白）
    data
}

fn ranged_params() -> TrainParams {
    TrainParams {
        roi_xywh: (32, 32, 176, 176),
        angle_min: -20.0,
        angle_max: 20.0,
        angle_step: 8.0,
        scale_min: 0.8,
        scale_max: 1.2,
        scale_step: 0.2,
        detector: None,
    }
}

fn trained_model_bytes() -> Vec<u8> {
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    train_model(reference, &ranged_params()).expect("默认参数训练失败")
}

#[test]
fn full_chain_default_params() {
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    let bytes = train_model(reference, &ranged_params()).expect("训练失败");
    assert!(!bytes.is_empty() && bytes.len() <= 32 * 1024 * 1024);
    let model = Model::from_bytes(&bytes).expect("模型加载失败");

    // 元数据：参考尺寸=原图、角度/缩放回显、默认三参回填 128/30/60
    let metadata = model.metadata().expect("元数据查询失败");
    assert_eq!(metadata.reference_width, SIZE as i32);
    assert_eq!(metadata.reference_height, SIZE as i32);
    assert_eq!(metadata.roi_xywh, (32, 32, 176, 176));
    assert_eq!(metadata.angle_min, -20.0);
    assert_eq!(metadata.angle_max, 20.0);
    assert_eq!(metadata.angle_step, 8.0);
    assert_eq!(metadata.scale_min, 0.8);
    assert_eq!(metadata.scale_max, 1.2);
    assert_eq!(metadata.scale_step, 0.2);
    assert_eq!(metadata.feature_count, 128);
    assert_eq!(metadata.weak_th, 30.0);
    assert_eq!(metadata.strong_th, 60.0);

    // 同图匹配：≥1 候选，分数/中心界内
    let candidates = model
        .match_(reference, 60.0, 16)
        .expect("同图匹配失败");
    assert!(!candidates.is_empty(), "同图匹配应 ≥1 候选");
    for candidate in &candidates {
        assert!(
            (60.0..=100.0).contains(&candidate.score),
            "候选分数越界：{}",
            candidate.score
        );
        assert!(
            candidate.center.0 >= 0.0
                && candidate.center.0 < SIZE as f64
                && candidate.center.1 >= 0.0
                && candidate.center.1 < SIZE as f64,
            "候选中心越出图像：{:?}",
            candidate.center
        );
    }
}

#[test]
fn match_batch_two_models_same_frame() {
    let bytes = trained_model_bytes();
    let model_a = Model::from_bytes(&bytes).expect("模型 A 加载失败");
    let model_b = Model::from_bytes(&bytes).expect("模型 B 加载失败");
    let pixels = synth_reference();
    let frame = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成帧构造失败");
    let requests = [
        BatchRequest {
            model: &model_a,
            threshold: 60.0,
            max_candidates: 16,
        },
        BatchRequest {
            model: &model_b,
            threshold: 80.0,
            max_candidates: 4,
        },
    ];
    let results = match_batch(&requests, frame).expect("批量编排失败");
    assert_eq!(results.len(), 2);
    for (index, result) in results.iter().enumerate() {
        let candidates = result.as_ref().unwrap_or_else(|error| {
            panic!("批量项 {index} 应成功：{error}");
        });
        assert!(!candidates.is_empty(), "批量项 {index} 应 ≥1 候选");
    }
}

#[test]
fn batch_per_item_isolation() {
    let bytes = trained_model_bytes();
    let healthy = Model::from_bytes(&bytes).expect("健康模型加载失败");
    let closed = Model::from_bytes(&bytes).expect("待关闭模型加载失败");
    closed.close().expect("显式关闭失败");
    let pixels = synth_reference();
    let frame = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成帧构造失败");
    let requests = [
        BatchRequest {
            model: &healthy,
            threshold: 60.0,
            max_candidates: 16,
        },
        BatchRequest {
            model: &closed,
            threshold: 60.0,
            max_candidates: 16,
        },
        BatchRequest {
            model: &healthy,
            threshold: f64::NAN,
            max_candidates: 16,
        },
        BatchRequest {
            model: &healthy,
            threshold: 60.0,
            max_candidates: 1,
        },
    ];
    let results = match_batch(&requests, frame).expect("混合批次编排失败");
    assert_eq!(results.len(), 4);
    // 项 0：健康项成功且不受他项影响
    let candidates = results[0].as_ref().expect("健康项应成功");
    assert!(!candidates.is_empty());
    // 项 1：已关闭模型逐项失败
    assert!(
        matches!(&results[1], Err(e) if e.message() == "模板模型已关闭"),
        "已关闭项应逐项失败，实际：{:?}",
        results[1]
    );
    // 项 2：NaN 阈值逐项失败
    assert!(
        matches!(&results[2], Err(ShapeError::InvalidInput(m)) if m.contains("阈值")),
        "NaN 阈值项应逐项失败，实际：{:?}",
        results[2]
    );
    // 项 3：容量 1 逐项失败
    assert!(
        matches!(&results[3], Err(ShapeError::InvalidInput(m)) if m.contains("2–256")),
        "容量 1 项应逐项失败，实际：{:?}",
        results[3]
    );
}

#[test]
fn error_paths() {
    // 坏字节：原生 load 返回 1（非容器且非旧单级格式）
    let garbage = b"definitely not a model payload".as_slice();
    let error = Model::from_bytes(garbage).expect_err("坏字节应加载失败");
    assert_eq!(error.native_status(), Some(1), "实际：{error}");
    // 空字节：绑定层拒绝
    let error = Model::from_bytes(&[]).expect_err("空字节应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));
    // 超预算字节：绑定层拒绝
    let huge = vec![0u8; 32 * 1024 * 1024 + 1];
    let error = Model::from_bytes(&huge).expect_err("超预算字节应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));

    let bytes = trained_model_bytes();
    let model = Model::from_bytes(&bytes).expect("模型加载失败");
    let pixels = synth_reference();
    let frame = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成帧构造失败");
    // NaN 阈值：绑定层拒绝（不进原生）
    let error = model
        .match_(frame, f64::NAN, 16)
        .expect_err("NaN 阈值应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(m) if m.contains("阈值")));
    // 容量 1：绑定层拒绝
    let error = model.match_(frame, 60.0, 1).expect_err("容量 1 应拒绝");
    assert!(
        matches!(&error, ShapeError::InvalidInput(m) if m.contains("2–256")),
        "实际：{error}"
    );
    // 图像合同：长度不符 / 宽高越界
    let error = BgrImage::new(100, 100, &pixels).expect_err("长度不符应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));
    let error =
        BgrImage::new(10, 10, &[0u8; 300]).expect_err("宽高越界应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));
    // 批量条数越界：外层拒绝
    let error = match_batch(&[], frame).expect_err("空批量应外层拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(m) if m.contains("1–16")));
}

#[test]
fn close_and_drop_are_idempotent() {
    let bytes = trained_model_bytes();
    // 显式关闭幂等：二次 close 为无操作；关闭后调用返回"已关闭"；Drop 再触发也无害
    let model = Model::from_bytes(&bytes).expect("模型加载失败");
    model.close().expect("首次关闭失败");
    model.close().expect("重复关闭应幂等成功");
    let error = model.metadata().expect_err("关闭后元数据应失败");
    assert!(matches!(error, ShapeError::InvalidInput(m) if m == "模板模型已关闭"));
    drop(model);
    // 未显式关闭直接 Drop（close 幂等由原生保证，进程收尾干净）
    let model = Model::from_bytes(&bytes).expect("第二模型加载失败");
    drop(model);
}
