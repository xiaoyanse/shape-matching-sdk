//! ABI 链核心组（Rust 版）：移植 tests/abi_chain_test.cpp 与 scripts/test_main_chain.py
//! 的核心断言组——全链/元数据往返/detector 往返/批量混合/生命周期。合成图画法复刻
//! smoke.rs（同 C++ synth_reference：320×320 黑底 + 白框 + 内部块）。
//!
//! 与 C++/Python 侧的口径差异：Rust 公共 API 不暴露已加载句柄的重导出（无
//! per-handle export），detector 往返组以「导出容器首段字节解析 + 首段单级加载
//! 回填 + 两级容器加载回填」三路对齐替代 C++ 的 load→export→load 二次往返链。

use shape_sdk::{
    match_batch, train_model, BatchRequest, BgrImage, Model, ModelMetadata, ShapeError,
    TrainParams,
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

fn reference_view(pixels: &[u8]) -> BgrImage<'_> {
    BgrImage::new(SIZE as i32, SIZE as i32, pixels).expect("合成图构造失败")
}

/// 多位姿范围参数（对齐 C++ a/b/g 组 ranged 配置）
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

/// 自定义 detector 三参 (32,20,40) 单位姿配置（对齐 C++ c/g 组 custom 配置）
fn custom_params() -> TrainParams {
    TrainParams {
        roi_xywh: (32, 32, 176, 176),
        angle_min: 0.0,
        angle_max: 0.0,
        angle_step: 1.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: Some(shape_sdk::DetectorParams {
            feature_count: 32,
            weak_th: 20.0,
            strong_th: 40.0,
        }),
    }
}

fn read_u64_le(bytes: &[u8], at: usize) -> u64 {
    let raw: [u8; 8] = bytes[at..at + 8].try_into().expect("u64 读取越界");
    u64::from_le_bytes(raw)
}

/// 容器首精修段提取（布局对齐 abi_chain_support.h：`VF2S3` 魔数 + u64 段数 n +
/// (n+3)×u64 长度表 + 数据连续块；段 0..n-1=精修、n=粗、n+1=模板 PNG、n+2=meta）
fn first_fine_segment(container: &[u8]) -> Vec<u8> {
    assert!(
        container.len() >= 13 && container.starts_with(b"VF2S3"),
        "容器 magic/长度不符"
    );
    let count = read_u64_le(container, 5) as usize;
    assert!((1..=64).contains(&count), "容器段数非法：{count}");
    let start = 13 + 8 * (count + 3);
    let length = read_u64_le(container, 13) as usize;
    assert!(start + length <= container.len(), "首段数据越界");
    container[start..start + length].to_vec()
}

/// 单级段 detector 三参解析（段内布局：签名 NUL + 1 + 12（schema/宽/高）+ 64
/// （训练配置）之后为 `<Idd`：u32 feature_count + f64 weak + f64 strong，小端）
fn segment_detector(segment: &[u8]) -> (u32, f64, f64) {
    let zero = segment.iter().position(|&b| b == 0).expect("段签名缺少 NUL 结尾");
    let offset = zero + 1 + 12 + 64;
    assert!(offset + 20 <= segment.len(), "段过短，detector 参数越界");
    let features: [u8; 4] = segment[offset..offset + 4].try_into().expect("u32 越界");
    let weak: [u8; 8] = segment[offset + 4..offset + 12].try_into().expect("f64 越界");
    let strong: [u8; 8] = segment[offset + 12..offset + 20].try_into().expect("f64 越界");
    (
        u32::from_le_bytes(features),
        f64::from_le_bytes(weak),
        f64::from_le_bytes(strong),
    )
}

/// metadata 三参（feature_count, weak_th, strong_th）
fn detector_of(metadata: &ModelMetadata) -> (u32, f64, f64) {
    (
        metadata.feature_count as u32,
        metadata.weak_th,
        metadata.strong_th,
    )
}

/// a 组：全链——训练→导出→加载→metadata（默认参回填 128/30/60）→同图匹配→close
#[test]
fn group_full_chain() {
    let pixels = synth_reference();
    let reference = reference_view(&pixels);
    let bytes = train_model(reference, &ranged_params()).expect("默认参数训练失败");
    assert!(!bytes.is_empty() && bytes.len() <= 32 * 1024 * 1024);
    // 导出侧：默认参首段序列化为实际三参（非零占位）
    assert_eq!(
        segment_detector(&first_fine_segment(&bytes)),
        (128, 30.0, 60.0),
        "默认参首段序列化漂移"
    );

    let model = Model::from_bytes(&bytes).expect("模型加载失败");
    let metadata = model.metadata().expect("元数据查询失败");
    assert_eq!(
        (metadata.reference_width, metadata.reference_height),
        (SIZE as i32, SIZE as i32),
        "参考尺寸非法"
    );
    assert_eq!(metadata.roi_xywh, (32, 32, 176, 176), "ROI 回显失真");
    assert_eq!(detector_of(&metadata), (128, 30.0, 60.0), "默认参 metadata 三值未物化");

    let hits = model.match_(reference, 60.0, 16).expect("同图匹配失败");
    assert!(!hits.is_empty(), "同图匹配命中 {} 应 ≥1", hits.len());
    for candidate in &hits {
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
    let best = hits
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .expect("候选非空已断言");
    assert!(
        (best.center.0 - 120.0).abs() < 40.0 && (best.center.1 - 120.0).abs() < 40.0,
        "命中中心偏移：{:?} 应近 ROI 中心 (120,120)",
        best.center
    );
    model.close().expect("全链收尾关闭失败");
}

/// b 组：容器往返元数据不变量（角度/缩放逐字段相等 + 参考尺寸=原图）
#[test]
fn group_metadata_roundtrip() {
    let pixels = synth_reference();
    let reference = reference_view(&pixels);
    let params = ranged_params();
    let bytes = train_model(reference, &params).expect("往返训练失败");
    let model = Model::from_bytes(&bytes).expect("往返加载失败");
    let metadata = model.metadata().expect("往返元数据失败");
    assert_eq!(metadata.angle_min, params.angle_min, "往返后 angle_min 漂移");
    assert_eq!(metadata.angle_max, params.angle_max, "往返后 angle_max 漂移");
    assert_eq!(metadata.angle_step, params.angle_step, "往返后 angle_step 漂移");
    assert_eq!(metadata.scale_min, params.scale_min, "往返后 scale_min 漂移");
    assert_eq!(metadata.scale_max, params.scale_max, "往返后 scale_max 漂移");
    assert_eq!(metadata.scale_step, params.scale_step, "往返后 scale_step 漂移");
    assert_eq!(
        (metadata.reference_width, metadata.reference_height),
        (SIZE as i32, SIZE as i32),
        "往返后参考尺寸应为原图"
    );
}

/// c 组：自定义参 (32,20,40) detector 三路对齐——导出容器首段字节解析、
/// 首段单级加载 metadata 回填、两级容器加载 metadata 回填，逐次相等
#[test]
fn group_detector_roundtrip() {
    let pixels = synth_reference();
    let reference = reference_view(&pixels);
    let container = train_model(reference, &custom_params()).expect("自定义参训练失败");
    let expected = (32u32, 20.0f64, 40.0f64);

    let first = segment_detector(&first_fine_segment(&container));
    assert_eq!(first, expected, "首次导出 detector 参数漂移：{first:?} 应为 (32,20,40)");

    let single = Model::from_bytes(&first_fine_segment(&container)).expect("首段单级加载失败");
    assert_eq!(
        detector_of(&single.metadata().expect("单级元数据失败")),
        expected,
        "单级加载 metadata 未回填"
    );

    let two_level = Model::from_bytes(&container).expect("两级容器加载失败");
    assert_eq!(
        detector_of(&two_level.metadata().expect("容器元数据失败")),
        expected,
        "两级容器 metadata 三字段漂移"
    );
}

/// g 组：批量混合——默认两级 + 默认参单级 + 自定义参单级同帧全通；坏容量项隔离
#[test]
fn group_batch_mix() {
    let pixels = synth_reference();
    let frame = reference_view(&pixels);
    let default_container = train_model(frame, &ranged_params()).expect("批量两级训练失败");
    let custom_container = train_model(frame, &custom_params()).expect("批量自定义参训练失败");
    let two_level = Model::from_bytes(&default_container).expect("两级模型加载失败");
    let single_default =
        Model::from_bytes(&first_fine_segment(&default_container)).expect("默认参单级加载失败");
    let single_custom =
        Model::from_bytes(&first_fine_segment(&custom_container)).expect("自定义参单级加载失败");

    let requests = [
        BatchRequest {
            model: &two_level,
            threshold: 60.0,
            max_candidates: 16,
        },
        BatchRequest {
            model: &single_default,
            threshold: 60.0,
            max_candidates: 16,
        },
        BatchRequest {
            model: &single_custom,
            threshold: 60.0,
            max_candidates: 16,
        },
    ];
    let results = match_batch(&requests, frame).expect("批量编排失败");
    for (index, result) in results.iter().enumerate() {
        let candidates = result.as_ref().unwrap_or_else(|error| {
            panic!("批量项 {index} 失败：{error}");
        });
        assert!(!candidates.is_empty(), "批量项 {index} 命中 {} 应 ≥1", candidates.len());
    }

    // 坏容量项隔离：容量 1 项逐项失败，同批健康项不受影响
    let mixed = [
        BatchRequest {
            model: &single_default,
            threshold: 60.0,
            max_candidates: 1,
        },
        BatchRequest {
            model: &single_default,
            threshold: 60.0,
            max_candidates: 16,
        },
    ];
    let isolated = match_batch(&mixed, frame).expect("隔离批次编排失败");
    assert!(
        matches!(&isolated[0], Err(ShapeError::InvalidInput(m)) if m.contains("2–256")),
        "容量 1 项应独立失败，实际：{:?}",
        isolated[0]
    );
    let healthy = isolated[1].as_ref().expect("容量越界项不应影响成功项");
    assert!(!healthy.is_empty(), "隔离后健康项应 ≥1 候选");
}

/// h 组：生命周期——坏 bytes 拒绝、NaN 阈值拒绝、close 幂等与关闭后受控失败
#[test]
fn group_lifecycle() {
    // 坏 bytes：原生 load 拒绝（status 1，非容器且非旧单级格式）
    let garbage = b"definitely not a model payload".as_slice();
    let error = Model::from_bytes(garbage).expect_err("坏字节应加载失败");
    assert_eq!(error.native_status(), Some(1), "实际：{error}");

    let pixels = synth_reference();
    let frame = reference_view(&pixels);
    let bytes = train_model(frame, &custom_params()).expect("生命周期训练失败");
    let model = Model::from_bytes(&bytes).expect("模型加载失败");
    // NaN 阈值：绑定层拒绝（不进原生）
    let error = model
        .match_(frame, f64::NAN, 16)
        .expect_err("NaN 阈值应拒绝");
    assert!(
        matches!(error, ShapeError::InvalidInput(ref m) if m.contains("阈值")),
        "实际：{error}"
    );
    // close 幂等 + 关闭后调用受控失败；Drop 再触发亦无害
    model.close().expect("首次关闭失败");
    model.close().expect("重复关闭应幂等成功");
    let error = model.metadata().expect_err("关闭后元数据应失败");
    assert!(
        matches!(error, ShapeError::InvalidInput(ref m) if m == "模板模型已关闭"),
        "实际：{error}"
    );
    drop(model);
}
