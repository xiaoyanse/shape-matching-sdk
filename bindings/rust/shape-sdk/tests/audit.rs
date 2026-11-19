//! 体检与审计绑定合同（Rust 版）：health 分级与自匹配、audit 分布单调与推荐
//! 区间数学（手工算对照）、INSEPARABLE 噪声构造、NO_NEG 边界。运行时取本升级
//! 批独立构建 build/t2b（visionforge_shape.dll + manifest，由构建脚本装配）。
//! fact 语义天然正负组：帧 1/3/4 有锚=正、帧 2 无锚=负。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shape_sdk::{
    audit_confidence, audit_flag, health_verdict, train_model, AuditReport, BgrImage,
    HealthReport, Model, ShapeError, ShapeLibrary, TrainParams,
};

const SIZE: usize = 480;
const ROI: (i32, i32, i32, i32) = (100, 100, 176, 176);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate 应位于仓内 bindings/rust/shape-sdk")
        .to_path_buf()
}

/// 体检/审计接口只存在于本升级批制品（build/t2b）；全部调用须绑定该实例，
/// 不得走全局单例（其默认发现 dist/runtime 旧制品，无新导出）
fn task_library() -> Arc<ShapeLibrary> {
    let dir = repo_root().join("build").join("t2b").join("runtime");
    let library = ShapeLibrary::open_at(&dir).expect("体检/审计测试须先装配 build/t2b/runtime");
    Arc::new(library)
}

fn fill(data: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, value: u8) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            let base = (y * SIZE + x) * 3;
            data[base..base + 3].fill(value);
        }
    }
}

/// 480×480 黑底：白色方框 (100..219)² 与内部灰块（形状与 ROI 边界留黑边距）
fn synth_reference() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    fill(&mut data, 100, 100, 219, 219, 255);
    fill(&mut data, 150, 150, 169, 169, 90);
    data
}

/// 模板 patch = 参考图 ROI 裁剪（176×176）
fn roi_patch(reference: &[u8]) -> Vec<u8> {
    let (x, y, w, h) = (ROI.0 as usize, ROI.1 as usize, ROI.2 as usize, ROI.3 as usize);
    let mut patch = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let start = ((y + row) * SIZE + x) * 3;
        patch.extend_from_slice(&reference[start..start + w * 3]);
    }
    patch
}

/// fact 语义帧：480×480 黑底 + patch 落点位（无锚帧传空 patch）
fn fact_frame(patch: &[u8], x: usize, y: usize) -> Vec<u8> {
    let mut frame = vec![0u8; SIZE * SIZE * 3];
    for row in 0..176 {
        let src = row * 176 * 3;
        let dst = ((y + row) * SIZE + x) * 3;
        frame[dst..dst + 176 * 3].copy_from_slice(&patch[src..src + 176 * 3]);
    }
    frame
}

/// 确定性伪噪声帧（LCG，与 Python/原生测试的种子口径互不影响）
fn noise_frame(size: usize) -> Vec<u8> {
    let mut seed = 0x2026_0317u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 16) as u8
    };
    let mut frame = vec![0u8; size * size * 3];
    for pixel in frame.iter_mut() {
        *pixel = next();
    }
    frame
}

fn audit_params() -> TrainParams {
    TrainParams {
        roi_xywh: ROI,
        angle_min: 0.0,
        angle_max: 0.0,
        angle_step: 1.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: None,
    }
}

fn read_u64_le(bytes: &[u8], at: usize) -> u64 {
    let raw: [u8; 8] = bytes[at..at + 8].try_into().expect("u64 读取越界");
    u64::from_le_bytes(raw)
}

/// 容器首精修段提取（单级资产来源，布局对齐 abi_chain.rs）
fn first_fine_segment(container: &[u8]) -> Vec<u8> {
    assert!(container.len() >= 13 && container.starts_with(b"VF2S3"), "容器 magic 不符");
    let count = read_u64_le(container, 5) as usize;
    let start = 13 + 8 * (count + 3);
    let length = read_u64_le(container, 13) as usize;
    container[start..start + length].to_vec()
}

fn expect_math(report: &AuditReport, what: &str) {
    let pos_stats = [
        report.pos_min,
        report.pos_p25,
        report.pos_median,
        report.pos_p75,
        report.pos_max,
    ];
    assert!(pos_stats.windows(2).all(|w| w[0] <= w[1]), "{what} 正组分布应单调：{pos_stats:?}");
    if report.neg_count > 0 {
        let neg_stats = [
            report.neg_min,
            report.neg_p25,
            report.neg_median,
            report.neg_p75,
            report.neg_max,
        ];
        assert!(neg_stats.windows(2).all(|w| w[0] <= w[1]), "{what} 负组分布应单调");
    }
    assert_eq!(report.separation, report.pos_p25 - report.neg_max, "{what} 分离度口径漂移");
    let lo = 30.0f64.max(report.neg_max + 5.0);
    let hi = 85.0f64.min(report.pos_p25 - 5.0);
    assert!(
        report.recommend_lo == lo && report.recommend_hi == hi,
        "{what} 推荐区间数学漂移：[{},{}] 应为 [{lo},{hi}]",
        report.recommend_lo,
        report.recommend_hi
    );
    assert_eq!(
        report.flag == audit_flag::INSEPARABLE,
        lo > hi,
        "{what} INSEPARABLE 判定应与 lo>hi 一致"
    );
    let low = report.pos_count < 5 || report.neg_count < 5;
    assert_eq!(
        report.confidence == audit_confidence::LOW,
        low,
        "{what} 置信度应与样本门槛一致"
    );
}

#[test]
fn health_reports_structure_and_self_score() {
    let library = task_library();
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    let bytes = train_model(reference, &audit_params()).expect("训练失败");
    let model = Model::from_library(&library, &bytes).expect("模型加载失败");

    let report: HealthReport = model.health().expect("容器体检失败");
    assert_eq!(report.verdict, health_verdict::HEALTHY, "健康模板应 HEALTHY");
    assert_eq!(health_verdict::name(report.verdict), "HEALTHY");
    assert!(report.self_score_valid && report.self_score >= 70.0, "自匹配应实测且 ≥70");
    assert!(report.pose_count >= 1, "姿态数应 ≥1");
    assert!(
        4 <= report.fine_min_features
            && report.fine_min_features <= report.fine_max_features
            && report.fine_max_features <= 512,
        "fine 特征数口径漂移"
    );
    assert!(report.coarse_features >= 4, "粗层特征数应 ≥4");
    assert_eq!(report.template_width, 176, "模板尺寸应=训练 ROI");
    assert_eq!(report.template_height, 176, "模板尺寸应=训练 ROI");
    assert_eq!(
        (report.detector_features, report.detector_weak, report.detector_strong),
        (128, 30.0, 60.0),
        "默认三参快照漂移"
    );

    let single_bytes = first_fine_segment(&bytes);
    let single = Model::from_library(&library, &single_bytes).expect("单级加载失败");
    let solo = single.health().expect("单级体检失败");
    assert!(!solo.self_score_valid && solo.self_score == 0.0, "单级自匹配应未实测");
    assert_eq!(solo.coarse_features, 0, "单级粗层特征数应 0");
    assert_eq!(solo.verdict, health_verdict::HEALTHY, "单级特征充足应 HEALTHY");
}

#[test]
fn audit_fact_groups_math_and_boundaries() {
    let library = task_library();
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    let bytes = train_model(reference, &audit_params()).expect("训练失败");
    let model = Model::from_library(&library, &bytes).expect("模型加载失败");

    let patch = roi_patch(&pixels);
    let buffers = [
        fact_frame(&patch, 40, 40),
        vec![0u8; SIZE * SIZE * 3],
        fact_frame(&patch, 260, 60),
        fact_frame(&patch, 60, 260),
    ];
    let views: Vec<BgrImage<'_>> = buffers
        .iter()
        .map(|data| BgrImage::new(SIZE as i32, SIZE as i32, data).expect("帧构造失败"))
        .collect();
    let pos = [views[0], views[2], views[3]];
    let neg = [views[1]];

    let report = model
        .audit_discriminability(&pos, &neg)
        .expect("fact 正负组审计失败");
    assert_eq!(report.pos_count, 3, "正组应 3 帧全有分");
    assert_eq!(report.pos_zero_hits, 0, "正组不应零命中");
    assert_eq!(report.neg_count, 0, "负组空白应零命中");
    assert_eq!(report.neg_zero_hits, 1, "负组空白应零命中 1 帧");
    assert!(report.pos_max >= 70.0, "正组最高分应 ≥70");
    expect_math(&report, "fact 正负组");
    assert_eq!(report.flag, audit_flag::RECOMMENDED, "可分离组应 RECOMMENDED");

    let no_neg = model.audit_discriminability(&pos, &[]).expect("无负组审计失败");
    assert_eq!(no_neg.flag, audit_flag::NO_NEGATIVE, "缺负组应 NO_NEGATIVE");
    assert_eq!(no_neg.neg_count, 0, "缺负组负样本数应 0");
    assert_eq!(no_neg.recommend_lo, 0.0, "缺负组不应产推荐区间");
    assert_eq!(no_neg.recommend_hi, 0.0, "缺负组不应产推荐区间");
    assert_eq!(no_neg.pos_count, 3, "无负组正组分布应保留");

    let noise = noise_frame(320);
    let noise_view = BgrImage::new(320, 320, &noise).expect("噪声帧构造失败");
    let inseparable = model
        .audit_discriminability(&[noise_view, noise_view], &[noise_view, noise_view])
        .expect("噪声双组审计失败");
    assert_eq!(inseparable.flag, audit_flag::INSEPARABLE, "噪声双组应 INSEPARABLE");

    let error = model
        .audit_discriminability(&[], &neg)
        .expect_err("正组空应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)), "正组空应绑定层拒绝：{error}");
}

#[test]
fn audit_layout_asserts_hold() {
    // 布局合同在 audit.rs 编译期断言（72/136）；此处复核枚举码值不漂移
    assert_eq!(health_verdict::HEALTHY, 0);
    assert_eq!(health_verdict::DEGRADED, 1);
    assert_eq!(health_verdict::WEAK, 2);
    assert_eq!(audit_flag::RECOMMENDED, 0);
    assert_eq!(audit_flag::INSEPARABLE, 1);
    assert_eq!(audit_flag::NO_NEGATIVE, 2);
    assert_eq!(audit_confidence::HIGH, 0);
    assert_eq!(audit_confidence::LOW, 1);
    assert_eq!(health_verdict::name(7), "UNKNOWN");
    assert_eq!(audit_flag::name(7), "UNKNOWN");
    assert_eq!(audit_confidence::name(7), "UNKNOWN");
}
