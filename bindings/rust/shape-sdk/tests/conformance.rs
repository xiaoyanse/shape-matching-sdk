//! 四帧事实基准（复刻 benchmarks/run_facts.py 全语义，Rust 绑定一致性见证）。
//!
//! 语义：帧 1 裁锚邻域 260×260 作参考训练（±30°/5°、单尺度、ROI (30,30,200,200)），
//! 四帧以 spec 阈值逐个匹配，对照 shape-facts-v1 spec 预期真值判定——锚在场帧须
//! 命中且最佳中心距锚 ≤8px（网格量化 4px 的两倍容差），帧 2 无锚须零命中。
//! 预期真值运行时读取 spec JSON（锚点/阈值与 Python 侧同源）。
//! 运行时按 crate 默认发现（仓内 dist/runtime）。

use std::path::Path;

use shape_sdk::{train_model, BgrImage, Model, TrainParams};

const CROP_HALF: i32 = 130;
/// 裁剪邻域内居中图案（对齐 spec 200px+余量 30）
const TEMPLATE_ROI: (i32, i32, i32, i32) = (30, 30, 200, 200);
/// 网格量化 4px 的两倍容差（对齐 run_facts.CENTER_TOLERANCE_PX）
const CENTER_TOLERANCE_PX: f64 = 8.0;
/// 对齐 run_facts.FRAMES（shape-facts-v1 帧文件名）
const FRAMES: [(&str, &str); 4] = [
    ("1", "synthetic-f0001-positive.png"),
    ("2", "synthetic-f0002-anchor-absent-target-positive.png"),
    ("3", "synthetic-f0003-target-negative.png"),
    ("4", "synthetic-f0004-positive-new-pose.png"),
];

fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate 应位于仓内 bindings/rust/shape-sdk")
}

/// 帧锚真值：((中心 x, 中心 y), 锚角)
type FrameAnchor = ((f64, f64), f64);

/// spec 预期真值：匹配阈值 + 帧尺寸 + 各帧锚（None=无锚）
struct SpecTruth {
    threshold: f64,
    frame_size: (i32, i32),
    anchors: Vec<(&'static str, Option<FrameAnchor>)>,
}

fn spec_truth() -> SpecTruth {
    let path = repo_root()
        .join("benchmarks")
        .join("shape-facts-v1")
        .join("template-and-frame-spec-v1.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("spec 读取失败：{error}"));
    let spec: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|error| panic!("spec 解析失败：{error}"));
    let threshold = spec["template_algorithm"]["score_threshold"]
        .as_f64()
        .expect("spec 缺 score_threshold");
    let size = spec["frame_spec"]["size_px"]
        .as_array()
        .and_then(|items| {
            let width = items.first()?.as_i64()?;
            let height = items.get(1)?.as_i64()?;
            Some((width as i32, height as i32))
        })
        .expect("spec 缺 size_px");
    let anchors = &spec["frame_spec"]["anchor_center_and_angle"];
    let truth = FRAMES
        .iter()
        .map(|(frame_id, _)| {
            let entry = &anchors[*frame_id];
            let anchor = if entry.is_null() {
                None
            } else {
                let center = entry.get(0).and_then(|v| v.as_array()).expect("锚缺中心");
                let x = center.first().and_then(|v| v.as_f64()).expect("锚心缺 x");
                let y = center.get(1).and_then(|v| v.as_f64()).expect("锚心缺 y");
                let angle = entry.get(1).and_then(|v| v.as_f64()).expect("锚缺角度");
                Some(((x, y), angle))
            };
            (*frame_id, anchor)
        })
        .collect();
    SpecTruth {
        threshold,
        frame_size: size,
        anchors: truth,
    }
}

/// 解码基准帧 PNG 为连续 BGR（image crate 输出 RGB，交换 R/B 对齐 cv2.IMREAD_COLOR）
fn load_bgr_frame(filename: &str) -> (i32, i32, Vec<u8>) {
    let path = repo_root()
        .join("benchmarks")
        .join("shape-facts-v1")
        .join("frames")
        .join(filename);
    let decoded = image::open(&path)
        .unwrap_or_else(|error| panic!("基准帧缺失或解码失败：{filename}：{error}"));
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut bgr = Vec::with_capacity(rgb.as_raw().len());
    for pixel in rgb.pixels() {
        bgr.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    (width as i32, height as i32, bgr)
}

/// 连续 BGR 帧内裁剪子窗（行优先，输出保持连续）
fn crop_bgr(data: &[u8], frame_width: usize, x0: usize, y0: usize, width: usize, height: usize) -> Vec<u8> {
    let stride = frame_width * 3;
    let mut out = Vec::with_capacity(width * height * 3);
    for row in 0..height {
        let start = (y0 + row) * stride + x0 * 3;
        out.extend_from_slice(&data[start..start + width * 3]);
    }
    out
}

fn facts_train_params() -> TrainParams {
    TrainParams {
        roi_xywh: TEMPLATE_ROI,
        angle_min: -30.0,
        angle_max: 30.0,
        angle_step: 5.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 0.05,
        detector: None,
    }
}

#[test]
fn four_frame_facts_benchmark() {
    let spec = spec_truth();

    // 帧 1 锚邻域裁剪作参考训练（同 run_facts：锚心 ±130px，260×260）
    let (frame1_w, frame1_h, frame1) = load_bgr_frame(FRAMES[0].1);
    assert_eq!(
        (frame1_w, frame1_h),
        spec.frame_size,
        "帧1 尺寸与 spec 不符，疑似资产漂移"
    );
    let ((anchor_x, anchor_y), _) = spec.anchors[0].1.expect("帧1 spec 缺锚");
    let x0 = (anchor_x as i32 - CROP_HALF) as usize;
    let y0 = (anchor_y as i32 - CROP_HALF) as usize;
    let patch = crop_bgr(
        &frame1,
        frame1_w as usize,
        x0,
        y0,
        (CROP_HALF * 2) as usize,
        (CROP_HALF * 2) as usize,
    );
    let reference = BgrImage::new(CROP_HALF * 2, CROP_HALF * 2, &patch)
        .expect("锚邻域裁剪图构造失败");
    let model_bytes = train_model(reference, &facts_train_params()).expect("四帧基准训练失败");
    let model = Model::from_bytes(&model_bytes).expect("基准模型加载失败");

    // 四帧逐个匹配，对照 spec 预期真值（锚在场定位 ≤8px；无锚零命中）
    for (frame_id, filename) in FRAMES {
        let (width, height, data) = load_bgr_frame(filename);
        assert_eq!((width, height), spec.frame_size, "帧{frame_id} 尺寸与 spec 不符");
        let frame = BgrImage::new(width, height, &data).expect("基准帧构造失败");
        let hits = model
            .match_(frame, spec.threshold, 16)
            .unwrap_or_else(|error| panic!("帧{frame_id} 匹配异常：{error}"));
        let (_, expected) = spec
            .anchors
            .iter()
            .find(|(id, _)| *id == frame_id)
            .expect("spec 真值缺帧");
        match expected {
            None => {
                assert!(
                    hits.is_empty(),
                    "帧{frame_id}（{filename}）预期无锚，实际误报 {} 个（最高分 {:.1}）",
                    hits.len(),
                    hits.iter().map(|c| c.score).fold(f64::NEG_INFINITY, f64::max),
                );
            }
            Some(((expect_x, expect_y), _)) => {
                assert!(
                    !hits.is_empty(),
                    "帧{frame_id}（{filename}）FAIL（漏检）：预期命中但无候选"
                );
                let best = hits
                    .iter()
                    .max_by(|a, b| a.score.total_cmp(&b.score))
                    .expect("候选非空已断言");
                let offset =
                    (best.center.0 - expect_x).hypot(best.center.1 - expect_y);
                assert!(
                    offset <= CENTER_TOLERANCE_PX,
                    "帧{frame_id}（{filename}）FAIL（定位超差）：中心偏差 {offset:.2}px \
                     （容差 {CENTER_TOLERANCE_PX}），实际中心 ({:.1},{:.1}) vs 预期 \
                     ({expect_x:.0},{expect_y:.0})，分数 {:.1}，角度 {:.1}°",
                    best.center.0,
                    best.center.1,
                    best.score,
                    best.angle,
                );
            }
        }
    }
    model.close().expect("基准模型关闭失败");
}
