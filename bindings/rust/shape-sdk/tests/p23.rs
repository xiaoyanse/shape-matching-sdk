//! P2/P3 合同：suggest_contours_ex 参数化建议（默认一致/阈值/epsilon/拒绝码）+
//! estimate_training_ex 多边形预算（bbox 账本一致/公式版本/fill ratio/拒绝码）。
//! 运行时装载走 SHAPE_SDK_RUNTIME 或 dist/runtime（与其他集成测试同约定）。

use shape_sdk::{
    error_code, suggest_contours, suggest_contours_ex, train_polygon_model, BgrImage, Model,
    PolygonTrainParams, PolygonVertex, SuggestParams, ShapeLibrary, TrainParams, TrainingShape,
};

const SIZE: usize = 400;

fn fill(data: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, value: u8) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            let base = (y * SIZE + x) * 3;
            data[base..base + 3].fill(value);
        }
    }
}

/// 黑底：亮矩形 (60..180)² + 暗矩形 gray=20 (240..360)²
fn suggest_scene() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    fill(&mut data, 60, 60, 180, 180, 255);
    fill(&mut data, 240, 240, 360, 360, 20);
    data
}

/// 黑底：圆心 (200,200) 半径 60 白色圆盘
fn circle_scene() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as i32 - 200;
            let dy = y as i32 - 200;
            if dx * dx + dy * dy <= 3600 {
                let base = (y * SIZE + x) * 3;
                data[base..base + 3].fill(255);
            }
        }
    }
    data
}

fn image(pixels: &[u8]) -> BgrImage<'_> {
    BgrImage::new(SIZE as i32, SIZE as i32, pixels).expect("合成图构造失败")
}

fn max_vertices(suggestions: &[Vec<shape_sdk::PolygonVertex>]) -> usize {
    suggestions.iter().map(Vec::len).max().unwrap_or(0)
}

#[test]
fn suggest_ex_defaults_match_v1() {
    let pixels = suggest_scene();
    let scene = image(&pixels);
    let v1 = suggest_contours(scene, None, 8, 8).expect("v1 建议失败");
    assert!(!v1.is_empty(), "亮矩形应产出建议");
    let ex = suggest_contours_ex(scene, None, 8, 8, None).expect("_ex 建议失败");
    assert_eq!(v1, ex, "默认参数 _ex 与 v1 输出不一致");

    let circle_pixels = circle_scene();
    let circle = image(&circle_pixels);
    assert_eq!(
        suggest_contours(circle, None, 8, 8).expect("圆盘 v1 失败"),
        suggest_contours_ex(circle, None, 8, 8, None).expect("圆盘 _ex 失败"),
        "圆盘默认输出不一致"
    );
}

#[test]
fn suggest_ex_canny_tuning() {
    let pixels = suggest_scene();
    let scene = image(&pixels);
    let defaults = SuggestParams {
        canny_low: None,
        canny_high: None,
        epsilon_factor: None,
    };
    let default = suggest_contours_ex(scene, None, 8, 8, Some(&defaults)).expect("默认建议失败");
    assert_eq!(default.len(), 1, "默认阈值应只见亮矩形，实际 {} 条", default.len());
    let tuned = SuggestParams {
        canny_low: Some(8.0),
        canny_high: Some(24.0),
        epsilon_factor: None,
    };
    let dim = suggest_contours_ex(scene, None, 8, 8, Some(&tuned)).expect("低阈值建议失败");
    assert!(dim.len() >= 2, "低阈值应见暗矩形（≥2 条），实际 {} 条", dim.len());
    let again = suggest_contours_ex(scene, None, 8, 8, Some(&tuned)).expect("复跑建议失败");
    assert_eq!(dim, again, "低阈值两次调用输出不一致（确定性破坏）");
}

#[test]
fn suggest_ex_epsilon_tuning() {
    let circle_pixels = circle_scene();
    let circle = image(&circle_pixels);
    let default = suggest_contours_ex(circle, None, 8, 32, None).expect("默认建议失败");
    let fine = SuggestParams {
        canny_low: None,
        canny_high: None,
        epsilon_factor: Some(0.005),
    };
    let fine = suggest_contours_ex(circle, None, 8, 32, Some(&fine)).expect("细逼近失败");
    let default_vertices = max_vertices(&default);
    let fine_vertices = max_vertices(&fine);
    assert!(
        fine_vertices > default_vertices,
        "小 epsilon 应更细（{fine_vertices} 顶点 vs 默认 {default_vertices}）"
    );
}

#[test]
fn suggest_ex_rejects_carry_code() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let pixels = suggest_scene();
    let scene = image(&pixels);
    let cases = [
        SuggestParams {
            canny_low: Some(-1.0),
            canny_high: None,
            epsilon_factor: None,
        },
        SuggestParams {
            canny_low: Some(5.0),
            canny_high: Some(2.0),
            epsilon_factor: None,
        },
        SuggestParams {
            canny_low: None,
            canny_high: None,
            epsilon_factor: Some(0.0005),
        },
        SuggestParams {
            canny_low: None,
            canny_high: None,
            epsilon_factor: Some(0.5),
        },
    ];
    for params in &cases {
        let error = suggest_contours_ex(scene, None, 8, 8, Some(params))
            .expect_err("非法参数应拒绝");
        assert_eq!(error.native_status(), Some(1), "实际：{error}");
        let info = library.last_error_info().expect("错误码查询失败");
        assert_eq!(info.code, error_code::DETECTOR_PARAMS, "错误码应为 8");
        assert!(info.param_count >= 1, "缺数值上下文");
    }
}

fn estimate_size() -> (i32, i32) {
    (480, 480)
}

fn rect_params(roi: (i32, i32, i32, i32)) -> TrainParams {
    TrainParams {
        roi_xywh: roi,
        angle_min: -30.0,
        angle_max: 30.0,
        angle_step: 15.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: None,
    }
}

fn polygon_params(outline: Vec<PolygonVertex>, holes: Vec<Vec<PolygonVertex>>) -> PolygonTrainParams {
    PolygonTrainParams {
        outline,
        holes,
        angle_min: -30.0,
        angle_max: 30.0,
        angle_step: 15.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: None,
    }
}

fn ledger_matches(a: &shape_sdk::BudgetReport, b: &shape_sdk::BudgetReport, label: &str) {
    assert_eq!(a.pose_used, b.pose_used, "{label} pose 漂移");
    assert_eq!(a.pose_limit, b.pose_limit, "{label} pose_limit 漂移");
    assert_eq!(a.canvas_used, b.canvas_used, "{label} canvas 漂移");
    assert_eq!(a.pixel_used, b.pixel_used, "{label} pixel 漂移");
    assert_eq!(a.pixel_limit, b.pixel_limit, "{label} pixel_limit 漂移");
    assert_eq!(a.over_budget, b.over_budget, "{label} over_budget 漂移");
}

#[test]
fn estimate_ex_rect_matches_v1() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let (w, h) = estimate_size();
    let rect = rect_params((90, 90, 220, 220));
    let v1 = library.estimate_training(&rect, w, h).expect("v1 预算失败");
    assert_eq!(v1.formula_version, 1, "v1 公式版本应 1");
    let ex_rect = library
        .estimate_training_ex(w, h, TrainingShape::Rect(&rect))
        .expect("_ex 矩形预算失败");
    assert_eq!(ex_rect, v1, "_ex 矩形路径与 v1 逐字段不一致");
    assert_eq!(ex_rect.polygon_fill_ratio, 1.0, "矩形 fill ratio 应 1.0");
}

#[test]
fn estimate_ex_polygon_matches_bbox_rect_ledger() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let (w, h) = estimate_size();
    let vertex = |x: f64, y: f64| PolygonVertex { x, y };

    let square = polygon_params(
        vec![vertex(90.0, 90.0), vertex(310.0, 90.0), vertex(310.0, 310.0), vertex(90.0, 310.0)],
        Vec::new(),
    );
    let square_report = library
        .estimate_training_ex(w, h, TrainingShape::Polygon(&square))
        .expect("多边形预算失败");
    assert_eq!(square_report.formula_version, 2, "多边形路径公式版本应 2");
    let rect_report = library
        .estimate_training(&rect_params((90, 90, 220, 220)), w, h)
        .expect("矩形预算失败");
    ledger_matches(&square_report, &rect_report, "正方形");
    assert!((square_report.polygon_fill_ratio - 1.0).abs() < 1e-9, "正方形 fill ratio 应 1.0");

    let triangle = polygon_params(
        vec![vertex(90.0, 90.0), vertex(290.0, 90.0), vertex(90.0, 290.0)],
        Vec::new(),
    );
    let triangle_report = library
        .estimate_training_ex(w, h, TrainingShape::Polygon(&triangle))
        .expect("三角形预算失败");
    let triangle_rect = library
        .estimate_training(&rect_params((90, 90, 200, 200)), w, h)
        .expect("三角形外接矩形预算失败");
    ledger_matches(&triangle_report, &triangle_rect, "三角形");
    assert!(
        (triangle_report.polygon_fill_ratio - 0.5).abs() < 1e-9,
        "三角形 fill ratio 应 0.5，实际 {}",
        triangle_report.polygon_fill_ratio
    );
}

#[test]
fn estimate_ex_holed_fill_ratio() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let (w, h) = estimate_size();
    let vertex = |x: f64, y: f64| PolygonVertex { x, y };
    let holed = polygon_params(
        vec![vertex(90.0, 90.0), vertex(310.0, 90.0), vertex(310.0, 310.0), vertex(90.0, 310.0)],
        vec![vec![
            vertex(140.0, 140.0),
            vertex(180.0, 140.0),
            vertex(180.0, 180.0),
            vertex(140.0, 180.0),
        ]],
    );
    let report = library
        .estimate_training_ex(w, h, TrainingShape::Polygon(&holed))
        .expect("带洞预算失败");
    // 手算：外环 220²=48400，洞 40²=1600，(48400-1600)/48400
    let want = (48400.0 - 1600.0) / 48400.0;
    assert!(
        (report.polygon_fill_ratio - want).abs() < 1e-9,
        "带洞 fill ratio 手算不符：{} vs {want}",
        report.polygon_fill_ratio
    );
    assert_eq!(report.formula_version, 2);
}

#[test]
fn estimate_ex_polygon_rejects_carry_code() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let (w, h) = estimate_size();
    let vertex = |x: f64, y: f64| PolygonVertex { x, y };
    let bowtie = polygon_params(
        vec![
            vertex(100.0, 150.0),
            vertex(300.0, 150.0),
            vertex(200.0, 100.0),
            vertex(200.0, 300.0),
        ],
        Vec::new(),
    );
    let error = library
        .estimate_training_ex(w, h, TrainingShape::Polygon(&bowtie))
        .expect_err("自交轮廓应拒绝");
    assert_eq!(error.native_status(), Some(1), "实际：{error}");
    let info = library.last_error_info().expect("错误码查询失败");
    assert_eq!(info.code, error_code::INVALID_POLYGON, "错误码应为 17");
}

#[test]
fn estimate_ex_matches_training_shape() {
    // 预算与实际多边形训练消费同参：训练后元数据 ROI = 外接矩形，账本同式同序
    let vertex = |x: f64, y: f64| PolygonVertex { x, y };
    let triangle = polygon_params(
        vec![vertex(90.0, 90.0), vertex(370.0, 90.0), vertex(90.0, 370.0)],
        Vec::new(),
    );
    let mut reference = vec![0u8; SIZE * SIZE * 3];
    for y in 101..380usize {
        for x in 101..380usize {
            // 三角形 (90,90),(370,90),(90,370)：白填充 x+y ≤ 460（hypotenuse 内侧）
            if x + y <= 460 {
                let base = (y * SIZE + x) * 3;
                reference[base..base + 3].fill(255);
            }
        }
    }
    let scene = BgrImage::new(SIZE as i32, SIZE as i32, &reference).expect("合成图失败");
    let bytes = train_polygon_model(scene, &triangle).expect("三角形训练失败");
    let model = Model::from_bytes(&bytes).expect("加载失败");
    let metadata = model.metadata().expect("元数据查询失败");
    assert_eq!(metadata.roi_xywh, (90, 90, 280, 280), "训练 ROI 应为外接矩形");

    let library = ShapeLibrary::global().expect("运行时装载失败");
    let report = library
        .estimate_training_ex(SIZE as i32, SIZE as i32, TrainingShape::Polygon(&triangle))
        .expect("预算失败");
    let rect = TrainParams {
        roi_xywh: metadata.roi_xywh,
        angle_min: -30.0,
        angle_max: 30.0,
        angle_step: 15.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: None,
    };
    let rect_report = library
        .estimate_training(&rect, SIZE as i32, SIZE as i32)
        .expect("矩形预算失败");
    ledger_matches(&report, &rect_report, "预算与训练");
}
