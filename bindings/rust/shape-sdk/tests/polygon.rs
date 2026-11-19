//! 多边形合同：train_polygon→元数据往返（绕向归一+洞数）→错误码→
//! suggest_contours（建议/确定性/先验框/非法参数）。合成图全内存，图像设计
//! 与 tests/polygon_test.cpp 同纪律：绘制形状与轮廓间留黑边距（vendor 掩码
//! 腐蚀会吃掉贴边边界特征）。

use shape_sdk::{
    error_code, suggest_contours, train_polygon_model, BgrImage, Model, PolygonTrainParams,
    PolygonVertex, ShapeError, ShapeLibrary,
};

const SIZE: usize = 480;

fn fill(data: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize, value: u8) {
    for y in y0..=y1 {
        for x in x0..=x1 {
            let base = (y * SIZE + x) * 3;
            data[base..base + 3].fill(value);
        }
    }
}

/// 黑底白色方框 (100..219)² + 内部灰块（形状四周留黑边距）
fn synth_reference() -> Vec<u8> {
    let mut data = vec![0u8; SIZE * SIZE * 3];
    fill(&mut data, 100, 100, 219, 219, 255);
    fill(&mut data, 150, 150, 169, 169, 90);
    data
}

fn vertex(x: f64, y: f64) -> PolygonVertex {
    PolygonVertex { x, y }
}

fn square_outline() -> Vec<PolygonVertex> {
    vec![vertex(90.0, 310.0), vertex(310.0, 310.0), vertex(310.0, 90.0), vertex(90.0, 90.0)]
}

/// 与 SDK 同式的绕向归一期望（有向面积为负时反序）
fn normalized(points: &[PolygonVertex]) -> Vec<PolygonVertex> {
    let mut area = 0.0;
    for (index, p) in points.iter().enumerate() {
        let q = points[(index + 1) % points.len()];
        area += p.x * q.y - q.x * p.y;
    }
    let mut points = points.to_vec();
    if area < 0.0 {
        points.reverse();
    }
    points
}

fn polygon_params(outline: Vec<PolygonVertex>) -> PolygonTrainParams {
    PolygonTrainParams {
        outline,
        holes: Vec::new(),
        angle_min: -30.0,
        angle_max: 30.0,
        angle_step: 15.0,
        scale_min: 1.0,
        scale_max: 1.0,
        scale_step: 1.0,
        detector: None,
    }
}

#[test]
fn polygon_train_and_metadata_roundtrip() {
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    // 反转输入绕向，验证 SDK 归一承诺
    let mut outline = square_outline();
    outline.reverse();
    let bytes = train_polygon_model(reference, &polygon_params(outline)).expect("多边形训练失败");
    let model = Model::from_bytes(&bytes).expect("模型加载失败");

    let candidates = model.match_(reference, 70.0, 16).expect("自匹配失败");
    assert!(!candidates.is_empty(), "多边形自匹配应命中");
    assert!(
        candidates.iter().any(|c| c.score >= 70.0),
        "自匹配分应 ≥70"
    );

    let metadata = model.polygon_metadata().expect("多边形元数据失败");
    let want = normalized(&square_outline());
    assert_eq!(metadata.outline.len(), want.len(), "顶点数漂移");
    for (got, expected) in metadata.outline.iter().zip(&want) {
        assert_eq!(got.x, expected.x, "顶点 x 字节不等");
        assert_eq!(got.y, expected.y, "顶点 y 字节不等");
    }
    assert_eq!(metadata.hole_count, 0, "洞数应 0");
}

#[test]
fn polygon_holed_roundtrip() {
    let mut pixels = synth_reference();
    // 洞区挖黑 + 洞心灰块（洞内强边缘不产特征的语义由原生端掩蔽保证）
    fill(&mut pixels, 140, 140, 180, 180, 0);
    fill(&mut pixels, 152, 152, 168, 168, 90);
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");
    let mut params = polygon_params(vec![
        vertex(90.0, 90.0),
        vertex(230.0, 90.0),
        vertex(230.0, 230.0),
        vertex(90.0, 230.0),
    ]);
    params.holes = vec![vec![
        vertex(140.0, 140.0),
        vertex(180.0, 140.0),
        vertex(180.0, 180.0),
        vertex(140.0, 180.0),
    ]];
    let bytes = train_polygon_model(reference, &params).expect("带洞训练失败");
    let model = Model::from_bytes(&bytes).expect("模型加载失败");
    let metadata = model.polygon_metadata().expect("带洞元数据失败");
    assert_eq!(metadata.hole_count, 1, "洞数应 1");
    let want = normalized(&params.outline);
    assert_eq!(metadata.outline, want, "带洞往返顶点漂移");
}

#[test]
fn polygon_rejections_carry_code() {
    let library = ShapeLibrary::global().expect("运行时装载失败");
    let pixels = synth_reference();
    let reference = BgrImage::new(SIZE as i32, SIZE as i32, &pixels).expect("合成图构造失败");

    // 自交（非零面积交叉四边形）→ INVALID_POLYGON(17) 细分 4
    let bowtie = polygon_params(vec![
        vertex(100.0, 150.0),
        vertex(300.0, 150.0),
        vertex(200.0, 100.0),
        vertex(200.0, 300.0),
    ]);
    let error = train_polygon_model(reference, &bowtie).expect_err("自交应拒绝");
    assert_eq!(error.native_status(), Some(1), "实际：{error}");
    let info = library.last_error_info().expect("错误码查询失败");
    assert_eq!(info.code, error_code::INVALID_POLYGON, "错误码应为 17");
    assert!(info.param_count >= 1 && info.params[0] as i32 == 4, "细分类型应为 4");

    // 外接矩形 <24px → 细分 9
    let tiny = polygon_params(vec![
        vertex(20.0, 20.0),
        vertex(42.0, 20.0),
        vertex(20.0, 42.0),
    ]);
    let error = train_polygon_model(reference, &tiny).expect_err("外接矩形<24px 应拒绝");
    assert_eq!(error.native_status(), Some(1));
    let info = library.last_error_info().expect("错误码查询失败");
    assert_eq!(info.code, error_code::INVALID_POLYGON);
    assert!(info.param_count >= 1 && info.params[0] as i32 == 9, "细分类型应为 9");
    assert_eq!(error_code::name(error_code::INVALID_POLYGON), "INVALID_POLYGON");
}

#[test]
fn suggest_contours_contract() {
    // 两个分离白矩形（轴对齐规避 Canny 斜边阶梯断裂）
    const SCENE: usize = 400;
    let mut data = vec![0u8; SCENE * SCENE * 3];
    for y in 60..=160 {
        for x in 60..=180 {
            let base = (y * SCENE + x) * 3;
            data[base..base + 3].fill(255);
        }
    }
    for y in 240..=340 {
        for x in 240..=360 {
            let base = (y * SCENE + x) * 3;
            data[base..base + 3].fill(255);
        }
    }
    let scene = BgrImage::new(SCENE as i32, SCENE as i32, &data).expect("场景图构造失败");

    let first = suggest_contours(scene, None, 8, 8).expect("轮廓建议失败");
    assert!(first.len() >= 2, "建议条数应 ≥2，实际 {}", first.len());
    for suggestion in &first {
        assert!(
            (3..=8).contains(&suggestion.len()),
            "建议顶点数越 3..8：{}",
            suggestion.len()
        );
    }
    let again = suggest_contours(scene, None, 8, 8).expect("第二次轮廓建议失败");
    assert_eq!(first, again, "两次建议输出不一致（确定性破坏）");

    let hinted = suggest_contours(scene, Some([30, 30, 220, 220]), 8, 8).expect("先验框建议失败");
    assert!(!hinted.is_empty(), "先验框内应有建议");
    for suggestion in &hinted {
        for v in suggestion {
            assert!(
                (30.0..=250.0).contains(&v.x) && (30.0..=250.0).contains(&v.y),
                "先验框建议顶点越框：({},{})",
                v.x,
                v.y
            );
        }
    }

    let error =
        suggest_contours(scene, Some([300, 300, 200, 200]), 4, 8).expect_err("越界先验框应拒绝");
    assert_eq!(error.native_status(), Some(1), "实际：{error}");
    let error = suggest_contours(scene, None, 0, 8).expect_err("建议条数 0 应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));
    let error = suggest_contours(scene, None, 4, 2).expect_err("顶点限 2 应拒绝");
    assert!(matches!(error, ShapeError::InvalidInput(_)));

    let blank_data = vec![0u8; 200 * 200 * 3];
    let blank = BgrImage::new(200, 200, &blank_data).expect("空图构造失败");
    let empty = suggest_contours(blank, None, 4, 8).expect("空图建议失败");
    assert!(empty.is_empty(), "空图应无建议（非错误）");
}
