// 两级打包训练（fines 分组 + coarse + 锚点偏置）。从 two_stage.cpp 按职责
// 拆出；容器编解码见 container_codec.cpp，匹配见 two_stage.cpp。

#include "two_stage.hpp"
#include "coarse_image.hpp"
#include "polygon.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace vfshape {
namespace {

int effective_k(int roi_w, int roi_h) {
    // Python 对齐：阈值 56px（K8→K4→K2 降档）
    const int longest = std::max(roi_w, roi_h);
    for (int k : {8, 4, 2})
        if (longest >= 56 * k) return k;
    return 2;
}

// 恒等（或指定角）自匹配测锚点偏置：Python _measure_anchor_bias 对齐
void measure_bias(Model& model, const cv::Mat& image_bgr,
                  const cv::Rect& roi, double angle_deg, double* bx, double* by) {
    // 注意：target 必须独立分配——`cv::Mat target = image_bgr` 浅拷贝共享数据，
    // warpAffine 的 src/dst 同 Mat 是未定义行为（实测旋转图损坏 74% 像素差异，
    // 组偏置被测成 (19,-14) 级离谱值）。
    // 性能：旋转与匹配限制在 ROI 邻域窗（目标位置已知=ROI 中心±模板半径），
    // 全图 match ~400ms → 窗口 ~30ms（偏置测量 4 次/训练）。
    const double cx = roi.x + roi.width / 2.0, cy = roi.y + roi.height / 2.0;
    const double radius = std::hypot(double(roi.width), double(roi.height)) / 2.0 + 64.0;
    const int wx0 = int(std::max(0.0, cx - radius)), wy0 = int(std::max(0.0, cy - radius));
    const int wx1 = int(std::min(double(image_bgr.cols), cx + radius + 1.0));
    const int wy1 = int(std::min(double(image_bgr.rows), cy + radius + 1.0));
    if (wx1 - wx0 < 64 || wy1 - wy0 < 64) {
        *bx = 0;
        *by = 0;
        return;
    }
    cv::Mat patch = image_bgr(cv::Rect(wx0, wy0, wx1 - wx0, wy1 - wy0));
    cv::Mat target;
    const double pcx = cx - wx0, pcy = cy - wy0;  // ROI 中心的窗口系坐标
    if (std::abs(angle_deg) > 1e-9) {
        auto m = cv::getRotationMatrix2D(cv::Point2d(pcx, pcy), angle_deg, 1.0);
        cv::warpAffine(patch, target, m, patch.size());
    } else {
        target = patch;
    }
    // F06 去异常化：无 try/catch。空候选本身是预期降级信号（零偏置）；
    // 真异常（bad_alloc/OpenCV 运行时/预算合同）自然穿透训练报错
    auto cands = match(model, target, 50.0);
    if (cands.empty()) {
        // 降级语义：锚点校准依赖 ROI 自匹配，低对比/弱特征场景落空是预期路径
        *bx = 0;
        *by = 0;
        return;
    }
    const auto& best = *std::max_element(
        cands.begin(), cands.end(),
        [](const vf_shape_candidate& a, const vf_shape_candidate& b) {
            return a.score < b.score;
        });
    *bx = best.center[0] - pcx;
    *by = best.center[1] - pcy;
}

}  // namespace

// —— 两级训练（Python train_packaged 对齐）——
std::shared_ptr<TwoStageModel> train_two_stage(
    const cv::Mat& image_bgr, const vf_shape_train_config& config,
    const PolygonShape* polygon) {
    const int w = image_bgr.cols, h = image_bgr.rows;
    validate_config(config, w, h);
    const cv::Rect roi = polygon ? polygon->bounds
                                 : cv::Rect(config.roi_x, config.roi_y,
                                            config.roi_width, config.roi_height);

    // 角度分组：group_span=16°（8° 步长+8° 余量），重叠 8°
    const double fine_step = std::min(config.angle_step, 0.5);
    const double total = config.angle_max - config.angle_min;
    const double group_span = kCoarseAngleStep * 2;
    const int n_groups =
        (total > group_span) ? int(std::ceil(total / group_span)) : 1;

    auto model = std::make_shared<TwoStageModel>();
    model->base_config = config;
    // R01 对齐：cfg 三参持久化实际生效值（默认零在训练侧物化），与单级 codec
    // 的回填同口径——新容器不携带零字段，解码侧按持久化值严格校验不再冲突
    const auto tuned = validate_detector_params(config.feature_count, config.weak_th,
                                                config.strong_th);
    model->base_config.feature_count = tuned.features;
    model->base_config.weak_th = tuned.weak;
    model->base_config.strong_th = tuned.strong;
    model->ref_width = w;   // F01：原始参考图尺寸（容器 ref 键）
    model->ref_height = h;
    model->groups.resize(n_groups);

    auto train_fine = [&](double lo, double hi) {
        vf_shape_train_config fc = config;
        fc.angle_min = lo;
        fc.angle_max = hi;
        fc.angle_step = fine_step;
        fc.scale_min = 1.0;
        fc.scale_max = 1.0;
        fc.scale_step = 1.0;
        // 相邻组来自同一张参考图，重叠角度复用完全相同的特征。
        return train(image_bgr, fc, model->fines.empty() ? nullptr : model->fines.back().get(),
                     polygon);
    };

    if (n_groups <= 1) {
        model->fines.push_back(train_fine(config.angle_min, config.angle_max));
        model->groups[0] = {config.angle_min, config.angle_max, 0, 0};
    } else {
        const double overlap = kCoarseAngleStep;
        const double stride = total / n_groups;
        for (int i = 0; i < n_groups; ++i) {
            double lo = config.angle_min + i * stride - (i ? overlap : 0);
            double hi = config.angle_min + (i + 1) * stride + (i < n_groups - 1 ? overlap : 0);
            lo = std::max(config.angle_min, lo);
            hi = std::min(config.angle_max, hi);
            model->fines.push_back(train_fine(lo, hi));
            model->groups[i] = {lo, hi, 0, 0};
        }
    }

    // 粗层：K 逐档重试（Python 对齐：56 阈值 + 失败显式抛错）
    const int k0 = effective_k(config.roi_width, config.roi_height);
    const int coarse_steps = std::min(
        3, std::max(1, int(std::lround((config.scale_max - config.scale_min) /
                                        std::max(0.05, config.scale_step / 2))) + 1));
    bool coarse_ok = false;
    std::string coarse_failure;
    int coarse_tried = 0;
    for (int k : {8, 4, 2}) {
        if (k > k0) continue;
        ++coarse_tried;
        cv::Mat small;
        resize_coarse_image(image_bgr, small, k);
        vf_shape_train_config cc = config;
        PolygonShape coarse_polygon;
        const PolygonShape* coarse_shape = nullptr;
        if (polygon) {
            // 多边形粗层：顶点除 k 缩放，ROI=缩放后整包界并截到缩图内
            //（fillPoly 自裁剪越界部分；相似缩放保持已校验的几何关系）
            coarse_polygon = scale_polygon(*polygon, 1.0 / k, small.cols, small.rows);
            coarse_shape = &coarse_polygon;
            cc.roi_x = coarse_polygon.bounds.x;
            cc.roi_y = coarse_polygon.bounds.y;
            cc.roi_width = coarse_polygon.bounds.width;
            cc.roi_height = coarse_polygon.bounds.height;
        } else {
            // 缩图尺寸同为 floor 整除（coarse_image.cpp），ROI 同口径映射：
            // 原点允许 0（整图 ROI），宽高对实际缩图裁限
            cc.roi_x = config.roi_x / k;
            cc.roi_y = config.roi_y / k;
            cc.roi_width = std::min(config.roi_width / k, small.cols - cc.roi_x);
            cc.roi_height = std::min(config.roi_height / k, small.rows - cc.roi_y);
        }
        cc.angle_step = kCoarseAngleStep;
        cc.scale_step = std::max(0.05, (config.scale_max - config.scale_min) /
                                            std::max(1, coarse_steps - 1));
        if (cc.roi_width < 8 || cc.roi_height < 8) {
            coarse_failure = "K=" + std::to_string(k) + ": coarse ROI " +
                             std::to_string(cc.roi_width) + "x" + std::to_string(cc.roi_height) +
                             " below 8px minimum";
            continue;
        }
        try {
            model->coarse = train(small, cc, nullptr, coarse_shape);
            model->k = k;
            coarse_ok = true;
            break;
        } catch (const std::invalid_argument& ex) {
            // F06 异常精确分类：仅输入合同类失败（require 抛 invalid_argument：
            // 特征不足/几何/预算）可换档重试，各档留真实失败原因；
            // bad_alloc/OpenCV 运行时等其他异常立即重抛穿透，不吞成"换档可解决"
            coarse_failure = "K=" + std::to_string(k) + ": " + ex.what();
        }
    }
    if (!coarse_ok)
        fail(VF_SHAPE_COARSE_TRAIN_FAILED, {double(coarse_tried)},
             ("coarse model training failed at all K levels (" + coarse_failure + ")").c_str());

    // 模板灰度 + B7 点
    cv::Mat roi_bgr = image_bgr(roi);
    cv::cvtColor(roi_bgr, model->tmpl_gray, cv::COLOR_BGR2GRAY);
    model->b7 = extract_b7_points(model->tmpl_gray);

    // 组级偏置：每组在其中心角合成旋转图上测（Python 对齐）
    for (size_t i = 0; i < model->groups.size(); ++i) {
        const double mid = (model->groups[i].lo + model->groups[i].hi) / 2.0;
        measure_bias(*model->fines[i], image_bgr, roi, mid,
                     &model->groups[i].bias_x, &model->groups[i].bias_y);
    }
    if (polygon) {
        auto asset = std::make_shared<PolygonAsset>();
        asset->outline = flatten_vertices(polygon->outline);
        for (const auto& hole : polygon->holes) asset->holes.push_back(flatten_vertices(hole));
        model->polygon = std::move(asset);
    }
    return model;
}

}  // namespace vfshape
