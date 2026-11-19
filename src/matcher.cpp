#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vfshape {
cv::Mat checked_image(const uint8_t* image, uint64_t size, int width, int height, uint64_t stride) {
    require_coded(image && width >= 24 && height >= 24 && width <= max_edge && height <= max_edge,
                  VF_SHAPE_INVALID_IMAGE, "image must be 24..4096 pixels on each edge",
                  double(width), double(height));
    require_coded(stride == uint64_t(width) * 3 && size == stride * height,
                  VF_SHAPE_INVALID_IMAGE, "expected contiguous BGR8 image",
                  double(stride), double(size));
    return cv::Mat(height, width, CV_8UC3, const_cast<uint8_t*>(image), size_t(stride));
}

void Model::build_pose_adjacency() {
    // 排列契约（train 循环）：外层 scale 内层 angle —— 同 scale 组 id 连续。
    std::vector<std::array<int, 2>> adj(poses.size(), {-1, -1});
    for (size_t id = 0; id < poses.size(); ++id) {
        const double s = poses[id].scale, a = poses[id].angle;
        auto is_neighbor = [&](size_t other, bool greater) {
            return other < poses.size() && poses[other].scale == s &&
                   ((greater && poses[other].angle > a) || (!greater && poses[other].angle < a));
        };
        if (id > 0 && is_neighbor(id - 1, false)) adj[id][0] = int(id) - 1;
        if (id + 1 < poses.size() && is_neighbor(id + 1, true)) adj[id][1] = int(id) + 1;
    }
    detector.pose_adjacency["shape"] = std::move(adj);
}

std::vector<double> samples(double low, double high, double step, double min, double max,
                            vf_shape_error_code range_code, double wrap) {
    require_coded(std::isfinite(low) && std::isfinite(high) && std::isfinite(step) &&
                  low >= min && high <= max && high >= low && step > 0,
                  range_code, "invalid training range", low, high, step);
    const double intervals = (high - low) / step;
    // 单轴档数超限归入该轴范围码：参数为当前步长、档数上限、建议步长
    require_coded(std::isfinite(intervals) && intervals < max_poses, range_code,
                  "pose budget exceeded", step, double(max_poses), (high - low) / 349.0);
    std::vector<double> values;
    for (int n = 0; n <= int(std::floor(intervals + 1e-9)); ++n) values.push_back(low + n * step);
    if (high - values.back() > 1e-7) values.push_back(high);
    // 首尾等价（差为 wrap 周期整数倍，如 -180≡180）时去重末点
    if (wrap > 0 && values.size() >= 2 &&
        std::abs(std::remainder(values.back() - values.front(), wrap)) < 1e-7) values.pop_back();
    return values;
}

DetectorTuning validate_detector_params(int features, double weak, double strong) {
    const int f = features == 0 ? feature_count : features;
    const double w = weak == 0 ? weak_threshold : weak;
    const double s = strong == 0 ? strong_threshold : strong;
    require_coded(std::isfinite(w) && std::isfinite(s), VF_SHAPE_DETECTOR_PARAMS,
                  "detector thresholds must be finite", w, s);
    require_coded(w > 0 && s > 0, VF_SHAPE_DETECTOR_PARAMS,
                  "detector thresholds must be positive", w, s);
    require_coded(w <= s, VF_SHAPE_DETECTOR_PARAMS,
                  "weak threshold must not exceed strong threshold", w, s);
    require_coded(s <= kMaxStrongThreshold, VF_SHAPE_DETECTOR_PARAMS,
                  "strong threshold exceeds reachable gradient magnitude (1020)", s,
                  kMaxStrongThreshold);
    require_coded(f >= 16 && f <= max_features_per_level, VF_SHAPE_DETECTOR_PARAMS,
                  "feature count per level must be 16-512", double(f), 16,
                  double(max_features_per_level));
    // 训练入口同样防 float 下溢：极小正值转 float 归零会产出导出后无法加载的资产
    //（训练侧归一化零=默认合法，但下溢拒绝与持久化侧口径一致）
    const float fw = float(w), fs = float(s);
    require_coded(std::isfinite(fw) && std::isfinite(fs) && fw > 0 && fs > 0,
                  VF_SHAPE_DETECTOR_PARAMS, "detector thresholds not representable as positive float",
                  w, s);
    return {f, fw, fs};
}

// 与训练侧语义相反：序列化字段是实际生效值，零=损坏须拒绝而非默认化；并防
// double→float 下溢（极小正值转 float 归零）导致接受后再加载被默认化洗值。
DetectorTuning validate_persisted_detector_params(int features, double weak, double strong) {
    require_coded(std::isfinite(weak) && std::isfinite(strong), VF_SHAPE_MODEL_TAMPER,
                  "detector thresholds must be finite", weak, strong);
    require_coded(weak > 0 && strong > 0, VF_SHAPE_MODEL_TAMPER,
                  "persisted detector thresholds must be positive", weak, strong);
    require_coded(weak <= strong, VF_SHAPE_MODEL_TAMPER,
                  "weak threshold must not exceed strong threshold", weak, strong);
    require_coded(strong <= kMaxStrongThreshold, VF_SHAPE_MODEL_TAMPER,
                  "strong threshold exceeds reachable gradient magnitude (1020)", strong,
                  kMaxStrongThreshold);
    require_coded(features >= 16 && features <= max_features_per_level, VF_SHAPE_MODEL_TAMPER,
                  "feature count per level must be 16-512", double(features), 16,
                  double(max_features_per_level));
    const float fw = static_cast<float>(weak), fs = static_cast<float>(strong);
    require_coded(std::isfinite(fw) && std::isfinite(fs) && fw > 0 && fs > 0,
                  VF_SHAPE_MODEL_TAMPER, "detector thresholds not representable as positive float",
                  weak, strong);
    return {features, fw, fs};
}

void validate_config_domain(const vf_shape_train_config& c, int width, int height) {
    require(c.struct_size == sizeof(c), "invalid training ABI structure size");
    validate_detector_params(c.feature_count, c.weak_th, c.strong_th);
    require_coded(c.roi_x >= 0 && c.roi_y >= 0 && c.roi_width >= 24 && c.roi_height >= 24 &&
                  c.roi_width <= width && c.roi_height <= height &&
                  c.roi_x <= width - c.roi_width && c.roi_y <= height - c.roi_height,
                  VF_SHAPE_INVALID_ROI, "invalid training ROI", double(c.roi_width),
                  double(c.roi_height));
    samples(c.angle_min, c.angle_max, c.angle_step, -180, 180, VF_SHAPE_INVALID_ANGLE_RANGE);
    samples(c.scale_min, c.scale_max, c.scale_step, .25, 2, VF_SHAPE_INVALID_SCALE_RANGE);
}

// 训练像素总量上限（预算公式 v1；账本口径与训练拒绝同一常量）
constexpr uint64_t kMaxTrainingPixels = 250000000;

vf_shape_budget_report account_training(const vf_shape_train_config& c, int width, int height) {
    (void)width; (void)height;  // 预留：账本公式后续版本可加入图像相关约束
    vf_shape_budget_report report{};
    report.struct_size = sizeof(report);
    report.formula_version = 1;
    const auto angles = samples(c.angle_min, c.angle_max, c.angle_step, -180, 180,
                                VF_SHAPE_INVALID_ANGLE_RANGE);
    const auto scales = samples(c.scale_min, c.scale_max, c.scale_step, .25, 2,
                                VF_SHAPE_INVALID_SCALE_RANGE);
    report.pose_used = uint32_t(angles.size() * scales.size());
    report.pose_limit = uint32_t(max_poses);
    uint64_t pixels = 0;
    uint32_t canvas = 0;
    for (double scale : scales) {
        const int sw = cvRound(c.roi_width * scale), sh = cvRound(c.roi_height * scale);
        const int padding = int(std::ceil(std::max(sw, sh) * std::sqrt(2.) / 2.)) + 24;
        const int cw = sw + 2 * padding, ch = sh + 2 * padding;
        canvas = std::max(canvas, uint32_t(std::max(cw, ch)));
        pixels += uint64_t(cw) * ch * angles.size();
    }
    report.canvas_used = canvas;
    report.canvas_limit = uint32_t(max_edge);
    report.pixel_used = pixels;
    report.pixel_limit = kMaxTrainingPixels;
    report.pose_over = report.pose_used > report.pose_limit;
    report.canvas_over = report.canvas_used > report.canvas_limit;
    report.pixel_over = report.pixel_used > report.pixel_limit;
    report.over_budget = report.pose_over
                             ? int32_t(VF_SHAPE_POSE_BUDGET)
                             : (report.canvas_over ? int32_t(VF_SHAPE_CANVAS_BUDGET)
                                                   : (report.pixel_over ? int32_t(VF_SHAPE_PIXEL_BUDGET)
                                                                        : int32_t(VF_SHAPE_OK)));
    return report;
}

vf_shape_budget_report estimate_training(const vf_shape_train_config& c, int width, int height) {
    validate_config_domain(c, width, height);
    return account_training(c, width, height);
}

void validate_config(const vf_shape_train_config& c, int width, int height) {
    validate_config_domain(c, width, height);
    const auto report = account_training(c, width, height);
    require_coded(report.pose_used <= report.pose_limit, VF_SHAPE_POSE_BUDGET,
                  "pose budget exceeded", double(report.pose_used), double(report.pose_limit));
    require_coded(report.canvas_used <= report.canvas_limit, VF_SHAPE_CANVAS_BUDGET,
                  "training canvas budget exceeded", double(report.canvas_used),
                  double(report.canvas_limit));
    require_coded(report.pixel_used <= report.pixel_limit, VF_SHAPE_PIXEL_BUDGET,
                  "training pixel budget exceeded", double(report.pixel_used),
                  double(report.pixel_limit));
}

std::shared_ptr<Model> train(const cv::Mat& image, const vf_shape_train_config& c,
                             const Model* previous_group) {
    validate_config(c, image.cols, image.rows);
    if (previous_group) {
        const auto& p = previous_group->config;
        require(previous_group->width == image.cols && previous_group->height == image.rows &&
                    p.roi_x == c.roi_x && p.roi_y == c.roi_y && p.roi_width == c.roi_width &&
                    p.roi_height == c.roi_height && p.weak_th == c.weak_th &&
                    p.strong_th == c.strong_th && p.feature_count == c.feature_count,
                "overlapping groups must share reference geometry and detector parameters");
    }
    const auto tuning = validate_detector_params(c.feature_count, c.weak_th, c.strong_th);
    auto model = std::make_shared<Model>(tuning.features, tuning.weak, tuning.strong);
    model->config = c; model->width = image.cols; model->height = image.rows;
    const auto angles = samples(c.angle_min, c.angle_max, c.angle_step, -180, 180,
                                VF_SHAPE_INVALID_ANGLE_RANGE, 360);
    const auto scales = samples(c.scale_min, c.scale_max, c.scale_step, .25, 2,
                                VF_SHAPE_INVALID_SCALE_RANGE);
    const auto crop = image(cv::Rect(c.roi_x, c.roi_y, c.roi_width, c.roi_height));
    for (double scale : scales) {
        cv::Mat scaled;
        cv::resize(crop, scaled, {}, scale, scale, scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
        const int padding = int(std::ceil(std::max(scaled.cols, scaled.rows) * std::sqrt(2.) / 2.)) + 24;
        cv::Mat canvas(scaled.rows + padding * 2, scaled.cols + padding * 2, CV_8UC3, cv::Scalar::all(0));
        const cv::Rect roi(padding, padding, scaled.cols, scaled.rows);
        scaled.copyTo(canvas(roi));
        cv::Mat mask(canvas.size(), CV_8UC1, cv::Scalar(0));
        mask(roi).setTo(255);
        for (double angle : angles) {
            if (previous_group) {
                const auto& poses = previous_group->poses;
                const auto cached = std::find_if(poses.begin(), poses.end(), [&](const Pose& pose) {
                    return pose.angle == angle && pose.scale == scale;
                });
                if (cached != poses.end()) {
                    const int id = int(std::distance(poses.begin(), cached));
                    model->detector.append(previous_group->detector.getTemplates("shape", id));
                    model->poses.push_back(*cached);
                    continue;
                }
            }
            const auto rotation = cv::getRotationMatrix2D(
                cv::Point2f(canvas.cols / 2.f, canvas.rows / 2.f), angle, 1);
            cv::Mat rotated, rotated_mask;
            cv::warpAffine(canvas, rotated, rotation, canvas.size());
            cv::warpAffine(mask, rotated_mask, rotation, mask.size(), cv::INTER_NEAREST);
            const int id = model->detector.addTemplate(rotated, "shape", rotated_mask);
            if (id < 0) continue;
            for (const auto& level : model->detector.getTemplates("shape", id))
                require_coded(level.features.size() >= 4 && level.features.size() <= max_features_per_level,
                              VF_SHAPE_DETECTOR_PARAMS, "training feature budget exceeded",
                              double(level.features.size()), 4, double(max_features_per_level));
            Pose pose; pose.angle = angle; pose.scale = scale;
            pose.canvas_width = canvas.cols; pose.canvas_height = canvas.rows;
            for (int row = 0; row < 2; ++row) {
                const double a = rotation.at<double>(row, 0), b = rotation.at<double>(row, 1);
                pose.transform[row * 3] = a * scaled.cols / c.roi_width;
                pose.transform[row * 3 + 1] = b * scaled.rows / c.roi_height;
                pose.transform[row * 3 + 2] = rotation.at<double>(row, 2)
                    + a * (padding - double(scaled.cols) * c.roi_x / c.roi_width)
                    + b * (padding - double(scaled.rows) * c.roi_y / c.roi_height);
            }
            model->poses.push_back(pose);
        }
    }
    require(!model->poses.empty(), "insufficient edge features in reference ROI");
    // 角度插值负结果（口径不对齐）暂不消费——不填 adjacency，vendor
    // 邻居响应计算随之跳过（Codex 性能项：未消费计算白费）。机制保留，
    // 口径对齐实验时重新启用此处。
    // model->build_pose_adjacency();
    return model;
}

namespace {
vf_shape_candidate candidate(const Model& model, const line2Dup::Match& hit, int padding) {
    vf_shape_candidate result{};
    result.score = hit.similarity; result.pose_id = hit.template_id;
    const auto& pose = model.poses.at(hit.template_id);
    const auto& templ = model.detector.getTemplates("shape", hit.template_id).front();
    // 负结果登记：相邻档 16×16 邻域峰值与主体亚像素精化的分数口径
    // 不对齐，插值引入系统噪声（非档位真值实测 2 好 3 差，12.3° 误差 0.48°
    // 劣于最近档 0.2°）。角度插值暂不消费（vendor 侧 adjacency/offset 机制
    // 保留），待邻居响应与主精化口径对齐后复评。float 坐标修复（B5 亚像素
    // 输出不再被 int 截断）为确定收益，保留。
    result.angle = pose.angle;
    result.scale = pose.scale;
    std::copy(pose.transform.begin(), pose.transform.end(), result.transform);
    result.transform[2] += hit.x - padding - templ.tl_x;
    result.transform[5] += hit.y - padding - templ.tl_y;
    const auto& c = model.config;
    const double points[8] = {double(c.roi_x), double(c.roi_y), double(c.roi_x + c.roi_width),
        double(c.roi_y), double(c.roi_x + c.roi_width), double(c.roi_y + c.roi_height),
        double(c.roi_x), double(c.roi_y + c.roi_height)};
    double left = 1e20, top = 1e20, right = -1e20, bottom = -1e20;
    for (int n = 0; n < 4; ++n) {
        const double x = result.transform[0] * points[2 * n] + result.transform[1] * points[2 * n + 1] + result.transform[2];
        const double y = result.transform[3] * points[2 * n] + result.transform[4] * points[2 * n + 1] + result.transform[5];
        result.corners[2 * n] = x; result.corners[2 * n + 1] = y;
        result.center[0] += x / 4; result.center[1] += y / 4;
        left = std::min(left, x); top = std::min(top, y); right = std::max(right, x); bottom = std::max(bottom, y);
    }
    result.bbox[0] = left; result.bbox[1] = top; result.bbox[2] = right - left; result.bbox[3] = bottom - top;
    return result;
}
bool same_object(const vf_shape_candidate& a, const vf_shape_candidate& b) {
    const double iw = std::max(0., std::min(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2]) - std::max(a.bbox[0], b.bbox[0]));
    const double ih = std::max(0., std::min(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3]) - std::max(a.bbox[1], b.bbox[1]));
    const double overlap = iw * ih;
    const double area = a.bbox[2] * a.bbox[3] + b.bbox[2] * b.bbox[3] - overlap;
    return area > 0 && overlap / area > .4;
}
}

std::vector<vf_shape_candidate> match(Model& model, const cv::Mat& image, double threshold) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    validate_search_budget(model, image.cols, image.rows);
    // 帧准备必须用模型同参 detector（量化/特征选择同参，防静默错配）
    return match_prepared(model, prepare_frame(image, &model.detector), threshold);
}

void validate_search_budget(const Model& model, int width, int height) {
    const int padded_width = (width + PreparedFrame::padding * 2 + 15) / 16 * 16;
    const int padded_height = (height + PreparedFrame::padding * 2 + 15) / 16 * 16;
    const auto used = uint64_t(padded_width / 16) * (padded_height / 16) * model.poses.size();
    require_coded(used <= 4000000, VF_SHAPE_SEARCH_BUDGET, "matching search budget exceeded",
                  double(used), 4000000.0);
}

PreparedFrame prepare_frame(const cv::Mat& image, const Detector* detector) {
    // Borders permit partial edge targets and satisfy upstream pyramid alignment.
    constexpr int padding = PreparedFrame::padding;
    const int width = (image.cols + padding * 2 + 15) / 16 * 16;
    const int height = (image.rows + padding * 2 + 15) / 16 * 16;
    cv::Mat padded(height, width, CV_8UC3, cv::Scalar::all(0));
    image.copyTo(padded(cv::Rect(padding, padding, image.cols, image.rows)));
    Detector fallback;  // 无模型上下文时用默认参（训练前的独立帧准备）
    const Detector& d = detector ? *detector : fallback;
    return {d.prepareOwned(std::move(padded)), image.cols, image.rows, width, height};
}

// 姿态歧义分数差豁免阈（0-100 相似度）：簇内姿态差超限的两解分差 ≥ 此值时
// 高分解明显领先，豁免歧义；分差小于此值才是无法分辨的多姿态解
constexpr double kAmbiguityScoreGap = 10.;

std::vector<vf_shape_candidate> match_prepared(Model& model, const PreparedFrame& frame, double threshold) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    const auto blocks = uint64_t(frame.padded_width / 16) * (frame.padded_height / 16) * model.poses.size();
    require_coded(blocks <= 4000000, VF_SHAPE_SEARCH_BUDGET, "matching search budget exceeded",
                  double(blocks), 4000000.0);
    auto hits = model.detector.matchPrepared(frame.features, float(threshold), {"shape"});
    require_coded(hits.size() <= 200000, VF_SHAPE_SEARCH_BUDGET, "candidate budget exceeded",
                  double(hits.size()), 200000.0);
    std::vector<vf_shape_candidate> objects;
    uint64_t comparisons = 0;
    for (const auto& hit : hits) {
        auto current = candidate(model, hit, PreparedFrame::padding);
        if (current.center[0] < 0 || current.center[0] >= frame.width ||
            current.center[1] < 0 || current.center[1] >= frame.height) continue;
        bool merged = false;
        for (auto& object : objects) {
            require_coded(++comparisons <= 4000000, VF_SHAPE_SEARCH_BUDGET,
                          "candidate clustering budget exceeded", double(comparisons), 4000000.0);
            if (!same_object(object, current)) continue;
            const double angle = std::abs(std::remainder(object.angle - current.angle, 360.));
            // 姿态差超限且两解分数接近（差 < kAmbiguityScoreGap）才是无法分辨的
            // 多姿态解，置歧义；分差 ≥ 豁免阈时高分解明显领先，信任高分解不置。
            // 流式合并标志只置位不清除：簇内只要出现过"姿态差超限且分数接近"的
            // 组合就保持歧义（后续高分命中不能翻案），仅当姿态差大的解与簇代表
            // 分数始终拉开时才豁免（比较基 = 当前簇代表分 vs 新命中分）。
            if ((angle >= 15 || std::abs(std::log(object.scale / current.scale)) >= .14) &&
                std::abs(object.score - current.score) < kAmbiguityScoreGap)
                object.pose_ambiguous = 1;
            // 簇内最高分为代表：整体替换输出数据并保留已置位的歧义标志
            if (current.score > object.score) {
                current.pose_ambiguous = object.pose_ambiguous;
                object = current;
            }
            merged = true; break;
        }
        if (!merged) objects.push_back(current);
    }
    return objects;
}
}
