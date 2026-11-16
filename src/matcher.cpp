#include "model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vfshape {
cv::Mat checked_image(const uint8_t* image, uint64_t size, int width, int height, uint64_t stride) {
    require(image && width >= 24 && height >= 24 && width <= max_edge && height <= max_edge,
            "image must be 24..4096 pixels on each edge");
    require(stride == uint64_t(width) * 3 && size == stride * height, "expected contiguous BGR8 image");
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

std::vector<double> samples(double low, double high, double step, double min, double max, double wrap) {
    require(std::isfinite(low) && std::isfinite(high) && std::isfinite(step) &&
            low >= min && high <= max && high >= low && step > 0,
            "invalid training range");
    const double intervals = (high - low) / step;
    require(std::isfinite(intervals) && intervals < max_poses, "pose budget exceeded");
    std::vector<double> values;
    for (int n = 0; n <= int(std::floor(intervals + 1e-9)); ++n) values.push_back(low + n * step);
    if (high - values.back() > 1e-7) values.push_back(high);
    // 首尾等价（差为 wrap 周期整数倍，如 -180≡180）时去重末点
    if (wrap > 0 && values.size() >= 2 &&
        std::abs(std::remainder(values.back() - values.front(), wrap)) < 1e-7) values.pop_back();
    return values;
}

void validate_config(const vf_shape_train_config& c, int width, int height) {
    require(c.struct_size == sizeof(c), "invalid training ABI structure size");
    require(c.weak_th >= 0 && c.strong_th >= 0, "detector thresholds must be non-negative");
    require(c.weak_th == 0 || c.strong_th == 0 || c.weak_th <= c.strong_th,
            "weak threshold must not exceed strong threshold");
    require(c.feature_count == 0 || (16 <= c.feature_count && c.feature_count <= 512),
            "feature count per level must be 16-512");
    require(c.roi_x >= 0 && c.roi_y >= 0 && c.roi_width >= 24 && c.roi_height >= 24 &&
            c.roi_width <= width && c.roi_height <= height &&
            c.roi_x <= width - c.roi_width && c.roi_y <= height - c.roi_height, "invalid training ROI");
    const auto angles = samples(c.angle_min, c.angle_max, c.angle_step, -180, 180);
    const auto scales = samples(c.scale_min, c.scale_max, c.scale_step, .25, 2);
    require(angles.size() * scales.size() <= max_poses, "pose budget exceeded");
    uint64_t pixels = 0;
    for (double scale : scales) {
        const int sw = cvRound(c.roi_width * scale), sh = cvRound(c.roi_height * scale);
        const int padding = int(std::ceil(std::max(sw, sh) * std::sqrt(2.) / 2.)) + 24;
        const int cw = sw + 2 * padding, ch = sh + 2 * padding;
        require(cw <= max_edge && ch <= max_edge, "training canvas budget exceeded");
        pixels += uint64_t(cw) * ch * angles.size();
    }
    require(pixels <= 250000000, "training pixel budget exceeded");
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
    // 特征筛选参数化：0=历史默认（128/30/60）；模型重建（codec 读回）
    // 时三值从序列化 config 复原，帧准备与模型特征选择始终同参。
    const int features = c.feature_count > 0 ? c.feature_count : feature_count;
    const float weak_th = c.weak_th > 0 ? float(c.weak_th) : weak_threshold;
    const float strong_th = c.strong_th > 0 ? float(c.strong_th) : strong_threshold;
    auto model = std::make_shared<Model>(features, weak_th, strong_th);
    model->config = c; model->width = image.cols; model->height = image.rows;
    const auto angles = samples(c.angle_min, c.angle_max, c.angle_step, -180, 180, 360);
    const auto scales = samples(c.scale_min, c.scale_max, c.scale_step, .25, 2);
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
                require(level.features.size() >= 4 && level.features.size() <= max_features_per_level,
                        "training feature budget exceeded");
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
    require(std::isfinite(threshold) && threshold > 0 && threshold <= 100, "invalid similarity threshold");
    validate_search_budget(model, image.cols, image.rows);
    // 帧准备必须用模型同参 detector（量化/特征选择同参，防静默错配）
    return match_prepared(model, prepare_frame(image, &model.detector), threshold);
}

void validate_search_budget(const Model& model, int width, int height) {
    const int padded_width = (width + PreparedFrame::padding * 2 + 15) / 16 * 16;
    const int padded_height = (height + PreparedFrame::padding * 2 + 15) / 16 * 16;
    require(uint64_t(padded_width / 16) * (padded_height / 16) * model.poses.size() <= 4000000,
            "matching search budget exceeded");
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
    require(std::isfinite(threshold) && threshold > 0 && threshold <= 100, "invalid similarity threshold");
    require(uint64_t(frame.padded_width / 16) * (frame.padded_height / 16) * model.poses.size() <= 4000000,
            "matching search budget exceeded");
    auto hits = model.detector.matchPrepared(frame.features, float(threshold), {"shape"});
    require(hits.size() <= 200000, "candidate budget exceeded");
    std::vector<vf_shape_candidate> objects;
    uint64_t comparisons = 0;
    for (const auto& hit : hits) {
        auto current = candidate(model, hit, PreparedFrame::padding);
        if (current.center[0] < 0 || current.center[0] >= frame.width ||
            current.center[1] < 0 || current.center[1] >= frame.height) continue;
        bool merged = false;
        for (auto& object : objects) {
            require(++comparisons <= 4000000, "candidate clustering budget exceeded");
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
