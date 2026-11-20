#include "rescue_score.hpp"

#include <algorithm>
#include <cmath>

namespace vfshape {

int rescue_collect_gate(double threshold) {
    return int(std::ceil(kRescueBeta * threshold));
}

double rescue_adjusted_score(double raw, int visible, int nf) {
    const double floor_denom = std::ceil(kRescueBeta * double(nf));
    const double denom = std::max(double(std::max(visible, 0)), floor_denom);
    return std::min(raw * double(nf) / denom, 100.0);
}

cv::Mat build_rescue_occupancy(const cv::Mat& padded_bgr, const Detector& detector) {
    // 量化链路与 vendor prepareMemories 同参同序：构造即 update（高斯平滑→
    // 最强通道梯度→迟滞门控 8 朝向标签），无掩码路径直接消费 angle 标签图
    const auto& modality = detector.getModalities();
    const line2Dup::ColorGradientPyramid quantizer(padded_bgr, cv::Mat(),
                                                   modality->weak_threshold,
                                                   modality->num_features,
                                                   modality->strong_threshold);
    const int T = detector.getT(0);
    cv::Mat binary, horizontal = cv::Mat::zeros(quantizer.angle.size(), CV_8U);
    cv::threshold(quantizer.angle, binary, 0, 1, cv::THRESH_BINARY);
    cv::Mat occupancy = cv::Mat::zeros(quantizer.angle.size(), CV_8U);
    for (int c = 0; c < T; ++c)
        cv::bitwise_or(binary(cv::Rect(c, 0, binary.cols - c, binary.rows)),
                       horizontal(cv::Rect(0, 0, binary.cols - c, binary.rows)),
                       horizontal(cv::Rect(0, 0, binary.cols - c, binary.rows)));
    for (int r = 0; r < T; ++r)
        cv::bitwise_or(horizontal(cv::Rect(0, r, binary.cols, binary.rows - r)),
                       occupancy(cv::Rect(0, 0, binary.cols, binary.rows - r)),
                       occupancy(cv::Rect(0, 0, binary.cols, binary.rows - r)));
    return occupancy;
}

int count_visible_features(const cv::Mat& occupancy, const line2Dup::Template& templ,
                           double match_x, double match_y, int T, int offset) {
    const int origin_x = int(std::lround(match_x)) - offset;
    const int origin_y = int(std::lround(match_y)) - offset;
    int visible = 0;
    for (const auto& feature : templ.features) {
        const int x = origin_x + feature.x;
        const int y = origin_y + feature.y;
        if (x < 0 || y < 0 || x >= occupancy.cols || y >= occupancy.rows) continue;
        if (occupancy.at<uchar>(y, x)) ++visible;
    }
    return visible;
}

std::vector<vf_shape_candidate> match_prepared_rescue(Model& model, const PreparedFrame& frame,
                                                      const cv::Mat& window_bgr, double threshold,
                                                      double* collect_best, int* rescued,
                                                      std::optional<double> legacy_empty_gate) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    validate_search_budget(model, frame.width, frame.height);
    const int gate = rescue_collect_gate(threshold);
    if (gate >= threshold) return {};
    const auto collect = [&](float cutoff) {
        auto hits = model.detector.matchPrepared(frame.features, cutoff, {"shape"});
        require_coded(hits.size() <= 200000, VF_SHAPE_SEARCH_BUDGET, "candidate budget exceeded",
                      double(hits.size()), 200000.0);
        return hits;
    };
    if (legacy_empty_gate && !collect(float(*legacy_empty_gate)).empty()) return {};

    auto hits = collect(float(gate));
    int saved = 0;
    double best_raw = 0.0;
    cv::Mat occupancy;
    const int T0 = model.detector.getT(0);
    const int offset = T0 / 2 + (T0 % 2 - 1);
    for (auto& hit : hits) {
        const double raw = hit.similarity;
        best_raw = std::max(best_raw, raw);
        hit.similarity = -1.f;
        if (raw >= threshold) continue;
        const auto& pyramid = model.detector.getTemplates("shape", hit.template_id);
        if (pyramid.empty() || pyramid.front().features.empty()) continue;
        if (occupancy.empty())
            occupancy = build_rescue_occupancy(pad_for_detector(window_bgr), model.detector);
        const auto& templ = pyramid.front();
        const int visible = count_visible_features(occupancy, templ, hit.x, hit.y, T0, offset);
        const double adjusted = rescue_adjusted_score(raw, visible, int(templ.features.size()));
        if (adjusted < threshold || float(adjusted) < threshold) continue;
        hit.similarity = float(adjusted);
        ++saved;
    }
    hits.erase(std::remove_if(hits.begin(), hits.end(),
                              [](const line2Dup::Match& m) { return m.similarity < 0.f; }),
               hits.end());
    if (collect_best) *collect_best = best_raw;
    if (rescued) *rescued = saved;
    return collect_hit_objects(model, hits, frame);
}

}
