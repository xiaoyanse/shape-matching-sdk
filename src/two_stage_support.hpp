#pragma once
// 两级匹配编排支撑（自 two_stage.cpp 匿名空间按职责拆出，逐字平移）：组选择/
// 重投影/跨候选聚类/精修窗重试数据面/遮挡救分窗口收尾/早停计数。仅匹配编排
// 翻译单元与文本包含 two_stage.cpp 的测试使用；符号库隐藏可见性，不进公共 ABI。
#include "matching_profile.hpp"
#include "rescue_score.hpp"
#include "two_stage.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <vector>

namespace vfshape {

// 组选择（Python _fine_for_angle 对齐：重叠区取覆盖最多的一组）
inline const Model* fine_for_angle(const TwoStageModel& model, double angle, const GroupSpec** spec) {
    const Model* best = model.fines.front().get();
    const GroupSpec* best_spec = &model.groups.front();
    double best_cover = -1.0;
    for (size_t i = 0; i < model.fines.size(); ++i) {
        const auto& g = model.groups[i];
        if (g.lo - 1e-9 <= angle && angle <= g.hi + 1e-9) {
            const double cover = std::min(g.hi, angle + 180) - std::max(g.lo, angle - 180);
            if (cover > best_cover) {
                best_cover = cover;
                best = model.fines[i].get();
                best_spec = &g;
            }
        }
    }
    *spec = best_spec;
    return best;
}

inline void reproject_candidate(vf_shape_candidate& c, int x0, int y0,
                                const GroupSpec& g, double cs) {
    // Python _reproject 对齐（ds=1）：扣组偏置 → ×cs → +窗口原点；
    // transform 同链复合；scale=输出×cs。
    const double up = cs;
    c.center[0] = (c.center[0] - g.bias_x) * up + x0;
    c.center[1] = (c.center[1] - g.bias_y) * up + y0;
    for (int i = 0; i < 4; ++i) {
        c.corners[2 * i] = (c.corners[2 * i] - g.bias_x) * up + x0;
        c.corners[2 * i + 1] = (c.corners[2 * i + 1] - g.bias_y) * up + y0;
    }
    c.bbox[0] = (c.bbox[0] - g.bias_x) * up + x0;
    c.bbox[1] = (c.bbox[1] - g.bias_y) * up + y0;
    c.bbox[2] *= up;
    c.bbox[3] *= up;
    const double a = c.transform[0], b = c.transform[1], t_x = c.transform[2];
    const double d = c.transform[3], e = c.transform[4], t_y = c.transform[5];
    c.transform[0] = a * up;
    c.transform[1] = b * up;
    c.transform[2] = (t_x - g.bias_x) * up + x0;
    c.transform[3] = d * up;
    c.transform[4] = e * up;
    c.transform[5] = (t_y - g.bias_y) * up + y0;
    c.scale = c.scale * cs;
}

// 跨候选聚类（对齐单级路径 matcher.cpp 的同对象合并：重叠比 >0.4 合并，
// 姿态差异置歧义标志，簇内最高分为代表）——两级多窗口/跨组候选统一去重。
inline bool same_object(const vf_shape_candidate& a, const vf_shape_candidate& b) {
    const double iw =
        std::max(0., std::min(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2]) -
                        std::max(a.bbox[0], b.bbox[0]));
    const double ih =
        std::max(0., std::min(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3]) -
                        std::max(a.bbox[1], b.bbox[1]));
    const double overlap = iw * ih;
    const double area = a.bbox[2] * a.bbox[3] + b.bbox[2] * b.bbox[3] - overlap;
    return area > 0 && overlap / area > .4;
}

// 姿态歧义分数差豁免阈（0-100 相似度）：与 matcher.cpp 同值同语义，两处聚类
// 的豁免口径必须一致，改动需同步
constexpr double kAmbiguityScoreGap = 10.;

// 多目标搜索预算：粗候选按分数保留上限 / 精修窗调度上限。前两窗恒跑（保
// 双目标），其余窗串行补搜索并在达标去重候选数达 max_candidates 后早停
constexpr size_t kMaxCoarseCandidates = 16;
constexpr size_t kMaxFineWindows = 8;
constexpr size_t kMaxCachedRescueWindows = 2;
constexpr size_t kMaxCachedRescuePixels = 1024 * 1024;

inline cv::Mat scale_fine_window(const cv::Mat& window, double scale) {
    cv::Mat scaled = window;
    cv::resize(window, scaled,
               cv::Size(std::max(24, int(window.cols / scale)),
                        std::max(24, int(window.rows / scale))),
               0, 0, scale > 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return scaled;
}

struct RescueWindow {
    Model* fine = nullptr;
    const GroupSpec* spec = nullptr;
    cv::Mat source;
    std::optional<PreparedFrame> prepared;
    int x = 0, y = 0;
    double scale = 1.0;
    bool legacy_empty = false;
    bool resize_source = false;
};

inline std::vector<vf_shape_candidate> cluster_candidates(std::vector<vf_shape_candidate>&& raw) {
    std::vector<vf_shape_candidate> objects;
    for (auto& current : raw) {
        if (current.center[0] < 0 || current.center[1] < 0) continue;
        bool merged = false;
        for (auto& object : objects) {
            if (!same_object(object, current)) continue;
            const double angle =
                std::abs(std::remainder(object.angle - current.angle, 360.));
            // 语义对齐 matcher.cpp：姿态差超限且两解分数接近（差 < 豁免阈）才置
            // 歧义，分差拉开时高分解明显领先不置；标志只置位不清除（流式合并）
            if ((angle >= 15 || std::abs(std::log(object.scale / current.scale)) >= .14) &&
                std::abs(object.score - current.score) < kAmbiguityScoreGap)
                object.pose_ambiguous = 1;
            if (current.score > object.score) {
                current.pose_ambiguous = object.pose_ambiguous;
                object = current;
            }
            merged = true;
            break;
        }
        if (!merged) objects.push_back(current);
    }
    return objects;
}

inline std::vector<vf_shape_candidate> rescue_misses(
    std::array<RescueWindow, kMaxFineWindows>& misses, double fine_th, double threshold,
    MatchProfile& profile) {
    std::vector<vf_shape_candidate> results;
    for (size_t ci = 0; ci < misses.size(); ++ci) {
        auto& miss = misses[ci];
        if (!miss.fine) continue;
        const cv::Mat source = miss.resize_source ? scale_fine_window(miss.source, miss.scale) : miss.source;
        if (!miss.prepared) miss.prepared.emplace(prepare_frame(source, &miss.fine->detector));
        int rescued = 0;
        auto candidates = match_prepared_rescue(
            *miss.fine, *miss.prepared, source, threshold, &profile.window_collect_best[ci],
            &rescued, miss.legacy_empty ? std::optional<double>(fine_th) : std::nullopt);
        miss.prepared.reset();
        profile.rescued_candidates.fetch_add(rescued, std::memory_order_relaxed);
        for (auto& candidate : candidates) {
            reproject_candidate(candidate, miss.x, miss.y, *miss.spec, miss.scale);
            results.push_back(candidate);
        }
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    return cluster_candidates(std::move(results));
}

// 早停计数与最终输出口径的单一实现（F08）：对候选集副本执行与收尾路径
// 完全相同的序列（分数降序排序→聚类→计过阈值数）。具名供见证测试
// 构造级对拍（tests/witness_test.cpp 直调）——勿在别处另行实现计数。
inline size_t count_qualified_objects(std::vector<vf_shape_candidate> candidates, double threshold) {
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    candidates = cluster_candidates(std::move(candidates));
    return size_t(std::count_if(candidates.begin(), candidates.end(),
                                [threshold](const vf_shape_candidate& c) {
                                    return c.score >= threshold;
                                }));
}

}  // namespace vfshape
