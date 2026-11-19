// 两级匹配：粗到精窗口精修 + 跨候选聚类 + B7 几何精修。
// 训练见 two_stage_train.cpp，容器编解码见 container_codec.cpp。

#include "two_stage.hpp"
#include "matching_profile.hpp"
#include "coarse_image.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <vector>

namespace vfshape {

namespace {

// 组选择（Python _fine_for_angle 对齐：重叠区取覆盖最多的一组）
const Model* fine_for_angle(const TwoStageModel& model, double angle, const GroupSpec** spec) {
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

void reproject_candidate(vf_shape_candidate& c, int x0, int y0,
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
bool same_object(const vf_shape_candidate& a, const vf_shape_candidate& b) {
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

std::vector<vf_shape_candidate> cluster_candidates(std::vector<vf_shape_candidate>&& raw) {
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

// 早停计数与最终输出口径的单一实现（F08）：对候选集副本执行与收尾路径
// 完全相同的序列（分数降序排序→聚类→计过阈值数）。具名供见证测试
// 构造级对拍（tests/witness_test.cpp 直调）——勿在别处另行实现计数。
size_t count_qualified_objects(std::vector<vf_shape_candidate> candidates, double threshold) {
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    candidates = cluster_candidates(std::move(candidates));
    return size_t(std::count_if(candidates.begin(), candidates.end(),
                                [threshold](const vf_shape_candidate& c) {
                                    return c.score >= threshold;
                                }));
}
}  // namespace

// —— 两级匹配（Python TwoStageMatcher._match_locked 对齐）——
std::vector<vf_shape_candidate> match_two_stage(
    TwoStageModel& model, const cv::Mat& image_bgr, double threshold, int max_candidates,
    MatchTraceSnapshot* trace) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    MatchProfile profile;
    const int w = image_bgr.cols, h = image_bgr.rows;
    // 粗层：1/k 缩图（4096 接口限制在编排内自然放开——缩图与窗口均小）
    cv::Mat small;
    MatchProfile::measure(profile.coarse_resize_ms, [&] {
      resize_coarse_image(image_bgr, small, model.k);
    });
    std::vector<vf_shape_candidate> coarse_cands;
    // 异常穿透到 ABI boundary 报错：执行失败不得伪装成"无匹配"空结果
    coarse_cands = MatchProfile::measure(profile.coarse_ms, [&] {
        return match(*model.coarse, small, std::max(45.0, threshold - 20.0));
    });
    profile.coarse_candidates = int(coarse_cands.size());
    if (coarse_cands.empty()) {
        if (trace) *trace = profile.snapshot();
        return {};
    }
    std::sort(coarse_cands.begin(), coarse_cands.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    if (coarse_cands.size() > kMaxCoarseCandidates)
        coarse_cands.resize(kMaxCoarseCandidates);
    // 前两窗恒跑（Codex P1 教训：分差启发在双目标场景丢第二目标）；
    // 其余窗补足多目标覆盖，达标去重候选数达 max_candidates 早停
    const size_t use = std::min<size_t>(kMaxFineWindows, coarse_cands.size());
    profile.count = int(use);

    // 精修半窗自适应：模板对角/2 + 4k（窗口必须装得下模板）
    double fine_half = 300.0;
    if (!model.tmpl_gray.empty()) {
        const double diag = std::hypot(double(model.tmpl_gray.cols), double(model.tmpl_gray.rows));
        fine_half = std::max(240.0, diag / 2.0 + 4.0 * model.k);
    }
    const double fine_th = std::max(50.0, threshold * 0.85);
    auto refine_window = [&](size_t ci) {
        profile.windows_searched.fetch_add(1, std::memory_order_relaxed);
        // 剖析槽仅 2 个（MatchProfile 契约）：前两窗入槽，其余窗记到本地空槽
        MatchProfile::Window overflow_trace;
        MatchProfile::Window& trace_window =
            ci < profile.windows.size() ? profile.windows[ci] : overflow_trace;
        std::vector<vf_shape_candidate> window_results;
        const auto& cc = coarse_cands[ci];
        const double cx = cc.center[0] * model.k, cy = cc.center[1] * model.k;
        double cs = (cc.scale > 0) ? cc.scale : 1.0;
        double ca = cc.angle;
        if (ca > 180) ca -= 360;
        const GroupSpec* spec = nullptr;
        const Model* fine_model = fine_for_angle(model, ca, &spec);

        const double half = fine_half * std::max(1.0, cs);
        const int x0 = int(std::max(0.0, cx - half)), y0 = int(std::max(0.0, cy - half));
        const int x1 = int(std::min(double(w), x0 + 2 * half));
        const int y1 = int(std::min(double(h), y0 + 2 * half));
        if (x1 - x0 < 24 || y1 - y0 < 24) return window_results;
        trace_window.width = x1 - x0; trace_window.height = y1 - y0;
        trace_window.scale = cs; trace_window.angle = ca;
        cv::Mat window = image_bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));

        cv::Mat window_scaled = window;
        const bool needs_scale = std::abs(cs - 1.0) > 0.02;
        if (needs_scale) {
            MatchProfile::Timer resize_timer(trace_window.resize_ms);
            cv::resize(window, window_scaled,
                       cv::Size(std::max(24, int(window.cols / cs)),
                                std::max(24, int(window.rows / cs))),
                       0, 0, cs > 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
        }
        cv::Mat effective = window_scaled;
        std::vector<vf_shape_candidate> refined;
        // 精修异常穿透（含并行窗 future.get 重抛）：失败窗口不得静默产出空结果
        auto best_score = [](const auto& candidates) {
            double score = 0;
            for (const auto& candidate : candidates) score = std::max(score, candidate.score);
            return score;
        };
        auto prepare = [&](const cv::Mat& image) {
            ++trace_window.prepares;
            return MatchProfile::measure(trace_window.prepare_ms, [&] {
                return prepare_frame(image, &fine_model->detector);
            });
        };
        auto evaluate = [&](Model& m, const PreparedFrame& frame) {
            const int ready = frame.features.preparedLevels();
            auto matches = MatchProfile::measure(trace_window.match_ms, [&] {
                return match_prepared(m, frame, fine_th);
            });
            trace_window.deferred_levels += frame.features.preparedLevels() - ready;
            return matches;
        };
        auto prepared = prepare(window_scaled);
        refined = evaluate(*const_cast<Model*>(fine_model), prepared);
        if (best_score(refined) < threshold && needs_scale) {
            ++trace_window.scale_retries;
            auto retry_frame = prepare(window);
            auto retry = evaluate(*const_cast<Model*>(fine_model), retry_frame);
            if (best_score(retry) > best_score(refined)) {
                cs = 1.0;
                effective = window;
                prepared = std::move(retry_frame);
                refined = std::move(retry);
            }
        }
        if (best_score(refined) < threshold && model.fines.size() > 1) {
            for (size_t gi = 0; gi < model.fines.size() && best_score(refined) < threshold; ++gi) {
                auto& alternate = *model.fines[gi];
                if (&alternate == fine_model) continue;
                ++trace_window.group_retries;
                const auto& cfg = alternate.config;
                const auto& original = fine_model->config;
                std::vector<vf_shape_candidate> retry;
                if (cfg.weak_th == original.weak_th && cfg.strong_th == original.strong_th &&
                    cfg.feature_count == original.feature_count) {
                    retry = evaluate(alternate, prepared);
                } else {
                    retry = match(alternate, effective, fine_th);
                }
                if (best_score(retry) > best_score(refined)) {
                    refined = std::move(retry);
                    spec = &model.groups[gi];
                }
            }
        }
        profile.window_best[ci] = best_score(refined);
        for (auto& r : refined) {
            reproject_candidate(r, x0, y0, *spec, cs);
            window_results.push_back(r);
        }
        return window_results;
    };
    std::vector<vf_shape_candidate> results;
    // F08：早停计数走 count_qualified_objects（与收尾路径同代码序列的具名
    // 单一实现）；IoU 非传递使任何单独实现的去重计数都与排序聚类不等价
    auto qualified_objects = [&results, threshold]() {
        return count_qualified_objects(results, threshold);
    };
    {
    MatchProfile::Timer window_timer(profile.windows_ms);
#ifdef VF_SHAPE_PARALLEL_WINDOWS
    std::future<std::vector<vf_shape_candidate>> second;
    if (use >= 2) second = std::async(std::launch::async, refine_window, 1);
    results = refine_window(0);
    if (second.valid()) {
        auto candidates = second.get();
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
    for (size_t ci = 2; ci < use && qualified_objects() < size_t(max_candidates); ++ci) {
        auto candidates = refine_window(ci);
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
#else
    for (size_t ci = 0; ci < use && (ci < 2 || qualified_objects() < size_t(max_candidates)); ++ci) {
        auto candidates = refine_window(ci);
        results.insert(results.end(), candidates.begin(), candidates.end());
    }
#endif
    }
    MatchProfile::Timer finish_timer(profile.finish_ms);
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });
    results = cluster_candidates(std::move(results));
    results.erase(std::remove_if(results.begin(), results.end(),
        [threshold](const auto& candidate) { return candidate.score < threshold; }), results.end());
    if (!results.empty() && !model.b7.px.empty()) {
        auto& best = results.front();
        double angle = best.angle > 180 ? best.angle - 360 : best.angle;
        const auto refined = b7_refine(image_bgr, model.b7, best.center[0], best.center[1],
                                       angle, best.scale);
        if (refined.valid >= kB7MinValidPts &&
            std::hypot(refined.x - best.center[0], refined.y - best.center[1]) <=
                2.0 * kB7SearchRadius) {
            apply_geometric_refinement(best, refined);
        }
    }
    if (results.size() > size_t(max_candidates)) results.resize(max_candidates);
    profile.final_candidates = int(results.size());
    if (trace) *trace = profile.snapshot();
    return results;
}

}  // namespace vfshape
