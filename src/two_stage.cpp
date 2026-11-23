// 两级匹配：粗到精窗口精修 + 跨候选聚类 + B7 几何精修。
// 训练见 two_stage_train.cpp，容器编解码见 container_codec.cpp。

#include "two_stage.hpp"
#include "matching_profile.hpp"
#include "rescue_score.hpp"
#include "coarse_image.hpp"
#include "two_stage_support.hpp"

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
#include "gpu/coarse_service.hpp"
#include "gpu/fine_service.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <optional>
#include <stdexcept>
#include <vector>

namespace vfshape {

namespace {

// gpu_seam=true 时两级接缝全走 GPU 服务（入口一次决策，整调用同后端）；GPU 路径
// 失败由 match_two_stage 包装层捕获回退
std::vector<vf_shape_candidate> match_two_stage_run(
    TwoStageModel& model, const cv::Mat& image_bgr, double threshold, int max_candidates,
    MatchTraceSnapshot* trace, bool gpu_seam) {
    MatchProfile profile;
    const int w = image_bgr.cols, h = image_bgr.rows;
    // 粗层：1/k 缩图（4096 接口限制在编排内自然放开——缩图与窗口均小）
    cv::Mat coarse_img;
    MatchProfile::measure(profile.coarse_resize_ms, [&] {
      resize_coarse_image(image_bgr, coarse_img, model.k);
    });
    std::vector<vf_shape_candidate> coarse_cands;
    // 异常穿透到 ABI boundary 报错：执行失败不得伪装成"无匹配"空结果
    coarse_cands = MatchProfile::measure(profile.coarse_ms, [&] {
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
        if (gpu_seam)
            return gpu::match_coarse_gpu(*model.coarse, coarse_img, std::max(45.0, threshold - 20.0));
#endif
        return match(*model.coarse, coarse_img, std::max(45.0, threshold - 20.0));
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
    const bool can_rescue = rescue_collect_gate(threshold) < threshold;
    std::array<RescueWindow, kMaxFineWindows> misses;
    auto refine_window = [&](size_t ci) {
        profile.windows_searched.fetch_add(1, std::memory_order_relaxed);
        // 剖析槽仅 2 个（MatchProfile 契约）：前两窗入槽，其余窗记到本地空槽
        MatchProfile::Window overflow_trace;
        MatchProfile::Window& trace_window =
            ci < profile.windows.size() ? profile.windows[ci] : overflow_trace;
        std::vector<vf_shape_candidate> window_results;
        // 窗口推导单一实现（主路径与 W6 预发射共用，同输入必同输出=预发射位级前提）
        struct DerivedWindow {
            bool valid = false;
            double cs = 1.0, ca = 0.0;
            const Model* fine = nullptr;
            const GroupSpec* spec = nullptr;
            int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            bool needs_scale = false;
            cv::Mat window, window_scaled;
        };
        auto derive_window = [&](size_t wi, MatchProfile::Window* trace) {
            DerivedWindow d;
            const auto& cc = coarse_cands[wi];
            d.cs = (cc.scale > 0) ? cc.scale : 1.0;
            d.ca = cc.angle;
            if (d.ca > 180) d.ca -= 360;
            d.fine = fine_for_angle(model, d.ca, &d.spec);
            const double half = fine_half * std::max(1.0, d.cs);
            d.x0 = int(std::max(0.0, cc.center[0] * model.k - half));
            d.y0 = int(std::max(0.0, cc.center[1] * model.k - half));
            d.x1 = int(std::min(double(w), d.x0 + 2 * half));
            d.y1 = int(std::min(double(h), d.y0 + 2 * half));
            if (d.x1 - d.x0 < 24 || d.y1 - d.y0 < 24) return d;
            d.window = image_bgr(cv::Rect(d.x0, d.y0, d.x1 - d.x0, d.y1 - d.y0));
            d.window_scaled = d.window;
            d.needs_scale = std::abs(d.cs - 1.0) > 0.02;
            if (d.needs_scale) {
                if (trace) {
                    MatchProfile::Timer resize_timer(trace->resize_ms);
                    d.window_scaled = scale_fine_window(d.window, d.cs);
                } else {
                    d.window_scaled = scale_fine_window(d.window, d.cs);
                }
            }
            d.valid = true;
            return d;
        };
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
        auto prefetch_key = [](const Model& m, const DerivedWindow& d, double th) {
            uint64_t h = 1469598103934665603ull;
            auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
            uint64_t cs_bits = 0, th_bits = 0;
            std::memcpy(&cs_bits, &d.cs, sizeof(cs_bits));
            std::memcpy(&th_bits, &th, sizeof(th_bits));
            mix(uintptr_t(&m));
            mix(uint64_t(unsigned(d.x0)));
            mix(uint64_t(unsigned(d.y0)));
            mix(uint64_t(unsigned(d.x1)));
            mix(uint64_t(unsigned(d.y1)));
            mix(cs_bits);
            mix(th_bits);
            return h;
        };
#endif
        const DerivedWindow dw = derive_window(ci, &trace_window);
        if (!dw.valid) return window_results;
        double cs = dw.cs;
        const GroupSpec* spec = dw.spec;
        const Model* fine_model = dw.fine;
        trace_window.width = dw.x1 - dw.x0; trace_window.height = dw.y1 - dw.y0;
        trace_window.scale = dw.cs; trace_window.angle = dw.ca;
        const cv::Mat& window = dw.window;

        const cv::Mat& window_scaled = dw.window_scaled;
        const bool needs_scale = dw.needs_scale;
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
        auto evaluate = [&](Model& m, const PreparedFrame& frame, const cv::Mat& gpu_image,
                            uint64_t pkey) {
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
            // GPU 路径：vendor PreparedFrame 不参与，全窗墙钟记 match_ms
            if (gpu_seam)
                return MatchProfile::measure(trace_window.match_ms, [&] {
                    return gpu::match_fine_window_gpu(m, gpu_image, fine_th, pkey);
                });
#endif
            const int ready = frame.features.preparedLevels();
            auto matches = MatchProfile::measure(trace_window.match_ms, [&] {
                return match_prepared(m, frame, fine_th);
            });
            trace_window.deferred_levels += frame.features.preparedLevels() - ready;
            return matches;
        };
        auto* selected = const_cast<Model*>(fine_model);
        bool selected_uses_prepared = !gpu_seam;
        PreparedFrame prepared;
        if (!gpu_seam) prepared = prepare(window_scaled);
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
        const uint64_t own_key = prefetch_key(*fine_model, dw, fine_th);
#else
        const uint64_t own_key = 0;
#endif
        refined = evaluate(*selected, prepared, window_scaled, own_key);
        if (best_score(refined) < threshold && needs_scale) {
            ++trace_window.scale_retries;
            PreparedFrame retry_frame;
            if (!gpu_seam) retry_frame = prepare(window);
            auto retry = evaluate(*selected, retry_frame, window, 0);
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
                const bool reuse = cfg.weak_th == original.weak_th &&
                                   cfg.strong_th == original.strong_th &&
                                   cfg.feature_count == original.feature_count;
                std::vector<vf_shape_candidate> retry;
                if (reuse || gpu_seam) {
                    // GPU 路径 prepared 无意义：两分支统一走 GPU 直调 alternate
                    //（同窗帧驻留复用，等价性论证见 W3 波2 报告；重试不领预发射）
                    retry = evaluate(alternate, prepared, effective, 0);
                } else {
                    retry = match(alternate, effective, fine_th);
                }
                if (best_score(retry) > best_score(refined)) {
                    refined = std::move(retry);
                    spec = &model.groups[gi];
                    selected = &alternate;
                    selected_uses_prepared = !gpu_seam && reuse;
                }
            }
        }
        profile.window_best[ci] = best_score(refined);
        bool eligible = can_rescue && profile.window_best[ci] < threshold;
        for (auto& r : refined) {
            reproject_candidate(r, dw.x0, dw.y0, *spec, cs);
            if (r.pose_ambiguous || r.center[0] < 0 || r.center[1] < 0) eligible = false;
            window_results.push_back(r);
        }
        if (eligible) {
            auto& miss = misses[ci];
            miss = {selected, spec, window, std::nullopt, dw.x0, dw.y0, cs, refined.empty(),
                    std::abs(cs - 1.0) > 0.02};
            if (selected_uses_prepared && ci < kMaxCachedRescueWindows &&
                size_t(prepared.padded_width) * prepared.padded_height <= kMaxCachedRescuePixels) {
                miss.source = effective;
                miss.resize_source = false;
                miss.prepared.emplace(std::move(prepared));
            }
        }
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
        // W6 窗批合并：预发射下一窗（H2D+全链入流，单槽待领）。决策序零改动——
        // 早停不跑则发射产物作废复用面；PARALLEL 构建下窗 1 由 async 线程自理
        //（本点预发射白跑反优化），仅串行段（ci>=1）预发。单窗内发射序不变。
        if (gpu_seam && ci + 1 < use && gpu::window_prefetch_enabled()
#if defined(VF_SHAPE_PARALLEL_WINDOWS) && VF_SHAPE_PARALLEL_WINDOWS
            && ci >= 1
#endif
        ) {
            const DerivedWindow nd = derive_window(ci + 1, nullptr);
            if (nd.valid)
                gpu::fine_prefetch(*const_cast<Model*>(nd.fine), nd.window_scaled, fine_th,
                                   prefetch_key(*nd.fine, nd, fine_th));
        }
#endif
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
    const bool ambiguous = std::any_of(results.begin(), results.end(), [](const auto& candidate) {
        return candidate.pose_ambiguous != 0;
    });
    results.erase(std::remove_if(results.begin(), results.end(),
        [threshold](const auto& candidate) { return candidate.score < threshold; }), results.end());
    if (results.empty() && !ambiguous) results = rescue_misses(misses, fine_th, threshold, profile);
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

}  // namespace

// —— 两级匹配（Python TwoStageMatcher._match_locked 对齐）——
// auto 态（W4 默认）：GPU 路径任一步编码错误 → 连续失败计数+1 并整调用 CPU 重跑
//（输入纯函数幂等，W3 已证逐位等价），不向调用方报错；连续 3 次失败转粘性禁用
// 至进程结束。gpu 态失败直报（对拍语义不变），cpu 态不触 GPU。
std::vector<vf_shape_candidate> match_two_stage(
    TwoStageModel& model, const cv::Mat& image_bgr, double threshold, int max_candidates,
    MatchTraceSnapshot* trace) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
    const gpu::Backend mode = gpu::current_backend();
    if (mode == gpu::Backend::Auto) {
        if (gpu::resolve_backend() == gpu::Backend::Gpu) {
            try {
                auto results = match_two_stage_run(model, image_bgr, threshold, max_candidates,
                                                   trace, true);
                gpu::note_gpu_call_ok();
                return results;
            } catch (const std::invalid_argument&) {
                // 仅 GPU 故障类（INTERNAL）计失败并回退；输入性编码错误（搜索
                // 预算等同码同参在 CPU 路径必然同样抛出）直抛调用方，不计数
                // ——否则连续无效输入会粘性禁用 GPU（语义错误）
                if (g_last_error.code != int32_t(VF_SHAPE_INTERNAL)) throw;
                gpu::note_gpu_call_failed();
                return match_two_stage_run(model, image_bgr, threshold, max_candidates, trace,
                                           false);
            }
        }
    } else if (mode == gpu::Backend::Gpu) {
        return match_two_stage_run(model, image_bgr, threshold, max_candidates, trace, true);
    }
#endif
    return match_two_stage_run(model, image_bgr, threshold, max_candidates, trace, false);
}

}  // namespace vfshape
