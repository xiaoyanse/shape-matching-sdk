// 粗层 GPU 服务实现（coarse_service.hpp 语义）：进程级单上下文（共享状态见
// service_shared.hpp，W3 起与精修窗服务同体）+ Model 指纹守驻留缓存 + 帧路径
// pad→upload→chain(VendorExact)→extract，收尾与 match() 同链（collect_hit_objects
// 复用）。失败一律编码错误直报。
#include "coarse_service.hpp"

#include "coarse_extract.hpp"
#include "coarse_kernels.hpp"
#include "error_code.hpp"
#include "gpu_backend.hpp"
#include "service_shared.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace vfshape::gpu {
namespace {

CoarseServiceStats& stats_slot() {
    static CoarseServiceStats value;
    return value;
}

bool build_extract_poses(const Model& coarse, ExtractPoses& poses, std::string& err) {
    return svc::build_extract_poses(coarse, poses, err);
}

// matcher.cpp 匿名空间同名实现不可达，最小平移（两处需同步改）
vf_shape_candidate candidate(const Model& model, const line2Dup::Match& hit, int padding) {
    vf_shape_candidate result{};
    result.score = hit.similarity; result.pose_id = hit.template_id;
    const auto& pose = model.poses.at(hit.template_id);
    const auto& templ = model.detector.getTemplates("shape", hit.template_id).front();
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

}  // namespace

Backend current_backend() {
    const char* value = std::getenv("VF_SHAPE_BACKEND");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0)
        return Backend::Auto;
    return std::strcmp(value, "gpu") == 0 ? Backend::Gpu : Backend::Cpu;
}

Backend resolve_backend() {
    const Backend mode = current_backend();
    if (mode != Backend::Auto) return mode;
    svc::FallbackState& fb = svc::fallback();
    if (fb.sticky_disabled.load(std::memory_order_acquire)) return Backend::Cpu;
    if (!fb.probed.load(std::memory_order_acquire)) svc::probe_device();
    return fb.gpu_usable.load(std::memory_order_acquire) ? Backend::Gpu : Backend::Cpu;
}

void note_gpu_call_ok() {
    svc::fallback().chain_fail_streak.store(0, std::memory_order_relaxed);
}

void note_gpu_call_failed() {
    svc::FallbackState& fb = svc::fallback();
    const int streak = fb.chain_fail_streak.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (streak >= 3 && !fb.sticky_disabled.exchange(true, std::memory_order_acq_rel))
        svc::gpu_log("vfshape: gpu fallback (sticky disable: 3 consecutive runtime failures)");
}

const CoarseServiceStats& last_call_stats() {
    return stats_slot();
}

std::vector<vf_shape_candidate> match_coarse_gpu(Model& coarse, const cv::Mat& small,
                                                 double threshold) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    validate_search_budget(coarse, small.cols, small.rows);

    std::lock_guard<std::mutex> lock(svc::service_mutex());
    stats_slot() = {};

    const svc::ResidencyFingerprint fp = svc::fingerprint_of(coarse);
    const cv::Mat padded = pad_for_detector(small);
    const int pw = padded.cols, ph = padded.rows;

    std::string err;
    if (!svc::ensure_context(pw, ph, fp.poses, fp.features, err)) svc::gpu_fail(err);
    if (!svc::ensure_residency(0, coarse, fp, err)) svc::gpu_fail(err);
    if (!upload_frame_capacity(svc::state().ctx, padded.data, pw, ph, err)) svc::gpu_fail(err);

    CoarseResult result;
    CoarseTiming timing;
    if (svc::consume_chain_fail_hook()) svc::gpu_fail("injected chain failure");
    // 精益模式：提取只消费 response[1]+spread[0]，跳过 L0 全图响应与冗余回读
    if (!run_coarse_chain(svc::state().ctx, fp.poses, coarse.detector.tuned_weak,
                          CoarseMode::VendorExact, true, result, &timing, err,
                          /*need_l0_response=*/false))
        svc::gpu_fail(err);

    ExtractPoses extract_poses;
    if (!build_extract_poses(coarse, extract_poses, err)) svc::gpu_fail(err);
    ExtractResult extract_out;
    if (!extract_coarse_candidates(result.response[1], result.resp_w[1], result.resp_h[1],
                                   result.maps.spread[0], result.maps.w[0], result.maps.h[0],
                                   extract_poses, pw, ph, float(threshold), extract_out, err)) {
        // 预算溢出与 CPU 路径同码（match_prepared 同额 200000 报 SEARCH_BUDGET），
        // 其余提取失败属 GPU 后端故障按 INTERNAL
        if (err == "candidate budget exceeded")
            fail(VF_SHAPE_SEARCH_BUDGET, {double(extract_out.initial_count), 200000.0},
                 err.c_str());
        svc::gpu_fail(err);
    }

    CoarseServiceStats& stats = stats_slot();
    stats.gpu_chain_ms = timing.wall_ms;
    stats.extract_ms = extract_out.extract_ms;
    stats.initial_count = extract_out.initial_count;

    PreparedFrame frame;  // 收尾只消费帧宽高（center 越界过滤），features 留空
    frame.width = small.cols;
    frame.height = small.rows;
    frame.padded_width = pw;
    frame.padded_height = ph;
    return collect_hit_objects(coarse, extract_out.matches, frame);
}

}  // namespace vfshape::gpu
