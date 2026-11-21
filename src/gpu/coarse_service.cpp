// 粗层 GPU 服务实现（coarse_service.hpp 语义）：上下文容量制懒创建、Model
// 指纹守驻留缓存、帧路径 pad→upload→chain(VendorExact)→extract，收尾与
// match() 同链（collect_hit_objects 复用）。失败一律编码错误直报。
#include "coarse_service.hpp"

#include "coarse_extract.hpp"
#include "coarse_kernels.hpp"
#include "error_code.hpp"
#include "gpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace vfshape::gpu {
namespace {

constexpr int kContextCandidates = 16;

Backend backend_from_env() {
    const char* value = std::getenv("VF_SHAPE_BACKEND");
    return value != nullptr && std::strcmp(value, "gpu") == 0 ? Backend::Gpu : Backend::Cpu;
}

struct ServiceState {
    GpuContext* ctx = nullptr;
    int cap_w = 0, cap_h = 0, cap_poses = 0, cap_feats = 0;
    const void* model_id = nullptr;
    int resident_poses = 0, resident_features = 0;
    long long resident_checksum = 0;
};

ServiceState& state() {
    static ServiceState value;
    return value;
}

std::mutex& service_mutex() {
    static std::mutex value;
    return value;
}

CoarseServiceStats& stats_slot() {
    static CoarseServiceStats value;
    return value;
}

[[noreturn]] void gpu_fail(const std::string& message) {
    fail(VF_SHAPE_INTERNAL, {}, message.c_str());
}

struct ResidencyFingerprint {
    int poses = 0, features = 0;
    long long checksum = 0;
};

// 指纹带特征内容校验和：仅 pose 数+特征总数时，同址复用的新模型会假命中陈旧驻留
ResidencyFingerprint fingerprint_of(const Model& model) {
    ResidencyFingerprint fp;
    fp.poses = int(model.poses.size());
    for (int id = 0; id < fp.poses; ++id)
        for (const auto& level : model.detector.getTemplates("shape", id))
            for (const auto& f : level.features) {
                ++fp.features;
                fp.checksum += f.x * 2 + f.y * 4 + f.label;
            }
    return fp;
}

// upload_frame 要求帧档与上下文 L0 档严格相等：帧槽尺寸恒取当前帧档，帧档
// 变化（含缩小）一律重建；位姿/特征容量只增不减；重建即驻留失效（重传）
bool ensure_context(int pw, int ph, int poses, int features, std::string& err) {
    ServiceState& s = state();
    const bool needs_rebuild = s.ctx == nullptr || pw != s.cap_w || ph != s.cap_h ||
                               poses > s.cap_poses || features > s.cap_feats;
    if (!needs_rebuild) return true;
    const int want_poses = std::max(s.cap_poses, poses);
    const int want_feats = std::max(s.cap_feats, features);
    destroy_context(s.ctx);
    s.ctx = create_context(pw, ph, want_poses, want_feats, kContextCandidates,
                           (pw / 4) * (ph / 4), err);
    if (s.ctx == nullptr) {
        s = ServiceState{};
        return false;
    }
    s.cap_w = pw;
    s.cap_h = ph;
    s.cap_poses = want_poses;
    s.cap_feats = want_feats;
    s.model_id = nullptr;
    return true;
}

bool ensure_residency(const Model& coarse, const ResidencyFingerprint& fp, std::string& err) {
    ServiceState& s = state();
    if (s.ctx != nullptr && s.model_id == &coarse && s.resident_poses == fp.poses &&
        s.resident_features == fp.features && s.resident_checksum == fp.checksum)
        return true;
    if (!upload_model_features(s.ctx, coarse, err)) return false;
    s.model_id = &coarse;
    s.resident_poses = fp.poses;
    s.resident_features = fp.features;
    s.resident_checksum = fp.checksum;
    return true;
}

bool build_extract_poses(const Model& coarse, ExtractPoses& poses, std::string& err) {
    const int n = int(coarse.poses.size());
    poses.class_id = "shape";
    poses.level1.resize(size_t(n));
    poses.level0.resize(size_t(n));
    for (int id = 0; id < n; ++id) {
        const auto& pyramid = coarse.detector.getTemplates("shape", id);
        if (pyramid.size() != 2 || pyramid[0].pyramid_level != 0 ||
            pyramid[1].pyramid_level != 1) {
            err = "template pyramid layout mismatch";
            return false;
        }
        for (int level = 0; level < 2; ++level) {
            const auto& templ = pyramid[size_t(level)];
            auto& dst = level == 1 ? poses.level1[size_t(id)] : poses.level0[size_t(id)];
            dst.width = templ.width;
            dst.height = templ.height;
            dst.feats.reserve(templ.features.size());
            for (const auto& f : templ.features) dst.feats.push_back({f.x, f.y, f.label});
        }
    }
    return true;
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
    return backend_from_env();
}

const CoarseServiceStats& last_call_stats() {
    return stats_slot();
}

std::vector<vf_shape_candidate> match_coarse_gpu(Model& coarse, const cv::Mat& small,
                                                 double threshold) {
    require_coded(std::isfinite(threshold) && threshold > 0 && threshold <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", threshold);
    validate_search_budget(coarse, small.cols, small.rows);

    std::lock_guard<std::mutex> lock(service_mutex());
    stats_slot() = {};

    const ResidencyFingerprint fp = fingerprint_of(coarse);
    const cv::Mat padded = pad_for_detector(small);
    const int pw = padded.cols, ph = padded.rows;

    std::string err;
    if (!ensure_context(pw, ph, fp.poses, fp.features, err)) gpu_fail(err);
    if (!ensure_residency(coarse, fp, err)) gpu_fail(err);
    if (!upload_frame(state().ctx, padded.data, pw, ph, err)) gpu_fail(err);

    CoarseResult result;
    CoarseTiming timing;
    // 精益模式：提取只消费 response[1]+spread[0]，跳过 L0 全图响应与冗余回读
    if (!run_coarse_chain(state().ctx, fp.poses, coarse.detector.tuned_weak,
                          CoarseMode::VendorExact, true, result, &timing, err,
                          /*need_l0_response=*/false))
        gpu_fail(err);

    ExtractPoses extract_poses;
    if (!build_extract_poses(coarse, extract_poses, err)) gpu_fail(err);
    ExtractResult extract_out;
    if (!extract_coarse_candidates(result.response[1], result.resp_w[1], result.resp_h[1],
                                   result.maps.spread[0], result.maps.w[0], result.maps.h[0],
                                   extract_poses, pw, ph, float(threshold), extract_out, err)) {
        // 预算溢出与 CPU 路径同码（match_prepared 同额 200000 报 SEARCH_BUDGET），
        // 其余提取失败属 GPU 后端故障按 INTERNAL
        if (err == "candidate budget exceeded")
            fail(VF_SHAPE_SEARCH_BUDGET, {double(extract_out.initial_count), 200000.0},
                 err.c_str());
        gpu_fail(err);
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
