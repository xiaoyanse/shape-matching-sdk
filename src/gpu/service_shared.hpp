#pragma once
// GPU 服务层共享内核（W3 波2 自 coarse_service.cpp 提取）：进程级单上下文状态、
// 双模型驻留槽（[0]=coarse/[1]=fine 窗，各自指纹失效判定）、帧槽容量制 ensure、
// 失败编码错误直报。仅 src/gpu 服务层翻译单元包含，不进公共 ABI。
#include "gpu_backend.hpp"

#include "../error_code.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

namespace vfshape::gpu {
namespace svc {

constexpr int kContextCandidates = 256;  // ≥W3 窗分分流阈值：免首窗批量触发重建

struct SlotResidency {
    const void* model_id = nullptr;
    int poses = 0, features = 0;
    long long checksum = 0;
};

struct ServiceState {
    GpuContext* ctx = nullptr;
    int cap_w = 0, cap_h = 0, cap_poses = 0, cap_feats = 0;
    // 指纹按 [model_slot][frame_slot] 双份：物理驻留随 frame_slot 复制，单份指纹
    // 会对另一工作面的空副本假命中（跳上传→链路报未驻留）
    SlotResidency slot[2][2];
};

inline ServiceState& state() {
    static ServiceState value;
    return value;
}

inline std::mutex& service_mutex() {
    static std::mutex value;
    return value;
}

// 流水化管线（W3 波2 补完）：专用流 + frame_slot 交替分配 + 在途 job 计数。
// 双工作面（face/response/centers/scores/fine 特征 ×2）使并发 job 设备写全隔离，
// 发射无依赖链、end 各自同步自身事件；ensure_context 重建与在途 job 冲突时直报。
struct PipelineState {
    void* stream = nullptr;
    bool stream_ready = false;
    bool slot_busy[2] = {false, false};  // 槽分配互斥下标记在途占用（双槽满=直报）
    std::atomic<int> active_jobs{0};
};

inline PipelineState& pipeline() {
    static PipelineState value;
    return value;
}

[[noreturn]] inline void gpu_fail(const std::string& message) {
    fail(VF_SHAPE_INTERNAL, {}, message.c_str());
}

// ensure_context 重建期间禁止在途 job（重建会销毁全部工作面）——正常时序下
// 粗层先建、首窗扩容后容量恒定，此防护只拦异常交错
inline void guard_rebuild_against_inflight() {
    if (pipeline().active_jobs.load(std::memory_order_acquire) > 0)
        gpu_fail("context rebuild while fine window job in flight");
}

struct ResidencyFingerprint {
    int poses = 0, features = 0;
    long long checksum = 0;
};

// 指纹带特征内容校验和：仅 pose 数+特征总数时，同址复用的新模型会假命中陈旧驻留
inline ResidencyFingerprint fingerprint_of(const Model& model) {
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

// 容量制上下文：帧档/位姿/特征任一超容量才重建，新容量取 max（只增不减）；
// 帧档缩小不再触发重建（W2-B 严格相等契约的 W3 放宽，超容量仍沿 ensure 重建）。
// 重建即两槽驻留全部失效（设备缓冲重分配）。
inline bool ensure_context(int pw, int ph, int poses, int features, std::string& err) {
    ServiceState& s = state();
    const bool needs_rebuild = s.ctx == nullptr || pw > s.cap_w || ph > s.cap_h ||
                               poses > s.cap_poses || features > s.cap_feats;
    if (!needs_rebuild) return true;
    guard_rebuild_against_inflight();
    const int want_w = std::max(s.cap_w, pw);
    const int want_h = std::max(s.cap_h, ph);
    const int want_poses = std::max(s.cap_poses, poses);
    const int want_feats = std::max(s.cap_feats, features);
    destroy_context(s.ctx);
    s.ctx = create_context(want_w, want_h, want_poses, want_feats, kContextCandidates,
                           (want_w / 4) * (want_h / 4), err);
    if (s.ctx == nullptr) {
        s = ServiceState{};
        return false;
    }
    s.cap_w = want_w;
    s.cap_h = want_h;
    s.cap_poses = want_poses;
    s.cap_feats = want_feats;
    for (int ms = 0; ms < 2; ++ms)
        for (int fs = 0; fs < 2; ++fs) s.slot[ms][fs] = {};
    return true;
}

// slot=1（fine 窗）按 frame_slot 取双份特征面之一；slot=0（coarse）frame_slot 恒 0
inline bool ensure_residency(int slot, const Model& model, const ResidencyFingerprint& fp,
                             std::string& err, int frame_slot = 0) {
    ServiceState& s = state();
    const SlotResidency& r = s.slot[slot][slot == 1 ? frame_slot : 0];
    if (s.ctx != nullptr && r.model_id == &model && r.poses == fp.poses &&
        r.features == fp.features && r.checksum == fp.checksum)
        return true;
    if (!upload_model_features(s.ctx, model, err, nullptr, slot, frame_slot)) return false;
    s.slot[slot][slot == 1 ? frame_slot : 0] = {&model, fp.poses, fp.features, fp.checksum};
    return true;
}

}  // namespace svc
}  // namespace vfshape::gpu
