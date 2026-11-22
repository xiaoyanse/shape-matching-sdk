#pragma once
// GPU 服务层共享内核（W3 波2 自 coarse_service.cpp 提取）：进程级单上下文状态、
// 双模型驻留槽（[0]=coarse/[1]=fine 窗，各自指纹失效判定）、帧槽容量制 ensure、
// 失败编码错误直报。仅 src/gpu 服务层翻译单元包含，不进公共 ABI。
#include "gpu_backend.hpp"

#include "../error_code.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
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

// —— W4 回退兜底状态面（进程级，粗精两服务共享；不进公共 ABI）——

// 日志 sink 可注入（测试捕获）；缺省 stderr 单行，消息自带 "vfshape: " 前缀
using GpuLogSink = void (*)(const char*);

inline GpuLogSink& log_sink() {
    static GpuLogSink value = nullptr;
    return value;
}

inline void gpu_log(const char* message) {
    if (const GpuLogSink sink = log_sink()) sink(message);
    else std::fprintf(stderr, "%s\n", message);
}

// 测试失败注入（仅测试目标置位；原子面为并行窗并发扣减所需）
struct TestHooks {
    int inject_probe_fail_branch = 0;       // 1/2/3 = probe 三分支注入
    int inject_init_fail = 0;               // 首次上下文创建注入失败次数（互斥内扣减）
    std::atomic<int> inject_chain_fail{0};  // 链路调用点注入失败次数
};

inline TestHooks& test_hooks() {
    static TestHooks value;
    return value;
}

// 探测/回退粘性状态：探测结论与粘性禁用存续至进程结束，运行中失败按
// 连续计数（成功调用清零），两次字段读写跨线程，故用原子
struct FallbackState {
    std::atomic<bool> probed{false};
    std::atomic<bool> gpu_usable{false};
    std::atomic<int> chain_fail_streak{0};
    std::atomic<bool> sticky_disabled{false};
};

inline FallbackState& fallback() {
    static FallbackState value;
    return value;
}

inline int consume_probe_fail_branch() {
    const int branch = test_hooks().inject_probe_fail_branch;
    test_hooks().inject_probe_fail_branch = 0;
    return branch;
}

inline bool consume_init_fail_hook() {
    int& remaining = test_hooks().inject_init_fail;
    if (remaining <= 0) return false;
    --remaining;
    return true;
}

inline bool consume_chain_fail_hook() {
    std::atomic<int>& remaining = test_hooks().inject_chain_fail;
    int current = remaining.load(std::memory_order_acquire);
    while (current > 0 &&
           !remaining.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel)) {
    }
    return current > 0;
}

// 三分支设备探测（进程内一次，粘性）：无卡/驱动缺、算力集外、上下文初始化
// 失败 → 永久 CPU + stderr 单行注明原因。env=cpu 时调用方旁路本函数
inline void probe_device() {
    FallbackState& fb = fallback();
    std::lock_guard<std::mutex> lock(service_mutex());
    if (fb.probed.load(std::memory_order_acquire)) return;
    fb.probed.store(true, std::memory_order_release);
    if (const int branch = consume_probe_fail_branch()) {
        const char* reason = branch == 1 ? "no cuda device: injected probe failure"
            : branch == 2 ? "sm_00 outside supported set {86,89,120}: injected probe failure"
                          : "context init failed: injected probe failure";
        char line[128];
        std::snprintf(line, sizeof line, "vfshape: gpu fallback (%s)", reason);
        gpu_log(line);
        return;
    }
    std::string name, err;
    int major = 0, minor = 0;
    if (!device_info(name, major, minor, err)) {
        gpu_log(("vfshape: gpu fallback (no cuda device: " + err + ")").c_str());
        return;
    }
    const int sm = major * 10 + minor;
    if (sm != 86 && sm != 89 && sm != 120) {
        char line[96];
        std::snprintf(line, sizeof line,
                      "vfshape: gpu fallback (sm_%d outside supported set {86,89,120})", sm);
        gpu_log(line);
        return;
    }
    GpuContext* probe_ctx =
        create_context(64, 64, 4, 256, kContextCandidates, 256, err);
    if (probe_ctx == nullptr) {
        gpu_log(("vfshape: gpu fallback (context init failed: " + err + ")").c_str());
        return;
    }
    destroy_context(probe_ctx);
    fb.gpu_usable.store(true, std::memory_order_release);
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
    if (s.ctx == nullptr && consume_init_fail_hook()) {
        err = "injected context init failure";
        return false;
    }
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
