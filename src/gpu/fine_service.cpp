// 精修窗 GPU 服务实现（fine_service.hpp 语义，W3 流水化）：begin 发射段
// （H2D/驻留/链路全入专用流后即返回）+ end 同步段（等 e_resp1 → 宿主 initial
// 扫描（与 L0 链重叠）→ 窗分内核流内发射（分块）→ 等 e_d2h → 外部窗分消费收尾）。
// 双工作面（frame_slot ×2）使并发 job 设备写全隔离——PARALLEL 双窗下窗 1 的链路
// 与窗 0 的宿主提取真重叠；串行调用退化为窗内流水。回读走 pinned 池（按槽借用，
// 同槽由槽分配互斥保证无并发）。同内核同数据同发射序=位级零变化（护栏=全部既有
// 对拍测试）。失败编码错误直报。
#include "fine_service.hpp"

#include "coarse_extract.hpp"
#include "coarse_kernels.hpp"
#include "error_code.hpp"
#include "fine_windows.hpp"
#include "service_shared.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace vfshape::gpu {
namespace {

// 窗分分流阈值（候选量，0=恒走内核）：harness 实测宿主路径每次调用必付线性内存
// 预烙 2.15ms/窗（608 档）而内核路径整体旁路之，纯窗分交叉带 32→64 候选——
// 路径级对比 GPU 全胜，恒走内核；保留常量供复核调档
constexpr int kWindowKernelMinCandidates = 0;

FineWindowStats& stats_slot() {
    static FineWindowStats value;
    return value;
}

std::mutex& stats_mutex() {
    static std::mutex value;
    return value;
}

// pinned 回读池（按 frame_slot 借用：同槽由槽分配互斥保证串行）
struct ReadbackPool {
    uint16_t* resp1 = nullptr;
    size_t resp1_cap = 0;
    uint16_t* scores = nullptr;
    size_t scores_cap = 0;
};

ReadbackPool& pool(int fs) {
    static ReadbackPool pools[2];
    return pools[fs];
}

bool ensure_pinned(uint16_t** ptr, size_t* cap, size_t bytes, std::string& err) {
    if (*ptr && *cap >= bytes) return true;
    if (*ptr) pinned_free(*ptr);
    *ptr = static_cast<uint16_t*>(pinned_alloc(bytes, err));
    if (!*ptr) {
        *cap = 0;
        return false;
    }
    *cap = bytes;
    return true;
}

bool build_extract_poses(const Model& fine, ExtractPoses& poses, std::string& err) {
    const int n = int(fine.poses.size());
    poses.class_id = "shape";
    poses.level1.resize(size_t(n));
    poses.level0.resize(size_t(n));
    for (int id = 0; id < n; ++id) {
        const auto& pyramid = fine.detector.getTemplates("shape", id);
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

// —— 发射段实体：事件 RAII + 在途/槽占用计数回收 ——
struct FineJob {
    int fs = 0;
    int poses = 0;
    cv::Mat padded;  // 异步 H2D 源存活保障（e_resp1 同步前不得析构）
    uint16_t* resp1_pinned = nullptr;
    size_t resp1_bytes = 0;
    int resp_w1 = 0, resp_h1 = 0;
    int pw = 0, ph = 0;
    const uint8_t* spread0_dev = nullptr;
    int sw = 0, sh = 0;
    uint16_t* scores_pinned = nullptr;
    size_t scores_bytes = 0;
    void* e_chain_begin = nullptr;
    void* e_resp1 = nullptr;
    void* e_chain_end = nullptr;
    void* e_k0 = nullptr;
    void* e_k1 = nullptr;
    void* e_d2h = nullptr;

    ~FineJob() {
        event_destroy(e_chain_begin);
        event_destroy(e_resp1);
        event_destroy(e_chain_end);
        event_destroy(e_k0);
        event_destroy(e_k1);
        event_destroy(e_d2h);
        std::lock_guard<std::mutex> lock(svc::service_mutex());
        svc::pipeline().slot_busy[fs] = false;
        svc::pipeline().active_jobs.fetch_sub(1, std::memory_order_acq_rel);
    }
};

bool create_job_events(FineJob& job, std::string& err) {
    job.e_chain_begin = event_create(err);
    if (!job.e_chain_begin) return false;
    job.e_resp1 = event_create(err);
    if (!job.e_resp1) return false;
    job.e_chain_end = event_create(err);
    if (!job.e_chain_end) return false;
    job.e_k0 = event_create(err);
    if (!job.e_k0) return false;
    job.e_k1 = event_create(err);
    if (!job.e_k1) return false;
    job.e_d2h = event_create(err);
    return job.e_d2h != nullptr;
}

}  // namespace

const FineWindowStats& last_fine_stats() {
    return stats_slot();
}

// 发射段：校验/指纹/上下文/槽分配（service_mutex 短临界）→ H2D/驻留/链路发射入
// 流后即返回（进行中）
static std::unique_ptr<FineJob> fine_window_begin(Model& fine, const cv::Mat& window,
                                                  double fine_th) {
    const svc::ResidencyFingerprint fp = svc::fingerprint_of(fine);
    const cv::Mat padded = pad_for_detector(window);
    const int pw = padded.cols, ph = padded.rows;
    // match_prepared 同式前置：搜索预算按补边档 16 格 × 位姿数核算
    const uint64_t blocks = uint64_t(pw / 16) * uint64_t(ph / 16) * uint64_t(fp.poses);
    require_coded(blocks <= 4000000, VF_SHAPE_SEARCH_BUDGET, "matching search budget exceeded",
                  double(blocks), 4000000.0);

    auto job = std::make_unique<FineJob>();
    {
        std::string err;
        std::lock_guard<std::mutex> lock(svc::service_mutex());
        if (!svc::ensure_context(pw, ph, fp.poses, fp.features, err)) svc::gpu_fail(err);
        svc::PipelineState& pipe = svc::pipeline();
        if (!pipe.stream_ready) {
            pipe.stream = stream_create(err);
            if (!pipe.stream) svc::gpu_fail(err);
            pipe.stream_ready = true;
        }
        if (pipe.slot_busy[0] && pipe.slot_busy[1])
            svc::gpu_fail("no free frame slot for concurrent fine window job");
        job->fs = pipe.slot_busy[0] ? 1 : 0;
        pipe.slot_busy[job->fs] = true;
        pipe.active_jobs.fetch_add(1, std::memory_order_acq_rel);
    }
    job->padded = padded;
    job->poses = int(fp.poses);
    job->pw = pw;
    job->ph = ph;
    std::string err;
    if (!create_job_events(*job, err)) svc::gpu_fail(err);
    if (!svc::ensure_residency(1, fine, fp, err, job->fs)) svc::gpu_fail(err);
    ReadbackPool& pl = pool(job->fs);
    const int cells = (pw / 2 / 8) * (ph / 2 / 8);
    job->resp1_bytes = size_t(fp.poses) * size_t(cells) * sizeof(uint16_t);
    if (!ensure_pinned(&pl.resp1, &pl.resp1_cap, job->resp1_bytes, err)) svc::gpu_fail(err);
    job->resp1_pinned = pl.resp1;
    if (!upload_frame_capacity_async(svc::state().ctx, job->padded.data, pw, ph,
                                     svc::pipeline().stream, err, job->fs))
        svc::gpu_fail(err);
    if (svc::consume_chain_fail_hook()) svc::gpu_fail("injected chain failure");
    if (!run_coarse_chain_async(svc::state().ctx, fp.poses, fine.detector.tuned_weak,
                                svc::pipeline().stream, job->resp1_pinned, job->resp1_bytes,
                                job->resp_w1, job->resp_h1, &job->spread0_dev, job->sw,
                                job->sh, job->e_chain_begin, job->e_resp1, job->e_chain_end,
                                /*model_slot=*/1, job->fs, err))
        svc::gpu_fail(err);
    return job;
}

// 同步段：等 e_resp1 → resp1 拷出 + initial 扫描（与 L0 链重叠）→ 窗分内核流内
// 发射（按候选容量分块）→ 等 e_d2h → 外部窗分消费收尾
static std::vector<vf_shape_candidate> fine_window_end(std::unique_ptr<FineJob> job,
                                                       Model& fine, const cv::Mat& window,
                                                       double fine_th) {
    std::string err;
    ExtractPoses extract_poses;
    if (!build_extract_poses(fine, extract_poses, err)) svc::gpu_fail(err);

    const auto scan_t0 = std::chrono::steady_clock::now();
    if (!event_sync(job->e_resp1, err)) svc::gpu_fail(err);
    const int cells = job->resp_w1 * job->resp_h1;
    std::vector<std::vector<uint16_t>> resp1(size_t(job->poses));
    for (int id = 0; id < job->poses; ++id)
        resp1[size_t(id)].assign(job->resp1_pinned + size_t(id) * cells,
                                 job->resp1_pinned + size_t(id + 1) * cells);
    std::vector<InitialCandidate> cands;
    if (!extract_initial_candidates(resp1, job->resp_w1, job->resp_h1, extract_poses, job->pw,
                                    job->ph, float(fine_th), cands, err)) {
        if (err == "candidate budget exceeded")
            fail(VF_SHAPE_SEARCH_BUDGET, {200000.0, 200000.0}, err.c_str());
        svc::gpu_fail(err);
    }
    const double scan_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scan_t0)
            .count();

    const bool kernel_path = int(cands.size()) >= kWindowKernelMinCandidates;
    double kernel_ms = 0, d2h_ms = 0;
    std::vector<uint16_t> scores;
    if (kernel_path && !cands.empty()) {
        scores.reserve(cands.size() * size_t(256));
        std::vector<WindowCandidate> wc(cands.size());
        std::vector<int> cp(cands.size());
        for (size_t i = 0; i < cands.size(); ++i) {
            wc[i] = {cands[i].x, cands[i].y};
            cp[i] = cands[i].pose_id;
        }
        ReadbackPool& pl = pool(job->fs);
        const int step = std::max(1, context_candidate_capacity(svc::state().ctx));
        for (size_t begin = 0; begin < cands.size(); begin += size_t(step)) {
            const int n = int(std::min(size_t(step), cands.size() - begin));
            job->scores_bytes = size_t(n) * 256 * sizeof(uint16_t);
            if (!ensure_pinned(&pl.scores, &pl.scores_cap, job->scores_bytes, err))
                svc::gpu_fail(err);
            job->scores_pinned = pl.scores;
            if (svc::consume_chain_fail_hook()) svc::gpu_fail("injected chain failure");
            if (!run_window_scores_async(svc::state().ctx, nullptr, job->pw, job->ph,
                                         context_pose_capacity(svc::state().ctx),
                                         wc.data() + begin, cp.data() + begin, n,
                                         job->scores_pinned, job->scores_bytes, job->e_k0,
                                         job->e_k1, job->e_d2h, 1, svc::pipeline().stream,
                                         err,
                                         job->fs))
                svc::gpu_fail(err);
            if (!event_sync(job->e_d2h, err)) svc::gpu_fail(err);
            bool ok_k = false, ok_d = false;
            kernel_ms += event_elapsed_ms(job->e_k0, job->e_k1, ok_k);
            d2h_ms += event_elapsed_ms(job->e_k1, job->e_d2h, ok_d);
            scores.insert(scores.end(), job->scores_pinned,
                          job->scores_pinned + size_t(n) * 256);
        }
    }

    // 外部窗分路径不消费宿主 spread 内容（仅尺寸校验），链路 spread 留设备端供
    // 窗分内核直读，宿主侧仅按尺寸占位
    const auto extract_t0 = std::chrono::steady_clock::now();
    ExtractResult extract_out;
    std::vector<uint8_t> spread_host(size_t(job->sw) * size_t(job->sh));
    if (!extract_coarse_candidates(resp1, job->resp_w1, job->resp_h1, spread_host, job->sw,
                                   job->sh, extract_poses, job->pw, job->ph, float(fine_th),
                                   extract_out, err, kernel_path ? &scores : nullptr))
        svc::gpu_fail(err);

    {
        std::lock_guard<std::mutex> lock(stats_mutex());
        if (!event_sync(job->e_chain_end, err)) svc::gpu_fail(err);
        bool ok_chain = false;
        const double chain_ms =
            event_elapsed_ms(job->e_chain_begin, job->e_chain_end, ok_chain);
        FineWindowStats& stats = stats_slot();
        stats.chain_ms = ok_chain ? chain_ms : 0.0;
        stats.extract_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      extract_t0).count();
        stats.initial_scan_ms = scan_ms;
        stats.window_kernel_ms = kernel_ms;
        stats.scores_d2h_ms = d2h_ms;
        stats.initial_count = int(cands.size());
        stats.window_kernel_used = kernel_path ? 1 : 0;
    }

    PreparedFrame frame;  // 收尾只消费窗口宽高（center 越界过滤），features 留空
    frame.width = window.cols;
    frame.height = window.rows;
    frame.padded_width = job->pw;
    frame.padded_height = job->ph;
    return collect_hit_objects(fine, extract_out.matches, frame);
}

std::vector<vf_shape_candidate> match_fine_window_gpu(Model& fine, const cv::Mat& window,
                                                      double fine_th) {
    require_coded(std::isfinite(fine_th) && fine_th > 0 && fine_th <= 100,
                  VF_SHAPE_THRESHOLD_INVALID, "invalid similarity threshold", fine_th);
    auto job = fine_window_begin(fine, window, fine_th);
    return fine_window_end(std::move(job), fine, window, fine_th);
}

}  // namespace vfshape::gpu
