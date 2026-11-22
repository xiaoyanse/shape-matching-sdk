// W4 回退兜底门（shape_gpu_fallback，真装配形态同 engine_bitexact）：probe 三
// 分支注入→永久 CPU+日志行+结果逐位；初始化失败 auto 回退逐位+gpu 态直报分叉
// +成功清零；运行中失败整调用 CPU 重跑逐位+连续 3 次粘性禁用+第 4 次不触 GPU；
// auto 默认（删除 env）有卡真跑 GPU+env=cpu 旁路不探测。失败注入经 gpu_test_
// hooks（仅测试置位）；stderr 日志断言经 log sink 注入捕获（缺省路径共用同一
// 消息串）。退出码：0=过；3=门禁失败；77=无 CUDA 构建/无设备（ctest Skip）。
#include <windows.h>

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
#include "../src/coarse_image.hpp"
#include "../src/gpu/coarse_service.hpp"
#include "../src/gpu/gpu_device.hpp"
#include "../src/gpu/service_shared.hpp"
#include "../src/two_stage.hpp"

#include "gpu_engine_scenes.hpp"

#include <opencv2/core.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace et = vfshape::engine_test;
namespace gpu = vfshape::gpu;
using et::check;

std::vector<std::string> g_log_lines;

void capture_log(const char* line) { g_log_lines.emplace_back(line); }

bool log_contains(const char* needle) {
    for (const auto& line : g_log_lines)
        if (line.find(needle) != std::string::npos) return true;
    return false;
}

void reset_gpu_state() {
    gpu::svc::test_hooks().inject_probe_fail_branch = 0;
    gpu::svc::test_hooks().inject_init_fail = 0;
    gpu::svc::test_hooks().inject_chain_fail.store(0);
    std::lock_guard<std::mutex> lock(gpu::svc::service_mutex());
    gpu::destroy_context(gpu::svc::state().ctx);
    gpu::svc::state() = gpu::svc::ServiceState{};
    gpu::svc::PipelineState& pipe = gpu::svc::pipeline();
    if (pipe.stream_ready) gpu::stream_destroy(pipe.stream);
    pipe.stream = nullptr;
    pipe.stream_ready = false;
    pipe.slot_busy[0] = pipe.slot_busy[1] = false;
    pipe.active_jobs.store(0);
    gpu::svc::fallback().probed.store(false);
    gpu::svc::fallback().gpu_usable.store(false);
    gpu::svc::fallback().chain_fail_streak.store(0);
    gpu::svc::fallback().sticky_disabled.store(false);
}

const char* first_diff_field(const vf_shape_candidate& a, const vf_shape_candidate& b) {
    if (a.score != b.score) return "score";
    if (a.pose_id != b.pose_id) return "pose_id";
    if (a.pose_ambiguous != b.pose_ambiguous) return "pose_ambiguous";
    for (int i = 0; i < 2; ++i)
        if (a.center[i] != b.center[i]) return "center";
    if (a.angle != b.angle) return "angle";
    if (a.scale != b.scale) return "scale";
    for (int i = 0; i < 8; ++i)
        if (a.corners[i] != b.corners[i]) return "corners";
    for (int i = 0; i < 4; ++i)
        if (a.bbox[i] != b.bbox[i]) return "bbox";
    for (int i = 0; i < 6; ++i)
        if (a.transform[i] != b.transform[i]) return "transform";
    return nullptr;
}

void compare_bitexact(const char* tag, const std::vector<vf_shape_candidate>& cpu,
                      const std::vector<vf_shape_candidate>& rerun) {
    if (cpu.size() != rerun.size())
        throw std::runtime_error(std::string(tag) + ": candidate count cpu=" +
                                 std::to_string(cpu.size()) + " rerun=" +
                                 std::to_string(rerun.size()));
    for (size_t i = 0; i < cpu.size(); ++i) {
        const char* field = first_diff_field(cpu[i], rerun[i]);
        if (field != nullptr)
            throw std::runtime_error(std::string(tag) + " candidate[" + std::to_string(i) +
                                     "] field " + field + " differs");
    }
    std::printf("[fallback] %s: n=%zu bitexact OK\n", tag, cpu.size());
}

std::vector<vf_shape_candidate> match_cpu(vfshape::TwoStageModel& model, const cv::Mat& frame) {
    et::set_backend("cpu");
    return vfshape::match_two_stage(model, frame, et::kThreshold, 16);
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        std::string err, device_name;
        int major = 0, minor = 0;
        if (!gpu::device_info(device_name, major, minor, err)) {
            std::printf("[fallback] skip: no usable CUDA device (%s)\n", err.c_str());
            return et::kSkipCode;
        }
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120) {
            std::printf("[fallback] skip: device sm_%d outside supported set {86,89,120}\n", sm);
            return et::kSkipCode;
        }
        std::printf("[fallback] device: %s (sm_%d)\n", device_name.c_str(), sm);
        gpu::svc::log_sink() = &capture_log;

        const auto patch = et::target_patch(128);
        const auto reference = et::make_reference(patch);
        auto model = vfshape::train_two_stage(
            reference, et::train_config(-30.0, 30.0, {80, 80, 96, 96}));
        check(model != nullptr && !model->coarse->poses.empty(), "training failed");
        std::printf("[fallback] model: coarse_poses=%zu k=%d\n", model->coarse->poses.size(),
                    model->k);

        std::vector<cv::Mat> frames;
        frames.push_back(et::paste(patch, et::kFrameW, et::kFrameH, 96, 56, 0xa11ce));
        frames.push_back(et::paste(patch, et::kFrameW, et::kFrameH, 20, 16, 0x5eed3));
        const auto cpu_baseline0 = match_cpu(*model, frames[0]);
        const auto cpu_baseline1 = match_cpu(*model, frames[1]);

        // 组1：probe 三分支注入 → 永久 CPU + 单行日志（原因）+ 结果与 CPU 逐位
        for (int branch = 1; branch <= 3; ++branch) {
            reset_gpu_state();
            gpu::svc::test_hooks().inject_probe_fail_branch = branch;
            et::set_backend("auto");
            g_log_lines.clear();
            const auto got = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
            check(g_log_lines.size() == 1, "probe branch must log exactly one line");
            const char* needle = branch == 1 ? "no cuda device"
                : branch == 2 ? "outside supported set" : "context init failed";
            check(log_contains(needle), "probe branch log reason missing");
            check(gpu::svc::fallback().probed.load() && !gpu::svc::fallback().gpu_usable.load(),
                  "probe branch must conclude permanently CPU");
            const std::string tag = "probe branch " + std::to_string(branch);
            compare_bitexact(tag.c_str(), cpu_baseline0, got);
        }

        // 组2：初始化失败注入 —— auto 回退逐位+计数，gpu 态同注入=直报（语义分叉）
        reset_gpu_state();
        gpu::svc::test_hooks().inject_init_fail = 1;
        et::set_backend("auto");
        const auto got2 = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        check(gpu::svc::test_hooks().inject_init_fail == 0,
              "init hook must be consumed by the GPU attempt");
        check(gpu::svc::fallback().chain_fail_streak.load() == 1,
              "init failure must count toward streak");
        compare_bitexact("init-fail rerun", cpu_baseline0, got2);
        const auto got2b = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        check(gpu::svc::fallback().chain_fail_streak.load() == 0,
              "successful GPU call must reset streak");
        compare_bitexact("init-fail recovery", cpu_baseline0, got2b);
        reset_gpu_state();
        gpu::svc::test_hooks().inject_init_fail = 1;
        et::set_backend("gpu");
        bool reported = false;
        try {
            (void)vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        } catch (const std::invalid_argument& ex) {
            reported = std::strstr(ex.what(), "injected context init failure") != nullptr;
        }
        check(reported, "gpu mode must report injected init failure (no fallback)");
        reset_gpu_state();

        // 组3：运行中失败 —— 整调用 CPU 重跑逐位；连续 3 次粘性；第 4 次不触 GPU
        reset_gpu_state();
        et::set_backend("auto");
        for (int round = 1; round <= 3; ++round) {
            gpu::svc::test_hooks().inject_chain_fail.store(1);
            g_log_lines.clear();
            const auto got3 = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
            check(gpu::svc::test_hooks().inject_chain_fail.load() == 0,
                  "chain hook must be consumed by the GPU attempt");
            check(gpu::svc::fallback().chain_fail_streak.load() == round, "streak must advance");
            if (round == 3) {
                check(log_contains("sticky disable"), "sticky log line missing");
                check(gpu::svc::fallback().sticky_disabled.load(), "sticky flag must set at 3");
            } else {
                check(!gpu::svc::fallback().sticky_disabled.load(),
                      "sticky must not set before 3rd failure");
                check(g_log_lines.empty(), "no sticky log before 3rd failure");
            }
            const std::string tag = "chain-fail rerun round " + std::to_string(round);
            compare_bitexact(tag.c_str(), cpu_baseline0, got3);
        }
        gpu::svc::test_hooks().inject_chain_fail.store(5);
        const auto got4 = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        check(gpu::svc::test_hooks().inject_chain_fail.load() == 5,
              "sticky disable must bypass GPU entirely (hook untouched)");
        compare_bitexact("post-sticky", cpu_baseline0, got4);
        reset_gpu_state();

        // 组4：auto 默认（删除 env）有卡真跑 GPU；env=cpu 旁路不探测
        reset_gpu_state();
        check(_putenv("VF_SHAPE_BACKEND=") == 0, "unset VF_SHAPE_BACKEND failed");
        check(std::getenv("VF_SHAPE_BACKEND") == nullptr, "env must be unset");
        g_log_lines.clear();
        const auto got5 = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        const auto got6 = vfshape::match_two_stage(*model, frames[1], et::kThreshold, 16);
        check(g_log_lines.empty(), "healthy device must not log fallback");
        check(gpu::svc::fallback().gpu_usable.load(), "probe must pass on real device");
        check(gpu::last_call_stats().gpu_chain_ms > 0.0,
              "auto default must run coarse on GPU");
        compare_bitexact("auto default f0", cpu_baseline0, got5);
        compare_bitexact("auto default f1", cpu_baseline1, got6);
        gpu::svc::test_hooks().inject_probe_fail_branch = 3;
        et::set_backend("cpu");
        const auto got7 = vfshape::match_two_stage(*model, frames[0], et::kThreshold, 16);
        check(gpu::svc::test_hooks().inject_probe_fail_branch == 3,
              "cpu mode must bypass probe (armed injection untouched)");
        check(g_log_lines.empty(), "cpu mode must not log");
        compare_bitexact("cpu bypass", cpu_baseline0, got7);
        gpu::svc::test_hooks().inject_probe_fail_branch = 0;

        std::printf("[fallback] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gpu fallback test failed: %s\n", ex.what());
        return et::kFailCode;
    } catch (...) {
        std::fprintf(stderr, "gpu fallback test failed: unknown exception\n");
        return et::kFailCode;
    }
}

#else  // !VF_SHAPE_CUDA

#include <cstdio>

int main() {
    std::printf("[fallback] skip: built without VF_SHAPE_CUDA\n");
    return 77;
}

#endif
