// W3 波3 引擎口径精修窗段性能门（shape_engine_bench）：trace.windows_ms 全窗段
// ——CPU=refine_window 段（per-window prepare+match 全链）vs GPU=fine_service 全
// 窗墙钟——热稳态预热 300×100 中位，门 ≥2.0 进测试（不足退出码 3）；同负载
// match_two_stage 全帧端到端 CPU vs GPU 数据行只测不门（第三单全帧裁决待定稿，
// 不足不凑数如实出值）。memo 单列：last_fine_stats 执行面字段 + coarse_resize_ms。
// 退出码：0=过；3=门禁失败；77=无 CUDA 构建/无可用设备（ctest SKIP_RETURN_CODE）。
#include <windows.h>

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
#include "../src/gpu/coarse_service.hpp"
#include "../src/gpu/fine_service.hpp"
#include "../src/matching_profile.hpp"
#include "../src/two_stage.hpp"

#include "gpu_engine_scenes.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace et = vfshape::engine_test;
using et::check;
using et::kThreshold;
using et::median_of;
using et::set_backend;
using et::wall_ms;

struct CallSample {
    double windows_ms = 0, wall_ms = 0, coarse_resize_ms = 0;
    int windows_searched = 0, scale_retries = 0, group_retries = 0;
};

// 1080P 典型负载（扰动基准形态=平坦工厂面+部件）：强目标（窗 0）+ 45% 遮挡
// 目标（候选 #2 进前两窗，窗内 fine 不达阈 → group 重试路径进本轮运行）。
// 平坦面压掉纹理假候选洪泛，候选量回到典型量级。
cv::Mat bench_scene(const cv::Mat& patch) {
    cv::Mat scene(1080, 1920, CV_8UC3, cv::Scalar(120, 118, 116));
    patch.copyTo(scene(cv::Rect(1216, 396, patch.cols, patch.rows)));
    patch.copyTo(scene(cv::Rect(900, 660, patch.cols, patch.rows)));
    cv::rectangle(scene, {900, 660}, {900 + patch.cols * 45 / 100, 660 + patch.rows},
                  {120, 118, 116}, -1);
    return scene;
}

CallSample run_side(const char* backend, vfshape::TwoStageModel& model, const cv::Mat& scene,
                    int warmup, int loops) {
    set_backend(backend);
    for (int i = 0; i < warmup; ++i)
        (void)vfshape::match_two_stage(model, scene, kThreshold, 16);
    std::vector<CallSample> samples;
    for (int i = 0; i < loops; ++i) {
        vfshape::MatchTraceSnapshot trace;
        const auto t0 = std::chrono::steady_clock::now();
        const auto cands = vfshape::match_two_stage(model, scene, kThreshold, 16, &trace);
        const auto t1 = std::chrono::steady_clock::now();
        CallSample s;
        s.windows_ms = trace.windows_ms;
        s.wall_ms = wall_ms(t0, t1);
        s.coarse_resize_ms = trace.coarse_resize_ms;
        s.windows_searched = trace.windows_searched;
        s.scale_retries = trace.scale_retries;
        s.group_retries = trace.group_retries;
        samples.push_back(s);
        check(!cands.empty(), "bench load produced no candidates");
    }
    std::vector<double> w, t;
    for (const auto& s : samples) {
        w.push_back(s.windows_ms);
        t.push_back(s.wall_ms);
    }
    CallSample med = samples[samples.size() / 2];
    med.windows_ms = median_of(w);
    med.wall_ms = median_of(t);
    return med;
}

void part_window_gate(vfshape::TwoStageModel& model, const cv::Mat& patch) {
    constexpr int kWarmup = 300, kLoops = 100;
    constexpr double kGate = 2.0;
    const cv::Mat scene = bench_scene(patch);
    std::printf("[bench] load %dx%d k=%d threshold=%.0f warmup=%d loops=%d\n", scene.cols,
                scene.rows, model.k, kThreshold, kWarmup, kLoops);

    const CallSample gpu = run_side("gpu", model, scene, kWarmup, kLoops);
    const vfshape::gpu::FineWindowStats stats = vfshape::gpu::last_fine_stats();
    const CallSample cpu = run_side("cpu", model, scene, kWarmup, kLoops);

    const double speedup = cpu.windows_ms / gpu.windows_ms;
    std::printf("[bench] window gate cpu=%8.3fms gpu=%7.3fms speedup=%6.2fx [gate >=%.1fx]\n",
                cpu.windows_ms, gpu.windows_ms, speedup, kGate);
    // 全帧端到端数据行（第三单裁决待定稿：只测不门，不足如实出值）
    const double frame_ratio = cpu.wall_ms / gpu.wall_ms;
    std::printf("[bench] full-frame cpu=%8.3fms gpu=%7.3fms ratio=%6.2fx [data row, no gate]\n",
                cpu.wall_ms, gpu.wall_ms, frame_ratio);
    std::printf("[bench] load shape: windows=%d scale_retries=%d group_retries=%d\n",
                cpu.windows_searched, cpu.scale_retries, cpu.group_retries);
    std::printf("[bench] memo gpu last-window: chain=%7.3fms extract=%7.3fms initial=%d "
                "kernel_used=%d (excl)\n",
                stats.chain_ms, stats.extract_ms, stats.initial_count,
                stats.window_kernel_used);
    std::printf("[bench] memo coarse_resize cpu=%7.3fms gpu=%7.3fms (excl)\n",
                cpu.coarse_resize_ms, gpu.coarse_resize_ms);
    check(speedup >= kGate,
          "window segment speedup below gate: " + std::to_string(speedup));
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        std::string err, device_name;
        int major = 0, minor = 0;
        if (!vfshape::gpu::device_info(device_name, major, minor, err)) {
            std::printf("[bench] skip: no usable CUDA device (%s)\n", err.c_str());
            return et::kSkipCode;
        }
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120) {
            std::printf("[bench] skip: device sm_%d outside supported set {86,89,120}\n", sm);
            return et::kSkipCode;
        }
        std::printf("[bench] device: %s (sm_%d)\n", device_name.c_str(), sm);
        set_backend("cpu");
        check(_putenv("VF_SHAPE_TRACE_QUIET=1") == 0, "set VF_SHAPE_TRACE_QUIET failed");

        const auto patch = et::target_patch(128);
        const auto reference = et::make_reference(patch);
        auto model = vfshape::train_two_stage(
            reference, et::train_config(-30.0, 30.0, {80, 80, 96, 96}));
        check(model != nullptr && !model->coarse->poses.empty(), "training failed");
        std::printf("[bench] model: coarse_poses=%zu fine_groups=%zu k=%d\n",
                    model->coarse->poses.size(), model->fines.size(), model->k);
        part_window_gate(*model, patch);

        std::printf("[bench] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "engine bench failed: %s\n", ex.what());
        return et::kFailCode;
    } catch (...) {
        std::fprintf(stderr, "engine bench failed: unknown exception\n");
        return et::kFailCode;
    }
}

#else  // !VF_SHAPE_CUDA

#include <cstdio>

int main() {
    std::printf("[bench] skip: built without VF_SHAPE_CUDA\n");
    return 77;
}

#endif
