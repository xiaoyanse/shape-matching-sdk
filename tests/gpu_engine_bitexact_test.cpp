// W2-B/W3 引擎级逐位对拍门（shape_engine_bitexact）：match_two_stage 全链真装配，
// 进程内翻转 VF_SHAPE_BACKEND 对拍 GPU vs vendor CPU 后端，最终候选集逐位一致
//（边缘帧咬 LM 行尾续读带回归；W3 增重试路径专帧咬合+trace 计数断言）；接缝级
// 直调对拍隔离定位；coarse_ms 双档性能门（k=8 parity 0.90 / k=2 大工作面 1.15，
// 缩图/链路/提取单列不上门）。退出码：0=过；3=门禁失败；77=无 CUDA 构建/无设备
//（ctest SKIP_RETURN_CODE）。共用场景件见 gpu_engine_scenes.hpp。
#include <windows.h>

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
#include "../src/coarse_image.hpp"
#include "../src/gpu/coarse_service.hpp"
#include "../src/matching_profile.hpp"
#include "../src/two_stage.hpp"

#include "gpu_engine_scenes.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace et = vfshape::engine_test;
using et::check;
using et::kPatch;
using et::kThreshold;
using et::median_of;
using et::set_backend;

std::vector<et::EngineFrame> build_frames(const cv::Mat& patch) {
    std::vector<et::EngineFrame> frames;
    frames.push_back({"anchor-center", et::paste(patch, et::kFrameW, et::kFrameH, 96, 56, 0xa11ce)});
    {
        cv::Mat rotated;
        const auto m = cv::getRotationMatrix2D(cv::Point2d(et::kFrameW / 2, et::kFrameH / 2),
                                               12.0, 1.0);
        cv::warpAffine(frames.back().image, rotated, m, frames.back().image.size(),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        frames.push_back({"rotated-12deg", rotated});
    }
    {
        cv::Mat noisy = et::paste(patch, et::kFrameW, et::kFrameH, 96, 56, 0xa11ce);
        uint64_t state = 0x707;
        for (int y = 0; y < noisy.rows; ++y) {
            uint8_t* row = noisy.ptr<uint8_t>(y);
            for (int x = 0; x < noisy.cols * 3; ++x)
                row[x] = cv::saturate_cast<uint8_t>(int(row[x]) + int(et::lcg(state) % 9) - 4);
        }
        frames.push_back({"noisy", noisy});
    }
    {
        cv::Mat multi = et::paste(patch, et::kFrameW, et::kFrameH, 20, 16, 0x5eed3);
        patch.copyTo(multi(cv::Rect(178, 104, kPatch, kPatch)));
        frames.push_back({"multi-target", multi});
    }
    frames.push_back({"empty-scene", et::background(et::kFrameW, et::kFrameH, 0x0ddba11)});
    frames.push_back({"edge-right",
                      et::paste(patch, et::kFrameW, et::kFrameH,
                                et::kFrameW - kPatch + 10, 60, 0xe122e)});
    frames.push_back({"edge-bottom",
                      et::paste(patch, et::kFrameW, et::kFrameH, 100,
                                et::kFrameH - kPatch + 8, 0xe130e)});
    return frames;
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

void compare_bitexact(const char* tag, const std::vector<vf_shape_candidate>& left,
                      const std::vector<vf_shape_candidate>& right) {
    if (left.size() != right.size())
        throw std::runtime_error(std::string(tag) + ": candidate count " +
                                 std::to_string(right.size()) + " != " +
                                 std::to_string(left.size()));
    for (size_t i = 0; i < left.size(); ++i) {
        const char* field = first_diff_field(left[i], right[i]);
        if (field == nullptr) continue;
        auto value_of = [field](const vf_shape_candidate& c) {
            if (std::strcmp(field, "score") == 0) return c.score;
            if (std::strcmp(field, "pose_id") == 0) return double(c.pose_id);
            if (std::strcmp(field, "pose_ambiguous") == 0) return double(c.pose_ambiguous);
            if (std::strcmp(field, "center") == 0) return c.center[0];
            if (std::strcmp(field, "angle") == 0) return c.angle;
            if (std::strcmp(field, "scale") == 0) return c.scale;
            if (std::strcmp(field, "corners") == 0) return c.corners[0];
            if (std::strcmp(field, "bbox") == 0) return c.bbox[0];
            return c.transform[0];
        };
        std::printf("[engine] %s candidate[%zu] field %s: left=%.17g right=%.17g\n", tag, i,
                    field, value_of(left[i]), value_of(right[i]));
        throw std::runtime_error(std::string(tag) + " candidate[" + std::to_string(i) +
                                 "] field " + field + " differs");
    }
}

void part5_engine_bitexact(vfshape::TwoStageModel& model,
                           const std::vector<et::EngineFrame>& frames) {
    std::vector<std::vector<vf_shape_candidate>> pass1_cpu(frames.size());
    for (int pass = 1; pass <= 2; ++pass) {
        const bool cpu_first = pass == 1;
        for (size_t i = 0; i < frames.size(); ++i) {
            set_backend(cpu_first ? "cpu" : "gpu");
            auto a = vfshape::match_two_stage(model, frames[i].image, kThreshold, 16);
            set_backend(cpu_first ? "gpu" : "cpu");
            auto b = vfshape::match_two_stage(model, frames[i].image, kThreshold, 16);
            auto* cpu = cpu_first ? &a : &b;
            auto* gpu = cpu_first ? &b : &a;
            const std::string tag =
                std::string(frames[i].name) + " pass" + std::to_string(pass);
            compare_bitexact(tag.c_str(), *cpu, *gpu);
            if (cpu_first)
                pass1_cpu[i] = *cpu;
            else
                compare_bitexact((tag + " order").c_str(), pass1_cpu[i], *cpu);
            std::printf("[engine] %s: cpu_n=%zu gpu_n=%zu bitexact OK\n", tag.c_str(),
                        cpu->size(), gpu->size());
        }
    }
}

void part5_seam_bitexact(vfshape::TwoStageModel& model,
                         const std::vector<et::EngineFrame>& frames) {
    const double coarse_th = std::max(45.0, kThreshold - 20.0);
    for (const auto& frame : frames) {
        cv::Mat coarse_img;
        vfshape::resize_coarse_image(frame.image, coarse_img, model.k);
        set_backend("cpu");
        const auto cpu_cands = vfshape::match(*model.coarse, coarse_img, coarse_th);
        set_backend("gpu");
        const auto gpu_cands =
            vfshape::gpu::match_coarse_gpu(*model.coarse, coarse_img, coarse_th);
        const std::string tag = std::string("seam ") + frame.name;
        compare_bitexact(tag.c_str(), cpu_cands, gpu_cands);
        std::printf("[engine] %s: cpu_n=%zu gpu_n=%zu bitexact OK\n", tag.c_str(),
                    cpu_cands.size(), gpu_cands.size());
    }
}

// W3 重试路径咬合：多尺度粗层模型（0.5/0.75/1.0 档）专帧强制 scale/group 重试
// 路径进本轮运行，trace 计数显式断言（>0）+ 双后端计数一致 + 结果逐位。
void part5_retry_paths(const cv::Mat& patch, const cv::Mat& reference) {
    auto model = vfshape::train_two_stage(
        reference, et::train_config(-30.0, 30.0, {32, 32, 192, 192}, 0.5, 1.0, 0.5));
    check(model != nullptr && !model->coarse->poses.empty(), "retry model training failed");
    // 两专帧平坦背景确定性构造。scale 帧：ROI 裁剪内容 ×0.5（与粗层 0.5 档模板
    // 逐像素同构 → cs 必报 0.5 且为最强候选），精修 1.0 档模板对半尺寸目标必
    // <60 → scale 重试必触发；group 帧：1.0 档半出下缘目标（cs=1 无缩放重试、
    // 可见特征不足 <60）→ group 重试必触发。
    std::vector<et::EngineFrame> frames;
    {
        cv::Mat scene(et::kFrameH, et::kFrameW, CV_8UC3, cv::Scalar(128, 128, 128));
        cv::Mat scaled_roi;
        cv::resize(reference(cv::Rect(32, 32, 192, 192)), scaled_roi, cv::Size(), 0.5, 0.5,
                   cv::INTER_AREA);
        scaled_roi.copyTo(scene(cv::Rect(60, 60, scaled_roi.cols, scaled_roi.rows)));
        frames.push_back({"retry-scale", scene});
    }
    {
        cv::Mat scene(et::kFrameH, et::kFrameW, CV_8UC3, cv::Scalar(128, 128, 128));
        patch(cv::Rect(0, 0, kPatch, kPatch / 2))
            .copyTo(scene(cv::Rect(96, et::kFrameH - kPatch / 2, kPatch, kPatch / 2)));
        frames.push_back({"retry-group", scene});
    }
    for (const auto& frame : frames) {
        // scale 帧以高判定阈运行：fine_th 抬到 72.25，修正窗内容必低于该线 →
        // 首次精修必然不足 → scale 重试确定性触发（粗层 65 线下强候选仍存活）
        const double th =
            std::strcmp(frame.name, "retry-scale") == 0 ? 85.0 : kThreshold;
        vfshape::MatchTraceSnapshot cpu_trace, gpu_trace;
        set_backend("cpu");
        const auto cpu_cands = vfshape::match_two_stage(*model, frame.image, th, 16, &cpu_trace);
        set_backend("gpu");
        const auto gpu_cands = vfshape::match_two_stage(*model, frame.image, th, 16, &gpu_trace);
        const std::string tag = "retry " + std::string(frame.name);
        compare_bitexact(tag.c_str(), cpu_cands, gpu_cands);
        check(cpu_trace.scale_retries == gpu_trace.scale_retries &&
                  cpu_trace.group_retries == gpu_trace.group_retries,
              tag + ": retry counters differ between backends");
        const double scale0 = cpu_cands.empty() ? -1.0 : cpu_cands.front().scale;
        std::printf("[engine] %s: cpu_n=%zu gpu_n=%zu scale_retries=%d group_retries=%d "
                    "windows=%d scale0=%.2f wbest=%.1f/%.1f bitexact OK\n",
                    tag.c_str(), cpu_cands.size(), gpu_cands.size(), cpu_trace.scale_retries,
                    cpu_trace.group_retries, cpu_trace.windows_searched, scale0,
                    cpu_trace.window_best[0], cpu_trace.window_best[1]);
        if (std::strcmp(frame.name, "retry-scale") == 0)
            check(cpu_trace.scale_retries > 0, "scale retry path not exercised this round");
        else
            check(cpu_trace.group_retries > 0, "group retry path not exercised this round");
    }
}

// 双档性能门（分档裁决口径）：k=8 小工作面 parity 档 / k=2 大工作面档；
// 两档都测完由调用方判门，失败报两档数值。缩图与 chain/extract 单列不上门。
double part5_bench_tier(const char* tag, vfshape::TwoStageModel& model, const cv::Mat& patch,
                        double gate) {
    constexpr int kWarmup = 10, kLoops = 100;
    const double coarse_th = std::max(45.0, kThreshold - 20.0);
    cv::Mat scene = et::sparse_background(1920, 1080);
    patch.copyTo(scene(cv::Rect(1216, 396, patch.cols, patch.rows)));
    cv::Mat coarse_img;
    vfshape::resize_coarse_image(scene, coarse_img, model.k);

    for (int i = 0; i < kWarmup; ++i) vfshape::match(*model.coarse, coarse_img, coarse_th);
    std::vector<double> cpu_ms;
    size_t cpu_sink = 0;
    for (int i = 0; i < kLoops; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto cands = vfshape::match(*model.coarse, coarse_img, coarse_th);
        cpu_ms.push_back(et::wall_ms(t0, std::chrono::steady_clock::now()));
        cpu_sink += cands.size();
    }

    set_backend("gpu");
    for (int i = 0; i < kWarmup; ++i)
        vfshape::gpu::match_coarse_gpu(*model.coarse, coarse_img, coarse_th);
    std::vector<double> gpu_ms, chain_ms, extract_ms, resize_ms;
    std::vector<int> initial;
    size_t gpu_sink = 0;
    for (int i = 0; i < kLoops; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto cands = vfshape::gpu::match_coarse_gpu(*model.coarse, coarse_img, coarse_th);
        gpu_ms.push_back(et::wall_ms(t0, std::chrono::steady_clock::now()));
        gpu_sink += cands.size();
        chain_ms.push_back(vfshape::gpu::last_call_stats().gpu_chain_ms);
        extract_ms.push_back(vfshape::gpu::last_call_stats().extract_ms);
        initial.push_back(vfshape::gpu::last_call_stats().initial_count);
    }
    set_backend("cpu");

    for (int i = 0; i < kLoops; ++i) {
        cv::Mat shrink;
        const auto t0 = std::chrono::steady_clock::now();
        vfshape::resize_coarse_image(scene, shrink, model.k);
        resize_ms.push_back(et::wall_ms(t0, std::chrono::steady_clock::now()));
    }
    std::sort(initial.begin(), initial.end());

    const double cpu_med = median_of(cpu_ms), gpu_med = median_of(gpu_ms);
    const double speedup = cpu_med / gpu_med;
    std::printf("[engine-bench] %s scene %dx%d small %dx%d coarse_poses=%zu k=%d "
                "sink=%zu/%zu\n",
                tag, scene.cols, scene.rows, coarse_img.cols, coarse_img.rows,
                model.coarse->poses.size(), model.k, cpu_sink, gpu_sink);
    std::printf("[engine-bench] %s coarse_ms cpu=%8.3fms gpu=%7.3fms speedup=%6.2fx "
                "[gate >=%.2fx]\n",
                tag, cpu_med, gpu_med, speedup, gate);
    std::printf("[engine-bench] %s memo resize_coarse_image=%7.3fms (cpu 1/k shrink, excl)\n",
                tag, median_of(resize_ms));
    std::printf("[engine-bench] %s memo gpu_chain=%7.3fms extract=%7.3fms initial_count=%d "
                "(excl)\n",
                tag, median_of(chain_ms), median_of(extract_ms), initial[initial.size() / 2]);
    return speedup;
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        std::string err, device_name;
        int major = 0, minor = 0;
        if (!vfshape::gpu::device_info(device_name, major, minor, err)) {
            std::printf("[engine] skip: no usable CUDA device (%s)\n", err.c_str());
            return et::kSkipCode;
        }
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120) {
            std::printf("[engine] skip: device sm_%d outside supported set {86,89,120}\n", sm);
            return et::kSkipCode;
        }
        std::printf("[engine] device: %s (sm_%d)\n", device_name.c_str(), sm);
        set_backend("cpu");

        const auto patch = et::target_patch(128);
        const auto reference = et::make_reference(patch);
        auto model = vfshape::train_two_stage(
            reference, et::train_config(-30.0, 30.0, {80, 80, 96, 96}));
        check(model != nullptr && !model->coarse->poses.empty(), "part5a training failed");
        std::printf("[engine] part5a model: coarse_poses=%zu fine_groups=%zu k=%d\n",
                    model->coarse->poses.size(), model->fines.size(), model->k);
        const auto frames = build_frames(patch);
        part5_engine_bitexact(*model, frames);
        part5_seam_bitexact(*model, frames);
        part5_retry_paths(patch, reference);

        const auto bench_patch = et::target_patch(512);
        cv::Mat bench_ref = et::sparse_background(1024, 1024);
        bench_patch.copyTo(bench_ref(cv::Rect(256, 256, 512, 512)));
        auto bench_model = vfshape::train_two_stage(
            bench_ref, et::train_config(-136.0, 136.0, {256, 256, 512, 512}));
        check(bench_model != nullptr && !bench_model->coarse->poses.empty(),
              "bench training failed");
        std::printf("[engine] bench model: coarse_poses=%zu k=%d\n",
                    bench_model->coarse->poses.size(), bench_model->k);
        const double speed_k8 = part5_bench_tier("bench-k8", *bench_model, bench_patch, 0.90);
        const double speed_k2 = part5_bench_tier("bench-k2", *model, patch, 1.15);
        check(speed_k8 >= 0.90,
              "bench-k8 coarse_ms speedup below gate: " + std::to_string(speed_k8));
        check(speed_k2 >= 1.15,
              "bench-k2 coarse_ms speedup below gate: " + std::to_string(speed_k2));

        std::printf("[engine] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "engine bitexact failed: %s\n", ex.what());
        return et::kFailCode;
    } catch (...) {
        std::fprintf(stderr, "engine bitexact failed: unknown exception\n");
        return et::kFailCode;
    }
}

#else  // !VF_SHAPE_CUDA

#include <cstdio>

int main() {
    std::printf("[engine] skip: built without VF_SHAPE_CUDA\n");
    return 77;
}

#endif
