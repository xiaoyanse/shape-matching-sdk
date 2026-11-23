// W2 Phase A 对拍门（shape_gpu_bitexact）：src/gpu 粗层三核独立对拍，不依赖引擎
// 接缝。三段：
//  Part 1 fastAtan2 复刻穷举——整数 Sobel 梯度对全网格（±1020，416 万对）对
//         cv::phase（本机 DLL 实际派发路径）逐位一致 + 量化对 convertTo 逐位一致；
//  Part 2 链路逐位对拍——GPU 内核 vs CPU 参考，round1=Spike 口径（复刻前基线，
//         证内核平移本身逐位），round2=Vendor 口径（复刻后，含 fastAtan2+回卷+
//         span 置零，全图 100% 逐位）；
//  Part 3 复刻前后差异归因——CPU 侧 Vendor vs Spike 差异分布：角度桶差与响应
//         差分离 Interior（span 内）与边带（回卷带+span 外），Interior 响应差仅
//         允许源自角度桶翻转。
//  --bench：粗层段（pyrDown+L1 三段）内核级 GPU vs CPU 参考，预热 10×循环 100
//         中位，下限门 ≥8×（尖峰实测 12-39× 取低段余量）。
// 退出码：0=过；3=门禁失败；77=无 CUDA 构建/无可用设备（ctest SKIP_RETURN_CODE）。
// 纪律：SetErrorMode 关系统错误弹窗；异常全捕获退出码化。
#include <windows.h>

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA
#include "../src/gpu/coarse_kernels.hpp"
#include "../src/gpu/fast_atan2.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_coarse_reference.h"

namespace {

constexpr int kSkipCode = 77, kFailCode = 3;
constexpr int kPadding = 32;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t lcg(uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 33);
}

int padded_dim(int value) { return (value + kPadding * 2 + 15) / 16 * 16; }

// —— Part 1：fastAtan2 复刻穷举（vs cv::phase DLL + convertTo）——
void part1_fast_atan2() {
    constexpr int kLim = 1020;  // vendor 3×3 Sobel 单方向上界 4*255（model.hpp 注）
    std::vector<float> yv, xv;
    yv.reserve(size_t(2 * kLim + 1) * (2 * kLim + 1));
    xv.reserve(yv.capacity());
    for (int y = -kLim; y <= kLim; ++y)
        for (int x = -kLim; x <= kLim; ++x) {
            yv.push_back(float(y));
            xv.push_back(float(x));
        }
    const int n = int(yv.size());
    cv::Mat ys(1, n, CV_32F, yv.data()), xs(1, n, CV_32F, xv.data()), angles;
    cv::phase(xs, ys, angles, true);
    cv::Mat quantized;
    angles.convertTo(quantized, CV_8U, 16.0 / 360.0);
    long long angle_diff = 0, quant_diff = 0, mask_diff = 0;
    for (int i = 0; i < n; ++i) {
        const float ref = angles.at<float>(i);
        const float got = vfshape::gpu::fast_atan2_deg(yv[size_t(i)], xv[size_t(i)]);
        if (got != ref) ++angle_diff;
        // 量化双查：原始 16 桶对 convertTo 逐位（vendor 口径）；&7 掩码结构自检
        const int q_ref = int(quantized.at<uchar>(i));
        const int q_mine = int(std::lrintf(ref * (float)(16.0 / 360.0)));
        if (q_mine != q_ref) ++quant_diff;
        if ((q_mine & 7) != vfshape::gpu::quantize_orientation(ref)) ++mask_diff;
    }
    std::printf("[bitexact] part1 atan2 grid=%d pairs: angle_diff=%lld quant16_diff=%lld "
                "mask7_diff=%lld\n",
                n, angle_diff, quant_diff, mask_diff);
    check(angle_diff == 0, "fast_atan2_deg differs from cv::phase on integer grid");
    check(quant_diff == 0, "quantization differs from convertTo on integer grid");
    check(mask_diff == 0, "quantize_orientation &7 masking inconsistent");
}

// —— 合成数据：确定性纹理帧 + 混合规格位姿（宽模板触发整行右缘回卷带、超幅
//    模板走出界跳过路径、特征数跨 64 桶两侧）——
cv::Mat synthetic_frame(int w, int h, uint64_t seed) {
    cv::Mat img(h, w, CV_8UC3);
    uint64_t state = seed;
    for (int y = 0; y < h; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const uint32_t r = lcg(state);
            row[3 * x] = uint8_t((x * 255) / (w - 1) ^ (r & 7));
            row[3 * x + 1] = uint8_t((y * 255) / (h - 1));
            row[3 * x + 2] = uint8_t(((x / 24) + (y / 24)) % 2 * 255 - (r % 37));
        }
    }
    cv::circle(img, cv::Point(w / 4, h / 3), std::min(w, h) / 6, cv::Scalar(200, 40, 120), -4);
    cv::circle(img, cv::Point(2 * w / 3, 3 * h / 4), std::min(w, h) / 8, cv::Scalar(30, 220, 90), -3);
    cv::rectangle(img, cv::Point(w / 2 - 60, h / 2 - 28), cv::Point(w / 2 + 72, h / 2 + 30), cv::Scalar(255, 255, 255), 5);
    return img;
}

struct PoseSpec {
    int w1, h1, nf;  // L1 模板宽/高/特征数；L0 同构两倍（引擎两级口径）
};

std::vector<vfshape::gpu_test::RefModel> synthetic_poses(const std::vector<PoseSpec>& specs,
                                                         uint64_t seed) {
    namespace t = vfshape::gpu_test;
    std::vector<t::RefModel> poses;
    uint64_t state = seed;
    for (const auto& spec : specs) {
        t::RefModel pose;
        pose.level[1].width = spec.w1;
        pose.level[1].height = spec.h1;
        pose.level[0].width = spec.w1 * 2;
        pose.level[0].height = spec.h1 * 2;
        for (int f = 0; f < spec.nf; ++f)
            pose.level[1].feats.push_back({int(lcg(state) % spec.w1), int(lcg(state) % spec.h1),
                                           int(lcg(state) % 8)});
        const int nf0 = spec.nf < 64 ? spec.nf * 2 : spec.nf;  // 特征数跨 64 桶两侧
        for (int f = 0; f < nf0; ++f)
            pose.level[0].feats.push_back({int(lcg(state) % (spec.w1 * 2)),
                                           int(lcg(state) % (spec.h1 * 2)),
                                           int(lcg(state) % 8)});
        poses.push_back(std::move(pose));
    }
    return poses;
}

long long diff_count(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    check(a.size() == b.size(), "map size mismatch in diff");
    long long count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++count;
    return count;
}

long long diff_count_u16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    check(a.size() == b.size(), "response size mismatch in diff");
    long long count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) ++count;
    return count;
}

struct RoundDiffs {
    long long angle[2] = {0, 0}, spread[2] = {0, 0}, response[2] = {0, 0};
};

RoundDiffs compare_chain(const char* name, const vfshape::gpu_test::RefChainOutput& ref,
                         const vfshape::gpu::CoarseResult& gpu, int poses, bool gate) {
    namespace t = vfshape::gpu_test;
    RoundDiffs diffs;
    for (int level = 0; level < 2; ++level) {
        if (ref.response[level].empty()) continue;
        diffs.angle[level] = diff_count(ref.maps.angle[level], gpu.maps.angle[level]);
        diffs.spread[level] = diff_count(ref.maps.spread[level], gpu.maps.spread[level]);
        for (int id = 0; id < poses; ++id)
            diffs.response[level] += diff_count_u16(ref.response[level][size_t(id)],
                                                    gpu.response[level][size_t(id)]);
    }
    std::printf("[bitexact] %s: angle L1/L0=%lld/%lld spread=%lld/%lld resp=%lld/%lld\n", name,
                diffs.angle[1], diffs.angle[0], diffs.spread[1], diffs.spread[0],
                diffs.response[1], diffs.response[0]);
    if (gate)
        check(diffs.angle[0] == 0 && diffs.angle[1] == 0 && diffs.spread[0] == 0 &&
                  diffs.spread[1] == 0 && diffs.response[0] == 0 && diffs.response[1] == 0,
              std::string(name) + ": chain maps must be bit-identical");
    return diffs;
}

struct FrameCase {
    const char* name;
    int raw_w, raw_h;
    uint64_t seed;
};

void part2_chain_bitexact(const std::vector<FrameCase>& frames) {
    namespace t = vfshape::gpu_test;
    namespace g = vfshape::gpu;
    for (const auto& frame : frames) {
        const int pw = padded_dim(frame.raw_w), ph = padded_dim(frame.raw_h);
        cv::Mat padded(ph, pw, CV_8UC3, cv::Scalar::all(0));
        synthetic_frame(frame.raw_w, frame.raw_h, frame.seed)
            .copyTo(padded(cv::Rect(kPadding, kPadding, frame.raw_w, frame.raw_h)));

        // 混合规格：常规（右缘整行回卷带天然覆盖）+ 高模板压 span_y + 超幅
        // （模板宽超图宽→vendor 出界跳过路径 + 负 span）
        const std::vector<PoseSpec> specs = {
            {56, 48, 96},
            {88, 24, 128},
            {40, 80, 40},
            {pw / 2 + 16, 32, 120},
            {24, ph / 4 + 8, 72},
        };
        const auto poses = synthetic_poses(specs, frame.seed ^ 0x9e3779b9ull);
        const int n = int(poses.size());
        int feats_total = 0;
        for (const auto& pose : poses)
            feats_total += int(pose.level[0].feats.size() + pose.level[1].feats.size());

        std::string err;
        auto* ctx = g::create_context(pw, ph, n, feats_total, 16, (pw / 4) * (ph / 4), err);
        check(ctx != nullptr, "create_context failed: " + err);
        // 驻留：feats/begin/count/dims 按 W1 布局打包（前半 L1 后半 L0）
        std::vector<g::FeatureRecord> feats;
        std::vector<int> begin(size_t(n) * 2), count(size_t(n) * 2);
        std::vector<int> widths(size_t(n) * 2, 1), heights(size_t(n) * 2, 1);
        for (int level = 1; level >= 0; --level)
            for (int id = 0; id < n; ++id) {
                const int slot = level == 1 ? id : n + id;
                begin[size_t(slot)] = int(feats.size());
                count[size_t(slot)] = int(poses[size_t(id)].level[level].feats.size());
                widths[size_t(slot)] = poses[size_t(id)].level[level].width;
                heights[size_t(slot)] = poses[size_t(id)].level[level].height;
                for (const auto& f : poses[size_t(id)].level[level].feats)
                    feats.push_back({f.x, f.y, f.label});
            }
        check(g::upload_features(ctx, feats.data(), int(feats.size()), begin.data(), count.data(),
                                 int(begin.size()), n, err),
              "upload_features failed: " + err);
        check(g::upload_template_dims(ctx, widths.data(), heights.data(), int(widths.size()),
                                      err),
              "upload_template_dims failed: " + err);
        check(g::upload_frame(ctx, padded.data, pw, ph, err), "upload_frame failed: " + err);

        // round1 复刻前基线：Spike 口径两侧（内核平移本身逐位）
        t::RefChainOutput ref_spike;
        t::run_reference_chain(padded.data, pw, ph, 30.f, t::RefMode::Spike, poses, true,
                               ref_spike);
        g::CoarseResult gpu_spike;
        check(g::run_coarse_chain(ctx, n, 30.f, g::CoarseMode::SpikeRef, true, gpu_spike,
                                  nullptr, err),
              "run_coarse_chain(SpikeRef) failed: " + err);
        compare_chain((std::string(frame.name) + " round1 spike/spike").c_str(), ref_spike,
                      gpu_spike, n, true);

        // round2 复刻后：Vendor 口径两侧（fastAtan2+回卷+span 置零，全图逐位）
        t::RefChainOutput ref_vendor;
        t::run_reference_chain(padded.data, pw, ph, 30.f, t::RefMode::Vendor, poses, true,
                               ref_vendor);
        g::CoarseResult gpu_vendor;
        check(g::run_coarse_chain(ctx, n, 30.f, g::CoarseMode::VendorExact, true, gpu_vendor,
                                  nullptr, err),
              "run_coarse_chain(VendorExact) failed: " + err);
        compare_chain((std::string(frame.name) + " round2 vendor/vendor").c_str(), ref_vendor,
                      gpu_vendor, n, true);

        // Part 3 归因：Vendor vs Spike 的 CPU 侧差异定位（Interior/边带）
        for (int level = 1; level >= 0; --level) {
            const long long angle_diff = diff_count(ref_spike.maps.angle[level],
                                                    ref_vendor.maps.angle[level]);
            long long interior = 0, edge = 0;
            const int T = level ? 8 : 4;
            const int level_w = level ? pw / 2 : pw;
            for (int id = 0; id < n; ++id) {
                const auto& a = ref_spike.response[level][size_t(id)];
                const auto& b = ref_vendor.response[level][size_t(id)];
                const auto& lv = poses[size_t(id)].level[level];
                const int W = int(a.size()) ? level_w / T : 0;
                const int H = W ? int(a.size()) / W : 0;
                const int wf = (lv.width - 1) / T + 1, hf = (lv.height - 1) / T + 1;
                const int tp = (H - hf) * W + (W - wf) + 1;
                for (size_t i = 0; i < a.size(); ++i)
                    if (a[i] != b[i]) {
                        if (int(i) >= tp)
                            ++edge;  // span 外置零带（Spike 全图写 vs vendor 零）
                        else if (int(i) % W >= W - wf)
                            ++edge;  // 右缘整行回卷带（px > span_x）
                        else
                            ++interior;
                    }
            }
            std::printf("[bitexact] %s L%d attribution: angle_flip=%lld resp interior=%lld "
                        "edge=%lld\n",
                        frame.name, level, angle_diff, interior, edge);
            check(interior == 0 || angle_diff > 0,
                  "interior response diffs require angle flips to attribute");
        }
        g::destroy_context(ctx);
    }
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void part4_bench() {
    namespace t = vfshape::gpu_test;
    namespace g = vfshape::gpu;
    const FrameCase cases[] = {
        {"1080P-padded", 1920, 1080, 0x5eed1}, {"native-padded", 1280, 960, 0x5eed2},
    };
    constexpr int kWarmup = 10, kLoops = 100;
    constexpr double kSpeedupGate = 8.0;
    for (const auto& frame : cases) {
        const int pw = padded_dim(frame.raw_w), ph = padded_dim(frame.raw_h);
        cv::Mat padded(ph, pw, CV_8UC3, cv::Scalar::all(0));
        synthetic_frame(frame.raw_w, frame.raw_h, frame.seed)
            .copyTo(padded(cv::Rect(kPadding, kPadding, frame.raw_w, frame.raw_h)));
        // 13 位姿锚规格（±30°/5° 同量级）：L1 128 特征 / L0 128 特征
        std::vector<PoseSpec> specs;
        for (int i = 0; i < 13; ++i) specs.push_back({64 + (i % 5) * 8, 40 + (i % 3) * 12, 128});
        const auto poses = synthetic_poses(specs, 0xabcd1234ull);
        const int n = int(poses.size());
        int feats_total = 0;
        for (const auto& pose : poses)
            feats_total += int(pose.level[0].feats.size() + pose.level[1].feats.size());

        std::string err;
        auto* ctx = g::create_context(pw, ph, n, feats_total, 16, (pw / 4) * (ph / 4), err);
        check(ctx != nullptr, "bench create_context failed: " + err);
        std::vector<g::FeatureRecord> feats;
        std::vector<int> begin(size_t(n) * 2), count(size_t(n) * 2);
        std::vector<int> widths(size_t(n) * 2, 1), heights(size_t(n) * 2, 1);
        for (int level = 1; level >= 0; --level)
            for (int id = 0; id < n; ++id) {
                const int slot = level == 1 ? id : n + id;
                begin[size_t(slot)] = int(feats.size());
                count[size_t(slot)] = int(poses[size_t(id)].level[level].feats.size());
                widths[size_t(slot)] = poses[size_t(id)].level[level].width;
                heights[size_t(slot)] = poses[size_t(id)].level[level].height;
                for (const auto& f : poses[size_t(id)].level[level].feats)
                    feats.push_back({f.x, f.y, f.label});
            }
        check(g::upload_features(ctx, feats.data(), int(feats.size()), begin.data(), count.data(),
                                 int(begin.size()), n, err),
              "bench upload_features failed: " + err);
        check(g::upload_template_dims(ctx, widths.data(), heights.data(), int(widths.size()),
                                      err),
              "bench upload_template_dims failed: " + err);
        check(g::upload_frame(ctx, padded.data, pw, ph, err), "bench upload_frame failed: " + err);

        g::CoarseResult result;
        g::CoarseTiming timing;
        for (int i = 0; i < kWarmup; ++i)
            check(g::run_coarse_chain(ctx, n, 30.f, g::CoarseMode::VendorExact, false, result,
                                      &timing, err),
                  "bench warmup failed: " + err);
        std::vector<double> gpu_coarse, gpu_wall, cpu_coarse;
        std::vector<double> gpu_stage[4], cpu_stage[4];
        for (int i = 0; i < kLoops; ++i) {
            check(g::run_coarse_chain(ctx, n, 30.f, g::CoarseMode::VendorExact, false, result,
                                      &timing, err),
                  "bench gpu loop failed: " + err);
            gpu_coarse.push_back(timing.coarse_ms());
            gpu_wall.push_back(timing.wall_ms);
            gpu_stage[0].push_back(timing.pyrdown_ms);
            gpu_stage[1].push_back(timing.grad1_ms);
            gpu_stage[2].push_back(timing.spread1_ms);
            gpu_stage[3].push_back(timing.resp1_ms);
        }
        t::RefChainOutput ref;
        for (int i = 0; i < kWarmup; ++i)
            t::run_reference_chain(padded.data, pw, ph, 30.f, t::RefMode::Vendor, poses, false,
                                   ref);
        for (int i = 0; i < kLoops; ++i) {
            t::RefChainOutput run;
            t::run_reference_chain(padded.data, pw, ph, 30.f, t::RefMode::Vendor, poses, false,
                                   run);
            cpu_coarse.push_back(run.coarse_ms());
            cpu_stage[0].push_back(run.pyrdown_ms);
            cpu_stage[1].push_back(run.grad1_ms);
            cpu_stage[2].push_back(run.spread1_ms);
            cpu_stage[3].push_back(run.resp1_ms);
        }
        const char* stage_names[4] = {"pyrdown", "grad_L1", "spread_L1", "resp_L1"};
        std::printf("[bench] %s (%dx%d padded, %d poses)\n", frame.name, pw, ph, n);
        for (int s = 0; s < 4; ++s)
            std::printf("[bench]   %-9s cpu=%8.3fms gpu=%7.3fms speedup=%6.2fx\n", stage_names[s],
                        median_of(cpu_stage[s]), median_of(gpu_stage[s]),
                        median_of(cpu_stage[s]) / median_of(gpu_stage[s]));
        const double cpu_med = median_of(cpu_coarse), gpu_med = median_of(gpu_coarse);
        const double speedup = cpu_med / gpu_med;
        std::printf("[bench]   coarse    cpu=%8.3fms gpu=%7.3fms (wall %7.3fms) speedup=%6.2fx "
                    "[gate >=%.1fx]\n",
                    cpu_med, gpu_med, median_of(gpu_wall), speedup, kSpeedupGate);
        check(speedup >= kSpeedupGate, "coarse speedup below 8x gate on " +
                                           std::string(frame.name) + ": " +
                                           std::to_string(speedup));
        g::destroy_context(ctx);
    }
}

}  // namespace

int main(int argc, char** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        std::string err, device_name;
        int major = 0, minor = 0;
        if (!vfshape::gpu::device_info(device_name, major, minor, err)) {
            std::printf("[bitexact] skip: no usable CUDA device (%s)\n", err.c_str());
            return kSkipCode;
        }
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120) {
            std::printf("[bitexact] skip: device sm_%d outside supported set {86,89,120}\n", sm);
            return kSkipCode;
        }
        std::printf("[bitexact] device: %s (sm_%d)\n", device_name.c_str(), sm);
        const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
        part1_fast_atan2();
        if (bench) {
            part4_bench();
        } else {
            const std::vector<FrameCase> frames = {
                {"small-640x480", 640, 480, 0x51de},
                {"1080p-1920x1080", 1920, 1080, 0x51df},
            };
            part2_chain_bitexact(frames);
        }
        std::printf("[bitexact] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gpu bitexact failed: %s\n", ex.what());
        return kFailCode;
    } catch (...) {
        std::fprintf(stderr, "gpu bitexact failed: unknown exception\n");
        return kFailCode;
    }
}

#else  // !VF_SHAPE_CUDA

#include <cstdio>

int main() {
    // VF_SHAPE_CUDA=OFF：目标形态照抄规划——无 CUDA 构建该目标返回 Skip（77），
    // ctest 显示 Skipped 而非缺席（防假绿：OFF 构建不会静默变成"无此测试"）
    std::printf("[bitexact] skip: built without VF_SHAPE_CUDA\n");
    return 77;
}

#endif
