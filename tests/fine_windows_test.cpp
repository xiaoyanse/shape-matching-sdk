// W3 波1 精修窗批量评分内核对拍（shape_fine_windows）：GPU run_window_scores
// vs 宿主 LM 镜像参考逐位（右/下缘行尾续读与 Mat 尾取模锚，场景6 同式咬合）+
// 乱序 pose 批量 + 边界 + 接缝等价 + 耗时 memo。退出码 0=过 3=失败 77=Skip。
#include <windows.h>

#if defined(VF_SHAPE_CUDA) && VF_SHAPE_CUDA

#include "../src/gpu/coarse_extract.hpp"
#include "../src/gpu/coarse_kernels.hpp"
#include "../src/gpu/fine_windows.hpp"
#include "../src/gpu/similarity_lut.h"

#include <mipp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSkipCode = 77, kFailCode = 3;
constexpr int kPadding = 32, kT0 = 4, kWinCells = 256;

static_assert(mipp::N<uint8_t>() <= 32, "mipp accumulate twin requires the 256-bit path");

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t lcg(uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 33);
}

int padded_dim(int value) { return (value + kPadding * 2 + 15) / 16 * 16; }

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// 确定性条纹+渐变纹理（Sobel 量化后多角度码分布，spread 非平凡）
void fill_frame(std::vector<uint8_t>& bgr, int w, int h, uint64_t seed) {
    bgr.assign(size_t(w) * h * 3, 0);
    uint64_t state = seed;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint32_t r = lcg(state);
            const size_t i = (size_t(y) * w + x) * 3;
            bgr[i] = uint8_t((x / 8 % 2) * 255 ^ (r & 7));
            bgr[i + 1] = uint8_t((y / 12 % 2) * 200 + x % 55);
            bgr[i + 2] = uint8_t(((x + y) / 16 % 2) * 255 - (r % 31));
        }
}

using vfshape::gpu::ExtractPoses;
using vfshape::gpu::ExtractResult;
using vfshape::gpu::FeatureRecord;
using vfshape::gpu::InitialCandidate;
using vfshape::gpu::WindowCandidate;

struct PoseSet {
    ExtractPoses extract;  // 宿主提取视图（与设备驻留同源）
    std::vector<FeatureRecord> feats;
    std::vector<int> begin, count, widths, heights;
};

// 驻留打包（前半 L1 slot=id、后半 L0 slot=pose_capacity+id，W1 布局）
void pack_residency(PoseSet& set, int pose_count) {
    set.begin.resize(size_t(pose_count) * 2);
    set.count = set.begin;
    set.widths.assign(size_t(pose_count) * 2, 1);
    set.heights = set.widths;
    for (int lv = 1; lv >= 0; --lv)
        for (int id = 0; id < pose_count; ++id) {
            const int slot = lv == 1 ? id : pose_count + id;
            const auto& src = lv == 1 ? set.extract.level1[size_t(id)] : set.extract.level0[size_t(id)];
            set.begin[size_t(slot)] = int(set.feats.size());
            set.count[size_t(slot)] = int(src.feats.size());
            set.widths[size_t(slot)] = src.width;
            set.heights[size_t(slot)] = src.height;
            set.feats.insert(set.feats.end(), src.feats.begin(), src.feats.end());
        }
}

ExtractPoses::Level rand_level(uint64_t& state, int width, int height, int nf) {
    ExtractPoses::Level lv;
    lv.width = width;
    lv.height = height;
    for (int i = 0; i < nf; ++i)
        lv.feats.push_back({int(lcg(state) % width), int(lcg(state) % height), int(lcg(state) % 8)});
    return lv;
}

ExtractPoses::Level fixed_level(int width, int height, const std::vector<FeatureRecord>& feats) {
    ExtractPoses::Level lv;
    lv.width = width;
    lv.height = height;
    lv.feats = feats;
    return lv;
}

// 混合规格：常规随机 + 贴右/下/角缘锚（特征越模板框 = 场景6 同式构造）+ 全出界
PoseSet build_poses(uint64_t seed) {
    PoseSet set;
    uint64_t state = seed;
    set.extract.level1.push_back(rand_level(state, 30, 30, 64));
    set.extract.level0.push_back(rand_level(state, 60, 60, 96));
    set.extract.level1.push_back(rand_level(state, 44, 24, 72));
    set.extract.level0.push_back(rand_level(state, 88, 48, 128));
    set.extract.level1.push_back(fixed_level(30, 30, {{1, 1, 0}}));
    set.extract.level0.push_back(
        fixed_level(60, 60, {{120, 10, 0}, {123, 13, 0}, {119, 9, 0}, {30, 30, 0}}));
    set.extract.level1.push_back(fixed_level(30, 30, {{1, 1, 0}}));
    set.extract.level0.push_back(
        fixed_level(60, 60, {{10, 120, 0}, {13, 123, 0}, {9, 119, 0}, {30, 30, 0}}));
    set.extract.level1.push_back(fixed_level(30, 30, {{1, 1, 0}}));
    set.extract.level0.push_back(
        fixed_level(60, 60, {{123, 123, 0}, {120, 123, 1}, {123, 120, 2}, {59, 59, 3}}));
    set.extract.level1.push_back(fixed_level(30, 30, {{1, 1, 0}}));
    set.extract.level0.push_back(fixed_level(60, 60, {{-60, 10, 0}, {2000, 10, 1}, {10, 2000, 2}}));
    pack_residency(set, int(set.extract.level1.size()));
    return set;
}

// 宿主参考（build_linear_memory 同布局微缩，结构异构于内核闭式索引）
struct RefLM {
    int W = 0, Hc = 0;
    size_t plane = 0, label_stride = 0;
    std::vector<uint8_t> data;
};

void ref_build_lm(const std::vector<uint8_t>& spread, int w0, int h0, RefLM& lm) {
    lm.W = w0 / kT0;
    lm.Hc = h0 / kT0;
    lm.plane = size_t(lm.W) * size_t(lm.Hc);
    const size_t mirror = size_t(16) * size_t(lm.W) + 64;
    lm.label_stride = size_t(kT0 * kT0) * lm.plane + mirror;
    lm.data.assign(size_t(8) * lm.label_stride, 0);
    for (int label = 0; label < 8; ++label) {
        const unsigned char* lut = SIMILARITY_LUT + 32 * label;
        uint8_t* base = lm.data.data() + size_t(label) * lm.label_stride;
        for (int p = 0; p < kT0 * kT0; ++p) {
            uint8_t* row = base + size_t(p) * lm.plane;
            size_t cell = 0;
            for (int cy = 0; cy < lm.Hc; ++cy)
                for (int cx = 0; cx < lm.W; ++cx, ++cell) {
                    const uint8_t b = spread[size_t(cy * kT0 + p / kT0) * w0 + cx * kT0 + p % kT0];
                    row[cell] = uint8_t(std::max(lut[b & 15], lut[16 + (b >> 4)]));
                }
        }
        std::copy_n(base, mirror, base + size_t(kT0 * kT0) * lm.plane);
    }
}

// accumulate_window 同构指针平推（行尾续读由镜像承接）；simd=true 走 MIPP 行加
void ref_accumulate(const RefLM& lm, const std::vector<FeatureRecord>& feats, int x, int y,
                    int w0, int h0, int16_t* window, bool simd) {
    const int offset_x = (x / kT0 - 8) * kT0, offset_y = (y / kT0 - 8) * kT0;
    const mipp::Reg<uint8_t> zero_v(uint8_t(0));
    for (const auto& f : feats) {
        const int fx = f.x + offset_x, fy = f.y + offset_y;
        if (fx < 0 || fy < 0 || fx >= w0 || fy >= h0) continue;
        const uint8_t* ptr = lm.data.data() + size_t(f.label) * lm.label_stride +
                             size_t((fy % kT0) * kT0 + (fx % kT0)) * lm.plane +
                             size_t((fy / kT0) * lm.W + (fx / kT0));
        for (int r = 0; r < 16; ++r) {
            int16_t* dst = window + size_t(r) * 16;
            const uint8_t* src = ptr + size_t(r) * lm.W;
            if (simd) {
                for (int col = 0; col < 16; col += mipp::N<int16_t>()) {
                    const mipp::Reg<uint8_t> src8_v(src + col);
                    const mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
                    const mipp::Reg<int16_t> dst_v(dst + col);
                    (src16_v + dst_v).store(dst + col);
                }
            } else {
                for (int c = 0; c < 16; ++c) dst[c] = int16_t(dst[c] + src[c]);
            }
        }
    }
}

// 链路真跑填充设备 spread0（lean 模式）并回读宿主副本——两侧同源
void chain_fill(vfshape::gpu::GpuContext* ctx, int poses, int pw, int ph,
                const std::vector<uint8_t>& padded, std::vector<uint8_t>& host_spread,
                std::vector<std::vector<uint16_t>>& resp1, std::string& err) {
    namespace g = vfshape::gpu;
    check(g::upload_frame(ctx, padded.data(), pw, ph, err), "upload_frame: " + err);
    g::CoarseResult out;
    check(g::run_coarse_chain(ctx, poses, 30.f, g::CoarseMode::VendorExact, true, out, nullptr,
                              err, /*need_l0_response=*/false),
          "run_coarse_chain: " + err);
    host_spread = std::move(out.maps.spread[0]);
    resp1 = std::move(out.response[1]);
}

void part_window_bitexact() {
    namespace g = vfshape::gpu;
    const int pw = padded_dim(512), ph = pw, rw = pw / 2 / 8, rh = ph / 2 / 8;
    std::vector<uint8_t> padded;
    fill_frame(padded, pw, ph, 0x5eedf111ull);
    PoseSet set = build_poses(0x9e3779b9ull);
    const int n = int(set.extract.level1.size());
    std::string err;
    g::GpuContext* ctx =
        g::create_context(pw, ph, n, int(set.feats.size()), 65536, (pw / kT0) * (ph / kT0), err);
    check(ctx != nullptr, "create_context: " + err);
    const bool up_ok =
        g::upload_features(ctx, set.feats.data(), int(set.feats.size()), set.begin.data(),
                           set.count.data(), int(set.begin.size()), n, err) &&
        g::upload_template_dims(ctx, set.widths.data(), set.heights.data(),
                                int(set.widths.size()), err);
    check(up_ok, "residency upload: " + err);
    std::vector<uint8_t> host_spread;
    std::vector<std::vector<uint16_t>> resp1;
    chain_fill(ctx, n, pw, ph, padded, host_spread, resp1, err);
    RefLM lm;
    ref_build_lm(host_spread, pw, ph, lm);

    // 乱序混排；锚点：角缘相位15取模 / 右/下缘 LM 续读 / 内部 / 全出界（500=clamp 上界）
    const std::vector<WindowCandidate> cands = {
        {500, 500}, {100, 80}, {500, 100}, {100, 500}, {400, 300}, {60, 60}, {500, 500}};
    const std::vector<int> cand_pose = {4, 0, 2, 3, 1, 5, 0};
    std::vector<uint16_t> scores;
    check(g::run_window_scores(ctx, nullptr, pw, ph, nullptr, nullptr, nullptr,
                               g::context_pose_capacity(ctx), cands.data(), cand_pose.data(),
                               int(cands.size()), scores, nullptr, err),
          "run_window_scores: " + err);
    check(scores.size() == cands.size() * size_t(kWinCells), "scores size mismatch");
    long long total_diff = 0;
    std::vector<int16_t> ref(kWinCells);
    for (size_t ci = 0; ci < cands.size(); ++ci) {
        std::fill(ref.begin(), ref.end(), 0);
        ref_accumulate(lm, set.extract.level0[size_t(cand_pose[ci])].feats, cands[ci].x,
                       cands[ci].y, pw, ph, ref.data(), false);
        const uint16_t* row = scores.data() + ci * kWinCells;
        long long diff = 0, sum = 0;
        for (int i = 0; i < kWinCells; ++i) {
            sum += row[i];
            if (row[i] != uint16_t(ref[i])) ++diff;
        }
        total_diff += diff;
        std::printf("[fine] cand%zu pose%d (%d,%d): diff=%lld sum=%lld\n", ci, cand_pose[ci],
                    cands[ci].x, cands[ci].y, diff, sum);
        if (ci == 5) check(sum == 0, "all-out-of-frame window must score zero");
        if (ci == 0 || ci == 2 || ci == 3) check(sum > 0, "edge-anchor control lost bite");
    }
    check(total_diff == 0, "gpu window scores differ from host LM reference");

    std::vector<uint16_t> empty_scores;
    check(g::run_window_scores(ctx, nullptr, pw, ph, nullptr, nullptr, nullptr,
                               g::context_pose_capacity(ctx), nullptr, nullptr, 0, empty_scores,
                               nullptr, err) &&
              empty_scores.empty(),
          "cand_count=0 must succeed empty");

    // 接缝等价：纯宿主路径 vs initial 出口回填 GPU 窗分的主入口消费
    ExtractResult pure;
    check(g::extract_coarse_candidates(resp1, rw, rh, host_spread, pw, ph, set.extract, pw, ph,
                                       15.f, pure, err),
          "pure extract: " + err);
    std::vector<InitialCandidate> icands;
    check(g::extract_initial_candidates(resp1, rw, rh, set.extract, pw, ph, 15.f, icands, err),
          "extract_initial_candidates: " + err);
    check(icands.size() == size_t(pure.initial_count), "initial count mismatch");
    check(!icands.empty(), "seam fixture produced no candidates");
    std::vector<WindowCandidate> wc(icands.size());
    std::vector<int> cp(icands.size());
    for (size_t i = 0; i < icands.size(); ++i) {
        wc[i] = {icands[i].x, icands[i].y};
        cp[i] = icands[i].pose_id;
    }
    const bool seam_gpu_ok =
        g::run_window_scores(ctx, nullptr, pw, ph, nullptr, nullptr, nullptr,
                             g::context_pose_capacity(ctx), wc.data(), cp.data(), int(wc.size()),
                             scores, nullptr, err);
    check(seam_gpu_ok, "seam run_window_scores: " + err);
    ExtractResult ext;
    check(g::extract_coarse_candidates(resp1, rw, rh, host_spread, pw, ph, set.extract, pw, ph,
                                       15.f, ext, err, &scores),
          "external extract: " + err);
    check(pure.matches == ext.matches, "external-score matches differ from pure host path");
    std::printf("[fine] seam: %zu candidates, %zu matches, pure==gpu-consumed\n", icands.size(),
                ext.matches.size());
    auto short_scores = scores;
    short_scores.resize(scores.size() - size_t(kWinCells));
    ExtractResult rejected;
    check(!g::extract_coarse_candidates(resp1, rw, rh, host_spread, pw, ph, set.extract, pw, ph,
                                        15.f, rejected, err, &short_scores) &&
              err == "external window scores count mismatch",
          "short external scores must fail: " + err);
    g::destroy_context(ctx);
}

void part_memo() {
    namespace g = vfshape::gpu;
    const int pw = padded_dim(544), ph = pw;  // 典型窗 600 档 pad 到 608（L1 304%8=0）
    std::vector<uint8_t> padded;
    fill_frame(padded, pw, ph, 0x5eedf222ull);
    PoseSet set;
    uint64_t state = 0xabcdef01ull;
    set.extract.level1.push_back(rand_level(state, 32, 32, 64));
    set.extract.level0.push_back(rand_level(state, 64, 64, 128));
    pack_residency(set, 1);
    std::string err;
    g::GpuContext* ctx =
        g::create_context(pw, ph, 1, int(set.feats.size()), 2048, (pw / kT0) * (ph / kT0), err);
    check(ctx != nullptr, "memo create_context: " + err);
    const bool memo_up_ok =
        g::upload_features(ctx, set.feats.data(), int(set.feats.size()), set.begin.data(),
                           set.count.data(), int(set.begin.size()), 1, err) &&
        g::upload_template_dims(ctx, set.widths.data(), set.heights.data(),
                                int(set.widths.size()), err);
    check(memo_up_ok, "memo residency upload: " + err);
    std::vector<uint8_t> host_spread;
    std::vector<std::vector<uint16_t>> resp1;
    chain_fill(ctx, 1, pw, ph, padded, host_spread, resp1, err);
    RefLM lm;
    ref_build_lm(host_spread, pw, ph, lm);

    std::vector<int16_t> window(kWinCells);
    std::vector<uint16_t> scores;
    double gpu_ms = 0;
    // 任务书口径 16 候选 + 批量摊销参照 1024 候选（波3 §3 门值取证数据点，非门）
    for (int batch : {16, 1024}) {
        std::vector<WindowCandidate> cands;
        std::vector<int> cand_pose;
        for (int i = 0; i < batch; ++i) {
            cands.push_back({32 + (i % 4) * 128, 32 + (i / 4 % 4) * 128});
            cand_pose.push_back(0);
        }
        std::vector<double> gpu_times;
        for (int i = 0; i < 60; ++i) {
            const bool ok =
                g::run_window_scores(ctx, nullptr, pw, ph, nullptr, nullptr, nullptr, 1,
                                     cands.data(), cand_pose.data(), batch, scores, &gpu_ms, err);
            check(ok, "memo gpu loop: " + err);
            if (i >= 10) gpu_times.push_back(gpu_ms);
        }
        std::vector<double> host_times;
        for (int i = 0; i < 55; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            for (const auto& c : cands) {
                std::memset(window.data(), 0, window.size() * sizeof(int16_t));
                ref_accumulate(lm, set.extract.level0[0].feats, c.x, c.y, pw, ph, window.data(),
                               true);
            }
            if (i >= 5) host_times.push_back(std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - t0).count());
        }
        const double gms = median_of(gpu_times), hms = median_of(host_times);
        std::printf("[memo] window 608x608 %4dcand x128feat: gpu=%.4fms host_mipp=%.4fms (%.2fus/win) speedup=%.1fx\n",
                    batch, gms, hms, hms * 1000.0 / batch, hms / gms);
    }
    g::destroy_context(ctx);
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        std::string err, name;
        int major = 0, minor = 0;
        if (!vfshape::gpu::device_info(name, major, minor, err)) {
            std::printf("[fine_windows] skip: no usable CUDA device (%s)\n", err.c_str());
            return kSkipCode;
        }
        const int sm = major * 10 + minor;
        if (sm != 86 && sm != 89 && sm != 120) {
            std::printf("[fine_windows] skip: sm_%d outside supported set\n", sm);
            return kSkipCode;
        }
        std::printf("[fine_windows] device: %s (sm_%d)\n", name.c_str(), sm);
        part_window_bitexact();
        part_memo();
        std::printf("[fine_windows] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fine_windows failed: %s\n", ex.what());
        return kFailCode;
    } catch (...) {
        std::fprintf(stderr, "fine_windows failed: unknown exception\n");
        return kFailCode;
    }
}

#else
#include <cstdio>

int main() {
    std::printf("[fine_windows] skip: built without VF_SHAPE_CUDA\n");
    return 77;
}
#endif
