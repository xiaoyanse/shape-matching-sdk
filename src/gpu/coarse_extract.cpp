// 宿主侧粗层候选提取/精修（coarse_extract.hpp 语义的实现）：引擎粗层段
// match(coarse, small, th) 的 GPU 等价收尾。vendor 行号引用均指
// vendor/shape_based_matching/line2Dup.cpp（语义真相源）。
#include "coarse_extract.hpp"

#include "similarity_lut.h"

#include <mipp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace vfshape::gpu {
namespace {

static_assert(mipp::N<uint8_t>() <= 32,
              "window accumulation requires the 256-bit mipp path (vendor line2Dup.cpp same)");

// 两级金字塔档位（引擎 Detector 构造同款：L0 T=4 / L1 T=8）
constexpr int kPyramidLevels = 2;
constexpr int kT[kPyramidLevels] = {4, 8};
constexpr int kLabels = 8;
constexpr int kInitialBudget = 200000;
constexpr int kMaxFeatures = 8192;

inline constexpr int grid_offset(int T) { return T / 2 + (T % 2 - 1); }

// 响应查表：vendor responseLookup() 同式（低/高半字节取大）
inline int similarity_response(int label, uint8_t mask) {
    return std::max(SIMILARITY_LUT[32 * label + (mask & 15)],
                    SIMILARITY_LUT[32 * label + 16 + (mask >> 4)]);
}

struct CoarseCand {
    float score = 0;
    int x = 0, y = 0;  // L1 像素坐标（含档位偏移）
    int pose_id = 0;
};

// initialMatches 护栏同式（:1332-1336）：非顶层 border=16*T、顶层 border=1
bool template_fits(const ExtractPoses::Level& l0, const ExtractPoses::Level& l1,
                   int w0, int h0, int w1, int h1) {
    if (l0.width + 16 * kT[0] >= w0 || l0.height + 16 * kT[0] >= h0) return false;
    if (l1.width + 1 >= w1 || l1.height + 1 >= h1) return false;
    return true;
}

// initialMatches 同式（:1347-1358）：score = v*100/(4*nf)，>= threshold 出候选
void initial_candidates(const std::vector<uint16_t>& resp, int W, int H,
                        const ExtractPoses::Level& level1, float threshold, int pose_id,
                        std::vector<CoarseCand>& out) {
    const int T = kT[1], offset = grid_offset(T);
    const int nf = int(level1.feats.size());
    const uint16_t* map = resp.data();
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            const float score = (*map++ * 100.f) / (4 * nf);
            if (score >= threshold) out.push_back({score, c * T + offset, r * T + offset, pose_id});
        }
}

// 每帧一次线性内存预烙（vendor linearizeResponses :820-841 同布局）：8 label ×
// T*T 相位行 × W*Hc 格连续平面，值 = 响应查表预烙。平面尾镜像首段 = 行尾 flat
// 续读的确定性化——越界读 ≡ 回卷相位行 0（W2-A 先例口径），镜像按字节等值
// 展开，SIMD 块读免逐格边界分支（镜像宽度覆盖最大越界读：块起点 ≤ 平面尾-1 +
// 15W，块宽 ≤ 32B）。
struct LinearMemory {
    int W = 0, Hc = 0;
    size_t plane = 0, label_stride = 0;
    std::vector<uint8_t> data;
};

void build_linear_memory(const std::vector<uint8_t>& spread0, int w0, int h0, LinearMemory& lm) {
    const int T = kT[0];
    lm.W = w0 / T;
    lm.Hc = h0 / T;
    lm.plane = size_t(lm.W) * size_t(lm.Hc);
    const size_t mirror = size_t(16) * lm.W + 64;
    lm.label_stride = size_t(T) * T * lm.plane + mirror;
    lm.data.assign(size_t(kLabels) * lm.label_stride, 0);
    uint8_t remap[kLabels][256];
    for (int label = 0; label < kLabels; ++label)
        for (int b = 0; b < 256; ++b)
            remap[label][b] = uint8_t(similarity_response(label, uint8_t(b)));
    for (int label = 0; label < kLabels; ++label) {
        uint8_t* plane_base = lm.data.data() + size_t(label) * lm.label_stride;
        for (int p = 0; p < T * T; ++p) {
            const int py = p / T, px = p % T;
            uint8_t* row = plane_base + size_t(p) * lm.plane;
            size_t cell = 0;
            for (int cy = 0; cy < lm.Hc; ++cy) {
                const uint8_t* src = spread0.data() + size_t(cy * T + py) * w0 + px;
                for (int cx = 0; cx < lm.W; ++cx, ++cell)
                    row[cell] = remap[label][src[size_t(cx) * T]];
            }
        }
        std::copy_n(plane_base, mirror, plane_base + size_t(T) * T * lm.plane);
    }
}

// 窗分 LM 直加（vendor similarityLocal :936-985 同式）：逐特征 LM 行指针 +
// 16×16 字节累加（uint16 精确，SIMD 与标量位级同值）。行尾 flat 续读由镜像
// 平面天然承接（LM 布局即连续数组，回卷闭式在索引层消解）；出界特征跳过
//（vendor :942 不变）。
void accumulate_window(const LinearMemory& lm, const ExtractPoses::Level& level0, int x, int y,
                       int16_t* window) {
    const int T = kT[0];
    const int w0 = lm.W * T, h0 = lm.Hc * T;
    const int offset_x = (x / T - 8) * T;
    const int offset_y = (y / T - 8) * T;
    const mipp::Reg<uint8_t> zero_v(uint8_t(0));
    for (const auto& f : level0.feats) {
        const int fx = f.x + offset_x, fy = f.y + offset_y;
        if (fx < 0 || fy < 0 || fx >= w0 || fy >= h0) continue;
        const uint8_t* ptr = lm.data.data() + size_t(f.label) * lm.label_stride +
                             size_t((fy % T) * T + (fx % T)) * lm.plane +
                             size_t((fy / T) * lm.W + (fx / T));
        for (int r = 0; r < 16; ++r) {
            int16_t* dst_row = window + r * 16;
            const uint8_t* src = ptr + size_t(r) * lm.W;
            for (int col = 0; col < 16; col += mipp::N<int16_t>()) {
                const mipp::Reg<uint8_t> src8_v(src + col);
                const mipp::Reg<int16_t> src16_v(mipp::interleavelo(src8_v, zero_v).r);
                const mipp::Reg<int16_t> dst_v(dst_row + col);
                (src16_v + dst_v).store(dst_row + col);
            }
        }
    }
}

}  // namespace

bool extract_coarse_candidates(
    const std::vector<std::vector<uint16_t>>& resp1, int resp_w1, int resp_h1,
    const std::vector<uint8_t>& spread0, int spread_w, int spread_h,
    const ExtractPoses& poses, int frame_w0, int frame_h0,
    float threshold, ExtractResult& out, std::string& err) {
    out.matches.clear();
    out.extract_ms = 0;
    out.initial_count = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const auto fail = [&](const std::string& msg) {
        err = msg;
        out.matches.clear();
        out.extract_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        return false;
    };

    const int pose_count = int(poses.level1.size());
    if (pose_count == 0 || int(poses.level0.size()) != pose_count)
        return fail("extract poses: level1/level0 pose count mismatch");
    if (int(resp1.size()) != pose_count) return fail("extract resp1 pose count mismatch");
    for (int id = 0; id < pose_count; ++id)
        if (resp1[size_t(id)].size() != size_t(resp_w1) * size_t(resp_h1))
            return fail("extract resp buffer size mismatch");
    if (spread_w <= 0 || spread_h <= 0 || frame_w0 <= 0 || frame_h0 <= 0)
        return fail("extract dims must be positive");
    if (spread_w != frame_w0 || spread_h != frame_h0)
        return fail("extract spread dims must match frame dims");
    if (spread0.size() != size_t(spread_w) * size_t(spread_h))
        return fail("extract spread buffer size mismatch");
    if (frame_w0 % kT[0] || frame_h0 % kT[0])
        return fail("extract frame dims must be multiples of coarse T");
    // L1 帧= L0 半幅（pyrDown 同参），响应网格= L1 帧/T1
    const int frame_w1 = frame_w0 / 2, frame_h1 = frame_h0 / 2;
    if (resp_w1 != frame_w1 / kT[1] || resp_h1 != frame_h1 / kT[1])
        return fail("extract resp grid dims mismatch frame dims");

    // initial 全 pose 提取 + 预算（vendor matchClass :1373-1376 同式：逐 pose 累计，
    // 超 200000 中断；GPU 路径语义=清空报错，与引擎 search budget 报错口径一致）
    std::vector<CoarseCand> candidates;
    for (int id = 0; id < pose_count; ++id) {
        const auto& level1 = poses.level1[size_t(id)];
        const auto& level0 = poses.level0[size_t(id)];
        if (int(level1.feats.size()) >= kMaxFeatures || int(level0.feats.size()) >= kMaxFeatures)
            return fail("extract feature count too large");
        if (!template_fits(level0, level1, frame_w0, frame_h0, frame_w1, frame_h1)) continue;
        if (level1.feats.empty()) continue;
        initial_candidates(resp1[size_t(id)], resp_w1, resp_h1, level1, threshold, id, candidates);
        out.initial_count = int(candidates.size());
        if (out.initial_count > kInitialBudget) return fail("candidate budget exceeded");
    }

    // 精化环（vendor matchClass :1409-1498，pyramid_levels=2 → 单 l=0 轮）+
    // MatchPredicate 过滤（:1501-1503 同式）；窗分消费每帧预烙线性内存。
    // 分片按 pose 边界（候选 pose-major 连续），片结果按序拼接 = 单线程全同；
    // 候选独立计算、全局 sort 在拼接后，线程数不影响输出序列。
    LinearMemory lm;
    build_linear_memory(spread0, frame_w0, frame_h0, lm);
    const int T0 = kT[0], offset0 = grid_offset(T0);
    const int border0 = 8 * T0;
    auto refine_slice = [&](size_t begin, size_t end, std::vector<line2Dup::Match>& sink) {
        for (size_t i = begin; i < end; ++i) {
            const auto& cand = candidates[i];
            const auto& level0 = poses.level0[size_t(cand.pose_id)];
            const int nf0 = int(level0.feats.size());
            int x = std::max(cand.x * 2 + 1, border0);
            int y = std::max(cand.y * 2 + 1, border0);
            x = std::min(x, frame_w0 - level0.width - border0);
            y = std::min(y, frame_h0 - level0.height - border0);
            int16_t window[256];
            std::memset(window, 0, sizeof(window));
            accumulate_window(lm, level0, x, y, window);
            // 行主序首最大（严格 >，vendor :1453-1470 同式）
            float best_score = 0;
            int best_r = -1, best_c = -1;
            for (int r = 0; r < 16; ++r)
                for (int c = 0; c < 16; ++c) {
                    const int score_int = window[size_t(r) * 16 + c];
                    const float score = (score_int * 100.f) / (4 * nf0);
                    if (score > best_score) {
                        best_score = score;
                        best_r = r;
                        best_c = c;
                    }
                }
            if (best_r < 0) continue;  // 空窗：vendor 路径该候选 similarity=0 必被过滤，净效果同
            line2Dup::Match m(float((x / T0 - 8 + best_c) * T0 + offset0),
                              float((y / T0 - 8 + best_r) * T0 + offset0), best_score,
                              poses.class_id, cand.pose_id);
            // B5 抛物线亚像素（vendor :1478-1497 逐式：denom==0 与非峰判据短路）
            if (best_r > 0 && best_r + 1 < 16 && best_c > 0 && best_c + 1 < 16) {
                const float c_l = float(window[size_t(best_r) * 16 + best_c - 1]);
                const float c_m = float(window[size_t(best_r) * 16 + best_c]);
                const float c_r = float(window[size_t(best_r) * 16 + best_c + 1]);
                const float r_u = float(window[size_t(best_r - 1) * 16 + best_c]);
                const float r_d = float(window[size_t(best_r + 1) * 16 + best_c]);
                const float denom_c = c_l - 2 * c_m + c_r;
                const float denom_r = r_u - 2 * c_m + r_d;
                const float dc = (denom_c != 0 && c_m >= c_l && c_m >= c_r)
                                     ? 0.5f * (c_l - c_r) / denom_c : 0.f;
                const float dr = (denom_r != 0 && c_m >= r_u && c_m >= r_d)
                                     ? 0.5f * (r_u - r_d) / denom_r : 0.f;
                m.x = float((x / T0 - 8 + best_c + dc) * T0 + offset0);
                m.y = float((y / T0 - 8 + best_r + dr) * T0 + offset0);
            }
            if (m.similarity >= threshold) sink.push_back(std::move(m));
        }
    };

    // 分片多线程实测判负（本机 8 线程创建开销 ~0.5ms > 该负载并行收益，
    // extract 0.89→1.2ms）：保留单线程直算，数据上仍按片函数形态以备复评
    refine_slice(size_t(0), candidates.size(), out.matches);

    // 收尾（vendor matchPrepared :1310-1312 同式）：vendor operator</operator==
    // 排序去重——下游流式聚类合并次序依赖此序
    std::sort(out.matches.begin(), out.matches.end());
    const auto new_end = std::unique(out.matches.begin(), out.matches.end());
    out.matches.erase(new_end, out.matches.end());

    out.extract_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace vfshape::gpu
