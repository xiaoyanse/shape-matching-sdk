#ifndef GPU_SPIKE_COMMON_H
#define GPU_SPIKE_COMMON_H
// GPU 尖峰共享定义：模板/中间图结构、相似度查表、候选提取与精修共享实现。
// CPU 参考与 GPU 两侧必须走同一套提取/精修代码，保证对拍差异只来自内核。
#include "similarity_lut.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace gpuspike {

// 与引擎 src/model.hpp Detector 构造一致：两级金字塔，level0 T=4 / level1 T=8
constexpr int kPyramidLevels = 2;
constexpr int kT[kPyramidLevels] = {4, 8};
// 引擎 matchClass 精修档位偏移：T/2 + (T%2 - 1)
inline constexpr int grid_offset(int T) { return T / 2 + (T % 2 - 1); }

struct TemplFeature {
    int32_t x = 0, y = 0, label = 0;
};
struct TemplLevel {
    int width = 0, height = 0, tl_x = 0, tl_y = 0;
    std::vector<TemplFeature> feats;
};
struct PoseModel {
    double angle = 0, scale = 1;
    double transform[6] = {1, 0, 0, 0, 1, 0};  // 训练位姿变换（容器原样）
    TemplLevel level[kPyramidLevels];
};

// 响应查表：与 vendor responseLookup() 同式（低/高半字节取大）
inline int similarity_response(int label, uint8_t mask) {
    return std::max(SIMILARITY_LUT[32 * label + (mask & 15)],
                    SIMILARITY_LUT[32 * label + 16 + (mask >> 4)]);
}

struct Plane {
    int w = 0, h = 0;
    std::vector<uint8_t> data;
};
struct LevelMaps {
    Plane angle;   // 迟滞后 one-hot 方向码
    Plane spread;  // 因果 T×T OR 扩散
};
struct CoarseCand {
    float score = 0;
    int x = 0, y = 0;  // level1 像素坐标（含档位偏移）
    int pose_id = 0;
};
struct RefinedCand {
    float score = 0;
    float x = 0, y = 0;  // level0 亚像素坐标
    int pose_id = 0;
};

enum class Stage { PyrDown = 0, Grad1, Spread1, Resp1, Grad0, Spread0, Resp0, Count };
struct StageTimes {
    double ms[static_cast<size_t>(Stage::Count)] = {};
    double total() const {
        double sum = 0;
        for (double v : ms) sum += v;
        return sum;
    }
};
inline const char* stage_name(Stage s) {
    switch (s) {
        case Stage::PyrDown: return "pyrDown";
        case Stage::Grad1: return "grad_L1";
        case Stage::Spread1: return "spread_L1";
        case Stage::Resp1: return "resp_L1";
        case Stage::Grad0: return "grad_L0";
        case Stage::Spread0: return "spread_L0";
        case Stage::Resp0: return "resp_L0(win)";
        default: return "?";
    }
}

struct ChainOutput {
    LevelMaps maps[kPyramidLevels];
    std::vector<std::vector<uint16_t>> response;  // 每 pose 的 level1 响应图
    int resp_W = 0, resp_H = 0;
    std::vector<CoarseCand> candidates;
    std::vector<RefinedCand> refined;
    StageTimes times;
    double wall_ms = 0;          // 链路墙钟（含 H2D，视 upload 开关）
    double host_extract_ms = 0;  // GPU 路径的宿主侧候选提取开销
};

// 模板尺寸护栏（initialMatches 同式：顶层 border=1，非顶层 border=16*T）
inline bool template_fits(const PoseModel& pose, int w0, int h0, int w1, int h1) {
    for (int level = 0; level < kPyramidLevels; ++level) {
        const int border = level + 1 < kPyramidLevels ? 16 * kT[level] : 1;
        const int w = level ? w1 : w0, h = level ? h1 : h0;
        const auto& t = pose.level[level];
        if (t.width + border >= w || t.height + border >= h) return false;
    }
    return true;
}

// 粗层候选提取（与 initialMatches 同式；响应图由调用侧给出）
inline std::vector<CoarseCand> extract_candidates(
    const std::vector<std::vector<uint16_t>>& response, int W, int H,
    const std::vector<PoseModel>& poses, float threshold) {
    std::vector<CoarseCand> out;
    const int T = kT[1], offset = grid_offset(T);
    for (size_t id = 0; id < poses.size(); ++id) {
        const int nf = int(poses[id].level[1].feats.size());
        if (nf == 0 || response[id].size() != size_t(W) * H) continue;  // 不合规模板无响应
        const uint16_t* map = response[id].data();
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                const float score = (*map++) * 100.f / (4 * nf);
                if (score >= threshold) out.push_back({score, c * T + offset, r * T + offset, int(id)});
            }
        if (out.size() > 200000) return {};
    }
    return out;
}

// 精修窗中心夹取（vendor matchClass 同式）：x=cand*2+1 后按 border/模板宽高夹
inline void clamp_center(const CoarseCand& cand, const TemplLevel& templ, int w0, int h0,
                         int& x, int& y) {
    const int border = 8 * kT[0];
    x = std::max(cand.x * 2 + 1, border);
    y = std::max(cand.y * 2 + 1, border);
    x = std::min(x, w0 - templ.width - border);
    y = std::min(y, h0 - templ.height - border);
}

// 16×16 局部窗分数批量计算后共享精修：坐标映射/夹取/行主序首最大/B5 抛物线
// 亚像素与 vendor matchClass 精修环同式；scores 布局 = 候选×256（行主序窗格）
inline std::vector<RefinedCand> refine_from_scores(
    const std::vector<CoarseCand>& candidates, const std::vector<PoseModel>& poses,
    int w0, int h0, float threshold, const std::vector<uint16_t>& scores) {
    const int T = kT[0], offset = grid_offset(T);
    std::vector<RefinedCand> out;
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const auto& cand = candidates[ci];
        const auto& templ = poses[cand.pose_id].level[0];
        const int nf = int(templ.feats.size());
        int x = 0, y = 0;
        clamp_center(cand, templ, w0, h0, x, y);
        const uint16_t* scores16 = scores.data() + ci * 256;
        float best_score = 0;
        int best_r = -1, best_c = -1;
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) {
                const float score = scores16[r * 16 + c] * 100.f / (4 * nf);
                if (score > best_score) { best_score = score; best_r = r; best_c = c; }
            }
        if (best_r < 0) continue;
        RefinedCand m{best_score, float((x / T - 8 + best_c) * T + offset),
                      float((y / T - 8 + best_r) * T + offset), cand.pose_id};
        if (best_r > 0 && best_r + 1 < 16 && best_c > 0 && best_c + 1 < 16) {
            const float c_l = scores16[best_r * 16 + best_c - 1];
            const float c_m = scores16[best_r * 16 + best_c];
            const float c_r = scores16[best_r * 16 + best_c + 1];
            const float r_u = scores16[(best_r - 1) * 16 + best_c];
            const float r_d = scores16[(best_r + 1) * 16 + best_c];
            const float denom_c = c_l - 2 * c_m + c_r;
            const float denom_r = r_u - 2 * c_m + r_d;
            const float dc = (denom_c != 0 && c_m >= c_l && c_m >= c_r)
                                 ? 0.5f * (c_l - c_r) / denom_c : 0.f;
            const float dr = (denom_r != 0 && c_m >= r_u && c_m >= r_d)
                                 ? 0.5f * (r_u - r_d) / denom_r : 0.f;
            m.x = (x / T - 8 + best_c + dc) * T + offset;
            m.y = (y / T - 8 + best_r + dr) * T + offset;
        }
        if (m.score >= threshold) out.push_back(m);
    }
    return out;
}

}  // namespace gpuspike

#endif
