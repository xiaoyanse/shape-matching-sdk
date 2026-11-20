// 粗层链 CPU 参考实现（对拍真相源之一）：Spike 模式为尖峰 cpu_chain.cpp 直译；
// Vendor 模式为 vendor line2Dup.cpp 语义直译（quantizedOrientations 标量路径 +
// linearizeResponses/similarity 线性内存 flat 累加，含回卷与 template_positions
// 外置零；尾格越界读 vendor 为 Mat 分配越界 UB，此处按 T*T 取模确定性化，与
// GPU 内核 coarse_kernels.cu k_sim_full 同式）。两模式 blur/pyrDown 共用 OpenCV
// 同参调用（尖峰实测与 vendor 逐字节一致）。
#include "gpu_coarse_reference.h"

#include "../src/gpu/fast_atan2.h"
#include "../src/gpu/similarity_lut.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace vfshape::gpu_test {
namespace {

constexpr int kT[2] = {4, 8};

class ScopedTimer {
public:
    explicit ScopedTimer(double& ms) : ms_(ms), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                         start_).count();
    }

private:
    double& ms_;
    std::chrono::steady_clock::time_point start_;
};

int similarity_response(int label, uint8_t mask) {
    return std::max(SIMILARITY_LUT[32 * label + (mask & 15)],
                    SIMILARITY_LUT[32 * label + 16 + (mask >> 4)]);
}

// strongestGradient 标量直译 + 迟滞前量化（模式分叉：Spike=精确 atan2+double 域
// 舍入；Vendor=fastAtan2 复刻+float 域舍入，fast_atan2.h 同式）
void gradient_quantize(const uint8_t* bgr, int w, int h, float weak, RefMode mode,
                       std::vector<uint8_t>& angle) {
    const cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(bgr));
    cv::Mat smoothed;
    // vendor quantizedOrientations 同参：7×7 高斯，BORDER_REPLICATE
    cv::GaussianBlur(src, smoothed, cv::Size(7, 7), 0, 0, cv::BORDER_REPLICATE);

    std::vector<float> mag(size_t(w) * h);
    std::vector<uint8_t> quantized(size_t(w) * h, 0);
    const float weak_sq = weak * weak;
    for (int y = 0; y < h; ++y) {
        const int y_top = std::max(0, y - 1), y_bot = std::min(h - 1, y + 1);
        const uint8_t* top = smoothed.ptr<uint8_t>(y_top);
        const uint8_t* mid = smoothed.ptr<uint8_t>(y);
        const uint8_t* bot = smoothed.ptr<uint8_t>(y_bot);
        for (int x = 0; x < w; ++x) {
            const size_t idx = size_t(y) * w + x;
            const int left = 3 * std::max(0, x - 1);
            const int right = 3 * std::min(w - 1, x + 1);
            const int center = 3 * x;
            int best = -1, best_x = 0, best_y = 0;
            for (int ch = 0; ch < 3; ++ch) {
                const int l = left + ch, r = right + ch, c = center + ch;
                const int gx = top[r] - top[l] + 2 * (mid[r] - mid[l]) + bot[r] - bot[l];
                const int gy = bot[l] - top[l] + 2 * (bot[c] - top[c]) + bot[r] - top[r];
                const int norm = gx * gx + gy * gy;
                if (norm > best) {
                    best = norm;
                    best_x = gx;
                    best_y = gy;
                }
            }
            mag[idx] = float(best);
            if (x == 0 || y == 0 || x == w - 1 || y == h - 1) continue;  // 边界置零
            if (mode == RefMode::Vendor) {
                quantized[idx] = uint8_t(vfshape::gpu::quantize_orientation(
                    vfshape::gpu::fast_atan2_deg(float(best_y), float(best_x))));
            } else {
                float ang = std::atan2(float(best_y), float(best_x)) * 57.29577951308232f;
                if (ang < 0.f) ang += 360.f;
                quantized[idx] = uint8_t(int(std::rint(double(ang) * (16.0 / 360.0))) & 7);
            }
        }
    }

    // hysteresisGradient 直译：3×3 直方图首最大，≥5 票成 one-hot；边界零
    angle.assign(size_t(w) * h, 0);
    for (int r = 1; r < h - 1; ++r) {
        for (int c = 1; c < w - 1; ++c) {
            const size_t idx = size_t(r) * w + c;
            if (mag[idx] <= weak_sq) continue;
            int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int dy = -1; dy <= 1; ++dy) {
                const uint8_t* patch = quantized.data() + size_t(r + dy) * w + c - 1;
                histogram[patch[0]]++;
                histogram[patch[1]]++;
                histogram[patch[2]]++;
            }
            int max_votes = 0, index = -1;
            for (int i = 0; i < 8; ++i) {
                if (max_votes < histogram[i]) {
                    index = i;
                    max_votes = histogram[i];
                }
            }
            if (max_votes >= 5) angle[idx] = uint8_t(1 << index);
        }
    }
}

void spread_maps(const std::vector<uint8_t>& angle, int w, int h, int T,
                 std::vector<uint8_t>& spread) {
    spread.assign(angle.size(), 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = angle.data() + size_t(y) * w;
        uint8_t* dst = spread.data() + size_t(y) * w;
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int dx = 0; dx < T && x + dx < w; ++dx) v |= src[x + dx];
            dst[x] = v;
        }
    }
    std::vector<uint8_t> horizontal = spread;
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = horizontal.data() + size_t(y) * w;
        uint8_t* dst = spread.data() + size_t(y) * w;
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int dy = 0; dy < T && y + dy < h; ++dy) v |= src[size_t(dy) * w + x];
            dst[x] = v;
        }
    }
}

// Spike 口径：尖峰 cpu_similarity_full 直译（出界特征/读跳过，全图写）
void similarity_full_spike(const std::vector<uint8_t>& spread, int w, int h, int T,
                           const RefLevel& level, std::vector<uint16_t>& dst) {
    const int W = w / T, H = h / T;
    dst.assign(size_t(W) * H, 0);
    for (const auto& f : level.feats) {
        if (f.x < 0 || f.x >= w || f.y < 0 || f.y >= h) continue;
        for (int py = 0; py < H; ++py) {
            const int fy = py * T + f.y;
            if (fy >= h) break;
            const uint8_t* spread_row = spread.data() + size_t(fy) * w;
            uint16_t* dst_row = dst.data() + size_t(py) * W;
            for (int px = 0; px < W; ++px) {
                const int fx = px * T + f.x;
                if (fx >= w) break;
                dst_row[px] = uint16_t(dst_row[px] + similarity_response(f.label, spread_row[fx]));
            }
        }
    }
}

// Vendor 口径：linearizeResponses + similarity 线性内存 flat 语义直译
void similarity_full_vendor(const std::vector<uint8_t>& spread, int w, int h, int T,
                            const RefLevel& level, std::vector<uint16_t>& dst) {
    const int W = w / T, H = h / T;
    const size_t cells = size_t(W) * H;
    // linearizeResponses 直译：lm[label][grid_row=gy*T+gx][pos]（行连续 flat）
    std::vector<uint8_t> lm(8ull * T * T * cells);
    for (int ori = 0; ori < 8; ++ori)
        for (int y = 0; y < T; ++y)
            for (int x = 0; x < T; ++x) {
                const int row = y * T + x;
                size_t p = 0;
                uint8_t* output = lm.data() + (size_t(ori) * T * T + row) * cells;
                for (int r = y; r < h; r += T)
                    for (int c = x; c < w; c += T)
                        output[p++] =
                            uint8_t(similarity_response(ori, spread[size_t(r) * w + c]));
            }
    // similarity 直译：dst=Mat::zeros；flat j < template_positions 累加
    // lm_ptr[j] 越过本格行尾续读下一格（Mat 行连续）；尾格越界按 T*T 取模
    const int wf = (level.width - 1) / T + 1, hf = (level.height - 1) / T + 1;
    const int template_positions = (H - hf) * W + (W - wf) + 1;
    dst.assign(cells, 0);
    if (template_positions <= 0) return;
    const size_t label_cells = size_t(T) * T * cells;
    for (const auto& f : level.feats) {
        if (f.x < 0 || f.x >= w || f.y < 0 || f.y >= h) continue;
        const size_t grid = size_t(f.y % T) * T + f.x % T;
        const size_t lm_index = size_t(f.y / T) * W + f.x / T;
        const uint8_t* base = lm.data() + size_t(f.label) * label_cells;
        for (int j = 0; j < template_positions; ++j) {
            size_t offset = grid * cells + lm_index + size_t(j);
            if (offset >= label_cells) offset %= label_cells;  // 尾格 UB 确定性化
            dst[size_t(j)] = uint16_t(dst[size_t(j)] + base[offset]);
        }
    }
}

}  // namespace

void run_reference_chain(const uint8_t* padded_bgr, int w, int h, float weak, RefMode mode,
                         const std::vector<RefModel>& poses, bool l0_too,
                         RefChainOutput& out) {
    const cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(padded_bgr));
    cv::Mat level1;
    {
        ScopedTimer timer(out.pyrdown_ms);
        cv::pyrDown(src, level1, cv::Size(w / 2, h / 2));
    }
    const int w1 = level1.cols, h1 = level1.rows;
    {
        ScopedTimer timer(out.grad1_ms);
        gradient_quantize(level1.data, w1, h1, weak, mode, out.maps.angle[1]);
    }
    {
        ScopedTimer timer(out.spread1_ms);
        spread_maps(out.maps.angle[1], w1, h1, kT[1], out.maps.spread[1]);
    }
    {
        ScopedTimer timer(out.resp1_ms);
        out.resp_w[1] = w1 / kT[1];
        out.resp_h[1] = h1 / kT[1];
        out.response[1].clear();
        for (const auto& pose : poses) {
            out.response[1].emplace_back();
            if (mode == RefMode::Vendor)
                similarity_full_vendor(out.maps.spread[1], w1, h1, kT[1], pose.level[1],
                                       out.response[1].back());
            else
                similarity_full_spike(out.maps.spread[1], w1, h1, kT[1], pose.level[1],
                                      out.response[1].back());
        }
    }
    out.maps.w[1] = w1;
    out.maps.h[1] = h1;
    if (!l0_too) return;

    {
        ScopedTimer timer(out.grad0_ms);
        gradient_quantize(padded_bgr, w, h, weak, mode, out.maps.angle[0]);
    }
    {
        ScopedTimer timer(out.spread0_ms);
        spread_maps(out.maps.angle[0], w, h, kT[0], out.maps.spread[0]);
    }
    {
        ScopedTimer timer(out.resp0_ms);
        out.resp_w[0] = w / kT[0];
        out.resp_h[0] = h / kT[0];
        out.response[0].clear();
        for (const auto& pose : poses) {
            out.response[0].emplace_back();
            if (mode == RefMode::Vendor)
                similarity_full_vendor(out.maps.spread[0], w, h, kT[0], pose.level[0],
                                       out.response[0].back());
            else
                similarity_full_spike(out.maps.spread[0], w, h, kT[0], pose.level[0],
                                      out.response[0].back());
        }
    }
    out.maps.w[0] = w;
    out.maps.h[0] = h;
}

}  // namespace vfshape::gpu_test
