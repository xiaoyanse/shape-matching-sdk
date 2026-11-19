#include "cpu_chain.h"
#include "model_source.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

#include <chrono>

namespace gpuspike {
namespace {
class ScopedTimer {
public:
    explicit ScopedTimer(double& ms) : ms_(ms), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    double& ms_;
    std::chrono::steady_clock::time_point start_;
};
}  // namespace

void cpu_gradient_quantize(const uint8_t* bgr, int w, int h, float weak, Plane& angle) {
    const cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(bgr));
    cv::Mat smoothed;
    // vendor quantizedOrientations 同参：7×7 高斯，BORDER_REPLICATE
    cv::GaussianBlur(src, smoothed, cv::Size(7, 7), 0, 0, cv::BORDER_REPLICATE);

    // strongestGradient 标量直译（vendor 非 SIMD 分支同式，整数精确）
    cv::Mat dx(h, w, CV_32F), dy(h, w, CV_32F), mag(h, w, CV_32F);
    for (int y = 0; y < h; ++y) {
        const int y_top = std::max(0, y - 1), y_bot = std::min(h - 1, y + 1);
        const uint8_t* top = smoothed.ptr<uint8_t>(y_top);
        const uint8_t* mid = smoothed.ptr<uint8_t>(y);
        const uint8_t* bot = smoothed.ptr<uint8_t>(y_bot);
        float* dx_r = dx.ptr<float>(y);
        float* dy_r = dy.ptr<float>(y);
        float* mag_r = mag.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            const int left = 3 * std::max(0, x - 1);
            const int right = 3 * std::min(w - 1, x + 1);
            const int center = 3 * x;
            int best = -1, best_x = 0, best_y = 0;
            for (int channel = 0; channel < 3; ++channel) {
                const int l = left + channel, r = right + channel, c = center + channel;
                const int gx = top[r] - top[l] + 2 * (mid[r] - mid[l]) + bot[r] - bot[l];
                const int gy = bot[l] - top[l] + 2 * (bot[c] - top[c]) + bot[r] - top[r];
                const int norm = gx * gx + gy * gy;
                if (norm > best) {
                    best = norm;
                    best_x = gx;
                    best_y = gy;
                }
            }
            dx_r[x] = float(best_x);
            dy_r[x] = float(best_y);
            mag_r[x] = float(best);
        }
    }

    // phase（度域 [0,360)）+ 16 桶量化 + 迟滞（vendor hysteresisGradient 同式）。
    // 角度用精确 atan2f 直算（vendor 走 cv::phase=fastAtan2 近似，~0.3° 表误差；
    // 参考与 GPU 同式可把对拍差异收敛到纯浮点噪声，引擎侧差异在报告边界说明）
    cv::Mat quantized(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
        const float* dx_r = dx.ptr<float>(y);
        const float* dy_r = dy.ptr<float>(y);
        uint8_t* q_r = quantized.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            float angle = std::atan2(dy_r[x], dx_r[x]) * 57.29577951308232f;
            if (angle < 0.f) angle += 360.f;
            q_r[x] = uint8_t(int(rint(double(angle) * (16.0 / 360.0))) & 7);
        }
    }
    const int qstep = int(quantized.step1());
    std::memset(quantized.ptr(), 0, size_t(qstep));
    std::memset(quantized.ptr(h - 1), 0, size_t(qstep));
    for (int r = 0; r < h; ++r) {
        quantized.at<uint8_t>(r, 0) = 0;
        quantized.at<uint8_t>(r, w - 1) = 0;
    }
    for (int r = 1; r < h - 1; ++r) {
        uint8_t* row = quantized.ptr<uint8_t>(r);
        for (int c = 1; c < w - 1; ++c) row[c] &= 7;
    }

    angle.w = w;
    angle.h = h;
    angle.data.assign(size_t(w) * h, 0);
    const float weak_sq = weak * weak;
    for (int r = 1; r < h - 1; ++r) {
        const float* mag_r = mag.ptr<float>(r);
        uint8_t* out_r = angle.data.data() + size_t(r) * w;
        for (int c = 1; c < w - 1; ++c) {
            if (mag_r[c] <= weak_sq) continue;
            int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const uint8_t* patch = quantized.ptr<uint8_t>(r - 1) + c - 1;
            histogram[patch[0]]++;
            histogram[patch[1]]++;
            histogram[patch[2]]++;
            patch += qstep;
            histogram[patch[0]]++;
            histogram[patch[1]]++;
            histogram[patch[2]]++;
            patch += qstep;
            histogram[patch[0]]++;
            histogram[patch[1]]++;
            histogram[patch[2]]++;
            int max_votes = 0, index = -1;
            for (int i = 0; i < 8; ++i) {
                if (max_votes < histogram[i]) {
                    index = i;
                    max_votes = histogram[i];
                }
            }
            if (max_votes >= 5) out_r[c] = uint8_t(1 << index);
        }
    }
}

void cpu_spread(const Plane& angle, int T, Plane& spread) {
    const int w = angle.w, h = angle.h;
    spread.w = w;
    spread.h = h;
    spread.data.assign(angle.data.size(), 0);
    // 因果窗（右/下方向 T），与 vendor spread 的两次单向 OR 等价
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = angle.data.data() + size_t(y) * w;
        uint8_t* dst = spread.data.data() + size_t(y) * w;
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int dx = 0; dx < T && x + dx < w; ++dx) v |= src[x + dx];
            dst[x] = v;
        }
    }
    std::vector<uint8_t> horizontal = spread.data;
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = horizontal.data() + size_t(y) * w;
        uint8_t* dst = spread.data.data() + size_t(y) * w;
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            for (int dy = 0; dy < T && y + dy < h; ++dy) v |= src[size_t(dy) * w + x];
            dst[x] = v;
        }
    }
}

void cpu_similarity_full(const Plane& spread, const TemplLevel& level, std::vector<uint16_t>& dst) {
    const int w = spread.w, h = spread.h, T = kT[1];
    const int W = w / T, H = h / T;
    dst.assign(size_t(W) * H, 0);
    for (const auto& f : level.feats) {
        if (f.x < 0 || f.x >= w || f.y < 0 || f.y >= h) continue;
        for (int py = 0; py < H; ++py) {
            const int fy = py * T + f.y;
            if (fy >= h) break;
            const uint8_t* spread_row = spread.data.data() + size_t(fy) * w;
            uint16_t* dst_row = dst.data() + size_t(py) * W;
            for (int px = 0; px < W; ++px) {
                const int fx = px * T + f.x;
                if (fx >= w) break;
                dst_row[px] = uint16_t(dst_row[px] + similarity_response(f.label, spread_row[fx]));
            }
        }
    }
}

void cpu_compute_windows(const Plane& spread0, const std::vector<PoseModel>& poses,
                         const std::vector<CoarseCand>& candidates, int w0, int h0,
                         std::vector<uint16_t>& scores) {
    scores.assign(candidates.size() * 256, 0);
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const int T = kT[0];
        const auto& cand = candidates[ci];
        const TemplLevel& level = poses[size_t(cand.pose_id)].level[0];
        int x = 0, y = 0;
        clamp_center(cand, level, w0, h0, x, y);
        const int offset_x = (x / T - 8) * T;
        const int offset_y = (y / T - 8) * T;
        const int w = spread0.w, h = spread0.h;
        uint16_t* out = scores.data() + ci * 256;
        for (int row = 0; row < 16; ++row)
            for (int col = 0; col < 16; ++col) {
                unsigned sum = 0;
                for (const auto& f : level.feats) {
                    const int fx = f.x + offset_x, fy = f.y + offset_y;
                    if (fx < 0 || fy < 0 || fx >= w || fy >= h) continue;
                    const int sx = fx + col * T, sy = fy + row * T;
                    if (sx >= w || sy >= h) continue;
                    sum += unsigned(similarity_response(f.label, spread0.data[size_t(sy) * w + sx]));
                }
                out[row * 16 + col] = uint16_t(sum);
            }
    }
}

void run_chain_cpu(const uint8_t* padded_bgr, int w, int h, float weak,
                   const std::vector<PoseModel>& poses,
                   float coarse_th, float fine_th, ChainOutput& out) {
    const auto start_wall = std::chrono::steady_clock::now();
    StageTimes& t = out.times;
    // pyrDown：vendor prepareImpl 同参（(w/2,h/2)）
    FrameData level1;
    {
        ScopedTimer timer(t.ms[size_t(Stage::PyrDown)]);
        const cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t*>(padded_bgr));
        cv::Mat dst;
        cv::pyrDown(src, dst, cv::Size(w / 2, h / 2));
        level1.w = dst.cols;
        level1.h = dst.rows;
        level1.bgr.assign(dst.data, dst.data + dst.total() * 3);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Grad1)]);
        cpu_gradient_quantize(level1.bgr.data(), level1.w, level1.h, weak, out.maps[1].angle);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Spread1)]);
        cpu_spread(out.maps[1].angle, kT[1], out.maps[1].spread);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Resp1)]);
        out.resp_W = out.maps[1].spread.w / kT[1];
        out.resp_H = out.maps[1].spread.h / kT[1];
        for (const auto& pose : poses) {
            if (!template_fits(pose, w, h, level1.w, level1.h)) {
                out.response.emplace_back();
                continue;
            }
            out.response.emplace_back();
            cpu_similarity_full(out.maps[1].spread, pose.level[1], out.response.back());
        }
        out.candidates = extract_candidates(out.response, out.resp_W, out.resp_H, poses, coarse_th);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Grad0)]);
        cpu_gradient_quantize(padded_bgr, w, h, weak, out.maps[0].angle);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Spread0)]);
        cpu_spread(out.maps[0].angle, kT[0], out.maps[0].spread);
    }
    {
        ScopedTimer timer(t.ms[size_t(Stage::Resp0)]);
        std::vector<uint16_t> scores;
        cpu_compute_windows(out.maps[0].spread, poses, out.candidates, w, h, scores);
        out.refined = refine_from_scores(out.candidates, poses, w, h, fine_th, scores);
    }
    out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            start_wall)
                      .count();
}

}  // namespace gpuspike
