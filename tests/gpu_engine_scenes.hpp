// W2-B/W3 引擎级 GPU 测试共用场景件：确定性纹理帧/目标贴片/训练配置/计时
// 小工具。头内 inline，供 shape_engine_bitexact 与 shape_engine_bench 复用；
// 生成逻辑与已验收的 W2-B 口径逐字节一致，改动须双侧重跑逐位门。
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace vfshape::engine_test {

inline constexpr int kSkipCode = 77, kFailCode = 3;
inline constexpr int kFrameW = 320, kFrameH = 240, kPatch = 128;
inline constexpr double kThreshold = 60.0;

inline void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void set_backend(const char* backend) {
    const std::string spec = std::string("VF_SHAPE_BACKEND=") + backend;
    check(_putenv(spec.c_str()) == 0, "set VF_SHAPE_BACKEND failed");
}

inline uint32_t lcg(uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 33);
}

inline double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

inline double wall_ms(std::chrono::steady_clock::time_point from,
                      std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

inline cv::Mat background(int w, int h, uint64_t seed) {
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
    return img;
}

inline cv::Mat sparse_background(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        uint8_t* row = img.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            row[3 * x] = uint8_t((x * 255) / (w - 1));
            row[3 * x + 1] = uint8_t((y * 255) / (h - 1));
            row[3 * x + 2] = uint8_t(((x / 48) + (y / 48)) % 2 * 200);
        }
    }
    cv::rectangle(img, {w / 6, h / 5}, {w / 3, 2 * h / 3}, {90, 160, 60}, 3);
    cv::circle(img, {3 * w / 4, h / 4}, h / 6, {60, 60, 200}, 4);
    return img;
}

inline cv::Mat target_patch(int size) {
    cv::Mat img = background(size, size, 0x7a11);
    const int cx = size / 2, cy = size / 2;
    cv::circle(img, {size / 3, size / 3}, size / 6, {200, 40, 120}, -4);
    cv::circle(img, {2 * size / 3, 2 * size / 3}, size / 5, {30, 220, 90}, -3);
    cv::rectangle(img, {cx - size * 40 / 128, cy - size * 18 / 128},
                  {cx + size * 44 / 128, cy + size * 22 / 128}, {255, 255, 255}, 5);
    return img;
}

inline cv::Mat paste(const cv::Mat& patch, int w, int h, int x, int y, uint64_t bg_seed) {
    cv::Mat scene = background(w, h, bg_seed);
    const int x0 = std::max(0, x), y0 = std::max(0, y);
    const int x1 = std::min(w, x + patch.cols), y1 = std::min(h, y + patch.rows);
    patch(cv::Rect(x0 - x, y0 - y, x1 - x0, y1 - y0))
        .copyTo(scene(cv::Rect(x0, y0, x1 - x0, y1 - y0)));
    return scene;
}

struct EngineFrame {
    const char* name;
    cv::Mat image;
};

inline cv::Mat make_reference(const cv::Mat& patch) {
    cv::Mat ref = background(256, 256, 0xfeecee);
    patch.copyTo(ref(cv::Rect(64, 64, kPatch, kPatch)));
    return ref;
}

inline vf_shape_train_config train_config(double angle_min, double angle_max, cv::Rect roi,
                                          double scale_min = 1.0, double scale_max = 1.0,
                                          double scale_step = 1.0) {
    vf_shape_train_config c{};
    c.struct_size = sizeof(vf_shape_train_config);
    c.roi_x = roi.x;
    c.roi_y = roi.y;
    c.roi_width = roi.width;
    c.roi_height = roi.height;
    c.angle_min = angle_min;
    c.angle_max = angle_max;
    c.angle_step = 5.0;
    c.scale_min = scale_min;
    c.scale_max = scale_max;
    c.scale_step = scale_step;
    return c;
}

}  // namespace vfshape::engine_test
