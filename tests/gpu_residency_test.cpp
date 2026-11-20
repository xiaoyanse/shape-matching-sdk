// GPU 骨架冒烟（W1，规划 gpu-port-phase0.md 验收门③）：sm_120 上创建上下文 +
// 从引擎内存 Model 提特征驻留 + 1080P 帧上传。断言口径：
//  1) 驻留特征数与引擎 Detector 模板逐项一致（pose 数/每层特征数/总数字段级对账，
//     设备回读 x/y/label 逐字段比对，begin/count 布局沿尖峰前半 L1 后半 L0）；
//  2) 帧 H2D→D2H 往返逐字节一致（引擎 prepare 面同式补边帧）；
//  3) LUT 同源：gen_lut.py 产物（vendor 逐字提取）上传 __constant__ 后回读比对；
//  4) 容量护栏负路径（pose 超容量/帧尺寸不符/未驻留回读）拒绝且不崩。
// 纪律：SetErrorMode 关系统错误弹窗；异常路径全 try/catch 退出码化（0=过 3=错）。
#include "../src/gpu/gpu_backend.hpp"
#include "../src/gpu/similarity_lut.h"

#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.hpp"

namespace {

constexpr int kFrameW = 1920, kFrameH = 1080;  // 1080P 冒烟帧（验收口径）
constexpr int kPadding = vfshape::PreparedFrame::padding;
constexpr int kEngineMaxPoses = 350;              // 引擎 model.hpp max_poses
constexpr int kMaxFeaturesPerLevel = 512;         // 引擎 model.hpp 上限

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int padded_dim(int value) {
    return (value + kPadding * 2 + 15) / 16 * 16;  // 引擎 prepare_frame 同式
}

// 训练图案与档位（沿 clustering_ambiguity_test 的构造：非对称双矩形，边缘充足）
cv::Mat reference_bgr() {
    cv::Mat img(96, 96, CV_8UC1, cv::Scalar(0));
    cv::rectangle(img, {2, 2}, {93, 93}, cv::Scalar(255), 6);
    cv::rectangle(img, {16, 16}, {39, 39}, cv::Scalar(255), -1);
    cv::Mat canvas(128, 128, CV_8UC1, cv::Scalar(0));
    img.copyTo(canvas(cv::Rect(16, 16, 96, 96)));
    cv::Mat bgr;
    cv::merge(std::vector<cv::Mat>{canvas, canvas, canvas}, bgr);
    return bgr;
}

vf_shape_train_config train_config() {
    vf_shape_train_config config{};
    config.struct_size = sizeof(config);
    config.roi_x = 16;
    config.roi_y = 16;
    config.roi_width = 96;
    config.roi_height = 96;
    config.angle_min = -20;  // 三档 -20/0/20：多姿态驻留对账
    config.angle_max = 20;
    config.angle_step = 20;
    config.scale_min = 1;
    config.scale_max = 1;
    config.scale_step = 1;
    config.weak_th = 0;
    config.strong_th = 0;
    config.feature_count = 0;  // 库默认参（128/30/60）
    return config;
}

// 1080P 确定性纹理帧（渐变 + 棋盘 + 圆：非平凡字节分布，回读比对有区分度）
cv::Mat frame_1080p() {
    cv::Mat image(kFrameH, kFrameW, CV_8UC3);
    for (int y = 0; y < kFrameH; ++y) {
        uint8_t* row = image.ptr<uint8_t>(y);
        for (int x = 0; x < kFrameW; ++x) {
            row[3 * x] = uint8_t((x * 255) / (kFrameW - 1));            // B 渐变
            row[3 * x + 1] = uint8_t((y * 255) / (kFrameH - 1));        // G 渐变
            row[3 * x + 2] = uint8_t(((x / 32) + (y / 32)) % 2 * 255);  // R 棋盘
        }
    }
    cv::circle(image, {kFrameW / 2, kFrameH / 2}, 240, {64, 200, 32}, -3);
    return image;
}

}  // namespace

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        // —— 设备探测：支持集 {sm_86, sm_89, sm_120}（三代 fatbin 口径）——
        std::string err, device_name;
        int major = 0, minor = 0;
        check(vfshape::gpu::device_info(device_name, major, minor, err),
              "device_info failed: " + err);
        const int sm = major * 10 + minor;
        check(sm == 86 || sm == 89 || sm == 120,
              "device compute capability not in {sm_86, sm_89, sm_120}: sm_" + std::to_string(sm));
        std::printf("[gpu-smoke] device: %s (sm_%d%d)\n", device_name.c_str(), major, minor);

        // —— LUT 同源：宿主表（gen_lut.py 从 vendor 逐字提取）上传后回读比对 ——
        unsigned char device_lut[256] = {};
        check(vfshape::gpu::lut_upload(SIMILARITY_LUT, err), "lut_upload failed: " + err);
        check(vfshape::gpu::lut_readback(device_lut, err), "lut_readback failed: " + err);
        check(std::memcmp(device_lut, SIMILARITY_LUT, 256) == 0,
              "device LUT residency differs from gen_lut.py host table");
        std::printf("[gpu-smoke] lut residency: 256/256 bytes identical\n");

        // —— 训练三档模型，统计引擎侧期望驻留量 ——
        const auto model = vfshape::train(reference_bgr(), train_config());
        const int pose_count = int(model->poses.size());
        check(pose_count == 3, "expected 3 trained poses, got " + std::to_string(pose_count));
        std::vector<std::vector<int>> expected(2);  // [level][pose] 特征数
        int expected_total[2] = {0, 0};
        for (int level = 0; level < 2; ++level)
            for (int id = 0; id < pose_count; ++id) {
                const auto& pyramid = model->detector.getTemplates("shape", id);
                check(pyramid.size() == 2 &&
                          pyramid[size_t(level)].pyramid_level == level,
                      "engine template pyramid layout mismatch");
                const int count = int(pyramid[size_t(level)].features.size());
                expected[size_t(level)].push_back(count);
                expected_total[level] += count;
            }

        // —— 上下文（容量=引擎上限 × 1080P padded 帧档）+ 特征驻留 ——
        const int padded_w = padded_dim(kFrameW), padded_h = padded_dim(kFrameH);
        auto* ctx = vfshape::gpu::create_context(
            padded_w, padded_h, kEngineMaxPoses, kEngineMaxPoses * 2 * kMaxFeaturesPerLevel,
            16384, (padded_w / 8) * (padded_h / 8), err);
        check(ctx != nullptr, "create_context failed: " + err);
        vfshape::gpu::ResidencyStats stats;
        check(vfshape::gpu::upload_model_features(ctx, *model, err, &stats),
              "upload_model_features failed: " + err);
        check(stats.pose_count == pose_count &&
                  stats.features_level0 == expected_total[0] &&
                  stats.features_level1 == expected_total[1] &&
                  stats.feature_total == expected_total[0] + expected_total[1],
              "residency stats mismatch with engine template counts");
        std::printf("[gpu-smoke] residency: %d poses, %d+%d features (L0+L1)\n",
                    stats.pose_count, stats.features_level0, stats.features_level1);

        // —— 驻留回读逐字段对账：布局前半 L1（slot=id）/后半 L0（slot=cap+id）——
        std::vector<vfshape::gpu::FeatureRecord> feats;
        std::vector<int> begin, count;
        check(vfshape::gpu::readback_features(ctx, feats, begin, count, err),
              "readback_features failed: " + err);
        check(feats.size() == size_t(stats.feature_total) &&
                  begin.size() == size_t(kEngineMaxPoses) * 2 &&
                  count.size() == size_t(kEngineMaxPoses) * 2,
              "readback buffer sizes mismatch");
        for (int level = 0; level < 2; ++level) {
            for (int id = 0; id < pose_count; ++id) {
                const int slot = level == 1 ? id : kEngineMaxPoses + id;
                const int n = expected[size_t(level)][size_t(id)];
                check(count[size_t(slot)] == n,
                      "pose " + std::to_string(id) + " level " + std::to_string(level) +
                          " feature count mismatch");
                const auto& pyramid = model->detector.getTemplates("shape", id);
                const auto& src = pyramid[size_t(level)].features;
                check(begin[size_t(slot)] + n <= int(feats.size()), "feature range out of buffer");
                for (int f = 0; f < n; ++f) {
                    const auto& got = feats[size_t(begin[size_t(slot)] + f)];
                    check(got.x == src[size_t(f)].x && got.y == src[size_t(f)].y &&
                              got.label == src[size_t(f)].label,
                          "feature field mismatch at pose/level/idx " +
                              std::to_string(id) + "/" + std::to_string(level) + "/" +
                              std::to_string(f));
                }
            }
        }
        std::printf("[gpu-smoke] residency readback: all feature fields identical\n");

        // —— 1080P 帧：引擎 prepare 面同式补边 → H2D → D2H 逐字节 ——
        const cv::Mat image = frame_1080p();
        cv::Mat padded(padded_h, padded_w, CV_8UC3, cv::Scalar::all(0));
        image.copyTo(padded(cv::Rect(kPadding, kPadding, image.cols, image.rows)));
        check(vfshape::gpu::upload_frame(ctx, padded.data, padded_w, padded_h, err),
              "upload_frame failed: " + err);
        cv::Mat roundtrip(padded_h, padded_w, CV_8UC3, cv::Scalar::all(0));
        check(vfshape::gpu::readback_frame(ctx, roundtrip.data, padded_w, padded_h, err),
              "readback_frame failed: " + err);
        check(std::memcmp(padded.data, roundtrip.data, size_t(padded_w) * padded_h * 3) == 0,
              "frame roundtrip bytes differ");
        std::printf("[gpu-smoke] frame %dx%d -> padded %dx%d roundtrip: byte-identical\n",
                    kFrameW, kFrameH, padded_w, padded_h);

        // —— 容量护栏负路径：拒绝且不崩（退出码化，不触发弹窗/异常逃逸）——
        std::string neg;
        check(!vfshape::gpu::upload_frame(ctx, padded.data, padded_w - 16, padded_h, neg) &&
                  !neg.empty(),
              "frame size mismatch must be rejected");
        neg.clear();
        // 变量名避开 small：windows.h -> rpcndr.h 定义 #define small char 宏
        auto* overflow_ctx = vfshape::gpu::create_context(padded_w, padded_h, 2,
                                                          kMaxFeaturesPerLevel * 4, 16, 4096,
                                                          err);
        check(overflow_ctx != nullptr, "overflow create_context failed: " + err);
        check(!vfshape::gpu::upload_model_features(overflow_ctx, *model, neg) && !neg.empty(),
              "pose overflow must be rejected");
        std::vector<vfshape::gpu::FeatureRecord> no_feats;
        std::vector<int> no_begin, no_count;
        neg.clear();
        check(!vfshape::gpu::readback_features(overflow_ctx, no_feats, no_begin, no_count,
                                               neg) &&
                  !neg.empty(),
              "readback before residency must be rejected");
        vfshape::gpu::destroy_context(overflow_ctx);
        vfshape::gpu::destroy_context(ctx);
        std::printf("[gpu-smoke] guard rails: size/pose/residency rejections ok\n");
        std::printf("[gpu-smoke] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "gpu residency smoke failed: %s\n", ex.what());
        return 3;
    } catch (...) {
        std::fprintf(stderr, "gpu residency smoke failed: unknown exception\n");
        return 3;
    }
}
