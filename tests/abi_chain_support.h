// ABI 合同测试共享支撑：合成图、训练/导出/加载/匹配辅助、容器段寻址与
// 篡改重构、detector 参数字段读写。仅测试域使用，非交付库代码。
#pragma once

#include "shape_matching.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace abitest {

inline char g_error[2048] = {};

inline void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 句柄异常展开时自动释放，避免失败路径泄漏注册表槽位
struct HandleGuard {
    uint64_t handle = 0;
    ~HandleGuard() {
        if (handle) vf_shape_close(handle, nullptr, 0);
    }
};

// 多句柄批量释放（容量隔离组等动态句柄场景）
struct HandleSet {
    std::vector<uint64_t> handles;
    ~HandleSet() {
        for (uint64_t handle : handles) vf_shape_close(handle, nullptr, 0);
    }
};

// 合成参考图 320×320：白外框 + 内部块（与绑定侧冒烟数据同构造）
inline cv::Mat synth_reference() {
    cv::Mat a(320, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(a, {42, 42}, {47, 197}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(a, {42, 42}, {197, 47}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(a, {42, 192}, {197, 197}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(a, {192, 42}, {197, 197}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(a, {65, 65}, {94, 94}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(a, {90, 120}, {154, 154}, cv::Scalar(200, 200, 200), -1);
    cv::rectangle(a, {165, 80}, {170, 179}, cv::Scalar(255, 255, 255), -1);
    return a;
}

inline vf_shape_train_config make_config(const cv::Rect& roi, double amin, double amax,
                                         double astep, double smin, double smax, double sstep,
                                         double weak = 0, double strong = 0, int features = 0) {
    vf_shape_train_config c{};
    c.struct_size = sizeof(c);
    c.roi_x = roi.x;
    c.roi_y = roi.y;
    c.roi_width = roi.width;
    c.roi_height = roi.height;
    c.angle_min = amin;
    c.angle_max = amax;
    c.angle_step = astep;
    c.scale_min = smin;
    c.scale_max = smax;
    c.scale_step = sstep;
    c.weak_th = weak;
    c.strong_th = strong;
    c.feature_count = features;
    return c;
}

inline uint64_t image_bytes(const cv::Mat& image) { return image.total() * image.elemSize(); }

inline uint64_t train_handle(const cv::Mat& image, const vf_shape_train_config& config,
                             const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc = vf_shape_train(image.data, image_bytes(image), image.cols, image.rows,
                                      image.step[0], &config, &handle, g_error, sizeof(g_error));
    check(rc == 0, what + " 训练失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(handle != 0, what + " 训练返回空句柄");
    return handle;
}

inline std::vector<uint8_t> export_bytes(uint64_t handle, const std::string& what) {
    uint64_t size = 0;
    int32_t rc = vf_shape_export(handle, nullptr, 0, &size, g_error, sizeof(g_error));
    check(rc == 0 && size > 0, what + " 导出长度查询失败 rc=" + std::to_string(rc) + "：" + g_error);
    std::vector<uint8_t> bytes(size_t(size), 0);
    uint64_t written = 0;
    rc = vf_shape_export(handle, bytes.data(), size, &written, g_error, sizeof(g_error));
    check(rc == 0, what + " 导出失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(written == size, what + " 两段导出长度不一致：查询 " + std::to_string(size) + " 实写 " +
                               std::to_string(written));
    return bytes;
}

inline uint64_t load_handle(const std::vector<uint8_t>& bytes, const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc =
        vf_shape_load(bytes.data(), bytes.size(), &handle, g_error, sizeof(g_error));
    check(rc == 0, what + " 加载失败 rc=" + std::to_string(rc) + "：" + g_error);
    return handle;
}

inline uint32_t match_count(uint64_t handle, const cv::Mat& image, double threshold,
                            std::vector<vf_shape_candidate>& out, const std::string& what) {
    uint32_t count = 0;
    const int32_t rc =
        vf_shape_match(handle, image.data, image_bytes(image), image.cols, image.rows,
                       image.step[0], threshold, out.data(), uint32_t(out.size()), &count,
                       g_error, sizeof(g_error));
    check(rc == 0, what + " 匹配失败 rc=" + std::to_string(rc) + "：" + g_error);
    return count;
}

inline vf_shape_batch_item make_item(uint64_t handle, double threshold,
                                     std::vector<vf_shape_candidate>& out) {
    vf_shape_batch_item item{};
    item.handle = handle;
    item.threshold = threshold;
    item.candidates = out.data();
    item.capacity = uint32_t(out.size());
    item.status = -1;
    return item;
}

inline int32_t run_batch(const cv::Mat& image, std::vector<vf_shape_batch_item>& items) {
    return vf_shape_match_batch(image.data, image_bytes(image), image.cols, image.rows,
                                image.step[0], items.data(), uint32_t(items.size()),
                                sizeof(vf_shape_batch_item), g_error, sizeof(g_error));
}

inline uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}

// 容器按索引取段：0..n-1=精修段，n=粗段，n+1=模板 PNG，n+2=meta
inline std::vector<uint8_t> segment_view(const std::vector<uint8_t>& container, size_t index) {
    check(container.size() >= 13 && std::memcmp(container.data(), "VF2S3", 5) == 0,
          "容器 magic/长度不符");
    const size_t n = size_t(read_u64_le(container.data() + 5));
    check(n >= 1 && n <= 64 && index < n + 3, "容器段索引越界");
    size_t cursor = 13 + 8 * (n + 3);
    for (size_t i = 0; i < index; ++i)
        cursor += size_t(read_u64_le(container.data() + 13 + 8 * i));
    const uint64_t len = read_u64_le(container.data() + 13 + 8 * index);
    check(cursor + len <= container.size(), "段数据越界");
    return {container.begin() + ptrdiff_t(cursor), container.begin() + ptrdiff_t(cursor + len)};
}

inline size_t container_segments(const std::vector<uint8_t>& container) {
    return size_t(read_u64_le(container.data() + 5)) + 3;
}

// 首精修段（detector 参数定位与单级资产构造用）
inline std::vector<uint8_t> first_fine_segment(const std::vector<uint8_t>& container) {
    return segment_view(container, 0);
}

// 替换指定段并重写长度表（篡改样本重构）；格式对齐 encoder：长度表连续块+数据连续块
inline std::vector<uint8_t> replace_segment(const std::vector<uint8_t>& container, size_t index,
                                            const std::vector<uint8_t>& replacement) {
    const size_t total = container_segments(container);
    std::vector<std::vector<uint8_t>> parts;
    for (size_t i = 0; i < total; ++i)
        parts.push_back(i == index ? replacement : segment_view(container, i));
    std::vector<uint8_t> out(container.begin(), container.begin() + ptrdiff_t(13));
    for (const auto& part : parts) {
        for (int shift = 0; shift < 64; shift += 8)
            out.push_back(uint8_t(uint64_t(part.size()) >> shift));
    }
    for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

// meta 段文本替换（子串唯一命中，未命中即失败）
inline std::vector<uint8_t> with_meta_text(const std::vector<uint8_t>& container,
                                           const std::string& from, const std::string& to) {
    std::vector<uint8_t> meta = segment_view(container, container_segments(container) - 1);
    std::string text(meta.begin(), meta.end());
    const size_t at = text.find(from);
    check(at != std::string::npos, "meta 段未找到目标文本：" + from);
    text.replace(at, from.size(), to);
    return replace_segment(container, container_segments(container) - 1,
                           std::vector<uint8_t>(text.begin(), text.end()));
}

// 单级段 detector 三值定位：签名 NUL + schema/宽/高(12) + 训练配置(64)
inline size_t detector_offset(const std::vector<uint8_t>& segment) {
    size_t zero = 0;
    while (zero < segment.size() && segment[zero] != 0) ++zero;
    check(zero < segment.size(), "段签名缺少 NUL 结尾");
    const size_t offset = zero + 1 + 12 + 64;
    check(offset + 20 <= segment.size(), "段过短，detector 参数越界");
    return offset;
}

struct DetectorParams {
    int features;
    double weak, strong;
};

inline DetectorParams segment_detector(const std::vector<uint8_t>& segment) {
    const size_t offset = detector_offset(segment);
    auto read_f64 = [&](size_t at) {
        uint64_t bits = read_u64_le(segment.data() + at);
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    return {int(read_u64_le(segment.data() + offset)), read_f64(offset + 4),
            read_f64(offset + 12)};
}

inline std::vector<uint8_t> mutate_detector(const std::vector<uint8_t>& segment, int features,
                                            double weak, double strong) {
    std::vector<uint8_t> out = segment;
    const size_t offset = detector_offset(out);
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) out[at + i] = uint8_t(v >> (8 * i));
    };
    auto put64 = [&](size_t at, uint64_t v) {
        for (int i = 0; i < 8; ++i) out[at + i] = uint8_t(v >> (8 * i));
    };
    put32(offset, uint32_t(features));
    uint64_t bits;
    std::memcpy(&bits, &weak, sizeof(bits));
    put64(offset + 4, bits);
    std::memcpy(&bits, &strong, sizeof(bits));
    put64(offset + 12, bits);
    return out;
}

inline bool detector_is(const DetectorParams& p, int features, double weak, double strong) {
    return p.features == features && p.weak == weak && p.strong == strong;
}

inline std::string params_text(const DetectorParams& p) {
    return "(" + std::to_string(p.features) + "," + std::to_string(p.weak) + "," +
           std::to_string(p.strong) + ")";
}

inline void expect_load_reject(const std::vector<uint8_t>& bytes, const std::string& needle,
                               const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc =
        vf_shape_load(bytes.data(), bytes.size(), &handle, g_error, sizeof(g_error));
    check(rc == 1 && std::string(g_error).find(needle) != std::string::npos,
          what + " 预期拒绝含「" + needle + "」，实际 rc=" + std::to_string(rc) + "：" + g_error);
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
}

inline void expect_train_reject(const cv::Mat& image, const vf_shape_train_config& config,
                                const std::string& needle, const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc = vf_shape_train(image.data, image_bytes(image), image.cols, image.rows,
                                      image.step[0], &config, &handle, g_error, sizeof(g_error));
    check(rc == 1 && std::string(g_error).find(needle) != std::string::npos,
          what + " 预期拒绝含「" + needle + "」，实际 rc=" + std::to_string(rc) + "：" + g_error);
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
}

inline void check_backfill(uint64_t handle, int features, double weak, double strong,
                           const std::string& what) {
    vf_shape_train_config meta{};
    meta.struct_size = sizeof(meta);
    int32_t width = 0, height = 0;
    const int32_t rc =
        vf_shape_metadata(handle, &meta, &width, &height, g_error, sizeof(g_error));
    check(rc == 0, what + " 元数据失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(meta.feature_count == features && meta.weak_th == weak && meta.strong_th == strong,
          what + " 三字段漂移：(" + std::to_string(meta.feature_count) + "," +
              std::to_string(meta.weak_th) + "," + std::to_string(meta.strong_th) + ") 应为 (" +
              std::to_string(features) + "," + std::to_string(weak) + "," +
              std::to_string(strong) + ")");
}

// 目标放置帧：1600×1600 黑底，patch 复制到各点位（多目标召回/容量合同共用）
inline cv::Mat place_targets(const cv::Mat& reference, const cv::Point places[], size_t count) {
    cv::Mat frame(1600, 1600, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat patch = reference(cv::Rect(32, 32, 176, 176));
    for (size_t i = 0; i < count; ++i)
        patch.copyTo(frame(cv::Rect(places[i], cv::Size(176, 176))));
    return frame;
}

}  // namespace abitest
