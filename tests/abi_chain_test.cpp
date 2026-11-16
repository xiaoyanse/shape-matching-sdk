// ABI 主链合同测试：链接真实 SDK 库（visionforge_shape），覆盖训练→元数据→
// 导出→加载→单/批量匹配→释放的对外主链。合成图进程内构造，无文件依赖。
// 分组独立函数独立判定：部分组合同以在途修复为目标，当前失败按组定位待验轨道。
#include "shape_matching.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace {

char g_error[2048] = {};

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 句柄异常展开时自动释放，避免失败路径泄漏注册表槽位
struct HandleGuard {
    uint64_t handle = 0;
    ~HandleGuard() {
        if (handle) vf_shape_close(handle, nullptr, 0);
    }
};

// 合成参考图 320×320：白外框 + 内部块（与绑定侧冒烟数据同构造）
cv::Mat synth_reference() {
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

vf_shape_train_config make_config(const cv::Rect& roi, double amin, double amax, double astep,
                                  double smin, double smax, double sstep, double weak = 0,
                                  double strong = 0, int features = 0) {
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

uint64_t image_bytes(const cv::Mat& image) { return image.total() * image.elemSize(); }

uint64_t train_handle(const cv::Mat& image, const vf_shape_train_config& config,
                      const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc = vf_shape_train(image.data, image_bytes(image), image.cols, image.rows,
                                      image.step[0], &config, &handle, g_error, sizeof(g_error));
    check(rc == 0, what + " 训练失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(handle != 0, what + " 训练返回空句柄");
    return handle;
}

std::vector<uint8_t> export_bytes(uint64_t handle, const std::string& what) {
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

uint64_t load_handle(const std::vector<uint8_t>& bytes, const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc =
        vf_shape_load(bytes.data(), bytes.size(), &handle, g_error, sizeof(g_error));
    check(rc == 0, what + " 加载失败 rc=" + std::to_string(rc) + "：" + g_error);
    return handle;
}

uint32_t match_count(uint64_t handle, const cv::Mat& image, double threshold,
                     std::vector<vf_shape_candidate>& out, const std::string& what) {
    uint32_t count = 0;
    const int32_t rc =
        vf_shape_match(handle, image.data, image_bytes(image), image.cols, image.rows,
                       image.step[0], threshold, out.data(), uint32_t(out.size()), &count,
                       g_error, sizeof(g_error));
    check(rc == 0, what + " 匹配失败 rc=" + std::to_string(rc) + "：" + g_error);
    return count;
}

vf_shape_batch_item make_item(uint64_t handle, double threshold,
                              std::vector<vf_shape_candidate>& out) {
    vf_shape_batch_item item{};
    item.handle = handle;
    item.threshold = threshold;
    item.candidates = out.data();
    item.capacity = uint32_t(out.size());
    item.status = -1;
    return item;
}

int32_t run_batch(const cv::Mat& image, std::vector<vf_shape_batch_item>& items) {
    return vf_shape_match_batch(image.data, image_bytes(image), image.cols, image.rows,
                                image.step[0], items.data(), uint32_t(items.size()),
                                sizeof(vf_shape_batch_item), g_error, sizeof(g_error));
}

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
    return v;
}

// 容器首精修段（detector 参数定位与单级资产构造用）
std::vector<uint8_t> first_fine_segment(const std::vector<uint8_t>& container) {
    check(container.size() >= 13 && std::memcmp(container.data(), "VF2S3", 5) == 0,
          "容器 magic/长度不符");
    const uint64_t n = read_u64_le(container.data() + 5);
    check(n >= 1 && n <= 64 && container.size() >= 13 + 8 * (n + 3), "容器段表越界");
    const uint64_t first = read_u64_le(container.data() + 13);
    const size_t start = size_t(13 + 8 * (n + 3));
    check(start + first <= container.size(), "首段越界");
    return {container.begin() + ptrdiff_t(start),
            container.begin() + ptrdiff_t(start + first)};
}

struct DetectorParams {
    int features;
    double weak, strong;
};

// 单级段：NUL 结尾签名 + schema/宽/高(12) + 训练配置(64) 后即 detector 三值
DetectorParams segment_detector(const std::vector<uint8_t>& segment) {
    size_t zero = 0;
    while (zero < segment.size() && segment[zero] != 0) ++zero;
    check(zero < segment.size(), "段签名缺少 NUL 结尾");
    const size_t offset = zero + 1 + 12 + 64;
    check(offset + 20 <= segment.size(), "段过短，detector 参数越界");
    auto read_f64 = [&](size_t at) {
        uint64_t bits = read_u64_le(segment.data() + at);
        double v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    };
    return {int(read_u64_le(segment.data() + offset)), read_f64(offset + 4),
            read_f64(offset + 12)};
}

std::string params_text(const DetectorParams& p) {
    return "(" + std::to_string(p.features) + "," + std::to_string(p.weak) + "," +
           std::to_string(p.strong) + ")";
}

// a 组：默认参数训练→元数据→导出（两段式）→加载→匹配→关闭全链
void group_full_chain() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard trained{train_handle(ref, config, "默认参数")};
    vf_shape_train_config meta{};
    meta.struct_size = sizeof(meta);
    int32_t width = 0, height = 0;
    int32_t rc = vf_shape_metadata(trained.handle, &meta, &width, &height, g_error,
                                   sizeof(g_error));
    check(rc == 0, "训练句柄元数据失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(width > 0 && height > 0, "参考尺寸非法：" + std::to_string(width) + "x" +
                                       std::to_string(height));
    const auto bytes = export_bytes(trained.handle, "默认参数");
    HandleGuard loaded{load_handle(bytes, "默认参数容器")};
    rc = vf_shape_metadata(loaded.handle, &meta, &width, &height, g_error, sizeof(g_error));
    check(rc == 0, "加载句柄元数据失败 rc=" + std::to_string(rc) + "：" + g_error);
    std::vector<vf_shape_candidate> out(16);
    const uint32_t count = match_count(loaded.handle, ref, 60.0, out, "默认参数同图");
    check(count >= 1, "同图匹配命中 " + std::to_string(count) + " 应 ≥1");
    for (uint32_t i = 0; i < count; ++i) {
        check(out[i].score >= 60.0 && out[i].score <= 100.0,
              "候选分数越界：" + std::to_string(out[i].score));
        check(out[i].center[0] >= 0 && out[i].center[0] < ref.cols && out[i].center[1] >= 0 &&
                  out[i].center[1] < ref.rows,
              "候选中心越出图像：" + std::to_string(out[i].center[0]) + "," +
                  std::to_string(out[i].center[1]));
    }
}

// b 组：容器往返元数据不变量（角度/缩放逐字段相等）
void group_metadata_roundtrip() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard trained{train_handle(ref, config, "往返")};
    const auto bytes = export_bytes(trained.handle, "往返");
    HandleGuard loaded{load_handle(bytes, "往返容器")};
    vf_shape_train_config meta{};
    meta.struct_size = sizeof(meta);
    int32_t width = 0, height = 0;
    const int32_t rc =
        vf_shape_metadata(loaded.handle, &meta, &width, &height, g_error, sizeof(g_error));
    check(rc == 0, "往返元数据失败 rc=" + std::to_string(rc) + "：" + g_error);
    const std::pair<const char*, double> expected[] = {
        {"angle_min", config.angle_min}, {"angle_max", config.angle_max},
        {"angle_step", config.angle_step}, {"scale_min", config.scale_min},
        {"scale_max", config.scale_max}, {"scale_step", config.scale_step}};
    const double actual[] = {meta.angle_min, meta.angle_max, meta.angle_step,
                             meta.scale_min, meta.scale_max, meta.scale_step};
    for (int i = 0; i < 6; ++i)
        check(actual[i] == expected[i].second,
              std::string("往返后 ") + expected[i].first + " 漂移：实际 " +
                  std::to_string(actual[i]) + " 期望 " + std::to_string(expected[i].second));
}

bool detector_is(const DetectorParams& p, int features, double weak, double strong) {
    return p.features == features && p.weak == weak && p.strong == strong;
}

std::string expected_text(int features, double weak, double strong) {
    return "(" + std::to_string(features) + "," + std::to_string(weak) + "," +
           std::to_string(strong) + ")";
}

void check_backfill(uint64_t handle, const std::string& what) {
    vf_shape_train_config meta{};
    meta.struct_size = sizeof(meta);
    int32_t width = 0, height = 0;
    const int32_t rc =
        vf_shape_metadata(handle, &meta, &width, &height, g_error, sizeof(g_error));
    check(rc == 0, what + " 元数据失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(meta.feature_count == 32 && meta.weak_th == 20 && meta.strong_th == 40,
          what + " 三字段未回填实际值：(" + std::to_string(meta.feature_count) + "," +
              std::to_string(meta.weak_th) + "," + std::to_string(meta.strong_th) +
              ") 应为 (32,20,40)");
}

// c 组：自定义参资产 load→export→load 三参逐次相等，metadata 回填实际值
void group_detector_roundtrip() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "自定义参")};
    const auto container = export_bytes(trained.handle, "自定义参");
    const DetectorParams original = segment_detector(first_fine_segment(container));
    check(detector_is(original, 32, 20, 40), "首次导出 detector 参数漂移：" +
                                                 params_text(original) + " 应为 (32,20,40)");
    HandleGuard single1{load_handle(first_fine_segment(container), "首段单级")};
    check_backfill(single1.handle, "单级一次加载");
    const auto reexport1 = export_bytes(single1.handle, "单级一次重导出");
    const DetectorParams second = segment_detector(reexport1);
    check(detector_is(second, 32, 20, 40), "一次往返导出参数漂移：" + params_text(second) +
                                               " 应为 (32,20,40)");
    HandleGuard single2{load_handle(reexport1, "二段单级")};
    check_backfill(single2.handle, "单级二次加载");
    const DetectorParams third = segment_detector(export_bytes(single2.handle, "单级二次重导出"));
    check(detector_is(third, 32, 20, 40), "二次往返导出参数漂移：" + params_text(third) +
                                              " 应为 (32,20,40)");
    HandleGuard reloaded{load_handle(container, "容器重载")};
    check_backfill(reloaded.handle, "两级容器加载");
    const DetectorParams recontainer =
        segment_detector(first_fine_segment(export_bytes(reloaded.handle, "容器重导出")));
    check(detector_is(recontainer, 32, 20, 40), "容器重导出首段参数漂移：" +
                                                    params_text(recontainer) +
                                                    " 应为 (32,20,40)");
}

// d 组：阈值合同——上限内 (30,300) 成功往返；超限/倒挂/非有限训练即拒
void group_threshold_contract() {
    const cv::Mat ref = synth_reference();
    const auto bound = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 30, 300, 32);
    HandleGuard trained{train_handle(ref, bound, "strong=300")};
    const auto segment = first_fine_segment(export_bytes(trained.handle, "strong=300"));
    const DetectorParams fields = segment_detector(segment);
    check(detector_is(fields, 32, 30, 300), "strong=300 序列化字段漂移：" + params_text(fields) +
                                                " 应为 (32,30,300)");
    HandleGuard loaded{load_handle(segment, "strong=300 单级段")};
    struct Rejection {
        vf_shape_train_config config;
        const char* text;
    };
    const Rejection rejections[] = {
        {make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 30, 2000, 32),
         "reachable gradient magnitude"},
        {make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 100, 0, 32),
         "weak threshold must not exceed strong threshold"},
        {make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, INFINITY, 60, 32), "finite"},
        {make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 30, NAN, 32), "finite"},
    };
    for (const auto& rejection : rejections) {
        uint64_t handle = 0;
        const int32_t rc =
            vf_shape_train(ref.data, image_bytes(ref), ref.cols, ref.rows, ref.step[0],
                           &rejection.config, &handle, g_error, sizeof(g_error));
        check(rc == 1 && std::string(g_error).find(rejection.text) != std::string::npos,
              std::string("预期拒绝含「") + rejection.text + "」，实际 rc=" +
                  std::to_string(rc) + " 错误：" + g_error);
    }
    const auto healthy = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard ok{train_handle(ref, healthy, "合法参数对照")};
}

// e 组：整图 ROI 训练成功
void group_full_image_roi() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({0, 0, 320, 320}, 0, 0, 1, 1, 1, 1, 30, 60, 32);
    HandleGuard trained{train_handle(ref, config, "整图 ROI")};
}

// f 组：1600×1600 三目标召回 ≥3（容量 16）
void group_multi_target_recall() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "多目标模板")};
    cv::Mat frame(1600, 1600, CV_8UC3, cv::Scalar(0, 0, 0));
    const cv::Mat patch = ref(cv::Rect(32, 32, 176, 176));
    const cv::Point places[] = {{100, 100}, {800, 100}, {1350, 1350}};
    for (const auto& place : places)
        patch.copyTo(frame(cv::Rect(place, cv::Size(176, 176))));
    std::vector<vf_shape_candidate> out(16);
    const uint32_t count = match_count(trained.handle, frame, 60.0, out, "三目标帧");
    check(count >= 3, "三目标召回 " + std::to_string(count) + " 应 ≥3（固定窗口预算静默遗漏）");
    for (int i = 0; i < 3; ++i) {
        bool covered = false;
        for (uint32_t j = 0; j < count && !covered; ++j) {
            const double dx = out[j].center[0] - (places[i].x + 87.5);
            const double dy = out[j].center[1] - (places[i].y + 87.5);
            covered = std::hypot(dx, dy) < 40.0;
        }
        check(covered, "目标 " + std::to_string(i) + "（" +
                           std::to_string(places[i].x + 87.5) + "," +
                           std::to_string(places[i].y + 87.5) + "）近旁 40px 无候选");
    }
}

// g 组：批量——默认两级/默认参单级/自定义参单级 + 错误项混合 status 语义
void group_batch_mix() {
    const cv::Mat ref = synth_reference();
    const auto ranged = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard two_level{train_handle(ref, ranged, "批量两级")};
    const auto container = export_bytes(two_level.handle, "批量两级");
    HandleGuard single_default{load_handle(first_fine_segment(container), "默认参单级")};
    const auto custom = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard custom_trained{train_handle(ref, custom, "批量自定义参")};
    const auto custom_container = export_bytes(custom_trained.handle, "批量自定义参");
    HandleGuard single_custom{load_handle(first_fine_segment(custom_container), "自定义参单级")};

    std::vector<vf_shape_candidate> out_two(16), out_default(16), out_custom(16), out_bad(4);
    std::vector<vf_shape_batch_item> items{make_item(two_level.handle, 60.0, out_two),
                                           make_item(single_default.handle, 60.0, out_default),
                                           make_item(single_custom.handle, 60.0, out_custom)};
    const int32_t rc = run_batch(ref, items);
    check(rc == 0, "批量编排失败 rc=" + std::to_string(rc) + "：" + g_error);
    for (size_t i = 0; i < items.size(); ++i) {
        check(items[i].status == 0,
              "批量项 " + std::to_string(i) + " status=" + std::to_string(items[i].status) +
                  "：" + items[i].error);
        check(items[i].count >= 1,
              "批量项 " + std::to_string(i) + " 命中 " + std::to_string(items[i].count) +
                  " 应 ≥1");
    }
    std::vector<vf_shape_batch_item> mixed{make_item(0xDEADBEEF, 60.0, out_bad),
                                           make_item(single_default.handle, 60.0, out_default)};
    const int32_t mixed_rc = run_batch(ref, mixed);
    check(mixed_rc == 0, "混合批次编排失败 rc=" + std::to_string(mixed_rc) + "：" + g_error);
    check(mixed[0].status != 0 && mixed[0].count == 0,
          "未知句柄项应独立失败且无候选：status=" + std::to_string(mixed[0].status) +
              " count=" + std::to_string(mixed[0].count));
    check(mixed[1].status == 0 && mixed[1].count >= 1,
          "错误项不应影响成功项：status=" + std::to_string(mixed[1].status) + " count=" +
              std::to_string(mixed[1].count));
    std::vector<vf_shape_candidate> out_tiny(1);
    std::vector<vf_shape_batch_item> isolated{make_item(single_default.handle, 60.0, out_tiny),
                                              make_item(single_default.handle, 60.0, out_default)};
    isolated[0].capacity = 1;
    const int32_t isolated_rc = run_batch(ref, isolated);
    check(isolated_rc == 0, "容量越界隔离批次失败 rc=" + std::to_string(isolated_rc) + "：" +
                               g_error);
    check(isolated[0].status != 0 && isolated[0].count == 0,
          "capacity=1 项应独立失败：status=" + std::to_string(isolated[0].status));
    check(isolated[1].status == 0 && isolated[1].count >= 1,
          "容量越界项不应影响成功项：status=" + std::to_string(isolated[1].status));
}

// h 组：生命周期——重复/未知句柄 close 幂等，非法输入返回码 1
void group_lifecycle() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    const uint64_t trained = train_handle(ref, config, "生命周期");
    int32_t rc = vf_shape_close(trained, g_error, sizeof(g_error));
    check(rc == 0, "首次关闭失败 rc=" + std::to_string(rc) + "：" + g_error);
    rc = vf_shape_close(trained, g_error, sizeof(g_error));
    check(rc == 0, "重复关闭应幂等成功，rc=" + std::to_string(rc) + "：" + g_error);
    rc = vf_shape_close(0xDEADBEEF, g_error, sizeof(g_error));
    check(rc == 0, "未知句柄关闭应幂等成功，rc=" + std::to_string(rc) + "：" + g_error);
    HandleGuard fresh{train_handle(ref, config, "生命周期二次")};
    std::vector<vf_shape_candidate> out(4);
    uint32_t count = 0;
    rc = vf_shape_match(fresh.handle, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], std::nan(""), out.data(), uint32_t(out.size()), &count,
                        g_error, sizeof(g_error));
    check(rc == 1, "NaN 阈值应返回码 1，实际 rc=" + std::to_string(rc));
    rc = vf_shape_match(fresh.handle, ref.data, image_bytes(ref), 0, ref.rows, ref.step[0],
                        60.0, out.data(), uint32_t(out.size()), &count, g_error,
                        sizeof(g_error));
    check(rc == 1, "零宽图应返回码 1，实际 rc=" + std::to_string(rc));
    rc = vf_shape_match(fresh.handle, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], 60.0, out.data(), uint32_t(out.size()), &count, g_error,
                        sizeof(g_error));
    check(rc == 0 && count >= 1, "合法匹配应成功：rc=" + std::to_string(rc) + " count=" +
                                     std::to_string(count) + " " + g_error);
}

struct Group {
    const char* name;
    void (*fn)();
};

}  // namespace

int main() {
    const Group groups[] = {
        {"a_full_chain", group_full_chain},
        {"b_metadata_roundtrip", group_metadata_roundtrip},
        {"c_detector_roundtrip", group_detector_roundtrip},
        {"d_threshold_contract", group_threshold_contract},
        {"e_full_image_roi", group_full_image_roi},
        {"f_multi_target_recall", group_multi_target_recall},
        {"g_batch_mix", group_batch_mix},
        {"h_lifecycle", group_lifecycle},
    };
    int failures = 0;
    for (const auto& group : groups) {
        try {
            group.fn();
            std::cout << "[PASS] " << group.name << "\n";
        } catch (const std::exception& error) {
            ++failures;
            std::cout << "[FAIL] " << group.name << ": " << error.what() << "\n";
        }
    }
    const int total = int(sizeof(groups) / sizeof(groups[0]));
    std::cout << "abi chain: " << (total - failures) << "/" << total << " groups passed\n";
    return failures ? 1 : 0;
}
