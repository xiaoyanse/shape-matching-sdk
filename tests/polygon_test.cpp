// 多边形模板能力 ABI 合同测试：训练/自匹配、洞掩蔽行为、反例族（码+细分参数）、
// 容器 poly 往返与删键回退、耗时承诺、suggest_contours 合同。验收基准一律程序
// 合成图（常设纪律）。
#include "abi_chain_support.h"
#include "shape_matching.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace abitest;

namespace {

struct ErrorInfo {
    int32_t code = 0;
    std::vector<double> params;
};

ErrorInfo last_error() {
    int32_t code = 0, count = 0;
    double params[6] = {};
    vf_shape_last_error_info(&code, params, &count);
    return {code, std::vector<double>(params, params + std::max(0, count))};
}

// 多边形训练用例构建器：finalize 后指针视图才有效（勿在 add_hole 后引用旧视图）
struct PolygonCase {
    std::vector<vf_shape_vertex> outline;
    std::vector<std::vector<vf_shape_vertex>> holes;
    std::vector<vf_shape_polygon> hole_views;
    vf_shape_polygon_train_params params{};

    void finalize(double amin = -30, double amax = 30, double astep = 15, double smin = 1,
                  double smax = 1, double sstep = 1, double weak = 0, double strong = 0,
                  int features = 0) {
        params = vf_shape_polygon_train_params{};
        params.struct_size = sizeof(params);
        params.outline = {outline.data(), uint32_t(outline.size())};
        hole_views.clear();
        for (auto& hole : holes) hole_views.push_back({hole.data(), uint32_t(hole.size())});
        params.holes = hole_views.data();
        params.hole_count = uint32_t(holes.size());
        params.angle_min = amin;
        params.angle_max = amax;
        params.angle_step = astep;
        params.scale_min = smin;
        params.scale_max = smax;
        params.scale_step = sstep;
        params.weak_th = weak;
        params.strong_th = strong;
        params.feature_count = features;
    }
};

vf_shape_vertex vertex(double x, double y) { return {x, y}; }

uint64_t train_polygon(const cv::Mat& image, const vf_shape_polygon_train_params& params,
                       const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc =
        vf_shape_train_polygon(image.data, image_bytes(image), image.cols, image.rows,
                               image.step[0], &params, &handle, g_error, sizeof(g_error));
    check(rc == 0, what + " 多边形训练失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(handle != 0, what + " 训练返回空句柄");
    return handle;
}

void expect_polygon_reject(const cv::Mat& image, const vf_shape_polygon_train_params& params,
                           int32_t subtype, double primary, const std::string& what) {
    uint64_t handle = 0;
    const int32_t rc =
        vf_shape_train_polygon(image.data, image_bytes(image), image.cols, image.rows,
                               image.step[0], &params, &handle, g_error, sizeof(g_error));
    check(rc == 1, what + " 预期拒绝 rc=1，实际 " + std::to_string(rc) + "：" + g_error);
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
    const ErrorInfo info = last_error();
    check(info.code == int32_t(VF_SHAPE_INVALID_POLYGON),
          what + " 错误码应为 INVALID_POLYGON(17)，实际 " + std::to_string(info.code));
    check(!info.params.empty() && int32_t(info.params[0]) == subtype,
          what + " 细分类型应为 " + std::to_string(subtype) + "，实际 " +
              (info.params.empty() ? "(空)" : std::to_string(int32_t(info.params[0]))));
    if (primary != -12345.0)
        check(info.params.size() > 1 && info.params[1] == primary,
              what + " 主参数应为 " + std::to_string(primary) + "，实际 " +
                  (info.params.size() > 1 ? std::to_string(info.params[1]) : "(缺)"));
}

// 与 polygon.cpp 同式的有向面积（测试侧独立实现，断言绕向归一方向）
double signed_area_test(const std::vector<vf_shape_vertex>& polygon) {
    double sum = 0;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const vf_shape_vertex& a = polygon[i];
        const vf_shape_vertex& b = polygon[(i + 1) % polygon.size()];
        sum += a.x * b.y - b.x * a.y;
    }
    return sum / 2.0;
}

std::vector<vf_shape_vertex> normalized_expectation(std::vector<vf_shape_vertex> polygon) {
    if (signed_area_test(polygon) < 0)
        std::reverse(polygon.begin(), polygon.end());
    return polygon;
}

struct PolygonMeta {
    std::vector<vf_shape_vertex> outline;
    uint32_t hole_count = 0;
};

PolygonMeta read_polygon_metadata(uint64_t handle, const std::string& what) {
    uint32_t count = 0, holes = 0;
    int32_t rc = vf_shape_polygon_metadata(handle, nullptr, 0, &count, &holes, g_error,
                                           sizeof(g_error));
    check(rc == 0, what + " 元数据查询失败 rc=" + std::to_string(rc) + "：" + g_error);
    std::vector<vf_shape_vertex> outline(count);
    rc = vf_shape_polygon_metadata(handle, outline.data(), count, &count, &holes, g_error,
                                   sizeof(g_error));
    check(rc == 0, what + " 元数据取回失败 rc=" + std::to_string(rc) + "：" + g_error);
    return {outline, holes};
}

void check_metadata_equal(const PolygonMeta& got, const std::vector<vf_shape_vertex>& want,
                          uint32_t hole_count, const std::string& what) {
    check(got.outline.size() == want.size(),
          what + " 顶点数漂移：实际 " + std::to_string(got.outline.size()) + " 应为 " +
              std::to_string(want.size()));
    check(std::memcmp(got.outline.data(), want.data(),
                      want.size() * sizeof(vf_shape_vertex)) == 0,
          what + " 顶点字节不等（坐标或绕向漂移）");
    check(got.hole_count == hole_count,
          what + " 洞数漂移：实际 " + std::to_string(got.hole_count) + " 应为 " +
              std::to_string(hole_count));
}

// 缩放合成参考图上的黑底白形（多边形轮廓由调用方按同坐标给出）
cv::Mat polygon_reference(int size, const std::vector<std::vector<cv::Point>>& shapes) {
    cv::Mat image(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    for (const auto& shape : shapes) {
        std::vector<std::vector<cv::Point>> poly{shape};
        cv::fillPoly(image, poly, cv::Scalar(255, 255, 255));
    }
    return image;
}

void group_polygon_training() {
    // 三角形：轮廓大于绘制形状（黑边距防掩码腐蚀吃掉贴边边界特征）
    cv::Mat image(480, 480, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Point> tri{{140, 120}, {360, 200}, {180, 360}};
    cv::fillPoly(image, std::vector<std::vector<cv::Point>>{tri}, cv::Scalar(255, 255, 255));
    cv::rectangle(image, {210, 190}, {250, 230}, cv::Scalar(120, 120, 120), -1);
    PolygonCase tri_case;
    tri_case.outline = {vertex(110, 90), vertex(400, 210), vertex(150, 400)};
    tri_case.finalize();
    {
        HandleGuard handle;
        handle.handle = train_polygon(image, tri_case.params, "三角形");
        std::vector<vf_shape_candidate> out(16);
        const uint32_t count = match_count(handle.handle, image, 70.0, out, "三角形自匹配");
        check(count >= 1, "三角形自匹配应命中");
        double best = 0;
        size_t best_index = 0;
        for (uint32_t i = 0; i < count; ++i)
            if (out[i].score > best) {
                best = out[i].score;
                best_index = i;
            }
        check(best >= 70.0, "三角形自匹配分应 ≥70，实际 " + std::to_string(best));
        // 命中中心=外接矩形四角映射均值，即外接矩形中心
        const double cx = (110 + 400) / 2.0, cy = (90 + 400) / 2.0;
        check(std::abs(out[best_index].center[0] - cx) <= 8 &&
                  std::abs(out[best_index].center[1] - cy) <= 8,
              "三角形命中中心应近外接矩形中心，实际 (" +
                  std::to_string(out[best_index].center[0]) + "," +
                  std::to_string(out[best_index].center[1]) + ") 期望约 (" +
                  std::to_string(cx) + "," + std::to_string(cy) + ")");
    }

    // L 形（绘制形状留边距；反转输入绕向验证 SDK 归一承诺）
    PolygonCase l_case;
    l_case.outline = {vertex(100, 100), vertex(300, 100), vertex(300, 180),
                      vertex(180, 180), vertex(180, 300), vertex(100, 300)};
    // 反转制造非归一绕向输入，验证 SDK 归一承诺
    std::reverse(l_case.outline.begin(), l_case.outline.end());
    l_case.finalize();
    {
        HandleGuard handle;
        handle.handle = train_polygon(polygon_reference(480, {{{120, 120}, {280, 120},
                                                               {280, 180}, {180, 180},
                                                               {180, 280}, {120, 280}}}),
                                      l_case.params, "L形");
        const PolygonMeta meta = read_polygon_metadata(handle.handle, "L形往返");
        check_metadata_equal(meta, normalized_expectation(
                                       {vertex(100, 100), vertex(300, 100), vertex(300, 180),
                                        vertex(180, 180), vertex(180, 300), vertex(100, 300)}),
                            0, "L形绕向归一");
    }

    // 带洞方形：绘制形状留边距；自匹配高分 + 元数据洞数回读
    PolygonCase holed_case;
    holed_case.outline = {vertex(90, 90), vertex(230, 90), vertex(230, 230), vertex(90, 230)};
    holed_case.holes.push_back({vertex(140, 140), vertex(180, 140), vertex(180, 180),
                                vertex(140, 180)});
    holed_case.finalize();
    cv::Mat holed_image(480, 480, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(holed_image, {100, 100}, {219, 219}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(holed_image, {150, 150}, {169, 169}, cv::Scalar(90, 90, 90), -1);
    {
        HandleGuard handle;
        handle.handle = train_polygon(holed_image, holed_case.params, "带洞方形");
        std::vector<vf_shape_candidate> out(16);
        const uint32_t count = match_count(handle.handle, holed_image, 70.0, out, "带洞自匹配");
        check(count >= 1, "带洞方形自匹配应命中");
        double best = 0;
        for (uint32_t i = 0; i < count; ++i) best = std::max(best, out[i].score);
        check(best >= 70.0, "带洞方形自匹配分应 ≥70，实际 " + std::to_string(best));
        const PolygonMeta meta = read_polygon_metadata(handle.handle, "带洞往返");
        check_metadata_equal(meta, holed_case.outline, 1, "带洞往返");
    }
}

void group_invalid_polygons() {
    const cv::Mat image = polygon_reference(480, {{{100, 100}, {300, 100}, {300, 300},
                                                   {100, 300}}});
    // 细长小三角形：外接矩形 <24px
    {
        PolygonCase c;
        c.outline = {vertex(20, 20), vertex(42, 20), vertex(20, 42)};
        c.finalize();
        expect_polygon_reject(image, c.params, 9, 22, "外接矩形<24px");
    }
    // 自交（非零面积的交叉四边形：边 0-1 与 2-3 正交于 (200,150)）
    {
        PolygonCase c;
        c.outline = {vertex(100, 150), vertex(300, 150), vertex(200, 100), vertex(200, 300)};
        c.finalize();
        expect_polygon_reject(image, c.params, 4, 0, "外环自交");
    }
    // 顶点越坐标域
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(500, 100), vertex(300, 300)};
        c.finalize();
        expect_polygon_reject(image, c.params, 2, 1, "顶点越界(下标1)");
    }
    // 顶点数 2 / 65
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(300, 100)};
        c.finalize();
        expect_polygon_reject(image, c.params, 1, 2, "顶点数2");
    }
    {
        PolygonCase c;
        for (int i = 0; i < 65; ++i) {
            const double angle = 2.0 * 3.14159265358979323846 * i / 65.0;
            c.outline.push_back(vertex(200 + 100 * std::cos(angle),
                                       200 + 100 * std::sin(angle)));
        }
        c.finalize();
        expect_polygon_reject(image, c.params, 1, 65, "顶点数65");
    }
    // 洞越界（在外环外）
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(300, 100), vertex(300, 300), vertex(100, 300)};
        c.holes.push_back({vertex(400, 400), vertex(430, 400), vertex(400, 430)});
        c.finalize();
        expect_polygon_reject(image, c.params, 8, 0, "洞在外环外");
    }
    // 洞面积不足 64px²
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(300, 100), vertex(300, 300), vertex(100, 300)};
        c.holes.push_back({vertex(150, 150), vertex(168, 150), vertex(150, 157)});
        c.finalize();
        expect_polygon_reject(image, c.params, 7, 0, "洞面积63px²");
    }
    // 洞顶点数 2
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(300, 100), vertex(300, 300), vertex(100, 300)};
        c.holes.push_back({vertex(150, 150), vertex(200, 150)});
        c.finalize();
        expect_polygon_reject(image, c.params, 6, 0, "洞顶点数2");
    }
    // 洞数 9
    {
        PolygonCase c;
        c.outline = {vertex(60, 60), vertex(420, 60), vertex(420, 420), vertex(60, 420)};
        for (int i = 0; i < 9; ++i) {
            const double x = 80.0 + (i % 3) * 40.0, y = 80.0 + (i / 3) * 40.0;
            c.holes.push_back({vertex(x, y), vertex(x + 10, y), vertex(x, y + 10)});
        }
        c.finalize();
        expect_polygon_reject(image, c.params, 5, 9, "洞数9");
    }
    // struct_size 错
    {
        PolygonCase c;
        c.outline = {vertex(100, 100), vertex(300, 100), vertex(300, 300)};
        c.finalize();
        vf_shape_polygon_train_params bad = c.params;
        bad.struct_size = 4;
        uint64_t handle = 0;
        const int32_t rc = vf_shape_train_polygon(image.data, image_bytes(image), image.cols,
                                                  image.rows, image.step[0], &bad, &handle,
                                                  g_error, sizeof(g_error));
        check(rc == 1, "struct_size 错应拒绝");
        check(std::string(g_error).find("structure size") != std::string::npos,
              "struct_size 错误文案异常：" + std::string(g_error));
        if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
    }
}

std::vector<uint8_t> strip_poly_key(const std::vector<uint8_t>& container) {
    std::vector<uint8_t> meta = segment_view(container, container_segments(container) - 1);
    std::string text(meta.begin(), meta.end());
    const size_t at = text.find("\"poly\":");
    check(at != std::string::npos && at > 0, "meta 段缺少 poly 键");
    check(text.back() == '}', "meta 段结尾异常");
    text = text.substr(0, at - 1) + "}";
    return replace_segment(container, container_segments(container) - 1,
                           std::vector<uint8_t>(text.begin(), text.end()));
}

void group_container_contract() {
    const cv::Mat image = polygon_reference(480, {{{100, 100}, {300, 100}, {300, 300},
                                                   {100, 300}}});
    PolygonCase c;
    c.outline = {vertex(90, 310), vertex(310, 310), vertex(310, 90), vertex(90, 90)};
    c.holes.push_back({vertex(180, 260), vertex(220, 260), vertex(220, 220), vertex(180, 220)});
    c.finalize();
    std::vector<uint8_t> container;
    {
        HandleGuard handle;
        handle.handle = train_polygon(image, c.params, "往返用例");
        container = export_bytes(handle.handle, "往返用例");
        // 逐字段往返：归一化绕向后逐字节相等 + 洞数
        const PolygonMeta meta = read_polygon_metadata(handle.handle, "容器往返");
        check_metadata_equal(meta, normalized_expectation(c.outline), 1, "容器往返");
    }
    // 删 poly 键（模拟旧方言容器）：加载受容，回读回退外接矩形
    {
        const std::vector<uint8_t> stripped = strip_poly_key(container);
        HandleGuard handle;
        handle.handle = load_handle(stripped, "删poly键容器");
        const PolygonMeta meta = read_polygon_metadata(handle.handle, "删poly回退");
        check_metadata_equal(meta,
                             {vertex(90, 90), vertex(310, 90), vertex(310, 310),
                              vertex(90, 310)},
                             0, "删poly外接矩形回退");
    }
    // 矩形资产回退：vf_shape_train 产物回读 = ROI 四角 + 洞数 0
    {
        const cv::Mat reference = abitest::synth_reference();
        HandleGuard handle;
        handle.handle = train_handle(reference,
                                     make_config(cv::Rect(42, 42, 156, 156), -30, 30, 5, 1, 1,
                                                 1),
                                     "矩形资产");
        const PolygonMeta meta = read_polygon_metadata(handle.handle, "矩形回退");
        check_metadata_equal(meta,
                             {vertex(42, 42), vertex(198, 42), vertex(198, 198),
                              vertex(42, 198)},
                             0, "矩形ROI四角回退");
        // 缓冲不足拒绝
        vf_shape_vertex tiny[3] = {};
        uint32_t count = 0, holes = 0;
        const int32_t rc = vf_shape_polygon_metadata(handle.handle, tiny, 3, &count, &holes,
                                                     g_error, sizeof(g_error));
        check(rc == 1, "元数据缓冲不足应拒绝");
    }
}

void group_timing_commitment() {
    // 1080P + 35 位姿，外接矩形 ≤400px：训练 ≤2s（超限即报，不许静默）；
    // 绘制三角形小于轮廓（黑边距防掩码腐蚀吃特征）
    cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<cv::Point> tri{{200, 150}, {520, 150}, {200, 390}};
    cv::fillPoly(image, std::vector<std::vector<cv::Point>>{tri}, cv::Scalar(255, 255, 255));
    cv::rectangle(image, {260, 200}, {340, 280}, cv::Scalar(110, 110, 110), -1);
    cv::rectangle(image, {300, 240}, {380, 320}, cv::Scalar(200, 200, 200), -1);
    PolygonCase c;
    c.outline = {vertex(180, 130), vertex(540, 130), vertex(180, 410)};
    c.finalize(-12, 12, 4, 1.0, 1.6, 0.15);
    const auto started = std::chrono::steady_clock::now();
    HandleGuard handle;
    handle.handle = train_polygon(image, c.params, "耗时背书");
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    std::cout << "polygon timing: bbox=320x240 poses=35 elapsed=" << elapsed << "ms\n";
    check(elapsed <= 2000.0, "多边形训练耗时承诺超限：" + std::to_string(elapsed) + "ms > 2000ms");
}

void group_suggest_contours() {
    cv::Mat image(400, 400, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::fillPoly(image,
                 std::vector<std::vector<cv::Point>>{
                     {{50, 50}, {200, 50}, {50, 200}}},
                 cv::Scalar(255, 255, 255));
    cv::fillPoly(image,
                 std::vector<std::vector<cv::Point>>{
                     {{250, 250}, {380, 250}, {380, 320}, {320, 320}, {320, 380}, {250, 380}}},
                 cv::Scalar(255, 255, 255));
    constexpr uint32_t kMaxSuggestions = 8;
    constexpr uint32_t kMaxVertices = 8;
    std::vector<vf_shape_vertex> vertices(size_t(kMaxSuggestions) * kMaxVertices);
    std::vector<uint32_t> counts(kMaxSuggestions);
    auto run_suggest = [&](const int32_t* hint) {
        return vf_shape_suggest_contours(image.data, image_bytes(image), image.cols, image.rows,
                                         image.step[0], hint, kMaxSuggestions, kMaxVertices,
                                         vertices.data(), counts.data(), g_error,
                                         sizeof(g_error));
    };
    const int32_t rc = run_suggest(nullptr);
    check(rc >= 2, "建议条数应 ≥2（三角+L），实际 " + std::to_string(rc) + "：" + g_error);
    check(rc <= int32_t(kMaxSuggestions), "建议条数越容量");
    uint32_t written = 0;
    for (int32_t i = 0; i < rc; ++i) {
        check(counts[size_t(i)] >= 3 && counts[size_t(i)] <= kMaxVertices,
              "建议顶点数越 3..8：第 " + std::to_string(i) + " 条 " +
                  std::to_string(counts[size_t(i)]));
        written += counts[size_t(i)];
    }
    // 面积降序：三角（约 11250）先于 L（约 13100？L 面积 130*70+60*60=12700）
    // —— 只锁"面积降序"关系本身，不锁具体面积值
    auto area_of = [&](size_t index) {
        double sum = 0;
        uint32_t offset = 0;
        for (size_t k = 0; k < size_t(index); ++k) offset += counts[k];
        for (uint32_t i = 0; i < counts[index]; ++i) {
            const vf_shape_vertex& a = vertices[offset + i];
            const vf_shape_vertex& b = vertices[offset + (i + 1) % counts[index]];
            sum += a.x * b.y - b.x * a.y;
        }
        return std::abs(sum / 2.0);
    };
    for (int32_t i = 1; i < rc; ++i)
        check(area_of(size_t(i - 1)) >= area_of(size_t(i)), "建议未按面积降序");
    // 确定性：同图两次调用逐字节一致
    std::vector<vf_shape_vertex> vertices_again = vertices;
    std::vector<uint32_t> counts_again = counts;
    const int32_t rc_again =
        vf_shape_suggest_contours(image.data, image_bytes(image), image.cols, image.rows,
                                  image.step[0], nullptr, kMaxSuggestions, kMaxVertices,
                                  vertices_again.data(), counts_again.data(), g_error,
                                  sizeof(g_error));
    check(rc_again == rc, "两次调用建议条数不一致");
    check(std::memcmp(vertices.data(), vertices_again.data(),
                      written * sizeof(vf_shape_vertex)) == 0 &&
              std::memcmp(counts.data(), counts_again.data(), rc * sizeof(uint32_t)) == 0,
          "两次调用输出字节不等（确定性破坏）");
    // ROI 先验框：圈三角后建议顶点均落框内，L 不入结果
    const int32_t hint[4] = {25, 25, 225, 225};
    const int32_t rc_hint = run_suggest(hint);
    check(rc_hint >= 1, "先验框内应有建议");
    uint32_t checked = 0;
    for (int32_t i = 0; i < rc_hint; ++i)
        for (uint32_t j = 0; j < counts[size_t(i)]; ++j) {
            const vf_shape_vertex& v = vertices[checked++];
            check(v.x >= 25 && v.x <= 250 && v.y >= 25 && v.y <= 250,
                  "先验框建议顶点越框：(" + std::to_string(v.x) + "," + std::to_string(v.y) +
                      ") 建议 " + std::to_string(i));
        }
    // 非法参数与空图
    uint32_t bad_counts[1] = {};
    check(vf_shape_suggest_contours(image.data, image_bytes(image), image.cols, image.rows,
                                    image.step[0], nullptr, 0, 8, vertices.data(), bad_counts,
                                    g_error, sizeof(g_error)) == -1,
          "max_suggestions=0 应 -1");
    check(vf_shape_suggest_contours(image.data, image_bytes(image), image.cols, image.rows,
                                    image.step[0], nullptr, 4, 2, vertices.data(), bad_counts,
                                    g_error, sizeof(g_error)) == -1,
          "max_vertices_per=2 应 -1");
    const int32_t bad_hint[4] = {300, 300, 200, 200};
    check(vf_shape_suggest_contours(image.data, image_bytes(image), image.cols, image.rows,
                                    image.step[0], bad_hint, 4, 8, vertices.data(),
                                    bad_counts, g_error, sizeof(g_error)) == -1,
          "越界先验框应 -1");
    cv::Mat blank(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    check(vf_shape_suggest_contours(blank.data, image_bytes(blank), blank.cols, blank.rows,
                                    blank.step[0], nullptr, 4, 8, vertices.data(), bad_counts,
                                    g_error, sizeof(g_error)) == 0,
          "空图应 0 建议（非错误）");
}

}  // namespace

int main() {
    try {
        group_polygon_training();
        group_invalid_polygons();
        group_container_contract();
        group_timing_commitment();
        group_suggest_contours();
        std::cout << "polygon: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
