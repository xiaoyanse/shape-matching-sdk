// ABI 主链合同测试：链接真实 SDK 库（visionforge_shape），覆盖训练→元数据→
// 导出→加载→单/批量匹配→释放的对外主链。合成图进程内构造，无文件依赖。
// 分组独立函数独立判定：失败按组定位，共享支撑见 abi_chain_support.h。
#include "abi_chain_support.h"

#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace abitest;

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
    check(width == ref.cols && height == ref.rows,
          "往返后参考尺寸应为原图：" + std::to_string(width) + "x" + std::to_string(height) +
              " 期望 " + std::to_string(ref.cols) + "x" + std::to_string(ref.rows));
}

// c 组：自定义参资产 load→export→load 三参逐次相等，metadata 回填实际值
void group_detector_roundtrip() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "自定义参")};
    const auto container = export_bytes(trained.handle, "自定义参");
    const DetectorParams original = segment_detector(first_fine_segment(container));
    check(detector_is(original, 32, 20, 40),
          "首次导出 detector 参数漂移：" + params_text(original) + " 应为 (32,20,40)");
    HandleGuard single1{load_handle(first_fine_segment(container), "首段单级")};
    check_backfill(single1.handle, 32, 20, 40, "单级一次加载");
    const auto reexport1 = export_bytes(single1.handle, "单级一次重导出");
    const DetectorParams second = segment_detector(reexport1);
    check(detector_is(second, 32, 20, 40),
          "一次往返导出参数漂移：" + params_text(second) + " 应为 (32,20,40)");
    HandleGuard single2{load_handle(reexport1, "二段单级")};
    check_backfill(single2.handle, 32, 20, 40, "单级二次加载");
    const DetectorParams third = segment_detector(export_bytes(single2.handle, "单级二次重导出"));
    check(detector_is(third, 32, 20, 40),
          "二次往返导出参数漂移：" + params_text(third) + " 应为 (32,20,40)");
    HandleGuard reloaded{load_handle(container, "容器重载")};
    check_backfill(reloaded.handle, 32, 20, 40, "两级容器加载");
    const DetectorParams recontainer =
        segment_detector(first_fine_segment(export_bytes(reloaded.handle, "容器重导出")));
    check(detector_is(recontainer, 32, 20, 40),
          "容器重导出首段参数漂移：" + params_text(recontainer) + " 应为 (32,20,40)");
}

// d 组：阈值合同——上限内 (30,300) 成功往返；超限/倒挂/非有限训练即拒
void group_threshold_contract() {
    const cv::Mat ref = synth_reference();
    const auto bound = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 30, 300, 32);
    HandleGuard trained{train_handle(ref, bound, "strong=300")};
    const auto segment = first_fine_segment(export_bytes(trained.handle, "strong=300"));
    const DetectorParams fields = segment_detector(segment);
    check(detector_is(fields, 32, 30, 300),
          "strong=300 序列化字段漂移：" + params_text(fields) + " 应为 (32,30,300)");
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
        expect_train_reject(ref, rejection.config, rejection.text,
                            "阈值合同 strong=" + std::to_string(rejection.config.strong_th) +
                                "/weak=" + std::to_string(rejection.config.weak_th));
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

// f 组：1600×1600 三目标召回 ≥3（容量 16），逐目标 40px 覆盖
void group_multi_target_recall() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "多目标模板")};
    const cv::Point places[] = {{100, 100}, {800, 100}, {1350, 1350}};
    const cv::Mat frame = place_targets(ref, places, 3);
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
    check(isolated_rc == 0,
          "容量越界隔离批次失败 rc=" + std::to_string(isolated_rc) + "：" + g_error);
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
    check(rc == 0 && count >= 1,
          "合法匹配应成功：rc=" + std::to_string(rc) + " count=" + std::to_string(count) +
              " " + g_error);
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
