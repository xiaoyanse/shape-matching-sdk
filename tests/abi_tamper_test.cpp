// P2 批反例永久回归：codec 持久化参数合同（篡改序列化字段）、容器跨段一致性
// 篡改、默认参物化、批量容量/预算合同、K 档失败原因透传。反例均进程内内存
// 篡改重构，无文件依赖；共享支撑见 abi_chain_support.h。
#include "abi_chain_support.h"

#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace abitest;

// 自定义参固定姿态容器（跨段篡改/预算组的公共模板产物）
std::vector<uint8_t> custom_container() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "篡改模板")};
    return export_bytes(trained.handle, "篡改模板");
}

// r01 组：持久化参数与训练默认语义分离——零/下溢拒绝，端点往返稳定
void group_r01_codec_tamper() {
    const auto segment = first_fine_segment(custom_container());
    expect_load_reject(mutate_detector(segment, 32, 0.0, 40.0),
                       "persisted detector thresholds must be positive", "序列化 weak=0");
    expect_load_reject(mutate_detector(segment, 32, 20.0, 0.0),
                       "persisted detector thresholds must be positive", "序列化 strong=0");
    expect_load_reject(mutate_detector(segment, 32, 1e-50, 40.0),
                       "not representable as positive float", "序列化 weak 下溢");
    expect_load_reject(mutate_detector(segment, 32, 20.0, 1020.000001), "1020",
                       "序列化 strong 超上限");
    const auto bound = mutate_detector(segment, 32, 20.0, 1020.0);
    HandleGuard loaded{load_handle(bound, "端点 (20,1020)")};
    check_backfill(loaded.handle, 32, 20, 1020, "端点加载");
    const DetectorParams reexport =
        segment_detector(export_bytes(loaded.handle, "端点重导出"));
    check(detector_is(reexport, 32, 20, 1020),
          "端点往返漂移：" + params_text(reexport) + " 应为 (32,20,1020)");
}

// train 组：训练入口同样拒绝 float 下溢（极小正值归零产出不可加载资产）
void group_train_underflow() {
    const cv::Mat ref = synth_reference();
    expect_train_reject(ref, make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 1e-50, 40, 32),
                        "not representable as positive float", "训练 weak 下溢");
    HandleGuard ok{train_handle(ref, make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32),
                                "下溢对照")};
}

// 默认参物化组：零值训练配置物化为实际生效 (128,30,60)，往返不再见零
void group_default_params_materialize() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard trained{train_handle(ref, config, "默认参")};
    check_backfill(trained.handle, 128, 30, 60, "默认参训练句柄");
    const auto container = export_bytes(trained.handle, "默认参");
    const DetectorParams serialized = segment_detector(first_fine_segment(container));
    check(detector_is(serialized, 128, 30, 60),
          "默认参首段序列化漂移：" + params_text(serialized) + " 应为 (128,30,60)");
    HandleGuard loaded{load_handle(container, "默认参容器")};
    check_backfill(loaded.handle, 128, 30, 60, "默认参加载句柄");
}

// 容器跨段篡改组：meta 字段与模板 PNG 的跨段一致性拒绝 + 头部边长控件
void group_container_tamper() {
    const auto container = custom_container();
    expect_load_reject(with_meta_text(container, "\"k\":2", "\"k\":4"),
                       "coarse segment disagrees with reference scale", "k 改档");
    expect_load_reject(with_meta_text(container, "\"cfg\":[32,32,176,176",
                                     "\"cfg\":[32,32,175,176"),
                       "template image disagrees with original training ROI", "ROI 宽度改写");
    expect_load_reject(with_meta_text(container, "\"cfg\":[32,32,176,176",
                                     "\"cfg\":[0,32,176,176"),
                       "fine segment disagrees with original training ROI", "ROI 原点改写");
    cv::Mat oversize(24, 4097, CV_8UC1, cv::Scalar(0));
    std::vector<uint8_t> png;
    cv::imencode(".png", oversize, png);
    expect_load_reject(replace_segment(container, container_segments(container) - 2, png),
                       "template image header exceeds edge budget", "PNG 4097 超边长");
    cv::Mat edge(24, 4096, CV_8UC1, cv::Scalar(0));
    png.clear();
    cv::imencode(".png", edge, png);
    uint64_t handle = 0;
    const auto control = replace_segment(container, container_segments(container) - 2, png);
    const int32_t rc =
        vf_shape_load(control.data(), control.size(), &handle, g_error, sizeof(g_error));
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
    const std::string error_text(g_error);
    check(rc == 1 && error_text.find("disagrees with original training ROI") != std::string::npos &&
              error_text.find("header exceeds") == std::string::npos,
          "PNG 4096 控件应通过头部预检后死在 ROI 一致性，实际 rc=" + std::to_string(rc) +
              "：" + g_error);
}

// 容量合同组：四目标容量 4 必须返回 4（防去重缺失回退）；坏容量时垃圾 count 清零
void group_capacity_contract() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "容量模板")};
    const cv::Point places[] = {{100, 100}, {300, 100}, {800, 100}, {1350, 1350}};
    const cv::Mat frame = place_targets(ref, places, 4);
    std::vector<vf_shape_candidate> out(4);
    const uint32_t count = match_count(trained.handle, frame, 60.0, out, "四目标容量4");
    check(count == 4, "四目标容量 4 返回 " + std::to_string(count) + " 应恰为 4");
    for (int i = 0; i < 4; ++i) {
        bool covered = false;
        for (uint32_t j = 0; j < count && !covered; ++j) {
            const double dx = out[j].center[0] - (places[i].x + 87.5);
            const double dy = out[j].center[1] - (places[i].y + 87.5);
            covered = std::hypot(dx, dy) < 40.0;
        }
        check(covered, "目标 " + std::to_string(i) + " 近旁 40px 无候选");
    }
    uint32_t stale = 999;
    const int32_t rc = vf_shape_match(
        trained.handle, ref.data, image_bytes(ref), ref.cols, ref.rows, ref.step[0], 60.0,
        out.data(), 1, &stale, g_error, sizeof(g_error));
    check(rc == 1 && stale == 0,
          "坏容量应返回码 1 且清零垃圾 count：rc=" + std::to_string(rc) + " count=" +
              std::to_string(stale));
}

// 预算组隔离组：9 参数组同帧批量，预算外组独立失败不影响预算内组
void group_batch_budget_groups() {
    const auto segment = first_fine_segment(custom_container());
    const std::vector<std::pair<int, double>> weaks = {
        {16, 10}, {24, 15}, {32, 20}, {48, 25}, {64, 30}, {96, 35}, {128, 40}, {192, 45},
        {256, 50}};
    HandleSet models;
    std::vector<std::vector<vf_shape_candidate>> outs(weaks.size(),
                                                      std::vector<vf_shape_candidate>(4));
    std::vector<vf_shape_batch_item> items;
    for (size_t i = 0; i < weaks.size(); ++i) {
        const auto tuned =
            mutate_detector(segment, weaks[i].first, weaks[i].second, weaks[i].second * 2);
        models.handles.push_back(load_handle(tuned, "预算组 " + std::to_string(i)));
        items.push_back(make_item(models.handles.back(), 60.0, outs[i]));
    }
    cv::Mat frame(4096, 4096, CV_8UC3, cv::Scalar(0, 0, 0));
    const int32_t rc = run_batch(frame, items);
    check(rc == 0, "预算批量编排失败 rc=" + std::to_string(rc) + "：" + g_error);
    for (size_t i = 0; i + 1 < items.size(); ++i)
        check(items[i].status == 0,
              "预算内组 " + std::to_string(i) + " status=" + std::to_string(items[i].status) +
                  "：" + items[i].error);
    check(items.back().status == 1 &&
              std::string(items.back().error).find("pixel budget exceeded") != std::string::npos,
          "第 9 组应因备帧像素预算独立失败：status=" + std::to_string(items.back().status) +
              "：" + items.back().error);
}

// K 档原因组：精修层可训练但粗层几何越下限，透传真实原因而非特征不足
void group_k_fallback_reason() {
    cv::Mat dense(320, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < 40; y += 10)
        for (int x = 0; x < 40; x += 10)
            cv::rectangle(dense, {x, y}, {x + 7, y + 7}, cv::Scalar(255, 255, 255), -1);
    const auto config = make_config({0, 0, 40, 40}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    uint64_t handle = 0;
    const int32_t rc = vf_shape_train(dense.data, image_bytes(dense), dense.cols, dense.rows,
                                      dense.step[0], &config, &handle, g_error, sizeof(g_error));
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
    const std::string error_text(g_error);
    check(rc == 1 && error_text.find("invalid training ROI") != std::string::npos &&
              error_text.find("features insufficient") == std::string::npos,
          "小 ROI 整图失败应报真实几何原因（invalid training ROI，非 features insufficient），"
          "实际 rc=" + std::to_string(rc) + "：" + g_error);
}

struct Group {
    const char* name;
    void (*fn)();
};

}  // namespace

int main() {
    const Group groups[] = {
        {"i_r01_codec_tamper", group_r01_codec_tamper},
        {"j_train_underflow", group_train_underflow},
        {"k_default_params_materialize", group_default_params_materialize},
        {"l_container_tamper", group_container_tamper},
        {"m_capacity_contract", group_capacity_contract},
        {"n_batch_budget_groups", group_batch_budget_groups},
        {"o_k_fallback_reason", group_k_fallback_reason},
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
    std::cout << "abi tamper: " << (total - failures) << "/" << total << " groups passed\n";
    return failures ? 1 : 0;
}
