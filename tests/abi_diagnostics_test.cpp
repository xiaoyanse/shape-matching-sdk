// 诊断三件套合同测试：稳定错误码+数值参数、预算查询与训练拒绝边界一致、
// 最近匹配诊断 trace。链接真实 SDK 库，共享支撑复用 abi_chain_support.h。
#include "abi_chain_support.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using namespace abitest;

// 触发一次预期失败并断言线程局部错误码与参数前缀
void expect_coded_failure(int32_t expected_code, const std::string& what) {
    int32_t code = -1, count = -1;
    double params[6] = {};
    vf_shape_last_error_info(&code, params, &count);
    check(code == expected_code,
          what + " 错误码漂移：" + std::to_string(code) + " 应为 " +
              std::to_string(expected_code));
    check(count >= 0 && count <= 6, what + " 参数个数越界：" + std::to_string(count));
}

void group_error_codes() {
    const cv::Mat ref = synth_reference();
    // 角度步过细 → INVALID_ANGLE_RANGE，参数=当前步长/档数上限/建议步长
    const auto over_fine_angle = make_config({32, 32, 176, 176}, -20, 20, 0.05,
                                             1, 1, 1, 20, 40, 32);
    expect_train_reject(ref, over_fine_angle, "pose budget exceeded", "角度步过细");
    {
        int32_t code = -1, count = -1;
        double params[6] = {};
        vf_shape_last_error_info(&code, params, &count);
        check(code == VF_SHAPE_INVALID_ANGLE_RANGE,
              "角度步过细错误码漂移：" + std::to_string(code) + " 应为 INVALID_ANGLE_RANGE(3)");
        check(count == 3, "角度步过细参数个数漂移：" + std::to_string(count) + " 应为 3");
        check(params[0] == 0.05, "参数[0]应为当前步长 0.05");
        check(params[1] == 350.0, "参数[1]应为档数上限 350");
        const double suggested = 40.0 / 349.0;
        check(std::abs(params[2] - suggested) < 1e-9,
              "参数[2]应为建议步长 " + std::to_string(suggested));
    }
    // 缩放步过细 → INVALID_SCALE_RANGE
    expect_train_reject(ref, make_config({32, 32, 176, 176}, 0, 0, 1, 0.25, 2, 0.001, 20, 40, 32),
                        "pose budget exceeded", "缩放步过细");
    expect_coded_failure(VF_SHAPE_INVALID_SCALE_RANGE, "缩放步过细");
    // ROI 越界 → INVALID_ROI；图像合同 → INVALID_IMAGE
    expect_train_reject(ref, make_config({300, 300, 100, 100}, 0, 0, 1, 1, 1, 1, 20, 40, 32),
                        "invalid training ROI", "ROI 越界");
    expect_coded_failure(VF_SHAPE_INVALID_ROI, "ROI 越界");
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "诊断模板")};
    std::vector<vf_shape_candidate> out(4);
    uint32_t count = 0;
    int32_t rc = vf_shape_match(trained.handle, ref.data, image_bytes(ref), 23, ref.rows,
                                ref.step[0], 60.0, out.data(), uint32_t(out.size()), &count,
                                g_error, sizeof(g_error));
    check(rc == 1, "23px 宽度应拒绝");
    expect_coded_failure(VF_SHAPE_INVALID_IMAGE, "图像过窄");
    // 阈值/容量/句柄
    rc = vf_shape_match(trained.handle, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], std::nan(""), out.data(), uint32_t(out.size()), &count,
                        g_error, sizeof(g_error));
    check(rc == 1, "NaN 阈值应拒绝");
    expect_coded_failure(VF_SHAPE_THRESHOLD_INVALID, "NaN 阈值");
    rc = vf_shape_match(trained.handle, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], 60.0, out.data(), 1, &count, g_error, sizeof(g_error));
    check(rc == 1, "capacity=1 应拒绝");
    expect_coded_failure(VF_SHAPE_CAPACITY_INVALID, "容量越界");
    rc = vf_shape_match(0xDEADBEEF, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], 60.0, out.data(), uint32_t(out.size()), &count,
                        g_error, sizeof(g_error));
    check(rc == 1, "未知句柄应拒绝");
    expect_coded_failure(VF_SHAPE_HANDLE_INVALID, "未知句柄");
    // detector 三参
    expect_train_reject(ref, make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 30, 2000, 32),
                        "reachable gradient magnitude", "strong 超限");
    expect_coded_failure(VF_SHAPE_DETECTOR_PARAMS, "detector 三参越界");
    // 模型格式/尺寸/篡改
    const std::vector<uint8_t> garbage(64, uint8_t('x'));
    expect_load_reject(garbage, "unsupported model source/format", "坏格式字节");
    expect_coded_failure(VF_SHAPE_MODEL_FORMAT, "坏格式字节");
    const std::vector<uint8_t> oversized(32 * 1024 * 1024 + 1, uint8_t(0));
    expect_load_reject(oversized, "invalid model buffer or size", "超 32MiB 字节");
    expect_coded_failure(VF_SHAPE_MODEL_SIZE, "超 32MiB 字节");
    const auto ranged = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard ranged_model{train_handle(ref, ranged, "容器来源")};
    const auto container = export_bytes(ranged_model.handle, "容器来源");
    const auto tampered = with_meta_text(container, "\"k\":2", "\"k\":6");
    expect_load_reject(tampered, "invalid coarse factor", "meta k 改档");
    expect_coded_failure(VF_SHAPE_MODEL_TAMPER, "meta k 改档");
    // 成功调用不清除上次记录：成功元数据查询后再查，码仍在
    vf_shape_train_config meta{};
    meta.struct_size = sizeof(meta);
    int32_t width = 0, height = 0;
    rc = vf_shape_metadata(trained.handle, &meta, &width, &height, g_error, sizeof(g_error));
    check(rc == 0, "元数据成功调用应通过");
    expect_coded_failure(VF_SHAPE_MODEL_TAMPER, "成功调用后记录保留");
}

// 预算查询（三账本）与训练拒绝的边界一致：查询报超限 ↔ 训练拒绝
vf_shape_budget_report estimate(const vf_shape_train_config& config, const cv::Mat& image,
                                const std::string& what) {
    vf_shape_budget_report report{};
    report.struct_size = sizeof(report);
    const int32_t rc = vf_shape_estimate_training(&config, image.cols, image.rows, &report,
                                                  g_error, sizeof(g_error));
    check(rc == 0, what + " 预算查询失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(report.formula_version == 1, what + " 公式版本应为 1");
    return report;
}

void expect_budget_boundary(const vf_shape_train_config& config, const cv::Mat& image,
                            bool should_train, int32_t factor, const std::string& what) {
    const auto report = estimate(config, image, what);
    check(report.over_budget == (should_train ? 0 : factor),
          what + " 首要超限因子漂移：" + std::to_string(report.over_budget) + " 应为 " +
              std::to_string(should_train ? 0 : factor));
    uint64_t handle = 0;
    const int32_t rc = vf_shape_train(image.data, image_bytes(image), image.cols, image.rows,
                                      image.step[0], &config, &handle, g_error, sizeof(g_error));
    if (rc == 0 && handle) vf_shape_close(handle, nullptr, 0);
    check((rc == 0) == should_train,
          what + " 查询与训练边界不一致：查询超限=" +
              std::to_string(report.over_budget != 0) + " 训练 rc=" + std::to_string(rc) +
              " " + g_error);
}

void group_budget_boundary() {
    // 姿态档数恰好压线：175 角度 × 2 缩放 = 350 通过；176 × 2 = 352 超限
    const cv::Mat small_ref = synth_reference();
    expect_budget_boundary(make_config({32, 32, 176, 176}, -87, 87, 1, 0.5, 1.0, 0.5, 20, 40, 32),
                           small_ref, true, 0, "姿态 350 恰好压线");
    expect_budget_boundary(make_config({32, 32, 176, 176}, -87.5, 87.5, 1, 0.5, 1.0, 0.5, 20, 40, 32),
                           small_ref, false, VF_SHAPE_POSE_BUDGET, "姿态 352 超限");
    // 画布账本：1600² 参考图上 800² ROI 压线通过、850² 超限（缩放 2）
    cv::Mat big(1600, 1600, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(big, {600, 600}, {655, 755}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(big, {600, 600}, {755, 655}, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(big, {600, 700}, {755, 755}, cv::Scalar(255, 255, 255), -1);
    expect_budget_boundary(make_config({375, 375, 800, 800}, 0, 0, 1, 1.0, 2.0, 1.0, 20, 40, 32),
                           big, true, 0, "画布 3912 压线");
    expect_budget_boundary(make_config({375, 375, 850, 850}, 0, 0, 1, 1.0, 2.0, 1.0, 20, 40, 32),
                           big, false, VF_SHAPE_CANVAS_BUDGET, "画布 4154 超限");
    // 像素账本：多档缩放累计超 2.5e8；少档同 ROI 通过
    expect_budget_boundary(make_config({450, 450, 700, 700}, -8, 8, 8, 1.0, 2.0, 0.05, 20, 40, 32),
                           big, false, VF_SHAPE_PIXEL_BUDGET, "像素累计超限");
    expect_budget_boundary(make_config({450, 450, 700, 700}, -8, 8, 8, 1.0, 1.2, 0.1, 20, 40, 32),
                           big, true, 0, "像素累计压线通过");
    // 域不合法与训练同判拒绝（返回 1 + 文本，不产报告）
    const auto invalid = make_config({32, 32, 176, 176}, 0, 0, 0, 1, 1, 1, 20, 40, 32);
    vf_shape_budget_report report{};
    report.struct_size = sizeof(report);
    const int32_t rc = vf_shape_estimate_training(&invalid, small_ref.cols, small_ref.rows,
                                                  &report, g_error, sizeof(g_error));
    check(rc == 1, "step=0 域不合法应拒绝");
    check(std::string(g_error).find("invalid training range") != std::string::npos,
          "域拒绝文本漂移");
    // ABI 守卫：struct_size 不符拒绝
    vf_shape_budget_report bad_report{};
    const int32_t abi_rc = vf_shape_estimate_training(
        &invalid, small_ref.cols, small_ref.rows, &bad_report, g_error, sizeof(g_error));
    check(abi_rc == 1, "struct_size 未置应拒绝");
}

void expect_legacy_trace(uint64_t handle, const vf_shape_match_trace& trace) {
    constexpr uint32_t legacy_size = 112;
    constexpr unsigned char sentinel = 0xAB;
    struct {
        vf_shape_match_trace trace;
        std::array<unsigned char, 16> guard;
    } buffer{};
    std::memset(&buffer, sentinel, sizeof(buffer));
    buffer.trace.struct_size = legacy_size;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&buffer);
    const auto* expected = reinterpret_cast<const unsigned char*>(&trace);
    constexpr auto size_field = sizeof(buffer.trace.struct_size);
    for (int attempt = 1; attempt <= 2; ++attempt) {
        const int32_t rc = vf_shape_last_match_info(handle, &buffer.trace, g_error, sizeof(g_error));
        const std::string label = "v1 trace 调用 " + std::to_string(attempt);
        check(rc == 0, label + " 失败：" + g_error);
        check(buffer.trace.struct_size == legacy_size, label + " 未回显原始声明尺寸");
        check(std::memcmp(bytes + size_field, expected + size_field, legacy_size - size_field) == 0,
              label + " 前缀字段与 v2 快照不一致");
        check(std::all_of(bytes + legacy_size, bytes + sizeof(buffer),
                          [sentinel](unsigned char byte) { return byte == sentinel; }),
              label + " 改写了声明范围外的尾哨兵");
    }
}

void group_match_trace() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "trace 模板")};
    const cv::Point places[] = {{100, 100}, {800, 100}, {1350, 1350}};
    const cv::Mat frame = place_targets(ref, places, 3);
    std::vector<vf_shape_candidate> out(16);
    const uint32_t hits = match_count(trained.handle, frame, 60.0, out, "三目标 trace 帧");
    check(hits >= 3, "三目标召回 " + std::to_string(hits) + " 应 ≥3");
    vf_shape_match_trace trace{};
    trace.struct_size = sizeof(trace);
    int32_t rc = vf_shape_last_match_info(trained.handle, &trace, g_error, sizeof(g_error));
    check(rc == 0, "trace 查询失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(trace.struct_size == sizeof(trace), "trace struct_size 回显漂移");
    check(trace.windows_searched >= 2,
          "三目标场景精修窗数 " + std::to_string(trace.windows_searched) + " 应 ≥2");
    check(trace.windows_searched <= 8 && trace.coarse_candidates >= trace.windows_searched,
          "窗数/粗候选数口径漂移：windows=" + std::to_string(trace.windows_searched) +
              " coarse=" + std::to_string(trace.coarse_candidates));
    check(trace.final_candidates == hits,
          "最终候选数 " + std::to_string(trace.final_candidates) + " 应与返回 " +
              std::to_string(hits) + " 一致");
    for (int n = 0; n < int(trace.windows_searched) && n < 8; ++n) {
        check(trace.window_best_scores[n] > 0.0 && trace.window_best_scores[n] <= 100.0,
              "窗 " + std::to_string(n) + " 最佳分越界：" +
                  std::to_string(trace.window_best_scores[n]));
    }
    for (double ms : {trace.coarse_ms, trace.coarse_resize_ms, trace.windows_ms, trace.finish_ms})
        check(ms == 0.0, "常规构建耗时字段应为 0");
    // 二次调用不清除：快照逐字节一致
    vf_shape_match_trace again{};
    again.struct_size = sizeof(again);
    rc = vf_shape_last_match_info(trained.handle, &again, g_error, sizeof(g_error));
    check(rc == 0 && std::memcmp(&trace, &again, sizeof(trace)) == 0,
          "二次 trace 查询应得同一快照");
    for (double score : trace.window_collect_best_scores)
        check(score == 0.0, "清洁命中未尝试救分，v2 各窗收集分应为 0");
    check(trace.rescued_candidates == 0, "清洁命中未尝试救分，v2 救活候选数应为 0");
    expect_legacy_trace(trained.handle, trace);
    // 单级模型：仅 final_candidates 有效
    const auto ranged = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2);
    HandleGuard two_level{train_handle(ref, ranged, "单级来源")};
    HandleGuard single{load_handle(first_fine_segment(export_bytes(two_level.handle, "单级来源")),
                                   "单级模型")};
    std::vector<vf_shape_candidate> single_out(16);
    const uint32_t single_hits = match_count(single.handle, ref, 60.0, single_out, "单级同图");
    check(single_hits >= 1, "单级同图应命中");
    vf_shape_match_trace single_trace{};
    single_trace.struct_size = sizeof(single_trace);
    rc = vf_shape_last_match_info(single.handle, &single_trace, g_error, sizeof(g_error));
    check(rc == 0 && single_trace.final_candidates == single_hits &&
              single_trace.windows_searched == 0 && single_trace.coarse_candidates == 0,
          "单级 trace 口径漂移：final=" + std::to_string(single_trace.final_candidates) +
              " windows=" + std::to_string(single_trace.windows_searched));
    // 失败匹配不动档案：NaN 阈值拒绝后 trace 仍为上次成功快照
    uint32_t count = 0;
    rc = vf_shape_match(single.handle, ref.data, image_bytes(ref), ref.cols, ref.rows,
                        ref.step[0], std::nan(""), single_out.data(),
                        uint32_t(single_out.size()), &count, g_error, sizeof(g_error));
    check(rc == 1, "NaN 阈值应拒绝");
    vf_shape_match_trace preserved{};
    preserved.struct_size = sizeof(preserved);
    rc = vf_shape_last_match_info(single.handle, &preserved, g_error, sizeof(g_error));
    check(rc == 0 && preserved.final_candidates == single_hits,
          "失败匹配不应清除上次诊断快照");
    // 句柄 ABI 守卫：struct_size 未置拒绝
    vf_shape_match_trace bad_trace{};
    rc = vf_shape_last_match_info(trained.handle, &bad_trace, g_error, sizeof(g_error));
    check(rc == 1, "trace struct_size 未置应拒绝");
}

}  // namespace

int main() {
    const struct Group {
        const char* name;
        void (*fn)();
    } groups[] = {
        {"a_error_codes", group_error_codes},
        {"b_budget_boundary", group_budget_boundary},
        {"c_match_trace", group_match_trace},
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
    std::cout << "diagnostics: " << (total - failures) << "/" << total << " groups passed\n";
    return failures ? 1 : 0;
}
