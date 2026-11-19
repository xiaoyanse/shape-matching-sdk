// 体检与审计合同测试：health 结构量与分级、错误码 18/19 站点、区分度审计
// 分布单调与推荐区间数学（手工算对照）、INSEPARABLE/NO_NEG 边界。
// 链接真实 SDK 库，共享支撑复用 abi_chain_support.h。
#include "abi_chain_support.h"

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

namespace {

using namespace abitest;

// fact 语义天然正负组：帧 1/3/4 有锚（模板 patch 放不同点位）=正，帧 2 无锚=负
cv::Mat fact_frame(const cv::Mat& reference, const cv::Point& place) {
    cv::Mat frame(480, 480, CV_8UC3, cv::Scalar(0, 0, 0));
    reference(cv::Rect(32, 32, 176, 176)).copyTo(frame(cv::Rect(place, cv::Size(176, 176))));
    return frame;
}

cv::Mat noise_frame(int size) {
    cv::Mat frame(size, size, CV_8UC3);
    cv::RNG rng(20260317);
    rng.fill(frame, cv::RNG::UNIFORM, 0, 256);
    return frame;
}

void expect_coded(int32_t expected, double first_param, const std::string& what) {
    int32_t code = -1, count = -1;
    double params[6] = {};
    vf_shape_last_error_info(&code, params, &count);
    check(code == expected,
          what + " 错误码漂移：" + std::to_string(code) + " 应为 " + std::to_string(expected));
    check(count >= 1 && params[0] == first_param,
          what + " 参数漂移：count=" + std::to_string(count) + " params[0]=" +
              std::to_string(params[0]) + " 应为 " + std::to_string(first_param));
}

void group_health() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "体检容器")};
    vf_shape_health_report report{};
    report.struct_size = sizeof(report);
    int32_t rc = vf_shape_model_health(trained.handle, &report, g_error, sizeof(g_error));
    check(rc == 0, "容器体检失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(report.struct_size == sizeof(report), "体检 struct_size 回显漂移");
    check(report.verdict == VF_SHAPE_HEALTH_HEALTHY,
          "健康模板应分级 HEALTHY，实际 " + std::to_string(report.verdict));
    check(report.self_score_valid == 1 && report.self_score >= 70.0,
          "容器自匹配应实测且 ≥70：" + std::to_string(report.self_score));
    check(report.pose_count >= 1, "姿态数应 ≥1");
    check(report.fine_min_features >= 4 && report.fine_max_features <= 512 &&
              report.fine_min_features <= report.fine_max_features,
          "fine 特征数口径漂移：" + std::to_string(report.fine_min_features) + ".." +
              std::to_string(report.fine_max_features));
    check(report.coarse_features >= 4, "粗层特征数应 ≥4");
    check(report.detector_features == 32 && report.detector_weak == 20.0 &&
              report.detector_strong == 40.0,
          "detector 三参快照漂移");
    check(report.template_width == 176 && report.template_height == 176,
          "模板尺寸应=训练 ROI 176×176");
    // 单级资产：无容器模板图，自匹配未实测，仅按特征数分级
    HandleGuard single{load_handle(first_fine_segment(export_bytes(trained.handle, "单级来源")),
                                   "单级模型")};
    vf_shape_health_report single_report{};
    single_report.struct_size = sizeof(single_report);
    rc = vf_shape_model_health(single.handle, &single_report, g_error, sizeof(g_error));
    check(rc == 0, "单级体检失败 rc=" + std::to_string(rc) + "：" + g_error);
    check(single_report.self_score_valid == 0 && single_report.self_score == 0.0,
          "单级自匹配应为未实测 0");
    check(single_report.coarse_features == 0, "单级粗层特征数应 0");
    check(single_report.verdict == VF_SHAPE_HEALTH_HEALTHY,
          "单级特征充足应 HEALTHY，实际 " + std::to_string(single_report.verdict));
    check(single_report.template_width > 0 && single_report.template_height > 0,
          "单级模板尺寸应取首姿态模板");
    // ABI 守卫：struct_size 未置拒绝
    vf_shape_health_report bad{};
    rc = vf_shape_model_health(trained.handle, &bad, g_error, sizeof(g_error));
    check(rc == 1, "health struct_size 未置应拒绝");
}

void group_coded_failures() {
    const cv::Mat flat(320, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto no_edges = make_config({32, 32, 176, 176}, 0, 0, 1, 1, 1, 1);
    uint64_t handle = 0;
    int32_t rc = vf_shape_train(flat.data, image_bytes(flat), flat.cols, flat.rows, flat.step[0],
                                &no_edges, &handle, g_error, sizeof(g_error));
    check(rc == 1 && std::string(g_error).find("insufficient edge features") != std::string::npos,
          "无边缘 ROI 预期拒绝，实际 rc=" + std::to_string(rc) + "：" + g_error);
    expect_coded(VF_SHAPE_INSUFFICIENT_FEATURES, 4.0, "特征不足");

    // 粗层全败：ROI 40px → K 降档只试 K=2，粗层 ROI 20px 违反 24px 合同 → 全败
    const cv::Mat ref = synth_reference();
    const auto tiny_roi = make_config({120, 120, 40, 40}, 0, 0, 1, 1, 1, 1);
    rc = vf_shape_train(ref.data, image_bytes(ref), ref.cols, ref.rows, ref.step[0],
                        &tiny_roi, &handle, g_error, sizeof(g_error));
    check(rc == 1 &&
              std::string(g_error).find("coarse model training failed at all K levels") !=
                  std::string::npos,
          "粗层全败预期拒绝，实际 rc=" + std::to_string(rc) + "：" + g_error);
    expect_coded(VF_SHAPE_COARSE_TRAIN_FAILED, 1.0, "粗层全败");
}

void check_group_stats(const vf_shape_audit_report& report, const std::string& what) {
    check(report.pos_min <= report.pos_p25 && report.pos_p25 <= report.pos_median &&
              report.pos_median <= report.pos_p75 && report.pos_p75 <= report.pos_max,
          what + " 正组分布应单调：" + std::to_string(report.pos_min) + "/" +
              std::to_string(report.pos_p25) + "/" + std::to_string(report.pos_median) + "/" +
              std::to_string(report.pos_p75) + "/" + std::to_string(report.pos_max));
    if (report.neg_count > 0) {
        check(report.neg_min <= report.neg_p25 && report.neg_p25 <= report.neg_median &&
                  report.neg_median <= report.neg_p75 && report.neg_p75 <= report.neg_max,
              what + " 负组分布应单调");
    }
    check(report.separation == report.pos_p25 - report.neg_max,
          what + " 分离度应=pos_p25−neg_max");
}

void expect_math(const vf_shape_audit_report& report, const std::string& what) {
    check_group_stats(report, what);
    const double lo = std::max(30.0, report.neg_max + 5.0);
    const double hi = std::min(85.0, report.pos_p25 - 5.0);
    check(report.recommend_lo == lo && report.recommend_hi == hi,
          what + " 推荐区间数学漂移：[" + std::to_string(report.recommend_lo) + "," +
              std::to_string(report.recommend_hi) + "] 应为 [" + std::to_string(lo) + "," +
              std::to_string(hi) + "]");
    check((lo > hi) == (report.flag == VF_SHAPE_AUDIT_INSEPARABLE),
          what + " INSEPARABLE 判定应与 lo>hi 一致");
    const bool low = report.pos_count < 5 || report.neg_count < 5;
    check((report.confidence == VF_SHAPE_AUDIT_CONFIDENCE_LOW) == low,
          what + " 置信度应与样本门槛一致");
}

void run_audit(uint64_t handle, std::vector<const uint8_t*> pos, int pw, int ph,
               std::vector<const uint8_t*> neg, int nw, int nh,
               vf_shape_audit_report& report, const std::string& what) {
    report.struct_size = sizeof(report);
    const uint64_t pos_stride = uint64_t(pw) * 3, neg_stride = uint64_t(nw) * 3;
    const int32_t rc = vf_shape_audit_discriminability(
        handle, pos.data(), pw, ph, pos_stride, uint32_t(pos.size()),
        neg.empty() ? nullptr : neg.data(), nw, nh, neg_stride, uint32_t(neg.size()),
        &report, g_error, sizeof(g_error));
    check(rc == 0, what + " 审计失败 rc=" + std::to_string(rc) + "：" + g_error);
}

void group_audit() {
    const cv::Mat ref = synth_reference();
    const auto config = make_config({32, 32, 176, 176}, -20, 20, 8, 0.8, 1.2, 0.2, 20, 40, 32);
    HandleGuard trained{train_handle(ref, config, "审计容器")};
    const cv::Mat p1 = fact_frame(ref, {40, 40});
    const cv::Mat p3 = fact_frame(ref, {260, 60});
    const cv::Mat p4 = fact_frame(ref, {60, 260});
    const cv::Mat n2(480, 480, CV_8UC3, cv::Scalar(0, 0, 0));
    // fact 正负组：正组 3 帧全命中，负组空白背景零命中；样本 <5 → LOW
    vf_shape_audit_report report{};
    run_audit(trained.handle, {p1.data, p3.data, p4.data}, 480, 480, {n2.data}, 480, 480,
              report, "fact 正负组");
    check(report.pos_count == 3 && report.pos_zero_hits == 0,
          "正组应 3 帧全有分：count=" + std::to_string(report.pos_count));
    check(report.neg_count == 0 && report.neg_zero_hits == 1,
          "负组空白应零命中：count=" + std::to_string(report.neg_count) + " zero=" +
              std::to_string(report.neg_zero_hits));
    check(report.pos_max >= 70.0, "正组最高分应 ≥70：" + std::to_string(report.pos_max));
    expect_math(report, "fact 正负组");
    check(report.flag == VF_SHAPE_AUDIT_RECOMMENDED, "可分离组应 RECOMMENDED");

    // NO_NEG：负组缺省只报正组分布，不推荐区间
    vf_shape_audit_report no_neg{};
    run_audit(trained.handle, {p1.data, p3.data, p4.data}, 480, 480, {}, 0, 0, no_neg, "无负组");
    check(no_neg.flag == VF_SHAPE_AUDIT_NO_NEGATIVE, "缺负组应 NO_NEGATIVE");
    check(no_neg.neg_count == 0 && no_neg.recommend_lo == 0.0 && no_neg.recommend_hi == 0.0,
          "缺负组不应产推荐区间");
    check(no_neg.pos_count == 3, "无负组正组分布应保留");

    // INSEPARABLE：两组同为噪声帧，分数重叠无法给出可用区间
    const cv::Mat noise = noise_frame(320);
    vf_shape_audit_report inseparable{};
    run_audit(trained.handle, {noise.data, noise.data}, 320, 320, {noise.data, noise.data},
              320, 320, inseparable, "噪声双组");
    check(inseparable.flag == VF_SHAPE_AUDIT_INSEPARABLE, "噪声双组应 INSEPARABLE");

    // 单级资产：扫描阈值取全分布，数学合同一致
    HandleGuard single{load_handle(first_fine_segment(export_bytes(trained.handle, "单级来源")),
                                   "单级模型")};
    vf_shape_audit_report single_report{};
    run_audit(single.handle, {p1.data, p3.data, p4.data}, 480, 480, {n2.data}, 480, 480,
              single_report, "单级审计");
    check_group_stats(single_report, "单级审计");
    check(single_report.pos_count == 3 && single_report.pos_zero_hits == 0,
          "单级正组应 3 帧有分");
    check(single_report.neg_count == 0 && single_report.neg_zero_hits == 1,
          "单级空白负组应零命中");
    const double lo = std::max(30.0, single_report.neg_max + 5.0);
    const double hi = std::min(85.0, single_report.pos_p25 - 5.0);
    check(single_report.recommend_lo == lo && single_report.recommend_hi == hi,
          "单级推荐区间数学漂移");
    check(single_report.flag == VF_SHAPE_AUDIT_RECOMMENDED, "单级可分离组应 RECOMMENDED");

    // 边界与 ABI 守卫：正组空、负组数组缺失、struct_size 未置均拒绝
    const uint8_t* one_image = p1.data;
    const uint8_t* neg_image = n2.data;
    int32_t rc = vf_shape_audit_discriminability(
        trained.handle, nullptr, 480, 480, 1440, 0, nullptr, 480, 480, 1440, 0, &report,
        g_error, sizeof(g_error));
    check(rc == 1, "正组空应拒绝");
    rc = vf_shape_audit_discriminability(
        trained.handle, &one_image, 480, 480, 1440, 1, nullptr, 480, 480, 1440, 2, &report,
        g_error, sizeof(g_error));
    check(rc == 1, "负组计数>0 而数组缺失应拒绝");
    vf_shape_audit_report bad{};
    rc = vf_shape_audit_discriminability(
        trained.handle, &one_image, 480, 480, 1440, 1, &neg_image, 480, 480, 1440, 1, &bad,
        g_error, sizeof(g_error));
    check(rc == 1, "audit struct_size 未置应拒绝");
}

}  // namespace

int main() {
    const struct Group {
        const char* name;
        void (*fn)();
    } groups[] = {
        {"a_health", group_health},
        {"b_coded_failures", group_coded_failures},
        {"c_audit", group_audit},
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
    std::cout << "audit: " << (total - failures) << "/" << total << " groups passed\n";
    return failures ? 1 : 0;
}
