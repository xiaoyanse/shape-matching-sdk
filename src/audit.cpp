// 体检与审计内部实现：健康报告从容器模板图与模型内部量导出（含一次自
// 匹配）；审计逐帧取现有匹配路径最高分构成分布，核算建议性阈值区间。

#include "audit.hpp"

#include <algorithm>
#include <vector>

namespace vfshape {
namespace {

constexpr double kHealthSelfThreshold = 70.0;
constexpr double kHealthDegradedSelfScore = 90.0;
// 审计扫描阈值：(0,100] 域的下界实践值，只用于让现有匹配不做阈值过滤；
// 两级路径内部粗/精门槛（45/50）语义不受其影响
constexpr double kAuditSweepThreshold = 1e-6;
constexpr int kAuditMaxCandidates = 256;
constexpr double kRecommendFloor = 30.0;
constexpr double kRecommendCeiling = 85.0;
constexpr double kNegMargin = 5.0;
constexpr double kPosMargin = 5.0;
constexpr uint32_t kConfidenceMinSamples = 5;

// 该模型各金字塔层特征数的最小/最大（pose id 与模板一一对应；空模型保持 0）
void collect_level_features(const Model& model, uint32_t& mn, uint32_t& mx) {
    bool first = true;
    for (uint32_t id = 0; id < model.poses.size(); ++id) {
        for (const auto& level : model.detector.getTemplates("shape", int(id))) {
            const auto count = uint32_t(level.features.size());
            if (first) {
                mn = mx = count;
                first = false;
            } else {
                mn = std::min(mn, count);
                mx = std::max(mx, count);
            }
        }
    }
}

double linear_quantile(const std::vector<double>& sorted, double q) {
    if (sorted.size() == 1) return sorted.front();
    const double position = q * double(sorted.size() - 1);
    const size_t lo = size_t(position);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (position - double(lo));
}

double frame_top_score(Model* single, TwoStageModel* two_stage, const cv::Mat& image) {
    std::vector<vf_shape_candidate> found;
    if (two_stage)
        found = match_two_stage(*two_stage, image, kAuditSweepThreshold, kAuditMaxCandidates);
    else
        found = match(*single, image, kAuditSweepThreshold);
    double top = 0;
    for (const auto& candidate : found) top = std::max(top, candidate.score);
    return top;
}

struct GroupStats {
    uint32_t count = 0, zero_hits = 0;
    double mn = 0, p25 = 0, median = 0, p75 = 0, mx = 0;
};

GroupStats eval_group(Model* single, TwoStageModel* two_stage,
                      const uint8_t* const* images, int width, int height, uint64_t stride,
                      uint32_t count) {
    GroupStats stats;
    std::vector<double> scores;
    scores.reserve(count);
    for (uint32_t n = 0; n < count; ++n) {
        const auto frame = checked_image(images[n], stride * uint64_t(height), width, height,
                                         stride);
        const double top = frame_top_score(single, two_stage, frame);
        if (top <= 0) {
            ++stats.zero_hits;
            continue;
        }
        scores.push_back(top);
    }
    stats.count = uint32_t(scores.size());
    if (scores.empty()) return stats;
    std::sort(scores.begin(), scores.end());
    stats.mn = scores.front();
    stats.mx = scores.back();
    stats.p25 = linear_quantile(scores, 0.25);
    stats.median = linear_quantile(scores, 0.5);
    stats.p75 = linear_quantile(scores, 0.75);
    return stats;
}

}

vf_shape_health_report model_health(Model* single, TwoStageModel* two_stage) {
    vf_shape_health_report report{};
    report.struct_size = sizeof(report);
    uint32_t fine_mn = 0, fine_mx = 0, coarse_mn = 0;
    if (two_stage) {
        bool first_group = true;
        for (const auto& fine : two_stage->fines) {
            uint32_t mn = 0, mx = 0;
            collect_level_features(*fine, mn, mx);
            if (first_group) {
                fine_mn = mn;
                fine_mx = mx;
                first_group = false;
            } else {
                fine_mn = std::min(fine_mn, mn);
                fine_mx = std::max(fine_mx, mx);
            }
            report.pose_count += uint32_t(fine->poses.size());
        }
        uint32_t coarse_mx = 0;
        collect_level_features(*two_stage->coarse, coarse_mn, coarse_mx);
        report.template_width = uint32_t(two_stage->tmpl_gray.cols);
        report.template_height = uint32_t(two_stage->tmpl_gray.rows);
        report.self_score_valid = 1;
        cv::Mat template_bgr;
        cv::cvtColor(two_stage->tmpl_gray, template_bgr, cv::COLOR_GRAY2BGR);
        const auto candidates = match_two_stage(*two_stage, template_bgr, kHealthSelfThreshold,
                                                kAuditMaxCandidates);
        for (const auto& candidate : candidates)
            report.self_score = std::max(report.self_score, candidate.score);
        const auto& tuned = two_stage->fines.front()->detector;
        report.detector_features = tuned.tuned_features;
        report.detector_weak = tuned.tuned_weak;
        report.detector_strong = tuned.tuned_strong;
    } else {
        collect_level_features(*single, fine_mn, fine_mx);
        report.pose_count = uint32_t(single->poses.size());
        if (!single->poses.empty()) {
            const auto& base = single->detector.getTemplates("shape", 0);
            if (!base.empty()) {
                report.template_width = uint32_t(base.front().width);
                report.template_height = uint32_t(base.front().height);
            }
        }
        const auto& tuned = single->detector;
        report.detector_features = tuned.tuned_features;
        report.detector_weak = tuned.tuned_weak;
        report.detector_strong = tuned.tuned_strong;
    }
    report.fine_min_features = fine_mn;
    report.fine_max_features = fine_mx;
    report.coarse_features = coarse_mn;
    // 分级取更重：任一层（fine 与非零 coarse）触下限归 WEAK，<8 归 DEGRADED
    const uint32_t overall_min = coarse_mn > 0 ? std::min(fine_mn, coarse_mn) : fine_mn;
    const bool self_weak = report.self_score_valid &&
                           report.self_score < kHealthSelfThreshold;
    const bool self_degraded = report.self_score_valid &&
                               report.self_score < kHealthDegradedSelfScore;
    if (self_weak || overall_min <= 4)
        report.verdict = VF_SHAPE_HEALTH_WEAK;
    else if (self_degraded || overall_min < 8)
        report.verdict = VF_SHAPE_HEALTH_DEGRADED;
    else
        report.verdict = VF_SHAPE_HEALTH_HEALTHY;
    return report;
}

vf_shape_audit_report audit_discriminability(
    Model* single, TwoStageModel* two_stage,
    const uint8_t* const* pos_images, int pos_width, int pos_height, uint64_t pos_stride,
    uint32_t pos_count,
    const uint8_t* const* neg_images, int neg_width, int neg_height, uint64_t neg_stride,
    uint32_t neg_count) {
    vf_shape_audit_report report{};
    report.struct_size = sizeof(report);
    const GroupStats pos = eval_group(single, two_stage, pos_images, pos_width, pos_height,
                                      pos_stride, pos_count);
    report.pos_count = pos.count;
    report.pos_zero_hits = pos.zero_hits;
    report.pos_min = pos.mn;
    report.pos_p25 = pos.p25;
    report.pos_median = pos.median;
    report.pos_p75 = pos.p75;
    report.pos_max = pos.mx;
    if (neg_count == 0) {
        report.flag = VF_SHAPE_AUDIT_NO_NEGATIVE;
        report.confidence = pos.count < kConfidenceMinSamples
                                ? VF_SHAPE_AUDIT_CONFIDENCE_LOW
                                : VF_SHAPE_AUDIT_CONFIDENCE_HIGH;
        return report;
    }
    const GroupStats neg = eval_group(single, two_stage, neg_images, neg_width, neg_height,
                                      neg_stride, neg_count);
    report.neg_count = neg.count;
    report.neg_zero_hits = neg.zero_hits;
    report.neg_min = neg.mn;
    report.neg_p25 = neg.p25;
    report.neg_median = neg.median;
    report.neg_p75 = neg.p75;
    report.neg_max = neg.mx;
    report.separation = pos.p25 - neg.mx;
    const double lo = std::max(kRecommendFloor, neg.mx + kNegMargin);
    const double hi = std::min(kRecommendCeiling, pos.p25 - kPosMargin);
    report.recommend_lo = lo;
    report.recommend_hi = hi;
    report.flag = lo > hi ? VF_SHAPE_AUDIT_INSEPARABLE : VF_SHAPE_AUDIT_RECOMMENDED;
    report.confidence =
        (pos.count < kConfidenceMinSamples || neg.count < kConfidenceMinSamples)
            ? VF_SHAPE_AUDIT_CONFIDENCE_LOW
            : VF_SHAPE_AUDIT_CONFIDENCE_HIGH;
    return report;
}

}
