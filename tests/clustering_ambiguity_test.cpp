// 聚类歧义分数差豁免契约测试：锁定 matcher.cpp 聚簇合并的置歧义条件——
// 同簇两候选姿态差 ≥15°（-20/0/20 三档训练、相邻档差 20°）且分数差 < 豁免阈
// kAmbiguityScoreGap(=10) → pose_ambiguous=1（无法分辨的多姿态解）；分数差
// ≥ 豁免阈 → 高分解明显领先，豁免不置位。防空洞：聚类合并会吞掉簇内候选，
// 先经 detector.matchPrepared 枚举聚类前的原始命中，确认同目标确有姿态差
// 超限、分差落在预期侧的双命中，再断言聚类输出标志（排除"单档命中→无合并
// →标志恒 0"的空洞通过）。--probe 打印各旋转角的原始分表供校准复用。
#include "model.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kPattern = 96;                       // 目标边长（对齐 Python 聚类契约测试图案）
constexpr int kFrameW = 320, kFrameH = 240;
constexpr double kTargetX = kFrameW / 2., kTargetY = kFrameH / 2.;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

cv::Mat pattern() {  // 白外框 + 左上实心块：非对称、各金字塔层边缘特征充足
    cv::Mat img(kPattern, kPattern, CV_8UC1, cv::Scalar(0));
    cv::rectangle(img, {2, 2}, {kPattern - 3, kPattern - 3}, cv::Scalar(255), 6);
    cv::rectangle(img, {16, 16}, {39, 39}, cv::Scalar(255), -1);
    return img;
}

cv::Mat to_bgr(const cv::Mat& gray) {
    cv::Mat bgr;
    cv::merge(std::vector<cv::Mat>{gray, gray, gray}, bgr);
    return bgr;
}

cv::Mat reference_bgr() {
    cv::Mat canvas(128, 128, CV_8UC1, cv::Scalar(0));
    pattern().copyTo(canvas(cv::Rect(16, 16, kPattern, kPattern)));
    return to_bgr(canvas);
}

// 旋转 angle 度并扩展画布（对齐 Python 测试 PIL rotate(expand=True) 构造）
cv::Mat rotated_pattern(double angle) {
    const cv::Mat src = pattern();
    if (std::abs(angle) < 1e-9) return src;
    const double rad = angle * CV_PI / 180.;
    const int side = int(std::ceil(kPattern * (std::cos(rad) + std::sin(rad))));
    const cv::Mat rot =
        cv::getRotationMatrix2D(cv::Point2f(src.cols / 2.f, src.rows / 2.f), angle, 1);
    cv::Mat dst(side, side, CV_8UC1, cv::Scalar(0));
    cv::warpAffine(src, dst, rot, dst.size());
    return dst;
}

cv::Mat frame_bgr(double angle) {  // 单目标置于帧中央
    cv::Mat canvas(kFrameH, kFrameW, CV_8UC1, cv::Scalar(0));
    const cv::Mat patch = rotated_pattern(angle);
    const cv::Rect roi(int(kTargetX - patch.cols / 2.), int(kTargetY - patch.rows / 2.),
                       patch.cols, patch.rows);
    patch.copyTo(canvas(roi));
    return to_bgr(canvas);
}

vf_shape_train_config train_config() {
    vf_shape_train_config config{};
    config.struct_size = sizeof(config);
    config.roi_x = 16;
    config.roi_y = 16;
    config.roi_width = kPattern;
    config.roi_height = kPattern;
    config.angle_min = -20;  // 档位 -20/0/20：相邻档角差 20° ≥ 歧义阈 15°
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

// 聚类前原始命中：各姿态档在帧上的最高分（单目标空底帧，最佳命中即目标处）
std::map<double, double> pose_scores(const vfshape::Model& model,
                                     const vfshape::PreparedFrame& frame, double threshold) {
    std::map<double, double> best;  // pose 角度 → 最高相似度
    for (const auto& hit :
         model.detector.matchPrepared(frame.features, float(threshold), {"shape"})) {
        if (hit.template_id < 0 || size_t(hit.template_id) >= model.poses.size()) continue;
        double& slot = best[model.poses[size_t(hit.template_id)].angle];
        slot = std::max(slot, double(hit.similarity));
    }
    return best;
}

bool near_target(const vf_shape_candidate& candidate, double tolerance = 25.) {
    return std::hypot(candidate.center[0] - kTargetX, candidate.center[1] - kTargetY) <=
           tolerance;
}

// 单用例：给定旋转角与阈值，先核原始双命中分差落在预期侧，再断言聚类标志
void check_ambiguity(vfshape::Model& model, double rotation, double threshold,
                     bool expect_ambiguous) {
    const cv::Mat image = frame_bgr(rotation);
    const auto frame = vfshape::prepare_frame(image, &model.detector);
    const auto scores = pose_scores(model, frame, threshold);
    check(scores.count(0.) == 1 && scores.count(20.) == 1,
          "rotation " + std::to_string(rotation) + ": 0/20 档未同时命中，校准失效");
    const double gap = scores.at(0.) - scores.at(20.);
    check(std::abs(20. - 0.) >= 15., "档间角差必须 ≥15° 才构成姿态冲突");
    if (expect_ambiguous)
        check(gap < 10., "校准失效：分差 " + std::to_string(gap) + " 应 <10 才能验证维持歧义");
    else
        check(gap >= 10., "校准失效：分差 " + std::to_string(gap) + " 应 ≥10 才能验证豁免");

    const auto found = vfshape::match_prepared(model, frame, threshold);
    check(!found.empty() && near_target(found.front()), "目标处必须有聚类代表输出");
    if (!expect_ambiguous)
        check(found.size() == 1, "同目标双命中必须合并为单簇，否则豁免断言空洞");
    check(found.front().pose_ambiguous == (expect_ambiguous ? 1u : 0u),
          std::string("rotation ") + std::to_string(rotation) + " 分差 " +
              std::to_string(gap) + " 的歧义标志与预期不符");
    std::cout << "rotation=" << rotation << " threshold=" << threshold
              << " gap=" << gap << " ambiguous=" << found.front().pose_ambiguous << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto model = vfshape::train(reference_bgr(), train_config());
        if (argc > 1 && std::string(argv[1]) == "--probe") {  // 校准表：各旋转角原始分
            for (int rotation = 0; rotation <= 16; rotation += 1) {
                const auto frame =
                    vfshape::prepare_frame(frame_bgr(rotation), &model->detector);
                std::cout << "rotation " << rotation << ":";
                for (const auto& [angle, score] : pose_scores(*model, frame, 1.))
                    std::cout << " " << angle << "deg=" << score;
                std::cout << "\n";
            }
            return 0;
        }
        // 旋转 11°：0/20 档分数 80/78（分差 ≈1.7 < 10）→ 维持歧义；
        // 旋转 4°：0/20 档分数 98/52（分差 ≈46.6 ≥ 10）→ 高分领先豁免。
        // -20 档在两旋转下分别为 30/39 分，被阈值 40 排除，不干扰簇内配对
        check_ambiguity(*model, 11., 40., true);
        check_ambiguity(*model, 4., 40., false);
        std::cout << "clustering ambiguity tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
