// 遮挡救分：重整、占用测量、旧决策与拒绝路径回归。
// --v-audit / --fp-probe 保持原离线审计入口。
#include "matching_profile.hpp"
#include "rescue_score.hpp"
#include "two_stage.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr int kPattern = 96;
constexpr int kFrameW = 320, kFrameH = 240;
constexpr double kTargetX = kFrameW / 2., kTargetY = kFrameH / 2.;

cv::Mat pattern() {
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

cv::Mat frame_bgr(double occlusion_ratio = 0.0) {
    cv::Mat canvas(kFrameH, kFrameW, CV_8UC1, cv::Scalar(0));
    pattern().copyTo(canvas(cv::Rect(int(kTargetX - kPattern / 2.), int(kTargetY - kPattern / 2.),
                                     kPattern, kPattern)));
    if (occlusion_ratio > 0.0) {
        const int edge = int(std::round(kPattern * std::sqrt(occlusion_ratio)));
        const int x0 = int(kTargetX - edge / 2.), y0 = int(kTargetY - edge / 2.);
        canvas(cv::Rect(x0, y0, edge, edge)).setTo(128);
    }
    return to_bgr(canvas);
}

vf_shape_train_config train_config() {
    vf_shape_train_config config{};
    config.struct_size = sizeof(config);
    config.roi_x = 16;
    config.roi_y = 16;
    config.roi_width = kPattern;
    config.roi_height = kPattern;
    config.angle_min = -20;
    config.angle_max = 20;
    config.angle_step = 20;
    config.scale_min = 1;
    config.scale_max = 1;
    config.scale_step = 1;
    return config;
}

void same_candidates(const std::vector<vf_shape_candidate>& a,
                     const std::vector<vf_shape_candidate>& b, const std::string& what) {
    check(a.size() == b.size(), what + " 候选数漂移：" + std::to_string(a.size()) + " vs " +
                                    std::to_string(b.size()));
    for (size_t i = 0; i < a.size(); ++i) {
        check(a[i].score == b[i].score && a[i].center[0] == b[i].center[0] &&
                  a[i].center[1] == b[i].center[1] && a[i].angle == b[i].angle &&
                  a[i].scale == b[i].scale && a[i].pose_id == b[i].pose_id &&
                  std::equal(std::begin(a[i].corners), std::end(a[i].corners), b[i].corners) &&
                  std::equal(std::begin(a[i].bbox), std::end(a[i].bbox), b[i].bbox) &&
                  std::equal(std::begin(a[i].transform), std::end(a[i].transform), b[i].transform) &&
                  a[i].pose_ambiguous == b[i].pose_ambiguous,
              what + " 候选 " + std::to_string(i) + " 逐值不等");
    }
}

void group_pure_functions() {
    check(vfshape::rescue_collect_gate(70.0) == 42, "收集门槛 β·T=70 应为 42");
    check(vfshape::rescue_collect_gate(71.0) == 43, "收集门槛 β·T=71 应为 43");
    check(vfshape::rescue_collect_gate(50.0) == 30, "收集门槛 β·T=50 应为 30");
    const double adj_mid = vfshape::rescue_adjusted_score(60.0, 180, 256);
    check(std::abs(adj_mid - 60.0 * 256 / 180) < 1e-9, "重整公式漂移");
    // 分母下限随 β 联动：v=0 与 v=下限同托底，不硬编码 β 旧值（如 0.70 时代的 180）
    const int floor_denom = int(std::ceil(vfshape::kRescueBeta * 256.0));
    check(vfshape::rescue_adjusted_score(60.0, 0, 256) ==
              vfshape::rescue_adjusted_score(60.0, floor_denom, 256),
          "可见数为 0 应按分母下限托底");
    check(vfshape::rescue_adjusted_score(60.0, 256, 256) == 60.0, "全可见应为恒等");
    check(vfshape::rescue_adjusted_score(69.9, 100, 256) <= 100.0, "重整分应截断 100");
    for (int v = 0; v <= 256; v += 8) {
        double previous = -1.0;
        for (double raw = 0.0; raw < 70.0; raw += 3.5) {
            const double adj = vfshape::rescue_adjusted_score(raw, v, 256);
            check(adj <= 100.0, "重整上界失效 raw=" + std::to_string(raw) + " v=" +
                                    std::to_string(v));
            check(adj >= raw, "重整不应降分 raw=" + std::to_string(raw) + " v=" +
                                  std::to_string(v));
            check(adj >= previous, "重整应对 raw 单调不减");
            previous = adj;
        }
    }
}

void group_occupancy_estimator() {
    vfshape::Detector detector;
    cv::Mat scene(kFrameH, kFrameW, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::rectangle(scene, {int(kTargetX) - 28, int(kTargetY) - 28, 56, 56},
                  cv::Scalar(255, 255, 255), 2);
    const auto occupancy = vfshape::build_rescue_occupancy(scene, detector);
    check(occupancy.at<uchar>(int(kTargetY), int(kTargetX) - 28) != 0, "边缘像素应有占用");
    check(occupancy.at<uchar>(int(kTargetY), int(kTargetX)) == 0, "平坦内部不应有占用");

    line2Dup::Template templ;
    const int cy = int(kTargetY);
    const int left = int(kTargetX) - 28, right = int(kTargetX) + 28;
    for (int x : {left, left + 1, right - 1, right - 2})
        templ.features.push_back(line2Dup::Feature(x, cy, 0));
    constexpr int T0 = 4;
    constexpr int kOffset = T0 / 2 + (T0 % 2 - 1);
    check(vfshape::count_visible_features(occupancy, templ, double(kOffset), double(kOffset),
                                          T0, kOffset) == 4,
          "全可见合成帧计数应为 4");
    // 位移 +12px：越过边缘梯度尾（平滑+Sobel 约 ±4px）与 spread 前向支撑 T
    check(vfshape::count_visible_features(occupancy, templ, double(kOffset) + 12.0,
                                          double(kOffset), T0, kOffset) == 0,
          "错位候选计数应为 0");

    cv::Mat occluded = scene.clone();
    occluded(cv::Rect(int(kTargetX), 0, kFrameW - int(kTargetX), kFrameH))
        .setTo(cv::Scalar(128, 128, 128));
    const auto occluded_occ = vfshape::build_rescue_occupancy(occluded, detector);
    const int half = vfshape::count_visible_features(occluded_occ, templ, double(kOffset),
                                                     double(kOffset), T0, kOffset);
    check(half == 2, "半覆盖合成帧计数应为 2，实际 " + std::to_string(half));
}

void check_rescue_skipped(const vfshape::MatchTraceSnapshot& trace) {
    check(trace.rescued_candidates == 0 &&
          std::all_of(trace.window_collect_best.begin(), trace.window_collect_best.end(),
                      [](double score) { return score == 0; }),
          "an unexecuted rescue must not populate v2 diagnostics");
}

cv::Mat rotated_frame(double angle, cv::Point2d* rendered_center = nullptr) {
    const double rad = angle * CV_PI / 180.0;
    const int side = int(std::ceil(kPattern * (std::cos(rad) + std::sin(rad))));
    const cv::Mat rot = cv::getRotationMatrix2D({kPattern / 2.f, kPattern / 2.f}, angle, 1);
    cv::Mat patch;
    cv::warpAffine(pattern(), patch, rot, cv::Size(side, side));
    cv::Mat canvas(kFrameH, kFrameW, CV_8UC1, cv::Scalar(0));
    const cv::Point origin(int(kTargetX - side / 2.), int(kTargetY - side / 2.));
    if (rendered_center) *rendered_center = {origin.x + kPattern / 2., origin.y + kPattern / 2.};
    patch.copyTo(canvas(cv::Rect(origin.x, origin.y, side, side)));
    return to_bgr(canvas);
}

std::shared_ptr<vfshape::TwoStageModel> two_stage_fixture() {
    cv::Mat reference(128, 128, CV_8UC1, cv::Scalar(0));
    pattern().copyTo(reference(cv::Rect(16, 16, kPattern, kPattern)));
    const cv::Mat bgr = to_bgr(reference);
    auto config = train_config();
    config.angle_min = config.angle_max = 0;
    auto model = vfshape::train_two_stage(bgr, config);
    model->fines.front() = vfshape::train(bgr, train_config());
    model->b7 = {};  // Isolate discrete decisions; the DLL differential covers B7 geometry.
    return model;
}

void check_overlap(vfshape::TwoStageModel& model, double rotation, uint32_t ambiguity) {
    constexpr double threshold = 70.0;
    cv::Point2d rendered_center{};
    const cv::Mat image = rotated_frame(rotation, &rendered_center);
    auto& fine = *model.fines.front();
    const auto frame = vfshape::prepare_frame(image, &fine.detector);
    auto legacy = vfshape::match_prepared(fine, frame, 59.5);
    legacy.erase(std::remove_if(legacy.begin(), legacy.end(), [threshold](const auto& c) {
        return c.score < threshold;
    }), legacy.end());
    check(legacy.size() == 1 && legacy.front().pose_ambiguous == ambiguity,
          "overlap fixture must have one legacy representative with the expected ambiguity");
    const auto hits = fine.detector.matchPrepared(frame.features, 49.f, {"shape"});
    std::ostringstream detail;
    detail << "fixture did not expose a competing pose: rotation=" << rotation
           << " collect_gate=49 threshold=" << threshold << " expected_other_score="
           << (ambiguity ? ">=threshold" : "<threshold") << " legacy_pose=" << legacy.front().pose_id
           << " legacy_score=" << legacy.front().score << " hits=" << hits.size()
           << " rendered_center=(" << rendered_center.x << ',' << rendered_center.y << ')';
    bool has_competing_pose = false;
    for (const auto& hit : hits) {
        const auto pose = vfshape::collect_hit_objects(fine, {hit}, frame);
        detail << " [pose=" << hit.template_id << " raw=" << hit.similarity;
        // 判距基准=实际渲染中心：扩画布后名义中心不再落在目标上
        const double distance = pose.empty() ? std::numeric_limits<double>::infinity() :
            std::hypot(pose.front().center[0] - rendered_center.x,
                       pose.front().center[1] - rendered_center.y);
        detail << " distance_to_rendered_center=" << distance << ']';
        if (hit.template_id != legacy.front().pose_id && distance <= 8.0 &&
            (ambiguity ? hit.similarity >= threshold : hit.similarity < threshold)) {
            has_competing_pose = true;
            break;
        }
    }
    check(has_competing_pose, detail.str());
    vfshape::MatchTraceSnapshot trace;
    const auto results = vfshape::match_two_stage(model, image, threshold, 2, &trace);
    check(trace.windows_searched > 0, "fixture did not search a fine window");
    same_candidates(legacy, results, "same target must retain its legacy decision");
    check_rescue_skipped(trace);
}

void group_overlap_and_ambiguity() {
    auto model = two_stage_fixture();
    // Use the same coordinate origin for the single-stage oracle and two-stage result.
    model->groups.front().bias_x = model->groups.front().bias_y = 0;
    check_overlap(*model, 4.0, 0);
    check_overlap(*model, 11.0, 1);

    const cv::Mat image = rotated_frame(11.0);
    auto& fine = *model->fines.front();
    const auto prepared = vfshape::prepare_frame(image, &fine.detector);
    const auto legacy = vfshape::match_prepared(fine, prepared, 59.5);
    check(legacy.size() == 1 && legacy.front().pose_ambiguous && legacy.front().score < 99.,
          "ambiguous miss fixture must have an ambiguous legacy representative");
    const double threshold = legacy.front().score + 1.;
    vfshape::MatchTraceSnapshot trace;
    const auto results = vfshape::match_two_stage(*model, image, threshold, 2, &trace);
    check(results.empty() && trace.window_best[0] == legacy.front().score,
          "the ambiguous legacy representative must stay below threshold");
    check_rescue_skipped(trace);
}

void group_geometry_rejection() {
    auto model = two_stage_fixture();
    const cv::Mat image = frame_bgr();
    vfshape::MatchTraceSnapshot trace;
    check(!vfshape::match_two_stage(*model, image, 70., 2, &trace).empty(),
          "geometry fixture must match before shifting poses");
    check_rescue_skipped(trace);
    auto& fine = *model->fines.front();
    for (auto& pose : fine.poses) pose.transform[2] -= 10000.;
    const auto prepared = vfshape::prepare_frame(image, &fine.detector);
    check(!fine.detector.matchPrepared(prepared.features, 59.5f, {"shape"}).empty(),
          "geometry fixture must produce legacy raw hits");
    check(vfshape::match_two_stage(*model, image, 70., 2, &trace).empty() &&
          trace.windows_searched > 0 && trace.window_best[0] == 0.,
          "raw hits rejected by legacy geometry must not turn into rescue misses");
    check_rescue_skipped(trace);
    const cv::Mat blank(kFrameH, kFrameW, CV_8UC3, cv::Scalar(40, 40, 40));
    check(vfshape::match_two_stage(*model, blank, 70., 2, &trace).empty() && trace.windows_searched == 0,
          "blank frame must not widen the coarse search");
    check_rescue_skipped(trace);
}

template <typename Call>
void check_rejected(vf_shape_error_code expected, Call&& call) {
    const auto ticket = vfshape::begin_error_scope();
    try {
        call();
    } catch (const std::invalid_argument&) {
        check(vfshape::g_last_error.generation == ticket && vfshape::g_last_error.code == expected,
              "the current call did not produce the expected error code");
        return;
    }
    check(false, "invalid request was returned as an ordinary match result");
}

void group_rejections() {
    auto model = two_stage_fixture();
    const cv::Mat image = frame_bgr();
    check_rejected(VF_SHAPE_INVALID_IMAGE, [&] {
        vfshape::checked_image(image.data, image.total() * image.elemSize() - 1,
                               image.cols, image.rows, image.step[0]);
    });
    check_rejected(VF_SHAPE_THRESHOLD_INVALID, [&] {
        vfshape::match_two_stage(*model, image, std::numeric_limits<double>::quiet_NaN(), 2);
    });
    vfshape::MatchTraceSnapshot trace;
    check(!vfshape::match_two_stage(*model, image, 70., 2, &trace).empty(),
          "budget fixture did not match");
    check_rescue_skipped(trace);
    const auto prior = trace;
    // Inflate only the budget accounting; the real detector must reject before reading a pose.
    model->fines.front()->poses.resize(100000);
    check_rejected(VF_SHAPE_SEARCH_BUDGET, [&] {
        vfshape::match_two_stage(*model, image, 70., 2, &trace);
    });
    check(trace.final_candidates == prior.final_candidates && trace.window_best == prior.window_best,
          "budget rejection must not overwrite the last successful trace");
    check_rescue_skipped(trace);
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

struct AuditModel {
    std::shared_ptr<vfshape::TwoStageModel> two_stage;
    vfshape::PreparedFrame prepare(const cv::Mat& frame) const {
        return vfshape::prepare_frame(frame, &two_stage->fines.front()->detector);
    }
};

AuditModel train_audit_model(const std::string& ref_path) {
    const cv::Mat ref = cv::imread(ref_path, cv::IMREAD_COLOR);
    check(!ref.empty(), "参考帧读取失败：" + ref_path);
    const cv::Mat patch = ref(cv::Rect(800, 380, 320, 320)).clone();
    vf_shape_train_config config{};
    config.struct_size = sizeof(config);
    config.roi_x = 32;
    config.roi_y = 32;
    config.roi_width = 256;
    config.roi_height = 256;
    config.angle_min = -85;
    config.angle_max = 85;
    config.angle_step = 5;
    config.scale_min = 1.0;
    config.scale_max = 1.0;
    config.scale_step = 0.05;
    return {vfshape::train_two_stage(patch, config)};
}

// 逐案在真值邻域取最高原始分命中（低门槛保底收集），输出 v/raw/adj 测量行
int run_v_audit(const std::string& ref_path, const std::string& manifest_path) {
    const AuditModel audit = train_audit_model(ref_path);
    const auto& detector = audit.two_stage->fines.front()->detector;
    const int T0 = detector.getT(0);
    const int offset = T0 / 2 + (T0 % 2 - 1);
    std::ifstream manifest(manifest_path);
    check(manifest.good(), "清单读取失败：" + manifest_path);
    std::string line;
    std::getline(manifest, line);
    std::cout << "id,ratio,anchor,seed,raw,v,nf,v_ratio,adj,rescued\n";
    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        check(fields.size() >= 5, "清单行字段不足：" + line);
        const cv::Mat frame = cv::imread(fields[4], cv::IMREAD_COLOR);
        check(!frame.empty(), "帧读取失败：" + fields[4]);
        const double cx = std::stod(fields[5]), cy = std::stod(fields[6]);
        const auto prepared = audit.prepare(frame);
        double best_raw = 0.0;
        float best_x = 0.f, best_y = 0.f;
        size_t best_fine = 0;
        int best_id = -1;
        for (size_t gi = 0; gi < audit.two_stage->fines.size(); ++gi) {
            const auto& fine = *audit.two_stage->fines[gi];
            for (const auto& hit : fine.detector.matchPrepared(prepared.features, 20.f,
                                                               {"shape"})) {
                // 命中中心换算与 matcher.cpp candidate() 同式：命中位置是模板
                // 裁剪原点，须加位姿变换下的 ROI 中心偏移再对真值判邻域
                const auto& pose = fine.poses[size_t(hit.template_id)];
                const auto& templ = fine.detector.getTemplates("shape", hit.template_id).front();
                const double center_x = pose.transform[0] * 160.0 + pose.transform[1] * 160.0 +
                                        pose.transform[2] + hit.x - 32.0 - templ.tl_x;
                const double center_y = pose.transform[3] * 160.0 + pose.transform[4] * 160.0 +
                                        pose.transform[5] + hit.y - 32.0 - templ.tl_y;
                if (std::hypot(center_x - cx, center_y - cy) > 30.0) continue;
                if (double(hit.similarity) > best_raw) {
                    best_raw = double(hit.similarity);
                    best_x = hit.x;
                    best_y = hit.y;
                    best_fine = gi;
                    best_id = hit.template_id;
                }
            }
        }
        if (best_id < 0) {
            std::cout << fields[0] << "," << fields[1] << "," << fields[2] << "," << fields[3]
                      << ",0,-1,0,0,0,0\n";
            continue;
        }
        const auto& fine = *audit.two_stage->fines[best_fine];
        const auto occupancy = vfshape::build_rescue_occupancy(vfshape::pad_for_detector(frame),
                                                               fine.detector);
        const auto& templ = fine.detector.getTemplates("shape", best_id).front();
        const int nf = int(templ.features.size());
        const int v = vfshape::count_visible_features(occupancy, templ, best_x, best_y, T0,
                                                      offset);
        const double adj = vfshape::rescue_adjusted_score(best_raw, v, nf);
        std::cout << fields[0] << "," << fields[1] << "," << fields[2] << "," << fields[3]
                  << "," << best_raw << "," << v << "," << nf << "," << double(v) / nf << ","
                  << adj << "," << (best_raw < 70.0 && adj >= 70.0 ? 1 : 0) << "\n";
    }
    return 0;
}

// 负样本假峰采样：中心 1024² 裁片按编译期收集门槛（=β·T，T=70）全模板枚举
int run_fp_probe(const std::string& ref_path, const std::string& manifest_path) {
    const AuditModel audit = train_audit_model(ref_path);
    std::ifstream manifest(manifest_path);
    check(manifest.good(), "清单读取失败：" + manifest_path);
    std::string line;
    std::getline(manifest, line);
    std::cout << "id,pose_angle,x,y,raw\n";
    while (std::getline(manifest, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        check(fields.size() >= 2, "清单行字段不足：" + line);
        const cv::Mat frame = cv::imread(fields[1], cv::IMREAD_COLOR);
        check(!frame.empty(), "帧读取失败：" + fields[1]);
        const int crop_w = std::min(1024, frame.cols), crop_h = std::min(1024, frame.rows);
        const cv::Mat crop =
            frame(cv::Rect((frame.cols - crop_w) / 2, (frame.rows - crop_h) / 2, crop_w,
                           crop_h));
        try {
            const auto prepared = audit.prepare(crop);
            for (const auto& fine : audit.two_stage->fines) {
                for (const auto& hit : fine->detector.matchPrepared(
                         prepared.features, float(vfshape::rescue_collect_gate(70.0)),
                         {"shape"})) {
                    const auto& pose = fine->poses[size_t(hit.template_id)];
                    std::cout << fields[0] << "," << pose.angle << "," << hit.x << "," << hit.y
                              << "," << hit.similarity << "\n";
                }
            }
        } catch (const std::exception& error) {
            std::cout << fields[0] << ",error,0,0,0\n";
            std::cerr << "probe error " << fields[0] << ": " << error.what() << "\n";
        }
    }
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "--v-audit") {
            check(argc >= 4, "--v-audit 需要 <参考帧.png> <清单.csv>");
            return run_v_audit(argv[2], argv[3]);
        }
        if (mode == "--fp-probe") {
            check(argc >= 4, "--fp-probe 需要 <参考帧.png> <清单.csv>");
            return run_fp_probe(argv[2], argv[3]);
        }
        group_pure_functions();
        group_occupancy_estimator();
        group_overlap_and_ambiguity();
        group_geometry_rejection();
        group_rejections();
        std::cout << "occlusion rescue tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
