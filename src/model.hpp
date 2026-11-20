#pragma once
#include "shape_matching.h"
#include "error_code.hpp"
#include "line2Dup.h"
#include <array>
#include <mutex>
#include <stdexcept>

namespace vfshape {
constexpr uint64_t max_model_bytes = 32ULL * 1024 * 1024;
constexpr int max_edge = 4096;
constexpr int max_poses = 350;
constexpr int feature_count = 128;
constexpr int max_features_per_level = 512;
constexpr float weak_threshold = 30.0f;
constexpr float strong_threshold = 60.0f;
// 上限依据：vendor 3×3 Sobel 经 7×7 高斯平滑（凸组合，值域仍 0..255）单方向
// 响应 ≤ 4×255=1020；阈值在平方域比较，幅度范数上界 1020√2≈1442，取保守的单方向口径。
constexpr double kMaxStrongThreshold = 1020.;

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

class Detector final : public line2Dup::Detector {
public:
    // 参数化特征筛选：weak/strong=梯度幅度迟滞阈，features=每层
    // 特征数上限——虚警治理的建模期入口。默认值=历史行为（128/30/60）。
    explicit Detector(int features = feature_count, float weak_th = weak_threshold,
                      float strong_th = strong_threshold)
        : line2Dup::Detector(features, {4, 8}, weak_th, strong_th),
          tuned_features(features), tuned_weak(weak_th), tuned_strong(strong_th) {}
    // 生效参数快照：编码导出（模型往返参数稳定）与批量帧分组（同参共帧）的
    // 唯一取参来源——config 三字段为零值默认语义，此处始终是实际生效值。
    int tuned_features;
    float tuned_weak, tuned_strong;
    void append(std::vector<line2Dup::Template> pyramid) {
        class_templates["shape"].push_back(std::move(pyramid));
    }
};

struct Pose {
    double angle = 0, scale = 1;
    int canvas_width = 0, canvas_height = 0;
    std::array<double, 6> transform{};
};

struct Model {
    // 参数化：模型与其后的帧准备必须同参（modality 参数同时控制
    // 帧量化与模板特征选择，跨参匹配会静默错乱）。
    explicit Model(int features = feature_count, float weak_th = weak_threshold,
                   float strong_th = strong_threshold)
        : detector(features, weak_th, strong_th) {}
    Detector detector;
    vf_shape_train_config config{};
    int width = 0, height = 0;
    std::vector<Pose> poses;

    // 按位姿排列（外层 scale 内层 angle）推导相邻角度档映射——
    // 同 scale 组内 id±1 为相邻角度，组首尾无邻居（-1）。训练后与解码重建后
    // 各调用一次回填 detector.pose_adjacency。
    void build_pose_adjacency();
    std::mutex mutex;
};

struct PreparedFrame {
    line2Dup::Detector::PreparedFrame features;
    int width = 0, height = 0;
    int padded_width = 0, padded_height = 0;
    static constexpr int padding = 32;
};

cv::Mat checked_image(const uint8_t* image, uint64_t size, int width, int height,
                      uint64_t stride);
std::vector<double> samples(double low, double high, double step, double min, double max,
                            vf_shape_error_code range_code, double wrap = 0);
// 配置域检查（结构/Detector 三参/ROI/角度缩放范围），不含预算判定
void validate_config_domain(const vf_shape_train_config& config, int width, int height);
// 预算核算纯函数（域合法前提；与 validate_config 拒绝顺序同式同序）
vf_shape_budget_report account_training(const vf_shape_train_config& config,
                                        int width, int height);
// 预算查询：域检查 + 核算（不判超限失败，超限进 report.over_budget）
vf_shape_budget_report estimate_training(const vf_shape_train_config& config,
                                         int width, int height);
void validate_config(const vf_shape_train_config& config, int width, int height);
struct DetectorTuning { int features; float weak, strong; };
DetectorTuning validate_detector_params(int features, double weak, double strong);
// 持久化三参严格校验（零=损坏拒绝，防 float 下溢洗白链）——容器/codec 加载端用
DetectorTuning validate_persisted_detector_params(int features, double weak, double strong);
std::shared_ptr<Model> train(const cv::Mat& image, const vf_shape_train_config& config,
                             const Model* previous_group = nullptr,
                             const struct PolygonShape* polygon = nullptr);
std::vector<uint8_t> encode(const Model& model);
std::shared_ptr<Model> decode(const uint8_t* bytes, uint64_t size);
std::vector<vf_shape_candidate> match(Model& model, const cv::Mat& image, double threshold);
PreparedFrame prepare_frame(const cv::Mat& image, const Detector* detector = nullptr);
void validate_search_budget(const Model& model, int width, int height);
std::vector<vf_shape_candidate> match_prepared(Model& model, const PreparedFrame& frame, double threshold);
// 帧填充：零边距补到 16 对齐（prepare_frame 与遮挡救分占用图共用同参重填充）
cv::Mat pad_for_detector(const cv::Mat& image);
// 命中→候选聚类（match_prepared 与遮挡救分路径共用的单一实现）
std::vector<vf_shape_candidate> collect_hit_objects(Model& model,
                                                    const std::vector<line2Dup::Match>& hits,
                                                    const PreparedFrame& frame);
}
