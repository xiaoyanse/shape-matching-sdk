#pragma once
#include "shape_matching.h"
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

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

class Detector final : public line2Dup::Detector {
public:
    // 参数化特征筛选：weak/strong=梯度幅度迟滞阈，features=每层
    // 特征数上限——虚警治理的建模期入口。默认值=历史行为（128/30/60）。
    explicit Detector(int features = feature_count, float weak_th = weak_threshold,
                      float strong_th = strong_threshold)
        : line2Dup::Detector(features, {4, 8}, weak_th, strong_th) {}
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
                            double wrap = 0);
void validate_config(const vf_shape_train_config& config, int width, int height);
std::shared_ptr<Model> train(const cv::Mat& image, const vf_shape_train_config& config,
                             const Model* previous_group = nullptr);
std::vector<uint8_t> encode(const Model& model);
std::shared_ptr<Model> decode(const uint8_t* bytes, uint64_t size);
std::vector<vf_shape_candidate> match(Model& model, const cv::Mat& image, double threshold);
PreparedFrame prepare_frame(const cv::Mat& image, const Detector* detector = nullptr);
void validate_search_budget(const Model& model, int width, int height);
std::vector<vf_shape_candidate> match_prepared(Model& model, const PreparedFrame& frame, double threshold);
}
