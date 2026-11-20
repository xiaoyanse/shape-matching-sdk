#pragma once
// 遮挡救分（rescue-only 补分制）：精修后低于判定阈的候选按可见特征占比重整。
// adj = raw×nf/max(v, ceil(β·nf))；raw≥阈值候选零改动，重整后仍不足阈值者剔除。

#include "model.hpp"

#include <optional>
#include <vector>

namespace vfshape {

// 分母下限系数：重整上限 1/β≈1.67×，收集门槛=ceil(β·T) 单参数联动
// β=0.70→0.60（2026-11-21）：50% 档 7 案缺口实测 2/7<门 3/7，降 β 同步放宽
// 收集带与重整上限；每档变更须重跑假峰带采样并全套复验后才能定值
constexpr double kRescueBeta = 0.60;

int rescue_collect_gate(double threshold);

// 纯函数：重整分（截断 100）；visible 按分母下限托底
double rescue_adjusted_score(double raw, int visible, int nf);

// 场景 8 朝向占用图：标签图非零位的前向 T×T 扩展（vendor spread 同语义，
// 输入须为 pad_for_detector 产物，输出 CV_8U 0/1）。仅细层（level 0）口径。
cv::Mat build_rescue_occupancy(const cv::Mat& padded_bgr, const Detector& detector);

// 模板细层特征在匹配位置的可见计数：场景坐标 = lround(match)-offset + 特征坐标，
// 越界特征不计（与 vendor 局部精化的越界丢弃口径一致）
int count_visible_features(const cv::Mat& occupancy, const line2Dup::Template& templ,
                           double match_x, double match_y, int T, int offset);

// legacy_empty_gate distinguishes an actual collection miss from raw hits rejected by geometry.
std::vector<vf_shape_candidate> match_prepared_rescue(
    Model& model, const PreparedFrame& frame, const cv::Mat& window_bgr, double threshold,
    double* collect_best = nullptr, int* rescued = nullptr,
    std::optional<double> legacy_empty_gate = std::nullopt);

}
