#pragma once
// 两级匹配（粗到精+几何精修）。
//
// 结构：容器字节流（与 Python coarse_to_fine.py 的 VF2S3 格式字节级兼容，
// 互为可加载）解析为 TwoStageModel——fines 角度分组模型 + coarse 粗层模型 +
// 模板灰度图 + 组级锚点偏置 + B7 轮廓点/法线。匹配走：粗层 K 自适应缩图
// 定位 → 原图窗口精修（按粗层锁定角度选组）→ B7 连续几何精修（top1）。
//
// 单级向后兼容：非容器格式（旧资产）回退单级路径。

#include "model.hpp"

#include <memory>
#include <vector>

namespace vfshape {

struct MatchTraceSnapshot;  // matching_profile.hpp（诊断快照，指针透传）

struct GroupSpec {
    double lo = 0, hi = 0;
    double bias_x = 0, bias_y = 0;
};

struct B7Model {
    // 模板轮廓点（相对模板中心）+法线（归一化梯度方向）
    std::vector<double> px, py, nx, ny;
};
B7Model extract_b7_points(const cv::Mat& template_gray);

// 容器 poly 段资产（多边形训练持久化形态）：顶点已绕向归一，坐标=参考图系
struct PolygonAsset {
    std::vector<double> outline;             // x0,y0,x1,y1,...
    std::vector<std::vector<double>> holes;  // 每洞 x0,y0,...
};

struct TwoStageModel {
    std::vector<std::shared_ptr<Model>> fines;   // 角度分组精修模型
    std::shared_ptr<Model> coarse;               // 粗层模型
    std::vector<GroupSpec> groups;               // 组界+组级偏置
    int k = 8;                                   // 粗层缩放（训练期 ROI 自适应）
    cv::Mat tmpl_gray;                           // 训练 ROI 灰度图（B7 基准）
    B7Model b7;                                  // B7 轮廓点（惰性或训练期提取）
    vf_shape_train_config base_config{};         // 原始训练配置（供 B7 角度步长等）
    int ref_width = 0, ref_height = 0;           // 原始参考图尺寸；0=旧容器（元数据回退模板灰图尺寸）
    std::shared_ptr<PolygonAsset> polygon;       // 多边形资产；空=矩形资产
};

// 容器编解码（VF2S3 字节级兼容；返回 nullptr 表示非容器格式）
std::shared_ptr<TwoStageModel> decode_two_stage(const uint8_t* bytes, uint64_t size);
std::vector<uint8_t> encode_two_stage(const TwoStageModel& model);

// 两级打包训练（等价 Python train_packaged）：fines 分组 + coarse + 偏置 + 容器；
// polygon 非空时训练掩码为多边形（config ROI 须等于其外接矩形）
std::shared_ptr<TwoStageModel> train_two_stage(
    const cv::Mat& image_bgr, const vf_shape_train_config& config,
    const struct PolygonShape* polygon = nullptr);

// 两级匹配（等价 Python TwoStageMatcher.match，含 B7 top1 精修）；
// trace 非空时回填最近一次匹配的诊断快照（计数常开，毫秒仅计时构建非 0）
std::vector<vf_shape_candidate> match_two_stage(
    TwoStageModel& model, const cv::Mat& image_bgr, double threshold, int max_candidates,
    MatchTraceSnapshot* trace = nullptr);

// B7 连续几何精修（内部；也用于对拍测试）
struct B7Result {
    double x = 0, y = 0, angle = 0, scale = 1;
    int valid = 0;
};
B7Result b7_refine(const cv::Mat& gray, const B7Model& b7,
                   double cx, double cy, double angle_deg, double scale);
void apply_geometric_refinement(vf_shape_candidate& candidate, const B7Result& refined);

// Python 侧参数对齐（单一真相源在容器 meta；此处仅训练期使用）
constexpr double kCoarseAngleStep = 8.0;
constexpr int kFineAngleStepMilli = 500;  // 0.5°（毫度表示避免浮点比较）
constexpr double kB7SearchRadius = 4.0;
constexpr int kB7MinValidPts = 24;
constexpr double kB7EdgeGradMin = 30.0;
constexpr int kB7GnIters = 4;
constexpr int kB7OutlierSigmaX10 = 25;    // 2.5（十分之一表示）

}  // namespace vfshape
