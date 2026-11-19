#pragma once
// 多边形模板能力：几何校验、训练掩码构造与轮廓建议。校验/建议/容器 poly 段
// 加载共用同一几何口径（INVALID_POLYGON 细分类型见公共头注释）。

#include "shape_matching.h"

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vfshape {

// 训练态多边形：绕向已归一（有向面积为正），坐标为传入训练图像素坐标系
struct PolygonShape {
    std::vector<cv::Point2d> outline;
    std::vector<std::vector<cv::Point2d>> holes;
    cv::Rect bounds{};  // 外接矩形（整数像素，训练 ROI 等价物）
};

// 参数结构解析+几何校验+绕向归一；失败按 INVALID_POLYGON 细分类型抛出
PolygonShape validate_polygon(const vf_shape_polygon_train_params& params,
                              int width, int height);

// 已解析顶点集合的几何校验（容器 poly 段加载复用同口径，防篡改洗白）
PolygonShape validate_polygon_vertices(std::vector<cv::Point2d> outline,
                                       std::vector<std::vector<cv::Point2d>> holes,
                                       int width, int height);

// 训练掩码：外环填充 255、洞填充 0；顶点按 (p-origin)*scale+padding 映射后光栅化
inline void fill_polygon_mask(cv::Mat& mask, const PolygonShape& polygon,
                              double origin_x, double origin_y, double scale, int padding) {
    auto to_canvas = [&](cv::Point2d p) {
        return cv::Point(cvRound((p.x - origin_x) * scale) + padding,
                         cvRound((p.y - origin_y) * scale) + padding);
    };
    std::vector<cv::Point> points;
    points.reserve(polygon.outline.size());
    for (const auto& v : polygon.outline) points.push_back(to_canvas(v));
    cv::fillPoly(mask, points, 255);
    for (const auto& hole : polygon.holes) {
        points.clear();
        for (const auto& v : hole) points.push_back(to_canvas(v));
        cv::fillPoly(mask, points, 0);
    }
}

// 粗层缩放：顶点除 k 后取整包界并截到缩图内（fillPoly 自裁剪越界部分）
PolygonShape scale_polygon(const PolygonShape& polygon, double factor,
                           int max_width, int max_height);

// 训练参数结构 → 矩形等价训练配置（ROI=外接矩形，预算/位姿映射全复用矩形路径）
vf_shape_train_config config_from_polygon(const vf_shape_polygon_train_params& params,
                                          const PolygonShape& shape);

// 持久化扁平形态（容器 poly 键）与训练态互转
std::vector<double> flatten_vertices(const std::vector<cv::Point2d>& points);
std::vector<cv::Point2d> unflatten_vertices(const std::vector<double>& flat);
std::vector<std::vector<cv::Point2d>> unflatten_holes(
    const std::vector<std::vector<double>>& flats);

// 轮廓建议（vf_shape_suggest_contours 实现）；返回建议条数
uint32_t suggest_contours(const cv::Mat& image, const int32_t* roi_hint,
                          uint32_t max_suggestions, uint32_t max_vertices_per,
                          vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts);

}
