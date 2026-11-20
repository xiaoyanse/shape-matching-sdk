#include "polygon.hpp"

#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>

namespace vfshape {
namespace {

constexpr double kDegenerateArea = 1e-9;  // 外环有向面积近零判退化
constexpr double kMinHoleArea = 64.0;     // 洞面积下限 px²
constexpr uint32_t kMaxVertices = 64;
constexpr uint32_t kMaxHoles = 8;
constexpr double kCrossEpsilon = 1e-9;

constexpr double kVertexCountLimit = 1;
constexpr double kVertexDomain = 2;
constexpr double kDegenerate = 3;
constexpr double kSelfIntersect = 4;
constexpr double kHoleCountLimit = 5;
constexpr double kHoleGeometry = 6;
constexpr double kHoleArea = 7;
constexpr double kHoleOutside = 8;
constexpr double kBboxContract = 9;

double signed_area(const std::vector<cv::Point2d>& polygon) {
    double sum = 0;
    for (size_t i = 0; i < polygon.size(); ++i) {
        const cv::Point2d& a = polygon[i];
        const cv::Point2d& b = polygon[(i + 1) % polygon.size()];
        sum += a.x * b.y - b.x * a.y;
    }
    return sum / 2.0;
}

double cross(double ax, double ay, double bx, double by) { return ax * by - ay * bx; }

// 严格线段相交（数值过零；端点邻近情形由落边分支兜底）
bool segments_cross(cv::Point2d a1, cv::Point2d a2, cv::Point2d b1, cv::Point2d b2) {
    const double d1 = cross(b2.x - b1.x, b2.y - b1.y, a1.x - b1.x, a1.y - b1.y);
    const double d2 = cross(b2.x - b1.x, b2.y - b1.y, a2.x - b1.x, a2.y - b1.y);
    const double d3 = cross(a2.x - a1.x, a2.y - a1.y, b1.x - a1.x, b1.y - a1.y);
    const double d4 = cross(a2.x - a1.x, a2.y - a1.y, b2.x - a1.x, b2.y - a1.y);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0)) &&
           std::abs(d1) > kCrossEpsilon && std::abs(d2) > kCrossEpsilon &&
           std::abs(d3) > kCrossEpsilon && std::abs(d4) > kCrossEpsilon;
}

bool point_on_segment(cv::Point2d p, cv::Point2d a, cv::Point2d b) {
    return std::abs(cross(b.x - a.x, b.y - a.y, p.x - a.x, p.y - a.y)) <= kCrossEpsilon &&
           std::min(a.x, b.x) - kCrossEpsilon <= p.x &&
           p.x <= std::max(a.x, b.x) + kCrossEpsilon &&
           std::min(a.y, b.y) - kCrossEpsilon <= p.y &&
           p.y <= std::max(a.y, b.y) + kCrossEpsilon;
}

enum class ContourFault { None, Degenerate, SelfIntersect };

struct ContourCheck {
    ContourFault fault = ContourFault::None;
    double area = 0;
    size_t edge_a = 0, edge_b = 0;
};

// 退化（|有向面积| < area_limit）与自交/重入：非相邻边严格相交、端点落边、
// 重复顶点即判自交；相邻边共享端点是合法拓扑
ContourCheck check_contour(const std::vector<cv::Point2d>& polygon, double area_limit) {
    ContourCheck result;
    result.area = std::abs(signed_area(polygon));
    if (result.area < area_limit) {
        result.fault = ContourFault::Degenerate;
        return result;
    }
    const size_t n = polygon.size();
    for (size_t i = 0; i < n; ++i) {
        if (polygon[i] == polygon[(i + 1) % n]) {
            result.fault = ContourFault::SelfIntersect;
            result.edge_a = i;
            result.edge_b = (i + 1) % n;
            return result;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const cv::Point2d a1 = polygon[i], a2 = polygon[(i + 1) % n];
        for (size_t j = i + 1; j < n; ++j) {
            if (j == i + 1 || (i == 0 && j == n - 1)) continue;
            const cv::Point2d b1 = polygon[j], b2 = polygon[(j + 1) % n];
            if (segments_cross(a1, a2, b1, b2) || point_on_segment(a1, b1, b2) ||
                point_on_segment(a2, b1, b2) || point_on_segment(b1, a1, a2) ||
                point_on_segment(b2, a1, a2)) {
                result.fault = ContourFault::SelfIntersect;
                result.edge_a = i;
                result.edge_b = j;
                return result;
            }
        }
    }
    return result;
}

// 奇偶射线（PNPOLY）：配合"洞与外环边界无任何接触"前置条件判定洞在内
bool point_inside(cv::Point2d p, const std::vector<cv::Point2d>& polygon) {
    bool inside = false;
    const size_t n = polygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const cv::Point2d a = polygon[j];
        const cv::Point2d b = polygon[i];
        if ((a.y > p.y) != (b.y > p.y)) {
            const double x_at = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
            if (p.x < x_at) inside = !inside;
        }
    }
    return inside;
}

void normalize_orientation(std::vector<cv::Point2d>& polygon) {
    if (signed_area(polygon) < 0) std::reverse(polygon.begin(), polygon.end());
}

cv::Rect bounding_box(const std::vector<cv::Point2d>& polygon) {
    double min_x = std::numeric_limits<double>::max();
    double min_y = min_x, max_x = -min_x, max_y = -min_x;
    for (const auto& v : polygon) {
        min_x = std::min(min_x, v.x);
        min_y = std::min(min_y, v.y);
        max_x = std::max(max_x, v.x);
        max_y = std::max(max_y, v.y);
    }
    const int x0 = int(std::floor(min_x)), y0 = int(std::floor(min_y));
    const int x1 = int(std::ceil(max_x)), y1 = int(std::ceil(max_y));
    return {x0, y0, x1 - x0, y1 - y0};
}

bool contours_touch(const std::vector<cv::Point2d>& hole,
                    const std::vector<cv::Point2d>& outline) {
    for (size_t i = 0; i < hole.size(); ++i) {
        const cv::Point2d h1 = hole[i], h2 = hole[(i + 1) % hole.size()];
        for (size_t j = 0; j < outline.size(); ++j) {
            const cv::Point2d o1 = outline[j], o2 = outline[(j + 1) % outline.size()];
            if (segments_cross(h1, h2, o1, o2) || point_on_segment(h1, o1, o2) ||
                point_on_segment(h2, o1, o2) || point_on_segment(o1, h1, h2) ||
                point_on_segment(o2, h1, h2))
                return true;
        }
    }
    return false;
}

}  // namespace

PolygonShape validate_polygon_vertices(std::vector<cv::Point2d> outline,
                                       std::vector<std::vector<cv::Point2d>> holes,
                                       int width, int height) {
    require_coded(outline.size() >= 3 && outline.size() <= kMaxVertices,
                  VF_SHAPE_INVALID_POLYGON, "polygon outline vertex count out of 3..64",
                  kVertexCountLimit, double(outline.size()), double(kMaxVertices));
    require_coded(holes.size() <= kMaxHoles, VF_SHAPE_INVALID_POLYGON,
                  "polygon hole count exceeds 8", kHoleCountLimit, double(holes.size()));
    double index = 0;
    auto check_domain = [&](const std::vector<cv::Point2d>& polygon) {
        for (const auto& v : polygon) {
            require_coded(std::isfinite(v.x) && std::isfinite(v.y) && v.x >= 0 &&
                              v.x <= width && v.y >= 0 && v.y <= height,
                          VF_SHAPE_INVALID_POLYGON,
                          "polygon vertex outside reference image domain", kVertexDomain,
                          index);
            ++index;
        }
    };
    check_domain(outline);
    for (size_t h = 0; h < holes.size(); ++h) {
        require_coded(holes[h].size() >= 3 && holes[h].size() <= kMaxVertices,
                      VF_SHAPE_INVALID_POLYGON, "polygon hole vertex count out of 3..64",
                      kHoleGeometry, double(h), kVertexCountLimit);
        check_domain(holes[h]);
    }

    const ContourCheck outline_check = check_contour(outline, kDegenerateArea);
    if (outline_check.fault == ContourFault::Degenerate)
        fail(VF_SHAPE_INVALID_POLYGON, {kDegenerate, outline_check.area},
             "polygon outline degenerate");
    if (outline_check.fault == ContourFault::SelfIntersect)
        fail(VF_SHAPE_INVALID_POLYGON,
             {kSelfIntersect, double(outline_check.edge_a), double(outline_check.edge_b)},
             "polygon outline self-intersects");

    for (size_t h = 0; h < holes.size(); ++h) {
        auto& hole = holes[h];
        const ContourCheck check = check_contour(hole, kMinHoleArea);
        if (check.fault == ContourFault::Degenerate)
            fail(VF_SHAPE_INVALID_POLYGON, {kHoleArea, double(h), check.area},
                 "polygon hole area below 64px2");
        if (check.fault == ContourFault::SelfIntersect)
            fail(VF_SHAPE_INVALID_POLYGON, {kHoleGeometry, double(h), kSelfIntersect},
                 "polygon hole self-intersects");
        normalize_orientation(hole);
    }
    normalize_orientation(outline);

    for (size_t h = 0; h < holes.size(); ++h) {
        require_coded(!contours_touch(holes[h], outline) &&
                          point_inside(holes[h].front(), outline),
                      VF_SHAPE_INVALID_POLYGON, "polygon hole not fully inside outline",
                      kHoleOutside, double(h));
    }

    const cv::Rect bounds = bounding_box(outline);
    require_coded(bounds.width >= 24 && bounds.height >= 24 && bounds.width <= max_edge &&
                      bounds.height <= max_edge,
                  VF_SHAPE_INVALID_POLYGON,
                  "polygon bounding box violates 24..4096 contract", kBboxContract,
                  double(bounds.width), double(bounds.height));
    return {std::move(outline), std::move(holes), bounds};
}

PolygonShape validate_polygon(const vf_shape_polygon_train_params& params,
                              int width, int height) {
    require(params.struct_size == sizeof(params),
            "invalid polygon training ABI structure size");
    require(params.outline.vertices || params.outline.count == 0,
            "missing polygon outline vertices");
    require(params.hole_count == 0 || params.holes, "missing polygon hole array");
    auto to_points = [](const vf_shape_vertex* vertices, uint32_t count) {
        std::vector<cv::Point2d> points;
        points.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            points.push_back({vertices[i].x, vertices[i].y});
        return points;
    };
    std::vector<cv::Point2d> outline = to_points(params.outline.vertices, params.outline.count);
    std::vector<std::vector<cv::Point2d>> holes;
    holes.reserve(params.hole_count);
    for (uint32_t i = 0; i < params.hole_count; ++i) {
        const vf_shape_polygon& hole = params.holes[i];
        require(hole.vertices || hole.count == 0, "missing polygon hole vertices");
        holes.push_back(to_points(hole.vertices, hole.count));
    }
    return validate_polygon_vertices(std::move(outline), std::move(holes), width, height);
}

double polygon_mask_area(const PolygonShape& polygon) {
    double area = std::abs(signed_area(polygon.outline));
    for (const auto& hole : polygon.holes) area -= std::abs(signed_area(hole));
    return std::max(0.0, area);
}

vf_shape_budget_report estimate_training_ex(const vf_shape_train_config& config, int width,
                                            int height,
                                            const vf_shape_polygon_train_params* params) {
    if (!params) return estimate_training(config, width, height);
    // 多边形路径与 vf_shape_train_polygon 同参语义：params 全量字段为准，
    // config 在该路径不参与核算（合同见公共头 _ex 注释）
    require(params->struct_size == sizeof(*params),
            "invalid polygon training ABI structure size");
    const PolygonShape shape = validate_polygon(*params, width, height);
    vf_shape_budget_report report = account_training(config_from_polygon(*params, shape),
                                                     width, height);
    report.formula_version = 2;
    const double bbox_area = double(shape.bounds.width) * double(shape.bounds.height);
    report.polygon_fill_ratio = polygon_mask_area(shape) / bbox_area;
    return report;
}

PolygonShape scale_polygon(const PolygonShape& polygon, double factor,
                           int max_width, int max_height) {
    PolygonShape out;
    auto scale_points = [factor](const std::vector<cv::Point2d>& source) {
        std::vector<cv::Point2d> points;
        points.reserve(source.size());
        for (const auto& v : source) points.push_back({v.x * factor, v.y * factor});
        return points;
    };
    out.outline = scale_points(polygon.outline);
    out.holes.reserve(polygon.holes.size());
    for (const auto& hole : polygon.holes) out.holes.push_back(scale_points(hole));
    const cv::Rect scaled = bounding_box(out.outline);
    const int x0 = std::max(0, scaled.x), y0 = std::max(0, scaled.y);
    const int x1 = std::min(max_width, scaled.x + scaled.width);
    const int y1 = std::min(max_height, scaled.y + scaled.height);
    out.bounds = {x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0)};
    return out;
}

vf_shape_train_config config_from_polygon(const vf_shape_polygon_train_params& params,
                                          const PolygonShape& shape) {
    vf_shape_train_config c{};
    c.struct_size = sizeof(c);
    c.roi_x = shape.bounds.x;
    c.roi_y = shape.bounds.y;
    c.roi_width = shape.bounds.width;
    c.roi_height = shape.bounds.height;
    c.angle_min = params.angle_min;
    c.angle_max = params.angle_max;
    c.angle_step = params.angle_step;
    c.scale_min = params.scale_min;
    c.scale_max = params.scale_max;
    c.scale_step = params.scale_step;
    c.weak_th = params.weak_th;
    c.strong_th = params.strong_th;
    c.feature_count = params.feature_count;
    return c;
}

std::vector<double> flatten_vertices(const std::vector<cv::Point2d>& points) {
    std::vector<double> flat;
    flat.reserve(points.size() * 2);
    for (const auto& v : points) {
        flat.push_back(v.x);
        flat.push_back(v.y);
    }
    return flat;
}

std::vector<cv::Point2d> unflatten_vertices(const std::vector<double>& flat) {
    std::vector<cv::Point2d> points;
    points.reserve(flat.size() / 2);
    for (size_t i = 0; i + 1 < flat.size(); i += 2) points.push_back({flat[i], flat[i + 1]});
    return points;
}

std::vector<std::vector<cv::Point2d>> unflatten_holes(
    const std::vector<std::vector<double>>& flats) {
    std::vector<std::vector<cv::Point2d>> holes;
    holes.reserve(flats.size());
    for (const auto& flat : flats) holes.push_back(unflatten_vertices(flat));
    return holes;
}

uint32_t suggest_contours(const cv::Mat& image, const int32_t* roi_hint,
                          uint32_t max_suggestions, uint32_t max_vertices_per,
                          const vf_shape_suggest_params* params,
                          vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts) {
    require_coded(max_suggestions >= 1 && max_suggestions <= 256, VF_SHAPE_CAPACITY_INVALID,
                  "max_suggestions must be 1..256", double(max_suggestions), 1.0, 256.0);
    require_coded(max_vertices_per >= 3 && max_vertices_per <= kMaxVertices,
                  VF_SHAPE_CAPACITY_INVALID, "max_vertices_per must be 3..64",
                  double(max_vertices_per), 3.0, double(kMaxVertices));
    // Canny 阈值与训练 detector 三参同类约束（正且递增的边缘阈值族），同码拒绝
    constexpr double kDefaultCannyLow = 50.0, kDefaultCannyHigh = 150.0;
    constexpr double kMinEpsilonFactor = 0.001, kMaxEpsilonFactor = 0.2;
    double canny_low = kDefaultCannyLow, canny_high = kDefaultCannyHigh;
    double epsilon_factor = 0.02;
    if (params) {
        require(params->struct_size == sizeof(vf_shape_suggest_params),
                "invalid suggestion params ABI structure size");
        require_coded(std::isfinite(params->canny_low) && params->canny_low >= 0,
                      VF_SHAPE_DETECTOR_PARAMS, "suggest canny low threshold must be positive",
                      params->canny_low);
        canny_low = params->canny_low > 0 ? params->canny_low : canny_low;
        require_coded(std::isfinite(params->canny_high) && params->canny_high >= 0,
                      VF_SHAPE_DETECTOR_PARAMS, "suggest canny high threshold must be positive",
                      params->canny_high);
        canny_high = params->canny_high > 0 ? params->canny_high : canny_high;
        require_coded(canny_high >= canny_low, VF_SHAPE_DETECTOR_PARAMS,
                      "suggest canny high threshold must not fall below low",
                      canny_low, canny_high);
        require_coded(std::isfinite(params->epsilon_factor) &&
                          (params->epsilon_factor == 0 || (params->epsilon_factor >= kMinEpsilonFactor &&
                                                           params->epsilon_factor <= kMaxEpsilonFactor)),
                      VF_SHAPE_DETECTOR_PARAMS, "suggest epsilon factor out of 0.001..0.2",
                      params->epsilon_factor, kMinEpsilonFactor, kMaxEpsilonFactor);
        epsilon_factor = params->epsilon_factor > 0 ? params->epsilon_factor : epsilon_factor;
    }
    cv::Rect search(0, 0, image.cols, image.rows);
    if (roi_hint) {
        require_coded(roi_hint[0] >= 0 && roi_hint[1] >= 0 && roi_hint[2] >= 1 &&
                          roi_hint[3] >= 1 && roi_hint[0] + roi_hint[2] <= image.cols &&
                          roi_hint[1] + roi_hint[3] <= image.rows,
                      VF_SHAPE_INVALID_ROI, "invalid suggestion ROI hint",
                      double(roi_hint[2]), double(roi_hint[3]));
        search = cv::Rect(roi_hint[0], roi_hint[1], roi_hint[2], roi_hint[3]);
    }
    cv::Mat gray;
    cv::cvtColor(image(search), gray, cv::COLOR_BGR2GRAY);
    // 固定 3×3 高斯：闭合斜边阶梯栅格造成的 Canny 断边/回折（确定性不变）
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0);
    cv::Mat edges;
    cv::Canny(gray, edges, canny_low, canny_high);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    struct Suggestion {
        std::vector<cv::Point2d> vertices;
        double area = 0;
    };
    std::vector<Suggestion> accepted;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinHoleArea) continue;
        double epsilon = std::max(1.0, epsilon_factor * cv::arcLength(contour, true));
        std::vector<cv::Point> approx;
        bool fits = false;
        while (epsilon <= 4.0 * (image.cols + image.rows)) {
            cv::approxPolyDP(contour, approx, epsilon, true);
            if (approx.size() >= 3 && approx.size() <= size_t(max_vertices_per)) {
                fits = true;
                break;
            }
            epsilon *= 1.5;
        }
        if (!fits) continue;
        Suggestion suggestion;
        suggestion.area = area;
        suggestion.vertices.reserve(approx.size());
        for (const auto& p : approx)
            suggestion.vertices.push_back({double(p.x + search.x), double(p.y + search.y)});
        accepted.push_back(std::move(suggestion));
    }
    std::sort(accepted.begin(), accepted.end(),
              [](const Suggestion& a, const Suggestion& b) { return a.area > b.area; });
    if (accepted.size() > max_suggestions) accepted.resize(max_suggestions);
    uint32_t written = 0;
    for (uint32_t i = 0; i < accepted.size(); ++i) {
        out_vertex_counts[i] = uint32_t(accepted[i].vertices.size());
        for (const auto& v : accepted[i].vertices) {
            out_vertices[written].x = v.x;
            out_vertices[written].y = v.y;
            ++written;
        }
    }
    return uint32_t(accepted.size());
}
}
