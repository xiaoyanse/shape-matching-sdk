// 洞掩蔽取证（构造级）：直连 vfshape::train 多边形路径，比较同一轮廓"有洞 vs
// 无洞"的特征总数——洞覆盖强边缘结构时特征数必须严格下降（洞内像素不产特征
// 的实证，验收门 2）。特征数取 0 号姿态全金字塔层之和。
// 图像设计注意：绘制形状与掩码边界之间必须留黑边距——vendor extractTemplate
// 会腐蚀掩码 1px，贴边形状的边界梯度全被吃掉（贴边=无特征，非多边形路径问题）。
#include "model.hpp"
#include "polygon.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 外环 140×140（90..230），内部白色方框 100..219 与棋盘块，形状四周留黑边距
const cv::Rect kBounds(90, 90, 140, 140);

cv::Mat pattern_reference() {
    cv::Mat img(320, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(img, {100, 100}, {219, 219}, cv::Scalar(255, 255, 255), -1);
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            cv::rectangle(img, {136 + col * 12, 136 + row * 12, 8, 8},
                          cv::Scalar((row + col) % 2 ? 60 : 240, (row + col) % 2 ? 60 : 240,
                                     (row + col) % 2 ? 60 : 240), -1);
    return img;
}

vf_shape_train_config train_config() {
    vf_shape_train_config c{};
    c.struct_size = sizeof(c);
    c.roi_x = kBounds.x;
    c.roi_y = kBounds.y;
    c.roi_width = kBounds.width;
    c.roi_height = kBounds.height;
    c.angle_min = 0;
    c.angle_max = 0;
    c.angle_step = 1;
    c.scale_min = 1;
    c.scale_max = 1;
    c.scale_step = 1;
    return c;
}

size_t pose_feature_total(const vfshape::Model& model) {
    size_t total = 0;
    for (const auto& level : model.detector.getTemplates("shape", 0))
        total += level.features.size();
    return total;
}

vfshape::PolygonShape plain_polygon() {
    vfshape::PolygonShape shape;
    shape.outline = {{90, 90}, {230, 90}, {230, 230}, {90, 230}};
    shape.bounds = kBounds;
    return shape;
}

int add_template_probe(const char* tag, const cv::Mat& canvas, const cv::Mat& mask) {
    vfshape::Detector det;
    const int id = det.addTemplate(canvas, "shape", mask);
    std::cout << "probe[" << tag << "] id=" << id;
    if (id >= 0)
        for (const auto& t : det.getTemplates("shape", id))
            std::cout << " feats=" << t.features.size();
    std::cout << "\n";
    return id;
}

}  // namespace

int main() {
    try {
        const cv::Mat image = pattern_reference();
        const vf_shape_train_config config = train_config();

        // coarse 链路复算（K=4，bbox 240→60）：小图+缩放多边形+fillPoly 掩码
        {
            cv::Mat small;
            cv::resize(image, small, cv::Size(120, 120), 0, 0, cv::INTER_AREA);
            const vfshape::PolygonShape coarse =
                vfshape::scale_polygon(plain_polygon(), 0.25, small.cols, small.rows);
            std::cout << "probe coarse bounds=(" << coarse.bounds.x << "," << coarse.bounds.y
                      << "," << coarse.bounds.width << "," << coarse.bounds.height << ")\n";
            const cv::Mat ccrop = small(coarse.bounds);
            const int cpadding =
                int(std::ceil(std::max(ccrop.cols, ccrop.rows) * std::sqrt(2.) / 2.)) + 24;
            cv::Mat ccanvas(ccrop.rows + cpadding * 2, ccrop.cols + cpadding * 2, CV_8UC3,
                            cv::Scalar::all(0));
            ccrop.copyTo(ccanvas(cv::Rect(cpadding, cpadding, ccrop.cols, ccrop.rows)));
            cv::Mat cmask(ccanvas.size(), CV_8UC1, cv::Scalar(0));
            vfshape::fill_polygon_mask(cmask, coarse, coarse.bounds.x, coarse.bounds.y, 1.0,
                                       cpadding);
            std::cout << "probe coarse mask nonzero=" << cv::countNonZero(cmask) << "\n";
            add_template_probe("coarse-k4", ccanvas, cmask);
        }

        // 无洞（矩形路径基线）
        const auto rect_model = vfshape::train(image, config);
        const size_t rect_features = pose_feature_total(*rect_model);
        std::cout << "rect trained\n";

        // 同轮廓多边形、无洞（多边形路径基线）
        vfshape::PolygonShape plain = plain_polygon();
        const auto poly_model = vfshape::train(image, config, nullptr, &plain);
        const size_t poly_features = pose_feature_total(*poly_model);
        std::cout << "poly trained\n";

        // 带洞：洞盖住中心棋盘块区
        vfshape::PolygonShape holed = plain;
        holed.holes.push_back({{132, 132}, {188, 132}, {188, 188}, {132, 188}});
        const auto holed_model = vfshape::train(image, config, nullptr, &holed);
        const size_t holed_features = pose_feature_total(*holed_model);
        std::cout << "holed trained\n";

        std::cout << "mask features: rect=" << rect_features << " poly=" << poly_features
                  << " holed=" << holed_features << "\n";
        check(rect_features >= 100, "矩形基线特征数异常少：" + std::to_string(rect_features));
        check(poly_features >= 100, "多边形无洞特征数异常少：" + std::to_string(poly_features));
        check(holed_features < poly_features,
              "洞掩蔽失效：有洞特征数 " + std::to_string(holed_features) + " 未低于无洞 " +
                  std::to_string(poly_features));

        // 带洞模型自匹配仍应命中（掩蔽不破坏可定位性）
        const auto hits = vfshape::match(*holed_model, image, 70.0);
        check(!hits.empty() && hits.front().score >= 70.0, "带洞模型自匹配应高分命中");

        std::cout << "polygon_mask: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
