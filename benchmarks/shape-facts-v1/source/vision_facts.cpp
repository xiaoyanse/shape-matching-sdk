// 合成图案绘制 + line2Dup 模板训练/匹配实现
#include "vision_facts.h"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace shape_poc {

cv::Point2f mapPoint(const cv::Mat &m, const cv::Point2f &p) {
    return {static_cast<float>(m.at<double>(0, 0) * p.x + m.at<double>(0, 1) * p.y + m.at<double>(0, 2)),
            static_cast<float>(m.at<double>(1, 0) * p.x + m.at<double>(1, 1) * p.y + m.at<double>(1, 2))};
}

// 确定性背景纹理：全帧梯度避免平坦区导致响应图退化，噪点弱于图案梯度不干扰匹配
void drawBackground(cv::Mat &img, uint64_t seed) {
    img.setTo(cv::Scalar(150, 150, 150));
    cv::RNG rng(seed);
    for (int i = 0; i < 900; ++i) {
        cv::Point p(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const int shade = rng.uniform(105, 195);
        cv::circle(img, p, rng.uniform(1, 3), cv::Scalar(shade, shade, shade), cv::LINE_8);
    }
}

// 锚点图案：方框+偏心圆+折线，不对称组合保证任意角度下方向可辨
void drawAnchorPattern(cv::Mat &img, cv::Point2f center, double angleDeg) {
    const cv::Mat rot = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    const cv::Scalar ink(45, 45, 45);
    const cv::Point2f local[] = {{-80, -80}, {80, -80}, {80, 80}, {-80, 80}};
    cv::Point box[4];
    for (int i = 0; i < 4; ++i) box[i] = mapPoint(rot, center + local[i]);
    for (int i = 0; i < 4; ++i) cv::line(img, box[i], box[(i + 1) % 4], ink, 8, cv::LINE_8);
    cv::circle(img, mapPoint(rot, center + cv::Point2f(36, -30)), 26, ink, -1);
    const cv::Point2f poly[3] = {mapPoint(rot, center + cv::Point2f(-56, 48)),
                                 mapPoint(rot, center + cv::Point2f(10, -58)),
                                 mapPoint(rot, center + cv::Point2f(58, 40))};
    for (int i = 0; i < 3; ++i) {
        cv::Point a(poly[i]), b(poly[(i + 1) % 3]);
        cv::line(img, a, b, ink, 10, cv::LINE_8);
    }
}

// 目标图案：实心圆+偏心方块+圆环（SOP 检测对象）
void drawTargetPattern(cv::Mat &img, cv::Point2f center, double angleDeg) {
    const cv::Mat rot = cv::getRotationMatrix2D(center, angleDeg, 1.0);
    const cv::Scalar ink(45, 45, 45);
    cv::circle(img, center, 48, ink, -1);
    const cv::Point2f localQuad[4] = {{-14, -14}, {14, -14}, {14, 14}, {-14, 14}};
    std::vector<cv::Point> quad;
    for (int i = 0; i < 4; ++i) quad.push_back(mapPoint(rot, center + cv::Point2f(30, 22) + localQuad[i]));
    std::vector<std::vector<cv::Point>> polys{quad};
    cv::fillPoly(img, polys, ink);
    cv::circle(img, mapPoint(rot, center + cv::Point2f(-24, -20)), 34, ink, 9);
}

// 图案静止画在训练画布中心 → 裁剪进黑底源 + 矩形 mask 建零度模板 → ±25°/5° 旋转模板族
TemplateBank trainTemplate(const std::string &classId, PatternDraw pattern) {
    cv::Mat canvas(kTrainCanvas, kTrainCanvas, CV_8UC3, cv::Scalar::all(0));
    pattern(canvas, cv::Point2f(kTrainCanvas / 2.0f, kTrainCanvas / 2.0f), 0.0);
    cv::Mat source(kTrainCanvas, kTrainCanvas, CV_8UC3, cv::Scalar::all(0));
    cv::Mat mask(kTrainCanvas, kTrainCanvas, CV_8UC1, cv::Scalar(0));
    const cv::Rect crop(kCropOrigin, kCropOrigin, kCropHalf * 2, kCropHalf * 2);
    canvas(crop).copyTo(source(crop));
    cv::rectangle(mask, crop, cv::Scalar(255), cv::FILLED);

    TemplateBank bank;
    bank.classId = classId;
    const int zeroId = bank.detector.addTemplate(source, classId, mask);
    if (zeroId < 0) throw std::runtime_error("addTemplate failed for " + classId);
    bank.angles.assign(1, 0.0f);
    for (int deg = -25; deg <= 25; deg += 5) {
        if (deg == 0) continue;
        const int id = bank.detector.addTemplate_rotate(classId, zeroId, static_cast<float>(deg), bank.modelCenter);
        if (id < 0) throw std::runtime_error("addTemplate_rotate failed for " + classId);
        if (static_cast<int>(bank.angles.size()) <= id) bank.angles.resize(id + 1, NAN);
        bank.angles[id] = static_cast<float>(deg);
    }
    return bank;
}

// 真实 native matcher 调用：全图搜索取最高分，反算图案中心
// 中心 = R(angle)·modelCenter + (match.xy - templ.tl)，modelCenter 为旋转中心故角度量化误差不放大中心误差
Hit matchOne(const cv::Mat &frame, const TemplateBank &bank, float threshold) {
    Hit hit;
    const std::vector<line2Dup::Match> ms = bank.detector.match(frame, threshold, {bank.classId});
    hit.candidates = static_cast<int>(ms.size());
    for (const auto &m : ms) {
        if (hit.ok && m.similarity <= hit.score) continue;
        const auto &templ = bank.detector.getTemplates(bank.classId, m.template_id).front();
        const float ang = bank.angles.at(m.template_id);
        const cv::Mat rot = cv::getRotationMatrix2D(bank.modelCenter, ang, 1.0);
        hit.center = mapPoint(rot, bank.modelCenter) + cv::Point2f(m.x - templ.tl_x, m.y - templ.tl_y);
        hit.ok = true;
        hit.score = m.similarity;
        hit.angle = ang;
        hit.templateId = m.template_id;
    }
    return hit;
}

cv::Mat padToAligned16(const cv::Mat &frame) {
    const int padW = (16 - frame.cols % 16) % 16;
    const int padH = (16 - frame.rows % 16) % 16;
    if (padW == 0 && padH == 0) return frame;
    // copyMakeBorder 只在右/底补零：原点不动，命中坐标即原图坐标
    cv::Mat padded;
    cv::copyMakeBorder(frame, padded, 0, padH, 0, padW, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

void confineToOriginalRegion(Hit &hit, int origCols, int origRows) {
    if (hit.ok && (hit.center.x < 0 || hit.center.y < 0 || hit.center.x >= origCols || hit.center.y >= origRows)) {
        hit.ok = false;
        hit.score = -1.0f;
        hit.candidates = 0;
        hit.templateId = -1;
    }
}

std::vector<cv::Point2f> mapModelVertices(const TemplateBank &bank, const Hit &hit,
                                          const std::vector<cv::Point2f> &localVertices) {
    std::vector<cv::Point2f> out;
    if (!hit.ok) return out;
    const cv::Mat rot = cv::getRotationMatrix2D(bank.modelCenter, hit.angle, 1.0);
    const cv::Point2f shift = hit.center - mapPoint(rot, bank.modelCenter);
    out.reserve(localVertices.size());
    for (const auto &v : localVertices) out.push_back(mapPoint(rot, bank.modelCenter + v) + shift);
    return out;
}

}  // namespace shape_poc
