// 合成图案绘制 + line2Dup 模板训练/匹配：FrameVisualFacts 的视觉事实来源（真实 native matcher）
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "line2Dup.h"

namespace shape_poc {

constexpr int kTrainCanvas = 540;  // 训练源画布边长（图案居中，裁剪块含余量）
constexpr int kCropHalf = 130;     // 图案裁剪半宽（图案 200px + 余量 30）
constexpr int kCropOrigin = (kTrainCanvas - kCropHalf * 2) / 2;
constexpr int kFrameW = 1280;      // 匹配帧 16 像素对齐（金字塔下采样整除约束）
constexpr int kFrameH = 960;

using PatternDraw = void (*)(cv::Mat &, cv::Point2f, double);

cv::Point2f mapPoint(const cv::Mat &m, const cv::Point2f &p);
void drawBackground(cv::Mat &img, uint64_t seed);
void drawAnchorPattern(cv::Mat &img, cv::Point2f center, double angleDeg);
void drawTargetPattern(cv::Mat &img, cv::Point2f center, double angleDeg);

struct TemplateBank {
    line2Dup::Detector detector;
    std::vector<float> angles;  // template_id → 训练角
    cv::Point2f modelCenter{kTrainCanvas / 2.0f, kTrainCanvas / 2.0f};
    std::string classId;
};

struct Hit {
    bool ok = false;
    float score = -1.0f;
    float angle = 0.0f;  // 命中模板的训练角（5° 档量化），非连续角度估计
    cv::Point2f center{};
    int templateId = -1;
    int candidates = 0;
};

TemplateBank trainTemplate(const std::string &classId, PatternDraw pattern);
Hit matchOne(const cv::Mat &frame, const TemplateBank &bank, float threshold);

// 匹配输入适配：仅右侧/底部补零至 16 倍数（不缩放、不移动原点，坐标仍为原图坐标系）；已对齐则原样返回
cv::Mat padToAligned16(const cv::Mat &frame);

// padding 区有效性过滤：命中中心落在原图区域之外 → 判为未命中（padding 不得作为有效 ROI/目标依据）
void confineToOriginalRegion(Hit &hit, int origCols, int origRows);

// 锚点模型坐标（相对图案中心）顶点 → 帧坐标：R(angle)·(modelCenter+v) + shift
std::vector<cv::Point2f> mapModelVertices(const TemplateBank &bank, const Hit &hit,
                                          const std::vector<cv::Point2f> &localVertices);

}  // namespace shape_poc
