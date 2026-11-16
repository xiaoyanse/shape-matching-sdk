#pragma once
#include <opencv2/core.hpp>

namespace vfshape {
bool resize_coarse_image(const cv::Mat& source, cv::Mat& destination, int factor);
}
