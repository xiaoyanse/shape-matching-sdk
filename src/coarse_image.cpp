#include "coarse_image.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <algorithm>
#include <array>

namespace vfshape {
bool resize_coarse_image(const cv::Mat& source, cv::Mat& destination, int factor) {
    CV_Assert(factor == 2 || factor == 4 || factor == 8);
    const cv::Size size(std::max(24, source.cols / factor), std::max(24, source.rows / factor));
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 10
    const double scale_y = 1.0 / (double(size.height) / source.rows);
    if (source.type() == CV_8UC3 && source.cols == size.width * 8 &&
        source.rows % size.height != 0 && scale_y > 1 && scale_y < 9 &&
        !cv::ipp::useIPP_NotExact()) {
        const cv::Mat input = source;
        cv::Mat result(size, CV_8UC3);
        struct Row { const uchar* pixels; float weight; };
        const double stripes = std::max(1.0, std::min(double(cv::getNumThreads()), input.total() / 262144.0));
        cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                const double begin = y * scale_y, end = begin + scale_y;
                const double width = std::min(scale_y, input.rows - begin);
                const int last = std::min(cvFloor(end), input.rows - 1);
                const int first = std::min(cvCeil(begin), last);
                std::array<Row, 11> rows{};
                int count = 0;
                if (first - begin > 1e-3)
                    rows[count++] = {input.ptr(first - 1), float((first - begin) / width)};
                for (int sy = first; sy < last; ++sy)
                    rows[count++] = {input.ptr(sy), float(1.0 / width)};
                if (end - last > 1e-3)
                    rows[count++] = {input.ptr(last), float(std::min(std::min(end - last, 1.0), width) / width)};
                uchar* output = result.ptr(y);
                for (int x = 0; x < size.width; ++x) {
                    std::array<float, 3> sum{};
                    for (int r = 0; r < count; ++r) {
                        const uchar* p = rows[r].pixels + x * 24;
                        for (int c = 0; c < 3; ++c) {
                            const int value = p[c] + p[c + 3] + p[c + 6] + p[c + 9] +
                                              p[c + 12] + p[c + 15] + p[c + 18] + p[c + 21];
                            const float average = value * 0.125f;
                            sum[c] += rows[r].weight * average;
                        }
                    }
                    for (int c = 0; c < 3; ++c) output[3 * x + c] = cv::saturate_cast<uchar>(sum[c]);
                }
            }
        }, stripes);
        destination = std::move(result);
        return true;
    }
#endif
    cv::resize(source, destination, size, 0, 0, cv::INTER_AREA);
    return false;
}
}
