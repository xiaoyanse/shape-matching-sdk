#include "coarse_image.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        cv::RNG random(71213);
        int fast = 0, checked = 0;
        for (const cv::Size size : {cv::Size(192, 191), cv::Size(320, 213), cv::Size(384, 257),
                 cv::Size(640, 513), cv::Size(641, 513), cv::Size(4096, 2730), cv::Size(192, 192)}) {
            cv::Mat storage(size.height + 4, size.width + 7, CV_8UC3);
            random.fill(storage, cv::RNG::UNIFORM, 0, 256);
            const cv::Mat image = storage(cv::Rect(3, 2, size.width, size.height));
            for (int factor : {2, 4, 8}) {
                cv::Mat expected, actual;
                cv::resize(image, expected, {std::max(24, image.cols / factor),
                    std::max(24, image.rows / factor)}, 0, 0, cv::INTER_AREA);
                fast += vfshape::resize_coarse_image(image, actual, factor);
                if (cv::norm(expected, actual, cv::NORM_INF) != 0)
                    throw std::runtime_error("coarse resize differs from OpenCV INTER_AREA");
                ++checked;
            }
        }
        for (int value : {0, 127, 128, 255}) {
            cv::Mat image(213, 320, CV_8UC3, cv::Scalar::all(value)), expected;
            cv::resize(image, expected, {40, 26}, 0, 0, cv::INTER_AREA);
            vfshape::resize_coarse_image(image, image, 8);
            if (cv::norm(image, expected, cv::NORM_INF) != 0)
                throw std::runtime_error("coarse resize alias or constant image failed");
        }
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 10
        if (!cv::ipp::useIPP_NotExact() && !fast)
            throw std::runtime_error("coarse optimized branch was not exercised");
#endif
        std::cout << "coarse resize: " << checked << " pixel-identical cases, " << fast << " optimized\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
