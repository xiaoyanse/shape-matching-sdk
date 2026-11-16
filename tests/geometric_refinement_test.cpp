#include "two_stage.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_pose(double old_angle, double angle, double scale, double dx, double dy) {
    vf_shape_candidate candidate{};
    const auto initial = cv::getRotationMatrix2D(cv::Point2d(90, 80), old_angle, 1.0);
    const std::array<double, 8> points{10, 20, 170, 20, 170, 140, 10, 140};
    for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 3; ++col)
            candidate.transform[row * 3 + col] = initial.at<double>(row, col);
    for (int i = 0; i < 4; ++i) {
        for (int row = 0; row < 2; ++row)
            candidate.corners[2 * i + row] = initial.at<double>(row, 0) * points[2 * i] +
                initial.at<double>(row, 1) * points[2 * i + 1] + initial.at<double>(row, 2);
    }
    candidate.center[0] = 90;
    candidate.center[1] = 80;
    candidate.angle = old_angle;
    candidate.scale = 1;
    vfshape::B7Result refined;
    refined.x = 90 + dx;
    refined.y = 80 + dy;
    refined.angle = angle;
    refined.scale = scale;
    vfshape::apply_geometric_refinement(candidate, refined);
    const auto expected = cv::getRotationMatrix2D(cv::Point2d(90, 80), angle, scale);
    for (int i = 0; i < 4; ++i) {
        for (int row = 0; row < 2; ++row) {
            const double x = points[2 * i], y = points[2 * i + 1];
            const double truth = expected.at<double>(row, 0) * x + expected.at<double>(row, 1) * y +
                expected.at<double>(row, 2) + (row ? dy : dx);
            const double mapped = candidate.transform[row * 3] * x +
                candidate.transform[row * 3 + 1] * y + candidate.transform[row * 3 + 2];
            check(std::abs(mapped - truth) < 1e-9, "matrix does not match refined pose");
            check(std::abs(candidate.corners[2 * i + row] - truth) < 1e-9,
                  "corners do not match refined pose");
        }
    }
    check(std::abs(candidate.center[0] - refined.x) < 1e-9, "wrong center");
    check(std::abs(candidate.scale - scale) < 1e-9, "wrong scale");
}

void check_solver(double angle) {
    cv::Mat reference(320, 320, CV_8UC1, cv::Scalar(0));
    cv::rectangle(reference, cv::Rect(75, 90, 140, 120), cv::Scalar(240), 3, cv::LINE_AA);
    cv::circle(reference, cv::Point(118, 137), 26, cv::Scalar(220), 3, cv::LINE_AA);
    cv::line(reference, cv::Point(85, 200), cv::Point(201, 101), cv::Scalar(180), 4, cv::LINE_AA);
    cv::rectangle(reference, cv::Rect(178, 171, 21, 22), cv::Scalar(200), -1);
    const auto model = vfshape::extract_b7_points(reference);
    auto transform = cv::getRotationMatrix2D(cv::Point2d(160, 160), angle, 1.01);
    transform.at<double>(0, 2) += 2.3;
    transform.at<double>(1, 2) -= 1.8;
    cv::Mat image;
    cv::warpAffine(reference, image, transform, reference.size());
    const auto refined = vfshape::b7_refine(image, model, 162, 158, angle + 0.3, 1.0);
    check(refined.valid >= vfshape::kB7MinValidPts, "solver did not activate");
    check(std::hypot(refined.x - 162.3, refined.y - 158.2) < 0.6,
          "solver translation error");
    check(std::abs(refined.angle - angle) < 0.2, "solver rotation convention error");
    check(std::abs(refined.scale - 1.01) < 0.01, "solver scale error");
}
}

int main() {
    try {
        check_pose(0, 0.4, 1.002, 0.2, -0.3);
        check_pose(-29.5, -30, 0.98, -2.1, 1.7);
        check_pose(29.5, 30.1, 1.03, 1.2, -1.3);
        check_pose(179.8, -179.9, 1, 0, 0);
        check_solver(25);
        check_solver(-25);
        vfshape::B7Model empty;
        const auto invalid = vfshape::b7_refine(cv::Mat(32, 32, CV_8UC1), empty, 16, 16, 20, 1.1);
        check(invalid.valid == 0 && invalid.x == 16 && invalid.scale == 1.1,
              "empty model changed pose");
        std::cout << "7 geometry contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
