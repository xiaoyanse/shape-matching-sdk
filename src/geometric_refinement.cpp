#include "two_stage.hpp"
#include <algorithm>
#include <cmath>

namespace vfshape {
namespace {
double gradient_projection(const cv::Mat& gx, const cv::Mat& gy, double x, double y,
                           double nx, double ny) {
    const int sx = std::min(int(x), gx.cols - 2), sy = std::min(int(y), gx.rows - 2);
    const double ax = x - sx, ay = y - sy;
    auto sample = [&](const cv::Mat& field) {
        return (1 - ay) * ((1 - ax) * field.at<float>(sy, sx) + ax * field.at<float>(sy, sx + 1)) +
               ay * ((1 - ax) * field.at<float>(sy + 1, sx) + ax * field.at<float>(sy + 1, sx + 1));
    };
    return sample(gx) * nx + sample(gy) * ny;
}
}

B7Model extract_b7_points(const cv::Mat& tmpl_gray) {
    B7Model b7;
    cv::Mat gx, gy, mag;
    cv::Sobel(tmpl_gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(tmpl_gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    const int h = tmpl_gray.rows, w = tmpl_gray.cols;
    constexpr int kTargetPts = 160;
    const int step = std::max(4, int(std::sqrt(double(h * w) / kTargetPts)));
    const double cx = w / 2.0, cy = h / 2.0;
    for (int by = 0; by + step <= h; by += step) {
        for (int bx = 0; bx + step <= w; bx += step) {
            cv::Mat block = mag(cv::Rect(bx, by, step, step));
            double block_max;
            cv::Point block_at;
            cv::minMaxLoc(block, nullptr, &block_max, nullptr, &block_at);
            const int x = bx + block_at.x, y = by + block_at.y;
            if (block_max > 40 && x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
                double gvx = gx.at<float>(y, x), gvy = gy.at<float>(y, x);
                const double norm = std::hypot(gvx, gvy) + 1e-9;
                const double nx = gvx / norm, ny = gvy / norm;
                const double left = gradient_projection(gx, gy, x - nx, y - ny, nx, ny);
                const double mid = gradient_projection(gx, gy, x, y, nx, ny);
                const double right = gradient_projection(gx, gy, x + nx, y + ny, nx, ny);
                const double denominator = left - 2 * mid + right;
                const double offset = denominator < -1e-9 && mid >= left && mid >= right
                    ? 0.5 * (left - right) / denominator : 0;
                b7.px.push_back(x + nx * offset - cx);
                b7.py.push_back(y + ny * offset - cy);
                b7.nx.push_back(nx);
                b7.ny.push_back(ny);
            }
        }
    }
    return b7;
}
B7Result b7_refine(const cv::Mat& gray, const B7Model& b7,
                   double cx, double cy, double angle_deg, double scale) {
    B7Result result;
    result.x = cx;
    result.y = cy;
    result.angle = angle_deg;
    result.scale = scale;
    result.valid = 0;
    const int n = int(b7.px.size());
    if (n < kB7MinValidPts) return result;
    const int h = gray.rows, w = gray.cols;
    // 候选局部包围盒（Sobel 局部化）
    double radius_max = 0;
    for (int i = 0; i < n; ++i)
        radius_max = std::max(radius_max, std::hypot(b7.px[i], b7.py[i]));
    const int radius = int(radius_max * scale + kB7SearchRadius + 2);
    const int bx0 = int(std::max(0.0, cx - radius)), by0 = int(std::max(0.0, cy - radius));
    const int bx1 = int(std::min(double(w), cx + radius + 1));
    const int by1 = int(std::min(double(h), cy + radius + 1));
    if (bx1 - bx0 < 3 * int(kB7SearchRadius) + 8 || by1 - by0 < 3 * int(kB7SearchRadius) + 8)
        return result;
    cv::Mat patch = gray(cv::Rect(bx0, by0, bx1 - bx0, by1 - by0));
    cv::Mat patch_gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, patch_gray, cv::COLOR_BGR2GRAY);
        patch = patch_gray;
    }
    cv::Mat gx, gy;
    cv::Sobel(patch, gx, CV_32F, 1, 0, 3);
    cv::Sobel(patch, gy, CV_32F, 0, 1, 3);
    const int ph = patch.rows, pw = patch.cols;

    // OpenCV 正角度在图像坐标中对应负数学角度。
    double theta = -angle_deg * CV_PI / 180.0;
    double tx = cx, ty = cy;
    const int K = int(kB7SearchRadius) * 2 + 1;
    std::vector<double> d_off(K);
    for (int d = 0; d < K; ++d) d_off[d] = -kB7SearchRadius + d;

    for (int iter = 0; iter < kB7GnIters; ++iter) {
        const double ct = std::cos(theta), st = std::sin(theta);
        // 世界坐标点/法线（局部系）
        std::vector<double> proj(n * K, -1e9);
        std::vector<char> inb(n * K, 0);
        for (int i = 0; i < n; ++i) {
            // p_world = s·R·p + t（局部）
            const double wx = scale * (ct * b7.px[i] - st * b7.py[i]) + tx - bx0;
            const double wy = scale * (st * b7.px[i] + ct * b7.py[i]) + ty - by0;
            // n_world = R·n
            const double nxw = ct * b7.nx[i] - st * b7.ny[i];
            const double nyw = st * b7.nx[i] + ct * b7.ny[i];
            for (int d = 0; d < K; ++d) {
                const double fx = wx + nxw * d_off[d], fy = wy + nyw * d_off[d];
                if (fx >= 1 && fy >= 1 && fx < pw - 2 && fy < ph - 2) {
                    proj[i * K + d] = gradient_projection(gx, gy, fx, fy, nxw, nyw);
                    inb[i * K + d] = 1;
                }
            }
        }
        // 每点极性 argmax + 亚像素（弱纹理自适应：标准阈值点数不足时半阈重收
        // 一次——img2 实测标准阈值下有效点 <24 → B7 不激活 → 旋转例 5-10px 残留）
        std::vector<int> idx;
        std::vector<double> d_stars;
        auto collect = [&](double grad_min) {
            idx.clear();
            d_stars.clear();
            for (int i = 0; i < n; ++i) {
                int best_d = -1;
                double best_m = -1e9;
                for (int d = 0; d < K; ++d)
                    if (inb[i * K + d] && proj[i * K + d] > best_m) {
                        best_m = proj[i * K + d];
                        best_d = d;
                    }
                if (best_d < 0 || best_m < grad_min) continue;
                const double p0 = proj[i * K + best_d];
                const double p1 = proj[i * K + std::min(best_d + 1, K - 1)];
                const double pm = proj[i * K + std::max(best_d - 1, 0)];
                const double denom = pm - 2 * p0 + p1;
                double shift = 0;
                if (denom != 0 && best_d > 0 && best_d < K - 1)
                    shift = 0.5 * (pm - p1) / denom;
                idx.push_back(i);
                d_stars.push_back(d_off[best_d] + shift);
            }
        };
        collect(kB7EdgeGradMin);
        if (int(idx.size()) < kB7MinValidPts) collect(kB7EdgeGradMin / 2);
        const int m = int(idx.size());
        if (m < kB7MinValidPts) break;
        // 以零残差为目标估计鲁棒尺度，正负位移必须对称处理。
        std::vector<double> r(m);
        for (int j = 0; j < m; ++j) r[j] = -d_stars[j];
        std::vector<double> abs_r;
        abs_r.reserve(r.size());
        for (double residual : r) abs_r.push_back(std::abs(residual));
        std::sort(abs_r.begin(), abs_r.end());
        const double mad = abs_r[m / 2];
        const double sigma = std::max(1.4826 * mad, 0.3);
        // 物理下限 1px：初值不确定度大时（d_stars 方差大、median 小）纯 MAD
        // 会把正常大残差点误剔（实测 57 点 → 3 点）；亚像素级剔除无意义。
        const double cut = std::max(kB7OutlierSigmaX10 / 10.0 * sigma, 1.0);
        std::vector<int> inlier;
        for (int j = 0; j < m; ++j)
            if (std::abs(r[j]) < cut) inlier.push_back(j);
        if (int(inlier.size()) < kB7MinValidPts) break;
        // 4 参 GN：J=[∂r/∂θ, ∂r/∂tx, ∂r/∂ty, ∂r/∂s]
        cv::Mat A = cv::Mat::zeros(4, 4, CV_64F), bvec = cv::Mat::zeros(4, 1, CV_64F);
        for (int j : inlier) {
            const int i = idx[j];
            const double wx = b7.px[i], wy = b7.py[i];
            const double nxw = ct * b7.nx[i] - st * b7.ny[i];
            const double nyw = st * b7.nx[i] + ct * b7.ny[i];
            const double dth_x = -scale * (st * wx + ct * wy);
            const double dth_y = scale * (ct * wx - st * wy);
            const double jth = nxw * dth_x + nyw * dth_y;
            const double rpx = ct * wx - st * wy;
            const double rpy = st * wx + ct * wy;
            const double js = scale * (nxw * rpx + nyw * rpy);
            const double J[4] = {jth, nxw, nyw, js};
            for (int a = 0; a < 4; ++a) {
                bvec.at<double>(a) += J[a] * (-r[j]);
                for (int c = 0; c < 4; ++c) A.at<double>(a, c) += J[a] * J[c];
            }
        }
        // 病态检查（对角占优近似：θ 列范数过小时退化）
        const double diag_min =
            std::min({A.at<double>(0, 0), A.at<double>(1, 1), A.at<double>(2, 2)});
        if (!(diag_min > 0) || cv::invert(A, A, cv::DECOMP_SVD) == 0) break;
        cv::Mat delta = A * bvec;
        const double d_theta = delta.at<double>(0);
        const double d_tx = delta.at<double>(1);
        const double d_ty = delta.at<double>(2);
        const double d_s = delta.at<double>(3);
        if (!std::isfinite(d_theta) || !std::isfinite(d_tx) || !std::isfinite(d_ty) ||
            !std::isfinite(d_s) || std::abs(d_s) >= 0.1) break;
        theta += d_theta;
        tx += d_tx;
        ty += d_ty;
        scale *= 1.0 + d_s;
        result.valid = int(inlier.size());
        if (std::hypot(d_tx, d_ty) < 0.01 && std::abs(d_theta) < 1e-4 && std::abs(d_s) < 1e-4)
            break;
    }
    if (result.valid < kB7MinValidPts) {
        result.x = cx;
        result.y = cy;
        result.angle = angle_deg;
        result.valid = 0;
        return result;
    }
    result.x = tx;
    result.y = ty;
    result.angle = -theta * 180.0 / CV_PI;
    result.scale = scale;
    return result;
}
void apply_geometric_refinement(vf_shape_candidate& c, const B7Result& refined) {
    const double theta = (refined.angle - c.angle) * CV_PI / 180.0;
    const double ratio = refined.scale / c.scale;
    const double a = ratio * std::cos(theta), b = ratio * std::sin(theta);
    const double tx = refined.x - a * c.center[0] - b * c.center[1];
    const double ty = refined.y + b * c.center[0] - a * c.center[1];
    const auto old = std::array<double, 6>{c.transform[0], c.transform[1], c.transform[2],
                                         c.transform[3], c.transform[4], c.transform[5]};
    for (int col = 0; col < 3; ++col) {
        c.transform[col] = a * old[col] + b * old[col + 3];
        c.transform[col + 3] = -b * old[col] + a * old[col + 3];
    }
    c.transform[2] += tx;
    c.transform[5] += ty;
    double left = 1e20, top = 1e20, right = -1e20, bottom = -1e20;
    for (int i = 0; i < 4; ++i) {
        const double x = c.corners[2 * i], y = c.corners[2 * i + 1];
        c.corners[2 * i] = a * x + b * y + tx;
        c.corners[2 * i + 1] = -b * x + a * y + ty;
        left = std::min(left, c.corners[2 * i]);
        right = std::max(right, c.corners[2 * i]);
        top = std::min(top, c.corners[2 * i + 1]);
        bottom = std::max(bottom, c.corners[2 * i + 1]);
    }
    c.center[0] = refined.x;
    c.center[1] = refined.y;
    c.angle = refined.angle;
    c.scale = refined.scale;
    c.bbox[0] = left;
    c.bbox[1] = top;
    c.bbox[2] = right - left;
    c.bbox[3] = bottom - top;
}
}  // namespace vfshape
