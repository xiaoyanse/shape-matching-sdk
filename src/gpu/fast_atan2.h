#ifndef VF_SHAPE_GPU_FAST_ATAN2_H
#define VF_SHAPE_GPU_FAST_ATAN2_H
// cv::phase 度域 fastAtan2 复刻（W2 拍板项 1·甲）：公式逐字取自 OpenCV 4.10
// mathfuncs_core.simd.hpp v_atan_f32——多项式近似，FMA 单次舍入链（本机 DLL 派发
// AVX2+FMA；整数 Sobel 梯度对全网格 ±1020 共 416 万对穷举逐位一致，复跑口径见
// tests/gpu_bitexact_test.cpp Part 1）。host/device 双端同式。
#include <cmath>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define VF_GPU_HD __host__ __device__
#else
#define VF_GPU_HD
#endif

namespace vfshape::gpu {
namespace fast_atan2_detail {
constexpr float kRadToDeg = (float)(180.0 / 3.1415926535897932384626433832795);
constexpr float kP1 = 0.9997878412794807f * kRadToDeg;
constexpr float kP3 = -0.3258083974640975f * kRadToDeg;
constexpr float kP5 = 0.1555786518463281f * kRadToDeg;
constexpr float kP7 = -0.04432655554792128f * kRadToDeg;
constexpr float kEps = (float)2.2204460492503131e-16;  // (float)DBL_EPSILON
}  // namespace fast_atan2_detail

// 度域 [0,360)，与 cv::phase(x, y, angle, true) 同值（穷举网格 0 差）
VF_GPU_HD inline float fast_atan2_deg(float y, float x) {
    using namespace fast_atan2_detail;
    const float ax = fabsf(x), ay = fabsf(y);
    const float c = fminf(ax, ay) / (fmaxf(ax, ay) + kEps);
    const float c2 = c * c;
#if defined(__CUDA_ARCH__)
    const float t = __fmaf_rn(__fmaf_rn(__fmaf_rn(c2, kP7, kP5), c2, kP3), c2, kP1) * c;
#else
    const float t = std::fma(std::fma(std::fma(c2, kP7, kP5), c2, kP3), c2, kP1) * c;
#endif
    float a = ax >= ay ? t : 90.f - t;
    if (x < 0) a = 180.f - a;
    if (y < 0) a = 360.f - a;
    return a;
}

// vendor 量化口径（hysteresis 前的 16 桶）：angle.convertTo(CV_8U, 16.0/360.0) =
// round-half-even(angle * (float)(16.0/360.0))，float 域缩放（convert_scale.simd.hpp
// cvt_32f wtype=float；b=0 时 v_fma 与单乘同值——网格穷举 0 差）。返回值已 &7。
VF_GPU_HD inline int quantize_orientation(float angle_deg) {
#if defined(__CUDA_ARCH__)
    return __float2int_rn(angle_deg * (float)(16.0 / 360.0)) & 7;
#else
    return int(std::lrintf(angle_deg * (float)(16.0 / 360.0))) & 7;
#endif
}

}  // namespace vfshape::gpu
#endif
