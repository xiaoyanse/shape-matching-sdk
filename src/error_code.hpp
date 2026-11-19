#pragma once

// 线程局部错误码记录：fail/require_coded 在抛出前写码与数值上下文，
// ABI boundary 捕获时按作用域新鲜度保留记录或落 INTERNAL。每线程独立，
// 跨线程不串（导出查询 vf_shape_last_error_info 只见本线程最近一次失败）。

#include "shape_matching.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace vfshape {

struct last_error_record {
    int32_t code = int32_t(VF_SHAPE_OK);
    int32_t param_count = 0;
    double params[6] = {};
    uint64_t generation = 0;
};

inline thread_local last_error_record g_last_error;
inline thread_local uint64_t g_error_scope = 0;

// 无抛出版落账入口：批量编排等不抛异常的失败路径直接记录
inline void record_error(vf_shape_error_code code, std::initializer_list<double> params) {
    g_last_error.code = int32_t(code);
    int32_t count = 0;
    for (double value : params) {
        if (count >= 6) break;
        g_last_error.params[count++] = value;
    }
    g_last_error.param_count = count;
    g_last_error.generation = g_error_scope;
}

[[noreturn]] inline void fail(vf_shape_error_code code, std::initializer_list<double> params,
                              const char* message) {
    record_error(code, params);
    throw std::invalid_argument(message);
}

// 数值参数按序取到首个 NaN 为止（缺省占位即"未提供"），语义随检查点不同
inline void require_coded(bool condition, vf_shape_error_code code, const char* message,
                          double p1 = std::nan(""), double p2 = std::nan(""),
                          double p3 = std::nan("")) {
    if (condition) return;
    if (std::isnan(p1)) fail(code, {}, message);
    if (std::isnan(p2)) fail(code, {p1}, message);
    if (std::isnan(p3)) fail(code, {p1, p2}, message);
    fail(code, {p1, p2, p3}, message);
}

inline uint64_t begin_error_scope() { return ++g_error_scope; }

// 捕获侧收口：本次作用域内有码记录则保留，否则按 INTERNAL 落账（保证
// "最近一次失败"口径始终指向当前这次调用）
inline void finish_error_scope(uint64_t ticket) {
    if (g_last_error.generation == ticket) return;
    g_last_error.code = int32_t(VF_SHAPE_INTERNAL);
    g_last_error.param_count = 0;
    g_last_error.generation = ticket;
}

}
