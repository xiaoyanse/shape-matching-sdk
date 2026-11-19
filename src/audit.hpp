#pragma once

// 模型健康体检与区分度审计的内部实现（ABI 导出在 abi.cpp，输出结构合同
// 在 include/shape_matching.h）。单级与两级容器模型二选一（另一指针为空）。

#include "model.hpp"
#include "shape_matching.h"
#include "two_stage.hpp"

#include <cstdint>

namespace vfshape {

vf_shape_health_report model_health(Model* single, TwoStageModel* two_stage);

vf_shape_audit_report audit_discriminability(
    Model* single, TwoStageModel* two_stage,
    const uint8_t* const* pos_images, int pos_width, int pos_height, uint64_t pos_stride,
    uint32_t pos_count,
    const uint8_t* const* neg_images, int neg_width, int neg_height, uint64_t neg_stride,
    uint32_t neg_count);

}
