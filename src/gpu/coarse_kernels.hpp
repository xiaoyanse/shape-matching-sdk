#pragma once
// 粗层链 GPU 后端对中性面（W2 Phase A）：L1 pyrDown→grad→spread→sim_full 全图
// 响应 + L0 grad→spread→sim_full 全图响应。语义双模（对拍归因设计，见
// coarse_kernels.cu 头注释）：SpikeRef=尖峰参考口径（复刻前基线）；
// VendorExact=vendor 逐位口径（fastAtan2 复刻 + 线性内存回卷 + span 外置零，
// W2 拍板项两项复刻，Phase B 生产口径）。
// 前置：upload_frame/upload_features/upload_template_dims 已就绪（gpu_device.hpp）。
// 仅 VF_SHAPE_CUDA=ON 时有实现；OFF 构建不引用本头。
#include "gpu_device.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vfshape::gpu {

enum class CoarseMode {
    SpikeRef,     // 尖峰参考口径：精确 atan2 + 出界跳过 + 全图写
    VendorExact,  // vendor 逐位口径：fastAtan2 复刻 + flat 回卷 + span 外置零
};

// 中间图（对拍归因/诊断回读；[0]=L0 [1]=L1）
struct CoarseMaps {
    int w[2] = {0, 0}, h[2] = {0, 0};
    std::vector<uint8_t> angle[2], spread[2];
};

struct CoarseResult {
    std::vector<std::vector<uint16_t>> response[2];  // 每 pose 每层全图响应
    int resp_w[2] = {0, 0}, resp_h[2] = {0, 0};
    CoarseMaps maps;  // readback_maps=true 时填充
};

struct CoarseTiming {
    double pyrdown_ms = 0, grad1_ms = 0, spread1_ms = 0, resp1_ms = 0;
    double grad0_ms = 0, spread0_ms = 0, resp0_ms = 0;
    // 粗层段（W2 验收门口径 = 引擎 coarse_resize+coarse：pyrDown + L1 三段）
    double coarse_ms() const { return pyrdown_ms + grad1_ms + spread1_ms + resp1_ms; }
    double wall_ms = 0;
};

// 跑一次粗层链（帧已 upload_frame；features+templ_dims 已驻留）。timing 可空
//（对拍轮次免事件开销）。response 缓冲两级复用，L1 段先回读再算 L0。
bool run_coarse_chain(GpuContext* ctx, int poses, float weak, CoarseMode mode,
                      bool readback_maps, CoarseResult& out, CoarseTiming* timing,
                      std::string& err);

}  // namespace vfshape::gpu
