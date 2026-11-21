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

// 跑一次粗层链（帧已 upload_frame；目标槽 features+templ_dims 已驻留）。timing
// 可空（对拍轮次免事件开销）。response 缓冲两级复用，L1 段先回读再算 L0。
// need_l0_response=false = 宿主提取精益模式：跳过 L0 全图响应内核+回读与
// L1 angle/spread 回读，只产出 response[1]+spread[0]（引擎接缝口径）；
// 默认 true = W2-A 对拍行为零变化。model_slot 选模型驻留槽（0=coarse，
// 1=fine 窗，W3）；帧档取 upload_frame_capacity 记录的当前档（未记录则回退
// 建档档，W2 直驱对拍路径行为不变）。
bool run_coarse_chain(GpuContext* ctx, int poses, float weak, CoarseMode mode,
                      bool readback_maps, CoarseResult& out, CoarseTiming* timing,
                      std::string& err, bool need_l0_response = true, int model_slot = 0,
                      int frame_slot = 0);

// 精修窗流水化专用异步链（W3 波2 补完）：与同步版同内核同参数同序（lean 口径
// L1 链+L1 全图响应+L0 链，无 L0 全图响应/无 spread 回读），全部发射入指定流后
// 即返回（发射即返回，设备完成由事件标记）；resp1 逐 pose 拼接异步回读到调用方
// pinned 缓冲（resp1_bytes = poses*W1*H1*2）完成时记录 e_resp1（宿主 initial 扫描
// 可与 L0 链重叠）；e_chain_begin/e_chain_end 包住全链供链路耗时测量。返回设备侧
// L0 spread 指针与当前档网格尺寸（供流内窗分内核直接消费，不经宿主）。
// 流/事件以 opaque 句柄传递（中立面不引入 CUDA 头）。
bool run_coarse_chain_async(GpuContext* ctx, int poses, float weak, void* stream_handle,
                            uint16_t* resp1_pinned, size_t resp1_bytes, int& resp_w1,
                            int& resp_h1, const uint8_t** spread0_dev, int& spread_w0,
                            int& spread_h0, void* e_chain_begin, void* e_resp1,
                            void* e_chain_end, int model_slot, int frame_slot,
                            std::string& err);

}  // namespace vfshape::gpu
