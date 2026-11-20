#pragma once
// 粗层链 CPU 参考（tests 内部）：尖峰 cpu_chain.cpp 的本仓平移 + vendor 逐位
// 口径（W2 拍板项复刻的 CPU 侧同式实现），供 shape_gpu_bitexact 对拍与归因。
// 双模：Spike=尖峰参考口径；Vendor=vendor 逐位口径（fastAtan2 复刻 +
// linearizeResponses/similarity 线性内存 flat 语义直译，含回卷与 span 外置零）。
// 与尖峰 cpu_chain 的唯一偏差：不做 template_fits 整体跳过（该守卫在引擎
// initialMatches 层，不属内核语义；本参考逐 pose 全算，对拍口径更严）。
#include "../src/gpu/gpu_device.hpp"

#include <cstdint>
#include <vector>

namespace vfshape::gpu_test {

enum class RefMode { Spike, Vendor };

struct RefLevel {
    int width = 0, height = 0;  // 模板宽高（vendor wf/hf 之源）
    std::vector<vfshape::gpu::FeatureRecord> feats;
};
struct RefModel {  // 单位姿两级模板（合成模型与引擎模板同构）
    RefLevel level[2];  // [0]=L0(T=4) [1]=L1(T=8)
};

struct RefMaps {
    int w[2] = {0, 0}, h[2] = {0, 0};
    std::vector<uint8_t> angle[2], spread[2];
};

struct RefChainOutput {
    RefMaps maps;
    std::vector<std::vector<uint16_t>> response[2];  // [pose] 每层
    int resp_w[2] = {0, 0}, resp_h[2] = {0, 0};
    double pyrdown_ms = 0, grad1_ms = 0, spread1_ms = 0, resp1_ms = 0;
    double grad0_ms = 0, spread0_ms = 0, resp0_ms = 0;
    double coarse_ms() const { return pyrdown_ms + grad1_ms + spread1_ms + resp1_ms; }
};

// padded_bgr：引擎 prepare 面同式补边帧（尺寸为 16 倍数）。l0_too=false 时只跑
// L1 链（基准循环提速；L0 段计时/响应不填）。
void run_reference_chain(const uint8_t* padded_bgr, int w, int h, float weak, RefMode mode,
                         const std::vector<RefModel>& poses, bool l0_too,
                         RefChainOutput& out);

}  // namespace vfshape::gpu_test
