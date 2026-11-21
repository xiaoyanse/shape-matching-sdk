#pragma once
// 精修窗 GPU 服务（W3 波2）：refine_window evaluate 接缝的 match_prepared GPU
// 等价——等价 match_prepared(fine, prepare_frame(window), fine_th) 的候选集
//（collect_hit_objects 同链收尾）。window 契约同 match_prepared 前置（阈值域 +
// 搜索预算同式校验）；失败编码错误直报（W2 口径，不回退）。
// 仅 VF_SHAPE_CUDA=ON 时参与编译。
#include "../model.hpp"

#include "gpu_device.hpp"

#include <string>
#include <vector>

namespace vfshape::gpu {

// 最近一次精修窗 GPU 调用的单列证据（W3 流水化口径分列）：chain_ms=GPU 链路
// 完成耗时（事件测）；initial_scan_ms=宿主 initial 扫描（与 L0 链重叠段）；
// window_kernel_ms/scores_d2h_ms=窗分内核与回读（事件测）；extract_ms=消费收尾
// CPU 墙钟；initial_count/window_kernel_used 同前。
struct FineWindowStats {
    double chain_ms = 0, extract_ms = 0;
    double initial_scan_ms = 0, window_kernel_ms = 0, scores_d2h_ms = 0;
    int initial_count = 0;
    int window_kernel_used = 0;
};
const FineWindowStats& last_fine_stats();

// match_prepared(fine, prepare_frame(window), fine_th) 的 GPU 等价；window 为
// 调用方已缩放/裁剪的窗口图（补边/上传/链路在服务内完成）
std::vector<vf_shape_candidate> match_fine_window_gpu(Model& fine, const cv::Mat& window,
                                                      double fine_th);

}  // namespace vfshape::gpu
