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

// 逐窗环形记录（W6 先决量化）：每次 match_fine_window_gpu 收尾写入最近 16 窗
// 分列计时；fine_stats_ring_reset 截断基线，fine_stats_since 读取自基线条目
//（观测方：bench 逐窗 GPU/宿主占比）。非线程安全口径=stats_slot 同款（观测用）。
int fine_stats_since(FineWindowStats* out, int cap);
void fine_stats_ring_reset();

// match_prepared(fine, prepare_frame(window), fine_th) 的 GPU 等价；window 为
// 调用方已缩放/裁剪的窗口图（补边/上传/链路在服务内完成）。
// prefetch_key：W6 窗批合并预发射领取凭据（编排层由窗口推导参数混 hash 生成，
// 同窗推导必同 key）；0=不走预发射缓存（同时作废既有预发射残留）。命中判定含
// 模型指针——组重试换模型即未命中作废，位级零变化。
std::vector<vf_shape_candidate> match_fine_window_gpu(Model& fine, const cv::Mat& window,
                                                      double fine_th,
                                                      uint64_t prefetch_key = 0);

// 预发射一窗（W6 窗批合并）：发射段与 match_fine_window_gpu 的 begin 完全同构
//（H2D/驻留/链路全入流），产物缓存待领；投机路径——任何失败静默返回 false
//（主流程回退同步发射，语义=回滚开关 OFF）。同 key 再预发射覆盖旧残留。
bool fine_prefetch(Model& fine, const cv::Mat& window, double fine_th, uint64_t prefetch_key);

// W6 窗批合并回滚开关：VF_SHAPE_WINDOW_PREFETCH（默认开；=0 回 W5 终态行为，
// 预发射与领取全禁用）。每调用读 env，测试进程内可翻转。
bool window_prefetch_enabled();

// W6 微波2 回滚开关：VF_SHAPE_WINDOW_STREAMS（默认开=每 frame_slot 专用流，
// 跨窗链/DMA 跨流并行；=0 回单流排序=W5+微波1 终态）。每调用读 env。
bool window_streams_enabled();

}  // namespace vfshape::gpu
