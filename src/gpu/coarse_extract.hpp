#pragma once
// 宿主侧粗层候选提取/精修（W2-B）：消费 GPU 粗层链产物（L1 全图响应 + L0 spread
// 图），复刻 vendor matchClass 全链（initial→精化→过滤→sort+unique）。窗分从
// L0 spread 图直算——vendor similarityLocal 对窗口偏移后出界特征跳过
// （line2Dup.cpp:942 continue），与全图回卷语义不等值，不可切片全图响应；
// 窗格越 LM 行尾为 vendor LM flat 行尾续读复刻（相位回卷闭式），Mat 尾 UB
// 按 W2-A 先例取模确定化（实现见 coarse_extract.cpp compute_window）。
// B6 亚像素角度精修不实现：引擎 build_pose_adjacency 调用点已注释
//（matcher.cpp:240），adjacency 恒空，vendor 分支不可达；pose_offset 恒 0。
// 纯 C++ 宿主代码，无 CUDA 依赖；仅 VF_SHAPE_CUDA=ON 时随 GPU 后端编译。
#include "gpu_device.hpp"
#include <cstdint>
#include <string>
#include <vector>

#include "line2Dup.h"

namespace vfshape::gpu {

struct ExtractPoses {
    // 每 pose 两级模板（与上传 GPU 的驻留数据同源）：L1 特征驱动 initial 提取，
    // L0 特征+模板宽高驱动窗分。宿主侧自持有（引擎侧从 Model::detector.getTemplates 取）
    struct Level { int width = 0, height = 0; std::vector<FeatureRecord> feats; };
    std::vector<Level> level1, level0;  // 同下标同 pose
    // vendor matchClass 的 class_id 来源（matchPrepared 传入），参与 Match
    // 相等比较与去重；引擎为单类 "shape"
    std::string class_id = "shape";
};

struct ExtractResult {
    std::vector<line2Dup::Match> matches;  // 与 matchPrepared 输出同序同语义（sort+unique 后）
    double extract_ms = 0;                  // 宿主提取耗时（单列记录用）
    int initial_count = 0;                  // initialMatches 阶段候选数（诊断）
};

// initial 候选（clamp 后窗中心 + pose，与 extract_coarse_candidates 内部
// initial 阶段完全同序；GPU 批量评分 run_window_scores 的候选形态）
struct InitialCandidate {
    int x = 0, y = 0;  // clamp 后窗中心（L0 像素坐标，vendor matchClass 同式）
    int pose_id = 0;
};

// threshold：引擎粗层段传入的匹配阈（two_stage.cpp:192 的 max(45, T-20)）
// 消费 L1 全图响应 + L0 spread 图，复刻 matchClass 全链（initial→精化→过滤→sort+unique）。
// 预算超限（累计 initial 候选 >200000，vendor :1376 同式）= 清空+err 报错返回 false。
// external_scores 非空（W3）：跳过宿主窗分自算（LM 预烙+accumulate_window），
// 按 initial 候选序直接消费扁平窗分（run_window_scores 产物同布局：候选×256
// uint16 行主序窗格；值域 <32768 由特征数护栏保证）——先用
// extract_initial_candidates 取候选回填 GPU 窗分再传入。默认空 = 宿主自算，
// 纯宿主路径零变化。
// resp1_flat 非空（W6 波B 跨度消费口径）：指向 pose-major 扁平缓冲直读（每 pose
// resp_w1*resp_h1 格步进，格数=pose 数×网格），忽略 resp1 参数并免逐 pose 尺寸
// 校验（服务内部调用方契约：缓冲由 pinned 回读池保证）；默认空 = 原 vector 口径
// 零变化。
bool extract_coarse_candidates(
    const std::vector<std::vector<uint16_t>>& resp1, int resp_w1, int resp_h1,
    const std::vector<uint8_t>& spread0, int spread_w, int spread_h,
    const ExtractPoses& poses, int frame_w0, int frame_h0,
    float threshold, ExtractResult& out, std::string& err,
    const std::vector<uint16_t>* external_scores = nullptr,
    const uint16_t* resp1_flat = nullptr);

// initial 阶段独立出口：返回与主入口 initial 阶段同序同量的候选（含 clamp 后
// 窗中心），供提取方先回填外部窗分。预算/特征数护栏语义同主入口；不消费
// spread0（不做 spread 校验）。resp1_flat 口径同主入口。
bool extract_initial_candidates(
    const std::vector<std::vector<uint16_t>>& resp1, int resp_w1, int resp_h1,
    const ExtractPoses& poses, int frame_w0, int frame_h0, float threshold,
    std::vector<InitialCandidate>& out_cands, std::string& err,
    const uint16_t* resp1_flat = nullptr);

}  // namespace vfshape::gpu
