#pragma once
// GPU 设备面（W1 骨架的 CUDA 中立子集）：不依赖引擎头（model.hpp/OpenCV/
// vendor），只做设备上下文（容量制缓冲）+ POD 特征驻留 + 帧 H2D + LUT 驻留。
// 实现见 gpu_device.cu（仅 VF_SHAPE_CUDA=ON 时编译；.cu 保持无 OpenCV 头，
// nvcc 坑面最小）。Model 级驻留在 gpu_backend.hpp 适配层。
#include <cstdint>
#include <string>
#include <vector>

namespace vfshape::gpu {

// 不透明设备上下文：两级帧工作面/特征驻留/响应窗缓冲按容量一次性分配
//（沿尖峰 gpu_create_context 形态，基准循环内复用）
struct GpuContext;

// 驻留特征单条记录（设备侧 int4 布局的宿主视图；theta 不参与响应计算，不驻留）
struct FeatureRecord {
    int32_t x = 0, y = 0, label = 0;
};

// 探测设备名与算力（支持集 {sm_86, sm_89, sm_120} 的校验归调用方/回退层 W4）
bool device_info(std::string& name, int& major, int& minor, std::string& err);

// 上下文容量查询（Model 适配层预算校验/窗分分块步长用）
int context_pose_capacity(const GpuContext* ctx);
int context_candidate_capacity(const GpuContext* ctx);

// 创建/销毁。容量口径同尖峰：max_w/max_h 为 L0 帧档（L1 自动减半），
// max_feats 为两层特征总数上限，max_candidates 为精修窗容量（W3），
// max_positions 为单 pose 响应格数容量——W2 粗层按 L0 全图口径
//（(max_w/4)*(max_h/4)，L1 复用同一缓冲前缀）。
GpuContext* create_context(int max_w, int max_h, int max_poses, int max_feats,
                           int max_candidates, int max_positions, std::string& err);
void destroy_context(GpuContext* ctx);

// POD 特征驻留：feats/feat_total 为拼接后的特征缓冲；begin/count 为容量布局
//（长度 2*pose_capacity，前半 L1 slot=id、后半 L0 slot=pose_capacity+id）。
// model_slot=0/1 选模型驻留槽（0=coarse，1=fine 窗），fine 槽按 frame_slot 双份
//（流水化双 job 并存），默认 0/0 沿 W2 行为。
bool upload_features(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                     const int* begin, const int* count, int layout_len, int poses,
                     std::string& err, int model_slot = 0, int frame_slot = 0);

// 模板尺寸驻留（W2）：每 pose 每层模板 (width,height)，slot 布局同 begin/count
//（长度 2*pose_capacity）。vendor 相似度的 span 写入边界（wf/hf）与线性内存
// 出界回卷语义依赖模板宽高——粗层内核 run_coarse_chain 前置要求已驻留。
bool upload_template_dims(GpuContext* ctx, const int* widths, const int* heights, int len,
                          std::string& err, int model_slot = 0, int frame_slot = 0);

// 特征驻留回读（一致性断言/诊断）：feats 为驻留量条数，begin/count 为容量
// 布局（2*pose_capacity 项，驻留 slot 外的值未定义）。
bool readback_features(const GpuContext* ctx, std::vector<FeatureRecord>& feats,
                       std::vector<int>& begin, std::vector<int>& count, std::string& err);

// 帧 H2D：padded BGR（引擎 prepare 面同式补边帧），尺寸须等于上下文 L0 档
bool upload_frame(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h, std::string& err);

// 帧 H2D 容量制（W3 R4）：帧档 ≤ 上下文 L0 容量即接受（帧档变化免重建上下文），
// 当前档记入 ctx（按 frame_slot）供链路/窗分内核按当前档启动；写入区域外陈旧
// 数据不参与计算。严格等档版 upload_frame 保留（W1 冒烟负路径契约）。
bool upload_frame_capacity(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h,
                           std::string& err, int frame_slot = 0);

// —— 流内异步上传面（W3 流水化）：与同步版同校验同目标缓冲，拷贝入指定流，
// 完成由流序/事件门控（无独立同步点）；data 须保持可读至流内拷贝完成。
// 流/事件以 opaque 句柄传递（中立面不引入 CUDA 头，实现在 gpu_device.cu）——
bool upload_frame_capacity_async(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h,
                                 void* stream, std::string& err, int frame_slot = 0);
bool upload_features_async(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                           const int* begin, const int* count, int layout_len, int poses,
                           int model_slot, void* stream, std::string& err, int frame_slot = 0);
bool upload_template_dims_async(GpuContext* ctx, const int* widths, const int* heights, int len,
                                int model_slot, void* stream, std::string& err,
                                int frame_slot = 0);

// —— 流/事件/pinned 薄包装（W3 流水化服务层编排用；CUDA API 不出 .cu）——
void* stream_create(std::string& err);
void stream_destroy(void* stream);
void* event_create(std::string& err);
void event_destroy(void* event);
void event_record(void* event, void* stream);
bool event_sync(void* event, std::string& err);
double event_elapsed_ms(void* begin_ev, void* end_ev, bool& ok);
void* pinned_alloc(size_t bytes, std::string& err);
void pinned_free(void* ptr);

// 帧 D2H 回读（冒烟/诊断，与 upload_frame 同尺寸约束）
bool readback_frame(const GpuContext* ctx, uint8_t* padded_bgr, int w, int h,
                    std::string& err);

// LUT 同源驻留：宿主表（gen_lut.py 产物）上传设备 __constant__ 后回读
bool lut_upload(const unsigned char host_lut[256], std::string& err);
bool lut_readback(unsigned char host_lut[256], std::string& err);

}  // namespace vfshape::gpu
