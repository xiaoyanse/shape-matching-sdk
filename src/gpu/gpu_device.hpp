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

// 上下文容量查询（Model 适配层预算校验用）
int context_pose_capacity(const GpuContext* ctx);

// 创建/销毁。容量口径同尖峰：max_w/max_h 为 L0 帧档（L1 自动减半），
// max_feats 为两层特征总数上限，max_candidates/max_positions 为精修窗/
// 响应图容量（W2/W3 内核工作面，缓冲随骨架一次分配）。
GpuContext* create_context(int max_w, int max_h, int max_poses, int max_feats,
                           int max_candidates, int max_positions, std::string& err);
void destroy_context(GpuContext* ctx);

// POD 特征驻留：feats/feat_total 为拼接后的特征缓冲；begin/count 为容量布局
//（长度 2*pose_capacity，前半 L1 slot=id、后半 L0 slot=pose_capacity+id）。
bool upload_features(GpuContext* ctx, const FeatureRecord* feats, int feat_total,
                     const int* begin, const int* count, int layout_len, int poses,
                     std::string& err);

// 特征驻留回读（一致性断言/诊断）：feats 为驻留量条数，begin/count 为容量
// 布局（2*pose_capacity 项，驻留 slot 外的值未定义）。
bool readback_features(const GpuContext* ctx, std::vector<FeatureRecord>& feats,
                       std::vector<int>& begin, std::vector<int>& count, std::string& err);

// 帧 H2D：padded BGR（引擎 prepare 面同式补边帧），尺寸须等于上下文 L0 档
bool upload_frame(GpuContext* ctx, const uint8_t* padded_bgr, int w, int h, std::string& err);

// 帧 D2H 回读（冒烟/诊断，与 upload_frame 同尺寸约束）
bool readback_frame(const GpuContext* ctx, uint8_t* padded_bgr, int w, int h,
                    std::string& err);

// LUT 同源驻留：宿主表（gen_lut.py 产物）上传设备 __constant__ 后回读
bool lut_upload(const unsigned char host_lut[256], std::string& err);
bool lut_readback(unsigned char host_lut[256], std::string& err);

}  // namespace vfshape::gpu
