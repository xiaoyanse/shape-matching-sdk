#ifndef GPU_SPIKE_MODEL_SOURCE_H
#define GPU_SPIKE_MODEL_SOURCE_H
// 数据源：VF2S3 两级容器解码 + fact 帧读取/缩放/补边（引擎零修改，独立解码）。
#include "common.h"

namespace gpuspike {

struct ModelSet {
    std::vector<PoseModel> coarse;
    std::vector<std::vector<PoseModel>> fines;  // 每角度组
    std::vector<double> group_lo, group_hi;     // 组界（由各组首末位姿角推导）
    int k = 2;
    double weak = 30, strong = 60;
    int coarse_roi[4] = {0, 0, 0, 0};  // 粗段训练 ROI（缩图坐标系，k 缩放后）
    int fine_roi[4] = {0, 0, 0, 0};    // 精修段训练 ROI（参考图原坐标）
};

struct FrameData {
    std::vector<uint8_t> bgr;  // 原始帧（BGR 连续）
    int w = 0, h = 0;
};

// 读容器文件（train_model 导出的原始字节）；失败返回 false 并置 err
bool load_model_set(const char* path, ModelSet& out, std::string& err);
// 读 PNG/JPG 帧；target_w/h=0 表示原始尺寸，否则双线性缩放（1080P 基准口径）
bool load_frame(const char* path, int target_w, int target_h, FrameData& out, std::string& err);
// 整数因子缩小（cv INTER_AREA 语义由 CPU 侧 cv::resize 承担）
bool shrink_frame(const FrameData& src, int factor, FrameData& out, std::string& err);
// 引擎 prepare_frame 同式补边：32px 零边 + 16 对齐
void pad_frame(const FrameData& src, FrameData& out);
// 引擎 refine_window 同式取窗（half 由调用侧给出）
bool crop_window(const FrameData& src, double cx, double cy, double half, FrameData& out);
// 组选择（fine_for_angle 同式：覆盖最多的一组）
int pick_fine_group(const ModelSet& model, double angle);

}  // namespace gpuspike

#endif
