// Model 级驻留适配（W1）：遍历引擎 Detector 模板（class_templates "shape"）
// 转设备面 POD 布局。CUDA API 细节在 gpu_device.cu，本文件只做取数与拼接，
// 两侧可独立演进（W2 内核面接入时无需动引擎模型读取逻辑）。
#include "gpu_backend.hpp"

#include <vector>

namespace vfshape::gpu {

bool upload_model_features(GpuContext* ctx, const Model& model, std::string& err,
                           ResidencyStats* stats, int model_slot, int frame_slot) {
    if (!ctx) {
        err = "null context";
        return false;
    }
    const auto& poses = model.poses;
    const int n = int(poses.size());
    const int capacity = context_pose_capacity(ctx);
    if (n <= 0 || n > capacity) {
        err = "pose count out of capacity";
        return false;
    }
    if (n != model.detector.numTemplates("shape")) {
        err = "pose/template count mismatch";
        return false;
    }
    std::vector<FeatureRecord> feats;
    std::vector<int> begin, count, widths, heights;
    int total_l1 = 0, total_l0 = 0;
    begin.assign(size_t(capacity) * 2, 0);
    count.assign(size_t(capacity) * 2, 0);
    widths.assign(size_t(capacity) * 2, 1);
    heights.assign(size_t(capacity) * 2, 1);
    for (int level = 1; level >= 0; --level) {  // 先 L1 后 L0 拼接（尖峰布局）
        for (int id = 0; id < n; ++id) {
            const auto& pyramid = model.detector.getTemplates("shape", id);
            if (pyramid.size() != 2 || pyramid[size_t(level)].pyramid_level != level) {
                err = "template pyramid layout mismatch";
                return false;
            }
            const auto& features = pyramid[size_t(level)].features;
            const int slot = level == 1 ? id : capacity + id;
            begin[size_t(slot)] = int(feats.size());
            count[size_t(slot)] = int(features.size());
            widths[size_t(slot)] = pyramid[size_t(level)].width;
            heights[size_t(slot)] = pyramid[size_t(level)].height;
            for (const auto& f : features) feats.push_back({f.x, f.y, f.label});
            (level ? total_l1 : total_l0) += int(features.size());
        }
    }
    if (!upload_features(ctx, feats.data(), int(feats.size()), begin.data(), count.data(),
                         int(begin.size()), n, err, model_slot, frame_slot))
        return false;
    if (!upload_template_dims(ctx, widths.data(), heights.data(), int(widths.size()), err,
                              model_slot, frame_slot))
        return false;
    if (stats) {
        stats->pose_count = n;
        stats->features_level0 = total_l0;
        stats->features_level1 = total_l1;
        stats->feature_total = total_l0 + total_l1;
    }
    return true;
}

}  // namespace vfshape::gpu
