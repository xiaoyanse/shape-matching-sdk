#include "model.hpp"
#include "two_stage.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

namespace {
std::mutex registry_mutex;

// ABI v3：句柄可为单级（旧资产）或两级容器模型。
struct ModelHolder {
    std::shared_ptr<vfshape::Model> single;            // 单级（旧格式资产）
    std::shared_ptr<vfshape::TwoStageModel> two_stage; // 两级容器（VF2S3）
    std::mutex mutex;
};
std::unordered_map<uint64_t, std::shared_ptr<ModelHolder>> registry;
uint64_t next_handle = 1;
constexpr size_t max_handles = 128;

void copy_error(char* dest, size_t dest_size, const char* message) {
    const auto length = std::min(dest_size - 1, std::strlen(message));
    std::memcpy(dest, message, length); dest[length] = '\0';
}

template<class Function>
int32_t boundary(char* error, uint32_t capacity, Function&& body) noexcept {
    if (error && capacity) error[0] = '\0';
    try { body(); return 0; }
    catch (const std::exception& ex) {
        if (error && capacity) copy_error(error, capacity, ex.what());
        return 1;
    } catch (...) {
        if (error && capacity) copy_error(error, capacity, "unknown native matching failure");
        return 2;
    }
}

// 批量备帧累计预算按填充后像素计价：每帧常驻约 5.75B/px（两层源图金字塔
// 3.75B/px、末层 8 方向线性内存 2B/px），匹配期惰性展开响应图再增约一半；
// item_count 上限 16 组同备 4096² 帧峰值可超 2GB。上限取 8 张 4096² 等效
// （填充后 4160²）把峰值压到约 1GB，超限组整组拒绝而非整批失败；两级项
// 不经共享帧不受此限。
constexpr uint64_t kMaxBatchPreparedPixels = 8ULL * 4160 * 4160;

uint64_t padded_frame_pixels(int width, int height) {
    const auto padded = [](int edge) {
        return uint64_t((edge + 2 * vfshape::PreparedFrame::padding + 15) / 16 * 16);
    };
    return padded(width) * padded(height);
}
std::shared_ptr<ModelHolder> get(uint64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto found = registry.find(handle);
    vfshape::require(found != registry.end(), "model handle is closed or unknown");
    return found->second;
}
void put(std::shared_ptr<ModelHolder> model, uint64_t* handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    vfshape::require(registry.size() < max_handles && next_handle != 0, "native handle budget exceeded");
    *handle = next_handle++; registry.emplace(*handle, std::move(model));
}
}

uint32_t vf_shape_abi_version() { return 3; }  // v3: 两级编排与几何精修全量下沉；训练默认两级打包
int32_t vf_shape_train(const uint8_t* image, uint64_t image_size, int32_t width, int32_t height,
    uint64_t stride, const vf_shape_train_config* config, uint64_t* handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(handle && config, "missing training output/config"); *handle = 0;
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        auto holder = std::make_shared<ModelHolder>();
        holder->two_stage = vfshape::train_two_stage(frame, *config);
        put(std::move(holder), handle);
    });
}
int32_t vf_shape_load(const uint8_t* bytes, uint64_t size, uint64_t* handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(handle, "missing model handle output"); *handle = 0;
        auto holder = std::make_shared<ModelHolder>();
        holder->two_stage = vfshape::decode_two_stage(bytes, size);
        if (!holder->two_stage) holder->single = vfshape::decode(bytes, size);
        vfshape::require(holder->single || holder->two_stage,
                         "model is neither two-stage container nor single-level format");
        put(std::move(holder), handle);
    });
}
int32_t vf_shape_export(uint64_t handle, uint8_t* bytes, uint64_t capacity, uint64_t* size,
    char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(size, "missing model size output");
        auto model = get(handle); std::lock_guard<std::mutex> lock(model->mutex);
        std::vector<uint8_t> encoded;
        if (model->two_stage)
            encoded = vfshape::encode_two_stage(*model->two_stage);
        else
            encoded = vfshape::encode(*model->single);
        *size = encoded.size();
        if (!bytes && capacity == 0) return;
        vfshape::require(bytes && capacity >= encoded.size(), "model output buffer too small");
        std::memcpy(bytes, encoded.data(), encoded.size());
    });
}
int32_t vf_shape_match(uint64_t handle, const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, double threshold, vf_shape_candidate* candidates,
    uint32_t capacity, uint32_t* count, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(count, "missing candidate count output");
        *count = 0;
        vfshape::require(candidates && capacity >= 2 && capacity <= 256, "candidate capacity must be 2..256");
        const auto frame = vfshape::checked_image(image, image_size, width, height, stride);
        auto model = get(handle); std::lock_guard<std::mutex> lock(model->mutex);
        std::vector<vf_shape_candidate> found;
        if (model->two_stage)
            found = vfshape::match_two_stage(*model->two_stage, frame, threshold, capacity);
        else
            found = vfshape::match(*model->single, frame, threshold);
        *count = uint32_t(std::min(size_t(capacity), found.size()));
        std::copy_n(found.begin(), *count, candidates);
    });
}
int32_t vf_shape_metadata(uint64_t handle, vf_shape_train_config* config, int32_t* width,
    int32_t* height, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        vfshape::require(config && width && height && config->struct_size == sizeof(*config),
                         "invalid metadata output ABI");
        auto model = get(handle);
        std::lock_guard<std::mutex> lock(model->mutex);
        if (model->two_stage) {
            *config = model->two_stage->base_config;
            // 参考尺寸须与 config 同处原图坐标系；旧容器 ref 字段为 0，回退模板灰图尺寸
            *width = model->two_stage->ref_width > 0
                         ? model->two_stage->ref_width : model->two_stage->tmpl_gray.cols;
            *height = model->two_stage->ref_height > 0
                          ? model->two_stage->ref_height : model->two_stage->tmpl_gray.rows;
        } else {
            *config = model->single->config; *width = model->single->width; *height = model->single->height;
        }
    });
}
int32_t vf_shape_match_batch(const uint8_t* image, uint64_t image_size, int32_t width,
    int32_t height, uint64_t stride, vf_shape_batch_item* items, uint32_t item_count,
    uint32_t item_size, char* error, uint32_t error_capacity) {
    return boundary(error, error_capacity, [&] {
        vfshape::require(items && item_count >= 1 && item_count <= 16 &&
            item_size == sizeof(vf_shape_batch_item), "invalid batch output ABI or count");
        for (uint32_t n = 0; n < item_count; ++n) {
            items[n].count = 0; items[n].status = 1; items[n].error[0] = '\0';
        }
        const auto image_view = vfshape::checked_image(image, image_size, width, height, stride);
        std::vector<std::shared_ptr<ModelHolder>> models(item_count);
        bool any_valid = false;
        for (uint32_t n = 0; n < item_count; ++n) {
            auto& item = items[n];
            item.status = boundary(item.error, sizeof(item.error), [&] {
                vfshape::require(item.candidates && item.capacity >= 2 && item.capacity <= 256,
                    "candidate capacity must be 2..256");
                vfshape::require(std::isfinite(item.threshold) && item.threshold > 0 && item.threshold <= 100,
                    "invalid similarity threshold");
                auto model = get(item.handle);
                // 搜索预算仅约束单级共享帧匹配；两级匹配内部自管缩图与窗口预算
                if (model->single)
                    vfshape::validate_search_budget(*model->single, width, height);
                models[n] = std::move(model);
                any_valid = true;
            });
        }
        if (!any_valid) return;
        // 单级帧量化与模板特征选择必须同参：按生效 detector 参数分组共帧，
        // 跨参共享 PreparedFrame 会触发 vendor 同参断言；两级匹配内部自行缩图，不共享。
        std::map<std::tuple<int, float, float>, vfshape::PreparedFrame> frames;
        std::set<std::tuple<int, float, float>> failed_groups;
        auto fail_group = [&](const std::tuple<int, float, float>& key, const char* message) {
            failed_groups.insert(key);
            for (uint32_t m = 0; m < item_count; ++m) {
                if (!models[m] || !models[m]->single) continue;
                const auto& detector = models[m]->single->detector;
                if (std::make_tuple(detector.tuned_features, detector.tuned_weak,
                                    detector.tuned_strong) != key)
                    continue;
                items[m].status = 1;
                copy_error(items[m].error, sizeof(items[m].error), message);
            }
        };
        const auto frame_pixels = padded_frame_pixels(width, height);
        uint64_t prepared_pixels = 0;
        for (uint32_t n = 0; n < item_count; ++n) {
            if (!models[n] || !models[n]->single) continue;
            const auto& detector = models[n]->single->detector;
            auto key = std::make_tuple(detector.tuned_features, detector.tuned_weak,
                                       detector.tuned_strong);
            if (frames.count(key) || failed_groups.count(key)) continue;
            if (prepared_pixels + frame_pixels > kMaxBatchPreparedPixels) {
                const auto message = "batch prepared-frame pixel budget exceeded: budget "
                    + std::to_string(kMaxBatchPreparedPixels) + " pixels, prepared "
                    + std::to_string(prepared_pixels) + " pixels, frame "
                    + std::to_string(frame_pixels) + " pixels";
                fail_group(key, message.c_str());
                continue;
            }
            try {
                // emplace 对已存在键仍会求值实参——先查再插，保证同参组只备一次帧
                frames.emplace(key, vfshape::prepare_frame(image_view, &detector));
            } catch (const std::exception& ex) {
                fail_group(key, ex.what());
                continue;
            } catch (...) {
                fail_group(key, "unknown native frame preparation failure");
                continue;
            }
            prepared_pixels += frame_pixels;
        }
        for (uint32_t n = 0; n < item_count; ++n) {
            if (!models[n]) continue;
            auto& item = items[n];
            if (models[n]->single) {
                const auto& detector = models[n]->single->detector;
                if (failed_groups.count(std::make_tuple(detector.tuned_features,
                                                        detector.tuned_weak,
                                                        detector.tuned_strong)))
                    continue;  // status/error 已在备帧隔离时标记
            }
            item.status = boundary(item.error, sizeof(item.error), [&] {
                std::lock_guard<std::mutex> lock(models[n]->mutex);
                std::vector<vf_shape_candidate> found;
                if (models[n]->two_stage) {
                    found = vfshape::match_two_stage(*models[n]->two_stage, image_view,
                                                     item.threshold, int(item.capacity));
                } else {
                    const auto& detector = models[n]->single->detector;
                    const auto& frame = frames.find(std::make_tuple(
                        detector.tuned_features, detector.tuned_weak, detector.tuned_strong))->second;
                    found = vfshape::match_prepared(*models[n]->single, frame, item.threshold);
                }
                item.count = uint32_t(std::min(size_t(item.capacity), found.size()));
                std::copy_n(found.begin(), item.count, item.candidates);
            });
        }
    });
}
int32_t vf_shape_close(uint64_t handle, char* error, uint32_t capacity) {
    return boundary(error, capacity, [&] {
        std::lock_guard<std::mutex> lock(registry_mutex);
        registry.erase(handle);
    });
}
