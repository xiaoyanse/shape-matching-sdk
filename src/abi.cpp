#include "model.hpp"
#include "two_stage.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
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

template<class Function>
int32_t boundary(char* error, uint32_t capacity, Function&& body) noexcept {
    if (error && capacity) error[0] = '\0';
    try { body(); return 0; }
    catch (const std::exception& ex) {
        if (error && capacity) {
            const auto length = std::min(size_t(capacity - 1), std::strlen(ex.what()));
            std::memcpy(error, ex.what(), length); error[length] = '\0';
        }
        return 1;
    } catch (...) {
        if (error && capacity) {
            const char* message = "unknown native matching failure";
            const auto length = std::min(size_t(capacity - 1), std::strlen(message));
            std::memcpy(error, message, length); error[length] = '\0';
        }
        return 2;
    }
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
        vfshape::require(count && candidates && capacity >= 2 && capacity <= 256, "candidate capacity must be 2..256");
        *count = 0;
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
            *width = model->two_stage->tmpl_gray.cols;
            *height = model->two_stage->tmpl_gray.rows;
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
                vfshape::require(!model->two_stage,
                    "two-stage models do not support batch; match individually");
                vfshape::validate_search_budget(*model->single, width, height);
                models[n] = std::move(model);
                any_valid = true;
            });
        }
        if (!any_valid) return;
        const auto frame = vfshape::prepare_frame(image_view);
        for (uint32_t n = 0; n < item_count; ++n) {
            if (!models[n]) continue;
            auto& item = items[n];
            item.status = boundary(item.error, sizeof(item.error), [&] {
                std::lock_guard<std::mutex> lock(models[n]->mutex);
                const auto found = vfshape::match_prepared(*models[n]->single, frame, item.threshold);
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
