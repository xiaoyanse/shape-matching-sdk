#pragma once
#include <stdint.h>

#ifdef _WIN32
#define VF_SHAPE_API __declspec(dllexport)
#else
#define VF_SHAPE_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t struct_size;
    int32_t roi_x, roi_y, roi_width, roi_height;
    double angle_min, angle_max, angle_step;
    double scale_min, scale_max, scale_step;
    /* 特征筛选参数（0=库默认 128/30/60）。weak/strong=梯度幅度迟滞阈，
       feature_count=每层特征数上限。ABI v2 起。 */
    double weak_th, strong_th;
    int32_t feature_count;
} vf_shape_train_config;

typedef struct {
    double score;
    int32_t pose_id;
    uint32_t pose_ambiguous;
    double center[2], angle, scale, corners[8], bbox[4], transform[6];
} vf_shape_candidate;

typedef struct {
    uint64_t handle;
    double threshold;
    vf_shape_candidate* candidates;
    uint32_t capacity, count;
    int32_t status;
    char error[256];
} vf_shape_batch_item;

VF_SHAPE_API uint32_t vf_shape_abi_version(void);
VF_SHAPE_API int32_t vf_shape_train(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride, const vf_shape_train_config* config,
    uint64_t* handle, char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_load(const uint8_t* bytes, uint64_t size,
    uint64_t* handle, char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_export(uint64_t handle, uint8_t* bytes, uint64_t capacity,
    uint64_t* size, char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_metadata(uint64_t handle, vf_shape_train_config* config,
    int32_t* width, int32_t* height, char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_match(uint64_t handle, const uint8_t* image,
    uint64_t image_size, int32_t width, int32_t height, uint64_t stride, double threshold,
    vf_shape_candidate* candidates, uint32_t capacity, uint32_t* count,
    char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_close(uint64_t handle, char* error, uint32_t error_capacity);
VF_SHAPE_API int32_t vf_shape_match_batch(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride, vf_shape_batch_item* items,
    uint32_t item_count, uint32_t item_size, char* error, uint32_t error_capacity);

#ifdef __cplusplus
}
#endif
