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

/* VisionForge 形状模板匹配 SDK —— 公开 C 接口（ABI v3）。
 *
 * 版本协商：任何其他调用之前先取 vf_shape_abi_version()，返回值非 3 时
 * 不得继续使用本接口。产品版本与 ABI 版本分属不同命名空间：产品版本由
 * 构建清单 runtime-manifest.json 的 sdk_version 字段声明，不与 ABI 数值
 * 强制相等。
 *
 * 错误分层（除 vf_shape_abi_version 外统一：该入口返回 ABI 版本号本身，
 * 无失败路径，不适用 0/1/2 约定）：
 * - 返回 0：成功。仅此时允许消费输出参数（句柄、count、config、size）。
 * - 返回 1：可报告失败（std::exception 族：参数非法、资源超限、解码失败等）。
 * - 返回 2：未分类原生失败。非 0 时的错误文本是尽力诊断信息，不构成
 *   可程序化的错误分类。
 * - error 缓冲由调用方提供：库在入口清空它，写入按容量截断并保证 NUL
 *   结尾；error 为 NULL 或 capacity 为 0 时不写文本，错误仍以返回码表达。
 *
 * 所有权与生命周期：
 * - 全部输入/输出缓冲由调用方分配并保证可写长度；库只在调用期间借用，
 *   返回后不保留任何调用方指针。
 * - 句柄仅来自返回 0 的 vf_shape_train / vf_shape_load；非零返回码后
 *   句柄输出无意义。活跃句柄上限 128 个，超限返回 1。
 * - vf_shape_close 对未知或已关闭句柄同样返回 0（幂等）。
 *
 * 并发：
 * - 同一句柄上的调用由库串行化；不同句柄可并行调用。
 * - 与在途调用并发 close 不产生悬垂访问等内存安全问题（模型寿命由内部
 *   所有权维持），但竞态中的在途调用可能以"句柄已关闭或未知"类错误
 *   返回 1：寿命安全不等于该调用业务成功，句柄自此失效。
 * - 匹配/训练期间库（含内部并行窗口）只读访问输入图像；调用返回之前
 *   调用方不得改写图像缓冲。
 *
 * 图像格式（train/match/batch 的 image 统一）：
 * - 连续 BGR 8bit 三通道；stride 必须等于 3×width，image_size 必须等于
 *   stride×height；宽高均须在 24..4096 像素。
 * - 坐标原点左上，x 向右 y 向下，单位像素。
 */

/* 训练配置。struct_size 必须先置为 sizeof(vf_shape_train_config)。
 * roi_x/roi_y/roi_width/roi_height：参考图坐标系内的训练 ROI，像素，
 *   ≥24px 且须完整落在图像内。
 * angle_min/angle_max/angle_step：角度，单位度；范围限制在 [-180,180]，
 *   step>0；正角与 OpenCV getRotationMatrix2D 同向（逆时针）；首尾按
 *   360° 周期等价时自动去重。
 * scale_min/scale_max/scale_step：缩放系数，范围限制在 [0.25,2]，step>0。
 * weak_th/strong_th：梯度幅度迟滞阈，正数且 weak≤strong≤1020。
 * feature_count：金字塔每层特征数上限，16..512。
 *   三者置 0 使用库默认：feature_count=128、weak_th=30、strong_th=60。
 * 角度×缩放的姿态总数受内部预算约束，超限训练返回 1。
 */
typedef struct {
    uint32_t struct_size;
    int32_t roi_x, roi_y, roi_width, roi_height;
    double angle_min, angle_max, angle_step;
    double scale_min, scale_max, scale_step;
    double weak_th, strong_th;
    int32_t feature_count;
} vf_shape_train_config;

/* 单个匹配结果，几何均位于输入帧坐标系，单位像素。
 *
 * score：0..100 的特征方向相似度（模板特征方向投票得分归一化到百分制），
 *   用于阈值过滤，不是校准过的概率。两级模型的 top1 候选经 B7 几何精修
 *   更新位姿几何，但 score/pose_id 保留离散匹配身份，不得解释为对精修后
 *   位姿重算的置信度。
 * pose_id：命中姿态在模型内的索引。单级模型为全模型姿态索引；两级容器
 *   模型为命中 fine 组内的局部索引，跨组不可比。
 * pose_ambiguous：0/1。1 表示该候选所在聚类曾合并姿态差超阈（角度差 ≥15°
 *   或缩放比偏离超过约 ±14%）且分数接近（分差 <10）的多解，目标身份不可
 *   唯一判定。
 * center[2]：训练 ROI 四角映射均值的落点；两级模型 top1 为 B7 亚像素精修
 *   后的中心。
 * angle：单位度，与训练角度同口径；两级 top1 为 B7 精修角，可小幅偏离
 *   离散档位。
 * scale：相对训练 ROI 的缩放系数。
 * corners[8]：训练 ROI 四角的映射落点，顺序为左上、右上、右下、左下，
 *   每角一对 (x,y)。
 * bbox[4]：四角外接框，x,y,w,h（左上角坐标与宽高）。
 * transform[6]：2×3 行优先仿射矩阵 [a b tx; c d ty]，把原始参考图坐标
 *   映射到输入帧坐标：x' = a·x + b·y + tx，y' = c·x + d·y + ty。参考图
 *   坐标是含 ROI 偏移的整幅参考图坐标，不是 ROI 局部坐标。
 * 坐标系基准即参考图口径：以 vf_shape_metadata 返回的参考尺寸为准——
 *   新容器模型记录原始参考图尺寸；旧格式容器无该记录，回退为模板灰图
 *   尺寸，此时参考图几何为派生值。
 */
typedef struct {
    double score;
    int32_t pose_id;
    uint32_t pose_ambiguous;
    double center[2], angle, scale, corners[8], bbox[4], transform[6];
} vf_shape_candidate;

/* 批量条目：vf_shape_match_batch 的输入输出，缓冲全部由调用方分配。
 * handle/threshold/candidates/capacity：语义同单模型匹配，capacity 须在
 *   2..256，threshold 须为 (0,100] 内的有限值。
 * count/status/error 为输出：status 0=成功，1/2 同全局错误分层；error
 *   固定 256 字节；count 仅在该条 status=0 时有效。
 */
typedef struct {
    uint64_t handle;
    double threshold;
    vf_shape_candidate* candidates;
    uint32_t capacity, count;
    int32_t status;
    char error[256];
} vf_shape_batch_item;

/* 返回 ABI 版本（当前为 3）。版本协商入口，无失败路径。 */
VF_SHAPE_API uint32_t vf_shape_abi_version(void);

/* 用参考图训练两级容器模型并注册句柄，成功后 *handle 为新句柄（入口先
 * 置 0）。库始终训练两级容器模型；单级模型只来自 vf_shape_load 的旧格式
 * 资产。图像合同见文件头注释。 */
VF_SHAPE_API int32_t vf_shape_train(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride, const vf_shape_train_config* config,
    uint64_t* handle, char* error, uint32_t error_capacity);

/* 加载模型资产并注册句柄：两级容器格式优先识别，非容器字节按旧单级格式
 * 解码，两者皆非返回 1；字节数不得超过 32MiB。成功后 *handle 为新句柄。 */
VF_SHAPE_API int32_t vf_shape_load(const uint8_t* bytes, uint64_t size,
    uint64_t* handle, char* error, uint32_t error_capacity);

/* 导出模型资产。两次调用模式：先以 bytes=NULL 且 capacity=0 调用查询
 * *size，再以不小于 *size 的缓冲取回；缓冲不足返回 1 且不写内容。
 * 两级模型导出为容器格式，单级模型导出为旧格式，均可被 vf_shape_load 回读。 */
VF_SHAPE_API int32_t vf_shape_export(uint64_t handle, uint8_t* bytes, uint64_t capacity,
    uint64_t* size, char* error, uint32_t error_capacity);

/* 查询模型训练配置与参考图尺寸（像素）。调用方须先把 config->struct_size
 * 置为 sizeof(vf_shape_train_config)。宽高为参考图坐标系口径，见候选结构注释。 */
VF_SHAPE_API int32_t vf_shape_metadata(uint64_t handle, vf_shape_train_config* config,
    int32_t* width, int32_t* height, char* error, uint32_t error_capacity);

/* 单模型匹配。threshold 为 (0,100] 内的有限值；score 达阈的聚类代表写入
 * candidates，实际数量写 *count（不超过 capacity）。capacity 须在 2..256；
 * *count 在容量校验之前置零，失败路径不承诺 candidates 内容。候选为聚类
 * 代表（簇内最高分），输出顺序无承诺；两级模型仅 top1 候选经几何精修。 */
VF_SHAPE_API int32_t vf_shape_match(uint64_t handle, const uint8_t* image,
    uint64_t image_size, int32_t width, int32_t height, uint64_t stride, double threshold,
    vf_shape_candidate* candidates, uint32_t capacity, uint32_t* count,
    char* error, uint32_t error_capacity);

/* 释放句柄。幂等：对未知或已关闭句柄同样返回 0。 */
VF_SHAPE_API int32_t vf_shape_close(uint64_t handle, char* error, uint32_t error_capacity);

/* 同一输入帧对多个模型匹配。item_count 须在 1..16，item_size 必须等于
 * sizeof(vf_shape_batch_item)。返回 0 仅表示批量流程完成，每个条目的
 * status 须单独检查；库先把全部条目的 count 置零、status 置 1、error 清空，
 * 再逐项执行。调用方必须先检查整体返回码、仅当为 0 时再消费逐项结果：
 * 除 items 为空、条数或条目尺寸不符外，公共图像校验失败与批量编排级
 * 资源异常（如缓冲分配失败）同样使整体返回非 0，此时逐项 status 只是
 * 初始化状态，不构成完整结果。 */
VF_SHAPE_API int32_t vf_shape_match_batch(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride, vf_shape_batch_item* items,
    uint32_t item_count, uint32_t item_size, char* error, uint32_t error_capacity);

#ifdef __cplusplus
}
#endif
