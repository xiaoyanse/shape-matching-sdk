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
 * - 错误码：返回 1 的可报告失败大多附带稳定错误码（vf_shape_error_code）
 *   与数值上下文，经 vf_shape_last_error_info 按线程查询；错误文本合同
 *   不变，码与文本并存。返回 2 与未覆盖的失败按 INTERNAL 归类。
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

/* 稳定错误码：与错误文本并存，供程序化分类与人话映射。语义为"哪类约束
 * 被违反"，命名与既有错误文本关键词一一挂钩；未覆盖的失败按 INTERNAL。
 * 码值经由 vf_shape_last_error_info 查询（线程局部，见该入口注释）。 */
typedef enum {
    VF_SHAPE_OK = 0,
    VF_SHAPE_INVALID_IMAGE = 1,
    VF_SHAPE_INVALID_ROI = 2,
    VF_SHAPE_INVALID_ANGLE_RANGE = 3,
    VF_SHAPE_INVALID_SCALE_RANGE = 4,
    VF_SHAPE_POSE_BUDGET = 5,
    VF_SHAPE_PIXEL_BUDGET = 6,
    VF_SHAPE_CANVAS_BUDGET = 7,
    VF_SHAPE_DETECTOR_PARAMS = 8,
    VF_SHAPE_THRESHOLD_INVALID = 9,
    VF_SHAPE_CAPACITY_INVALID = 10,
    VF_SHAPE_HANDLE_INVALID = 11,
    VF_SHAPE_HANDLE_BUDGET = 12,
    VF_SHAPE_MODEL_FORMAT = 13,
    VF_SHAPE_MODEL_SIZE = 14,
    VF_SHAPE_MODEL_TAMPER = 15,
    VF_SHAPE_SEARCH_BUDGET = 16,
    VF_SHAPE_INTERNAL = 99
} vf_shape_error_code;

/* 训练预算账本（vf_shape_estimate_training 输出）。
 * formula_version：预算公式版本，当前恒为 1；核算公式变化时递增。
 * pose_used/pose_limit：角度×缩放档位总数与其上限（350）。
 * canvas_used/canvas_limit：缩放后最大画布单边像素与其上限（4096）。
 * pixel_used/pixel_limit：全部姿态画布像素总量与其上限（2.5e8）。
 * pose_over/canvas_over/pixel_over：0/1，该账本是否超限；超限账本仍报告
 *   实际用量。over_budget：首要超限因子，判定顺序与训练拒绝顺序一致
 *   （POSE_BUDGET → CANVAS_BUDGET → PIXEL_BUDGET），未超限为 VF_SHAPE_OK。
 */
typedef struct {
    uint32_t struct_size;
    uint32_t formula_version;
    uint32_t pose_used, pose_limit, pose_over;
    uint32_t canvas_used, canvas_limit, canvas_over;
    uint32_t pixel_over;
    uint64_t pixel_used, pixel_limit;
    int32_t over_budget;
} vf_shape_budget_report;

/* 最近一次匹配的诊断快照（vf_shape_last_match_info 输出）。
 * coarse_candidates：两级模型粗层命中数（预算截断前）；windows_searched：
 *   实际执行精修的窗口数；window_best_scores：各精修窗最佳分（槽位上限
 *   8，未执行窗为 0）；final_candidates：聚类过滤后的最终候选数（输出
 *   容量截断前）。计数常开、零开销。
 * coarse_ms/coarse_resize_ms/windows_ms/finish_ms：分阶段耗时（毫秒），
 *   仅诊断计时构建（VF_SHAPE_PROFILE_STAGES）填充，常规构建恒为 0。
 * 单级模型无两级编排：仅 final_candidates 有效，其余为 0。句柄尚未执行
 * 过匹配时全部字段为 0（struct_size 回显除外）。
 */
typedef struct {
    uint32_t struct_size;
    uint32_t coarse_candidates, windows_searched, final_candidates;
    double window_best_scores[8];
    double coarse_ms, coarse_resize_ms, windows_ms, finish_ms;
} vf_shape_match_trace;

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

/* 查询本线程最近一次失败调用的错误码与数值参数。code/params/param_count
 * 均可为 NULL 跳过对应输出；params 前 param_count（0..6）项有效，语义随
 * 错误码与检查点不同（如角度步长过细时依次为：当前步长、档数上限、建议
 * 步长）。成功调用不清除上次记录（跨调用仍可查询）；记录每线程独立，
 * 跨线程不可见；本线程无失败历史时 code 为 VF_SHAPE_OK。 */
VF_SHAPE_API void vf_shape_last_error_info(int32_t* code, double params[6],
    int32_t* param_count);

/* 训练预算查询：按训练同一公式核算配置将占用的姿态数、最大画布与像素
 * 总量。纯计算，不建句柄不碰图像。config->struct_size 与 out->struct_size
 * 须先置为 sizeof。配置域检查（结构尺寸/Detector 三参/ROI/角度缩放范围）
 * 与训练同判：不合法返回 1 并写错误文本；合法配置返回 0，*out 报告用量
 * 与首要超限因子（超限本身不算查询失败，见 vf_shape_budget_report）。 */
VF_SHAPE_API int32_t vf_shape_estimate_training(const vf_shape_train_config* config,
    int32_t image_w, int32_t image_h, vf_shape_budget_report* out,
    char* error, uint32_t error_capacity);

/* 查询句柄最近一次 match/match_batch 的诊断快照。非阻塞快照：句柄锁下
 * 读取，与在途匹配串行。out->struct_size 须先置为 sizeof，返回后库按其
 * 回显。返回码分层同文件头。 */
VF_SHAPE_API int32_t vf_shape_last_match_info(uint64_t handle,
    vf_shape_match_trace* out, char* error, uint32_t error_capacity);

#ifdef __cplusplus
}
#endif
