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

/* 多边形顶点与轮廓：坐标为参考图（原图）坐标系，单位像素，原点左上。
 * 顶点数组由调用方分配，库只在调用期间借用。 */
typedef struct {
    double x, y;
} vf_shape_vertex;

typedef struct {
    const vf_shape_vertex* vertices;
    uint32_t count;
} vf_shape_polygon;

/* 多边形训练参数（独立结构，vf_shape_train_config 不动）。
 * struct_size 必须先置为 sizeof(vf_shape_polygon_train_params)。
 * outline：外环 3..64 顶点，任意绕向（SDK 归一为固定绕向后持久化与回读）；
 *   顶点须落在参考图坐标域 [0,width]×[0,height] 内，轮廓不得自交或退化
 *   （有向面积非零）。
 * holes/hole_count：洞 0..8 个，每洞 3..64 顶点、面积 ≥64px²、须完全在外环
 *   内（含顶点不得触及外环边界）；洞语义=训练特征提取期掩蔽（洞内像素不产
 *   特征），洞间重叠不校验（按并集掩蔽）。
 * 外环外接矩形须满足训练 ROI 同款 24..4096 像素合同，并作为姿态/画布/像素
 *   预算的记账基（与矩形训练同式同序）；短边 <48px 时两级粗层无法满足最小
 *   ROI 合同，训练将以错误拒绝（与矩形训练同一行为）。
 * angle/scale/weak_th/strong_th/feature_count：与 vf_shape_train_config 同名
 *   字段语义一致（0=默认）。 */
typedef struct {
    uint32_t struct_size;
    vf_shape_polygon outline;
    const vf_shape_polygon* holes;
    uint32_t hole_count;
    double angle_min, angle_max, angle_step;
    double scale_min, scale_max, scale_step;
    double weak_th, strong_th;
    int32_t feature_count;
} vf_shape_polygon_train_params;

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
    /* 多边形训练参数几何非法（vf_shape_train_polygon 与容器 poly 段加载校验
     * 共用）。params 依次为：失败细分类型、主参数、次参数。细分类型：
     * 1=外环顶点数越界 3..64（主=实际数，次=上限）；
     * 2=顶点坐标非有限或越参考图坐标域（主=顶点全局下标，外环在前、
     *   各洞顶点依序接续，从 0 起）；
     * 3=外环退化（有向面积近零，主=面积绝对值）；
     * 4=外环自交或重入（主/次=相交边下标，闭边计最后一条）；
     * 5=洞数越界 ≤8（主=实际数）；
     * 6=洞几何非法（主=洞下标，次=细分类型，复用 1/2/3/4 语义）；
     * 7=洞面积不足 64px²（主=洞下标，次=实际面积）；
     * 8=洞未完全在外环内（主=洞下标）；
     * 9=外接矩形违反 24..4096 合同（主=宽，次=高）。 */
    VF_SHAPE_INVALID_POLYGON = 17,
    /* 内容类训练失败：参考 ROI 内边缘特征不足，姿态模板数为空（params：
     * 特征数下限 4）。 */
    VF_SHAPE_INSUFFICIENT_FEATURES = 18,
    /* 两级容器粗层训练在全部 K 档失败（params：实际尝试的 K 档数）。 */
    VF_SHAPE_COARSE_TRAIN_FAILED = 19,
    /* 带版本戳模型与当前运行时 OpenCV 三段版本不等（params 依次为：模型
     * major/minor/revision、运行时 major/minor/revision）。分级合同：无戳
     * 模型（≤v0.4.0 产物）放行+一次性告警，仅带戳模型严格拒载。 */
    VF_SHAPE_MODEL_OPENCV_MISMATCH = 20,
    VF_SHAPE_INTERNAL = 99
} vf_shape_error_code;

/* 训练预算账本（vf_shape_estimate_training / _ex 输出）。版本演进规则：
 * 尾部加性追加 + formula_version 递增，既有字段语义与偏移永不变化。
 * formula_version：预算公式版本。矩形核算路径恒为 1（含 _ex 矩形路径，与
 *   旧入口逐字段一致）；仅 _ex 多边形路径报 2，消费方按版本分支。
 * pose_used/pose_limit：角度×缩放档位总数与其上限（350）。
 * canvas_used/canvas_limit：缩放后最大画布单边像素与其上限（4096）。
 * pixel_used/pixel_limit：全部姿态画布像素总量与其上限（2.5e8）。
 * pose_over/canvas_over/pixel_over：0/1，该账本是否超限；超限账本仍报告
 *   实际用量。over_budget：首要超限因子，判定顺序与训练拒绝顺序一致
 *   （POSE_BUDGET → CANVAS_BUDGET → PIXEL_BUDGET），未超限为 VF_SHAPE_OK。
 * polygon_fill_ratio（v2 尾部追加，formula_version=2 时有效）：多边形掩码
 *   面积/外接矩形面积比（外环面积−各洞面积绝对值直减，洞间重叠不消重，
 *   建议性密度提示非合同值）；矩形路径=1.0。struct_size 兼容两种布局：
 *   v1（追加前，64 字节，库只回写前缀字段）与 v2（追加后，72 字节）。
 */
typedef struct {
    uint32_t struct_size;
    uint32_t formula_version;
    uint32_t pose_used, pose_limit, pose_over;
    uint32_t canvas_used, canvas_limit, canvas_over;
    uint32_t pixel_over;
    uint64_t pixel_used, pixel_limit;
    int32_t over_budget;
    double polygon_fill_ratio;
} vf_shape_budget_report;

/* 最近一次匹配的诊断快照（vf_shape_last_match_info 输出）。
 * coarse_candidates：两级模型粗层命中数（预算截断前）；windows_searched：
 *   实际执行精修的窗口数；window_best_scores：各精修窗最佳分（槽位上限
 *   8，未执行窗为 0）；final_candidates：聚类过滤后的最终候选数（输出
 *   容量截断前）。计数常开、零开销。
 * coarse_ms/coarse_resize_ms/windows_ms/finish_ms：分阶段耗时（毫秒），
 *   仅诊断计时构建（VF_SHAPE_PROFILE_STAGES）填充，常规构建恒为 0。
 * window_collect_best_scores（v2 尾部追加）：仅记录实际执行救分收集的窗口，
 *   值为重整前最高原始分。未尝试救分的窗填 0（不是实测低分）；尝试后无候选也为 0。
 * rescued_candidates（v2）：实际救分中低于判定阈且重整后过阈的候选数；未尝试为 0。
 * 单级模型无两级编排：仅 final_candidates 有效，其余为 0。句柄尚未执行
 * 过匹配时全部字段为 0（struct_size 回显除外）。
 * struct_size 兼容 v1（追加前 112 字节，库只回写前缀字段）与 v2 两种布局。
 */
typedef struct {
    uint32_t struct_size;
    uint32_t coarse_candidates, windows_searched, final_candidates;
    double window_best_scores[8];
    double coarse_ms, coarse_resize_ms, windows_ms, finish_ms;
    double window_collect_best_scores[8];
    int32_t rescued_candidates;
} vf_shape_match_trace;

/* 模型体检结论分级（vf_shape_model_health 输出 verdict），取更重原则：
 * - WEAK=2：实测自匹配分 <70，或任一层特征数触训练下限（≤4）；
 * - DEGRADED=1：未达 WEAK，且实测自匹配分 <90，或任一层特征数 <8；
 * - HEALTHY=0：其余。
 * 单级旧格式资产无容器模板图（self_score_valid=0），自匹配项不参与分级，
 * 仅按特征数判定。 */
typedef enum {
    VF_SHAPE_HEALTH_HEALTHY = 0,
    VF_SHAPE_HEALTH_DEGRADED = 1,
    VF_SHAPE_HEALTH_WEAK = 2
} vf_shape_health_verdict;

/* 零输入体检报告（vf_shape_model_health 输出）。struct_size 须先置为
 * sizeof 并由库回显。verdict 分级合同见 vf_shape_health_verdict。
 * pose_count：姿态模板总数（两级容器为各 fine 组之和，含组间 8° 重叠
 *   复用姿态；单级为全模型姿态数）。
 * fine_min/fine_max_features：fine 侧（单级即该模型）各金字塔层特征数
 *   的最小/最大；coarse_features：粗层各层特征数最小值，单级恒 0。
 * detector_features/weak/strong：生效 detector 三参快照。
 * self_score：容器内模板灰图渲染 BGR 后对模型自身匹配一次（阈值 70，
 *   只取最高分）的最高分；无候选=0（等价低于 70）。self_score_valid=0
 *   表示未实测（单级资产）。
 * template_width/height：两级=容器模板灰图尺寸；单级=首姿态最细层模板
 *   尺寸。 */
typedef struct {
    uint32_t struct_size;
    int32_t verdict;
    uint32_t pose_count;
    uint32_t fine_min_features, fine_max_features, coarse_features;
    int32_t detector_features;
    double detector_weak, detector_strong;
    double self_score;
    uint32_t self_score_valid;
    uint32_t template_width, template_height;
} vf_shape_health_report;

/* 区分度审计结论（vf_shape_audit_discriminability 输出 flag）。
 * RECOMMENDED=0：推荐区间有效；INSEPARABLE=1：lo>hi，正负分数重叠，
 *   区间字段仅为计算中间值（lo>hi），不得消费；NO_NEGATIVE=2：负组未
 *   提供，只报正组分布，区间字段为 0。 */
typedef enum {
    VF_SHAPE_AUDIT_RECOMMENDED = 0,
    VF_SHAPE_AUDIT_INSEPARABLE = 1,
    VF_SHAPE_AUDIT_NO_NEGATIVE = 2
} vf_shape_audit_flag;

/* 推荐区间置信度：HIGH=0（两组有效分数样本均 ≥5）；LOW=1（任一组样本
 * <5，含零命中导致的样本不足）。 */
typedef enum {
    VF_SHAPE_AUDIT_CONFIDENCE_HIGH = 0,
    VF_SHAPE_AUDIT_CONFIDENCE_LOW = 1
} vf_shape_audit_confidence;

/* 区分度审计报告（vf_shape_audit_discriminability 输出）。struct_size
 * 须先置为 sizeof 并由库回显。
 * 逐帧取现有匹配路径输出的最高分构成组内分布（count=有分数帧数，
 * zero_hits=零命中帧数，零命中帧不进分布；两级匹配内部粗/精门槛以下
 * 的帧计为零命中，属诚实口径而非缺陷）；min/p25/median/p75/max 为
 * 线性插值分位（q·(n−1) 插值）。
 * separation=pos_p25−neg_max（可负）。推荐区间（建议性，advisory，非
 * 合同承诺）：lo=max(30, neg_max+5)、hi=min(85, pos_p25−5)；lo>hi 时
 * flag=INSEPARABLE，只消费各自分布。负组可选（neg_count=0 时只报正组
 * 分布，flag=NO_NEGATIVE，不推荐区间）。 */
typedef struct {
    uint32_t struct_size;
    int32_t flag;
    int32_t confidence;
    uint32_t pos_count, pos_zero_hits;
    double pos_min, pos_p25, pos_median, pos_p75, pos_max;
    uint32_t neg_count, neg_zero_hits;
    double neg_min, neg_p25, neg_median, neg_p75, neg_max;
    double separation, recommend_lo, recommend_hi;
} vf_shape_audit_report;

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

/* 当前运行时链接的 OpenCV 三段版本（major/minor/revision），out_version[3]
 * 由库回写。无失败路径。 */
VF_SHAPE_API void vf_shape_runtime_opencv_version(int32_t out_version[3]);

/* 模型字节（VFSHAPE 单级模型段；两级容器的 fine/coarse 段同构）的 OpenCV
 * 版本戳查询。*has_stamp 回写 0（无戳=戳机制前产物，out_version 置零）或
 * 1（带戳，out_version 回写训练时版本）。bytes 非法（签名/结构损坏）返回 1
 * 并写错误文本，语义与装载校验同一口径；本查询不执行版本拦截（只读）。 */
VF_SHAPE_API int32_t vf_shape_model_opencv_version(const uint8_t* bytes, uint64_t size,
    int32_t out_version[3], int32_t* has_stamp, char* error, uint32_t error_capacity);

/* 训练预算查询：按训练同一公式核算配置将占用的姿态数、最大画布与像素
 * 总量。纯计算，不建句柄不碰图像。config->struct_size 与 out->struct_size
 * 须先置为 sizeof。配置域检查（结构尺寸/Detector 三参/ROI/角度缩放范围）
 * 与训练同判：不合法返回 1 并写错误文本；合法配置返回 0，*out 报告用量
 * 与首要超限因子（超限本身不算查询失败，见 vf_shape_budget_report）。 */
VF_SHAPE_API int32_t vf_shape_estimate_training(const vf_shape_train_config* config,
    int32_t image_w, int32_t image_h, vf_shape_budget_report* out,
    char* error, uint32_t error_capacity);

/* 多边形训练预算查询（_ex）：params=NULL 时与 vf_shape_estimate_training
 * 完全一致（formula_version=1、逐字段相等）；params 非空时按多边形路径核算
 * ——与 vf_shape_train_polygon 同参语义（params 全量字段为准，config 仅承担
 * 结构校验不参与核算）：validate_polygon 几何校验（INVALID_POLYGON 复用
 * 细分）、外接矩形记账（与矩形训练同式同序）、formula_version=2 且
 * polygon_fill_ratio 有效。out->struct_size 接受 v1/v2 两种布局，库按其
 * 只回写前缀字段。 */
VF_SHAPE_API int32_t vf_shape_estimate_training_ex(const vf_shape_train_config* config,
    int32_t image_w, int32_t image_h, const vf_shape_polygon_train_params* params,
    vf_shape_budget_report* out, char* error, uint32_t error_capacity);

/* 多边形模板训练：与 vf_shape_train 同链路（两级容器模型），仅把训练掩码从
 * 矩形 ROI 换为多边形（外环填充、洞掩蔽），位姿映射/预算/导出/加载/匹配
 * 全部复用矩形路径（外接矩形即 ROI）。成功后 *handle 为新句柄；导出为容器
 * 格式，meta 段携带可选 poly 键（归一化外环顶点+洞分段+外接矩形）；旧版本
 * 库加载该容器时忽略 poly 键、按外接矩形回退解读（兼容承诺）。图像合同见
 * 文件头注释。 */
VF_SHAPE_API int32_t vf_shape_train_polygon(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride,
    const vf_shape_polygon_train_params* params, uint64_t* handle,
    char* error, uint32_t error_capacity);

/* 自动轮廓提取建议：Canny→外轮廓→多边形逼近，按面积降序输出前
 * max_suggestions 条建议（固定阈值，无随机：同图两次调用输出逐字节一致）。
 * roi_hint 非空时仅在 xywh 先验框内搜索（须完整落在图像内），顶点坐标仍为
 * 原图坐标系；NULL=全图。max_vertices_per 限每条建议顶点数（3..64），无法
 * 化简到限内的轮廓不输出；面积 <64px² 的碎轮廓不输出。out_vertices 容量须
 * 为 max_suggestions×max_vertices_per 个顶点，out_vertex_counts 容量须为
 * max_suggestions，第 i 条建议占 out_vertices 前缀累计偏移。
 * 本入口返回值语义特殊：≥0 为成功（值=建议条数，0=无建议，非错误），
 * -1=可报告失败，-2=未分类原生失败；失败时错误文本照常写入 error。 */
VF_SHAPE_API int32_t vf_shape_suggest_contours(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride,
    const int32_t roi_hint[4],
    uint32_t max_suggestions, uint32_t max_vertices_per,
    vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts,
    char* error, uint32_t error_capacity);

/* 轮廓建议参数（vf_shape_suggest_contours_ex 输入，NULL=全默认）。
 * struct_size 必须先置为 sizeof(vf_shape_suggest_params)。
 * canny_low/canny_high：Canny 迟滞阈，0=默认 50/150；非零须为有限正数且
 *   生效值满足 high≥low（默认参与比较）。阈值语义与训练 detector 三参同类
 *   （正且递增的边缘阈值族），越界拒绝码 VF_SHAPE_DETECTOR_PARAMS：can­ny_low
 *   非法 params=[canny_low]；high<low params=[生效 low,生效 high]。
 * epsilon_factor：多边形逼近精度（×轮廓周长，0=默认 0.02），有效域
 *   [0.001,0.2]；越界拒绝码 VF_SHAPE_DETECTOR_PARAMS，params=[epsilon,下界,上界]。 */
typedef struct {
    uint32_t struct_size;
    double canny_low, canny_high;
    double epsilon_factor;
} vf_shape_suggest_params;

/* 参数化轮廓建议：行为与 vf_shape_suggest_contours 一致，仅阈值与逼近精度
 * 可调（params=NULL 时两者逐字节一致）。其余合同（返回值 ≥0 条数/-1/-2 分层、
 * 确定性、缓冲容量）见 vf_shape_suggest_contours。 */
VF_SHAPE_API int32_t vf_shape_suggest_contours_ex(const uint8_t* image, uint64_t image_size,
    int32_t width, int32_t height, uint64_t stride, const int32_t roi_hint[4],
    uint32_t max_suggestions, uint32_t max_vertices_per,
    const vf_shape_suggest_params* params,
    vf_shape_vertex* out_vertices, uint32_t* out_vertex_counts,
    char* error, uint32_t error_capacity);

/* 多边形元数据回读：多边形资产返回持久化的归一化外环顶点（与训练输入经
 * 绕向归一后逐字段一致）与洞数；矩形资产（含旧格式与单级资产）回退为训练
 * ROI 四角（左上、右上、右下、左下）且洞数为 0。两次调用模式同
 * vf_shape_export：先以 outline=NULL 且 capacity=0 查询 *outline_count，再以
 * 不小于该值的缓冲取回；缓冲不足返回 1 且不写顶点内容。 */
VF_SHAPE_API int32_t vf_shape_polygon_metadata(uint64_t handle,
    vf_shape_vertex* outline, uint32_t capacity, uint32_t* outline_count,
    uint32_t* hole_count, char* error, uint32_t error_capacity);

/* 查询句柄最近一次 match/match_batch 的诊断快照。非阻塞快照：句柄锁下
 * 读取，与在途匹配串行。out->struct_size 须先置为 sizeof（v2）或 v1 布局
 * 尺寸 112，库按其只回写对应前缀。返回码分层同文件头。 */
VF_SHAPE_API int32_t vf_shape_last_match_info(uint64_t handle,
    vf_shape_match_trace* out, char* error, uint32_t error_capacity);

/* 模型健康体检（零输入）：从容器模板图与模型内部量导出结构与质量指标，
 * 不消费图像输入。报告字段与分级合同见 vf_shape_health_report；
 * out->struct_size 须先置为 sizeof。体检含一次自匹配（容器模板图对模型
 * 自身，阈值 70 取最高分），单级资产无模板图时该项未实测（valid=0）。
 * 返回码分层同文件头。 */
VF_SHAPE_API int32_t vf_shape_model_health(uint64_t handle,
    vf_shape_health_report* out, char* error, uint32_t error_capacity);

/* 区分度审计与阈值推荐（⑤-5/⑤-7 合一）：两组图像（正=含目标、负=无
 * 目标背景）各以指针数组+共享尺寸三元组传入（格式同文件头 BGR8 连续，
 * 组内宽高/步长一致），逐帧取现有匹配路径最高分构成分布并给出建议性
 * 阈值区间。pos_images/pos_count 须至少 1 帧；neg 可选（neg_count=0 时
 * neg_images 可为 NULL）。out->struct_size 须先置为 sizeof。区间为统计
 * 建议（advisory），消费方结合场景调整，不构成合同承诺。返回码分层同
 * 文件头。 */
VF_SHAPE_API int32_t vf_shape_audit_discriminability(uint64_t handle,
    const uint8_t* const* pos_images, int32_t pos_width, int32_t pos_height,
    uint64_t pos_stride, uint32_t pos_count,
    const uint8_t* const* neg_images, int32_t neg_width, int32_t neg_height,
    uint64_t neg_stride, uint32_t neg_count,
    vf_shape_audit_report* out, char* error, uint32_t error_capacity);

#ifdef __cplusplus
}
#endif
