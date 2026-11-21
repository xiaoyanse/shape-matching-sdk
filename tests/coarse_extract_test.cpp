// coarse_extract 宿主提取单测（确定性构造，无随机）：五场景
//  1 出界跳过语义（R2 回归锚）：窗偏移后出界特征跳过 vs 全图回卷——
//    在回卷目标像素放置响应位，若实现回卷则 argmax 坐标改变必抓；
//  2 行主序首最大 + B5 亚像素：已知峰形窗（单元格值手推）；
//  3 sort+unique 复刻：同 pose 重复候选去重、跨 pose 分数降序；
//  4 门槛三处：initial 阈（恰等 >=）、精化后过滤阈、护栏 border 拒绝；
//  5 预算：initial 候选累计 >200000 报错路径。
// 退出码：0=过；3=门禁失败。
#include "gpu/coarse_extract.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kFailCode = 3;
constexpr float kThreshold = 45.f;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 帧 512×512（L1 256×256 → 响应网格 32×32）；模板 L0 60×60 / L1 30×30
constexpr int kW0 = 512, kH0 = 512;
constexpr int kW1 = 256, kH1 = 256;
constexpr int kRespW = kW1 / 8, kRespH = kH1 / 8;
constexpr int kL0W = 60, kL0H = 60, kL1W = 30, kL1H = 30;

using vfshape::gpu::ExtractPoses;
using vfshape::gpu::ExtractResult;
using vfshape::gpu::FeatureRecord;

std::vector<uint16_t> make_resp(const std::vector<std::pair<int, int>>& cells, uint16_t value) {
    std::vector<uint16_t> resp(size_t(kRespW) * kRespH, 0);
    for (const auto& cell : cells) resp[size_t(cell.first) * kRespW + cell.second] = value;
    return resp;
}

// (sy,sx) 处直接放扩散后字节（跳过 angle→spread 中间步，几何等价：单像素
// angle 位扩散的因果 4×4 块不与本场景其他采样点相交，见各场景手推）
std::vector<uint8_t> make_spread(
    const std::vector<std::pair<std::pair<int, int>, uint8_t>>& bytes) {
    std::vector<uint8_t> spread(size_t(kW0) * kH0, 0);
    for (const auto& item : bytes)
        spread[size_t(item.first.first) * kW0 + item.first.second] = item.second;
    return spread;
}

ExtractPoses::Level make_level(int width, int height, const std::vector<FeatureRecord>& feats) {
    ExtractPoses::Level level;
    level.width = width;
    level.height = height;
    level.feats = feats;
    return level;
}

bool run(const std::vector<std::vector<uint16_t>>& resp1,
         const std::vector<uint8_t>& spread0, const ExtractPoses& poses,
         ExtractResult& out, std::string& err) {
    return vfshape::gpu::extract_coarse_candidates(resp1, kRespW, kRespH, spread0, kW0, kH0,
                                                   poses, kW0, kH0, kThreshold, out, err);
}

// err 进断言消息须在调用返回后取值（参数先求值，勿内联进 check 首参）
void check_run(bool ok, const std::string& scenario, const std::string& err) {
    check(ok, scenario + " extract failed: " + err);
}

// —— 场景 1：出界跳过（R2 回归锚）——
// 候选 (5,2)→精修中心 (87,39)，窗偏移 (52,4)，gx0=13/gy0=1。
// 特征 A(10,10,l0)→(62,14) 命中格 (2,3)；A2(10,10,l1) 命中 (0,3)；
// B(-60,10,l0) 偏移后 (-8,14) 出界必须跳过（vendor :942）；其回卷目标像素
// (14,120) 放响应位——若实现走全图回卷，格 (0,0) 记 4，argmax 变 (0,0) 即抓。
// D(440,10,l0)→(492,14)：窗列 >=5 像素出界（vendor flat 续读 UB 带，尖峰口径
// 跳过）；其回卷目标像素 (30,{0,4,8}) 放 0xFF——若回卷，格 (0,5..7) 记 4 即抓。
// 跳过口径唯一非零格 (2,3)=8（A 4 + A2 4，位 (22,74)=0x03 两标签各记 4）。
void test_out_of_bounds_skip() {
    const std::vector<uint8_t> spread = make_spread(
        {{{22, 74}, 0x03}, {{14, 120}, 0x01}, {{30, 0}, 0xFF}, {{30, 4}, 0xFF}, {{30, 8}, 0xFF}});
    ExtractPoses poses;
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
    poses.level0.push_back(make_level(
        kL0W, kL0H, {{10, 10, 0}, {-60, 10, 0}, {10, 10, 1}, {440, 10, 0}}));
    const auto resp1 = std::vector<std::vector<uint16_t>>{make_resp({{2, 5}}, 2)};

    ExtractResult out;
    std::string err;
    check_run(run(resp1, spread, poses, out, err), "scenario1", err);
    check(out.initial_count == 1, "scenario1 initial_count=" + std::to_string(out.initial_count));
    check(out.matches.size() == 1, "scenario1 matches=" + std::to_string(out.matches.size()));
    const auto& m = out.matches[0];
    // 窗分 8*100/(4*4)=50 恰等实现值；回卷实现 argmax 会落 (0,0)→x=53
    check(m.similarity == 50.f, "scenario1 similarity");
    // argmax (2,3)：x=(gx0+3)*4+1=(13+3)*4+1=65，y=(gy0+2)*4+1=(1+2)*4+1=13
    check(m.x == 65.f && m.y == 13.f, "scenario1 coords (skip anchor)");
    check(m.template_id == 0 && m.class_id == "shape", "scenario1 id/class");
}

// —— 场景 2：首最大 + 亚像素 ——
// 特征 P(48,96,l0)→(100,100)、Q(88,96,l0)→(140,100)，响应位全 0x01（label0 各记 4）：
//  a) 位 (108,104)+(108,108)+(108,148)+(112,148)：格 (2,1)=4,(2,2)=8,(2,12)=4,
//     (3,2)=4,(3,12)=4 → argmax (2,2)；亚像素 c_l=4/c_r=0/r_u=0/r_d=4；
//  b) 去掉 (108,108)：格 (2,1)=(2,2)=(2,12)=(3,2)=(3,12)=4 平峰 → 首最大 (2,1)。
void test_argmax_and_subpixel() {
    ExtractPoses poses;
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{48, 96, 0}, {88, 96, 0}}));
    const auto resp1 = std::vector<std::vector<uint16_t>>{make_resp({{2, 5}}, 2)};

    // a) 峰形 + 亚像素
    {
        const std::vector<uint8_t> spread = make_spread(
            {{{108, 104}, 0x01}, {{108, 108}, 0x01}, {{108, 148}, 0x01}, {{112, 148}, 0x01}});
        ExtractResult out;
        std::string err;
        check_run(run(resp1, spread, poses, out, err), "scenario2a", err);
        check(out.matches.size() == 1, "scenario2a matches=" + std::to_string(out.matches.size()));
        const auto& m = out.matches[0];
        check(m.similarity == 100.f, "scenario2a similarity");
        // 亚像素 B5 同式：dc=0.5*(4-0)/(4-16+0)，dr=0.5*(0-4)/(0-16+4)
        const float c_l = 4.f, c_m = 8.f, c_r = 0.f, r_u = 0.f, r_d = 4.f;
        const float denom_c = c_l - 2 * c_m + c_r, denom_r = r_u - 2 * c_m + r_d;
        const float dc = 0.5f * (c_l - c_r) / denom_c;
        const float dr = 0.5f * (r_u - r_d) / denom_r;
        check(m.x == float((13 + 2 + dc) * 4 + 1), "scenario2a subpixel x");
        check(m.y == float((1 + 2 + dr) * 4 + 1), "scenario2a subpixel y");
        check(m.x != 61.f, "scenario2a subpixel must be non-integer");
    }
    // b) 平峰首最大：若取尾最大 (2,12)/(3,12)，坐标跳到 x=101 必抓
    {
        const std::vector<uint8_t> spread =
            make_spread({{{108, 104}, 0x01}, {{108, 148}, 0x01}, {{112, 148}, 0x01}});
        ExtractResult out;
        std::string err;
        check_run(run(resp1, spread, poses, out, err), "scenario2b", err);
        check(out.matches.size() == 1, "scenario2b matches=" + std::to_string(out.matches.size()));
        const auto& m = out.matches[0];
        // (2,1) 亚像素：c_l=0/c_m=4/c_r=4 → dc=0.5*(0-4)/(0-8+4)=0.5；dr=0
        check(m.similarity == 50.f, "scenario2b similarity");
        // argmax=格(2,1)（行主序首最大）：dc=0.5*(0-4)/(0-8+4)=0.5、dr=0
        // → x=(13+1+0.5)*4+1=59，y=(1+2+0)*4+1=13
        check(m.x == 59.f && m.y == 13.f,
              "scenario2b first-max coords x=" + std::to_string(m.x) + " y=" + std::to_string(m.y));
    }
}

// —— 场景 3：sort+unique 复刻 ——
// 单特征 F(12,24,l0)，位 (52,72)=0x01。pose0 两个 initial 候选 (5,2)/(5,3)，
// 窗偏移 y 差 16 → 峰格 (6,2)/(2,2) 同指绝对像素 → 精修后同分同坐标
// (53,29,100) → vendor operator== 去重留一；pose1 同构造 → template_id 异
// 不等不去重；同分 sort 按 template_id 升序（Match::operator< 次序）。
void test_sort_unique() {
    const std::vector<uint8_t> spread = make_spread({{{52, 72}, 0x01}});
    ExtractPoses poses;
    for (int pose = 0; pose < 2; ++pose) {
        poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
        poses.level0.push_back(make_level(kL0W, kL0H, {{12, 24, 0}}));
    }
    const auto resp1 = std::vector<std::vector<uint16_t>>{
        make_resp({{2, 5}, {3, 5}}, 2), make_resp({{2, 5}}, 2)};

    ExtractResult out;
    std::string err;
    check_run(run(resp1, spread, poses, out, err), "scenario3", err);
    check(out.initial_count == 3, "scenario3 initial_count=" + std::to_string(out.initial_count));
    check(out.matches.size() == 2, "scenario3 matches=" + std::to_string(out.matches.size()));
    check(out.matches[0].template_id == 0 && out.matches[1].template_id == 1,
          "scenario3 order (similarity desc, template_id asc tie)");
    for (const auto& m : out.matches) {
        // x=(gx0+2)*4+1=(13+2)*4+1=61；y 峰格绝对同指 → (1+6)*4+1=(5+2)*4+1=29
        check(m.x == 61.f && m.y == 29.f && m.similarity == 100.f, "scenario3 coords/score");
    }
}

// —— 场景 4：门槛三处 ——
// pose0：resp 值 9、nf1=5 → initial 分 9*100/20=45 恰过（>= 同式）；L0 位
// (28,64)=0x01 → 精修 100 → 留。pose1：resp 值 8 → 40 不过 → 无候选。
// pose2：L0 特征 (24,40,l0)→(76,44)，位 (44,76)=0x02 → 窗分 3*100/8=37.5
// 低于精化阈 → MatchPredicate 剔除。pose3：L0 宽 448+64>=512 护栏拒绝。
void test_thresholds_and_guard() {
    const std::vector<uint8_t> spread = make_spread({{{28, 64}, 0x01}, {{44, 76}, 0x02}});
    ExtractPoses poses;
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0},
                                                    {5, 5, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{12, 24, 0}}));
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0},
                                                    {5, 5, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{12, 24, 0}}));
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0},
                                                    {5, 5, 0}}));
    // nf0=2：窗分 3*100/8=37.5 低于阈 45 → 精化环 MatchPredicate 剔除；
    // 第二特征远离全部响应位（窗分恒 0，只抬分母）
    poses.level0.push_back(make_level(kL0W, kL0H, {{24, 40, 0}, {200, 200, 1}}));
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0},
                                                    {5, 5, 0}}));
    poses.level0.push_back(make_level(448, kL0H, {{12, 24, 0}}));
    const auto resp1 = std::vector<std::vector<uint16_t>>{make_resp({{2, 5}}, 9),
                                                          make_resp({{2, 5}}, 8),
                                                          make_resp({{2, 5}}, 9),
                                                          make_resp({{2, 5}}, 9)};

    ExtractResult out;
    std::string err;
    check_run(run(resp1, spread, poses, out, err), "scenario4", err);
    check(out.initial_count == 2, "scenario4 initial_count=" + std::to_string(out.initial_count));
    check(out.matches.size() == 1, "scenario4 matches=" + std::to_string(out.matches.size()));
    const auto& m = out.matches[0];
    check(m.template_id == 0 && m.x == 53.f && m.y == 5.f && m.similarity == 100.f,
          "scenario4 survivor");
}

// —— 场景 5：预算 ——
// 4 pose × 224×224 全格候选 = 200704 > 200000（帧 3584×3584 → 网格 224×224）
void test_budget() {
    constexpr int kW = 3584, kH = 3584, kRW = kW / 16, kRH = kH / 16;
    const std::vector<uint8_t> spread(size_t(kW) * kH, 0);
    ExtractPoses poses;
    std::vector<std::vector<uint16_t>> resp1;
    for (int pose = 0; pose < 4; ++pose) {
        poses.level1.push_back(make_level(30, 30, {{1, 1, 0}, {2, 2, 0}, {3, 3, 0}, {4, 4, 0},
                                                   {5, 5, 0}}));
        poses.level0.push_back(make_level(60, 60, {{12, 24, 0}}));
        resp1.push_back(std::vector<uint16_t>(size_t(kRW) * kRH, 9));
    }
    ExtractResult out;
    std::string err;
    const bool ok = vfshape::gpu::extract_coarse_candidates(resp1, kRW, kRH, spread, kW, kH,
                                                            poses, kW, kH, kThreshold, out, err);
    check(!ok, "scenario5 must fail on budget");
    check(err == "candidate budget exceeded", "scenario5 err=" + err);
    check(out.matches.empty(), "scenario5 matches must be cleared");
    check(out.initial_count == 4 * kRW * kRH, "scenario5 initial_count");
}

// —— 场景 6：边缘带 LM flat 行尾续读（审计修正的回归锚）——
// 三 pose 同候选 (5,2)（窗偏移 (52,4)，W=Hc=128，plane=16384），特征贴右下缘：
//  pose0 F2=(457,9,l0)→(509,13)：px=1,py=1,lm=(127,3)。格 (2,1)：
//    j=3*128+127+2*128+1=768<plane → cell(6,0) → 采样 (1,25)，字节 (25,1)=0x01
//    → 4（右缘同平面回卷：旧"跳过"语义采样 (513,21) 越界弃之，窗分全零）；
//  pose1 F3=(457,505,l0)→(509,509)：lm=(127,127)。格 (0,1)：j=16384=plane
//    → p=1*4+1+1=6、j'=0 → 采样 (2,1)，字节 (1,2)=0x01 → 4（平面尾溢出相位
//    进位：旧语义跳过）；
//  pose2 F4=(459,507,l0)→(511,511)：p0=3*4+3=15。格 (0,1)：j=16384 → p=16
//    ≥T*T → 取模 → 0 → 采样 (0,0)，字节 (0,0)=0x01 → 4（Mat 尾 UB 确定性化：
//    vendor 此处真越界读，不可复现，W2-A 先例口径）。
// 旧"跳过"语义下三窗全零候选全弃（0 匹配）；新语义 3 匹配——断言咬住分歧。
// 采样坐标按相位字节类分离（sx ≡1/≡2/≡0 mod 4），字节互不误中。
void test_edge_wrap_backread() {
    const std::vector<uint8_t> spread =
        make_spread({{{25, 1}, 0x01}, {{1, 2}, 0x01}, {{0, 0}, 0x01}});
    ExtractPoses poses;
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{457, 9, 0}}));
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{457, 505, 0}}));
    poses.level1.push_back(make_level(kL1W, kL1H, {{1, 1, 0}}));
    poses.level0.push_back(make_level(kL0W, kL0H, {{459, 507, 0}}));
    const auto resp1 = std::vector<std::vector<uint16_t>>{make_resp({{2, 5}}, 2),
                                                          make_resp({{2, 5}}, 2),
                                                          make_resp({{2, 5}}, 2)};

    ExtractResult out;
    std::string err;
    check_run(run(resp1, spread, poses, out, err), "scenario6", err);
    check(out.initial_count == 3, "scenario6 initial_count=" + std::to_string(out.initial_count));
    check(out.matches.size() == 3, "scenario6 matches=" + std::to_string(out.matches.size()));
    check(out.matches[0].template_id == 0 && out.matches[1].template_id == 1 &&
              out.matches[2].template_id == 2,
          "scenario6 order");
    // 三窗唯一峰格：pose0 (2,1)→(57,13)（邻零亚像素 dc=dr=0）；pose1/2 (0,1) 边界行
    // 无亚像素→(57,5)；nf0=1 → 窗分 4*100/4=100
    check(out.matches[0].x == 57.f && out.matches[0].y == 13.f, "scenario6 pose0 right-edge wrap");
    check(out.matches[1].x == 57.f && out.matches[1].y == 5.f,
          "scenario6 pose1 plane-tail carry");
    check(out.matches[2].x == 57.f && out.matches[2].y == 5.f,
          "scenario6 pose2 mat-tail determinized");
    for (const auto& m : out.matches) check(m.similarity == 100.f, "scenario6 score");
}

}  // namespace

int main() {
    try {
        test_out_of_bounds_skip();
        std::printf("[coarse_extract] scenario1 out-of-bounds skip PASS\n");
        test_argmax_and_subpixel();
        std::printf("[coarse_extract] scenario2 argmax first-max + subpixel PASS\n");
        test_sort_unique();
        std::printf("[coarse_extract] scenario3 sort+unique PASS\n");
        test_thresholds_and_guard();
        std::printf("[coarse_extract] scenario4 thresholds + guard PASS\n");
        test_budget();
        std::printf("[coarse_extract] scenario5 candidate budget PASS\n");
        test_edge_wrap_backread();
        std::printf("[coarse_extract] scenario6 edge LM-flat wrap backread PASS\n");
        std::printf("[coarse_extract] PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "coarse_extract failed: %s\n", ex.what());
        return kFailCode;
    }
}
