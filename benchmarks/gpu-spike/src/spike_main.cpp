// GPU 可行性尖峰入口：两级最小链路（粗层全图搜 + 精修窗）CPU 参考 vs CUDA 对拍与基准。
// 用法：spike_bench.exe --model <bin> --frame <png> [--width 1920 --height 1080]
//       [--threshold 70] [--warmup 10] [--iters 100] [--mode both|bench|correctness]
#include "cpu_chain.h"
#include "gpu_chain.h"
#include "model_source.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gpuspike;

namespace {

struct Args {
    const char* model = "build/gpu-spike/data/anchor-model.bin";
    const char* frame = "benchmarks/shape-facts-v1/frames/synthetic-f0001-positive.png";
    int width = 1920, height = 1080;
    double threshold = 70;
    int warmup = 10, iters = 100;
    const char* mode = "both";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        const auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "参数 %s 缺值\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "--model") args.model = next("--model");
        else if (key == "--frame") args.frame = next("--frame");
        else if (key == "--width") args.width = std::atoi(next("--width"));
        else if (key == "--height") args.height = std::atoi(next("--height"));
        else if (key == "--threshold") args.threshold = std::atof(next("--threshold"));
        else if (key == "--warmup") args.warmup = std::atoi(next("--warmup"));
        else if (key == "--iters") args.iters = std::atoi(next("--iters"));
        else if (key == "--mode") args.mode = next("--mode");
        else {
            std::fprintf(stderr, "未知参数 %s\n", key.c_str());
            std::exit(2);
        }
    }
    return args;
}

double median_of(std::vector<double> values) {
    if (values.empty()) return 0;
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

double agreement(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    if (a.size() != b.size() || a.empty()) return -1;
    size_t same = 0;
    for (size_t i = 0; i < a.size(); ++i) same += a[i] == b[i];
    return double(same) / a.size();
}

double pearson(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    if (a.size() != b.size() || a.empty()) return -2;
    const size_t n = a.size();
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; ++i) {
        sa += a[i];
        sb += b[i];
    }
    const double ma = sa / n, mb = sb / n;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = double(a[i]) - ma, y = double(b[i]) - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if (da == 0 || db == 0) return da == db ? 1.0 : -2;
    return num / std::sqrt(da * db);
}

// 响应图首个最大值位置（行主序，与共享提取同打破平局规则）
bool argmax_position(const std::vector<uint16_t>& map, int W, int H, double& x, double& y) {
    if (map.size() != size_t(W) * H) return false;
    size_t best = 0;
    for (size_t i = 1; i < map.size(); ++i)
        if (map[i] > map[best]) best = i;
    const int r = int(best) / W, c = int(best) % W;
    x = c * kT[1] + grid_offset(kT[1]);
    y = r * kT[1] + grid_offset(kT[1]);
    return true;
}

struct ChainCompare {
    double angle_agree[2] = {-1, -1};
    double spread_agree[2] = {-1, -1};
    double resp_corr_min = 2;
    double resp_max_diff = 0;
    double argmax_max_dev_px = -1;
    int compared_poses = 0;
    bool top_match = false;
    double top_dev_px = -1;
};

ChainCompare compare_chains(const ChainOutput& cpu, const ChainOutput& gpu) {
    ChainCompare result;
    for (int level = 0; level < 2; ++level) {
        result.angle_agree[level] = agreement(cpu.maps[level].angle.data,
                                              gpu.maps[level].angle.data);
        result.spread_agree[level] = agreement(cpu.maps[level].spread.data,
                                               gpu.maps[level].spread.data);
    }
    for (size_t id = 0; id < cpu.response.size(); ++id) {
        if (cpu.response[id].empty() || gpu.response[id].empty()) continue;
        result.resp_corr_min = std::min(result.resp_corr_min,
                                        pearson(cpu.response[id], gpu.response[id]));
        double max_diff = 0;
        for (size_t i = 0; i < cpu.response[id].size(); ++i)
            max_diff = std::max(max_diff,
                                double(std::abs(int(cpu.response[id][i]) - int(gpu.response[id][i]))));
        result.resp_max_diff = std::max(result.resp_max_diff, max_diff);
        double cx = 0, cy = 0, gx = 0, gy = 0;
        if (argmax_position(cpu.response[id], cpu.resp_W, cpu.resp_H, cx, cy) &&
            argmax_position(gpu.response[id], gpu.resp_W, gpu.resp_H, gx, gy)) {
            const double dev = std::hypot(cx - gx, cy - gy);
            result.argmax_max_dev_px = std::max(result.argmax_max_dev_px, dev);
        }
        ++result.compared_poses;
    }
    auto top_of = [](const std::vector<RefinedCand>& list) {
        int best = -1;
        for (int i = 0; i < int(list.size()); ++i)
            if (best < 0 || list[i].score > list[best].score) best = i;
        return best;
    };
    const int ct = top_of(cpu.refined), gt = top_of(gpu.refined);
    if (ct >= 0 && gt >= 0) {
        result.top_match = cpu.refined[ct].pose_id == gpu.refined[gt].pose_id;
        result.top_dev_px = std::hypot(cpu.refined[ct].x - gpu.refined[gt].x,
                                       cpu.refined[ct].y - gpu.refined[gt].y);
    } else {
        result.top_match = ct < 0 && gt < 0;
        result.top_dev_px = ct < 0 && gt < 0 ? 0 : -1;
    }
    return result;
}

void print_compare(const char* name, const ChainCompare& c) {
    std::printf("[%s] 对拍：方向一致率 L0=%.4f%% L1=%.4f%% | spread 一致率 L0=%.4f%% L1=%.4f%%\n",
                name, c.angle_agree[0] * 100, c.angle_agree[1] * 100,
                c.spread_agree[0] * 100, c.spread_agree[1] * 100);
    std::printf("[%s] 响应图：min 相关系数=%.6f 最大绝对差=%.0f 对比位姿=%d | 每 pose argmax 偏差 max=%.1fpx\n",
                name, c.resp_corr_min, c.resp_max_diff, c.compared_poses, c.argmax_max_dev_px);
    std::printf("[%s] 链顶候选：同位姿=%d 位置偏差=%.2fpx\n", name, int(c.top_match), c.top_dev_px);
}

struct BenchRecord {
    double wall_ms = 0;
    double stage_ms[static_cast<size_t>(Stage::Count)] = {};
    double host_extract_ms = 0;
};

BenchRecord median_records(std::vector<BenchRecord>& records) {
    BenchRecord out;
    std::vector<double> wall;
    for (auto& r : records) wall.push_back(r.wall_ms);
    out.wall_ms = median_of(wall);
    for (size_t s = 0; s < static_cast<size_t>(Stage::Count); ++s) {
        std::vector<double> v;
        v.reserve(records.size());
        for (auto& r : records) v.push_back(r.stage_ms[s]);
        out.stage_ms[s] = median_of(v);
    }
    std::vector<double> v;
    for (auto& r : records) v.push_back(r.host_extract_ms);
    out.host_extract_ms = median_of(v);
    return out;
}

void print_bench(const char* name, const BenchRecord& r) {
    std::printf("[%s] 链路墙钟 %.3f ms（宿主提取 %.3f ms）| 阶段：", name, r.wall_ms,
                r.host_extract_ms);
    for (size_t s = 0; s < static_cast<size_t>(Stage::Count); ++s)
        std::printf("%s=%.3f", stage_name(Stage(s)), r.stage_ms[s]);
    std::printf("\n");
}

struct ChainCase {
    std::string name;
    FrameData padded;
    const std::vector<PoseModel>* poses = nullptr;
    float coarse_th = 0, fine_th = 0;
};

bool run_gpus(GpuChainContext* ctx, const ChainCase& chain, float weak, bool upload,
              bool correctness, ChainOutput& out, std::string& err) {
    return gpu_run_chain(ctx, chain.padded.bgr.data(), chain.padded.w, chain.padded.h, weak,
                         *chain.poses, chain.coarse_th, chain.fine_th, upload, correctness, out,
                         err);
}

}  // namespace

int main(int argc, char** argv) {
    // 弹窗红线：异常路径退出码化，禁止系统级错误弹窗
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    const Args args = parse_args(argc, argv);
    try {
        ModelSet model;
        std::string err;
        if (!load_model_set(args.model, model, err)) {
            std::fprintf(stderr, "模型加载失败：%s\n", err.c_str());
            return 3;
        }
        std::string device_name;
        int major = 0, minor = 0;
        if (!gpu_device_info(device_name, major, minor, err)) {
            std::fprintf(stderr, "GPU 探测失败：%s\n", err.c_str());
            return 3;
        }
        FrameData frame;
        if (!load_frame(args.frame, args.width, args.height, frame, err)) {
            std::fprintf(stderr, "帧加载失败：%s\n", err.c_str());
            return 3;
        }
        std::printf("环境：%s (sm_%d%d) | 帧 %dx%d | k=%d | coarse %zu 位姿 | fine 组位姿",
                    device_name.c_str(), major, minor, frame.w, frame.h, model.k,
                    model.coarse.size());
        for (auto& group : model.fines) std::printf(" %zu", group.size());
        std::printf("\n");

        // —— 链 A：粗层（k 缩图全图搜）——
        FrameData small;
        if (!shrink_frame(frame, model.k, small, err)) {
            std::fprintf(stderr, "缩图失败：%s\n", err.c_str());
            return 3;
        }
        ChainCase chain_a;
        pad_frame(small, chain_a.padded);
        chain_a.name = "A 粗层";
        chain_a.poses = &model.coarse;
        chain_a.coarse_th = chain_a.fine_th = float(std::max(45.0, args.threshold - 20.0));

        ChainOutput cpu_a;
        run_chain_cpu(chain_a.padded.bgr.data(), chain_a.padded.w, chain_a.padded.h,
                      float(model.weak), *chain_a.poses, chain_a.coarse_th, chain_a.fine_th, cpu_a);
        if (cpu_a.refined.empty()) {
            std::fprintf(stderr, "粗层链无精修候选（阈值 %.1f 过高或锚未命中）\n",
                         double(chain_a.coarse_th));
            return 3;
        }
        const auto best_a =
            std::max_element(cpu_a.refined.begin(), cpu_a.refined.end(),
                             [](const RefinedCand& a, const RefinedCand& b) { return a.score < b.score; });

        // —— 链 B：精修窗（引擎 candidate() 同式 ROI 中心定位，CPU 结果保证对拍同窗）——
        const double diag = std::hypot(200.0, 200.0);
        const double half = std::max(240.0, diag / 2.0 + 4.0 * model.k);
        double t[6];
        std::copy(std::begin(model.coarse[size_t(best_a->pose_id)].transform),
                  std::end(model.coarse[size_t(best_a->pose_id)].transform), t);
        t[2] += best_a->x - 32.0 - model.coarse[size_t(best_a->pose_id)].level[0].tl_x;
        t[5] += best_a->y - 32.0 - model.coarse[size_t(best_a->pose_id)].level[0].tl_y;
        const double roi_pts[8] = {double(model.coarse_roi[0]), double(model.coarse_roi[1]),
                                   double(model.coarse_roi[0] + model.coarse_roi[2]),
                                   double(model.coarse_roi[1]),
                                   double(model.coarse_roi[0] + model.coarse_roi[2]),
                                   double(model.coarse_roi[1] + model.coarse_roi[3]),
                                   double(model.coarse_roi[0]),
                                   double(model.coarse_roi[1] + model.coarse_roi[3])};
        double center_small_x = 0, center_small_y = 0;
        for (int corner = 0; corner < 4; ++corner) {
            center_small_x += (t[0] * roi_pts[2 * corner] + t[1] * roi_pts[2 * corner + 1] + t[2]) / 4;
            center_small_y += (t[3] * roi_pts[2 * corner] + t[4] * roi_pts[2 * corner + 1] + t[5]) / 4;
        }
        const double cx = center_small_x * model.k, cy = center_small_y * model.k;
        ChainCase chain_b;
        FrameData window;
        if (!crop_window(frame, cx, cy, half, window)) {
            std::fprintf(stderr, "精修窗裁剪失败\n");
            return 3;
        }
        pad_frame(window, chain_b.padded);
        chain_b.name = "B 精修窗";
        double anchor_angle = model.coarse[size_t(best_a->pose_id)].angle;
        if (anchor_angle > 180) anchor_angle -= 360;
        chain_b.poses = &model.fines[size_t(pick_fine_group(model, anchor_angle))];
        chain_b.coarse_th = chain_b.fine_th = float(std::max(50.0, args.threshold * 0.85));

        std::printf("链 A：缩图 %dx%d → 补边 %dx%d，位姿 %zu，阈 %.1f（候选 %zu 精修 %zu top 分 %.1f 角 %.1f）\n",
                    small.w, small.h, chain_a.padded.w, chain_a.padded.h, model.coarse.size(),
                    double(chain_a.coarse_th), cpu_a.candidates.size(), cpu_a.refined.size(),
                    double(best_a->score), anchor_angle);
        std::printf("链 B：窗 %dx%d @ (%.0f,%.0f) → 补边 %dx%d，位姿 %zu，阈 %.1f\n", window.w,
                    window.h, cx, cy, chain_b.padded.w, chain_b.padded.h, chain_b.poses->size(),
                    double(chain_b.coarse_th));

        // —— GPU 上下文 ×2 ——
        const auto ctx_of = [&](const ChainCase& c) {
            const int poses = int(c.poses->size());
            int feats = 0;
            for (auto& pose : *c.poses)
                for (int l = 0; l < kPyramidLevels; ++l) feats += int(pose.level[l].feats.size());
            const int W = (c.padded.w / 2) / kT[1], H = (c.padded.h / 2) / kT[1];
            return gpu_create_context(c.padded.w, c.padded.h, poses, feats, 16384, W * H, err);
        };
        GpuChainContext* ctx_a = ctx_of(chain_a);
        GpuChainContext* ctx_b = ctx_of(chain_b);
        if (!ctx_a || !ctx_b) {
            std::fprintf(stderr, "GPU 上下文创建失败：%s\n", err.c_str());
            return 3;
        }
        bool ok = gpu_upload_features(ctx_a, *chain_a.poses, err) &&
                  gpu_upload_features(ctx_b, *chain_b.poses, err);
        if (!ok) {
            std::fprintf(stderr, "特征驻留失败：%s\n", err.c_str());
            return 3;
        }

        // —— 对拍 ——
        bool gates_pass = true;
        const bool want_correctness = std::string(args.mode) != "bench";
        ChainCompare cmp_a, cmp_b;
        if (want_correctness) {
            ChainOutput gpu_a, gpu_b, cpu_b;
            if (!run_gpus(ctx_a, chain_a, float(model.weak), true, true, gpu_a, err)) {
                std::fprintf(stderr, "GPU 链 A 失败：%s\n", err.c_str());
                return 3;
            }
            run_chain_cpu(chain_b.padded.bgr.data(), chain_b.padded.w, chain_b.padded.h,
                          float(model.weak), *chain_b.poses, chain_b.coarse_th, chain_b.fine_th,
                          cpu_b);
            if (!run_gpus(ctx_b, chain_b, float(model.weak), true, true, gpu_b, err)) {
                std::fprintf(stderr, "GPU 链 B 失败：%s\n", err.c_str());
                return 3;
            }
            cmp_a = compare_chains(cpu_a, gpu_a);
            cmp_b = compare_chains(cpu_b, gpu_b);
            print_compare("A 粗层", cmp_a);
            print_compare("B 精修窗", cmp_b);
            if (!cpu_b.refined.empty()) {
                const auto top_b = std::max_element(
                    cpu_b.refined.begin(), cpu_b.refined.end(),
                    [](const RefinedCand& a, const RefinedCand& b) { return a.score < b.score; });
                const int wx0 = int(std::max(0.0, cx - half)), wy0 = int(std::max(0.0, cy - half));
                // 引擎 candidate() 同式：fine 位姿 transform 映射 ROI 四角取中心
                const auto& pose = chain_b.poses->at(size_t(top_b->pose_id));
                double ft[6];
                std::copy(std::begin(pose.transform), std::end(pose.transform), ft);
                ft[2] += top_b->x - 32.0 - pose.level[0].tl_x + wx0;
                ft[5] += top_b->y - 32.0 - pose.level[0].tl_y + wy0;
                const double pts[8] = {double(model.fine_roi[0]), double(model.fine_roi[1]),
                                       double(model.fine_roi[0] + model.fine_roi[2]),
                                       double(model.fine_roi[1]),
                                       double(model.fine_roi[0] + model.fine_roi[2]),
                                       double(model.fine_roi[1] + model.fine_roi[3]),
                                       double(model.fine_roi[0]),
                                       double(model.fine_roi[1] + model.fine_roi[3])};
                double ox = 0, oy = 0;
                for (int corner = 0; corner < 4; ++corner) {
                    ox += (ft[0] * pts[2 * corner] + ft[1] * pts[2 * corner + 1] + ft[2]) / 4;
                    oy += (ft[3] * pts[2 * corner] + ft[4] * pts[2 * corner + 1] + ft[5]) / 4;
                }
                std::printf("[B 精修窗] 链顶：分数 %.1f 角 %.1f° 目标中心原帧坐标 (%.1f, %.1f)\n",
                            double(top_b->score), pose.angle, ox, oy);
            }
            for (const ChainCompare* c : {&cmp_a, &cmp_b}) {
                gates_pass &= c->angle_agree[0] >= 0.99 && c->angle_agree[1] >= 0.99;
                gates_pass &= c->resp_corr_min >= 0.99;
                gates_pass &= c->top_dev_px >= 0 && c->top_dev_px <= 2.0;
            }
            std::printf("门禁（方向≥99%% | 相关≥0.99 | 顶位偏差≤2px）：%s\n",
                        gates_pass ? "PASS" : "FAIL");
        }

        // —— 基准 ——
        const bool want_bench = std::string(args.mode) != "correctness";
        if (want_bench) {
            const auto bench_loop = [&](const char* tag, const ChainCase& chain,
                                        GpuChainContext* ctx, bool cpu_side, bool upload) {
                std::vector<BenchRecord> records;
                for (int i = 0; i < args.warmup + args.iters; ++i) {
                    ChainOutput out;
                    if (cpu_side) {
                        run_chain_cpu(chain.padded.bgr.data(), chain.padded.w, chain.padded.h,
                                      float(model.weak), *chain.poses, chain.coarse_th,
                                      chain.fine_th, out);
                    } else if (!run_gpus(ctx, chain, float(model.weak), upload, false, out, err)) {
                        std::fprintf(stderr, "GPU 基准失败：%s\n", err.c_str());
                        std::exit(3);
                    }
                    if (i >= args.warmup) {
                        BenchRecord r;
                        r.wall_ms = out.wall_ms;
                        std::copy(std::begin(out.times.ms), std::end(out.times.ms),
                                  std::begin(r.stage_ms));
                        r.host_extract_ms = out.host_extract_ms;
                        records.push_back(r);
                    }
                }
                return median_records(records);
            };
            const BenchRecord cpu_a_rec = bench_loop("cpu", chain_a, ctx_a, true, false);
            const BenchRecord cpu_b_rec = bench_loop("cpu", chain_b, ctx_b, true, false);
            const BenchRecord gpu_a_xfer = bench_loop("gpu", chain_a, ctx_a, false, true);
            const BenchRecord gpu_a_comp = bench_loop("gpu", chain_a, ctx_a, false, false);
            const BenchRecord gpu_b_xfer = bench_loop("gpu", chain_b, ctx_b, false, true);
            const BenchRecord gpu_b_comp = bench_loop("gpu", chain_b, ctx_b, false, false);
            print_bench("A CPU 单线程", cpu_a_rec);
            print_bench("A GPU 含传输", gpu_a_xfer);
            print_bench("A GPU 纯计算", gpu_a_comp);
            print_bench("B CPU 单线程", cpu_b_rec);
            print_bench("B GPU 含传输", gpu_b_xfer);
            print_bench("B GPU 纯计算", gpu_b_comp);
            const auto speedup = [](const BenchRecord& cpu, const BenchRecord& gpu) {
                return gpu.wall_ms > 0 ? cpu.wall_ms / gpu.wall_ms : 0.0;
            };
            std::printf("加速比：A 含传输 %.1fx / 纯计算 %.1fx | B 含传输 %.1fx / 纯计算 %.1fx | A+B 含传输 %.1fx\n",
                        speedup(cpu_a_rec, gpu_a_xfer), speedup(cpu_a_rec, gpu_a_comp),
                        speedup(cpu_b_rec, gpu_b_xfer), speedup(cpu_b_rec, gpu_b_comp),
                        (cpu_a_rec.wall_ms + cpu_b_rec.wall_ms) / (gpu_a_xfer.wall_ms + gpu_b_xfer.wall_ms));
        }

        gpu_destroy_context(ctx_a);
        gpu_destroy_context(ctx_b);
        return gates_pass || !want_correctness ? 0 : 2;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "异常退出：%s\n", ex.what());
        return 3;
    } catch (...) {
        std::fprintf(stderr, "未知异常退出\n");
        return 3;
    }
}
