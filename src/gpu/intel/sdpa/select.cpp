/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/sdpa/select.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <unordered_map>

#include "common/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

namespace {

struct coef_entry_t {
    const char *name;
    double fwd_model_coefs_t::*member;
};

// Declaration order of fwd_model_coefs_t, tooling indexes by it
const coef_entry_t coef_table[] = {
        {"clock_ghz", &fwd_model_coefs_t::clock_ghz},
        {"mma_cycles_per_flop", &fwd_model_coefs_t::mma_cycles_per_flop},
        {"fma_cycles_per_flop", &fwd_model_coefs_t::fma_cycles_per_flop},
        {"unroll_overhead", &fwd_model_coefs_t::unroll_overhead},
        {"softmax_cycles_per_elem",
                &fwd_model_coefs_t::softmax_cycles_per_elem},
        {"slm_cycles_per_byte", &fwd_model_coefs_t::slm_cycles_per_byte},
        {"load_cycles_per_byte", &fwd_model_coefs_t::load_cycles_per_byte},
        {"align4_penalty", &fwd_model_coefs_t::align4_penalty},
        {"unaligned_penalty", &fwd_model_coefs_t::unaligned_penalty},
        {"dequant_cycles_per_elem",
                &fwd_model_coefs_t::dequant_cycles_per_elem},
        {"barrier_cycles", &fwd_model_coefs_t::barrier_cycles},
        {"wg_fixed_cycles", &fwd_model_coefs_t::wg_fixed_cycles},
        {"io_cycles_per_byte", &fwd_model_coefs_t::io_cycles_per_byte},
        {"overlap_threads", &fwd_model_coefs_t::overlap_threads},
        {"partial_wave_c0", &fwd_model_coefs_t::partial_wave_c0},
        {"partial_wave_c1", &fwd_model_coefs_t::partial_wave_c1},
        {"mem_bw_discrete", &fwd_model_coefs_t::mem_bw_discrete},
        {"mem_bw_integrated", &fwd_model_coefs_t::mem_bw_integrated},
        {"l3_bw", &fwd_model_coefs_t::l3_bw},
        {"l3_fraction", &fwd_model_coefs_t::l3_fraction},
        {"launch_cycles", &fwd_model_coefs_t::launch_cycles},
        {"grf_k_load", &fwd_model_coefs_t::grf_k_load},
        {"mix_power", &fwd_model_coefs_t::mix_power},
        {"dpas_issue_cycles", &fwd_model_coefs_t::dpas_issue_cycles},
        {"dpas_latency_cycles", &fwd_model_coefs_t::dpas_latency_cycles},
        {"msg_latency_cycles", &fwd_model_coefs_t::msg_latency_cycles},
        {"msg_bytes_per_cycle", &fwd_model_coefs_t::msg_bytes_per_cycle},
        {"inflight_depth", &fwd_model_coefs_t::inflight_depth},
        {"mem_latency_cycles", &fwd_model_coefs_t::mem_latency_cycles},
        {"barrier_sg_cycles", &fwd_model_coefs_t::barrier_sg_cycles},
        {"causal_tail_frac", &fwd_model_coefs_t::causal_tail_frac},
};
constexpr int coef_count = sizeof(coef_table) / sizeof(coef_table[0]);

// Seeds: first-principles rates per EU, product bandwidths over the clock,
// latencies of tens of cycles. Swept architectures override them with
// fitted values below; xe_lp and xe_hp are not fitted yet
fwd_model_coefs_t seed_coefs(compute::gpu_arch_t arch) {
    fwd_model_coefs_t m {};
    m.clock_ghz = 2.0;
    m.mma_cycles_per_flop = 1.0 / 256;
    m.fma_cycles_per_flop = 1.0 / 16;
    m.unroll_overhead = 0.08;
    m.softmax_cycles_per_elem = 0.25;
    m.slm_cycles_per_byte = 1.0 / 128;
    m.load_cycles_per_byte = 1.0 / 64;
    m.align4_penalty = 2.0;
    m.unaligned_penalty = 4.0;
    m.dequant_cycles_per_elem = 0.1;
    m.barrier_cycles = 200;
    m.wg_fixed_cycles = 2000;
    m.io_cycles_per_byte = 1.0 / 32;
    m.overlap_threads = 4.0; // A750 sweeps favour 32-subgroup work-groups
    m.partial_wave_c0 = 1.0; // a partial wave still takes a whole wave
    m.partial_wave_c1 = 0.0;
    m.mem_bw_discrete = 250; // 512 GB/s at 2.05 GHz (A750)
    m.mem_bw_integrated = 64; // 136 GB/s at ~2.1 GHz (LNL class)
    m.l3_bw = 1000;
    m.l3_fraction = 0.5;
    m.launch_cycles = 8000;
    m.grf_k_load = 32;
    m.mix_power = 3.0;
    m.dpas_issue_cycles = 2.0;
    m.dpas_latency_cycles = 24;
    m.msg_latency_cycles = 40;
    m.msg_bytes_per_cycle = 32;
    m.inflight_depth = 2.0;
    m.mem_latency_cycles = 1500;
    m.barrier_sg_cycles = 20;
    m.causal_tail_frac = 0.0;

    switch (arch) {
        case compute::gpu_arch_t::xe_hpg:
            // Fitted on A750: 116 problems, 2677 configs, 4-fold CV regret
            // 1.082 (legacy table 1.259), in-sample 1.079
            m.softmax_cycles_per_elem = 2.74;
            m.slm_cycles_per_byte = 0.002638;
            m.load_cycles_per_byte = 6.088;
            m.barrier_cycles = 1965.0;
            m.wg_fixed_cycles = 665.2;
            m.io_cycles_per_byte = 0.1216;
            m.overlap_threads = 12.58;
            m.partial_wave_c0 = 1.185;
            m.partial_wave_c1 = 0.1569;
            m.mem_bw_discrete = 286.1;
            m.l3_bw = 330.9;
            m.l3_fraction = 3.011;
            m.launch_cycles = 22561.6;
            m.mix_power = 2.26;
            m.dpas_issue_cycles = 128.0;
            m.dpas_latency_cycles = 271.0;
            m.msg_latency_cycles = 570.7;
            m.msg_bytes_per_cycle = 21.64;
            m.inflight_depth = 0.7576;
            m.mem_latency_cycles = 7026.0;
            m.barrier_sg_cycles = 4.78149e-10;
            m.causal_tail_frac = 2.113;
            m.mma_cycles_per_flop = 0.007439;
            break;
        case compute::gpu_arch_t::xe_hpc:
            // Fitted on PVC: 129 problems, 3404 configs, 4-fold CV regret
            // 1.122 (legacy table 1.209), in-sample 1.091
            m.clock_ghz = 1.6;
            m.mma_cycles_per_flop = 0.002518;
            m.fma_cycles_per_flop = 0.05549;
            m.unroll_overhead = 0.4825;
            m.softmax_cycles_per_elem = 1.532;
            m.slm_cycles_per_byte = 0.01288;
            m.load_cycles_per_byte = 3.176;
            m.align4_penalty = 2;
            m.unaligned_penalty = 4;
            m.dequant_cycles_per_elem = 0.1769;
            m.barrier_cycles = 1151.0;
            m.wg_fixed_cycles = 627.9;
            m.io_cycles_per_byte = 0.0006272;
            m.overlap_threads = 4.658;
            m.partial_wave_c0 = 1.519;
            m.partial_wave_c1 = 0.05398;
            m.mem_bw_discrete = 1113.0;
            m.mem_bw_integrated = 64;
            m.l3_bw = 1472.0;
            m.l3_fraction = 0.3033;
            m.launch_cycles = 6466.0;
            m.grf_k_load = 32;
            m.mix_power = 1.82;
            m.dpas_issue_cycles = 2.0;
            m.dpas_latency_cycles = 14.56;
            m.msg_latency_cycles = 14.72;
            m.msg_bytes_per_cycle = 9.026;
            m.inflight_depth = 3.565;
            m.mem_latency_cycles = 909.8;
            m.barrier_sg_cycles = 0.8353;
            m.causal_tail_frac = 1.07;
            break;
        // Separate cases so bake_fit.py replaces one architecture at a time
        case compute::gpu_arch_t::xe2:
            // Fitted on BMG+LNL: 882 problems, 12560 configs, 4-fold CV regret
            // 1.090 (legacy table 1.387), in-sample 1.081
            m.clock_ghz = 2.5;
            m.mma_cycles_per_flop = 0.01011;
            m.fma_cycles_per_flop = 0.1401;
            m.mem_bw_discrete = 222.2;
            m.mem_bw_integrated = 34.83;
            m.l3_bw = 2991.0;
            m.softmax_cycles_per_elem = 1.917;
            m.slm_cycles_per_byte = 0.005909;
            m.load_cycles_per_byte = 6.259;
            m.barrier_cycles = 1502.0;
            m.wg_fixed_cycles = 1.57607e-10;
            m.io_cycles_per_byte = 0.006478;
            m.overlap_threads = 3.879;
            m.partial_wave_c0 = 0.4064;
            m.partial_wave_c1 = 0.2531;
            m.l3_fraction = 1.316;
            m.launch_cycles = 6613.0;
            m.mix_power = 1.275;
            m.dpas_issue_cycles = 2.348;
            m.dpas_latency_cycles = 29.16;
            m.msg_latency_cycles = 44.27;
            m.msg_bytes_per_cycle = 9.078;
            m.inflight_depth = 3.704;
            m.mem_latency_cycles = 1657.0;
            m.barrier_sg_cycles = 19.11;
            m.causal_tail_frac = 0.1854;
            m.unroll_overhead = 0.08;
            m.dequant_cycles_per_elem = 0.2718;
            break;
        case compute::gpu_arch_t::xe3:
            // Fitted on PTL: 87 problems, 1994 configs, 4-fold CV regret
            // 1.086 (legacy table 1.303), in-sample 1.055
            m.clock_ghz = 2.5;
            m.mma_cycles_per_flop = 0.005309;
            m.fma_cycles_per_flop = 1.0 / 32;
            m.mem_bw_discrete = 180.0;
            m.mem_bw_integrated = 90.68;
            m.l3_bw = 1500.0;
            m.softmax_cycles_per_elem = 1.847;
            m.slm_cycles_per_byte = 0.01288;
            m.load_cycles_per_byte = 3.823;
            m.barrier_cycles = 1478.0;
            m.wg_fixed_cycles = 446.3;
            m.io_cycles_per_byte = 0.03125;
            m.overlap_threads = 4.0;
            m.partial_wave_c0 = 1.649;
            m.partial_wave_c1 = 3.39827e-15;
            m.l3_fraction = 2.241;
            m.launch_cycles = 6.06605e-06;
            m.mix_power = 0.6694;
            m.dpas_issue_cycles = 2.0;
            m.dpas_latency_cycles = 65.24;
            m.msg_latency_cycles = 65.95;
            m.msg_bytes_per_cycle = 11.77;
            m.inflight_depth = 8.963;
            m.mem_latency_cycles = 909.8;
            m.barrier_sg_cycles = 12.13;
            m.causal_tail_frac = 1.983;
            break;
        case compute::gpu_arch_t::xe3p:
            // Fitted on NVL+CRI: 622 problems, 9470 configs, 4-fold CV regret
            // 1.055 (legacy table 1.859), in-sample 1.049
            m.clock_ghz = 2.5;
            m.mma_cycles_per_flop = 0.004135;
            m.fma_cycles_per_flop = 1.0 / 32;
            m.mem_bw_discrete = 806.7;
            m.mem_bw_integrated = 80.02;
            m.l3_bw = 1926.0;
            m.softmax_cycles_per_elem = 0.5997;
            m.slm_cycles_per_byte = 0.00369;
            m.load_cycles_per_byte = 3.374;
            m.barrier_cycles = 1016.0;
            m.wg_fixed_cycles = 1213.0;
            m.io_cycles_per_byte = 0.08494;
            m.overlap_threads = 3.115;
            m.partial_wave_c0 = 1.0;
            m.partial_wave_c1 = 1.18611e-14;
            m.l3_fraction = 1.359;
            m.launch_cycles = 1083.0;
            m.mix_power = 2.062;
            m.dpas_issue_cycles = 2.0;
            m.dpas_latency_cycles = 24.0;
            m.msg_latency_cycles = 65.95;
            m.msg_bytes_per_cycle = 6.302;
            m.inflight_depth = 11.51;
            m.mem_latency_cycles = 909.8;
            m.barrier_sg_cycles = 9.45;
            m.causal_tail_frac = 1.203;
            break;
        default: break;
    }
    return m;
}

// Smallest k block a strategy loads, used by the feasibility gate
constexpr int fwd_min_k_load = 16;

struct grf_estimate_t {
    int kq, vs; // registers of each microkernel: A, B blocks and C tile
    int host_reserve; // host values live across the calls: C tiles, softmax
};

// with_upconvert counts the wider copy of a sub-16-bit operand as gemmstone
// does; the feasibility gate skips it (shipped int8 64-row VS tiles build)
grf_estimate_t estimate_regs(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, int k_load, bool with_upconvert = true) {
    const int grf = hw.grf_bytes > 0 ? hw.grf_bytes : 32;
    const int compute_bits = p.f32 ? 32 : 16;
    auto regs = [&](double bytes) { return (int)std::ceil(bytes / grf); };
    // A and B blocks, their upconverted copies and the C tile
    auto ukernel_regs
            = [&](int um, int un, int a_bits, int b_bits, int c_bytes) {
        int a = regs(um * k_load * a_bits / 8.0);
        int b = regs(un * k_load * b_bits / 8.0);
        if (with_upconvert && a_bits != compute_bits)
            a += regs(um * k_load * compute_bits / 8.0);
        if (with_upconvert && b_bits != compute_bits)
            b += regs(un * k_load * compute_bits / 8.0);
        return a + b + regs((double)um * un * c_bytes);
    };
    const int kq_c_bytes = p.kq_f16_acc ? 2 : 4;
    const int vs_c_bytes = p.vs_f16_acc ? 2 : 4;
    grf_estimate_t e;
    e.kq = ukernel_regs(
            c.unroll_m_kq, c.unroll_n_kq, p.k_bits, p.q_bits, kq_c_bytes);
    e.vs = ukernel_regs(
            c.unroll_m_vs, c.unroll_n_vs, p.v_bits, p.q_bits, vs_c_bytes);
    e.host_reserve = regs(c.unroll_m_kq * c.unroll_n_kq * kq_c_bytes
            + c.unroll_m_vs * c.unroll_n_vs * vs_c_bytes
            + 3 * c.unroll_n_kq * (int)sizeof(float));
    return e;
}

int align_class(int a) {
    if (a >= 64) return 64;
    if (a >= 16) return 16;
    if (a >= 4) return 4;
    return a > 0 ? a : 1;
}

std::string trim(const std::string &s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Built-in entries, "key config" per line, from tune_fwd_config.py
// --table-out, verified with benchdnn correctness. A key may end in
// ":eu<N>" to apply only to devices with N EUs, preferred over the
// untagged arch-wide line
// clang-format off
const char *const fwd_table_data[] = {
        nullptr,
};
// clang-format on

void add_table_line(
        std::unordered_map<std::string, fwd_config_t> &t, std::string line) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    line = trim(line);
    if (line.empty()) return;
    std::istringstream ss(line);
    std::string key, cfg;
    if (!(ss >> key >> cfg)) return;
    fwd_config_t c {};
    if (fwd_parse_config(cfg, c)) t[key] = c;
}

// Device-tagged line first, arch-wide line second
bool lookup_in(const std::unordered_map<std::string, fwd_config_t> &t,
        const std::string &key, int eu_count, fwd_config_t &c,
        bool *device_specific) {
    auto it = t.find(key + ":eu" + std::to_string(eu_count));
    const bool tagged = (it != t.end());
    if (!tagged) it = t.find(key);
    if (it == t.end()) return false;
    c = it->second;
    if (device_specific) *device_specific = tagged;
    return true;
}

const std::unordered_map<std::string, fwd_config_t> &fwd_table() {
    static const std::unordered_map<std::string, fwd_config_t> table = []() {
        std::unordered_map<std::string, fwd_config_t> t;
        for (const char *const e : fwd_table_data)
            if (e) add_table_line(t, e);
        return t;
    }();
    return table;
}

} // namespace

int fwd_hw_t::max_wg_items(int grfs) const {
    const int tpe = threads_per_eu(grfs);
    const int base = threads_per_eu_128 > 0 ? threads_per_eu_128 : 1;
    const long device_limit = (long)max_wg_items_128 * tpe / base;
    const long sg_limit = (long)tpe * eus_per_subslice * subgroup_size;
    return (int)std::min(device_limit, sg_limit);
}

const std::vector<std::string> &fwd_model_coef_names() {
    static const std::vector<std::string> names = []() {
        std::vector<std::string> n;
        for (const auto &e : coef_table)
            n.emplace_back(e.name);
        return n;
    }();
    return names;
}

bool fwd_model_coef_set(
        fwd_model_coefs_t &m, const std::string &name, double v) {
    for (const auto &e : coef_table)
        if (name == e.name) {
            m.*(e.member) = v;
            return true;
        }
    return false;
}

bool fwd_model_coef_set(fwd_model_coefs_t &m, int idx, double v) {
    if (idx < 0 || idx >= coef_count) return false;
    m.*(coef_table[idx].member) = v;
    return true;
}

double fwd_model_coef_get(const fwd_model_coefs_t &m, int idx) {
    if (idx < 0 || idx >= coef_count) return 0;
    return m.*(coef_table[idx].member);
}

fwd_model_coefs_t fwd_model_coefs(compute::gpu_arch_t arch) {
    return seed_coefs(arch);
}

int fwd_slm_bytes(const fwd_config_t &c, const fwd_problem_t &p) {
    // micro.cl layout: Q_slm + S_slm + S_sum_slm + S_max_slm; the
    // microkernels request no SLM of their own
    const int q_bytes = std::max(p.q_bits, 8) / 8;
    const int q_tile = c.unroll_n_kq * c.wg_n_kq;
    const int kv_tile = c.unroll_m_kq * c.wg_m_kq;
    const int q_slm = p.d_max_kq * q_tile * q_bytes;
    const int s_slm = kv_tile * q_tile * q_bytes;
    const int s_sum = q_tile * c.wg_m_kq * (int)sizeof(float);
    const int s_max = q_tile * (int)sizeof(float);
    return q_slm + s_slm + s_sum + s_max;
}

int fwd_estimate_grfs(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, const fwd_model_coefs_t &m) {
    const grf_estimate_t e
            = estimate_regs(c, p, hw, std::max(1, (int)m.grf_k_load));
    // Below XeHPC the host's live values share the 128-GRF budget
    const int reserve
            = (hw.arch < compute::gpu_arch_t::xe_hpc) ? e.host_reserve : 0;
    const int budget = 128 - reserve;
    return (e.kq <= budget && e.vs <= budget) ? 128 : 256;
}

bool fwd_describe(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, const fwd_model_coefs_t &m, fwd_candidate_t &out,
        std::string *reason) {
    auto fail = [&](const char *why) {
        if (reason) *reason = why;
        return false;
    };
    const int vals[] = {c.unroll_m_kq, c.unroll_n_kq, c.unroll_m_vs,
            c.unroll_n_vs, c.wg_m_kq, c.wg_n_kq, c.wg_m_vs, c.wg_n_vs};
    for (int v : vals)
        if (v <= 0) return fail("non-positive config value");

    out = fwd_candidate_t {};
    out.config = c;
    out.kv_tile = c.unroll_m_kq * c.wg_m_kq;
    out.q_tile = c.unroll_n_kq * c.wg_n_kq;
    out.v_tile = c.unroll_m_vs * c.wg_m_vs;
    out.sg_per_wg = c.wg_m_kq * c.wg_n_kq;

    // Dispatch identities from micro.cpp, plus the implicit one: both GEMM
    // grids are indexed with the same subgroup id
    if (out.q_tile != c.unroll_n_vs * c.wg_n_vs)
        return fail("KQ and VS work-group tile N differ");
    if (c.unroll_n_kq % c.unroll_n_vs != 0)
        return fail("KQ subgroup tile N not divisible by VS subgroup tile N");
    if (out.v_tile < p.d_v)
        return fail("VS work-group tile M below the value head size");
    // Stricter than the kernel: the hand-tuned xe2 decode config
    // 64,16,16,16,8,1,4,1 runs VS on fewer subgroups and stays legacy
    if (out.sg_per_wg != c.wg_m_vs * c.wg_n_vs)
        return fail("KQ and VS subgroup counts differ");
    // Shipped configs stop at 32 subgroups, 64 failed to generate on A750
    if (out.sg_per_wg > 32) return fail("more than 32 subgroups");
    // Mirrors the problem-level dispatch check in micro.cpp
    if (utils::one_of(
                hw.arch, compute::gpu_arch_t::xe2, compute::gpu_arch_t::xe3)
            && p.k_bits == 16 && p.k_align < 4 && p.queries == 1 && p.keys > 64)
        return fail("decode over 2-byte aligned K rows on xe2/xe3");

    const int min_unroll = (hw.arch < compute::gpu_arch_t::xe_hpc) ? 8 : 16;
    const int unrolls[]
            = {c.unroll_m_kq, c.unroll_n_kq, c.unroll_m_vs, c.unroll_n_vs};
    for (int u : unrolls)
        if (u % min_unroll != 0 || u > 64)
            return fail("unroll outside the supported range");

    // Generator ranges, from the shipped table and A750 sweeps
    const int max_unroll_kq = (hw.arch < compute::gpu_arch_t::xe_hpc) ? 32 : 64;
    if (c.unroll_m_kq > max_unroll_kq || c.unroll_n_kq > max_unroll_kq)
        return fail("KQ unroll above the generator range for this arch");
    if (c.unroll_n_vs > 32) return fail("VS unroll N above 32");
    if (c.wg_n_kq > 8 || c.wg_n_vs > 8)
        return fail("more than 8 subgroups along N");
    if (c.wg_m_kq > 32 || c.wg_m_vs > 32)
        return fail("more than 32 subgroups along M");
    const int kq_elems = c.unroll_m_kq * c.unroll_n_kq;
    // The cooperative K load cannot split a tile over 3, 5 or 6 subgroups
    // along N ("Cooperative operation cannot be split evenly"): A750 wg 6x4
    // in either layout, PVC coverage and compute sweeps for 3, 5 and 6
    if ((c.wg_n_kq & (c.wg_n_kq - 1)) != 0)
        return fail("KQ needs a power-of-two subgroup count along N");
    if (hw.arch < compute::gpu_arch_t::xe_hpc) {
        // A750: 64-row VS tiles only build with sub-16-bit V, 32-row KQ tiles
        // need more than one subgroup along N
        if (c.unroll_m_vs > 32 && p.v_bits >= 16)
            return fail("64-row VS tile with 16-bit V below XeHPC");
        if (c.unroll_m_kq == 32 && c.wg_n_kq == 1)
            return fail("32-row KQ tile with one subgroup along N below XeHPC");
        // KPI sweeps: a 512-element KQ tile over a 32-column Q tile has no
        // strategy for 2-byte aligned K rows (K of 77, 1025, ...) and runs
        // out of resources with 32 subgroups from D=256 on; an odd
        // work-group dimension fails preflight in the systolic generator
        // unless the other dimension is 1 (5x1 builds, 5x2 and 4x5 do not)
        if (p.k_align < 4 && kq_elems > 256 && out.q_tile < 64)
            return fail(
                    "512-element KQ tile over a 32-column Q tile with "
                    "2-byte aligned K below XeHPC");
        if (p.d_max_kq >= 256 && kq_elems > 256 && out.sg_per_wg >= 32)
            return fail(
                    "512-element KQ tile with 32 subgroups at D >= 256 "
                    "below XeHPC");
        // Coverage sweep: with K transposed (head dimension contiguous) a
        // 512-element KQ tile fails the kernel build with CL_OUT_OF_RESOURCES
        // over a Q tile below 64 columns, and from D=128 on also with 24 or
        // more subgroups; the microkernels themselves generate
        if (p.transpose_k && kq_elems > 256
                && (out.q_tile < 64
                        || (out.sg_per_wg >= 24 && p.d_max_kq >= 128)))
            return fail("512-element KQ tile with transposed K below XeHPC");
        // D=16/32 sweep: the VS microkernel fails preflight with a single
        // subgroup along M and several along N (wg 1x8); PVC builds those
        if (c.wg_m_vs == 1 && c.wg_n_vs > 1)
            return fail(
                    "VS tile with one subgroup along M and several along "
                    "N below XeHPC");
        // D=512 coverage sweep: with 32 subgroups the kernel build runs out
        // of resources once the KQ tile reaches 256 elements and the VS
        // tile 512, or with transposed K; 16-subgroup work-groups build
        if (p.d_max_kq >= 512 && out.sg_per_wg >= 32 && kq_elems >= 256
                && (p.transpose_k || c.unroll_m_vs * c.unroll_n_vs >= 512))
            return fail(
                    "KQ and VS tiles too large for 32 subgroups at D >= 512 "
                    "below XeHPC");
        if (!p.fma) {
            auto odd_pair = [](int a, int b) {
                return a > 1 && b > 1 && (a % 2 != 0 || b % 2 != 0);
            };
            if (odd_pair(c.wg_m_kq, c.wg_n_kq)
                    || odd_pair(c.wg_m_vs, c.wg_n_vs))
                return fail("odd work-group dimension below XeHPC");
        }
    }
    // PVC D=512 sweep: a 16-column VS tile over 32 subgroups along M does
    // not generate with 16-bit V; int8 and f32 (shipped decode) entries do
    if (hw.arch >= compute::gpu_arch_t::xe_hpc && p.v_bits == 16
            && c.unroll_n_vs == 16 && c.wg_m_vs == 32)
        return fail(
                "16-column VS tile with 32 subgroups along M and 16-bit V "
                "at or above XeHPC");

    // PVC K=77 landscape and D=512 decode: a 1024-element KQ tile (64x16 or
    // 32x32) with one subgroup along N has no strategy for 2-byte aligned K
    // rows ("Insufficient registers"); two subgroups along N and aligned K
    // build, the 32x32 tile over 32 subgroups is a D=512 landscape winner;
    // BMG builds and verifies the same tiles
    if (hw.arch == compute::gpu_arch_t::xe_hpc && p.k_align < 4
            && c.wg_n_kq == 1 && kq_elems >= 1024)
        return fail(
                "1024-element KQ tile with one subgroup along N and 2-byte "
                "aligned K on XeHPC");

    // PVC, BMG and PTL f32 landscapes: every 64-row VS tile and 64-column KQ tile
    // failed
    if ((hw.arch == compute::gpu_arch_t::xe_hpc
                || hw.arch == compute::gpu_arch_t::xe2
                || hw.arch == compute::gpu_arch_t::xe3)
            && p.fma && p.f32 && (c.unroll_m_vs > 32 || c.unroll_n_kq > 32))
        return fail("64-row VS or 64-column KQ tile with f32 FMA");
    // CRI f32 landscape: the KQ microkernel runs out of registers ("No
    // matching kernel") with any KQ tile wider than 16 columns
    if (hw.arch == compute::gpu_arch_t::xe3p && p.fma && p.f32
            && (c.unroll_m_vs > 32 || c.unroll_n_kq > 16))
        return fail("64-row VS or wide KQ tile with f32 FMA on Xe3p");
    // "Cooperative operation cannot be split evenly" with 8 along N
    if ((hw.arch == compute::gpu_arch_t::xe_hpc
                || hw.arch == compute::gpu_arch_t::xe2
                || hw.arch == compute::gpu_arch_t::xe3
                || hw.arch == compute::gpu_arch_t::xe3p)
            && p.fma && p.f32 && (c.wg_n_kq > 4 || c.wg_n_vs > 4))
        return fail("more than 4 subgroups along N with f32 FMA");
    // CRI f16 sweeps: wrong results on every D=32 graph that ran it
    if (hw.arch == compute::gpu_arch_t::xe3p && p.d_max_v <= 32
            && c.unroll_m_vs == 32 && c.unroll_n_vs == 32 && c.wg_m_vs == 1)
        return fail(
                "32x32 VS tile with one subgroup along M at D=32 on "
                "Xe3p");

    // BMG, PTL, NVL and PVC (f32) checks: these build but fail correctness
    // (D=80 wg 5, D=96 wg 3, 6)
    if (hw.arch >= compute::gpu_arch_t::xe_hpc
            && (c.wg_m_kq & (c.wg_m_kq - 1)) != 0)
        return fail(
                "KQ subgroup count along M not a power of two from XeHPC on");

    // Microkernels plus host state must fit the large GRF mode even with
    // the smallest k block
    const grf_estimate_t fit
            = estimate_regs(c, p, hw, fwd_min_k_load, /*with_upconvert=*/false);
    if (fit.kq + fit.host_reserve > 256 || fit.vs + fit.host_reserve > 256)
        return fail("register footprint exceeds the 256-GRF mode");

    out.slm_bytes = fwd_slm_bytes(c, p);
    if (hw.slm_per_wg > 0 && out.slm_bytes > hw.slm_per_wg)
        return fail("SLM exceeds the work-group limit");

    out.grfs = fwd_estimate_grfs(c, p, hw, m);
    const int items = out.sg_per_wg * hw.subgroup_size;
    if (hw.max_wg_items_128 > 0 && items > hw.max_wg_items(out.grfs))
        return fail("work-group exceeds the device limit");

    int by_threads = 64, by_slm = 64;
    if (hw.eus_per_subslice > 0)
        by_threads = hw.eus_per_subslice * hw.threads_per_eu(out.grfs)
                / out.sg_per_wg;
    if (hw.slm_per_subslice > 0)
        by_slm = hw.slm_per_subslice / std::max(out.slm_bytes, 1);
    out.wg_per_subslice = std::min(by_threads, by_slm);
    if (out.wg_per_subslice < 1) return fail("no work-group fits a subslice");

    out.cost_us = fwd_estimate_cost(p, hw, out, m);
    return true;
}

bool fwd_config_valid(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, std::string *reason) {
    fwd_candidate_t unused;
    return fwd_describe(c, p, hw, fwd_model_coefs(hw.arch), unused, reason);
}

double fwd_estimate_cost(const fwd_problem_t &p, const fwd_hw_t &hw,
        const fwd_candidate_t &c, const fwd_model_coefs_t &m) {
    const double sg = std::max(c.sg_per_wg, 1);
    const double q_eff = (double)std::max<dim_t>(p.effective_queries(), 1);
    const double n_q_tiles = std::ceil(q_eff / c.q_tile);
    const double n_wg = (double)std::max<dim_t>(p.batch_heads, 1) * n_q_tiles;
    const double keys = (double)std::max<dim_t>(p.keys, 1);

    /* Key iterations per work-group, causal masks stop each q tile early */
    double iters_mean = std::ceil(keys / c.kv_tile);
    double iters_max = iters_mean;
    if (p.mask == fwd_mask_t::causal_top_left
            || p.mask == fwd_mask_t::causal_bottom_right) {
        const dim_t nq = (dim_t)n_q_tiles;
        const dim_t stride = std::max<dim_t>(1, nq / 1024);
        double sum = 0;
        dim_t count = 0;
        iters_max = 0;
        for (dim_t j = 0; j < nq; j += stride) {
            const double q_end = std::min(q_eff, (double)(j + 1) * c.q_tile);
            double k_end = (p.mask == fwd_mask_t::causal_top_left)
                    ? std::min(keys, q_end)
                    : std::min(keys, q_end + (keys - q_eff));
            k_end = std::max(k_end, 0.0);
            const double it = std::ceil(k_end / c.kv_tile);
            sum += it;
            iters_max = std::max(iters_max, it);
            count++;
        }
        iters_mean = sum / (double)std::max<dim_t>(count, 1);
    }

    /* Per-iteration work of one subgroup, in cycles */
    const auto &cfg = c.config;
    const int kblk = p.f32 ? 8 : 16; // k elements per systolic block
    const double kb_kq = std::ceil((double)p.d_qk / kblk);
    const double kb_vs = std::ceil((double)c.kv_tile / kblk);

    double t_core;
    if (p.fma) {
        // FMA kernels: throughput with a small-tile penalty, not swept yet
        auto shape = [&](int um, int un) {
            return 1.0 + m.unroll_overhead * 32.0 * (1.0 / um + 1.0 / un);
        };
        const double flops_kq = 2.0 * c.kv_tile * c.q_tile * p.d_qk;
        const double flops_vs = 2.0 * c.v_tile * c.q_tile * c.kv_tile;
        t_core = (flops_kq * shape(cfg.unroll_m_kq, cfg.unroll_n_kq)
                         + flops_vs * shape(cfg.unroll_m_vs, cfg.unroll_n_vs))
                / sg * m.fma_cycles_per_flop;
    } else {
        // Systolic k-block chains: (um/8) x (un/simd) dpas per block, at least
        // the pipe latency each
        const double simd = std::max(hw.subgroup_size, 1);
        const double n_dpas_kq
                = (cfg.unroll_m_kq / 8.0) * (cfg.unroll_n_kq / simd);
        const double n_dpas_vs
                = (cfg.unroll_m_vs / 8.0) * (cfg.unroll_n_vs / simd);
        t_core = kb_kq
                        * std::max(n_dpas_kq * m.dpas_issue_cycles,
                                m.dpas_latency_cycles)
                + kb_vs
                        * std::max(n_dpas_vs * m.dpas_issue_cycles,
                                m.dpas_latency_cycles);
        // Throughput floor: a fit that drives the chain to zero on
        // latency-bound data still charges compute-bound shapes
        const double flops = 2.0 * c.kv_tile * c.q_tile * p.d_qk
                + 2.0 * c.v_tile * c.q_tile * c.kv_tile;
        t_core = std::max(t_core, flops / sg * m.mma_cycles_per_flop);
    }

    const double s_elems = (double)c.kv_tile * c.q_tile;
    const double t_sm = s_elems / sg * m.softmax_cycles_per_elem;
    const double t_slm
            = 2.0 * s_elems * (p.q_bits / 8.0) * m.slm_cycles_per_byte;

    auto align_pen = [&](int a) {
        return a >= 16 ? 1.0
                       : (a >= 4 ? m.align4_penalty : m.unaligned_penalty);
    };
    // K/V loads: one message per k-block of unroll_m rows, misalignment
    // lowers the bytes per cycle
    const double msg_kq = cfg.unroll_m_kq * kblk * p.k_bits / 8.0;
    const double msg_vs = cfg.unroll_m_vs * kblk * p.v_bits / 8.0;
    const double bpc = std::max(m.msg_bytes_per_cycle, 1e-3);
    double t_ld = kb_kq
                    * std::max(m.msg_latency_cycles,
                            msg_kq * align_pen(p.k_align) / bpc)
            + kb_vs
                    * std::max(m.msg_latency_cycles,
                            msg_vs * align_pen(p.v_align) / bpc);
    // Mask, scales and zero points, per byte
    double extra_bytes = 0;
    if (p.mask == fwd_mask_t::buffer)
        extra_bytes += c.kv_tile * (p.mask_broadcast_q ? 1.0 : (double)c.q_tile)
                * p.mask_bits / 8.0;
    if (p.k_quantized)
        extra_bytes += c.kv_tile
                * (p.k_group_size > 0 ? (double)p.d_qk / p.k_group_size : 1.0)
                * 3.0;
    if (p.v_quantized)
        extra_bytes += c.kv_tile
                * (p.v_group_size > 0 ? (double)p.d_v / p.v_group_size : 1.0)
                * 3.0;
    t_ld += extra_bytes / sg * m.load_cycles_per_byte;

    double dq_elems = 0;
    if (p.k_quantized) dq_elems += c.kv_tile * (double)p.d_qk;
    if (p.v_quantized) dq_elems += c.kv_tile * (double)p.d_v;
    const double t_dq = dq_elems / sg * m.dequant_cycles_per_elem;

    // Two barriers per iteration with their SLM reductions
    const double t_bar = 2.0 * (m.barrier_cycles + m.barrier_sg_cycles * sg);
    const double t_iter = t_core + t_sm + t_slm + t_ld + t_dq + t_bar;

    const double io_bytes = (double)p.d_max_kq * c.q_tile * p.q_bits / 8.0
            + (double)p.d_v * c.q_tile * p.dst_bits / 8.0;
    const double t_wg = m.wg_fixed_cycles + io_bytes * m.io_cycles_per_byte
            + iters_mean * t_iter;

    /* Occupancy: resident work-groups share a subslice's EUs, a wave takes
       a whole work-group time however full it is */
    const double subslices = std::max(hw.subslices(), 1);
    const double wg_per_ss = std::max(c.wg_per_subslice, 1);
    const double eus_per_ss = std::max(hw.eus_per_subslice, 1);
    auto contention = [&](double resident_wgs) {
        return std::max(1.0,
                resident_wgs * sg / eus_per_ss
                        / std::max(m.overlap_threads, 1e-3));
    };
    const double capacity = subslices * wg_per_ss;
    const double full = std::floor(n_wg / capacity);
    const double rem = n_wg - full * capacity;
    double t_compute = full * t_wg * contention(wg_per_ss);
    if (rem > 0)
        t_compute += t_wg * contention(std::ceil(rem / subslices))
                * (m.partial_wave_c0 + m.partial_wave_c1 * rem / capacity);
    // Causal tail: the longest work-groups run last, partly alone
    if (iters_max > iters_mean)
        t_compute += m.causal_tail_frac * (iters_max - iters_mean) * t_iter
                * contention(1.0);

    /* Memory: K/V are re-read once per q tile, L3 absorbs the re-reads
       while the resident heads' K/V fit */
    const double kv_row_bytes
            = ((double)p.d_qk * p.k_bits + (double)p.d_v * p.v_bits) / 8.0;
    const double kv_head_bytes = keys * kv_row_bytes;
    const double total_kv = n_wg * iters_mean * c.kv_tile * kv_row_bytes;
    const double kv_heads = (p.queries == 1)
            ? (double)p.batch_heads
            : (double)p.batch_heads / std::max(p.kv_group_size, 1);
    const double unique_kv = std::min(total_kv, kv_heads * kv_head_bytes);
    const double qio_bytes = (double)p.batch_heads * q_eff
            * ((double)p.d_qk * p.q_bits + (double)p.d_v * p.dst_bits) / 8.0;
    const double resident_heads
            = std::min((double)std::max<dim_t>(p.batch_heads, 1),
                    std::max(1.0, std::ceil(capacity / n_q_tiles)));
    const bool l3_reuse = hw.l3_bytes > 0
            && resident_heads * kv_head_bytes
                    <= m.l3_fraction * (double)hw.l3_bytes;
    const double bw = hw.integrated ? m.mem_bw_integrated : m.mem_bw_discrete;
    double t_mem = l3_reuse
            ? (unique_kv + qio_bytes) / bw + (total_kv - unique_kv) / m.l3_bw
            : (total_kv + qio_bytes) / bw;
    // Little's law: achieved bandwidth = bytes in flight / latency
    const double active_threads = std::min(n_wg, capacity) * sg;
    const double inflight = active_threads
            * (cfg.unroll_m_kq * 16 * p.k_bits / 8.0
                    + cfg.unroll_m_vs * 16 * p.v_bits / 8.0)
            * m.inflight_depth;
    const double bw_eff
            = std::min(bw, inflight / std::max(m.mem_latency_cycles, 1.0));
    t_mem *= bw / std::max(bw_eff, 1e-6);

    /* Smooth max of compute and memory keeps the ranking strict */
    const double pw = std::max(m.mix_power, 1.0);
    const double mix
            = std::pow(std::pow(t_compute, pw) + std::pow(t_mem, pw), 1.0 / pw);
    const double cycles = m.launch_cycles + mix;
    return cycles / (m.clock_ghz * 1e3);
}

std::vector<fwd_candidate_t> fwd_enumerate(const fwd_problem_t &p,
        const fwd_hw_t &hw, const fwd_model_coefs_t &m) {
    std::vector<int> unrolls = (hw.arch < compute::gpu_arch_t::xe_hpc)
            ? std::vector<int> {8, 16, 32, 64}
            : std::vector<int> {16, 32, 64};
    std::vector<int> wgs = {1, 2, 4, 8, 16, 32};
    // Exact covers of a non-power-of-two value head size (80 = 16 x 5)
    for (int u : unrolls) {
        if (p.d_v <= 0 || p.d_v % u != 0) continue;
        const int w = p.d_v / u;
        if (w >= 1 && w <= 32
                && std::find(wgs.begin(), wgs.end(), w) == wgs.end())
            wgs.push_back(w);
    }
    std::sort(wgs.begin(), wgs.end());

    const int max_v_tile = 2 * std::max(p.d_max_v, p.d_v);
    const int max_items
            = hw.max_wg_items_128 > 0 ? hw.max_wg_items(128) : INT_MAX;

    std::vector<fwd_candidate_t> out;
    for (int um_kq : unrolls)
        for (int un_kq : unrolls)
            for (int wm_kq : wgs)
                for (int wn_kq : wgs) {
                    const int sg = wm_kq * wn_kq;
                    if (sg * hw.subgroup_size > max_items) continue;
                    const int q_tile = un_kq * wn_kq;
                    for (int um_vs : unrolls)
                        for (int un_vs : unrolls) {
                            if (un_kq % un_vs != 0 || q_tile % un_vs != 0)
                                continue;
                            const int wn_vs = q_tile / un_vs;
                            if (sg % wn_vs != 0) continue;
                            const int wm_vs = sg / wn_vs;
                            const int v_tile = um_vs * wm_vs;
                            if (v_tile < p.d_v || v_tile >= max_v_tile)
                                continue;
                            const fwd_config_t c {um_kq, un_kq, um_vs, un_vs,
                                    wm_kq, wn_kq, wm_vs, wn_vs};
                            fwd_candidate_t cand;
                            if (fwd_describe(c, p, hw, m, cand))
                                out.push_back(cand);
                        }
                }
    std::stable_sort(out.begin(), out.end(),
            [](const fwd_candidate_t &a, const fwd_candidate_t &b) {
        return a.cost_us < b.cost_us;
    });
    return out;
}

fwd_key_t fwd_make_key(const fwd_problem_t &p, const fwd_hw_t &hw) {
    fwd_key_t k {};
    k.arch = hw.arch;
    k.d_qk = p.d_qk;
    k.d_v = p.d_v;
    k.keys = utils::rnd_up_pow2(std::max<dim_t>(p.keys, 32));
    const dim_t q = p.effective_queries();
    k.queries = q <= 1                  ? 1
            : q <= fwd_thin_q_threshold ? fwd_thin_q_threshold
                                        : utils::rnd_up_pow2(q);
    k.batch_heads = utils::rnd_up_pow2(std::max<dim_t>(p.batch_heads, 1));
    k.mask = (int)p.mask;
    k.q_bits = p.q_bits;
    k.k_bits = p.k_bits;
    k.v_bits = p.v_bits;
    k.dst_bits = p.dst_bits;
    k.k_quantized = p.k_quantized;
    k.v_quantized = p.v_quantized;
    k.k_group_size = p.k_quantized ? p.k_group_size : 0;
    k.v_group_size = p.v_quantized ? p.v_group_size : 0;
    k.fma = p.fma;
    k.f32 = p.f32;
    k.kq_f16_acc = p.kq_f16_acc;
    k.vs_f16_acc = p.vs_f16_acc;
    k.q_align = align_class(p.q_align);
    k.k_align = align_class(p.k_align);
    k.v_align = align_class(p.v_align);
    k.dst_align = align_class(p.dst_align);
    k.integrated = hw.integrated;
    k.training = p.training;
    k.transpose_k = p.transpose_k;
    return k;
}

std::string fwd_key_t::str() const {
    std::ostringstream s;
    s << compute::to_string(arch) << ":d" << d_qk << "x" << d_v << ":k" << keys
      << ":q" << queries << ":bh" << batch_heads << ":m" << mask << ":t"
      << q_bits << "/" << k_bits << "/" << v_bits << "/" << dst_bits << ":qz"
      << (int)k_quantized << (int)v_quantized << ":g" << k_group_size << "/"
      << v_group_size << ":" << (fma ? "fma" : "sys") << (f32 ? 32 : 16)
      << ":acc" << (kq_f16_acc ? 16 : 32) << "/" << (vs_f16_acc ? 16 : 32)
      << ":al" << q_align << "/" << k_align << "/" << v_align << "/"
      << dst_align << ":i" << (int)integrated << ":tr" << (int)training << ":tk"
      << (int)transpose_k;
    return s.str();
}

bool fwd_parse_config(const std::string &s, fwd_config_t &c) {
    int v[8];
    int n = 0;
    std::stringstream ss(s);
    int x;
    while (n < 8 && ss >> x) {
        v[n++] = x;
        if (ss.peek() == ',') ss.ignore();
    }
    if (n != 8) return false;
    c = fwd_config_t {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    return true;
}

std::string fwd_config_str(const fwd_config_t &c) {
    std::ostringstream s;
    s << c.unroll_m_kq << "," << c.unroll_n_kq << "," << c.unroll_m_vs << ","
      << c.unroll_n_vs << "," << c.wg_m_kq << "," << c.wg_n_kq << ","
      << c.wg_m_vs << "," << c.wg_n_vs;
    return s.str();
}

std::string fwd_hw_str(const fwd_hw_t &hw) {
    std::ostringstream s;
    s << "arch=" << compute::to_string(hw.arch) << ";eu=" << hw.eu_count
      << ";ss=" << hw.eus_per_subslice << ";tpe=" << hw.threads_per_eu_128
      << "/" << hw.threads_per_eu_256 << ";grf=" << hw.grf_bytes
      << ";slm=" << hw.slm_per_wg << "/" << hw.slm_per_subslice
      << ";wg=" << hw.max_wg_items_128 << ";l3=" << hw.l3_bytes
      << ";sg=" << hw.subgroup_size << ";integ=" << (int)hw.integrated
      << ";sys=" << (int)hw.systolic;
    return s.str();
}

std::string fwd_problem_str(const fwd_problem_t &p) {
    std::ostringstream s;
    s << "d=" << p.d_qk << "/" << p.d_v << ";dmax=" << p.d_max_kq << "/"
      << p.d_max_v << ";k=" << p.keys << ";q=" << p.queries
      << ";bh=" << p.batch_heads << ";g=" << p.kv_group_size
      << ";mask=" << (int)p.mask << ";mbq=" << (int)p.mask_broadcast_q
      << ";bits=" << p.q_bits << "/" << p.k_bits << "/" << p.v_bits << "/"
      << p.dst_bits << "/" << p.mask_bits << ";qz=" << (int)p.k_quantized << "/"
      << (int)p.v_quantized << ";gs=" << p.k_group_size << "/" << p.v_group_size
      << ";acc=" << (int)p.kq_f16_acc << "/" << (int)p.vs_f16_acc
      << ";f32=" << (int)p.f32 << ";fma=" << (int)p.fma << ";al=" << p.q_align
      << "/" << p.k_align << "/" << p.v_align << "/" << p.dst_align
      << ";tk=" << (int)p.transpose_k << ";tr=" << (int)p.training
      << ";do=" << (int)p.dropout;
    return s.str();
}

namespace {

// "a=b;c=d" into a map, false on a token without '='
bool parse_kv(const std::string &s,
        std::unordered_map<std::string, std::string> &kv) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ';')) {
        item = trim(item);
        if (item.empty()) continue;
        const auto eq = item.find('=');
        if (eq == std::string::npos) return false;
        kv[item.substr(0, eq)] = item.substr(eq + 1);
    }
    return !kv.empty();
}

// "1/2/3" into ints, missing fields stay at their defaults
void parse_slashed(const std::unordered_map<std::string, std::string> &kv,
        const char *name, std::vector<long long> &out) {
    const auto it = kv.find(name);
    if (it == kv.end()) return;
    std::stringstream ss(it->second);
    std::string tok;
    size_t i = 0;
    while (std::getline(ss, tok, '/') && i < out.size())
        out[i++] = std::atoll(tok.c_str());
}

long long kv_int(const std::unordered_map<std::string, std::string> &kv,
        const char *name, long long def) {
    const auto it = kv.find(name);
    return it == kv.end() ? def : std::atoll(it->second.c_str());
}

} // namespace

bool fwd_parse_hw(const std::string &s, fwd_hw_t &hw) {
    std::unordered_map<std::string, std::string> kv;
    if (!parse_kv(s, kv)) return false;
    hw = fwd_hw_t {};
    const auto arch = kv.find("arch");
    if (arch == kv.end()) return false;
    hw.arch = compute::str2gpu_arch(arch->second.c_str());
    if (hw.arch == compute::gpu_arch_t::unknown) return false;
    hw.eu_count = (int)kv_int(kv, "eu", 0);
    hw.eus_per_subslice = (int)kv_int(kv, "ss", 0);
    std::vector<long long> tpe(2, 0), slm(2, 0);
    parse_slashed(kv, "tpe", tpe);
    parse_slashed(kv, "slm", slm);
    hw.threads_per_eu_128 = (int)tpe[0];
    hw.threads_per_eu_256 = (int)tpe[1];
    hw.grf_bytes = (int)kv_int(kv, "grf", 0);
    hw.slm_per_wg = (int)slm[0];
    hw.slm_per_subslice = (int)slm[1];
    hw.max_wg_items_128 = (int)kv_int(kv, "wg", 0);
    hw.l3_bytes = (size_t)kv_int(kv, "l3", 0);
    hw.subgroup_size = (int)kv_int(kv, "sg", 16);
    hw.integrated = kv_int(kv, "integ", 0) != 0;
    hw.systolic = kv_int(kv, "sys", 1) != 0;
    return true;
}

bool fwd_parse_problem(const std::string &s, fwd_problem_t &p) {
    std::unordered_map<std::string, std::string> kv;
    if (!parse_kv(s, kv)) return false;
    p = fwd_problem_t {};
    std::vector<long long> d(2, 0), dmax(2, 0), bits(5, 0), qz(2, 0), gs(2, 0),
            acc(2, 0), al(4, 0);
    parse_slashed(kv, "d", d);
    parse_slashed(kv, "dmax", dmax);
    parse_slashed(kv, "bits", bits);
    parse_slashed(kv, "qz", qz);
    parse_slashed(kv, "gs", gs);
    parse_slashed(kv, "acc", acc);
    parse_slashed(kv, "al", al);
    if (d[0] <= 0) return false;
    p.d_qk = (int)d[0];
    p.d_v = (int)d[1];
    p.d_max_kq = (int)dmax[0];
    p.d_max_v = (int)dmax[1];
    p.keys = kv_int(kv, "k", 0);
    p.queries = kv_int(kv, "q", 0);
    p.batch_heads = kv_int(kv, "bh", 0);
    p.kv_group_size = (int)kv_int(kv, "g", 1);
    p.mask = (fwd_mask_t)kv_int(kv, "mask", 0);
    p.mask_broadcast_q = kv_int(kv, "mbq", 1) != 0;
    p.q_bits = (int)bits[0];
    p.k_bits = (int)bits[1];
    p.v_bits = (int)bits[2];
    p.dst_bits = (int)bits[3];
    p.mask_bits = (int)bits[4];
    p.k_quantized = qz[0] != 0;
    p.v_quantized = qz[1] != 0;
    p.k_group_size = (int)gs[0];
    p.v_group_size = (int)gs[1];
    p.kq_f16_acc = acc[0] != 0;
    p.vs_f16_acc = acc[1] != 0;
    p.f32 = kv_int(kv, "f32", 0) != 0;
    p.fma = kv_int(kv, "fma", 0) != 0;
    p.q_align = (int)al[0];
    p.k_align = (int)al[1];
    p.v_align = (int)al[2];
    p.dst_align = (int)al[3];
    p.transpose_k = kv_int(kv, "tk", 0) != 0;
    p.training = kv_int(kv, "tr", 0) != 0;
    p.dropout = kv_int(kv, "do", 0) != 0;
    return true;
}

bool fwd_lookup(const fwd_key_t &key, int eu_count, fwd_config_t &c,
        bool *device_specific) {
    return lookup_in(fwd_table(), key.str(), eu_count, c, device_specific);
}

bool fwd_select(
        const fwd_problem_t &p, const fwd_hw_t &hw, fwd_selection_t &out) {
    fwd_config_t c {};
    bool device_specific = false;
    if (!fwd_lookup(fwd_make_key(p, hw), hw.eu_count, c, &device_specific))
        return false;
    if (!fwd_describe(c, p, hw, fwd_model_coefs(hw.arch), out.info))
        return false;
    out.config = c;
    out.source = device_specific ? "table_dev" : "table";
    return true;
}

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
