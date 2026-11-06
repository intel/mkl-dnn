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

// Forward SDPA tile selection: problem/hardware description, a lookup
// table on the full key that micro.cpp consults before the hand-tuned
// table, and a tiler plus cost model that enumerate and rank valid configs
// for the sweeps in scripts/sdpa_tuner. Plain data and pure functions,
// usable without an engine; micro.cpp fills fwd_problem_t / fwd_hw_t

#ifndef GPU_INTEL_SDPA_SELECT_HPP
#define GPU_INTEL_SDPA_SELECT_HPP

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/sdpa/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

// Queries at or below this count are thin (decode-like)
constexpr int fwd_thin_q_threshold = 16;

// Device facts that constrain and rank forward configs
struct fwd_hw_t {
    compute::gpu_arch_t arch = compute::gpu_arch_t::unknown;
    int eu_count = 0;
    int eus_per_subslice = 0;
    int threads_per_eu_128 = 0; // thread slots per EU in the 128-GRF mode
    int threads_per_eu_256 = 0; // thread slots per EU in the 256-GRF mode
    int grf_bytes = 0; // register width: 32 B before XeHPC, 64 B after
    int slm_per_wg = 0; // bytes one work-group may allocate
    int slm_per_subslice = 0; // bytes shared by the work-groups on a subslice
    int max_wg_items_128 = 0; // work-group size limit in the 128-GRF mode
    size_t l3_bytes = 0;
    int subgroup_size = 16;
    bool integrated = false;
    bool systolic = true;

    int threads_per_eu(int grfs) const {
        return grfs > 128 ? threads_per_eu_256 : threads_per_eu_128;
    }
    // Mirrors device_info_t::max_wg_size(grfs, subgroup_size)
    int max_wg_items(int grfs) const;
    int subslices() const {
        return eus_per_subslice > 0 ? eu_count / eus_per_subslice : 0;
    }
};

enum class fwd_mask_t : int {
    none = 0,
    causal_top_left,
    causal_bottom_right,
    buffer,
};

// Problem features forward selection may depend on
struct fwd_problem_t {
    int d_qk = 0, d_v = 0; // head sizes
    int d_max_kq = 0, d_max_v = 0; // power-of-two blocks baked into the kernel
    dim_t keys = 0;
    dim_t queries = 0; // per head; 1 for decode
    dim_t batch_heads = 0; // work-groups dispatched along batch x heads
    int kv_group_size = 1; // query heads per kv head
    fwd_mask_t mask = fwd_mask_t::none;
    bool mask_broadcast_q = true; // explicit mask has a single query row
    int q_bits = 16, k_bits = 16, v_bits = 16, dst_bits = 16, mask_bits = 0;
    bool k_quantized = false, v_quantized = false;
    int k_group_size = 0, v_group_size = 0; // 0: common scale/zero point
    bool kq_f16_acc = false, vs_f16_acc = false;
    bool f32 = false; // f32 compute (FMA kernel)
    bool fma = false; // non-systolic microkernels
    int q_align = 0, k_align = 0, v_align = 0, dst_align = 0; // bytes
    bool transpose_k = false;
    bool training = false, dropout = false;

    // GQA decode folds the kv group into the query dimension
    dim_t effective_queries() const {
        return queries == 1 ? kv_group_size : queries;
    }
    bool thin_q() const { return effective_queries() <= fwd_thin_q_threshold; }
    bool quantized() const { return k_quantized || v_quantized; }
};

// Cost model coefficients per architecture, in GPU cycles or bytes per
// cycle; estimates convert to microseconds with clock_ghz
struct fwd_model_coefs_t {
    double clock_ghz;
    double mma_cycles_per_flop; // one subgroup, systolic, at peak (unused)
    double fma_cycles_per_flop; // one subgroup, FMA, at peak
    double unroll_overhead; // operand traffic penalty for small unrolls (FMA)
    double softmax_cycles_per_elem; // per S element per subgroup
    double slm_cycles_per_byte; // S tile write + read through SLM
    double load_cycles_per_byte; // mask / scale loads per subgroup
    double align4_penalty; // load multiplier when only 4-byte aligned
    double unaligned_penalty; // load multiplier below 4-byte alignment
    double dequant_cycles_per_elem; // per dequantized K/V element
    double barrier_cycles; // one work-group barrier with its SLM reduce
    double wg_fixed_cycles; // prologue + epilogue per work-group
    double io_cycles_per_byte; // Q load/pack to SLM and output store
    double overlap_threads; // threads per EU that overlap without slowdown
    double partial_wave_c0, partial_wave_c1; // tail wave cost model
    double mem_bw_discrete, mem_bw_integrated; // bytes per cycle
    double l3_bw; // bytes per cycle for L3 hits
    double l3_fraction; // share of L3 the K/V working set may occupy
    double launch_cycles;
    double grf_k_load; // k elements a microkernel loads per A/B block
    double mix_power; // p of the (compute^p + memory^p)^(1/p) smooth max
    // Systolic k-block chain: n_dpas issues per block, at least the pipe
    // latency, so small tiles are latency bound
    double dpas_issue_cycles, dpas_latency_cycles;
    // K/V loads: one message per k-block, latency or bandwidth share
    double msg_latency_cycles, msg_bytes_per_cycle;
    // Achieved bandwidth bounded by bytes in flight (Little's law)
    double inflight_depth, mem_latency_cycles;
    // Barrier cost grows with the subgroups taking part in the reduction;
    // lets an architecture prefer narrower work-groups at equal occupancy
    double barrier_sg_cycles;
    // Causal shapes: the longest work-groups are issued last, so part of
    // the gap between the longest and the mean work-group runs on an
    // otherwise idle machine; 0 switches the term off
    double causal_tail_frac;
};

// Coefficients for an architecture
fwd_model_coefs_t fwd_model_coefs(compute::gpu_arch_t arch);
// Coefficient names in declaration order, for tooling
const std::vector<std::string> &fwd_model_coef_names();
bool fwd_model_coef_set(
        fwd_model_coefs_t &m, const std::string &name, double v);
bool fwd_model_coef_set(fwd_model_coefs_t &m, int idx, double v);
double fwd_model_coef_get(const fwd_model_coefs_t &m, int idx);

// One valid config with the quantities the model is built from
struct fwd_candidate_t {
    fwd_config_t config {};
    int kv_tile = 0, q_tile = 0, v_tile = 0, sg_per_wg = 0;
    int slm_bytes = 0;
    int grfs = 128; // GRF mode the microkernel estimate lands in
    int wg_per_subslice = 0;
    double cost_us = 0; // model estimate, lower is better
};

// SLM the host kernel allocates for this config (micro.cl layout)
int fwd_slm_bytes(const fwd_config_t &c, const fwd_problem_t &p);
// Expected GRF mode, mirrors chooseMicrokernelGRFMode in gemmstone
int fwd_estimate_grfs(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, const fwd_model_coefs_t &m);

// Fills the derived fields and the cost; false with a reason when the
// config cannot run this problem on this device
bool fwd_describe(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, const fwd_model_coefs_t &m, fwd_candidate_t &out,
        std::string *reason = nullptr);
bool fwd_config_valid(const fwd_config_t &c, const fwd_problem_t &p,
        const fwd_hw_t &hw, std::string *reason = nullptr);

double fwd_estimate_cost(const fwd_problem_t &p, const fwd_hw_t &hw,
        const fwd_candidate_t &c, const fwd_model_coefs_t &m);

// Every valid config for the problem, best estimate first
std::vector<fwd_candidate_t> fwd_enumerate(
        const fwd_problem_t &p, const fwd_hw_t &hw, const fwd_model_coefs_t &m);

// Lookup key: exact on discrete features, bucketed on sizes
struct fwd_key_t {
    compute::gpu_arch_t arch;
    int d_qk, d_v;
    dim_t keys, queries, batch_heads; // bucketed
    int mask;
    int q_bits, k_bits, v_bits, dst_bits;
    bool k_quantized, v_quantized;
    int k_group_size, v_group_size;
    bool fma, f32, kq_f16_acc, vs_f16_acc;
    int q_align, k_align, v_align, dst_align; // alignment classes
    bool integrated, training;
    bool transpose_k; // K layout: the kernel differs, decode time by up to 3x

    std::string str() const;
};
fwd_key_t fwd_make_key(const fwd_problem_t &p, const fwd_hw_t &hw);
bool fwd_parse_config(const std::string &s, fwd_config_t &c);
std::string fwd_config_str(const fwd_config_t &c);
// "name=value;..." descriptions for the verbose line (sdpa_model.py)
std::string fwd_hw_str(const fwd_hw_t &hw);
std::string fwd_problem_str(const fwd_problem_t &p);
// Inverses of the two above, false on a malformed string
bool fwd_parse_hw(const std::string &s, fwd_hw_t &hw);
bool fwd_parse_problem(const std::string &s, fwd_problem_t &p);
// Table hit on the built-in entries ("key config" lines). A line's key may
// end in ":eu<N>" to apply only to devices with N EUs; such a line is
// preferred over the arch-wide one and sets device_specific
bool fwd_lookup(const fwd_key_t &key, int eu_count, fwd_config_t &c,
        bool *device_specific = nullptr);

struct fwd_selection_t {
    fwd_config_t config {};
    const char *source = "none"; // "legacy", "table", "table_dev", "env"
    fwd_candidate_t info {};
};

// Table hit that passes the tiler's constraints; false when the caller
// falls back to the hand-tuned table. The cost model is not a selection
// tier: sweeps found it 1.08x off the best config out of sample where
// legacy is 1.26x, but a config the generator rejects fails primitive
// creation, and only the table entries have been built and verified
bool fwd_select(
        const fwd_problem_t &p, const fwd_hw_t &hw, fwd_selection_t &out);

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
