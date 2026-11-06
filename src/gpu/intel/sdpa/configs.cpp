/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/sdpa/configs.hpp"

#include <algorithm>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

inline property operator|(property a, property b) {
    return (property)((int)a | (int)b);
}
inline property operator&(property a, property b) {
    return (property)((int)a & (int)b);
}
inline property operator^(property a, property b) {
    return (property)((int)a ^ (int)b);
}

inline property &operator|=(property &a, property b) {
    return (property &)((int &)a |= (int)b);
}
inline property &operator&=(property &a, property b) {
    return (property &)((int &)a &= (int)b);
}
inline property &operator^=(property &a, property b) {
    return (property &)((int &)a ^= (int)b);
}

std::string to_string(const config_query_t &q) {
    std::stringstream s;
    s << "arch:" << std::to_string((int)q.arch) << " hs:" << q.head_size
      << " seq:" << q.seq_len
      << " prop:" << ((bool)(q.prop & property::second_token) ? " thinq" : "")
      << ((bool)(q.prop & property::quantized) ? " quant" : "")
      << ((bool)(q.prop & property::integrated) ? " [integ]" : "")
      << ((bool)(q.prop & property::fma) ? " fma" : "")
      << ((bool)(q.prop & property::f32) ? " [f32]" : "")
      << ((bool)(q.prop & property::f16_accumulate) ? " [f16_acc]" : "");
    return s.str();
}
std::string to_string(const config_criteria_t &c) {
    std::stringstream s;
    s << "arch:" << std::to_string((int)c.arch) << " hs:" << c.head_size
      << " seq:" << c.seq_len
      << " prop:" << ((bool)(c.prop & property::second_token) ? " thinq" : "")
      << ((bool)(c.prop & property::quantized) ? " quant" : "")
      << ((bool)(c.prop & property::integrated) ? " integ" : "")
      << ((bool)(c.prop & property::fma) ? " fma" : "")
      << ((bool)(c.prop & property::f32) ? " f32" : "")
      << ((bool)(c.prop & property::f16_accumulate) ? " f16_acc" : "");
    return s.str();
}
std::string to_string(const fwd_config_t &c) {
    std::stringstream s;
    s << c.unroll_m_kq << "," << c.unroll_n_kq << "," << c.unroll_m_vs << ","
      << c.unroll_n_vs << "," << c.wg_m_kq << "," << c.wg_n_kq << ","
      << c.wg_m_vs << "," << c.wg_n_vs;
    return s.str();
}
std::string to_string(const bwd_config_t &c) {
    std::stringstream s;
    s << c.unroll_m_BcBr << "," << c.unroll_n_BcBr << "," << c.unroll_m_DBc
      << "," << c.unroll_n_DBc << "," << c.unroll_m_DBr << "," << c.unroll_n_DBr
      << "," << c.wg_m_BcBr << "," << c.wg_n_BcBr << "," << c.wg_m_DBc << ","
      << c.wg_n_DBc << "," << c.wg_m_DBr << "," << c.wg_n_DBr;
    return s.str();
}

// A matching config is a combination of mandatory and optional requirements
// it is assumed the query criteria are specific whereas key criteria may be approximate
// sorting all available configs will order them from most to least specific allowing
// exact matches to happen before approxiate matches
// head size and sequence length must strictly match the inequality with a caveat for
// the more general key criteria "any = -1"
// properties must match exactly if they are specified in the key criteria
bool criteria_matches(
        const config_criteria_t &key, const config_query_t &query) {
    return ((query.arch == key.arch) && (query.head_size <= key.head_size)
            && ((key.seq_len == -1)
                    || (key.seq_len != -1 && query.seq_len <= key.seq_len))
            && ((((query.prop & property::second_token)
                        == (key.prop & property::second_token)))
                    && (((query.prop & property::quantized)
                            == (key.prop & property::quantized)))
                    && (((query.prop & property::fma)
                            == (key.prop & property::fma)))
                    && (((key.prop & property::f32) == property::none)
                            || ((query.prop & property::f32)
                                    == (key.prop & property::f32)))
                    && (((key.prop & property::f16_accumulate)
                                == property::none)
                            || ((query.prop & property::f16_accumulate)
                                    == (key.prop & property::f16_accumulate)))
                    && (((key.prop & property::integrated) == property::none)
                            || ((query.prop & property::integrated)
                                    == (key.prop & property::integrated)))));
}

bool operator==(const fwd_config_record_t &key, const config_query_t &query) {
    return criteria_matches(key.criteria, query);
}

bool operator==(const bwd_config_record_t &key, const config_query_t &query) {
    return criteria_matches(key.criteria, query);
}

template <typename Enum>
int popcount(Enum e) {
    using U = typename std::underlying_type<Enum>::type;
    U x = static_cast<U>(e);
    int count = 0;
    while (x) {
        x &= (x - 1); // clear lowest set bit
        ++count;
    }
    return count;
}

bool operator<(const config_criteria_t &lhs, const config_criteria_t &rhs) {
    auto num_set_fields = [](const config_criteria_t &crit) {
        int set_fields = 0;
        if (crit.arch != compute::gpu_arch_t::unknown) { set_fields++; }
        if (crit.head_size != -1) { set_fields++; }
        const int n_props = popcount<property>(crit.prop);
        set_fields += n_props;
        return set_fields;
    };

    auto noprops = [](const config_criteria_t &crit) {
        const int n_props = popcount<property>(crit.prop);
        return (n_props == 0);
    };

    int l_set_fields = num_set_fields(lhs);
    int r_set_fields = num_set_fields(rhs);

    // SWO, first sort by arch
    if (lhs.arch != rhs.arch) return lhs.arch < rhs.arch;
    // then by head size
    else if (lhs.head_size != rhs.head_size)
        return lhs.head_size < rhs.head_size;
    else if (noprops(lhs) != noprops(rhs))
        return noprops(rhs);
    // then by most->least set properties (ignores seq len)
    else if (l_set_fields != r_set_fields)
        return (l_set_fields > r_set_fields);
    // then by sequence length (if both defined)
    else if (lhs.seq_len != rhs.seq_len && lhs.seq_len != -1
            && rhs.seq_len != -1)
        return lhs.seq_len < rhs.seq_len;
    // then if single seq_len == -1 prefer defined seq_len
    else if (lhs.seq_len != rhs.seq_len)
        return lhs.seq_len != -1;
    // ensure consistent order if # fields identical
    else if ((lhs.prop & property::fma) != (rhs.prop & property::fma))
        return static_cast<bool>(lhs.prop & property::fma);
    else if ((lhs.prop & property::quantized)
            != (rhs.prop & property::quantized))
        return static_cast<bool>(lhs.prop & property::quantized);
    else if ((lhs.prop & property::second_token)
            != (rhs.prop & property::second_token))
        return static_cast<bool>(lhs.prop & property::second_token);
    else if ((lhs.prop & property::integrated)
            != (rhs.prop & property::integrated))
        return static_cast<bool>(lhs.prop & property::integrated);
    else if ((lhs.prop & property::f32) != (rhs.prop & property::f32))
        return static_cast<bool>(lhs.prop & property::f32);
    else if ((lhs.prop & property::f16_accumulate)
            != (rhs.prop & property::f16_accumulate))
        return static_cast<bool>(lhs.prop & property::f16_accumulate);
    return false;
}

bool operator<(const fwd_config_record_t &lhs, const fwd_config_record_t &rhs) {
    return lhs.criteria < rhs.criteria;
}

bool operator<(const bwd_config_record_t &lhs, const bwd_config_record_t &rhs) {
    return lhs.criteria < rhs.criteria;
}

static auto constexpr second_token = property::second_token;
static auto constexpr quantized = property::quantized;
static auto constexpr integrated = property::integrated;
static auto constexpr fma = property::fma;
static auto constexpr f32 = property::f32;
static auto constexpr f16_accumulate = property::f16_accumulate;

// Kernel configurations: [ arch, head_size, {sequence length}, {properties} ] -> config
static std::vector<fwd_config_record_t> sorted_configs = []() {
    // clang-format off
    std::vector<fwd_config_record_t> configs = {
        // xe_hpg
        {{compute::gpu_arch_t::xe_hpg, 16, fma | f32},    {8, 16, 8, 16, 2, 4, 2, 4}},

        {{compute::gpu_arch_t::xe_hpg, 32},               {16, 16, 16, 16, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, 256},          {16, 16, 16, 16, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, 64},           {16, 16, 16, 8, 4, 4, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, 32},           {8, 8, 8, 8, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpg, 32, second_token}, {8, 32, 16, 8, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe_hpg, 32, quantized},                {32, 16, 16, 16, 2, 8, 2, 8}},
        {{compute::gpu_arch_t::xe_hpg, 32, quantized | second_token}, {32, 16, 8, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 32, fma},                {16, 16, 8, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 32, fma | second_token}, {8, 32, 16, 8, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe_hpg, 32, fma | f32},                            {16, 16, 16, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 32, quantized | fma | f32},                { 8, 16, 16, 16, 16, 1, 16, 1 }},
        {{compute::gpu_arch_t::xe_hpg, 32, second_token | quantized | fma | f32}, {16, 16,  8,  8, 16, 1,  8, 2 }},

        {{compute::gpu_arch_t::xe_hpg, 64},               {32, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128},          {16, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 64},           {16, 16, 8, 8, 16, 4, 8, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, second_token}, {8, 16, 16, 8, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpg, 64, fma},                {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpg, 64, fma | second_token}, {16, 16, 32, 8, 32, 1, 16, 2}},

        {{compute::gpu_arch_t::xe_hpg, 64,      quantized}, {32, 16, 16, 16, 4, 8, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128, quantized}, {16, 16, 16, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 64, 64,  quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpg, 64, 32,  quantized}, {8, 8, 16, 8, 4, 8, 4, 8}},

        {{compute::gpu_arch_t::xe_hpg, 64,      quantized | second_token}, {16, 16, 8, 8, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 64, 128, quantized | second_token}, {16, 8, 8, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 64, 64,  quantized | second_token}, {8, 8, 8, 8, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpg, 64, fma | f32},                            {16, 32, 16, 16,  8, 1,  4, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 64, second_token | fma | f32},             { 8, 16,  8, 16, 16, 1, 16, 1 }},
        {{compute::gpu_arch_t::xe_hpg, 64, quantized | fma | f32},                { 8, 16, 16,  8,  8, 1,  4, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 64, second_token | quantized | fma | f32}, {16, 16,  8,  8, 32, 1, 16, 2 }},


        {{compute::gpu_arch_t::xe_hpg, 80, fma}, {16, 16, 16, 16, 5, 4, 5, 4}},
        {{compute::gpu_arch_t::xe_hpg, 80, fma | f16_accumulate}, {8, 16, 16, 16, 8, 4, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 112},                    {16, 16, 32, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpg, 112, 32},                {16, 16, 16, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 128},                    {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 32},                {16, 16, 16, 8, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 256, second_token}, {8, 16, 32, 8, 8, 1, 4, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128, second_token},      {8, 16, 16, 8, 16, 1, 8, 2}},

        {{compute::gpu_arch_t::xe_hpg, 128,      quantized}, {8, 32, 16, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128, 32,  quantized}, {16, 8, 16, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 64,  quantized}, {8, 8, 16, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 512, quantized}, {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 96,  quantized | second_token}, {8, 8, 8, 8, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128,      quantized | second_token}, {16, 16, 16, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 128, fma},                {8, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 448, fma},           {8, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 128, 448, fma | f16_accumulate}, {8, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpg, 128, fma | second_token}, {32, 16, 32, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 128, fma | f32},                            { 8, 16, 32,  8,  8, 2,  4, 4 }},
        {{compute::gpu_arch_t::xe_hpg, 128, second_token | fma | f32},             { 8, 16, 32,  8, 16, 1,  8, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 128, quantized | fma | f32},                { 8, 16, 16, 16,  8, 1,  8, 1 }},
        {{compute::gpu_arch_t::xe_hpg, 128, second_token | quantized | fma | f32}, {16, 16, 16,  8, 32, 1, 16, 2 }},

        {{compute::gpu_arch_t::xe_hpg, 256},      {16, 16, 32, 8, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 128}, {8, 16, 32, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 32},  {8, 16, 32, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 256,      quantized}, {16, 16, 64, 8, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe_hpg, 256, 512, quantized}, {16, 16, 32, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 64,  quantized}, {8, 8, 32, 8, 8, 4, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 256,     second_token}, {16, 8, 16, 8, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpg, 256, 64, second_token}, {16, 8, 16, 8, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpg, 256, 32, second_token}, {16, 16, 32, 8, 16, 1, 8, 2}},

        {{compute::gpu_arch_t::xe_hpg, 256,     second_token | quantized}, {32, 8, 32, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 256, 96, second_token | quantized}, {8, 8, 16, 8, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpg, 256, fma},                {8, 16, 32, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpg, 256, fma | second_token}, {16, 8, 16, 8, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe_hpg, 256, fma | f32},                            {32,  8, 32,  8, 16, 2, 16, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 256, second_token | fma | f32},             { 8, 16, 32,  8, 16, 1,  8, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 256, quantized | fma | f32},                {16,  8, 32,  8,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe_hpg, 256, second_token | quantized | fma | f32}, { 8, 16, 32,  8, 16, 2,  8, 4 }},

        {{compute::gpu_arch_t::xe_hpg, 512, 64,  quantized}, {8, 8, 64, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 512, 128, quantized}, {8, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, 256, quantized}, {16, 8, 64, 8, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpg, 512,      quantized}, {8, 16, 64, 8, 16, 2, 8, 4}},

        {{compute::gpu_arch_t::xe_hpg, 512, 64,  second_token | quantized}, {8, 16, 32, 8, 32, 1, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, 256, second_token | quantized}, {16, 8, 32, 8, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512,      second_token | quantized}, {16, 8, 16, 8, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe_hpg, 512},               {8, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpg, 512, second_token}, {8, 8, 32, 8, 16, 1, 16, 1}},

        // xe_hpc
        {{compute::gpu_arch_t::xe_hpc, 32},               {16, 64, 32, 16, 4, 2, 1, 8}},
        {{compute::gpu_arch_t::xe_hpc, 32, second_token}, {16, 64, 16, 16, 8, 1, 2, 4}},
        {{compute::gpu_arch_t::xe_hpc, 32, 32},           {16, 16, 16, 16, 2, 4, 2, 4}},
        {{compute::gpu_arch_t::xe_hpc, 32, 32, second_token}, {16, 64, 16, 16, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe_hpc, 64},                   {16, 64, 32, 16, 8, 2, 2, 8}},
        {{compute::gpu_arch_t::xe_hpc, 64, 64},               {32, 32, 32, 16, 4, 2, 2, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 32},               {16, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe_hpc, 64,     second_token}, {32, 32, 32, 16, 4, 1, 2, 2}},
        {{compute::gpu_arch_t::xe_hpc, 64, 64, second_token}, {16, 16, 16, 16, 4, 1, 4, 1}},

        {{compute::gpu_arch_t::xe_hpc, 64, 1024, quantized}, {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 384,  quantized}, {16, 64, 16, 32, 8, 2, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64, 64,   quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe_hpc, 64,       quantized}, {16, 64, 16, 32, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpc, 64, 1152, second_token | quantized}, {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64, 256,  second_token | quantized}, {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64, 96,   second_token | quantized}, {16, 16, 16, 16, 8, 1, 4, 1}},
        {{compute::gpu_arch_t::xe_hpc, 64,       second_token | quantized}, {64, 16, 16, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128},               {16, 64, 32, 16, 16, 2, 4, 8}},
        {{compute::gpu_arch_t::xe_hpc, 128, 64},           {16, 32, 32, 32, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 32},           {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, second_token}, {32, 32, 32, 16, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128,      quantized},              {16, 64, 16, 32, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 128, quantized},              {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 128, 32,  quantized},              {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 128, integrated | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 128,      second_token | quantized}, {16, 16, 16, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128, 512, second_token | quantized}, {16, 16, 16, 16, 32, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128, 512, second_token | quantized | f32 }, {16, 16, 16, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128, 96,  second_token | quantized}, {16, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 128, integrated | second_token | quantized}, {16, 16, 16, 16, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe_hpc, 256},                                 {16, 32, 32, 32, 8,  4,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 256, 64},                             {16, 32, 32, 32, 8,  1,  8, 1}},
        {{compute::gpu_arch_t::xe_hpc, 256,       second_token},             {16, 16, 16, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc, 256,       fma | second_token},       {16, 16, 32, 16, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe_hpc, 256,       quantized},                {16, 32, 32, 32,  8, 4,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 256, 64,   quantized},                {16, 16, 16, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpc, 256,       quantized | second_token}, {16, 16, 16, 16, 32, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 256, 1152, quantized | second_token}, {16, 16, 16, 16, 32, 1, 32, 1}},
        {{compute::gpu_arch_t::xe_hpc, 256, 196,  quantized | second_token}, {16, 16, 16, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc, 512},      {32, 16, 64, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 512, 128}, {16, 16, 64, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 512, 32},  {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 512,       second_token}, {32, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, 1024, second_token}, {64, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, 512,  second_token}, {32, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512, 128,  second_token}, {16, 16, 64, 16, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe_hpc, 576},               {16, 32, 32, 32, 32, 1, 32, 1}},
        {{compute::gpu_arch_t::xe_hpc, 576, second_token}, {32, 16, 32, 16, 32, 1, 31, 1}},

        {{compute::gpu_arch_t::xe_hpc, 512,      quantized}, {16, 32, 64, 16, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 512, 128, quantized}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 512, second_token | quantized}, {16, 16, 32, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe_hpc, 32, fma | second_token}, {16, 16, 16, 16, 32, 1, 32, 1}},

        {{compute::gpu_arch_t::xe_hpc,  32, 1024, fma }, {32, 32, 16, 16,  8, 2,  4, 4}},
        {{compute::gpu_arch_t::xe_hpc,  32,       fma }, {32, 32, 16, 16,  8, 2,  4, 4}},

        {{compute::gpu_arch_t::xe_hpc,  64,  384, fma }, {16, 16, 16, 16,  8, 4,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc,  64, 1024, fma }, {16, 32, 16, 16, 16, 2,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc,  64,       fma }, {16, 32, 16, 16, 16, 2,  8, 4}},

        {{compute::gpu_arch_t::xe_hpc, 128, 384,  fma }, {32, 32, 16, 16, 16, 1, 8, 2 }},
        {{compute::gpu_arch_t::xe_hpc, 128, 1024, fma }, {32, 32, 16, 32,  8, 2,  8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128,       fma }, {32, 32, 16, 32,  8, 2,  8, 2}},

        {{compute::gpu_arch_t::xe_hpc, 256,  384, fma }, {16, 32, 16, 16, 32, 1, 16, 2}},
        {{compute::gpu_arch_t::xe_hpc, 256, 1024, fma }, {16, 16, 32, 16,  8, 4,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 256,       fma }, {16, 16, 32, 16,  8, 4,  8, 4}},

        {{compute::gpu_arch_t::xe_hpc, 512, 1024, fma }, {16, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe_hpc, 512,       fma }, {16, 16, 32, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe_hpc,  16,   f32 | fma }, {16, 32, 16, 16, 4,  1,  2, 2}},
        {{compute::gpu_arch_t::xe_hpc,  32,   f32 | fma }, {16, 32, 16, 16, 4 , 2,  2, 4}},
        {{compute::gpu_arch_t::xe_hpc,  64,   f32 | fma }, {16, 16, 16, 16, 4,  2,  4, 2}},
        {{compute::gpu_arch_t::xe_hpc,  64, 384,   f32 | fma }, {16, 16, 16, 16, 8,  2,  8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 128,   f32 | fma }, {32, 32, 16, 32, 8,  2,  8, 2}},
        {{compute::gpu_arch_t::xe_hpc, 256,   f32 | fma }, {16, 16, 32, 16, 8,  4,  8, 4}},
        {{compute::gpu_arch_t::xe_hpc, 512,   f32 | fma }, {16, 16, 32, 16, 16, 1, 16, 1}},

        // xe2
        {{compute::gpu_arch_t::xe2, 32},               {16, 64, 32, 16, 4, 2, 1, 8}},
        {{compute::gpu_arch_t::xe2, 32, second_token}, {16, 64, 16, 16, 8, 1, 2, 4}},
        {{compute::gpu_arch_t::xe2, 32, 32},           {16, 16, 16, 16, 2, 4, 2, 4}},
        {{compute::gpu_arch_t::xe2, 32, 32, second_token}, {16, 64, 16, 16, 8, 1, 2, 4}},

        {{compute::gpu_arch_t::xe2, 32,       quantized}, {16, 64, 16, 32, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe2, 32, 512,  quantized}, {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 32, 384,  quantized}, {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe2, 32, 128,  quantized}, {16, 64, 16, 32, 8, 1, 4, 2}},
        {{compute::gpu_arch_t::xe2, 32, 32,   quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe2, 32, 1024, integrated | quantized}, {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 32, 384,  integrated | quantized}, {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe2, 32, 128,  integrated | quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},

        {{compute::gpu_arch_t::xe2, 32,      second_token | quantized}, {16, 16, 16, 16, 16, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 32, 768, second_token | quantized}, {64, 16, 16, 16, 16, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 32, 512, second_token | quantized}, {64, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 32, 384, second_token | quantized}, {16, 16, 16, 16, 16, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 32, 128, second_token | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 32, 64,  second_token | quantized}, {16, 16, 16, 16, 4, 2, 4, 2}},

        {{compute::gpu_arch_t::xe2, 32,      integrated | second_token | quantized}, {16, 16, 16, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 32, 384, integrated | second_token | quantized}, {64, 16, 16, 16, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 32, 96,  integrated | second_token | quantized}, {16, 16, 16, 16, 8, 1, 4, 1}},

        {{compute::gpu_arch_t::xe2, 64},                   {16, 64, 32, 16, 8, 2, 2, 8}},
        {{compute::gpu_arch_t::xe2, 64, 64},               {32, 32, 32, 16, 4, 2, 2, 4}},
        {{compute::gpu_arch_t::xe2, 64, 32},               {16, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64,     second_token}, {64, 16, 16, 16, 8, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 64, 64, second_token}, {16, 16, 16, 16, 4, 2, 4, 2}},

        {{compute::gpu_arch_t::xe2, 64,       quantized}, {16, 64, 16, 32, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 512,  quantized}, {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 64, 384,  quantized}, {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 128,  quantized}, {16, 64, 16, 32, 8, 1, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64, 32,   quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 1024, integrated | quantized}, {16, 64, 16, 32, 8, 4, 4, 8}},
        {{compute::gpu_arch_t::xe2, 64, 384,  integrated | quantized}, {16, 64, 16, 16, 16, 1, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 128,  integrated | quantized}, {16, 16, 16, 16, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe2, 64, 96,   integrated | quantized}, {16, 64, 16, 32, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe2, 64,      second_token | quantized}, {16, 16, 16, 16, 16, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 512, second_token | quantized}, {64, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 384, second_token | quantized}, {16, 16, 16, 16, 16, 2, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64, 128, second_token | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 64,  second_token | quantized}, {16, 16, 16, 16, 4, 2, 4, 2}},

        {{compute::gpu_arch_t::xe2, 64,      integrated | second_token | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 64, 384, integrated | second_token | quantized}, {64, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe2, 64, 96,  integrated | second_token | quantized}, {16, 16, 16, 16, 8, 2, 4, 2}},

        {{compute::gpu_arch_t::xe2, 72, 1024}, {16, 16, 16, 16, 5, 2, 5, 2}},
        {{compute::gpu_arch_t::xe2, 72},       {32, 32, 16, 32, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 128},               {16, 64, 32, 16, 16, 2, 4, 8}},
        {{compute::gpu_arch_t::xe2, 128, 64},           {16, 32, 32, 32, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe2, 128, 32},           {16, 16, 16, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 128, second_token}, {32, 32, 32, 16, 8, 1, 4, 2}},

        {{compute::gpu_arch_t::xe2, 128,      quantized},              {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 128, 128, quantized},              {16, 16, 16, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 128, 32,  quantized},              {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 128, 128, integrated | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 128,      second_token | quantized}, {16, 16, 16, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 128, 512, second_token | quantized}, {16, 16, 16, 16, 16, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 128, 96,  second_token | quantized}, {16, 16, 16, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 128, integrated | second_token | quantized}, {16, 16, 16, 16, 16, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 256, 1024, second_token}, {64, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 256, 512,  second_token}, {32, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 256, 128,  second_token}, {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 256,       second_token}, {32, 16, 64, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 256,     second_token | quantized}, {16, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 256, 64, second_token | quantized}, {16, 16, 64, 16, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe2, 256,      integrated}, {16, 16, 16, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 256, 128, integrated}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 256,       integrated | second_token}, {16, 16, 64, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 256, 256,  integrated | second_token}, {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 256, 1024, integrated | second_token}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 256,      quantized}, {16, 64, 16, 32, 32, 1, 16, 2}},
        {{compute::gpu_arch_t::xe2, 256, 384, quantized}, {16, 32, 32, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 128, quantized}, {16, 32, 32, 32, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 256, 64,  quantized}, {16, 32, 64, 16, 8, 2, 4, 4}},

        {{compute::gpu_arch_t::xe2, 256, 128, integrated | quantized}, {16, 32, 32, 32, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 64,  integrated | quantized}, {16, 16, 16, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe2, 256,       integrated | second_token | quantized}, {32, 16, 64, 16, 4, 1, 4, 1}},
        // TODO: restore to seq <= 1152 instead of seq < 1152?
        {{compute::gpu_arch_t::xe2, 256, 1151, integrated | second_token | quantized}, {16, 16, 64, 16, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe2, 256, 511,  integrated | second_token | quantized}, {32, 32, 32, 16, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe2, 256, 383,  integrated | second_token | quantized}, {16, 16, 16, 16, 16, 2, 16, 2}},

        {{compute::gpu_arch_t::xe2, 256},               {16, 32, 32, 32, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 256, 64},           {16, 32, 32, 32, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe2, 512},     {32, 16, 64, 16, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 512, 64}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 512, 1024, second_token}, {64, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 512, 512,  second_token}, {32, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512, 128,  second_token}, {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 512,       second_token}, {32, 16, 64, 16, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512,      quantized}, {16, 32, 64, 16, 16, 2, 8, 4}},
        {{compute::gpu_arch_t::xe2, 512, 128, quantized}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 512,     second_token | quantized}, {16, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512, 64, second_token | quantized}, {16, 16, 64, 16, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe2, 512,      integrated}, {32, 16, 64, 16, 8, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, 128, integrated}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 512,       integrated | second_token}, {16, 16, 64, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe2, 512, 256,  integrated | second_token}, {16, 16, 64, 16, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe2, 512, 1024, integrated | second_token}, {16, 16, 64, 16, 8, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 512, integrated | quantized}, {16, 32, 32, 32, 16, 1, 16, 1}},

        {{compute::gpu_arch_t::xe2, 512,       integrated | second_token | quantized}, {32, 16, 64, 16, 8, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512, 1024, integrated | second_token | quantized}, {16, 16, 64, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe2, 512, 512,  integrated | second_token | quantized}, {16, 16, 64, 16, 4, 4, 8, 4}},
        {{compute::gpu_arch_t::xe2, 512, 256,  integrated | second_token | quantized}, {16, 32, 64, 32, 16, 2, 8, 2}},
        {{compute::gpu_arch_t::xe2, 512, 128,  integrated | second_token | quantized}, {16, 16, 64, 16, 8, 1, 32, 1}},
        {{compute::gpu_arch_t::xe2, 512, 64,   integrated | second_token | quantized}, {16, 32, 64, 32, 16, 2, 8, 2}},

        {{compute::gpu_arch_t::xe2, 576}, {16, 32, 32, 32, 32, 1, 32, 1}},


        {{compute::gpu_arch_t::xe2,  32, 384, fma }, { 32, 16, 16, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2,  64, 384, fma }, { 16, 16, 16, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 128, 384, fma }, { 16, 16, 32, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, 384, fma }, { 16, 16, 32, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2, 512, 384, fma }, { 16, 16, 32, 16, 16, 2, 16, 2 }},

        {{compute::gpu_arch_t::xe2,  16, fma }, { 16, 16, 16, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2,  32, fma }, { 32, 16, 16, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2,  64, fma }, { 16, 16, 16, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 128, fma }, { 16, 16, 32, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, fma }, { 16, 16, 32, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2, 512, fma }, { 16, 16, 32, 16, 16, 2, 16, 2 }},

        {{compute::gpu_arch_t::xe2,  32, 385, fma | second_token }, { 16, 16, 16, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2,  64, 385, fma | second_token }, { 32, 16, 16, 16,  8, 4,  8, 4 }},
        {{compute::gpu_arch_t::xe2, 128, 385, fma | second_token }, { 16, 16, 32, 16, 16, 1, 16, 1 }},
        {{compute::gpu_arch_t::xe2, 256, 385, fma | second_token }, { 32, 16, 32, 16, 16, 2, 16, 2 }},

        {{compute::gpu_arch_t::xe2,  32, fma | second_token }, { 16, 16, 16, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2,  64, fma | second_token }, { 32, 32, 16, 32,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 128, fma | second_token }, { 16, 16, 32, 16, 16, 1, 16, 1 }},
        {{compute::gpu_arch_t::xe2, 128, f32 | fma | second_token }, { 16, 16, 32, 16, 8, 2, 8, 2 }},
        {{compute::gpu_arch_t::xe2, 256, fma | second_token }, { 16, 16, 16, 16, 16, 2, 16, 2 }},
        {{compute::gpu_arch_t::xe2, 512, fma | second_token }, { 16, 16, 32, 16, 16, 2, 16, 2 }},


        {{compute::gpu_arch_t::xe2,  32, 384, fma | quantized }, { 16, 16, 32, 16, 4, 2, 4, 2 }},
        {{compute::gpu_arch_t::xe2,  64, 384, fma | quantized }, { 32, 16, 32, 16, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 128, 384, fma | quantized }, { 16, 16, 32, 16, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, 384, fma | quantized }, { 32, 16, 16, 16,32, 1,32, 1 }},

        {{compute::gpu_arch_t::xe2,  32, fma | quantized }, { 16, 16, 32, 16, 4, 2, 4, 2 }},
        {{compute::gpu_arch_t::xe2,  64, fma | quantized }, { 32, 16, 32, 16, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 128, fma | quantized }, { 16, 16, 32, 16, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, fma | quantized }, { 16, 16, 32, 16, 8, 4, 8, 4 }},

        {{compute::gpu_arch_t::xe2,  32, fma | second_token | quantized }, { 32, 32, 16, 16,  8, 2,  4, 4}},
        {{compute::gpu_arch_t::xe2,  64, fma | second_token | quantized }, { 16, 32, 16, 16, 16, 2,  8, 4}},
        {{compute::gpu_arch_t::xe2, 128, fma | second_token | quantized }, { 16, 16, 32, 16,  8, 2,  8, 2}},
        {{compute::gpu_arch_t::xe2, 256, fma | second_token | quantized }, { 16, 16, 16, 16, 16, 2, 16, 2}},


        {{compute::gpu_arch_t::xe2,  32, 384, fma | integrated }, { 32, 32, 16, 32,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2,  64, 384, fma | integrated }, { 16, 16, 16, 16,  8, 4,  8, 4 }},
        {{compute::gpu_arch_t::xe2, 128, 384, fma | integrated }, { 16, 16, 32, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, 384, fma | integrated }, { 16, 16, 32, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2, 512, 384, fma | integrated }, { 16, 16, 32, 16, 16, 2, 16, 2 }},

        {{compute::gpu_arch_t::xe2,  16, fma | integrated }, { 16, 16, 16, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2,  32, fma | integrated }, { 32, 16, 16, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2,  64, fma | integrated }, { 16, 16, 32, 16,  2, 4,  2, 4 }},
        {{compute::gpu_arch_t::xe2, 128, fma | integrated }, { 16, 16, 32, 16,  4, 4,  4, 4 }},
        {{compute::gpu_arch_t::xe2, 256, fma | integrated }, { 16, 16, 32, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2, 512, fma | integrated }, { 16, 16, 32, 16, 16, 2, 16, 2 }},

        {{compute::gpu_arch_t::xe2, 128, 385, fma | second_token | integrated }, { 16, 16, 32, 16, 16, 1, 16, 1 }},

        {{compute::gpu_arch_t::xe2,  32, fma | second_token | integrated }, { 16, 16, 16, 16,  8, 2,  8, 2 }},
        {{compute::gpu_arch_t::xe2,  64, fma | second_token | integrated }, { 16, 16, 16, 16,  4, 2,  4, 2 }},
        {{compute::gpu_arch_t::xe2, 128, fma | second_token | integrated }, { 16, 16, 32, 16, 16, 1, 16, 1 }},
        {{compute::gpu_arch_t::xe2, 128, f32 | fma | second_token | integrated }, { 16, 16, 32, 16, 4, 2, 4, 2 }},
        {{compute::gpu_arch_t::xe2, 256, fma | second_token | integrated }, { 16, 16, 32, 16, 8, 2, 8, 2 }},

        {{compute::gpu_arch_t::xe3, 32},     {16, 64, 32, 16, 4, 2, 1, 8}},
        {{compute::gpu_arch_t::xe3, 32, 32}, {16, 16, 16, 16, 2, 4, 2, 4}},
        {{compute::gpu_arch_t::xe3, 64},     {16, 64, 32, 16, 8, 2, 2, 8}},
        {{compute::gpu_arch_t::xe3, 64, 64}, {32, 32, 32, 16, 4, 2, 2, 4}},
        {{compute::gpu_arch_t::xe3, 64, 32}, {16, 16, 16, 16, 4, 2, 4, 2}},
        {{compute::gpu_arch_t::xe3, 72, 64}, {16, 16, 16, 16, 5, 2, 5, 2}},
        {{compute::gpu_arch_t::xe3, 72},     {16, 32, 32, 32, 4, 2, 4, 2}},

        {{compute::gpu_arch_t::xe3p,  32, fma },                            { 16, 16, 16, 16,  2, 2,  2, 2}},
        {{compute::gpu_arch_t::xe3p,  64 },                                 { 32, 32, 32, 32, 2, 4, 2, 4}},
        {{compute::gpu_arch_t::xe3p,  64, quantized },                      { 32, 32, 32, 16, 4, 2, 2, 4}},
        {{compute::gpu_arch_t::xe3p,  64, fma | second_token },             { 16, 16, 16, 16,  8, 4,  8, 4}},
        {{compute::gpu_arch_t::xe3p,  32, fma | second_token | quantized }, { 16, 16, 16, 16,  2, 2,  2, 2}},
        {{compute::gpu_arch_t::xe3p,  64, fma | second_token | quantized }, { 16, 16, 16, 16,  8, 4,  8, 4}},
        {{compute::gpu_arch_t::xe3p, 128 },                         {32, 32, 32, 32, 4, 4, 4, 4}},
        {{compute::gpu_arch_t::xe3p, 128, second_token },           {32, 32, 32, 32, 4, 1, 4, 1}},
        {{compute::gpu_arch_t::xe3p, 128, quantized },              {32, 32, 32, 16, 8, 2, 4, 4}},
        {{compute::gpu_arch_t::xe3p, 128, second_token | quantized },           {32, 32, 32, 32, 4, 1, 4, 1}},

        {{compute::gpu_arch_t::xe3p, 256 },                         {32, 32, 32, 32, 8, 4, 8, 4}},
        {{compute::gpu_arch_t::xe3p, 256, second_token },           {32, 32, 32, 32, 8, 1, 8, 1}},
        {{compute::gpu_arch_t::xe3p, 256, quantized},               {16, 32, 32, 16, 16, 1, 8, 2}},
        {{compute::gpu_arch_t::xe3p, 256, second_token | quantized },           {32, 32, 32, 32, 8, 1, 8, 1}},

        {{compute::gpu_arch_t::xe3p, 512},                                        {32, 16, 32, 16, 16, 2, 16, 2}},
        {{compute::gpu_arch_t::xe3p, 512, second_token },                         {32, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe3p, 512, second_token | quantized },             {32, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe3p, 512, fma | quantized },                      {16, 16, 32, 16, 16, 1, 16, 1}},
        {{compute::gpu_arch_t::xe3p, 512, fma | f32 | second_token | quantized }, {16, 16, 32, 16, 16, 1, 16, 1}},
    };
    // clang-format on

    // ensures configs appear in order of most to least defined/desirable
    std::sort(std::begin(configs), std::end(configs));
    return configs;
}();

property set_properties(bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32, bool is_f16_accumulate) {
    property properties = property::none;
    if (is_thin_q) { properties |= property::second_token; }
    if (is_quantized) { properties |= property::quantized; }
    if (is_integrated) { properties |= property::integrated; }
    if (is_fma) { properties |= property::fma; }
    if (is_f32) { properties |= property::f32; }
    if (is_f16_accumulate) { properties |= property::f16_accumulate; }
    return properties;
}

fwd_config_t *choose_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32, bool is_f16_accumulate) {
    // quantized FMA for f16 on MTL not implemented in gemmstone
    if (arch == compute::gpu_arch_t::xe_hpg && is_fma && !is_f32
            && is_quantized)
        return nullptr;
    // f32 and fma on MTL requires too many registers for head sizes >= 256
    if (arch == compute::gpu_arch_t::xe_hpg && (is_fma || is_f32)
            && head_size > 256)
        return nullptr;
    // no valid quantized configs w/head size = 512 on xe2
    if (arch == compute::gpu_arch_t::xe2 && is_fma && is_quantized
            && head_size > 256)
        return nullptr;

    compute::gpu_arch_t arch_query = (arch == compute::gpu_arch_t::xe3)
            ? compute::gpu_arch_t::xe2
            : arch;

    // TODO: remove this when Xe3 configs are added. Currently the Xe2
    // non-integrated configs perform better than integrated Xe2 configs.
    if (arch == compute::gpu_arch_t::xe3) { is_integrated = false; }
    auto arch_str = std::string(to_string(arch));
    std::string s = gpu_utils::dev_getenv("SDPA_SELECT_CONFIG_HW", arch_str);
    if (!s.empty()) {
        compute::gpu_arch_t forced_arch = compute::str2gpu_arch(s.c_str());
        if (forced_arch == compute::gpu_arch_t::unknown) {
            VDEBUGINFO(0, primitive, sdpa,
                    "Invalid SDPA_SELECT_CONFIG_HW value '%s', expected one of "
                    "'xe2', 'xe3', 'xe3p', 'xe_hpc' or 'xe_hpg'.",
                    s.c_str());
        }
        if (forced_arch != arch_query) {
            VDEBUGINFO(4, primitive, sdpa,
                    "Overriding architecture for config selection: %s -> %s.",
                    to_string(arch_query), to_string(forced_arch));
            arch_query = forced_arch;
        }
    }

    bool fallback = true;
    while (fallback) {
        property query_properties = set_properties(is_thin_q, is_quantized,
                is_integrated, is_fma, is_f32, is_f16_accumulate);
        config_query_t query(arch_query, static_cast<int>(head_size),
                static_cast<int>(seq), query_properties);
        auto it = find(begin(sorted_configs), end(sorted_configs), query);
        if (it != end(sorted_configs)) {
            VDEBUGINFO(4, primitive, sdpa,
                    "config search: {query %s} -> {%s config:%s},",
                    to_string(query).c_str(), to_string(it->criteria).c_str(),
                    to_string(it->config).c_str());
            return &it->config;
        } else {
            VDEBUGINFO(4, primitive, sdpa, "config search failed: {query %s},",
                    to_string(query).c_str());
        }
        switch (arch_query) {
            case compute::gpu_arch_t::xe3p:
                arch_query = compute::gpu_arch_t::xe3;
                break;
            case compute::gpu_arch_t::xe3:
                arch_query = compute::gpu_arch_t::xe2;
                break;
            case compute::gpu_arch_t::xe2: fallback = false; break;
            case compute::gpu_arch_t::xe_hpc: fallback = false; break;
            default: fallback = false; break;
        }
    }
    return nullptr;
}

// adjust heuristic intervals to match the tuned intervals according
// to the sequence length and gpu architecture
// this way recompilation both matches the tuned intervals and avoids
// excessive recompilation with smaller power of 2 sizes
dim_t nearest_conf_seq_interval(compute::gpu_arch_t arch, dim_t head_size,
        dim_t seq, bool is_thin_q, bool is_quantized, bool is_integrated,
        bool is_fma, bool is_f32, bool is_f16_accumulate) {
    property query_properties = set_properties(is_thin_q, is_quantized,
            is_integrated, is_fma, is_f32, is_f16_accumulate);

    compute::gpu_arch_t arch_query = (arch >= compute::gpu_arch_t::xe3)
            ? compute::gpu_arch_t::xe2
            : arch;

    config_query_t query(arch_query, static_cast<int>(head_size),
            static_cast<int>(seq), query_properties);
    auto it = find(begin(sorted_configs), end(sorted_configs), query);
    if (it != end(sorted_configs)) { return it->criteria.seq_len; }
    return utils::rnd_up_pow2(seq);
}

// Backward pass kernel configurations:
// [ arch, head_size, {sequence length}, {properties} ] -> bwd_config
// bwd_config_t fields: {unroll_m_BcBr, unroll_n_BcBr,
//                        unroll_m_DBc,  unroll_n_DBc,
//                        unroll_m_DBr,  unroll_n_DBr,
//                        wg_m_BcBr, wg_n_BcBr,
//                        wg_m_DBc,  wg_n_DBc,
//                        wg_m_DBr,  wg_n_DBr}
static std::vector<bwd_config_record_t> sorted_bwd_configs = []() {
    // clang-format off
    std::vector<bwd_config_record_t> configs = {
        // xe_hpc
        {{compute::gpu_arch_t::xe_hpc, 32},      { 16, 16, 16, 16, 16, 16, 2, 2, 2, 2, 2, 2 }},
        {{compute::gpu_arch_t::xe_hpc, 32, 128}, { 16, 16, 16, 16, 16, 16, 2, 8, 2, 2, 2, 8 }},
        {{compute::gpu_arch_t::xe_hpc, 32, 256}, { 16, 16, 16, 16, 16, 16, 2, 8, 2, 2, 2, 8 }},

        {{compute::gpu_arch_t::xe_hpc, 64},      { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe_hpc, 64, 49},  { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe_hpc, 64, 256}, { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},

        {{compute::gpu_arch_t::xe_hpc, 88},      { 16, 16, 16, 16, 16, 32, 4, 8, 8, 4, 8, 4 }},

        {{compute::gpu_arch_t::xe_hpc, 128},      { 16, 16, 16, 16, 16, 32, 4, 8, 8, 4, 8, 4 }},
        {{compute::gpu_arch_t::xe_hpc, 128, 256}, { 16, 16, 16, 32, 32, 32, 2, 4, 8, 1, 4, 2 }},
        {{compute::gpu_arch_t::xe_hpc, 128, 320}, { 16, 16, 32, 16, 32, 16, 4, 4, 4, 4, 4, 4 }},

        {{compute::gpu_arch_t::xe_hpc, 32, second_token},  { 16, 16, 16, 16, 32, 16, 1, 2, 2, 1, 1, 2 }},
        {{compute::gpu_arch_t::xe_hpc, 64, second_token},  { 32, 16, 16, 32, 32, 32, 1, 4, 4, 1, 2, 2 }},
        {{compute::gpu_arch_t::xe_hpc, 128, second_token}, { 16, 16, 16, 16, 32, 32, 2, 8, 8, 2, 4, 4 }},

        {{compute::gpu_arch_t::xe_hpc,  32, f32 | fma},  { 32, 32, 16, 16, 32, 32, 1, 4, 2, 2, 1, 4 }},
        {{compute::gpu_arch_t::xe_hpc,  64, f32 | fma},  { 16, 32, 16, 16, 16, 32, 4, 4, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe_hpc, 128, f32 | fma},  { 16, 16, 16, 32, 32, 32, 2, 4, 8, 1, 4, 2 }},

        // xe2
        {{compute::gpu_arch_t::xe2, 32},       { 32, 32, 16, 32, 16, 32, 2, 2, 2, 2, 2, 2 }},
        {{compute::gpu_arch_t::xe2, 32, 128},  { 16, 32, 16, 16, 16, 16, 4, 2, 2, 4, 2, 4 }},

        {{compute::gpu_arch_t::xe2, 64},       { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 64, 64},   { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},
        {{compute::gpu_arch_t::xe2, 64, 128},  { 16, 16, 16, 16, 16, 32, 2, 4, 4, 2, 4, 2 }},
        {{compute::gpu_arch_t::xe2, 64, 320},  { 16, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 4 }},

        {{compute::gpu_arch_t::xe2, 88},       { 16, 16, 16, 32, 32, 32, 2, 4, 8, 1, 4, 2 }},

        {{compute::gpu_arch_t::xe2, 128},      { 16, 16, 32, 16, 32, 32, 2, 4, 4, 2, 4, 2 }},
        {{compute::gpu_arch_t::xe2, 128, 256}, { 16, 16, 16, 16, 16, 64, 2, 8, 8, 2, 8, 2 }},

        {{compute::gpu_arch_t::xe2, 32, integrated},  { 16, 64, 16, 16, 32, 32, 2, 2, 2, 2, 1, 4 }},
        {{compute::gpu_arch_t::xe2, 64, integrated},  { 16, 32, 16, 16, 32, 32, 2, 4, 4, 2, 2, 4 }},
        {{compute::gpu_arch_t::xe2, 128, integrated}, { 16, 32, 16, 16, 32, 32, 4, 8, 8, 4, 4, 8 }},

    };
    // clang-format on

    // ensures configs appear in order of most to least defined/desirable
    std::sort(std::begin(configs), std::end(configs));
    return configs;
}();

bwd_config_t *choose_bwd_config(compute::gpu_arch_t arch, dim_t head_size,
        dim_t qry, dim_t seq, dim_t batch_heads, bool is_thin_q,
        bool is_quantized, bool is_integrated, bool is_fma, bool is_f32,
        bool is_causal) {
    // limit configs to tuned architectures and problem sizes
    if (arch >= compute::gpu_arch_t::xe3) return nullptr;
    if (compute::gpu_arch_t::xe2 == arch && is_integrated) return nullptr;

    if (compute::gpu_arch_t::xe2 == arch && !is_causal
            && (head_size >= 64 && std::min(qry, seq) <= head_size)) {
        return nullptr;
    }

    const bool is_f16_accumulate = false;
    property query_properties = set_properties(is_thin_q, is_quantized,
            is_integrated, is_fma, is_f32, is_f16_accumulate);

    config_query_t query(arch, static_cast<int>(head_size),
            static_cast<int>(seq), query_properties);
    auto it = find(begin(sorted_bwd_configs), end(sorted_bwd_configs), query);
    if (it != end(sorted_bwd_configs)) {
        // Primitive-based backward is generally faster for small problems.
        // Direct dQ and fully occupied tiles lower that crossover for
        // sufficiently parallel Xe HPC configurations.
        static constexpr dim_t xe_hpc_bwd_min_fused_area = 128 * 192;
        static constexpr dim_t xe_hpc_bwd_min_batch_heads = 256;
        static constexpr dim_t xe_hpc_bwd_min_main_wgs = 2048;
        if (arch == compute::gpu_arch_t::xe_hpc && head_size > 32) {
            const auto &config = it->config;
            const dim_t tile_k = (dim_t)config.unroll_m_BcBr * config.wg_m_BcBr;
            const dim_t tile_q = (dim_t)config.unroll_n_BcBr * config.wg_n_BcBr;
            const dim_t tile_d = (dim_t)config.unroll_m_DBc * config.wg_m_DBc;
            const bool large_problem = qry * seq >= xe_hpc_bwd_min_fused_area;
            const bool full_d64 = head_size <= 64 && head_size == tile_d;
            const bool efficient_direct_dq = full_d64 && !is_thin_q
                    && seq <= tile_k && tile_q <= 64
                    && batch_heads >= xe_hpc_bwd_min_batch_heads;
            const dim_t key_blocks = utils::div_up(seq, tile_k);
            const bool efficient_full_tiles = full_d64 && !is_thin_q
                    && seq % tile_k == 0 && qry % tile_q == 0
                    && batch_heads * key_blocks >= xe_hpc_bwd_min_main_wgs;
            const bool efficient_small_problem = is_causal
                    ? !is_thin_q
                    : efficient_direct_dq || efficient_full_tiles;
            if (!large_problem && !efficient_small_problem) return nullptr;
        }

        VDEBUGINFO(4, primitive, sdpa,
                "bwd config search: {query %s} -> {%s config:%s},",
                to_string(query).c_str(), to_string(it->criteria).c_str(),
                to_string(it->config).c_str());
        return &it->config;
    }
    return nullptr;
}

void deserialize_config_to_gemmstone(micro::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        micro::GEMMOptions &opts_kq, micro::GEMMOptions &opts_vs,
        gemmstone::SizeParams &sizes_kq, gemmstone::SizeParams &sizes_vs,
        const micro_fwd_ukernel_params_t &ukernel_config) {

    // hardware info
    hwInfo.gmdid = ukernel_config.hwinfo.gmdid;
    hwInfo.euCount = ukernel_config.hwinfo.euCount;
    hwInfo.systolicAvailable = ukernel_config.hwinfo.systolicAvailable;
    hwInfo.isEfficient64Bit = ukernel_config.hwinfo.isEfficient64Bit;

    // options kq, vs
    auto deserialize_options
            = [](micro::GEMMOptions &gemmstone_opts,
                      const ukernel_serialized_opts_t &serialized_opts) {
        gemmstone_opts.localA = serialized_opts.localA;
        gemmstone_opts.localB = serialized_opts.localB;
        gemmstone_opts.slmPtr = serialized_opts.slmPtr;
        gemmstone_opts.scaleA = serialized_opts.scaleA;
        gemmstone_opts.offsetA = serialized_opts.offsetA;
    };

    deserialize_options(opts_kq, ukernel_config.opts_kq);
    deserialize_options(opts_vs, ukernel_config.opts_vs);

    // problems kq, vs
    auto deserialize_problem
            = [](gemmstone::GEMMProblem &problem,
                      const ukernel_serialized_problem_t &serialized_problem) {
        problem.Ta_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Ta_ext)};
        problem.Tb_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tb_ext)};
        problem.Ta
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ta)};
        problem.Tb
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tb)};
        problem.Tc_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tc_ext)};
        problem.Tc
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tc)};
        problem.Ts
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ts)};
        problem.A.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_layout);

        problem.Ta_scale = {static_cast<gemmstone::Type::_Type>(
                serialized_problem.Ta_scale)};
        problem.A_scale.setAlignment(serialized_problem.A_scale_alignment);
        problem.A_scale.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_scale_layout);
        problem.asPtrDims = serialized_problem.asPtrDims;
        problem.Tao
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tao)};
        problem.AO.setAlignment(serialized_problem.AO_alignment);
        problem.AO.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.AO_layout);
        problem.aoPtrDims = serialized_problem.aoPtrDims;
        problem.aOffset
                = static_cast<gemmstone::ABOffset>(serialized_problem.aOffset);
        problem.aqGroupM = serialized_problem.aqGroupM;
        problem.aqGroupK = serialized_problem.aqGroupK;

        problem.B.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.B_layout);
        problem.C.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.C_layout);
        problem.A.setAlignment(serialized_problem.A_alignment);
        problem.A.crosspack = serialized_problem.A_crosspack;
        problem.A.tileR = serialized_problem.A_tileR;
        problem.A.tileC = serialized_problem.A_tileC;

        problem.B.setAlignment(serialized_problem.B_alignment);
        problem.B.crosspack = serialized_problem.B_crosspack;
        problem.B.tileR = serialized_problem.B_tileR;
        problem.B.tileC = serialized_problem.B_tileC;
    };
    deserialize_problem(problem_kq, ukernel_config.problem_kq);
    deserialize_problem(problem_vs, ukernel_config.problem_vs);

    // sizes kq, vs
    auto deserialize_sizes
            = [](gemmstone::SizeParams &sizes,
                      const ukernel_serialized_sizes_t &serialized_sizes) {
        sizes.m = serialized_sizes.m;
        sizes.n = serialized_sizes.n;
        sizes.k = serialized_sizes.k;
        sizes.batch = serialized_sizes.batch;
    };
    deserialize_sizes(sizes_kq, ukernel_config.sizes_kq);
    deserialize_sizes(sizes_vs, ukernel_config.sizes_vs);
}

void deserialize_config_to_gemmstone(micro::HWInformation &hwInfo,
        gemmstone::GEMMProblem &problem_kq, gemmstone::GEMMProblem &problem_vs,
        gemmstone::GEMMProblem &problem_vtdA,
        gemmstone::GEMMProblem &problem_ktq,
        gemmstone::GEMMProblem &problem_qdSt, micro::GEMMOptions &opts_kq,
        micro::GEMMOptions &opts_vs, micro::GEMMOptions &opts_vtdA,
        micro::GEMMOptions &opts_ktq, micro::GEMMOptions &opts_qdSt,
        gemmstone::SizeParams &sizes_kq, gemmstone::SizeParams &sizes_vs,
        gemmstone::SizeParams &sizes_vtdA, gemmstone::SizeParams &sizes_ktq,
        gemmstone::SizeParams &sizes_qdSt,
        const micro_bwd_ukernel_params_t &ukernel_config) {

    // hardware info
    hwInfo.gmdid = ukernel_config.hwinfo.gmdid;
    hwInfo.euCount = ukernel_config.hwinfo.euCount;
    hwInfo.systolicAvailable = ukernel_config.hwinfo.systolicAvailable;
    hwInfo.isEfficient64Bit = ukernel_config.hwinfo.isEfficient64Bit;

    // options kq, vs
    auto deserialize_options
            = [](micro::GEMMOptions &gemmstone_opts,
                      const ukernel_serialized_opts_t &serialized_opts) {
        gemmstone_opts.localA = serialized_opts.localA;
        gemmstone_opts.localB = serialized_opts.localB;
        gemmstone_opts.slmPtr = serialized_opts.slmPtr;
        gemmstone_opts.scaleA = serialized_opts.scaleA;
        gemmstone_opts.offsetA = serialized_opts.offsetA;
    };
    deserialize_options(opts_kq, ukernel_config.opts_kq);
    deserialize_options(opts_vs, ukernel_config.opts_vs);
    deserialize_options(opts_vtdA, ukernel_config.opts_vtdA);
    deserialize_options(opts_ktq, ukernel_config.opts_ktq);
    deserialize_options(opts_qdSt, ukernel_config.opts_qdSt);

    // problems kq, vs
    auto deserialize_problem
            = [](gemmstone::GEMMProblem &problem,
                      const ukernel_serialized_problem_t &serialized_problem) {
        problem.Ta_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Ta_ext)};
        problem.Tb_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tb_ext)};
        problem.Ta
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ta)};
        problem.Tb
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tb)};
        problem.Tc_ext = {
                static_cast<gemmstone::Type::_Type>(serialized_problem.Tc_ext)};
        problem.Tc
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tc)};
        problem.Ts
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Ts)};
        problem.A.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_layout);

        problem.Ta_scale = {static_cast<gemmstone::Type::_Type>(
                serialized_problem.Ta_scale)};
        problem.A_scale.setAlignment(serialized_problem.A_scale_alignment);
        problem.A_scale.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.A_scale_layout);
        problem.asPtrDims = serialized_problem.asPtrDims;
        problem.Tao
                = {static_cast<gemmstone::Type::_Type>(serialized_problem.Tao)};
        problem.AO.setAlignment(serialized_problem.AO_alignment);
        problem.AO.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.AO_layout);
        problem.aoPtrDims = serialized_problem.aoPtrDims;
        problem.aOffset
                = static_cast<gemmstone::ABOffset>(serialized_problem.aOffset);
        problem.aqGroupM = serialized_problem.aqGroupM;
        problem.aqGroupK = serialized_problem.aqGroupK;

        problem.B.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.B_layout);
        problem.C.layout = static_cast<gemmstone::MatrixLayout>(
                serialized_problem.C_layout);
        problem.A.setAlignment(serialized_problem.A_alignment);
        problem.A.crosspack = serialized_problem.A_crosspack;
        problem.A.tileR = serialized_problem.A_tileR;
        problem.A.tileC = serialized_problem.A_tileC;

        problem.B.setAlignment(serialized_problem.B_alignment);
        problem.B.crosspack = serialized_problem.B_crosspack;
        problem.B.tileR = serialized_problem.B_tileR;
        problem.B.tileC = serialized_problem.B_tileC;
    };
    deserialize_problem(problem_kq, ukernel_config.problem_kq);
    deserialize_problem(problem_vs, ukernel_config.problem_vs);
    deserialize_problem(problem_vtdA, ukernel_config.problem_vtdA);
    deserialize_problem(problem_ktq, ukernel_config.problem_ktq);
    deserialize_problem(problem_qdSt, ukernel_config.problem_qdSt);

    // sizes kq, vs
    auto deserialize_sizes
            = [](gemmstone::SizeParams &sizes,
                      const ukernel_serialized_sizes_t &serialized_sizes) {
        sizes.m = serialized_sizes.m;
        sizes.n = serialized_sizes.n;
        sizes.k = serialized_sizes.k;
        sizes.batch = serialized_sizes.batch;
    };
    deserialize_sizes(sizes_kq, ukernel_config.sizes_kq);
    deserialize_sizes(sizes_vs, ukernel_config.sizes_vs);
    deserialize_sizes(sizes_vtdA, ukernel_config.sizes_vtdA);
    deserialize_sizes(sizes_ktq, ukernel_config.sizes_ktq);
    deserialize_sizes(sizes_qdSt, ukernel_config.sizes_qdSt);
}

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
