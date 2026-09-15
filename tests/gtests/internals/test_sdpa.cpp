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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include "sdpa_internal.hpp"
#include "test_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#include <algorithm>
#include <memory>
#include <random>

#define DEBUG_PRINT_MEM 0

using mdt = memory::data_type;
using dnnl::accumulation_mode;

enum class mask_type { no_mask, oneD, twoD, causal_br, causal_tl };
enum class scale_type { host_side, device_side };
constexpr scale_type default_scale_type = scale_type::device_side;

std::ostream &operator<<(std::ostream &ss, const mask_type &p) {
    ss << "mask:";
    switch (p) {
        case mask_type::no_mask: ss << "no mask"; break;
        case mask_type::oneD: ss << "1D"; break;
        case mask_type::twoD: ss << "2D"; break;
        case mask_type::causal_br: ss << "causal bottom right"; break;
        case mask_type::causal_tl: ss << "causal top left"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const scale_type &p) {
    ss << "scale:";
    switch (p) {
        case scale_type::device_side: ss << "device_side"; break;
        case scale_type::host_side: ss << "host_side"; break;
    }
    return ss;
}

struct tag_t {
    dnnl::memory::format_tag t;
    tag_t() : t(dnnl::memory::format_tag::undef) {}
    tag_t(dnnl::memory::format_tag t_) : t(t_) {}
    operator dnnl::memory::format_tag() const { return t; }
    bool operator==(const memory::format_tag &other) const {
        return t == other;
    }
};

std::ostream &operator<<(std::ostream &ss, const tag_t &p) {
    ss << "key tag:";
    switch (p.t) {
        case dnnl::memory::format_tag::abcd: ss << "abcd"; break;
        case dnnl::memory::format_tag::abdc: ss << "abdc"; break;
        default: ss << "undef"; break;
    }
    return ss;
}

bool is_quantized(mdt dt, mdt sdt, mdt zpdt, quantize_type qtype) {
    if (qtype == quantize_type::no_quantization) return false;

    if (dt == mdt::f16 || dt == mdt::bf16 || dt == mdt::f32
            || dt == mdt::f8_e4m3 || dt == mdt::f8_e5m2) {
        return false;
    }
    if (sdt == mdt::undef && zpdt == mdt::undef) return false;
    return true;
}

struct tensor_type_t {
    std::string name;
    mdt dt;
    mdt sdt; // scaled data type
    mdt zpdt; // zero point data type
    tensor_type_t() = default;
    tensor_type_t(const tensor_type_t &other) = default;
    tensor_type_t(tensor_type_t &&other) = default;
    tensor_type_t &operator=(const tensor_type_t &other) = default;
    tensor_type_t &operator=(tensor_type_t &&other) = default;
    tensor_type_t(std::string name_, memory::data_type t,
            memory::data_type st = mdt::undef,
            memory::data_type zpt = mdt::undef)
        : name(std::move(name_)), dt(t), sdt(st), zpdt(zpt) {}
    //operator memory::data_type() const { return dt; }
};

std::ostream &operator<<(std::ostream &ss, const tensor_type_t &p) {
    ss << p.name << ":" << p.dt;
    if (is_quantized(p.dt, p.sdt, p.zpdt, quantize_type::per_token)) {
        ss << "x" << p.sdt;
        ss << "x" << p.zpdt;
    }
    return ss;
}

struct head_group_size_t {
    memory::dim head_size;
    int kgroup_size;
    int vgroup_size;
};

std::ostream &operator<<(
        std::ostream &ss, const head_group_size_t &head_group) {
    ss << "Head Size(D)=" << head_group.head_size;
    if (head_group.head_size != head_group.kgroup_size
            || head_group.head_size != head_group.vgroup_size) {
        ss << " Group Size=";
        if (head_group.kgroup_size == head_group.vgroup_size) {
            ss << head_group.kgroup_size;
        } else {
            ss << "(" << head_group.kgroup_size << "x" << head_group.vgroup_size
               << ")";
        }
    }
    return ss;
}

struct seq_len_size_t {
    memory::dim q;
    memory::dim kv;
};

std::ostream &operator<<(std::ostream &ss, const seq_len_size_t &seq_len) {
    ss << "Sequence Length";
    if (seq_len.q == seq_len.kv) {
        ss << "(K/Q)=" << seq_len.q;
    } else {
        ss << "Q:" << seq_len.q << " K/V:" << seq_len.kv;
    }
    return ss;
}

struct num_heads_t {
    memory::dim q;
    memory::dim kv;
};

std::ostream &operator<<(std::ostream &ss, const num_heads_t &heads) {
    ss << "Number of Heads(N)=";
    if (heads.q == heads.kv) {
        ss << heads.q;
    } else {
        ss << "Q:" << heads.q << " K/V:" << heads.kv;
    }
    return ss;
}

struct mask_config_t {
    mask_type type;
    memory::data_type dt;
};

std::ostream &operator<<(std::ostream &ss, const mask_config_t &m) {
    ss << "Mask =";
    switch (m.type) {
        case mask_type::no_mask: ss << "no mask"; break;
        case mask_type::oneD: ss << "1D:" << m.dt; break;
        case mask_type::twoD: ss << "2D:" << m.dt; break;
        case mask_type::causal_br: ss << "causalbr"; break;
        case mask_type::causal_tl: ss << "causaltl"; break;
    }
    return ss;
}

struct accumulation_t {
    dnnl::accumulation_mode kq_acc;
    dnnl::accumulation_mode vs_acc;
};

std::ostream &operator<<(std::ostream &ss, const accumulation_t &accs) {
    ss << "Acc(KQ/VS) =";
    std::string kq_str
            = (accs.kq_acc == accumulation_mode::f16) ? "(f16," : "(f32,";
    std::string vs_str
            = (accs.vs_acc == accumulation_mode::f16) ? "f16)" : "f32)";
    ss << kq_str << vs_str;
    return ss;
}

// Mirrors benchdnn attr_t::dropout_t (--attr-dropout=PROB[:SEED[:TAG[:OFFSET[:HOST_SCALARS]]]]).
struct dropout_config_t {
    float probability = 0.0f;
    int64_t seed = 0; // 64-bit Philox seed
    memory::data_type seed_dt = mdt::s64; // s32 or s64
    int64_t offset = 0; // 64-bit RNG offset; 0 = unset
    bool use_host_scalars = false;
    memory::format_tag mask_tag
            = memory::format_tag::any; // any = output mask; undef = no mask

    dropout_config_t() = default;
    dropout_config_t(float probability_, int64_t seed_,
            memory::data_type seed_dt_ = mdt::s64, int64_t offset_ = 0,
            bool use_host_scalars_ = false,
            memory::format_tag mask_tag_ = memory::format_tag::any)
        : probability(probability_)
        , seed(seed_)
        , seed_dt(seed_dt_)
        , offset(offset_)
        , use_host_scalars(use_host_scalars_)
        , mask_tag(mask_tag_) {}

    bool enabled() const { return probability > 0.0f; }
    bool has_output_mask() const {
        return mask_tag != memory::format_tag::undef;
    }
};

std::ostream &operator<<(std::ostream &ss, const dropout_config_t &d) {
    if (!d.enabled()) {
        ss << "dropout_none";
    } else {
        // Replace '.' in probability float with 'p' so the name stays
        // alphanumeric-plus-underscore (GTest requirement).
        std::string prob_str = std::to_string(d.probability);
        for (char &c : prob_str)
            if (c == '.') c = 'p';
        ss << "dropout_" << prob_str << "_s" << d.seed;
        if (d.offset != 0) ss << "_o" << d.offset;
        if (!d.has_output_mask()) ss << "_nomask";
    }
    return ss;
}

static const dropout_config_t no_dropout {};

using sdpa_dims_t_tuple = std::tuple<int, num_heads_t, seq_len_size_t,
        head_group_size_t, tensor_type_t, tensor_type_t, tensor_type_t,
        quantize_type, tag_t, mask_config_t, scale_type, accumulation_t,
        dropout_config_t>;

struct sdpa_dims_t {
    memory::dim mb;
    num_heads_t heads;
    seq_len_size_t seq_len;

    head_group_size_t head_group;

    tensor_type_t dt;
    tensor_type_t key;
    tensor_type_t value;

    quantize_type qtype;
    memory::format_tag key_format_tag;
    mask_config_t mask;
    scale_type stype;
    accumulation_t acc_modes;
    dropout_config_t dropout;

    sdpa_dims_t() = default;
    sdpa_dims_t(memory::dim mb_, memory::dim head_num_,
            memory::dim kv_head_num_, memory::dim seq_len_,
            memory::dim query_num_, memory::dim head_size_, int kgroup_size_,
            int vgroup_size_, memory::data_type dt_, memory::data_type kdt_,
            memory::data_type ksdt_, memory::data_type kzpdt_,
            memory::data_type vdt_, memory::data_type vsdt_,
            memory::data_type vzpdt_, memory::data_type mskdt_,
            quantize_type qtype_ = quantize_type::no_quantization,
            dnnl::memory::format_tag key_format_tag_
            = dnnl::memory::format_tag::abcd,
            mask_type mask_ = mask_type::no_mask,
            scale_type stype_ = default_scale_type,
            accumulation_mode kq_acc_ = accumulation_mode::f32,
            accumulation_mode vs_acc_ = accumulation_mode::f32,
            dropout_config_t dropout_ = no_dropout)
        : mb(mb_)
        , heads {head_num_, kv_head_num_}
        , seq_len {query_num_, seq_len_}
        , head_group {head_size_, kgroup_size_, vgroup_size_}
        , dt("Q", dt_)
        , key("K", kdt_, ksdt_, kzpdt_)
        , value("V", vdt_, vsdt_, vzpdt_)
        , qtype(qtype_)
        , key_format_tag(key_format_tag_)
        , mask {mask_, mskdt_}
        , stype(stype_)
        , acc_modes {kq_acc_, vs_acc_}
        , dropout(dropout_) {}

    sdpa_dims_t(const sdpa_dims_t_tuple &dims)
        : mb(std::get<0>(dims))
        , heads(std::get<1>(dims))
        , seq_len(std::get<2>(dims))
        , head_group(std::get<3>(dims))
        , dt(std::get<4>(dims))
        , key(std::get<5>(dims))
        , value(std::get<6>(dims))
        , qtype(std::get<7>(dims))
        , key_format_tag(std::get<8>(dims))
        , mask(std::get<9>(dims))
        , stype(std::get<10>(dims))
        , acc_modes(std::get<11>(dims))
        , dropout(std::get<12>(dims)) {}
};

struct sdpa_tensors_t {
    memory m_query, m_mask, m_output;
    // sum_k prob_k*|V_k|; sizes the absolute error floor in check_memory.
    memory m_output_absmag;
    memory m_key_quantized, m_value_quantized, m_output_quantized;
    memory m_scale; // tested sdpa arg, can be host-side scalar
    memory m_scale_prim; // reference (prim) sdpa arg
    memory m_dropout_prob, m_dropout_seed, m_dropout_offset;

    memory m_diff_query, m_diff_key, m_diff_value, m_diff_output, m_dS;
    memory m_diff_query_quantized, m_diff_key_quantized, m_diff_value_quantized;

    memory m_key_scales, m_key_zp, m_value_scales, m_value_zp;
    dnnl::primitive_attr sdpa_attr_quantized, sdpa_kq_attr_quantized,
            sdpa_vs_attr_quantized;

    int kq_mask, vs_mask;
    memory::dims kq_groups, vs_groups;
};

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t &p) {
    ss << "mb" << p.mb;
    if (p.heads.kv != p.heads.q) { ss << "_KVN" << p.heads.kv; }
    ss << "_N" << p.heads.q;
    ss << "_D" << p.head_group.head_size;
    if (p.head_group.head_size != p.head_group.kgroup_size
            || p.head_group.head_size != p.head_group.vgroup_size) {
        ss << "_" << p.head_group.kgroup_size << "x";
        ss << p.head_group.vgroup_size;
    }
    if (p.key_format_tag == memory::format_tag::abdc)
        ss << "_T";
    else
        ss << "_";
    ss << "K" << p.seq_len.kv;
    ss << "_Q" << p.seq_len.q;
    ss << "_Qdt_" << p.dt.dt;
    ss << "_Kdt_" << p.key.dt;
    if (is_quantized(p.key.dt, p.key.sdt, p.key.zpdt, p.qtype)) {
        ss << "x" << p.key.sdt;
        ss << "x" << p.key.zpdt;
    }
    ss << "_Vdt_" << p.value.dt;
    if (is_quantized(p.value.dt, p.value.sdt, p.value.zpdt, p.qtype)) {
        ss << "x" << p.value.sdt;
        ss << "x" << p.value.zpdt;
    }
    switch (p.mask.type) {
        case mask_type::no_mask: ss << "_no_mask"; break;
        case mask_type::oneD: ss << "_mask1D_" << p.mask.dt; break;
        case mask_type::twoD: ss << "_mask2D_" << p.mask.dt; break;
        case mask_type::causal_br: ss << "_maskcausalbr"; break;
        case mask_type::causal_tl: ss << "_maskcausaltl"; break;
    }
    switch (p.qtype) {
        case quantize_type::no_quantization: ss << "_noq"; break;
        case quantize_type::per_token_with_groups: ss << "_ptwg"; break;
        case quantize_type::per_token: ss << "_pt"; break;
        case quantize_type::per_tensor: ss << "_ptensor"; break;
        case quantize_type::per_tensor1: ss << "_ptensor1"; break;
        case quantize_type::per_tensor3: ss << "_ptensor3"; break;
    }
    if (p.stype == scale_type::host_side)
        ss << "_hostscale";
    else
        ss << "_devicescale";
    if (p.acc_modes.kq_acc == accumulation_mode::f16) ss << "_kqf16acc";
    if (p.acc_modes.vs_acc == accumulation_mode::f16) ss << "_vsf16acc";
    if (p.dropout.enabled()) ss << "_" << p.dropout;
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t_tuple &p) {
    ss << sdpa_dims_t(p);
    return ss;
}

std::string print_to_string(const ::testing::TestParamInfo<sdpa_dims_t> &info) {
    dnnl::impl::stringstream_t ss;
    ss << info.param;
    return ss.str();
}

std::string print_to_string2(
        const ::testing::TestParamInfo<sdpa_dims_t_tuple> &info) {
    dnnl::impl::stringstream_t ss;
    ss << sdpa_dims_t(info.param);
    return ss.str();
}

void print_table_header() {
    std::cout << "| mb | Q Heads | KV Heads |   D |    K  |    Q | Kdt | "
                 "Vdt | scale | "
                 "mask | quant |  time (ns) | BW eff/actual (Gbps) | "
                 "gemm/total FLOPs (GFLOPs) |\n";
}

std::string print_row(const sdpa_dims_t &p) {
    dnnl::impl::stringstream_t ss;

    ss << "|" << p.mb;
    ss << "|" << p.heads.q;
    ss << "|" << p.heads.kv;
    ss << "|" << p.head_group.head_size;
    ss << "|" << p.seq_len.kv;
    ss << "|" << p.seq_len.q;
    ss << "|" << p.key.dt;
    if (!(p.key.dt == mdt::f16 || p.key.dt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.key.sdt;
        ss << "/" << p.key.zpdt;
    }
    ss << "|" << p.value.dt;
    if (!(p.value.dt == mdt::f16 || p.value.dt == mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.value.sdt;
        ss << "/" << p.value.zpdt;
    }
    ss << "|";
    switch (p.stype) {
        case scale_type::device_side: ss << "device"; break;
        case scale_type::host_side: ss << "host"; break;
    }
    ss << "|";
    switch (p.mask.type) {
        case mask_type::no_mask: ss << "no"; break;
        case mask_type::oneD: ss << "1D"; break;
        case mask_type::twoD: ss << "2D"; break;
        case mask_type::causal_br: ss << "causalbr"; break;
        case mask_type::causal_tl: ss << "causaltl"; break;
    }
    ss << "|" << p.qtype;
    return ss.str();
}

using dnnl::algorithm;
using dnnl::matmul;
using dnnl::memory;
using dnnl::primitive_attr;
using dnnl::softmax_forward;

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#undef CHECK
#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, const memory::desc &desc) {
    const auto &dims = desc.get_dims();
    if (dims.empty()) return;
    size_t seq_len = dims[3];
    size_t query_num = dims[2];
    size_t batches = dims[1] * dims[0];
    for (size_t b = 0; b < batches; b++) {
        for (size_t q = 0; q < query_num; q++) {
            for (size_t i = 0; i < seq_len; i++) {
                if (i <= q) {
                    mask[b * query_num * seq_len + q * seq_len + i] = 0;
                    // = (float)i + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + i]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)i + (float)q / 100.f);
                }
            }
        }
    }
}

void fill_causal_mask(
        std::vector<float> &mask, const memory::desc &desc, mask_type mask_t) {
    const auto &dims = desc.get_dims();
    if (dims.empty()) return;
    int64_t seq_len = dims[3];
    int64_t query_num = dims[2];
    int64_t batches = dims[1] * dims[0];
    for (int64_t b = 0; b < batches; b++) {
        for (int64_t q = 0; q < query_num; q++) {
            for (int64_t k = 0; k < seq_len; k++) {
                if (mask_t == mask_type::causal_br
                                ? ((q + seq_len - query_num) >= k)
                                : (q >= k)) {
                    mask[b * query_num * seq_len + q * seq_len + k] = 0;
                    // = (float)k + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + k]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)k + (float)q / 100.f);
                }
            }
        }
    }
}

memory::dims double_mb(const memory::dims &dims) {
    memory::dims ret = dims;
    if (!ret.empty()) ret[0] *= 2;
    return ret;
}

/// This function creates a large tensor double the size requested by /p desc and
/// fills it with NaN values. It then creates a new memory object backed by
/// the first memory handle but with the size of the original memory descriptor.
///
/// This function allows us to identify situations where the SDPA kernel is
/// accessing data out-of-bounds
memory double_and_resize(const memory::desc &desc, dnnl::engine &eng,
        dnnl::stream &strm, std::vector<dnnl_memory_t> &doubled_memory) {
    memory::dims dims2 = double_mb(desc.get_dims());
    auto desc2 = memory::desc(dims2, desc.get_data_type(), desc.get_strides());

    dnnl_memory_t mem2;
    CHECK(dnnl_memory_create(
            &mem2, desc2.get(), eng.get(), DNNL_MEMORY_ALLOCATE));
    doubled_memory.push_back(mem2);

    void *handle;
    CHECK(dnnl_memory_get_data_handle(mem2, &handle));
    if (desc2.get_size()) {
        void *mapped_ptr = nullptr;
        strm.wait();
        CHECK(dnnl_memory_map_data(mem2, &mapped_ptr));
        memset(mapped_ptr, 0xFF, desc2.get_size());
        CHECK(dnnl_memory_unmap_data(mem2, mapped_ptr));
        strm.wait();
    }

    auto out = memory(desc, eng, handle);
    return out;
}

sdpa_tensors_t get_descriptors(dnnl::engine &eng, dnnl::stream &strm,
        const sdpa_dims_t &p, std::vector<dnnl_memory_t> &doubled_memory) {
    sdpa_tensors_t out;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz
            = {p.mb, p.heads.q, p.seq_len.q, p.head_group.head_size};
    const memory::dims k_sz
            = {p.mb, p.heads.kv, p.head_group.head_size, p.seq_len.kv};
    const memory::dims k_stride
            = {p.mb, p.heads.kv, p.head_group.head_size, p.seq_len.kv * 2};
    const memory::dims k_t_stride
            = {p.mb, p.heads.kv, p.seq_len.kv * 2, p.head_group.head_size};
    const memory::dims v_sz
            = {p.mb, p.heads.kv, p.seq_len.kv, p.head_group.head_size};
    const memory::dims scale_sz = {1, 1, 1, 1};

    bool with_host_scale = p.stype == scale_type::host_side;

    const memory::dims key_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {k_sz[0], k_sz[1],
                        k_sz[2] / p.head_group.kgroup_size, k_sz[3]};
            case quantize_type::per_token:
                return memory::dims {k_sz[0], k_sz[1], 1, k_sz[3]};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {k_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {k_sz[0], k_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();
    const memory::dims val_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {v_sz[0], v_sz[1], v_sz[2],
                        v_sz[3] / p.head_group.vgroup_size};
            case quantize_type::per_token:
                return memory::dims {v_sz[0], v_sz[1], v_sz[2], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {v_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {v_sz[0], v_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();

    auto def_scale_value = [&] { return std::sqrt(p.head_group.head_size); }();

    memory::dims mask_sz;
    switch (p.mask.type) {
        case mask_type::no_mask: mask_sz = {}; break;
        case mask_type::oneD: mask_sz = {1, 1, 1, p.seq_len.kv}; break;
        case mask_type::causal_br:
        case mask_type::causal_tl:
        case mask_type::twoD:
            mask_sz = {1, 1, p.seq_len.q, p.seq_len.kv};
            break;
    }

    auto ksdt = p.key.sdt == mdt::undef ? p.key.dt : p.key.sdt;
    auto kzpdt = p.key.zpdt == mdt::undef ? mdt::s8 : p.key.zpdt;
    auto vsdt = p.value.sdt == mdt::undef ? p.value.dt : p.value.sdt;
    auto vzpdt = p.value.zpdt == mdt::undef ? mdt::s8 : p.value.zpdt;

    dnnl::memory::format_tag abcd = dnnl::memory::format_tag::abcd;
    dnnl::memory::format_tag abdc = dnnl::memory::format_tag::abdc;
    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    // clang-format off
    auto query_md            = memory::desc(q_sz,          p.dt.dt,      abcd);

    auto query_test_md       = memory::desc(q_sz,          p.dt.dt,      abcd);

    auto key_quantized_md    = memory::desc(k_sz,          p.key.dt,   p.key_format_tag);
    auto key_scales_md       = memory::desc(key_scales_sz, ksdt,       abcd);
    auto key_zp_md           = memory::desc(key_scales_sz, kzpdt,      abcd);

    auto val_quantized_md    = memory::desc(v_sz,          p.value.dt, abcd);
    auto val_t_quantized_md  = memory::desc(v_sz,          p.value.dt, abdc);
    auto val_scales_md       = memory::desc(val_scales_sz, vsdt,      abcd);
    auto val_zp_md           = memory::desc(val_scales_sz, vzpdt,     abcd);


    auto mask_md             = memory::desc(mask_sz,       p.mask.dt != mdt::undef ? p.mask.dt : p.dt.dt,   abcd);
    auto output_md           = memory::desc(q_sz,          p.dt.dt,     abcd);
    auto output_quantized_md = memory::desc(q_sz,          p.dt.dt,     abcd);
    // clang-format on

    // Create memory objects
    out.m_query = double_and_resize(query_md, eng, strm, doubled_memory);
    out.m_diff_query = double_and_resize(query_md, eng, strm, doubled_memory);
    out.m_diff_query_quantized
            = double_and_resize(query_md, eng, strm, doubled_memory);

    out.m_key_quantized
            = double_and_resize(key_quantized_md, eng, strm, doubled_memory);
    out.m_diff_key
            = double_and_resize(key_quantized_md, eng, strm, doubled_memory);
    out.m_diff_key_quantized
            = double_and_resize(key_quantized_md, eng, strm, doubled_memory);

    out.m_key_scales
            = double_and_resize(key_scales_md, eng, strm, doubled_memory);
    out.m_key_zp = double_and_resize(key_zp_md, eng, strm, doubled_memory);

    out.m_value_quantized
            = double_and_resize(val_quantized_md, eng, strm, doubled_memory);
    out.m_diff_value
            = double_and_resize(val_quantized_md, eng, strm, doubled_memory);
    out.m_diff_value_quantized
            = double_and_resize(val_quantized_md, eng, strm, doubled_memory);

    out.m_value_scales
            = double_and_resize(val_scales_md, eng, strm, doubled_memory);
    out.m_value_zp = double_and_resize(val_zp_md, eng, strm, doubled_memory);
    out.m_mask = double_and_resize(mask_md, eng, strm, doubled_memory);

    out.m_output = double_and_resize(output_md, eng, strm, doubled_memory);
    // Always f32, regardless of the output dtype.
    auto output_absmag_md = memory::desc(q_sz, mdt::f32, abcd);
    out.m_output_absmag
            = double_and_resize(output_absmag_md, eng, strm, doubled_memory);
    out.m_output_quantized
            = double_and_resize(output_quantized_md, eng, strm, doubled_memory);
    out.m_diff_output = double_and_resize(output_md, eng, strm, doubled_memory);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz), 0.f);
    std::vector<float> scale_data(product(scale_sz), def_scale_value);
    std::vector<float> key_quantized_data(product(k_sz), 0);
    std::vector<float> val_quantized_data(product(v_sz), 0);
    std::vector<float> key_scale_data(product(key_scales_sz), std::nanf("1"));
    std::vector<float> val_scale_data(product(val_scales_sz), std::nanf("1"));
    std::vector<float> diff_output_data(product(q_sz), 0.f);

    std::vector<int> key_zp_data_signed(product(key_scales_sz), INT_MAX);
    std::vector<int> val_zp_data_signed(product(val_scales_sz), INT_MAX);

    std::vector<unsigned> key_zp_data_unsigned(product(key_scales_sz), INT_MAX);
    std::vector<unsigned> val_zp_data_unsigned(product(val_scales_sz), INT_MAX);

    std::vector<float> mask_data(product(mask_sz), NAN);
    std::vector<float> output_data(product(q_sz), NAN);

    out.sdpa_attr_quantized.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    out.kq_mask = 0;
    out.vs_mask = 0;
    out.kq_groups = {};
    out.vs_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            out.kq_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.kq_groups = {p.head_group.kgroup_size, 1};
            out.vs_groups = {1, p.head_group.vgroup_size};
            break;
        case quantize_type::per_token:
            out.kq_mask = 1 << 3 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 0 | 1 << 1 | 1 << 2;
            break;
        case quantize_type::per_tensor3:
            out.kq_mask = 3;
            out.vs_mask = 3;
            break;
        case quantize_type::per_tensor1:
            out.kq_mask = 1;
            out.vs_mask = 1;
            break;
        case quantize_type::per_tensor:
            out.kq_mask = 0;
            out.vs_mask = 0;
            break;
        case quantize_type::no_quantization: break;
    }

    out.sdpa_kq_attr_quantized.set_accumulation_mode(p.acc_modes.kq_acc);
    out.sdpa_vs_attr_quantized.set_accumulation_mode(p.acc_modes.vs_acc);

    if (p.qtype != quantize_type::no_quantization) {
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.sdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.key.sdt);
        }

        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.sdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.value.sdt);
        }

        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.key.zpdt);
        }

        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.value.zpdt);
        }
    }

    if (p.dropout.enabled()) {
        // Mask descriptor: u8, same 4D shape as the score matrix
        // {mb, heads_q, seq_len_q, seq_len_kv} — matches the SDPA query rank
        // and satisfies the documentation requirement that the mask has the
        // same dimensions as the softmax src/dst (the score tensor).
        memory::desc mask_desc;
        if (p.dropout.has_output_mask()) {
            const memory::dims score_sz_4d
                    = {p.mb, p.heads.q, p.seq_len.q, p.seq_len.kv};
            mask_desc = memory::desc(
                    score_sz_4d, mdt::u8, memory::format_tag::abcd);
        }
        out.sdpa_attr_quantized.set_dropout(mask_desc, p.dropout.seed_dt,
                p.dropout.offset != 0, p.dropout.use_host_scalars);

        if (p.dropout.use_host_scalars) {
            out.m_dropout_prob = memory(
                    memory::desc::host_scalar(mdt::f32), p.dropout.probability);
            if (p.dropout.seed_dt != mdt::s64) {
                throw std::runtime_error(
                        "SDPA dropout host scalar seed_dt must be s64");
            }
            out.m_dropout_seed = memory(memory::desc::host_scalar(mdt::s64),
                    static_cast<int64_t>(p.dropout.seed));
            if (p.dropout.offset != 0) {
                out.m_dropout_offset
                        = memory(memory::desc::host_scalar(mdt::s64),
                                static_cast<int64_t>(p.dropout.offset));
            }
        } else {
            auto prob_md = memory::desc({1}, mdt::f32, memory::format_tag::a);
            auto seed_md = memory::desc(
                    {1}, p.dropout.seed_dt, memory::format_tag::a);
            out.m_dropout_prob = memory(prob_md, eng);
            out.m_dropout_seed = memory(seed_md, eng);
            float prob = p.dropout.probability;
            int64_t seed = p.dropout.seed;
            write_to_dnnl_memory(&prob, out.m_dropout_prob, eng, strm);
            write_to_dnnl_memory(&seed, out.m_dropout_seed, eng, strm);
            if (p.dropout.offset != 0) {
                auto offset_md
                        = memory::desc({1}, mdt::s64, memory::format_tag::a);
                out.m_dropout_offset = memory(offset_md, eng);
                int64_t off = p.dropout.offset;
                write_to_dnnl_memory(&off, out.m_dropout_offset, eng, strm);
            }
        }
    }

    fill_random(query_data, query_md);
    fill_random(diff_output_data, output_md);
    fill_random_quantized(key_quantized_data, key_quantized_md,
            (p.key.dt == mdt::u4 || p.key.dt == mdt::u8));
    fill_random_quantized(val_quantized_data, val_quantized_md,
            (p.value.dt == mdt::u4 || p.value.dt == mdt::u8));
    if (p.qtype != quantize_type::no_quantization) {
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.sdt != mdt::undef) {
            fill_random_scales(key_scale_data, key_scales_md);
        } else {
            fill_value(key_scale_data, key_scales_md, 1.f);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.sdt != mdt::undef) {
            fill_random_scales(val_scale_data, val_scales_md);
        } else {
            fill_value(val_scale_data, val_scales_md, 1.f);
        }
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_signed, key_zp_md);
        } else {
            fill_value(key_zp_data_signed, key_zp_md, 0);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_signed, val_zp_md);
        } else {
            fill_value(val_zp_data_signed, val_zp_md, 0);
        }
        if (p.key.dt != mdt::f16 && p.key.dt != mdt::bf16
                && p.key.zpdt != mdt::undef) {
            fill_random_quantized(key_zp_data_unsigned, key_zp_md);
        } else {
            fill_value(key_zp_data_unsigned, key_zp_md, 0U);
        }
        if (p.value.dt != mdt::f16 && p.value.dt != mdt::bf16
                && p.value.zpdt != mdt::undef) {
            fill_random_quantized(val_zp_data_unsigned, val_zp_md);
        } else {
            fill_value(val_zp_data_unsigned, val_zp_md, 0U);
        }
    }

    if (p.mask.type == mask_type::causal_br
            || p.mask.type == mask_type::causal_tl) {
        fill_causal_mask(mask_data, mask_md, p.mask.type);
    } else {
        fill_mask(mask_data, mask_md);
    }

/// This section allows setting the values of the tensors using environment variables.
/// Syntax:
///    <Tensor Name>[<S for scales, Z for zero points>]<R for row C for column>
///
/// KR=3 KC=1 Set the value in the  Key tensor at (3, 1) to 1 and all other values should be zero
/// VSR=1 VSC=2  Set the scale for the Value tensor at (1, 2) to 1 and all other values to zero
#if 0
    auto &Q = query_data;
    auto &K = key_quantized_data;
    auto &V = val_quantized_data;
    auto &Ks = key_scale_data;
    auto &Vs = val_scale_data;
    auto &Kz = key_zp_data_signed;
    auto &Vz = val_zp_data_signed;
    auto d = p.head_group.head_size;
    auto k = p.seq_len.kv;
    auto q = p.seq_len.q;

    int kr = -1, kc = -1, qr = -1, qc = -1, vr = -1, vc = -1, mr = -1, mc = -1,
        xb = 0;
    int ksr = -1, ksc = -1, kzr = -1, kzc = -1, vsr = -1, vscales = -1,
        vzr = -1, vzc = -1;
    if (getenv("KR")) kr = atoi(getenv("KR"));
    if (getenv("KC")) kc = atoi(getenv("KC"));
    if (getenv("KSR")) ksr = atoi(getenv("KSR"));
    if (getenv("KSC")) ksc = atoi(getenv("KSC"));
    if (getenv("KZR")) kzr = atoi(getenv("KZR"));
    if (getenv("KZC")) kzc = atoi(getenv("KZC"));
    if (getenv("QR")) qr = atoi(getenv("QR"));
    if (getenv("QC")) qc = atoi(getenv("QC"));
    if (getenv("VR")) vr = atoi(getenv("VR"));
    if (getenv("VC")) vc = atoi(getenv("VC"));
    if (getenv("VSR")) vsr = atoi(getenv("VSR"));
    if (getenv("VScaleC")) vscales = atoi(getenv("VScaleC"));
    if (getenv("VZR")) vzr = atoi(getenv("VZR"));
    if (getenv("VZC")) vzc = atoi(getenv("VZC"));
    if (getenv("XB")) xb = atoi(getenv("XB"));

    if (getenv("MR")) mr = atoi(getenv("MR"));
    if (getenv("MC")) mc = atoi(getenv("MC"));

    if (mr >= 0 || mc >= 0) {
        mr = std::max(mr, 0);
        mc = std::max(mc, 0);
        for (auto &m : mask_data)
            m = 0;
        mask_data[mr * p.seq_len.kv + mc] = -999;
    }
    if (kr >= 0 || kc >= 0) {
        kr = std::max(kr, 0);
        kc = std::max(kc, 0);
        if (getenv("KX")) {
            for (int kr_ = 0; kr_ < d; kr_++)
                for (int kc_ = 0; kc_ < k; kc_++)
                    if (kr_ >= kr || kc_ >= kc) K[kr_ * k + kc_] = 0;
        } else {
            for (auto &k : K)
                k = 0;
            K[xb * d * k + kr * k + kc] = 1;
        }
    }
    if (ksr >= 0 || ksc >= 0) {
        ksr = std::max(ksr, 0);
        ksc = std::max(ksc, 0);
        for (auto &ks : Ks)
            ks = 0;
        Ks[(xb * d / p.kgroup_size * k + ksr * k) + ksc] = 1;
    }
    if (kzr >= 0 || kzc >= 0) {
        kzr = std::max(kzr, 0);
        kzc = std::max(kzc, 0);
        for (auto &kz : Kz)
            kz = 0;
        Kz[(xb * d * k + kzr * d) / p.kgroup_size + kzc] = 2;
    }
    if (qr >= 0 || qc >= 0) {
        qr = std::max(qr, 0);
        qc = std::max(qc, 0);
        if (getenv("QX")) {
            for (int qr_ = 0; qr_ < d; qr_++)
                for (int qc_ = 0; qc_ < q; qc_++)
                    if (qr_ >= qr || qc_ >= qc) Q[qr_ * d + qc_] = 0;
        } else {
            for (auto &q : Q)
                q = 0;
            Q[xb * d * q + qr * d + qc] = 1;
        }
    }
    if (vr >= 0 || vc >= 0) {
        vr = std::max(vr, 0);
        vc = std::max(vc, 0);
        if (getenv("VX")) {
            for (int vr_ = 0; vr_ < k; vr_++)
                for (int vc_ = 0; vc_ < d; vc_++)
                    if (vr_ >= vr || vc_ >= vc) V[vr_ * d + vc_] = 0;
        } else {
            for (auto &v : V)
                v = 0;
            V[xb * d * k + vr * d + vc] = 1;
        }
    }
    if (vsr >= 0 || vscales >= 0) {
        vsr = std::max(vsr, 0);
        vscales = std::max(vscales, 0);
        for (auto &vs : Vs)
            vs = 0;
        Vs[(xb * d * k + vscales * d) / p.vgroup_size + vsr] = 1;
    }
    if (vzr >= 0 || vzc >= 0) {
        vzr = std::max(vzr, 0);
        vzc = std::max(vzc, 0);
        for (auto &vz : Vz)
            vz = 0;
        Vz[(xb * d * k + vzc * d) / p.vgroup_size + vzr] = 1;
    }
#endif

    if (p.mask.type != mask_type::no_mask)
        write_to_dnnl_memory(mask_data.data(), out.m_mask, eng, strm);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(query_data.data(), out.m_query, eng, strm);

    write_to_dnnl_memory(diff_output_data.data(), out.m_diff_output, eng, strm);

    write_to_dnnl_memory(
            key_quantized_data.data(), out.m_key_quantized, eng, strm);

    write_to_dnnl_memory(
            val_quantized_data.data(), out.m_value_quantized, eng, strm);
    if (p.key.zpdt == mdt::s4 || p.key.zpdt == mdt::s8) {
        write_to_dnnl_memory(
                key_zp_data_signed.data(), out.m_key_zp, eng, strm);
    } else {
        write_to_dnnl_memory(
                key_zp_data_unsigned.data(), out.m_key_zp, eng, strm);
    }
    if (p.value.zpdt == mdt::s4 || p.value.zpdt == mdt::s8) {
        write_to_dnnl_memory(
                val_zp_data_signed.data(), out.m_value_zp, eng, strm);
    } else {
        write_to_dnnl_memory(
                val_zp_data_unsigned.data(), out.m_value_zp, eng, strm);
    }
    write_to_dnnl_memory(key_scale_data.data(), out.m_key_scales, eng, strm);
    write_to_dnnl_memory(val_scale_data.data(), out.m_value_scales, eng, strm);
    write_to_dnnl_memory(output_data.data(), out.m_output, eng, strm);
    write_to_dnnl_memory(output_data.data(), out.m_output_quantized, eng, strm);

    auto setup_device_scale = [&](memory *outmem) {
        auto scale_md = memory::desc(scale_sz, p.dt.dt, abcd);
        *outmem = double_and_resize(scale_md, eng, strm, doubled_memory);
        write_to_dnnl_memory(scale_data.data(), *outmem, eng, strm);
    };

    if (with_host_scale) {
        auto scale_md = memory::desc::host_scalar(p.dt.dt);
        float scale_val = (float)def_scale_value;
        switch (p.dt.dt) {
            case mdt::f32:
                out.m_scale = dnnl::memory(scale_md, (float)scale_val);
                break;
            case mdt::f16:
                out.m_scale = dnnl::memory(scale_md, (float16_t)scale_val);
                break;
            case mdt::bf16:
                out.m_scale = dnnl::memory(scale_md, (bfloat16_t)scale_val);
                break;
            default:
                throw std::runtime_error("Scale data type not supported\n");
        }
    } else
        setup_device_scale(&out.m_scale);
    setup_device_scale(&out.m_scale_prim);

    return out;
}

static std::unique_ptr<dnnl::engine> sdpa_eng;

dnnl::engine get_sdpa_test_engine() {
    return *sdpa_eng;
}

memory as(dnnl::stream &strm, memory &mem, memory::data_type dt) {
    const memory::dims sz = mem.get_desc().get_dims();

    auto md = memory::desc(sz, dt, mem.get_desc().get_strides());
    auto out = memory(md, mem.get_engine());
    dnnl::reorder(mem, out).execute(strm, mem, out);
    return out;
}

memory reshape(dnnl::stream &strm, memory &mem, const memory::desc &md) {
    auto out = memory(md, mem.get_engine());
    strm.wait();
    void *mem_ptr_ = (void *)mem.map_data();
    if (mem_ptr_ == nullptr)
        throw std::runtime_error("Failed to map mem in resize");
    void *out_ptr_ = (void *)out.map_data();
    if (out_ptr_ == nullptr)
        throw std::runtime_error("Failed to map out in resize");
    memcpy(out_ptr_, mem_ptr_, mem.get_desc().get_size());
    mem.unmap_data(mem_ptr_);
    out.unmap_data(out_ptr_);
    return out;
}

std::pair<dnnl::reorder, memory> dequantize_prim(const engine &eng, mdt dt,
        const memory::desc &desc, int mask, const memory::dims &groups, mdt sdt,
        mdt zpdt,
        dnnl::memory::format_tag tag = dnnl::memory::format_tag::abcd) {
    auto dequantized_md = memory::desc(desc.get_dims(), dt, tag);
    primitive_attr dequantized_attr;

    if (sdt != mdt::undef) {
        dequantized_attr.set_scales(DNNL_ARG_FROM, mask, groups, sdt);
    }
    if (zpdt != mdt::undef) {
        dequantized_attr.set_zero_points(DNNL_ARG_SRC, mask, groups, zpdt);
    }

    auto dequantize_pd = dnnl::reorder::primitive_desc(
            eng, desc, eng, dequantized_md, dequantized_attr, false);

    memory dequantized_mem = memory(
            {desc.get_dims(), dt, dnnl::memory::format_tag::abcd}, eng);
    return std::make_pair(dnnl::reorder(dequantize_pd), dequantized_mem);
}

void prim_sdpa_quant(const sdpa_dims_t &p, const sdpa_tensors_t &t,
        dnnl::engine &eng, dnnl::stream &strm, dnnl::memory &query,
        dnnl::memory &key, dnnl::memory &key_scales, dnnl::memory &key_zp,
        dnnl::memory::data_type scale_dt, dnnl::memory &scale_device,
        dnnl::memory &mask, dnnl::memory &value, dnnl::memory &value_scales,
        dnnl::memory &value_zp, dnnl::memory &output, bool invert_scale,
        dnnl::memory *dropout_mask_out,
        std::vector<dnnl_memory_t> &doubled_memory,
        dnnl::memory *output_absmag = nullptr) {
    using namespace dnnl;
    primitive_attr bmm1_attr;
    post_ops bmm1_po;
    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    auto mask_f32 = as(strm, mask, mdt::f32);
    auto mask_sz = mask.get_desc().get_dims();
    auto scale_f32 = as(strm, scale_device, mdt::f32);

    if (scale_dt != mdt::undef) {
        scale_f32 = reshape(strm, scale_f32,
                {{1, 1, 1, 1, 1}, mdt::f32, memory::format_tag::abcde});
        if (invert_scale)
            bmm1_po.append_binary(algorithm::binary_div, scale_f32.get_desc());
        else
            bmm1_po.append_binary(algorithm::binary_mul, scale_f32.get_desc());
    }
    if (p.mask.type != mask_type::no_mask) {
        mask_f32 = reshape(strm, mask_f32,
                {{mask_sz[0], 1, 1, mask_sz[2], mask_sz[3]}, mdt::f32,
                        memory::format_tag::abcde});
        bmm1_po.append_binary(algorithm::binary_add, mask_f32.get_desc());
    }

    int head_kv_group_size = 0;
    int head_q_group_size = 0;
    int head_group_batches = 0;
    if (p.heads.kv == p.heads.q) {
        head_kv_group_size = p.heads.kv;
        head_q_group_size = p.heads.q;
        head_group_batches = 1;
    } else {
        head_kv_group_size = 1;
        head_q_group_size = p.heads.q / p.heads.kv;
        head_group_batches = p.heads.kv;
    }

    auto original_k_sz = key.get_desc().get_dims();
    const memory::dims k_sz {p.mb, head_group_batches, head_kv_group_size,
            original_k_sz[2], original_k_sz[3]};
    const memory::dims v_sz {p.mb, head_group_batches, head_kv_group_size,
            p.seq_len.kv, p.head_group.head_size};
    const memory::dims q_sz {p.mb, head_group_batches, head_q_group_size,
            p.seq_len.q, p.head_group.head_size};
    memory::desc grouped_key_md(k_sz, p.dt.dt, memory::format_tag::abcde);
    memory::desc grouped_value_md(v_sz, mdt::f32, memory::format_tag::abcde);
    memory::desc grouped_query_md(q_sz, p.dt.dt, memory::format_tag::abcde);

    memory key_dequantized;
    if ((key.get_desc().get_data_type() != mdt::f16
                && key.get_desc().get_data_type() != mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {

        dnnl::reorder key_dequantize_prim;
        std::tie(key_dequantize_prim, key_dequantized)
                = dequantize_prim(eng, p.dt.dt, key.get_desc(), t.kq_mask,
                        t.kq_groups, p.key.sdt, p.key.zpdt);

        std::unordered_map<int, memory> key_dequantize_args = {
                {DNNL_ARG_FROM, key},
                {DNNL_ARG_TO, key_dequantized},
        };
        if (p.key.sdt != mdt::undef) {
            key_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = key_scales;
        }
        if (p.key.zpdt != mdt::undef)
            key_dequantize_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM]
                    = key_zp;
        key_dequantize_prim.execute(strm, key_dequantize_args);
        key_dequantized = reshape(strm, key_dequantized, grouped_key_md);
    } else {
        auto keytmp = as(strm, key, p.dt.dt);
        grouped_key_md = p.key_format_tag == memory::format_tag::abcd
                ? memory::desc(k_sz, p.dt.dt, memory::format_tag::abcde)
                : memory::desc(k_sz, p.dt.dt, memory::format_tag::abced);

        key_dequantized = reshape(strm, keytmp, grouped_key_md);
    }

    memory value_dequantized;
    if (value.get_desc().get_data_type() != mdt::f16
            && value.get_desc().get_data_type() != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        dnnl::reorder value_dequantize_prim;
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f32, value.get_desc(), t.vs_mask,
                        t.vs_groups, p.value.sdt, p.value.zpdt);

        std::unordered_map<int, memory> value_dequantize_args = {
                {DNNL_ARG_FROM, value},
                {DNNL_ARG_TO, value_dequantized},
        };
        if (p.value.sdt != mdt::undef) {
            value_dequantize_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM]
                    = value_scales;
        }
        if (p.value.zpdt != mdt::undef)
            value_dequantize_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM]
                    = value_zp;
        value_dequantize_prim.execute(strm, value_dequantize_args);
        value_dequantized = reshape(strm, value_dequantized, grouped_value_md);
    } else {
        auto value32 = as(strm, value, mdt::f32);
        value_dequantized = reshape(strm, value32, grouped_value_md);
    }

    memory grouped_query = reshape(strm, query, grouped_query_md);

    const memory::dims score_sz = {p.mb, head_group_batches, head_q_group_size,
            p.seq_len.q, p.seq_len.kv};
    memory::desc score_md {score_sz, mdt::f32, memory::format_tag::abcde};
    memory::desc score_f16_md {score_sz, mdt::f16, memory::format_tag::abcde};

    auto score = memory(score_md, eng);
    auto score_f16 = memory(score_f16_md, eng);
    auto score2 = memory(score_md, eng);
    auto score2_f16 = memory(score_f16_md, eng);

    const bool is_kq_acc_f16 = (p.acc_modes.kq_acc == accumulation_mode::f16);
    const bool is_vs_acc_f16 = (p.acc_modes.vs_acc == accumulation_mode::f16);

    // matmul primitive for KQ
    if (is_kq_acc_f16) {
        bmm1_attr.set_accumulation_mode(dnnl::accumulation_mode::f16);
    } else {
        bmm1_attr.set_post_ops(bmm1_po);
    }
    auto bmm1_pd = is_kq_acc_f16
            ? matmul::primitive_desc(eng, grouped_query_md,
                      key_dequantized.get_desc(), score_f16_md, bmm1_attr)
            : matmul::primitive_desc(eng, grouped_query_md,
                      key_dequantized.get_desc(), score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    // reorder primitive to convert f16 to f32 after bmm1
    auto f16_to_f32_pd
            = dnnl::reorder::primitive_desc(eng, score_f16_md, eng, score_md);
    auto f16_to_f32_prim = dnnl::reorder(f16_to_f32_pd);

    // binary primitive for scaling
    primitive_attr binary_attr;
    auto scale_algo
            = invert_scale ? algorithm::binary_div : algorithm::binary_mul;
    auto bin_pd = binary::primitive_desc(eng, scale_algo, score_md,
            scale_f32.get_desc(), score_md, binary_attr);
    auto bin_prim = binary(bin_pd);

    // binary primitive for mask addition if needed
    binary mask_prim;
    if (p.mask.type != mask_type::no_mask) {
        // reshape mask to match score dimensions
        mask_f32 = reshape(strm, mask_f32,
                {{mask_sz[0], 1, 1, mask_sz[2], mask_sz[3]}, mdt::f32,
                        memory::format_tag::abcde});

        auto mask_bin_pd = binary::primitive_desc(eng, algorithm::binary_add,
                score_md, mask_f32.get_desc(), score_md, binary_attr);
        mask_prim = binary(mask_bin_pd);
    }

    // softmax primitive
    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::library);
    if (p.dropout.enabled()) {
        memory::desc mask_desc;
        if (p.dropout.has_output_mask()) {
            // Dropout mask must match the softmax src/dst shape (score_sz),
            // not q_sz (which has head_size in the last dim, not seq_len.kv).
            // score_sz is 5D so always use abcde.
            mask_desc = memory::desc(
                    score_sz, mdt::u8, memory::format_tag::abcde);
        }
        softmax_attr.set_dropout(mask_desc, p.dropout.seed_dt,
                p.dropout.offset != 0, p.dropout.use_host_scalars);
    }
    softmax_forward::primitive_desc softmax_pd;
    softmax_forward softmax_prim;
    softmax_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_inference,
            (algorithm)dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
            score.get_desc(), score.get_desc(), 4, softmax_attr);
    softmax_prim = softmax_forward(softmax_pd);

    auto f32_to_f16_pd
            = dnnl::reorder::primitive_desc(eng, score_md, eng, score_f16_md);
    auto f32_to_f16_prim = dnnl::reorder(f32_to_f16_pd);

    // attention_output = attention_probs x value
    primitive_attr bmm2_attr;

    bmm2_attr.set_scratchpad_mode(scratchpad_mode::library);
    if (is_vs_acc_f16) {
        bmm2_attr.set_accumulation_mode(dnnl::accumulation_mode::f16);
    }
    memory::desc grouped_output_f16_md(
            grouped_query_md.get_dims(), mdt::f16, memory::format_tag::abcde);
    auto grouped_output_f16 = memory(grouped_output_f16_md, eng);
    auto grouped_output
            = double_and_resize(grouped_query_md, eng, strm, doubled_memory);

    // matmul primitive for VS
    auto bmm2_pd = is_vs_acc_f16
            ? matmul::primitive_desc(eng, score_f16_md, grouped_value_md,
                      grouped_output_f16_md, bmm2_attr)
            : matmul::primitive_desc(eng, score_md, grouped_value_md,
                      grouped_query_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // reorder primitive to convert f16 to f32 after bmm2
    auto f16_to_f32_output_pd = dnnl::reorder::primitive_desc(
            eng, grouped_output_f16_md, eng, grouped_query_md);
    auto f16_to_f32_output_prim = dnnl::reorder(f16_to_f32_output_pd);

    // setup args
    std::unordered_map<int, memory> bmm1_args = {{DNNL_ARG_SRC, grouped_query},
            {DNNL_ARG_WEIGHTS, key_dequantized}, {DNNL_ARG_DST, score}};
    if (is_kq_acc_f16) {
        bmm1_args[DNNL_ARG_DST] = score_f16;
    } else {
        if (scale_dt != mdt::undef) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                    = std::move(scale_f32);
            if (p.mask.type != mask_type::no_mask) {
                bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                        = std::move(mask_f32);
            }
        } else {
            if (p.mask.type != mask_type::no_mask) {
                bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                        = std::move(mask_f32);
            }
        }
    }

    std::unordered_map<int, memory> scale_args = {{DNNL_ARG_SRC_0, score},
            {DNNL_ARG_SRC_1, scale_f32}, {DNNL_ARG_DST, score}};

    std::unordered_map<int, memory> mask_args = {{DNNL_ARG_SRC_0, score},
            {DNNL_ARG_SRC_1, mask_f32}, {DNNL_ARG_DST, score}};

    std::unordered_map<int, memory> bmm2_args
            = {{DNNL_ARG_SRC, score2}, {DNNL_ARG_WEIGHTS, value_dequantized},
                    {DNNL_ARG_DST, grouped_output}};
    if (is_vs_acc_f16) {
        bmm2_args[DNNL_ARG_SRC] = score2_f16;
        bmm2_args[DNNL_ARG_DST] = grouped_output_f16;
    }

    const auto loop = [&]() {
        // KQ
        if (is_kq_acc_f16) {
            // f16 accumulation needs separate binary post-ops
            bmm1_prim.execute(strm, bmm1_args);

            // convert f16 to f32 for binary operations
            f16_to_f32_prim.execute(
                    strm, {{DNNL_ARG_FROM, score_f16}, {DNNL_ARG_TO, score}});

            // binary scale
            bin_prim.execute(strm, scale_args);

            // execute masking if needed
            if (p.mask.type != mask_type::no_mask) {
                mask_prim.execute(strm, mask_args);
            }
        } else {
            bmm1_prim.execute(strm, bmm1_args);
        }

        // softmax
        std::unordered_map<int, memory> softmax_args
                = {{DNNL_ARG_SRC, score}, {DNNL_ARG_DST, score2}};
        if (p.dropout.enabled()) {
            softmax_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            softmax_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0) {
                softmax_args[DNNL_ARG_ATTR_DROPOUT_OFFSET] = t.m_dropout_offset;
            }
            if (p.dropout.has_output_mask()) {
                auto dropout_mask_md = softmax_pd.query_md(
                        dnnl::query::exec_arg_md, DNNL_ARG_ATTR_DROPOUT_MASK);
                *dropout_mask_out = memory(dropout_mask_md, eng);
                softmax_args[DNNL_ARG_ATTR_DROPOUT_MASK] = *dropout_mask_out;
            }
        }
        softmax_prim.execute(strm, softmax_args);

        if (is_vs_acc_f16) {
            // convert f16 output back to f32
            f32_to_f16_prim.execute(
                    strm, {{DNNL_ARG_FROM, score2}, {DNNL_ARG_TO, score2_f16}});
        }

        // SV
        bmm2_prim.execute(strm, bmm2_args);

        if (is_vs_acc_f16) {
            // convert f16 output back to f32
            f16_to_f32_output_prim.execute(strm,
                    {{DNNL_ARG_FROM, grouped_output_f16},
                            {DNNL_ARG_TO, grouped_output}});
        }
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    strm.wait();

    void *output_ptr_ = (void *)output.map_data();
    void *grouped_output_ptr_ = (void *)grouped_output.map_data();
    memcpy(output_ptr_, grouped_output_ptr_, grouped_query_md.get_size());
    grouped_output.unmap_data(grouped_output_ptr_);
    output.unmap_data(output_ptr_);
    strm.wait();

    // = matmul(score2, |V|) since prob >= 0; score2 carries dropout scaling.
    if (output_absmag) {
        memory::desc grouped_absmag_md(grouped_query_md.get_dims(), mdt::f32,
                memory::format_tag::abcde);
        auto abs_value = memory(grouped_value_md, eng);
        auto abs_pd = eltwise_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::eltwise_abs,
                grouped_value_md, grouped_value_md, 0.f, 0.f);
        eltwise_forward(abs_pd).execute(strm,
                {{DNNL_ARG_SRC, value_dequantized}, {DNNL_ARG_DST, abs_value}});

        auto absmag_pd = matmul::primitive_desc(
                eng, score_md, grouped_value_md, grouped_absmag_md);
        auto grouped_absmag = memory(grouped_absmag_md, eng);
        matmul(absmag_pd).execute(strm,
                {{DNNL_ARG_SRC, score2}, {DNNL_ARG_WEIGHTS, abs_value},
                        {DNNL_ARG_DST, grouped_absmag}});
        strm.wait();

        void *dst = (void *)output_absmag->map_data();
        void *src = (void *)grouped_absmag.map_data();
        memcpy(dst, src, grouped_absmag_md.get_size());
        grouped_absmag.unmap_data(src);
        output_absmag->unmap_data(dst);
        strm.wait();
    }
}

std::vector<std::chrono::nanoseconds> timeit(
        const std::function<void()> &func, dnnl::stream &str, int iterations) {
    using namespace std::chrono;
    func();
    func();
    std::vector<std::chrono::nanoseconds> times;
    for (int j = 0; j < 5; j++) {
        auto e = steady_clock::now();
        str.wait();
        auto s = steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        str.wait();
        e = steady_clock::now();
        times.push_back(std::chrono::duration_cast<nanoseconds>(e - s));
    }
    return times;
}

std::chrono::nanoseconds prim_sdpa_quant_bwd(const sdpa_dims_t &p,
        const sdpa_tensors_t &t, dnnl::engine &eng, dnnl::stream &strm,
        dnnl::memory &query, dnnl::memory &key,
        dnnl::memory::data_type scale_dt, dnnl::memory &scale,
        dnnl::memory &mask, dnnl::memory &value, dnnl::memory &output,
        dnnl::memory &diff_output, bool invert_scale,
        std::vector<dnnl_memory_t> &doubled_memory, dnnl::memory &diff_query,
        dnnl::memory &diff_key, dnnl::memory &diff_value,
        bool with_timing = false) {

    using namespace dnnl;

    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    post_ops bmm1_po;
    auto scale_f32 = as(strm, scale, mdt::f32);
    auto mask_f32 = as(strm, mask, mdt::f32);
    auto mask_sz = mask.get_desc().get_dims();

    if (scale_dt != mdt::undef) {
        scale_f32 = reshape(strm, scale_f32,
                {{1, 1, 1, 1, 1}, mdt::f32, memory::format_tag::abcde});
        if (invert_scale)
            bmm1_po.append_binary(algorithm::binary_div, scale_f32.get_desc());
        else
            bmm1_po.append_binary(algorithm::binary_mul, scale_f32.get_desc());
    }
    if (p.mask.type != mask_type::no_mask) {
        mask_f32 = reshape(strm, mask_f32,
                {{mask_sz[0], 1, 1, mask_sz[2], mask_sz[3]}, mdt::f32,
                        memory::format_tag::abcde});
        bmm1_po.append_binary(algorithm::binary_add, mask_f32.get_desc());
    }
    bmm1_attr.set_post_ops(bmm1_po);
    // Keep a copy of the 5D scale for the backward pass.
    // bmm1_args will std::move scale_f32, so save it before that happens.
    memory scale_f32_bwd = scale_f32;

    int head_kv_group_size = 0;
    int head_q_group_size = 0;
    int head_group_batches = 0;
    if (p.heads.kv == p.heads.q) {
        head_kv_group_size = p.heads.kv;
        head_q_group_size = p.heads.q;
        head_group_batches = 1;
    } else {
        head_kv_group_size = 1;
        head_q_group_size = p.heads.q / p.heads.kv;
        head_group_batches = p.heads.kv;
    }

    auto original_k_sz = key.get_desc().get_dims();
    const memory::dims k_sz {p.mb, head_group_batches, head_kv_group_size,
            original_k_sz[2], original_k_sz[3]};
    const memory::dims v_sz {p.mb, head_group_batches, head_kv_group_size,
            p.seq_len.kv, p.head_group.head_size};
    const memory::dims q_sz {p.mb, head_group_batches, head_q_group_size,
            p.seq_len.q, p.head_group.head_size};

    memory::desc grouped_key_md(k_sz, p.dt.dt, memory::format_tag::abcde);
    memory::desc grouped_value_md(v_sz, p.dt.dt, memory::format_tag::abcde);
    memory::desc grouped_query_md(q_sz, p.dt.dt, memory::format_tag::abcde);

    memory key_dequantized;
    auto keytmp = as(strm, key, p.dt.dt);
    grouped_key_md = p.key_format_tag == memory::format_tag::abcd
            ? memory::desc(k_sz, p.dt.dt, memory::format_tag::abcde)
            : memory::desc(k_sz, p.dt.dt, memory::format_tag::abced);

    key_dequantized = reshape(strm, keytmp, grouped_key_md);

    memory value_dequantized;
    auto value32 = as(strm, value, p.dt.dt);
    value_dequantized = reshape(strm, value32, grouped_value_md);

    memory grouped_query = reshape(strm, query, grouped_query_md);

    const memory::dims score_sz = {p.mb, head_group_batches, head_q_group_size,
            p.seq_len.q, p.seq_len.kv};
    memory::desc score_md {score_sz, mdt::f32, memory::format_tag::abcde};
    memory::desc score_f16_md {score_sz, mdt::f16, memory::format_tag::abcde};

    auto score = memory(score_md, eng);
    auto score_f16 = memory(score_f16_md, eng);
    auto score2 = memory(score_md, eng);
    auto score2_f16 = memory(score_f16_md, eng);

    // matmul primitive for QK
    auto bmm1_pd = matmul::primitive_desc(eng, grouped_query_md,
            key_dequantized.get_desc(), score_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    // softmax forward
    primitive_attr softmax_attr;
    auto softmax_fwd_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_training,
            (algorithm)dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
            score.get_desc(), score.get_desc(), 4, softmax_attr);
    auto softmax_prim = softmax_forward(softmax_fwd_pd);

    // Apply dropout as a separate eltwise (identity) primitive so that
    // score2 retains the un-dropped softmax output P for the backward pass.
    auto score2_dropped = p.dropout.enabled() ? memory(score_md, eng) : score2;
    eltwise_forward dropout_eltwise;
    if (p.dropout.enabled()) {
        primitive_attr dropout_attr;
        dropout_attr.set_scratchpad_mode(scratchpad_mode::library);
        memory::desc mask_desc = {};
        dropout_attr.set_dropout(mask_desc, p.dropout.seed_dt,
                p.dropout.offset != 0, p.dropout.use_host_scalars);
        auto dropout_pd = eltwise_forward::primitive_desc(eng,
                prop_kind::forward_inference, algorithm::eltwise_linear,
                score_md, score_md, 1.0f, 0.0f, dropout_attr);
        dropout_eltwise = eltwise_forward(dropout_pd);
    }

    // attention_output = attention_probs x value
    primitive_attr bmm2_attr;

    bmm2_attr.set_scratchpad_mode(scratchpad_mode::library);
    auto grouped_output
            = double_and_resize(grouped_query_md, eng, strm, doubled_memory);
    auto bmm2_pd = matmul::primitive_desc(
            eng, score_md, grouped_value_md, grouped_query_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    // setup args
    std::unordered_map<int, memory> bmm1_args = {{DNNL_ARG_SRC, grouped_query},
            {DNNL_ARG_WEIGHTS, key_dequantized}, {DNNL_ARG_DST, score}};

    if (scale_dt != mdt::undef) {
        bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                = std::move(scale_f32);
        if (p.mask.type != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = std::move(mask_f32);
        }
    } else {
        if (p.mask.type != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                    = std::move(mask_f32);
        }
    }

    std::unordered_map<int, memory> bmm2_args = {{DNNL_ARG_SRC, score2_dropped},
            {DNNL_ARG_WEIGHTS, value_dequantized},
            {DNNL_ARG_DST, grouped_output}};

    const auto fwd_loop = [&]() {
        // QK
        bmm1_prim.execute(strm, bmm1_args);

        // softmax (no dropout — keeps un-dropped P in score2)
        softmax_prim.execute(
                strm, {{DNNL_ARG_SRC, score}, {DNNL_ARG_DST, score2}});

        // apply dropout separately: score2 -> score2_dropped
        if (p.dropout.enabled()) {
            std::unordered_map<int, memory> dropout_args
                    = {{DNNL_ARG_SRC, score2}, {DNNL_ARG_DST, score2_dropped}};
            dropout_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            dropout_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0) {
                dropout_args[DNNL_ARG_ATTR_DROPOUT_OFFSET] = t.m_dropout_offset;
            }
            dropout_eltwise.execute(strm, dropout_args);
        }

        // SV (uses dropped P)
        bmm2_prim.execute(strm, bmm2_args);
    };

    // Warmup run.
    // Execute primitives of sdpa.
    fwd_loop();
    strm.wait();

#if DEBUG_PRINT_MEM
    print_mem(grouped_query, "FWD grouped_query");
    print_mem(key_dequantized, "FWD keq_deq");
    print_mem(scale, "FWD scale");
    if (p.mask.type != mask_type::no_mask) { print_mem(mask, "FWD mask"); }
    print_mem(score, "FWD intermediate score");
    print_mem(score2, "FWD intermediate score2");
#endif

    void *output_ptr_ = (void *)output.map_data();
    void *grouped_output_ptr_ = (void *)grouped_output.map_data();
    memcpy(output_ptr_, grouped_output_ptr_, grouped_query_md.get_size());
    grouped_output.unmap_data(grouped_output_ptr_);
    output.unmap_data(output_ptr_);

    strm.wait();

    // end forward portion
    // backwards pass of SDPA

    // init incoming gradient, dO
    memory::desc grouped_output_md = grouped_output.get_desc();
    memory dO_mem = reshape(strm, diff_output, grouped_output_md);

#if DEBUG_PRINT_MEM
    printf("\n\n================\n\n\n");
    print_mem(dO_mem, "BWD incoming dO");
#endif

    ///////// final MM
    // matmul forward, O = s2 * v

    // init v, score2 gradients
    memory::desc diff_score2_md(score_sz, mdt::f32, memory::format_tag::abcde);
    memory dV_mem(grouped_value_md, eng);
    memory diff_score2_mem(diff_score2_md, eng);

    // backwards pass gradient of s2 (dS2 = dO * v^t)
    memory::desc v_t_md
            = memory::desc({v_sz[0], v_sz[1], v_sz[2], v_sz[4], v_sz[3]},
                    /*dO.get_dt*/ p.dt.dt, memory::format_tag::abced);
    primitive_attr mm_bwd_dS2_attr;
    mm_bwd_dS2_attr.set_scratchpad_mode(scratchpad_mode::library);
    matmul::primitive_desc mm_bwd_dS2_pd(
            eng, grouped_output_md, v_t_md, diff_score2_md, mm_bwd_dS2_attr);
    matmul mm_bwd_dS2(mm_bwd_dS2_pd);
    std::unordered_map<int, memory> mm_bwd_dS2_args
            = {{DNNL_ARG_SRC, dO_mem}, {DNNL_ARG_WEIGHTS, value_dequantized},
                    {DNNL_ARG_DST, diff_score2_mem}};
    mm_bwd_dS2.execute(strm, mm_bwd_dS2_args);
    strm.wait();

#if DEBUG_PRINT_MEM
    print_mem(diff_score2_mem, "BWD dS2");
#endif

    // backwards pass gradient of v (dV = s2^t * dO)

    // downcast score to p.dt.dt
    memory::desc s2_t_md = memory::desc(
            {score_sz[0], score_sz[1], score_sz[2], score_sz[4], score_sz[3]},
            p.dt.dt, memory::format_tag::abced);

    // dV uses P_dropped (score2_dropped) to match the kernel
    memory score2_downcast;
    auto score32 = as(strm, score2_dropped, p.dt.dt);
    score2_downcast = reshape(strm, score32, s2_t_md);

    const bool is_gqa = (head_q_group_size != head_kv_group_size);
    memory::dims dV_full_dims = {v_sz[0], v_sz[1],
            is_gqa ? head_q_group_size : head_kv_group_size, v_sz[3], v_sz[4]};
    memory::desc dV_full_md(dV_full_dims, p.dt.dt, memory::format_tag::abcde);
    memory dV_full_mem = is_gqa ? memory(dV_full_md, eng) : dV_mem;
    matmul::primitive_desc mm_bwd_dV_pd(
            eng, s2_t_md, grouped_output_md, dV_full_mem.get_desc());
    matmul mm_bwd_dV(mm_bwd_dV_pd);
    mm_bwd_dV.execute(strm,
            {{DNNL_ARG_SRC, score2_downcast}, {DNNL_ARG_WEIGHTS, dO_mem},
                    {DNNL_ARG_DST, dV_full_mem}});
    strm.wait();
    // reduce [mb, groups, hq_group, kv, D] -> [mb, groups, 1, kv, D] for GQA
    dnnl::reduction dV_reduce;
    if (is_gqa) {
        dnnl::reduction::primitive_desc dV_reduce_pd(eng,
                algorithm::reduction_sum, dV_full_md, grouped_value_md, 0.f,
                0.f);
        dV_reduce = dnnl::reduction(dV_reduce_pd);
        dV_reduce.execute(
                strm, {{DNNL_ARG_SRC, dV_full_mem}, {DNNL_ARG_DST, dV_mem}});
        strm.wait();
    }

#if DEBUG_PRINT_MEM
    print_mem(dV_mem, "BWD dV");
#endif

    ///////// intermediate softmax(QK)

    // compute softmax backward and scale in f32 to match kernel/graph
    memory::desc diff_score_f32_md(
            score_sz, mdt::f32, memory::format_tag::abcde);
    memory diff_score_f32_mem(diff_score_f32_md, eng);

    // Apply dropout Jacobian to dP_raw: dP = dP_raw * Z / q
    if (p.dropout.enabled()) {
        std::unordered_map<int, memory> dropout_bwd_args
                = {{DNNL_ARG_SRC, diff_score2_mem},
                        {DNNL_ARG_DST, diff_score2_mem}};
        dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
        dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
        if (p.dropout.offset != 0) {
            dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_OFFSET] = t.m_dropout_offset;
        }
        dropout_eltwise.execute(strm, dropout_bwd_args);
        strm.wait();
    }

    // backwards pass gradient of softmax (uses un-dropped P as DST)
    softmax_backward::primitive_desc softmax_bwd_pd(eng,
            algorithm::softmax_accurate, diff_score_f32_md, diff_score2_md,
            score_md, 4, softmax_fwd_pd);
    softmax_backward softmax_bwd(softmax_bwd_pd);
    softmax_bwd.execute(strm,
            {{DNNL_ARG_DST, score2}, {DNNL_ARG_DIFF_DST, diff_score2_mem},
                    {DNNL_ARG_DIFF_SRC, diff_score_f32_mem}});

    binary scale_bwd_prim;
    if (scale_dt != mdt::undef) {
        auto scale_algo_bwd
                = invert_scale ? algorithm::binary_div : algorithm::binary_mul;
        auto scale_bwd_pd = binary::primitive_desc(eng, scale_algo_bwd,
                diff_score_f32_md, scale_f32_bwd.get_desc(), diff_score_f32_md);
        scale_bwd_prim = binary(scale_bwd_pd);
        scale_bwd_prim.execute(strm,
                {{DNNL_ARG_SRC_0, diff_score_f32_mem},
                        {DNNL_ARG_SRC_1, scale_f32_bwd},
                        {DNNL_ARG_DST, diff_score_f32_mem}});
    }

    // downcast dS from f32 to p.dt.dt for dQ/dK matmuls
    memory::desc diff_score_md(score_sz, p.dt.dt, memory::format_tag::abcde);
    memory diff_score_mem(diff_score_md, eng);
    dnnl::reorder diff_score_cast(diff_score_f32_mem, diff_score_mem);
    diff_score_cast.execute(strm, diff_score_f32_mem, diff_score_mem);

    strm.wait();
    // print softmax gradient
#if DEBUG_PRINT_MEM
    print_mem(diff_score_mem, "BWD dS");
#endif

    ///////// first MM
    // matmul forward, s = q * k

    // init q,k gradients
    memory dQ_mem(grouped_query_md, eng);
    memory dK_mem(key_dequantized.get_desc(), eng);

    // backwards pass gradient of q (dQ = dS * k^t)
    // k^t requires transposed format and dims
    auto k_t_fmt = p.key_format_tag == memory::format_tag::abcd
            ? memory::format_tag::abced
            : memory::format_tag::abcde;
    memory::desc k_t_md = memory::desc(
            {k_sz[0], k_sz[1], k_sz[2], k_sz[4], k_sz[3]}, p.dt.dt, k_t_fmt);
    matmul::primitive_desc mm_bwd_dq_pd(
            eng, diff_score_md, k_t_md, grouped_query_md);
    matmul mm_bwd_dq(mm_bwd_dq_pd);
    mm_bwd_dq.execute(strm,
            {{DNNL_ARG_SRC, diff_score_mem},
                    {DNNL_ARG_WEIGHTS, key_dequantized},
                    {DNNL_ARG_DST, dQ_mem}});
    strm.wait();

    // backwards pass gradient of k (dK = q^t * dS)
    memory::desc q_t_md({q_sz[0], q_sz[1], q_sz[2], q_sz[4], q_sz[3]}, p.dt.dt,
            memory::format_tag::abced);

    // for GQA cases, perform additional reduction of dK along headq_group
    auto key_fmt = p.key_format_tag == memory::format_tag::abcd
            ? memory::format_tag::abcde
            : memory::format_tag::abced;
    memory::dims dK_full_dims = {k_sz[0], k_sz[1],
            is_gqa ? head_q_group_size : head_kv_group_size, k_sz[3], k_sz[4]};
    memory::desc dK_full_md(dK_full_dims, p.dt.dt, key_fmt);
    memory dK_full_mem = is_gqa ? memory(dK_full_md, eng) : dK_mem;
    matmul::primitive_desc mm_bwd_dk_pd(
            eng, q_t_md, diff_score_md, dK_full_mem.get_desc());
    matmul mm_bwd_dk(mm_bwd_dk_pd);
    mm_bwd_dk.execute(strm,
            {{DNNL_ARG_SRC, grouped_query}, {DNNL_ARG_WEIGHTS, diff_score_mem},
                    {DNNL_ARG_DST, dK_full_mem}});
    strm.wait();
    // reduce [mb, groups, hq_group, D, K] -> [mb, groups, 1, D, K] for GQA cases
    dnnl::reduction dK_reduce;
    if (is_gqa) {
        dnnl::reduction::primitive_desc dK_reduce_pd(eng,
                algorithm::reduction_sum, dK_full_md,
                key_dequantized.get_desc(), 0.f, 0.f);
        dK_reduce = dnnl::reduction(dK_reduce_pd);
        dK_reduce.execute(
                strm, {{DNNL_ARG_SRC, dK_full_mem}, {DNNL_ARG_DST, dK_mem}});
        strm.wait();
    }

    const auto bwd_loop = [&]() {
        strm.wait();
        mm_bwd_dS2.execute(strm, mm_bwd_dS2_args);
        strm.wait();
        mm_bwd_dV.execute(strm,
                {{DNNL_ARG_SRC, score2_downcast}, {DNNL_ARG_WEIGHTS, dO_mem},
                        {DNNL_ARG_DST, dV_full_mem}});
        if (is_gqa)
            dV_reduce.execute(strm,
                    {{DNNL_ARG_SRC, dV_full_mem}, {DNNL_ARG_DST, dV_mem}});

        // Apply dropout Jacobian to dP_raw in bwd_loop
        if (p.dropout.enabled()) {
            std::unordered_map<int, memory> dropout_bwd_args
                    = {{DNNL_ARG_SRC, diff_score2_mem},
                            {DNNL_ARG_DST, diff_score2_mem}};
            dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY]
                    = t.m_dropout_prob;
            dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0) {
                dropout_bwd_args[DNNL_ARG_ATTR_DROPOUT_OFFSET]
                        = t.m_dropout_offset;
            }
            dropout_eltwise.execute(strm, dropout_bwd_args);
        }

        softmax_bwd.execute(strm,
                {{DNNL_ARG_DST, score2}, {DNNL_ARG_DIFF_DST, diff_score2_mem},
                        {DNNL_ARG_DIFF_SRC, diff_score_f32_mem}});

        if (scale_dt != mdt::undef) {
            scale_bwd_prim.execute(strm,
                    {{DNNL_ARG_SRC_0, diff_score_f32_mem},
                            {DNNL_ARG_SRC_1, scale_f32_bwd},
                            {DNNL_ARG_DST, diff_score_f32_mem}});
        }

        diff_score_cast.execute(strm, diff_score_f32_mem, diff_score_mem);

        mm_bwd_dq.execute(strm,
                {{DNNL_ARG_SRC, diff_score_mem},
                        {DNNL_ARG_WEIGHTS, key_dequantized},
                        {DNNL_ARG_DST, dQ_mem}});
        mm_bwd_dk.execute(strm,
                {{DNNL_ARG_SRC, grouped_query},
                        {DNNL_ARG_WEIGHTS, diff_score_mem},
                        {DNNL_ARG_DST, dK_full_mem}});
        if (is_gqa)
            dK_reduce.execute(strm,
                    {{DNNL_ARG_SRC, dK_full_mem}, {DNNL_ARG_DST, dK_mem}});
        strm.wait();
    };
    bwd_loop();
    strm.wait();

    using namespace std::chrono;
    nanoseconds qtime_bwd {};

    if (with_timing) {
        auto loop_bwd_prim = [&] { bwd_loop(); };

        int iterations = 20;
        auto quantized_bwd_time = timeit(loop_bwd_prim, strm, iterations);

        auto min_time = [](const std::vector<nanoseconds> &a) {
            return *std::min_element(a.begin(), a.end());
        };

        qtime_bwd = min_time(quantized_bwd_time) / iterations;
        printf("qtime_training_backwards_prim %f\n", (float)qtime_bwd.count());
    }

    // print q,k gradients
#if DEBUG_PRINT_MEM
    print_mem(dQ_mem, "BWD dQ");
    print_mem(dK_mem, "BWD dK");
#endif

    // reshape gradients from grouped layout back to 4d
    void *diff_query_ptr = (void *)diff_query.map_data();
    void *dQ_ptr = (void *)dQ_mem.map_data();
    memcpy(diff_query_ptr, dQ_ptr, grouped_query_md.get_size());
    dQ_mem.unmap_data(dQ_ptr);
    diff_query.unmap_data(diff_query_ptr);

    void *diff_key_ptr = (void *)diff_key.map_data();
    void *dK_ptr = (void *)dK_mem.map_data();
    memcpy(diff_key_ptr, dK_ptr, key_dequantized.get_desc().get_size());
    dK_mem.unmap_data(dK_ptr);
    diff_key.unmap_data(diff_key_ptr);

    void *diff_value_ptr = (void *)diff_value.map_data();
    void *dV_ptr = (void *)dV_mem.map_data();
    memcpy(diff_value_ptr, dV_ptr, grouped_value_md.get_size());
    dV_mem.unmap_data(dV_ptr);
    diff_value.unmap_data(diff_value_ptr);

    return qtime_bwd;
}

// Upper bound on max|V| after dequantization. Fill ranges: codes and
// zero-points [0,6] unsigned / [-4,4] signed, scales [-2,2].
static float forward_value_magnitude(const sdpa_dims_t &p) {
    const bool quantized
            = is_quantized(p.value.dt, p.value.sdt, p.value.zpdt, p.qtype);
    if (!quantized) return 4.0f;
    // max|V| <= (|code|max + |zp|max) * |scale|max.
    const bool code_unsigned = (p.value.dt == mdt::u8 || p.value.dt == mdt::u4);
    const bool zp_unsigned
            = !(p.value.zpdt == mdt::s8 || p.value.zpdt == mdt::s4);
    const float code_max = code_unsigned ? 6.0f : 4.0f;
    const float zp_max = zp_unsigned ? 6.0f : 4.0f;
    return (code_max + zp_max) * 2.0f; // 16 signed, 20 mixed, 24 unsigned
}

template <typename T>
void check_memory(dnnl::stream &strm, memory &gold, memory &test,
        float max_diff_threshold = 0.03f, float fthreshold = 0.001466,
        float max_abs_value = 1.0f, memory *ref_magnitude = nullptr) {
    T *mapped_ptr_gold = nullptr;
    T *mapped_ptr_test = nullptr;
    mapped_ptr_gold = (T *)gold.map_data();
    mapped_ptr_test = (T *)test.map_data();
    strm.wait();

    auto dims = gold.get_desc().get_dims();
    auto strides = gold.get_desc().get_strides();

    int mismatches = 0;
    int total = 0;

    float max_diff = std::numeric_limits<float>::min();
    std::map<int, std::map<int, int>> hist;
    const bool verbose = false;

    // Catch systematic errors the per-element floor misses.
    double sumsq_diff = 0.0, sumsq_gold = 0.0, sum_signed_diff = 0.0;

    // Error is bounded by fthreshold * the conditioning magnitude, independent
    // of |O|: a relative gate explodes when cancellation drives |O| -> 0.
    float *mapped_ptr_mag
            = ref_magnitude ? (float *)ref_magnitude->map_data() : nullptr;

    // Keeps the per-element budget from underflowing where mag is tiny.
    const float acc_floor = 64.0f * 1.1920929e-7f * max_abs_value; // 64*eps_f32

    // Sizes the max_diff cap when only a flat magnitude is available.
    float max_abs_gold = 0.f;

    for_(int l = 0; l < dims[0]; l++)
    for_(int k = 0; k < dims[1]; k++)
    for_(int j = 0; j < dims[2]; j++)
    for (int i = 0; i < dims[3]; i++) {
        auto offset = l * strides[0] + k * strides[1] + j * strides[2]
                + i * strides[3];
        auto o_gold = (float)mapped_ptr_gold[offset];
        auto o_test = (float)mapped_ptr_test[offset];
        total++;

        auto min_val = fmin(o_gold, o_test);
        auto max_val = fmax(o_gold, o_test);
        float abs_diff = abs(max_val - min_val);
        bool is_nan = isnan(o_gold) || isnan(o_test);

        if (!is_nan) {
            const double d = (double)o_test - (double)o_gold;
            sumsq_diff += d * d;
            sumsq_gold += (double)o_gold * (double)o_gold;
            sum_signed_diff += d;
        }

        max_abs_gold = fmax(max_abs_gold, fabsf(o_gold));

        // Per-element conditioning magnitude when available, else flat bound.
        const float mag
                = mapped_ptr_mag ? mapped_ptr_mag[offset] : max_abs_value;
        const float atol
                = std::min(fthreshold * mag + acc_floor, max_diff_threshold);
        // Relative arm, ORed with the floor (as benchdnn does). Only needed
        // for a flat mag; per-element absmag >= |gold| makes it redundant.
        const float rtol = mapped_ptr_mag ? 0.f : fthreshold * fabsf(o_gold);
        const float elem_threshold = fmax(atol, rtol);
        bool is_mismatch = is_nan || (abs_diff > elem_threshold);
        if (max_diff < abs_diff) {
            if (verbose) {
                printf("new max(%d,%d,%d,%d): test: %f vs gold: %f "
                       "(diff: %f thresh: %f mag: %f)\n",
                        l, k, j, i, o_test, o_gold, abs_diff, elem_threshold,
                        mag);
            }
            max_diff = abs_diff;
        }
        if (is_mismatch) {
            hist[0][l]++;
            hist[1][k]++;
            hist[2][j]++;
            hist[3][i]++;
        }
        if (is_mismatch && mismatches++ < 32) {
            if (verbose)
                printf("Mismatch at (%d,%d,%d,%d): test %f "
                       "vs. gold %f (diff: %f thresh: %f mag: %f)\n",
                        l, k, j, i, o_test, o_gold, abs_diff, elem_threshold,
                        mag);
        }
    }

    gold.unmap_data(mapped_ptr_gold);
    test.unmap_data(mapped_ptr_test);
    if (mapped_ptr_mag) ref_magnitude->unmap_data(mapped_ptr_mag);

    // Sized as multiples of the per-element budget so they stay dtype-aware.
    const double l2_rel = sumsq_gold > 0.0 ? std::sqrt(sumsq_diff / sumsq_gold)
                                           : std::sqrt(sumsq_diff);
    const double bias = total > 0 ? sum_signed_diff / total : 0.0;
    const double l2_threshold = 8.0 * fthreshold;
    const double bias_threshold = 0.5 * fthreshold * max_abs_value;

    // A flat magnitude cannot bound the largest element: one f16 ulp above
    // 1024 is 1.0, vs a 0.3 bwd cap.
    const float max_diff_cap = ref_magnitude
            ? max_diff_threshold
            : fmax(max_diff_threshold, fthreshold * max_abs_gold);

    ASSERT_EQ(mismatches, 0) << mismatches << " out of: " << total;
    ASSERT_LE(max_diff, max_diff_cap);
    ASSERT_LE(l2_rel, l2_threshold) << "L2 rel error " << l2_rel;
    ASSERT_LE(std::abs(bias), bias_threshold) << "systematic bias " << bias;
}

template <typename T>
void check_dropout_mask_memory(dnnl::stream &strm, memory &gold, memory &test) {
    T *mapped_ptr_gold = (T *)gold.map_data();
    T *mapped_ptr_test = (T *)test.map_data();
    strm.wait();

    auto dims = gold.get_desc().get_dims();
    auto gold_strides = gold.get_desc().get_strides();
    auto test_dims = test.get_desc().get_dims();
    auto test_strides = test.get_desc().get_strides();

    ASSERT_EQ(dims, test_dims)
            << "[DROPOUT_MASK] Shape mismatch: reference and tested must have "
               "the same dimensions before comparison";

    std::vector<int64_t> index(dims.size(), 0);
    int mismatches = 0;
    int total = 0;

    while (true) {
        int64_t gold_offset = 0;
        int64_t test_offset = 0;
        for (size_t d = 0; d < dims.size(); d++) {
            gold_offset += index[d] * gold_strides[d];
            test_offset += index[d] * test_strides[d];
        }

        total++;
        if (mapped_ptr_gold[gold_offset] != mapped_ptr_test[test_offset]) {
            mismatches++;
        }

        int dim = (int)dims.size() - 1;
        while (dim >= 0) {
            index[dim]++;
            if (index[dim] < dims[dim]) break;
            index[dim] = 0;
            dim--;
        }
        if (dim < 0) break;
    }

    gold.unmap_data(mapped_ptr_gold);
    test.unmap_data(mapped_ptr_test);

    ASSERT_EQ(mismatches, 0)
            << "[DROPOUT_MASK] " << mismatches << " out of: " << total;
}

int to_attn_mask_type(mask_type t) {
    using namespace dnnl::impl::attn_mask_type;
    auto attn_mask = buffer;
    switch (t) {
        case mask_type::causal_tl: attn_mask = top_left; break;
        case mask_type::causal_br: attn_mask = bottom_right; break;
        default:;
    }
    return static_cast<int>(attn_mask);
}

template <typename O, typename I>
O magnitude_cast(I input) {
    using ratio = std::ratio_divide<typename I::ratio, typename O::ratio>;
    return input.value * ratio::num / ratio::den;
}

template <class Unit = std::ratio<1>>
class byte_t {
public:
    using ratio = Unit;
    float value;
    byte_t(float v) : value(v) {}

    byte_t(memory::data_type dt)
        : value(dnnl_data_type_size((dnnl_data_type_t)dt)
                  / ((dt == mdt::s4 || dt == mdt::u4) ? 2.f : 1.f)) {}

    template <typename OR>
    byte_t(byte_t<OR> o) : value(magnitude_cast<Unit>(o).value) {}

    operator float() { return value; }
};

template <class Unit = std::ratio<1>>
class num_ops_t {
public:
    using ratio = Unit;
    float value;
    num_ops_t(float v) : value(v) {}

    template <typename OR>
    num_ops_t(num_ops_t<OR> o) : value(magnitude_cast<Unit>(o).value) {}

    operator float() { return value; }
};

using kilobyte = byte_t<std::ratio<1024>>;
using megabyte = byte_t<std::ratio<1024 * 1024, 1>>;
using gigabyte = byte_t<std::ratio<1024 * 1024 * 1024, 1>>;

using kiloops = num_ops_t<std::ratio<1000>>;
using megaops = num_ops_t<std::ratio<1000 * 1000, 1>>;
using gigaops = num_ops_t<std::ratio<1000 * 1000 * 1000, 1>>;

template <typename BYTES, typename TIME>
float bandwidth(BYTES bytes, TIME duration) {
    return (bytes.value
            / std::chrono::duration_cast<std::chrono::duration<float>>(duration)
                      .count());
}

template <typename OPS, typename TIME>
float compute(OPS ops, TIME duration) {
    return (ops.value
            / std::chrono::duration_cast<std::chrono::duration<float>>(duration)
                      .count());
}

static std::once_flag header_flag;

template <typename T>
class sdpa_test_t : public ::testing::TestWithParam<T> {
public:
    // Testing reusable functionality requires shared engine between tests.
    static void SetUpTestSuite() {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "SDPA primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "SDPA primitive tests do not support HIP";
#endif
#ifndef DNNL_TEST_WITH_ENGINE_PARAM
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "SDPA tests require gpus.");
        sdpa_eng.reset(new dnnl::engine(engine::kind::gpu, 0));
#endif
    }

    void SetUp() override {
#ifdef DNNL_SYCL_CUDA
        GTEST_SKIP() << "SDPA primitive tests do not support CUDA";
#endif
#ifdef DNNL_SYCL_HIP
        GTEST_SKIP() << "SDPA primitive tests do not support HIP";
#endif
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
        SKIP_IF(get_test_engine_kind() != dnnl::engine::kind::gpu,
                "This test requires GPU engine");
        eng = get_test_engine();
#else
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "SDPA tests require gpus.");
        eng = get_sdpa_test_engine();
#endif
        strm = dnnl::stream(eng);
        p = sdpa_test_t<T>::GetParam();
        doubled_memory.reserve(30);
        t = get_descriptors(eng, strm, p, doubled_memory);
        scale_dt = t.m_query.get_desc().get_data_type();
    }

    void TearDown() override {
        for (dnnl_memory_t &mem : doubled_memory) {
            CHECK(dnnl_memory_destroy(mem));
        }
    }

    static void TearDownTestSuite() {
#ifndef DNNL_TEST_WITH_ENGINE_PARAM
        sdpa_eng.reset();
#endif
    }

    void compare() {
        using namespace dnnl::impl;
        auto mask = t.m_mask.get_desc();
        memory ref_dropout_mask;
        memory test_dropout_mask;

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
            case mask_type::no_mask:
            case mask_type::causal_tl:
            case mask_type::causal_br: mask_ptr = nullptr; break;
            case mask_type::oneD:
            case mask_type::twoD: mask_ptr = &mask; break;
        }

        sdpa::primitive_desc sdpa_quantized_pd;
        sdpa sdpa_quantized_p;
        try {
            sdpa_quantized_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    invert_scale, p.heads.kv, to_attn_mask_type(p.mask.type),
                    dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                    prop_kind::forward_inference, t.sdpa_attr_quantized,
                    t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);
            sdpa_quantized_p = sdpa(sdpa_quantized_pd);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented)
                GTEST_SKIP() << "Unimplemented: " << e.what();
            else
                throw;
        }

        std::unordered_map<int, memory> sdpa_args
                = {{{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized}}};

        sdpa_args[DNNL_ARG_KEYS] = t.m_key_quantized;

        if (scale_dt != mdt::undef) { sdpa_args[DNNL_ARG_SCALE] = t.m_scale; }

        bool k_is_float = ((p.key.dt == mdt::f16) || (p.key.dt == mdt::bf16));
        bool v_is_float
                = ((p.value.dt == mdt::f16) || (p.value.dt == mdt::bf16));
        if (!k_is_float && p.qtype != quantize_type::no_quantization) {
            if (p.key.sdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS]
                        = t.m_key_scales;
            if (p.key.zpdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS]
                        = t.m_key_zp;
        }
        if (!v_is_float && p.qtype != quantize_type::no_quantization) {
            if (p.value.sdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES]
                        = t.m_value_scales;
            if (p.value.zpdt != mdt::undef)
                sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
                        = t.m_value_zp;
        }
        if (mask_ptr) { sdpa_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }
        if (p.dropout.enabled()) {
            sdpa_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            sdpa_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0)
                sdpa_args[DNNL_ARG_ATTR_DROPOUT_OFFSET] = t.m_dropout_offset;
            // Query mask md from pd (benchdnn pattern); only when has_output_mask.
            if (p.dropout.has_output_mask()) {
                auto dropout_mask_md = sdpa_quantized_pd.query_md(
                        dnnl::query::exec_arg_md, DNNL_ARG_ATTR_DROPOUT_MASK);
                test_dropout_mask = memory(dropout_mask_md, eng);
                sdpa_args[DNNL_ARG_ATTR_DROPOUT_MASK] = test_dropout_mask;
            }
        }

        sdpa_quantized_p.execute(strm, sdpa_args);

        prim_sdpa_quant(p, t, eng, strm, t.m_query, t.m_key_quantized,
                t.m_key_scales, t.m_key_zp, scale_dt, t.m_scale_prim, t.m_mask,
                t.m_value_quantized, t.m_value_scales, t.m_value_zp, t.m_output,
                invert_scale,
                p.dropout.has_output_mask() ? &ref_dropout_mask : nullptr,
                doubled_memory, &t.m_output_absmag);

#if 0
    if (::getenv("SKIP_CHECK")) return;
#endif
        float max_diff_threshold = 0.03f;
        // bf16 uses its unit roundoff, doubled for the extra s4 dequant path.
        // The f16/f32 value is empirical, deliberately not eps-derived.
        constexpr float eps_bf16 = 0.0078125f; // 2^-7, bf16 unit roundoff
        float fthreshold = 0.f;
        if (p.dt.dt == mdt::bf16) {
            const bool kv_s4 = (p.key.dt == mdt::s4 || p.value.dt == mdt::s4);
            fthreshold = kv_s4 ? 2.f * eps_bf16 : eps_bf16;
        } else {
            fthreshold = 0.001466f;
        }

        // f16 accumulation carries ~bf16 precision regardless of I/O dtype.
        if (p.acc_modes.kq_acc == dnnl::accumulation_mode::f16
                || p.acc_modes.vs_acc == dnnl::accumulation_mode::f16) {
            fthreshold = eps_bf16;
        }

        if (p.key.dt == mdt::s4 || p.value.dt == mdt::s4) {
            max_diff_threshold = 0.063f;
        }
        // m_output_absmag bounds the error for all forward cases, quantized
        // included. value_mag is the flat fallback and sizes the bias bound.
        float value_mag = forward_value_magnitude(p);
        if (p.dropout.enabled()) value_mag /= (1.0f - p.dropout.probability);
        memory *ref_mag = &t.m_output_absmag;
        if (t.m_output.get_desc().get_data_type() == mdt::f16)
            check_memory<float16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag, ref_mag);
        else if (t.m_output.get_desc().get_data_type() == mdt::bf16)
            check_memory<bfloat16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag, ref_mag);
        else if (t.m_output.get_desc().get_data_type() == mdt::f32)
            check_memory<float_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag, ref_mag);

        if (p.dropout.enabled() && p.dropout.has_output_mask()) {
            // The reference path (prim_sdpa_quant) reshapes Q/K/V into a 5D
            // grouped layout {mb, N_kv, N_q/N_kv, seq_q, seq_kv} to handle
            // GQA batching, so the softmax dropout mask it produces is also 5D.
            // The SDPA primitive outputs the mask in the score shape derived
            // from the original 4D query: {mb, N_q, seq_q, seq_kv}.
            // Both have the same element count; reshape the reference mask to
            // the 4D score shape so the comparison uses exact same-shape indexing.
            const memory::dims score_sz_4d
                    = {p.mb, p.heads.q, p.seq_len.q, p.seq_len.kv};
            auto ref_mask_4d = reshape(strm, ref_dropout_mask,
                    memory::desc(
                            score_sz_4d, mdt::u8, memory::format_tag::abcd));
            check_dropout_mask_memory<unsigned char>(
                    strm, ref_mask_4d, test_dropout_mask);
        }

#if 0
    for (auto &kv : hist) {
        for (auto &kv2 : kv.second) {
            printf("hist[%d][%d] = %d\n", kv.first, kv2.first, kv2.second);
        }
    }
#endif
    }

    void compare_bwd() {
        using namespace dnnl::impl;

        auto mask = t.m_mask.get_desc();

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
            case mask_type::no_mask:
            case mask_type::causal_tl:
            case mask_type::causal_br: mask_ptr = nullptr; break;
            case mask_type::oneD:
            case mask_type::twoD: mask_ptr = &mask; break;
        }

        auto dS_desc
                = memory::desc({p.mb, p.heads.q, p.seq_len.q, p.seq_len.kv},
                        p.dt.dt, memory::format_tag::abcd);
        memory::desc *dS_ptr = nullptr;
        //dS_ptr = &dS_desc; // uncomment for optional dS output
        if (dS_ptr)
            t.m_dS = double_and_resize(*dS_ptr, eng, strm, doubled_memory);

        // fwd sdpa primitive to populate dst, col_maxes
        sdpa::primitive_desc sdpa_fwd_pd;
        sdpa sdpa_fwd;

        sdpa_backward::primitive_desc sdpa_bwd_pd;
        sdpa_backward sdpa_bwd;
        try {
            sdpa_fwd_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    invert_scale, p.heads.kv, to_attn_mask_type(p.mask.type),
                    dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                    prop_kind::forward_training, t.sdpa_attr_quantized,
                    t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);

            // create bwd pd before compiling fwd so unsupported cases
            // skip without paying for a fwd kernel build
            sdpa_bwd_pd = sdpa_backward::primitive_desc(eng,
                    t.m_query.get_desc(), t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    t.m_diff_query_quantized.get_desc(),
                    t.m_diff_key_quantized.get_desc(),
                    t.m_diff_value_quantized.get_desc(),
                    t.m_diff_output.get_desc(), dS_ptr, invert_scale,
                    p.heads.kv, to_attn_mask_type(p.mask.type),
                    dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                    sdpa_fwd_pd, t.sdpa_attr_quantized);
            sdpa_fwd = sdpa(sdpa_fwd_pd);
            sdpa_bwd = sdpa_backward(sdpa_bwd_pd);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented)
                GTEST_SKIP() << "Unimplemented: " << e.what();
            else
                throw;
        }

        auto sdpa_fwd_workspace_memory
                = memory(sdpa_fwd_pd.workspace_desc(), eng);

        std::unordered_map<int, memory> sdpa_fwd_args
                = {{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_KEYS, t.m_key_quantized},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized},
                        {DNNL_ARG_WORKSPACE, sdpa_fwd_workspace_memory}};

        if (scale_dt != mdt::undef) {
            sdpa_fwd_args[DNNL_ARG_SCALE] = t.m_scale;
        }
        if (mask_ptr) { sdpa_fwd_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }
        if (p.dropout.enabled()) {
            sdpa_fwd_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            sdpa_fwd_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0) {
                sdpa_fwd_args[DNNL_ARG_ATTR_DROPOUT_OFFSET]
                        = t.m_dropout_offset;
            }
        }

        strm.wait();
        sdpa_fwd.execute(strm, sdpa_fwd_args);
        strm.wait();

        std::unordered_map<int, memory> sdpa_bwd_args
                = {{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_KEYS, t.m_key_quantized},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized},
                        {DNNL_ARG_DIFF_DST, t.m_diff_output},
                        {DNNL_ARG_DIFF_QUERIES, t.m_diff_query_quantized},
                        {DNNL_ARG_DIFF_KEYS, t.m_diff_key_quantized},
                        {DNNL_ARG_DIFF_VALUES, t.m_diff_value_quantized},
                        {DNNL_ARG_WORKSPACE, sdpa_fwd_workspace_memory}};
        if (dS_ptr) sdpa_bwd_args[DNNL_ARG_DS] = t.m_dS;

#if DEBUG_PRINT_MEM
        print_mem(t.m_query, "input t_m_query");
        print_mem(t.m_key_quantized, "input t_m_key");
        print_mem(t.m_value_quantized, "input t_m_value");
        print_mem(t.m_diff_output, "input dA");
#endif

        if (scale_dt != mdt::undef) {
            sdpa_bwd_args[DNNL_ARG_SCALE] = t.m_scale;
        }
        if (mask_ptr) { sdpa_bwd_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }
        if (p.dropout.enabled()) {
            sdpa_bwd_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            sdpa_bwd_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0) {
                sdpa_bwd_args[DNNL_ARG_ATTR_DROPOUT_OFFSET]
                        = t.m_dropout_offset;
            }
        }

        sdpa_bwd.execute(strm, sdpa_bwd_args);
        strm.wait();

#if DEBUG_PRINT_MEM
        if (dS_ptr) print_mem(t.m_dS, "computed dS");
        print_mem(t.m_diff_value_quantized, "dV bwd out");
        printf("--------------  Primitives based implementation -------------- "
               "\n");
#endif

        // perform primitives based backwards sdpa pass to generate "gold" gradient outputs
        prim_sdpa_quant_bwd(p, t, eng, strm, t.m_query, t.m_key_quantized,
                scale_dt, t.m_scale, t.m_mask, t.m_value_quantized, t.m_output,
                t.m_diff_output, invert_scale, doubled_memory, t.m_diff_query,
                t.m_diff_key, t.m_diff_value,
                /*with_timing=*/false);

        float max_diff_threshold = 0.3f;
        // backward thresholds are higher than forward due to chained matmuls
        // softmax backward potential catastrophic cancellation S*(dP - Di)
        // and atomic adds across dQ accumulation
        float fthreshold = 0.f;
        if (p.dt.dt == mdt::bf16) {
            fthreshold = 0.1f;
        } else if (p.dt.dt == mdt::f16) {
            fthreshold = 0.035f;
        } else {
            fthreshold = 0.002f;
        }

        strm.wait();

        // GQA atomic adds across kv_group_size Q-heads introduce
        // nondeterministic rounding in dK/dV. Relax threshold by sqrt(group)
        const int kv_group_size = static_cast<int>(p.heads.q / p.heads.kv);
        const float gqa_fthreshold
                = fthreshold * std::sqrt(static_cast<float>(kv_group_size));

        // Operand-scale proxies: dV ~ P^T*dO -> max|dO|, the rest -> max|V|.
        // The relative arm in check_memory is the primary gate.
        float value_mag = forward_value_magnitude(p);
        float dO_mag = 4.0f;
        // Dropout scales surviving probabilities by 1/(1-p).
        if (p.dropout.enabled()) {
            const float inv_keep = 1.0f / (1.0f - p.dropout.probability);
            value_mag *= inv_keep;
            dO_mag *= inv_keep;
        }

        if (t.m_output.get_desc().get_data_type() == mdt::f16) {
            check_memory<float16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag);
            check_memory<float16_t>(strm, t.m_diff_query,
                    t.m_diff_query_quantized, max_diff_threshold, fthreshold,
                    value_mag);
            check_memory<float16_t>(strm, t.m_diff_key, t.m_diff_key_quantized,
                    max_diff_threshold, gqa_fthreshold, value_mag);
            check_memory<float16_t>(strm, t.m_diff_value,
                    t.m_diff_value_quantized, max_diff_threshold,
                    gqa_fthreshold, dO_mag);

        } else if (t.m_output.get_desc().get_data_type() == mdt::bf16) {
            check_memory<bfloat16_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag);
            check_memory<bfloat16_t>(strm, t.m_diff_query,
                    t.m_diff_query_quantized, max_diff_threshold, fthreshold,
                    value_mag);
            check_memory<bfloat16_t>(strm, t.m_diff_key, t.m_diff_key_quantized,
                    max_diff_threshold, gqa_fthreshold, value_mag);
            check_memory<bfloat16_t>(strm, t.m_diff_value,
                    t.m_diff_value_quantized, max_diff_threshold,
                    gqa_fthreshold, dO_mag);

        } else if (t.m_output.get_desc().get_data_type() == mdt::f32) {
            check_memory<float_t>(strm, t.m_output, t.m_output_quantized,
                    max_diff_threshold, fthreshold, value_mag);
            check_memory<float_t>(strm, t.m_diff_query,
                    t.m_diff_query_quantized, max_diff_threshold, fthreshold,
                    value_mag);
            check_memory<float_t>(strm, t.m_diff_key, t.m_diff_key_quantized,
                    max_diff_threshold, gqa_fthreshold, value_mag);
            check_memory<float_t>(strm, t.m_diff_value,
                    t.m_diff_value_quantized, max_diff_threshold,
                    gqa_fthreshold, dO_mag);
        }

#if DEBUG_PRINT_MEM
        print_mem(t.m_output, "gold m_output");
        print_mem(t.m_output_quantized, "test m_output");
        printf("----------\n\n");

        print_mem(t.m_diff_query, "gold m_diff_query");
        print_mem(t.m_diff_query_quantized, "test m_diff_query");
        printf("----------\n\n");

        print_mem(t.m_diff_key, "gold m_diff_key");
        print_mem(t.m_diff_key_quantized, "test m_diff_key");
        printf("----------\n\n");

        print_mem(t.m_diff_value, "gold m_diff_value");
        print_mem(t.m_diff_value_quantized, "test m_diff_value");
        printf("----------\n\n");
#endif
    }

    void perf() {
        using namespace dnnl::impl;
        auto mask = t.m_mask.get_desc();

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
            case mask_type::no_mask:
            case mask_type::causal_tl:
            case mask_type::causal_br: mask_ptr = nullptr; break;
            case mask_type::oneD:
            case mask_type::twoD: mask_ptr = &mask; break;
        }

        sdpa::primitive_desc sdpa_quantized_pd;
        sdpa sdpa_quantized_p;
        try {
            sdpa_quantized_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    invert_scale, p.heads.kv, to_attn_mask_type(p.mask.type),
                    alg_kind::softmax_accurate_inf_as_zero,
                    prop_kind::forward_inference, t.sdpa_attr_quantized,
                    t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);
            sdpa_quantized_p = sdpa(sdpa_quantized_pd);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented)
                GTEST_SKIP() << "Unimplemented: " << e.what();
            else
                throw;
        }

        std::unordered_map<int, memory> sdpa_args
                = {{{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized}}};

        sdpa_args[DNNL_ARG_KEYS] = t.m_key_quantized;
        if (scale_dt != mdt::undef) { sdpa_args[DNNL_ARG_SCALE] = t.m_scale; }

        if (p.key.dt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
            sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
            sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
        }
        if (p.value.dt != mdt::f16
                && p.qtype != quantize_type::no_quantization) {
            sdpa_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES]
                    = t.m_value_scales;
            sdpa_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES]
                    = t.m_value_zp;
        }
        if (mask_ptr) { sdpa_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }
        if (p.dropout.enabled()) {
            sdpa_args[DNNL_ARG_ATTR_DROPOUT_PROBABILITY] = t.m_dropout_prob;
            sdpa_args[DNNL_ARG_ATTR_DROPOUT_SEED] = t.m_dropout_seed;
            if (p.dropout.offset != 0)
                sdpa_args[DNNL_ARG_ATTR_DROPOUT_OFFSET] = t.m_dropout_offset;
            if (p.dropout.has_output_mask()) {
                auto dropout_mask_md = sdpa_quantized_pd.query_md(
                        dnnl::query::exec_arg_md, DNNL_ARG_ATTR_DROPOUT_MASK);
                sdpa_args[DNNL_ARG_ATTR_DROPOUT_MASK]
                        = memory(dropout_mask_md, eng);
            }
        }

        auto loop_quantized
                = [&] { sdpa_quantized_p.execute(strm, sdpa_args); };

        int iterations = 20;
        auto quantized_time = timeit(loop_quantized, strm, iterations);

        using namespace std::chrono;
        auto min_time = [](const std::vector<nanoseconds> &a) {
            return *std::min_element(a.begin(), a.end());
        };

        auto qtime = min_time(quantized_time) / iterations;

        // total number of bytes of all tensors
        byte_t<> total_bytes = t.m_query.get_desc().get_size()

                + t.m_key_quantized.get_desc().get_size() / 2
                + t.m_key_scales.get_desc().get_size()
                + t.m_key_zp.get_desc().get_size()

                + t.m_value_quantized.get_desc().get_size() / 2
                + t.m_value_scales.get_desc().get_size()
                + t.m_value_zp.get_desc().get_size()

                + t.m_output.get_desc().get_size()
                + (mask_ptr ? t.m_mask.get_desc().get_size() : 0);

        auto mask_slice_elements = 0;
        switch (p.mask.type) {
            case mask_type::twoD:
                mask_slice_elements = p.seq_len.kv * p.seq_len.q;
                break;
            case mask_type::oneD: mask_slice_elements = p.seq_len.kv; break;
            default: mask_slice_elements = 0; break;
        }

        size_t kv_slice_tensor_elements
                = (p.head_group.head_size * p.seq_len.kv);
        size_t batch_elements = p.mb * std::max(p.heads.q, p.heads.kv);

        // Total number of bytes read by the micro_sdpa kernel. This calculation
        // is different from total_bytes because it expands tensors like masks
        // to match the batches of kvq tensors. Typically this is bigger than
        // total bytes.
        byte_t<> total_bytes_effective
                = (batch_elements
                          * (byte_t<>(p.key.dt) * kv_slice_tensor_elements
                                  + byte_t<>(p.value.dt)
                                          * kv_slice_tensor_elements
                                  + byte_t<>(p.dt.dt)
                                          * (2 * p.head_group.head_size
                                                  * p.seq_len.q)
                                  + (mask_ptr ? byte_t<>(p.mask.dt)
                                                          * mask_slice_elements
                                              : 0)))
                + t.m_key_scales.get_desc().get_size()
                + t.m_key_zp.get_desc().get_size()
                + t.m_value_scales.get_desc().get_size()
                + t.m_value_zp.get_desc().get_size();

        // All flops even for causal mask cases
        num_ops_t<> total_flops = std::max<size_t>(p.heads.kv, p.heads.q) * p.mb
                * (2.f
                                * (2.f * p.head_group.head_size * p.seq_len.kv
                                        * p.seq_len.q)
                        + (scale_dt != mdt::undef ? (p.seq_len.kv * p.seq_len.q)
                                                  : 0)
                        + (p.mask.type != mask_type::no_mask
                                        ? (p.seq_len.kv * p.seq_len.q)
                                        : 0)
                        + (5 * p.seq_len.kv * p.seq_len.q));

        // Ignores softmax/mask/scale and does not count masked out values in causal mask cases
        num_ops_t<> flash_flops
                = (4.f * p.mb * p.heads.q * p.seq_len.kv * p.seq_len.q
                          * p.head_group.head_size)
                / ((p.mask.type == mask_type::causal_tl
                           || p.mask.type == mask_type::causal_br)
                                ? 2.f
                                : 1.f);

        std::call_once(header_flag, print_table_header);
        std::cout << print_row(p) << "|" << qtime.count() << "|"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes_effective),
                             qtime)
                  << "/"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes), qtime)
                  << "|" << compute(magnitude_cast<gigaops>(flash_flops), qtime)
                  << "/" << compute(magnitude_cast<gigaops>(total_flops), qtime)
                  << "|" << std::endl;
    }

    void perf_bwd(bool time_reference = false) {
        using namespace dnnl::impl;
        auto mask = t.m_mask.get_desc();

        memory::desc *mask_ptr = nullptr;

        switch (p.mask.type) {
            case mask_type::no_mask:
            case mask_type::causal_tl:
            case mask_type::causal_br: mask_ptr = nullptr; break;
            case mask_type::oneD:
            case mask_type::twoD: mask_ptr = &mask; break;
        }

        // Forward training pass (needed for workspace)
        sdpa::primitive_desc sdpa_fwd_pd;
        sdpa sdpa_fwd;

        sdpa_backward::primitive_desc sdpa_bwd_pd;
        sdpa_backward sdpa_bwd;
        try {
            sdpa_fwd_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
                    t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    invert_scale, p.heads.kv, to_attn_mask_type(p.mask.type),
                    dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                    prop_kind::forward_training, t.sdpa_attr_quantized,
                    t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);

            sdpa_bwd_pd = sdpa_backward::primitive_desc(eng,
                    t.m_query.get_desc(), t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr,
                    t.m_scale.get_desc(), t.m_output_quantized.get_desc(),
                    t.m_diff_query_quantized.get_desc(),
                    t.m_diff_key_quantized.get_desc(),
                    t.m_diff_value_quantized.get_desc(),
                    t.m_diff_output.get_desc(), nullptr, invert_scale,
                    p.heads.kv, to_attn_mask_type(p.mask.type),
                    dnnl::impl::alg_kind::softmax_accurate_inf_as_zero,
                    sdpa_fwd_pd, t.sdpa_attr_quantized);
            sdpa_fwd = sdpa(sdpa_fwd_pd);
            sdpa_bwd = sdpa_backward(sdpa_bwd_pd);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented)
                GTEST_SKIP() << "Unimplemented: " << e.what();
            else
                throw;
        }

        // Execute forward pass to populate workspace
        auto sdpa_fwd_workspace_memory
                = memory(sdpa_fwd_pd.workspace_desc(), eng);

        std::unordered_map<int, memory> sdpa_fwd_args
                = {{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_KEYS, t.m_key_quantized},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized},
                        {DNNL_ARG_WORKSPACE, sdpa_fwd_workspace_memory}};

        if (scale_dt != mdt::undef) {
            sdpa_fwd_args[DNNL_ARG_SCALE] = t.m_scale;
        }
        if (mask_ptr) { sdpa_fwd_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

        sdpa_fwd.execute(strm, sdpa_fwd_args);
        strm.wait();

        // Build backward args
        std::unordered_map<int, memory> sdpa_bwd_args
                = {{DNNL_ARG_QUERIES, t.m_query},
                        {DNNL_ARG_KEYS, t.m_key_quantized},
                        {DNNL_ARG_VALUES, t.m_value_quantized},
                        {DNNL_ARG_DST, t.m_output_quantized},
                        {DNNL_ARG_DIFF_DST, t.m_diff_output},
                        {DNNL_ARG_DIFF_QUERIES, t.m_diff_query_quantized},
                        {DNNL_ARG_DIFF_KEYS, t.m_diff_key_quantized},
                        {DNNL_ARG_DIFF_VALUES, t.m_diff_value_quantized},
                        {DNNL_ARG_WORKSPACE, sdpa_fwd_workspace_memory}};

        if (scale_dt != mdt::undef) {
            sdpa_bwd_args[DNNL_ARG_SCALE] = t.m_scale;
        }
        if (mask_ptr) { sdpa_bwd_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

        // Time backward pass
        auto loop_bwd = [&] { sdpa_bwd.execute(strm, sdpa_bwd_args); };

        int iterations = 20;
        auto bwd_time = timeit(loop_bwd, strm, iterations);

        using namespace std::chrono;
        auto min_time = [](const std::vector<nanoseconds> &a) {
            return *std::min_element(a.begin(), a.end());
        };

        auto qtime = min_time(bwd_time) / iterations;
        printf("qtimebwd %f\n", (float)qtime.count() / 1e6);

        // Backward reads: Q, K, V, O, dO, workspace(logsumexp)
        // Backward writes: dQ, dK, dV
        byte_t<> total_bytes = t.m_query.get_desc().get_size()
                + t.m_key_quantized.get_desc().get_size()
                + t.m_value_quantized.get_desc().get_size()
                + t.m_output_quantized.get_desc().get_size()
                + t.m_diff_output.get_desc().get_size()
                + t.m_diff_query_quantized.get_desc().get_size()
                + t.m_diff_key_quantized.get_desc().get_size()
                + t.m_diff_value_quantized.get_desc().get_size()
                + (mask_ptr ? t.m_mask.get_desc().get_size() : 0);

        size_t kv_slice_tensor_elements
                = (p.head_group.head_size * p.seq_len.kv);
        auto mask_slice_elements = 0;
        switch (p.mask.type) {
            case mask_type::twoD:
                mask_slice_elements = p.seq_len.kv * p.seq_len.q;
                break;
            case mask_type::oneD: mask_slice_elements = p.seq_len.kv; break;
            default: mask_slice_elements = 0; break;
        }
        size_t batch_elements = p.mb * std::max(p.heads.q, p.heads.kv);

        // Effective bytes: expand broadcast tensors to full batch size
        // Reads: Q, K, V, O, dO (all expanded); Writes: dQ, dK, dV
        byte_t<> total_bytes_effective = (batch_elements
                * (byte_t<>(p.key.dt) * kv_slice_tensor_elements
                        + byte_t<>(p.value.dt) * kv_slice_tensor_elements
                        + byte_t<>(p.dt.dt)
                                * (5 * p.head_group.head_size * p.seq_len.q)
                        + byte_t<>(p.dt.dt) * (2 * kv_slice_tensor_elements)
                        + (mask_ptr ? byte_t<>(p.mask.dt) * mask_slice_elements
                                    : 0)));

        // Backward has 5 matmuls: S=K*Q, dS2=V*dO, dV=S*dO, dK=dS*Q, dQ=dS*K
        // plus softmax_bwd, scale, Di reduction
        float causal_divisor = (p.mask.type == mask_type::causal_tl
                                       || p.mask.type == mask_type::causal_br)
                ? 2.f
                : 1.f;

        // 5 matmuls, flash-attention paper scales FWD flops by 5/2
        // softmax_bwd (~5*K*Q), scale, Di
        num_ops_t<> total_flops = std::max<size_t>(p.heads.kv, p.heads.q) * p.mb
                * (2.f
                                * (2.5f * 2.f * p.head_group.head_size
                                        * p.seq_len.kv * p.seq_len.q)
                        + (scale_dt != mdt::undef
                                        ? (2 * p.seq_len.kv * p.seq_len.q)
                                        : 0)
                        + (p.mask.type != mask_type::no_mask
                                        ? (p.seq_len.kv * p.seq_len.q)
                                        : 0)
                        + (5 * p.seq_len.kv * p.seq_len.q))
                / causal_divisor;

        // Just the 5 matmul flops
        num_ops_t<> flash_flops
                = (2.5f * 2.f * p.mb * p.heads.q * p.seq_len.kv * p.seq_len.q
                          * p.head_group.head_size)
                / causal_divisor;

        std::call_once(header_flag, print_table_header);
        std::cout << "BWD" << print_row(p) << "|" << qtime.count() << "|"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes_effective),
                             qtime)
                  << "/"
                  << bandwidth(magnitude_cast<gigabyte>(total_bytes), qtime)
                  << "|" << compute(magnitude_cast<gigaops>(flash_flops), qtime)
                  << "/" << compute(magnitude_cast<gigaops>(total_flops), qtime)
                  << "|" << std::endl;

        if (time_reference) {
            auto ref_time = prim_sdpa_quant_bwd(p, t, eng, strm, t.m_query,
                    t.m_key_quantized, scale_dt, t.m_scale, t.m_mask,
                    t.m_value_quantized, t.m_output, t.m_diff_output,
                    invert_scale, doubled_memory, t.m_diff_query, t.m_diff_key,
                    t.m_diff_value,
                    /*with_timing=*/true);

            float speedup = ref_time.count() > 0
                    ? (float)ref_time.count() / (float)qtime.count()
                    : 0.f;
            std::cout << "REF" << print_row(p) << "|" << ref_time.count() << "|"
                      << speedup << "x speedup|" << std::endl;
        }
    }

protected:
    dnnl::engine eng;
    dnnl::stream strm;
    sdpa_dims_t p;
    sdpa_tensors_t t;
    std::vector<dnnl_memory_t> doubled_memory;

    bool invert_scale = true;
    memory::data_type scale_dt;
};

memory::format_tag with_key_transposed = memory::format_tag::abdc;
memory::format_tag no_key_transposed = memory::format_tag::abcd;

using sdpa_test = sdpa_test_t<sdpa_dims_t>;
using sdpa_test_datatypes = sdpa_test_t<sdpa_dims_t_tuple>;
using sdpa_bwd_test = sdpa_test_t<sdpa_dims_t>;
using sdpa_bwd_test_datatypes = sdpa_test_t<sdpa_dims_t_tuple>;
using sdpa_bwd_perf_test = sdpa_test_t<sdpa_dims_t>;

// clang-format off

INSTANTIATE_TEST_SUITE_P(ScaleTypes_f16, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f16)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}), // mask_type
                ::testing::Values(scale_type::device_side,scale_type::host_side), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);
INSTANTIATE_TEST_SUITE_P(ScaleTypes_bf16, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::bf16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::bf16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::bf16)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}), // mask_type
                ::testing::Values(scale_type::device_side,scale_type::host_side), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);
INSTANTIATE_TEST_SUITE_P(ScaleTypes_f32, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f32)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f32)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f32)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}), // mask_type
                ::testing::Values(scale_type::device_side,scale_type::host_side), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s8, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8) /*, tensor_type_t("K", mdt::s8, mdt::undef, mdt::s8)*/), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f16), tensor_type_t("V", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8) /*, tensor_type_t("V", mdt::s8, mdt::undef, mdt::s8) */), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abcd, dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s8_kf16, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s8, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8) /*, tensor_type_t("V", mdt::s8, mdt::undef, mdt::s8) */), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abcd, dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(PerHeadQuant_f16_s8, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}, num_heads_t {8, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s8, mdt::f16, mdt::undef)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s8, mdt::f16, mdt::undef)), // vdt
                ::testing::Values(quantize_type::per_tensor3), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask, mdt::undef}, mask_config_t {mask_type::causal_tl, mdt::undef}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s4, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s4, mdt::f16, mdt::undef), tensor_type_t("K", mdt::s4, mdt::f16, mdt::s8)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f16), tensor_type_t("V", mdt::s4, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::f16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f16_s4_kf16, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s4, mdt::f16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::f16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f32}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_bf16_s4, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::bf16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s4, mdt::bf16, mdt::undef), tensor_type_t("K", mdt::s4, mdt::bf16, mdt::s8)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::bf16), tensor_type_t("V", mdt::s4, mdt::bf16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::bf16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::bf16}, mask_config_t {mask_type::twoD, mdt::bf16}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_bf16_s4_kbf16, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 386}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::bf16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::bf16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s4, mdt::bf16, mdt::undef), tensor_type_t("V", mdt::s4, mdt::bf16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::oneD, mdt::bf16}, mask_config_t {mask_type::twoD, mdt::bf16}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(DataTypes_f32, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}, seq_len_size_t {1024, 1024}, seq_len_size_t {1, 1025}), // seq_len
                ::testing::Values(head_group_size_t {32, 32, 32}, head_group_size_t {64, 64, 64}, head_group_size_t {128, 128, 128}, head_group_size_t {256, 256, 256}, head_group_size_t {512, 512, 512}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f32)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f32)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f32)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abcd, dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::twoD, mdt::f32}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(AllMaskTypes, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {64, 64, 64}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}, mask_config_t {mask_type::causal_tl}, mask_config_t {mask_type::causal_br}, mask_config_t {mask_type::oneD, mdt::f16}, mask_config_t {mask_type::twoD, mdt::f16}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(AllMaskTypes_D128, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {2, 2}), // hd_num
                ::testing::Values(seq_len_size_t {384, 384}, seq_len_size_t {1, 385}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::s8, mdt::f16, mdt::s8)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::s8, mdt::f16, mdt::s8)), // vdt
                ::testing::Values(quantize_type::per_token), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}, mask_config_t {mask_type::causal_tl}, mask_config_t {mask_type::causal_br}, mask_config_t {mask_type::twoD, mdt::f16}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f32, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);

INSTANTIATE_TEST_SUITE_P(f16_accumulation, sdpa_test_datatypes,
        testing::Combine(::testing::Values(1), // mb
                ::testing::Values(num_heads_t {16, 16}, num_heads_t {12, 2}), // hd_num
                ::testing::Values(seq_len_size_t {1024, 1024}, seq_len_size_t {407, 407}), // seq_len
                ::testing::Values(head_group_size_t {128, 128, 128}, head_group_size_t {80, 80, 80}), // hd_size
                ::testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                ::testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                ::testing::Values(tensor_type_t("V", mdt::f16)), // vdt
                ::testing::Values(quantize_type::no_quantization), // qtype
                ::testing::Values(dnnl::memory::format_tag::abdc), // key_format_tag
                ::testing::Values(mask_config_t {mask_type::no_mask}, mask_config_t {mask_type::twoD}, mask_config_t {mask_type::causal_tl}), // mask_type
                ::testing::Values(default_scale_type), // scale_type
                ::testing::Values(accumulation_t {accumulation_mode::f16, accumulation_mode::f16}, accumulation_t {accumulation_mode::f32, accumulation_mode::f16}, accumulation_t {accumulation_mode::f16, accumulation_mode::f32}), // accumulation_mode
                ::testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);


////llama-2-7b-chat shape: Q [1x32xSEQ_LENx128] KV [1x32xSEQ_LENx128]
////llama-3-8b shape: Q [1x32xSEQ_LENx128] KV [1x8xSEQ_LENx128]
////minicpm-1b-sft shape:  Q [1x24xSEQ_LENx64]  KV [1x8xSEQ_LENx64]
////qwen2-7b shape: Q [1x28xSEQ_LENx128] KV [1x4xSEQ_LENx128]
////phi3-mini-4k-instruct shape: Q [1x32xSEQ_LENx96] KV [1x32xSEQ_LENx96]

INSTANTIATE_TEST_SUITE_P(llama_2_7b_chat,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       kdt,        ksdt,      kzpdt,        vdt,       vsdt,      vzpdt,     mskdt, qtype
    ::testing::Values(
                    sdpa_dims_t{   1,    32,       32,    384,    384,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    385,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    512,    512,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,    513,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1024,   1024,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   1025,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2048,   2048,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl },
                    sdpa_dims_t{   1,    32,       32,   2049,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal_tl }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(llama_3_8b,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,       kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    ::testing::Values(
                    sdpa_dims_t{   1,    32,        8,    384,    384,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    385,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    512,    512,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,    513,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1024,   1024,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   1025,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2048,   2048,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    32,        8,   2049,      1,    128,   128,     128, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(minicpm_1b_st,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,     kdt,      ksdt,      kzpdt,      vdt,     vsdt,      vzpdt,    mskdt, qtype
    ::testing::Values(
                    sdpa_dims_t{   1,    24,        8,    384,    384,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    385,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    512,    512,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,    513,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1024,   1024,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   1025,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2048,   2048,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    24,        8,   2049,      1,     64,    64,      64, mdt::f16, mdt::s8,  mdt::f16,    mdt::s8,  mdt::s8, mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(qwen2_7b,
    sdpa_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,  vzpdt,    mskdt, qtype
    ::testing::Values(
                    sdpa_dims_t{   1,    28,        4,    384,    384,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    385,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    512,    512,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,    513,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1024,   1024,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   1025,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2048,   2048,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,    28,        4,   2049,      1,    128,   128,     128, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(phi3_mini_4k_instruct,
    sdpa_test,
                               // mb, hd_num,kv_grp_sz,seq_len, qry_num,hd_size,  kg_sz, vgrp_sz,       dt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt,    mskdt, qtype
    ::testing::Values(
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::oneD },
                    sdpa_dims_t{   1,     32,       32,    384,     384,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   1,     32,       32,    385,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    512,     512,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,    513,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   1024,    1024,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   1025,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   2048,    2048,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD },
                    sdpa_dims_t{   1,     32,       32,   2049,       1,     96,     96,      96, mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::twoD }
    ), &print_to_string);

INSTANTIATE_TEST_SUITE_P(bwd_perf,
    sdpa_bwd_perf_test,
                               // mb,hd_num,kv_hd_num,seq_len,qry_num,hd_size, kg_sz, vgrp_sz,       dt,    kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    32,      32,     32, mdt::f32, mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    64,      64,     64, mdt::f32, mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,   128,     128,    128, mdt::f32, mdt::f32,  mdt::undef, mdt::undef,  mdt::f32, mdt::undef, mdt::undef, mdt::f32, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    32,      32,     32, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    64,      64,     64, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    128,      128,     128, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::no_mask },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    32,      32,     32, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    64,      64,     64, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::causal_tl },
                    sdpa_dims_t{   4,    4,        4,      4096,       4096,    128,      128,     128, mdt::f16, mdt::f16,  mdt::undef, mdt::undef,  mdt::f16, mdt::undef, mdt::undef, mdt::f16, quantize_type::no_quantization,  no_key_transposed, mask_type::causal_tl }
   ), &print_to_string);

// clang-format on

GPU_TEST_P(sdpa_test, compare) {
    compare();
}

GPU_TEST_P(sdpa_test_datatypes, compare) {
    compare();
}

GPU_TEST_P(sdpa_bwd_test_datatypes, compare_bwd) {
    compare_bwd();
}

GPU_TEST_P(sdpa_bwd_test, compare_bwd) {
    compare_bwd();
}

GPU_TEST_P(sdpa_test, DISABLED_perf) {
    perf();
}

GPU_TEST_P(sdpa_bwd_perf_test, DISABLED_perf_bwd) {
    const bool time_reference = true;
    perf_bwd(time_reference);
}

// Equivalent of benchdnn --attr-dropout=PROBABILITY[:SEED[:TAG[:OFFSET[:HOST_SCALARS]]]]
// Case 1: 0.5:12345678           -> no tag field => benchdnn default tag=any => has_output_mask
// Case 2: 0.75:12345678:undef    -> tag=undef => no mask buffer written
// Case 3: 0.25:843921:any:1238976:true -> output mask + non-zero offset + host scalars
#define SDPA_DIMS_DROPOUT_SHARED_BASE \
    1, 1, 1, 32, 32, 32, 32, 32, mdt::f16, mdt::f16, mdt::undef, mdt::undef, \
            mdt::f16, mdt::undef, mdt::undef, mdt::f16, \
            quantize_type::no_quantization, no_key_transposed, \
            mask_type::no_mask, default_scale_type, accumulation_mode::f32, \
            accumulation_mode::f32
#define SDPA_DIMS_DROPOUT_BASE_TRANSPOSED \
    1, 1, 1, 32, 32, 32, 32, 32, mdt::f16, mdt::f16, mdt::undef, mdt::undef, \
            mdt::f16, mdt::undef, mdt::undef, mdt::f16, \
            quantize_type::no_quantization, with_key_transposed, \
            mask_type::no_mask, default_scale_type, accumulation_mode::f32, \
            accumulation_mode::f32
#define SDPA_DIMS_DROPOUT_MULTI \
    2, 2, 2, 32, 32, 32, 32, 32, mdt::f16, mdt::f16, mdt::undef, mdt::undef, \
            mdt::f16, mdt::undef, mdt::undef, mdt::f16, \
            quantize_type::no_quantization, no_key_transposed, \
            mask_type::no_mask, default_scale_type, accumulation_mode::f32, \
            accumulation_mode::f32
#define SDPA_DIMS_DROPOUT_LARGE \
    1, 1, 1, 32, 256, 256, 32, 32, mdt::f16, mdt::f16, mdt::undef, mdt::undef, \
            mdt::f16, mdt::undef, mdt::undef, mdt::f16, \
            quantize_type::no_quantization, no_key_transposed, \
            mask_type::no_mask, default_scale_type, accumulation_mode::f32, \
            accumulation_mode::f32
#define SDPA_DIMS_DROPOUT_SHARED_GQA \
    2, 4, 2, 128, 128, 64, 64, 64, mdt::f16, mdt::f16, mdt::undef, mdt::undef, \
            mdt::f16, mdt::undef, mdt::undef, mdt::f16, \
            quantize_type::no_quantization, no_key_transposed, \
            mask_type::no_mask, default_scale_type, accumulation_mode::f32, \
            accumulation_mode::f32
INSTANTIATE_TEST_SUITE_P(dropout_forward_minimal, sdpa_test,
        testing::Values(
                // 0.5:12345678:abcd — concrete tag => blocked format => has_output_mask
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                        dropout_config_t {0.5f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::abcd}},
                // 0.75:12345678:undef — explicit undef => no mask output
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                        dropout_config_t {0.75f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                // 0.25:843921:abcd:1238976:true — output mask, offset, host scalars, concrete tag
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                        dropout_config_t {0.25f, 843921, mdt::s64, 1238976,
                                true, memory::format_tag::abcd}},
                // SDPA_DIMS with key transposed
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.5f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::abcd}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.75f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.25f, 843921, mdt::s64, 1238976,
                                true, memory::format_tag::abcd}},
                // multi-batch, multi-head with all three variations
                sdpa_dims_t {SDPA_DIMS_DROPOUT_MULTI,
                        dropout_config_t {0.5f, 42, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_MULTI,
                        dropout_config_t {0.5f, 42, mdt::s64, 0, false,
                                memory::format_tag::abcd}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_MULTI,
                        dropout_config_t {0.5f, 42, mdt::s64, 1238976, true,
                                memory::format_tag::abcd}},
                // larger sequence (256x256) — all three variations
                sdpa_dims_t {SDPA_DIMS_DROPOUT_LARGE,
                        dropout_config_t {0.3f, 99999, mdt::s64, 0, false,
                                memory::format_tag::abcd}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_LARGE,
                        dropout_config_t {0.3f, 99999, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_LARGE,
                        dropout_config_t {0.3f, 99999, mdt::s64, 1238796, true,
                                memory::format_tag::abcd}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.5f, 4242, mdt::s64, 0, false,
                                memory::format_tag::abcd}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.75f, 4343, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.25f, 4444, mdt::s64, 123456, true,
                                memory::format_tag::abcd}}),
        &print_to_string);
#undef SDPA_DIMS_DROPOUT_MULTI
#undef SDPA_DIMS_DROPOUT_LARGE

INSTANTIATE_TEST_SUITE_P(dropout_backward_minimal, sdpa_bwd_test,
        testing::Values(sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                                dropout_config_t {0.5f, 12345678, mdt::s64, 0,
                                        false, memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                        dropout_config_t {0.75f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_BASE,
                        dropout_config_t {0.25f, 843921, mdt::s64, 1238976,
                                true, memory::format_tag::undef}},
                // Base casae with key transposed
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.5f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.75f, 12345678, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_BASE_TRANSPOSED,
                        dropout_config_t {0.25f, 843921, mdt::s64, 1238976,
                                true, memory::format_tag::undef}},
                // GQA: concrete tag
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.5f, 4242, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.75f, 4343, mdt::s64, 0, false,
                                memory::format_tag::undef}},
                sdpa_dims_t {SDPA_DIMS_DROPOUT_SHARED_GQA,
                        dropout_config_t {0.25f, 4444, mdt::s64, 123456, true,
                                memory::format_tag::undef}}),
        &print_to_string);
#undef SDPA_DIMS_DROPOUT_SHARED_BASE_TRANSPOSED
#undef SDPA_DIMS_DROPOUT_SHARED_BASE
#undef SDPA_DIMS_DROPOUT_SHARED_GQA

// clang-format off

INSTANTIATE_TEST_SUITE_P(bwd_smoke, sdpa_bwd_test_datatypes,
        testing::Combine(testing::Values(1), // mb
                testing::Values(num_heads_t {2, 2}), // heads
                testing::Values(seq_len_size_t {64, 64}), // seq_len
                testing::Values(head_group_size_t {32, 32, 32}), // head_size
                testing::Values(tensor_type_t("Q", mdt::f16)), // dt
                testing::Values(tensor_type_t("K", mdt::f16)), // kdt
                testing::Values(tensor_type_t("V", mdt::f16)), // vdt
                testing::Values(quantize_type::no_quantization), // qtype
                testing::Values(dnnl::memory::format_tag::abcd), // key_format_tag
                testing::Values(mask_config_t {mask_type::causal_tl}), // mask_type
                testing::Values(default_scale_type), // scale_type
                testing::Values(
                        accumulation_t {accumulation_mode::f32,
                                accumulation_mode::f32}), // accumulation_mode
                testing::Values(no_dropout) // dropout
                ),
        &print_to_string2);
// clang-format on

// An in-primitive Q descale must match folding the same scale into Q on the host
static void test_query_descale_equivalence(
        memory::dim mb, memory::dim H_q, memory::dim H_kv, memory::dim S_q) {
    using namespace dnnl::impl;

    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "SDPA tests require gpus.");
    dnnl::engine eng(engine::kind::gpu, 0);
    dnnl::stream strm(eng);

    const memory::dim D = 64, S_kv = 128;
    const memory::dims q_sz = {mb, H_q, S_q, D}, k_sz = {mb, H_kv, D, S_kv},
                       v_sz = {mb, H_kv, S_kv, D}, o_sz = {mb, H_q, S_q, D};

    memory::desc q_md(q_sz, mdt::f16, memory::format_tag::abcd);
    memory::desc k_md(k_sz, mdt::f16, memory::format_tag::abcd);
    memory::desc v_md(v_sz, mdt::f16, memory::format_tag::abcd);
    memory::desc o_md(o_sz, mdt::f16, memory::format_tag::abcd);
    memory::desc scale_md({1}, mdt::f32, memory::format_tag::a);
    memory::desc qs_md({mb, H_q, 1, 1}, mdt::f32, memory::format_tag::abcd);

    std::vector<float> q_data(product(q_sz)), k_data(product(k_sz)),
            v_data(product(v_sz));
    fill_random(q_data, q_md);
    fill_random(k_data, k_md);
    fill_random(v_data, v_md);

    std::vector<float> qs(mb * H_q);
    const float pow2[] = {0.5f, 2.f, 0.25f, 4.f};
    for (memory::dim b = 0; b < mb; b++)
        for (memory::dim h = 0; h < H_q; h++)
            qs[b * H_q + h] = pow2[h % 4] * (b == 0 ? 1.f : 8.f);

    std::vector<float> q_folded(q_data.size());
    for (memory::dim bh = 0; bh < mb * H_q; bh++)
        for (memory::dim i = 0; i < S_q * D; i++)
            q_folded[bh * S_q * D + i] = q_data[bh * S_q * D + i] * qs[bh];

    bool unimplemented = false;
    auto run = [&](const std::vector<float> &qsrc,
                       bool with_q_scales) -> std::vector<float> {
        memory qm(q_md, eng), km(k_md, eng), vm(v_md, eng), om(o_md, eng),
                sm(scale_md, eng), qsm;
        write_to_dnnl_memory(qsrc.data(), qm, eng, strm);
        write_to_dnnl_memory(k_data.data(), km, eng, strm);
        write_to_dnnl_memory(v_data.data(), vm, eng, strm);
        const float one = 1.f;
        write_to_dnnl_memory(&one, sm, eng, strm);

        primitive_attr kq_attr;
        if (with_q_scales) {
            kq_attr.set_scales(DNNL_ARG_SRC, /* mask = per (batch, head) */ 3,
                    {}, mdt::f32);
            qsm = memory(qs_md, eng);
            write_to_dnnl_memory(qs.data(), qsm, eng, strm);
        }

        sdpa::primitive_desc pd;
        try {
            pd = sdpa::primitive_desc(eng, q_md, k_md, v_md, nullptr, scale_md,
                    o_md, /* invert_scale = */ false, H_kv,
                    to_attn_mask_type(mask_type::no_mask),
                    alg_kind::softmax_accurate, prop_kind::forward_inference,
                    primitive_attr(), kq_attr);
        } catch (const dnnl::error &e) {
            if (e.status == dnnl_unimplemented) {
                unimplemented = true;
                return {};
            }
            throw;
        }
        sdpa prim(pd);

        std::unordered_map<int, memory> args = {{DNNL_ARG_QUERIES, qm},
                {DNNL_ARG_KEYS, km}, {DNNL_ARG_VALUES, vm},
                {DNNL_ARG_SCALE, sm}, {DNNL_ARG_DST, om}};
        if (with_q_scales) args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_QUERIES] = qsm;
        prim.execute(strm, args);
        strm.wait();

        memory of32({o_sz, mdt::f32, memory::format_tag::abcd}, eng);
        dnnl::reorder(om, of32).execute(strm, om, of32);
        strm.wait();
        std::vector<float> out(product(o_sz));
        void *p = of32.map_data();
        std::memcpy(out.data(), p, out.size() * sizeof(float));
        of32.unmap_data(p);
        return out;
    };

    const auto scaled_in_primitive = run(q_data, /* with_q_scales = */ true);
    if (unimplemented) GTEST_SKIP() << "Q descale unimplemented";
    const auto folded_into_q = run(q_folded, /* with_q_scales = */ false);
    if (unimplemented) GTEST_SKIP() << "baseline unimplemented";

    ASSERT_EQ(scaled_in_primitive.size(), folded_into_q.size());
    for (size_t i = 0; i < folded_into_q.size(); i++) {
        ASSERT_NEAR(scaled_in_primitive[i], folded_into_q[i],
                1e-3f * std::max(1.f, std::abs(folded_into_q[i])))
                << "mismatch at " << i;
    }
}

TEST(sdpa_query_scales, PerQueryHead) {
    test_query_descale_equivalence(
            /* mb = */ 2, /* H_q = */ 4, /* H_kv = */ 4, /* S_q = */ 128);
}

TEST(sdpa_query_scales, PerQueryHeadGQA) {
    test_query_descale_equivalence(
            /* mb = */ 2, /* H_q = */ 8, /* H_kv = */ 2, /* S_q = */ 128);
}

TEST(sdpa_query_scales, PerQueryHeadGQASingleQueryRejected) {
    using namespace dnnl::impl;
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "SDPA tests require gpus.");
    dnnl::engine eng(engine::kind::gpu, 0);

    const memory::dim mb = 1, H_q = 8, H_kv = 2, D = 64, S_kv = 128;
    memory::desc q_md({mb, H_q, 1, D}, mdt::f16, memory::format_tag::abcd);
    memory::desc k_md({mb, H_kv, D, S_kv}, mdt::f16, memory::format_tag::abcd);
    memory::desc v_md({mb, H_kv, S_kv, D}, mdt::f16, memory::format_tag::abcd);
    memory::desc o_md({mb, H_q, 1, D}, mdt::f16, memory::format_tag::abcd);
    memory::desc scale_md({1}, mdt::f32, memory::format_tag::a);

    primitive_attr kq_attr;
    kq_attr.set_scales(
            DNNL_ARG_SRC, /* mask = per (batch, head) */ 3, {}, mdt::f32);

    try {
        sdpa::primitive_desc pd(eng, q_md, k_md, v_md, nullptr, scale_md, o_md,
                /* invert_scale = */ false, H_kv,
                to_attn_mask_type(mask_type::no_mask),
                alg_kind::softmax_accurate, prop_kind::forward_inference,
                primitive_attr(), kq_attr);
        ADD_FAILURE() << "expected dnnl_unimplemented";
    } catch (const dnnl::error &e) { EXPECT_EQ(e.status, dnnl_unimplemented); }
}

// fp8 SDPA end to end, against a host model of the defined numerics: f32
// scores and softmax, probabilities quantized to e4m3, then P*V.
namespace {

constexpr float vs_s_fp8_scale = 448.f;

std::vector<float> read_as_f32(
        dnnl::memory &mem, dnnl::engine &eng, dnnl::stream &strm) {
    const auto md = mem.get_desc();
    dnnl::memory f32_mem({md.get_dims(), mdt::f32, md.get_strides()}, eng);
    dnnl::reorder(mem, f32_mem).execute(strm, mem, f32_mem);
    strm.wait();
    std::vector<float> out(product(md.get_dims()));
    void *p = f32_mem.map_data();
    std::memcpy(out.data(), p, out.size() * sizeof(float));
    f32_mem.unmap_data(p);
    return out;
}

// Round-trip through a real reorder so the reference rounds the way the
// hardware does rather than through a hand-rolled model of e4m3
std::vector<float> quantize_f8_e4m3(
        const std::vector<float> &in, dnnl::engine &eng, dnnl::stream &strm) {
    const memory::dims d = {static_cast<memory::dim>(in.size())};
    memory f32_mem({d, mdt::f32, memory::format_tag::a}, eng);
    memory f8_mem({d, mdt::f8_e4m3, memory::format_tag::a}, eng);
    write_to_dnnl_memory(in.data(), f32_mem, eng, strm);
    dnnl::reorder(f32_mem, f8_mem).execute(strm, f32_mem, f8_mem);
    strm.wait();
    return read_as_f32(f8_mem, eng, strm);
}

// Broadcast a per (batch, head) descale over a 4D tensor and fold it in
std::vector<float> fold_descale(const std::vector<float> &data,
        const std::vector<float> &scales, memory::dim per_head_elems) {
    std::vector<float> out(data.size());
    for (size_t bh = 0; bh < scales.size(); bh++)
        for (memory::dim i = 0; i < per_head_elems; i++)
            out[bh * per_head_elems + i]
                    = data[bh * per_head_elems + i] * scales[bh];
    return out;
}

std::vector<float> pow2_scales(memory::dim n_bh) {
    std::vector<float> s(n_bh);
    const float pow2[] = {0.5f, 2.f, 0.25f, 4.f};
    for (memory::dim i = 0; i < n_bh; i++)
        s[i] = pow2[i % 4];
    return s;
}

std::vector<float> decomposed_reference(const std::vector<float> &q_deq,
        const std::vector<float> &k_deq, const std::vector<float> &v_deq,
        memory::dim mb, memory::dim H_q, memory::dim H_kv, memory::dim S_q,
        memory::dim S_kv, memory::dim D, dnnl::engine &eng, dnnl::stream &strm,
        std::vector<float> *absmag) {
    const auto abcd = memory::format_tag::abcd;
    const memory::dim kv_group = H_q / H_kv;

    // Expand the KV heads so both matmuls are plain batched ones
    auto expand = [&](const std::vector<float> &in, memory::dim per_head) {
        std::vector<float> out((size_t)mb * H_q * per_head);
        for (memory::dim b = 0; b < mb; b++)
            for (memory::dim h = 0; h < H_q; h++)
                std::memcpy(&out[((size_t)b * H_q + h) * per_head],
                        &in[((size_t)b * H_kv + h / kv_group) * per_head],
                        per_head * sizeof(float));
        return out;
    };
    const auto k_e = expand(k_deq, D * S_kv);
    const auto v_e = expand(v_deq, S_kv * D);

    memory::desc q_md({mb, H_q, S_q, D}, mdt::f32, abcd);
    memory::desc k_md({mb, H_q, D, S_kv}, mdt::f32, abcd);
    memory::desc s_md({mb, H_q, S_q, S_kv}, mdt::f32, abcd);
    memory::desc v_md({mb, H_q, S_kv, D}, mdt::f32, abcd);
    memory::desc o_md({mb, H_q, S_q, D}, mdt::f32, abcd);

    memory qm(q_md, eng), km(k_md, eng), scores(s_md, eng), probs(s_md, eng),
            vm(v_md, eng), om(o_md, eng);
    write_to_dnnl_memory(q_deq.data(), qm, eng, strm);
    write_to_dnnl_memory(k_e.data(), km, eng, strm);
    write_to_dnnl_memory(v_e.data(), vm, eng, strm);

    dnnl::matmul(dnnl::matmul::primitive_desc(eng, q_md, k_md, s_md))
            .execute(strm,
                    {{DNNL_ARG_SRC, qm}, {DNNL_ARG_WEIGHTS, km},
                            {DNNL_ARG_DST, scores}});
    dnnl::softmax_forward(dnnl::softmax_forward::primitive_desc(eng,
                                  dnnl::prop_kind::forward_inference,
                                  dnnl::algorithm::softmax_accurate, s_md, s_md,
                                  /* axis */ 3))
            .execute(strm, {{DNNL_ARG_SRC, scores}, {DNNL_ARG_DST, probs}});
    strm.wait();

    // Quantize and dequantize the normalized probabilities, as the graph does
    // 1/vs_s_fp8_scale maps [0, 1] onto the full e4m3 range
    auto p = read_as_f32(probs, eng, strm);
    for (auto &x : p)
        x *= vs_s_fp8_scale;
    p = quantize_f8_e4m3(p, eng, strm);
    for (auto &x : p)
        x /= vs_s_fp8_scale;
    write_to_dnnl_memory(p.data(), probs, eng, strm);

    dnnl::matmul(dnnl::matmul::primitive_desc(eng, s_md, v_md, o_md))
            .execute(strm,
                    {{DNNL_ARG_SRC, probs}, {DNNL_ARG_WEIGHTS, vm},
                            {DNNL_ARG_DST, om}});
    strm.wait();
    const auto out = read_as_f32(om, eng, strm);

    // Conditioning magnitude: the probabilities are non-negative, so
    // sum_j p_j |v_j| bounds how much an e4m3 rounding of p can move the
    // output. Cancellation in the real sum makes |out| a useless yardstick.
    auto v_abs = v_e;
    for (auto &x : v_abs)
        x = std::abs(x);
    memory vam(v_md, eng), amm(o_md, eng);
    write_to_dnnl_memory(v_abs.data(), vam, eng, strm);
    dnnl::matmul(dnnl::matmul::primitive_desc(eng, s_md, v_md, o_md))
            .execute(strm,
                    {{DNNL_ARG_SRC, probs}, {DNNL_ARG_WEIGHTS, vam},
                            {DNNL_ARG_DST, amm}});
    strm.wait();
    *absmag = read_as_f32(amm, eng, strm);
    return out;
}
} // namespace

static void test_fp8_sdpa(memory::dim mb, memory::dim H_q, memory::dim H_kv,
        memory::dim S_q, mdt dst_dt, float tolerance) {
    using namespace dnnl::impl;

    SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
            "SDPA tests require gpus.");
    dnnl::engine eng(engine::kind::gpu, 0);
    dnnl::stream strm(eng);

    const memory::dim D = 64, S_kv = 128;
    const memory::dims q_sz = {mb, H_q, S_q, D}, k_sz = {mb, H_kv, D, S_kv},
                       v_sz = {mb, H_kv, S_kv, D}, o_sz = {mb, H_q, S_q, D};
    const auto abcd = memory::format_tag::abcd;

    memory::desc q8_md(q_sz, mdt::f8_e4m3, abcd),
            k8_md(k_sz, mdt::f8_e4m3, abcd), v8_md(v_sz, mdt::f8_e4m3, abcd);
    memory::desc o_md(o_sz, dst_dt, abcd);
    memory::desc scale_md({1}, mdt::f32, memory::format_tag::a);
    memory::desc qs_md({mb, H_q, 1, 1}, mdt::f32, abcd);
    memory::desc kvs_md({mb, H_kv, 1, 1}, mdt::f32, abcd);

    std::vector<float> q_raw(product(q_sz)), k_raw(product(k_sz)),
            v_raw(product(v_sz));
    fill_random(q_raw, memory::desc(q_sz, mdt::f16, abcd));
    fill_random(k_raw, memory::desc(k_sz, mdt::f16, abcd));
    fill_random(v_raw, memory::desc(v_sz, mdt::f16, abcd));

    // Round-trip through fp8 so the reference uses the values actually
    // stored, not a model of e4m3 rounding
    memory qm(q8_md, eng), km(k8_md, eng), vm(v8_md, eng);
    write_to_dnnl_memory(q_raw.data(), qm, eng, strm);
    write_to_dnnl_memory(k_raw.data(), km, eng, strm);
    write_to_dnnl_memory(v_raw.data(), vm, eng, strm);
    const auto q_stored = read_as_f32(qm, eng, strm);
    const auto k_stored = read_as_f32(km, eng, strm);
    const auto v_stored = read_as_f32(vm, eng, strm);

    const auto q_scales = pow2_scales(mb * H_q);
    const auto k_scales = pow2_scales(mb * H_kv);
    const auto v_scales = pow2_scales(mb * H_kv);

    std::vector<float> absmag;
    const auto ref_out
            = decomposed_reference(fold_descale(q_stored, q_scales, S_q * D),
                    fold_descale(k_stored, k_scales, D * S_kv),
                    fold_descale(v_stored, v_scales, S_kv * D), mb, H_q, H_kv,
                    S_q, S_kv, D, eng, strm, &absmag);

    memory om(o_md, eng), sm(scale_md, eng), qsm(qs_md, eng), ksm(kvs_md, eng),
            vsm(kvs_md, eng);
    const float one = 1.f;
    write_to_dnnl_memory(&one, sm, eng, strm);
    write_to_dnnl_memory(q_scales.data(), qsm, eng, strm);
    write_to_dnnl_memory(k_scales.data(), ksm, eng, strm);
    write_to_dnnl_memory(v_scales.data(), vsm, eng, strm);

    primitive_attr kq_attr, vs_attr;
    kq_attr.set_scales(DNNL_ARG_SRC, 3, {}, mdt::f32);
    kq_attr.set_scales(DNNL_ARG_WEIGHTS, 3, {}, mdt::f32);
    vs_attr.set_scales(DNNL_ARG_WEIGHTS, 3, {}, mdt::f32);

    sdpa::primitive_desc pd;
    try {
        pd = sdpa::primitive_desc(eng, q8_md, k8_md, v8_md, nullptr, scale_md,
                o_md, /* invert_scale = */ false, H_kv,
                to_attn_mask_type(mask_type::no_mask),
                alg_kind::softmax_accurate, prop_kind::forward_inference,
                primitive_attr(), kq_attr, vs_attr);
    } catch (const dnnl::error &e) {
        if (e.status == dnnl_unimplemented)
            GTEST_SKIP() << "fp8 sdpa unimplemented: " << e.what();
        throw;
    }
    sdpa prim(pd);
    prim.execute(strm,
            {{DNNL_ARG_QUERIES, qm}, {DNNL_ARG_KEYS, km}, {DNNL_ARG_VALUES, vm},
                    {DNNL_ARG_SCALE, sm}, {DNNL_ARG_DST, om},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_QUERIES, qsm},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS, ksm},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES, vsm}});
    strm.wait();
    const auto got = read_as_f32(om, eng, strm);

    ASSERT_EQ(got.size(), ref_out.size());
    double max_rel = 0.;
    for (size_t i = 0; i < ref_out.size(); i++) {
        ASSERT_TRUE(std::isfinite(got[i])) << "non-finite at " << i;
        const float denom = std::max(1e-6f, absmag[i]);
        max_rel = std::max<double>(
                max_rel, std::abs(got[i] - ref_out[i]) / denom);
        ASSERT_NEAR(got[i], ref_out[i], tolerance * denom)
                << "mismatch at " << i;
    }
    ::testing::Test::RecordProperty(
            "max_relative_deviation", std::to_string(max_rel));
}

TEST(sdpa_fp8, E4M3PerHeadF16Dst) {
    test_fp8_sdpa(/* mb = */ 2, /* H_q = */ 4, /* H_kv = */ 4, /* S_q = */ 128,
            mdt::f16, /* tolerance = */ 0.125f);
}

// Only the store rounds differently, so the budget is one bf16 ulp (2^-8)
TEST(sdpa_fp8, E4M3PerHeadBf16Dst) {
    test_fp8_sdpa(/* mb = */ 2, /* H_q = */ 4, /* H_kv = */ 4, /* S_q = */ 128,
            mdt::bf16, /* tolerance = */ 0.125f);
}

TEST(sdpa_fp8, E4M3PerHeadGQA) {
    test_fp8_sdpa(/* mb = */ 2, /* H_q = */ 8, /* H_kv = */ 2, /* S_q = */ 128,
            mdt::bf16, /* tolerance = */ 0.125f);
}
