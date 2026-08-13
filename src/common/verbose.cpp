/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#include <atomic>
#include <map>
#include <regex>
#include <sstream>
#include <type_traits>

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_version.h"
#include "oneapi/dnnl/dnnl_version_hash.h"

#include "c_types_map.hpp"
#include "verbose.hpp"

#include "batch_normalization_pd.hpp"
#include "binary_pd.hpp"
#include "concat_pd.hpp"
#include "convolution_pd.hpp"
#include "deconvolution_pd.hpp"
#include "eltwise_pd.hpp"
#include "gated_mlp_pd.hpp"
#include "gemm_pd.hpp"
#include "group_normalization_pd.hpp"
#include "inner_product_pd.hpp"
#include "layer_normalization_pd.hpp"
#include "lrn_pd.hpp"
#include "matmul_pd.hpp"
#include "pooling_pd.hpp"
#include "prelu_pd.hpp"
#include "reduction_pd.hpp"
#include "reorder_pd.hpp"
#include "resampling_pd.hpp"
#include "rnn_pd.hpp"
#include "sdpa_pd.hpp"
#include "shuffle_pd.hpp"
#include "softmax_pd.hpp"
#include "sum_pd.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "xpu/ocl/verbose.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "xpu/sycl/verbose.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "xpu/ze/verbose.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL
#include "common/experimental.hpp"
#endif

namespace dnnl {
namespace impl {

// Versioning is used to communicate breaking verbose lines changes to
// verbose_converter tool for maintaining compatibility smoother.
// Numeration uses integers only and goes linearly from 0 to infinity.
static constexpr char verbose_version[] = "v1";

static setting_t<uint32_t> verbose {0};

// Component filters help manage verbose output by parsing and printing for
// matching components. The filter status is tracked from verbose initializaton,
// allowing queries for the component type during verbose printing.
filter_status_t &filter_status() {
    static filter_status_t filter_status;
    return filter_status;
}

void print_header() noexcept {
    static std::atomic_flag version_printed = ATOMIC_FLAG_INIT;
    if (!version_printed.test_and_set()) {
        verbose_printf("info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        verbose_printf("info,cpu,runtime:%s,nthr:%d\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime),
                dnnl_get_max_threads());
        verbose_printf("info,cpu,isa:%s\n", cpu::platform::get_isa_info());
#endif
        verbose_printf("info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
        // Printing the header generally requires iterating over devices/backends,
        // which may involve an allocation. Use a try/catch block in case
        // these fail (not printing a header is reasonable in this case)
        try {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
            xpu::ocl::print_verbose_header();
#endif
#ifdef DNNL_WITH_SYCL
            xpu::sycl::print_verbose_header();
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
            xpu::ze::print_verbose_header();
#endif
#ifdef ONEDNN_BUILD_GRAPH
            graph::utils::print_verbose_header();
#endif
        } catch (...) {
            verbose_printf("info,exception while printing verbose header\n");
        }
#ifdef DNNL_EXPERIMENTAL
        verbose_printf("info,experimental features are enabled\n");
        verbose_printf("info,use batch_normalization stats one pass is %s\n",
                experimental::use_bnorm_stats_one_pass() ? "enabled"
                                                         : "disabled");
        verbose_printf("info,GPU convolution v2 is %s\n",
                experimental::use_gpu_conv_v2() ? "enabled" : "disabled");
#endif
#ifdef DNNL_EXPERIMENTAL_LOGGING
        const log_manager_t &log_manager = log_manager_t::get_log_manager();
        if (log_manager.is_logger_enabled())
            verbose_printf(
                    "info,experimental functionality for logging is enabled\n");
#endif
#ifdef DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER
        verbose_printf("info,experimental SYCL kernel compiler is enabled\n");
#endif
        verbose_printf(
                "primitive,info,template:%soperation,engine,primitive,"
                "implementation,prop_kind,memory_descriptors,attributes,"
                "auxiliary,problem_desc,exec_time\n",
                get_verbose_timestamp() ? "timestamp," : "");

#ifdef ONEDNN_BUILD_GRAPH
        verbose_printf(
                "graph,info,template:%soperation,engine,partition_id,"
                "partition_kind,op_names,data_formats,logical_tensors,fpmath_"
                "mode,implementation,backend,exec_time\n",
                get_verbose_timestamp() ? "timestamp," : "");
#endif
        if (filter_status().status == filter_status_t::flags::valid)
            verbose_printf(
                    "common,info,filter format is enabled, hit components: "
                    "%s\n",
                    filter_status().components.c_str());
        else if (filter_status().status == filter_status_t::flags::invalid)
            verbose_printf(
                    "common,error,filter format is ill-formed and is not "
                    "applied, error: %s\n",
                    filter_status().err_msg.c_str());
    }
}

// hint parameter is the kind of verbose we are querying for
uint32_t get_verbose(verbose_t::flag_kind verbosity_kind,
        component_t::flag_kind filter_kind) noexcept {
#if defined(DISABLE_VERBOSE)
    return verbose_t::none;
#endif
    // we print all verbose by default
    static int flags = component_t::all;

    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        static std::string user_opt = getenv_string_user("VERBOSE");
        auto update_kind = [&](const std::string &s, int &k) {
            // Legacy: we accept values 0,1,2
            // 0 and none erase previously set flags, including error
            if (s == "0" || s == "none") k = verbose_t::none;
            if (s == "1") k |= verbose_t::level1;
            if (s == "2") k |= verbose_t::level2;
            if (s == "all" || s == "-1") k |= verbose_t::all;
            if (s == "error") k |= verbose_t::error;
            if (s == "check")
                k |= verbose_t::create_check | verbose_t::exec_check;
            if (s == "dispatch") k |= verbose_t::create_dispatch;
            if (s == "profile")
                k |= verbose_t::create_profile | verbose_t::exec_profile;
            if (s == "profile_create") k |= verbose_t::create_profile;
            if (s == "profile_exec") k |= verbose_t::exec_profile;
            // Enable profiling to external libraries
            if (s == "profile_externals") k |= verbose_t::profile_externals;
            if (s == "warn") k |= verbose_t::warn;
            // we extract debug info debuginfo=XX. ignore if debuginfo is invalid.
            if (s.rfind("debuginfo=", 0) == 0)
                k |= verbose_t::make_debuginfo(
                        std::strtol(s.c_str() + 10, nullptr, 10));
        };

        auto update_filter = [&](const std::string &s) -> int {
            int k = component_t::none;
            try {
                std::regex regexp = std::regex(s);

#define REGEX_SEARCH(k, component, regexp) \
    if (std::regex_search("" #component "", regexp)) { \
        (k) |= component_t::component; \
        filter_status().components += "" #component ","; \
    }
                REGEX_SEARCH(k, primitive, regexp);
                REGEX_SEARCH(k, reorder, regexp);
                REGEX_SEARCH(k, shuffle, regexp);
                REGEX_SEARCH(k, concat, regexp);
                REGEX_SEARCH(k, sum, regexp);
                REGEX_SEARCH(k, convolution, regexp);
                REGEX_SEARCH(k, deconvolution, regexp);
                REGEX_SEARCH(k, eltwise, regexp);
                REGEX_SEARCH(k, lrn, regexp);
                REGEX_SEARCH(k, batch_normalization, regexp);
                REGEX_SEARCH(k, inner_product, regexp);
                REGEX_SEARCH(k, rnn, regexp);
                REGEX_SEARCH(k, binary, regexp);
                REGEX_SEARCH(k, matmul, regexp);
                REGEX_SEARCH(k, resampling, regexp);
                REGEX_SEARCH(k, pooling, regexp);
                REGEX_SEARCH(k, reduction, regexp);
                REGEX_SEARCH(k, prelu, regexp);
                REGEX_SEARCH(k, softmax, regexp);
                REGEX_SEARCH(k, layer_normalization, regexp);
                REGEX_SEARCH(k, group_normalization, regexp);
                REGEX_SEARCH(k, graph, regexp);
                REGEX_SEARCH(k, gemm_api, regexp);
                REGEX_SEARCH(k, ukernel, regexp);
#undef REGEX_SEARCH
            } catch (const std::exception &e) {
                filter_status().status = filter_status_t::flags::invalid;
                filter_status().err_msg = e.what();
                return component_t::all;
            }

            // filter enabled and at least one component is hit
            if (!filter_status().components.empty()) {
                // pop out the last comma
                filter_status().components.pop_back();
                filter_status().status = filter_status_t::flags::valid;
            } else {
                filter_status().status = filter_status_t::flags::invalid;
                filter_status().err_msg
                        = "component with name \'" + s + "\' not found";
            }
            return k;
        };

        // we always enable error by default
        int val = verbose_t::error;
        for (size_t pos_st = 0, pos_en = user_opt.find_first_of(',', pos_st);
                true; pos_st = pos_en + 1,
                    pos_en = user_opt.find_first_of(',', pos_st)) {
            std::string tok = user_opt.substr(pos_st, pos_en - pos_st);
            // update verbose flags
            update_kind(tok, val);
            // update filter flags
            if (tok.rfind("filter=", 0) == 0) {
                auto filter_str = tok.substr(7);
                if (!filter_str.empty()) { flags = update_filter(filter_str); }
            }
            if (pos_en == std::string::npos) break;
        }

        // We parse for explicit flags
        verbose.set(val);

#ifdef DNNL_EXPERIMENTAL_LOGGING
        const log_manager_t &log_manager = log_manager_t::get_log_manager();
        if (log_manager.is_logger_enabled())
            log_manager.set_log_level(user_opt);
#endif
    }

    int result = verbose.get() & verbosity_kind;
    if (verbosity_kind == verbose_t::debuginfo)
        result = verbose_t::get_debuginfo(verbose.get());
    bool filter_result = flags & filter_kind;
    return filter_result ? result : 0;
}

static setting_t<bool> verbose_timestamp {false};
bool get_verbose_timestamp() {
#if defined(DISABLE_VERBOSE)
    return false;
#endif
    if (verbose.get() == 0) return false;

    if (!verbose_timestamp.initialized()) {
        // Assumes that all threads see the same environment
        static bool val
                = getenv_int_user("VERBOSE_TIMESTAMP", verbose_timestamp.get());
        verbose_timestamp.set(val);
    }
    return verbose_timestamp.get();
}

std::ostream &operator<<(std::ostream &ss, engine_kind_t eng_kind) {
    ss << dnnl_engine_kind2str(eng_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const engine_t *engine) {
    ss << dnnl_engine_kind2str(engine->kind());
    if (dnnl_engine_get_count(engine->kind()) > 1)
        ss << ":" + std::to_string(engine->index());
    return ss;
}

const char *prim_kind2str(primitive_kind_t prim_kind) {
    switch ((int)prim_kind) {
        case primitive_kind::zero_pad: return "zero_pad";
        default: return dnnl_prim_kind2str(prim_kind);
    }
}

std::ostream &operator<<(std::ostream &ss, primitive_kind_t prim_kind) {
    ss << prim_kind2str(prim_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, prop_kind_t prop_kind) {
    ss << dnnl_prop_kind2str(prop_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, data_type_t data_type) {
    ss << dnnl_dt2str(data_type);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, alg_kind_t alg) {
    ss << dnnl_alg_kind2str(alg);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, format_kind_t format_kind) {
    ss << dnnl_fmt_kind2str(format_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, sparse_encoding_t encoding) {
    ss << dnnl_sparse_encoding2str(encoding);
    return ss;
}

std::string normalization_flags2str(unsigned flags) {
    std::string s;
    if (flags & normalization_flags::use_global_stats) s += "G";
    if (flags & normalization_flags::use_scale) s += "C";
    if (flags & normalization_flags::use_shift) s += "H";
    if (flags & normalization_flags::fuse_norm_relu) s += "R";
    if (flags & normalization_flags::fuse_norm_add_relu) s += "A";
    if (flags & normalization_flags::rms_norm) s += "M";
    return s;
}

std::string rnn_flags2str(unsigned flags) {
    std::string s;
    if (flags & rnn_flags::diff_weights_overwrite) s += "O";
    return s;
}

std::string cublasltfmt2str(const memory_desc_t *md) {
    if (md->format_desc.cublaslt_blocked_desc.cublaslt_format
            == cublaslt_memory_format_t::col32_2r_4r4) {
        return ":col32_2r_4r4";
    }
    return "";
}

std::ostream &operator<<(std::ostream &ss, const memory_extra_desc_t &extra) {
    using namespace memory_extra_flags;

    ss << ":f" << extra.flags;
    if (extra.flags & compensation_conv_s8s8)
        ss << ":s8m" << extra.compensation_mask;
    if (extra.flags & compensation_conv_asymmetric_src)
        ss << ":zpm" << extra.asymm_compensation_mask;
    if (extra.flags & compensation_gpu_conv_asymmetric_src) {
        ss << ":zid" << extra.idhw[0];
        ss << ":zih" << extra.idhw[1];
        ss << ":ziw" << extra.idhw[2];
        ss << ":zod" << extra.odhw[0];
        ss << ":zoh" << extra.odhw[1];
        ss << ":zow" << extra.odhw[2];
        ss << ":zpd" << extra.pdhw[0];
        ss << ":zph" << extra.pdhw[1];
        ss << ":zpw" << extra.pdhw[2];
        ss << ":zdd" << extra.ddhw[0];
        ss << ":zdh" << extra.ddhw[1];
        ss << ":zdw" << extra.ddhw[2];
        ss << ":zs" << extra.dst_size;
    }
    if (extra.flags & scale_adjust && extra.scale_adjust != 1.f)
        ss << ":sa" << extra.scale_adjust;
    return ss;
}

std::string md2fmt_tag_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);

    // Can't report meaningful tag for runtime dimensions.
    if (mdw.has_runtime_strides()) return "*";

    struct sort_key_t {
        uint64_t stride_order;
        dim_t outer_block;
        int idx;
        char dim_char;
    };

    dims_t blocks = {0};
    mdw.compute_blocks(blocks);

    std::vector<sort_key_t> sort_keys(mdw.ndims());
    const auto &pdims = mdw.padded_dims();
    const auto &blk = mdw.blocking_desc();
    for (int i = 0; i < mdw.ndims(); ++i)
        // Assume that any dimension with stride 0 is outer relative to other
        // dimensions. Use (uint64_t)(stride - 1) to sort a stride of 0 highest.
        // Multiple dimensions with stride 0 is ambiguous.
        sort_keys[i] = {(uint64_t)(blk.strides[i] - 1), pdims[i] / blocks[i], i,
                (char)((blocks[i] == 1 ? 'a' : 'A') + i)};

    // Old approach: utils::simultaneous_sort(strides, outer_blocks, dim_chars)
    //   input tag: acdb
    //   dims: 5x8x0x2
    //   strides: 0x1x16x8
    //   output tag: cdba
    //
    // New approach with std::sort and sort keys:
    //   input tag: acdb
    //   dims: 5x8x0x2
    //   "stride orders": (BIG NUMBER)x0x15x7
    //   output tag: acdb
    std::sort(sort_keys.begin(), sort_keys.end(),
            [](const sort_key_t &left, const sort_key_t &right) {
        if (left.stride_order < right.stride_order) return false;
        if (left.stride_order == right.stride_order) {
            // WLOG, we can assume a dimension of size 1 has the same
            // stride as the next outermost dimension. Sort the one with
            // the non-unit outer block as the outer dimension. Multiple
            // dimensions of size 1 with the same stride is ambiguous.
            if (left.outer_block < right.outer_block) return false;
            if (left.outer_block == right.outer_block)
                // Sort 1x1x... outer blocks to (arbitrarily) list them
                // in alphabetical order.
                return left.idx < right.idx;
        }
        return true;
    });

    char dim_chars[DNNL_MAX_NDIMS + 1];
    for (int i = 0; i < mdw.ndims(); ++i)
        dim_chars[i] = sort_keys[i].dim_char;
    dim_chars[mdw.ndims()] = '\0';

    std::string s(dim_chars);

    if (!mdw.is_plain()) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            char c = ('a' + (char)blk.inner_idxs[iblk]);
            s += (std::to_string(blk.inner_blks[iblk]) + c);
        }
    }

    return s;
}

std::string md2fmt_strides_str(const memory_desc_t *md) {
    memory_desc_wrapper mdw(md);
    std::string s;

    // Print strides if non-dense descriptor with defined dims/strides was
    // provided.
    if (mdw.has_runtime_dims_or_strides() || mdw.is_dense(true)) return s;

    // Note: there's no API to create a memory desc with strides and blocks
    // together, thus, this non-plain md will dump strides for info purpose.
    s += md2dim_str(md, dims_type_t::strides);
    return s;
}

// Forms a format string for a given memory descriptor.
//
// There are two formats:
// - dense: defined as: 'dt:[a|p|o|0]:fmt_kind:fmt:strides:extra'.
// - sparse: defined as: 'dt:[a|p|o|0]:fmt_kind:encoding:extra'.
// Here:
//  - dt       -- data type
//  - a        -- indicates memory desc was created with fmt_kind `any`.
//  - p        -- indicates there is non-trivial padding
//  - o        -- indicates there is non-trivial padding offset
//  - 0        -- indicates there is non-trivial offset0
//  - fmt_kind -- format kind (blocked, wino, etc...)
//  - encoding -- [sparse_desc only] sparse encoding (csr, etc...)
//  - fmt      -- [blocking_desc only] extended format string
//  - strides  -- [blocking_desc only] non-dense strides string (dims style)
//  - extra    -- shows extra fields (underspecified)
//
// Note: `user_format` is an information that is not available in memory descs
// from `pd` since those are initialized by implementations. The knowledge about
// original user format specified is kept in pd->desc()->xxx_desc and this is
// the info provided to this call.
// On the other hand, just a user memory descriptor can't be passed because it
// is not initialized b the library, and format information will be missed.
std::string md2fmt_str(
        const char *name, const memory_desc_t *md, format_kind_t user_format) {
    stringstream_t ss;
    ss << name << ":";
    if (!md || types::is_zero_md(md)) {
        ss << data_type::undef << "::" << format_kind::undef << ":::";
        return ss.str();
    }

    memory_desc_wrapper mdw(md);
    ss << mdw.data_type() << ":";

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < mdw.ndims(); ++d) {
        if (mdw.dims()[d] != mdw.padded_dims()[d]) padded_dims = true;
        if (mdw.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = mdw.offset0();
    ss << (user_format == format_kind::any ? "a" : "");
    ss << (padded_dims ? "p" : "");
    ss << (padded_offsets ? "o" : "");
    ss << (offset0 ? "0" : "");
    ss << ":" << mdw.format_kind();

    // Cast is required to pass through compiler error:
    // error: case value ‘256’ not in enumerated type
    // ‘dnnl::impl::format_kind_t’ {aka ‘dnnl_format_kind_t’}
    switch (static_cast<int>(mdw.format_kind())) {
        case format_kind::blocked:
            ss << ":" << md2fmt_tag_str(md) << ":" << md2fmt_strides_str(md);
            break;
        case format_kind::cublaslt_blocked: ss << cublasltfmt2str(md); break;
        case format_kind::wino:
        case format_kind::rnn_packed:
        case format_kind::zen_packed:
        case format_kind::opaque: ss << "::"; break;
        case format_kind::sparse: ss << ":" << mdw.encoding();
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
            if (mdw.is_grouped_desc()) {
                ss << ":" << mdw.sparse_desc().grouped_desc.variable_dim_idx;
                ss << ":" << mdw.sparse_desc().grouped_desc.group_count;
            }
#endif
            ss << ":";
            break;
        case format_kind::any: ss << ":any:"; break;
        default:
            assert(!"unsupported format_kind");
            ss << "::";
            break;
    }

    ss << mdw.extra();

    return ss.str();
}

// Puts memory_desc information into stream without dimensions
std::ostream &operator<<(std::ostream &ss, const memory_desc_t *md) {
    assert(!"unexpected call to the operator<<");
    return ss;
}

template <typename T>
static std::string get_val_str(T val) {
    static_assert(
            std::is_arithmetic<T>::value, "T must be an arithmetic type.");
    if (is_runtime_value(val)) return std::string("*");
    return std::to_string(val);
}

// Returns string with dimensions from a given memory descriptor.
// The format is defined as: dim0xdim1x...xdimN, with RT values signed as `*`.
std::string md2dim_str(const memory_desc_t *md, dims_type_t dims_type) {
    if (md == nullptr || md->ndims == 0) return "";

    memory_desc_wrapper mdw(md);
    std::string s;

    assert(dims_type == dims_type_t::dims || dims_type == dims_type_t::strides);
    const auto &dims_obj
            = dims_type == dims_type_t::dims ? mdw.dims() : mdw.strides();

    s += get_val_str(dims_obj[0]);
    for (int d = 1; d < mdw.ndims(); ++d)
        s += ("x" + get_val_str(dims_obj[d]));

    return s;
}

// Returns string with descriptor style from memory_desc.
std::string md2desc_str(const memory_desc_t *md) {
    const auto dims = md->dims;
    std::string s;
    if (md->ndims >= 6) return md2dim_str(md);

    if (md->ndims == 1) {
        s += "x" + std::to_string(dims[0]);
        return s;
    }

    s += "mb" + std::to_string(dims[0]) + "ic" + std::to_string(dims[1]);
    if (md->ndims >= 5) s += "id" + std::to_string(dims[md->ndims - 3]);
    if (md->ndims >= 4) s += "ih" + std::to_string(dims[md->ndims - 2]);
    if (md->ndims >= 3) s += "iw" + std::to_string(dims[md->ndims - 1]);
    return s;
}

std::ostream &operator<<(
        std::ostream &ss, const rnn_create_time_scales_t &rnn_scales) {
    ss << rnn_scales.mask_;
    const float val = rnn_scales.scales_[0];
    // Can't use scientific flags since it breaks parsing on converter and
    // benchdnn side.
    if (rnn_scales.mask_ == 0 || is_runtime_value(val))
        ss << ":" << get_val_str(val);
    return ss;
}

namespace {
int get_runtime_mask(const memory_desc_t *md) {
    int mask = 0;
    for (int d = md->ndims - 1; d >= 0; --d) {
        mask += is_runtime_value(md->dims[d]) ? 1 << d : 0;
    }
    return mask;
}

int get_arg_index(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return arg - DNNL_ARG_MULTIPLE_SRC;
    switch (arg) {
        case DNNL_ARG_SRC_0: return 0;
        case DNNL_ARG_SRC_1: return 1;
        case DNNL_ARG_SRC_2: return 2;
        default: return -1;
    }
    return -1;
}

std::string get_arg(int arg) {
    if (arg & DNNL_ARG_MULTIPLE_SRC) return "msrc";

    std::string s;
    switch (arg) {
        case DNNL_ARG_SRC: // DNNL_ARG_SRC_0
        case DNNL_ARG_SRC_1:
        case DNNL_ARG_SRC_2: s = "src"; break;
        case DNNL_ARG_DST: s = "dst"; break;
        case DNNL_ARG_WEIGHTS: // DNNL_ARG_WEIGHTS_0
        case DNNL_ARG_WEIGHTS_1:
        case DNNL_ARG_WEIGHTS_2: s = "wei"; break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_DST:
            s = "attr_post_op_dw_dst";
            break;
        case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
            s = "attr_post_op_dw_wei";
            break;
        default: assert(!"unsupported arg"); s = "unsupported arg";
    }
    return s;
}
} // namespace

std::string arg2str(int arg) {
    std::string s = get_arg(arg);
    const int idx = get_arg_index(arg);
    if (idx != -1) s += std::to_string(idx);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const primitive_attr_t *attr) {
    struct {
        const char *operator()() {
            current[0] = next;
            next = ' ';
            return current;
        }

    private:
        char current[2] = {};
        char next = 0;
    } field_delim;

    std::string empty_delim, attr_delim = "+";

    // scratchpad and fpmath mode are not a part of
    // has_default_values(). Check them first.
    const scratchpad_mode_t &spm = attr->scratchpad_mode_;
    if (spm != scratchpad_mode_t::dnnl_scratchpad_mode_library) {
        ss << field_delim()
           << "attr-scratchpad:" << dnnl_scratchpad_mode2str(spm);
    }
    const fpmath_t &fpm = attr->fpmath_;
    if (fpm.mode_ != fpmath_mode_t::dnnl_fpmath_mode_strict
            || fpm.apply_to_int_) {
        ss << field_delim()
           << "attr-fpmath:" << dnnl_fpmath_mode2str(fpm.mode_);
        if (fpm.apply_to_int_) ss << ":true";
    }

    const accumulation_mode_t &am = attr->acc_mode_;
    if (am != accumulation_mode::strict) {
        ss << field_delim()
           << "attr-acc-mode:" << dnnl_accumulation_mode2str(am);
    }

    const auto &rm = attr->rounding_mode_;
    if (!rm.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-rounding-mode:";
        for (const auto &e : rm.rounding_modes_map_) {
            // TODO: add support for diff tensors in arg2str when
            // support is added
            if (!rm.has_default_values(e.first))
                ss << delim << arg2str(e.first) << ":"
                   << dnnl_rounding_mode2str(e.second);
            delim = attr_delim;
        }
    }

    const bool deterministic = attr->deterministic_;
    if (deterministic) {
        ss << field_delim() << "attr-deterministic:" << deterministic;
    }

    // Fast exit if rest attributes were not specified.
    if (attr->has_default_values()) return ss;

    const scales_t &scales = attr->scales_;
    if (!scales.has_default_values()) {
        ss << field_delim() << "attr-scales:" << scales.get_verbose();
    }

    const zero_points_t &zero_points = attr->zero_points_;
    if (!zero_points.has_default_values()) {
        ss << field_delim() << "attr-zero-points:" << zero_points.get_verbose();
    }

    const precomputed_reductions_t &pr = attr->precomputed_reductions_;
    if (!pr.has_default_values()) {
        ss << field_delim()
           << "attr-precomputed-reductions:" << pr.get_verbose();
    }

    const post_ops_t &po = attr->post_ops_;
    if (!po.has_default_values()) {
        std::string delim = empty_delim;
        ss << field_delim() << "attr-post-ops:";
        for (int i = 0; i < po.len(); ++i) {
            const post_ops_t::entry_t &e = po.entry_[i];
            switch (e.kind) {
                case primitive_kind::sum: {
                    const auto &s = e.sum;
                    ss << delim << "sum";
                    if (s.scale != 1.f || s.zero_point != 0
                            || s.dt != data_type::undef)
                        ss << ":" << s.scale;
                    if (s.zero_point != 0 || s.dt != data_type::undef)
                        ss << ":" << s.zero_point;
                    if (s.dt != data_type::undef) ss << ":" << s.dt;
                } break;
                case primitive_kind::convolution: {
                    using namespace data_type;
                    const auto &c = e.depthwise_conv;
                    ss << delim << "dw:k" << c.kernel << "s" << c.stride << "p"
                       << c.padding;
                    if (c.wei_dt == s8 || c.dst_dt != f32)
                        ss << ":" << c.dst_dt;
                } break;
                case primitive_kind::eltwise: {
                    const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                    ss << delim << ew.alg;
                    if (ew.alpha != 0.f || ew.beta != 0.f || ew.scale != 1.f)
                        ss << ":" << ew.alpha;
                    if (ew.beta != 0.f || ew.scale != 1.f) ss << ":" << ew.beta;
                    if (ew.scale != 1.f) ss << ":" << ew.scale;
                } break;
                case primitive_kind::binary: {
                    const post_ops_t::entry_t::binary_t &eb = e.binary;
                    const auto &md = eb.user_src1_desc;
                    ss << delim << eb.alg << ":" << md.data_type;

                    auto dump = [&](const memory_desc_t &md) {
                        int mask = 0;
                        for (int d = 0; d < md.ndims; ++d)
                            mask += md.dims[d] != 1 ? (1 << d) : 0;
                        ss << ":" << mask;
                        const memory_desc_wrapper mdw(md);
                        switch (mdw.format_kind()) {
                            case format_kind::blocked:
                                if (!mdw.count_non_unit_dims(1)) {
                                    ss << ":" << md2fmt_tag_str(&eb.src1_desc);
                                    const auto &strides_str
                                            = md2fmt_strides_str(&eb.src1_desc);
                                    if (!strides_str.empty())
                                        ss << ":" << strides_str;
                                }
                                break;
                            case format_kind::sparse:
                                if (mdw.is_grouped_desc()) {
                                    ss << ":grouped";
                                    break;
                                }
                                assert(!"unsupported sparse encoding");
                                break;
                            case format_kind::any: ss << ":any"; break;
                            default: assert(!"unsupported format_kind");
                        }
                    };

                    dump(md);
                    if (e.is_binary_with_ternary_op()) dump(eb.user_src2_desc);
                } break;
                case primitive_kind::prelu: {
                    const auto &ep = e.prelu;
                    ss << delim << "prelu" << ":" << ep.mask;
                } break;
                default: assert(!"unsupported post op primitive kind!"); break;
            }
            delim = attr_delim;
        }
    }

    const rnn_data_qparams_t &rnn_qp = attr->rnn_data_qparams_;
    if (!rnn_qp.has_default_values()) {
        ss << field_delim() << "rnn_data_qparams:" << rnn_qp.scale_ << ":"
           << rnn_qp.shift_ << ";";
    }

    if (!attr->dropout_.has_default_values()) {
        ss << field_delim() << "attr-dropout";
        const memory_desc_wrapper mdw(attr->dropout_.user_dropout_desc_);
        switch (mdw.format_kind()) {
            case format_kind::blocked:
                if (!mdw.count_non_unit_dims(1))
                    ss << ":" << md2fmt_tag_str(&attr->dropout_.dropout_desc_);
                break;
            case format_kind::any: ss << ":any"; break;
            case format_kind::undef: ss << ":undef"; break;
            default: assert(!"unsupported format_kind");
        }
        ss << ":" << dnnl_dt2str(attr->dropout_.seed_dt_);
        ss << ":" << attr->dropout_.use_offset_;
        ss << ":" << attr->dropout_.use_host_scalars_;
    }
    return ss;
}

std::string attr2str(const primitive_attr_t *attr) {
    stringstream_t ss;
    ss << attr;
    return ss.str();
}

/* init_info section */
namespace {

template <typename pd_t>
std::string init_info_batch_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();

    ss << md2fmt_str("data", src_md, pd->src_md(0, true)->format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_binary(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src0_md = pd->invariant_src_md(0);
    auto src1_md = pd->invariant_src_md(1);
    auto src2_md = pd->invariant_src_md(2);
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src0_md, pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str("src", src1_md, pd->invariant_src_user_format_kind(1))
       << " ";
    if (pd->desc()->alg_kind == alg_kind_t::dnnl_binary_select) {
        ss << md2fmt_str("src", src2_md, pd->invariant_src_user_format_kind(2))
           << " ";
    }

    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2dim_str(src0_md) << ":" << md2dim_str(src1_md);
    if (pd->desc()->alg_kind == alg_kind_t::dnnl_binary_select)
        ss << ":" << md2dim_str(src2_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_concat(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << md2fmt_str("src", src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->desc()->concat_dimension << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << md2dim_str(src_i_md);
        if (i < pd->n_inputs() - 1) ss << ":";
    }

    return ss.str();
}

template <typename pd_t>
std::string init_info_convolution(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("wei", wei_md, pd->invariant_wei_user_format_kind())
       << " ";
    ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind())
       << " ";

    // `has_fused_dw` modifies the convolution output in the following way:
    // * It provides additional src, wei and bia md to show their presence and
    //   wei format (for convenience, it can't be used directly).
    // * It makes output spatial dimensions same as input since first conv is
    //   always 1x1 for such fusion. This is to make a problem benchdnn
    //   compatible.
    // * Note: Queried `dst_md` with final dimensions after fusion will reside
    //   in fused conv pd. The op_desc for it is created on the library side and
    //   filled with already blocked formats compatible with precedeing 1x1
    //   convolution. It means that it can't identify if original dst_md was
    //   created with `format_kind::any` or not. For purposes of re-construction
    //   of benchdnn line, intermediate "src_fused" is required - it gives an
    //   info about data type to pass into benchdnn and also whether 1x1 conv
    //   dst was created with `format_kind::any`.
    // * Note: DW-post op is the only reason why `arg_md` got `user_input`
    //   argument and also takes argument. This is due to dw post-op mds can
    //   be queried only through `arg_md` interface.
    const bool has_fused_dw
            = pd->attr()->post_ops_.find(primitive_kind::convolution) >= 0;
    if (has_fused_dw) {
        auto src_fused_md = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC);
        auto wei_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        auto bia_fused_md
                = pd->arg_md(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);
        // User-provided dst memory descriptor.
        ss << md2fmt_str("src_fused", src_fused_md,
                pd->invariant_dst_user_format_kind(
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_SRC))
           << " ";
        // Not user-provided memory descriptors.
        ss << md2fmt_str("wei_fused", wei_fused_md, format_kind::undef) << " ";
        ss << md2fmt_str("bia_fused", bia_fused_md, format_kind::undef) << " ";
        ss << md2fmt_str("dst", dst_md, format_kind::undef);
    } else {
        ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    if (pd->with_groups()) ss << "g" << pd->G();
    ss << "mb" << pd->MB() << "_" << "ic" << pd->IC() << "oc" << pd->OC()
       << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << (has_fused_dw ? pd->ID() : pd->OD())
           << "kd" << pd->KD() << "sd" << pd->KSD() << "dd" << pd->KDD() << "pd"
           << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << (has_fused_dw ? pd->IH() : pd->OH())
           << "kh" << pd->KH() << "sh" << pd->KSH() << "dh" << pd->KDH() << "ph"
           << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << (has_fused_dw ? pd->IW() : pd->OW())
       << "kw" << pd->KW() << "sw" << pd->KSW() << "dw" << pd->KDW() << "pw"
       << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_deconvolution(const engine_t *e, const pd_t *pd) {
    return init_info_convolution(e, pd);
}

template <typename pd_t>
std::string init_info_eltwise(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->use_dst() ? pd->dst_md(0) : pd->src_md(0);
    auto user_data_format_kind = pd->use_dst()
            ? pd->dst_md(0, true)->format_kind
            : pd->src_md(0, true)->format_kind;
    auto diff_src_md = pd->diff_src_md();

    ss << md2fmt_str("data", data_md, user_data_format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", diff_src_md,
                      pd->invariant_src_user_format_kind(0));
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " alpha:" << pd->desc()->alpha
       << " beta:" << pd->desc()->beta << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_gated_mlp(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    ss << md2fmt_str("src", pd->arg_md(DNNL_ARG_SRC), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_gate", pd->arg_md(DNNL_ARG_WEIGHTS_GATE), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_up", pd->arg_md(DNNL_ARG_WEIGHTS_UP), format_kind::undef)
       << " ";
    ss << md2fmt_str(
            "wei_down", pd->arg_md(DNNL_ARG_WEIGHTS_DOWN), format_kind::undef)
       << " ";
    ss << md2fmt_str("dst", pd->arg_md(DNNL_ARG_DST), format_kind::undef);

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->activation() << ",";
    ss << "mb" << pd->MB() << "ic" << pd->IC() << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_gemm(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_a_md = pd->invariant_src_md(0);
    auto src_b_md = pd->invariant_src_md(1);
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    ss << md2fmt_str("src_a", src_a_md, pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str("src_b", src_b_md, pd->invariant_src_user_format_kind(1))
       << " ";
    if (pd->with_bias()) {
        ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind());
        ss << "_mask" << get_bia_mask();
        ss << " ";
    }
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2dim_str(src_a_md) << ":" << md2dim_str(src_b_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_group_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("src", src_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff_src", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << "g" << pd->desc()->groups << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_inner_product(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("wei", wei_md, pd->invariant_wei_user_format_kind())
       << " ";
    ss << md2fmt_str("bia", bia_md, pd->invariant_bia_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";

    ss << md2desc_str(src_md);
    ss << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_layer_normalization(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->invariant_dst_md();
    auto stats_md = pd->is_fwd() && !pd->stats_are_src() ? pd->dst_md(1)
                                                         : pd->src_md(1);
    auto user_stats_format_kind = pd->is_fwd() && !pd->stats_are_src()
            ? pd->dst_md(1, true)->format_kind
            : pd->src_md(1, true)->format_kind;
    auto scaleshift_md = pd->weights_md(0);
    auto diff_scaleshift_md = pd->diff_weights_md(0);

    ss << md2fmt_str("src", src_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind())
       << " ";
    ss << md2fmt_str("stats", stats_md, user_stats_format_kind);
    if (pd->use_scale()) {
        ss << " "
           << md2fmt_str("scale", scaleshift_md,
                      pd->weights_md(0, true)->format_kind);
    }
    if (pd->use_shift()) {
        ss << " "
           << md2fmt_str("shift", scaleshift_md,
                      pd->weights_md(0, true)->format_kind);
    }
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff_src", pd->diff_src_md(0),
                      pd->diff_src_md(0, true)->format_kind);
    }
    if (!pd->is_fwd() && pd->use_scale()) {
        ss << " "
           << md2fmt_str("diff_scale", diff_scaleshift_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }
    if (!pd->is_fwd() && pd->use_shift()) {
        ss << " "
           << md2fmt_str("diff_shift", diff_scaleshift_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "flags:" << normalization_flags2str(pd->desc()->flags) << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_lrn(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();

    ss << md2fmt_str("data", data_md, pd->src_md(0, true)->format_kind);
    if (!pd->is_fwd()) {
        ss << " "
           << md2fmt_str("diff", diff_src_md,
                      pd->invariant_src_user_format_kind(0));
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2desc_str(data_md);
    ss << "ls" << pd->desc()->local_size << "beta" << pd->desc()->lrn_beta;

    return ss.str();
}

std::string mds2str_matmul(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *wei_md,
        format_kind_t wei_user_format_kind, const memory_desc_t *bia_md,
        format_kind_t bia_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    stringstream_t ss;

    ss << md2fmt_str("src", src_md, src_user_format_kind) << " ";
    ss << md2fmt_str("wei", wei_md, wei_user_format_kind) << " ";
    if (!memory_desc_wrapper(bia_md).is_zero()) {
        ss << md2fmt_str("bia", bia_md, bia_user_format_kind);
        ss << "_mask" << get_bia_mask();
        ss << " ";
    }
    ss << md2fmt_str("dst", dst_md, dst_user_format_kind);

    std::string s = ss.str();
    return s;
}

std::string dims2fmt_str_matmul(
        const memory_desc_t *src_md, const memory_desc_t *wei_md) {
    return md2dim_str(src_md) + ":" + md2dim_str(wei_md);
}

template <typename pd_t>
std::string init_info_matmul(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto wei_md = pd->invariant_wei_md();
    auto bia_md = pd->invariant_bia_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_matmul(src_md, pd->invariant_src_user_format_kind(), wei_md,
            pd->invariant_wei_user_format_kind(), bia_md,
            pd->invariant_bia_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime_dims_masks:" << get_runtime_mask(src_md) << ":"
           << get_runtime_mask(wei_md);
    }
    ss << "," << dims2fmt_str_matmul(src_md, wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_pooling(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();
    auto ws_md = pd->workspace_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());
    if (!memory_desc_wrapper(ws_md).is_zero()) {
        ss << " " << md2fmt_str("ws", ws_md, format_kind::undef);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->IC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << pd->OD() << "kd" << pd->KD() << "sd"
           << pd->KSD() << "dd" << pd->KDD() << "pd" << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << pd->OH() << "kh" << pd->KH() << "sh"
           << pd->KSH() << "dh" << pd->KDH() << "ph" << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW() << "kw" << pd->KW() << "sw"
       << pd->KSW() << "dw" << pd->KDW() << "pw" << pd->padL();

    return ss.str();
}

template <typename pd_t>
std::string init_info_prelu(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md(0);
    auto wei_md = pd->weights_md(0);
    auto diff_data_md = pd->diff_src_md(0);
    auto diff_wei_md = pd->diff_weights_md(0);

    ss << md2fmt_str("data", data_md, pd->src_md(0, true)->format_kind) << " ";
    ss << md2fmt_str("wei", wei_md, pd->weights_md(0, true)->format_kind);
    if (!memory_desc_wrapper(diff_data_md).is_zero()) {
        ss << " "
           << md2fmt_str("diff", diff_data_md,
                      pd->diff_src_md(0, true)->format_kind);
    }
    if (!memory_desc_wrapper(diff_wei_md).is_zero()) {
        ss << " "
           << md2fmt_str("diff_wei", diff_wei_md,
                      pd->diff_weights_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(data_md) << ":" << md2dim_str(wei_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_reduction(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " p:" << pd->desc()->p
       << " eps:" << pd->desc()->eps << ",";
    ss << md2dim_str(src_md) << ":" << md2dim_str(dst_md);

    return ss.str();
}

std::string mds2str_reorder(const memory_desc_t *src_md,
        format_kind_t src_user_format_kind, const memory_desc_t *dst_md,
        format_kind_t dst_user_format_kind) {
    std::string s;
    s += md2fmt_str("src", src_md, src_user_format_kind);
    s += " ";
    s += md2fmt_str("dst", dst_md, dst_user_format_kind);
    return s;
}

std::string dims2fmt_str_reorder(const memory_desc_t *src_md) {
    return md2dim_str(src_md);
}

template <typename pd_t>
std::string init_info_reorder(const engine_t *e, pd_t *pd) {
    stringstream_t ss;

    const auto src_ek = pd->desc()->src_engine_kind;
    const auto dst_ek = pd->desc()->dst_engine_kind;

    if (src_ek != dst_ek)
        ss << src_ek << "2" << dst_ek;
    else
        ss << e;

    ss << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << mds2str_reorder(src_md, pd->invariant_src_user_format_kind(), dst_md,
            pd->invariant_dst_user_format_kind());
    ss << "," << pd->attr() << ",";

    if (pd->has_runtime_dims_or_strides()) {
        ss << "runtime-dim-mask:" << get_runtime_mask(src_md);
    }
    ss << "," << dims2fmt_str_reorder(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_resampling(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->invariant_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->C() << "_";
    if (pd->ndims() >= 5) ss << "id" << pd->ID() << "od" << pd->OD() << "_";
    if (pd->ndims() >= 4) ss << "ih" << pd->IH() << "oh" << pd->OH() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW();

    return ss.str();
}

template <typename pd_t>
std::string init_info_rnn(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    // TODO: shorten the names to consume fewer characters on verbose output.
    ss << md2fmt_str(
            "src_layer", pd->src_md(0), pd->src_md(0, true)->format_kind)
       << " ";
    if (pd->with_src_iter())
        ss << md2fmt_str(
                "src_iter", pd->src_md(1), pd->src_md(1, true)->format_kind)
           << " ";
    ss << md2fmt_str("wei_layer", pd->weights_md(0),
            pd->weights_md(0, true)->format_kind)
       << " ";
    ss << md2fmt_str(
            "wei_iter", pd->weights_md(1), pd->weights_md(1, true)->format_kind)
       << " ";
    if (pd->is_lstm_peephole())
        ss << md2fmt_str("wei_peephole", pd->weights_md(2),
                pd->weights_md(2, true)->format_kind)
           << " ";
    // TODO: separate methods for aux weights?
    if (pd->is_lstm_projection()) {
        auto proj_idx = 2 + pd->is_lstm_peephole();
        ss << md2fmt_str("wei_proj", pd->weights_md(proj_idx),
                pd->weights_md(proj_idx, true)->format_kind)
           << " ";
    }
    if (pd->with_bias()) {
        auto bias_idx = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
        ss << md2fmt_str("bias", pd->weights_md(bias_idx),
                pd->weights_md(bias_idx, true)->format_kind)
           << " ";
    }
    ss << md2fmt_str(
            "dst_layer", pd->dst_md(0), pd->dst_md(0, true)->format_kind);
    if (pd->with_dst_iter())
        ss << " "
           << md2fmt_str("dst_iter", pd->dst_md(1),
                      pd->dst_md(1, true)->format_kind);

    if (!pd->is_fwd()) {
        ss << " ";
        ss << md2fmt_str("diff_src_layer", pd->diff_src_md(0),
                pd->diff_src_md(0, true)->format_kind)
           << " ";
        if (pd->with_src_iter())
            ss << md2fmt_str("diff_src_iter", pd->diff_src_md(1),
                    pd->diff_src_md(1, true)->format_kind)
               << " ";
        ss << md2fmt_str("diff_wei_layer", pd->diff_weights_md(0),
                pd->diff_weights_md(0, true)->format_kind)
           << " ";
        ss << md2fmt_str("diff_wei_iter", pd->diff_weights_md(1),
                pd->diff_weights_md(1, true)->format_kind)
           << " ";
        if (pd->is_lstm_peephole())
            ss << md2fmt_str("diff_wei_peephole", pd->diff_weights_md(2),
                    pd->diff_weights_md(2, true)->format_kind)
               << " ";
        if (pd->is_lstm_projection()) {
            auto proj_idx = 2 + pd->is_lstm_peephole();
            ss << md2fmt_str("diff_wei_proj", pd->weights_md(proj_idx),
                    pd->weights_md(proj_idx, true)->format_kind)
               << " ";
        }
        if (pd->with_bias()) {
            auto bias_idx
                    = 2 + pd->is_lstm_peephole() + pd->is_lstm_projection();
            ss << md2fmt_str("diff_bias", pd->weights_md(bias_idx),
                    pd->weights_md(bias_idx, true)->format_kind)
               << " ";
        }
        ss << md2fmt_str("diff_dst_layer", pd->diff_dst_md(0),
                pd->diff_dst_md(0, true)->format_kind);
        if (pd->with_dst_iter())
            ss << " "
               << md2fmt_str("diff_dst_iter", pd->diff_dst_md(1),
                          pd->diff_dst_md(1, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->cell_kind()
       << " direction:" << dnnl_rnn_direction2str(pd->direction())
       << " activation:" << pd->activation_kind()
       << " flags:" << rnn_flags2str(pd->desc()->flags) << ",";

    ss << "l" << pd->L() << "t" << pd->T() << "mb" << pd->MB() << "sic"
       << pd->SIC() << "slc" << pd->SLC() << "dhc" << pd->DHC() << "dic"
       << pd->DIC();

    return ss.str();
}

template <typename pd_t>
std::string init_info_shuffle(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->invariant_src_md();

    ss << md2fmt_str("data", data_md, pd->invariant_src_user_format_kind());

    ss << "," << pd->attr() << ",";
    ss << "axis:" << pd->axis() << " group:" << pd->group_size() << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_softmax(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->invariant_src_md();
    auto dst_md = pd->dst_md();
    auto diff_dst_md = pd->diff_dst_md();

    ss << md2fmt_str("src", src_md, pd->invariant_src_user_format_kind())
       << " ";
    ss << md2fmt_str("dst", dst_md, pd->dst_md(0, true)->format_kind);
    if (!types::is_zero_md(diff_dst_md)) {
        ss << " "
           << md2fmt_str("diff_dst", diff_dst_md,
                      pd->diff_dst_md(0, true)->format_kind);
    }

    ss << "," << pd->attr() << ",";
    ss << "alg:" << pd->alg_kind() << " axis:" << pd->axis() << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_sum(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->invariant_src_md(i);
        ss << md2fmt_str("src", src_i_md, pd->invariant_src_user_format_kind(i))
           << " ";
    }
    auto dst_md = pd->invariant_dst_md();
    ss << md2fmt_str("dst", dst_md, pd->invariant_dst_user_format_kind());

    ss << "," << pd->attr() << ",,";
    ss << md2dim_str(dst_md);

    return ss.str();
}

template <typename pd_t>
std::string init_info_sdpa(const engine_t *e, const pd_t *pd) {
    stringstream_t ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    const sdpa_desc_t *desc = pd->desc();
    ss << md2fmt_str(
            "query", desc->qry_md(), pd->invariant_src_user_format_kind(0))
       << " ";
    ss << md2fmt_str(
            "key", desc->key_md(), pd->invariant_src_user_format_kind(1))
       << " ";
    ss << md2fmt_str(
            "val", desc->val_md(), pd->invariant_src_user_format_kind(2))
       << " ";
    if (pd->with_attn_mask())
        ss << md2fmt_str("msk", desc->attn_mask_md(),
                pd->invariant_src_user_format_kind(3))
           << " ";
    ss << md2fmt_str("dst", pd->dst_md(), pd->invariant_dst_user_format_kind())
       << ",";

    std::string delimiter;
    if (pd->with_key_scales() || pd->with_value_scales()) {
        ss << delimiter << "attr-scales:";
        delimiter = "";
        if (pd->with_key_scales()) {
            ss << delimiter << "key:" << desc->kq_scales;
            delimiter = "+";
        }
        if (pd->with_value_scales()) {
            ss << delimiter << "val:" << desc->vs_scales;
            delimiter = "+";
        }
        delimiter = " ";
    }
    if (pd->with_key_zp() || pd->with_value_zp()) {
        ss << delimiter << "attr-zero-points:";
        delimiter = "";
        if (pd->with_key_zp()) {
            ss << delimiter << "key:" << desc->kq_zero_points;
            delimiter = "+";
        }
        if (pd->with_value_zp()) {
            ss << delimiter << "val:" << desc->vs_zero_points;
            delimiter = "+";
        }
        delimiter = " ";
    }
    ss << delimiter << pd->attr();

    delimiter = " ";
    ss << ",alg:" << desc->softmax_alg;
    if (pd->with_attn_mask()) {
        auto *md = desc->attn_mask_md();
        ss << delimiter << "msk:buffer_" << (md->dims[2] == 1 ? 1 : 2) << 'd';
    } else if (pd->with_causal_mask()) {
        ss << delimiter;
        if (desc->mask_type == attn_mask_type::top_left)
            ss << "msk:causal_top_left";
        else
            ss << "msk:causal_bottom_right";
    }
    if (pd->with_attn_scale()) {
        ss << delimiter << "scl:";
        if (desc->invert_scale)
            ss << "div:";
        else
            ss << "mul:";
        ss << dnnl_dt2str(desc->scale_md()->data_type) << ":";
        if (pd->with_host_scale())
            ss << "host";
        else
            ss << "device";
    }

    ss << "," << md2dim_str(desc->qry_md()) << ":" << md2dim_str(desc->key_md())
       << ":" << md2dim_str(desc->val_md());

    return ss.str();
}

} // namespace

std::string rt_mds2str(primitive_kind_t prim_kind, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bia_md,
        const memory_desc_t *dst_md) {
    // Note: pass format_kind::undef since runtime dims-ed mds can't have
    // format_kind::any at any stage.
    std::string s;
#if defined(DISABLE_VERBOSE)
    return s;
#endif

    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = mds2str_matmul(src_md, format_kind::undef, wei_md,
                    format_kind::undef, bia_md, format_kind::undef, dst_md,
                    format_kind::undef);
            break;
        case primitive_kind::reorder:
            s = mds2str_reorder(
                    src_md, format_kind::undef, dst_md, format_kind::undef);
            break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

// Designated function to prepend a verbose marker and correspondent version.
// Note: intended to be called inside `verbose_printf_impl` only!
std::string prepend_identifier_and_version(const char *fmt_str) {
    assert(std::string(fmt_str).find("onednn_verbose") == std::string::npos);
    std::string s
            = "onednn_verbose," + std::string(verbose_version) + "," + fmt_str;
    return s;
}

void verbose_printf_impl(const char *raw_fmt_str, verbose_t::flag_kind kind) {
#if defined(DISABLE_VERBOSE)
    return;
#endif

    if (get_verbose(kind)) print_header();

    const auto &fmt_str = prepend_identifier_and_version(raw_fmt_str);

#ifdef DNNL_EXPERIMENTAL_LOGGING
    const log_manager_t &log_manager = log_manager_t::get_log_manager();

    if (log_manager.is_logger_enabled()) {
        static const std::map<verbose_t::flag_kind, log_manager_t::log_level_t>
                verbose_to_log_map {
                        {verbose_t::all, log_manager_t::trace},
                        {verbose_t::debuginfo, log_manager_t::debug},
                        {verbose_t::level1, log_manager_t::info},
                        {verbose_t::level2, log_manager_t::info},
                        {verbose_t::create_dispatch, log_manager_t::info},
                        {verbose_t::create_check, log_manager_t::info},
                        {verbose_t::create_profile, log_manager_t::info},
                        {verbose_t::profile_externals, log_manager_t::info},
                        {verbose_t::exec_profile, log_manager_t::info},
                        {verbose_t::exec_check, log_manager_t::error},
                        {verbose_t::error, log_manager_t::critical},
                        {verbose_t::warn, log_manager_t::warn},
                        {verbose_t::none, log_manager_t::off},
                };

        log_manager.log(fmt_str.c_str(), verbose_to_log_map.at(kind));
    }
    if (log_manager.is_console_enabled()) {
        printf("%s", fmt_str.c_str());
        fflush(stdout);
    }
#else
    printf("%s", fmt_str.c_str());
    fflush(stdout);
#endif
}

std::string rt_dims2fmt_str(primitive_kind_t prim_kind,
        const memory_desc_t *src_md, const memory_desc_t *wei_md,
        const memory_desc_t *dst_md) {
    std::string s;
#if defined(DISABLE_VERBOSE)
    return s;
#endif

    switch ((int)prim_kind) {
        case primitive_kind::matmul:
            s = dims2fmt_str_matmul(src_md, wei_md);
            break;
        case primitive_kind::reorder: s = dims2fmt_str_reorder(src_md); break;

        case primitive_kind::batch_normalization:
        case primitive_kind::binary:
        case primitive_kind::concat:
        case primitive_kind::convolution:
        case primitive_kind::deconvolution:
        case primitive_kind::eltwise:
        case primitive_kind::inner_product:
        case primitive_kind::layer_normalization:
        case primitive_kind::lrn:
        case primitive_kind::pooling:
        case primitive_kind::prelu:
        case primitive_kind::reduction:
        case primitive_kind::resampling:
        case primitive_kind::rnn:
        case primitive_kind::shuffle:
        case primitive_kind::softmax:
        case primitive_kind::sum: assert(!"unsupported primitive kind"); break;
        default: assert(!"unknown primitive kind");
    }
    return s;
}

void pd_info_t::init(const engine_t *engine, const primitive_desc_t *pd) {
    // Handles VERBOSE_DISABLE since `is_initialized_` is set to `true`.
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
    // clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        str_ = init_info_##kind(engine, (const kind##_pd_t *)pd); \
        break

        switch ((int)pd->kind()) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(concat);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(gated_mlp);
            CASE(gemm);
            CASE(group_normalization);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(matmul);
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            CASE(sum);
            CASE(sdpa);
            case primitive_kind::zero_pad:
              str_ = "zero_pad, unknown info";
              break;
            default:
              str_ = "unknown primitive info";
              assert(!"unknown primitive kind");
        }
#undef CASE
        // clang-format on

        is_initialized_ = true;
    });
}

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_set_verbose(int level) {
    using namespace dnnl::impl::status;
    using namespace dnnl::impl;
    if (level < 0 || level > 2) return invalid_arguments;

    uint32_t verbose_level = verbose_t::none;
    if (level == 1) verbose_level = verbose_t::level1;
    if (level == 2) verbose_level = verbose_t::level2;
    // we put the lower byte of level as devinfo to preserve backward
    // compatibility with historical VERBOSE={1,2}
    if (level == 1 || level == 2) verbose_level |= (level << 24);
    verbose.set(verbose_level);

#ifdef DNNL_EXPERIMENTAL_LOGGING
    // if logging is enabled, this also updates the logging level for the outputs
    const log_manager_t &log_manager = log_manager_t::get_log_manager();
    if (log_manager.is_logger_enabled())
        log_manager.set_log_level(std::to_string(level));
#endif

    return success;
}

dnnl_status_t dnnl_verbose_profiling_enabled(int *enabled) {
    using namespace dnnl::impl::status;
    using namespace dnnl::impl;
    if (enabled == nullptr) return invalid_arguments;

    const uint32_t profiling
            = verbose_t::create_profile | verbose_t::exec_profile;
    *enabled = get_verbose(static_cast<verbose_t::flag_kind>(profiling));

    return success;
}

const dnnl_version_t *dnnl_version(void) {
    static const dnnl_version_t ver
            = {DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                    DNNL_VERSION_HASH, DNNL_CPU_RUNTIME, DNNL_GPU_RUNTIME};
    return &ver;
}
