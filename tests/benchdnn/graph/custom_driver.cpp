/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <random>
#include "common.hpp"
#include "deserialize.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "oneapi/dnnl/dnnl.h"
#include "utils/parallel.hpp"
#include "utils/perf_report.hpp"
#include "utils/settings.hpp"

#include "custom_driver.hpp"

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

namespace custom {

namespace genindex {
// GENINDEX OP
// DNNL_ARG_SRC: src
// DNNL_ARG_DST: dst

std::vector<int> exec_args = {
        DNNL_ARG_SRC,
        DNNL_ARG_DST,
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res) {

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second;

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                // For GenIndex op, the input value doesn't affect the output
                // value, it doesn't matter what value we fill in.
                SAFE(::custom::fill_mem(mem, ref_mem, 0, 0, res), WARN);
                break;
            default: break;
        }
    }
    return OK;
}

int execute(const prb_t *prb, const args_t &args, res_t *res) {
    dnn_mem_t &dst = const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST));
    auto ndims = dst.ndims();
    const size_t axis = prb->axis < 0 ? (prb->axis + ndims) : prb->axis;
    auto dims = dst.dims();
    auto strides = dst.strides();

    benchdnn_parallel_nd(dst.nelems(), [&](int64_t index) {
        // This function resembles dnn_mem_t::get_idx but has a format + axis
        // peculiarity which can't be covered in get_idx function without
        // sacrificing performance, and the current code as is.
        size_t offdst = 0, result = 0;
        for (int i = 0; i < ndims; i++) {
            // calculate the idx on each dimension
            int idx = index % dims[i];
            if ((size_t)i == axis) result = idx;
            index /= dims[i];

            // accumulate offset for each arg
            offdst += strides[i] * idx;
            if (index == 0) break;
        }
        dst.set_elem(offdst, result);
    });
    return OK;
}
} // namespace genindex

namespace transpose {
// TRANSPOSE OP
// DNNL_ARG_SRC: src
// DNNL_ARG_DST: dst
// order attribute: a permutation defines how to transpose

std::vector<int> exec_args = {
        DNNL_ARG_SRC,
        DNNL_ARG_DST,
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res) {

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second;

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                // Use `7` not to mess with scales for s8 which may create a
                // `8 * 8 (= 128) = -128` for s8.
                SAFE(::custom::fill_mem(mem, ref_mem, -8, 7, res), WARN);
                break;
            default: break;
        }
    }
    return OK;
}

int execute(const prb_t *prb, const args_t &args, res_t *res) {
    const auto &src = args.find(DNNL_ARG_SRC);
    const auto &tag = ::std::get<0>(prb->arg_mds_.at(DNNL_ARG_SRC));
    dnn_mem_t pad(src, src.dt(), tag, get_test_engine());
    ::graph::permute_md(pad, prb->order);
    return const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST)).reorder(pad, res);
}
} // namespace transpose

namespace reshape {
// RESHAPE OP
// DNNL_ARG_SRC: src
// DNNL_ARG_DST: dst

std::vector<int> exec_args = {
        DNNL_ARG_SRC,
        DNNL_ARG_DST,
};

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        const prb_t *prb, res_t *res) {

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        auto &mem = entry.second;

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                // Use `7` not to mess with scales for s8 which may create a
                // `8 * 8 (= 128) = -128` for s8.
                SAFE(::custom::fill_mem(mem, ref_mem, -8, 7, res), WARN);
                break;
            default: break;
        }
    }
    return OK;
}

int execute(const prb_t *prb, const args_t &args, res_t *res) {
    const dnn_mem_t &src = args.find(DNNL_ARG_SRC);
    dnn_mem_t &dst = const_cast<dnn_mem_t &>(args.find(DNNL_ARG_DST));
    // generate dense stride
    dnn_mem_t pad(src.md_, src.dt(), tag::abx, get_test_engine(),
            /* prefill = */ true);
    SAFE(pad.reorder(src, res), WARN);

    // update output shape with dense stride
    dnnl_memory_desc_destroy(pad.md_);
    dnnl_memory_desc_create_with_string_tag(&pad.md_, dst.ndims(), dst.dims(),
            dst.dt(), normalize_tag(tag::abx, dst.ndims()).c_str());
    return dst.reorder(pad, res);
}
} // namespace reshape

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    return dnnl_success;
}

std::vector<int> prb_t::supported_exec_args(bool) const {
    std::vector<int> exec_args;
    switch (alg) {
        case GENINDEX: return ::custom::genindex::exec_args;
        case TRANSPOSE: return ::custom::transpose::exec_args;
        case RESHAPE: return ::custom::reshape::exec_args;
        default: assert(!"unknown alg"); break;
    }
    return exec_args;
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const auto *prb = prb_t::from(base_prb);
    switch (prb->alg) {
        case GENINDEX:
        case TRANSPOSE:
        case RESHAPE: cmp.set_zero_trust_percent(100.f); break;
        default: assert(!"unknown alg"); break;
    }
}

int fill_mem(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, int f_min, int f_max,
        res_t *res) {

    const auto dt = mem_dt.dt();
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
            && !is_integral_dt(dt)) {
        // Use data filled by benchdnn for `no_ref_memory`, except some
        // customized operations in Graph API which expect the input
        // values to indicate indexing information, especially for integral
        // inputs. Hence we need to be limited the input value to the
        // provided range.
        return OK;
    }

    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    f_min = (dt == dnnl_u8 && f_min < 0) ? 0 : f_min;
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand int_seed(nelems + idx_start + 1);
        std::uniform_int_distribution<> gen(f_min, f_max);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = gen(int_seed);
            mem_fp.set_elem(idx, round_to_nearest_representable(dt, value));
        }
    });
    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

void init_memory_args(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        bool override_dir_with_fwd, res_t *, const engine_t &test_engine) {
    const auto *prb = prb_t::from(base_prb);
    const auto &supported_exec_args
            = base_prb->supported_exec_args(override_dir_with_fwd);
    for (const auto &exec_arg : supported_exec_args) {
        if (prb->arg_mds_.find(exec_arg) == prb->arg_mds_.end()) {
            assert(!"missing required args");
            SAFE_V(FAIL);
        }
        auto arg_mds_ = prb->arg_mds_.find(exec_arg)->second;
        dnnl_dims_t dnnl_dims {};
        auto dims = ::std::get<1>(arg_mds_);
        for (size_t i = 0; i < dims.size(); i++) {
            dnnl_dims[i] = dims[i];
        }

        mem_map.emplace(exec_arg,
                dnn_mem_t(static_cast<int>(dims.size()), dnnl_dims,
                        std::get<2>(arg_mds_), ::std::get<0>(arg_mds_),
                        test_engine, /* prefill = */ true));
    }
}

void init_memory_args_native(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        const graph::deserialized_op_t &, const engine_t &ref_eng) {
    init_memory_args(mem_map, base_prb, /*override_dir_with_fwd=*/false,
            nullptr, ref_eng);
}

// Account for the test allocation and its mapped, reference, and comparison
// buffers. Custom operations provide dimensions separately because they do not
// have a primitive descriptor from which the reference layout can be queried.
void add_md_size(const_dnnl_memory_desc_t md, const dims_t &dims,
        check_mem_size_args_t &mem_size_args) {
    const auto mem_size = dnnl_memory_desc_get_size(md);
    if (mem_size == 0 || mem_size == DNNL_RUNTIME_SIZE_VAL) return;

    mem_size_args.sizes.push_back(mem_size);
    mem_size_args.total_size_device += mem_size;

    const bool mapped_mem_factor = !is_cpu()
            && !has_bench_mode_modifier(mode_modifier_t::no_ref_memory);
    mem_size_args.total_size_mapped += mapped_mem_factor * mem_size;

    // `dims` were already used to create `md`, so a failure is not expected.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> ref_md;
    DNN_SAFE_V(dnn_mem_t::init_md(ref_md, static_cast<int>(dims.size()),
            dims.data(), dnnl_f32, tag::abx));
    const auto ref_md_size = dnnl_memory_desc_get_size(ref_md);
    const size_t ref_mem_idx = mem_size_args.want_input ? 0 : 1;
    mem_size_args.total_ref_md_size[ref_mem_idx] = ref_md_size;

    const bool is_corr = has_bench_mode_bit(mode_bit_t::corr);
    const bool is_bitwise = has_bench_mode_bit(mode_bit_t::bitwise);
    mem_size_args.total_size_ref += is_corr * ref_md_size;

    const bool is_output = !mem_size_args.want_input;
    mem_size_args.total_size_compare
            += is_output * ((is_corr || is_bitwise) + is_bitwise) * ref_md_size;
}

// Rebuild memory estimates from custom operation arguments. Unlike regular
// drivers, custom operations do not create a primitive whose descriptors can be
// collected by the common memory accounting path.
void collect_mem_size(
        check_mem_size_args_t &mem_size_args, const base_prb_t *base_prb) {
    mem_size_args = {};
    const auto *prb = prb_t::from(base_prb);
    for (const auto exec_arg : prb->supported_exec_args(false)) {
        const auto &arg_md = prb->arg_mds_.at(exec_arg);
        const auto &tag = std::get<0>(arg_md);
        const auto &dims = std::get<1>(arg_md);
        const auto dt = std::get<2>(arg_md);
        // `tag` is derived from the logical tensor strides and always matches
        // `dims`, so a failure is not expected.
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
        DNN_SAFE_V(dnn_mem_t::init_md(
                md, static_cast<int>(dims.size()), dims.data(), dt, tag));
        mem_size_args.want_input = exec_arg != DNNL_ARG_DST;
        add_md_size(md, dims, mem_size_args);
    }
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t) {
    const auto *prb = prb_t::from(base_prb);
    switch (prb->alg) {
        case GENINDEX:
            SAFE(::custom::genindex::init_ref_memory_args(
                         ref_mem_map, mem_map, prb, res),
                    WARN);
            break;
        case TRANSPOSE:
            SAFE(::custom::transpose::init_ref_memory_args(
                         ref_mem_map, mem_map, prb, res),
                    WARN);
            break;
        case RESHAPE:
            SAFE(::custom::reshape::init_ref_memory_args(
                         ref_mem_map, mem_map, prb, res),
                    WARN);
            break;
        default: assert(!"unknown alg"); break;
    }
    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {}

int execute(const base_prb_t *base_prb, const args_t &args, res_t *res) {
    const auto *prb = prb_t::from(base_prb);
    int ret = FAILED;
    switch (prb->alg) {
        case GENINDEX: ret = ::custom::genindex::execute(prb, args, res); break;
        case TRANSPOSE:
            ret = ::custom::transpose::execute(prb, args, res);
            break;
        case RESHAPE: ret = ::custom::reshape::execute(prb, args, res); break;
        default: assert(!"unknown alg"); break;
    }
    return ret;
}

} // namespace custom
