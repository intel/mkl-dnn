/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <algorithm>
#include <cstring>

#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// TODO: refactor the driver to avoid using extra flags of a memory descriptor.
#include "common/memory_desc.hpp"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reorder.hpp"

namespace reorder {

int fill_mem(int exec_arg, const prb_t *prb, data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    const auto conf = prb->get_conf(kind);

    const float gen[] = {
            conf->max, // saturate to max
            conf->min, // saturate to min
            0.f,
            0.25f,
            0.5f,
            1.f,
            1.5f,
            2.f,
            16.f,
            64.f,
    };
    const auto table_size = sizeof(gen) / sizeof(gen[0]);
    // MIOpen doesn't work properly when tensors are filled with 0xFF.
    const bool zero_out_wa = is_amd_gpu();

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const int64_t table_idx = kind == SRC
                ? (i % table_size)
                : ((i * (table_size + 1) / table_size) % table_size);
        mem_fp.set_f32_elem(
                i, round_to_nearest_representable(conf->dt, gen[table_idx]));
        if (zero_out_wa) mem_dt.set_elem(i, 0);
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

dnn_mem_t setup_compensation_memory(const prb_t *prb, flag_bit_t flag) {
    dnn_mem_t m;
    if (prb->is_reorder_with_compensation(flag)) {
        dims_t dims = prb->get_compensation_dims(flag);
        int ndims = static_cast<int>(dims.size());
        // Compensation dims are derived from the problem dims, so a plain s32
        // descriptor is not expected to fail.
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
        DNN_SAFE_V(
                dnn_mem_t::init_md(md, ndims, dims.data(), dnnl_s32, tag::abx));
        m = dnn_mem_t(md, get_cpu_engine(), /* prefill = */ false);
    }
    return m;
}

int compare_compensation(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    // Note: following check relies on certain assumptions on CPU. These
    // assumptions may not hold for GPU. In addition, it's prohibit to work
    // with raw pointers directly for buffer type of memory.
    if (!is_cpu(get_test_engine())) return FAIL;

    const auto &mem_got = mem_map[DNNL_ARG_DST];
    const auto &mem_s8_comp_ref = ref_mem_map[DNNL_ARG_SRC_1];
    const auto &mem_zp_comp_ref = ref_mem_map[DNNL_ARG_SRC_2];

    const auto padded_nelems = mem_got.nelems(true);
    // Note: internally offset is aligned on 4, otherwise it's UB.
    size_t first_comp_offset = rnd_up(padded_nelems, 4);
    int *comp_handle
            = reinterpret_cast<int *>((char *)mem_got + first_comp_offset);

    const auto cmp_compensation = [&](const dnn_mem_t &mem_ref, int comp_mask) {
        // Idea behind this check:
        // Using knowledge from the library where `comp_handle` starts, and that
        // memory utilizes blocking over OC and G, if present, we wrap that
        // piece of memory which is described by shortened tag coming from prb
        // into a separate memory and reorder it to plain so that it is a
        // straight comparison of values in native plain layout.
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> comp_md;
        DNN_SAFE(dnn_mem_t::init_md(comp_md, mem_ref.ndims(), mem_ref.dims(),
                         mem_ref.dt(), trim_tag_by_mask(prb->dtag, comp_mask)),
                WARN);
        // Predefined handle_info won't use prefilling no matter what.
        dnn_mem_t comp_m(comp_md, mem_ref.engine(), /* prefill = */ false,
                {false, comp_handle});

        compare::compare_t cmp;
        cmp.set_zero_trust_percent(100.f); // No sense in zero trust test.
        int status = cmp.compare(mem_ref, comp_m, attr_t(), res);

        // Shift original compensation pointer for next compensation
        comp_handle += comp_m.nelems(true);
        return status;
    };

    if (mem_s8_comp_ref.ndims())
        SAFE(cmp_compensation(mem_s8_comp_ref,
                     prb->get_compensation_mask(FLAG_S8S8_COMP)),
                WARN);
    if (mem_zp_comp_ref.ndims())
        SAFE(cmp_compensation(
                     mem_zp_comp_ref, prb->get_compensation_mask(FLAG_ZP_COMP)),
                WARN);

    return res->state == FAILED ? FAIL : OK;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    auto dims = prb->dims;
    for (int d = 0; d < prb->ndims; ++d)
        if (prb->runtime_dim_mask & (1 << d)) dims[d] = DNNL_RUNTIME_DIM_VAL;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_d, prb->ndims, dims.data(),
            force_f32_dt ? dnnl_f32 : prb->src_dt(), prb->stag,
            prb->strides[0]));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_d, prb->ndims, dims.data(),
            force_f32_dt ? dnnl_f32 : prb->dst_dt(), prb->dtag,
            prb->strides[1]));

    // Prepare and assign extra for dst_md.
    auto &extra = static_cast<dnnl_memory_desc_t>(dst_d)->extra;
    extra.flags = dnnl::impl::memory_extra_flags::none;
    if (prb->is_reorder_with_compensation(FLAG_ANY)) {
        for (const auto &i_oflag : prb->oflag) {
            if (i_oflag.first & FLAG_S8S8_COMP) {
                extra.flags |= dnnl::impl::memory_extra_flags::
                        compensation_conv_s8s8;
                extra.compensation_mask = i_oflag.second;

                const float s8_scale_factor = reorder_rescale_factor();
                const bool need_rescale = s8_scale_factor != 1.f;
                if (need_rescale) {
                    extra.flags |= dnnl::impl::memory_extra_flags::scale_adjust;
                    extra.scale_adjust = s8_scale_factor;
                }
            }
            if (i_oflag.first & FLAG_ZP_COMP) {
                extra.flags |= dnnl::impl::memory_extra_flags::
                        compensation_conv_asymmetric_src;
                extra.asymm_compensation_mask = i_oflag.second;
            }
        }
    }

    auto src_engine = init_pd_args.engine;
    auto dst_engine = init_pd_args.engine;
    if (is_gpu()) {
        switch (prb->cross_engine) {
            case CPU2GPU: src_engine = get_cpu_engine(); break;
            case GPU2CPU: dst_engine = get_cpu_engine(); break;
            default: break;
        }
    }

    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args_t(), prb->ndims));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_reorder_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.src_md ? init_pd_args.src_md : src_d,
            src_engine, dst_d, dst_engine, dnnl_attr)));

    return dnnl_success;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    const auto sdt = prb->src_dt();
    const auto ddt = prb->dst_dt();
    skip_unimplemented_data_type({sdt, ddt}, prb->dir, res);

    if (is_gpu() && prb->runtime_dim_mask != 0) {
        BENCHDNN_PRINTF(
                2, "%s", "[SKIP]: GPU doesn't support runtime dimensions.");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    // This condition supposed to be true only for GPU, however,
    // ref/simple reorders don't support all configurations with compensation,
    // thus, allow any unimplemented to fall into this category until it's
    // resolved.
    // TODO: restore original `is_gpu()` part of the condition.
    if (prb->is_reorder_with_compensation(FLAG_ANY)) {
        // Reorders w/ compensation are not supported by design: zp_comp is
        // done in kernels directly, but s8s8 instructions are available in HW.
        // BENCHDNN_PRINTF(2, "%s",
        //         "[SKIP]: GPU doesn't support cases with compensation.");
        // TODO: replace with the condition above
        BENCHDNN_PRINTF(2, "%s",
                "[SKIP]: given case with compensation is unsupported.");
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
}

void prb_t::skip_invalid(res_t *res) const {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_NONE
    const prb_t *prb = this; // Kept to avoid mass update
    auto cross_engine = prb->cross_engine;
    if (cross_engine == CPU2GPU || cross_engine == GPU2CPU) {
        BENCHDNN_PRINT(2,
                "[INVALID][%s:%d]: Cross-engine case isn't supported when just "
                "one runtime is enabled.\n",
                __FILE__, __LINE__);
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
#endif
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    // This value can be exact without scales. Scales may affect the value.
    // Avoid any scales logic involved until needed.
    cmp.set_zero_trust_percent(80.f);

    // `f8_e4m3` range is very short which makes inputs convert into NaNs.
    cmp.set_op_output_has_nans(prb->src_dt() == dnnl_f8_e4m3);

    // Additional check to avoid false-positive result from f32->s32 conversion
    // in case of sum post-op on GPU happening when two max_dt values
    // are summed together.
    const auto reorder_add_check
            = [&](const compare::compare_t::driver_check_func_args_t &args) {
        if (args.dt == dnnl_s32 && args.got == max_dt(args.dt) && is_gpu()) {
            // 128.f = float(INT_MAX) - BENCHDNN_S32_TO_F32_SAT_CONST;
            return args.diff == 128.f;
        }
        return false;
    };
    cmp.set_driver_check_function(reorder_add_check);
}

std::vector<int> prb_t::supported_exec_args(bool) const {
    static const std::vector<int> exec_args = {
            DNNL_ARG_FROM,
            DNNL_ARG_TO,
    };
    return exec_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_FROM: {
                SAFE(fill_mem(exec_arg, prb, SRC, mem, ref_mem, res), WARN);
                // Additional inputs to compare compensation buffers.
                ref_mem_map.emplace(DNNL_ARG_SRC_1,
                        setup_compensation_memory(prb, FLAG_S8S8_COMP));
                ref_mem_map.emplace(DNNL_ARG_SRC_2,
                        setup_compensation_memory(prb, FLAG_ZP_COMP));
            } break;
            case DNNL_ARG_TO: {
                const auto &po = prb->attr.post_ops;
                const int sum_idx = po.find(attr_t::post_ops_t::SUM);
                // MIOpen doesn't work properly when tensors are filled with 0xFF.
                if (sum_idx >= 0 || is_amd_gpu()) {
                    SAFE(fill_mem(exec_arg, prb, DST, mem, ref_mem, res), WARN);

                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem, res), WARN);
                    }
                }
            } break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(mem_map, prb, prim, res), WARN);
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        // Remove extra desc so that reorders with compensation could have
        // proper reorder from blocked layout to plain for comparison.
        auto &dst_dt = mem_map[DNNL_ARG_DST];
        dnnl::impl::memory_extra_desc_t empty_extra {};
        const auto orig_dst_extra = dst_dt.md_->extra;
        dst_dt.md_->extra = empty_extra;

        // Validate main reorder part.
        check_correctness(prb, {DST}, args, ref_args, compute_ref, setup_cmp,
                res, prb->dir);

        // Restore extra for compensation comparison and performance mode.
        dst_dt.md_->extra = orig_dst_extra;

        // Validate compensated reorder part.
        if (prb->is_reorder_with_compensation(FLAG_ANY)) {
            compare_compensation(prb, mem_map, ref_mem_map, res);
        }
    }
    SAFE(check_bitwise(prim, {DST}, args, prb->attr, prb->inplace, res), WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace reorder
