/*******************************************************************************
* Copyright 2018 Intel Corporation
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

/*
  General architecture

  for diff states, we have n_states + 1 as we have n_states diff
  to propagate to the previous iteration and 1 states to propagate
  to the previous layer
  index 0 is dh for cell(t-1, l) to consume // replaced by diff_src_iter
  index 1 is dc for cell(t-1, l) to consume // replaced by diff_src_iter_c
  index 2 is dh for cell(t, l-1) to consume // replace by diff_src_layer
  this indexing enables to have the same indexing for states in elemwise
  function
  only the cell execution function should be impacted

 */

#include "common/dnnl_thread.hpp"
#include "common/matmul_pd.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/stream.hpp"

#include "cpu/simple_q10n.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/gemm_pack.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::memory_tracking::names;
using namespace rnn_utils;
#define AOC array_offset_calculator

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
status_t dnnl::impl::cpu::ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::pd_t::init_ref(engine_t *engine) {

    using namespace prop_kind;
    using namespace utils;
    using namespace format_tag;
    using namespace rnn_utils;
    const alg_kind_t cell_kind = this->desc()->cell_kind;

    const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    const data_type_t weights_iter_dt
            = this->desc()->weights_iter_desc.data_type;
    const data_type_t weights_layer_dt
            = this->desc()->weights_layer_desc.data_type;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                    alg_kind::vanilla_gru, alg_kind::lbr_gru,
                    alg_kind::vanilla_augru, alg_kind::lbr_augru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          one_of(this->desc()->prop_kind, forward_training,
                                  forward_inference)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(src_layer_dt == src_type, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(everyone_is(weights_type, weights_iter_dt, weights_layer_dt),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(this->set_default_params() == status::success,
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    rnn_ = zero<decltype(rnn_)>();
    rnn_.is_brgemm = false;
    VDISPATCH_RNN(init_conf<class_name>(rnn_, *this->desc(), *this->attr(),
                          this->src_md(0), this->src_md(1), this->src_md(2),
                          this->weights_md(0), this->weights_md(1),
                          this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                          this->dst_md(0), this->dst_md(1), this->dst_md(2),
                          this->arg_md(DNNL_ARG_BIAS)),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "rnn");

    VDISPATCH_RNN(IMPLICATION(rnn_.is_f16_conf(), !rnn_.is_training),
            VERBOSE_UNSUPPORTED_FEATURE, "f16 training not supported");

    if (rnn_.is_xf16_conf()) {
        VDISPATCH_RNN(
                !(!utils::one_of(rnn_.bias_dt, src_type, data_type::f32)
                        || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                        || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                                src_type, data_type::f32)),
                VERBOSE_UNSUPPORTED_DT_CFG);
    } else {
        VDISPATCH_RNN(!(rnn_.bias_dt != data_type::f32
                              || !utils::one_of(rnn_.src_iter_c_dt,
                                      data_type::undef, data_type::f32)
                              || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt),
                VERBOSE_UNSUPPORTED_DT_CFG);
    }

    /* check that no data shift have been passed to s8s8 lstm */
    VDISPATCH_RNN(IMPLICATION(rnn_.is_signed_int8_conf(),
                          this->attr()->rnn_data_qparams_.shift_ == 0.f),
            VERBOSE_UNSUPPORTED_FEATURE,
            "s8s8 lstm does not support data shift");

    /* INT8 cases with non-trivial strides are not supported */
    VDISPATCH_RNN(!(rnn_.is_int8_conf()
                          && !(rnn_.src_layer_is_trivial_stride
                                  && rnn_.dst_layer_is_trivial_stride)),
            VERBOSE_NONTRIVIAL_STRIDE);

    /* check that only supported attr have been passed */
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8)
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_projection_qparams;

    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // Set weights descriptors to desired format
    memory_desc_t new_weights_layer_md = *this->weights_md(0);
    CHECK(set_expected_desc(
            rnn_, new_weights_layer_md, rnn_utils::weights_type_t::layer));
    if (this->weights_layer_md_.format_kind == format_kind::any) {
        this->weights_layer_md_ = new_weights_layer_md;
    } else if (this->weights_layer_md_.format_kind == format_kind::rnn_packed) {
        VDISPATCH_RNN(this->weights_layer_md_ == new_weights_layer_md,
                VERBOSE_INCONSISTENT_MDS, "weights_layer", "new_weights_layer");
    }

    memory_desc_t new_weights_iter_md = *this->weights_md(1);
    CHECK(set_expected_desc(
            rnn_, new_weights_iter_md, rnn_utils::weights_type_t::iter));
    if (this->weights_iter_md_.format_kind == format_kind::any) {
        this->weights_iter_md_ = new_weights_iter_md;
    } else if (this->weights_iter_md_.format_kind == format_kind::rnn_packed) {
        VDISPATCH_RNN(this->weights_iter_md_ == new_weights_iter_md,
                VERBOSE_INCONSISTENT_MDS, "weights_iter", "new_weights_iter");
    }

    if (rnn_.is_lstm_projection) {
        memory_desc_t new_weights_projection_md
                = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
        CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                rnn_utils::weights_type_t::projection));
        if (this->weights_projection_md_.format_kind == format_kind::any) {
            this->weights_projection_md_ = new_weights_projection_md;
        } else if (this->weights_projection_md_.format_kind
                == format_kind::rnn_packed) {
            VDISPATCH_RNN(
                    this->weights_projection_md_ == new_weights_projection_md,
                    VERBOSE_INCONSISTENT_MDS, "weights_projection",
                    "new_weights_projection");
        }
    }

    VDISPATCH_RNN(this->check_layout_consistency(false /*is_brgemm*/)
                    == status::success,
            "layout consistency check failed");

    set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
            this->weights_md(1), this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
            this->diff_weights_md(0), this->diff_weights_md(1),
            this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
    set_workspace_sizes<class_name>(rnn_, *this->desc());

    // INIT MATMULS
    auto init_matmul_pd
            = [&](std::shared_ptr<primitive_desc_t> &mpd, dim_t M, dim_t N,
                      dim_t K, dim_t LDA, dim_t LDB, dim_t LDC, bool sum_po) {
        memory_desc_t src_desc;
        const dims_t src_dims = {M, K};
        const dims_t src_strides = {LDA, 1};
        CHECK(memory_desc_init_by_strides(
                src_desc, 2, src_dims, src_type, src_strides));

        memory_desc_t wei_desc;
        const dims_t wei_dims = {K, N};
        const dims_t wei_strides = {LDB, 1};
        CHECK(memory_desc_init_by_strides(
                wei_desc, 2, wei_dims, weights_type, wei_strides));

        memory_desc_t dst_desc;
        const dims_t dst_dims = {M, N};
        const dims_t dst_strides = {LDC, 1};
        CHECK(memory_desc_init_by_strides(
                dst_desc, 2, dst_dims, scratch_type, dst_strides));

        matmul_desc_t matmul_desc;
        CHECK(matmul_desc_init(
                &matmul_desc, &src_desc, &wei_desc, nullptr, &dst_desc));
        post_ops_t po;
        CHECK(po.append_sum(1.0f));
        primitive_attr_t attr;
        CHECK(attr.set_post_ops(po));
        primitive_desc_iterator_t it(engine, (op_desc_t *)(&matmul_desc),
                sum_po ? &attr : nullptr, nullptr);
        if (!it.is_initialized()) return status::out_of_memory;

        while (++it != it.end()) {
            mpd = *it;
            const bool ok = mpd->weights_md()->extra.flags == 0;
            if (ok) return status::success;
        }
        return status::unimplemented;
    };

    if (rnn_.use_matmul) {
        { // init layer matmuls
            const dim_t M = rnn_.mb;
            const dim_t N = static_cast<dim_t>(rnn_.n_gates) * rnn_.dhc;
            const dim_t K = rnn_.slc;
            const dim_t LDA1 = rnn_.src_layer_ld_;
            const dim_t LDA2 = rnn_.ws_states_layer_ld;
            const dim_t LDA3 = rnn_.dst_iter_ld_;
            const dim_t LDB = rnn_.weights_layer_ld;
            const dim_t LDC = rnn_.scratch_gates_ld;
            const bool do_sum = false;
            if (LDA1 >= K)
                CHECK(init_matmul_pd(
                        matmul_layer_1_pd_, M, N, K, LDA1, LDB, LDC, do_sum));
            if (LDA2 >= K && LDA2 != LDA1)
                CHECK(init_matmul_pd(
                        matmul_layer_2_pd_, M, N, K, LDA2, LDB, LDC, do_sum));
            if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                CHECK(init_matmul_pd(
                        matmul_layer_3_pd_, M, N, K, LDA3, LDB, LDC, do_sum));
        }

        { // init iter matmuls
            const dim_t M = rnn_.mb;
            const dim_t N = static_cast<dim_t>(rnn_.dhc)
                    * (rnn_.n_gates - rnn_.is_orig_gru);
            const dim_t K = rnn_.sic;
            const dim_t LDA1 = rnn_.src_iter_ld_;
            const dim_t LDA2 = rnn_.ws_states_iter_ld;
            const dim_t LDA3 = rnn_.dst_layer_ld_;
            const dim_t LDB = rnn_.weights_iter_ld;
            const dim_t LDC = rnn_.scratch_gates_ld;
            const bool do_sum = !rnn_.is_lbr;
            if (LDA1 >= K)
                CHECK(init_matmul_pd(
                        matmul_iter_1_pd_, M, N, K, LDA1, LDB, LDC, do_sum));
            if (LDA2 >= K && LDA2 != LDA1)
                CHECK(init_matmul_pd(
                        matmul_iter_2_pd_, M, N, K, LDA2, LDB, LDC, do_sum));
            if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                CHECK(init_matmul_pd(
                        matmul_iter_3_pd_, M, N, K, LDA3, LDB, LDC, do_sum));

            if (rnn_.is_orig_gru) {
                const dim_t N_part2 = rnn_.dhc;
                const dim_t LDA1 = rnn_.ws_states_layer_ld;
                const dim_t LDA2 = rnn_.ws_states_iter_ld;
                const dim_t LDA3 = rnn_.dst_layer_ld_;
                const dim_t LDA4 = rnn_.dst_iter_ld_;
                if (LDA1 >= K)
                    CHECK(init_matmul_pd(matmul_part2_1_pd_, M, N_part2, K,
                            LDA1, LDB, LDC, do_sum));
                if (LDA2 >= K && LDA2 != LDA1)
                    CHECK(init_matmul_pd(matmul_part2_2_pd_, M, N_part2, K,
                            LDA2, LDB, LDC, do_sum));
                if (LDA3 >= K && !utils::one_of(LDA3, LDA1, LDA2))
                    CHECK(init_matmul_pd(matmul_part2_3_pd_, M, N_part2, K,
                            LDA3, LDB, LDC, do_sum));
                if (LDA4 >= K && !utils::one_of(LDA4, LDA1, LDA2, LDA3))
                    CHECK(init_matmul_pd(matmul_part2_4_pd_, M, N_part2, K,
                            LDA4, LDB, LDC, do_sum));
            }
        }
    }
    return status::success;
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
status_t
ref_rnn_common_t<aprop, src_type, weights_type, acc_type>::pd_t::init_brgemm(
        engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace format_tag;
    using namespace rnn_utils;
#if DNNL_X64
    using namespace x64;
    const alg_kind_t cell_kind = this->desc()->cell_kind;

    const data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    const data_type_t weights_iter_dt
            = this->desc()->weights_iter_desc.data_type;
    const data_type_t weights_layer_dt
            = this->desc()->weights_layer_desc.data_type;
    bool is_f32 = everyone_is(
            data_type::f32, src_layer_dt, weights_iter_dt, weights_layer_dt);
    bool is_impl_bf16 = everyone_is(data_type::bf16, src_type, weights_type);
    bool is_fpmath_bf16 = one_of(
            this->attr()->fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any);
    bool allow_down_conversion_to_bf16
            = is_f32 && is_fpmath_bf16 && is_impl_bf16;

    // Initialized rnn_ early to get correct verbose output
    rnn_ = zero<decltype(rnn_)>();
    rnn_.is_brgemm = true;
    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                    alg_kind::vanilla_gru, alg_kind::lbr_gru,
                    alg_kind::vanilla_augru, alg_kind::lbr_augru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          one_of(this->desc()->prop_kind, forward_training,
                                  forward_inference)),
            VERBOSE_BAD_PROPKIND);
    // LBR is not supported for training in brgemm
    VDISPATCH_RNN(IMPLICATION(one_of(cell_kind, alg_kind::lbr_gru,
                                      alg_kind::lbr_augru),
                          this->desc()->prop_kind == forward_inference),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    // TODO: Enable diff_weights_overwrite support
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          this->diff_weights_overwrite() == false),
            VERBOSE_BAD_PROPKIND);
    // cell_type (or src_type) and primitive data type should
    // match, except for the bf32 case.
    VDISPATCH_RNN(IMPLICATION(!allow_down_conversion_to_bf16,
                          src_layer_dt == src_type
                                  && everyone_is(weights_type, weights_iter_dt,
                                          weights_layer_dt)),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(this->set_default_params() == status::success,
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    VDISPATCH_RNN(init_conf<class_name>(rnn_, *this->desc(), *this->attr(),
                          this->src_md(0), this->src_md(1), this->src_md(2),
                          this->weights_md(0), this->weights_md(1),
                          this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                          this->dst_md(0), this->dst_md(1), this->dst_md(2),
                          this->arg_md(DNNL_ARG_BIAS)),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "rnn");

    VDISPATCH_RNN(IMPLICATION(one_of(this->desc()->prop_kind, forward_training,
                                      backward),
                          (rnn_.is_xf16_conf() || rnn_.is_f32_conf())),
            VERBOSE_PROPKIND_DT_MISMATCH);

    // Support for GRU / AUGRU cell in BRGEMM-based implementation is
    // limited by forward_inference pass for now. all_f32 is enabled only
    // for small shapes on avx2, where it avoids the per-call overhead of
    // the nested matmul in the reference implementation. For larger shapes
    // it is slower, as the cell is parallelized over M blocks only.
    // TODO: Improve GRU / AUGRU coverage in BRGEMM-based implementation
    const dim_t max_c = nstl::max(rnn_.slc, nstl::max(rnn_.sic, rnn_.dhc));
    const bool is_small_f32_avx2 = !mayiuse(avx512_core)
            && ((rnn_.mb <= 4 && max_c <= 128) || max_c <= 64);
    VDISPATCH_RNN(IMPLICATION(rnn_.is_orig_gru,
                          this->desc()->prop_kind == forward_inference
                                  && IMPLICATION(rnn_.is_cell_dt_f32(),
                                          is_small_f32_avx2)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "gru/augru cell in brgemm-based forward inference");

    VDISPATCH_RNN(!(rnn_.is_cell_dt_f32()
                          && utils::one_of(this->desc()->prop_kind, backward,
                                  forward_training)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "f32 datatype in brgemm-based implementation");

    VDISPATCH_RNN((IMPLICATION((cell_kind == alg_kind::vanilla_lstm
                                       && rnn_.is_lstm_projection),
                          this->desc()->prop_kind == forward_inference)),
            "bad algorithm for lstm projection for forward inference");

    if (rnn_.is_bf16_conf()) {
        const bool isa_dt_not_ok = (!mayiuse(avx512_core_bf16)
                || !utils::one_of(rnn_.bias_dt, data_type::bf16, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                        data_type::bf16, data_type::f32));
        VDISPATCH_RNN(!isa_dt_not_ok, VERBOSE_ISA_DT_MISMATCH);
    } else if (rnn_.is_f16_conf()) {
        const bool isa_dt_not_ok = (!mayiuse(avx512_core_amx_fp16)
                || !utils::one_of(rnn_.bias_dt, data_type::f16, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt
                || !utils::one_of(rnn_.src_iter_c_dt, data_type::undef,
                        data_type::f16, data_type::f32));
        VDISPATCH_RNN(!isa_dt_not_ok, VERBOSE_ISA_DT_MISMATCH);
    } else {
        const bool dt_not_ok = (rnn_.bias_dt != data_type::f32
                || !utils::one_of(
                        rnn_.src_iter_c_dt, data_type::undef, data_type::f32)
                || rnn_.src_iter_c_dt != rnn_.dst_iter_c_dt);
        VDISPATCH_RNN(!dt_not_ok, VERBOSE_UNSUPPORTED_DT_CFG);
    }
    const auto isa = get_max_cpu_isa();
    VDISPATCH_RNN(
            !(rnn_.is_signed_int8_conf() && !is_superset(isa, avx512_core_amx)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_RNN(!(rnn_.is_int8_conf() && !is_superset(isa, avx2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_RNN(!(rnn_.is_f32_conf() && !is_superset(isa, avx2)),
            VERBOSE_ISA_DT_MISMATCH);

    /* check that no shift have been passed to s8s8 amx lstm */
    VDISPATCH_RNN(IMPLICATION(rnn_.is_signed_int8_conf(),
                          this->attr()->rnn_data_qparams_.shift_ == 0),
            VERBOSE_UNSUPPORTED_FEATURE,
            "s8s8 amx lstm does not support shift");

    /* INT8 cases with non-trivial strides are not supported */
    VDISPATCH_RNN(!(rnn_.is_int8_conf()
                          && !(rnn_.src_layer_is_trivial_stride
                                  && rnn_.dst_layer_is_trivial_stride)),
            VERBOSE_NONTRIVIAL_STRIDE);

    /* check that only supported attr have been passed */
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8)
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_projection_qparams
                | primitive_attr_t::skip_mask_t::fpmath_mode;
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
            this->weights_md(1), this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
            this->diff_weights_md(0), this->diff_weights_md(1),
            this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

    CHECK(ref_rnn_brgemm_t::configure_brgemm(rnn_, this->desc()->cell_kind,
            sizeof(src_layer_t), sizeof(scratch_t)));

    // must be called after configure_brgemm()
    set_workspace_sizes<class_name>(rnn_, *this->desc());

    // Only AMX LSTM supports s8s8 now
    VDISPATCH_RNN(!(rnn_.is_signed_int8_conf() && !rnn_.is_cell_int8_amx()),
            VERBOSE_UNSUPPORTED_DT);

    // Set weights descriptors to desired format
    memory_desc_t new_weights_layer_md = *this->weights_md(0);
    CHECK(set_expected_desc(
            rnn_, new_weights_layer_md, rnn_utils::weights_type_t::layer));
    if (this->weights_layer_md_.format_kind == format_kind::any) {
        this->weights_layer_md_ = new_weights_layer_md;
    } else {
        VDISPATCH_RNN(this->weights_layer_md_ == new_weights_layer_md,
                VERBOSE_INCONSISTENT_MDS, "weights_layer", "new_weights_layer");
    }

    memory_desc_t new_weights_iter_md = *this->weights_md(1);
    CHECK(set_expected_desc(
            rnn_, new_weights_iter_md, rnn_utils::weights_type_t::iter));
    if (this->weights_iter_md_.format_kind == format_kind::any) {
        this->weights_iter_md_ = new_weights_iter_md;
    } else {
        VDISPATCH_RNN(this->weights_iter_md_ == new_weights_iter_md,
                VERBOSE_INCONSISTENT_MDS, "weights_iter", "new_weights_iter");
    }
    if (rnn_.is_lstm_projection) {
        memory_desc_t new_weights_projection_md
                = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
        CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                rnn_utils::weights_type_t::projection));
        if (this->weights_projection_md_.format_kind == format_kind::any) {
            this->weights_projection_md_ = new_weights_projection_md;
        } else {
            VDISPATCH_RNN(
                    this->weights_projection_md_ == new_weights_projection_md,
                    VERBOSE_INCONSISTENT_MDS, "weights_projection",
                    "new_weights_projection");
        }
    }
    if (rnn_.is_unsigned_int8_conf()) {
        const memory_desc_wrapper &weights_layer_d(this->weights_layer_md_);
        const memory_desc_wrapper &weights_iter_d(this->weights_iter_md_);
        const auto &pdims_l = weights_layer_d.padded_dims();
        const auto &pdims_i = weights_iter_d.padded_dims();
        rnn_.weights_layer_comp_offset = rnn_.n_layer * rnn_.n_dir
                * rnn_.n_gates * pdims_l[2] * pdims_l[4];
        rnn_.weights_iter_comp_offset = rnn_.n_layer * rnn_.n_dir * rnn_.n_gates
                * pdims_i[2] * pdims_i[4];
        if (rnn_.is_lstm_projection) {
            const memory_desc_wrapper &weights_proj_d(
                    this->weights_projection_md_);
            const auto &pdims_p = weights_proj_d.padded_dims();
            rnn_.weights_projection_comp_offset
                    = rnn_.n_layer * rnn_.n_dir * pdims_p[2] * pdims_p[3];
        } else {
            rnn_.weights_projection_comp_offset = 0;
        }
    }
    VDISPATCH_RNN(this->check_layout_consistency(true /*is_brgemm*/)
                    == status::success,
            "layout consistency check failed");

    if (rnn_.is_bf32()) {
        const memory_desc_wrapper weights_layer_d(this->weights_layer_md_);
        memory_desc_t weights_layer_md;
        const memory_desc_wrapper weights_iter_d(this->weights_iter_md_);
        memory_desc_t weights_iter_md;

        const auto bf16_tag = rnn_.n_block == 64 ? format_tag::ldgOI64o2i
                                                 : format_tag::ldgOI32o2i;
        CHECK(memory_desc_init_by_tag(weights_layer_md, weights_layer_d.ndims(),
                weights_layer_d.dims(), data_type::bf16, bf16_tag));
        CHECK(reorder_primitive_desc_create(bf32_wei_layer_reorder_pd_, engine,
                weights_layer_d.md_, &weights_layer_md, nullptr));

        CHECK(memory_desc_init_by_tag(weights_iter_md, weights_iter_d.ndims(),
                weights_iter_d.dims(), data_type::bf16, bf16_tag));
        CHECK(reorder_primitive_desc_create(bf32_wei_iter_reorder_pd_, engine,
                weights_iter_d.md_, &weights_iter_md, nullptr));
    }

    return status::success;
#else
    return status::unimplemented;
#endif
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
status_t ref_rnn_common_t<aprop, src_type, weights_type, acc_type>::pd_t::init(
        engine_t *engine) {
    status_t st = init_brgemm(engine);
    if (st != status::success) {
        rnn_.is_brgemm = false;
        st = init_ref(engine);
    }
    if (st == status::success) {
        size_t scratchpad_sz {0}, ws_sz {0};
        get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

        init_scratchpad(scratchpad_sz);
        // initialize the workspace if needed
        if (rnn_.is_training) {
            dims_t ws_dims = {(dim_t)ws_sz};
            CHECK(memory_desc_init_by_tag(
                    this->ws_md_, 1, ws_dims, data_type::u8, format_tag::x));
        }
        rnn_.cell_kind = this->desc()->cell_kind;
    }
    return st;
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
void ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::pd_t::init_scratchpad(size_t scratchpad_sz) {
    using namespace memory_tracking::names;
    auto scratchpad = this->scratchpad_registry().registrar();

    {
        static constexpr size_t data_size
                = 1; // "true" data size already incorporated
        static constexpr size_t data_align
                = alignof(float); // "worst" case scenario
        static constexpr size_t perf_align = 4096;
        scratchpad.book(key_rnn_space, scratchpad_sz, data_size, data_align,
                perf_align);
    }

    const int max_nparts
            = utils::one_of(this->cell_kind(), alg_kind::vanilla_gru,
                      alg_kind::vanilla_augru)
            ? 2
            : 1;
    const dim_t ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
    scratchpad.template book<float *>(key_rnn_ptrs_wei_layer, ptr_wei_sz);
    scratchpad.template book<float *>(key_rnn_ptrs_wei_iter, ptr_wei_sz);
    scratchpad.template book<float *>(key_rnn_ptrs_wei_projection, ptr_wei_sz);

    const auto bias_dt_size
            = types::data_type_size(this->arg_md(DNNL_ARG_BIAS)->data_type);
    scratchpad.template book<void *>(
            key_rnn_ptrs_bia, ptr_wei_sz * bias_dt_size);

#if DNNL_X64
    if (rnn_.is_brgemm)
        ref_rnn_brgemm_t::init_scratchpad(
                rnn_, scratchpad, sizeof(gemm_acc_t), alignof(gemm_acc_t));
#endif

    // Below primitives may be run as part of execution.Fortunately, none of
    // them run simulataneously. So, we can re-use the same scratchpad across
    // all primitives. Iterate through them to find the largest scratchpad
    // required.
    const auto nested_pds = {matmul_layer_1_pd_, matmul_layer_2_pd_,
            matmul_layer_3_pd_, matmul_iter_1_pd_, matmul_iter_2_pd_,
            matmul_iter_3_pd_, matmul_part2_1_pd_, matmul_part2_2_pd_,
            matmul_part2_3_pd_, matmul_part2_4_pd_,
#if DNNL_X64
            bf32_wei_layer_reorder_pd_, bf32_wei_iter_reorder_pd_
#endif
    };

    size_t max_nested_scratchpad_size = 0;
    for (const auto &n_pd : nested_pds) {
        if (n_pd)
            max_nested_scratchpad_size = nstl::max(max_nested_scratchpad_size,
                    n_pd->scratchpad_registry().size());
    }

    scratchpad.template book<void *>(
            key_nested_multiple + 0, max_nested_scratchpad_size);
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
status_t dnnl::impl::cpu::ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::init(engine_t *engine) {
    /// @todo set max_feature_size assuming that we limit the number of
    /// iterations and layer to one if slc != dhc and sic != dhc
    /// respectively

    bias_preparation_func = &class_name::bias_prepare;
    bias_finalization_func = &class_name::bias_finalize;

    const auto set_gemm_funcs = [](bool packed_gemm, gemm_t &g,
                                        weights_assign_t &a, bool is_brgemm) {
        if (packed_gemm) {
            g = &class_name::packed_gemm;
            a = &class_name::assign_packed_weights;
        } else {
            g = (!is_brgemm) ? &class_name::gemm : nullptr;
            a = &class_name::assign_weights;
        }
    };
    set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
            weights_iter_assign_func, pd()->rnn_.is_brgemm);

    set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
            weights_layer_assign_func, pd()->rnn_.is_brgemm);

    if (pd()->rnn_.is_lstm_projection) {
        set_gemm_funcs(pd()->rnn_.use_projection_packed_gemm,
                gemm_projection_func, weights_projection_assign_func,
                pd()->rnn_.is_brgemm);
    } else {
        gemm_projection_func = nullptr;
        weights_projection_assign_func = nullptr;
    }

    rnn_postgemm_ = new postgemm_t(pd()->rnn_, pd());
    assert(rnn_postgemm_ != nullptr);
    CHECK(rnn_postgemm_->init(pd()->rnn_));
    if (pd()->rnn_.is_brgemm)
        cell_func = &class_name::cell_execution_brgemm;
    else {
        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_rnn:
            case alg_kind::vanilla_lstm:
                cell_func = &class_name::cell_execution_ref;
                break;
            case alg_kind::vanilla_gru:
            case alg_kind::vanilla_augru:
                cell_func = &class_name::cell_execution_gru;
                break;
            case alg_kind::lbr_augru:
            case alg_kind::lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                break;
            default: break;
        }
    }

    merged_layer_func = pd()->rnn_.is_brgemm && pd()->rnn_.merge_gemm_layer
                    && aprop == prop_kind::forward
            ? &class_name::merged_layer_brgemm
            : &class_name::merged_layer_execution_ref;
    grid_computation = &class_name::linear_execution;

    size_t scratchpad_size, workspace_size;
    rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_ht_offset_,
            ws_states_layer_offset_, ws_states_iter_offset_,
            ws_states_iter_c_offset_, ws_diff_states_layer_offset_,
            ws_diff_states_iter_offset_, ws_diff_states_iter_c_offset_,
            ws_grid_comp_offset_, ws_bias_offset_, scratch_gates_offset_,
            scratch_ht_offset_, scratch_diff_ht_offset_, scratch_cell_offset_,
            scratchpad_size, workspace_size);

#define CREATE_MATMUL(m) \
    if (pd()->m##pd_) { CHECK(pd()->m##pd_->create_primitive(m, engine)); }

    CREATE_MATMUL(matmul_layer_1_);
    CREATE_MATMUL(matmul_layer_2_);
    CREATE_MATMUL(matmul_layer_3_);
    CREATE_MATMUL(matmul_iter_1_);
    CREATE_MATMUL(matmul_iter_2_);
    CREATE_MATMUL(matmul_iter_3_);
    CREATE_MATMUL(matmul_part2_1_);
    CREATE_MATMUL(matmul_part2_2_);
    CREATE_MATMUL(matmul_part2_3_);
    CREATE_MATMUL(matmul_part2_4_);
#undef CREATE_MATMUL

#if DNNL_X64
    const auto rnn = pd()->rnn_;
    if (rnn.is_brgemm) {
        if (rnn.is_bf32()) {

            CHECK(pd()->bf32_wei_layer_reorder_pd_->create_primitive(
                    bf32_wei_layer_reorder_, engine));

            CHECK(pd()->bf32_wei_iter_reorder_pd_->create_primitive(
                    bf32_wei_iter_reorder_, engine));
        }
        return rnn_brgemm_.init_kernels(rnn, src_type, weights_type);
    }
#endif
    return status::success;
}

// GEMM functions wrapper definitions

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_gemm_sig((ref_rnn_fwd_t<src_type, weights_type, acc_type>::gemm)) {
    assert(!"non packed gemm is unavailable for this data type");
    return dnnl_unimplemented;
}

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_gemm_sig((ref_rnn_bwd_t<src_type, weights_type, acc_type>::gemm)) {
    assert(!"non packed gemm is unavailable for this data type");
    return dnnl_unimplemented;
}

template rnn_gemm_sig(ref_rnn_fwd_f16_t::gemm);
template rnn_gemm_sig(ref_rnn_bwd_f16_t::gemm);

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
const std::shared_ptr<primitive_t> &
dnnl::impl::cpu::ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::get_matmul_layer(cell_position_t cell_position) const {
    const auto &rnn = pd()->rnn_;
    const auto src_ld = rnn.src_layer_ld(cell_position);
    const auto LDB1 = rnn.src_layer_ld_;
    const auto LDB2 = rnn.ws_states_layer_ld;
    const auto LDB3 = rnn.dst_iter_ld_;
    MAYBE_UNUSED(LDB3);
    if (src_ld == LDB1)
        return matmul_layer_1_;
    else if (src_ld == LDB2)
        return matmul_layer_2_;
    else {
        assert(src_ld == LDB3);
        return matmul_layer_3_;
    }
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
const std::shared_ptr<primitive_t> &
dnnl::impl::cpu::ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::get_matmul_iter(cell_position_t cell_position) const {
    const auto &rnn = pd()->rnn_;
    const auto src_ld = rnn.src_iter_ld(cell_position);
    const auto LDB1 = rnn.src_iter_ld_;
    const auto LDB2 = rnn.ws_states_iter_ld;
    const auto LDB3 = rnn.dst_layer_ld_;
    MAYBE_UNUSED(LDB3);
    if (src_ld == LDB1)
        return matmul_iter_1_;
    else if (src_ld == LDB2)
        return matmul_iter_2_;
    else {
        assert(src_ld == LDB3);
        return matmul_iter_3_;
    }
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
const std::shared_ptr<primitive_t> &
dnnl::impl::cpu::ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::get_matmul_part2(cell_position_t cell_position) const {
    const auto &rnn = pd()->rnn_;
    const auto ldb = rnn.dst_iter_part2_ld(cell_position);
    const auto LDB1 = rnn.ws_states_layer_ld;
    const auto LDB2 = rnn.ws_states_iter_ld;
    const auto LDB3 = rnn.dst_layer_ld_;
    const auto LDB4 = rnn.dst_iter_ld_;
    MAYBE_UNUSED(LDB4);
    if (ldb == LDB1)
        return matmul_part2_1_;
    else if (ldb == LDB2)
        return matmul_part2_2_;
    else if (ldb == LDB3)
        return matmul_part2_3_;
    else {
        assert(ldb == LDB4);
        return matmul_part2_4_;
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_matmul_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::execute_matmul)) {

    // Service engine is just a global classic CPU engine that is used
    // when it's required to create memory_t objects for classic CPU
    // engine regardless of the CPU runtime. For example, SYCL CPU engine
    // cannot be used to create such objects.
    engine_t *service_engine = get_service_engine();
    constexpr auto mem_flag = memory_flags_t::use_runtime_ptr;

    // a_, b_ and c_ are regular, raw CPU pointers that can only be used with
    // memory_t objects created for the classic CPU engine.
    std::unique_ptr<memory_t, memory_deleter_t> src_mem;
    CHECK(safe_ptr_assign(src_mem,
            new memory_t(service_engine, matmul_prim->pd()->src_md(), mem_flag,
                    (void *)(a_))));
    std::unique_ptr<memory_t, memory_deleter_t> wei_mem;
    CHECK(safe_ptr_assign(wei_mem,
            new memory_t(service_engine, matmul_prim->pd()->weights_md(),
                    mem_flag, (void *)(b_))));
    std::unique_ptr<memory_t, memory_deleter_t> dst_mem;
    CHECK(safe_ptr_assign(dst_mem,
            new memory_t(service_engine, matmul_prim->pd()->dst_md(), mem_flag,
                    (void *)(c_))));

    exec_args_t matmul_args;
    // Note Matmul src and wei may not directly map to RNN primitive src and wei
    matmul_args[DNNL_ARG_SRC] = {wei_mem.get(), true};
    matmul_args[DNNL_ARG_WEIGHTS] = {src_mem.get(), true};
    matmul_args[DNNL_ARG_DST] = {dst_mem.get(), false};

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested_multiple, matmul_prim->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_prim->execute(matmul_ctx);
}

template <>
rnn_gemm_sig((ref_rnn_fwd_f32_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC, nullptr, pd()->rnn_.force_nocopy);
}

template <>
rnn_gemm_sig((ref_rnn_bwd_f32_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return extended_sgemm(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC, nullptr, pd()->rnn_.force_nocopy);
}

template <>
rnn_gemm_sig((ref_rnn_fwd_bf16_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return gemm_bf16bf16f32(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig((ref_rnn_bwd_bf16_t::gemm)) {
    assert(ldA * ldB * ldC != 0);
    return gemm_bf16bf16f32(&transA, &transB, &m, &n, &k, &alpha, a_, &ldA, b_,
            &ldB, &beta, c_, &ldC);
}

// packed GEMM functions wrapper definitions

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_gemm_sig((ref_rnn_fwd_t<src_type, weights_type, acc_type>::packed_gemm)) {
    assert(!"packed gemm is unavailable for this datatype");
    return dnnl_unimplemented;
}

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_gemm_sig((ref_rnn_bwd_t<src_type, weights_type, acc_type>::packed_gemm)) {
    assert(!"packed gemm is unavailable for this datatype");
    return dnnl_unimplemented;
}

template rnn_gemm_sig(ref_rnn_fwd_f16_t::packed_gemm);
template rnn_gemm_sig(ref_rnn_bwd_f16_t::packed_gemm);

template <>
rnn_gemm_sig(ref_rnn_fwd_f32_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return sgemm_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig(ref_rnn_bwd_f32_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return sgemm_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig((ref_rnn_fwd_bf16_t::packed_gemm)) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return gemm_bf16bf16f32_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig((ref_rnn_bwd_bf16_t::packed_gemm)) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    return gemm_bf16bf16f32_compute(
            "P", "N", &m, &n, &k, a_, &ldA, b_, &ldB, &beta, c_, &ldC);
}

template <>
rnn_gemm_sig(ref_rnn_fwd_u8s8_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    int32_t offsetc = 0;
    return gemm_s8u8s32_compute("P", "N", "F", &m, &n, &k, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, &offsetc);
}

template <>
rnn_gemm_sig(ref_rnn_fwd_s8s8_t::packed_gemm) {
    assert(transA == 'N' && transB == 'N' && alpha == 1.);
    int32_t offsetc = 0;
    return gemm_s8s8s32_compute("P", "N", "F", &m, &n, &k, a_, &ldA, b_, &ldB,
            &beta, c_, &ldC, &offsetc);
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_grid_execution_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::linear_execution)) {
    const AOC<src_layer_t, 4> ws_states_layer(ws_states_layer_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_states_layer_nld * rnn.ws_states_layer_ld);
    const AOC<const src_layer_t, 3> augru_attention(
            augru_attention_, rnn.n_iter, rnn.mb, 1);
    const AOC<src_iter_t, 4> ws_states_iter(ws_states_iter_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_states_iter_nld * rnn.ws_states_iter_ld);
    const auto ws_states_iter_c = rnn_utils::make_raw_aoc(ws_states_iter_c_,
            types::data_type_size(rnn.src_iter_c_dt), rnn.n_layer + 1,
            rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_diff_states_iter_c_nld * rnn.ws_diff_states_iter_c_ld);
    const AOC<gemm_acc_t, 4> ws_diff_states_layer(ws_diff_states_layer_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_diff_states_layer_nld * rnn.ws_diff_states_layer_ld);
    const AOC<gemm_acc_t, 3> diff_augru_attention(
            diff_augru_attention_, rnn.n_iter, rnn.mb, 1);
    const AOC<gemm_acc_t, 4> ws_diff_states_iter(ws_diff_states_iter_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_diff_states_iter_nld * rnn.ws_diff_states_iter_ld);
    const AOC<gemm_acc_t, 4> ws_diff_states_iter_c(ws_diff_states_iter_c_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1,
            rnn.ws_diff_states_iter_c_nld * rnn.ws_diff_states_iter_c_ld);
    const AOC<gates_t, 4> ws_gates(ws_gates_, rnn.n_layer, rnn.n_dir,
            rnn.n_iter, rnn.ws_gates_nld * rnn.ws_gates_ld);
    const AOC<dst_iter_t, 4> ws_ht(ws_ht_, rnn.n_layer, rnn.n_dir, rnn.n_iter,
            rnn.ws_ht_nld * rnn.ws_ht_ld);
    const AOC<weights_t *, 3> weights_layer(
            weights_layer_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_layer);
    const AOC<weights_t *, 3> weights_iter(
            weights_iter_, rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_iter);
    const AOC<weights_t *, 2> weights_projection(
            weights_projection_, rnn.n_layer, rnn.n_dir);
    const AOC<const float, 3> weights_peephole(
            weights_peephole_, rnn.n_layer, rnn.n_dir, 3 * rnn.dhc);
    bias_linear_exec_aoc_t bias(rnn, bias_);
    const AOC<gemm_acc_t, 3> diff_weights_layer(diff_weights_layer_,
            rnn.n_layer, rnn.n_dir,
            rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ld);
    const AOC<gemm_acc_t, 3> diff_weights_iter(diff_weights_iter_, rnn.n_layer,
            rnn.n_dir, rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ld);
    const AOC<float, 3> diff_weights_peephole(
            diff_weights_peephole_, rnn.n_layer, rnn.n_dir, 3 * rnn.dhc);
    const AOC<float, 3> diff_weights_projection(diff_weights_projection_,
            rnn.n_layer, rnn.n_dir,
            rnn.diff_weights_projection_nld * rnn.diff_weights_projection_ld);
    const AOC<float, 3> diff_bias(
            diff_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dhc);
    const AOC<gates_t, 4> ws_grid(
            ws_grid_, rnn.n_layer, rnn.n_dir, rnn.n_iter, (int)rnn.ws_per_cell);

    /* Raw inputs/outputs coming from the user */
    // Here we cannot use AOC as user's input can have arbitrary strides, so we use desc_wrapper.
    const auto src_layer_mdw = memory_desc_wrapper(pd()->src_md(0));
    const auto dst_layer_mdw = memory_desc_wrapper(pd()->dst_md(0));
    const auto src_iter_mdw = memory_desc_wrapper(pd()->src_md(1));
    const auto dst_iter_mdw = memory_desc_wrapper(pd()->dst_md(1));
    const auto src_iter_c_mdw = memory_desc_wrapper(pd()->src_md(2));
    const auto dst_iter_c_mdw = memory_desc_wrapper(pd()->dst_md(2));

// Since the function FN(...) returns by reference so an extra exception
// has to be made for nullptr argument
#define SAFE_PTR(FN, ...) CONCAT2(FN, _) ? &(FN(__VA_ARGS__)) : nullptr
    const auto compute_merged_layer_part_if_applicable
            = [&](prop_kind_t target_prop, int dir, int lay) {
        if (IMPLICATION(rnn.merge_gemm_layer, aprop != target_prop))
            return dnnl_success;

        cell_position_t cell_position = middle_cell;
        if (lay == 0) cell_position |= first_layer;
        cell_position |= merged_layer;

        const src_layer_t *src_layer = lay == 0 && rnn.skip_src_layer_copy()
                ? src_layer_
                : SAFE_PTR(ws_states_layer, lay, dir, 1, 0);
#if DNNL_X64
        CHECK((this->*merged_layer_func)(ctx, rnn, cell_position,
                SAFE_PTR(weights_layer, lay, dir, 0), src_layer, scratch_gates_,
                SAFE_PTR(ws_diff_states_layer, lay, dir, 0, 0),
                SAFE_PTR(diff_weights_layer, lay, dir, 0), amx_scratchpad,
                addr_batch_global));
#else
        CHECK((this->*merged_layer_func)(rnn, cell_position,
                SAFE_PTR(weights_layer, lay, dir, 0), src_layer, scratch_gates_,
                SAFE_PTR(ws_diff_states_layer, lay, dir, 0, 0),
                SAFE_PTR(diff_weights_layer, lay, dir, 0)));
#endif
        return dnnl_success;
    };

    // We run the grid of computation
    for_(int dir = 0; dir < rnn.n_dir; dir++)
    for (int j = 0; j < rnn.n_layer; j++) {
        const int lay = static_cast<int>(
                (aprop == prop_kind::forward) ? j : rnn.n_layer - j - 1);

        CHECK(compute_merged_layer_part_if_applicable(
                prop_kind::forward, dir, lay));

        // TODO: enable merging projection gemm in bwd lstm projection

        for (int i = 0; i < rnn.n_iter; i++) {
            const int iter = static_cast<int>(
                    (aprop == prop_kind::forward) ? i : rnn.n_iter - i - 1);

            // We set parameters to the cell execution call

            // dst_layer is equal to dst_iter. To avoid
            // duplication of memory access we hence use only
            // dst_layer and set dst_iter to nullptr, unless we
            // cannot for one of the following condition:
            // - in the last layer and last iteration, we need to
            //   copy ht in two tensors (dst_layer and dst_iter)
            dst_layer_t *cell_dst_layer
                    = &(ws_states_layer(lay + 1, dir, iter + 1, 0));
            dst_iter_t *cell_dst_iter = nullptr;
            const src_layer_t *cell_src_layer
                    = &(ws_states_layer(lay, dir, iter + 1, 0));
            const src_iter_t *cell_src_iter
                    = &(ws_states_iter(lay + 1, dir, iter, 0));

            void *cell_dst_iter_c = const_cast<void *>(
                    ws_states_iter_c(lay + 1, dir, iter + 1, 0));
            const void *cell_src_iter_c
                    = ws_states_iter_c(lay + 1, dir, iter, 0);

            // the cell_position is used only when skip_data_copy is
            // supported currently supported only for forward
            cell_position_t cell_position = middle_cell;
            if (iter == 0) cell_position |= first_iter;
            if (lay == 0) cell_position |= first_layer;
            if (iter == rnn.n_iter - 1) cell_position |= last_iter;
            if (lay == rnn.n_layer - 1) cell_position |= last_layer;

            // The dst_* paths should be before the src_* paths as
            // the later will override cell_src_layer and
            // cell_src_iter appropriately for 1st layer and 1st
            // iter.
            const bool last_iter_skip_copy
                    = rnn.skip_dst_iter_copy() && (cell_position & last_iter);
            if (last_iter_skip_copy) {
                cell_dst_layer = dst_iter_ + dst_iter_mdw.off(lay, dir, 0, 0);
                cell_src_layer
                        = dst_iter_ + dst_iter_mdw.off(lay - 1, dir, 0, 0);
            }

            if (rnn.skip_dst_layer_copy() && (cell_position & last_layer)) {
                // Note: for last layer and last iter, the output is in dst_layer
                // and still need to be copied to dst_iter
                cell_dst_layer = dst_layer_ + dst_layer_mdw.off(iter, 0, 0);
                cell_dst_iter = last_iter_skip_copy
                        ? dst_iter_ + dst_iter_mdw.off(lay, dir, 0, 0)
                        : nullptr;
                cell_src_iter = (iter != 0)
                        ? dst_layer_ + dst_layer_mdw.off(iter - 1, 0, 0)
                        : cell_src_iter;
            }
            if (rnn.skip_src_iter_copy() && (cell_position & first_iter))
                cell_src_iter = src_iter_ + src_iter_mdw.off(lay, dir, 0, 0);

            if (rnn.skip_src_layer_copy() && (cell_position & first_layer))
                cell_src_layer = src_layer_ + src_layer_mdw.off(iter, 0, 0);

            // because the c state is always f32 and require no
            // conversion, we can always skip to copy for the 1st
            // and last iteration
            if (iter == 0 && src_iter_c_) {
                cell_src_iter_c = inc_ptr(src_iter_c_, rnn.src_iter_c_dt,
                        src_iter_c_mdw.off(lay, dir, 0, 0));
                cell_position |= c_state_first_iter;
            }
            if (iter == rnn.n_iter - 1 && dst_iter_c_) {
                cell_dst_iter_c = inc_ptr(dst_iter_c_, rnn.dst_iter_c_dt,
                        dst_iter_c_mdw.off(lay, dir, 0, 0));
                cell_position |= c_state_last_iter;
            }
            const size_t sg_start_idx = rnn.n_iter_scratch_gates == 1
                    ? static_cast<size_t>(0)
                    : static_cast<size_t>(iter) * rnn.scratch_gates_nld
                            * rnn.scratch_gates_ld;
            const auto cell_scratch_gates = &scratch_gates_[sg_start_idx];

            dst_iter_t *proj_ht = nullptr;
            if (rnn.is_lstm_projection) {
                if (rnn.is_training)
                    proj_ht = &(ws_ht(lay, dir, iter, 0));
                else
                    proj_ht = scratch_ht_;
            }

#if DNNL_X64
            CHECK((this->*cell_func)(ctx, rnn, cell_position, cell_dst_layer,
                    cell_dst_iter_c,
                    SAFE_PTR(ws_diff_states_layer, lay, dir, iter, 0),
                    SAFE_PTR(diff_augru_attention, iter, 0, 0),
                    SAFE_PTR(ws_diff_states_iter, lay, dir, iter, 0),
                    SAFE_PTR(ws_diff_states_iter_c, lay, dir, iter, 0),
                    SAFE_PTR(weights_layer, lay, dir, 0),
                    SAFE_PTR(weights_iter, lay, dir, 0),
                    SAFE_PTR(weights_projection, lay, dir),
                    SAFE_PTR(weights_peephole, lay, dir, 0),
                    w_proj_comp ? w_proj_comp + (j * rnn.n_dir + dir) * rnn.dic
                                : nullptr,
                    bias(lay, dir), cell_src_layer,
                    SAFE_PTR(augru_attention, iter, 0, 0), cell_src_iter,
                    cell_src_iter_c,
                    SAFE_PTR(ws_diff_states_layer, lay + 1, dir, iter, 0),
                    SAFE_PTR(ws_diff_states_iter, lay, dir, iter + 1, 0),
                    SAFE_PTR(ws_diff_states_iter_c, lay, dir, iter + 1, 0),
                    SAFE_PTR(diff_weights_layer, lay, dir, 0),
                    SAFE_PTR(diff_weights_iter, lay, dir, 0),
                    SAFE_PTR(diff_weights_projection, lay, dir, 0),
                    SAFE_PTR(diff_weights_peephole, lay, dir, 0),
                    SAFE_PTR(diff_bias, lay, dir, 0),
                    SAFE_PTR(ws_gates, lay, dir, iter, 0), cell_scratch_gates,
                    proj_ht, scratch_diff_ht_,
                    SAFE_PTR(ws_grid, lay, dir, iter, 0), scratch_cell_,
                    scratch_gates_blocked_, scratch_src_layer_,
                    scratch_src_iter_, cell_dst_iter, amx_scratchpad,
                    addr_batch_global));
#else
            CHECK((this->*cell_func)(ctx, rnn, cell_position, cell_dst_layer,
                    cell_dst_iter_c,
                    SAFE_PTR(ws_diff_states_layer, lay, dir, iter, 0),
                    SAFE_PTR(diff_augru_attention, iter, 0, 0),
                    SAFE_PTR(ws_diff_states_iter, lay, dir, iter, 0),
                    SAFE_PTR(ws_diff_states_iter_c, lay, dir, iter, 0),
                    SAFE_PTR(weights_layer, lay, dir, 0),
                    SAFE_PTR(weights_iter, lay, dir, 0),
                    SAFE_PTR(weights_projection, lay, dir),
                    SAFE_PTR(weights_peephole, lay, dir, 0),
                    w_proj_comp ? w_proj_comp + (j * rnn.n_dir + dir) * rnn.dic
                                : nullptr,
                    bias(lay, dir), cell_src_layer,
                    SAFE_PTR(augru_attention, iter, 0, 0), cell_src_iter,
                    cell_src_iter_c,
                    SAFE_PTR(ws_diff_states_layer, lay + 1, dir, iter, 0),
                    SAFE_PTR(ws_diff_states_iter, lay, dir, iter + 1, 0),
                    SAFE_PTR(ws_diff_states_iter_c, lay, dir, iter + 1, 0),
                    SAFE_PTR(diff_weights_layer, lay, dir, 0),
                    SAFE_PTR(diff_weights_iter, lay, dir, 0),
                    SAFE_PTR(diff_weights_projection, lay, dir, 0),
                    SAFE_PTR(diff_weights_peephole, lay, dir, 0),
                    SAFE_PTR(diff_bias, lay, dir, 0),
                    SAFE_PTR(ws_gates, lay, dir, iter, 0), cell_scratch_gates,
                    proj_ht, scratch_diff_ht_,
                    SAFE_PTR(ws_grid, lay, dir, iter, 0), scratch_cell_,
                    cell_dst_iter, amx_scratchpad));
#endif
        }

        CHECK(compute_merged_layer_part_if_applicable(
                prop_kind::backward, dir, lay));

#undef SAFE_PTR

        if ((aprop == prop_kind::backward) && rnn.merge_gemm_iter) {
            // This is split in 3 pieces if we skip copies.
            // last iter in user mem, middle iters in ws, first iter in user mem
            // Note 1: here we assume no change in datatypes for src_iter, ws_iter and dst_iter

            const dst_iter_t *states_iter = nullptr;
            dim_t states_iter_ld = 0;
            dim_t niter_merge_gemm_iter = 0;

            states_iter = &(
                    ws_states_iter(lay + 1, dir, rnn.skip_src_iter_copy(), 0));
            states_iter_ld = rnn.ws_states_iter_ld;
            if (rnn.skip_dst_layer_copy()
                    && (lay == rnn.n_layer - 1)) { // last layer
                states_iter = dst_layer_;
                states_iter_ld = rnn.dst_layer_ld_;
            }
            niter_merge_gemm_iter = rnn.n_iter - rnn.skip_src_iter_copy();
            if (niter_merge_gemm_iter > 0) {
                CHECK(gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic,
                        rnn.mb * niter_merge_gemm_iter, 1.0,
                        (weights_t *)scratch_gates_
                                + rnn.skip_src_iter_copy()
                                        * rnn.scratch_gates_nld
                                        * rnn.scratch_gates_ld,
                        rnn.scratch_gates_ld, states_iter, states_iter_ld,
                        rnn.diff_weights_beta(cell_position_t::merged_iter),
                        &(diff_weights_iter(lay, dir, 0)),
                        rnn.diff_weights_iter_ld));
            }

            if (rnn.skip_src_iter_copy()) {
                states_iter = src_iter_ + src_iter_mdw.off(lay, dir, 0, 0);
                states_iter_ld = rnn.src_iter_ld_;
                CHECK(gemm('N', 'T', rnn.n_gates * rnn.dhc, rnn.sic, rnn.mb,
                        1.0, (weights_t *)scratch_gates_, rnn.scratch_gates_ld,
                        states_iter, states_iter_ld,
                        rnn.diff_weights_beta(niter_merge_gemm_iter
                                        ? cell_position_t::middle_cell
                                        : cell_position_t::merged_iter),
                        &(diff_weights_iter(lay, dir, 0)),
                        rnn.diff_weights_iter_ld));
            }
        }
    }
    return dnnl_success;
}

//********* GRID computations strategy: utility functions **********//

// for bf32 src_data_t(bf16) and input_data_t(f32) types can be different.
template <typename src_data_t, typename input_data_t>
void copy_init_layer_fwd_template(const rnn_conf_t &rnn,
        src_data_t *__restrict ws_states_layer_,
        const input_data_t *__restrict xt_, const memory_desc_wrapper &xt_d) {

    const AOC<src_data_t, 4> ws_states_layer(ws_states_layer_, rnn.n_dir,
            rnn.n_iter + 1, rnn.mb, rnn.ws_states_layer_ld);

    parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
        auto xxt = xt_ + xt_d.blk_off(it, b);
        src_data_t *ws_l2r_ptr = &(ws_states_layer(0, it + 1, b, 0));
        src_data_t *ws_r2l_ptr
                = &(ws_states_layer(rnn.n_dir - 1, rnn.n_iter - it, b, 0));
        if (rnn.exec_dir != r2l) {
            if (rnn.is_bf32()) {
                cvt_float_to_bfloat16(
                        (bfloat16_t *)ws_l2r_ptr, (const float *)xxt, rnn.slc);
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < rnn.slc; c++)
                    ws_l2r_ptr[c] = static_cast<src_data_t>(xxt[c]);
            }
        }
        if (rnn.exec_dir != l2r) {
            if (rnn.is_bf32()) {
                cvt_float_to_bfloat16(
                        (bfloat16_t *)ws_r2l_ptr, (const float *)xxt, rnn.slc);
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < rnn.slc; c++)
                    ws_r2l_ptr[c] = static_cast<src_data_t>(xxt[c]);
            }
        }
    });
}

template <typename acc_data_t>
void copy_init_layer_bwd_template(const rnn_conf_t &rnn,
        acc_data_t *ws_diff_states_layer_, const acc_data_t *diff_dst_layer_,
        const memory_desc_wrapper &diff_dst_layer_d) {
    const AOC<acc_data_t, 5> ws_diff_states_layer(ws_diff_states_layer_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_layer_ld);

    switch (rnn.exec_dir) {
        case bi_concat:
            parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
                const auto diff_dst_layer_x
                        = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
                for (int s = 0; s < rnn.dlc; s++) {
                    ws_diff_states_layer(rnn.n_layer, 0, it, b, s)
                            = diff_dst_layer_x[s];
                    ws_diff_states_layer(
                            rnn.n_layer, 1, rnn.n_iter - it - 1, b, s)
                            = diff_dst_layer_x[rnn.dlc + s];
                }
            });
            break;
        case bi_sum:
            parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
                const auto diff_dst_layer_x
                        = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
                for (int s = 0; s < rnn.dlc; s++) {
                    ws_diff_states_layer(rnn.n_layer, 0, it, b, s)
                            = diff_dst_layer_x[s];
                    ws_diff_states_layer(
                            rnn.n_layer, 1, rnn.n_iter - it - 1, b, s)
                            = diff_dst_layer_x[s];
                }
            });
            break;
        case l2r:
            parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
                const auto diff_dst_layer_x
                        = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
                for (int s = 0; s < rnn.dlc; s++) {
                    ws_diff_states_layer(rnn.n_layer, 0, it, b, s)
                            = diff_dst_layer_x[s];
                }
            });
            break;
        case r2l:
            parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
                const auto diff_dst_layer_x = diff_dst_layer_
                        + diff_dst_layer_d.blk_off(rnn.n_iter - it - 1, b);
                for (int s = 0; s < rnn.dlc; s++) {
                    ws_diff_states_layer(rnn.n_layer, 0, it, b, s)
                            = diff_dst_layer_x[s];
                }
            });
            break;
        default: assert(!"Unsupported direction"); break;
    }
}

#define RNN_DECL_COPY_INIT_LAYER_FWD(cname) \
    template <> \
    template <typename input_data_t> \
    void cname::copy_init_layer(const rnn_conf_t &rnn, \
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_, \
            const input_data_t *xt_, const gemm_acc_t *diff_dst_layer_) \
            const { \
        copy_init_layer_fwd_template(rnn, ws_states_layer_, xt_, \
                memory_desc_wrapper(pd()->src_md(0))); \
    }

RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_bf16_t)
RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_f16_t)
RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_INIT_LAYER_FWD(ref_rnn_common_fwd_s8s8_t)

#define RNN_DECL_COPY_INIT_LAYER_BWD(cname) \
    template <> \
    template <typename input_data_t> \
    void cname::copy_init_layer(const rnn_conf_t &rnn, \
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_, \
            const input_data_t *xt_, const gemm_acc_t *diff_dst_layer_) \
            const { \
        copy_init_layer_bwd_template(rnn, ws_diff_states_layer_, \
                diff_dst_layer_, memory_desc_wrapper(pd()->diff_dst_md(0))); \
    }

RNN_DECL_COPY_INIT_LAYER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_INIT_LAYER_BWD(ref_rnn_common_bwd_bf16_t)
RNN_DECL_COPY_INIT_LAYER_BWD(ref_rnn_common_bwd_f16_t)

/* For int8 configuration, input iteration states may be of types f32 or u8
 * Internally h_state is always stored in u8 and c_state is always stored in f32
 * If input states are of type u8 then h state is copied and c state is dequantized
 * If input states are of type f32 then h state is quantized and c_state is copied
 * */
template <typename src_data_t, typename input_data_t>
void copy_init_iter_fwd_template(const rnn_conf_t &rnn, const rnn_pd_t *pd,
        src_data_t *__restrict ws_states_iter_,
        void *__restrict ws_states_iter_c_,
        const input_data_t *__restrict src_iter_,
        const memory_desc_wrapper &src_iter_d,
        const void *__restrict src_iter_c_,
        const memory_desc_wrapper &src_iter_c_d) {
    const AOC<src_data_t, 5> ws_states_iter(ws_states_iter_, rnn.n_layer + 1,
            rnn.n_dir, rnn.n_iter + 1, rnn.mb, rnn.ws_states_iter_ld);
    const auto ws_states_iter_c_aoc = rnn_utils::make_raw_aoc(ws_states_iter_c_,
            types::data_type_size(rnn.src_iter_c_dt), rnn.n_layer + 1,
            rnn.n_dir, rnn.n_iter + 1, rnn.mb, rnn.ws_states_iter_c_ld);

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const bool quantize = rnn.is_int8_conf()
            && IMPLICATION(pd->with_src_iter(),
                    pd->src_md(1)->data_type == data_type::f32);
    const auto maybe_q = [&](input_data_t f) {
        if (quantize) {
            float qf = f * data_scale + data_shift;
            return q10n::qz_a1b0_t<float, src_data_t>()(qf);
        } else
            return (src_data_t)f;
    };
    const src_data_t zero = maybe_q(static_cast<input_data_t>(0.f));
    const auto zero_ws_iter_c
            = [&](dim_t lay, dim_t dir, dim_t mb_id, dim_t sic_id) {
        void *ws_states_iter_c = const_cast<void *>(
                ws_states_iter_c_aoc(lay, dir, 0, mb_id, sic_id));
        if (rnn.src_iter_c_dt == data_type::f32)
            *(static_cast<float *>(ws_states_iter_c)) = 0.0f;
        else if (rnn.src_iter_c_dt == data_type::bf16)
            *(static_cast<bfloat16_t *>(ws_states_iter_c)) = 0.0f;
        else if (rnn.src_iter_c_dt == data_type::f16)
            *(static_cast<float16_t *>(ws_states_iter_c)) = 0.0f;
    };

    if (src_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](dim_t lay, dim_t dir, dim_t b) {
            const auto *ss = &src_iter_[src_iter_d.blk_off(lay, dir, b, 0)];
            auto *dd = &ws_states_iter(lay + 1, dir, 0, b, 0);
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.sic; s++)
                dd[s] = maybe_q(ss[s]);
        });
    } else {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](dim_t lay, dim_t dir, dim_t b) {
            for (int j = 0; j < rnn.sic; j++)
                ws_states_iter(lay + 1, dir, 0, b, j) = zero;
            if (pd->cell_kind() == alg_kind::vanilla_lstm)
                for (int j = 0; j < rnn.dhc; j++)
                    zero_ws_iter_c(lay + 1, dir, b, j);
        });
    }
}

template <typename acc_data_t>
void copy_init_iter_bwd_template(const rnn_conf_t &rnn, const rnn_pd_t *pd,
        acc_data_t *ws_diff_states_iter_, acc_data_t *ws_diff_states_iter_c_,
        const acc_data_t *diff_dst_iter_,
        const memory_desc_wrapper diff_dst_iter_d,
        const float *diff_dst_iter_c_,
        const memory_desc_wrapper diff_dst_iter_c_d) {
    const AOC<acc_data_t, 5> ws_diff_states_iter(ws_diff_states_iter_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_iter_ld);
    const AOC<acc_data_t, 5> ws_diff_states_iter_c(ws_diff_states_iter_c_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_iter_c_ld);
    if (diff_dst_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](dim_t lay, dim_t dir, dim_t b) {
            array_copy(&(ws_diff_states_iter(lay, dir, rnn.n_iter, b, 0)),
                    diff_dst_iter_ + diff_dst_iter_d.blk_off(lay, dir, b),
                    rnn.dic);
            if (pd->cell_kind() == alg_kind::vanilla_lstm)
                array_copy(&(ws_diff_states_iter_c(lay, dir, rnn.n_iter, b, 0)),
                        diff_dst_iter_c_
                                + diff_dst_iter_c_d.blk_off(lay, dir, b),
                        rnn.dhc);
        });
    } else {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](dim_t lay, dim_t dir, dim_t i) {
            for (int j = 0; j < rnn.dic; j++)
                ws_diff_states_iter(lay, dir, rnn.n_iter, i, j) = 0.0f;
            if (pd->cell_kind() == alg_kind::vanilla_lstm)
                for (int j = 0; j < rnn.dhc; j++)
                    ws_diff_states_iter_c(lay, dir, rnn.n_iter, i, j) = 0.0f;
        });
    }
}

#define RNN_DECL_COPY_INIT_ITER_FWD(cname) \
    template <> \
    template <typename input_data_t> \
    void cname::copy_init_iter(const rnn_conf_t &rnn, \
            src_layer_t *__restrict ws_states_iter_, \
            void *__restrict ws_states_iter_c_, \
            gemm_acc_t *__restrict ws_diff_states_iter_, \
            gemm_acc_t *__restrict ws_diff_states_iter_c_, \
            const input_data_t *__restrict src_iter_, \
            const void *__restrict src_iter_c_, \
            const gemm_acc_t *__restrict diff_dst_iter_, \
            const float *__restrict diff_dst_iter_c_) const { \
        auto src_iter_d = memory_desc_wrapper(pd()->src_md(1)); \
        auto src_iter_c_d = memory_desc_wrapper(pd()->src_md(2)); \
        copy_init_iter_fwd_template(rnn, pd(), ws_states_iter_, \
                ws_states_iter_c_, src_iter_, src_iter_d, src_iter_c_, \
                src_iter_c_d); \
    }

RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_bf16_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_f16_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_INIT_ITER_FWD(ref_rnn_common_fwd_s8s8_t)

#define RNN_DECL_COPY_INIT_ITER_BWD(cname) \
    template <> \
    template <typename input_data_t> \
    void cname::copy_init_iter(const rnn_conf_t &rnn, \
            src_layer_t *ws_states_iter_, void *ws_states_iter_c_, \
            gemm_acc_t *ws_diff_states_iter_, \
            gemm_acc_t *ws_diff_states_iter_c_, const input_data_t *src_iter_, \
            const void *src_iter_c_, const gemm_acc_t *diff_dst_iter_, \
            const float *diff_dst_iter_c_) const { \
        auto diff_dst_iter_d = memory_desc_wrapper(pd()->diff_dst_md(1)); \
        auto diff_dst_iter_c_d = memory_desc_wrapper(pd()->diff_dst_md(2)); \
        copy_init_iter_bwd_template(rnn, pd(), ws_diff_states_iter_, \
                ws_diff_states_iter_c_, diff_dst_iter_, diff_dst_iter_d, \
                diff_dst_iter_c_, diff_dst_iter_c_d); \
    }

RNN_DECL_COPY_INIT_ITER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_INIT_ITER_BWD(ref_rnn_common_bwd_bf16_t)
RNN_DECL_COPY_INIT_ITER_BWD(ref_rnn_common_bwd_f16_t)

template <typename src_data_t, typename dst_layer_dt, typename dst_iter_dt>
void copy_res_layer_fwd_template(const rnn_conf_t &rnn, const rnn_pd_t *pd,
        dst_layer_dt *dst_layer_, memory_desc_wrapper &dst_layer_d,
        const dst_iter_dt *dst_iter_, const memory_desc_wrapper &dst_iter_d,
        const src_data_t *ws_states_layer_) {

    const AOC<const src_data_t, 5> ws_states_layer(ws_states_layer_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_states_layer_ld);
    const float shift = (pd->attr()->rnn_data_qparams_.shift_);
    const float scale = (pd->attr()->rnn_data_qparams_.scale_);

    const bool dequantize
            = pd->dst_md(0)->data_type == data_type::f32 && rnn.is_int8_conf();
    const bool dequantize_at_copy = dequantize && rnn.exec_dir != bi_sum;

    // minor optimization helper for a compiler
    static constexpr bool rnn_u8u8_case
            = std::is_same<dst_layer_dt, uint8_t>::value
            && std::is_same<src_data_t, uint8_t>::value;
    static constexpr bool rnn_s8s8_case
            = std::is_same<dst_layer_dt, int8_t>::value
            && std::is_same<src_data_t, int8_t>::value;

    const auto copy_vec = [&](dst_layer_dt *dd, const src_data_t *ss) {
        if (dequantize_at_copy) {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dlc; s++)
                dd[s] = (dst_layer_dt)(((float)ss[s] - shift) / scale);
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dlc; s++)
                dd[s] = (dst_layer_dt)ss[s];
        }
    };

    const auto acc_vec = [&](dst_layer_dt *dd, const src_data_t *ss) {
        if (dequantize) {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dlc; s++) {
                float val = (float)ss[s] + dd[s];
                val = q10n::qz_a1b0_t<float, src_data_t>()(val);
                dd[s] = (dst_layer_dt)((val - 2 * shift) / scale);
            }
        } else if (rnn_u8u8_case
                || rnn_s8s8_case) { // instead of checking for rnn.is_int8()
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dlc; s++)
                dd[s] = static_cast<dst_layer_dt>(
                        q10n::saturate<dst_layer_dt, int16_t>(
                                (int16_t)dd[s] + (int16_t)ss[s]));
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dlc; s++)
                dd[s] += (dst_layer_dt)ss[s];
        }
    };

    // if skip_dst_iter_copy, then the data for the last iteration is
    // in dst_iter, not in workspace
    parallel_nd(rnn.n_iter - (rnn.skip_dst_iter_copy() ? 1 : 0), rnn.mb,
            [&](dim_t it, dim_t b) {
        int dir = 0;
        if (rnn.exec_dir != r2l) {
            const auto *ss = &ws_states_layer(rnn.n_layer, dir, it + 1, b, 0);
            auto *dd = &dst_layer_[dst_layer_d.blk_off(it, b, dir * rnn.dlc)];
            copy_vec(dd, ss);
            dir = 1;
        }
        if (rnn.exec_dir != l2r) {
            const auto *ss
                    = &ws_states_layer(rnn.n_layer, dir, rnn.n_iter - it, b, 0);
            if (rnn.exec_dir == bi_sum) {
                auto *dd = &dst_layer_[dst_layer_d.blk_off(it, b, 0)];
                acc_vec(dd, ss);
            } else {
                auto *dd = &dst_layer_[dst_layer_d.blk_off(
                        it, b, dir * rnn.dlc)];
                copy_vec(dd, ss);
            }
        }
    });
    if (rnn.skip_dst_iter_copy()) {
        parallel_nd(rnn.mb, [&](dim_t b) {
            const dim_t it = rnn.n_iter - 1;
            int dir = 0;
            if (rnn.exec_dir != r2l) {
                const auto *ss = dst_iter_
                        + dst_iter_d.blk_off(rnn.n_layer - 1, dir, b, 0);
                auto *dd = &dst_layer_[dst_layer_d.blk_off(
                        it, b, dir * rnn.dlc)];
                copy_vec(dd, (src_data_t *)ss);
                dir = 1;
            }
            if (rnn.exec_dir != l2r) {
                const auto *ss = dst_iter_
                        + dst_iter_d.blk_off(rnn.n_layer - 1, dir, b, 0);
                if (rnn.exec_dir == bi_sum) {
                    auto *dd = &dst_layer_[dst_layer_d.blk_off(it, b, 0)];
                    acc_vec(dd, (src_data_t *)ss);
                } else {
                    auto *dd = &dst_layer_[dst_layer_d.blk_off(
                            it, b, dir * rnn.dlc)];
                    copy_vec(dd, (src_data_t *)ss);
                }
            }
        });
    }
}

template <typename acc_data_t>
void copy_res_layer_bwd_template(const rnn_conf_t &rnn,
        acc_data_t *diff_src_layer_, memory_desc_wrapper &diff_src_layer_d,
        const acc_data_t *ws_diff_states_layer_) {
    const AOC<const acc_data_t, 5> ws_diff_states_layer(ws_diff_states_layer_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_layer_ld);

    parallel_nd(rnn.n_iter, rnn.mb, [&](dim_t it, dim_t b) {
        int dir = 0;
        for (int s = 0; s < rnn.slc; s++) {
            acc_data_t *dst_addr = diff_src_layer_
                    + diff_src_layer_d.blk_off(
                            (rnn.exec_dir == r2l) ? rnn.n_iter - 1 - it : it, b,
                            dir * rnn.slc + s);
            acc_data_t res = ws_diff_states_layer(0, 0, it, b, s);
            if (rnn.n_dir - 1)
                res += ws_diff_states_layer(0, 1, rnn.n_iter - 1 - it, b, s);
            dst_addr[0] = res;
        }
    });
}

#define RNN_DECL_COPY_RES_LAYER_FWD(cname) \
    template <> \
    template <typename dst_layer_dt, typename dst_iter_dt> \
    void cname::copy_res_layer(const rnn_conf_t &rnn, \
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer, \
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_, \
            const gemm_acc_t *ws_diff_states_layer_) const { \
        auto dst_layer_d = memory_desc_wrapper(pd()->dst_md(0)); \
        auto dst_iter_d = memory_desc_wrapper(pd()->dst_md(1)); \
        copy_res_layer_fwd_template(rnn, pd(), dst_layer_, dst_layer_d, \
                dst_iter_, dst_iter_d, ws_states_layer_); \
    }

RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_bf16_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_f16_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_RES_LAYER_FWD(ref_rnn_common_fwd_s8s8_t)

#define RNN_DECL_COPY_RES_LAYER_BWD(cname) \
    template <> \
    template <typename dst_layer_dt, typename dst_iter_dt> \
    void cname::copy_res_layer(const rnn_conf_t &rnn, \
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer_, \
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_, \
            const gemm_acc_t *ws_diff_states_layer_) const { \
        auto diff_src_layer_d = memory_desc_wrapper(pd()->diff_src_md(0)); \
        copy_res_layer_bwd_template(rnn, diff_src_layer_, diff_src_layer_d, \
                ws_diff_states_layer_); \
    }

RNN_DECL_COPY_RES_LAYER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_RES_LAYER_BWD(ref_rnn_common_bwd_bf16_t)
RNN_DECL_COPY_RES_LAYER_BWD(ref_rnn_common_bwd_f16_t)

template <typename src_data_t, typename dst_iter_dt, typename dst_layer_dt>
void copy_res_iter_fwd_template(const rnn_conf_t &rnn, const rnn_pd_t *pd,
        dst_iter_dt *dst_iter_, memory_desc_wrapper &dst_iter_d,
        void *dst_iter_c_, memory_desc_wrapper dst_iter_c_d,
        const dst_layer_dt *dst_layer_, memory_desc_wrapper dst_layer_d,
        const src_data_t *ws_states_iter_, const void *ws_states_iter_c_) {
    if (dst_iter_ == nullptr) return;

    const AOC<const src_data_t, 5> ws_states_iter(ws_states_iter_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_states_iter_ld);

    const float data_shift = pd->attr()->rnn_data_qparams_.shift_;
    const float data_scale = pd->attr()->rnn_data_qparams_.scale_;

    const bool dequantize = pd->with_dst_iter()
            && pd->dst_md(1)->data_type == data_type::f32 && rnn.is_int8_conf();
    const auto copy_vec = [&](dst_iter_dt *dd, const src_data_t *ss) {
        if (dequantize) {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dic; s++)
                dd[s] = (dst_iter_dt)(((float)ss[s] - data_shift) / data_scale);
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t s = 0; s < rnn.dic; s++)
                dd[s] = (dst_iter_dt)ss[s];
        }
    };

    // If skip_dst_layer_copy, then the data to copy for the last
    // layer is in dst_layer, not in workspace.
    const auto n_layer_in_ws = rnn.n_layer - rnn.skip_dst_layer_copy();

    parallel_nd(n_layer_in_ws, rnn.n_dir, rnn.mb,
            [&](dim_t lay, dim_t dir, dim_t b) {
        const auto *ss = &ws_states_iter(lay + 1, dir, rnn.n_iter, b, 0);
        auto *dd = dst_iter_ + dst_iter_d.blk_off(lay, dir, b, 0);
        copy_vec(dd, ss);
    });

    if (rnn.skip_dst_layer_copy()) {
        parallel_nd(rnn.n_dir, rnn.mb, [&](dim_t dir, dim_t b) {
            const auto *ss
                    = &dst_layer_[dst_layer_d.blk_off(rnn.n_iter - 1, b, dir)];
            auto *dd = &dst_iter_[dst_iter_d.blk_off(
                    rnn.n_layer - 1, dir, b, 0)];
            copy_vec(dd, (src_data_t *)ss);
        });
    }
}

template <typename acc_data_t>
void copy_res_iter_bwd_template(const rnn_conf_t &rnn, const rnn_pd_t *pd,
        acc_data_t *diff_src_iter_, memory_desc_wrapper &diff_src_iter_d,
        float *diff_src_iter_c_, memory_desc_wrapper &diff_src_iter_c_d,
        const acc_data_t *ws_diff_states_iter_,
        const acc_data_t *ws_diff_states_iter_c_) {
    const AOC<const acc_data_t, 5> ws_diff_states_iter(ws_diff_states_iter_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_iter_ld);
    const AOC<const acc_data_t, 5> ws_diff_states_iter_c(ws_diff_states_iter_c_,
            rnn.n_layer + 1, rnn.n_dir, rnn.n_iter + 1, rnn.mb,
            rnn.ws_diff_states_iter_c_ld);
    if (diff_src_iter_) {
        parallel_nd(rnn.n_layer, rnn.n_dir, rnn.mb,
                [&](dim_t lay, dim_t dir, dim_t b) {
            for (int s = 0; s < rnn.sic; s++) {
                diff_src_iter_[diff_src_iter_d.blk_off(lay, dir, b, s)]
                        = ws_diff_states_iter(lay, dir, 0, b, s);
            }
            if (pd->cell_kind() == alg_kind::vanilla_lstm)
                for (int s = 0; s < rnn.dhc; s++) {
                    diff_src_iter_c_[diff_src_iter_c_d.blk_off(lay, dir, b, s)]
                            = ws_diff_states_iter_c(lay, dir, 0, b, s);
                }
        });
    }
}

#define RNN_DECL_COPY_RES_ITER_FWD(cname) \
    template <> \
    template <typename dst_iter_dt, typename dst_layer_dt> \
    void cname::copy_res_iter(const rnn_conf_t &rnn, dst_iter_dt *dst_iter_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_iter_, \
            float *diff_src_iter_c_, const dst_layer_dt *dst_layer_, \
            const src_layer_t *ws_states_layer_, \
            const void *ws_states_iter_c_, \
            const gemm_acc_t *ws_diff_states_iter_, \
            const gemm_acc_t *ws_diff_states_iter_c_) const { \
        auto dst_layer_d = memory_desc_wrapper(pd()->dst_md(0)); \
        auto dst_iter_d = memory_desc_wrapper(pd()->dst_md(1)); \
        auto dst_iter_c_d = memory_desc_wrapper(pd()->dst_md(2)); \
        copy_res_iter_fwd_template(rnn, pd(), dst_iter_, dst_iter_d, \
                dst_iter_c_, dst_iter_c_d, dst_layer_, dst_layer_d, \
                ws_states_layer_, ws_states_iter_c_); \
    }

RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_f32_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_bf16_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_f16_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_u8s8_t)
RNN_DECL_COPY_RES_ITER_FWD(ref_rnn_common_fwd_s8s8_t)

#define RNN_DECL_COPY_RES_ITER_BWD(cname) \
    template <> \
    template <typename output_data_t, typename dst_data_t> \
    void cname::copy_res_iter(const rnn_conf_t &rnn, output_data_t *dst_iter_, \
            void *dst_iter_c_, gemm_acc_t *diff_src_iter_, \
            float *diff_src_iter_c_, const dst_data_t *dst_layer_, \
            const src_layer_t *ws_states_layer_, \
            const void *ws_states_iter_c_, \
            const gemm_acc_t *ws_diff_states_iter_, \
            const gemm_acc_t *ws_diff_states_iter_c_) const { \
        auto diff_src_iter_d = memory_desc_wrapper(pd()->diff_src_md(1)); \
        auto diff_src_iter_c_d = memory_desc_wrapper(pd()->diff_src_md(2)); \
        copy_res_iter_bwd_template(rnn, pd(), diff_src_iter_, diff_src_iter_d, \
                diff_src_iter_c_, diff_src_iter_c_d, ws_diff_states_iter_, \
                ws_diff_states_iter_c_); \
    }

RNN_DECL_COPY_RES_ITER_BWD(ref_rnn_common_bwd_f32_t)
RNN_DECL_COPY_RES_ITER_BWD(ref_rnn_common_bwd_bf16_t)
RNN_DECL_COPY_RES_ITER_BWD(ref_rnn_common_bwd_f16_t)

rnn_bias_prepare_sig_templ(copy_bias_to_scratch) {
    const AOC<T, 3> scratch_bias(
            scratch_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dhc);

    parallel_nd(static_cast<dim_t>(rnn.n_layer) * rnn.n_dir, [&](dim_t i) {
        const dim_t off = i * rnn.n_bias * rnn.dhc;
        PRAGMA_OMP_SIMD()
        for (dim_t j = 0; j < rnn.n_bias * rnn.dhc; j++)
            scratch_bias_[off + j] = b_[off + j];
    });
}

rnn_bias_prepare_sig_templ(copy_bias_to_ws) {
    /* Original set of bias provided by the user */
    const AOC<const T, 5> b(b_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dhc);
    /* Array of pointers initialized in packing */
    const AOC<T *, 3> bias(bias_, rnn.n_layer, rnn.n_dir, rnn.n_parts_bias);
    const AOC<T, 3> scratch_bias(
            scratch_bias_, rnn.n_layer, rnn.n_dir, rnn.n_bias * rnn.dhc);

    for (int i = 0; i < rnn.n_layer; i++) {
        for (int d = 0; d < rnn.n_dir; d++) {
            int offset_bias = 0;
            for (int p = 0; p < rnn.n_parts_bias; p++) {
                bias(i, d, p) = rnn.copy_bias
                        ? const_cast<T *>(&scratch_bias(i, d, offset_bias))
                        : const_cast<T *>(&b(i, d, offset_bias));
                offset_bias += static_cast<int>(rnn.parts_bias[p] * rnn.dhc);
            }
        }
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_bias_prepare_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::bias_prepare)) {

    if (rnn.copy_bias) {
        if (rnn.bias_dt == data_type::f32)
            copy_bias_to_scratch(rnn, reinterpret_cast<float **>(bias_),
                    static_cast<const float *>(b_),
                    static_cast<float *>(scratch_bias_));
        else if (rnn.bias_dt == data_type::bf16)
            copy_bias_to_scratch(rnn, reinterpret_cast<bfloat16_t **>(bias_),
                    static_cast<const bfloat16_t *>(b_),
                    static_cast<bfloat16_t *>(scratch_bias_));
        else if (rnn.bias_dt == data_type::f16)
            copy_bias_to_scratch(rnn, reinterpret_cast<float16_t **>(bias_),
                    static_cast<const float16_t *>(b_),
                    static_cast<float16_t *>(scratch_bias_));
        else
            assert(!"Unsupported bias data type");
    }

    if (rnn.bias_dt == data_type::f32)
        copy_bias_to_ws(rnn, reinterpret_cast<float **>(bias_),
                static_cast<const float *>(b_),
                static_cast<float *>(scratch_bias_));
    else if (rnn.bias_dt == data_type::bf16)
        copy_bias_to_ws(rnn, reinterpret_cast<bfloat16_t **>(bias_),
                static_cast<const bfloat16_t *>(b_),
                static_cast<bfloat16_t *>(scratch_bias_));
    else if (rnn.bias_dt == data_type::f16)
        copy_bias_to_ws(rnn, reinterpret_cast<float16_t **>(bias_),
                static_cast<const float16_t *>(b_),
                static_cast<float16_t *>(scratch_bias_));
    else
        assert(!"Unsupported bias data type");
}

static void apply_bias_compensation(const rnn_utils::rnn_conf_t &rnn,
        float *scratch_bias_, const float *w_iter_comp,
        const float *w_layer_comp, const float data_shift,
        const float data_scale, const float *const weights_scales,
        const bool scale_per_oc) {

    for (int i = 0; i < rnn.n_layer * rnn.n_dir; i++)
        for (int j = 0; j < rnn.n_bias * rnn.dhc; j++) {
            const size_t off = i * rnn.n_bias * rnn.dhc + j;
            const float weights_scale
                    = scale_per_oc ? weights_scales[j] : weights_scales[0];
            scratch_bias_[off] -= (w_iter_comp[off] + w_layer_comp[off])
                    * data_shift / (weights_scale * data_scale);
        }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_bias_finalize_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::bias_finalize)) {
    if (rnn.is_unsigned_int8_conf()) {
        const float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
        const float data_scale = pd()->attr()->rnn_data_qparams_.scale_;
        const float *const weights_scales
                = pd()->attr()->rnn_weights_qparams_.scales_;
        const bool scale_per_oc = pd()->attr()->rnn_weights_qparams_.mask_ != 0;

        apply_bias_compensation(rnn, static_cast<float *>(scratch_bias_),
                w_iter_comp, w_layer_comp, data_shift, data_scale,
                weights_scales, scale_per_oc);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_weights_assign_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::assign_packed_weights)) {
    assert(md->format_kind == format_kind::rnn_packed);
    const auto packed_desc = md->format_desc.rnn_packed_desc;
    const AOC<weights_t *, 3> weights(
            weights_, rnn.n_layer, rnn.n_dir, packed_desc.n_parts);

    size_t offset_packed = 0;
    for (int l = 0; l < rnn.n_layer; l++)
        for (int d = 0; d < rnn.n_dir; d++) {
            for (int p = 0; p < packed_desc.n_parts; p++) {
                weights(l, d, p) = (weights_t *)&w_[offset_packed];
                offset_packed
                        += packed_desc.part_pack_size[p] / sizeof(weights_t);
            }
        }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
rnn_weights_assign_sig((ref_rnn_common_t<aprop, src_type, weights_type,
        acc_type>::assign_weights)) {
    assert(md->format_kind == format_kind::blocked);
    const auto &blk = md->format_desc.blocking;
    /* Original set of weights provided by the user */
    const AOC<const weights_t, 3> w(
            w_, rnn.n_layer, rnn.n_dir, (int)blk.strides[1]);
    /* Array of pointers for each part of weights */
    const AOC<weights_t *, 3> weights(
            weights_, rnn.n_layer, rnn.n_dir, n_parts);

    for (int i = 0; i < rnn.n_layer; i++)
        for (int d = 0; d < rnn.n_dir; d++) {
            size_t offset_weights = 0;
            for (int p = 0; p < n_parts; p++) {
                weights(i, d, p) = (weights_t *)&w(i, d, offset_weights);
                offset_weights += gates_per_part[p] * blk.strides[3];
            }
        }
}

//********************* Execution function *********************//
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type,
        data_type_t acc_type>
status_t ref_rnn_common_t<aprop, src_type, weights_type, acc_type>::execute(
        const exec_ctx_t &ctx) const {

    // Note: this is unconventional early exit in execute() for the library.
    // This RNN implementation is particularly hard for enabling asynchronous
    // threadpool runtime due to:
    // * Excessive unverified calls to Autogen GeMM which is disabled for
    //   async runtime.
    // * Using stack objects to submit parallel tasks from itself which doesn't
    //   make tasks to capture the internals of this object leading to
    //   stack-use-after-free errors.
    //   (e.g. src/cpu/x64/rnn/brgemm_cell_common_fwd.cpp::
    //         brgemm_dst_layer_iter_t::execute() (L110-120).
    //
    // The alternative approach is to introduce an API to register a threadpool
    // object before any primitive creation but this is considered as a high
    // obligation put on users to workaorund the implementation complexity.
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::threadpool_interop::threadpool_iface *tp;
    auto status = ctx.stream()->get_threadpool(&tp);
    bool ok = status == status::success && tp
            && !(tp->get_flags()
                    & dnnl::threadpool_interop::threadpool_iface::ASYNCHRONOUS);
    VCONDCHECK(primitive, create, dispatch, rnn, ok, status::unimplemented,
            "%s," VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME,
            pd()->info(ctx.stream()->engine()));
#endif

    const rnn_conf_t &rnn = this->pd()->rnn_;
    auto src_layer = CTX_IN_MEM(const src_layer_t *, DNNL_ARG_SRC_LAYER);
    auto augru_attention
            = CTX_IN_MEM(const src_layer_t *, DNNL_ARG_AUGRU_ATTENTION);
    auto src_iter = CTX_IN_MEM(const char *, DNNL_ARG_SRC_ITER);
    auto src_iter_c = CTX_IN_MEM(const void *, DNNL_ARG_SRC_ITER_C);
    auto layer_weights_n_comp
            = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS_LAYER);
    auto iter_weights_n_comp = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS_ITER);
    auto weights_peephole
            = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS_PEEPHOLE);
    auto projection_weights_n_comp
            = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS_PROJECTION);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);

    auto dst_layer = rnn.is_fwd
            ? CTX_OUT_MEM(char *, DNNL_ARG_DST_LAYER)
            : const_cast<char *>(CTX_IN_MEM(const char *, DNNL_ARG_DST_LAYER));
    auto dst_iter = rnn.is_fwd
            ? CTX_OUT_MEM(char *, DNNL_ARG_DST_ITER)
            : const_cast<char *>(CTX_IN_MEM(const char *, DNNL_ARG_DST_ITER));
    auto dst_iter_c = CTX_OUT_MEM(void *, DNNL_ARG_DST_ITER_C);

    auto diff_dst_layer
            = CTX_IN_MEM(const gemm_acc_t *, DNNL_ARG_DIFF_DST_LAYER);
    auto diff_dst_iter = CTX_IN_MEM(const gemm_acc_t *, DNNL_ARG_DIFF_DST_ITER);
    auto diff_dst_iter_c = CTX_IN_MEM(const float *, DNNL_ARG_DIFF_DST_ITER_C);

    auto w_layer = reinterpret_cast<const weights_t *>(layer_weights_n_comp);
    auto w_iter = reinterpret_cast<const weights_t *>(iter_weights_n_comp);
    auto w_projection
            = reinterpret_cast<const weights_t *>(projection_weights_n_comp);
    auto w_layer_comp = reinterpret_cast<const float *>(
            layer_weights_n_comp + rnn.weights_layer_comp_offset);
    auto w_iter_comp = reinterpret_cast<const float *>(
            iter_weights_n_comp + rnn.weights_iter_comp_offset);
    auto w_projection_comp = reinterpret_cast<const float *>(
            projection_weights_n_comp + rnn.weights_projection_comp_offset);
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    auto ptr_wei_layer
            = scratchpad.template get<weights_t *>(key_rnn_ptrs_wei_layer);
    auto ptr_wei_iter
            = scratchpad.template get<weights_t *>(key_rnn_ptrs_wei_iter);
    auto ptr_wei_projection
            = scratchpad.template get<weights_t *>(key_rnn_ptrs_wei_projection);
    auto ptr_bias = scratchpad.template get<void *>(key_rnn_ptrs_bia);
#if DNNL_X64
    const auto scratch_gates_blocked
            = scratchpad.template get<scratch_t>(key_rnn_gates_blocked);
    const auto scratch_src_layer
            = scratchpad.template get<scratch_t>(key_rnn_src_layer_trans);
    const auto scratch_src_iter
            = scratchpad.template get<scratch_t>(key_rnn_src_iter_trans);
#endif

    gemm_acc_t *amx_scratchpad = nullptr;
#if DNNL_X64
    x64::brgemm_batch_element_t *addr_batch_global = nullptr;
    if (rnn.is_brgemm && rnn.is_cell_amx()) {
        amx_scratchpad = scratchpad.template get<gemm_acc_t>(
                key_brgemm_primitive_buffer);
    }
    addr_batch_global = scratchpad.template get<x64::brgemm_batch_element_t>(
            key_brgemm_primitive_batch);
#endif
    // Fetching buffers from the workspace
    // if no workspace was provided we use the scratchpad
    char *scratch_ptr = scratchpad.template get<char>(key_rnn_space);
    char *ws_ptr = nullptr;
    if (rnn.use_workspace)
        ws_ptr = rnn.is_fwd ? CTX_OUT_MEM(char *, DNNL_ARG_WORKSPACE)
                            : const_cast<char *>(CTX_IN_MEM(
                                      const char *, DNNL_ARG_WORKSPACE));

    char *base_ptr = rnn.use_workspace ? ws_ptr : scratch_ptr;
    // ws_gates is only used to pass data from FWD to BWD.
    // assumption: in training, src_data_t and weights_t match
    gates_t *ws_gates = (gates_t *)(base_ptr + ws_gates_offset_);
    dst_iter_t *ws_ht = (dst_iter_t *)(base_ptr + ws_ht_offset_);
    src_layer_t *ws_states_layer
            = (src_layer_t *)(base_ptr + ws_states_layer_offset_);
    src_iter_t *ws_states_iter
            = (src_iter_t *)(base_ptr + ws_states_iter_offset_);
    void *ws_states_iter_c = (void *)(base_ptr + ws_states_iter_c_offset_);
    gemm_acc_t *ws_diff_states_layer
            = (gemm_acc_t *)(base_ptr + ws_diff_states_layer_offset_);
    gemm_acc_t *ws_diff_states_iter
            = (gemm_acc_t *)(base_ptr + ws_diff_states_iter_offset_);
    gemm_acc_t *ws_diff_states_iter_c
            = (gemm_acc_t *)(base_ptr + ws_diff_states_iter_c_offset_);
    gates_t *ws_grid = (gates_t *)(base_ptr + ws_grid_comp_offset_);

    auto diff_src_layer = CTX_OUT_MEM(gemm_acc_t *, DNNL_ARG_DIFF_SRC_LAYER);
    auto diff_src_iter = CTX_OUT_MEM(gemm_acc_t *, DNNL_ARG_DIFF_SRC_ITER);
    auto diff_src_iter_c = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_SRC_ITER_C);

    auto diff_augru_attention
            = CTX_OUT_MEM(gemm_acc_t *, DNNL_ARG_DIFF_AUGRU_ATTENTION);
    auto diff_weights_layer
            = CTX_OUT_MEM(gemm_acc_t *, DNNL_ARG_DIFF_WEIGHTS_LAYER);
    auto diff_weights_iter
            = CTX_OUT_MEM(gemm_acc_t *, DNNL_ARG_DIFF_WEIGHTS_ITER);
    auto diff_weights_projection
            = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_WEIGHTS_PROJECTION);
    auto diff_weights_peephole
            = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
    auto diff_bias = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);

    // Fetching extra buffers from scratchpad
    void *ws_bias = static_cast<void *>(scratch_ptr + ws_bias_offset_);
    /* Pack(if using packed gemm API) or copy(if input arrays have bad leading
     * dimension */
    (this->*bias_preparation_func)(rnn, ptr_bias, bias, ws_bias);

    // Here we use scratch_gates for the output of GEMMs on FWD and on input of GEMMs for BWD.
    // None of the values are kept for bwd
    auto scratch_gates = rnn.scratch_gates_size
            ? (scratch_t *)(scratch_ptr + scratch_gates_offset_)
            : nullptr;
    auto scratch_ht = rnn.scratch_ht_size
            ? (ht_t *)(scratch_ptr + scratch_ht_offset_)
            : nullptr;
    auto scratch_diff_ht = rnn.scratch_diff_ht_size
            ? (gemm_acc_t *)(scratch_ptr + scratch_diff_ht_offset_)
            : nullptr;
    auto scratch_cell = rnn.scratch_cell_size
            ? (scratch_t *)(scratch_ptr + scratch_cell_offset_)
            : nullptr;

    const memory_desc_t *weights_layer_md = pd()->weights_md(0);
    const memory_desc_t *weights_iter_md = pd()->weights_md(1);

    memory_desc_t wei_layer_desc;
    memory_desc_t wei_iter_desc;

#if DNNL_X64
    if (rnn.is_bf32()) {
        const auto tag = rnn.n_block == 64 ? format_tag::ldgOI64o2i
                                           : format_tag::ldgOI32o2i;
        CHECK(memory_desc_init_by_tag(wei_layer_desc, weights_layer_md->ndims,
                weights_layer_md->dims, data_type::bf16, tag));
        CHECK(memory_desc_init_by_tag(wei_iter_desc, weights_iter_md->ndims,
                weights_iter_md->dims, data_type::bf16, tag));

        if (rnn.is_augru) {
            const auto bf32_augru_attention
                    = scratchpad.template get<src_layer_t>(
                            key_rnn_bf32_attention_trans);
            cvt_float_to_bfloat16((bfloat16_t *)bf32_augru_attention,
                    (float *)augru_attention, rnn.n_iter * rnn.mb);
            augru_attention = bf32_augru_attention;
        }
        engine_t *engine = ctx.stream()->engine();
        auto wei_layer_mem
                = scratchpad.get_memory_storage(key_rnn_bf32_wei_layer_trans);
        auto wei_iter_mem
                = scratchpad.get_memory_storage(key_rnn_bf32_wei_iter_trans);
        {
            std::unique_ptr<memory_t, memory_deleter_t> reorder_dst;
            CHECK(safe_ptr_assign(reorder_dst,
                    new memory_t(engine, &wei_layer_desc,
                            std::move(wei_layer_mem))));
            exec_args_t reorder_args;
            reorder_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_WEIGHTS_LAYER);
            reorder_args[DNNL_ARG_DST] = {reorder_dst.get(), false};
            exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));
            auto *nested_grantor = create_nested_grantor(
                    ctx.get_scratchpad_grantor(), key_nested_multiple,
                    bf32_wei_layer_reorder_->pd()->scratchpad_registry());
            reorder_ctx.set_scratchpad_grantor(nested_grantor);
            CHECK(bf32_wei_layer_reorder_->execute(reorder_ctx));
            w_layer = scratchpad.template get<weights_t>(
                    key_rnn_bf32_wei_layer_trans);
            weights_layer_md = &wei_layer_desc;
        }

        {
            std::unique_ptr<memory_t, memory_deleter_t> reorder_dst;
            CHECK(safe_ptr_assign(reorder_dst,
                    new memory_t(
                            engine, &wei_iter_desc, std::move(wei_iter_mem))));
            exec_args_t reorder_args;
            reorder_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_WEIGHTS_ITER);
            reorder_args[DNNL_ARG_DST] = {reorder_dst.get(), false};
            exec_ctx_t reorder_ctx(ctx, std::move(reorder_args));
            auto *nested_grantor = create_nested_grantor(
                    ctx.get_scratchpad_grantor(), key_nested_multiple,
                    bf32_wei_iter_reorder_->pd()->scratchpad_registry());
            reorder_ctx.set_scratchpad_grantor(nested_grantor);
            CHECK(bf32_wei_iter_reorder_->execute(reorder_ctx));
            w_iter = scratchpad.template get<weights_t>(
                    key_rnn_bf32_wei_iter_trans);
            weights_iter_md = &wei_iter_desc;
        }
    }
#endif

    (this->*weights_iter_assign_func)(rnn, weights_iter_md,
            rnn.n_parts_weights_iter, rnn.parts_weights_iter, ptr_wei_iter,
            w_iter);
    (this->*weights_layer_assign_func)(rnn, weights_layer_md,
            rnn.n_parts_weights_layer, rnn.parts_weights_layer, ptr_wei_layer,
            w_layer);

    if (rnn.is_lstm_projection) {
        (this->*weights_projection_assign_func)(rnn,
                pd()->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                rnn.n_parts_weights_projection, rnn.parts_weights_projection,
                ptr_wei_projection, w_projection);
    }

    (this->*bias_finalization_func)(rnn, ws_bias, w_iter_comp, w_layer_comp);

    // we first need to copy the initial states and input into ws
    if (!(rnn.skip_src_layer_copy() && rnn.is_fwd)) {
        if (pd()->src_md(0)->data_type == data_type::f32)
            copy_init_layer(rnn, ws_states_layer, ws_diff_states_layer,
                    (const float *)src_layer, diff_dst_layer);
        else
            copy_init_layer(rnn, ws_states_layer, ws_diff_states_layer,
                    src_layer, diff_dst_layer);
    }

    if (!(rnn.skip_src_iter_copy() && rnn.is_fwd)) {
        if (pd()->src_md(1)->data_type == data_type::f32)
            copy_init_iter(rnn, ws_states_iter,
                    static_cast<void *>(ws_states_iter_c), ws_diff_states_iter,
                    ws_diff_states_iter_c, (const float *)src_iter, src_iter_c,
                    diff_dst_iter, diff_dst_iter_c);
        else
            copy_init_iter(rnn, ws_states_iter, ws_states_iter_c,
                    ws_diff_states_iter, ws_diff_states_iter_c,
                    (const src_iter_t *)src_iter, src_iter_c, diff_dst_iter,
                    diff_dst_iter_c);
    }

    // run the execution on the grid
#if DNNL_X64
    CHECK((this->*grid_computation)(ctx, rnn, ptr_wei_layer, ptr_wei_iter,
            ptr_wei_projection, weights_peephole, w_projection_comp, ptr_bias,
            src_layer, augru_attention, (const src_iter_t *)src_iter,
            src_iter_c, (dst_layer_t *)dst_layer, (dst_iter_t *)dst_iter,
            dst_iter_c, ws_states_layer, ws_states_iter, ws_states_iter_c,
            ws_diff_states_layer, ws_diff_states_iter, ws_diff_states_iter_c,
            ws_gates, ws_ht, ws_grid, scratch_gates, scratch_ht,
            scratch_diff_ht, scratch_cell, scratch_gates_blocked,
            scratch_src_layer, scratch_src_iter, diff_augru_attention,
            diff_weights_layer, diff_weights_iter, diff_weights_projection,
            diff_weights_peephole, diff_bias, amx_scratchpad,
            addr_batch_global));
#else
    CHECK((this->*grid_computation)(ctx, rnn, ptr_wei_layer, ptr_wei_iter,
            ptr_wei_projection, weights_peephole, w_projection_comp, ptr_bias,
            src_layer, augru_attention, (const src_iter_t *)src_iter,
            src_iter_c, (dst_layer_t *)dst_layer, (dst_iter_t *)dst_iter,
            dst_iter_c, ws_states_layer, ws_states_iter, ws_states_iter_c,
            ws_diff_states_layer, ws_diff_states_iter, ws_diff_states_iter_c,
            ws_gates, ws_ht, ws_grid, scratch_gates, scratch_ht,
            scratch_diff_ht, scratch_cell, diff_augru_attention,
            diff_weights_layer, diff_weights_iter, diff_weights_projection,
            diff_weights_peephole, diff_bias, amx_scratchpad));
#endif

    // Finally we copy the results to the result buffers
    if (!(rnn.skip_dst_layer_copy() && rnn.is_fwd)) {
        if (pd()->dst_md(0)->data_type == data_type::f32)
            copy_res_layer(rnn, (float *)dst_layer, diff_src_layer, dst_iter,
                    ws_states_layer, ws_diff_states_layer);
        else
            copy_res_layer(rnn, (dst_layer_t *)dst_layer, diff_src_layer,
                    dst_iter, ws_states_layer, ws_diff_states_layer);
    }

    if (!(rnn.skip_dst_iter_copy() && rnn.is_fwd)) {
        if (pd()->dst_md(1)->data_type == data_type::f32)
            copy_res_iter(rnn, (float *)dst_iter, dst_iter_c, diff_src_iter,
                    diff_src_iter_c, dst_layer, ws_states_iter,
                    ws_states_iter_c, ws_diff_states_iter,
                    ws_diff_states_iter_c);
        else
            copy_res_iter(rnn, (dst_iter_t *)dst_iter, dst_iter_c,
                    diff_src_iter, diff_src_iter_c, dst_layer, ws_states_iter,
                    ws_states_iter_c, ws_diff_states_iter,
                    ws_diff_states_iter_c);
    }

    return status::success;
}
/* Fix for MSVS warning C4661 */
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f32_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f32_t::merged_layer_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_bwd_f32_t::merged_layer_execution_ref);

template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_bf16_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_bf16_t::merged_layer_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_bwd_bf16_t::merged_layer_execution_ref);

template <>
rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f16_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_f16_t::merged_layer_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_bwd_f16_t::merged_layer_execution_ref);

template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_u8s8_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_u8s8_t::merged_layer_brgemm);

template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_ref);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_gru);
template <>
rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_gru_lbr);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_s8s8_t::merged_layer_execution_ref);
template <>
rnn_merged_layer_execution_sig(ref_rnn_fwd_s8s8_t::merged_layer_brgemm);

template struct ref_rnn_common_t<prop_kind::forward, data_type::f32,
        data_type::f32, data_type::f32>;
template struct ref_rnn_common_t<prop_kind::backward, data_type::f32,
        data_type::f32, data_type::f32>;

template struct ref_rnn_common_t<prop_kind::forward, data_type::bf16,
        data_type::bf16, data_type::f32>;
template struct ref_rnn_common_t<prop_kind::backward, data_type::bf16,
        data_type::bf16, data_type::f32>;

template struct ref_rnn_common_t<prop_kind::forward, data_type::f16,
        data_type::f16, data_type::f32>;
template struct ref_rnn_common_t<prop_kind::backward, data_type::f16,
        data_type::f16, data_type::f32>;

template struct ref_rnn_common_t<prop_kind::forward, data_type::u8,
        data_type::s8, data_type::s32>;
template struct ref_rnn_common_t<prop_kind::forward, data_type::s8,
        data_type::s8, data_type::s32>;

#undef AOC

} // namespace cpu
} // namespace impl
} // namespace dnnl
