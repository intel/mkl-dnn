/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>

#include "oneapi/dnnl/dnnl.h"

#include "utils/dnnl_query.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "rnn/rnn.hpp"
#include "rnn/rnn_aux.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "tests/test_thread_decl.hpp"
#endif

#if defined(DNNL_AARCH64_USE_ACL)
#include "tests/test_isa_common.hpp"
#endif

// Using hidden attr API for testing RNN
dnnl_status_t dnnl_primitive_attr_set_rnn_tparams(dnnl_primitive_attr_t attr,
        bool mode, dnnl_dim_t ngates, const float *scales, float cscale);

namespace {

// In order to have consistent filling across compilers and operating systems,
// we implement the equivalent of std::normal_distribution using the so-called
// Marsaglia polar method.
template <typename T>
class normal_distribution_t {
public:
    normal_distribution_t(T mean, T stddev)
        : gen(-1.f, 1.f)
        , is_odd_(false)
        , odd_(1.f)
        , mean_(mean)
        , stddev_(stddev) {
        static_assert(std::is_floating_point<T>::value,
                "T must be a floating point type.");
    }
    template <typename URNG>
    T operator()(URNG &g) {
        T r, r2, x, y;
        if (is_odd_) {
            is_odd_ = false;
            return odd_;
        }
        is_odd_ = true;
        do {
            x = gen(g); // x E [-1, 1)
            y = gen(g); // y E [-1, 1)
            r2 = x * x + y * y;
        } while (0.f == r2 || 1.f < r2); // r2 E (0, 1]
        r = stddev_ * std::sqrt(-2.f * std::log(r2) / r2);
        x = mean_ + x * r;
        y = mean_ + y * r;
        odd_ = x;
        return y;
    }

private:
    std::uniform_real_distribution<T> gen;
    bool is_odd_;
    T odd_;
    const T mean_;
    const T stddev_;
};

} // namespace

namespace rnn {

dnnl_primitive_attr_t create_dnnl_rnn_attr(const prb_t &prb) {
    dnnl_primitive_attr_t dnnl_attr = nullptr;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    if (prb.skip_nonlinear)
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_tparams(dnnl_attr, true,
                prb.n_gates(), prb.linear_scales, prb.linear_cscale));

    DNN_SAFE_V(dnnl_primitive_attr_set_rnn_weights_qparams(
            dnnl_attr, prb.wei_nscales, prb.wei_scales_mask, prb.wei_scales));

    if (prb.is_lstm_projection() && prb.is_int8())
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_weights_projection_qparams(
                dnnl_attr, prb.wei_proj_nscales, prb.wei_proj_scales_mask,
                prb.wei_proj_scales));

    if (prb.data_scale != 1.0 || prb.data_shift != 0.0)
        DNN_SAFE_V(dnnl_primitive_attr_set_rnn_data_qparams(
                dnnl_attr, prb.data_scale, prb.data_shift));

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, prb.attr.scratchpad_mode));

    DNN_SAFE_V(dnnl_primitive_attr_set_fpmath_mode_v2(dnnl_attr,
            prb.attr.fpmath_mode.mode, prb.attr.fpmath_mode.apply_to_int));

    DNN_SAFE_V(dnnl_primitive_attr_set_deterministic(
            dnnl_attr, prb.attr.deterministic.enabled));

    return dnnl_attr;
}

int check_s8s8_reorder(const prb_t &prb, rnn_data_kind_t kind,
        const dnn_mem_t &mem_dt, const dnn_mem_t &mem_fp, res_t *res) {
    // TODO: enable for all cpu_kind when supported
    if (is_gpu()) return OK;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_DPCPP
    // DPC++ does not provide a simple way to access the underlying
    // buffer alignment.
    return OK;
#endif

    // In the main test, we fill buffers with f32 and reorder to s8
    // with quantization.

    // The endgoal is to check that the reorder
    // f32_plain_nonquantized --reorder--> s8_packed_quantized
    // gives the same output as the sequence
    // f32_plain --quant--> s8_plain_quantized --reorder--> s8_packed_quantized

    // Here,
    // 1. we quantize the f32 plain memory to s8 plain memory,
    // 2. we reorder the s8 plain to s8 packed (queried from rnn primitive desc)
    // 3. we check that the two memory are bitwise identical.

    // Note: the two s8 packed memories need to have the same
    // alignment as packed buffer is aligned internally and the offset
    // is kept in the metadata.
    // Works fine with dnn_mem_t as it is align to 2MB large page boundary
    dnn_mem_t mem_s8_src(mem_fp.md_, dnnl_s8, tag::abx, get_cpu_engine(),
            /* prefill = */ true);
    dnn_mem_t mem_s8_dst(mem_dt.md_, get_test_engine(), /* prefill = */ true);

    /* 1. compute f32_plain --quant--> s8_plain_quantized */
    /* Do fixed partitioning to have same filling for any number of threads */
    auto nelems = mem_fp.nelems();
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);
    const auto quantize = [&](const float *scales, int nscales, float shift,
                                  int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            const float current_scale = scales[idx % nscales];
            float val_f32 = mem_fp.get_f32_elem(idx);
            int8_t val_s8
                    = maybe_saturate(dnnl_s8, val_f32 * current_scale + shift);
            mem_s8_src.set_elem(idx, val_s8);
        }
    };
    switch (kind) {
        case WEIGHTS_LAYER:
        case WEIGHTS_ITER:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                quantize(prb.wei_scales, prb.wei_nscales, 0, idx);
            });
            break;
        case WEIGHTS_PROJECTION:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                quantize(prb.wei_proj_scales, prb.wei_proj_nscales, 0, idx);
            });
            break;
        case SRC_LAYER:
        case SRC_ITER:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                quantize(&(prb.data_scale), 1, prb.data_shift, idx);
            });
            break;
        default: assert(!"unsupported kind");
    }

    /* 2. compute s8_plain_quantized --reorder--> s8_packed_quantized */
    SAFE(mem_s8_dst.reorder(mem_s8_src, res), WARN);

    /* 3. we check that the two memory are bitwise identical. */
    auto sz = mem_dt.size();
    uint8_t *s8_dst_handle = (uint8_t *)mem_s8_dst;
    uint8_t *mem_dt_handle = (uint8_t *)mem_dt;

    // check that both have the same size
    assert(mem_dt.size() == mem_s8_dst.size());
    // check that both have the same alignment modulo align_data in gemm_pack_storage.hpp
    assert((uint64_t)s8_dst_handle % 0x1000
            == (uint64_t)mem_dt_handle % 0x1000);
    for (size_t i = 0; i < sz; ++i) {
        if (s8_dst_handle[i] != mem_dt_handle[i]) { return FAIL; }
    }

    return OK;
}

int fill_memory(int exec_arg, const prb_t &prb, rnn_data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, dnnl_data_type_t dt, float mean,
        float stddev, float min, float max, res_t *res,
        const_dnnl_primitive_attr_t attr = nullptr, bool flip_sign = false) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    // For non-int8 RNN the data is filled according to cfg directly.
    // However, for int8 RNN we have slightly obscure logic, at least for now:
    // 1. cfg describes the quantized data;
    // 2. We fill first f32 de-quantized data, by inverse-applying the scale
    //    and shift to the data generated by cfg distribution;
    // 3. We reorder the data for the oneDNN RNN primitive
    // 4. Q10n of the data for reference benchdnn RNN:
    //    4.a. If the tensor is weights -- q10n it here;
    //    4.b. If the tensor is data -- reference benchdnn RNN will quantize it.

    // pass rnn attributes to f32 -> int8 reorders only
    const_dnnl_primitive_attr_t reorder_attr = nullptr;
    if (prb.is_int8() && (dt != dnnl_f32)) reorder_attr = attr;
    float default_scales[1] = {1.0f};
    float default_shift = 0.0f;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    // 2. We fill first f32 de-quantized data, by inverse-applying the scale
    //    and shift to the data generated by cfg distribution;
    auto fill_chunk = [&](const float *scales, int nscales, float shift,
                              int idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand msr;
        msr.seed(idx_start + kind);
        normal_distribution_t<float> gen(mean, stddev);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float val = round_to_nearest_representable(dt, gen(msr));
            val = MAX2(MIN2(val, max), min);
            val = (val - shift)
                    / scales[idx % nscales]; // change only int8-case

            // Vanilla RNN with RELU testing related only: flip the sign of
            // inputs for `mb` == 0 to test RELU part
            if (flip_sign) {
                assert(kind == SRC_LAYER || kind == SRC_ITER);
                auto ld = kind == SRC_LAYER ? prb.slc : prb.sic;
                if (idx % (prb.mb * ld) < ld) val *= -1;
            }
            mem_fp.set_f32_elem(idx, val);
        }
    };
    switch (kind) {
        case WEIGHTS_PROJECTION:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                fill_chunk(
                        prb.wei_proj_scales, prb.wei_proj_nscales, 0.0f, idx);
            });
            break;
        case WEIGHTS_LAYER:
        case WEIGHTS_ITER:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                fill_chunk(prb.wei_scales, prb.wei_nscales, 0.0f, idx);
            });
            break;
        case SRC_LAYER:
        case SRC_ITER:
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                fill_chunk(&(prb.data_scale), 1, prb.data_shift, idx);
            });
            break;
        default: // we do no scale/shift
            benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                fill_chunk(default_scales, 1, default_shift, idx);
            });
    }

    // 3. We reorder the data for the DNNL RNN primitive
    SAFE(mem_dt.reorder(mem_fp, res, reorder_attr), WARN);
    if ((reorder_attr != nullptr) && (dt == dnnl_s8))
        if (check_s8s8_reorder(prb, kind, mem_dt, mem_fp, res) != OK)
            return FAIL;

    // Bullet 4.a holds: quantize weights for int8 benchdnn reference RNN
    if (prb.is_int8()) {
        auto quantize_chunk
                = [&](const float *scales, int nscales, int idx_chunk) {
            int64_t idx_start = idx_chunk * chunk_size;
            int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                float current_scale = scales[idx % nscales];
                float val = ((float *)mem_fp)[idx];
                val = round(current_scale * val);
                mem_fp.set_f32_elem(idx, MAX2(MIN2(val, max), min));
            }
        };
        switch (kind) {
            case WEIGHTS_LAYER:
            case WEIGHTS_ITER:
                benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                    quantize_chunk(prb.wei_scales, prb.wei_nscales, idx);
                });
                break;
            case WEIGHTS_PROJECTION:
                benchdnn_parallel_nd(n_chunks, [&](int64_t idx) {
                    quantize_chunk(
                            prb.wei_proj_scales, prb.wei_proj_nscales, idx);
                });
                break;
            default: // Nothing to do
                break;
        }
    }

    return OK;
}

int fill_memory(int exec_arg, const prb_t &prb, rnn_data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res,
        const_dnnl_primitive_attr_t attr = nullptr, bool flip_sign = false) {
    const dt_conf_t::entry_t &c = prb.cfg[kind];
    return fill_memory(exec_arg, prb, kind, mem_dt, mem_fp, c.dt, c.f_mean,
            c.f_stddev, c.f_min, c.f_max, res, attr, flip_sign);
}

int fill_activation(int exec_arg, const prb_t &prb, rnn_data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res,
        const_dnnl_primitive_attr_t attr = nullptr) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    // In general, we mostly want to use positive values to avoid
    // cancellation from happening during computation.  The only case
    // where we actually want negative values to appear is for 1 layer
    // 1 iteration tests using vanilla_rnn and non-zero alpha. In that
    // case, we want to check that alpha is applied accordingly. Here
    // skip_nonlinear is checked as we want to test relu with non-zero
    // alpha, and not the linear function that would replace it under
    // skip_nonlinear=true.
    bool flip_sign = prb.skip_nonlinear == false && prb.alg == VANILLA_RNN
            && prb.activation == RELU
            && (kind == SRC_LAYER || kind == SRC_ITER);
    return fill_memory(
            exec_arg, prb, kind, mem_dt, mem_fp, res, attr, flip_sign);
}

int fill_src_iter_c(int exec_arg, const prb_t &prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res,
        const_dnnl_primitive_attr_t attr = nullptr) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;

    const bool special_case = prb.prop == dnnl_backward && prb.skip_nonlinear;
    if (!special_case)
        return fill_memory(
                exec_arg, prb, SRC_ITER_C, mem_dt, mem_fp, res, attr);

    // The scaling factors in tparams when testing backward are common for
    // for forward and backward passes, and computed as 1 over maximum of
    // the accumulation chain:
    // - ~n_gates on FWD
    // - ~dhc * n_gates on BWD_D
    // - ~mb * n_gates on BWD_W
    //
    // This makes tparam relatively small for the forward pass (compare to
    // the forward pass when we test forward only). This in turn, makes
    // src_iter_c converge relatively fast to the value ~ i_gate * c_gate,
    // which is (typically) way smaller than the original distribution for
    // src_iter_c.
    //
    // TODO: use different tparams for forward and
    //       backward passes when testing BWD_DW.
    //
    // The problem appears on backward pass. Consider diff_f_gate that
    // contributes to backward weights when batch or number of iterations
    // is big:
    //   diff_f_gate[iter] = src_iter_c[iter] * diff_dst[iter].
    //   diff_weights += ~diff_f_gate[iter].
    //
    // Assume, that diff_dst[iter] is always about the same for every iter.
    // Since src_iter_c[0] >> src_iter_c[iter] for iter > 0, this makes the
    // diff_weight be highly dependent on the order of accumulating the
    // diff_f_gate[iter].
    //
    // Originally we had something like:
    // diff_weights = v + v * 10^-5 + ... + v * 10^-5 (n_iter * MB summands).
    // Depending on the order of summation the difference might exceed the
    // typical bound approximation: coefficient * log(number_of_summands).
    //
    // Anyways, the algorithm below tries to put the first src_iter_c[iter = 0]
    // in the same ballpark as all the subsequent src_iter_c[iter > 0].
    //
    // The estimation is based on the following rough assumptions:
    //   src_iter_c[iter+1] = f_gate * src_iter_c[iter] + i_gate * c_gate
    //                     ~= f_gate * small_value      + i_gate * c_gate
    //                     ~=                             i_gate * c_gate.
    //   i_gate ~= tparams[i_gate] * (
    //              1 / ngates * mean_src_layer +
    //              1 / ngates * mean_src_iter  +
    //              mean_bias);
    //
    // Same for c_gate.
    // The (1 / ngates) factor is taken from fill_weights().

    float expect_gemm_output = (1.f / prb.n_gates()) * prb.cfg[SRC_LAYER].f_mean
            + (1.f / prb.n_gates()) * prb.cfg[SRC_ITER].f_mean
            + prb.cfg[BIAS].f_mean;
    float expect_i_gate = (float)prb.linear_scales[LSTM_I] * expect_gemm_output;
    float expect_c_gate = (float)prb.linear_scales[LSTM_C] * expect_gemm_output;
    float expect_src_iter_c_mean = expect_i_gate * expect_c_gate;

    float adjust_factor = 1;

    const bool need_adjust = expect_src_iter_c_mean < prb.cfg[SRC_ITER_C].f_mean
            && prb.cfg[SRC_ITER_C].f_mean != 0;
    if (need_adjust)
        adjust_factor = expect_src_iter_c_mean / prb.cfg[SRC_ITER_C].f_mean;

    const dt_conf_t::entry_t &c = prb.cfg[SRC_ITER_C];
    return fill_memory(exec_arg, prb, SRC_ITER_C, mem_dt, mem_fp, c.dt,
            c.f_mean * adjust_factor, c.f_stddev * adjust_factor,
            c.f_min * adjust_factor, c.f_max * adjust_factor, res, attr);
}

int fill_weights(int exec_arg, const prb_t &prb, rnn_data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res,
        const_dnnl_primitive_attr_t attr = nullptr) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    assert(kind == WEIGHTS_PROJECTION ? mem_fp.ndims() == 4
                                      : mem_fp.ndims() == 5);

    dnn_mem_t mem_pure_fp(mem_dt.md_, dnnl_f32, tag::abx, get_cpu_engine(),
            /* prefill = */ false);

    const auto dt = prb.cfg[kind].dt;
    const auto &dims = mem_fp.dims();
    const int64_t L = dims[0];
    const int64_t D = dims[1];
    const int64_t I = dims[2];
    const int64_t G = (kind == WEIGHTS_PROJECTION) ? 1 : dims[3];
    const int64_t O = (kind == WEIGHTS_PROJECTION) ? dims[3] : dims[4];
    const float gate_factor
            = (kind == WEIGHTS_PROJECTION) ? 1.f : 1.f / prb.n_gates();
    const auto scales = (kind == WEIGHTS_PROJECTION) ? prb.wei_proj_scales
                                                     : prb.wei_scales;
    const auto n_scales = (kind == WEIGHTS_PROJECTION) ? prb.wei_proj_nscales
                                                       : prb.wei_nscales;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        mem_fp.set_f32_elem(i, 0);
        mem_pure_fp.set_f32_elem(i, 0);
    });

    // Fill weights sparsely to avoid accumulation errors. Using two memories:
    // one is quantized for reference, another is for a reorder.
    // Regular approach with sparsity doesn't apply, `i` dim must have a single
    // element in it across whole buffer.
    benchdnn_parallel_nd(
            L, D, G, O, [&](int64_t l, int64_t d, int64_t g, int64_t o) {
        int64_t i_off = ((19 * o + 7 * g + 11 * d + 13 * l) % I);
        int64_t off = (((l * D + d) * I + i_off) * G + g) * O + o;
        float val = gate_factor;
        mem_pure_fp.set_f32_elem(off, val);
        if (prb.is_int8()) val *= scales[off % n_scales];
        mem_fp.set_f32_elem(off, round_to_nearest_representable(dt, val));
    });

    // Pass rnn attributes to f32 -> s8 reorders only
    const_dnnl_primitive_attr_t reorder_attr = nullptr;
    if (prb.is_int8()) reorder_attr = attr;
    SAFE(mem_dt.reorder(mem_pure_fp, res, reorder_attr), WARN);

    // Test that s8 -> s8 reorder works correctly
    if ((reorder_attr != nullptr) && (dt == dnnl_s8))
        return check_s8s8_reorder(prb, kind, mem_dt, mem_pure_fp, res);
    return OK;
}

// To reduce likelihood of cancellation happening in bwd by bias,
// (especially for GRU), we want diff_bias to be sparse
int fill_bias(int exec_arg, const prb_t &prb, rnn_data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand norm_seed(kind * nelems + idx_start + 1);
        norm_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::normal_distribution<float> gen(
                prb.cfg[kind].f_mean, prb.cfg[kind].f_stddev);
        std::bernoulli_distribution b_dist(0.05f);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = b_dist(b_seed);
            float gen_val = gen(norm_seed);
            float val = is_one * gen_val;
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(prb.cfg[kind].dt, val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

void compute_ref(const base_prb_t *base_prb, dir_t dir, const args_t &args,
        dnnl_primitive_t prim_ref) {
    const prb_t *prb = prb_t::from(base_prb);
    const prb_t &prb_ = *prb;
    if (prop2prop_kind(dir) != dnnl_backward)
        compute_ref_fwd(prb_, args);
    else
        compute_ref_bwd(prb_, args);
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t &prb = *prb_t::from(init_pd_args.base_prb);
    const dir_t dir = init_pd_args.dir;
    res_t *res = init_pd_args.res;

    dnnl_prop_kind_t fwd_prop = dnnl_prop_kind_undef;
    switch (prb.prop) {
        case dnnl_forward_training: fwd_prop = dnnl_forward_training; break;
        case dnnl_forward_inference: fwd_prop = dnnl_forward_inference; break;
        // If we are testing backward, we have to run forward training first
        // in order to generate a valid workspace.
        case dnnl_backward: fwd_prop = dnnl_forward_training; break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    const bool is_gru_lbr = prb.alg == LBR_GRU || prb.alg == LBR_AUGRU;
    // Enable testing with non trivial strides
    const int the_stride = prb.trivial_strides ? 0 : 1;

    // bidirectional = 2, s for lstm = 2, for all other = 1
    dnnl_dims_t weights_layer_dims
            = {prb.n_layer, prb.n_dir(), prb.slc, prb.n_gates(), prb.dhc};
    dnnl_dims_t weights_iter_dims
            = {prb.n_layer, prb.n_dir(), prb.sic, prb.n_gates(), prb.dhc};
    dnnl_dims_t attention_dims = {prb.n_iter, prb.mb, 1};
    dnnl_dims_t weights_peephole_dims = {prb.n_layer, prb.n_dir(), 3, prb.dhc};
    dnnl_dims_t weights_projection_dims
            = {prb.n_layer, prb.n_dir(), prb.dhc, prb.dic};
    dnnl_dims_t bias_dims
            = {prb.n_layer, prb.n_dir(), prb.n_gates() + is_gru_lbr, prb.dhc};
    // dnnl_tnc
    dnnl_dims_t dst_layer_dims = {prb.n_iter, prb.mb, prb.dlc(PRIMITIVE)};

    dnnl_dims_t src_layer_dims = {prb.n_iter, prb.mb, prb.slc};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_layer_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_layer_d, prb.ndims(SRC_LAYER),
            src_layer_dims, prb.cfg[SRC_LAYER].dt, prb.tag[0]));
    if (prb.tag[0] != tag::any) {
        dims_t src_layer_strides(query_md_ndims(src_layer_d));
        std::memcpy(src_layer_strides.data(), query_md_strides(src_layer_d),
                src_layer_strides.size() * sizeof(dnnl_dim_t));
        int biggest_stride_idx = 0;
        int64_t biggest_stride = src_layer_strides[biggest_stride_idx];
        for (int i = 1; i < query_md_ndims(src_layer_d); i++) {
            if (src_layer_strides[i] > biggest_stride) {
                biggest_stride = src_layer_strides[i];
                biggest_stride_idx = i;
            }
        }
        // Apply the extra +1 to the biggest stride to avoid modifying all of
        // them.
        src_layer_strides[biggest_stride_idx] += the_stride;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(src_layer_d,
                query_md_ndims(src_layer_d), query_md_dims(src_layer_d),
                query_md_data_type(src_layer_d), "", src_layer_strides));
    }

    dnnl_dims_t src_iter_dims = {prb.n_layer, prb.n_dir(), prb.mb, prb.sic};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_iter_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_iter_d, prb.ndims(SRC_ITER),
            src_iter_dims, prb.cfg[SRC_ITER].dt, tag::abx /* dnnl_ldnc */));
    // Adjust strides for src_iter_d.
    dims_t src_iter_strides(query_md_ndims(src_iter_d));
    std::memcpy(src_iter_strides.data(), query_md_strides(src_iter_d),
            src_iter_strides.size() * sizeof(dnnl_dim_t));
    src_iter_strides[2] = prb.sic + the_stride;
    for (int d = 1; d >= 0; --d)
        src_iter_strides[d]
                = src_iter_strides[d + 1] * query_md_dims(src_iter_d)[d + 1];
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_iter_d, query_md_ndims(src_iter_d),
            query_md_dims(src_iter_d), query_md_data_type(src_iter_d), "",
            src_iter_strides));

    dnnl_dims_t src_iter_c_dims = {prb.n_layer, prb.n_dir(), prb.mb, prb.dhc};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_iter_c_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_iter_c_d, prb.ndims(SRC_ITER_C),
            src_iter_c_dims, prb.cfg[SRC_ITER_C].dt, tag::abx /* dnnl_ldnc */));
    // Adjust strides for src_iter_c_d.
    dims_t src_iter_c_strides(query_md_ndims(src_iter_c_d));
    std::memcpy(src_iter_c_strides.data(), query_md_strides(src_iter_c_d),
            src_iter_c_strides.size() * sizeof(dnnl_dim_t));
    src_iter_c_strides[2] = prb.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        src_iter_c_strides[d] = src_iter_c_strides[d + 1]
                * query_md_dims(src_iter_c_d)[d + 1];
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_iter_c_d,
            query_md_ndims(src_iter_c_d), query_md_dims(src_iter_c_d),
            query_md_data_type(src_iter_c_d), "", src_iter_c_strides));

    // Forward and backward support different layouts for weights. When
    // testing backward, we cannot reliably use the supplied weights tag.
    bool has_service_prim = prb.prop == dnnl_backward;
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> weights_layer_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(weights_layer_d,
            prb.ndims(WEIGHTS_LAYER), weights_layer_dims,
            prb.cfg[WEIGHTS_LAYER].dt, has_service_prim ? "any" : prb.tag[1]));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> weights_iter_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(weights_iter_d, prb.ndims(WEIGHTS_ITER),
            weights_iter_dims, prb.cfg[WEIGHTS_ITER].dt,
            has_service_prim ? "any" : prb.tag[1]));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> attention_d {};
    if (prb.is_augru())
        DNN_SAFE_STATUS(dnn_mem_t::init_md(attention_d,
                prb.ndims(AUGRU_ATTENTION), attention_dims,
                prb.cfg[AUGRU_ATTENTION].dt, tag::abx /* dnnl_tnc */));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> weights_peephole_d {};
    if (prb.is_lstm_peephole())
        DNN_SAFE_STATUS(dnn_mem_t::init_md(weights_peephole_d,
                prb.ndims(WEIGHTS_PEEPHOLE), weights_peephole_dims,
                prb.cfg[WEIGHTS_PEEPHOLE].dt, tag::abx /* dnnl_ldgo */));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> weights_projection_d {};
    if (prb.is_lstm_projection())
        DNN_SAFE_STATUS(dnn_mem_t::init_md(weights_projection_d,
                prb.ndims(WEIGHTS_PROJECTION), weights_projection_dims,
                prb.cfg[WEIGHTS_PROJECTION].dt, tag::any));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bias_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(
            bias_d, prb.ndims(BIAS), bias_dims, prb.cfg[BIAS].dt, tag::any));

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_layer_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_layer_d, prb.ndims(DST_LAYER),
            dst_layer_dims, prb.cfg[DST_LAYER].dt, prb.tag[2]));
    if (prb.tag[2] != tag::any) {
        dims_t dst_layer_strides(query_md_ndims(dst_layer_d));
        std::memcpy(dst_layer_strides.data(), query_md_strides(dst_layer_d),
                dst_layer_strides.size() * sizeof(dnnl_dim_t));
        int biggest_stride_idx = 0;
        int64_t biggest_stride = dst_layer_strides[biggest_stride_idx];
        for (int i = 1; i < query_md_ndims(dst_layer_d); i++) {
            if (dst_layer_strides[i] > biggest_stride) {
                biggest_stride = dst_layer_strides[i];
                biggest_stride_idx = i;
            }
        }
        // Apply the extra +1 to the biggest stride to avoid modifying all of
        // them.
        dst_layer_strides[biggest_stride_idx] += the_stride;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_layer_d,
                query_md_ndims(dst_layer_d), query_md_dims(dst_layer_d),
                query_md_data_type(dst_layer_d), "", dst_layer_strides));
    }

    dnnl_dims_t dst_iter_dims = {prb.n_layer, prb.n_dir(), prb.mb, prb.dic};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_iter_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_iter_d, prb.ndims(DST_ITER),
            dst_iter_dims, prb.cfg[DST_ITER].dt, tag::abx /* dnnl_ldnc */));
    // Adjust strides for dst_iter_d.
    dims_t dst_iter_strides(query_md_ndims(dst_iter_d));
    std::memcpy(dst_iter_strides.data(), query_md_strides(dst_iter_d),
            dst_iter_strides.size() * sizeof(dnnl_dim_t));
    dst_iter_strides[2] = prb.dic + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_iter_strides[d]
                = dst_iter_strides[d + 1] * query_md_dims(dst_iter_d)[d + 1];
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_iter_d, query_md_ndims(dst_iter_d),
            query_md_dims(dst_iter_d), query_md_data_type(dst_iter_d), "",
            dst_iter_strides));

    dnnl_dims_t dst_iter_c_dims = {prb.n_layer, prb.n_dir(), prb.mb, prb.dhc};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_iter_c_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_iter_c_d, prb.ndims(DST_ITER_C),
            dst_iter_c_dims, prb.cfg[DST_ITER_C].dt, tag::abx /* dnnl_ldnc */));
    // Adjust strides for dst_iter_c_d.
    dims_t dst_iter_c_strides(query_md_ndims(dst_iter_c_d));
    std::memcpy(dst_iter_c_strides.data(), query_md_strides(dst_iter_c_d),
            dst_iter_c_strides.size() * sizeof(dnnl_dim_t));
    dst_iter_c_strides[2] = prb.dhc + the_stride;
    for (int d = 1; d >= 0; --d)
        dst_iter_c_strides[d] = dst_iter_c_strides[d + 1]
                * query_md_dims(dst_iter_c_d)[d + 1];
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_iter_c_d,
            query_md_ndims(dst_iter_c_d), query_md_dims(dst_iter_c_d),
            query_md_data_type(dst_iter_c_d), "", dst_iter_c_strides));

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(create_dnnl_rnn_attr(prb));

    // Initializing the forward pass
    // When inference, we use forward_inference
    // When training, we use forward_training
    if (dir & FLAG_FWD) {
        DNN_SAFE_STATUS(init_rnn_fwd_pd(&init_pd_args.pd, init_pd_args.engine,
                prb, fwd_prop, src_layer_d, src_iter_d, src_iter_c_d,
                attention_d, weights_layer_d, weights_iter_d,
                weights_peephole_d, weights_projection_d, bias_d, dst_layer_d,
                dst_iter_d, dst_iter_c_d, dnnl_attr, res));
    } else {
        // TODO: add stride support for diff_* tensors
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_src_layer_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_src_layer_d,
                prb.ndims(DIFF_SRC_LAYER), src_layer_dims,
                prb.cfg[DIFF_SRC_LAYER].dt, prb.tag[0]));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_src_iter_d;
        DNN_SAFE_STATUS(
                dnn_mem_t::init_md(diff_src_iter_d, prb.ndims(DIFF_SRC_ITER),
                        src_iter_dims, prb.cfg[DIFF_SRC_ITER].dt, tag::any));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_src_iter_c_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_src_iter_c_d,
                prb.ndims(DIFF_SRC_ITER_C), src_iter_c_dims,
                prb.cfg[DIFF_SRC_ITER_C].dt, tag::any));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_weights_layer_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_weights_layer_d,
                prb.ndims(DIFF_WEIGHTS_LAYER), weights_layer_dims,
                prb.cfg[DIFF_WEIGHTS_LAYER].dt, prb.tag[1]));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_weights_iter_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_weights_iter_d,
                prb.ndims(DIFF_WEIGHTS_ITER), weights_iter_dims,
                prb.cfg[DIFF_WEIGHTS_ITER].dt, prb.tag[1]));

        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_attention_d {};
        if (prb.is_augru())
            DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_attention_d,
                    prb.ndims(DIFF_AUGRU_ATTENTION), attention_dims,
                    prb.cfg[DIFF_AUGRU_ATTENTION].dt, tag::abx /* dnnl_tnc */));

        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_weights_peephole_d {};
        if (prb.is_lstm_peephole())
            DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_weights_peephole_d,
                    prb.ndims(DIFF_WEIGHTS_PEEPHOLE), weights_peephole_dims,
                    prb.cfg[DIFF_WEIGHTS_PEEPHOLE].dt,
                    tag::abx /* dnnl_ldgo */));

        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>
                diff_weights_projection_d {};
        if (prb.is_lstm_projection())
            DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_weights_projection_d,
                    prb.ndims(DIFF_WEIGHTS_PROJECTION), weights_projection_dims,
                    prb.cfg[DIFF_WEIGHTS_PROJECTION].dt, tag::any));

        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_bias_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_bias_d, prb.ndims(DIFF_BIAS),
                bias_dims, prb.cfg[DIFF_BIAS].dt, tag::any));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_dst_layer_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_dst_layer_d,
                prb.ndims(DIFF_DST_LAYER), dst_layer_dims,
                prb.cfg[DIFF_DST_LAYER].dt, prb.tag[2]));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_dst_iter_d;
        DNN_SAFE_STATUS(
                dnn_mem_t::init_md(diff_dst_iter_d, prb.ndims(DIFF_DST_ITER),
                        dst_iter_dims, prb.cfg[DIFF_DST_ITER].dt, tag::any));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_dst_iter_c_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_dst_iter_c_d,
                prb.ndims(DIFF_DST_ITER_C), dst_iter_c_dims,
                prb.cfg[DIFF_DST_ITER_C].dt, tag::any));

        DNN_SAFE_STATUS(init_rnn_bwd_pd(&init_pd_args.pd, init_pd_args.engine,
                prb, prb.prop, src_layer_d, src_iter_d, src_iter_c_d,
                attention_d, weights_layer_d, weights_iter_d,
                weights_peephole_d, weights_projection_d, bias_d, dst_layer_d,
                dst_iter_d, dst_iter_c_d, diff_src_layer_d, diff_src_iter_d,
                diff_src_iter_c_d, diff_attention_d, diff_weights_layer_d,
                diff_weights_iter_d, diff_weights_peephole_d,
                diff_weights_projection_d, diff_bias_d, diff_dst_layer_d,
                diff_dst_iter_d, diff_dst_iter_c_d, init_pd_args.hint,
                dnnl_attr, res));
    }

    return dnnl_success;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb_ = this;
    const prb_t &prb = *prb_;
    dir_t dir = str2dir(prop2str(prb.prop));
    skip_unimplemented_data_type({prb.cfg[SRC_LAYER].dt}, dir, res);
    skip_unimplemented_sum_po(prb.attr, res, dnnl_rnn, prb.cfg[SRC_LAYER].dt);
    skip_unimplemented_binary_po(prb.attr, res);
    skip_unimplemented_prelu_po(prb.attr, res, dnnl_rnn);

    if (is_cpu()) {
#if !defined(DNNL_X64) || DNNL_X64 == 0 \
        || DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        // int8 is not supported altogether since RNN relies on packed IGEMM
        // FIXME: this will disable int8 RNN testing if the library is built with
        //        Intel MKL that does have packed IGEMM
        if (prb.is_int8()) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
#endif
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        // RNN is not supported with async threadpool runtime.
        auto *tp = dnnl::testing::get_threadpool();
        if (tp
                && (tp->get_flags()
                        & dnnl::threadpool_interop::threadpool_iface::
                                ASYNCHRONOUS)) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
#endif
        const auto wei_tag
                = normalize_tag(prb.tag[1], prb.ndims(WEIGHTS_LAYER));
        // cpu backward only supports `any` layout for weights.
        if (prb.prop == dnnl_backward && wei_tag != tag::any) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }

        // f16 training is not yet fully supported.
        const bool is_f16_not_ok
                = prb.cfg[SRC_LAYER].dt == dnnl_f16 && !(dir & FLAG_INF);
        if (is_f16_not_ok) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }

#if defined(DNNL_AARCH64_USE_ACL)
        const bool is_acl_f16_not_ok = prb.cfg[SRC_LAYER].dt == dnnl_f16
                && dnnl::impl::cpu::platform::has_data_type_support(dnnl_f16);
        if (is_acl_f16_not_ok) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
#endif
    }

    // int8 weights reorder does not support non trivial strides;
    // only LSTM and GRU cell kinds support int8 so far;
    if (prb.is_int8()) {
        if (!prb.trivial_strides) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.alg != VANILLA_LSTM && prb.alg != VANILLA_GRU) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.prop != dnnl_forward_inference) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }

        if (is_cpu()) {
            const auto src_tag
                    = normalize_tag(prb.tag[0], prb.ndims(SRC_LAYER));
            const auto wei_tag
                    = normalize_tag(prb.tag[1], prb.ndims(WEIGHTS_LAYER));
            const auto dst_tag
                    = normalize_tag(prb.tag[2], prb.ndims(DST_LAYER));

            const bool tags_not_ok = src_tag != "abc" || wei_tag != tag::any
                    || dst_tag != "abc";
            if (tags_not_ok) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return;
            }
        }

        if (is_gpu() && prb.tag[1] != tag::any) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
    }

    // LSTM w/ projection is not supported for bf16
    if (prb.is_lstm_projection()
            && (prb.cfg[SRC_LAYER].dt == dnnl_bf16
                    || prb.cfg[SRC_LAYER].dt == dnnl_f16)) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    // GPU limitations for RNN
    if (is_gpu()) {
        bool is_AUGRU = prb.alg == VANILLA_AUGRU || prb.alg == LBR_AUGRU;
        if (is_AUGRU) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.is_lstm_projection() || prb.is_lstm_peephole()) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.is_int8() && prb.alg != VANILLA_LSTM) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.is_s8() && prb.alg == VANILLA_LSTM) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        // Implemented only for CPU
        if (prb.cfg[BIAS].dt == dnnl_bf16 || prb.cfg[SRC_ITER_C].dt == dnnl_bf16
                || prb.cfg[DST_ITER_C].dt == dnnl_bf16) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
        if (prb.flags != NONE) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
    }
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb_ = this;
    const prb_t &prb = *prb_;

    // Consistency validation.
    bool consistent_proj
            = IMPLICATION(!prb.with_projection, prb.dhc == prb.dic);
    bool consistent_L = IMPLICATION(prb.n_layer > 1, prb.slc == prb.dic);
    bool consistent_T = IMPLICATION(prb.n_iter > 1, prb.sic == prb.dic);
    bool is_GRU = prb.alg == VANILLA_GRU || prb.alg == LBR_GRU;
    bool consistent_GRU = IMPLICATION(is_GRU, prb.sic == prb.dic);
    bool is_AUGRU = prb.alg == VANILLA_AUGRU || prb.alg == LBR_AUGRU;
    bool consistent_AUGRU = IMPLICATION(is_AUGRU,
            prb.sic == prb.dic && prb.n_layer == 1
                    && prb.direction == dnnl_unidirectional_left2right);
    if (!consistent_proj || !consistent_L || !consistent_T || !consistent_GRU
            || !consistent_AUGRU) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Only LSTM supports peephole and projection layer.
    bool is_lstm_peephole
            = IMPLICATION(prb.with_peephole, prb.alg == VANILLA_LSTM);
    bool is_lstm_projection
            = IMPLICATION(prb.with_projection, prb.alg == VANILLA_LSTM);
    if (!is_lstm_peephole || !is_lstm_projection) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Bitwise backward requires the flag, otherwise, diff_weights accumulate
    // the output, which doesn't allow to validate numerical stability.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && (prb.prop == dnnl_backward)
            && prb.flags != DIFF_WEIGHTS_OVERWRITE) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Non-trivial strides modify existing strides, when the tag is defined.
    // With tag::any, strides are not defined.
    if (!prb.trivial_strides
            && (prb.tag[0] == tag::any || prb.tag[2] == tag::any)) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const auto rnn_kind = data_kind2rnn_data_kind(kind);
    const auto &cfg = prb->cfg[rnn_kind];
    // factor 2 is because of the sum of 2 GEMMs
    int64_t fwd_acc_dim = 2 * prb->n_gates() + 1;
    if (prb->alg == VANILLA_GRU || prb->alg == VANILLA_AUGRU)
        fwd_acc_dim *= prb->sic;
    int64_t bwdd_acc_dim = prb->n_gates() * prb->dhc;
    int64_t bwdw_acc_dim = prb->mb;
    int64_t acc_dim = fwd_acc_dim;
    if (prb->prop == dnnl_backward) acc_dim *= MAX2(bwdd_acc_dim, bwdw_acc_dim);
    // Here the factor 4 just gives some wiggle room for fp32 testing

    float trh = 4
            * (1 + (prb->prop == dnnl_backward)) // double wiggle room for bwd
            * ((prb->direction == dnnl_bidirectional_sum)
                    + 1) // double trh if bidir_sum
            * ceilf(log2f(acc_dim * prb->n_iter)) * cfg.eps;
    // expect exact value for int8
    if (cfg.dt == dnnl_u8 || cfg.dt == dnnl_s8) trh = 0.f;
    cmp.set_threshold(trh);

    // Note: we do an eltwise comparison only when:
    // - we use skip_nonlinear;
    // - we do not use skip_nonlinear and we test only one cell execution;
    // - for int8 computations the tensor is not DST_ITER_C;
    // If the above conditions are not met, we check only L1, L2 and L8.

    // Rough rationale for the `DST_ITER_C` exception in int8 case:
    // - The formula for one-step c-state is:
    //   c_t = f_t * c_{t−1} + i_t * c~_t.
    //   Here all computations happen in f32 (f_t, i_t, and c~_t are dequantized
    //   right before the computations + the corresponding bias added).
    // - In int8 case we don't have much control over these components and
    //   cannot surmount potential cancellations, if any.
    //   In practice, I observed that the relative element-wise error of values
    //   in `DST_ITER_C` was bigger (up-to 8e-5) whenever the values
    //   themselves were smaller (which indirectly means the problem is exactly
    //   in the cancellation). Unfortunately, this even happened with only one
    //   layer and one time stamp.
    // - So, for now the solution is to use l1- l2- and l_inf-norms to validate
    //   `DST_ITER_C`. When we switch testing on using precise
    //   integer arithmetic based on modulo operation in rnn_tparams (instead of
    //   current unreliable re-scaling), this testing weakness should go away.
    // - Just an obvious side note: `DST_LAYER` and `DST_ITER`
    //   are immediate dequantization of the corresponding u8 tensors. Hence,
    //   as long as we get precise u8 intermediate results (and so far we do),
    //   the f32 result should be pretty accurate -- the dequantization is just
    //   two simple ops: f32 = scale * u8 + shift.
    const bool disallow_norm_check = prb->skip_nonlinear
            || (prb->n_layer == 1 && prb->n_iter == 1)
            || (prb->is_int8() && rnn_kind == DST_ITER_C);
    cmp.set_allow_norm_check(!disallow_norm_check);

    const auto rnn_add_check =
            [&, prb](const compare::compare_t::driver_check_func_args_t &args) {
        // Limitation from current filling.
        // TODO: find a better filling to get rid of this...
        if ((prb->alg == VANILLA_GRU || prb->alg == LBR_AUGRU
                    || prb->alg == VANILLA_RNN || prb->alg == LBR_GRU
                    || prb->alg == VANILLA_LSTM)
                && prb->prop == dnnl_backward) {
            return args.diff < args.trh;
        }

        // When a problem uses int computations, DST_ITER(_C) is computed using
        // DST_LAYER. However, the library part can compute LAYER and ITER in
        // parallel, which can lead to off-by-1 issue for ITER part.
        // Reconstruct original DST_LAYER values on got and exp sides and if
        // they are off-by-1, let them through.
        if (prb->cfg.is_int8()
                && (args.dk == rnn_data_kind2data_kind(DST_ITER)
                        || args.dk == rnn_data_kind2data_kind(DST_ITER_C))) {
            const int exp_q = static_cast<int>(
                    args.exp * prb->data_scale + prb->data_shift);
            const int got_q = static_cast<int>(
                    args.got * prb->data_scale + prb->data_shift);
            return abs(got_q - exp_q) <= 1;
        }
        return false;
    };
    cmp.set_driver_check_function(rnn_add_check);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC_LAYER,
            DNNL_ARG_AUGRU_ATTENTION,
            DNNL_ARG_SRC_ITER,
            DNNL_ARG_SRC_ITER_C,
            DNNL_ARG_WEIGHTS_LAYER,
            DNNL_ARG_WEIGHTS_ITER,
            DNNL_ARG_WEIGHTS_PEEPHOLE,
            DNNL_ARG_WEIGHTS_PROJECTION,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST_LAYER,
            DNNL_ARG_DST_ITER,
            DNNL_ARG_DST_ITER_C,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC_LAYER,
            DNNL_ARG_AUGRU_ATTENTION,
            DNNL_ARG_SRC_ITER,
            DNNL_ARG_SRC_ITER_C,
            DNNL_ARG_WEIGHTS_LAYER,
            DNNL_ARG_WEIGHTS_ITER,
            DNNL_ARG_WEIGHTS_PEEPHOLE,
            DNNL_ARG_WEIGHTS_PROJECTION,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST_LAYER,
            DNNL_ARG_DST_ITER,
            DNNL_ARG_DST_ITER_C,
            DNNL_ARG_WORKSPACE,
            DNNL_ARG_DIFF_DST_LAYER,
            DNNL_ARG_DIFF_DST_ITER,
            DNNL_ARG_DIFF_DST_ITER_C,
            DNNL_ARG_DIFF_SRC_LAYER,
            DNNL_ARG_DIFF_AUGRU_ATTENTION,
            DNNL_ARG_DIFF_SRC_ITER,
            DNNL_ARG_DIFF_SRC_ITER_C,
            DNNL_ARG_DIFF_WEIGHTS_LAYER,
            DNNL_ARG_DIFF_WEIGHTS_ITER,
            DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE,
            DNNL_ARG_DIFF_WEIGHTS_PROJECTION,
            DNNL_ARG_DIFF_BIAS,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
                                                       : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    if (!ref_mem_map.empty()) { erase_unused_args(ref_mem_map, mem_map); }

    const auto &prb = *prb_t::from(base_prb);
    const auto &ref_engine = get_cpu_engine();

    auto const_pd = query_pd(prim);
    // for int8 RNN we need pass attributes for data q10n
    auto rnn_attr = query_attr(const_pd);
    const bool is_fwd_prim = is_fwd_prop_kind(query_prop_kind(const_pd));

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD && exec_arg != DNNL_ARG_WORKSPACE) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC_LAYER:
                SAFE(fill_activation(exec_arg, prb, SRC_LAYER, mem, ref_mem,
                             res, rnn_attr),
                        WARN);
                break;
            case DNNL_ARG_AUGRU_ATTENTION:
                SAFE(fill_activation(exec_arg, prb, AUGRU_ATTENTION, mem,
                             ref_mem, res, rnn_attr),
                        WARN);
                break;
            case DNNL_ARG_SRC_ITER:
                SAFE(fill_activation(exec_arg, prb, SRC_ITER, mem, ref_mem, res,
                             rnn_attr),
                        WARN);
                break;
            case DNNL_ARG_SRC_ITER_C:
                SAFE(fill_src_iter_c(
                             exec_arg, prb, mem, ref_mem, res, rnn_attr),
                        WARN);
                break;
            case DNNL_ARG_WEIGHTS_LAYER:
                if (is_fwd_prim)
                    SAFE(fill_weights(exec_arg, prb, WEIGHTS_LAYER, mem,
                                 ref_mem, res, rnn_attr),
                            WARN);
                break;
            case DNNL_ARG_WEIGHTS_ITER:
                if (is_fwd_prim)
                    SAFE(fill_weights(exec_arg, prb, WEIGHTS_ITER, mem, ref_mem,
                                 res, rnn_attr),
                            WARN);
                break;
            case DNNL_ARG_WEIGHTS_PEEPHOLE:
                if (is_fwd_prim)
                    SAFE(fill_memory(exec_arg, prb, WEIGHTS_PEEPHOLE, mem,
                                 ref_mem, res),
                            WARN);
                break;
            case DNNL_ARG_WEIGHTS_PROJECTION:
                if (is_fwd_prim)
                    SAFE(fill_weights(exec_arg, prb, WEIGHTS_PROJECTION, mem,
                                 ref_mem, res, rnn_attr),
                            WARN);
                break;
            case DNNL_ARG_BIAS:
                if (is_fwd_prim)
                    SAFE(fill_memory(exec_arg, prb, BIAS, mem, ref_mem, res),
                            WARN);
                break;
            case DNNL_ARG_DST_LAYER:
                if (is_fwd_prim)
                    SAFE(fill_activation(
                                 exec_arg, prb, DST_LAYER, mem, ref_mem, res),
                            WARN);
                break;
            case DNNL_ARG_DST_ITER:
                if (is_fwd_prim)
                    SAFE(fill_activation(
                                 exec_arg, prb, DST_ITER, mem, ref_mem, res),
                            WARN);
                break;
            case DNNL_ARG_DST_ITER_C:
                if (is_fwd_prim)
                    SAFE(fill_memory(
                                 exec_arg, prb, DST_ITER_C, mem, ref_mem, res),
                            WARN);
                break;
            case DNNL_ARG_SCRATCHPAD: /* Put internal allocations here */ break;
            case DNNL_ARG_WORKSPACE: /* Or here... */ break;
            case DNNL_ARG_DIFF_SRC_LAYER:
                SAFE(fill_activation(
                             exec_arg, prb, DIFF_SRC_LAYER, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_AUGRU_ATTENTION:
                SAFE(fill_activation(exec_arg, prb, DIFF_AUGRU_ATTENTION, mem,
                             ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_SRC_ITER:
                SAFE(fill_activation(
                             exec_arg, prb, DIFF_SRC_ITER, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_SRC_ITER_C:
                SAFE(fill_memory(
                             exec_arg, prb, DIFF_SRC_ITER_C, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_WEIGHTS_LAYER:
                SAFE(fill_weights(exec_arg, prb, DIFF_WEIGHTS_LAYER, mem,
                             ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_WEIGHTS_ITER:
                SAFE(fill_weights(exec_arg, prb, DIFF_WEIGHTS_ITER, mem,
                             ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE:
                SAFE(fill_memory(exec_arg, prb, DIFF_WEIGHTS_PEEPHOLE, mem,
                             ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_WEIGHTS_PROJECTION:
                SAFE(fill_memory(exec_arg, prb, DIFF_WEIGHTS_PROJECTION, mem,
                             ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_BIAS:
                SAFE(fill_bias(exec_arg, prb, DIFF_BIAS, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_DST_LAYER:
                SAFE(fill_activation(
                             exec_arg, prb, DIFF_DST_LAYER, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_DST_ITER:
                SAFE(fill_activation(
                             exec_arg, prb, DIFF_DST_ITER, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_DST_ITER_C:
                SAFE(fill_memory(
                             exec_arg, prb, DIFF_DST_ITER_C, mem, ref_mem, res),
                        WARN);
                break;
            default: break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if ((prb->dir & FLAG_FWD) && (dir & FLAG_FWD)) {
        check_kinds = {data_kind_t::DST, data_kind_t::DST_ITER};
        if (prb->alg == VANILLA_LSTM) {
            check_kinds.push_back(data_kind_t::DST_ITER_C);
        }
    } else if ((prb->dir & FLAG_BWD) && (dir & FLAG_BWD)) {
        check_kinds = {data_kind_t::DST, data_kind_t::DST_ITER,
                data_kind_t::SRC, data_kind_t::SRC_ITER, data_kind_t::WEI,
                data_kind_t::WEI_ITER, data_kind_t::BIA};
        if (prb->alg == VANILLA_LSTM) {
            check_kinds.push_back(data_kind_t::DST_ITER_C);
            check_kinds.push_back(data_kind_t::SRC_ITER_C);
        }
        if (prb->alg == VANILLA_AUGRU || prb->alg == LBR_AUGRU)
            check_kinds.push_back(data_kind_t::AUGRU_ATTENTION);
        if (prb->is_lstm_peephole())
            check_kinds.push_back(data_kind_t::WEI_PEEPHOLE);
        if (prb->is_lstm_projection())
            check_kinds.push_back(data_kind_t::WEI_PROJECTION);
    }
    // `check_kinds` is empty for `(prb->dir & FLAG_BWD) && (dir & FLAG_FWD)`.
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t &base_prb, res_t *res) {
    const prb_t &prb = static_cast<const prb_t &>(base_prb);
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb.ctx_init, v_prim[0], init_pd, &prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb.dir & FLAG_BWD),
            WARN);
    if (prb.dir & FLAG_BWD) {
        SAFE(init_prim(prb.ctx_init, v_prim[1], init_pd, &prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        // Set memory used for reference.
        auto &mem_args = res->mem_size_args;
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            mem_args.extra_size_driver = prb->get_ws_size(FLAG_FWD);
        }

        SAFE(check_total_size(res), WARN);

        if (v_prim[1]) {
            // For backward, need to accumulate extra space with forward.
            if (has_bench_mode_bit(mode_bit_t::corr)) {
                mem_args.extra_size_driver += prb->get_ws_size(FLAG_BWD);
            }

            SAFE(check_total_size(res), WARN);
        }
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
        if (v_prim[1]) {
            SAFE(check_caches(v_prim[1], prb->ctx_init, res), WARN);
        }
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t &base_prb, res_t *res) {
    const prb_t &prb = static_cast<const prb_t &>(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = prb.prop != dnnl_backward ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(
                 mem_map, &prb, v_prim[0], res, /*override_dir_with_fwd=*/true),
            WARN);
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, v_prim[0], &prb, res),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto st = run_execution(v_prim[0], args, res);
    if (st == FAIL) {
        prb.skip_unimplemented(res);
        if (res->state == SKIPPED || res->state == DEFERRED) return OK;
        return FAIL;
    }
#else
    SAFE(run_execution(v_prim[0], args, res), WARN);
#endif

    check_correctness(&prb, get_kinds_to_check(&prb, FLAG_FWD), args, ref_args,
            compute_ref, setup_cmp, res, FLAG_FWD);
    SAFE(check_bitwise(prim, get_kinds_to_check(&prb, FLAG_FWD), args, prb.attr,
                 prb.inplace, res),
            WARN);

    if (prb.prop == dnnl_backward) {
        // Pass same memory map as we need data from forward on backward.
        SAFE(init_memory_args(mem_map, &prb, v_prim[1], res), WARN);
        TIME_FILL(SAFE(init_ref_memory_args(
                               ref_mem_map, mem_map, v_prim[1], &prb, res),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(run_execution(v_prim[1], args, res), WARN);

        check_correctness(&prb, get_kinds_to_check(&prb, FLAG_BWD), args,
                ref_args, compute_ref, setup_cmp, res, prb.dir);
        SAFE(check_bitwise(prim, get_kinds_to_check(&prb, FLAG_BWD), args,
                     prb.attr, prb.inplace, res),
                WARN);
    }

    return measure_perf(prb.ctx_exe, res, prim, args);
}

} // namespace rnn
