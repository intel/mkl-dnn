/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <cmath>
#include <mutex>
#include <string>
#include <thread>

#include "common.hpp"
#include "utils/compare.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/norm.hpp"
#include "utils/parallel.hpp"
#include "utils/stringstream.hpp"

#include "eltwise/eltwise.hpp"

namespace compare {

namespace {

void dump_point_values(
        const std::string &kind_str, const compare_t::dump_point_ctx_t &ctx) {
    stringstream_t ss;
    dims_t l_dims = md2dims(ctx.md);
    dims_t dims_idx = off2dims_idx(l_dims, ctx.l_offset);
    ss << dims_idx;
    std::string ind_str = ss.str();

    BENCHDNN_PRINT(0,
            "[%4" PRId64
            "]%s[%s] exp_f32:%12g exp:%12g got:%12g diff:%8g rdiff:%8g\n",
            ctx.l_offset, kind_str.c_str(), ind_str.c_str(), ctx.exp_f32,
            ctx.exp, ctx.got, ctx.diff, ctx.rel_diff);
}

void dump_norm_values(
        const diff_norm_t &diff_norm, const std::string &kind_str) {
    BENCHDNN_PRINT(0,
            "%s[L0] = %g\n"
            "%s[L1] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L2] exp:%8g got:%8g diff:%8g rel_diff:%8g\n"
            "%s[L8] exp:%8g got:%8g diff:%8g rel_diff:%8g\n",
            kind_str.c_str(), diff_norm.rel_diff(norm_t::L0), kind_str.c_str(),
            diff_norm.a_[norm_t::L1], diff_norm.b_[norm_t::L1],
            diff_norm.diff_[norm_t::L1], diff_norm.rel_diff(norm_t::L1),
            kind_str.c_str(), diff_norm.a_[norm_t::L2],
            diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
            diff_norm.rel_diff(norm_t::L2), kind_str.c_str(),
            diff_norm.a_[norm_t::L8], diff_norm.b_[norm_t::L8],
            diff_norm.diff_[norm_t::L8], diff_norm.rel_diff(norm_t::L8));
}

bool has_binary_po_algs(const attr_t &attr,
        const std::vector<attr_t::post_ops_t::kind_t> &algs) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return false;

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (!e.is_binary_kind()) continue;

        if (std::any_of(algs.cbegin(), algs.cend(),
                    [&](const attr_t::post_ops_t::kind_t alg) {
            return e.kind == alg;
        }))
            return true;
    }
    return false;
}

bool has_binary_comparison_po(const attr_t &attr) {
    using alg_t = attr_t::post_ops_t::kind_t;
    static const std::vector<alg_t> cmp_alg = {alg_t::MAX, alg_t::MIN,
            alg_t::GE, alg_t::GT, alg_t::LE, alg_t::LT, alg_t::EQ, alg_t::NE};
    return has_binary_po_algs(attr, cmp_alg);
}

bool has_binary_compute_po(const attr_t &attr) {
    using alg_t = attr_t::post_ops_t::kind_t;
    static const std::vector<alg_t> cmp_alg
            = {alg_t::ADD, alg_t::SUB, alg_t::MUL, alg_t::DIV};
    return has_binary_po_algs(attr, cmp_alg);
}

bool negative_converts_to_zero(const attr_t &attr, dnnl_data_type_t target_dt) {
    using po_kind_t = attr_t::post_ops_t::kind_t;
    const auto &po = attr.post_ops;

    // Check for all post-ops that convert negative to zero
    std::vector<po_kind_t> non_neg_po {po_kind_t::ABS};
    std::vector<po_kind_t> non_neg_alpha_0_po {po_kind_t::CLIP,
            po_kind_t::CLIP_V2, po_kind_t::ELU, po_kind_t::RELU};
    for (int i = 0; i < po.len(); ++i) {
        const auto &e = po.entry[i];
        if (!e.is_eltwise_kind()) continue;

        auto k = e.kind;
        auto alpha = e.eltwise.alpha;

        if (std::any_of(non_neg_po.cbegin(), non_neg_po.cend(),
                    [k](const po_kind_t alg) { return alg == k; }))
            return true;

        if (std::any_of(non_neg_alpha_0_po.cbegin(), non_neg_alpha_0_po.cend(),
                    [k, alpha](const po_kind_t alg) {
            return alg == k && alpha == 0;
        }))
            return true;
    }
    // Check for u8 dst
    if (target_dt == dnnl_u8) return true;

    return false;
}

} // namespace

bool compare_extreme_values(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
        return true;
    return false;
}

compare_t::driver_check_func_args_t::driver_check_func_args_t(
        const dnn_mem_t &exp_mem, const dnn_mem_t &got_f32, const int64_t i,
        const dnnl_data_type_t data_type, const float trh, data_kind_t dk)
    : dt(data_type)
    , idx(i)
    , exp_f32(exp_mem.get_f32_elem(idx))
    , exp(round_to_nearest_representable(dt, exp_f32))
    , got(got_f32.get_f32_elem(idx))
    , diff(fabsf(exp - got))
    , rel_diff(diff / (fabsf(exp) > FLT_MIN ? fabsf(exp) : 1))
    , trh(trh)
    , dk(dk) {}

int compare_t::compare_norm(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    const auto nelems = got_mem.nelems();
    if (nelems == 0) {
        if (res->state == EXECUTED) res->state = PASSED;
        return OK;
    }

    res->total = 1;

    dnn_mem_t got_f32(got_mem, dnnl_f32, tag::abx, get_cpu_engine());
    const auto dt = got_mem.dt();

    // Idea is to pad nelems to mimic uniform load between available threads.
    // It allows to make a clear assumption when the last element is processed
    // and do a sync with global diff_norm object.
    // The better alternative is to expose `parallel` interface.
    const auto nthreads = benchdnn_get_max_threads();
    const auto nelems_per_thread = div_up(nelems, nthreads);
    const auto nelems_pad = nelems_per_thread * nthreads;

    diff_norm_t diff_norm;
    const bool need_dump = verbose >= 99;

    const auto compare_norm_values = [&](int64_t i) {
        if (i >= nelems) return;

        // Specifiers to keep data accumulated over several `i`.
        static thread_local diff_norm_t diff_norm_ithr;
        driver_check_func_args_t args(
                exp_mem, got_f32, i, dt, trh_norm_, kind_);

        if ((std::isnan(args.exp_f32)) || std::isinf(args.exp)) {
            // Don't include nan inf values into norm as they make it
            // irrelevant for validation.
            ;
        } else if (is_cpu() && dt == dnnl_s32 && args.exp == max_dt(dnnl_s32)
                && args.got >= BENCHDNN_S32_TO_F32_SAT_CONST
                && args.got < max_dt(dnnl_s32)) {
            // Don't include f32->s32 saturation values into norm as they make
            // it irrelevant for validation.
            ;
        } else {
            diff_norm_ithr.update(args.exp, args.got);
        }

        // Synchronization point, exchange to main diff_norm and reset thread's
        // diff_norm_ithr object.
        if (((i + 1) % nelems_per_thread == 0) || (i == nelems - 1)) {
            static std::mutex m;
            std::lock_guard<std::mutex> guard(m);
            diff_norm.update(diff_norm_ithr);
            diff_norm_ithr = diff_norm_t();
        }
    };

    // Parallel norm computation to speed up the process.
    benchdnn_parallel_nd(nelems_pad, compare_norm_values);

    diff_norm.done();

    bool ok = diff_norm.rel_diff(norm_t::L2) <= trh_norm_;
    if (!ok) res->errors = 1;

    const bool dump = need_dump || !ok;
    if (dump) {
        if (!need_dump) {
            // Forced dump was printed in p2p.
            dump_p2p_errors();
        }
        dump_norm_values(diff_norm, get_kind_str());
    }

    if (res->errors) res->state = FAILED;

    // Status may be propagated from previous tensor. Use stats from cur tensor.
    BENCHDNN_PRINT((res->errors ? 0 : 6),
            "[COMPARE_STATS]%s: trh=%g (compare against [L2] rel_diff)\n",
            get_kind_str().c_str(), trh_norm_);

    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

int compare_t::compare_p2p(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    const auto nelems = got_mem.nelems();
    if (nelems == 0) {
        if (res->state == EXECUTED) res->state = PASSED;
        return OK;
    }

    res->total += nelems;

    dnn_mem_t got_f32(got_mem, dnnl_f32, tag::abx, get_cpu_engine());
    dnn_mem_t exp_f32_plain;
    const bool is_prim_ref_dst_mem_f32_abx
            = query_md_data_type(exp_mem.md_) == dnnl_f32
            && IMPLICATION(has_prim_ref_,
                    check_md_consistency_with_tag(exp_mem.md_, tag::abx));
    if (!is_prim_ref_dst_mem_f32_abx) {
        exp_f32_plain
                = dnn_mem_t(exp_mem, dnnl_f32, tag::abx, get_cpu_engine());
    }
    const dnn_mem_t &exp_f32 = exp_f32_plain ? exp_f32_plain : exp_mem;

    // Note: the section below has some adjustable settings but they all seem
    // to target DST kind_.
    // TODO: add a list of allowed kinds to get into, and adjust settings to
    // apply only to kinds they aim. So far, play it safe and remove DST_SCALES
    // when not applicable.
    const auto dt = got_mem.dt();
    const bool kind_is_dst_scales = kind_ == DST_SCALES;
    const bool has_eltwise = !kind_is_dst_scales
            && (attr.post_ops.eltwise_index() != -1 || has_eltwise_post_op_);
    // Output fp8 data types expected to have dynamic dst scaling. When such
    // scaling is enabled, it saturates, doesn't overflow, thus, NaNs are not
    // expected in the output. No dynamic scaling - NaN may happen.
    const bool non_scaled_fp8_output
            = (dt == dnnl_f8_e5m2 || dt == dnnl_f8_e4m3)
            && !attr.scales.get(DNNL_ARG_DST).is_dynamic();
    const bool output_has_nans = op_output_has_nans_
            || eltwise::eltwise_alg_returns_nan_or_inf(attr)
            || has_binary_po_algs(attr, {attr_t::post_ops_t::kind_t::DIV})
            || dt == dnnl_f16 || non_scaled_fp8_output;
    const bool has_exp_eltwise = !kind_is_dst_scales
            && attr.post_ops.find(attr_t::post_ops_t::kind_t::EXP) >= 0;
    const auto &dst_scale = attr.scales.get(DNNL_ARG_DST);
    const bool has_dst_scale = !dst_scale.is_def();

    // Idea is to pad nelems to mimic uniform load between available threads.
    // It allows to make a clear assumption when the last element is processed
    // and do a sync with global diff_norm object.
    // The better alternative is to expose `parallel` interface.
    const auto nthreads = benchdnn_get_max_threads();
    const auto nelems_per_thread = div_up(nelems, nthreads);
    const auto nelems_pad = nelems_per_thread * nthreads;

    // These global metrics are updated at the synchronization point.
    int64_t zeros = 0;
    // "all_" stuff is across the whole tensor. "err_" stuff is just for points
    // that didn't pass any criteria.
    float all_max_rdiff = 0.f;
    float all_max_diff = 0.f;
    float err_max_rdiff = 0.f;
    float err_max_diff = 0.f;
    const bool need_dump = verbose >= 99;

    // Make thread_data static so that acquiring the thread data can be
    // performed in a static thread_local variable to minimize locking
    static struct {
        struct data_t {
            int64_t n_errors;
            std::vector<compare_t::dump_point_ctx_t> dumps;
        };

        data_t &get() {
            std::lock_guard<std::mutex> guard(m);
            return data[std::this_thread::get_id()];
        }
        void reset() {
            for (auto &d : data) {
                d.second.n_errors = 0;
                d.second.dumps.clear();
            }
        }

        std::unordered_map<std::thread::id, data_t> data;
        std::mutex m;
    } thread_data;

    // Clear data from previous runs for the static variable
    thread_data.reset();

    const auto compare_point_values = [&](int64_t i) {
        // Skip padded (non-existent) elements.
        if (i >= nelems) return;

        // Stats for all validated points per one thread.
        static thread_local int64_t ithr_zeros = 0;
        static thread_local float ithr_all_max_rdiff = 0.f;
        static thread_local float ithr_all_max_diff = 0.f;
        static thread_local float ithr_err_max_rdiff = 0.f;
        static thread_local float ithr_err_max_diff = 0.f;

        // This is valid because references to data are only invalidated by
        // erasing that element, but it does require that thread_data is a
        // static variable and the corresponding element isn't erased between
        // calls to this function.
        static thread_local auto &out_data = thread_data.get();

        const auto got_val = got_f32.get_f32_elem(i);
        bool ok = exp_f32.get_f32_elem(i) == got_val;

        static thread_local driver_check_func_args_t args;
        for (int z = ok; z < 1; z++) {
            args = driver_check_func_args_t(
                    exp_f32, got_f32, i, dt, trh_, kind_);

            if (std::isnan(args.exp_f32) && is_integral_dt(dt)) {
                // There's no single spec to comply with when it comes to the
                // implementation of NaN fp32 to int conversion. CPU backend
                // saturates the value to INT32_MAX, UINT8_MAX, INT8_MIN, while
                // GPU converts it to 0.
                ok = true;
                break;
            }

            // Discard tiny values very close to each other. It's impossible to
            // compare them reliably and fit into any criterion.
            //
            // Note: the only exception from this condition is `e8m0` dt since
            // its minimal value passes this criterion though in the way it's
            // written. The minimal value must be check for exactness and if it
            // wasn't, it should be reported. Skipping `e8m0` in this condition
            // allows to catch the error properly.
            ok = fabsf(args.exp) <= 1e-5f && args.diff < epsilon_dt(dnnl_f32)
                    && args.dt != dnnl_e8m0;
            if (ok) break;

            // Standard check for relative diff is under set threshold.
            //
            // Note: and same limitations for `e8m0`.
            ok = (fabsf(args.exp) > 1e-5f ? args.rel_diff : args.diff) <= trh_
                    && args.dt != dnnl_e8m0;
            if (ok) break;

            // When NaNs or infinity are allowed for the driver, check
            // that both exp and got are NaNs or infinity with same sign.
            ok = output_has_nans
                    && compare::compare_extreme_values(args.exp, got_val);
            if (ok) break;

            // Use hack to check not fully correct s32 saturation on CPU.
            ok = is_cpu() && dt == dnnl_s32 && args.exp == max_dt(dnnl_s32)
                    && got_val >= BENCHDNN_S32_TO_F32_SAT_CONST
                    && got_val < max_dt(dnnl_s32);
            if (ok) break;

            // Check driver's additional checks, if set.
            ok = driver_check_func_ && driver_check_func_(args);
            if (ok) break;

            // Check if there are eltwise post-ops, use very relaxed
            // comparison since we can't control inputs for each driver finely
            // or validate if the output value from operation satisfies the
            // check for catastrophic cancellation (see eltwise additional check
            // function). We rely on validation of pure eltwise and let some
            // big rdiff errors slip away hoping that absolute error is good
            // enough.
            // Note: two scenarios covered:
            // * When rdiff is bigger due to small output values but diff is
            //   small due to single point computation or short acc chain.
            // * When diff is no longer small due to longer acc chain, but rdiff
            //   is still small but greater than 0.
            // For integral dst, a small f32 difference before the final
            // rounding may flip the result by one unit; allow that for
            // eltwise post-ops.
            const float experimental_eltwise_trh_diff = is_integral_dt(dt)
                    ? 1.f
                    : std::max(epsilon_dt(dt), 2e-5f);
            const float experimental_eltwise_trh_rel_diff
                    = std::max(epsilon_dt(dt), 8e-6f);
            ok = has_eltwise
                    && (args.diff <= experimental_eltwise_trh_diff
                            || args.rel_diff
                                    <= experimental_eltwise_trh_rel_diff);
            if (ok) break;

            // For eltwise it also may happen that threshold is really small,
            // but absolute difference is really big. Also exponent is a special
            // transcendental post-op that has accuracy issues with older isa.
            ok = has_eltwise && (fabsf(args.exp) > 1e+5f || has_exp_eltwise)
                    && args.rel_diff <= std::max(epsilon_dt(dt), 5e-6f);
            if (ok) break;

            // f32 dst scale is used as a divisor to quantize data to dt.
            // Implementation might decide to pre-compute inverse value and
            // multiply on it in kernel. This difference might result in a
            // slight error comparing to a division operation.
            const float experimental_dst_scale_trh
                    = std::max(epsilon_dt(dt), 1e-5f);
            ok = has_dst_scale && dst_scale.dt == dnnl_f32
                    && args.rel_diff <= experimental_dst_scale_trh;
            if (ok) break;

            // Binary MAX, MIN and comparison operations post-ops may return
            // different results for different backends when NaN is one of
            // inputs. Depending on its position and implementation, either
            // first or second operand may be returned.
            ok = has_binary_comparison_po(attr) && output_has_nans;
            if (ok) break;

            // Binary Add/Sub/Mul/Div usually produce additional noise in the
            // output. For Add/Sub it's mostly catastrophic cancellation, for
            // Mul it's usually rounding, for Div it's usually different
            // precision level of instructions for different backends. Those are
            // hard to fix with filling adjustments. Since binary po filling
            // operates with integers or large numbers, it's safe to let some
            // minor diff error to exist under assumption that original problem
            // lacks those errors.
            //
            // Note: use specific dt and correspondent values not to mess with
            // broad set of supported data types.
            float binary_comp_po_diff_trh = 0.f;
            float binary_comp_po_rdiff_trh = 0.f;
            if (args.dt == dnnl_f16) binary_comp_po_diff_trh = 5e-3f;
            if (args.dt == dnnl_f32) {
                binary_comp_po_diff_trh = 4e-6f;
                binary_comp_po_rdiff_trh = 1e-5f;
            }
            ok = has_binary_compute_po(attr)
                    && (args.diff <= binary_comp_po_diff_trh
                            || args.rel_diff <= binary_comp_po_rdiff_trh);
            if (ok) break;

            // A check for `off-by-1` errors for both integral data types and
            // low-precision floating-point data types.
            ok = (is_integral_dt(args.dt) || bits_dt(args.dt) <= 8)
                    && is_off_by_one(args);
            if (ok) break;

            // Another class of `off-by-1` issues coming from optimized
            // reference when transcendental operation present in the chain. In
            // such cases, there's no way to test original output as both
            // outputs would be rounded to integer number.
            const auto is_int8_prim_ref_and_transcedental = [&]() -> bool {
                if (!has_prim_ref_) return false;
                if (fabsf(args.exp_f32 - got_val) != 1) return false;
                // TODO: update with transcendental eltwise ops only.
                return has_eltwise;
            };
            // There's a class of rounding issues happening around conversion
            // of NaN values into integer data type (see the first check with
            // NaNs) with optimized reference involved since it's impossible to
            // verify the original value against NaN.
            const auto is_nan_to_int_good = [&]() -> bool {
                if (!has_prim_ref_) return false;
                // CPU has prim_ref as well, but it's expected to return same
                // values.
                if (got_val != 0) return false;
                int exp_sat_value = 0;
                switch (args.dt) {
                    case dnnl_s32: exp_sat_value = INT_MAX; break;
                    case dnnl_s8: exp_sat_value = INT8_MIN; break;
                    case dnnl_u8: exp_sat_value = UINT8_MAX; break;
                    default: // Gated behind is_integral_dt(args.dt).
                        assert(!"unexpected data type");
                }
                if (fabsf(args.exp_f32 - exp_sat_value) != 0) return false;
                return true;
            };
            ok = is_integral_dt(args.dt)
                    && (is_int8_prim_ref_and_transcedental()
                            || is_nan_to_int_good());
            if (ok) break;

            // Nvidia backend with fpmath mode enabled returns not exact output
            // values (presumably on conversion to fp32), thus, make sure they
            // fit single ulp for a reduced data type.
            ok = is_nvidia_gpu()
                    && attr.fpmath_mode.mode != dnnl_fpmath_mode_strict
                    && args.diff
                            <= epsilon_dt(deduce_cfg_data_type(dt, attr, SRC));
            if (ok) break;
        }

        // Update compare stats.
        if (fabsf(got_val) == 0) ithr_zeros++;
        if (args.rel_diff > 0)
            ithr_all_max_rdiff = MAX2(ithr_all_max_rdiff, args.rel_diff);
        if (args.diff > 0)
            ithr_all_max_diff = MAX2(ithr_all_max_diff, args.diff);
        if (!ok) ithr_err_max_rdiff = MAX2(ithr_err_max_rdiff, args.rel_diff);
        if (!ok) ithr_err_max_diff = MAX2(ithr_err_max_diff, args.diff);

        if (!ok) out_data.n_errors++;

        const bool dump = need_dump
                || (!ok && (out_data.n_errors <= 10 || verbose >= 10));
        if (dump) {
            // Need to initialize `args` in case they weren't.
            if (args.dt == dnnl_data_type_undef)
                args = driver_check_func_args_t(
                        exp_f32, got_f32, i, dt, trh_, kind_);

            out_data.dumps.emplace_back(got_mem.md_, i, args.exp_f32, args.exp,
                    got_val, args.diff, args.rel_diff);
        }

        // Reset args for the next point if they were initialized.
        if (args.dt != dnnl_data_type_undef) args = driver_check_func_args_t();

        // Synchronization point, update global stats from thread stats.
        if (((i + 1) % nelems_per_thread == 0) || (i == nelems - 1)) {
            static std::mutex m;
            std::lock_guard<std::mutex> guard(m);

            zeros += ithr_zeros;
            // NaN would sneak due to MAX2 implementation picking the second
            // value in case of uncomparable value (which NaN is).
            if (!std::isnan(all_max_rdiff) && !std::isinf(all_max_rdiff))
                all_max_rdiff = MAX2(all_max_rdiff, ithr_all_max_rdiff);
            if (!std::isnan(all_max_diff) && !std::isinf(all_max_diff))
                all_max_diff = MAX2(all_max_diff, ithr_all_max_diff);
            if (!std::isnan(err_max_rdiff) && !std::isinf(err_max_rdiff))
                err_max_rdiff = MAX2(err_max_rdiff, ithr_err_max_rdiff);
            if (!std::isnan(err_max_diff) && !std::isinf(err_max_diff))
                err_max_diff = MAX2(err_max_diff, ithr_err_max_diff);
            ithr_zeros = 0;
            ithr_all_max_rdiff = 0.f;
            ithr_all_max_diff = 0.f;
            ithr_err_max_rdiff = 0.f;
            ithr_err_max_diff = 0.f;
        }
    };

    // parallel comparison to speed up the process
    // TODO: to speed up the dump process, each thread should prepare its dump
    // piece in a string object, then the master thread prints them in order.
    // With this logic, the block of code below won't be needed.
    benchdnn_parallel_nd(nelems_pad, compare_point_values);

    int64_t n_errors = 0;
    for (auto &d : thread_data.data) {
        n_errors += d.second.n_errors;
    }
    // serial comparison with enabled dumping when needed for nicer output.
    if (n_errors > 0 || need_dump) {
        for (auto &d : thread_data.data) {
            p2p_dumps_.insert(p2p_dumps_.end(), d.second.dumps.begin(),
                    d.second.dumps.end());
        }
        std::sort(p2p_dumps_.begin(), p2p_dumps_.end(),
                [](const compare_t::dump_point_ctx_t &a,
                        const compare_t::dump_point_ctx_t &b) {
            return a.l_offset < b.l_offset;
        });
        // If norm fallback is allowed, these dumps will be printed there.
        // This is done to avoid an output disturbance if p2p check fails but
        // norm passes.
        if (need_dump || !allow_norm_check_) dump_p2p_errors();
    }

    // Set state to FAILED in case of any errors.
    if (n_errors) res->errors = n_errors, res->state = FAILED;
    // State could be already FAILED, check zero trust for non-FAILED only.
    const float zeros_percent = 100.f * zeros / nelems;
    float zero_trust_percent = zero_trust_percent_;
    // Adjust default zero trust for cases when negative are converted into 0.
    if (zero_trust_percent_ == default_zero_trust_percent_
            && negative_converts_to_zero(attr, dt)) {
        // (100% - X%) / 2 + X%. X% is default. Each half represents positive
        // and negative in the output equally.
        zero_trust_percent = (100.f + zero_trust_percent_) / 2.f;
    }
    bool is_mistrusted = zeros_percent > zero_trust_percent && nelems >= 10;
    if (res->state != FAILED && is_mistrusted) res->state = MISTRUSTED;

    // Status may be propagated from previous tensor. Use stats from cur tensor.
    BENCHDNN_PRINT((n_errors ? 0 : 6),
            "[COMPARE_STATS]%s: trh=%g err_max_diff:%8g err_max_rdiff:%8g "
            "all_max_diff:%8g all_max_rdiff:%8g\n",
            get_kind_str().c_str(), trh_, err_max_diff, err_max_rdiff,
            all_max_diff, all_max_rdiff);

    BENCHDNN_PRINT((is_mistrusted ? 2 : 6),
            "[COMPARE_TRUST]%s: z:%2.0f%% (>%2.0f%%) (z: %ld, total: %ld)\n",
            get_kind_str().c_str(), zeros_percent, zero_trust_percent,
            (long)zeros, (long)nelems);

    // Set PASSED if no failure in current or previous checks happened and test
    // can be trusted.
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

void compare_t::dump_p2p_errors() const {
    size_t max_dump_size = (verbose >= 10 || p2p_dumps_.size() < 10)
            ? p2p_dumps_.size()
            : 10;
    for (size_t i = 0; i < max_dump_size; i++) {
        dump_point_values(get_kind_str(), p2p_dumps_[i]);
    }
}

bool compare_t::is_off_by_one(const driver_check_func_args_t &args) const {
    // Preserve testing floor/ceil for integral types and not a stricter
    // threshold
    if (is_integral_dt(args.dt)) {
        static constexpr float small_eps = 9e-6f;
        const float floor_val = floorf(args.exp_f32);
        const float ceil_val = ceilf(args.exp_f32);
        if (fabsf((floor_val + 0.5f) - args.exp_f32) >= small_eps) return false;
        if (args.exp == floor_val) return args.got == ceil_val;
        if (args.exp == ceil_val) return args.got == floor_val;
        return false;
    }

    // Low-precision floats (fp8/fp4): accept a single-ULP difference where
    // the ULP is computed for the actual magnitude of the values (including
    // subnormals), and the exact f32 value lies between `exp` and `got`.
    // This covers e.g. double rounding (f32->f16->f8) in implementations vs.
    // direct f32->f8 rounding in the reference.
    if (!std::isfinite(args.exp_f32) || !std::isfinite(args.exp)
            || !std::isfinite(args.got))
        return false;
    // Expected f32 value shouldn't be equal to expected `float` value.
    if (args.exp_f32 == args.exp) return false;
    if (args.got == args.exp) return false;

    const float lo = std::min(args.exp, args.got);
    const float hi = std::max(args.exp, args.got);
    // The exact value must lie strictly between the two neighbors.
    if (!(args.exp_f32 > lo && args.exp_f32 < hi)) return false;

    const float ulp = ulp_dt(args.dt, std::min(fabsf(lo), fabsf(hi)));
    if (ulp <= 0.f) return false;
    return (hi - lo) == ulp;
}

int compare_t::compare(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
        const attr_t &attr, res_t *res) const {
    std::string add_args
            = std::string(allow_norm_check_ ? "allow_norm:true;" : "")
            + std::string(op_output_has_nans_ ? "has_nans:true;" : "")
            + std::string(has_prim_ref_ ? "has_prim_ref:true;" : "");
    BENCHDNN_PRINT(6, "[COMPARE]%s: zero_trust%%=%.2f%% extra=%s\n",
            get_kind_str().c_str(), zero_trust_percent_, add_args.c_str());
    auto st = compare_p2p(exp_mem, got_mem, attr, res);
    if (st != OK && allow_norm_check_) {
        bool call_norm_check = true;
        // Note: the following code specifies additional driver's individual
        // desires when to enable norm check. This one purely depends on the
        // result of p2p comparison. So far graph is the only driver needing
        // such. When this becomes a trend, move it to a registered function
        // mechanism.
        if (driver_name == "graph") {
            // For graph driver there's additional runtime check based on the
            // number of failed points. This is done to limit the risk of hiding
            // issues. If the number of failed points is reasonably low, let it
            // try the norm approach.
            const size_t allowed_error_points = res->total / 1024;
            const bool norm_check_allowed = allowed_error_points >= res->errors;

            BENCHDNN_PRINT(0,
                    "[COMPARE_STATS] Norm check is %s; error_to_total_ratio: "
                    "%zu/%zu; allowed_ratio: %zu/%zu;\n",
                    norm_check_allowed ? "allowed" : "prohibited", res->errors,
                    res->total, allowed_error_points, res->total);

            call_norm_check = norm_check_allowed;
        }

        if (call_norm_check) {
            res->reset_stats(EXECUTED);
            st = compare_norm(exp_mem, got_mem, attr, res);
        } else {
            // Can be triggered by graph only if output wasn't requested.
            const bool need_dump = verbose >= 99;
            if (!need_dump) dump_p2p_errors();
        }
    }
    return st;
}

} // namespace compare
