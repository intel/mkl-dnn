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

#ifndef UTILS_PRB_HPP
#define UTILS_PRB_HPP

#include "dnn_types.hpp"
#include "utils/impl_filter.hpp"
#include "utils/res.hpp"
#include "utils/wrapper.hpp"

#include "tests/thread_context.hpp"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

struct base_perf_report_t;

// A base class for all driver-specific `prb_t` problem descriptors. It holds
// the members that are common across **every** driver and exposes a polymorphic
// interface for all functions to rely on it.
struct base_prb_t {
    base_prb_t() = default;
    base_prb_t(dir_t dir, bool inplace, const attr_t &attr,
            const thr_ctx_t &ctx_init, const thr_ctx_t &ctx_exe,
            const impl_filter_t &impl_filter)
        : dir(dir)
        , inplace(inplace)
        , attr(attr)
        , ctx_init(ctx_init)
        , ctx_exe(ctx_exe)
        , impl_filter(impl_filter) {}

    virtual ~base_prb_t() = default;

    dir_t dir = FLAG_FWD;
    bool inplace = false;
    attr_t attr;
    thr_ctx_t ctx_init, ctx_exe;
    impl_filter_t impl_filter;

    // Use to construct memory descriptor when dimensions are runtime since such
    // memory descriptors can't be used directly from a query call and memory
    // objects can't be constructed from such descriptors.
    // Drivers supporting runtime dimensions must override this call.
    virtual benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> get_md(int arg) const {
        assert(!"No runtime dimensions support for this driver!");
        return make_benchdnn_dnnl_wrapper<dnnl_memory_desc_t>(nullptr);
    }

    // Checks driver-specific cases of unimplemented functionality post pd
    // creation. Each driver overrides it with its own logic.
    virtual void skip_unimplemented(res_t *res) const {}
    virtual void skip_invalid(res_t *res) const {}

    // Returns the list of execution arguments supported by the driver. When
    // `override_dir_with_fwd` is set, the forward set is returned regardless of
    // the problem direction; this covers drivers that prepare a forward
    // primitive prior to the backward one. The direction is taken from the
    // `dir` member of this object. Drivers relying on the common memory args
    // machinery override it; the rest use the empty default.
    virtual std::vector<int> supported_exec_args(
            bool override_dir_with_fwd) const {
        return {};
    }

    // Pure virtual so that every driver must supply its `perf_report_t`, which
    // keeps the shared `task_t` independent of driver types.
    virtual std::unique_ptr<base_perf_report_t> create_perf_report(
            const char *perf_template) const
            = 0;

    const char *str() const { return repro.c_str(); }

protected:
    // Broadcasts a single-element vector `v` to `n_inputs` copies of `val`. If
    // `v` already contains more than one element, it is left untouched.
    // TODO: make `n_inputs` a `prb_t` method and drop it from the signature.
    template <typename T>
    void broadcast_vector(std::vector<T> &v, const T &val, const int n_inputs) {
        // If it's not a single element in vector, nothing to broadcast.
        if (v.size() != 1) return;

        // std::vector<T>::assign invalidates all iterators and references.
        // `this->stag.assign(prb_vdims.n_inputs(), stag[0]);` line can
        // crash depending on the implementation whether it saves `stag[0]`
        // before invalidating or not. Windows crashed.
        v.assign(n_inputs, val);
    }

    // Collects driver-specific reproducer information into a single line. Each
    // driver provides its own implementation and must call it last in the ctor
    // to assign the result to `repro`.
    virtual std::string set_repro_line() = 0;

    std::string repro;
};

#endif
