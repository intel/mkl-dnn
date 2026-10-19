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

#include "dnnl_common.hpp"

#include "utils/cold_cache.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/system.hpp"

cold_cache_input_t cold_cache_input;

const cold_cache_input_t &default_cold_cache_input() {
    static const cold_cache_input_t cold_cache_input;
    return cold_cache_input;
}

namespace cold_cache_utils {
// Returns `arg` index in `dnnl_args` since they packed in random order.
int get_arg_idx(const std::vector<dnnl_exec_arg_t> &dnnl_args, int arg) {
    for (int i = 0; i < static_cast<int>(dnnl_args.size()); ++i)
        if (dnnl_args[i].arg == arg) return i;
    return -1;
}

size_t get_arg_size(const std::vector<dnnl_exec_arg_t> &dnnl_args, int arg) {
    const int arg_idx = get_arg_idx(dnnl_args, arg);
    if (arg_idx < 0) return 0; // `arg` was not found, return empty size.

    const auto &mem = dnnl_args[arg_idx].memory;
    return dnnl_memory_desc_get_size(query_md(mem));
}
} // namespace cold_cache_utils

cold_cache_t::cold_cache_t(
        const std::vector<dnnl_exec_arg_t> &dnnl_args, dnnl_stream_t stream)
    : cold_cache_input_(cold_cache_input)
    , enabled_(use_cold_cache(dnnl_args))
    , n_buffers_top_limit_(
              is_gpu() ? gpu_n_buffers_top_limit_ : cpu_n_buffers_top_limit_) {

    // Note: there's an additional return from ctor below if it was identified
    // that no buffers are needed.
    if (!enabled_) return;

    static cpu_cache_args_t cpu_cache_args {};
    SAFE_V(get_cpu_cache_size(cpu_cache_args));
    const auto cpu_cache_capacity = cpu_cache_args.total_socket_size;
    // `3` potentially to cover both one and two socket scenarios.
    static const size_t cpu_cache_size_upper_bound = cpu_cache_capacity * 3;

    static size_t gpu_cache_capacity = 0;
    SAFE_V(get_gpu_cache_size(gpu_cache_capacity));
    static const size_t gpu_cache_size_upper_bound = gpu_cache_capacity * 4;

    const auto cache_capacity
            = is_gpu() ? gpu_cache_capacity : cpu_cache_capacity;
    const auto cache_size_upper_bound = is_gpu() ? gpu_cache_size_upper_bound
                                                 : cpu_cache_size_upper_bound;

    size_t full_args_size = 0;
    for (auto &e : dnnl_args) {
        if (!e.memory) continue;
        // Scratchpad is never meant to be cold, remove it from counting.
        if (e.arg == DNNL_ARG_SCRATCHPAD) continue;

        full_args_size += dnnl_memory_desc_get_size(query_md(e.memory));
    }
    size_t hot_args_size = full_args_size;
    size_t cold_args_size = 0;

    std::vector<int> cc_args; // future keys for cold_cache object.
    if (cold_cache_input_.cold_cache_mode_ == cold_cache_mode_t::wei) {
        cc_args = {DNNL_ARG_WEIGHTS};
        const auto wei_size
                = cold_cache_utils::get_arg_size(dnnl_args, DNNL_ARG_WEIGHTS);
        hot_args_size -= wei_size;
        cold_args_size += wei_size;
    } else if (cold_cache_input_.cold_cache_mode_ == cold_cache_mode_t::all) {
        cc_args.reserve(dnnl_args.size());
        for (auto &e : dnnl_args) {
            if (e.arg == DNNL_ARG_SCRATCHPAD) continue;

            cc_args.push_back(e.arg);
        }
        hot_args_size = 0;
        cold_args_size = full_args_size;
    } else if (cold_cache_input_.cold_cache_mode_
            == cold_cache_mode_t::custom) {
        const std::vector<int> user_args = {/* DNNL_ARG_WEIGHTS, ... */};
        cc_args = user_args;
        if (cc_args.empty()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: execution args for custom cold cache weren't "
                    "specified.");
            SAFE_V(FAIL);
        }
        for (int arg : cc_args) {
            const auto arg_size
                    = cold_cache_utils::get_arg_size(dnnl_args, arg);
            hot_args_size -= arg_size;
            cold_args_size += arg_size;
        }
    } else {
        assert(!"unknown cold cache mode!");
    }

    BENCHDNN_PRINT(3,
            "[COLD_CACHE]%s Size:%s; Limit:%s; Hot args:%s; Cold args:%s;\n",
            (is_gpu() ? "[GPU]" : "[CPU]"), smart_bytes(cache_capacity).c_str(),
            smart_bytes(cache_size_upper_bound).c_str(),
            smart_bytes(hot_args_size).c_str(),
            smart_bytes(cold_args_size).c_str());

    const size_t cold_mem_pool_size = cache_size_upper_bound > hot_args_size
            ? cache_size_upper_bound - hot_args_size
            : 0;

    size_t n_mem_pool_buffers = 0;
    // If `cold_args_size` are greater then allowed pool_size, it means there's
    // no sense in allocating any more buffers. Use original buffers only.
    if (cold_mem_pool_size > cold_args_size)
        n_mem_pool_buffers = div_up(cold_mem_pool_size, cold_args_size);

    n_buffers_ = MIN2(MAX2(n_mem_pool_buffers, n_buffers_bottom_limit_),
            n_buffers_top_limit_);
    override_n_buffers_ = n_mem_pool_buffers > n_buffers_top_limit_;

    BENCHDNN_PRINT(3,
            "[COLD_CACHE] n_buffer_limits: [%zu, %s]; n_mem_pool_buffers: "
            "%zu; n_buffers: %zu.\n",
            n_buffers_bottom_limit_,
            (n_buffers_top_limit_ == SIZE_MAX
                            ? "SIZE_MAX"
                            : std::to_string(n_buffers_top_limit_).c_str()),
            n_mem_pool_buffers, n_buffers_);
    if (cold_cache_input_.cold_tlb_) {
        BENCHDNN_PRINT(3, "[COLD_CACHE] tlb:enabled; size:%s;\n",
                smart_bytes(cold_cache_input_.cold_tlb_size_).c_str());
    }
    if (n_buffers_ <= 0) {
        // No buffers allocation needed, return to avoid scratching `cache_`
        // object. This allows to keep rest logic intact.
        return;
    }

    for (auto arg : cc_args) {
        const int idx = cold_cache_utils::get_arg_idx(dnnl_args, arg);
        if (idx < 0) {
            BENCHDNN_PRINT(0, "%s \'%d\' %s\n", "Error: execution argument",
                    idx, "requested for cold caching was not found!");
            SAFE_V(FAIL);
        }

        const auto &orig_mem = dnnl_args[idx].memory;
        // Empty memories don't get their cold cache entry.
        if (!orig_mem) continue;
        // Host scalars carry no buffer to thrash and can't be replicated.
        if (query_md_format_kind(query_md(orig_mem))
                == dnnl_format_kind_host_scalar)
            continue;

        auto &cc_entry = cache_[arg];
        cc_entry.resize(n_buffers_);
        auto orig_cc_mem_md = query_md(orig_mem);

        for (size_t i = 0; i < n_buffers_; i++) {
            // If there are multiple buffers, don't rely on reorders because it
            // takes a long time. Rely on a default initialization in this case.
            //
            // Note: despite `gpu_n_buffers_top_limit_` is taken as a limit, it
            // applies for CPU as well to avoid numerous reorders.
            const bool prefill = n_mem_pool_buffers > gpu_n_buffers_top_limit_;
            // Only buffers smaller than the large page get inflated by
            // it, so only those benefit from shrinking it; shrinking
            // larger buffers too adds run-to-run timing noise for no gain.
            const size_t alignment = dnnl_memory_desc_get_size(orig_cc_mem_md)
                            < dnnl_mem_large_page_size
                    ? dnnl_mem_page_size
                    : dnnl_mem_large_page_size;
            cc_entry[i] = dnn_mem_t(orig_cc_mem_md, get_test_engine(), prefill,
                    dnn_mem_t::handle_info_t::allocate(alignment));

            // Sparse memories require this call to replicate the exact original
            // data distribution because the data structure affects performance
            // in a direct way.
            if (cc_entry[i].format_kind() == dnnl_format_kind_sparse) {
                auto st = fill_random_real(
                        cc_entry[i], get_default_fill_cfg(), orig_mem);
                if (st != OK) {
                    BENCHDNN_PRINT(0,
                            "Error: filling for cold cache tensor %zu failed "
                            "(%s:%d)!\n",
                            i, __FILE__, __LINE__);
                    return;
                }
                if (cc_entry[i].is_mapped()) cc_entry[i].unmap();
                continue;
            }

            // Reorders are expensive. If there are multiple buffers to
            // fill, simply rely on default memory initialization.
            if (prefill) continue;

            if (cc_entry[i].is_mapped()) cc_entry[i].unmap();
            const auto &dst_memory = cc_entry[i].m_;
            benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> reorder_pdw;
            dnnl_primitive_desc_t reorder_pd {};
            dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                    &reorder_pd, orig_cc_mem_md, query_engine(orig_mem),
                    orig_cc_mem_md, query_engine(dst_memory), nullptr);
            if (status != dnnl_success) {
                BENCHDNN_PRINT(0,
                        "Error: cold-cache reorder failed for %s arg and "
                        "%zu buffer (out of %zu).\n",
                        data_kind2str(exec_arg2data_kind(arg)), i, n_buffers_);
                assert(status == dnnl_success);
                return;
            }
            reorder_pdw.reset(reorder_pd);

            benchdnn_dnnl_wrapper_t<dnnl_primitive_t> reorder_w;
            dnnl_primitive_t reorder {};
            status = dnnl_primitive_create(&reorder, reorder_pdw);
            assert(status == dnnl_success);
            if (status != dnnl_success) { return; }
            reorder_w.reset(reorder);

            std::vector<dnnl_exec_arg_t> dnnl_args;
            dnnl_args.resize(2);
            dnnl_args[0].arg = DNNL_ARG_FROM;
            dnnl_args[0].memory = orig_mem;
            dnnl_args[1].arg = DNNL_ARG_TO;
            dnnl_args[1].memory = dst_memory;

            status = dnnl_primitive_execute(
                    reorder_w, stream, (int)dnnl_args.size(), dnnl_args.data());
            assert(status == dnnl_success);
            if (status != dnnl_success) { return; }
            if (cc_entry[i].is_mapped()) cc_entry[i].unmap();
        }
    }

    // Refer to `gpu_n_buffers_top_limit_` comment.
    // Run reorder only if memory heuristic was overrided.
    if (override_n_buffers_) {
        // Exact cache size for src is needed to secure from potential
        // non-temporal dst stores or addresses collisions which would result
        // in part cache update.
        const size_t mem_size = cache_capacity;
        static constexpr size_t cache_line_size = 64;
        SAFE_V(thrash_reorder(mem_size, /* granularity = */ cache_line_size));
    }
}

cold_cache_t::~cold_cache_t() {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return;

    // Mapping memories after execution to have them destroyed gracefully.
    for (auto &e : cache_) {
        auto &cc_entry = e.second;
        for (size_t i = 0; i < cc_entry.size(); i++) {
            if (!cc_entry[i].is_mapped()) cc_entry[i].map();
        }
    }
}

cold_cache_t &cold_cache_t::operator=(cold_cache_t &&rhs) {
    if (&rhs == this) return *this;

    // Not expected to move a cold cache in the middle of the executions.
    assert(rhs.cc_counter_ == 0);

    enabled_ = rhs.enabled_;
    n_buffers_top_limit_ = rhs.n_buffers_top_limit_;
    n_buffers_bottom_limit_ = rhs.n_buffers_bottom_limit_;
    n_buffers_ = rhs.n_buffers_;
    override_n_buffers_ = rhs.override_n_buffers_;
    cold_cache_input_ = std::move(rhs.cold_cache_input_);
    cache_ = std::move(rhs.cache_);

    return *this;
}

// `mem_size` is the amount of memory in bytes for reorder.
// `granularity` defines the stride between elements to make an object of
//   `granularity` size to fit the specific cache. Mostly designed for
//   cache-line size of 64, or a page size of 4096.
int cold_cache_t::thrash_reorder(size_t mem_size, size_t granularity) const {
    const auto &engine = get_test_engine();

    const dnnl_dim_t nelems
            = static_cast<dnnl_dim_t>(div_up(mem_size, sizeof(float)));
    const dnnl_dim_t stride = granularity / sizeof(float);

    // Reduce the number of element by the the stride to keep the memory size
    // as requested.
    const dnnl_dims_t dims {div_up(nelems, stride)};
    const dnnl_dims_t strides {stride};

    // Since it's a thrash reorder, just need memory pages to go through, no
    // need to prefill data.
    dnn_mem_t src_m(1, dims, dnnl_f32, strides, engine, /* prefill = */ false);
    dnn_mem_t dst_m(1, dims, dnnl_f32, strides, engine, /* prefill = */ false);

    dnnl_primitive_desc_t r_pd {};
    dnnl_primitive_attr_t attr {};
    DNN_SAFE(dnnl_reorder_primitive_desc_create(
                     &r_pd, src_m.md_, engine, dst_m.md_, engine, attr),
            WARN);
    auto r_pd_w = make_benchdnn_dnnl_wrapper(r_pd);

    dnnl_primitive_t prim {};
    DNN_SAFE(dnnl_primitive_create(&prim, r_pd), WARN);
    auto prim_w = make_benchdnn_dnnl_wrapper(prim);

    args_t args;
    args.set(DNNL_ARG_FROM, src_m);
    args.set(DNNL_ARG_TO, dst_m);
    SAFE(execute_and_wait(prim, args), WARN);

    return OK;
}

bool cold_cache_t::update_dnnl_args(std::vector<dnnl_exec_arg_t> &dnnl_args) {
    if (!enabled_) return true;
    if (should_stop()) return false;

    for (const auto &cc_entry : cache_) {
        const int arg = cc_entry.first;
        const int dnnl_args_idx = cold_cache_utils::get_arg_idx(dnnl_args, arg);
        if (dnnl_args_idx < 0) return false;

        const auto &e = cc_entry.second;
        // Assumption that cache entries of the same size.
        if (cc_counter_ >= e.size()) cc_counter_ = 0;
        dnnl_args[dnnl_args_idx].memory = e[cc_counter_].m_;
    }

    // Need to sweep all pages from TLB. Must be done every time the stack of
    // memories was used for measurements as they hit TLB after being used.
    if (cold_cache_input_.cold_tlb_ && cc_counter_ == 0) {
        // A full size requested will be used for source AND destination in the
        // underlying reorder. Though it's double the size, addresses of the
        // buffers may coincide and TLB will be thrashed just by the `size`,
        // not the double size.
        //
        // In theory, a single buffer split in half should work, but currently
        // benchdnn abstractions are not ready to work with external pointers of
        // non-host nature.
        const size_t mem_size = cold_cache_input_.cold_tlb_size_;
        static constexpr size_t page_size = 4096; // 4 KB.
        auto st = thrash_reorder(mem_size, /* granularity = */ page_size);
        if (st != OK) return false;
    }

    // Update counter outside of the loop to make **all** arguments use same
    // order element from the cache.
    cc_counter_++;

    return true;
}

bool cold_cache_t::should_stop() const {
    return override_n_buffers_ && cc_counter_ == n_buffers_;
}

bool cold_cache_t::use_cold_cache(
        const std::vector<dnnl_exec_arg_t> &dnnl_args) const {
    // Cold cache is not supported in single-run/simulation mode.
    if (has_bench_mode_bit(mode_bit_t::sim)) return false;

    const bool cc_wei
            = cold_cache_input_.cold_cache_mode_ == cold_cache_mode_t::wei;
    const bool cc_all
            = cold_cache_input_.cold_cache_mode_ == cold_cache_mode_t::all;
    const bool cc_custom
            = cold_cache_input_.cold_cache_mode_ == cold_cache_mode_t::custom;
    const bool has_weights
            = cold_cache_utils::get_arg_idx(dnnl_args, DNNL_ARG_WEIGHTS) >= 0;
    static int warning_printed = 0;
    if (cc_wei && !has_weights && !warning_printed) {
        BENCHDNN_PRINT(0, "%s\n",
                "Warning: cold cache for weights was requested but weights "
                "were not identified in execution arguments. Cold cache will "
                "not be enabled.");
        warning_printed = 1;
    }

    return (cc_wei && has_weights) || cc_all || cc_custom;
}

std::ostream &operator<<(std::ostream &s, cold_cache_mode_t cold_cache_mode) {
    if (cold_cache_mode == cold_cache_mode_t::none)
        s << "";
    else if (cold_cache_mode == cold_cache_mode_t::wei)
        s << "wei";
    else if (cold_cache_mode == cold_cache_mode_t::all)
        s << "all";
    else if (cold_cache_mode == cold_cache_mode_t::custom)
        s << "custom";
    else { assert(!"unsupported cold cache mode"); }
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const cold_cache_input_t &cold_cache_input) {
    s << cold_cache_input.cold_cache_mode_;
    if (cold_cache_input.cold_tlb_) {
        s << "+tlb";
        if (cold_cache_input.cold_tlb_size_str_
                != default_cold_cache_input().cold_tlb_size_str_) {
            s << ":" << cold_cache_input.cold_tlb_size_str_;
        }
    }
    return s;
}
