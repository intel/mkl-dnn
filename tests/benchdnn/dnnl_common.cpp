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

#include <algorithm> // for std::reverse and std::copy
#include <functional> // for std::bind and std::placeholders
#include <future> // for std::promise and std::future
#include <list>
#include <numeric>
#include <string> // for std::string
#include <utility> // for std::pair
#include <vector> // for std::vector

#include <assert.h>

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "oneapi/dnnl/dnnl_ze.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif

// Uses only publicly available types.
#include "src/common/primitive_cache_test_api.hpp"

#include "cpu/platform.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "utils/check.hpp"
#include "utils/cold_cache.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/execution_mode.hpp"
#include "utils/fill.hpp"
#include "utils/parallel.hpp"
#include "utils/stream_kind.hpp"

#include "tests/test_thread_decl.hpp"

namespace {
// `fetch_impl` is responsible to provide a valid `pd` under certain conditions:
// 1. Either valid `pd` or `pd_it` were provided.
// 2a. It's a service primitive (fwd-for-bwd or cpu-for-gpu or
//     simple-prims-of-complex-prim).
// 2b. It's a tested primitive and not all implementations hit skip-impl option
//     values.
//
// Note: `res` can be empty when fetching impl for prim_ref support.
int fetch_impl(benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> &pdw,
        init_pd_args_t &init_pd_args, const impl_filter_t &impl_filter,
        res_t *res, bool is_service_prim) {
    if (!init_pd_args.pd) return FAIL;

    // Wrapper is expected to come empty.
    assert(!pdw);

    pdw.reset(init_pd_args.pd);

    // Service primitive is not supposed to utilize further logic.
    if (is_service_prim) return OK;

    while (true) {
        const auto impl_name = query_impl_info(pdw);
        if (!need_next_impl(impl_name, impl_filter)) return OK;

        BENCHDNN_PRINT(6, "[IMPL_FILTER] Implementation skipped: %s\n",
                impl_name.c_str());

        // Iterator is not supported, further logic is not applicable.
        if (!init_pd_args.is_iterator_supported) {
            if (res) res->state = SKIPPED;
            if (res) res->reason = reason_t::skip_impl_hit;
            return OK;
        }

        auto status = dnnl_primitive_desc_next_impl(pdw);
        if (status == dnnl_last_impl_reached) {
            BENCHDNN_PRINT(2, "%s\n",
                    "[IMPL_FILTER] All implementations were skipped!");
            if (res) res->state = SKIPPED;
            if (res) res->reason = reason_t::skip_impl_hit;
            pdw.reset(nullptr);
            return OK;
        } else if (status == dnnl_success) {
            continue;
        } else {
            BENCHDNN_PRINT(0, "%s\n",
                    "[IMPL_FILTER] Unexpected status from pd iterator.");
            return FAIL;
        }
    }

    // Unreached fail status.
    return FAIL;
}

int check_pd_w_and_wo_attr(dnnl_engine_t engine,
        const init_pd_func_t &init_pd_func, const base_prb_t *base_prb,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint) {
    if (!attr_same_pd_check || base_prb->attr.is_def()) return OK;

    if (base_prb->attr.post_ops.convolution_index() != -1) return OK;

    // Check that adding attributes doesn't cause a fall back to another impl.
    auto *prb_mutable = const_cast<base_prb_t *>(base_prb);
    auto old_attr = prb_mutable->attr;
    prb_mutable->attr = attr_t();
    init_pd_args_t init_pd_args_without_attr(
            res, engine, prb_mutable, dir, hint, /* src_md = */ nullptr);
    DNN_SAFE(init_pd_func(init_pd_args_without_attr), WARN);
    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw(
            init_pd_args_without_attr.pd);
    prb_mutable->attr = old_attr;
    SAFE(check_same_pd(pdw, res), WARN);
    return OK;
}

} // namespace

extern "C" dnnl_status_t dnnl_impl_notify_profiling_complete(
        dnnl_stream_t stream);

int check_pd_cache(const_dnnl_primitive_desc_t pd, res_t *res) {
    // Disable this check for threadpool. A threadpool is always defined in
    // validation infrastructure but creates primitives in a separate
    // environment that doesn't have threadpool enabled. Thus, it utilizes a
    // specified number of threads in a testing environment that is different
    // from the number of cores per socket (internal logic for primitive
    // creation). This will cause this check to fail as the number of threads
    // is used in the primitive cache key.
#if !defined(DNNL_DISABLE_PRIMITIVE_CACHE) \
        && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), FAIL);
    if (capacity && !dnnl_test_is_pd_in_cache(pd)) {
        res->state = FAILED;
        BENCHDNN_PRINT(0, "%s\n",
                "Error: primitive descriptor is expected to be fetched from "
                "the primitive cache");
        return FAIL;
    }
#endif
    return OK;
}

int check_primitive_cache(dnnl_primitive_t p, res_t *res) {
    // See the comment in `check_pd_cache`.
#if !defined(DNNL_DISABLE_PRIMITIVE_CACHE) \
        && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), WARN);
    if (capacity && !dnnl_test_is_primitive_in_cache(p)) {
        res->state = FAILED;
        BENCHDNN_PRINT(0, "%s\n",
                "Error: primitive is expected to be fetched from the primitive "
                "cache");
        return FAIL;
    }
#endif
    return OK;
}

int get_cache_blob_id(
        std::vector<uint8_t> &cache_blob_id, const_dnnl_primitive_desc_t pd) {
    dnnl_dim_t count;
    const uint8_t *c_id;
    DNN_SAFE(dnnl_primitive_desc_query(
                     pd, dnnl_query_cache_blob_id_size_s64, 0, (void *)&count),
            WARN);
    DNN_SAFE(dnnl_primitive_desc_query(
                     pd, dnnl_query_cache_blob_id, 0, (void **)&c_id),
            WARN);
    cache_blob_id = {c_id, c_id + count};
    return OK;
}

int get_cache_blob(std::vector<uint8_t> &cache_blob, dnnl_primitive_t prim) {
    size_t size = 0;
    DNN_SAFE(dnnl_primitive_get_cache_blob(prim, &size, nullptr), WARN);

    cache_blob.resize(size);
    DNN_SAFE(dnnl_primitive_get_cache_blob(prim, &size, cache_blob.data()),
            WARN);
    return OK;
}

struct lru_cache_t {
    lru_cache_t(size_t capacity) : capacity_(capacity) {}
    ~lru_cache_t() = default;

    const std::vector<uint8_t> &get(const std::vector<uint8_t> &key) {
        auto it = cache_mapper_.find(key);
        if (it == cache_mapper_.end()) { return dummy_; }
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return cache_list_.front().value_;
    }

    void add(const std::vector<uint8_t> &key,
            const std::vector<uint8_t> &value) {
        assert(!cache_mapper_.count(key));
        if (cache_mapper_.size() >= capacity_) {
            cache_mapper_.erase(cache_list_.back().key_);
            cache_list_.pop_back();
        }
        cache_list_.emplace_front(key, value);
        cache_mapper_.insert({key, cache_list_.begin()});
    }

private:
    lru_cache_t(const lru_cache_t &other) = delete;
    lru_cache_t(lru_cache_t &&other) = delete;
    lru_cache_t &operator=(const lru_cache_t &other) = delete;
    lru_cache_t &operator=(lru_cache_t &&other) = delete;

    struct entry_t {
        entry_t(const std::vector<uint8_t> &key,
                const std::vector<uint8_t> &value)
            : key_(key), value_(value) {}
        std::vector<uint8_t> key_;
        std::vector<uint8_t> value_;
    };

    size_t capacity_;
    std::list<entry_t> cache_list_;
    std::map<std::vector<uint8_t>, std::list<entry_t>::iterator> cache_mapper_;

    const std::vector<uint8_t> dummy_;
};

lru_cache_t &get_test_cache() {
    static lru_cache_t cache(1024);
    return cache;
}

int test_persistent_cache_api(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim, res_t *res) {

    auto pd = query_pd(prim);

    // 0. check memory descriptor serialization API
    // Limiting to weights md should have good md kind coverage and
    // limit overhead
    {
        dnnl_memory_desc_t new_md;
        size_t sz = 0;
        auto wei_md = query_md(pd, dnnl_query_weights_md);
        dnnl_memory_desc_get_blob(nullptr, &sz, wei_md);

        std::vector<uint8_t> md_blob(sz);
        dnnl_memory_desc_get_blob(md_blob.data(), &sz, wei_md);
        dnnl_memory_desc_create_with_blob(&new_md, md_blob.data());
        auto mew_mdw = make_benchdnn_dnnl_wrapper(new_md);

        if (dnnl_memory_desc_equal(wei_md, new_md) == 0) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }
    }

    // Start testing persistent cache API.
    if (!is_gpu() || !(is_opencl_engine() || is_ze_engine())) { return OK; }

    // 1. Disable primitive cache to make sure that the next primitive will
    // be created from the cache blob and not fetched from the primitive cache.
    const auto old_capacity
            = dnnl_test_set_primitive_cache_capacity_without_clearing(0);
    // 2. Get cache blob ID to use it as a key for the `test_cache`.
    std::vector<uint8_t> cache_blob_id;
    auto st = get_cache_blob_id(cache_blob_id, pd);
    if (st != OK) {
        res->state = FAILED;
        SAFE(FAIL, WARN);
    }
    // 3. Check if a cache blob for the obtained cache blob ID is present in the
    //    `test_cache`.
    //    a) If the cache blob is found the primitive is created from it.
    //    b) If the cache blob is not found then get it from the previously
    //       created primitive, store it in the cache and create the primitive
    //       from it.
    dnnl_primitive_t p {};
    auto &cache = get_test_cache();
    auto cache_value = cache.get(cache_blob_id);
    if (!cache_value.empty()) {
        const size_t size = cache_value.size();
        const uint8_t *cache_blob = cache_value.data();
        auto dnnl_st = dnnl_primitive_create_from_cache_blob(
                &p, pd, size, cache_blob);
        if (dnnl_st != dnnl_success) {
            res->state = FAILED;
            DNN_SAFE(dnnl_st, WARN);
        }
    } else {
        std::vector<uint8_t> cache_blob;
        st = get_cache_blob(cache_blob, prim);
        if (st != OK) {
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }

        // The cross-engine and direct copy reorders are special primitives that
        // may contain no kernels therefore the cache blob will always be empty,
        // which is the correct behavior.
        if (cache_blob.empty()) {
            dnnl_test_set_primitive_cache_capacity_without_clearing(
                    old_capacity);
            if (query_prim_kind(pd) == dnnl_reorder
                    && (res->impl_name.find("cross_engine") != std::string::npos
                            || res->impl_name.find("direct_copy")
                                    != std::string::npos))
                return OK;

            // If the operation is trivial, there may be no kernel in the cache.
            const auto dst_md = query_md(pd, DNNL_ARG_DST);
            for (int i = 0; i < query_md_ndims(dst_md); ++i)
                if (query_md_padded_dims(dst_md)[i] == 0) return OK;

            BENCHDNN_PRINT(
                    0, "error: %s\n", "cache blob is not expected to be empty");
            res->state = FAILED;
            SAFE(FAIL, WARN);
        }

        auto dnnl_st = dnnl_primitive_create_from_cache_blob(
                &p, pd, cache_blob.size(), cache_blob.data());
        if (dnnl_st != dnnl_success) {
            res->state = FAILED;
            DNN_SAFE(dnnl_st, WARN);
        }
        cache.add(cache_blob_id, cache_blob);
    }
    prim.reset(p);

    // 4. Restore the original primitive cache capacity to make it functional.
    dnnl_test_set_primitive_cache_capacity_without_clearing(old_capacity);

    return OK;
}

// CPU ISA specific hints : none by default
isa_hints_t hints {isa_hints_t::none};

memory_kind_ext_t memory_kind {default_memory_kind};

int default_num_streams = 1;
int num_streams = default_num_streams;

void init_isa_settings() {
    if (hints.get() == isa_hints_t::no_hints) {
        DNN_SAFE_V(dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints));
    } else if (hints.get() == isa_hints_t::prefer_ymm) {
        auto status = dnnl_set_cpu_isa_hints(dnnl_cpu_isa_prefer_ymm);
        // Unimplemented is a valid status for non-x64
        if (status == dnnl_success || status == dnnl_unimplemented) return;
        DNN_SAFE_V(status);
    } else {
        // Do nothing when hints == none
        assert(hints.get() == isa_hints_t::none);
    }
}

// This ctor is responsible to provide proper pointers to memory objects for
// correspondent arguments. It is important for in-place cases when a single
// object should be used as SRC and DST.
// `mem_map` object is an owner of memory objects and can't use the same object
// for SRC and DST while `args` is a proxy with pointers to memories and may
// easily change what to pick for a specific arg.
args_t::args_t(const dnn_mem_map_t &mem_map) {
    // Tells which arg's memory object serves `arg` when the latter was dropped
    // from the map, or `DNNL_ARG_UNDEF` when `arg` isn't one of those.
    const auto inplace_donor_arg = [](int arg) {
        if (arg == DNNL_ARG_DST) return DNNL_ARG_SRC;
        if (arg == DNNL_ARG_DIFF_SRC) return DNNL_ARG_DIFF_DST;
        // An in-place binary post-op reads the destination as its right-hand
        // side. Note that post-op args start past `..._POST_OP_BASE` while the
        // remainder tells which of the post-op inputs the arg stands for.
        if (arg >= DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE
                && arg % DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE == DNNL_ARG_SRC_1)
            return DNNL_ARG_DST;
        return DNNL_ARG_UNDEF;
    };

    for (const auto &map_entry : mem_map) {
        const dnn_mem_t *mem_ptr = &map_entry.second;
        // An empty memory means the arg was dropped in favor of another one.
        // Follow the chain of substitutions, as the arg serving a dropped one
        // may have been dropped itself.
        for (int arg = map_entry.first; !*mem_ptr;) {
            arg = inplace_donor_arg(arg);
            // Not an arg subject to the substitution, keep it as it is.
            if (arg == DNNL_ARG_UNDEF) break;

            auto it = mem_map.find(arg);
            // May happen that source argument is different.
            if (it == mem_map.end() && arg == DNNL_ARG_SRC)
                it = mem_map.find(DNNL_ARG_MULTIPLE_SRC);
            if (it == mem_map.end()) {
                BENCHDNN_PRINT(0, "%s\n", "Inplace substitution failed.");
                SAFE_V(FAIL);
            }

            mem_ptr = &((*it).second); // Update reference with in-place memory.
            // The donor either has a memory of its own or gets substituted in
            // turn; anything else leaves the arg with nothing to point at.
            assert(*mem_ptr || inplace_donor_arg(arg) != DNNL_ARG_UNDEF);
        }

        args_.emplace_back(map_entry.first, mem_ptr);
    }
}

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.emplace_back(arg, &mem);
    return *this;
}

const dnn_mem_t &args_t::find(int arg) const {
    static dnn_mem_t empty_stub;
    for (const auto &e : args_) {
        if (e.first == arg) return *(e.second);
    }
    return empty_stub;
}

void args_t::replace(int arg, const dnn_mem_t *mem) {
    for (auto &e : args_) {
        if (e.first == arg) {
            e.second = mem;
            break;
        }
    }
}

// Unmap before passing the memory to execute
void execute_unmap_args(
        const args_t &args, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    dnnl_args.resize(args.size());
    for (int i = 0; i < args.size(); ++i) {
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }
}

// Map the memory back after execute
void execute_map_args(const args_t &args) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return;

    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
}

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res) {
    stream_t stream(engine);
    std::vector<dnnl_exec_arg_t> dnnl_args;

    TIME_EXECUTE(execute_unmap_args(args, dnnl_args));

    dnnl_status_t status = dnnl_runtime_error;
    if (use_sycl_graph_exec(engine)) {
        std::function<void()> record_fn
                = std::bind(exec_func, std::ref(stream), std::cref(dnnl_args));
        // TIME_EXECUTE is done inside this call.
        int st = execute_in_graph_mode(stream, record_fn, res);
        // Update status only on success.
        if (st == OK) status = dnnl_success;
    } else {
        stream_staller_t staller(stream);
        TIME_EXECUTE(status = exec_func(stream, dnnl_args));
        staller.release();

        TIME_EXECUTE(DNN_SAFE(dnnl_stream_wait(stream), CRIT));
    }
    if (res) res->state = EXECUTED;

    TIME_EXECUTE(execute_map_args(args));
    if (status != dnnl_success) {
        if (res) res->state = FAILED;
        return FAIL;
    }

    return OK;
}

dnnl_status_t primitive_executor(dnnl_primitive_t prim,
        const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {
    return dnnl_primitive_execute(
            prim, stream, (int)dnnl_args.size(), dnnl_args.data());
}

int execute_and_wait(dnnl_primitive_t prim, const args_t &args, res_t *res) {
    perf_function_t exec_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);
    auto pd = query_pd(prim);
    auto engine = query_engine(pd);
    return execute_and_wait(exec_func, engine, args, res);
}

int run_execution(dnnl_primitive_t prim, const args_t &args, res_t *res) {
    if (!has_bench_mode_bit(mode_bit_t::exec)
            || has_bench_mode_bit(mode_bit_t::perf))
        return OK;
    return execute_and_wait(prim, args, res);
}

int run_execution(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res) {
    if (!has_bench_mode_bit(mode_bit_t::exec)
            || has_bench_mode_bit(mode_bit_t::perf))
        return OK;
    return execute_and_wait(exec_func, engine, args, res);
}

void reset_gpu_profiling(dnnl_stream_t stream) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    DNN_SAFE_V(dnnl_reset_profiling(stream));
#endif
}

int get_gpu_profiling_info(dnnl_stream_t stream, std::vector<uint64_t> &nsecs,
        std::vector<uint64_t> &cycles, int expected_num_entries) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    dnnl_profiling_data_kind_t undef_kind {};
    dnnl_profiling_data_kind_t time_kind {};

    // This is an internal data kind.
    dnnl_profiling_data_kind_t cycles_kind
            = dnnl::impl::profiling_data_kind::cycles;
#ifndef DNNL_EXPERIMENTAL_PROFILING
    undef_kind = 0;
    time_kind = 1;
#else
    undef_kind = dnnl_profiling_data_kind_undef;
    time_kind = dnnl_profiling_data_kind_time;
#endif

    int num_entries = 0;
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, undef_kind, &num_entries, nullptr),
            CRIT);
    if (expected_num_entries != -1 && num_entries != expected_num_entries) {
        BENCHDNN_PRINT(0,
                "ERROR: profiling entries mismatch, expected: %d entries but "
                "got %d entries\n",
                expected_num_entries, num_entries);
        return FAIL;
    }
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, time_kind, &num_entries, nsecs.data()),
            CRIT);
    nsecs.resize(num_entries);
    cycles.resize(num_entries);
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, time_kind, &num_entries, nsecs.data()),
            CRIT);
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, cycles_kind, &num_entries, cycles.data()),
            CRIT);
#endif
    return OK;
}

void notify_gpu_profiling_complete(dnnl_stream_t stream) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    DNN_SAFE_V(dnnl_impl_notify_profiling_complete(stream));
#endif
}

void finalize() {
    finalize_tbb();

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    // See: DUMMY_PARALLEL.
    if (is_cpu()) {
        auto tp = dnnl::testing::get_threadpool();
        tp->parallel_for(1, [](int, int) {});
    }
#endif
}

int update_timer_with_profiling_info(timer::timer_t &t, bool use_profiling,
        const std::vector<stream_t> &v_stream, int execute_count) {
    if (!use_profiling) {
        t.stamp(execute_count * num_streams);
        return OK;
    }

    std::vector<std::vector<uint64_t>> v_nsecs(num_streams);
    std::vector<std::vector<uint64_t>> v_cycles(num_streams);
    for (size_t j = 0; j < v_stream.size(); j++) {
        SAFE(get_gpu_profiling_info(
                     v_stream[j], v_nsecs[j], v_cycles[j], execute_count),
                CRIT);
        reset_gpu_profiling(v_stream[j]);

        // Profiling should have information to report, otherwise, stop.
        if (v_nsecs[j].empty()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "WARNING: no counters were found during profiling.");
            return FAIL;
        }
    }

    for_(size_t j = 0; j < v_stream.size(); j++)
    for (size_t i = 0; i < v_nsecs[j].size(); i++) {
        t.stop(1, (int64_t)v_cycles[j][i], v_nsecs[j][i] / 1e6);
    }

    return OK;
}

inline int measure_perf_individual(timer::timer_t &t, dnnl_stream_t stream,
        perf_function_t &perf_func, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    if (!has_bench_mode_bit(mode_bit_t::sim)) {
        DNN_SAFE(perf_func(stream, dnnl_args), WARN);
        DNN_SAFE(dnnl_stream_wait(stream), CRIT);
    }

    cold_cache_t cold_cache(dnnl_args, stream);

    t.reset();
    while (true) {
        if (!cold_cache.update_dnnl_args(dnnl_args)) break;
        t.start();
        DNN_SAFE(perf_func(stream, dnnl_args), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(timer::timer_t &t,
        const std::vector<stream_t> &v_stream, perf_function_t &perf_func,
        std::vector<std::vector<dnnl_exec_arg_t>> &dnnl_args) {
    std::vector<cold_cache_t> cold_cache(num_streams);

    // Nvidia/AMD don't support profiling.
    const bool use_profiling = is_gpu() && !is_nvidia_gpu() && !is_amd_gpu();

    // Warm-up run, this is not measured due to possibility the associated
    // kernel has not been built and skews the results.
    if (!has_bench_mode_bit(mode_bit_t::sim)) {
        for (size_t j = 0; j < v_stream.size(); j++) {
            DNN_SAFE(perf_func(v_stream[j], dnnl_args[j]), WARN);
            DNN_SAFE(dnnl_stream_wait(v_stream[j]), CRIT);
        }
    }

    for (size_t j = 0; j < v_stream.size(); j++) {
        cold_cache[j] = cold_cache_t(dnnl_args[j], v_stream[j]);
        if (use_profiling) reset_gpu_profiling(v_stream[j]);
    }

    int num_submissions = 0;
    if (fix_times_per_prb) {
        // If user specified the number of runs, just use it.
        num_submissions = fix_times_per_prb;
    } else {
        // Otherwise, use an estimate run, which is not a part of final time
        // collection. It's used to calculate the number of submissions to run
        // before synchronization. Done on a single stream.
        t.reset();
        DNN_SAFE(perf_func(v_stream[0], dnnl_args[0]), WARN);
        DNN_SAFE(dnnl_stream_wait(v_stream[0]), CRIT);
        SAFE(update_timer_with_profiling_info(t, use_profiling, v_stream, 1),
                WARN);

        double ms_min = t.total_ms();
        if (ms_min == 0.0) SAFE_V(FAIL);
        num_submissions
                = MAX2(min_times_per_prb, (int)(max_ms_per_prb / ms_min));
    }

    BENCHDNN_PRINT(
            4, "%s%d%s\n", "[PERF]: submissions: ", num_submissions, ";");
    // Measuring loop. A single synchronization point, called once, the number
    // of submissions determined above based on n_times or plain time criterion.
    t.reset();
    // Keep a separate variable due to a `break` inside the loop.
    int execute_count = 0;
    // Keep inner loop over streams for better submission overlapping.
    for_(int i = 0; i < num_submissions; i++)
    for (size_t j = 0; j < v_stream.size(); j++) {
        if (!cold_cache[j].update_dnnl_args(dnnl_args[j])) break;
        DNN_SAFE(perf_func(v_stream[j], dnnl_args[j]), WARN);
        execute_count++;
    }

    for (size_t j = 0; j < v_stream.size(); j++) {
        DNN_SAFE(dnnl_stream_wait(v_stream[j]), CRIT);
    }

    SAFE(update_timer_with_profiling_info(
                 t, use_profiling, v_stream, execute_count),
            WARN);

    if (use_profiling) {
        for (size_t j = 0; j < v_stream.size(); j++) {
            notify_gpu_profiling_complete(v_stream[j]);
        }

        t.filter_collection();
    }

    return OK;
}

int measure_perf(const thr_ctx_t &ctx, res_t *res, perf_function_t &perf_func,
        args_t &args) {
    if (!has_bench_mode_bit(mode_bit_t::perf)) return OK;

    const auto &engine = get_test_engine();
    std::vector<stream_t> v_stream(num_streams);
    for (int i = 0; i < num_streams; i++)
        v_stream[i] = stream_t(engine, ctx.get_interop_obj());

    std::vector<std::vector<dnnl_exec_arg_t>> dnnl_args(num_streams);
    std::vector<dnn_mem_map_t> mem_map(num_streams);
    std::vector<args_t> v_args(num_streams);
    v_args[0] = args;
    for (int j = 1; j < num_streams; j++) {
        for (int i = 0; i < args.size(); i++) {
            int arg = args.arg(i);
            const auto &m = args.dnn_mem(i);
            // Memory must be filled with some data for meaningful performance
            // numbers.
            mem_map[j].emplace(
                    arg, dnn_mem_t(m.md_, engine, /* prefill = */ true));
            SAFE(mem_map[j].at(arg).reorder(m, res), WARN);
        }
        v_args[j] = args_t(mem_map[j]);
        execute_unmap_args(v_args[j], dnnl_args[j]);
    }
    execute_unmap_args(args, dnnl_args[0]);

    auto &t = res->timer_map.perf_timer();
    // For non-DPCPP CPU: measure individual iterations.
    // For DPCPP CPU and GPU: measure iterations in batches to hide driver
    // overhead. DPCPP CPU follows the model of GPU, thus, handled similar.
    // For async threadpool CPU: use aggregate as well, similar to DPCPP CPU.
    int ret = OK;
    if (is_async(engine)) {
        std::function<int()> measure_perf_aggregate_fn = std::bind(
                measure_perf_aggregate, std::ref(t), std::cref(v_stream),
                std::ref(perf_func), std::ref(dnnl_args));
        ret = execute_in_thr_ctx(ctx, measure_perf_aggregate_fn);
    } else {
        std::function<int()> measure_perf_individual_fn = std::bind(
                measure_perf_individual, std::ref(t), std::ref(v_stream[0]),
                std::ref(perf_func), std::ref(dnnl_args[0]));
        ret = execute_in_thr_ctx(ctx, measure_perf_individual_fn);
    }

    res->state = (ret == OK ? EXECUTED : FAILED);
    execute_map_args(args);
    for (int j = 1; j < num_streams; j++) {
        execute_map_args(v_args[j]);
    }

    return ret;
}

int measure_perf(
        const thr_ctx_t &ctx, res_t *res, dnnl_primitive_t prim, args_t &args) {
    perf_function_t perf_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);

    return measure_perf(ctx, res, perf_func, args);
}

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off, int64_t group_id) {
    if (v_po_masks.empty()) return std::vector<float>();

    std::vector<float> v_vals(v_po_masks.size());

    for (size_t d = 0; d < v_po_masks.size(); ++d) {
        // For grouped memory, mask == 0 is overloaded to mean per-group:
        // a single value per concatenated group, indexed by group_id
        //
        // Note, that group_id is 0 for non-grouped use
        const auto po_offset = v_po_masks[d].second == 0
                ? group_id
                : dst_m.get_idx(dst_off, v_po_masks[d].second);
        const float val
                = args.find(v_po_masks[d].first).get_f32_elem(po_offset);
        v_vals[d] = val;
    }
    return v_vals;
}

bool check_md_consistency_with_tag(
        const_dnnl_memory_desc_t md, const std::string &tag) {
    // `tag` is either `tag::abx` or the problem tag already used in `init_pd`
    // for a tensor of the same rank, so a failure is not expected.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md_new_tag;
    DNN_SAFE_V(dnn_mem_t::init_md(md_new_tag, query_md_ndims(md),
            query_md_dims(md), query_md_data_type(md), tag));
    return dnnl_memory_desc_equal(md_new_tag, md);
}

// Note: CPU support can be simplified if prop_kind is passed, however, this
// prop kind must come from prb directly, not queried, as if not implemented,
// pd will come empty and there's nothing to query. prop_kind is not available
// in prb as of now and it will require dir_t -> prop_kind_t replacement
// refactor across the stack.
void skip_unimplemented_data_type(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res) {
    const bool has_f64_support = is_f64_supported();
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    // bf16 is supported on AVX512-CORE+
    const bool has_bf16_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_bf16)
                    && IMPLICATION(!(dir & FLAG_INF),
                            has_training_support(dnnl_bf16)));
    const bool has_f16_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f16)
                    && IMPLICATION(
                            !(dir & FLAG_INF), has_training_support(dnnl_f16)));
    const bool has_e8m0_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_e8m0));
    const bool has_f4_e2m1_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_f4_e2m1));
    const bool has_f8_e5m2_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f8_e5m2)
                    && IMPLICATION(!(dir & FLAG_INF),
                            has_training_support(dnnl_f8_e5m2)));
    const bool has_f8_e4m3_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f8_e4m3)
                    && IMPLICATION(!(dir & FLAG_INF),
                            has_training_support(dnnl_f8_e4m3)));
#else
    const bool has_bf16_support = is_gpu();
    // f16 is supported on GPU for inference only.
    const bool has_f16_support = is_gpu() && (dir & FLAG_FWD);
    const bool has_f4_e2m1_support = is_gpu();
    const bool has_e8m0_support = is_gpu();
    const bool has_f8_e5m2_support = is_gpu();
    const bool has_f8_e4m3_support = is_gpu();
#endif

    for (const auto &i_dt : v_dt) {
        bool need_skip = false;
        switch (i_dt) {
            case dnnl_bf16: need_skip = !has_bf16_support; break;
            case dnnl_f16: need_skip = !has_f16_support; break;
            case dnnl_f64: need_skip = !has_f64_support; break;
            case dnnl_e8m0: need_skip = !has_e8m0_support; break;
            case dnnl_f4_e2m1: need_skip = !has_f4_e2m1_support; break;
            case dnnl_f8_e5m2: need_skip = !has_f8_e5m2_support; break;
            case dnnl_f8_e4m3: need_skip = !has_f8_e4m3_support; break;
            default: break;
        }
        if (need_skip) {
            BENCHDNN_PRINTF(2, "%s%s%s", "[SKIP]: Data type \'", dt2str(i_dt),
                    "\' is not supported on this platform.");
            res->state = SKIPPED;
            res->reason = reason_t::skip_data_type;
            return;
        }
    }
}

void skip_unimplemented_sum_po(const attr_t &attr, res_t *res,
        dnnl_primitive_kind_t pkind, dnnl_data_type_t src_dt,
        dnnl_data_type_t dst_dt) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return;

    const int first_sum_idx = po.find(attr_t::post_ops_t::SUM);
    if (first_sum_idx == -1) return;

    const auto sum_dt = po.entry[first_sum_idx].sum.dt;

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (e.is_sum_kind()) {
            // API requirements
            if (e.sum.zero_point != 0) {
                // Sum with zero-point is only supported for int8
                if (!is_integral_dt(src_dt)) {
                    res->state = SKIPPED;
                    res->reason = reason_t::skip_not_supported;
                    return;
                } else {
                    // Only quantized sum operand can have zero point
                    const dnnl_data_type_t e_sum_dt
                            = e.sum.dt == dnnl_data_type_undef ? dst_dt
                                                               : e.sum.dt;
                    if (!is_integral_dt(e_sum_dt)) {
                        res->state = SKIPPED;
                        res->reason = reason_t::skip_not_supported;
                        return;
                    }
                }
            }

            // Sum with zero-point is not supported on GPU
            if (is_gpu() && e.sum.zero_point != 0) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                break;
            }
            // Each sum must have same data on CPU
            if (is_cpu() && e.sum.dt != sum_dt) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                break;
            }
            // Sum must have data type with the same size like dst on both
            if (dst_dt != dnnl_data_type_undef && sum_dt != dnnl_data_type_undef
                    && dnnl_data_type_size(dst_dt)
                            != dnnl_data_type_size(e.sum.dt)) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return;
            }
        }
    }
}

void skip_unimplemented_binary_po(const attr_t &attr, res_t *res) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return;

    std::vector<dnnl_data_type_t> dts;
    for (int i = 0; i < po.len(); i++) {
        const auto &e = po.entry[i];
        if (e.is_binary_kind()) {
            dts.push_back(e.binary.src1_dt);
            if (e.is_binary_kind_with_ternary_op())
                dts.push_back(e.binary.src2_dt);
        } else if (e.is_sum_kind()) {
            // Though sum is not binary, this code will be removed in a favor of
            // new status coming from the library. Just put sum here to avoid
            // major changes across all drivers.
            if (e.sum.dt != dnnl_data_type_undef) { dts.push_back(e.sum.dt); }
        }
    }
    skip_unimplemented_data_type(dts, FLAG_INF, res);
}

void skip_unimplemented_prelu_po(
        const attr_t &attr, res_t *res, dnnl_primitive_kind_t pkind) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return;

    const int first_prelu_idx = po.find(attr_t::post_ops_t::PRELU);
    if (first_prelu_idx == -1) return;

    switch (pkind) {
        case dnnl_convolution:
        case dnnl_deconvolution:
        case dnnl_inner_product:
        case dnnl_matmul: return; break;
        default:
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            break;
    }
}

void skip_unimplemented_arg_scale(const attr_t &attr, res_t *res) {
    for (const auto &arg_s : attr.scales.scales) {
        if (!arg_s.second.has_single_element()) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
    }
}

void skip_invalid_inplace(res_t *res, dnnl_data_type_t sdt,
        dnnl_data_type_t ddt, const std::string &stag,
        const std::string &dtag) {
    // Note: existing implementation of dnn_mem_t doesn't allow to track the
    // fact that two different objects pointing on the same SYCL memory should
    // not map/unmap both objects.
    // This leads to the restriction that memory descriptors should coincide,
    // thus, a single memory object would be used for in-place validation.
    // General limitation of in-place mode is having same amount of memory on
    // input and output.
    if (sdt != ddt) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    if (dtag == tag::any) return;
    if (stag != dtag) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

// Check ensures that attributes don't cause implementation fallback
int check_same_pd(const dnnl_primitive_desc_t &pd_no_attr, res_t *res) {
    const auto pd_no_attr_name = query_impl_info(pd_no_attr);
    if (res->impl_name == pd_no_attr_name) return OK;

    res->state = FAILED;
    BENCHDNN_PRINT(0,
            "ERROR: attributes caused impl fallback from [%s] to [%s]\n",
            pd_no_attr_name.c_str(), res->impl_name.c_str());
    return FAIL;
}

bool is_f64_supported(const engine_t &engine) {
    if (!is_gpu(engine)) return false;
    if (is_nvidia_gpu(engine) || is_amd_gpu(engine)) return false;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    if (is_sycl_engine(engine)) {
        auto eng = dnnl::engine(engine, true);
        auto dev = dnnl::sycl_interop::get_device(eng);
        return dev.has(::sycl::aspect::fp64);
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (is_opencl_engine(engine)) {
        auto eng = dnnl::engine(engine, true);
        cl_device_id dev = dnnl::ocl_interop::get_device(eng);
        size_t param_size = 0;
        cl_int err = clGetDeviceInfo(
                dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &param_size);
        if (err != CL_SUCCESS) return false;

        std::string extension_string(param_size, '\0');
        err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, param_size,
                &extension_string[0], &param_size);
        if (err != CL_SUCCESS) return false;

        return extension_string.find("cl_khr_fp64") != std::string::npos;
    }
#endif
    return false;
}

#if defined(_WIN32)
#include "windows.h"

size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
#include <unistd.h>
#include <sys/sysctl.h>

size_t get_cpu_ram_size() {
#ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
#else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
#endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

int get_gpu_ram_sizes(size_t &ram_size, size_t &max_alloc_size) {
    if (!is_gpu()) return OK;
    if (ram_size > 0 && max_alloc_size > 0) return OK;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_device_id ocl_dev = dnnl::ocl_interop::get_device(eng);

    cl_ulong ram_sz = 0;
    cl_int status = clGetDeviceInfo(ocl_dev, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    ram_size = (size_t)ram_sz;
    // For OCL runtime we allow allocation of buffers up to VRAM size,
    // with the usage of CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL flag.
    max_alloc_size = (size_t)ram_sz;
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    ram_size = (size_t)sycl_dev
                       .get_info<::sycl::info::device::global_mem_size>();
    max_alloc_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::max_mem_alloc_size>();
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;
    if (count > 1) {
        assert(!"Found more than a single entry for memory.");
        return FAIL;
    }

    ze_device_memory_properties_t ze_dev_mem_props {};
    ze_dev_mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, &ze_dev_mem_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    ram_size = (size_t)ze_dev_mem_props.totalSize;

    ze_device_properties_t ze_dev_props {};
    ze_dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_dev_props.pNext = nullptr;
    status = zeDeviceGetProperties(ze_dev, &ze_dev_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    max_alloc_size = (size_t)ze_dev_props.maxMemAllocSize;
    return OK;
#else
    assert(!"unsupported GPU runtime");
#endif
    ram_size = 0;
    max_alloc_size = 0;
    return OK;
}

int get_cpu_cache_size(cpu_cache_args_t &cache_args) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    cache_args.L2_size = get_per_core_cache_size(2);
    cache_args.L3_size = get_per_core_cache_size(3);
    cache_args.num_cores = get_num_cores();
    cache_args.total_socket_size
            = (cache_args.L2_size + cache_args.L3_size) * cache_args.num_cores;
#else
    // If functions are not available, just use 150 MiB.
    cache_args.total_socket_size = 150 * 1024 * 1024;
#endif
    return OK;
}

int get_gpu_cache_size(size_t &cache_size) {
    if (!is_gpu()) return OK;

    static size_t _cache_size = 0;
    if (_cache_size > 0) {
        cache_size = _cache_size;
        return OK;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_int status = CL_SUCCESS;
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong cache_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            sizeof(cl_ulong), &cache_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    _cache_size = (size_t)cache_sz;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    _cache_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::global_mem_cache_size>();
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetCacheProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    std::vector<ze_device_cache_properties_t> ze_cache_props(count);
    for (auto &e : ze_cache_props) {
        e.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    }

    status = zeDeviceGetCacheProperties(ze_dev, &count, ze_cache_props.data());
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    for (uint32_t i = 0; i < count; i++) {
        if (ze_cache_props[i].flags == 0) {
            _cache_size = ze_cache_props[i].cacheSize;
            break;
        }
    }
#else
    assert(!"unsupported GPU runtime");
#endif
    cache_size = _cache_size;
    return OK;
}

// `checkit` function verifies the amount of memory needed for the case is
// complied with the system limits.
//
// The function logic is the following:
// If `prim_ref` is provided, the check will include its requirement.
// If check passes, it's good to go with a `prim_ref` enabled.
// If not, `checkit` can process this scenario by dropping `prim_ref` (nulling
// it) and try the check again. It may happen that `prim_ref` used more memory
// than naive reference implementation. In that case the total consumption will
// be lower and might fit the limit.
// It the check without `prim_ref` passes, it will run without it.
// If not, the case will be skipped as it can't be run.
int check_total_size(res_t *res, dnnl_primitive_t prim_ref) {
    // Skip the check if it is disabled.
    if (!mem_check) return OK;

    // Skip the check if the test object won't be executed.
    if (!has_bench_mode_bit(mode_bit_t::exec)) return OK;

    static size_t cpu_device_capacity = get_cpu_ram_size();
    static size_t gpu_device_capacity = 0;
    static size_t gpu_max_alloc_capacity = 0;
    SAFE(get_gpu_ram_sizes(gpu_device_capacity, gpu_max_alloc_capacity), WARN);

    const size_t device_max_capacity
            = is_cpu() ? cpu_device_capacity : gpu_device_capacity;

    // `0.70` is taken mostly due to integrated graphics and the way service
    // reorders are handled by benchdnn. See a comment at `execute_reorder`.
    // It's always a subject to change in the future.
    const double capacity_factor = 0.70;
    const double benchdnn_device_limit = capacity_factor * device_max_capacity;
    // Note: there used to be a separate limit for combined memory pool, however
    // it didn't work even at 0.80 point due to mentioned reorder peculiarity,
    // and the way RNN allocates memory for ref computations.
    // Use same limit for combined cpu and pure cpu cases.
    const double benchdnn_cpu_limit = capacity_factor * cpu_device_capacity;
    assert(benchdnn_device_limit > 0 && benchdnn_cpu_limit > 0);

    auto dir_c_str = [&res]() {
        // TODO: always reports BWD when fwd-for-bwd case runs as the check
        // moved to `checkit` stage and always has `mem_size_args.dir` set to
        // BWD.
        assert(res->mem_size_args.dir != DIR_UNDEF);
        return (res->mem_size_args.dir & FLAG_FWD) ? "FWD" : "BWD";
    };

    const check_mem_size_args_t &check_mem_size_args = res->mem_size_args;

    if (is_gpu()) {
        // Mapped host buffers are live together with device allocations at peak
        // (and on some runtimes become device-resident while accessed), so
        // account for them in the device peak used for the device RAM fit check.
        const size_t device_peak_size = check_mem_size_args.total_size_device
                + check_mem_size_args.total_size_mapped;
        const bool fits_device_ram = device_peak_size <= benchdnn_device_limit;
        if (!fits_device_ram) {
            BENCHDNN_PRINT(1,
                    "[CHECK_MEM][%s]: Not enough device RAM for a problem.\n",
                    dir_c_str());
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_enough_ram;
        }

        const bool all_allocation_fit_limit
                = std::all_of(check_mem_size_args.sizes.cbegin(),
                        check_mem_size_args.sizes.cend(), [&](size_t s) {
            const bool fit = s < gpu_max_alloc_capacity;
            if (!fit) {
                BENCHDNN_PRINT(1,
                        "[CHECK_MEM][%s]: Allocation of size %s "
                        "doesn't fit allocation limit of %s.\n",
                        dir_c_str(), smart_bytes(s).c_str(),
                        smart_bytes(gpu_max_alloc_capacity).c_str());
            }
            return fit;
        });
        if (!all_allocation_fit_limit) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_enough_ram;
        }

        BENCHDNN_PRINT((!fits_device_ram ? 1 : 6),
                "[CHECK_MEM][%s]: Requested: %s (device: %s; mapped: %s); "
                "benchdnn_device_limit: %s; device_RAM_capacity: %s; "
                "gpu_max_alloc: %s;\n",
                dir_c_str(), smart_bytes(device_peak_size).c_str(),
                smart_bytes(check_mem_size_args.total_size_device).c_str(),
                smart_bytes(check_mem_size_args.total_size_mapped).c_str(),
                smart_bytes(benchdnn_device_limit).c_str(),
                smart_bytes(gpu_device_capacity).c_str(),
                smart_bytes(gpu_max_alloc_capacity).c_str());
    }

    // Note: in theory, `total_size_ref` itself can be smaller for a `prim_ref`
    // because stock reference uses f32 for estimation and best `prim_ref` tries
    // requested data types first which can be lower precision data types which
    // require less memory. However, during the filling both memories - the
    // original f32 and prim_ref memory are alive - and they might be huge. To
    // avoid potential overflow at that exact moment, ref sizes are combined.
    size_t total_size_ref = check_mem_size_args.total_size_ref;
    // Note: `prim_ref` can require extra memory buffer for comparison to
    // convert output to `abx` format.
    size_t total_size_compare = check_mem_size_args.total_size_compare;
    if (prim_ref) {
        // Collect memory sizes of prim_ref.
        check_mem_size_args_t prim_ref_mem_size_args;
        collect_mem_size(prim_ref_mem_size_args, query_pd(prim_ref), DIR_UNDEF,
                /* need_skip = */ false);
        // Add prim_ref reference size number.
        total_size_ref += std::accumulate(prim_ref_mem_size_args.sizes.begin(),
                prim_ref_mem_size_args.sizes.end(), 0ULL);
        // Update compare size number. It's twice of original memory since both
        // tensors expected to be of the same size.
        const auto const_pd = query_pd(prim_ref);
        const auto prop_kind = query_prop_kind(const_pd);
        const bool is_fwd = is_fwd_prop_kind(prop_kind);
        const_dnnl_memory_desc_t output_md {};
        if (is_fwd) {
            output_md = query_md(const_pd, dnnl_query_dst_md, 0);
        } else {
            const auto diff_src_md
                    = query_md(const_pd, dnnl_query_diff_src_md, 0);
            const auto diff_wei_md
                    = query_md(const_pd, dnnl_query_diff_weights_md, 0);
            const auto diff_src_md_size
                    = dnnl_memory_desc_get_size(diff_src_md);
            output_md = diff_src_md_size > 0 ? diff_src_md : diff_wei_md;
        }

        if (query_md_data_type(output_md) != dnnl_f32
                || !check_md_consistency_with_tag(output_md, tag::abx)) {
            total_size_compare *= 2;
        }
    }

    size_t total_size_cpu = total_size_ref + total_size_compare
            + check_mem_size_args.total_size_mapped
            + check_mem_size_args.extra_size_driver;

    // If the problem runs on CPU, the combined memory represents requirements
    // for the library and for the reference paths.
    // If the problem runs on a device, the combined memory represents potential
    // requirement for integrated devices that use CPU pool for both memories.
    size_t cpu_and_device_size
            = total_size_cpu + check_mem_size_args.total_size_device;
    bool fits_cpu_ram = cpu_and_device_size <= benchdnn_cpu_limit;

    // Save the expected value to set at `doit`, otherwise, the check doesn't
    // work correctly. See `zmalloc_expected_size` comment.
    res->mem_size_args.zmalloc_expected_size
            = is_cpu() ? cpu_and_device_size : total_size_cpu;

    if (!fits_cpu_ram) {
        std::string prim_ref_msg
                = prim_ref ? " with CPU primitive reference" : "";
        BENCHDNN_PRINT(1,
                "[CHECK_MEM][%s]: Not enough CPU RAM for a problem%s.\n",
                dir_c_str(), prim_ref_msg.c_str());
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_enough_ram;
    }

    if (!fits_cpu_ram) {
        // Try to catch a huge scratchpad size requested by the library.
        // Use following logic:
        //     scratch_size
        // ---------------------- <= 0.75 (pre-defined threshold).
        // io_size + scratch_size
        //
        // 0.75 value supposed to be experimental and might be adjusted.
        static constexpr float scratch_trh = 0.75f;
        if (is_cpu()
                && check_mem_size_args.scratchpad_size
                        > scratch_trh * check_mem_size_args.total_size_device) {
            BENCHDNN_PRINT(1,
                    "[CHECK_MEM][%s]: CPU scratchpad size `%zu` exceeded a "
                    "given threshold `%zu`.\n",
                    dir_c_str(), check_mem_size_args.scratchpad_size,
                    (size_t)(scratch_trh
                            * check_mem_size_args.total_size_device));
            res->state = FAILED;
        }
    }

    BENCHDNN_PRINT((!fits_cpu_ram ? 1 : 6),
            "[CHECK_MEM][%s]: benchdnn_CPU_limit: %s; CPU_RAM_capacity: %s;\n",
            dir_c_str(), smart_bytes(benchdnn_cpu_limit).c_str(),
            smart_bytes(cpu_device_capacity).c_str());

    std::string sizes_str;
    for (const auto sz : check_mem_size_args.sizes) {
        const bool is_scratchpad = sz == check_mem_size_args.scratchpad_size;
        sizes_str += smart_bytes(sz) + (is_scratchpad ? " (Scratchpad)" : "")
                + ", ";
    }
    BENCHDNN_PRINT(6, "[CHECK_MEM][%s]: Sizes: {%s};\n", dir_c_str(),
            sizes_str.c_str());

    std::string total_size_device_str = is_cpu()
            ? smart_bytes(check_mem_size_args.total_size_device) + " (Lib), "
            : "";
    BENCHDNN_PRINT((!fits_cpu_ram ? 1 : 6),
            "[CHECK_MEM][%s]: Requested: %s%s (Service), %s (combined);\n",
            dir_c_str(), total_size_device_str.c_str(),
            smart_bytes(total_size_cpu).c_str(),
            smart_bytes(cpu_and_device_size).c_str());

    return res->state == FAILED ? FAIL : OK;
}

void add_md_size(const_dnnl_memory_desc_t md,
        check_mem_size_args_t &check_mem_size_args) {
    size_t mem_size = SIZE_MAX;
    if (check_mem_size_args.use_logical_size) {
        mem_size = get_logical_size(md);
    } else {
        mem_size = dnnl_memory_desc_get_size(md);
    }
    // Runtime mem size is not defined.
    if (mem_size == 0 || mem_size == DNNL_RUNTIME_SIZE_VAL) return;

    check_mem_size_args.sizes.push_back(mem_size);

    // Original memory size.
    check_mem_size_args.total_size_device += mem_size;

    // GPU mapped memory factor.
    // All memory is mapped once it is created and unmapped only for primitive
    // execution mapped back again right after. Device memory requires
    // additional buffer for mapped memory allocated on host (CPU).
    //
    // Note: When oneDNN uses USM shared memory on an iGPU, additional buffers
    // are not required, so map factor could be equal to 0. This is not
    // accounted for to maintain simplicity.
    //
    // This might be improved by switching to lazy mapping when it happens upon
    // direct data accessing and unmapping right after the access completed.
    // In such case it requires only the biggest buffer among all to be
    // accounted for total memory check, with scratchpad not included among
    // those.
    // ANCHOR: LAZY_MAPPING.
    const bool mapped_mem_factor = !is_cpu()
            && !has_bench_mode_modifier(mode_modifier_t::no_ref_memory);

    // Mapped memory for GPU backend on CPU.
    check_mem_size_args.total_size_mapped += mapped_mem_factor * mem_size;

    const bool is_corr = has_bench_mode_bit(mode_bit_t::corr);
    const bool is_bitwise = has_bench_mode_bit(mode_bit_t::bitwise);
    // Reference memories are always tag::abx fp32, hence need re-creating
    // memory descriptor and take its size. The dims come from an existing
    // descriptor, so a failure is not expected.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> ref_md;
    DNN_SAFE_V(dnn_mem_t::init_md(
            ref_md, query_md_ndims(md), query_md_dims(md), dnnl_f32, tag::abx));
    const auto ref_md_size = dnnl_memory_desc_get_size(ref_md);

    const size_t ref_mem_idx = check_mem_size_args.want_input ? 0 : 1;
    check_mem_size_args.total_ref_md_size[ref_mem_idx] = ref_md_size;

    // A memory copy for ref_compute, happens only in correctness.
    check_mem_size_args.total_size_ref += is_corr * ref_md_size;

    // Comparison function allocates an additional tag::abx f32 memory.
    // This allocation holds for correctness and bitwise modes.
    const bool compare_mem_factor
            = !check_mem_size_args.want_input && (is_corr || is_bitwise);
    check_mem_size_args.total_size_compare += compare_mem_factor * ref_md_size;

    // Bitwise comparison allocates an additional tag::abx f32 memory from
    // the first run to compare results against it.
    const bool bitwise_compare_mem_factor
            = !check_mem_size_args.want_input && is_bitwise;
    check_mem_size_args.total_size_compare
            += bitwise_compare_mem_factor * ref_md_size;
}

bool is_fwd_training(dnnl_prop_kind_t prop_kind) {
    return prop_kind == dnnl_forward_training
            || prop_kind == dnnl_prop_kind_undef;
}

bool is_fwd_prop_kind(dnnl_prop_kind_t prop_kind) {
    return prop_kind == dnnl_forward_training
            || prop_kind == dnnl_forward_inference
            || prop_kind == dnnl_prop_kind_undef;
}

void get_memory_bytes(check_mem_size_args_t &check_mem_size_args) {
    auto const_pd = check_mem_size_args.pd;
    const int n_idx = check_mem_size_args.want_input
            ? query_n_inputs(const_pd)
            : query_n_outputs(const_pd);
    const auto prop_kind = query_prop_kind(const_pd);
    const bool is_fwd = is_fwd_prop_kind(prop_kind);

#define MD(name) dnnl_query_##name##_md
    std::vector<dnnl_query_t> query_fwd_in_mds {MD(src), MD(weights)};
    std::vector<dnnl_query_t> query_fwd_out_mds {MD(dst), MD(workspace)};

    std::vector<dnnl_query_t> query_bwd_in_mds {
            MD(src), MD(weights), MD(dst), MD(diff_dst), MD(workspace)};
    std::vector<dnnl_query_t> query_bwd_out_mds {
            MD(diff_src), MD(diff_weights)};
#undef MD

    const auto &query_in_mds = is_fwd ? query_fwd_in_mds : query_bwd_in_mds;
    const auto &query_out_mds = is_fwd ? query_fwd_out_mds : query_bwd_out_mds;
    const auto &query_mds
            = check_mem_size_args.want_input ? query_in_mds : query_out_mds;

    for_(const auto query : query_mds)
    for (int idx = 0; idx < n_idx; ++idx) {
        const auto &md = query_md(const_pd, query, idx);
        add_md_size(md, check_mem_size_args);
    }

    // Binary post-op memories counted as input.
    if (check_mem_size_args.want_input) {
        auto const_attr_po = query_post_ops(const_pd);
        auto po_len = dnnl_post_ops_len(const_attr_po);
        for (int idx = 0; idx < po_len; ++idx) {
            const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
            if (kind == dnnl_binary) {
                int po_arg
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;
                const auto &po_md = query_md(const_pd, po_arg);
                add_md_size(po_md, check_mem_size_args);

                if (query_post_ops_has_binary_alg_kind(
                            const_attr_po, dnnl_binary_select, idx)) {
                    int po_arg_src2 = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                            | DNNL_ARG_SRC_2;
                    const auto &po_md_src2 = query_md(const_pd, po_arg_src2);
                    add_md_size(po_md_src2, check_mem_size_args);
                }
            }
        }
    }
}

int check_mem_size(const_dnnl_memory_desc_t md, res_t *res) {
    if (!mem_check) return OK;

    const auto md_size = dnnl_memory_desc_get_size(md);
    res->mem_size_args.total_size_device = md_size;
    res->mem_size_args.sizes.push_back(md_size);
    return check_total_size(res);
}

int collect_mem_size(check_mem_size_args_t &mem_size_args,
        const_dnnl_primitive_desc_t const_pd, dir_t dir, bool need_skip) {
    // Skip the check if it is disabled.
    if (!mem_check) return OK;

    // Skip the check if the test object won't be executed.
    if (!has_bench_mode_bit(mode_bit_t::exec)) return OK;

    // Skip the check if it has already happened for the passed `dir`.
    // It saves from a repeated run when the second test object is created to
    // validate the primitive cache. At the same time it allows to verify both
    // test objects when a double-run driver executes the fwd-for-bwd object
    // first and the bwd object after.
    // ANCHOR: MEM_CHECK_ARGS_DIR;
    if (need_skip && mem_size_args.dir == dir) return OK;

    // Get input sizes.
    check_mem_size_args_t check_mem_size_args(
            const_pd, /* input = */ true, dir);
    get_memory_bytes(check_mem_size_args);

    // Get scratchpad size.
    // Since scratchpad modes are mutually excluded, get sizes of both modes as
    // either of them will report 0 size depending on the mode, and take the
    // biggest from them.
    const size_t library_scratchpad_size
            = static_cast<size_t>(query_mem_consumption(const_pd));
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);
    const size_t user_scratchpad_size
            = dnnl_memory_desc_get_size(scratchpad_md);
    const size_t scratchpad_size
            = MAX2(library_scratchpad_size, user_scratchpad_size);
    // Update same fields as `add_md_size` would. See details there.
    check_mem_size_args.sizes.push_back(scratchpad_size);
    check_mem_size_args.total_size_device += scratchpad_size;
    check_mem_size_args.scratchpad_size = scratchpad_size;
    //   Add mapped size for scratchpad due to it gets mapped unconditionally
    //   when it gets created and then after the execution ends followed by
    //   comparison which creates a memory object and may trigger a memory
    //   over-consumption error message.
    //   ANCHOR: LAZY_MAPPING.
    check_mem_size_args.total_size_mapped += scratchpad_size;

    // Get output sizes.
    check_mem_size_args.want_input = false;
    get_memory_bytes(check_mem_size_args);

    // Copy memory stats. It's required to accumulate them before performing
    // the check.
    mem_size_args = check_mem_size_args;
    return OK;
}

int get_memory_footprint(const_dnnl_primitive_desc_t const_pd, res_t *res) {
    check_mem_size_args_t check_mem_in_size_args(const_pd,
            /* want_input = */ true, DIR_UNDEF,
            /* use_logical_size */ true);
    get_memory_bytes(check_mem_in_size_args); // Get input bytes.
    check_mem_size_args_t check_mem_out_size_args(const_pd,
            /* want_input = */ false, DIR_UNDEF,
            /* use_logical_size */ true);
    get_memory_bytes(check_mem_out_size_args); // Get output bytes.

    // Sum post-ops include dst bytes as an input. Not included in
    // `get_memory_bytes` since it would cause `collect_mem_size` to
    // double-count dst bytes.
    auto const_attr_po = query_post_ops(const_pd);
    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind == dnnl_sum) {
            const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
            add_md_size(dst_md, check_mem_in_size_args);
        }
    }

    // Forward nearest neighbor down sampling (one or more bigger source
    // dimensions than in destination) cases by the nature of the operation
    // read only a part of a source tensor. Here we adjust the `ibytes` variable
    // so that reported bandwidth numbers reflect the reality, otherwise
    // final numbers are much higher than they should be.
    dnnl_primitive_kind_t kind = query_prim_kind(const_pd);
    dnnl_alg_kind_t alg = query_alg_kind(const_pd);
    dnnl_prop_kind_t prop = query_prop_kind(const_pd);
    if (is_fwd_prop_kind(prop) && kind == dnnl_resampling
            && alg == dnnl_resampling_nearest) {
        auto src_md = query_md(const_pd, DNNL_ARG_SRC);
        auto dst_md = query_md(const_pd, DNNL_ARG_DST);
        auto ndims = query_md_ndims(src_md);
        auto src_pdims = query_md_padded_dims(src_md);
        auto dst_pdims = query_md_padded_dims(dst_md);
        dnnl_dim_t total_elems = 1;
        dnnl_dim_t read_elems = 1;
        for (int i = 0; i < ndims; i++) {
            if (dst_pdims[i] < src_pdims[i]) {
                total_elems *= src_pdims[i];
                read_elems *= dst_pdims[i];
            }
        }
        const auto src_size = dnnl_memory_desc_get_size(src_md);
        const auto fixed_src_size = src_size / static_cast<size_t>(total_elems)
                * static_cast<size_t>(read_elems);
        check_mem_in_size_args.total_size_device += fixed_src_size - src_size;
    }

    const auto &adjust_src_bytes_for_stride
            = [&check_mem_in_size_args](const_dnnl_memory_desc_t src_md,
                      int spatial_dims, const dnnl_dims_t strides,
                      const dnnl_dims_t kernels) {
        double kernel_ratio = 1;
        for (int d = 0; d < spatial_dims; d++) {
            auto stride = strides[d];
            auto kernel = kernels[d];
            if (stride > kernel) {
                kernel_ratio *= static_cast<double>(stride) / kernel;
            }
        }
        const auto src_size = dnnl_memory_desc_get_size(src_md);
        const auto fixed_src_size = static_cast<size_t>(
                static_cast<double>(src_size) / kernel_ratio);
        check_mem_in_size_args.total_size_device += fixed_src_size - src_size;
    };

    // When a dimensional stride is bigger than kernel window, it means there
    // are less reads from source by stride/kernel times.
    if (is_fwd_prop_kind(prop) && kind == dnnl_convolution) {
        auto src_md = query_md(const_pd, DNNL_ARG_SRC);
        auto wei_md = query_md(const_pd, DNNL_ARG_WEIGHTS);
        auto src_ndims = query_md_ndims(src_md);
        auto wei_ndims = query_md_ndims(wei_md);
        auto with_groups = src_ndims < wei_ndims;
        auto wei_dims = query_md_dims(wei_md);

        auto strides = query_strides(const_pd);
        adjust_src_bytes_for_stride(
                src_md, src_ndims - 2, strides, &wei_dims[with_groups + 2]);
    }

    // When a dimensional stride is bigger than kernel window, it means there
    // are less reads from source by stride/kernel times.
    if (is_fwd_prop_kind(prop) && kind == dnnl_pooling) {
        auto src_md = query_md(const_pd, DNNL_ARG_SRC);
        auto ndims = query_md_ndims(src_md);

        auto strides = query_strides(const_pd);
        auto kernels = query_kernels(const_pd);
        adjust_src_bytes_for_stride(src_md, ndims - 2, strides, kernels);
    }

    res->ibytes = check_mem_in_size_args.total_size_device;
    res->obytes = check_mem_out_size_args.total_size_device;

    return OK;
}

memory_kind_ext_t str2memory_kind(const char *str) {
#define CASE(param) \
    if (!strcasecmp(#param, str)) return memory_kind_ext_t::param

    CASE(usm);
    CASE(buffer);
    CASE(usm_device);
    CASE(usm_shared);

#undef CASE

    assert(!"not expected");
    return memory_kind_ext_t::usm;
}

float reorder_rescale_factor() {
    float factor = 1.f;
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (is_cpu(get_test_engine()))
        factor = dnnl::impl::cpu::platform::s8s8_weights_scale_factor();
#endif
    return factor;
}

dims_t md2dims(const_dnnl_memory_desc_t md, int mask, bool extend_by_ones,
        const std::vector<int64_t> &groups) {
    auto ndims = query_md_ndims(md);
    dims_t dims;
    for (int d = 0; d < ndims; ++d) {
        if (mask & (1 << d)) {
            dims.push_back(query_md_dims(md)[d]);
            // Note: groups are done for matmul's last two dimensions.
            const auto group_dim = d - (ndims - 2);
            if (!groups.empty() && group_dim >= 0) {
                // If groups are passed, divide dims on the correspondent group
                // size. It's needed to pass proper memory objects.
                assert(dims.back() % groups[group_dim] == 0);
                dims.back() /= groups[group_dim];
            }
        } else if (extend_by_ones) {
            dims.push_back(1);
        }
    }
    return dims;
}

dims_t md2dims(const_dnnl_memory_desc_t md, int mask, bool extend_by_ones) {
    return md2dims(md, mask, extend_by_ones, {});
}

dnnl_data_type_t deduce_cfg_data_type(
        dnnl_data_type_t in_dt, const attr_t &attr, data_kind_t dk) {
    dnnl_data_type_t dt_ = in_dt;

    if ((dk == SRC || dk == WEI) && dt_ == dnnl_f32) {
        // Update data type based on fpmath-mode attribute
        switch (attr.fpmath_mode.mode) {
            case dnnl_fpmath_mode_strict: break;
            case dnnl_fpmath_mode_bf16: dt_ = dnnl_bf16; break;
            case dnnl_fpmath_mode_f16: dt_ = dnnl_f16; break;
            case dnnl_fpmath_mode_tf32: dt_ = dnnl_bf16; break;
            default: assert(!"unsupported_fpmath_mode"); SAFE_V(CRIT);
        }
    } else if (dk == DST) {
        // Sum post-op defines the type of filling destination.
        const int sum_idx = attr.post_ops.find(attr_t::post_ops_t::SUM);
        if (sum_idx >= 0) {
            auto sum_dt = attr.post_ops.entry[sum_idx].sum.dt;
            if (sum_dt != dnnl_data_type_undef) dt_ = sum_dt;
        }
    }

    return dt_;
}

// The function removes arguments that are unnecessary. Such arguments may come
// from the forward-for-backward primitive running. The library map removes them
// when allocating memories for the backward primitive and the reference map
// should do the same.
void erase_unused_args(
        dnn_mem_map_t &ref_mem_map, const dnn_mem_map_t &mem_map) {
    // Collection of keys is required as evicting members along the way
    // invalidates references in the modified object and makes further
    // traversing over the object undefined.
    std::vector<int> keys_to_erase;
    keys_to_erase.reserve(ref_mem_map.size());

    for (const auto &pair : ref_mem_map) {
        const auto key = pair.first;
        if (mem_map.find(key) == mem_map.end()) {
            // Correspondent argument is not found in a library mem map.
            // It means it should be removed.
            keys_to_erase.push_back(key);
        }
    }
    for (const auto &k : keys_to_erase) {
        ref_mem_map.erase(k);
    }
}

// Appends data kinds to check during comparison into `check_kinds` vector
// coming from extensions through `attr`.
void get_kinds_to_check_shared(
        std::vector<data_kind_t> &check_kinds, const attr_t &attr) {
    if (!attr.dropout.is_def() && attr.dropout.has_output_mask())
        check_kinds.push_back(DROPOUT_MASK);

    if (!attr.scales.get(DNNL_ARG_DST).is_def()
            && attr.scales.get(DNNL_ARG_DST).is_dynamic())
        check_kinds.push_back(DST_SCALES);
}

// This function handles cases when optimized CPU primitive is used as a
// reference for a problem. Optimized primitive means custom memory formats
// which require reorder to them. Since `ref_mem_map` is passed to optimized
// primitive, it's required to replace correspondent memory objects and update
// them with proper values to get the matched output.
// The function also handles cases when a target primitive doesn't need a
// scratchpad while the reference one does.
// Note: the last argument `swapped_dt` is a property of `driver::cfg_t`. `cfg`
// could be passed directly, but since it's tied to a `prb_t`, the function will
// requires a template argument. If more members from `cfg` would be needed,
// consider passing `cfg` directly.
int update_ref_mem_map_from_prim(dnnl_primitive_t prim_ref,
        const dnn_mem_t &library_mem, dnn_mem_map_t &ref_mem_map, int exec_arg,
        dnnl_data_type_t swapped_dt, res_t *res) {
    if (!prim_ref) return OK;

    const auto &ref_mem = ref_mem_map.at(exec_arg);
    const bool is_scratchpad = exec_arg == DNNL_ARG_SCRATCHPAD;

    // If `ref_mem` is empty (unless scratchpad since GPU may not have one
    // while CPU may), it means there's nothing to update.
    if (!ref_mem && !is_scratchpad) return OK;

    bool skip_replace = false;
    auto const_ref_pd = query_pd(prim_ref);
    const auto &ref_md = query_md(const_ref_pd, exec_arg);
    const auto &ref_engine = get_cpu_engine();
    // Since this goes to the library, it's desired to verify CPU implementation
    // handles memories correctly.
    dnn_mem_t prim_ref_mem(ref_md, ref_engine, /* prefill = */ true);

    // When queried memory comes empty, it may be attributes as library doesn't
    // have dedicated query mechanism for those. Process potential outcomes:
    while (query_md_ndims(ref_md) == 0) {
        bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
        // Scales received data type support in the library. The reference
        // primitive expects them in the same data type.
        if (is_scales_arg) {
            prim_ref_mem = dnn_mem_t(library_mem.md_, library_mem.dt(),
                    tag::abx, ref_engine, /* prefill = */ true);
            break;
        }

        bool is_zero_point_arg = (exec_arg & DNNL_ARG_ATTR_ZERO_POINTS);
        // Zero-points received data type support in the library. The reference
        // primitive expects them in the same data type.
        if (is_zero_point_arg) {
            prim_ref_mem = dnn_mem_t(library_mem.md_, library_mem.dt(),
                    tag::abx, ref_engine, /* prefill = */ true);
            break;
        }

        const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
        const bool is_post_ops_arg = (exec_arg & post_ops_range);
        const bool is_prelu_arg
                = is_post_ops_arg && (exec_arg & DNNL_ARG_WEIGHTS);
        // The library doesn't return a memory desc for prelu post-op. Prelu
        // requires `tag::axb` format, thus, need to put a desc into ref prim.
        if (is_prelu_arg) {
            prim_ref_mem = dnn_mem_t(library_mem.md_, dnnl_f32, tag::axb,
                    ref_engine, /* prefill = */ true);
            break;
        }

        // Rest arguments don't need special handling and should be
        // skipped as empty.
        skip_replace = true;
        break;
    }

    // Avoid reordering on empty memories;
    // Avoid replacing memories that have already been properly filled.
    if (skip_replace) return OK;

    if (!is_scratchpad) {
        const auto prim_ref_swapped_dt = query_md_data_type(ref_md) == dnnl_f32
                ? dnnl_data_type_undef
                : swapped_dt;
        SAFE(prim_ref_mem.reorder(ref_mem, res, prim_ref_swapped_dt), WARN);
    }
    ref_mem_map[exec_arg] = std::move(prim_ref_mem);

    return OK;
}

// This function provides a general filling for atributes across all drivers.
//
// It provides a default filling config for both binary and prelu post-op.
// The user has an option to override it by passing `fill_cfg_map` with
// correspondent argument and attached `fill_cfg_t` object to it.
// Default filling configs are simple to avoid floating-point rounding effects,
// but not cancellation effects. For latter ones, it's the user's responsibility
// to avoid them by supplying a proper fill_cfg for a given driver/problem.
int init_ref_memory_args_default_case(int exec_arg, dnn_mem_t &mem,
        dnn_mem_t &ref_mem, const attr_t &attr, res_t *res,
        const std::unordered_map<int, fill_cfg_t> &fill_cfg_map) {
    assert(exec_arg > 0); // Negative values will produce false-positive `true`.
    if (fill_from_file(exec_arg, mem, ref_mem, res)) return OK;

    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    const bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
    const bool is_zero_point_arg = (exec_arg & DNNL_ARG_ATTR_ZERO_POINTS);
    const bool is_dropout_p = (exec_arg == DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
    const bool is_dropout_seed = (exec_arg == DNNL_ARG_ATTR_DROPOUT_SEED);
    const bool is_dropout_offset = (exec_arg == DNNL_ARG_ATTR_DROPOUT_OFFSET);
    const bool is_rounding_seed = (exec_arg == DNNL_ARG_ATTR_ROUNDING_SEED);

    if (is_post_ops_arg) {
        const int bin_po_idx
                = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
        assert(bin_po_idx < attr.post_ops.len());
        const bool exact_match_for_src1_arg = !(exec_arg
                ^ (DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_po_idx)
                        | DNNL_ARG_SRC_1));
        const bool exact_match_for_src2_arg = !(exec_arg
                ^ (DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_po_idx)
                        | DNNL_ARG_SRC_2));

        if (exact_match_for_src1_arg) {
            // The right-hand side of an in-place Binary has no dedicated memory
            // to fill; current source of data - DST arg - is filled in the
            // driver body.
            if (attr.post_ops.entry[bin_po_idx].is_binary_kind_inplace()
                    && !mem)
                return OK;

            const auto alg = attr.post_ops.entry[bin_po_idx].kind;
            // Binary post-op filling for src1 tensor
            fill_cfg_t def_binary_cfg(mem.dt(), -16.f, 16.f,
                    /* int = */ true, alg, "def_binary_post_op_src1");
            const auto it = fill_cfg_map.find(DNNL_ARG_SRC_1);
            const bool has_external_cfg = it != fill_cfg_map.end();
            const fill_cfg_t &binary_fill_cfg
                    = has_external_cfg ? (*it).second : def_binary_cfg;
            TIME_FILL(SAFE(fill_random_real(mem, ref_mem, res, binary_fill_cfg),
                    WARN));
        } else if (exact_match_for_src2_arg) {
            assert(attr.post_ops.entry[bin_po_idx]
                            .is_binary_kind_with_ternary_op());
            const auto alg = attr.post_ops.entry[bin_po_idx].kind;
            // Binary post-op filling for src2 conditional tensor
            // Use values bigger than 1 to ensure it works correctly.
            fill_cfg_t def_binary_cfg(mem.dt(), 0, 16.f,
                    /* int = */ true, alg, "def_binary_post_op_src2");
            const auto it = fill_cfg_map.find(DNNL_ARG_SRC_2);
            const bool has_external_cfg = it != fill_cfg_map.end();
            const fill_cfg_t &binary_fill_cfg
                    = has_external_cfg ? (*it).second : def_binary_cfg;
            TIME_FILL(SAFE(fill_random_real(mem, ref_mem, res, binary_fill_cfg),
                    WARN));
        } else if (exec_arg & DNNL_ARG_WEIGHTS) {
            // Prelu post-op filling.
            fill_cfg_t def_prelu_fill_cfg(mem.dt(), -2.f, 2.f, /* int = */ true,
                    attr_t::post_ops_t::kind_t::PRELU, "def_prelu_post_op");
            const auto it = fill_cfg_map.find(DNNL_ARG_WEIGHTS);
            const bool has_external_cfg = it != fill_cfg_map.end();
            const fill_cfg_t &prelu_fill_cfg
                    = has_external_cfg ? (*it).second : def_prelu_fill_cfg;
            TIME_FILL(SAFE(
                    fill_random_real(mem, ref_mem, res, prelu_fill_cfg), WARN));
        }
    } else if (is_scales_arg) {
        int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
        TIME_FILL(SAFE(
                fill_scales(attr, local_exec_arg, mem, ref_mem, res), WARN));
    } else if (is_zero_point_arg) {
        int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_ZERO_POINTS;
        TIME_FILL(
                SAFE(fill_zero_points(attr, local_exec_arg, mem, ref_mem, res),
                        WARN));
    } else if (is_dropout_p) {
        ref_mem.set_f32_elem(0, attr.dropout.p);
        mem.set_elem(0, attr.dropout.p);
    } else if (is_dropout_seed) {
        ref_mem = dnn_mem_t(mem.md_, dnnl_s64, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        ref_mem.set_s64_elem(0, attr.dropout.seed);
        assert(mem.dt() == dnnl_s64);
        mem.set_s64_elem(0, attr.dropout.seed);
    } else if (is_dropout_offset) {
        ref_mem = dnn_mem_t(mem.md_, dnnl_s64, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        ref_mem.set_s64_elem(0, attr.dropout.offset);
        assert(mem.dt() == dnnl_s64);
        mem.set_s64_elem(0, attr.dropout.offset);
    } else if (is_rounding_seed) {
        ref_mem.set_elem(0, attr.rounding_mode.seed);
        TIME_FILL(SAFE(mem.reorder(ref_mem, res), WARN));
    }

    return OK;
}

// This function is responsible for performing the bitwise validation:
// * Saves the output of the first run for further comparison.
// * Refreshes the input data when the original data was corrupted, e.g.,
//   inplace mode or sum post-op.
// * Performs a second run of the original primitive and its inputs.
// * Compares both outputs on bitwise exactness.
//
// The function takes the following arguments:
// * A `prim` object, the same one used for the first run.
// * A vector of `kinds` to validate each output from the primitive. The exactly
//   same one is used for correctness validation.
// * `args` arguments with all memory objects used for the first run and also
//   with stashed memories when they got overwritten during execution.
// * `inplace` flags to help identify if data refresh is needed.
// * `res` object to save the state of the validation result.
//
int check_bitwise(dnnl_primitive_t prim, const std::vector<data_kind_t> &kinds,
        const args_t &args, const attr_t &attr, bool inplace, res_t *res) {
    // Fast exit for any modes but bitwise.
    if (!has_bench_mode_bit(mode_bit_t::bitwise)) return OK;

    // Forward-for-backward service primitives define `kinds` as empty to skip
    // validation. This is to avoid extra checks on higher level.
    if (kinds.empty()) return OK;

    // Collect copies of outputs.
    dnn_mem_map_t run1_mem_map;
    for (const auto &kind : kinds) {
        const int arg = data_kind2exec_arg(kind);
        assert(arg > 0);

        auto &mem = args.find(arg);
        if (!mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: output memory was not found among arguments.");
            res->state = FAILED;
            return FAIL;
        }
        // A memory used as reference for comparison, must be allocated on the
        // CPU engine.
        run1_mem_map.emplace(arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, get_cpu_engine(),
                        /* prefill = */ false));
        SAFE(run1_mem_map.at(arg).reorder(mem, res), WARN);
    }

    // Put original data into DST tensor if a post-op reading it is present.
    if (query_post_ops_has_kind(prim, dnnl_sum)
            || query_post_ops_has_binary_alg_kind(
                    prim, dnnl_binary_mul_inplace)) {
        const int query_arg = DNNL_ARG_DST;
        auto &dst_mem = const_cast<dnn_mem_t &>(args.find(query_arg));
        const auto &orig_dst_mem = args.find(-query_arg);
        SAFE_V(bool(orig_dst_mem) && bool(dst_mem) ? OK : FAIL);
        SAFE(dst_mem.reorder(orig_dst_mem, res), WARN);
    }

    // Put original data into SRC if inplace mode was specified.
    if (inplace) {
        const bool has_multiple_args = bool(args.find(DNNL_ARG_MULTIPLE_SRC));
        const auto prop_kind = query_prop_kind(query_pd(prim));
        const auto query_arg = is_fwd_prop_kind(prop_kind)
                ? (has_multiple_args ? DNNL_ARG_MULTIPLE_SRC : DNNL_ARG_SRC)
                : DNNL_ARG_DIFF_DST;
        auto &in_mem = const_cast<dnn_mem_t &>(args.find(query_arg));
        if (!in_mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input memory was not found among arguments.");
            res->state = FAILED;
            return FAIL;
        }
        const auto &orig_in_mem = args.find(-query_arg);
        if (!orig_in_mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: original input memory was not found among "
                    "arguments.");
            res->state = FAILED;
            return FAIL;
        }
        SAFE(in_mem.reorder(orig_in_mem, res), WARN);
    }

    // Perform a second run.
    SAFE(execute_and_wait(prim, args, res), WARN);

    // `args_t` has an interface to retrieve a memory object by the `arg`.
    args_t run1_args(run1_mem_map);
    compare::compare_t cmp;
    for (const auto &kind : kinds) {
        cmp.set_data_kind(kind);

        const int arg = data_kind2exec_arg(kind);
        assert(arg > 0);

        auto &mem = args.find(arg);
        auto &run1_mem = run1_args.find(arg);

        TIME_COMPARE(cmp.compare(run1_mem, mem, attr, res));
    }

    return OK;
}

bool should_stop(const timer::timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb
                    && t.times() >= static_cast<size_t>(fix_times_per_prb))
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= static_cast<size_t>(min_times_per_prb));
    return stop;
}

int check_caches(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &primw,
        const thr_ctx_t &ctx_init, res_t *res) {
    if (!primw) return OK;

    // Under assumption of a limited cache capacity, which is usually the case,
    // any check that rely on primitive (descriptor) resides in cache fails for
    // parallel creation modifier. There's no guarantee in a general case that
    // primitive stays in cache till the the moment the check happens. Even with
    // primitive saving a state of picked from cache or not, there's a time gap
    // between creation of two same primitives over different engines where
    // first instance could be evicted. This approach may work under assumption
    // of infinite cache though.
    if (!has_bench_mode_modifier(mode_modifier_t::par_create)) {
        const_dnnl_primitive_desc_t pd = query_pd(primw);
        std::function<int()> pd_cache_fn = std::bind(check_pd_cache, pd, res);
        SAFE(create_in_thr_ctx(ctx_init, pd_cache_fn), WARN);
        // Check primitive is picked up from the cache if applicable.
        std::function<int()> prim_cache_fn
                = std::bind(check_primitive_cache, std::ref(primw), res);
        SAFE(create_in_thr_ctx(ctx_init, prim_cache_fn), WARN);
    }

    // Check primitive is picked up from the persistent cache if applicable.
    // Note: primw get re-written here to put a primitive from cache blob, if
    // GPU backend is OCL.
    SAFE(test_persistent_cache_api(primw, res), WARN);

    return OK;
}

int check_dnnl_status(
        dnnl_status_t status, const base_prb_t *base_prb, res_t *res) {
    if (!res || status == dnnl_success) return OK;

    switch (status) {
        case dnnl_invalid_arguments: res->state = INVALID_ARGUMENTS; break;
        case dnnl_unimplemented: {
            // Unconditionally set all Nvidia backend unimplemented cases as
            // not supported.
            if (is_nvidia_gpu() || is_amd_gpu()) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return OK;
            }

            // Check driver specific cases of unimplemented functionality.
            // It means that the case is valid from API perspective but not
            // supported by any implementation for a specific backend.
            //
            // Note: since it's done post pd creation, code in these
            // driver-defined functions can end up being dead.
            base_prb->skip_unimplemented(res);
            if (res->state == SKIPPED || res->state == DEFERRED) return OK;

            // If the case is not known to be skipped, it is unimplemented.
            res->state = UNIMPLEMENTED;
        } break;
        default: assert(!"unexpected");
    }
    DNN_SAFE(status, WARN);
    return OK;
}

int create_primitive(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &primw,
        dnnl_engine_t engine, const init_pd_func_t &init_pd_func,
        const base_prb_t *base_prb, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint, bool is_service_prim,
        const_dnnl_memory_desc_t src_md, bool force_f32_dt, bool is_graph_ref) {
    dnnl_status_t status = dnnl_success;
    dnnl_primitive_t prim {};

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;

    init_pd_args_t init_pd_args(
            res, engine, base_prb, dir, hint, src_md, force_f32_dt);
    status = init_pd_func(init_pd_args);

    SAFE(check_dnnl_status(status, base_prb, res), WARN);
    if (res->state == SKIPPED) return OK;
    if (is_graph_ref && res->state == DEFERRED) return OK;

    // Fetch also checks if user requested to skip certain implementations.
    SAFE(fetch_impl(pdw, init_pd_args, base_prb->impl_filter, res,
                 is_service_prim),
            WARN);
    if (res->state == SKIPPED) return OK;

    // Note: Graph may contain more than one operation with identical `dir`.
    //   It's required to collect all memory sizes regardless of `dir`.
    SAFE(collect_mem_size(res->mem_size_args, pdw, dir,
                 /* need_skip = */ !is_graph_ref),
            WARN);

    // The library scratchpad is allocated at create_primitive stage. The memory
    // check is moved after the creation stage. It's necessary to check the
    // library scratchpad size against gpu_max_alloc, otherwise, out_of_memory
    // would be issued by the library.
    if (res->mem_size_args.scratchpad_size > 0 && is_gpu()
            && query_scratchpad_mode(query_attr(pdw))
                    == dnnl_scratchpad_mode_library) {
        static size_t gpu_device_capacity = 0;
        static size_t gpu_max_alloc_capacity = 0;
        SAFE(get_gpu_ram_sizes(gpu_device_capacity, gpu_max_alloc_capacity),
                WARN);
        const bool fit
                = res->mem_size_args.scratchpad_size < gpu_max_alloc_capacity;
        if (!fit) {
            BENCHDNN_PRINT(1,
                    "[CHECK_MEM]: Size of the scratchpad %s "
                    "doesn't fit the allocation limit of %s.\n",
                    smart_bytes(res->mem_size_args.scratchpad_size).c_str(),
                    smart_bytes(gpu_max_alloc_capacity).c_str());
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_enough_ram;
            return OK;
        }
    }

    TIME_C_PRIM(DNN_SAFE(dnnl_primitive_create(&prim, pdw), WARN));
    primw.reset(prim);

    return OK;
}

int init_prim(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const init_pd_func_t &init_pd_func, const base_prb_t *base_prb,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint,
        bool is_service_prim) {
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> primw;

    // Verify that the problem is formed correctly. `invalid` means that there
    // might be incompatible settings that the library can verify only in
    // runtime, and it usually doesn't do that leading to unexpected results
    // which is hard to debug, e.g., inplace mode with different-sized data
    // types - it will lead to a crash or incorrect result.
    //
    // This function MUST NOT take care of cases that return `invalid_arguments`
    // status. The library must return this status for all incorrect API calls
    // and such cases must be updated on the user side.
    base_prb->skip_invalid(res);
    if (res->state == SKIPPED) return OK;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE

    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), FAIL);
    if (capacity > 0) {
        // The idea is to create the requested primitive twice using different
        // engines but the same device and context in the case of OpenCL and DPCPP.
        // Rationale: make sure that the primitive cache is robust in the case
        // where CPU and GPU engines are re-created because this is a commonly
        // used scenario in the frameworks.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        engine_t engine(get_test_engine(), /* recreate_on_copy = */ true);

        // The first primitive creation using a temporary engine.
        SAFE(create_primitive(primw, engine, init_pd_func, base_prb, res, dir,
                     hint, is_service_prim, /* src_md = */ nullptr,
                     /* force_f32_dt = */ false),
                WARN);
        if (res->state == SKIPPED) return OK;
    }

#endif
    // The second (if the cache is enabled) primitive creation using the global
    // test engine. This primitive is expected to come from the cache.
    SAFE(create_primitive(primw, get_test_engine(), init_pd_func, base_prb, res,
                 dir, hint, is_service_prim, /* src_md = */ nullptr,
                 /* force_f32_dt = */ false),
            WARN);
    if (res->state == SKIPPED) return OK;

    // Further checks are only for tested primitives.
    if (is_service_prim) {
        user_prim.reset(primw.release());
        return OK;
    }

    auto pd = query_pd(primw);
    res->impl_name = query_impl_info(pd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    // Collect memory footprint (perf report) for a given primitive descriptor.
    SAFE(get_memory_footprint(pd, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        // Check if adding attributes doesn't cause a fall back to another impl.
        SAFE(check_pd_w_and_wo_attr(
                     get_test_engine(), init_pd_func, base_prb, res, dir, hint),
                WARN);
        // Check if unexpected ref impl was hit.
        SAFE(check_ref_impl_hit(res), WARN);
    }

    user_prim.reset(primw.release());
    return res->state = INITIALIZED, OK;
}

int init_prim(const thr_ctx_t &thr_ctx,
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const init_pd_func_t &init_pd_func, const base_prb_t *base_prb,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint,
        bool is_service_prim) {
    int (*f)(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &,
            const init_pd_func_t &, const base_prb_t *, res_t *, dir_t,
            const_dnnl_primitive_desc_t, bool)
            = init_prim;
    std::function<int()> call_fn = std::bind(f, std::ref(user_prim),
            init_pd_func, base_prb, res, dir, hint, is_service_prim);
    return create_in_thr_ctx(thr_ctx, call_fn);
}

void check_correctness(const base_prb_t *base_prb,
        const std::vector<data_kind_t> &kinds, const args_t &args,
        const args_t &ref_args, const compute_ref_func_t &compute_ref_func,
        const setup_cmp_func_t &setup_cmp_func, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref) {
    // Fast exit for any modes but correctness.
    if (!has_bench_mode_bit(mode_bit_t::corr)) return;

    // Report prim_ref run status for easier distinguishing between GPU failures
    // and ref CPU failures.
    if (prim_ref) {
        BENCHDNN_PRINT(1, "run ref: %s\n", res->prim_ref_repro.c_str());
    } else {
        BENCHDNN_PRINT(8, "%s\n", "[NAIVE_REF]: Start");
    }

    TIME_REF(compute_ref_func(base_prb, dir, ref_args, prim_ref));

    // Forward-for-backward service primitives define `kinds` as empty to skip
    // validation. This is to avoid extra checks on higher level.
    if (kinds.empty()) return;

    for (int i = 0; i < args.size(); ++i) {
        TIME_COMPARE(check_zero_padding(args.dnn_mem(i), args.arg(i), res));
        TIME_COMPARE(check_buffer_overwrite(args.dnn_mem(i), args.arg(i), res));
    }

    for (const auto &kind : kinds) {
        compare::compare_t cmp;
        cmp.set_data_kind(kind);
        cmp.set_has_prim_ref(bool(prim_ref));
        setup_cmp_func(cmp, base_prb, kind, ref_args);

        int arg = data_kind2exec_arg(kind);
        assert(arg > 0);

        const auto &mem_dt = args.find(arg);
        const auto &mem_fp = ref_args.find(arg);

        TIME_COMPARE(cmp.compare(mem_fp, mem_dt, base_prb->attr, res));
    }

    if (prim_ref && res->state == FAILED) {
        static cpu_cache_args_t cpu_cache_args {};
        SAFE_V(get_cpu_cache_size(cpu_cache_args));

        BENCHDNN_PRINT(0,
                "[PRIM_REF][INFO]: L2_size:%zu bytes; per_core_L3_size:%zu "
                "bytes; nthr:%d; impl_name:%s\n",
                cpu_cache_args.L2_size, cpu_cache_args.L3_size,
                benchdnn_get_max_threads(),
                query_impl_info(query_pd(prim_ref)).c_str());
    }
}

int init_memory_args(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        dnnl_primitive_t prim, res_t *res, bool override_dir_with_fwd,
        const engine_t &test_engine) {
    const std::vector<int> supported_exec_args
            = base_prb->supported_exec_args(override_dir_with_fwd);

    // Backward case when forward is required will have mem_map not empty.
    // Remove all memories that are not in `supported_exec_args` to save on
    // initializing reference memories.
    //
    // Note: this code is pretty similar to the one in `erase_unused_args` but
    // with a slight change where the key is checked and bitwise correction.
    if (!mem_map.empty()) {
        // Collection of keys is required as evicting members along the way
        // invalidates references and makes traversing over the object
        // undefined behavior.
        std::vector<int> keys_to_erase;
        for (const auto &pair : mem_map) {
            const auto key = pair.first;
            bool key_found_in_exec_args = false;
            for (const auto &arg : supported_exec_args) {
                if (arg == key) {
                    key_found_in_exec_args = true;
                    break;
                }
            }
            // Don't remove stashed memory for bitwise validation.
            const bool bitwise_stash
                    = has_bench_mode_bit(mode_bit_t::bitwise) && key < 0;
            bool add_key_to_erase = !key_found_in_exec_args && !bitwise_stash;
            if (add_key_to_erase) keys_to_erase.push_back(key);
        }
        for (const auto &k : keys_to_erase) {
            mem_map.erase(k);
        }
    }

    auto const_pd = query_pd(prim);
    auto const_po = query_post_ops(const_pd);
    const auto prim_kind = query_prim_kind(const_pd);
    const auto prop_kind = query_prop_kind(const_pd);

    const auto has_runtime_dims = [](const_dnnl_memory_desc_t md) -> bool {
        for (int d = 0; d < query_md_ndims(md); ++d)
            if (query_md_dims(md)[d] == DNNL_RUNTIME_DIM_VAL) return true;
        return false;
    };

    if (prim_kind == dnnl_reorder) {
        auto src_engine = query_engine(const_pd, dnnl_query_reorder_src_engine);
        auto dst_engine = query_engine(const_pd, dnnl_query_reorder_dst_engine);
        const auto &src_md = query_md(const_pd, DNNL_ARG_FROM);
        const auto &dst_md = query_md(const_pd, DNNL_ARG_TO);
        if (has_runtime_dims(src_md)) {
            mem_map.emplace(DNNL_ARG_FROM,
                    dnn_mem_t(base_prb->get_md(DNNL_ARG_FROM), src_engine,
                            /* prefill = */ true));
            mem_map.emplace(DNNL_ARG_TO,
                    dnn_mem_t(base_prb->get_md(DNNL_ARG_TO), dst_engine,
                            /* prefill = */ true));
        } else {
            mem_map.emplace(DNNL_ARG_FROM,
                    dnn_mem_t(src_md, src_engine, /* prefill = */ true));
            mem_map.emplace(DNNL_ARG_TO,
                    dnn_mem_t(dst_md, dst_engine, /* prefill = */ true));
        }
    } else {
        for (const auto &exec_arg : supported_exec_args) {
            if (exec_arg == DNNL_ARG_MULTIPLE_SRC) {
                // `DNNL_ARG_MULTIPLE_SRC` corresponds to a pack of inputs.
                const auto n_inputs = query_n_inputs(const_pd);
                for (int i = 0; i < n_inputs; i++) {
                    const auto &md = query_md(const_pd, exec_arg + i);
                    mem_map.emplace(exec_arg + i,
                            dnn_mem_t(md, test_engine, /* prefill = */ true));
                }
            } else {
                const bool is_arg_in_map
                        = mem_map.find(exec_arg) != mem_map.end();
                const auto &md = query_md(const_pd, exec_arg);
                // Check for ndims is needed when the driver supported args map
                // contains extra arguments for other purposes.
                const int ndims = query_md_ndims(md);
                if (is_arg_in_map) {
                    // It may happen that the map already has the argument but
                    // the library requires it in a different format, e.g., RNN
                    // BWD support on GPU (for better performance). It may also
                    // happen in a combination with `no_ref_memory` modifier,
                    // which requires the library memories map to handle such
                    // cases.
                    if (ndims > 0
                            && dnnl_memory_desc_equal(
                                       md, mem_map.at(exec_arg).md_)
                                    == 0) {
                        assert(!has_runtime_dims(md));
                        dnn_mem_t new_mem(
                                md, test_engine, /* prefill = */ true);
                        // Reorder user's data from the old memory to the new one.
                        SAFE(new_mem.reorder(mem_map.at(exec_arg), res), WARN);
                        mem_map[exec_arg] = std::move(new_mem);
                    }
                } else {
                    if (has_runtime_dims(md)) {
                        mem_map.emplace(exec_arg,
                                dnn_mem_t(base_prb->get_md(exec_arg),
                                        test_engine,
                                        /* prefill = */ true));
                    } else {
                        // In case when arguments get updated on backward when
                        // forward is required, `emplace` guarantees newly
                        // constructed element will be destroyed if an element
                        // with a key already present in the map. C++17 could
                        // use try_emplace instead to mitigate
                        // construction/destruction overhead.
                        mem_map.emplace(exec_arg,
                                dnn_mem_t(
                                        md, test_engine, /* prefill = */ true));
                    }
                }
            }
        }
    }

    // Drop "destination" memory for in-place case. `args` will take care of
    // setting proper pointers to make in-place mode happen.
    // Note: must precede bitwise stash memory insertion to keep numbers
    // estimated by memory checker correct.
    if (base_prb->inplace) {
        const bool inplace_fwd = (base_prb->dir & FLAG_FWD);
        const bool inplace_bwd
                = (base_prb->dir & FLAG_BWD) && !is_fwd_prop_kind(prop_kind);
        if (inplace_fwd || inplace_bwd) {
            const int inplace_dst_arg = (base_prb->dir & FLAG_FWD)
                    ? DNNL_ARG_DST
                    : DNNL_ARG_DIFF_SRC;
            mem_map[inplace_dst_arg] = dnn_mem_t();
        }
    }

    // Bitwise mode demands exactly the same inputs between two runs. There are
    // certain scenarios that affect original memory objects content. When such
    // scenarios occur, memory objects have their original content overwritten.
    // The logic below stashes additional memory objects for a copy of data
    // which will get reordered before the second run.
    //
    // An implementation detail:
    // All such memory objects' counterparts are created with the same arg value
    // but with a negative sign. This is the only guaranteed value that is not
    // used by the library.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // A sum post-op has the destination memory data overwritten by the
        // accumulation memory. An in-place binary post-op reads the destination
        // as its right-hand side, thus it's affected the very same way.
        if (query_post_ops_has_kind(const_po, dnnl_sum)
                || query_post_ops_has_binary_alg_kind(
                        const_po, dnnl_binary_mul_inplace)) {
            const int query_arg = DNNL_ARG_DST;
            const int insert_arg = -query_arg;
            const auto &md = query_md(const_pd, query_arg);
            if (has_runtime_dims(md)) {
                mem_map.emplace(insert_arg,
                        dnn_mem_t(base_prb->get_md(query_arg), test_engine,
                                /* prefill = */ true));
            } else {
                mem_map.emplace(insert_arg,
                        dnn_mem_t(md, test_engine, /* prefill = */ true));
            }
        }

        // An inplace mode uses the source memory object as the destination one.
        // It results in the source is overwritten after the operation is done.
        if (base_prb->inplace) {
            const bool has_multiple_args = std::any_of(
                    supported_exec_args.begin(), supported_exec_args.end(),
                    [](int arg) { return arg == DNNL_ARG_MULTIPLE_SRC; });
            const auto query_arg = is_fwd_prop_kind(prop_kind)
                    ? (has_multiple_args ? DNNL_ARG_MULTIPLE_SRC : DNNL_ARG_SRC)
                    : DNNL_ARG_DIFF_DST;
            const int insert_arg = -query_arg;
            const auto &md = query_md(const_pd, query_arg);
            if (has_runtime_dims(md)) {
                mem_map.emplace(insert_arg,
                        dnn_mem_t(base_prb->get_md(query_arg), test_engine,
                                /* prefill = */ true));
            } else {
                mem_map.emplace(insert_arg,
                        dnn_mem_t(md, test_engine, /* prefill = */ true));
            }
        }
    }

    const auto &scratch_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);
    mem_map.emplace(DNNL_ARG_SCRATCHPAD,
            dnn_mem_t(scratch_md, test_engine, /* prefill = */ true));

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    // Grouped max variable dim hint
    // Optional host scalar, created when src is a grouped descriptor
    if (has_grouped_encoding(query_md(const_pd, DNNL_ARG_SRC))) {
        auto hint_md = dnn_mem_t::init_host_scalar_md(dnnl_s32);
        mem_map.emplace(DNNL_ARG_HINT_MAX_GROUP_SIZE, dnn_mem_t(hint_md));
    }
#endif

    // Binary post-op.
    // TODO: currently run-time dimensions are not supported in binary post-op.
    for (int idx = 0; idx < dnnl_post_ops_len(const_po); ++idx) {
        if (dnnl_post_ops_get_kind(const_po, idx) != dnnl_binary) continue;

        int po_arg1 = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;

        // An in-place post-op reads the destination as its right-hand side.
        // Drop its "source" memory the same way the in-place mode drops the
        // destination one; `args` will take care of setting proper pointers.
        if (query_post_ops_has_binary_alg_kind(
                    const_po, dnnl_binary_mul_inplace, idx)) {
            mem_map[po_arg1] = dnn_mem_t();
            continue;
        }

        const auto &po_md1 = query_md(const_pd, po_arg1);
        mem_map.emplace(
                po_arg1, dnn_mem_t(po_md1, test_engine, /* prefill = */ true));

        if (!query_post_ops_has_binary_alg_kind(
                    const_po, dnnl_binary_select, idx))
            continue;
        int po_arg2 = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_2;
        const auto &po_md2 = query_md(const_pd, po_arg2);
        mem_map.emplace(
                po_arg2, dnn_mem_t(po_md2, test_engine, /* prefill = */ true));
    }

    // Prelu post-op.
    for (int idx = 0; idx < dnnl_post_ops_len(const_po); ++idx) {
        if (dnnl_post_ops_get_kind(const_po, idx) != dnnl_prelu) continue;

        const auto &orig_dst_md = query_md(const_pd, DNNL_ARG_DST);
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> prb_dst_md;
        if (has_runtime_dims(orig_dst_md)) {
            prb_dst_md = base_prb->get_md(DNNL_ARG_DST);
        }
        const auto &dst_md = prb_dst_md ? prb_dst_md : orig_dst_md;

        const auto ndims = query_md_ndims(dst_md);
        int mask = 0;
        dnnl_post_ops_get_params_prelu(const_po, idx, &mask);

        // Deduce prelu weights dims based on input policy.
        dims_t dims = md2dims(dst_md, mask);

        int po_arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_WEIGHTS;
        mem_map.emplace(po_arg,
                dnn_mem_t(ndims, dims.data(), dnnl_f32, tag::axb, test_engine,
                        /* prefill = */ true));
    }

    // Dropout
    if (is_fwd_training(prop_kind) && !base_prb->attr.dropout.is_def()) {
        const auto &dropout_md = query_md(const_pd, DNNL_ARG_ATTR_DROPOUT_MASK);
        mem_map.emplace(DNNL_ARG_ATTR_DROPOUT_MASK,
                dnn_mem_t(dropout_md, test_engine, /* prefill = */ true));

        if (base_prb->attr.dropout.use_host_scalars) {
            auto prob_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
            mem_map.emplace(
                    DNNL_ARG_ATTR_DROPOUT_PROBABILITY, dnn_mem_t(prob_md));
            auto seed_md = dnn_mem_t::init_host_scalar_md(dnnl_s64);
            mem_map.emplace(DNNL_ARG_ATTR_DROPOUT_SEED, dnn_mem_t(seed_md));
            if (base_prb->attr.dropout.offset != 0) {
                auto offset_md = dnn_mem_t::init_host_scalar_md(dnnl_s64);
                mem_map.emplace(
                        DNNL_ARG_ATTR_DROPOUT_OFFSET, dnn_mem_t(offset_md));
            }
        } else {
            int64_t count = 1;
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> prob_md;
            DNN_SAFE(dnn_mem_t::init_md(prob_md, 1, &count, dnnl_f32, tag::abx),
                    WARN);
            mem_map.emplace(DNNL_ARG_ATTR_DROPOUT_PROBABILITY,
                    dnn_mem_t(prob_md, test_engine, /* prefill = */ true));
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> seed_md;
            DNN_SAFE(dnn_mem_t::init_md(seed_md, 1, &count, dnnl_s64, tag::abx),
                    WARN);
            mem_map.emplace(DNNL_ARG_ATTR_DROPOUT_SEED,
                    dnn_mem_t(seed_md, test_engine, /* prefill = */ true));
            if (base_prb->attr.dropout.offset != 0) {
                benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> offset_md;
                DNN_SAFE(dnn_mem_t::init_md(
                                 offset_md, 1, &count, dnnl_s64, tag::abx),
                        WARN);
                mem_map.emplace(DNNL_ARG_ATTR_DROPOUT_OFFSET,
                        dnn_mem_t(
                                offset_md, test_engine, /* prefill = */ true));
            }
        }
    }

    // Scales.
    if (!base_prb->attr.scales.is_def()) {
        const auto &sc = base_prb->attr.scales;

        const auto &src_md = query_md(const_pd, DNNL_ARG_SRC);
        const auto &wei_md = query_md(const_pd, DNNL_ARG_WEIGHTS);
        const bool has_channel_groups
                = (query_md_ndims(src_md) + 1) == query_md_ndims(wei_md);

        const auto append_scales = [&](int exec_arg) {
            const int exec_sc_arg = DNNL_ARG_ATTR_SCALES | exec_arg;
            dims_t dims = {};
            int64_t ndims = 1;
            const auto &arg_md = query_md(const_pd, exec_arg);
            const auto mask = sc.get_mask(exec_arg, prim_kind,
                    query_md_ndims(arg_md), has_channel_groups);
            const auto &groups = sc.get(exec_arg).groups;

            if (mask > 0) {
                const auto &md = query_md(const_pd, exec_arg);
                if (has_runtime_dims(md)) {
                    const auto prb_md = base_prb->get_md(exec_arg);
                    dims = md2dims(prb_md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                } else {
                    dims = md2dims(md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                }
            } else {
                dims = {1};
                ndims = 1;
            }

            const auto dt = sc.get(exec_arg).dt;
            const auto policy = sc.get(exec_arg).policy;

            if (policy == attr_t::policy_t::HOST_SCALAR) {
                auto scales_md = dnn_mem_t::init_host_scalar_md(dt);
                mem_map.emplace(exec_sc_arg, dnn_mem_t(scales_md));
                return OK;
            }

            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> scales_md;
            DNN_SAFE(dnn_mem_t::init_md(
                             scales_md, ndims, dims.data(), dt, tag::abx),
                    WARN);
            mem_map.emplace(exec_sc_arg,
                    dnn_mem_t(scales_md, test_engine, /* prefill = */ true));
            return OK;
        };

        for (const auto &exec_arg : supported_exec_args) {
            if (exec_arg == DNNL_ARG_MULTIPLE_SRC) {
                // `DNNL_ARG_MULTIPLE_SRC` corresponds to a pack of inputs.
                const auto n_inputs = query_n_inputs(const_pd);
                for (int i = 0; i < n_inputs; i++) {
                    const auto i_exec_arg = exec_arg + i;
                    if (!sc.is_def(i_exec_arg))
                        SAFE(append_scales(i_exec_arg), WARN);
                }
            } else {
                if (!sc.is_def(exec_arg)) SAFE(append_scales(exec_arg), WARN);
            }
        }
    }

    // Zero points.
    if (!base_prb->attr.zero_points.is_def()) {
        const auto &zp = base_prb->attr.zero_points;

        const auto append_zero_points = [&](int exec_arg) {
            const int exec_zp_arg = DNNL_ARG_ATTR_ZERO_POINTS | exec_arg;
            const auto &e = zp.get(exec_arg);
            int64_t ndims = 1;
            dims_t dims = {};
            const auto &arg_md = query_md(const_pd, exec_arg);
            const auto mask
                    = zp.get_mask(exec_arg, prim_kind, query_md_ndims(arg_md));
            const auto &groups = e.groups;

            if (mask > 0) {
                const auto &md = query_md(const_pd, exec_arg);
                if (has_runtime_dims(md)) {
                    const auto prb_md = base_prb->get_md(exec_arg);
                    dims = md2dims(prb_md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                } else {
                    dims = md2dims(md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                }
            } else {
                dims = {1};
                ndims = 1;
            }

            if (e.policy == attr_t::policy_t::HOST_SCALAR) {
                auto zp_md = dnn_mem_t::init_host_scalar_md(e.dt);
                mem_map.emplace(exec_zp_arg, dnn_mem_t(zp_md));
                return OK;
            }

            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> zp_md;
            DNN_SAFE(dnn_mem_t::init_md(
                             zp_md, ndims, dims.data(), e.dt, tag::abx),
                    WARN);
            mem_map.emplace(exec_zp_arg,
                    dnn_mem_t(zp_md, test_engine, /* prefill = */ true));
            return OK;
        };

        for (const auto &exec_arg : supported_exec_args) {
            if (exec_arg == DNNL_ARG_MULTIPLE_SRC) {
                // `DNNL_ARG_MULTIPLE_SRC` corresponds to a pack of inputs.
                const auto n_inputs = query_n_inputs(const_pd);
                for (int i = 0; i < n_inputs; i++) {
                    const auto i_exec_arg = exec_arg + i;
                    if (!zp.is_def(i_exec_arg))
                        SAFE(append_zero_points(i_exec_arg), WARN);
                }
            } else {
                if (!zp.is_def(exec_arg))
                    SAFE(append_zero_points(exec_arg), WARN);
            }
        }
    }

    // Precomputed reductions.
    if (!base_prb->attr.precomputed_reductions.is_def()) {
        const auto &pr = base_prb->attr.precomputed_reductions;

        const auto append_precomputed_reductions = [&](int exec_arg) {
            const int exec_pr_arg
                    = DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS | exec_arg;
            const auto &e = pr.get(exec_arg);
            int64_t ndims = 1;
            dims_t dims = {};
            const auto &arg_md = query_md(const_pd, exec_arg);
            const auto mask
                    = pr.get_mask(exec_arg, prim_kind, query_md_ndims(arg_md));
            const auto &groups = e.groups;

            if (mask > 0) {
                const auto &md = query_md(const_pd, exec_arg);
                if (has_runtime_dims(md)) {
                    const auto prb_md = base_prb->get_md(exec_arg);
                    dims = md2dims(prb_md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                } else {
                    dims = md2dims(md, mask, false, groups);
                    ndims = static_cast<int>(dims.size());
                }
            } else {
                dims = {1};
                ndims = 1;
            }
            benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> pr_md;
            DNN_SAFE(dnn_mem_t::init_md(
                             pr_md, ndims, dims.data(), e.dt, tag::abx),
                    WARN);
            mem_map.emplace(exec_pr_arg,
                    dnn_mem_t(pr_md, test_engine, /* prefill = */ true));
            return OK;
        };

        for (const auto &exec_arg : supported_exec_args) {
            if (exec_arg == DNNL_ARG_MULTIPLE_SRC) {
                // `DNNL_ARG_MULTIPLE_SRC` corresponds to a pack of inputs.
                const auto n_inputs = query_n_inputs(const_pd);
                for (int i = 0; i < n_inputs; i++) {
                    const auto i_exec_arg = exec_arg + i;
                    if (!pr.is_def(i_exec_arg))
                        SAFE(append_precomputed_reductions(i_exec_arg), WARN);
                }
            } else {
                if (!pr.is_def(exec_arg))
                    SAFE(append_precomputed_reductions(exec_arg), WARN);
            }
        }
    }

    // rounding mode
    if (!base_prb->attr.rounding_mode.is_def()) {
        int64_t count = 1;
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> seed_md;
        DNN_SAFE(dnn_mem_t::init_md(seed_md, 1, &count, dnnl_s32, tag::abx),
                WARN);
        mem_map.emplace(DNNL_ARG_ATTR_ROUNDING_SEED,
                dnn_mem_t(seed_md, test_engine, /* prefill = */ true));
    }

    return OK;
}

int init_prim_ref_common(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const base_prb_t *base_prb_cpu, res_t *res,
        const init_pd_func_t &init_pd_func) {

    init_pd_args_t init_pd_args(
            /* res = */ nullptr, get_cpu_engine(), base_prb_cpu,
            base_prb_cpu->dir,
            /* hint = */ nullptr, /* src_md = */ nullptr);
    init_pd_func(init_pd_args);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;
    // `is_service_prim=true` prevents from filtering the implementation
    // by name which is intended through a `get_prim_ref_impl_filter()`.
    // As `fetch_impl` doesn't have any further logic related to it, it's
    // safe to set it to `false`.
    fetch_impl(pdw, init_pd_args, get_prim_ref_impl_filter(),
            /* res = */ nullptr,
            /* is_service_prim = */ false);

    // Prim desc wasn't created - try the next set...
    if (!pdw) return FAIL;

    dnnl_primitive_t prim_ref_ptr {};
    auto st = dnnl_primitive_create(&prim_ref_ptr, pdw);
    // Primitive wasn't created - try the next set...
    if (st != dnnl_success) return FAIL;

    BENCHDNN_PRINT(5, "CPU reference oneDNN implementation: %s\n",
            query_impl_info(pdw).c_str());

    res->prim_ref_repro = base_prb_cpu->str();
    // Replace engine kind for repro line from GPU to CPU.
    const auto eng_pos = res->prim_ref_repro.find("engine=gpu");
    if (eng_pos != std::string::npos) {
        // Replace `g` in `gpu` with `c`
        res->prim_ref_repro[eng_pos + 7] = 'c';
    }

    // Remove `--impl=XXX` as it doesn't affect prim_ref.
    const auto impl_pos = res->prim_ref_repro.find("--impl=");
    if (impl_pos != std::string::npos) {
        // Search for the next space starting from `impl_pos` as names' length
        // is variadic.
        const auto end_impl_pos
                = res->prim_ref_repro.find_first_of(" ", impl_pos);
        assert(end_impl_pos != std::string::npos);
        // `+ 1` is for extra space.
        res->prim_ref_repro.erase(impl_pos, end_impl_pos - impl_pos + 1);
    }

    prim_ref.reset(prim_ref_ptr);
    return OK;
}
