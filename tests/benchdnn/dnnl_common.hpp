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

#ifndef DNNL_COMMON_HPP
#define DNNL_COMMON_HPP

#include <functional>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"
#include "utils/dims.hpp"
#include "utils/engine.hpp"
#include "utils/fill.hpp"
#include "utils/prb.hpp"

#ifndef DNNL_EXPERIMENTAL_PROFILING
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
using dnnl_profiling_data_kind_t = int;
extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
extern "C" dnnl_status_t dnnl_query_profiling_data(dnnl_stream_t stream,
        dnnl_profiling_data_kind_t data_kind, int *num_entries, uint64_t *data);
#endif
#endif

int check_pd_cache(const_dnnl_primitive_desc_t pd, res_t *res);
int check_primitive_cache(dnnl_primitive_t p, res_t *res);

extern isa_hints_t hints;
extern int default_num_streams;
extern int num_streams;

bool is_f64_supported(const engine_t &engine = get_test_engine());

// Extended version of dnnl_sycl_interop_memory_kind_t enumeration.
enum class memory_kind_ext_t {
    usm, // Same as dnnl_sycl_interop_usm
    buffer, // Same as dnnl_sycl_interop_buffer
    usm_device, // USM allocated via malloc_device()
    usm_shared, // USM allocated via malloc_shared()
};

const memory_kind_ext_t default_memory_kind = memory_kind_ext_t::usm;

extern memory_kind_ext_t memory_kind;

void init_isa_settings();

struct args_t {
    args_t() = default;
    args_t(const dnn_mem_map_t &mem_map);

    args_t &set(int arg, const dnn_mem_t &mem);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    const dnn_mem_t &find(int arg) const;
    // Used in graph to link arguments together by updating current source with
    // previous destination.
    void replace(int arg, const dnn_mem_t *mem);

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

struct init_pd_args_t {
    init_pd_args_t(res_t *res, dnnl_engine_t engine, const base_prb_t *base_prb,
            dir_t dir, const_dnnl_primitive_desc_t hint,
            const_dnnl_memory_desc_t src_md, bool force_f32_dt)
        : pd(nullptr)
        , is_iterator_supported(true)
        , res(res)
        , engine(engine)
        , base_prb(base_prb)
        , dir(dir)
        , hint(hint)
        , src_md(src_md)
        , force_f32_dt(force_f32_dt) {}

    init_pd_args_t(res_t *res, dnnl_engine_t engine, const base_prb_t *base_prb,
            dir_t dir, const_dnnl_primitive_desc_t hint,
            const_dnnl_memory_desc_t src_md)
        : init_pd_args_t(res, engine, base_prb, dir, hint, src_md, false) {}

    // Output members
    dnnl_primitive_desc_t pd;

    bool is_iterator_supported;

    // Input members
    res_t *res;
    dnnl_engine_t engine;
    const base_prb_t *base_prb;
    // Used to specify the prop_kind of the pd. Required for double-run drivers
    // to differentiate between fwd-for-bwd pd and actual bwd pd.
    dir_t dir;
    const_dnnl_primitive_desc_t hint;
    // Use for memory propagation between pd. Nullptr will ignore the setting.
    const_dnnl_memory_desc_t src_md;
    // When `true`, overrides prb data type with f32 for ALL memory descriptors
    // when creating pd objects.
    bool force_f32_dt;
};

using init_pd_func_t = std::function<dnnl_status_t(init_pd_args_t &)>;

int check_total_size(res_t *res, dnnl_primitive_t prim_ref = nullptr);
bool is_fwd_training(dnnl_prop_kind_t prop_kind);
bool is_fwd_prop_kind(dnnl_prop_kind_t prop_kind);
int get_memory_footprint(const_dnnl_primitive_desc_t pd, res_t *res);
int check_same_pd(const dnnl_primitive_desc_t &pd_no_attr, res_t *res);
int test_persistent_cache_api(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim, res_t *res);
// This call is used in zeropad only and still does check inside, too.
int check_mem_size(const_dnnl_memory_desc_t md, res_t *res);
// Only collects memory sizes from an input `const_pd` and puts the result into
// `mem_size_args`.
int collect_mem_size(check_mem_size_args_t &mem_size_args,
        const_dnnl_primitive_desc_t const_pd, dir_t dir, bool need_skip = true);

bool should_stop(const timer::timer_t &t);

void skip_unimplemented_data_type(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res);
void skip_unimplemented_sum_po(const attr_t &attr, res_t *res,
        dnnl_primitive_kind_t pkind, dnnl_data_type_t src_dt,
        dnnl_data_type_t dst_dt = dnnl_data_type_undef);
void skip_unimplemented_binary_po(const attr_t &attr, res_t *res);
void skip_unimplemented_prelu_po(
        const attr_t &attr, res_t *res, dnnl_primitive_kind_t pkind);
void skip_invalid_inplace(res_t *res, dnnl_data_type_t sdt,
        dnnl_data_type_t ddt, const std::string &stag, const std::string &dtag);
void skip_unimplemented_arg_scale(const attr_t &attr, res_t *res);

int check_caches(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &primw,
        const thr_ctx_t &ctx_init, res_t *res);

// `check_dnnl_status` function is called to validate the result of primitive
// descriptor creation. Based on the status, it produces additional checks:
// * For `invalid_arguments` it just updates the `res` object with it.
// * For `unimplemented` it checks whether the lack of support is expected or
//   not. It relies on the `skip_unimplemented` virtual method overridden by
//   every driver's `prb_t` to dispatch to its driver-specific logic. If the
//   case is unknown, `UNIMPLEMENTED` status will be returned.
int check_dnnl_status(
        dnnl_status_t status, const base_prb_t *base_prb, res_t *res);

// Moves `init_pd_args.pd` into `pdw`, iterating to the impl `impl_filter`
// picks; `pdw` is empty if every impl is skipped.
int fetch_impl(benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> &pdw,
        init_pd_args_t &init_pd_args, const impl_filter_t &impl_filter,
        res_t *res, bool is_service_prim);

// This is an internal to `init_prim` function that utilizes the logic of
// creating a `pd` and `prim` and assign them to input wrappers. It allows to
// remove code duplication and keep all the logic in a single place.
int create_primitive(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &primw,
        dnnl_engine_t engine, const init_pd_func_t &init_pd_func,
        const base_prb_t *base_prb, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint, bool is_service_prim,
        const_dnnl_memory_desc_t src_md, bool force_f32_dt,
        bool is_graph_ref = false);

int init_prim(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const init_pd_func_t &init_pd_func, const base_prb_t *base_prb,
        res_t *res, dir_t dir = FLAG_FWD,
        const_dnnl_primitive_desc_t hint = nullptr,
        bool is_service_prim = false);

int init_prim(const thr_ctx_t &thr_ctx,
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const init_pd_func_t &init_pd_func, const base_prb_t *base_prb,
        res_t *res, dir_t dir = FLAG_FWD,
        const_dnnl_primitive_desc_t hint = nullptr,
        bool is_service_prim = false);

// `setup_cmp_func` function is defined in every driver.
// It takes:
// * A reference to a `compare_t` object which the function modifies based on
//   driver's needs.
// * A pointer to a `base_prb_t` problem.
// * `data_kind` value to help to setup threshold depending on output argument.
// * Driver's reference memory arguments since some drivers can't validate
//   certain scenarios for sure without additional memory arguments.
// Returns nothing since the object is modified by reference due to lifetime of
// the compare object is controlled by `check_correctness`.
using setup_cmp_func_t = std::function<void(
        compare::compare_t &, const base_prb_t *, data_kind_t, const args_t &)>;

using compute_ref_func_t = std::function<void(
        const base_prb_t *, dir_t, const args_t &, dnnl_primitive_t)>;

// `check_correctness` function is designed to be called from every driver where
// correctness validation is needed. It takes:
// * A pointer to a `prb_t` problem.
// * A vector of kinds to compare, to validate several outputs, if applicable.
// * Backend arguments to compare the output.
// * Driver's reference memory arguments to compute the reference path, then
//   setup a compare object, and, finally, compare the output.
// * A reference to function that sets up the compare object, see description
//   below.
// * A pointer to a `res_t` structure to update validation status.
// * An optional pointer to CPU primitive for speeding up reference path
//   computation on GPU.
//
// The function doesn't return status since we rely on `res` to contain all
// necessary information about validation results.
//
// The function performs several validation steps:
// * Checks that padded area of all memories are properly zeroed.
// * Checks that GPU backend haven't modified out-of-boundary memory regions.
// * Executes driver's reference path, using the problem, driver reference
//   arguments, and CPU primitive for GPU backend, if available.
// * For each kind to validate it:
//   - Creates and sets up the compare object. Setting is done with
//     `setup_cmp_func` (see above).
//   - Finds correspondent memory arguments from backend and reference and
//     compares them.
//   - Result of comparison is saved into `res` object.
void check_correctness(const base_prb_t *base_prb,
        const std::vector<data_kind_t> &kinds, const args_t &args,
        const args_t &ref_args, const compute_ref_func_t &compute_ref_func,
        const setup_cmp_func_t &setup_cmp_func, res_t *res, dir_t dir,
        dnnl_primitive_t prim_ref = nullptr);

using perf_function_t = std::function<dnnl_status_t(
        const dnnl_stream_t &, const std::vector<dnnl_exec_arg_t> &)>;

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res = nullptr);
int execute_and_wait(
        dnnl_primitive_t prim, const args_t &args, res_t *res = nullptr);

int run_execution(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res = nullptr);
int run_execution(dnnl_primitive_t prim, const args_t &args, res_t *res);

void reset_gpu_profiling(dnnl_stream_t stream);

void finalize();

int get_gpu_profiling_info(dnnl_stream_t stream, std::vector<uint64_t> &nsecs,
        std::vector<uint64_t> &cycles, int expected_num_entries);
int measure_perf(const thr_ctx_t &ctx, res_t *res, perf_function_t &perf_func,
        args_t &args);
int measure_perf(
        const thr_ctx_t &ctx, res_t *res, dnnl_primitive_t prim, args_t &args);

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off, int64_t group_id = 0);

bool check_md_consistency_with_tag(
        const_dnnl_memory_desc_t md, const std::string &tag);

memory_kind_ext_t str2memory_kind(const char *str);

float reorder_rescale_factor();

// The function converts a memory descriptor dims into a `dims_t` object under
// certain rules.
//
// `mask` argument picks what dimensions to put into a new object as is.
// `extend_by_ones` specifies the behavior with dimensions not matched by
//     `mask`. When set to `true` (the default), a dim value of `1` is used
//     for a not matched dimension. Thus, `ndims` of a new object will remain
//     the same as for original md. When set to `false`, a dim is skipped and
//     the final object could end up with smaller `ndims` (or `size()`) value.
// `groups` specify a vector of group values which decrease the final dimension
//     values by dividing on the group size.
dims_t md2dims(const_dnnl_memory_desc_t md, int mask = -1,
        bool extend_by_ones = true, const std::vector<int64_t> &groups = {});

// Function adjusts data type if fpmath mode is present or sum_dt is different
// from destination_dt. It is used in `cfg` objects that regulate filling.
dnnl_data_type_t deduce_cfg_data_type(
        dnnl_data_type_t in_dt, const attr_t &attr, data_kind_t dk);

// `init_memory_args` is responsible for:
// * Constructing all necessary `dnn_mem_t` objects needed by the library
//   primitive for the main operation and attributes.
//   All these memories must utilize `prefill_memory` flag of `dnn_mem_t` ctor
//   to verify reorders are working correctly and output memory was updated
//   completely.
// * Stashing them with a proper exec_arg ID in a `mem_map` object.
// Caller is responsible for constructing reference memories and filling both
// the library and reference memories by calling `init_ref_memory_args`.
//
// Note: unordered_map is taken over std::vector because vector invalidates its
// references once the object emplaced due to memory re-allocations happening
// internally, while map doesn't not invalidate its references when adding a new
// element which simplifies an implementation.
int init_memory_args(dnn_mem_map_t &mem_map, const base_prb_t *base_prb,
        dnnl_primitive_t prim, res_t *res, bool override_dir_with_fwd = false,
        const engine_t &test_engine = get_test_engine());

void erase_unused_args(
        dnn_mem_map_t &ref_mem_map, const dnn_mem_map_t &mem_map);

void get_kinds_to_check_shared(
        std::vector<data_kind_t> &check_kinds, const attr_t &attr);

int update_ref_mem_map_from_prim(dnnl_primitive_t prim_ref,
        const dnn_mem_t &library_mem, dnn_mem_map_t &ref_mem_map, int exec_arg,
        dnnl_data_type_t swapped_dt, res_t *res);

int init_ref_memory_args_default_case(int exec_arg, dnn_mem_t &mem,
        dnn_mem_t &ref_mem, const attr_t &attr, res_t *res,
        const std::unordered_map<int, fill_cfg_t> &fill_cfg_map = {});

int check_bitwise(dnnl_primitive_t prim, const std::vector<data_kind_t> &kinds,
        const args_t &args, const attr_t &attr, bool inplace, res_t *res);

int init_prim_ref_common(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const base_prb_t *base_prb_cpu, res_t *res,
        const init_pd_func_t &init_pd_func);

#endif
