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

#include <cassert>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.h"
#endif

#include "common/cache_hit_types.hpp"
#include "common/stream.hpp"
#include "common/verbose.hpp"

#include "graph/interface/allocator.hpp"
#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/partition.hpp"
#include "graph/interface/partition_cache.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#endif

using dnnl::impl::cache_state2str;
using dnnl::impl::cache_state_t;
using namespace dnnl::impl::graph;

/// This allows to create a partition directly with an op and an engine kind. In
/// order to not break backend API and change the existing graph and partition
/// implementation, we internally construct a temporal graph object, add the
/// operator to it, and then do partitioning on the graph. The workflow should
/// be the same as partitioning a normal user graph.
status_t dnnl_graph_partition_create_with_op(
        partition_t **partition, const op_t *op, engine_kind_t ekind) {
    using ltw = logical_tensor_wrapper_t;

    if (utils::any_null(partition, op)) return status::invalid_arguments;

    // new an empty partition
    *partition = new partition_t();

    status_t ret = status::success;

    // construct a single op graph
    graph_t g {ekind};
    ret = g.add_op(op);
    g.finalize();

    if (ret != status::success) return ret;

    // find opaque logical tensor in inputs
    const auto &input_vals = op->get_input_values();
    auto opaque_in_iter = std::find_if(input_vals.begin(), input_vals.end(),
            [](const std::shared_ptr<value_t> &it) {
        return ltw(it->get_logical_tensor()).is_opaque();
    });
    // find opaque layout tensors in outputs
    const auto &output_vals = op->get_output_values();
    auto opaque_out_iter = std::find_if(output_vals.begin(), output_vals.end(),
            [](const std::shared_ptr<value_t> &it) {
        return ltw(it->get_logical_tensor()).is_opaque();
    });

    // Case 1: all input/outputs are not opaque logical tensors. We need go
    // through all registered backends to get partitions
    if (opaque_in_iter == input_vals.end()
            && opaque_out_iter == output_vals.end()) {
        // get partition impl. by calling each backend
        std::vector<const backend_t *> &backends
                = backend_registry_t::get_singleton().get_registered_backends();
        for (const auto &cbkd : backends) {
            backend_t *bkd = const_cast<backend_t *>(cbkd);
            ret = bkd->get_partitions(g, partition_policy::fusion);
            if (ret != status::success) return ret;
        }
    } else {
        // Case 2: if input/output logical tensors have already embedded with
        // backend ID (e.g opaque layout), here we directly use the same backend
        // to get partitions
        bool in_has_valid_layout_id = opaque_in_iter != input_vals.end();
        bool out_has_valid_layout_id = opaque_out_iter != output_vals.end();
        size_t in_valid_layout_id = in_has_valid_layout_id
                ? ltw((*opaque_in_iter)->get_logical_tensor()).layout_id()
                : std::numeric_limits<size_t>::max();
        size_t out_valid_layout_id = out_has_valid_layout_id
                ? ltw((*opaque_out_iter)->get_logical_tensor()).layout_id()
                : std::numeric_limits<size_t>::max();
        if (in_has_valid_layout_id && out_has_valid_layout_id) {
            size_t in_backend_id = backend_registry_t::extract_backend_id(
                    in_valid_layout_id);
            size_t out_backend_id = backend_registry_t::extract_backend_id(
                    out_valid_layout_id);
            // input and output logical tensor have different backend IDs
            if (in_backend_id != out_backend_id) {
                assertm(false, "backends mismatch between inputs and outputs");
                return status::unimplemented;
            }
        }
        size_t valid_layout_id = in_has_valid_layout_id ? in_valid_layout_id
                                                        : out_valid_layout_id;
        backend_t *bkd = const_cast<backend_t *>(
                backend_registry_t::get_singleton().get_registered_backend(
                        valid_layout_id));
        assertm(bkd != nullptr,
                "backend is not valid since layout id maybe not correct.");
        ret = bkd->get_partitions(g, partition_policy::fusion);
        if (ret != status::success) return ret;
    }

    // check the partition impl.
    auto &partition_vec = g.get_partitions();
    assertm(partition_vec.size() == 1,
            "single op graph should contain only one partition");
    if (partition_vec[0]->get_assigned_backend() == nullptr) {
        return status::invalid_graph;
    }

    // wrap into the partition
    std::vector<partition_t *> parts {*partition};
    g.get_ordered_partitions(parts);
    return ret;
}

status_t dnnl_graph_partition_destroy(partition_t *partition) {
    delete partition;
    return status::success;
}

status_t dnnl_graph_partition_get_op_num(
        const partition_t *partition, size_t *num) {
    if (utils::any_null(partition, num)) return status::invalid_arguments;

    *num = partition->num_ops();
    return status::success;
}

status_t dnnl_graph_partition_get_ops(
        partition_t *partition, size_t num, size_t *ops) {
    if (utils::any_null(partition, ops)) { return status::invalid_arguments; }

    auto ids = partition->get_op_ids();
    if (ids.size() != num) { return status::invalid_arguments; }

    int idx = 0;
    for (auto it = ids.begin(); it != ids.end(); ++it, ++idx) {
        ops[idx] = *it;
    }

    return status::success;
}

status_t dnnl_graph_partition_get_id(const partition_t *partition, size_t *id) {
    if (utils::any_null(partition, id)) { return status::invalid_arguments; }

    *id = partition->id();
    return status::success;
}

status_t dnnl_graph_partition_compile(partition_t *partition,
        compiled_partition_t *compiled_partition, size_t in_num,
        const logical_tensor_t **inputs, size_t out_num,
        const logical_tensor_t **outputs, engine_t *engine) {
    if (utils::any_null(partition, compiled_partition, engine)) {
        return status::invalid_arguments;
    }

    if (!partition->is_supported()) return status::invalid_arguments;

    std::vector<const logical_tensor_t *> in {inputs, inputs + in_num};
    std::vector<const logical_tensor_t *> out {outputs, outputs + out_num};

    // The cache_state_t in the pair will track whether the compiled partition is from
    // global cache.
    std::pair<compiled_partition_t *, cache_state_t> cp {
            compiled_partition, cache_state_t::compiled_partition_hit};

    if (get_verbose(dnnl::impl::verbose_t::create_profile,
                dnnl::impl::component_t::graph)) {
        double start_ms = dnnl::impl::get_msec();
        CHECK(partition->compile(cp, in, out, engine));
        double duration_ms = dnnl::impl::get_msec() - start_ms;

        const char *cache_status = cache_state2str(cp.second);
        VPROF(start_ms, graph, compile, cache_status,
                compiled_partition->info(), duration_ms);
    } else {
        CHECK(partition->compile(cp, in, out, engine));
    }
    return status::success;
}

status_t dnnl_graph_partition_get_input_ports_num(
        const partition_t *partition, size_t *num) {
    if (utils::any_null(partition, num)) { return status::invalid_arguments; }

    *num = partition->get_inputs_num();
    return status::success;
}

status_t dnnl_graph_partition_get_output_ports_num(
        const partition_t *partition, size_t *num) {
    if (utils::any_null(partition, num)) { return status::invalid_arguments; }

    *num = partition->get_outputs_num();
    return status::success;
}

status_t dnnl_graph_partition_get_input_ports(
        const partition_t *partition, size_t num, logical_tensor_t *inputs) {
    if (utils::any_null(partition, inputs)
            || partition->get_inputs_num() != num) {
        return status::invalid_arguments;
    }

    auto &in = partition->get_inputs();
    for (size_t i = 0; i < num; ++i) {
        inputs[i] = in[i];
    }

    return status::success;
}

status_t dnnl_graph_partition_get_output_ports(
        const partition_t *partition, size_t num, logical_tensor_t *outputs) {
    if (utils::any_null(partition, outputs)
            || partition->get_outputs_num() != num) {
        return status::invalid_arguments;
    }

    auto &out = partition->get_outputs();
    for (size_t i = 0; i < num; ++i) {
        outputs[i] = out[i];
    }

    return status::success;
}

status_t dnnl_graph_partition_is_supported(
        const partition_t *partition, uint8_t *is_supported) {
    if (utils::any_null(partition, is_supported))
        return status::invalid_arguments;

    *is_supported = static_cast<uint8_t>(partition->is_supported());
    return status::success;
}

status_t dnnl_graph_partition_get_engine_kind(
        const partition_t *partition, engine_kind_t *kind) {
    if (utils::any_null(partition, kind)) { return status::invalid_arguments; }

    *kind = partition->get_pimpl()->get_engine_kind();
    return status::success;
}

status_t DNNL_API dnnl_graph_partition_get_kind(
        const partition_t *partition, partition_kind_t *kind) {
    if (utils::any_null(partition, kind)) { return status::invalid_arguments; }

    *kind = partition->get_kind();
    return status::success;
}

///
/// dnnl_graph_compiled_partition_t
///
status_t dnnl_graph_compiled_partition_create(
        compiled_partition_t **compiled_partition, partition_t *partition) {
    if (utils::any_null(compiled_partition, partition)) {
        return status::invalid_arguments;
    }

    *compiled_partition = new compiled_partition_t {*partition};
    return status::success;
}

status_t dnnl_graph_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs) {
    return dnnl_graph_compiled_partition_execute_v2(compiled_partition, stream,
            num_inputs, inputs, num_outputs, outputs, nullptr);
}

status_t dnnl_graph_compiled_partition_execute_v2(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const tensor_t *scratchpad) {
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_arguments;

    std::vector<tensor_t> ins, outs;
    ins.reserve(num_inputs);
    outs.reserve(num_outputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        ins.emplace_back(**(inputs + i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        outs.emplace_back(**(outputs + i));
    }

    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        bool block_on_wait = true;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto status = stream->get_threadpool(&tp);
        block_on_wait = status == status::success && tp
                && !(tp->get_flags()
                        & dnnl::threadpool_interop::threadpool_iface::
                                ASYNCHRONOUS);
#endif
        if (block_on_wait) stream->wait();
        double start_ms = dnnl::impl::get_msec();
        CHECK(compiled_partition->execute(stream, ins, outs, scratchpad));
        if (block_on_wait) stream->wait();
        double duration_ms = dnnl::impl::get_msec() - start_ms;
        auto info = compiled_partition->exec_info(scratchpad != nullptr);
        VPROF(start_ms, graph, exec, VERBOSE_profile, info.c_str(),
                duration_ms);
    } else {
        CHECK(compiled_partition->execute(stream, ins, outs, scratchpad));
    }
    return status::success;
}

status_t dnnl_graph_sycl_interop_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const void *deps, void *sycl_event) {
    return dnnl_graph_sycl_interop_compiled_partition_execute_v2(
            compiled_partition, stream, num_inputs, inputs, num_outputs,
            outputs, nullptr, deps, sycl_event);
}

status_t dnnl_graph_sycl_interop_compiled_partition_execute_v2(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const tensor_t *scratchpad, const void *deps,
        void *sycl_event) {
#ifdef DNNL_WITH_SYCL
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_arguments;
    if (stream->engine()->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_SYCL
        return status::invalid_arguments;
#endif
    } else {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
        return status::invalid_arguments;
#endif
    }

    std::vector<tensor_t> ins, outs;
    ins.reserve(num_inputs);
    outs.reserve(num_outputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        ins.emplace_back(**(inputs + i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        outs.emplace_back(**(outputs + i));
    }
    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        bool use_profiler = stream->is_verbose_profiler_enabled();
        if (!use_profiler) stream->wait();
        double start_ms = dnnl::impl::get_msec();
        if (deps != nullptr) {
            const auto &sycl_deps = *(const std::vector<::sycl::event> *)deps;
            CHECK(compiled_partition->execute_sycl(stream, ins, outs,
                    scratchpad, sycl_deps,
                    static_cast<::sycl::event *>(sycl_event)));
        } else {
            CHECK(compiled_partition->execute_sycl(stream, ins, outs,
                    scratchpad, {}, static_cast<::sycl::event *>(sycl_event)));
        }
        auto info = compiled_partition->exec_info(scratchpad != nullptr);
        if (!use_profiler) {
            stream->wait();
            double duration_ms = dnnl::impl::get_msec() - start_ms;
            VPROF(start_ms, graph, exec, VERBOSE_profile, info.c_str(),
                    duration_ms);
        } else {
            CHECK(stream->run_verbose_profiler(info, start_ms,
                    static_cast<uint64_t>(dnnl::impl::component_t::graph)));
        }
    } else {
        if (deps != nullptr) {
            const auto &sycl_deps = *(const std::vector<::sycl::event> *)deps;
            CHECK(compiled_partition->execute_sycl(stream, ins, outs,
                    scratchpad, sycl_deps,
                    static_cast<::sycl::event *>(sycl_event)));
        } else {
            CHECK(compiled_partition->execute_sycl(stream, ins, outs,
                    scratchpad, {}, static_cast<::sycl::event *>(sycl_event)));
        }
    }

    return status::success;
#else
    UNUSED(compiled_partition);
    UNUSED(stream);
    UNUSED(num_inputs);
    UNUSED(inputs);
    UNUSED(num_outputs);
    UNUSED(outputs);
    UNUSED(scratchpad);
    UNUSED(deps);
    UNUSED(sycl_event);
    return status::unimplemented;
#endif
}

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
using ocl_event_t = dnnl::impl::xpu::ocl::wrapper_t<cl_event>;

status_t dnnl_graph_ocl_interop_compiled_partition_execute(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const cl_event *deps, int ndeps,
        cl_event *ocl_event) {
    return dnnl_graph_ocl_interop_compiled_partition_execute_v2(
            compiled_partition, stream, num_inputs, inputs, num_outputs,
            outputs, nullptr, deps, ndeps, ocl_event);
}

status_t dnnl_graph_ocl_interop_compiled_partition_execute_v2(
        const compiled_partition_t *compiled_partition, stream_t *stream,
        size_t num_inputs, const tensor_t **inputs, size_t num_outputs,
        const tensor_t **outputs, const tensor_t *scratchpad,
        const cl_event *deps, int ndeps, cl_event *ocl_event) {
    if (utils::any_null(stream, compiled_partition, inputs, outputs))
        return status::invalid_arguments;
    if (stream->engine()->kind() != engine_kind::gpu) {
        return status::invalid_arguments;
    }

    std::vector<tensor_t> ins, outs;
    ins.reserve(num_inputs);
    outs.reserve(num_outputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        ins.emplace_back(**(inputs + i));
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        outs.emplace_back(**(outputs + i));
    }

    // Wrap raw user deps with retain.
    std::vector<ocl_event_t> wrapped_deps;
    if (deps != nullptr) {
        if (ndeps < 0) return status::invalid_arguments;
        wrapped_deps.reserve(static_cast<size_t>(ndeps));
        for (int i = 0; i < ndeps; i++)
            wrapped_deps.emplace_back(deps[i], true);
    }

    ocl_event_t ret_event;

    if (get_verbose(dnnl::impl::verbose_t::exec_profile,
                dnnl::impl::component_t::graph)) {
        bool use_profiler = stream->is_verbose_profiler_enabled();
        if (!use_profiler) stream->wait();
        double start_ms = dnnl::impl::get_msec();
        CHECK(compiled_partition->execute_ocl(
                stream, ins, outs, scratchpad, wrapped_deps, ret_event));
        // Transfer ownership to caller.
        if (ocl_event) *ocl_event = ret_event.release();
        auto info = compiled_partition->exec_info(scratchpad != nullptr);
        if (!use_profiler) {
            stream->wait();
            double duration_ms = dnnl::impl::get_msec() - start_ms;
            VPROF(start_ms, graph, exec, VERBOSE_profile, info.c_str(),
                    duration_ms);
        } else {
            CHECK(stream->run_verbose_profiler(info, start_ms,
                    static_cast<uint64_t>(dnnl::impl::component_t::graph)));
        }
    } else {
        CHECK(compiled_partition->execute_ocl(
                stream, ins, outs, scratchpad, wrapped_deps, ret_event));
        // Transfer ownership to caller.
        if (ocl_event) *ocl_event = ret_event.release();
    }

    return status::success;
}
#endif

status_t dnnl_graph_compiled_partition_destroy(
        compiled_partition_t *compiled_partition) {
    delete compiled_partition;
    return status::success;
}

status_t dnnl_graph_compiled_partition_query_logical_tensor(
        const compiled_partition_t *compiled_partition, size_t tid,
        logical_tensor_t *lt) {
    if (utils::any_null(compiled_partition, lt))
        return status::invalid_arguments;
    return compiled_partition->query_logical_tensor(tid, lt);
}

status_t dnnl_graph_compiled_partition_get_inplace_ports(
        const compiled_partition_t *compiled_partition,
        size_t *num_inplace_pairs, const inplace_pair_t **inplace_pairs) {
    if (utils::any_null(compiled_partition, num_inplace_pairs, inplace_pairs))
        return status::invalid_arguments;

    const auto &cp_inplace_pairs = compiled_partition->get_inplace_pairs();
    *num_inplace_pairs = cp_inplace_pairs.size();
    *inplace_pairs = cp_inplace_pairs.data();

    return status::success;
}

status_t dnnl_graph_compiled_partition_get_scratchpad_logical_tensor(
        const compiled_partition_t *compiled_partition, logical_tensor_t *lt) {
    if (utils::any_null(compiled_partition, lt))
        return status::invalid_arguments;

    const size_t scratchpad_size = compiled_partition->get_scratchpad_size();

    // Build a 1-D logical tensor with data_type u8 and strided layout
    *lt = logical_tensor_t();
    lt->id = std::numeric_limits<size_t>::max();
    lt->ndims = 1;
    lt->dims[0] = static_cast<dnnl_dim_t>(scratchpad_size);
    lt->data_type = data_type::u8;
    lt->property = property_type::variable;
    lt->layout_type = layout_type::strided;
    lt->layout.strides[0] = 1;

    return status::success;
}

status_t dnnl_graph_compiled_partition_query_impl_info(
        const compiled_partition_t *compiled_partition,
        const char **impl_info) {
    if (utils::any_null(compiled_partition, impl_info))
        return status::invalid_arguments;
    if (!compiled_partition->is_initialized()) return status::invalid_arguments;

    *impl_info = compiled_partition->impl_info();
    return status::success;
}

status_t dnnl_graph_partition::infer_shape(
        std::vector<const logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // check if shape is already known, if so, no need to do shape inference
    auto pos = std::find_if(outputs.begin(), outputs.end(),
            [&](const std::vector<logical_tensor_t *>::value_type &out)
                    -> bool {
        return logical_tensor_wrapper_t(out).is_shape_unknown();
    });
    if (pos == outputs.end()) { return status::success; }

    return pimpl_->infer_shape(inputs, outputs);
}

static status_t pre_process(std::vector<logical_tensor_t> &dst,
        std::vector<const logical_tensor_t *> &src, const backend_t *abackend) {
    using ltw = logical_tensor_wrapper_t;
    dst.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst.emplace_back(*src[i]);
        if (ltw(src[i]).is_opaque()) {
            size_t layout_id = src[i]->layout.layout_id;
            auto pair = backend_registry_t::decode_layout_id(layout_id);
            if (pair.second != abackend->get_id()) {
                // given opaque layout id must be generated by this
                // backend
                return status::invalid_arguments;
            }
            dst[i].layout.layout_id = pair.first;
        }
    }
    return status::success;
}

static status_t post_process(std::vector<logical_tensor_t> &dst,
        std::vector<logical_tensor_t> &src, const backend_t *abackend) {
    using ltw = logical_tensor_wrapper_t;
    UNUSED(src);

    for (size_t i = 0; i < dst.size(); i++) {
        if (ltw(dst[i]).is_opaque()) {
            size_t layout_id = dst[i].layout.layout_id;
            dst[i].layout.layout_id = backend_registry_t::encode_layout_id(
                    layout_id, abackend->get_id());
        }
    }
    return status::success;
}

static status_t pre_process(std::vector<tensor_t> &dst,
        const std::vector<tensor_t> &src, const backend_t *abackend) {
    using ltw = logical_tensor_wrapper_t;
    dst.reserve(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst.emplace_back(src[i]);
        auto &src_lt = src[i].get_logical_tensor();
        if (ltw(src_lt).is_opaque()) {
            size_t layout_id = src_lt.layout.layout_id;
            auto pair = backend_registry_t::decode_layout_id(layout_id);
            if (pair.second != abackend->get_id()) {
                return status::invalid_arguments;
            }
            auto &dst_lt = const_cast<logical_tensor_t &>(
                    dst[i].get_logical_tensor());

            dst_lt.layout.layout_id = pair.first;
        }
    }
    return status::success;
}

bool dnnl_graph_partition::is_supported() const {
    return (pimpl_ != nullptr)
            && (pimpl_->get_assigned_backend()->get_name() != "fake_backend");
}

status_t dnnl_graph_partition::compile(compiled_partition_t *cp,
        std::vector<const logical_tensor_t *> &inputs,
        std::vector<const logical_tensor_t *> &outputs,
        engine_t *aengine) const {
    status_t ret;

    if (!aengine || aengine->kind() != pimpl_->get_engine_kind())
        return status::invalid_arguments;

    const backend_t *backend = pimpl_->get_assigned_backend();
    if (!backend) return status::invalid_arguments;

    // Pre-process the given logical tensor. The pre-process includes
    // 1. decode backend id from the layout id and remove it
    std::vector<logical_tensor_t> tmp_inputs, tmp_outputs;
    ret = pre_process(tmp_inputs, inputs, backend);
    if (status::success != ret) return ret;

    ret = pre_process(tmp_outputs, outputs, backend);
    if (status::success != ret) return ret;

    // Count how many registered backends support the engine kind
    const engine_kind_t kind = aengine->kind();
    size_t effective_backends = 0;
    for (const auto &bkd :
            backend_registry_t::get_singleton().get_registered_backends()) {
        const bool is_not_fake = bkd->get_priority() > 0;
        if (is_not_fake && bkd->support_engine_kind(kind)) {
            effective_backends++;
        }
    }

    // If engine kind is GPU and only dnnl backend supports GPU, we can
    // safely use blocked layout to improve performance. Otherwise, we must
    // use plain layout, since: 1. plain layout usually give optimal layout
    // on CPU. 2. we don't want to pass blocked layout cross backends.
    const bool can_use_blocked_layout
            = effective_backends == 1 && kind == engine_kind::gpu;
    const_cast<partition_impl_t *>(pimpl_.get())
            ->set_use_blocked_layout(can_use_blocked_layout);

    if (dnnl::impl::graph::utils::get_graph_dump_mode(
                dnnl::impl::graph::graph_dump_mode_t::subgraph)) {
        if (!is_supported()) return status::unimplemented;
        // deep copy for graph serialization
        auto part = pimpl_->clone();
        const std::vector<std::shared_ptr<op_t>> &fused_op = part->get_ops();
        if (fused_op.empty()) return status::invalid_arguments;
        auto agraph = graph_t(fused_op, get_engine_kind(), get_attr());
        // set user given logical tensors and infer shape
        agraph.set_user_inputs_outputs(tmp_inputs, tmp_outputs);
        agraph.infer_shape();
        // hash logical tensors to generate unique filename
        partition_hashing::key_t key(this, aengine, inputs, outputs);
        size_t seed = 0;
        seed = partition_hashing::get_unordered_array_hash(seed, key.ins_);
        seed = partition_hashing::get_unordered_array_hash(seed, key.outs_);
        dnnl::impl::stringstream_t filename;
        filename << "graph-" << id() << "-" << seed << ".json";
        agraph.serialize(filename.str());
    }

    // The impl's compile will generate the compiled_partition_impl and
    // modify the given inputs outputs logical tensor
    ret = pimpl_->compile(cp, tmp_inputs, tmp_outputs, aengine);
    if (status::success != ret) return ret;

    // Post-process the modified logical tensor and store them
    // to compiled_partition_impl. The post-process includes
    // 1. encode backend id to generated layout id
    ret = post_process(cp->get_mutable_inputs(), tmp_inputs, backend);
    if (status::success != ret) return ret;

    ret = post_process(cp->get_mutable_outputs(), tmp_outputs, backend);
    if (status::success != ret) return ret;

    if (ret != status::success || !cp->is_initialized())
        return status::unimplemented;
    return status::success;
}

status_t dnnl_graph_partition::compile(
        std::pair<compiled_partition_t *, cache_state_t> &compiled_partition,
        std::vector<const logical_tensor_t *> &inputs,
        std::vector<const logical_tensor_t *> &outputs,
        engine_t *aengine) const {
    namespace partition_hashing = partition_hashing;
    auto &global_compiled_partition_cache = compiled_partition_cache();
    partition_hashing::key_t key(this, aengine, inputs, outputs);

    struct create_context_t {
        const partition_t *partition;
        std::vector<const logical_tensor_t *> &inputs;
        std::vector<const logical_tensor_t *> &outputs;
        engine_t *engine;
        cache_state_t cache_status;
    };
    create_context_t context {this, inputs, outputs, aengine,
            cache_state_t::compiled_partition_hit};

    compiled_partition_cache_t::create_func_ptr_t create = [](void *context) {
        auto &c = *static_cast<create_context_t *>(context);
        c.cache_status = cache_state_t::miss;
        std::shared_ptr<compiled_partition_t> cp
                = std::make_shared<compiled_partition_t>(*c.partition);
        status_t status
                = (c.partition)
                          ->compile(cp.get(), c.inputs, c.outputs, c.engine);
        return compiled_partition_cache_t::result_t {std::move(cp), status};
    };

    auto result = global_compiled_partition_cache.get_or_create(
            key, *create, &context);
    if (result.status != status::success) return result.status;

    compiled_partition.first->init(result.value->pimpl_);
    compiled_partition.second = context.cache_status;

    return result.status;
}

status_t dnnl_graph_compiled_partition::execute(stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const tensor_t *scratchpad) const {
    if (astream->engine()->kind() == engine_kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        return execute_sycl(astream, inputs, outputs, scratchpad, {}, nullptr);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        ocl_event_t unused_event;
        return execute_ocl(
                astream, inputs, outputs, scratchpad, {}, unused_event);
#else
        return status::runtime_error;
#endif
    } else {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        return execute_sycl(astream, inputs, outputs, scratchpad, {}, nullptr);
#else
        // TODO(xxx): need to improve the check of two engines. On dev-graph, there
        // is a match function.
        if (!astream
                || (astream->engine()->kind() != pimpl_->get_engine()->kind()))
            return status::invalid_arguments;

        const backend_t *backend = src_partition_.get_assigned_backend();
        if (!backend) return status::invalid_arguments;

        // Pre-process the given tensor. The pre-process includes
        // FIXME(xx) reduce overhead?
        // 1. decode backend id from the layout id and remove it
        std::vector<tensor_t> processed_inputs, processed_outputs;
        pre_process(processed_inputs, inputs, backend);
        pre_process(processed_outputs, outputs, backend);

        return pimpl_->execute(
                astream, processed_inputs, processed_outputs, scratchpad);
#endif
    }
}

#ifdef DNNL_WITH_SYCL
status_t dnnl_graph_compiled_partition::execute_sycl(stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) const {
    // TODO(xxx): need to improve the check of two engines. On dev-graph, there
    // is a match function.
    if (!astream || (astream->engine()->kind() != pimpl_->get_engine()->kind()))
        return status::invalid_arguments;

    const backend_t *backend = src_partition_.get_assigned_backend();
    if (!backend) return status::invalid_arguments;

    // Pre-process the given tensor. The pre-process includes
    // 1. decode backend id from the layout id and remove it
    std::vector<tensor_t> processed_inputs, processed_outputs;
    pre_process(processed_inputs, inputs, backend);
    pre_process(processed_outputs, outputs, backend);

    return pimpl_->execute_sycl(astream, processed_inputs, processed_outputs,
            scratchpad, sycl_deps, sycl_event);
}
#endif // DNNL_WITH_SYCL

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
// It looks very similar to execute_sycl(). Consider to merge them in the
// future.
status_t dnnl_graph_compiled_partition::execute_ocl(stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad,
        const std::vector<ocl_event_t> &ocl_deps,
        ocl_event_t &ocl_event) const {
    if (!astream || (astream->engine()->kind() != pimpl_->get_engine()->kind()))
        return status::invalid_arguments;

    const backend_t *backend = src_partition_.get_assigned_backend();
    if (!backend) return status::invalid_arguments;

    std::vector<tensor_t> processed_inputs, processed_outputs;
    pre_process(processed_inputs, inputs, backend);
    pre_process(processed_outputs, outputs, backend);

    return pimpl_->execute_ocl(astream, processed_inputs, processed_outputs,
            scratchpad, ocl_deps, ocl_event);
}
#endif
