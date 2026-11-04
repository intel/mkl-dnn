/*******************************************************************************
 * Copyright 2025 Intel Corporation
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

#include "graph/backend/dnnl/executables/concat.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#define VCHECK_OP_EXECUTABLE(cond, msg, ...) \
    if (!(cond)) { VERROR(graph, op_executable, msg, ##__VA_ARGS__); }

concat_executable_t::desc_t concat_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return {graph::utils::any_cast<dnnl::concat::primitive_desc>(
                        pd_cache.at(op.get())),
                false};
    }

    // Here we force to use plain-in-plain-out (acdb) for 4D case to make
    // sure good performance of DenseNet121 (reducing reorder overhead).
    // But for other cases like 2D/3D (e.g. DLRM), we just use default
    // format since there may be followed by a non-DNNL op which requires an
    // input with default format. Anyway it looks like a bit tricky.
    auto get_forced_format_tag = [](const dims &in_dims) -> format_tag {
        if (in_dims.size() == 4)
            return format_tag::acdb;
        else
            return get_ncx_format(in_dims);
    };

    const auto rank = op->get_output_logical_tensor(0).ndims;
    const auto res = utils::try_reverse_axis(
            op->get_attr<int64_t>(op_attr::axis), rank);
    VCHECK_OP_EXECUTABLE(res.first, "invalid axis for concat");
    const auto axis = res.second;

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    std::vector<memory::desc> src_mds;
    src_mds.reserve(op->num_inputs());
    for (const auto &in_val : op->get_input_values()) {
        const auto tmp_desc
                = make_dnnl_memory_desc(in_val->get_logical_tensor());
        src_mds.emplace_back(tmp_desc.get_dims(), tmp_desc.get_data_type(),
                get_forced_format_tag(tmp_desc.get_dims()));
    }
    const auto tmp_desc
            = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    auto dst = memory::desc {tmp_desc.get_dims(), tmp_desc.get_data_type(),
            get_forced_format_tag(tmp_desc.get_dims())};

    dnnl::concat::primitive_desc pd(
            p_engine, dst, static_cast<int>(axis), src_mds, prm_attr);
    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t concat_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_miso_op(op);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
