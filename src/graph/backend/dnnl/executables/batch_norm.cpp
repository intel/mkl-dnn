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

#include "graph/backend/dnnl/executables/batch_norm.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

bn_folding_t::bn_folding_t(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout) {
    desc_ = create_desc(op, p_engine, pd_cache, graph_attr, use_block_layout);
    add_prim_ = dnnl::binary(desc_.add_pd_);
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
    if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
        sqrt_prim_ = dnnl::eltwise_forward(desc_.sqrt_pd_);
    }
#endif
    mul_prim_ = dnnl::binary(desc_.mul_pd_);
    sub_prim_ = dnnl::binary(desc_.sub_pd_);
}

void bn_folding_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    UNUSED(args);

    auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
    auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
    auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
    auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
    auto mean = args.find(DNNL_ARG_MEAN)->second;
    auto variance = args.find(DNNL_ARG_VARIANCE)->second;
    auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

    auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
    auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

    // 0. split scratchpad buffer to specific intermediate memory
    // sqrt_variance
    char *buf_start = (char *)scratchpad.get_data_handle();
    memory sqrt_variance = make_dnnl_memory(
            variance.get_desc(), scratchpad.get_engine(), (void *)buf_start);
    buf_start += variance.get_desc().get_size();
    // zero_bias
    memory valid_bias = bias;
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        valid_bias = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += valid_bias.get_desc().get_size();
    }
    // epsilon
    memory epsilon_mem = make_dnnl_memory(
            desc_.epsilon_desc_, scratchpad.get_engine(), (void *)buf_start);

    // 1. sqrt_variance = sqrt(variance + epsilon)
    if (variance.get_engine().get_kind() == engine::kind::cpu) {
        float *ptr = (float *)epsilon_mem.get_data_handle();
        *ptr = desc_.epsilon_;
    } else {
        engine cpu_eng(engine::kind::cpu, 0);
        memory cpu_mem = make_dnnl_memory(
                desc_.epsilon_desc_, cpu_eng, (void *)&desc_.epsilon_);
        dnnl::reorder(cpu_mem, epsilon_mem)
                .execute(stream, cpu_mem, epsilon_mem);
    }

    add_prim_.execute(stream,
            {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                    {DNNL_ARG_DST, sqrt_variance}});

    // 2. updated_weight = weights * scale / sqrt_variance
    memory new_scale(
            desc_.new_scale_desc_, scale.get_engine(), scale.get_data_handle());
    memory new_sqrt_variance(desc_.new_variance_desc_,
            sqrt_variance.get_engine(), sqrt_variance.get_data_handle());
    mul_prim_.execute(stream,
            {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                    {DNNL_ARG_DST, updated_weights},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            new_sqrt_variance}});

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        // initialize the bias with zero value
        std::vector<float> zero(
                graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
        if (mean.get_engine().get_kind() == engine::kind::cpu) {
            std::memcpy(valid_bias.get_data_handle(), zero.data(),
                    valid_bias.get_desc().get_size());
        } else {
            engine cpu_eng(engine::kind::cpu, 0);
            memory cpu_mem = make_dnnl_memory(
                    variance.get_desc(), cpu_eng, zero.data());
            dnnl::reorder(cpu_mem, valid_bias)
                    .execute(stream, cpu_mem, valid_bias);
        }
    }

    sub_prim_.execute(stream,
            {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                    {DNNL_ARG_DST, updated_bias},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            sqrt_variance},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                            shift}});
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> bn_folding_t::execute_sycl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    UNUSED(args);

    auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
    auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
    auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
    auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
    auto mean = args.find(DNNL_ARG_MEAN)->second;
    auto variance = args.find(DNNL_ARG_VARIANCE)->second;
    auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

    auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
    auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

    // 0. split scratchpad buffer to specific intermediate memory
    // sqrt_variance
    char *buf_start = (char *)scratchpad.get_data_handle();
    memory sqrt_variance = make_dnnl_memory(
            variance.get_desc(), scratchpad.get_engine(), (void *)buf_start);
    buf_start += variance.get_desc().get_size();
    // zero_bias
    memory valid_bias = bias;
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        valid_bias = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += valid_bias.get_desc().get_size();
    }
    // epsilon
    memory epsilon_mem = make_dnnl_memory(
            desc_.epsilon_desc_, scratchpad.get_engine(), (void *)buf_start);

    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
    ::sycl::event sycl_deps;

    if (scratchpad.get_engine().get_kind() == engine::kind::gpu) {
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA

        buf_start += epsilon_mem.get_desc().get_size();

        // variance + epsilon
        memory variance_epsilon = make_dnnl_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        //auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        sycl_queue
                .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                        epsilon_mem.get_desc().get_size())
                .wait();

        auto sycl_deps0 = dnnl::sycl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, variance_epsilon}},
                deps);

        sycl_deps = dnnl::sycl_interop::execute(sqrt_prim_, stream,
                {{DNNL_ARG_SRC, variance_epsilon},
                        {DNNL_ARG_DST, sqrt_variance}},
                {sycl_deps0});
#else

        // 1. sqrt_variance = sqrt(variance + epsilon)
        //auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        sycl_queue
                .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                        epsilon_mem.get_desc().get_size())
                .wait();

        sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);

#endif
    } else {
        // 1. sqrt_variance = sqrt(variance + epsilon)
        sycl_queue
                .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                        epsilon_mem.get_desc().get_size())
                .wait();

        sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);
    }
    // 2. updated_weight = weights * scale / sqrt_variance
    memory new_scale(
            desc_.new_scale_desc_, scale.get_engine(), scale.get_data_handle());
    memory new_sqrt_variance(desc_.new_variance_desc_,
            sqrt_variance.get_engine(), sqrt_variance.get_data_handle());

    auto sycl_deps2 = dnnl::sycl_interop::execute(mul_prim_, stream,
            {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                    {DNNL_ARG_DST, updated_weights},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            new_sqrt_variance}},
            {sycl_deps});

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        // initialize the bias with zero value
        std::vector<float> zero(
                graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
        sycl_queue
                .memcpy(valid_bias.get_data_handle(), zero.data(),
                        valid_bias.get_desc().get_size())
                .wait();
        auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {sycl_deps2});
        if (stream.get_engine().get_kind() == engine::kind::cpu)
            sycl_deps3.wait();
        return sycl_deps3;
    }

    auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
            {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                    {DNNL_ARG_DST, updated_bias},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            sqrt_variance},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                            shift}},
            {sycl_deps2});
    if (stream.get_engine().get_kind() == engine::kind::cpu) sycl_deps3.wait();
    return sycl_deps3;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t bn_folding_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    UNUSED(args);

    auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
    auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
    auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
    auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
    auto mean = args.find(DNNL_ARG_MEAN)->second;
    auto variance = args.find(DNNL_ARG_VARIANCE)->second;
    auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

    auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
    auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

    // 0. split scratchpad buffer to specific intermediate memory
    // sqrt_variance

    char *buf_start = (char *)scratchpad.get_data_handle();
    memory sqrt_variance = dnnl::ocl_interop::make_memory(variance.get_desc(),
            scratchpad.get_engine(), dnnl::ocl_interop::memory_kind::usm,
            (void *)buf_start);
    buf_start += variance.get_desc().get_size();
    // zero_bias
    memory valid_bias = bias;
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        valid_bias = dnnl::ocl_interop::make_memory(variance.get_desc(),
                scratchpad.get_engine(), dnnl::ocl_interop::memory_kind::usm,
                (void *)buf_start);
        buf_start += valid_bias.get_desc().get_size();
    }
    // epsilon
    memory epsilon_mem = dnnl::ocl_interop::make_memory(desc_.epsilon_desc_,
            scratchpad.get_engine(), dnnl::ocl_interop::memory_kind::usm,
            (void *)buf_start);

    // 1. sqrt_variance = sqrt(variance + epsilon)
    cl_event e;
    xpu::ocl::usm::memcpy(stream.get(), epsilon_mem.get_data_handle(),
            &desc_.epsilon_, epsilon_mem.get_desc().get_size(), 0, nullptr, &e);
    xpu::ocl::clWaitForEvents(1, &e);
    xpu::ocl::clReleaseEvent(e);

    std::vector<cl_event> raw_deps(deps.begin(), deps.end());
    ocl_event_t ocl_deps(dnnl::ocl_interop::execute(add_prim_, stream,
            {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                    {DNNL_ARG_DST, sqrt_variance}},
            raw_deps));

    // 2. updated_weight = weights * scale / sqrt_variance
    memory new_scale = dnnl::ocl_interop::make_memory(desc_.new_scale_desc_,
            scale.get_engine(), dnnl::ocl_interop::memory_kind::usm,
            scale.get_data_handle());
    memory new_sqrt_variance = dnnl::ocl_interop::make_memory(
            desc_.new_variance_desc_, sqrt_variance.get_engine(),
            dnnl::ocl_interop::memory_kind::usm,
            sqrt_variance.get_data_handle());

    ocl_event_t ocl_deps2(dnnl::ocl_interop::execute(mul_prim_, stream,
            {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                    {DNNL_ARG_DST, updated_weights},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                            new_sqrt_variance}},
            {ocl_deps.get()}));

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
    if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
        // initialize the bias with zero value
        std::vector<float> zero(
                graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
        xpu::ocl::usm::memcpy(stream.get(), valid_bias.get_data_handle(),
                zero.data(), valid_bias.get_desc().get_size(), 0, nullptr, &e);
        xpu::ocl::clWaitForEvents(1, &e);
        xpu::ocl::clReleaseEvent(e);

        auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {ocl_deps2.get()});
        return ocl_event_t(ocl_deps3);
    }

    auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
            {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                    {DNNL_ARG_DST, updated_bias},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, scale},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                            sqrt_variance},
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                            shift}},
            {ocl_deps2.get()});
    return ocl_event_t(ocl_deps3);
}
#endif

bn_folding_t::desc_t bn_folding_t::create_desc(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, pd_cache_t &pd_cache,
        const graph_attr_t &graph_attr, bool use_block_layout) {
    UNUSED(pd_cache);
    UNUSED(graph_attr);
    UNUSED(use_block_layout);

    desc_t desc;

    desc.epsilon_ = op->get_attr<float>(op_attr::epsilon);
    desc.data_format_ = op->get_attr<std::string>(op_attr::data_format);
    desc.filter_format_ = op->get_attr<std::string>(op_attr::weights_format);
    desc.with_bias_ = op->get_attr<bool>(op_attr::with_bias);

    size_t in_idx = 0;
    auto weights
            = make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++));
    auto bias = desc.with_bias_
            ? make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++))
            : memory::desc();
    auto scale = make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++));
    auto shift = make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++));
    auto mean = make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++));
    auto variance
            = make_dnnl_memory_desc(op->get_input_logical_tensor(in_idx++));

    // 1. sqrt_variance = sqrt(variance + epsilon)

    // temp = variance + epsilon
    memory::dims epsilon_dims(variance.get_ndims(), 1);
    desc.epsilon_desc_ = memory::desc(
            epsilon_dims, memory::data_type::f32, memory::format_tag::a);

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU
    if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
        primitive_attr add_attr;
        desc.add_pd_
                = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
                        variance, desc.epsilon_desc_, variance, add_attr);

        primitive_attr sqrt_attr;
        desc.sqrt_pd_ = dnnl::eltwise_forward::primitive_desc(p_engine,
                prop_kind::forward, algorithm::eltwise_sqrt, variance, variance,
                0.0f, 0.0f, sqrt_attr);
    } else {
        post_ops add_post_ops;
        // sqrt_variance = sqrt(temp)
        add_post_ops.append_eltwise(algorithm::eltwise_sqrt, 0.0f, 0.0f);

        primitive_attr add_attr;
        add_attr.set_post_ops(add_post_ops);
        desc.add_pd_
                = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
                        variance, desc.epsilon_desc_, variance, add_attr);
    }

#else
    post_ops add_post_ops;
    // sqrt_variance = sqrt(temp)
    add_post_ops.append_eltwise(algorithm::eltwise_sqrt, 0.0f, 0.0f);

    primitive_attr add_attr;
    add_attr.set_post_ops(add_post_ops);
    desc.add_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_add,
            variance, desc.epsilon_desc_, variance, add_attr);

#endif
    // 2. updated_weight = weights * scale / sqrt_variance

    // expand 1D scale and variance to same ndims with weights
    desc.new_scale_desc_ = expand(scale, weights.get_ndims());
    desc.new_variance_desc_ = expand(variance, weights.get_ndims());

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of NXC format. But for NCX format, we
    // need permute c channel to the second dimension
    if (desc.filter_format_ == "NCX") { // matmul case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "NXC", "NCX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // after expand, the c channel is on the last dimension, which
    // meet the requirement of XIO format. But for OIX format, we
    // need permute c channel to the first dimension
    if (desc.filter_format_ == "OIX") { // conv case
        auto perm = dnnl_impl::utils::cast_to_int32(get_permutation(
                desc.new_scale_desc_.get_ndims(), "XIO", "OIX"));
        desc.new_scale_desc_ = desc.new_scale_desc_.permute_axes(perm);
        desc.new_variance_desc_ = desc.new_variance_desc_.permute_axes(perm);
    }

    // temp = weights * scale
    post_ops mul_post_ops;
    // updated_weight = temp / sqrt_variance
    mul_post_ops.append_binary(algorithm::binary_div, desc.new_variance_desc_);

    primitive_attr mul_attr;
    mul_attr.set_post_ops(mul_post_ops);
    desc.mul_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_mul,
            weights, desc.new_scale_desc_, weights, mul_attr);

    // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift

    // temp = bias - mean
    memory::desc valid_bias = bias.is_zero() ? mean : bias;

    post_ops sub_post_ops;
    // temp = temp * scale
    sub_post_ops.append_binary(algorithm::binary_mul, scale);
    // temp = temp / sqrt_variance
    sub_post_ops.append_binary(algorithm::binary_div, variance);
    // temp = temp + shift
    sub_post_ops.append_binary(algorithm::binary_add, shift);

    primitive_attr sub_attr;
    sub_attr.set_post_ops(sub_post_ops);
    desc.sub_pd_ = dnnl::binary::primitive_desc(p_engine, algorithm::binary_sub,
            valid_bias, mean, valid_bias, sub_attr);

    memory::dims scratchpad_dims = variance.get_dims();
    // sqrt_variance, zero_bias and others (like epsilon),
    // or no need to alloc bias
    // binary + sqrt post-op fusion is unsupported on NVIDIA GPU, so we need
    // one more scratchpad for sqrt
#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE \
        && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    size_t factor = 0;
    if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
        factor = bias.is_zero() ? 4 : 3;
    } else {
        factor = bias.is_zero() ? 3 : 2;
    }
#else
    size_t factor = bias.is_zero() ? 3 : 2;
#endif
    scratchpad_dims[0] *= factor;
    desc.scratchpad_desc_ = memory::desc(
            scratchpad_dims, variance.get_data_type(), memory::format_tag::a);

    return desc;
}

arg_indices_t bn_folding_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;

    size_t in_idx = 0;
    args.insert({DNNL_ARG_WEIGHTS, {indices_t::type_t::input, in_idx++}});
    if (op->get_attr<bool>(op_attr::with_bias)) {
        args.insert({DNNL_ARG_BIAS, {indices_t::type_t::input, in_idx++}});
    }
    args.insert({DNNL_ARG_WEIGHTS_1,
            {indices_t::type_t::input, in_idx++}}); // scale
    args.insert({DNNL_ARG_WEIGHTS_2,
            {indices_t::type_t::input, in_idx++}}); // shift
    args.insert({DNNL_ARG_MEAN, {indices_t::type_t::input, in_idx++}}); // mean
    args.insert({DNNL_ARG_VARIANCE,
            {indices_t::type_t::input, in_idx++}}); // variance

    // bind output memory
    size_t out_idx = 0;
    args.insert({DNNL_ARG_DST_0,
            {indices_t::type_t::output, out_idx++}}); // updated weight
    args.insert({DNNL_ARG_DST_1,
            {indices_t::type_t::output, out_idx++}}); // updated bias
    args.insert({DNNL_ARG_SCRATCHPAD,
            {indices_t::type_t::output, out_idx++}}); // scratchpad

    return args;
}

batchnorm_executable_t::desc_t batchnorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // for inference
    if (!op->get_attr<bool>(op_attr::is_training)) {
        flags |= dnnl::normalization_flags::use_global_stats;
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // for training, inputs: [src, mean, variance, gamma, beta]
        if (op->num_inputs() > 3) {
            flags |= dnnl::normalization_flags::use_scale;
            flags |= dnnl::normalization_flags::use_shift;
        }

        if (op->has_attr(op_attr::fuse_relu)
                && op->get_attr<bool>(op_attr::fuse_relu))
            flags |= dnnl::normalization_flags::fuse_norm_relu;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));
    dst = to_format_any(dst);

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto pkind = op->get_attr<bool>(op_attr::is_training)
            ? prop_kind::forward_training
            : prop_kind::forward_inference;

    dnnl::batch_normalization_forward::primitive_desc pd(
            p_engine, pkind, src, dst, epsilon, flags, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

void batchnorm_executable_t::execute(const stream &stream,
        const std::unordered_map<int, memory> &args) const {
    if (!is_training_) {
        prim_.execute(stream, args);
        return;
    }

    std::unordered_map<int, memory> exe_args = args;
    exe_args.erase(DNNL_ARG_SRC_1);
    exe_args.erase(DNNL_ARG_SRC_2);
    exe_args.erase(DNNL_ARG_DST_1);
    exe_args.erase(DNNL_ARG_DST_2);

    prim_.execute(stream, exe_args);

    // calculate running_mean and running_variance
    auto it_mean = args.find(DNNL_ARG_MEAN);
    auto it_var = args.find(DNNL_ARG_VARIANCE);
    auto it_src1 = args.find(DNNL_ARG_SRC_1);
    auto it_src2 = args.find(DNNL_ARG_SRC_2);
    auto it_dst1 = args.find(DNNL_ARG_DST_1);
    auto it_dst2 = args.find(DNNL_ARG_DST_2);

    if (graph::utils::one_of(args.end(), it_mean, it_var, it_src1, it_src2,
                it_dst1, it_dst2)) {
        assert(!"cannot find one of the required memories");
        return;
    }

    auto batch_mean = it_mean->second;
    auto batch_variance = it_var->second;
    auto old_running_mean = it_src1->second;
    auto old_running_variance = it_src2->second;
    auto new_running_mean = it_dst1->second;
    auto new_running_variance = it_dst2->second;

    dnnl::engine p_engine = stream.get_engine();
    // new_running_mean = momentum * old_running_mean +
    //                                      (1 - momentum) * batch_mean
    dnnl::sum({p_engine, scales_,
                      {old_running_mean.get_desc(), batch_mean.get_desc()}})
            .execute(stream,
                    {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                            {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                            {DNNL_ARG_DST, new_running_mean}});
    // new_running_variance = momentum * old_running_variance +
    //                                  (1 - momentum) * batch_variance
    dnnl::sum({p_engine, scales_,
                      {old_running_variance.get_desc(),
                              batch_variance.get_desc()}})
            .execute(stream,
                    {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                            {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                            {DNNL_ARG_DST, new_running_variance}});
}

#ifdef DNNL_WITH_SYCL
std::optional<::sycl::event> batchnorm_executable_t::execute_sycl(
        const stream &stream, const std::unordered_map<int, memory> &args,
        const std::vector<::sycl::event> &deps) const {
    if (!is_training_) {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }

    std::unordered_map<int, memory> exe_args = args;
    exe_args.erase(DNNL_ARG_SRC_1);
    exe_args.erase(DNNL_ARG_SRC_2);
    exe_args.erase(DNNL_ARG_DST_1);
    exe_args.erase(DNNL_ARG_DST_2);

    auto e0 = dnnl::sycl_interop::execute(prim_, stream, exe_args, deps);

    // calculate running_mean and running_variance
    auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
    auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
    auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
    auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
    auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
    auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

    dnnl::engine p_engine = stream.get_engine();
    // new_running_mean = momentum * old_running_mean +
    //                                      (1 - momentum) * batch_mean
    auto sum_prim_0 = dnnl::sum({p_engine, scales_,
            {old_running_mean.get_desc(), batch_mean.get_desc()}});
    auto e1 = dnnl::sycl_interop::execute(sum_prim_0, stream,
            {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                    {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                    {DNNL_ARG_DST, new_running_mean}},
            {e0});
    // new_running_variance = momentum * old_running_variance +
    //                                  (1 - momentum) * batch_variance
    auto sum_prim_1 = dnnl::sum({p_engine, scales_,
            {old_running_variance.get_desc(), batch_variance.get_desc()}});
    auto e2 = dnnl::sycl_interop::execute(sum_prim_1, stream,
            {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                    {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                    {DNNL_ARG_DST, new_running_variance}},
            {e1});
    if (stream.get_engine().get_kind() == engine::kind::cpu) e2.wait();
    return e2;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
ocl_event_t batchnorm_executable_t::execute_ocl(const stream &stream,
        const std::unordered_map<int, memory> &args,
        const std::vector<ocl_event_t> &deps) const {
    std::vector<cl_event> raw_deps(deps.begin(), deps.end());
    if (!is_training_) {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, raw_deps);
        return ocl_event_t(e);
    }

    std::unordered_map<int, memory> exe_args = args;
    exe_args.erase(DNNL_ARG_SRC_1);
    exe_args.erase(DNNL_ARG_SRC_2);
    exe_args.erase(DNNL_ARG_DST_1);
    exe_args.erase(DNNL_ARG_DST_2);

    ocl_event_t e0(
            dnnl::ocl_interop::execute(prim_, stream, exe_args, raw_deps));

    // calculate running_mean and running_variance
    auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
    auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
    auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
    auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
    auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
    auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

    dnnl::engine p_engine = stream.get_engine();
    // new_running_mean = momentum * old_running_mean +
    //                                      (1 - momentum) * batch_mean
    auto sum_prim_0 = dnnl::sum({p_engine, scales_,
            {old_running_mean.get_desc(), batch_mean.get_desc()}});
    ocl_event_t e1(dnnl::ocl_interop::execute(sum_prim_0, stream,
            {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                    {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                    {DNNL_ARG_DST, new_running_mean}},
            {e0.get()}));
    // new_running_variance = momentum * old_running_variance +
    //                                  (1 - momentum) * batch_variance
    auto sum_prim_1 = dnnl::sum({p_engine, scales_,
            {old_running_variance.get_desc(), batch_variance.get_desc()}});
    ocl_event_t e2(dnnl::ocl_interop::execute(sum_prim_1, stream,
            {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                    {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                    {DNNL_ARG_DST, new_running_variance}},
            {e1.get()}));
    return e2;
}
#endif

batchnorm_bwd_executable_t::desc_t batchnorm_bwd_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::batch_normalization_backward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    float epsilon = op->get_attr<float>(op_attr::epsilon);

    auto flags = dnnl::normalization_flags::none;
    // [diff_src, diff_scale, diff_shift, scratchpad]
    if (op->num_outputs() > 2) {
        flags |= dnnl::normalization_flags::use_scale;
        flags |= dnnl::normalization_flags::use_shift;
    } else {
        // [diff_src, scratchpad]
        flags |= dnnl::normalization_flags::use_global_stats;
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));

    if (src.get_inner_nblks() == 1 && src.get_inner_idxs()[0] == 1
            && src.get_inner_blks()[0] == 4) {
        // to default format
        src = to_ncx_format(src);
    }

    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
            p_engine, prop_kind::forward_training, src, src, epsilon, flags,
            prm_attr);

    dnnl::batch_normalization_backward::primitive_desc pd(p_engine,
            prop_kind::backward, src, forward_hints.dst_desc(), src, epsilon,
            flags, forward_hints, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

arg_indices_t batchnorm_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;

    size_t idx = 0;
    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, idx++}});
    if (!op->get_attr<bool>(op_attr::is_training)) { // inference
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_SHIFT, {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_MEAN, {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_VARIANCE, {indices_t::type_t::input, idx++}});
    } else { // training
        // running_mean/running_variance of last iteration
        args.insert({DNNL_ARG_SRC_1, {indices_t::type_t::input, idx++}});
        args.insert({DNNL_ARG_SRC_2, {indices_t::type_t::input, idx++}});

        if (op->num_inputs() > 3) {
            args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
            args.insert({DNNL_ARG_SHIFT, {indices_t::type_t::input, idx++}});
        }
    }

    idx = 0;
    args.insert({DNNL_ARG_DST, {indices_t::type_t::output, idx++}});
    if (op->get_attr<bool>(op_attr::is_training)) {
        // running_mean
        args.insert({DNNL_ARG_DST_1, {indices_t::type_t::output, idx++}});
        // running_variance
        args.insert({DNNL_ARG_DST_2, {indices_t::type_t::output, idx++}});
        // batch_mean
        args.insert({DNNL_ARG_MEAN, {indices_t::type_t::output, idx++}});
        // batch_variance
        args.insert({DNNL_ARG_VARIANCE, {indices_t::type_t::output, idx++}});
    }

    if (op->num_outputs() > idx) {
        args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, idx++}});
    }

    // workspace (for BatchNormForwardTraining with ReLU)
    if (op->num_outputs() > idx) {
        args.insert({DNNL_ARG_WORKSPACE, {indices_t::type_t::output, idx++}});
    }

    return args;
}

arg_indices_t batchnorm_bwd_executable_t::get_arg_indices(const op_t *op) {
    arg_indices_t args;
    size_t idx = 0;

    args.insert({DNNL_ARG_SRC, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_DIFF_DST, {indices_t::type_t::input, idx++}});

    args.insert({DNNL_ARG_MEAN, {indices_t::type_t::input, idx++}});
    args.insert({DNNL_ARG_VARIANCE, {indices_t::type_t::input, idx++}});
    if (op->num_outputs() > 2) {
        // oneDNN only need the scales now
        args.insert({DNNL_ARG_SCALE, {indices_t::type_t::input, idx++}});
    }

    idx = 0;
    args.insert({DNNL_ARG_DIFF_SRC, {indices_t::type_t::output, idx++}});
    // check if has diff_scale and diff_shift outputs
    if (op->num_outputs() > 2) {
        args.insert({DNNL_ARG_DIFF_SCALE, {indices_t::type_t::output, idx++}});
        args.insert({DNNL_ARG_DIFF_SHIFT, {indices_t::type_t::output, idx++}});
    }

    if (op->num_outputs() > idx) {
        args.insert({DNNL_ARG_SCRATCHPAD, {indices_t::type_t::output, idx++}});
    }

    return args;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
