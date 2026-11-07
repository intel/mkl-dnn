/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "graph/interface/shape_infer.hpp"

#define VCHECK_INVALID_SHAPE(cond, msg, ...) \
    VCONDCHECK(graph, create, check, compile, (cond), status::invalid_shape, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {

std::string dims2str(const dims &dims) {
    if (dims.empty()) return std::string("");

    std::string str;
    str += std::to_string(dims[0]);
    for (size_t d = 1; d < dims.size(); ++d)
        str += ("x" + std::to_string(dims[d]));
    return str;
}

/// convert shape to ncx or oix
dims canonicalize(const dims &shape, const std::string &format) {
    dims ret(shape);
    const size_t ndims = shape.size();

    if (ndims <= 2 || "NCX" == format || "OIX" == format) return ret;

    if ("NXC" == format) {
        // NXC -> NCX
        ret[1] = shape[ndims - 1]; // c
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i - 1];
        }
    } else if ("XIO" == format) {
        // XIO -> OIX
        ret[0] = shape[ndims - 1]; // oc
        ret[1] = shape[ndims - 2]; // ic
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i - 2];
        }
    } else if ("XOI" == format) {
        // XOI -> OIX
        ret[0] = shape[ndims - 2]; // oc
        ret[1] = shape[ndims - 1]; // ic
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i - 2];
        }
    } else if ("IOX" == format) {
        // IOX -> OIX
        ret[0] = shape[1]; // oc
        ret[1] = shape[0]; // ic
        for (size_t i = 2; i < ndims; ++i) {
            ret[i] = shape[i];
        }
    } else {
        assert(!"invalid format");
    }
    return ret;
}

inline dims ncx2nxc(const dims &shape) {
    const size_t ndims = shape.size();
    if (ndims <= 2) return shape;

    dims ret(shape);
    // x
    for (size_t i = 2; i < ndims; ++i) {
        ret[i - 1] = shape[i];
    }
    // c
    ret[ndims - 1] = shape[1];

    return ret;
}

/// make a dims according to the format. Only for data format ncx or nxc.
inline dims make_data_dims(const std::string &format, const dim_t n,
        const dim_t c, const dims &x) {
    dims ret;
    if (format == "NCX") {
        ret.push_back(n);
        ret.push_back(c);
        ret.insert(ret.end(), x.begin(), x.end());
    } else if (format == "NXC") {
        ret.push_back(n);
        ret.insert(ret.end(), x.begin(), x.end());
        ret.push_back(c);
    } else {
        assert(!"invalid format");
    }

    return ret;
}

/// make a dims according to the format. Only for filter format xio or oix.
inline dims make_filter_dims(const std::string &format, const dim_t i,
        const dim_t o, const dims &x) {
    dims ret;
    if (format == "XIO") {
        ret.insert(ret.begin(), x.begin(), x.end());
        ret.push_back(i);
        ret.push_back(o);
    } else if (format == "OIX") {
        ret.push_back(o);
        ret.push_back(i);
        ret.insert(ret.end(), x.begin(), x.end());
    } else {
        assert(!"invalid format");
    }

    return ret;
}

/// validate the inferred shape with the expected one.
bool validate(const dims &inferred, const dims &expected) {
    if (inferred.size() != expected.size()) { return false; }

    for (size_t i = 0; i < inferred.size(); ++i) {
        if (expected[i] != DNNL_GRAPH_UNKNOWN_DIM
                && inferred[i] != expected[i]) {
            return false;
        }
    }
    return true;
}

/// get the dense strides of a given shape
/// eg. (3, 4, 5) -> (20, 5, 1)
dims get_dense_strides(const dims &shape) {
    dims strides(shape.size());
    for (auto it = shape.begin(); it < shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), shape.end(), (dim_t)1, [](dim_t x, dim_t y) {
            // replace 0 in shape to 1 when computing the strides
            return std::max<dim_t>(x, 1) * std::max<dim_t>(y, 1);
        });
        const auto dist = std::distance(shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
    }
    return strides;
}

/// shapes of the logical tensors in the vector are known
inline bool every_shape_is_known(const std::vector<logical_tensor_t *> &lts) {
    bool ret = std::all_of(
            lts.cbegin(), lts.cend(), [](const logical_tensor_t *const lt) {
        return !logical_tensor_wrapper_t(lt).is_shape_unknown();
    });
    return ret;
}

inline bool verify_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end,
        const std::function<bool(const dims)> &validator) {
    for (size_t idx = begin; idx < end; ++idx) {
        const dims ltx_dims = logical_tensor_wrapper_t(lts[idx]).vdims();
        if (!validator(ltx_dims)) return false;
    }

    return true;
}

void set_shape_and_strides(logical_tensor_t &lt, const dims &shape) {
    utils::array_copy(lt.dims, shape.data(), shape.size());
    lt.ndims = static_cast<int32_t>(shape.size());

    auto ltw = logical_tensor_wrapper_t(lt);
    // don't overwrite strides provided by users
    if (ltw.is_strided() && ltw.is_stride_unknown()) {
        const dims strides = get_dense_strides(shape);
        utils::array_copy(lt.layout.strides, strides.data(), strides.size());
    }
}

inline void set_shapes_in_range(const std::vector<logical_tensor_t *> &lts,
        const size_t begin, const size_t end, const dims &shape) {
    for (auto idx = begin; idx < end; ++idx) {
        set_shape_and_strides(*lts[idx], shape);
    }
}

/// infer the padding sizes according auto_pad type
status_t infer_auto_pad(const dim_t input_size, const dim_t stride,
        const dim_t kernel, const dim_t dilation, const std::string &auto_pad,
        dim_t &pad_begin, dim_t &pad_end, bool is_deconv) {
    if (auto_pad == "VALID") {
        pad_begin = 0;
        pad_end = 0;
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        dim_t effective_kernel = (kernel - 1) * dilation + 1;
        dim_t total_padding_size = 0;
        // calculate total padding size
        if (is_deconv) {
            total_padding_size = effective_kernel - stride;
        } else {
            if (input_size % stride == 0) {
                total_padding_size = effective_kernel - stride;
            } else {
                total_padding_size = effective_kernel - (input_size % stride);
            }
        }
        // padding size should not be negative
        if (total_padding_size < 0) { total_padding_size = 0; }
        pad_begin = auto_pad == "SAME_LOWER" ? ((total_padding_size + 1) / 2)
                                             : (total_padding_size / 2);
        pad_end = total_padding_size - pad_begin;
    } else {
        if (auto_pad != "NONE") return status::invalid_arguments;
    }

    return status::success;
}

/// numpy broadcasting
/// TODO(xxx): 0-D broadcasting?
status_t broadcast(const dims &lhs, const dims &rhs, dims &broadcasted) {
    const size_t lhs_rank = lhs.size();
    const size_t rhs_rank = rhs.size();
    const size_t max_rank = std::max(lhs_rank, rhs_rank);

    broadcasted.resize(max_rank);
    const size_t bl = max_rank - lhs_rank;
    const size_t br = max_rank - rhs_rank;

    for (size_t index = 0; index < max_rank; ++index) {
        dim_t l = 1, r = 1;
        if (index >= bl) l = lhs[index - bl];
        if (index >= br) r = rhs[index - br];
        if (l != r) {
            if (l != 1 && r != 1) return status::invalid_shape;
            broadcasted[index] = (l == 1 ? r : l);
        } else {
            broadcasted[index] = l;
        }
    }

    return status::success;
}

//one-way broadcasting
status_t one_way_broadcast(const dims &dst_shape, const dims &src_shape) {
    const size_t dst_rank = dst_shape.size();
    const size_t src_rank = src_shape.size();

    if (dst_rank < src_rank)
        return status::invalid_shape;
    else {
        // case 1: two tensors have exactly the same shape
        // case 2: after rightmost alignment, the length of each
        // dimensions is either a common length or rhs's length is 1.
        const size_t br = dst_rank - src_rank;
        dim_t dst_dim = 1, src_dim = 1;
        for (size_t index = src_rank - 1; index < src_rank; --index) {
            src_dim = src_shape[index];
            dst_dim = dst_shape[index + br];
            if (dst_dim != src_dim && src_dim != 1)
                return status::invalid_shape;
            if (0 == index) break;
        }
    }

    return status::success;
}

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
inline void infer_conv_ncx_oix(const dims &src_dims, const dims &fil_dims,
        const dims &strides, const dims &dilations, const dims &pads_begin,
        const dims &pads_end, dims &output_dims) {
    output_dims[0] = src_dims[0]; // n
    output_dims[1] = fil_dims[0]; // c
    for (size_t i = 2; i < src_dims.size(); ++i) {
        dim_t padded = src_dims[i] + pads_begin[i - 2] + pads_end[i - 2];
        dim_t dilated = dilations[i - 2] * (fil_dims[i] - 1) + 1;
        output_dims[i] = ((padded - dilated) / strides[i - 2]) + 1;
    }
}

status_t infer_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]); // src
    auto in1 = logical_tensor_wrapper_t(inputs[1]); // filter
    auto out0 = logical_tensor_wrapper_t(outputs[0]); // output

    // get attr value
    const dim_t g = n->get_attr<dim_t>(op_attr::groups);
    const auto &strides = n->get_attr<dims>(op_attr::strides);
    const auto &dilations = n->get_attr<dims>(op_attr::dilations);
    const auto &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const auto &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string fil_fmt = n->get_attr<std::string>(op_attr::weights_format);
    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);

    // avoid dividing by zero at below.
    if (g == 0) return status::invalid_shape;
    // check if src channel / groups == weight input channel
    VCHECK_INVALID_SHAPE(
            (in0.get_src_c(src_fmt) / g == in1.get_weight_i(fil_fmt)),
            "%s, the source channel divided by groups should be equal to the "
            "weight input channels, given source input channel: %d, group: "
            "%d, weight input channel: %d ",
            op_t::kind2str(n->get_kind()).c_str(),
            static_cast<int>(in0.get_src_c(src_fmt)), static_cast<int>(g),
            static_cast<int>(in1.get_weight_i(fil_fmt)));

    // spatial dims
    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    const auto invalid_strides_and_dilations_cond
            = (strides.size() != src_sp.size()
                    || dilations.size() != fil_sp.size()
                    || new_pads_begin.size() != src_sp.size()
                    || new_pads_end.size() != src_sp.size());
    VCHECK_INVALID_SHAPE(!invalid_strides_and_dilations_cond,
            "%s, the strides and dilations are required and should be "
            "correctly provided ",
            op_t::kind2str(n->get_kind()).c_str());

    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto padding sizes
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], fil_sp[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }

        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    // infer output shape
    dims output_dims(in0.vdims());
    infer_conv_ncx_oix(canonicalize(in0.vdims(), src_fmt),
            canonicalize(in1.vdims(), fil_fmt), strides, dilations,
            new_pads_begin, new_pads_end, output_dims);
    // output shape should have the same format as input data
    if ("NXC" == src_fmt) { output_dims = ncx2nxc(output_dims); }

    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(output_dims, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], output_dims);
    return status::success;
}

status_t infer_conv_bprop_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in1 = logical_tensor_wrapper_t(inputs[1]); // filter
    auto out = logical_tensor_wrapper_t(outputs[0]); // dst
    dims output_shape(in1.ndims());
    if (!out.is_shape_unknown()) {
        // use output shape if known
        output_shape = out.vdims();
    } else {
        // not support shape inference with runtime shape tesnor
        if (inputs.size() > 2) return status::unimplemented;
        if (!n->has_attr(op_attr::dst_shape)) return status::unimplemented;
        output_shape = n->get_attr<dims>(op_attr::dst_shape);
    };

    // get attr value
    const auto &strides = n->get_attr<dims>(op_attr::strides);
    const auto &dilations = n->get_attr<dims>(op_attr::dilations);
    const auto &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const auto &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string fil_fmt = n->get_attr<std::string>(op_attr::weights_format);
    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);

    // spatial dims
    dims src_sp = output_shape;
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);
    if (src_fmt == "NCX") {
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
    } else if (src_fmt == "NXC") {
        src_sp.erase(src_sp.begin(), src_sp.begin() + 1);
        src_sp.erase(src_sp.end() - 1, src_sp.end());
    } else {
        return status::unimplemented;
    }

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    const auto invalid_strides_and_dilations_cond
            = (strides.size() != src_sp.size()
                    || dilations.size() != fil_sp.size()
                    || new_pads_begin.size() != src_sp.size()
                    || new_pads_end.size() != src_sp.size());
    VCHECK_INVALID_SHAPE(!invalid_strides_and_dilations_cond,
            "%s, the strides and dilations are required and should be "
            "correctly provided ",
            op_t::kind2str(n->get_kind()).c_str());

    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);

        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], fil_sp[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }

        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    set_shape_and_strides(*outputs[0], output_shape);

    return status::success;
}

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
inline void infer_convtranspose_ncx_oix(const dims &src_dims,
        const dims &fil_dims, const dims &strides, const dims &dilations,
        const dims &pads_begin, const dims &pads_end, dims &output_dims) {
    output_dims[0] = src_dims[0]; // n
    output_dims[1] = fil_dims[1]; // ic
    for (size_t i = 2; i < src_dims.size(); ++i) {
        dim_t padded = src_dims[i] + pads_begin[i - 2] + pads_end[i - 2];
        dim_t dilated = dilations[i - 2] * (fil_dims[i] - 1) + 1;
        output_dims[i] = ((padded - dilated) / strides[i - 2]) + 1;
    }
}

status_t infer_convtranspose_bprop_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]); // src
    auto in1 = logical_tensor_wrapper_t(inputs[1]); // filter
    auto out0 = logical_tensor_wrapper_t(outputs[0]); // output

    // get attr value
    const dim_t g = n->get_attr<dim_t>(op_attr::groups);
    const auto &strides = n->get_attr<dims>(op_attr::strides);
    const auto &dilations = n->get_attr<dims>(op_attr::dilations);
    const auto &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const auto &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string fil_fmt = n->get_attr<std::string>(op_attr::weights_format);
    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);

    // avoid dividing by zero at below.
    if (g == 0) return status::invalid_shape;
    // check if src channel / groups == weight output channel
    VCHECK_INVALID_SHAPE(
            (in0.get_src_c(src_fmt) / g == in1.get_weight_o(fil_fmt)),
            "%s, src channel divided by groups should be equal to weight "
            "output channel,  src channel:%d, g:%d, weight output channel: "
            "%d ",
            op_t::kind2str(n->get_kind()).c_str(),
            static_cast<int>(in0.get_src_c(src_fmt)), static_cast<int>(g),
            static_cast<int>(in1.get_weight_o(fil_fmt)));

    // spatial dims
    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    const auto invalid_strides_and_dilations_cond
            = (strides.size() != src_sp.size()
                    || dilations.size() != fil_sp.size()
                    || new_pads_begin.size() != src_sp.size()
                    || new_pads_end.size() != src_sp.size());
    VCHECK_INVALID_SHAPE(!invalid_strides_and_dilations_cond,
            "%s, the strides and dilations are required and should be "
            "correctly provided ",
            op_t::kind2str(n->get_kind()).c_str());

    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto padding sizes
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], fil_sp[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }

        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    // infer output shape
    dims output_dims(in0.vdims());
    infer_convtranspose_ncx_oix(canonicalize(in0.vdims(), src_fmt),
            canonicalize(in1.vdims(), fil_fmt), strides, dilations,
            new_pads_begin, new_pads_end, output_dims);
    // output shape should have the same format as input data
    if ("NXC" == src_fmt) { output_dims = ncx2nxc(output_dims); }

    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(output_dims, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], output_dims);
    return status::success;
}

status_t infer_conv_bprop_filters_output_shape_common(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs, const size_t in_num) {
    // src for conv and diff_dst for convtranspose
    auto in = logical_tensor_wrapper_t(inputs[in_num]);
    auto out = logical_tensor_wrapper_t(outputs[0]); // diff_wei
    dims filter_shape(in.ndims());
    if (!out.is_shape_unknown()) {
        // use output shape if known
        filter_shape = out.vdims();
    } else {
        // not support shape inference with runtime shape tesnor
        if (!n->has_attr(op_attr::weights_shape)) return status::unimplemented;
        filter_shape = n->get_attr<dims>(op_attr::weights_shape);
    };

    // get attr value
    const auto &strides = n->get_attr<dims>(op_attr::strides);
    const auto &dilations = n->get_attr<dims>(op_attr::dilations);
    const auto &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const auto &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string fil_fmt = n->get_attr<std::string>(op_attr::weights_format);
    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);

    // spatial dims
    dims src_sp = in.get_src_spatial_dims(src_fmt);
    dims fil_sp = filter_shape;
    if (fil_fmt == "OIX" || fil_fmt == "IOX") {
        fil_sp.erase(fil_sp.begin(), fil_sp.begin() + 2);
    } else if (fil_fmt == "XIO" || fil_fmt == "XOI") {
        fil_sp.erase(fil_sp.end() - 2, fil_sp.end());
    } else {
        return status::unimplemented;
    }

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    const auto invalid_strides_and_dilations_cond
            = (strides.size() != src_sp.size()
                    || dilations.size() != fil_sp.size()
                    || new_pads_begin.size() != src_sp.size()
                    || new_pads_end.size() != src_sp.size());
    VCHECK_INVALID_SHAPE(!invalid_strides_and_dilations_cond,
            "%s, the strides and dilations are required and should be "
            "correctly provided ",
            op_t::kind2str(n->get_kind()).c_str());

    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto padding sizes
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], fil_sp[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }

        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    set_shape_and_strides(*outputs[0], filter_shape);

    return status::success;
}

status_t infer_convtranspose_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t diff_dst_in_num = 1;
    return infer_conv_bprop_filters_output_shape_common(
            n, inputs, outputs, diff_dst_in_num);
}

status_t infer_conv_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t src_in_num = 0;
    return infer_conv_bprop_filters_output_shape_common(
            n, inputs, outputs, src_in_num);
}

status_t infer_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto in1 = logical_tensor_wrapper_t(inputs[1]);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    // get attr value
    const dim_t g = n->get_attr<dim_t>(op_attr::groups);
    const auto &strides = n->get_attr<dims>(op_attr::strides);
    const auto &dilations = n->get_attr<dims>(op_attr::dilations);
    const auto &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const auto &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string fil_fmt = n->get_attr<std::string>(op_attr::weights_format);
    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);

    // avoid dividing by zero at below.
    if (g == 0) return status::invalid_shape;

    if (!out0.is_shape_unknown()) {
        // check if dst channel / groups == weight output channel
        VCHECK_INVALID_SHAPE(
                (out0.get_src_c(src_fmt) / g == in1.get_weight_o(fil_fmt)),
                "%s, the dst channel divided by groups should be equal to "
                "weight output channel. dst channel: %d, group: %d, weight "
                "output channel: %d ",
                op_t::kind2str(n->get_kind()).c_str(),
                static_cast<int>(out0.get_src_c(src_fmt)), static_cast<int>(g),
                static_cast<int>(in1.get_weight_o(fil_fmt)));
    }

    dims src_sp = in0.get_src_spatial_dims(src_fmt);
    dims fil_sp = in1.get_weight_spatial_dims(fil_fmt);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }

    // strides and dilations are required and should be correctly provided.
    const auto invalid_strides_and_dilations_cond
            = (strides.size() != src_sp.size()
                    || dilations.size() != fil_sp.size()
                    || new_pads_begin.size() != src_sp.size()
                    || new_pads_end.size() != src_sp.size());
    VCHECK_INVALID_SHAPE(!invalid_strides_and_dilations_cond,
            "%s, the strides and dilations are required and should be "
            "correctly provided ",
            op_t::kind2str(n->get_kind()).c_str());

    dims output_padding(src_sp.size(), 0);
    if (n->has_attr(op_attr::output_padding)) {
        output_padding = n->get_attr<dims>(op_attr::output_padding);
    }

    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);

        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], fil_sp[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i],
                    true);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }

        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    dims output_sp;
    for (size_t i = 0; i < src_sp.size(); ++i) {
        dim_t padded = output_padding[i] - new_pads_begin[i] - new_pads_end[i];
        dim_t dilated = dilations[i] * (fil_sp[i] - 1) + 1;
        output_sp.push_back(strides[i] * (src_sp[i] - 1) + dilated + padded);
    }

    const dims out0_shape = make_data_dims(
            src_fmt, in0.get_src_n(), in1.get_weight_o(fil_fmt) * g, output_sp);

    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out0_shape, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], out0_shape);

    return status::success;
}

status_t infer_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    // get attr value
    const dims &strides = n->get_attr<dims>(op_attr::strides);
    const dims &kernel = n->get_attr<dims>(op_attr::kernel);
    const dims &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const dims &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string rounding_type = "floor";
    if (n->has_attr(op_attr::rounding_type)) {
        rounding_type = n->get_attr<std::string>(op_attr::rounding_type);
    }
    std::string src_format = n->get_attr<std::string>(op_attr::data_format);

    dims dilations(kernel.size(), 1);
    if (n->has_attr(op_attr::dilations)) {
        dilations = n->get_attr<dims>(op_attr::dilations);
    }

    const dims src_dims = in0.vdims();

    dims src_sp = in0.get_src_spatial_dims(src_format);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }
    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], kernel[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }
        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    dims output_sp;
    for (size_t i = 0; i < src_sp.size(); ++i) {
        dim_t padded = src_sp[i] + new_pads_begin[i] + new_pads_end[i];
        dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
        dim_t out_value;
        if (rounding_type == "ceil") {
            out_value = utils::div_up(padded - dilated, strides[i]) + 1;
        } else {
            out_value = (padded - dilated) / strides[i] + 1;
        }
        output_sp.push_back(out_value);
    }

    dims out_shape = make_data_dims(
            src_format, in0.get_src_n(), in0.get_src_c(src_format), output_sp);
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out_shape, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], out_shape);
    return status::success;
}

status_t infer_pool_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    const bool is_maxpool = n->get_kind() == op_kind::MaxPoolBackward;

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        dims src_shape;
        if (is_maxpool) {
            // maxpool bwd
            src_shape = in0.vdims();
        } else if (n->has_attr(op_attr::src_shape)) {
            // avgpool bwd with src_shape attribute
            src_shape = n->get_attr<dims>(op_attr::src_shape);
        } else {
            // should not reach here. As dnnl backend does not support runtime input for avgpool bwd now, should return unimplemented
            return status::unimplemented;
        }
        VCHECK_INVALID_SHAPE(validate(src_shape, out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    if (is_maxpool) {
        // We should compute output dense strides instead of directly copying
        // input strides to it
        set_shape_and_strides(*outputs[0], in0.vdims());
    } else {
        // AvgPoolBackward
        dims diff_src_shape(in0.ndims());
        if (!out0.is_shape_unknown()) {
            // use output shape if known
            diff_src_shape = out0.vdims();
        } else {
            // not support shape inference with runtime shape tesnor
            if (inputs.size() > 1) return status::unimplemented;
            if (!n->has_attr(op_attr::src_shape)) return status::unimplemented;
            diff_src_shape = n->get_attr<dims>(op_attr::src_shape);
        };
        set_shape_and_strides(*outputs[0], diff_src_shape);
    }

    // get attr value
    const dims &strides = n->get_attr<dims>(op_attr::strides);
    const dims &kernel = n->get_attr<dims>(op_attr::kernel);
    const dims &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const dims &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string src_format = n->get_attr<std::string>(op_attr::data_format);

    dims dilations(kernel.size(), 1);
    if (n->has_attr(op_attr::dilations)) {
        dilations = n->get_attr<dims>(op_attr::dilations);
    }

    // out0 is the diff_src, has same shape with src
    const dims src_dims = out0.vdims();
    dims src_sp = out0.get_src_spatial_dims(src_format);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }
    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], kernel[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);
            VCHECK_INVALID_SHAPE((ret == status::success),
                    "%s, auto padding attribute can only be set to the "
                    "following values: VALID, SAME_UPPER, SAME_LOWER, NONE. "
                    "given value: %s",
                    op_t::kind2str(n->get_kind()).c_str(), auto_pad.c_str());
        }
        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    return status::success;
}

status_t infer_matmul_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto in1 = logical_tensor_wrapper_t(inputs[1]);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    // get attr value
    bool transpose_a = false;
    if (n->has_attr(op_attr::transpose_a)) {
        transpose_a = n->get_attr<bool>(op_attr::transpose_a);
    }
    bool transpose_b = false;
    if (n->has_attr(op_attr::transpose_b)) {
        transpose_b = n->get_attr<bool>(op_attr::transpose_b);
    }
    const dims input0_dims = in0.vdims();
    const dims input1_dims = in1.vdims();
    size_t input0_rank = input0_dims.size();
    size_t input1_rank = input1_dims.size();
    dims updated_input0(input0_dims);
    dims updated_input1(input1_dims);
    if (transpose_a && input0_rank > 1) {
        std::swap(updated_input0[input0_rank - 2],
                updated_input0[input0_rank - 1]);
    }
    if (transpose_b && input1_rank > 1) {
        std::swap(updated_input1[input1_rank - 2],
                updated_input1[input1_rank - 1]);
    }

    dims inferred_out_shape;
    if (input0_rank == 1 && input1_rank == 1) {
        // matmul: incompatible arg shapes
        VCHECK_INVALID_SHAPE((updated_input0 == updated_input1),
                "%s, arg shapes are not compatible. input 0: %s, input 1: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str());

        // example: input0={1,1}, input1={1,1}, output={2}. According to spec,
        // output shape of two 1D tensors multiplication [S] x [S] is squeezed
        // to scalar.
        inferred_out_shape = {};
    } else if (input0_rank == 1) {
        // matmul: incompatible arg shapes
        VCHECK_INVALID_SHAPE(
                (updated_input0[0] == updated_input1[input1_rank - 2]),
                "%s, arg shapes are not compatible. input 0: %s, input 1: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str());

        // example: input0 shape {3}, input1 shape {2,3,4}, output shape {2,4}
        updated_input1.erase(
                updated_input1.begin() + static_cast<dim_t>(input1_rank) - 2);
        inferred_out_shape = std::move(updated_input1);
    } else if (input1_rank == 1) {
        // matmul: incompatible arg shapes
        VCHECK_INVALID_SHAPE(
                (updated_input1[0] == updated_input0[input0_rank - 1]),
                "%s, arg shapes are not compatible. input 0: %s, input 1: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str());
        // example: input0 shape {2,3,4}, input1 shape {4}, output shape {2,3}
        updated_input0.erase(
                updated_input0.begin() + static_cast<dim_t>(input0_rank) - 1);
        inferred_out_shape = std::move(updated_input0);
    } else if (input0_rank == 2 && input1_rank == 2) {
        // matmul: incompatible arg shapes
        VCHECK_INVALID_SHAPE((updated_input0[1] == updated_input1[0]),
                "%s, arg shapes are not compatible. input 0: %s, input 1: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str());
        // example: input0 shape {1, 3}, input1 shape {3, 2}, output shape {1,2}
        inferred_out_shape = {updated_input0[0], updated_input1[1]};
    } else {
        // matmul: incompatible arg shapes
        VCHECK_INVALID_SHAPE((updated_input0[input0_rank - 1]
                                     == updated_input1[input1_rank - 2]),
                "%s, arg shapes are not compatible. input 0: %s, input 1: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str());

        std::vector<int64_t> input0_batch_dims {
                updated_input0.begin(), updated_input0.end() - 2};
        std::vector<int64_t> input1_batch_dims {
                updated_input1.begin(), updated_input1.end() - 2};
        status_t ret = broadcast(
                input0_batch_dims, input1_batch_dims, inferred_out_shape);
        VCHECK_INVALID_SHAPE((ret == status::success),
                "%s, failed to implement numpy broadcasting",
                op_t::kind2str(n->get_kind()).c_str());

        inferred_out_shape.push_back(updated_input0[input0_rank - 2]);
        inferred_out_shape.push_back(updated_input1[input1_rank - 1]);
    }

    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_shape, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_dropout_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in0.vdims(), out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], in0.vdims());

    if (outputs.size() > 1) {
        auto out1 = logical_tensor_wrapper_t(outputs[1]);
        if (out1.ndims() != -1) {
            VCHECK_INVALID_SHAPE(validate(in0.vdims(), out1.vdims()),
                    "%s, input and mask shapes are not compatible",
                    op_t::kind2str(n->get_kind()).c_str());
        }
        set_shape_and_strides(*outputs[1], in0.vdims());
    }

    UNUSED(n);
    return status::success;
}

status_t infer_identity_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in0.vdims(), out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    // We should compute output dense strides instead of directly copying input
    // strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    UNUSED(n);
    return status::success;
}

status_t infer_softmax_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in0.vdims(), out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    // We should compute output dense strides instead of directly copying input
    // strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    if (outputs.size() == 1) return status::success;

    auto out1 = logical_tensor_wrapper_t(outputs[1]);
    dims out1_dims = in0.vdims();
    int64_t axis = n->get_attr<int64_t>(op_attr::axis);
    VCHECK_INVALID_SHAPE((axis >= -in0.ndims() && axis < in0.ndims()),
            "%s, axis should be in range [-%d, %d), given axis: %" PRId64 " ",
            op_t::kind2str(n->get_kind()).c_str(), in0.ndims(), in0.ndims(),
            axis);
    if (axis < 0) { axis += in0.ndims(); }
    out1_dims[axis] = 1;

    if (out1.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out1_dims, out1.vdims()),
                "%s, given stats shape is not compatible with inferred",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[1], out1_dims);

    return status::success;
}

status_t identity_output_shape_on_pos(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        std::vector<std::pair<uint32_t, uint32_t>> &positions) {
    for (auto &pos : positions) {
        std::vector<logical_tensor_t *> ins = {inputs[pos.first]};
        std::vector<logical_tensor_t *> outs = {outputs[pos.second]};
        auto status = infer_identity_output_shape(n, ins, outs);
        if (status != status::success) return status;
    }
    return status::success;
}

status_t infer_bias_backprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in = logical_tensor_wrapper_t(inputs[0]);
    dims input_dims = in.vdims();
    VCHECK_INVALID_SHAPE((input_dims.size() >= 4),
            "%s, should have at least 4 dims, given dims: %zu ",
            op_t::kind2str(n->get_kind()).c_str(), input_dims.size());

    auto out = logical_tensor_wrapper_t(outputs[0]);
    std::string fmt = n->has_attr(op_attr::data_format)
            ? n->get_attr<std::string>(op_attr::data_format)
            : "NXC";
    const auto channels = in.get_src_c(fmt);

    if (!out.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE((channels == out.vdims()[0]),
                "%s, given output shape is not compatible with channels",
                op_t::kind2str(n->get_kind()).c_str());
        return status::success;
    }
    dims new_out_dims = {channels};

    set_shape_and_strides(*outputs[0], new_out_dims);
    return status::success;
}

status_t infer_bias_add_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {

    auto in = logical_tensor_wrapper_t(inputs[0]);
    auto out = logical_tensor_wrapper_t(outputs[0]);
    if (!out.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(in.vdims(), out.vdims()),
                "%s, given input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        return status::success;
    }

    dims input_dims = in.vdims();
    VCHECK_INVALID_SHAPE((input_dims.size() >= 2),
            "%s, input should have at least 2 dims, given dims: %zu ",
            op_t::kind2str(n->get_kind()).c_str(), input_dims.size());

    auto bias = logical_tensor_wrapper_t(inputs[1]);
    VCHECK_INVALID_SHAPE((bias.ndims() == 1),
            "%s, the bias input should have exactly 1 dim, given dims: %d ",
            op_t::kind2str(n->get_kind()).c_str(), bias.ndims());

    // following the spec of convolution, nxc as default format
    std::string fmt = n->has_attr(op_attr::data_format)
            ? n->get_attr<std::string>(op_attr::data_format)
            : "NXC";

    const auto channels = in.get_src_c(fmt);
    dims bias_dims = bias.vdims();
    VCHECK_INVALID_SHAPE((bias_dims[0] == channels),
            "%s, the bias size should match input channel size, given bias "
            "size: %d ",
            op_t::kind2str(n->get_kind()).c_str(),
            static_cast<int>(bias_dims[0]));

    return infer_identity_output_shape(n, inputs, outputs);
}

status_t infer_norm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto status = infer_identity_output_shape(n, inputs, outputs);
    if (status != status::success) return status;

    const auto is_rms = n->get_kind() == op_kind::RMSNorm;
    if (is_rms) return status::success;

    const bool keep_stats = n->has_attr(op_attr::keep_stats)
            ? n->get_attr<bool>(op_attr::keep_stats)
            // Keep default value as which in op_schema
            : true;
    if (!keep_stats) return status::success;

    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    const dims input0_dims = in0.vdims();

    const dim_t begin_norm_axis = n->has_attr(op_attr::begin_norm_axis)
            ? n->get_attr<dim_t>(op_attr::begin_norm_axis)
            : -1;
    // The backend pattern only checks this against the rank known at add_op,
    // which can differ from the rank given at compile.
    VCHECK_INVALID_SHAPE(
            (begin_norm_axis >= -in0.ndims() && begin_norm_axis < in0.ndims()),
            "%s, begin_norm_axis should be in range [-%d, %d), given: %" PRId64
            " ",
            op_t::kind2str(n->get_kind()).c_str(), in0.ndims(), in0.ndims(),
            begin_norm_axis);

    auto out1 = logical_tensor_wrapper_t(outputs[1]);
    auto out2 = logical_tensor_wrapper_t(outputs[2]);
    dims output_dims(input0_dims);

    auto norm_starting_position
            = begin_norm_axis >= 0 ? output_dims.begin() : output_dims.end();
    output_dims.erase(
            norm_starting_position + begin_norm_axis, output_dims.end());

    // check if output shape is already known
    if (out1.is_shape_unknown()) {
        set_shape_and_strides(*outputs[1], output_dims);
    }

    // check if output shape is already known
    if (out2.is_shape_unknown()) {
        set_shape_and_strides(*outputs[2], output_dims);
    }
    return status::success;
}

status_t infer_norm_bprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    std::vector<std::pair<uint32_t, uint32_t>> identity_shapes_pos = {{0, 0}};
    if (n->has_attr(op_attr::use_affine)
            && n->get_attr<bool>(op_attr::use_affine) == true) {
        // when use_affine parameter is set,
        // there will be two additional outputs
        identity_shapes_pos.insert(identity_shapes_pos.end(), {{4, 1}, {4, 2}});
    }
    return identity_output_shape_on_pos(
            n, inputs, outputs, identity_shapes_pos);
}

status_t infer_select_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto in1 = logical_tensor_wrapper_t(inputs[1]);
    auto in2 = logical_tensor_wrapper_t(inputs[2]);

    const bool shapes_should_match = n->has_attr(op_attr::auto_broadcast)
            ? "none" == n->get_attr<std::string>(op_attr::auto_broadcast)
            : false;

    dims input0_dims = in0.vdims();
    dims input1_dims = in1.vdims();
    dims input2_dims = in2.vdims();
    dims inferred_out_shape;

    if (shapes_should_match) { // no broadcast
        VCHECK_INVALID_SHAPE(
                (input0_dims == input1_dims && input1_dims == input2_dims),
                "%s, all input dims should match each other if there is no "
                "broadcast. input0 dims: %s, input1 dims: %s, input2 dims: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str(),
                dims2str(input2_dims).c_str());
        inferred_out_shape = std::move(input0_dims);
    } else { // can broadcast
        status_t ret1 = broadcast(input1_dims, input2_dims, inferred_out_shape);
        VCHECK_INVALID_SHAPE((ret1 == status::success),
                "%s, failed to implement numpy broadcasting",
                op_t::kind2str(n->get_kind()).c_str());
        status_t ret2 = one_way_broadcast(inferred_out_shape, input0_dims);
        VCHECK_INVALID_SHAPE((ret2 == status::success),
                "%s, failed to implement one-way broadcasting",
                op_t::kind2str(n->get_kind()).c_str());
    }

    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    // check if given or partial set shape aligns with inferred shape
    if (!out0.is_shape_unknown() || out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_shape, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        if (!out0.is_shape_unknown()) return status::success;
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_elemwise_arithmetic_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto in1 = logical_tensor_wrapper_t(inputs[1]);
    // check if output shape is already known
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    const bool shapes_should_match = n->has_attr(op_attr::auto_broadcast)
            ? "none" == n->get_attr<std::string>(op_attr::auto_broadcast)
            : false;

    dims input0_dims = in0.vdims();
    dims input1_dims = in1.vdims();
    dims inferred_out_shape;
    if (shapes_should_match) {
        VCHECK_INVALID_SHAPE((input0_dims == input1_dims),
                "%s, incompatible input shapes (auto_broadcast=none) ",
                op_t::kind2str(n->get_kind()).c_str());

        inferred_out_shape = std::move(input0_dims);
    } else {
        status_t ret = broadcast(input0_dims, input1_dims, inferred_out_shape);
        VCHECK_INVALID_SHAPE((ret == status::success),
                "%s, failed to implement numpy broadcasting",
                op_t::kind2str(n->get_kind()).c_str());
    }
    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_shape, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_bn_fwd_train_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (every_shape_is_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper_t(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    // Graph API supports 0d spatial input of batchnorm at minimum,
    // of which the input dim size is 2
    VCHECK_INVALID_SHAPE((input_dims.size() >= 2),
            "%s, input dim size of batch norm should be at least 2, input dims "
            "size: %zu ",
            op_t::kind2str(n->get_kind()).c_str(), input_dims.size());

    std::string fmt = n->has_attr(op_attr::data_format)
            ? n->get_attr<std::string>(op_attr::data_format)
            : "NXC";

    const auto channels = in.get_src_c(fmt);
    const auto validator = [&channels](cvec_int64 &vec) {
        return vec.size() == 1 && vec[0] == channels;
    };
    if (!verify_shapes_in_range(inputs, 1, inputs.size(), validator))
        return status::invalid_shape;

    infer_identity_output_shape(n, inputs, outputs);
    cvec_int64 new_out_dims = {channels};
    set_shapes_in_range(outputs, 1, 5 /* output number*/, new_out_dims);
    return status::success;
}

status_t infer_bn_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using cvec_int64 = const std::vector<int64_t>;

    if (every_shape_is_known(outputs)) return status::success;

    const auto in = logical_tensor_wrapper_t(inputs[0]);
    cvec_int64 input_dims = in.vdims();
    const auto out_delta = logical_tensor_wrapper_t(inputs[1]);
    cvec_int64 out_delta_dims = out_delta.vdims();
    VCHECK_INVALID_SHAPE((input_dims.size() >= 4 && out_delta_dims.size() >= 4),
            "%s, dims range should not be less than 4, input dims size: %zu, "
            "output delta dims size: %zu",
            op_t::kind2str(n->get_kind()).c_str(), input_dims.size(),
            out_delta_dims.size());

    std::string fmt = n->has_attr(op_attr::data_format)
            ? n->get_attr<std::string>(op_attr::data_format)
            : "NXC";

    const auto channels = in.get_src_c(fmt);
    const auto validator = [&channels](cvec_int64 &vec) {
        return vec.size() == 1 && vec[0] == channels;
    };
    if (!verify_shapes_in_range(inputs, 2, inputs.size(), validator))
        return status::invalid_shape;

    infer_identity_output_shape(n, inputs, outputs);
    cvec_int64 new_out_dims = {channels};
    set_shapes_in_range(outputs, 1,
            std::min(outputs.size(),
                    static_cast<size_t>(3) /* max output number*/),
            new_out_dims);
    return status::success;
}

status_t infer_concat_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    // if only one tensor to concat, out_shape is same as input_shape
    if (inputs.size() == 1) {
        infer_identity_output_shape(n, inputs, outputs);
        return status::success;
    }
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto data_type = in0.data_type();
    if (data_type != out0.data_type()) return status::unimplemented;

    int64_t axis = n->get_attr<int64_t>(op_attr::axis);
    auto ndims = in0.ndims();
    auto dims = in0.dims();
    if (axis < -ndims || axis >= ndims) {
        return status::invalid_arguments;
    } else if (axis < 0) {
        axis += ndims;
    }

    int64_t sum = 0;
    for (auto iter = inputs.cbegin(); iter != inputs.cend(); iter++) {
        auto lt_inN = logical_tensor_wrapper_t(*iter);
        const auto &lt_inN_dims = lt_inN.vdims();
        VCHECK_INVALID_SHAPE((lt_inN.ndims() == ndims),
                "%s, input dims size should be equal to lt dims size. "
                "input dims size: %d, lt input size: %d  ",
                op_t::kind2str(n->get_kind()).c_str(), ndims, lt_inN.ndims());

        if (lt_inN.data_type() != data_type) { return status::unimplemented; }
        for (int32_t i = 0; i < ndims; i++) {
            if (i != axis) {
                // input dims should be same except axis dim.
                VCHECK_INVALID_SHAPE(
                        (dims[i] == lt_inN_dims[static_cast<size_t>(i)]),
                        "%s, input dims should be same except axis dim. "
                        "dims[i]: %d, dim in logical tensor: %d ",
                        op_t::kind2str(n->get_kind()).c_str(),
                        static_cast<int>(dims[i]),
                        static_cast<int>(lt_inN_dims[static_cast<size_t>(i)]));

            } else {
                sum += lt_inN_dims[static_cast<size_t>(axis)];
            }
        }
    };

    std::vector<int64_t> inferred_out_shape(dims, dims + ndims);
    inferred_out_shape[axis] = sum;

    // for known output shape, check if it's reasonable, which requires
    // inference as well
    if (!out0.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_shape, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        return status::success;
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_unsupported_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    UNUSED(inputs);
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    if (out0.is_shape_unknown()) return status::unimplemented;
    return status::success;
}

status_t infer_dummy_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    UNUSED(n);
    UNUSED(inputs);
    UNUSED(outputs);
    return status::success;
}

status_t infer_reduce_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);

    if (n->has_attr(op_attr::axes)) {
        auto axes = n->get_attr<dims>(op_attr::axes);
        // our backend doesn't support such a case
        if (axes.empty()) return status::unimplemented;

        auto shape = logical_tensor_wrapper_t(inputs[0]).vdims();
        auto ndim = static_cast<int64_t>(shape.size());

        if (std::any_of(axes.begin(), axes.end(), [&ndim](int64_t axis) {
            return axis < -ndim || axis >= ndim;
        }))
            return status::unimplemented;

        // convert negative axis to positive one
        std::transform(axes.begin(), axes.end(), axes.begin(),
                [&ndim](int64_t axis) -> int64_t {
            return axis < 0 ? ndim + axis : axis;
        });

        if (std::unordered_set<int64_t>(axes.begin(), axes.end()).size()
                < axes.size())
            return status::unimplemented;

        auto keep_dims = n->has_attr(op_attr::keep_dims)
                ? n->get_attr<bool>(op_attr::keep_dims)
                : false;
        for (auto axis : axes)
            shape[static_cast<size_t>(axis)] = (keep_dims) ? 1 : 0;
        if (!keep_dims)
            shape.erase(std::remove_if(shape.begin(), shape.end(),
                                [](int64_t d) { return d == 0; }),
                    shape.end());
        if (!out0.is_shape_unknown()) {
            VCHECK_INVALID_SHAPE(validate(shape, out0.vdims()),
                    "%s, inferred out shape and output shape are not "
                    "compatible",
                    op_t::kind2str(n->get_kind()).c_str());
        }

        set_shape_and_strides(*outputs[0], shape);

        return status::success;
    }

    // When the second input is an empty list, then this operation does nothing,
    // it is an identity. For dims.size() == 0, we set ndims to -1 and can't get
    // vdims. Therefore, this path is not supported.

    // since we don't have an access to the second input data, which contain
    // axis indices, we cannot calculate output shape
    return status::unimplemented;
}

status_t infer_static_reshape_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    const dims &in_dims = in0.vdims();
    dims out_dims = n->get_attr<dims>(op_attr::shape);
    // The inferred rank is taken directly from the user-provided shape
    // attribute, so bound it before it is written into the fixed-size dims of
    // the output logical tensor in set_shape_and_strides.
    VCHECK_INVALID_SHAPE(
            (out_dims.size() <= static_cast<size_t>(DNNL_MAX_NDIMS)),
            "%s, the number of output dims (%zu) should not exceed the maximum "
            "supported ndims (%d)",
            op_t::kind2str(n->get_kind()).c_str(), out_dims.size(),
            DNNL_MAX_NDIMS);
    const bool special_zero = n->get_attr<bool>(op_attr::special_zero);

    bool find_uncertain_dim = false; // shape contains -1
    size_t uncertain_axis = 0;
    for (size_t i = 0; i < out_dims.size(); i++) {
        VCHECK_INVALID_SHAPE((out_dims[i] >= -1),
                "%s, output dims should be larger than or equal to -1, output "
                "dim: %d  ",
                op_t::kind2str(n->get_kind()).c_str(),
                static_cast<int>(out_dims[i]));
        if (out_dims[i] == 0) {
            // handle special_zero: 0 means same as input shape in that
            // dimension if special_zero is true; or 0 as-is if special_zero is
            // false
            if (special_zero) {
                VCHECK_INVALID_SHAPE((i < static_cast<size_t>(in0.ndims())),
                        "%s, output dims size should be smaller than input "
                        "size for special zero, output dim size: %zu  ",
                        op_t::kind2str(n->get_kind()).c_str(), out_dims.size());
                out_dims[i] = in_dims[i];
            }
        } else if (out_dims[i] == -1) {
            // only allow at most one -1
            if (find_uncertain_dim) return status::invalid_shape;
            find_uncertain_dim = true;
            uncertain_axis = i;
        }
    }

    dim_t in_size = 1;
    dim_t out_size = 1;
    for (size_t i = 0; i < in_dims.size(); i++) {
        if (in_dims[i] >= 0) in_size *= in_dims[i];
    }
    for (size_t i = 0; i < out_dims.size(); i++) {
        if (out_dims[i] >= 0) out_size *= out_dims[i];
    }
    // handle -1 in output shape: the value is inferred from the size of the
    // input tensor and the remaining dimensions
    if (find_uncertain_dim) {
        VCHECK_INVALID_SHAPE((out_size != 0),
                "%s, output size is not allowed to be 0 for uncertain dims, "
                "output size: %" PRId64 " ",
                op_t::kind2str(n->get_kind()).c_str(), out_size);
        out_dims[uncertain_axis] = in_size / out_size;
    }

    // size of input should be same as output
    if (find_uncertain_dim == false) {
        VCHECK_INVALID_SHAPE((out_size == in_size),
                "%s, size of input should be same as output. input size: "
                "%" PRId64
                ", "
                "output size: %" PRId64 " ",
                op_t::kind2str(n->get_kind()).c_str(), in_size, out_size);
    } else {
        VCHECK_INVALID_SHAPE((out_size * out_dims[uncertain_axis] == in_size),
                "%s, the product of output size and output dim at uncertain "
                "axis should be equal to the input size. input size: %" PRId64
                ", "
                "output size: %" PRId64 ", output dim at uncertain axis: %d  ",
                op_t::kind2str(n->get_kind()).c_str(), in_size, out_size,
                static_cast<int>(out_dims[uncertain_axis]));
    }

    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    // check if given or partial set shape aligns with inferred shape
    if (!out0.is_shape_unknown() || out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out_dims, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        if (!out0.is_shape_unknown()) return status::success;
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], out_dims);
    return status::success;
}

status_t infer_static_transpose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = logical_tensor_wrapper_t(inputs[0]);

    const dims &in_dims = in0.vdims();
    const int32_t in_ndims = in0.ndims();
    std::vector<int64_t> order = n->get_attr<dims>(op_attr::order);
    std::vector<bool> order_covered_flg(in_ndims, false);
    // check order should be in [-n, n-1] and cover all input axis if order < 0,
    // convert it to positive order
    if (!order.empty()) {
        if (order.size() != static_cast<size_t>(in_ndims)) {
            return status::invalid_shape;
        }
        VCHECK_INVALID_SHAPE((order.size() == static_cast<size_t>(in_ndims)),
                "%s, order size should be equal to input dims size. order "
                "size: %zu, input dims size: %d  ",
                op_t::kind2str(n->get_kind()).c_str(), order.size(), in_ndims);

        for (int64_t &axis : order) {
            VCHECK_INVALID_SHAPE((!(axis < -in_ndims || axis > in_ndims - 1)),
                    "%s, order axis should be in appropriate range. axis: %d, "
                    "in_ndims: %d ",
                    op_t::kind2str(n->get_kind()).c_str(),
                    static_cast<int>(axis), in_ndims);

            if (axis < 0) axis += in_ndims;
            if (order_covered_flg[axis]) {
                return status::invalid_shape;
            } else {
                order_covered_flg[axis] = true;
            }
        }
    }

    dims out_dims;
    out_dims.reserve(in_ndims);
    if (order.empty()) {
        // If order is not given, will transpose to (n-1...0),
        for (int i = in_ndims - 1; i >= 0; --i)
            out_dims.push_back(in_dims[i]);
    } else {
        for (const int64_t &axis : order) {
            out_dims.push_back(
                    axis >= 0 ? in_dims[axis] : in_dims[axis + in_ndims]);
        }
    }

    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    // check if given or partial set shape aligns with inferred shape
    if (!out0.is_shape_unknown() || out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out_dims, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        if (!out0.is_shape_unknown()) return status::success;
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], out_dims);
    return status::success;
}

status_t infer_interpolate_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in = logical_tensor_wrapper_t(inputs[0]);
    auto in_dims = in.vdims();
    // Number of spatial dimensions
    int spatial_ndim = in.ndims() - 2;

    std::vector<int64_t> sizes;
    if (n->has_attr(op_attr::sizes)) {
        // sizes is set by user
        sizes = n->get_attr<std::vector<int64_t>>(op_attr::sizes);
    }
    std::vector<float> scales;
    if (n->has_attr(op_attr::scales)) {
        // scales is set by user
        scales = n->get_attr<std::vector<float>>(op_attr::scales);
    }

    std::string src_fmt = n->get_attr<std::string>(op_attr::data_format);
    int spatial_dim_start_axis = 0;
    if (src_fmt == "NXC") {
        // "X" start from in_dims[1]
        spatial_dim_start_axis = 1;
    } else if (src_fmt == "NCX") {
        // "X" start from in_dims[2]
        spatial_dim_start_axis = 2;
    } else {
        return status::invalid_arguments;
    }

    if (!scales.empty()) {
        // scales length should equal spatial_ndim
        if (scales.size() != static_cast<size_t>(spatial_ndim)) {
            return status::invalid_arguments;
        }

        // spatial_ndim
        for (size_t i = 0; i < static_cast<size_t>(spatial_ndim); i++) {
            const float d
                    = static_cast<float>(in_dims[i + spatial_dim_start_axis]);
            in_dims[i + spatial_dim_start_axis]
                    = static_cast<dim_t>(d * scales[i]);
        }
    }
    if (!sizes.empty()) {
        // sizes length should equal spatial_ndim
        if (sizes.size() != static_cast<size_t>(spatial_ndim)) {
            return status::invalid_arguments;
        }

        for (size_t i = 0; i < static_cast<size_t>(spatial_ndim); i++) {
            in_dims[i + spatial_dim_start_axis] = sizes[i];
        }
    }

    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    // if the output is given, check if it's reasonable
    if (!out0.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(in_dims, out0.vdims()),
                "%s, inferred output shape and shape from logical tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        return status::success;
    }

    set_shape_and_strides(*outputs[0], in_dims);
    return status::success;
}

status_t infer_prelu_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    std::vector<std::pair<uint32_t, uint32_t>> identity_shapes_pos
            = {{0, 0}, {1, 1}};
    return identity_output_shape_on_pos(
            n, inputs, outputs, identity_shapes_pos);
}

status_t infer_groupnorm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto status = infer_identity_output_shape(n, inputs, outputs);
    if (status != status::success) return status;

    const bool keep_stats = n->has_attr(op_attr::keep_stats)
            ? n->get_attr<bool>(op_attr::keep_stats)
            // Keep default value as which in op_schema
            : true;
    if (!keep_stats) return status::success;

    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    const dims input0_dims = in0.vdims();

    auto out1 = logical_tensor_wrapper_t(outputs[1]);
    auto out2 = logical_tensor_wrapper_t(outputs[2]);

    // GroupNorm
    if (!n->has_attr(op_attr::groups)) return status::invalid_arguments;
    const dim_t num_groups = n->get_attr<dim_t>(op_attr::groups);
    // output_dims[batch_size, num_groups]
    dims output_dims = {input0_dims[0], num_groups};

    // check if output shape is already known
    if (!out1.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(output_dims, out1.vdims()),
                "%s, `mean` inferred output shape and shape from logical "
                "tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    } else {
        set_shape_and_strides(*outputs[1], output_dims);
    }

    // check if output shape is already known
    if (!out2.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(output_dims, out2.vdims()),
                "%s, `variance` inferred output shape and shape from logical "
                "tensor are "
                "not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    } else {
        set_shape_and_strides(*outputs[2], output_dims);
    }

    return status::success;
}

using ltw = logical_tensor_wrapper_t;

static status_t infer_dnnl_conv_common_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        const size_t axis_with_groups) {
    bool canonicalized = n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized);
    const auto groups = n->get_attr<int64_t>(op_attr::groups);

    auto out = ltw(outputs[0]); // diff_wei
    if (canonicalized && groups > 1 && !out.is_shape_unknown()) {
        // convert the out shape to uncanonicalized form to reuse the frontend
        // shape infer function.
        auto out_dims = out.vdims();
        out_dims.erase(out_dims.begin());
        out_dims[axis_with_groups] *= out_dims[0];
        set_shape_and_strides(*outputs[0], out_dims);
    }

    // infer pad and filter shape (groups not included)
    const auto ret = infer_conv_bprop_filters_output_shape(n, inputs, outputs);
    if (ret != status::success) return ret;

    // add groups into weights shape
    if (canonicalized && groups > 1) {
        auto out_dims = ltw(outputs[0]).vdims();
        out_dims[axis_with_groups] /= groups;
        out_dims.insert(out_dims.begin(), groups);

        set_shape_and_strides(*outputs[0], out_dims);
    }

    return status::success;
}

status_t infer_dnnl_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = ltw(outputs[0]);
    const bool org_out_shape_unknown = out.is_shape_unknown();

    auto backup_wei_shape = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    if (n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized)
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_conv_output_shape(n, inputs, outputs);
    *inputs[1] = backup_wei_shape;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);

    // The following code will take effect only when fusing dw conv.
    // At this stage outputs[0] corresponds to conv_1x1 dst
    // we now just need to adjust oh and ow in case of dw_k3s2p1 post-op
    dims output_dims(ltw(outputs[0]).vdims());
    if (org_out_shape_unknown && n->has_attr(op_attr::dw_type)
            && n->get_attr<std::string>(op_attr::dw_type) == "k3s2p1") {
        const std::string src_fmt
                = n->get_attr<std::string>(op_attr::data_format);
        const size_t oh_offset
                = (src_fmt == "NCX") ? output_dims.size() - 2 : 1;
        const size_t ow_offset
                = (src_fmt == "NCX") ? output_dims.size() - 1 : 2;
        const dim_t stride = 2;
        const dim_t new_oh = (output_dims[oh_offset] + stride - 1) / stride;
        const dim_t new_ow = (output_dims[ow_offset] + stride - 1) / stride;
        output_dims[oh_offset] = new_oh;
        output_dims[ow_offset] = new_ow;
        set_shape_and_strides(*outputs[0], output_dims);
    }

    return status::success;
}

status_t infer_dnnl_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto backup = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    bool is_canonicalized = n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized);
    if (is_canonicalized
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        // [g, O/g, I/g, H, W]
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[2] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_convtranspose_output_shape(n, inputs, outputs);
    *inputs[1] = backup;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);
    return status::success;
}

status_t infer_dnnl_convtranspose_bwd_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto backup_wei_shape = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    if (n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized)
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[2] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_convtranspose_bprop_data_output_shape(n, inputs, outputs);
    *inputs[1] = backup_wei_shape;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);
    return status::success;
}

status_t infer_dnnl_convtranspose_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t axis_with_groups = 1;
    return infer_dnnl_conv_common_bwd_weight_output_shape(
            n, inputs, outputs, axis_with_groups);
}

status_t infer_dnnl_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    infer_pool_output_shape(n, inputs, outputs);
    return status::success;
}

status_t infer_permute_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = ltw(outputs[0]);

    auto in_dims = ltw(inputs[0]).vdims();
    auto perm = n->get_attr<std::vector<int64_t>>(op_attr::permutation);
    // validate permutation: correct rank, in-range indices, and no duplicates
    VCHECK_INVALID_SHAPE(perm.size() == in_dims.size(),
            "%s, permutation size (%zu) must match input rank (%zu)",
            op_t::kind2str(n->get_kind()).c_str(), perm.size(), in_dims.size());

    std::vector<bool> seen(perm.size(), false);
    for (size_t i = 0; i < perm.size(); ++i) {
        int64_t p = perm[i];
        VCHECK_INVALID_SHAPE(p >= 0 && p < static_cast<int64_t>(perm.size()),
                "%s, permutation index %ld at position %zu is out of range "
                "[0, %zu)",
                op_t::kind2str(n->get_kind()).c_str(), static_cast<long int>(p),
                i, perm.size());
        VCHECK_INVALID_SHAPE(!seen[static_cast<size_t>(p)],
                "%s, permutation index %ld appears more than once",
                op_t::kind2str(n->get_kind()).c_str(),
                static_cast<long int>(p));
        seen[static_cast<size_t>(p)] = true;
    }

    std::vector<dim_t> inferred_out_dims(perm.size(), DNNL_GRAPH_UNKNOWN_DIM);
    for (size_t i = 0; i < perm.size(); i++) {
        inferred_out_dims[perm[i]] = in_dims[i];
    }

    // check the given shape
    if (!out0.is_shape_unknown()) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_dims, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }
    set_shape_and_strides(*outputs[0], inferred_out_dims);

    return status::success;
}

status_t infer_to_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = ltw(outputs[0]);
    auto in0 = ltw(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    auto groups = n->get_attr<int64_t>(op_attr::groups);
    dims in_dims = in0.vdims();

    if (n->has_attr(op_attr::is_convtranspose)
            && n->get_attr<bool>(op_attr::is_convtranspose)) {
        in_dims[1] /= groups;
    } else {
        in_dims[0] /= groups;
    }
    in_dims.insert(in_dims.begin(), groups);

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], in_dims);
    UNUSED(n);
    return status::success;
}

status_t infer_from_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = ltw(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    const auto groups = n->get_attr<int64_t>(op_attr::groups);
    dims inferred_out_dims = ltw(inputs[0]).vdims();
    inferred_out_dims.erase(inferred_out_dims.begin());
    if (n->has_attr(op_attr::is_convtranspose)
            && n->get_attr<bool>(op_attr::is_convtranspose)) {
        inferred_out_dims[1] *= groups;
    } else {
        inferred_out_dims[0] *= groups;
    }

    set_shape_and_strides(*outputs[0], inferred_out_dims);

    return status::success;
}

status_t infer_unsqueeze_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto axes = (n->has_attr(op_attr::axes))
            ? n->get_attr<std::vector<int64_t>>(op_attr::axes)
            : std::vector<int64_t>();
    const auto in_dims = ltw(inputs[0]).vdims();

    const auto out_ndim = static_cast<int64_t>(in_dims.size() + axes.size());
    if (std::any_of(axes.begin(), axes.end(), [&out_ndim](int64_t axis) {
        return axis < -out_ndim || axis >= out_ndim;
    }))
        return status::unimplemented;

    // convert negative axis to positive one
    std::transform(axes.begin(), axes.end(), axes.begin(),
            [&out_ndim](int64_t axis) -> int64_t {
        return axis < 0 ? out_ndim + axis : axis;
    });

    if (std::unordered_set<int64_t>(axes.begin(), axes.end()).size()
            < axes.size())
        return status::unimplemented;

    std::vector<size_t> indices(out_ndim);
    std::iota(indices.begin(), indices.end(), 0);
    dims inferred_output_shape(out_ndim, 1);
    size_t in_dims_idx = 0;
    for (const auto i : indices) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end())
            inferred_output_shape[i] = in_dims[in_dims_idx++];
    }

    set_shape_and_strides(*outputs[0], inferred_output_shape);

    return status::success;
}

status_t infer_squeeze_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto in_dims = ltw(inputs[0]).vdims();
    auto in_ndim = in_dims.size();

    auto axes = (n->has_attr(op_attr::axes))
            ? n->get_attr<std::vector<int64_t>>(op_attr::axes)
            : std::vector<int64_t>();
    // convert negative axis to positive one
    std::transform(axes.begin(), axes.end(), axes.begin(),
            [&in_ndim](int64_t axis) -> int64_t {
        return axis < 0 ? axis + in_ndim : axis;
    });

    dims inferred_output_shape = {};
    for (size_t i = 0; i < in_ndim; ++i) {
        if (axes.empty() && in_dims[i] == 1) {
            continue;
        } else if (!axes.empty()
                && std::find(axes.begin(), axes.end(), i) != axes.end()) {
            if (in_dims[i] != 1) {
                // Dimension must be 1
                return status::invalid_arguments;
            }
        } else {
            inferred_output_shape.push_back(in_dims[i]);
        }
    }
    set_shape_and_strides(*outputs[0], inferred_output_shape);
    return status::success;
}

status_t infer_bn_folding_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = ltw(outputs[0]);
    auto out1 = ltw(outputs[1]);
    auto in0 = ltw(inputs[0]);
    auto in1 = ltw(inputs[1]);

    if (!out0.is_shape_unknown() && !out1.is_shape_unknown())
        return status::success;

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in0.vdims(), out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    if (out1.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in1.vdims(), out1.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    set_shape_and_strides(*outputs[1], in1.vdims());
    UNUSED(n);
    return status::success;
}

status_t infer_dnnl_conv_bwd_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto backup = *inputs[1];
    if (n->get_attr<int64_t>(op_attr::groups) > 1) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    auto ret = infer_conv_bprop_data_output_shape(n, inputs, outputs);
    if (ret != status::success) return ret;
    *inputs[1] = backup;
    return status::success;
}

status_t infer_dnnl_conv_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t axis_with_groups = 0;
    return infer_dnnl_conv_common_bwd_weight_output_shape(
            n, inputs, outputs, axis_with_groups);
}

status_t infer_dnnl_batchnorm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    status_t stat = status::success;
    if (n->get_attr<bool>(op_attr::is_training))
        stat = infer_bn_fwd_train_output_shape(n, inputs, outputs);
    else
        stat = infer_identity_output_shape(n, inputs, outputs);
    return stat;
}

status_t infer_dnnl_batchnorm_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // skip shape inference for scratchpad output
    // FIXME(wuxun): may remove this temporary solution after we refine op
    // definition to handle one or more optional input/outputs
    auto new_outputs = outputs;
    new_outputs.pop_back();
    infer_bn_bwd_output_shape(n, inputs, new_outputs);
    return status::success;
}

status_t infer_dnnl_constant_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // `dnnl_constant_[scales|zps]` op doesn't have any inputs
    auto out_shape = n->get_attr<std::vector<int64_t>>(op_attr::shape);
    set_shape_and_strides(*outputs[0], out_shape);

    return status::success;
}

status_t infer_dnnl_pool_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto diff_src_shape = n->get_attr<std::vector<int64_t>>(op_attr::src_shape);
    set_shape_and_strides(*outputs[0], diff_src_shape);

    // get attr value
    const dims &strides = n->get_attr<dims>(op_attr::strides);
    const dims &kernel = n->get_attr<dims>(op_attr::kernel);
    const dims &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const dims &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string src_format = n->get_attr<std::string>(op_attr::data_format);

    dims dilations(kernel.size(), 1);
    if (n->has_attr(op_attr::dilations)) {
        dilations = n->get_attr<dims>(op_attr::dilations);
        if (dilations.size() != kernel.size()) {
            return status::invalid_arguments;
        }
    }

    logical_tensor_wrapper_t diff_src_ltw(outputs[0]);
    dims src_sp = diff_src_ltw.get_src_spatial_dims(src_format);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }
    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], kernel[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);

            if (ret != status::success) return ret;
        }
        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    return status::success;
}

status_t infer_binary_select_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto in0 = ltw(inputs[0]);
    auto in1 = ltw(inputs[1]);
    auto in2 = ltw(inputs[2]);

    const bool shapes_should_match = n->has_attr(op_attr::auto_broadcast)
            ? "none" == n->get_attr<std::string>(op_attr::auto_broadcast)
            : false;

    dims input0_dims = in0.vdims();
    dims input1_dims = in1.vdims();
    dims input2_dims = in2.vdims();
    dims inferred_out_shape;

    if (shapes_should_match) { // no broadcast
        VCHECK_INVALID_SHAPE(
                (input0_dims == input1_dims && input1_dims == input2_dims),
                "%s, all input dims should match each other if there is no "
                "broadcast. input0 dims: %s, input1 dims: %s, input2 dims: %s ",
                op_t::kind2str(n->get_kind()).c_str(),
                dims2str(input0_dims).c_str(), dims2str(input1_dims).c_str(),
                dims2str(input2_dims).c_str());
        inferred_out_shape = std::move(input0_dims);
    } else { // can broadcast
        status_t ret1 = broadcast(input0_dims, input1_dims, inferred_out_shape);
        VCHECK_INVALID_SHAPE((ret1 == status::success),
                "%s, failed to implement numpy broadcasting",
                op_t::kind2str(n->get_kind()).c_str());
    }

    auto out0 = ltw(outputs[0]);
    // check if given or partial set shape aligns with inferred shape
    if (!out0.is_shape_unknown() || out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(inferred_out_shape, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
        if (!out0.is_shape_unknown()) return status::success;
    }

    set_shape_and_strides(*outputs[0], inferred_out_shape);
    return status::success;
}

status_t infer_dnnl_binary_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const bool is_bias_add = n->has_attr(op_attr::is_bias_add)
            && n->get_attr<bool>(op_attr::is_bias_add);
    const auto algo = n->get_attr<int64_t>(op_attr::alg_kind);
    if (algo == static_cast<int64_t>(alg_kind::binary_select)) {
        return infer_binary_select_output_shape(n, inputs, outputs);
    } else if (is_bias_add) {
        return infer_bias_add_output_shape(n, inputs, outputs);
    } else {
        return infer_elemwise_arithmetic_output_shape(n, inputs, outputs);
    }
}

status_t infer_dnnl_sdpa_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // [batch_size, num_heads_q, seq_len_q, head_size_qk]
    auto query = ltw(inputs[0]);
    // [batch_size, num_heads_q, head_size_qk, seq_len_kv,]
    auto key = ltw(inputs[1]);
    // [batch_size, num_heads_v, seq_len_kv, head_size_v]
    auto value = ltw(inputs[2]);
    // [batch_size, num_heads_q, seq_len_q, head_size_v]
    auto out0 = ltw(outputs[0]);

    dims query_dims = query.vdims();
    dims key_dims = key.vdims();
    dims value_dims = value.vdims();

    VCHECK_INVALID_SHAPE((query_dims.size() == key_dims.size()
                                 && key_dims.size() == value_dims.size()),
            "%s, all input dims should match each other. input0 dims: %s, "
            "input1 dims: %s, input2 dims: %s ",
            op_t::kind2str(n->get_kind()).c_str(), dims2str(query_dims).c_str(),
            dims2str(key_dims).c_str(), dims2str(value_dims).c_str());

    VCHECK_INVALID_SHAPE((query_dims.size() == 4),
            "%s, only support 4D input for all q/k/v. input0 dimension: %s, "
            "input1 dimension: %s, input2 dimension: %s ",
            op_t::kind2str(n->get_kind()).c_str(),
            std::to_string(query_dims.size()).c_str(),
            std::to_string(key_dims.size()).c_str(),
            std::to_string(value_dims.size()).c_str());

    VCHECK_INVALID_SHAPE((query_dims[3] == key_dims[2]),
            "%s, query head size should be match with key head size. query "
            "dims: %s, Key dims: %s",
            op_t::kind2str(n->get_kind()).c_str(), dims2str(query_dims).c_str(),
            dims2str(key_dims).c_str());

    VCHECK_INVALID_SHAPE((key_dims[3] == value_dims[2]),
            "%s, key sequence length should be match with value sequence "
            "length. key dims: %s, value dims: %s ",
            op_t::kind2str(n->get_kind()).c_str(), dims2str(key_dims).c_str(),
            dims2str(value_dims).c_str());

    dims inferred_output_shape;
    inferred_output_shape
            = {query_dims[0], query_dims[1], query_dims[2], value_dims[3]};

    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(inferred_output_shape, out0.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[0], inferred_output_shape);

    if (outputs.size() > 2) {
        auto out1 = logical_tensor_wrapper_t(outputs[2]);
        dims inferred_stats_shape
                = {query_dims[0], query_dims[1], query_dims[2], 1};

        if (out1.ndims() != -1) {
            VCHECK_INVALID_SHAPE(validate(inferred_stats_shape, out1.vdims()),
                    "%s, given stats shape is not compatible with inferred",
                    op_t::kind2str(n->get_kind()).c_str());
        }

        set_shape_and_strides(*outputs[2], inferred_stats_shape);
    }

    return status::success;
}

status_t infer_dnnl_host_scalar_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // host scalar output is always strided
    // with shape = {1}, strides = {1}
    outputs[0]->layout_type = layout_type::strided;
    set_shape_and_strides(*outputs[0], {1});
    return status::success;
}

status_t infer_dnnl_layernorm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const auto is_rms = n->has_attr(op_attr::is_rms)
            && n->get_attr<bool>(op_attr::is_rms);
    if (is_rms) {
        auto status = infer_identity_output_shape(n, inputs, outputs);
        if (status != status::success) return status;
    } else {
        auto status = infer_norm_output_shape(n, inputs, outputs);
        if (status != status::success) return status;
    }
    return status::success;
}

status_t infer_dnnl_softmax_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(in0.vdims(), out0.vdims()),
                "%s, input and output shapes are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }

    // We should compute output dense strides instead of directly copying input
    // strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    if (outputs.size() == 2) return status::success;

    // infer stats output shape
    auto out1 = logical_tensor_wrapper_t(outputs[2]);
    dims out1_dims = in0.vdims();
    int64_t axis = n->get_attr<int64_t>(op_attr::axis);
    VCHECK_INVALID_SHAPE((axis >= -in0.ndims() && axis < in0.ndims()),
            "%s, axis should be in range [-%d, %d), given axis: %" PRId64 " ",
            op_t::kind2str(n->get_kind()).c_str(), in0.ndims(), in0.ndims(),
            axis);
    if (axis < 0) { axis += in0.ndims(); }
    out1_dims[axis] = 1;

    if (out1.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(out1_dims, out1.vdims()),
                "%s, given stats shape is not compatible with inferred",
                op_t::kind2str(n->get_kind()).c_str());
    }

    set_shape_and_strides(*outputs[2], out1_dims);

    return status::success;
}

status_t infer_dnnl_sdpa_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // [batch_size, num_heads_q, seq_len_q, head_size_qk]
    auto query = ltw(inputs[0]);
    // [batch_size, num_heads_q, head_size_qk, seq_len_kv,]
    auto key = ltw(inputs[1]);
    // [batch_size, num_heads_v, seq_len_kv, head_size_v]
    auto value = ltw(inputs[2]);

    auto dquery = ltw(outputs[0]);
    auto dkey = ltw(outputs[1]);
    auto dvalue = ltw(outputs[2]);

    if (dquery.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(dquery.vdims(), query.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }
    set_shape_and_strides(*outputs[0], query.vdims());

    if (dkey.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(dkey.vdims(), key.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }
    set_shape_and_strides(*outputs[1], key.vdims());

    if (dvalue.ndims() != -1) {
        VCHECK_INVALID_SHAPE(validate(dvalue.vdims(), value.vdims()),
                "%s, inferred out shape and output shape are not compatible",
                op_t::kind2str(n->get_kind()).c_str());
    }
    set_shape_and_strides(*outputs[2], value.vdims());

    if (outputs.size() > 4) {
        // dmask exists
        auto dmask = ltw(outputs[4]);
        dims inferred_dmask_shape = query.vdims();
        size_t ndims = query.ndims();
        // [batch_size, num_heads_q, seq_len_q, seq_len_kv]
        inferred_dmask_shape[ndims - 1] = value.vdims()[ndims - 1];

        if (dmask.ndims() != -1) {
            VCHECK_INVALID_SHAPE(validate(inferred_dmask_shape, dmask.vdims()),
                    "%s, given dmask shape is not compatible with inferred",
                    op_t::kind2str(n->get_kind()).c_str());
        }

        set_shape_and_strides(*outputs[4], inferred_dmask_shape);
    }

    return status::success;
}

} // namespace graph
} // namespace impl
} // namespace dnnl
