/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <cctype>

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {

inline status_t memory_desc_sanity_check(int ndims, const dims_t dims,
        data_type_t data_type, format_kind_t format_kind) {
    using namespace data_type;

    if (ndims == 0) return status::success;

    VCHECK_MEMORY(dims != nullptr, status::invalid_arguments, "%s (dims)",
            VERBOSE_NULL_ARG);
    VCHECK_MEMORY(0 < ndims && ndims <= DNNL_MAX_NDIMS,
            status::invalid_arguments, VERBOSE_BAD_NDIMS, "tensor", ndims);
    VCHECK_MEMORY(utils::one_of(data_type, f4_e2m1, e8m0, f8_e5m2, f8_e4m3, f16,
                          bf16, f32, f64, s64, s32, s8, u8, s4, u4, u2, u3),
            status::invalid_arguments, VERBOSE_UNSUPPORTED_DT);

    // A bounds check on the dimensions ensures that the tensor size
    // computation does not trigger a overflow during memory creation.
    dim_t prod = 1;
    for (int d = 0; d < ndims; ++d) {
        if (!is_runtime_value(dims[d])) {
            if (dims[d] < 0)
                VCHECK_MEMORY(false, status::invalid_arguments, VERBOSE_BAD_DIM,
                        "dims", d);
            if (dims[d] > 0) {
                if (prod > std::numeric_limits<dim_t>::max() / dims[d])
                    VCHECK_MEMORY(false, status::invalid_arguments,
                            VERBOSE_INTEGRAL_OVERFLOW_DIM, "dims", d);
                prod *= dims[d];
            }
        }
    }

    bool has_runtime_dims = false;
    for (int d = 0; d < ndims; ++d) {
        if (is_runtime_value(dims[d])) has_runtime_dims = true;
    }

    VCHECK_MEMORY(
            IMPLICATION(has_runtime_dims, format_kind != format_kind::any),
            status::invalid_arguments, VERBOSE_UNSUPPORTED_FORMAT_KIND);

    return status::success;
}

status_t memory_desc_sanity_check(const memory_desc_t &md) {
    return memory_desc_sanity_check(
            md.ndims, md.dims, md.data_type, md.format_kind);
}

// Returns true if strides are compatible with memory_desc_t
status_t memory_desc_strides_check(
        const memory_desc_t &md, const dims_t strides) {
    if (strides == nullptr || md.ndims == 0
            || md.format_kind != format_kind::blocked)
        return status::success;

    dims_t blocks = {0};
    int perm[DNNL_MAX_NDIMS] = {0};
    for (int d = 0; d < md.ndims; ++d) {
        // no strides check needed for empty tensor
        if (md.padded_dims[d] == 0) return status::success;

        // no strides verification for runtime dims
        const bool has_runtime_dim
                = any_runtime_value(strides[d], md.padded_dims[d]);
        if (has_runtime_dim) return status::success;

        perm[d] = d;
        blocks[d] = 1;
    }

    dim_t block_size = 1;
    const auto &blk = md.format_desc.blocking;
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
        block_size *= blk.inner_blks[iblk];
    }

    // Returns `true` if @p idx is among blocked indices and @p pos_idx saves
    // its position when passed.
    auto is_inner_index = [&](const int idx, int *pos_idx = nullptr) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            if (blk.inner_idxs[iblk] == idx) {
                if (pos_idx) *pos_idx = idx;
                return true;
            }
        }
        return false;
    };

    // A custom comparator sorting perm dimension indices by strides.
    // Typically, strides are all different and sorting is smooth.
    // But there're situations, when they are same. Such situation involve:
    // * Dims of 1;
    // * Blocked dimension contains a single block, e.g., dim=16, block=32;
    //
    // In such case, prioritize blocked dimensions as more innermost over unit
    // dimensions.
    auto stride_sorter = [&](const int a, const int b) -> bool {
        if (strides[a] == strides[b]) {
            int pos_a = 0;
            int pos_b = 0;
            if (is_inner_index(a, &pos_a) + is_inner_index(b, &pos_b) == 2) {
                // When both dimensions are blocked, check how many blocks they
                // have. For same strides it means at least one of them has a
                // single block which should go first.
                // If strides are same and both blocks are unit, put in natural
                // blocking order.
                auto block_ratio_a = md.padded_dims[a] / blocks[a];
                auto block_ratio_b = md.padded_dims[b] / blocks[b];
                if (block_ratio_a == block_ratio_b)
                    return pos_a > pos_b;
                else
                    return block_ratio_a < block_ratio_b;
            } else if (is_inner_index(a) + is_inner_index(b) == 1) {
                // When it's just one, the one that blocked goes first.
                return is_inner_index(a);
            } else {
                // When none are blocked, the unit dim goes first, and if equal,
                // the natural order is used.
                if (md.padded_dims[a] == md.padded_dims[b])
                    return a < b;
                else
                    return md.padded_dims[a] < md.padded_dims[b];
            }
        } else {
            return strides[a] < strides[b];
        }
    };
    std::sort(perm, perm + md.ndims, stride_sorter);

    // tracks max stride for integral overflow checks
    dim_t max_stride = 1;
    int max_stride_d = 0;

    dim_t min_stride = block_size;

    for (int idx = 0; idx < md.ndims; ++idx) {
        const int d = perm[idx];

        // Make an exception for strides[d] == 0 as it has broadcast semantics
        // Note: owing to being sorted, these are the initial strides

        // FIXME: make an exception for dims[d] == 1 with the
        // assumption that no code applies that stride when the only
        // index accessed for that dimension is 0. This is because PT
        // can use "dummy" padding in those situations
        if ((strides[d] == 0) || (md.padded_dims[d] == 1))
            continue;
        else if (strides[d] < min_stride)
            VCONDCHECK(common, create, check, memory, false,
                    status::invalid_arguments, VERBOSE_INTEGRAL_OVERFLOW_DIM,
                    "strides", d);

        // Check for data type explicitly to avoid hitting potential assert.
        if (md.data_type != data_type::undef
                && types::data_type_bits(md.data_type) < 8) {
            const int dt_bits = types::data_type_bits(md.data_type);
            constexpr int bits_in_byte = 8;
            int elems_in_bytes = math::lcm(dt_bits, bits_in_byte) / dt_bits;
            VCONDCHECK(common, create, check, memory,
                    IMPLICATION(
                            strides[d] > 1, strides[d] % elems_in_bytes == 0),
                    status::invalid_arguments, VERBOSE_BAD_DIM, "strides", d);

            // The innermost block must hold whole storage units, otherwise a
            // unit straddles two blocks and cannot be owned by a single thread.
            if (blk.inner_nblks > 0
                    && d == blk.inner_idxs[blk.inner_nblks - 1]) {
                VCONDCHECK(common, create, check, memory,
                        blk.inner_blks[blk.inner_nblks - 1] % elems_in_bytes
                                == 0,
                        status::invalid_arguments, VERBOSE_BAD_DIM, "", d);
            }
        }

        // update min_stride for next iteration
        const auto padded_dim = md.padded_dims[d];

        const dim_t blocks_ratio = padded_dim / blocks[d];
        VCONDCHECK(common, create, check, memory,
                IMPLICATION(
                        strides[d] > 0 && block_size > 0 && blocks_ratio > 0,
                        strides[d] <= std::numeric_limits<dim_t>::max()
                                        / (block_size * blocks_ratio)),
                status::invalid_arguments, VERBOSE_INTEGRAL_OVERFLOW_DIM,
                "strides", d);

        // `strides` already account block_size in it, use block ratio to
        // increase the minimal stride requirement.
        min_stride = strides[d] * blocks_ratio;
        if (max_stride <= strides[d]) {
            max_stride = strides[d];
            max_stride_d = d;
        }
    }

    const size_t dt_size = types::data_type_size(md.data_type);

    // guard against integral overflow due to strides exceeding numeric limits
    if (!is_runtime_value(md.padded_dims[max_stride_d])) {
        size_t dim_val = static_cast<size_t>(
                md.padded_dims[max_stride_d] / blocks[max_stride_d]);
        dim_val = dim_val == (size_t)max_stride ? 1 : dim_val;
        VCONDCHECK(common, create, check, memory,
                (dim_val <= SIZE_MAX / max_stride), status::invalid_arguments,
                VERBOSE_INTEGRAL_OVERFLOW_DIM, "padded_dims", max_stride_d);
        VCONDCHECK(common, create, check, memory,
                (dt_size && ((dim_val * max_stride) <= SIZE_MAX / dt_size)),
                status::invalid_arguments, VERBOSE_INTEGRAL_OVERFLOW_DIM,
                "padded_dims", max_stride_d);
    }
    return status::success;
}

status_t memory_desc_init_host_scalar(
        memory_desc_t &memory_desc, data_type_t data_type) {
    memory_desc.ndims = 1;
    memory_desc.dims[0] = 1;
    memory_desc.data_type = data_type;
    memory_desc.format_kind = format_kind::host_scalar;

    CHECK(memory_desc_sanity_check(memory_desc.ndims, memory_desc.dims,
            memory_desc.data_type, memory_desc.format_kind));

    return success;
}

status_t memory_desc_init_by_tag(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (ndims == 0 || tag == format_tag::undef) {
        memory_desc = types::zero_md();
        return success;
    }

    format_kind_t format_kind = types::format_tag_to_kind(tag);

    /* memory_desc != 0 */
    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind));

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind;

    status_t status = success;
    if (tag == format_tag::any) {
        // nop
    } else if (format_kind == format_kind::blocked) {
        status = memory_desc_wrapper::compute_blocking(md, tag);
        CHECK(memory_desc_strides_check(md, md.format_desc.blocking.strides));
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success) memory_desc = md;

    return status;
}

status_t memory_desc_init_by_strides(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind::undef));

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::blocked;

    dims_t default_strides = {0};
    if (strides == nullptr) {
        bool has_runtime_strides = false;
        default_strides[md.ndims - 1] = 1;
        for (int d = md.ndims - 2; d >= 0; --d) {
            if (is_runtime_value(md.padded_dims[d])) has_runtime_strides = true;
            default_strides[d] = has_runtime_strides
                    ? runtime_value_for(default_strides[d])
                    : default_strides[d + 1] * md.padded_dims[d + 1];
        }
        strides = default_strides;
    }
    CHECK(memory_desc_strides_check(md, strides));

    array_copy(md.format_desc.blocking.strides, strides, md.ndims);

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_csr_encoding(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt, data_type_t pointers_dt) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    // This is the only number of dims that is supported at this point.
    VCHECK_MEMORY(ndims <= 2, unimplemented, VERBOSE_BAD_NDIMS, "", ndims);

    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind::undef));

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::csr;
    md.format_desc.sparse_desc.nnz = nnz;
    md.format_desc.sparse_desc.metadata_types[0] = indices_dt;
    md.format_desc.sparse_desc.metadata_types[1] = pointers_dt;

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_coo_encoding(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    // This is the only number of dims that is supported at this point.
    VCHECK_MEMORY(ndims <= 2, unimplemented, VERBOSE_BAD_NDIMS, "", ndims);

    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind::undef));

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::coo;
    md.format_desc.sparse_desc.nnz = nnz;
    md.format_desc.sparse_desc.metadata_types[0] = indices_dt;

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_packed_encoding(memory_desc_t &memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind::undef));

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::packed;
    md.format_desc.sparse_desc.nnz = nnz;

    memory_desc = md;

    return success;
}

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
// Creates a memory descriptor for a grouped encoding
//
// The grouped encoding represents a tensor where one dimension has variable
// size per group (e.g., different number of tokens per expert in MoE).
// Also could be perceived as a group of multiple tensors along
// the variable dimension.
//
// Parameters:
// - variable_dim_idx: Index of dimension with variable size
// - group_count: Number of groups
// - dims: Total dimensions
//
// Supported below:
// - 2D tensors only with variable dim 0 (e.g., [total_M, K]) or dim 1
//   (e.g., [M, total_K])
// - The variable dim is the outermost dim of the values
//   buffer: idx=0 -> row-major (ab), idx=1 -> col-major (ba)
// - Other dimensions must be uniform across all groups
status_t memory_desc_init_with_grouped_encoding(memory_desc_t &memory_desc,
        int ndims, const dims_t dims, data_type_t data_type,
        int variable_dim_idx, dim_t group_count, data_type_t offsets_dt) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    VCHECK_MEMORY(ndims == 2, unimplemented, VERBOSE_BAD_NDIMS, "", ndims);

    CHECK(memory_desc_sanity_check(ndims, dims, data_type, format_kind::undef));

    VCHECK_MEMORY(group_count > 0, invalid_arguments, VERBOSE_BAD_PARAM,
            "group_count");

    // Validate offsets data type
    VCHECK_MEMORY(offsets_dt == data_type::s32, invalid_arguments,
            VERBOSE_INVALID_DATATYPE, "offsets");

    // Validate variable dimension index
    VCHECK_MEMORY(variable_dim_idx >= 0 && variable_dim_idx < ndims,
            invalid_arguments, VERBOSE_BAD_PARAM, "variable_dim_idx");

    for (int d = 0; d < ndims; ++d) {
        VCHECK_MEMORY(!is_runtime_value(dims[d]), invalid_arguments,
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);
        VCHECK_MEMORY(
                dims[d] > 0, invalid_arguments, VERBOSE_BAD_DIM, "dims", d);
    }

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);

    // Grouped sparse encoding specifics
    // Uses format_kind::sparse because grouped layout is a form of multi-buffer
    // representation where values are stored together with metadata
    // describing the variable structure (similar to CSR with rowptr/colind)
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::grouped;
    md.format_desc.sparse_desc.nnz = utils::array_product(dims, ndims);
    md.format_desc.sparse_desc.metadata_types[0] = offsets_dt;
    auto &grouped = md.format_desc.sparse_desc.grouped_desc;
    grouped.group_count = group_count;
    grouped.variable_dim_idx = variable_dim_idx;
    // layout is determined by variable_dim_idx + dims:
    //   idx=0 row-major (ab): {dims[1], 1}
    //   idx=1 col-major (ba): {1, dims[0]}
    grouped.strides[variable_dim_idx] = dims[1 - variable_dim_idx];
    grouped.strides[1 - variable_dim_idx] = 1;

    memory_desc = md;

    return success;
}
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

status_t memory_desc_init_submemory(memory_desc_t &memory_desc,
        const memory_desc_t &parent_memory_desc, const dims_t dims,
        const dims_t offsets) {
    CHECK(memory_desc_sanity_check(parent_memory_desc));

    const memory_desc_wrapper src_d(parent_memory_desc);
    VCHECK_MEMORY((!src_d.has_runtime_dims_or_strides()), unimplemented,
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    for (int d = 0; d < src_d.ndims(); ++d) {
        VCHECK_MEMORY(!any_runtime_value(dims[d], offsets[d]), unimplemented,
                VERBOSE_RUNTIMEDIM_UNSUPPORTED);

        const bool dim_offsets_oob = (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims()[d]));
        VCHECK_MEMORY(
                !dim_offsets_oob, invalid_arguments, VERBOSE_BAD_DIM, "src", d);
    }

    VCHECK_MEMORY(src_d.format_kind() == format_kind::blocked, unimplemented,
            VERBOSE_UNSUPPORTED_TAG);

    dims_t blocks;
    src_d.compute_blocks(blocks);

    memory_desc_t dst_d = parent_memory_desc;
    auto &dst_d_blk = dst_d.format_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < src_d.ndims(); ++d) {
        const bool is_right_border = offsets[d] + dims[d] == src_d.dims()[d];

        /* very limited functionality for now */
        const bool ok = offsets[d] % blocks[d] == 0 /* [r1] */
                && src_d.padded_offsets()[d] == 0
                && IMPLICATION(!is_right_border,
                        (dims[d] % blocks[d] == 0 || dims[d] < blocks[d]));
        if (!ok) return unimplemented;

        dst_d.dims[d] = dims[d];
        dst_d.padded_dims[d] = is_right_border
                ? src_d.padded_dims()[d] - offsets[d]
                : dst_d.dims[d];
        dst_d.padded_offsets[d] = src_d.padded_offsets()[d];
        dst_d.offset0 += /* [r1] */
                offsets[d] / blocks[d] * dst_d_blk.strides[d];
    }

    memory_desc = dst_d;

    return success;
}

status_t memory_desc_reshape(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, int ndims, const dims_t dims) {
    auto volume = [](const dim_t *dims, int ndims) -> dim_t {
        dim_t prod = 1;
        for (int i = 0; i < ndims; ++i) {
            if (is_runtime_value(dims[i])) return runtime_value_for(prod);
            prod *= dims[i] > 0 ? dims[i] : 1;
        }
        return prod;
    };

    CHECK(memory_desc_sanity_check(in_memory_desc));
    CHECK(memory_desc_sanity_check(
            ndims, dims, in_memory_desc.data_type, in_memory_desc.format_kind));
    VCHECK_MEMORY(one_of(in_memory_desc.format_kind, format_kind::any,
                          format_kind::blocked),
            invalid_arguments, VERBOSE_UNSUPPORTED_TAG);
    VCHECK_MEMORY(!types::is_zero_md(&in_memory_desc), invalid_arguments,
            VERBOSE_NULL_ARG);
    VCHECK_MEMORY(volume(in_memory_desc.dims, in_memory_desc.ndims)
                    == volume(dims, ndims),
            invalid_arguments, VERBOSE_SHAPE_RESTRICTION);
    VCHECK_MEMORY(
            !memory_desc_wrapper(in_memory_desc).has_runtime_dims_or_strides(),
            invalid_arguments, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    VCHECK_MEMORY(in_memory_desc.extra.flags == 0, invalid_arguments,
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    if (in_memory_desc.format_kind == format_kind::any)
        return memory_desc_init_by_tag(out_memory_desc, ndims, dims,
                in_memory_desc.data_type, format_tag::any);

    assert(in_memory_desc.format_kind == format_kind::blocked);
    assert(in_memory_desc.extra.flags == 0);

    // temporary output
    auto md = in_memory_desc;

    md.ndims = ndims;
    array_copy(md.dims, dims, md.ndims);

    const int i_ndims = in_memory_desc.ndims;
    const int o_ndims = md.ndims;

    const auto &i_dims = in_memory_desc.dims,
               &i_pdims = in_memory_desc.padded_dims;
    const auto &o_dims = md.dims;

    const auto &i_bd = in_memory_desc.format_desc.blocking;
    auto &o_bd = md.format_desc.blocking;

    dims_t blocks = {0};
    memory_desc_wrapper(in_memory_desc).compute_blocks(blocks);

    enum class action_t { REMOVE_1, ADD_1, KEEP_DIM, REARRANGE_DIMS, FAIL };

    // Determine groups in input and output dims starting with the given
    // positions (going backwards) that satisfy one of the conditions:
    // - REMOVE_1
    //      input_group = {1}, output_group = empty
    // - ADD_1
    //      input_group = empty, output_group = {1}
    // - KEEP_DIM
    //      input_group = {x}, output_group = {x}
    // - REARRANGE_DIMS
    //      input_group = {x1, x2, .., xk}, output_group = {y1, y2, ..., ym}
    //      and product(x_i) = product(y_j), and the groups are minimal
    // - FAIL
    //      invalid configuration (return false)
    auto find_groups
            = [&](int &i_group_begin, int i_group_end, int &o_group_begin,
                      int o_group_end) -> action_t {
        // 1st step: check for `1` in the input dims
        if (i_group_end > 0 && i_dims[i_group_end - 1] == 1) {
            i_group_begin = i_group_end - 1;
            if (i_pdims[i_group_end - 1] == 1) {
                o_group_begin = o_group_end;
                return action_t::REMOVE_1;
            } else if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
                o_group_begin = o_group_end - 1;
                return action_t::KEEP_DIM;
            } else {
                return action_t::FAIL;
            }
        }

        // 2nd step: check for `1` in the output dims
        if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
            i_group_begin = i_group_end;
            o_group_begin = o_group_end - 1;
            return action_t::ADD_1;
        }

        // at this moment both groups cannot be empty
        if (i_group_end == 0 || o_group_end == 0) return action_t::FAIL;

        // 3rd step: find the non-trivial groups of the same volume
        i_group_begin = i_group_end - 1;
        o_group_begin = o_group_end - 1;

        dim_t i_volume = i_dims[i_group_begin];
        dim_t o_volume = o_dims[o_group_begin];

        while (i_volume != o_volume) {
            if (i_volume < o_volume) {
                if (i_group_begin == 0) return action_t::FAIL;
                i_volume *= i_dims[--i_group_begin];

                // do not allow `0` axis in the middle
                if (i_volume == 0) return action_t::FAIL;
            } else {
                if (o_group_begin == 0) return action_t::FAIL;
                o_volume *= o_dims[--o_group_begin];

                // do not allow `0` axis in the middle
                if (o_volume == 0) return action_t::FAIL;
            }
        }

        assert(i_volume == o_volume);
        assert(i_group_begin >= 0);
        assert(o_group_begin >= 0);

        return (i_group_begin + 1 == i_group_end
                       && o_group_begin + 1 == o_group_end)
                ? action_t::KEEP_DIM
                : action_t::REARRANGE_DIMS;
    };

    int i_group_begin = i_ndims, i_group_end = i_ndims;
    int o_group_begin = o_ndims, o_group_end = o_ndims;

    while (i_group_end != 0 || o_group_end != 0) {
        action_t action = find_groups(
                i_group_begin, i_group_end, o_group_begin, o_group_end);

        if (action == action_t::REMOVE_1) {
            // nop, padding is already taken into account by `find_groups()`
        } else if (action == action_t::ADD_1) {
            // get the stride from the right
            dim_t current_stride = 1;
            if (i_group_begin == i_ndims) {
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    current_stride *= i_bd.inner_blks[d];
            } else {
                // Add `1` to the left from axes with index `i_group_begin`
                current_stride
                        = i_bd.strides[i_group_begin] * i_dims[i_group_begin];
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    if (i_bd.inner_idxs[d] == i_group_begin)
                        current_stride /= i_bd.inner_blks[d];
            }
            md.padded_dims[o_group_begin] = 1;
            md.padded_offsets[o_group_begin] = 0;
            o_bd.strides[o_group_begin] = current_stride;
        } else if (action == action_t::KEEP_DIM) {
            // change the axis index from `i_group_begin` to `o_group_begin`
            assert(i_group_begin + 1 == i_group_end);
            assert(o_group_begin + 1 == o_group_end);

            md.padded_dims[o_group_begin]
                    = in_memory_desc.padded_dims[i_group_begin];
            md.padded_offsets[o_group_begin]
                    = in_memory_desc.padded_offsets[i_group_begin];
            o_bd.strides[o_group_begin] = i_bd.strides[i_group_begin];
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    o_bd.inner_idxs[d] = o_group_begin;
        } else if (action == action_t::REARRANGE_DIMS) {
            // check that input group is dense, sequential, and is not blocked
            for (int d = i_group_end - 1; d > i_group_begin; --d)
                if (i_dims[d] * i_bd.strides[d] != i_bd.strides[d - 1])
                    return invalid_arguments;

            // checked (i_group_begin, i_group_end), `i_group_begin` remains
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    return invalid_arguments;
            if (in_memory_desc.padded_dims[i_group_begin]
                    != i_dims[i_group_begin])
                return invalid_arguments;
            if (in_memory_desc.padded_offsets[i_group_begin] != 0)
                return invalid_arguments;

            // oK, fill output md according to
            // o_dims[o_group_begin .. o_group_end]

            dim_t current_stride = i_bd.strides[i_group_end - 1];
            for (int d = o_group_end - 1; d >= o_group_begin; --d) {
                md.padded_dims[d] = o_dims[d];
                md.padded_offsets[d] = 0;
                o_bd.strides[d] = current_stride;
                current_stride *= md.padded_dims[d];
            }
        } else {
            assert(action == action_t::FAIL);
            return invalid_arguments;
        }

        i_group_end = i_group_begin;
        o_group_end = o_group_begin;
    }

    out_memory_desc = md;
    return success;
}

status_t memory_desc_permute_axes(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, const int *perm) {
    CHECK(memory_desc_sanity_check(in_memory_desc));
    VCHECK_MEMORY(one_of(in_memory_desc.format_kind, format_kind::any,
                          format_kind::blocked),
            invalid_arguments, VERBOSE_UNSUPPORTED_TAG);
    VCHECK_MEMORY(!types::is_zero_md(&in_memory_desc), invalid_arguments,
            VERBOSE_NULL_ARG);
    VCHECK_MEMORY(
            !memory_desc_wrapper(in_memory_desc).has_runtime_dims_or_strides(),
            invalid_arguments, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    VCHECK_MEMORY(
            check_md_extra_flags_compensation_gpu(in_memory_desc.extra.flags),
            invalid_arguments, VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    // verify that perm is indeed a permutation of [0 .. ndims)
    unsigned occurrence_mask = 0;
    for (int d = 0; d < in_memory_desc.ndims; ++d)
        if (0 <= perm[d] && perm[d] < in_memory_desc.ndims)
            occurrence_mask |= (1u << perm[d]);
    VCHECK_MEMORY((occurrence_mask + 1 == (1u << in_memory_desc.ndims)),
            invalid_arguments, VERBOSE_BAD_NDIMS, "in_memory_desc",
            in_memory_desc.ndims);

    out_memory_desc = in_memory_desc;
    for (int d = 0; d < in_memory_desc.ndims; ++d) {
        if (perm[d] == d) continue;
        out_memory_desc.dims[perm[d]] = in_memory_desc.dims[d];
        out_memory_desc.padded_dims[perm[d]] = in_memory_desc.padded_dims[d];
        out_memory_desc.padded_offsets[perm[d]]
                = in_memory_desc.padded_offsets[d];
        if (in_memory_desc.format_kind == format_kind::blocked) {
            const auto &i_bd = in_memory_desc.format_desc.blocking;
            auto &o_bd = out_memory_desc.format_desc.blocking;

            o_bd.strides[perm[d]] = i_bd.strides[d];
            for (int blk = 0; blk < i_bd.inner_nblks; ++blk)
                if (i_bd.inner_idxs[blk] == d) o_bd.inner_idxs[blk] = perm[d];
        }
    }

    return success;
}

// This is only used by internal API that is used for testing only.
status_t memory_desc_init_by_string_tag(memory_desc_t &md, int ndims,
        const dims_t dims, data_type_t data_type, const std::string &tag) {
    // Copy to temporary to handle dims == md->dims case.
    dims_t tmp_dims;
    std::copy(dims, dims + ndims, tmp_dims);

    md.ndims = ndims;
    VCHECK_MEMORY(!(ndims < 0 || ndims > DNNL_MAX_NDIMS), invalid_arguments,
            VERBOSE_BAD_NDIMS, "", ndims);

    std::copy(tmp_dims, tmp_dims + ndims, md.dims);
    md.data_type = data_type;
    md.format_kind = format_kind::blocked;

    // Parse dimensions and their block sizes starting from the innermost one.
    std::vector<std::pair<int, int>> dim_blocks;
    int pos = (int)tag.size() - 1;
    int ndims_from_tag = -1;
    while (pos >= 0) {
        int pos0 = pos;

        --pos;
        while (pos >= 0 && std::isdigit(tag[pos]))
            pos--;

        int dim_idx = std::tolower(tag[pos0]) - 'a';
        VCHECK_MEMORY(dim_idx < ndims, invalid_arguments,
                "ndims deduced (%d) from the tag \'%s\' is inconsistent with "
                "provided ndims (%d)",
                dim_idx + 1, tag.c_str(), ndims);
        ndims_from_tag = std::max(dim_idx + 1, ndims_from_tag);
        int block_str_len = pos0 - pos - 1;
        bool is_blocked = block_str_len > 0;
        int block = is_blocked ? std::stoi(tag.substr(pos + 1, block_str_len))
                               : 1;
        if (is_blocked && block == 1) continue; // skip trivial blocks
        dim_blocks.emplace_back(dim_idx, block);
    }
    VCHECK_MEMORY((ndims_from_tag == ndims), invalid_arguments,
            "ndims deduced (%d) from the tag \'%s\' is inconsistent with "
            "provided ndims (%d)",
            ndims_from_tag, tag.c_str(), ndims);

    auto &blk = md.format_desc.blocking;

    // Compute strides and fill inner block sizes/indices.
    dim_t stride = 1;
    dims_t full_inner_blks;
    std::fill(full_inner_blks, full_inner_blks + ndims, 1);
    for (auto &p : dim_blocks) {
        int dim_idx = p.first;
        int block = p.second;
        if (block == 1) {
            assert(blk.strides[dim_idx] == 0);
            blk.strides[dim_idx] = stride;

            dim_t fib = full_inner_blks[dim_idx];
            const auto padded_dim = is_runtime_value(md.dims[dim_idx])
                    ? runtime_value_for(md.padded_dims[dim_idx])
                    : (md.dims[dim_idx] + fib - 1) / fib * fib;
            md.padded_dims[dim_idx] = padded_dim;
            if (any_runtime_value(padded_dim, stride))
                stride = runtime_value_for(stride);
            else
                stride *= (padded_dim / fib);
        } else {
            full_inner_blks[dim_idx] *= block;
            blk.inner_blks[blk.inner_nblks] = block;
            blk.inner_idxs[blk.inner_nblks] = dim_idx;
            blk.inner_nblks++;
            stride *= block;
        }
    }

    // Inner block sizes/indices are stored from the outermost to the innermost
    // so need to reverse them.
    std::reverse(blk.inner_blks, blk.inner_blks + blk.inner_nblks);
    std::reverse(blk.inner_idxs, blk.inner_idxs + blk.inner_nblks);

    return success;
}

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_memory_desc_create_with_tag(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_tag(*md, ndims, dims, data_type, tag));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_strides(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type,
        const dims_t strides) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_strides(*md, ndims, dims, data_type, strides));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_csr_encoding(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt, data_type_t pointers_dt) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_csr_encoding(
            *md, ndims, dims, data_type, nnz, indices_dt, pointers_dt));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_coo_encoding(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_coo_encoding(
            *md, ndims, dims, data_type, nnz, indices_dt));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_packed_encoding(
        memory_desc_t **memory_desc, int ndims, const dims_t dims,
        data_type_t data_type, dim_t nnz) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_packed_encoding(
            *md, ndims, dims, data_type, nnz));
    (*memory_desc) = md.release();
    return success;
}

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
status_t dnnl_memory_desc_create_with_grouped_encoding(
        memory_desc_t **memory_desc, int ndims, const dims_t dims,
        data_type_t data_type, int variable_dim_idx, dim_t group_count,
        data_type_t offsets_dt) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_with_grouped_encoding(*md, ndims, dims, data_type,
            variable_dim_idx, group_count, offsets_dt));
    (*memory_desc) = md.release();
    return success;
}
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

status_t dnnl_memory_desc_create_host_scalar(
        memory_desc_t **memory_desc, data_type_t data_type) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_host_scalar(*md, data_type));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_submemory(memory_desc_t **memory_desc,
        const memory_desc_t *parent_memory_desc, const dims_t dims,
        const dims_t offsets) {
    if (any_null(memory_desc, parent_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_submemory(*md, *parent_memory_desc, dims, offsets));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_reshape(memory_desc_t **out_memory_desc,
        const memory_desc_t *in_memory_desc, int ndims, const dims_t dims) {
    if (any_null(out_memory_desc, in_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_reshape(*md, *in_memory_desc, ndims, dims));
    (*out_memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_permute_axes(memory_desc_t **out_memory_desc,
        const memory_desc_t *in_memory_desc, const int *perm) {
    if (any_null(out_memory_desc, in_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_permute_axes(*md, *in_memory_desc, perm));
    (*out_memory_desc) = md.release();
    return success;
}

int dnnl_memory_desc_equal(const memory_desc_t *lhs, const memory_desc_t *rhs) {
    if (lhs == rhs) return 1;
    if (any_null(lhs, rhs)) return 0;
    return memory_desc_wrapper(*lhs) == memory_desc_wrapper(*rhs);
}

size_t dnnl_memory_desc_get_size(const memory_desc_t *md) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size();
}

size_t dnnl_memory_desc_get_size_v2(const memory_desc_t *md, int index) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size(index);
}

size_t dnnl_data_type_size(dnnl_data_type_t data_type) {
    return types::data_type_size(data_type);
}

status_t dnnl_memory_desc_query(
        const memory_desc_t *md, query_t what, void *result) {
    if (any_null(md, result)) return invalid_arguments;

    const bool is_blocked = md->format_kind == format_kind::blocked;

    switch (what) {
        case query::ndims_s32: *(int32_t *)result = md->ndims; break;
        case query::dims: *(const dims_t **)result = &md->dims; break;
        case query::data_type: *(data_type_t *)result = md->data_type; break;
        case query::submemory_offset_s64: *(dim_t *)result = md->offset0; break;
        case query::padded_dims:
            *(const dims_t **)result = &md->padded_dims;
            break;
        case query::padded_offsets:
            *(const dims_t **)result = &md->padded_offsets;
            break;
        case query::format_kind:
            switch ((int)md->format_kind) {
                case format_kind::rnn_packed:
                case format_kind::cublaslt_blocked:
                case format_kind::wino:
                case format_kind::zen_packed:
                    *(format_kind_t *)result = format_kind::opaque;
                    break;
                default: *(format_kind_t *)result = md->format_kind;
            }
            break;
        case query::strides:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.strides;
            break;
        case query::inner_nblks_s32:
            if (!is_blocked) return status::invalid_arguments;
            *(int32_t *)result = md->format_desc.blocking.inner_nblks;
            break;
        case query::inner_blks:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_blks;
            break;
        case query::inner_idxs:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_idxs;
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

status_t dnnl_memory_desc_query_v2(
        const memory_desc_t *md, query_t what, int index, void *result) {
    if (any_null(md, result)) return invalid_arguments;
    const bool is_sparse = md->format_kind == format_kind::sparse;
    VCHECK_MEMORY(
            !((!is_sparse && index > 0)
                    || (is_sparse && index > 0 && what != query::data_type)),
            invalid_arguments, VERBOSE_UNSUPPORTED_SPARSE_CFG);

    switch (what) {
        case query::sparse_encoding:
            if (!is_sparse) return status::invalid_arguments;
            *(sparse_encoding_t *)result = md->format_desc.sparse_desc.encoding;
            break;
        case query::nnz_s64:
            if (!is_sparse) return status::invalid_arguments;
            *(dim_t *)result = md->format_desc.sparse_desc.nnz;
            break;
        case query::data_type:
            if (index == 0) {
                *(data_type_t *)result = md->data_type;
            } else {
                if (!is_sparse) return status::invalid_arguments;
                VCHECK_MEMORY(index > 0, invalid_arguments,
                        VERBOSE_UNSUPPORTED_SPARSE_CFG);
                const auto &sd = md->format_desc.sparse_desc;
                const int meta_idx = sd.encoding == sparse_encoding_t::dnnl_coo
                        ? 0
                        : index - 1;
                VCHECK_MEMORY(meta_idx < sparse_desc_t::max_metadata_types,
                        invalid_arguments, VERBOSE_UNSUPPORTED_SPARSE_CFG);
                *(data_type_t *)result = sd.metadata_types[meta_idx];
            }
            break;
        case query::num_handles_s32:
            if (is_sparse) {
                switch (md->format_desc.sparse_desc.encoding) {
                    case sparse_encoding::csr:
                    case sparse_encoding::coo:
                        *(int *)result = md->ndims + 1;
                        break;
                    case sparse_encoding::packed: *(int *)result = 3; break;
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
                    case sparse_encoding::grouped: *(int *)result = 2; break;
#endif
                    default: assert(!"unknown encoding"); *(int *)result = 0;
                }
            } else
                *(int *)result = 1;
            break;
        default: return dnnl_memory_desc_query(md, what, result);
    }
    return status::success;
}

status_t dnnl_memory_desc_destroy(memory_desc_t *memory_desc) {
    delete memory_desc;
    return success;
}

status_t dnnl_memory_desc_clone(memory_desc_t **memory_desc,
        const memory_desc_t *existing_memory_desc) {
    (*memory_desc) = new memory_desc_t(*existing_memory_desc);
    return success;
}

status_t dnnl_memory_desc_get_blob(
        uint8_t *blob, size_t *size, const memory_desc_t *md) {
    if (md == nullptr || (blob == nullptr && size == nullptr))
        return invalid_arguments;
    if (blob != nullptr)
        memcpy(blob, md, *size);
    else if (size != nullptr)
        *size = sizeof(memory_desc_t);

    return success;
}

status_t dnnl_memory_desc_create_with_blob(
        memory_desc_t **md, const uint8_t *blob) {
    if (one_of(nullptr, md, blob)) return invalid_arguments;

    *md = new memory_desc_t();
    memcpy(*md, blob, sizeof(memory_desc_t));
    return success;
}

// This is an internal API that is used only for testing in benchdnn.
extern "C" status_t DNNL_API dnnl_memory_desc_create_with_string_tag(
        memory_desc_t **memory_desc, int ndims, const dims_t dims,
        data_type_t data_type, const char *tag) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_string_tag(*md, ndims, dims, data_type, tag));
    (*memory_desc) = md.release();
    return success;
}

extern "C" status_t DNNL_API dnnl_memory_desc_set_data_type(
        memory_desc_t *memory_desc, data_type_t data_type) {
    if (any_null(memory_desc)) return invalid_arguments;
    memory_desc->data_type = data_type;
    return success;
}
