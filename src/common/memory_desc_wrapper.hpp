/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef COMMON_MEMORY_DESC_WRAPPER_HPP
#define COMMON_MEMORY_DESC_WRAPPER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "type_helpers.hpp"

#include "verbose.hpp"

#define VCHECK_MEMORY(cond, stat, msg, ...) \
    VCONDCHECK(common, create, check, memory, (cond), stat, msg, ##__VA_ARGS__)

namespace dnnl {
namespace impl {

/** thin wrapper class over \struct memory_desc_t which allows easy
 * manipulations with underlying C structure, which is taken by reference */
// NOLINTNEXTLINE(readability-identifier-naming)
struct memory_desc_wrapper {
    const memory_desc_t *md_;

    /** constructor which takes a reference to a constant underlying C memory
     * descriptor \param md */
    memory_desc_wrapper(const memory_desc_t *md)
        : md_(md ? md : &glob_zero_md) {}
    memory_desc_wrapper(const memory_desc_t &md) : memory_desc_wrapper(&md) {}

    /* implementing attributes */
    int ndims() const { return md_->ndims; }
    const dims_t &dims() const { return md_->dims; }
    data_type_t data_type() const { return md_->data_type; }

    const dims_t &padded_dims() const { return md_->padded_dims; }
    const dims_t &padded_offsets() const { return md_->padded_offsets; }
    dim_t offset0() const { return md_->offset0; }

    format_kind_t format_kind() const { return md_->format_kind; }

    bool is_blocking_desc() const {
        return format_kind() == format_kind::blocked;
    }

    bool is_sparse_packed_desc() const {
        return is_sparse_desc()
                && sparse_desc().encoding == sparse_encoding::packed;
    }

    bool is_wino_desc() const { return format_kind() == format_kind::wino; }
    bool is_rnn_packed_desc() const {
        return format_kind() == format_kind::rnn_packed;
    }
    bool is_cublaslt_blocked_desc() const {
        return format_kind() == format_kind::cublaslt_blocked;
    }
    bool is_zen_packed_desc() const {
        return format_kind() == format_kind::zen_packed;
    }
    bool is_sparse_desc() const { return format_kind() == format_kind::sparse; }
    bool is_grouped_desc() const {
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
        return is_sparse_desc()
                && sparse_desc().encoding == sparse_encoding::grouped;
#else
        return false;
#endif
    }

    bool is_host_scalar_desc() const {
        return format_kind() == format_kind::host_scalar;
    }

    /**
     * Checks whether the memory descriptor represents a canonical (row-major)
     * layout.
     *
     * A canonical layout is defined as one where the strides follow the
     * standard row-major ordering of the remaining dimensions, ignoring any
     * dimensions of size 1.
     *
     * @note Format tags (e.g., "abc", "acb") may correspond to different
     *   stride arrays but the same physical layout when unit dimensions are
     *   present. Therefore, checking for canonical layout is more reliable than
     *   matching a specific tag.
     *
     * Example:
     *   dims    = [10, 12, 1]
     *   strides = [12, 1, 1]
     *   tag     = "abc"
     *   -> remove unit dims: dims = [10,12], strides = [12,1]
     *   -> canonical (true)
     *
     *   dims    = [10, 1, 12]
     *   strides = [12, 12, 1]
     *   tag     = "acb"
     *   -> remove unit dims: dims = [10,12], strides = [12,1]
     *   -> canonical (true)
     *
     *   dims    = [10, 12, 3]
     *   strides = [36, 1, 3]
     *   tag     = "acb"
     *   -> canonical (false)
     *
     * @return true if the descriptor has canonical layout, false otherwise.
     */
    bool is_canonical() const {
        if (!is_plain()) return false;

        dims_t normalized_dims = {};
        dims_t normalized_strides = {};

        int normalized_ndims = 0;
        for (int i = 0; i < ndims(); ++i) {
            if (dims()[i] != 1) {
                normalized_dims[normalized_ndims] = dims()[i];
                normalized_strides[normalized_ndims] = strides()[i];
                normalized_ndims++;
            }
        }

        if (normalized_ndims == 0) return true;

        dim_t expected_stride = 1;
        for (int i = normalized_ndims - 1; i >= 0; --i) {
            if (normalized_strides[i] != expected_stride) return false;
            expected_stride *= normalized_dims[i];
        }
        return true;
    }

    const blocking_desc_t &blocking_desc() const {
        assert(is_blocking_desc() || is_sparse_packed_desc());
        if (!is_sparse_desc()) return md_->format_desc.blocking;
        return sparse_desc().packed_desc;
    }
    const wino_desc_t &wino_desc() const {
        assert(is_wino_desc());
        return md_->format_desc.wino_desc;
    }
    const rnn_packed_desc_t &rnn_packed_desc() const {
        assert(is_rnn_packed_desc());
        return md_->format_desc.rnn_packed_desc;
    }
    const cublaslt_blocked_desc_t &cublaslt_blocked_desc() const {
        assert(is_cublaslt_blocked_desc());
        return md_->format_desc.cublaslt_blocked_desc;
    }

    const zen_packed_desc_t &zen_packed_desc() const {
        assert(is_zen_packed_desc());
        return md_->format_desc.zen_packed_desc;
    }

    const sparse_desc_t &sparse_desc() const {
        assert(is_sparse_desc());
        return md_->format_desc.sparse_desc;
    }

    data_type_t metadata_type(int idx = 0) const {
        assert(is_sparse_desc() && idx < sparse_desc_t::max_metadata_types);
        return sparse_desc().metadata_types[idx];
    }

    sparse_encoding_t encoding() const {
        assert(is_sparse_desc());
        return sparse_desc().encoding;
    }

    dim_t nnz() const {
        assert(is_sparse_desc());
        return sparse_desc().nnz;
    }

    const dims_t &strides() const {
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
        if (is_grouped_desc()) return sparse_desc().grouped_desc.strides;
#endif
        return blocking_desc().strides;
    }

    const memory_extra_desc_t &extra() const { return md_->extra; }

    /* some useful function */

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise. Counting starts from
     * \param start_dim, so a non-zero value skips the outer dimensions and
     * returns the product over dimensions `[start_dim, ndims())`. */
    dim_t nelems(bool with_padding = false, int start_dim = 0) const {
        if (is_zero()) return 0;
        if (has_runtime_dims()) return runtime_value_for<dim_t>();
        return utils::array_product(
                (with_padding ? padded_dims() : dims()) + start_dim,
                nstl::max(0, ndims() - start_dim));
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** returns true if memory descriptor contains zero as one of its dim */
    bool has_zero_dim() const {
        // to be consistent for when memory descriptor is zero with ndims = 0
        if (is_zero()) return true;

        for (int d = 0; d < ndims(); ++d)
            if (dims()[d] == 0) return true;
        return false;
    }

    /** return the size of data type (a shortcut) */
    size_t data_type_size() const { return types::data_type_size(data_type()); }

    /** return the size of data type of additional buffer */
    size_t additional_buffer_data_size(uint64_t flag_select) const {
        using namespace memory_extra_flags;
        if (flag_select & compensation_conv_s8s8) return sizeof(int32_t);
        if (flag_select & rnn_u8s8_compensation) return sizeof(float);
        if (flag_select & compensation_conv_asymmetric_src)
            return sizeof(int32_t);
        if (flag_select & compensation_gpu_conv_asymmetric_src)
            return sizeof(int32_t);
        return 0;
    }

    /** return true if memory format has additional buffer */
    bool is_additional_buffer() const {
        using namespace memory_extra_flags;
        return extra().flags
                & (compensation_conv_s8s8 | rnn_u8s8_compensation
                        | compensation_gpu_conv_asymmetric_src
                        | compensation_conv_asymmetric_src);
    }

    /** returns the size required for a particular extra memory buffer */
    size_t additional_buffer_size(memory_extra_flags_t flag) const {
        using namespace memory_extra_flags;
        const auto flags = extra().flags;
        if (!(flags & flag)) return 0;

        const auto ndims = this->ndims();
        const auto &pdims = padded_dims();

        auto calculate_size
                = [ndims, &pdims](int cmask, size_t buff_data_size) {
            assert(utils::one_of(cmask, 1, 2, 3, 5, 13, 27));
            dim_t prod = 1;
            for (int d = 0; d < ndims; ++d)
                if (cmask & (1 << d)) { prod *= pdims[d]; }
            return (size_t)prod * buff_data_size;
        };

        if (flag == compensation_conv_s8s8) {
            return calculate_size(extra().compensation_mask,
                    additional_buffer_data_size(flag));
        }
        if (flag == rnn_u8s8_compensation) {
            return calculate_size(extra().compensation_mask,
                    additional_buffer_data_size(flag));
        }
        if (flag == compensation_conv_asymmetric_src) {
            return calculate_size(extra().asymm_compensation_mask,
                    additional_buffer_data_size(flag));
        }
        if (flag == compensation_gpu_conv_asymmetric_src) {
            return extra().dst_size;
        }

        return 0;
    }

    dim_t blk_size() const {
        assert(is_blocking_desc() || is_sparse_packed_desc());
        const auto &bd = blocking_desc();
        return utils::array_product(bd.inner_blks, bd.inner_nblks);
    }

    /** returns the size of the appended buffer when the memory descriptor
     * requires extra space to hold compensation data */
    size_t additional_buffer_size() const {
        using namespace memory_extra_flags;

        size_t buff_size = 0;
        buff_size += additional_buffer_size(compensation_conv_s8s8);
        buff_size += additional_buffer_size(rnn_u8s8_compensation);
        buff_size += additional_buffer_size(compensation_conv_asymmetric_src);
        buff_size
                += additional_buffer_size(compensation_gpu_conv_asymmetric_src);
        return buff_size;
    }

    /** returns the number of elements spanned by the descriptor starting from
     * `start_dim`, i.e. one past the largest linear offset `off_v()` may return
     * when only dimensions `[start_dim, ndims())` are traversed. With the
     * default `start_dim == 0` it accounts for every dimension and exceeds
     * `nelems(true)` for strided descriptors. A non-zero `start_dim` ignores
     * the span contributed by the outer dimensions, which is useful to check
     * whether everything past those dimensions is dense. */
    size_t span(int start_dim = 0) const {
        if (utils::one_of(format_kind(), format_kind::undef, format_kind::any)
                || is_zero() || has_zero_dim())
            return 0;

        if (has_runtime_dims_or_strides()) return runtime_value_for<size_t>();

        if (is_blocking_desc()) {
            dims_t blocks = {0};
            compute_blocks(blocks);

            const auto &bd = blocking_desc();

            dim_t max_size = 0;
            for (int d = start_dim; d < ndims(); ++d) {
                dim_t strided_pdim = padded_dims()[d] / blocks[d];
                dim_t effective_stride = strided_pdim == 1 ? 1 : bd.strides[d];
                max_size = nstl::max(max_size, strided_pdim * effective_stride);
            }

            if (max_size == 1 && bd.inner_nblks != 0) max_size = blk_size();
            return static_cast<size_t>(max_size);
        } else {
            assert(!"unsupported format kind");
            return 0;
        }
    }

    /** returns the size required to store described memory note: does not
        include offset0 by default */
    size_t size(int index = 0, bool include_additional_size = true,
            bool include_offset0 = false) const {
        if (utils::one_of(format_kind(), format_kind::undef, format_kind::any)
                || is_zero() || has_zero_dim())
            return 0;

        if (utils::one_of(format_kind(), format_kind::blocked,
                    format_kind::wino, format_kind::rnn_packed,
                    format_kind::cublaslt_blocked, format_kind::zen_packed)
                && index != 0) {
            return 0;
        }

        if (has_runtime_dims_or_strides()) return runtime_value_for<size_t>();

        if (is_wino_desc()) {
            return wino_desc().size;
        } else if (is_rnn_packed_desc()) {
            return rnn_packed_desc().size;
        } else if (is_cublaslt_blocked_desc()) {
            return cublaslt_blocked_desc().size;
        } else if (is_zen_packed_desc()) {
            return zen_packed_desc().size;
        } else if (is_blocking_desc()) {
            const size_t max_size = span();

            size_t data_size = types::elements_to_bytes(data_type(), max_size);
            if (is_additional_buffer()) {
                // The additional buffers, typically of data type int32_t, float
                // are stored at the end of data. Pad the data, so that the
                // buffers are properly aligned to their data type.
                const size_t alignment_in_bytes = 4;
                data_size = utils::rnd_up(data_size, alignment_in_bytes);
            }
            return data_size
                    + (include_additional_size ? additional_buffer_size() : 0)
                    + (include_offset0 ? data_type_size() * offset0() : 0);
        } else if (is_sparse_desc()) {
            if (sparse_desc().encoding == sparse_encoding::csr) {
                switch (index) {
                    // Return size for values.
                    case 0: return nnz() * data_type_size();
                    // Return size for indices.
                    case 1: {
                        const auto idx_dt = metadata_type(0);
                        return nnz() * types::data_type_size(idx_dt);
                    }
                    // Return size for pointers.
                    case 2: {
                        const auto ptr_dt = metadata_type(1);
                        return (dims()[0] + 1) * types::data_type_size(ptr_dt);
                    }
                    default: assert(!"unknown index"); return 0;
                }
            } else if (sparse_desc().encoding == sparse_encoding::coo) {
                // Return size for values.
                if (index == 0) {
                    return nnz() * data_type_size();
                } else if (index > 0 && index <= ndims()) {
                    // Return size for index buffers.
                    const auto idx_dt = metadata_type(0);
                    return nnz() * types::data_type_size(idx_dt);
                } else {
                    assert(!"unknown index");
                    return 0;
                }
            } else if (sparse_desc().encoding == sparse_encoding::packed) {
                // If the size if queried from a user-created memory descriptor.
                if (blocking_desc().strides[0] == 0) return 0;

                switch (index) {
                    case 0:
                        // Return size for values.
                        return nnz() * data_type_size();
                    case 1: {
                        // Return size for offsets.
                        return (nelems(true) / blk_size()) * sizeof(int64_t);
                    }
                    case 2:
                        // Return size for bitmask. The bitmask has 1 bit
                        // per each value.
                        return utils::div_up(nelems(true), CHAR_BIT);
                    default: assert(!"unknown index"); return 0;
                }
            }
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
            else if (sparse_desc().encoding == sparse_encoding::grouped) {
                // Grouped encoding has values buffer and offsets buffer
                switch (index) {
                    case 0:
                        // Return size for values.
                        return types::elements_to_bytes(data_type(), nnz());
                    case 1: {
                        // Return size for offsets (group_count offsets).
                        const auto offsets_dt = metadata_type(0);
                        return (sparse_desc().grouped_desc.group_count)
                                * types::data_type_size(offsets_dt);
                    }
                    default: assert(!"unknown index"); return 0;
                }
            }
#endif
            else {
                assert(!"unknown sparse encoding");
                return 0;
            }
        } else if (is_host_scalar_desc()) {
            return data_type_size();
        } else {
            assert(!"unknown format kind");
            return 0;
        }
    }

    /** returns the true if some dim is broadcasted (stride == 0) */
    bool has_broadcast() const {
        const auto &bd = blocking_desc();
        for (int d = 0; d < ndims(); d++)
            if (bd.strides[d] == 0) return true;
        return false;
    }

    /** returns true if number of non unit dims is <= `n`. */
    bool count_non_unit_dims(int n) const {
        int non_unit_dims = 0;
        for (int d = 0; d < ndims(); d++) {
            if (dims()[d] != 1) non_unit_dims++;
        }
        return non_unit_dims <= n;
    }

    /** returns the index of the innermost (most densely packed) logical
     * dimension, i.e. the one that varies fastest in memory. When the format
     * has inner blocks, it is the dimension of the last block; otherwise it is
     * the dimension with the smallest stride, excluding unit and broadcast dims
     * whose strides would otherwise tie with and mask the real packed axis
     * (e.g. dims [1, N] with a `ba` layout has strides [1, 1], yet dim 1 is the
     * packed axis). */
    int innermost_dim() const {
        const auto &bd = blocking_desc();
        if (bd.inner_nblks > 0)
            return static_cast<int>(bd.inner_idxs[bd.inner_nblks - 1]);

        int inner_dim = ndims() - 1;
        dim_t min_stride = 0;
        for (int d = 0; d < ndims(); d++) {
            if (dims()[d] == 1 || bd.strides[d] == 0) continue;
            if (min_stride == 0 || bd.strides[d] < min_stride) {
                min_stride = bd.strides[d];
                inner_dim = d;
            }
        }
        return inner_dim;
    }

    /** returns true if data is dense in memory */
    bool is_dense(bool with_padding = false) const {
        if (utils::one_of(format_kind(), format_kind::undef, format_kind::any))
            return false;
        if (has_runtime_dims_or_strides() || has_broadcast()) return false;
        return types::elements_to_bytes(data_type(), nelems(with_padding))
                == size(0, /* include_additional_size = */ false);
    }

    /** returns true if format is set to `any` */
    bool format_any() const { return format_kind() == format_kind::any; }

    /** returns true if at least one dim is not known */
    bool has_runtime_dims() const {
        for (int d = 0; d < ndims(); ++d)
            if (is_runtime_value(dims()[d])) return true;
        return false;
    }

    /** returns true if at least one dim is not known */
    bool has_runtime_strides() const {
        if (!is_blocking_desc()) return false;
        for (int d = 0; d < ndims(); ++d)
            if (is_runtime_value(blocking_desc().strides[d])) return true;
        return false;
    }

    /** returns true if memory format is runtime_dims_or_strides-dependent */
    bool has_runtime_dims_or_strides() const {
        return has_runtime_dims() || has_runtime_strides();
    }

    /** returns true if the only (potentially) padded dim is \param dim */
    bool only_padded_dim(int dim) const {
        if (has_runtime_dims()) return false;
        for (int d = 0; d < ndims(); ++d)
            if (d != dim && dims()[d] != padded_dims()[d]) return false;
        return true;
    }

    /** returns true if memory desc has blocked layout and block dims are 1s */
    bool is_plain() const {
        if (!is_blocking_desc()) return false;
        return blocking_desc().inner_nblks == 0;
    }

    /** returns overall block sizes */
    void compute_blocks(dims_t blocks) const {
        if (!is_blocking_desc()) {
            utils::array_set(blocks, 0, ndims());
            return;
        }

        utils::array_set(blocks, 1, ndims());

        const auto &bd = blocking_desc();
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            blocks[bd.inner_idxs[iblk]] *= bd.inner_blks[iblk];
    }

    // XXX: for backward compatibility with v0.x
    // strides_compat[0]: stride between the first elements of adjacent blocks
    // strides_compat[1]: strides between elements in the same block
    //
    // For 2+ level blocking all inner blocks are treated as a single block.
    void compute_strides_compat(dims_t *strides_compat) const;

    /* comparison section */

    bool operator==(const memory_desc_wrapper &rhs) const {
        return *this->md_ == *rhs.md_;
    }
    bool operator!=(const memory_desc_wrapper &rhs) const {
        return !operator==(rhs);
    }
    bool operator==(const memory_desc_t &rhs) const {
        return operator==(memory_desc_wrapper(rhs));
    }
    bool operator!=(const memory_desc_t &rhs) const { return !operator==(rhs); }

    /** returns true if data (w/o padding if with_padding == false and w/
     * padding otherwise) have the same physical structure, i.e. dimensions,
     * strides, and blocked structure. Depending on with_data_type flag
     * data_type is taken or not taken into account. dim_start allows to check
     * similarity for the logical part of data [dim_start .. ndims()].
     * CAUTION: format kind any and undef are not similar to whatever, hence the
     * following statement might be true: lhs == rhs && !lhs.similar_to(rhs) */
    /* TODO: revise */
    bool similar_to(const memory_desc_wrapper &rhs, bool with_padding = true,
            bool with_data_type = true, int dim_start = 0) const;

    /** returns true if one memory can be reordered to another */
    bool consistent_with(const memory_desc_wrapper &rhs) const;

    /** returns true if the memory desc corresponds to the given format tag.
     * @sa memory_desc_matches_tag */
    bool matches_tag(format_tag_t tag, const dims_t strides = nullptr) const {
        return memory_desc_matches_tag(*md_, tag, strides);
    }

    /** returns matching tag (or undef if match is not found)
     * XXX: This is a workaround that eventually should go away! */
    template <typename... Tags>
    format_tag_t matches_one_of_tag(Tags... tags) const {
        for (const auto tag : {tags...}) {
            if (memory_desc_matches_tag(*md_, tag)) return tag;
        }
        return format_tag::undef;
    }

    template <typename... Tags>
    format_tag_t mb_stride_relaxed_match(Tags... tags) const {
        dims_t skip_mb_stride {};
        // See `memory_desc_matches_tag` comment.
        skip_mb_stride[0] = -1;
        for (const auto &tag : {tags...})
            if (matches_tag(tag, skip_mb_stride)) return tag;
        return format_tag::undef;
    }

    /* offset section */

    /** returns physical offset by logical one. logical offset is represented by
     * an array \param pos. if \param is_pos_padded is true \param pos
     * represents the position in already padded area */
    dim_t off_v(const dims_t pos, bool is_pos_padded = false) const {
        if (is_host_scalar_desc()) return 0;
        // Grouped values have no blocking, and an index along the variable
        // dim spans all groups
        if (is_grouped_desc()) {
            dim_t phys_offset = 0;
            for (int d = 0; d < ndims(); ++d)
                phys_offset += pos[d] * strides()[d];
            return phys_offset;
        }
        assert(is_blocking_desc() || is_sparse_packed_desc());
        const blocking_desc_t &blk = blocking_desc();

        dims_t pos_copy = {0};
        for (int d = 0; d < ndims(); ++d)
            pos_copy[d] = pos[d] + (is_pos_padded ? 0 : padded_offsets()[d]);

        dim_t phys_offset = offset0();

        if (blk.inner_nblks > 0) {
            dim_t blk_stride = 1;
            for (int iblk = blk.inner_nblks - 1; iblk >= 0; --iblk) {
                const dim_t d = blk.inner_idxs[iblk];

                dim_t p;
                /* switch to faster 32-bit division when possible.
                 * inner blocks always fit 32-bit. */
                if (pos_copy[d] <= INT32_MAX) {
                    p = (int32_t)pos_copy[d] % (int32_t)blk.inner_blks[iblk];
                    pos_copy[d] = (int32_t)pos_copy[d]
                            / (int32_t)blk.inner_blks[iblk];
                } else {
                    p = pos_copy[d] % blk.inner_blks[iblk];
                    pos_copy[d] /= blk.inner_blks[iblk];
                }

                phys_offset += p * blk_stride;

                blk_stride *= blk.inner_blks[iblk];
            }
        }

        for (int d = 0; d < ndims(); ++d) {
            const dim_t p = pos_copy[d];
            phys_offset += p * blk.strides[d];
        }

        return phys_offset;
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a scalar \param l_offset. if \param is_pos_padded is true, \param
     * l_offset represents logical offset in already padded area */
    dim_t off_l(dim_t l_offset, bool is_pos_padded = false) const {
        if (l_offset == 0) return offset0();
        dims_t dims_pos;
        const auto &cur_dims = is_pos_padded ? padded_dims() : dims();
        utils::l_dims_by_l_offset(dims_pos, l_offset, cur_dims, ndims());
        return off_v(dims_pos, is_pos_padded);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indices (\param xn, ..., \param x1, \param x0) */
    template <typename... Args>
    dim_t off(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = {args...};
        return off_v(pos, false);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indices (\param xn, ..., \param x1, \param x0) in already
     * padded area */
    template <typename... Args>
    dim_t off_padding(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = {args...};
        return off_v(pos, true);
    }

    /** returns physical offset by logical one. Logical offset is represented by
     * a tuple of block indices (\param bn, ..., \param b1, \param b0). It is a
     * user responsibility to adjust the result to get offset within blocks */
    template <typename... Args>
    dim_t blk_off(Args... args) const {
        return _blk_off<sizeof...(args), Args...>(args...);
    }

    template <bool skip_first, typename T, typename... Args>
    dim_t blk_off(T xn, Args... args) const {
        return skip_first ? blk_off<Args...>(args...)
                          : blk_off<T, Args...>(xn, args...);
    }

    /** returns physical offset by logical one. Logical offset is represented by
     * a tuple of block indices (\param bn, ..., \param b1, \param b0). It is a
     * user responsibility to adjust the result to get offset within blocks.
     * If @tparam sub_off0 is true, then offset0() will be subtracted
     * from result.*/
    template <bool skip_first, bool sub_off0, typename T, typename... Args>
    dim_t blk_off(T xn, Args... args) const {
        return blk_off<skip_first, Args...>(xn, args...) - sub_off0 * offset0();
    }

    /* static functions section */
    /* TODO: replace with non-static, once md_ becomes non-const ref */

    static status_t compute_blocking(
            memory_desc_t &memory_desc, format_tag_t tag);

private:
    /* TODO: put logical_offset in utils */
    template <typename T>
    dim_t logical_offset(T x0) const {
        return x0;
    }

    template <typename T, typename... Args>
    dim_t logical_offset(T xn, Args... args) const {
        const size_t n_args = sizeof...(args);
        return xn * utils::array_product<n_args>(&dims()[ndims() - n_args])
                + logical_offset(args...);
    }

    template <int ORIG_LEN, typename... Void>
    dim_t _blk_off() const {
        return offset0();
    }

    template <int ORIG_LEN, typename T, typename... Args>
    dim_t _blk_off(T xc, Args... args) const {
        assert(is_blocking_desc() || is_sparse_packed_desc());
        constexpr int dc = ORIG_LEN - sizeof...(args) - 1;
        return xc * blocking_desc().strides[dc]
                + _blk_off<ORIG_LEN, Args...>(args...);
    }
};

inline bool memory_desc_wrapper::similar_to(const memory_desc_wrapper &rhs,
        bool with_padding, bool with_data_type, int dim_start) const {
    using namespace utils;

    if (one_of(format_kind(), format_kind::undef, format_kind::any))
        return false;
    if (is_wino_desc() || is_rnn_packed_desc() || is_cublaslt_blocked_desc()
            || is_zen_packed_desc())
        return false;

    const int ds = dim_start;
    const auto &blk = blocking_desc();
    const auto &r_blk = rhs.blocking_desc();

    return ndims() == rhs.ndims() && dim_start <= ndims() /* guard */
            && format_kind() == rhs.format_kind()
            && IMPLICATION(with_data_type, data_type() == rhs.data_type())
            && array_cmp(dims() + ds, rhs.dims() + ds, ndims() - ds)
            && array_cmp(blk.strides + ds, r_blk.strides + ds, ndims() - ds)
            && blk.inner_nblks == r_blk.inner_nblks
            && array_cmp(blk.inner_blks, r_blk.inner_blks, blk.inner_nblks)
            && array_cmp(blk.inner_idxs, r_blk.inner_idxs, blk.inner_nblks)
            && IMPLICATION(with_padding,
                    true
                            && array_cmp(padded_dims() + ds,
                                    rhs.padded_dims() + ds, ndims() - ds)
                            && array_cmp(padded_offsets() + ds,
                                    rhs.padded_offsets() + ds, ndims() - ds));
}

inline bool memory_desc_wrapper::consistent_with(
        const memory_desc_wrapper &rhs) const {
    if (ndims() == rhs.ndims()) {
        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] != rhs.dims()[d]) return false;
        }
        return true;
    } else {
        /* TODO: revise.
         * is the following possible?
         * [1, a, b] <--reorder--> [a, b]
         * [a, 1, b] <--reorder--> [a, b]
         * not, at least for now */
        return false;
    }
}

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
