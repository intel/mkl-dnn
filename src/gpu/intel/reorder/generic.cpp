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

/*
 Fast, generic reorder kernel.

 Reorder kernel is strictly memory-bound. Performance is determined only by
 memory access efficiency.

 There's no perf difference between using single word reads and block read
 functions (intel_sub_group_read8 etc.). Only thing that matters is that when
 data is read/written from/to global memory, code must utilize whole cache
 lines. Using smaller chunks causes cache eviction and requires the same data
 to be accessed later again, wasting bandwidth.

 This kernel tries to load/store data in packets that are at least as large as
  cache line.

  Example: abc -> bca, 32x32x32, data type f32
  Assume SIMD16 and cache line size = 64B
  To fill cache line kernel must load 16 consecutive items (16c) and
  store 16 consecutive items (16a). So it needs to operate on a matrix of
  16a x 16c.
  It will load 16 non-adjacent (strided by A) sets of 16 adjacent data
  (strided by C, src' innermost dimension), perform internal transposition,
   then store 16 non-adjacent (strided by C) sets of 16 adjacent data (strided
   by A, dst's innermost dimension).

Difficulty is in determining how to achieve the above goal for
  any combination of tensor size and format tags.
*/

#include <algorithm>
#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/reorder/generic.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

using namespace dnnl::impl::memory_tracking::names;

struct dimension_t {
    dim_t size;
    dim_t step;
    dim_idx_t idx;
};

using dimensions_t = std::vector<dimension_t>;

// Return a description of dimensions sorted by stride, i.e., nesting order.
dimensions_t dims_by_stride(const memory_desc_wrapper &mdw) {
    const auto &desc = mdw.blocking_desc();
    const auto &strides = desc.strides;

    // Sort blocks by stride.
    const auto cmp = [&](const dimension_t &a, const dimension_t &b) {
        // Order by stride. Ties mean that we have at least one dim of size 1.
        // We don't care about the order of those dims, just that that the dim
        // with size > 1 is sorted last.
        const auto a_stride = strides[a.idx];
        const auto b_stride = strides[b.idx];
        return a_stride < b_stride || (a_stride == b_stride && a.size < b.size);
    };

    const int ndims = mdw.ndims();
    dimensions_t dims(ndims);
    for (int d = 0; d < ndims; ++d) {
        auto &blk = dims[d];
        blk.idx = d;
        blk.size = mdw.padded_dims()[d];
    }
    std::sort(dims.begin(), dims.end(), cmp);
    return dims;
}

// Returns description of blocks and dimensions that constitute the format tag
// of tensor, starting from innermost. Blocks, if exist, take precedence before
// dimensions. Order of dimensions is determined by sorting strides; smallest
// stride is innermost dimension. Dimensions of size 1 are ignored. This may
// lead to illegal tensor tags, where the innermost dim is the same as the
// outermost block:
//     8x8x1x1 aBcd8b becomes cdaB8b (note: "B8b").
// In such cases, this function combines the last dim with the first block(s),
// so xB8b becomes xb. Dimensions are treated like blocks, that is they don't
// report whole tensor size across given axis but rather number of underlying
// blocks for given dimension.
// Example: ABcd8b8a2b 32x48x1x7 will return description that amounts to...
// outermost-> 1c:4a:3b:7d:8b:8a:2b <-innermost
dimensions_t query_dims_and_blocks(const memory_desc_wrapper &mdw) {
    auto blocks = dims_by_stride(mdw);
    const dim_idx_t ndims = mdw.ndims();
    const auto &desc = mdw.blocking_desc();
    const dim_idx_t nblks = desc.inner_nblks;

    // Calculate info for inner blocks
    dimensions_t inner_blks(nblks);
    std::vector<dim_t> steps(ndims, 1);
    dim_t blks_size = 1;
    for (int i = nblks - 1; i >= 0; --i) {
        auto &blk = inner_blks[i];
        blk.idx = into<dim_idx_t>(desc.inner_idxs[i]);
        blk.size = desc.inner_blks[i];
        blk.step = steps[blk.idx];
        // steps increase in reverse order of how blocks are listed
        steps[blk.idx] *= blk.size;
        blks_size *= blk.size;
    }

    // Divide dim by its step to get block size
    for (auto &blk : blocks) {
        blk.step = steps[blk.idx];
        blk.size = utils::div_up(blk.size, blk.step);
    }

    // If we have any dims with block size 1, we ignore them.
    const auto size_1 = [](const dimension_t &b) { return b.size == 1; };
    const auto end = blocks.end();
    blocks.erase(std::remove_if(blocks.begin(), end, size_1), end);

    dim_t stride = blocks.empty() ? 1 : desc.strides[blocks[0].idx];
    for (auto &blk : inner_blks) {
        if (blk.size == 1) continue; // Can safely ignore blocks of size 1
        if (blocks.empty() || blocks[0].idx != blk.idx || blks_size != stride) {
            blocks.insert(blocks.begin(), blk);
        } else {
            // Combine blocks with repeated index if there is no extra padding
            blk.size *= blocks[0].size;
            blocks[0] = blk;
        }
        blks_size /= blk.size;
        stride = blks_size;
    }

    if (blocks.empty() && ndims > 0) {
        dimension_t blk;
        blk.idx = 0;
        blk.size = 1;
        blk.step = 1;
        blocks.push_back(blk);
    }
    return blocks;
}

dimensions_t query_dims_and_blocks(const memory_desc_t &md) {
    const memory_desc_wrapper mdw(md);
    return query_dims_and_blocks(mdw);
}

bool is_generic_faster_than_ref(
        const memory_desc_t &src_md, const memory_desc_t &dst_md) {
    const dim_t max_1d_ref_nelems = 512;
    const dim_t max_nd_ref_nelems = 512 * 512;
    auto nelems
            = std::max(utils::array_product(src_md.padded_dims, src_md.ndims),
                    utils::array_product(dst_md.padded_dims, dst_md.ndims));
    if (src_md.ndims == 1 && dst_md.ndims == 1)
        return nelems > max_1d_ref_nelems;
    auto src_blks = query_dims_and_blocks(src_md);
    auto dst_blks = query_dims_and_blocks(dst_md);
    if (src_blks.empty() || dst_blks.empty()) return false;
    auto src_inner_idx = src_blks[0].idx;
    auto dst_inner_idx = dst_blks[0].idx;
    auto scale = (src_inner_idx != dst_inner_idx) ? 2 : 1;
    return nelems > scale * max_nd_ref_nelems;
}

using dim_pair_t = std::array<dimension_t, 2>;

// Return whether the two blocks represent an equal part of the same dimension.
bool equal_blocks(const dim_pair_t &a, const dim_pair_t &b) {
    return (a[0].size == b[0].size && a[1].size == b[1].size);
}

// Combine dimension j into dimension i.
void combine(memory_desc_t &md, int i, int j) {
    const int new_ndims = md.ndims - 1;
    if (new_ndims == 0) return; // Don't delete the only dimension.
    auto &desc = md.format_desc.blocking;
    auto &strides = desc.strides;
    const int outer = strides[i] < strides[j] ? j : i;
    const int inner = strides[i] < strides[j] ? i : j;

    const auto outer_stride = strides[outer];
    const auto outer_size = md.padded_dims[outer];
    md.offset0 += strides[outer] * md.padded_offsets[outer];
    md.dims[i] = md.dims[outer] * md.padded_dims[inner];
    md.padded_dims[i] = md.padded_dims[i] * md.padded_dims[j];
    md.padded_offsets[i] = md.padded_offsets[inner];
    strides[i] = strides[inner];
    for (int k = j; k < new_ndims; ++k) {
        md.dims[k] = md.dims[k + 1];
        md.padded_dims[k] = md.padded_dims[k + 1];
        md.padded_offsets[k] = md.padded_offsets[k + 1];
        strides[k] = strides[k + 1];
    }
    md.dims[new_ndims] = 0;
    md.padded_dims[new_ndims] = 0;
    md.padded_offsets[new_ndims] = 0;
    strides[new_ndims] = 0;

    auto &idxs = desc.inner_idxs;
    auto &blks = desc.inner_blks;
    int nblks = desc.inner_nblks;
    auto blks_size = utils::array_product(blks, nblks);
    int count = 0;
    bool last_is_combined = false;
    dim_t blocks = 1;
    for (int k = 0; k < nblks; ++k) {
        if (idxs[k] == i || idxs[k] == j) {
            blocks *= blks[k];
            // Combine the innermost dim and outermost block when they have the
            // same index and no extra padding, e.g., ...A8a... -> ...a...
            if (count == 0 && strides[i] == blks_size) {
                md.dims[i] = md.padded_dims[i];
                strides[i] /= blks[k];
                blks_size /= blks[k];
            } else if (last_is_combined) {
                blks[count - 1] *= blks[k];
            } else {
                last_is_combined = true;
                blks[count] = blks[k];
                idxs[count] = i;
                count++;
            }
            continue;
        }
        last_is_combined = false;
        blks[count] = blks[k];
        idxs[count] = (idxs[k] > j ? idxs[k] - 1 : idxs[k]);
        count++;
    }
    // We've changed Nx1x...x1xM to 1x...x1xNM by combining dims, now fix the
    // strides of the size-1 dims by multiplying by the step of the size-N dim.
    auto outer_step = utils::div_up(outer_size, blocks);
    for (int k = 0; k < new_ndims; ++k) {
        if (strides[k] == outer_stride) strides[k] *= outer_step;
    }
    desc.inner_nblks = count;
    md.ndims = new_ndims;
}

void remove_bit(int &mask, int bit) {
    const int lower_bits = (1 << bit) - 1;
    mask = (mask & lower_bits) | ((mask >> 1) & ~lower_bits);
}

// For each dimension, determine if the inner dimensions do not account for its
// stride. We cannot combine a dimension that does not align with the stride of
// the next outer dimension.
int extended_dims(const memory_desc_t &md) {
    int mask = 0;
    const int ndims = md.ndims;
    const auto &blkg = md.format_desc.blocking;
    const int nblks = blkg.inner_nblks;

    auto dims = dims_by_stride(md);
    std::vector<dim_t> blocks(ndims, 1);
    dim_t expected_stride = 1;
    for (int i = 0; i < nblks; ++i) {
        auto idx = blkg.inner_idxs[i];
        auto blks = blkg.inner_blks[i];
        blocks[idx] *= blks;
        expected_stride *= blks;
    }

    for (int i = 0; i < ndims; ++i) {
        const auto &dim = dims[i];
        auto stride = blkg.strides[dim.idx];
        auto step = utils::div_up(dim.size, blocks[dim.idx]);
        if (stride != expected_stride) {
            mask |= (1 << dim.idx);
            // A size-1 dim carries no data; don't let it reset the expected
            // stride. Otherwise a size-1 dim whose stride matches the next
            // outer dim would absorb a real stride gap and hide it from that
            // outer dim, incorrectly allowing it to be combined.
            if (dim.size != 1) expected_stride = stride;
        }
        expected_stride *= step;
    }
    return mask;
}

struct pair_filter_t {
public:
    using value_type = dim_pair_t;

private:
    using const_dim_iterator_t = typename dimensions_t::const_iterator;
    using predicate_t = std::function<bool(const value_type &)>;

public:
    struct iterator_t {
        bool operator==(const iterator_t &o) const { return it == o.it; }
        bool operator!=(const iterator_t &o) const { return it != o.it; }
        value_type operator*() const { return {*it, *(it + 1)}; }
        iterator_t &operator++() {
            advance();
            return *this;
        }
        iterator_t operator++(int) {
            auto cpy = *this;
            advance();
            return cpy;
        }
        iterator_t(const_dim_iterator_t it, const_dim_iterator_t end,
                predicate_t pred)
            : it(it), end(end), pred(std::move(pred)) {
            advance(true);
        }

    private:
        void advance(bool check_first = false) {
            if (it == end || (check_first && pred(operator*()))) return;
            while (++it != end && !pred(operator*())) {}
        }

        const_dim_iterator_t it, end;
        predicate_t pred;
    };

    iterator_t begin() const { return {begin_, end_ - 1, pred}; }
    iterator_t end() const { return {end_ - 1, end_ - 1, pred}; }
    bool empty() const { return begin() == end(); }

    pair_filter_t(const dimensions_t &iter, const predicate_t &pred)
        : begin_(iter.begin()), end_(iter.end()), pred(pred) {}

private:
    const_dim_iterator_t begin_, end_;
    predicate_t pred;
};

#define NO_IDX dim_idx::invalid
// Find the index of the dimension that always and only follows the dimension
// with index idx. If none exists, return NO_IDX. If no dimension with index idx
// is present in the given block representation, return idx to delete the
// dimension
dim_idx_t successor(const dimensions_t &a, dim_idx_t idx) {
    dim_idx_t succ;
    auto match_idx = [&](const dim_pair_t &p) { return p[0].idx == idx; };
    auto match_xor = [&](const dim_pair_t &p) {
        return match_idx(p) ^ (p[1].idx == succ);
    };
    // idx is the index of outermost dim; it has no successor
    if (a.back().idx == idx) return NO_IDX;
    auto filtered = pair_filter_t(a, match_idx);
    // no dim with index idx appears in block representation; delete it
    if (filtered.empty()) return idx;
    succ = (*filtered.begin())[1].idx;
    // succ is the index of the innermost dim; it has no predecessor
    if (a.front().idx == succ) return NO_IDX;
    if (!pair_filter_t(a, match_xor).empty()) return NO_IDX;
    return succ;
}

// Find the index of the dimension that ALWAYS follows dimension `idx` in the
// given block representations. The successor dimension will be combined with
// the given dimension, or, in the case that the given dimension does not appear
// in the block representation, it will be deleted.
dim_idx_t successor(
        const dimensions_t &a, const dimensions_t &b, dim_idx_t idx) {
    auto succ = successor(a, idx);
    if (succ == NO_IDX || succ != successor(b, idx)) return NO_IDX;

    auto pred = [&](const dim_pair_t &p) { return p[0].idx == idx; };
    pair_filter_t iter_a(a, pred);
    pair_filter_t iter_b(b, pred);

    auto it_a = iter_a.begin();
    auto it_b = iter_b.begin();
    const auto end_a = iter_a.end();
    const auto end_b = iter_b.end();

    for (; it_a != end_a && it_b != end_b; ++it_a, ++it_b) {
        if (!equal_blocks(*it_a, *it_b)) return NO_IDX;
    }
    return (it_a != end_a || it_b != end_b) ? NO_IDX : succ;
}

bool can_be_combined(dim_idx_t idx, int mask) {
    return !(idx == NO_IDX || (mask & (1 << idx)));
}

void compress(memory_desc_t &a, memory_desc_t &b, int &a_mask, int &b_mask) {
    const auto blks_a = query_dims_and_blocks(a);
    const auto blks_b = query_dims_and_blocks(b);
    const int skip_mask = a_mask | b_mask | extended_dims(a) | extended_dims(b);

    const dim_idx_t ndims = a.ndims;
    std::vector<dim_idx_t> successors(ndims, NO_IDX);
    std::vector<dim_idx_t> aliases(ndims);
    for (dim_idx_t i = 0; i < ndims; ++i) {
        aliases[i] = i;
        if ((a_mask | b_mask) & (1 << i)) continue;
        auto succ = successor(blks_a, blks_b, i);
        if (!can_be_combined(succ, skip_mask)) continue;
        successors[i] = succ;
    }

    for (int i = ndims - 1; i >= 0; --i) {
        dim_idx_t succ = successors[i];
        if (succ == NO_IDX) continue;
        while (succ != aliases[succ])
            succ = aliases[succ];
        dim_idx_t from = std::max<dim_idx_t>(i, succ);
        dim_idx_t into = std::min<dim_idx_t>(i, succ);
        combine(a, into, from);
        combine(b, into, from);
        remove_bit(a_mask, from);
        remove_bit(b_mask, from);
        aliases[from] = into;
    }
}
#undef NO_IDX

void fix_steps(dimensions_t &blk, const dimensions_t &pkt) {
    dim_t steps[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < pkt.size(); i++) {
        steps[pkt[i].idx] *= pkt[i].size;
    }
    for (size_t i = 0; i < blk.size(); i++) {
        blk[i].step = steps[blk[i].idx];
        steps[blk[i].idx] *= blk[i].size;
    }
}

// Returns vector of blocks that were present in a but missing from b
dimensions_t find_missing_blocks(
        const dimensions_t &all, dimensions_t subset, bool round_up) {
    dimensions_t ret;
    for (size_t ia = 0; ia < all.size(); ia++) {
        dimension_t from_a = all[ia];
        for (size_t ib = 0; ib < subset.size(); ib++) {
            if (subset[ib].idx == from_a.idx) {
                auto smaller = std::min(from_a.size, subset[ib].size);
                if (round_up) {
                    from_a.size = utils::div_up(from_a.size, smaller);
                    subset[ib].size = utils::div_up(subset[ib].size, smaller);
                } else {
                    from_a.size /= smaller;
                    subset[ib].size /= smaller;
                }
            }
        }
        if (from_a.size > 1) { ret.push_back(from_a); }
    }
    return ret;
}

dimensions_t remainder(const dimensions_t &all, const dimensions_t &subset) {
    dimensions_t ret;
    for (size_t i = 0; i < all.size(); i++) {
        if (i < subset.size()) {
            if (all[i].size == subset[i].size) {
                continue;
            } else {
                dimension_t item;
                item.idx = all[i].idx;
                item.size = all[i].size / subset[i].size;
                item.step = all[i].step * subset[i].size;
                ret.push_back(item);
            }
        } else {
            ret.push_back(all[i]);
        }
    }
    return ret;
}

// Given format description, try to find formula for 16 adjacent items to
// vectorize across.
// Examples:
// For 1024x1024 ab, it will be (16b)
// for 16x16x16 ABc2a2b, it will be (4c2a2b)
bool fill_to_vect(
        int simd_size, const dimensions_t &all, dimensions_t &subset) {
    const int min_full_vecs = 5; // TODO: tune me
    dim_t current_size = 1;
    subset.clear();
    for (auto &dim : all) {
        dim_t next_size = current_size * dim.size;
        dim_t next_full_vecs = next_size / simd_size;
        if (next_full_vecs >= min_full_vecs || next_size % simd_size == 0) {
            // Vectorize innermost dim(s). If it's not divisible by simd size,
            // they will need to be padded. And for that the vectorised dim(s)
            // should be large enough because otherwise the padding would be
            // too significant fraction of tensor and it would hurt perf.
            dimension_t tmp = dim;
            tmp.size = simd_size / current_size;
            subset.push_back(tmp);
            return true;
        }
        // No hope of properly filling the vector.
        if (simd_size % next_size != 0) return false;
        current_size = next_size;
        subset.push_back(dim);
    }
    // there was not enough data in tensor to fill even a single packet
    return false;
}

bool add_to_vector(dimensions_t &v, const dimension_t &item) {
    if (v.empty() || item.idx != v.back().idx) {
        if (v.size() >= LOOP_NEST_LEVEL) { return false; }
        v.push_back(item);
        v.back().size = item.size;
    } else {
        v.back().size *= item.size;
    }
    return true;
}

bool no_more_such_idx(dimensions_t &vect, size_t iter) {
    const dim_idx_t idx_to_search_for = vect[iter].idx;
    for (size_t i = iter + 1; i < vect.size(); i++) {
        if (vect[i].idx == idx_to_search_for) { return false; }
    }
    return true;
}

// Given full description of tensor and subset of description,
// sort the subset in such way that it will describe longest possible
// sequence of continuous memory addresses.
// Example: full 32a32b4c4a, subset 12a2b4c,
// result = 3a2b4c4a, it gives 3 distant sets of 2*4*4 adjacent items
dimensions_t fix_order_to(dimensions_t input, dimensions_t ref) {
    dimensions_t ret;
    for (size_t i = 0; i < ref.size(); i++) {
        for (size_t j = 0; j < input.size(); j++) {
            if (ref[i].size != 1 && input[j].size != 1
                    && ref[i].idx == input[j].idx) {
                dim_t smaller = std::min(ref[i].size, input[j].size);
                if (no_more_such_idx(ref, i) || j == input.size() - 1) {
                    smaller = input[j].size;
                }
                dimension_t item = ref[i];
                item.size = smaller;
                ref[i].size = utils::div_up(ref[i].size, smaller);
                input[j].size = utils::div_up(input[j].size, smaller);
                add_to_vector(ret, item);
            }
        }
    }
    // It is possible that requested block on a dimension of src is bigger than
    // whole dimension in src. That happens when there's large padding in dst.
    // Add this block at the end, it will be handled by padding in opencl code.
    for (size_t i = 0; i < input.size(); i++) {
        if (input[i].size > 1) { add_to_vector(ret, input[i]); }
    }
    return ret;
}

dim_t check_size(const dimensions_t &block) {
    dim_t length = 1;
    for (size_t i = 0; i < block.size(); i++) {
        length *= block[i].size;
    }
    return length;
}

// Given full tensor description and subset of that description, find
// how many items are adjacent in memory
size_t check_burst_length(dimensions_t all, dimensions_t subset) {
    size_t length = 1;
    for (size_t i = 0; i < all.size(); i++) {
        for (size_t j = 0; j < subset.size(); j++) {
            if (all[i].idx == subset[j].idx) {
                auto smaller = std::min(all[i].size, subset[j].size);
                length *= (int)smaller;
                all[i].size /= smaller;
                subset[j].size /= smaller;
            }
        }
        if (all[i].size != 1) {
            return length;
        } // dim not covered in block, so burst ends
    }
    return length;
}

// Given full tensor description and subset of that description which
// determines how many items will be read in a burst, try to enlarge subset
// to increase burst size to achieve better cache line utilizaton.
// Example: full 32a32b4c4a, subset 12a2b2c,
// current burst size = 8 (2c*4a); enlarge subset to 12a2b4c to achieve
// burst size = 32 (2b*4c*4a)
bool increase_burst(dimensions_t all, dimensions_t &subset, dimensions_t &other,
        size_t itemlimit, size_t current_size, size_t optimal_size) {
    const dim_t space_coeff = itemlimit / check_size(subset);
    const dim_t request_coeff = utils::div_up(optimal_size, current_size);
    dimensions_t subset_copy = subset;
    if (space_coeff < 2) { return false; }
    for (size_t i = 0; i < all.size(); i++) {
        for (size_t j = 0; j < subset_copy.size(); j++) {
            if (all[i].idx == subset_copy[j].idx) {
                auto smaller = std::min(all[i].size, subset_copy[j].size);
                all[i].size /= smaller;
                subset_copy[j].size /= smaller;
            }
        }
        if (all[i].size != 1) {
            // add to subset new item or enlarge last item, if it was the same dim
            auto incr = std::min(space_coeff, all[i].size);
            incr = std::min(incr, request_coeff);
            all[i].size = incr;
            bool success = add_to_vector(subset, all[i]);
            if (!success) { return false; }
            add_to_vector(other, all[i]);
            return true;
        }
    }
    return false;
}

// "packet" - set of 16 adjacent data to be read in one go by a subgroup
// "block"  - how many iterations of packet read should a subgroup do
// This function splits tensor description into blocks and packets in such way
// that optimizes burst length.
bool split_into_blocks_and_packets(size_t vect, size_t optimal_burst_bytes,
        size_t memlimit_bytes, size_t sizeof_src, size_t sizeof_dst,
        const dimensions_t &src, const dimensions_t &dst,
        dimensions_t &src_packet, dimensions_t &src_block,
        dimensions_t &dst_packet, dimensions_t &dst_block) {

    // 1. determine composition of src and dst packet
    if (!fill_to_vect((int)vect, src, src_packet)) { return false; }
    if (!fill_to_vect((int)vect, dst, dst_packet)) { return false; }
    // 2. determine which parts of tensor format tag are left after taking away packet
    dimensions_t sremainder = remainder(src, src_packet);
    dimensions_t dremainder = remainder(dst, dst_packet);
    // 3. The same amount of data will be read and written. So, every dimension
    // that's in src packet and not in dst packet must be in dst block.
    src_block = find_missing_blocks(dst_packet, src_packet, true);
    dst_block = find_missing_blocks(src_packet, dst_packet, false);
    // 4a. Check how much continuous data will be read/written...
    size_t burst_size_src
            = vect * sizeof_src * check_burst_length(sremainder, src_block);
    size_t burst_size_dst
            = vect * sizeof_dst * check_burst_length(dremainder, dst_block);
    bool success = true;
    // TODO: use smaller of SRC_T, DST_T type to conserve local mem
    size_t itemlimit = memlimit_bytes / (vect * sizeof_src);
    // 4b. ... and determine if that's long enough to achieve good performance
    while (success
            && (burst_size_src < optimal_burst_bytes
                    || burst_size_dst < optimal_burst_bytes)) {
        // 5. If burst needs to be longer, attempt to increase block size (but
        // don't exceed local memory limits as that would hurt performance)
        if (burst_size_src < burst_size_dst) {
            success = increase_burst(sremainder, src_block, dst_block,
                    itemlimit, burst_size_src, optimal_burst_bytes);
        } else {
            success = increase_burst(dremainder, dst_block, src_block,
                    itemlimit, burst_size_dst, optimal_burst_bytes);
        }
        burst_size_src
                = vect * sizeof_src * check_burst_length(sremainder, src_block);
        burst_size_dst
                = vect * sizeof_dst * check_burst_length(dremainder, dst_block);
    }
    // 6. At this point contents of src block and dst blocks are not sorted.
    // Sort each of them according to tensor format tag to make longest
    // possible continuous memory accesses.
    src_block = fix_order_to(src_block, std::move(sremainder));
    dst_block = fix_order_to(dst_block, std::move(dremainder));
    fix_steps(src_block, src_packet);
    fix_steps(dst_block, dst_packet);
    return true;
}

bool fill_conf_vld(const memory_desc_wrapper &src,
        const memory_desc_wrapper &dst, int scale_mask, size_t memlimit_bytes,
        size_t optimal_burst_bytes, vectorize_last_dim_t &cfg,
        dim_idx_t &vect_dim, int &vect_size, dim_t *blocks) {

    const dimensions_t src_dims = query_dims_and_blocks(src);
    const dimensions_t dst_dims = query_dims_and_blocks(dst);
    dimensions_t src_packet, src_block, dst_packet, dst_block;
    bool success = split_into_blocks_and_packets(16, memlimit_bytes,
            optimal_burst_bytes, src.data_type_size(), dst.data_type_size(),
            src_dims, dst_dims, src_packet, src_block, dst_packet, dst_block);
    if (!success) { return false; }
    // Below: unpack std vectors into POD arrays

    cfg.src_vect_limit = (int)check_burst_length(src_packet, src_packet);
    cfg.dst_vect_limit = (int)check_burst_length(dst_packet, dst_packet);

    // reset packet and loop
    for (size_t i = 0; i < LOOP_NEST_LEVEL; i++) {
        cfg.src_vct[i].blk_size = 1;
        cfg.dst_vct[i].blk_size = 1;
        cfg.src_blk[i].blk_size = 1;
        cfg.dst_blk[i].blk_size = 1;
        cfg.src_vct[i].step_size = 1;
        cfg.dst_vct[i].step_size = 1;
        cfg.src_blk[i].step_size = 1;
        cfg.dst_blk[i].step_size = 1;
        cfg.src_vct[i].dim_idx = 0;
        cfg.dst_vct[i].dim_idx = 0;
        cfg.src_blk[i].dim_idx = 0;
        cfg.dst_blk[i].dim_idx = 0;
    }
    cfg.src_vct[0].blk_size = into<int>(src_packet[0].size);
    cfg.src_vct[0].dim_idx = src_packet[0].idx;
    cfg.dst_vct[0].blk_size = into<int>(dst_packet[0].size);
    cfg.dst_vct[0].dim_idx = dst_packet[0].idx;
    for (size_t i = 0; i < src_packet.size(); i++) {
        cfg.src_vct[i].dim_idx = src_packet[i].idx;
        cfg.src_vct[i].blk_size = into<int>(src_packet[i].size);
        cfg.src_vct[i].step_size = into<int>(src_packet[i].step);
    }
    for (size_t i = 0; i < dst_packet.size(); i++) {
        cfg.dst_vct[i].dim_idx = dst_packet[i].idx;
        cfg.dst_vct[i].blk_size = into<int>(dst_packet[i].size);
        cfg.dst_vct[i].step_size = into<int>(dst_packet[i].step);
    }

    // fill src's and dst's loop recipe
    for (size_t i = 0; i < src_block.size(); i++) {
        cfg.src_blk[i].dim_idx = src_block[i].idx;
        cfg.src_blk[i].blk_size = into<int>(src_block[i].size);
        cfg.src_blk[i].step_size = into<int>(src_block[i].step);
    }
    for (size_t i = 0; i < dst_block.size(); i++) {
        cfg.dst_blk[i].dim_idx = dst_block[i].idx;
        cfg.dst_blk[i].blk_size = into<int>(dst_block[i].size);
        cfg.dst_blk[i].step_size = into<int>(dst_block[i].step);
    }
    cfg.vector_dim = dst_packet[0].idx;
    vect_dim = dst_packet[0].idx;
    vect_size = 16;
    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        if (cfg.dst_blk[i].blk_size != 1) {
            blocks[cfg.dst_blk[i].dim_idx] *= cfg.dst_blk[i].blk_size;
        }
    }
    // Multiply by 16 the size of the dimension that will be vectorized.
    // This is workaround for 2 dispatcher problems:
    // - it doesn't allow vectorization of dims that are not divisible by 16
    // - vectorized dim's coordinate returned in openCL side is rounded to 16
    // Here we multiply the dim-to-be-vectorized by 16 and it immediately
    // solves 1st issue; we declare larger block on dim-to-be-vectorized to
    // prevent dispatcher from spawning too many work items over this enlarged
    // dim; and later on openCL side we'll divide this dim's coordinate by 16
    // to get fine-grained coordinates not rounded to 16.
    cfg.rescale_coeff = 16;

    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        auto db = cfg.dst_vct[i];
        blocks[db.dim_idx] *= db.blk_size;
    }

    return true;
}

status_t generic_t::pd_t::init_conf(const impl::engine_t *engine) {
    using namespace format_tag;

    size_t memlimit_bytes;
    size_t optimal_burst_bytes;

    const memory_desc_wrapper original_src_mdw(src_md());
    const memory_desc_wrapper original_dst_mdw(dst_md());
    quantization_t src_quant(attr(), original_src_mdw, DNNL_ARG_SRC);
    quantization_t dst_quant(attr(), original_dst_mdw, DNNL_ARG_DST);

    auto src_mask = src_quant.scale_mask();
    auto dst_mask = dst_quant.scale_mask();

    memory_desc_t new_a;
    memory_desc_t new_b;
    primitive_attr_t attr_copy = *attr();
    memcpy(&new_a, src_md(), sizeof(new_a));
    memcpy(&new_b, dst_md(), sizeof(new_b));
    compress(new_a, new_b, src_mask, dst_mask);
    if (src_mask >= 0) { CHECK(attr_copy.scales_.set(DNNL_ARG_SRC, src_mask)); }
    if (dst_mask >= 0) { CHECK(attr_copy.scales_.set(DNNL_ARG_DST, dst_mask)); }

    VDISPATCH_REORDER_IC(is_generic_faster_than_ref(new_a, new_b),
            VERBOSE_SKIP_PRIMITIVE_IMPL);

    const memory_desc_wrapper src_mdw(new_a);
    const memory_desc_wrapper dst_mdw(new_b);
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    conf.require_stateless_addressing = has_large_buffers();

    conf.src_quant = {&attr_copy, src_mdw, DNNL_ARG_SRC};
    conf.dst_quant = {&attr_copy, dst_mdw, DNNL_ARG_DST};
    conf.sum_quant = {&attr_copy};

    status_t status = status::success;

    const auto &padded_dims = dst_mdw.padded_dims();
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(padded_dims, conf.ndims);

    conf.sub_group_size = 1;
    if (conf.nelems == 0) { return status::success; }
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);

    // Theoretically, bursts should be at least big enough to span whole
    // cache line and bigger bursts should give better perf as long as
    // local mem capacity is not exceeded. However, all tests show that
    // burst size 64 gives best performance regardless of cache line size.
    memlimit_bytes = 2048;
    optimal_burst_bytes = 64;

    dim_t blocks[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    int vect_size = 1;
    dim_idx_t vect_dim = 0;

    VDISPATCH_REORDER_IC(
            fill_conf_vld(src_mdw, dst_mdw, src_mask | dst_mask, memlimit_bytes,
                    optimal_burst_bytes, conf.aux_data.vld, vect_dim, vect_size,
                    &blocks[0]),
            VERBOSE_BAD_PARAM, "conf_vld");

    conf.sub_group_size = vect_size;

    conf.dispatch = intel_engine->create_dispatch(dst_mdw.md_);

    for (dim_idx_t i = 0; i < MAX_NDIMS; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < into<dim_idx_t>(dst_mdw.ndims())) {
            uint64_t dim = padded_dims[i];
            // Pad vectorized dim to multiple of block size (to make sure that
            // enough work items will be generated to have only full subgroups,
            // no fractions) then multiply it by vector size (to work around
            // dispatcher's limitation that vectorized dim must be divisible by
            // vector size).
            if (i == vect_dim) {
                dim = utils::rnd_up(dim, blocks[i]);
                dim *= 16;
            }
            conf.dispatch.define_dim(dim_str, i, dim, blocks[i]);
        } else {
            conf.dispatch.define_dim(dim_str, 1);
        }
    }
    if (vect_size != 1) {
        const auto dim_str = utils::format("D%d", vect_dim);
        CHECK(conf.dispatch.vectorize_dim(dim_str, vect_size));
    }

    conf.dispatch.generate();

    return status;
}

status_t generic_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    if (conf.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.add_option("-cl-std=CL2.0");

    conf.src_quant.define_macros(kernel_ctx, "SRC");
    conf.dst_quant.define_macros(kernel_ctx, "DST");
    conf.sum_quant.define_macros(kernel_ctx, "SUM");

    def_dispatch(kernel_ctx, conf.dispatch);
    kernel_ctx.require_stateless_addressing(conf.require_stateless_addressing);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    kernel_ctx.define_int("GENERIC_REORDER", 1);
    kernel_ctx.define_int("VECT_DIM", conf.aux_data.vld.vector_dim);
    kernel_ctx.define_int("VECT_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("RESCALE_COEFF", conf.aux_data.vld.rescale_coeff);
    kernel_ctx.define_int("LIMIT_SSGID", conf.aux_data.vld.src_vect_limit);
    kernel_ctx.define_int("LIMIT_DSGID", conf.aux_data.vld.dst_vect_limit);
    compute::nd_range_t nd_range = conf.dispatch.nd_range();
    const auto &lws = nd_range.local_range();
    if (!lws) return status::runtime_error;
    kernel_ctx.define_int("SG_PER_WG", lws.nelems() / conf.sub_group_size);
    int i = 0;
    int cache_dim[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    while (i < LOOP_NEST_LEVEL) {
        cache_dim[conf.aux_data.vld.dst_vct[i].dim_idx]
                *= conf.aux_data.vld.dst_vct[i].blk_size;
        cache_dim[conf.aux_data.vld.dst_blk[i].dim_idx]
                *= conf.aux_data.vld.dst_blk[i].blk_size;
        kernel_ctx.define_int(std::string("S_BLK_SIZE_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].blk_size);
        kernel_ctx.define_int(std::string("S_BLK_STEP_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].step_size);
        kernel_ctx.define_int(std::string("S_BLK_IDX_") + std::to_string(i),
                conf.aux_data.vld.src_blk[i].dim_idx);
        kernel_ctx.define_int(std::string("D_BLK_SIZE_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].blk_size);
        kernel_ctx.define_int(std::string("D_BLK_STEP_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].step_size);
        kernel_ctx.define_int(std::string("D_BLK_IDX_") + std::to_string(i),
                conf.aux_data.vld.dst_blk[i].dim_idx);
        i++;
    }
    int cache_stride = 1;
    for (int i = 0; i < MAX_NDIMS; i++) {
        kernel_ctx.define_int(
                std::string("CACHE_STRIDE_") + std::to_string(i), cache_stride);
        cache_stride *= cache_dim[i];
    }
    int s_size_so_far = 1;
    int d_size_so_far = 1;
    for (int i = 0; i < LOOP_NEST_LEVEL; i++) {
        auto s = conf.aux_data.vld.src_vct[i];
        auto d = conf.aux_data.vld.dst_vct[i];
        kernel_ctx.define_int(
                std::string("S_MOD_") + std::to_string(i), s.blk_size);
        kernel_ctx.define_int(
                std::string("S_DIV_") + std::to_string(i), s_size_so_far);
        kernel_ctx.define_int(
                std::string("S_MUL_") + std::to_string(i), s.step_size);
        kernel_ctx.define_int(
                std::string("S_IDX_") + std::to_string(i), s.dim_idx);
        kernel_ctx.define_int(
                std::string("D_MOD_") + std::to_string(i), d.blk_size);
        kernel_ctx.define_int(
                std::string("D_DIV_") + std::to_string(i), d_size_so_far);
        kernel_ctx.define_int(
                std::string("D_MUL_") + std::to_string(i), d.step_size);
        kernel_ctx.define_int(
                std::string("D_IDX_") + std::to_string(i), d.dim_idx);

        s_size_so_far *= s.blk_size;
        d_size_so_far *= d.blk_size;
    }

    return status::success;
}

void generic_t::pd_t::init_scratchpad() {
    if (conf.src_quant.with_scale()) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_src_scales,
                conf.src_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
    if (conf.dst_quant.with_scale()) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_dst_scales,
                conf.dst_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
}

status_t generic_t::execute(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);
    CHECK(status);

    const auto &conf = pd()->conf;
    if (conf.nelems == 0) { return status::success; }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    arg_list.set(2, conf.src_quant.scales(ctx));
    arg_list.set(3, conf.src_quant.zero_points(ctx));
    arg_list.set(4, conf.dst_quant.scales(ctx));
    arg_list.set(5, conf.dst_quant.zero_points(ctx));

    arg_list.set(6, conf.sum_quant.scales());
    arg_list.set(7, conf.sum_quant.zero_points());

    auto nd_range = conf.dispatch.nd_range();

    return large_parallel_for(ctx, nd_range, kernel_, arg_list, 8);
}

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
