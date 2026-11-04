/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#ifndef GPU_INTEL_POST_OPS_HPP
#define GPU_INTEL_POST_OPS_HPP

#include "common/math_utils.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/intel/utils.hpp"

namespace gemmstone {
namespace dsl {
namespace ir {
class expr_t;
}
} // namespace dsl
} // namespace gemmstone

namespace dnnl {
namespace impl {

// Uses binary_t to represent prelu with alg_kind_t = eltwise_relu
namespace alg_kind {
const alg_kind_t binary_prelu = eltwise_relu;
};

namespace gpu {
namespace intel {

namespace post_op {

// Wrapper structure to enable extending specializations without requiring
// constructor API changes.
struct specializations_t {
    struct inline_mode_t {
        constexpr inline_mode_t() = default;
        static constexpr inline_mode_t always() { return {mode_t::always}; }
        static constexpr inline_mode_t never() { return {mode_t::never}; }
        static constexpr inline_mode_t if_zero() { return {mode_t::if_zero}; }

        // Alias for never() to clarify the intended reason inlining is dropped
        static constexpr inline_mode_t impl_managed() { return never(); }

        bool is_inlined() const {
            switch (mode_) {
                case mode_t::always: return true;
                case mode_t::never: return false;
                default: gpu_error_not_expected(); return true;
            }
        }

        template <typename T>
        bool is_inlined(T value) const {
            switch (mode_) {
                case mode_t::always: return true;
                case mode_t::never: return false;
                case mode_t::if_zero: return value == 0;
            }
            gpu_error_not_expected();
            return true;
        }

    private:
        enum class mode_t : uint8_t { always, never, if_zero };
        constexpr inline_mode_t(mode_t m) : mode_(m) {};

        mode_t mode_ = mode_t::always;
    };

    struct eltwise_t {
        constexpr eltwise_t() = default;
        constexpr eltwise_t(
                inline_mode_t scale, inline_mode_t alpha, inline_mode_t beta)
            : scale(scale), alpha(alpha), beta(beta) {};
        inline_mode_t scale;
        inline_mode_t alpha;
        inline_mode_t beta;
    } eltwise;

    struct sum_t {
        constexpr sum_t() = default;
        constexpr sum_t(inline_mode_t scale, inline_mode_t zero_point)
            : scale(scale), zero_point(zero_point) {};
        inline_mode_t scale;
        inline_mode_t zero_point;
    } sum;

    struct binary_t {
        constexpr binary_t() = default;
        constexpr binary_t(inline_mode_t src1_desc_layout)
            : src1_desc_layout(src1_desc_layout) {};
        inline_mode_t src1_desc_layout;
    } binary;
};

// Helper to extend the memory descriptor dimensions (e.g. NCW -> NCDHW).
struct ndim_normalizer_t {
    constexpr ndim_normalizer_t() = default;
    constexpr ndim_normalizer_t(int insert_idx, int bcast_ndims)
        : insert_idx(insert_idx), bcast_ndims(bcast_ndims) {}

    int ndims(const memory_desc_t &md) const { return md.ndims + bcast_ndims; }

    int dim_idx(int md_idx) const {
        return (md_idx < insert_idx) ? md_idx : md_idx + bcast_ndims;
    }

    dim_t dim(int idx, const memory_desc_t &md) const {
        auto &dims = md.dims;
        return (idx < insert_idx)
                ? dims[idx]
                : (idx < insert_idx + bcast_ndims ? 1
                                                  : dims[idx - bcast_ndims]);
    }

    dim_t stride(int idx, const memory_desc_t &md) const {
        const auto &strides = memory_desc_wrapper(md).strides();
        return (idx < insert_idx)
                ? strides[idx]
                : (idx < insert_idx + bcast_ndims ? 0
                                                  : strides[idx - bcast_ndims]);
    }

    // Position to insert broadcast dimensions, dimensions
    // are inserted before this index.
    int insert_idx = 0;
    // Number of broadcast dimensions to insert.
    int bcast_ndims = 0;
};

// New type to prevent misuse of relative_md_t indices with memory_desc_t indices
struct relative_idx_t {
    constexpr relative_idx_t() = default;
    constexpr relative_idx_t(int8_t v) : value_(v) {};
    constexpr bool operator==(const relative_idx_t &o) const {
        return value_ == o.value_;
    }
    constexpr bool is_innermost() const { return value_ == 0; }
    constexpr bool is_unset() const { return value_ < 0; }
    std::string str() const {
        if (is_unset()) return "(unset)";
        char name[2] = {into<char>(value_ + 'a'), 0};
        return name;
    }

protected:
    friend struct relative_md_t;
    constexpr int as_int() const { return value_; }

private:
    int8_t value_ = -1;
};

// Represents a memory with dimensions inferred from another memory descriptor.
// Indices are stored from inner-most to outer-most which is the reversed order
// when compared to memory descriptor. This allows us to omit storing ndims as
// we can assume all larger dimensions are implicitly broadcasted.
struct relative_md_t {
    using idx_t = relative_idx_t;
    static constexpr int to_md_idx(idx_t idx, int ndims) {
        return ndims - 1 - idx.as_int();
    }
    static idx_t from_md_idx(
            int idx, int ndims, const ndim_normalizer_t &ndim_normalizer) {
        return {into<int8_t>(ndims - 1 - ndim_normalizer.dim_idx(idx))};
    }

    // A compressed representation of the inner block. This cannot represent all
    // memory layouts from a memory descriptor, but blocked memory layouts are
    // created using format_tags which should all fit in this structure.
    struct blocking_t {
        static constexpr uint8_t unset_block = 0;
        static constexpr int max_dims = 4;

        bool empty() const { return idxs[0].is_unset(); }

        std::string str() const {
            ostringstream_t oss;
            for (int i = max_dims - 1; i >= 0; i--) {
                if (idxs[i].is_unset()) continue;
                oss << int(blocks[i]) << idxs[i].str();
            }
            return oss.str();
        }
#if __cplusplus >= 202002L
        bool operator==(const blocking_t &) const = default;
#endif

        std::array<uint8_t, max_dims> blocks
                = {unset_block, unset_block, unset_block, unset_block};
        std::array<idx_t, max_dims> idxs = {};
    };

    relative_md_t() = default;
    relative_md_t(int md_broadcast_mask, int md_inner_dim, int ndims,
            data_type_t dt, const ndim_normalizer_t &ndim_normalizer)
        : dt(dt)
        , broadcast_mask(uint16_t(0xFFFFu << ndims))
        , inner_dim(from_md_idx(md_inner_dim, ndims, ndim_normalizer)) {
        for (int i = 0; i < ndims; i++) {
            auto rmd_idx = from_md_idx(i, ndims, ndim_normalizer).as_int();
            broadcast_mask |= ((md_broadcast_mask >> i) & 1) << rmd_idx;
        }
    }
    static status_t make(relative_md_t &rmd, const memory_desc_t &md,
            const ndim_normalizer_t &ndim_normalizer) {
        memory_desc_wrapper mdw(md);
        VCONDCHECK(primitive, create, dispatch, post_ops,
                mdw.is_blocking_desc() || mdw.is_grouped_desc(),
                status::unimplemented, "%s,expected format_kind::blocked, got ",
                dnnl_fmt_kind2str(md.format_kind));

        rmd = relative_md_t();
        rmd.dt = md.data_type;

        auto ndims = ndim_normalizer.ndims(md);

        // Only blocked layouts have inner blocks.
        if (mdw.is_blocking_desc()) {
            auto &blocking = mdw.blocking_desc();
            gpu_assert(blocking.inner_nblks <= blocking_t::max_dims);

            for (dim_t i = 0; i < blocking.inner_nblks; i++) {
                auto rmd_i = blocking.inner_nblks - 1 - i;
                rmd.inner_layout.idxs[rmd_i]
                        = from_md_idx(into<int>(blocking.inner_idxs[i]), ndims,
                                ndim_normalizer);
                rmd.inner_layout.blocks[rmd_i]
                        = into<uint8_t>(blocking.inner_blks[i]);
            }
        }

        // Default all dimensions to broadcast
        rmd.broadcast_mask = ~0;
        uint16_t mask_bit = 1;
        for (int i = ndims - 1; i >= 0; i--) {
            auto d = ndim_normalizer.dim(i, md);
            if (d > 1 || d == DNNL_RUNTIME_DIM_VAL)
                rmd.broadcast_mask &= ~mask_bit;
            mask_bit = static_cast<uint16_t>(mask_bit << 1);
        }

        dim_t min_stride = std::numeric_limits<dim_t>::max();
        for (int i = 0; i < ndims; i++) {
            if (ndim_normalizer.dim(i, md) > 1
                    && ndim_normalizer.stride(i, md) <= min_stride) {
                rmd.inner_dim = from_md_idx(i, ndims, ndim_normalizer_t());
                min_stride = ndim_normalizer.stride(i, md);
            }
        }
        if (rmd.inner_dim.is_unset()) rmd.inner_dim = {0};

        return status::success;
    }

    std::string ocl_defines(const std::string &prefix,
            const std::array<std::string, MAX_NDIMS> &strides, int ndims) const;

    gemmstone::dsl::ir::expr_t get_offset(
            const std::vector<gemmstone::dsl::ir::expr_t> &dim_idxs,
            const std::vector<gemmstone::dsl::ir::expr_t> &strides) const;

    bool is_broadcast(int idx, int ndims) const {
        idx_t norm = from_md_idx(idx, ndims, {});
        if (norm.is_unset()) return true;
        return (1 << norm.as_int()) & broadcast_mask;
    }

    bool is_inner_dim(int idx, int ndims) const {
        return idx == to_md_idx(inner_dim, ndims);
    }

    // Implicitly removes size 1 outer-dimensions from the original memory
    // descriptor
    int ndims() const {
        size_t dim_mask = broadcast_mask ^ 0xffff;
        return math::ilog2q(dim_mask) + 1;
    }

    std::string str() const {
        if (broadcast_mask == 0xFFFF) {
            gpu_assert(inner_layout.empty());
            return std::string("{scalar}.") + dnnl_dt2str(dt);
        }

        // Use reverse ordering to align with oneDNN format tag semantics
        ostringstream_t oss;
        const char *prefix = "{";
        for (int i = 15; i >= 0; i--) {
            if (broadcast_mask & (1 << i)) continue;
            std::cout << prefix;
            oss << idx_t(into<int8_t>(i)).str();
            prefix = "";
        }
        oss << "}:";
        if (!inner_dim.is_unset()) oss << inner_dim.str();
        if (!inner_layout.empty()) oss << inner_layout.str();
        oss << "." << dnnl_dt2str(dt);
        return oss.str();
    }
#if __cplusplus >= 202002L
    bool operator==(const relative_md_t &) const = default;
#endif

    blocking_t inner_layout;
    data_type_t dt = data_type::undef;
    uint16_t broadcast_mask = 0;
    idx_t inner_dim;
    uint8_t pad[1] = {};
};

enum class kind_t {
    undef,
    sum,
    eltwise,
    conv,
    binary,
};

struct sum_t {
    sum_t() = default;
    sum_t(const post_ops_t::entry_t::sum_t &op,
            const specializations_t::sum_t &s)
        : dt(op.dt)
        , inline_scale(s.scale.is_inlined(op.scale))
        , inline_zero_point(s.zero_point.is_inlined(op.zero_point))
        , scale(inline_scale ? op.scale : NAN)
        , zero_point(inline_zero_point ? op.zero_point : -1) {}

#if __cplusplus >= 202002L
    bool operator==(const sum_t &) const = default;
#endif

    void serialize(serialization_stream_t &s) const {
        s.append(dt);
        s.append(inline_scale);
        s.append(inline_zero_point);
        s.append(scale);
        s.append(zero_point);
    }

    static sum_t deserialize(deserializer_t &d) {
        sum_t e {};
        d.pop(e.dt);
        d.pop(e.inline_scale);
        d.pop(e.inline_zero_point);
        d.pop(e.scale);
        d.pop(e.zero_point);
        return e;
    }

    data_type_t dt = data_type::undef;
    bool inline_scale;
    bool inline_zero_point;
    uint8_t pad[2] = {};
    float scale = 0;
    int zero_point = 0;
};

struct eltwise_t {
    eltwise_t() = default;
    eltwise_t(const post_ops_t::entry_t::eltwise_t &op,
            const specializations_t::eltwise_t &s)
        : alg(op.alg)
        , inline_scale(s.scale.is_inlined(op.scale))
        , inline_alpha(s.alpha.is_inlined(op.alpha))
        , inline_beta(s.beta.is_inlined(op.beta))
        , scale(inline_scale ? op.scale : NAN)
        , alpha(inline_alpha ? op.alpha : NAN)
        , beta(inline_beta ? op.beta : NAN) {}

#if __cplusplus >= 202002L
    bool operator==(const eltwise_t &) const = default;
#endif

    void serialize(serialization_stream_t &s) const {
        s.append(alg);
        s.append(inline_scale);
        s.append(inline_alpha);
        s.append(inline_beta);
        s.append(scale);
        s.append(alpha);
        s.append(beta);
    }
    static eltwise_t deserialize(deserializer_t &d) {
        eltwise_t e {};
        d.pop(e.alg);
        d.pop(e.inline_scale);
        d.pop(e.inline_alpha);
        d.pop(e.inline_beta);
        d.pop(e.scale);
        d.pop(e.alpha);
        d.pop(e.beta);
        return e;
    }

    alg_kind_t alg = alg_kind::undef;
    bool inline_scale = {};
    bool inline_alpha = {};
    bool inline_beta = {};
    uint8_t pad[1] = {};
    float scale = 0, alpha = 0, beta = 0;
};

struct depthwise_conv_t {
    depthwise_conv_t() = default;
    depthwise_conv_t(const post_ops_t::entry_t::depthwise_conv_t &op)
        : kernel(op.kernel)
        , stride(op.stride)
        , padding(op.padding)
        , wei_dt(op.wei_dt)
        , bias_dt(op.bias_dt)
        , dst_dt(op.dst_dt) {}

#if __cplusplus >= 202002L
    bool operator==(const depthwise_conv_t &) const = default;
#endif

    dim_t kernel = 0;
    dim_t stride = 0;
    dim_t padding = 0;
    data_type_t wei_dt = data_type::undef;
    data_type_t bias_dt = data_type::undef;
    data_type_t dst_dt = data_type::undef;
    uint8_t pad[4] = {};
};

struct binary_t {
    binary_t() = default;
    static status_t make(binary_t &b, const post_ops_t::entry_t::binary_t &op,
            const specializations_t::binary_t &s,
            const post_op::ndim_normalizer_t &ndim_normalizer) {
        if (s.src1_desc_layout.is_inlined())
            CHECK(relative_md_t::make(
                    b.src1_desc, op.src1_desc, ndim_normalizer));
        else
            b.src1_desc.dt = op.src1_desc.data_type;

        b.alg = op.alg;
        return status::success;
    }

    static status_t make(binary_t &b, const post_ops_t::entry_t::prelu_t &op,
            const memory_desc_wrapper &dst_md,
            const specializations_t::binary_t &s,
            const ndim_normalizer_t &ndim_normalizer) {
        if (s.src1_desc_layout.is_inlined()) {
            auto bcast_mask = ~op.mask;
            int inner_dim = 0;
            for (int i = 0; i < dst_md.ndims(); i++) {
                // Normalize size 1 dimensions to broadcasts.
                if (dst_md.dims()[i] == 1) bcast_mask |= 1 << i;
                // Prelu layout is axb, so b dimension is innermost
                if ((~bcast_mask & (1 << i)) && inner_dim != 1) inner_dim = i;
            }
            b.src1_desc = relative_md_t(bcast_mask, inner_dim, dst_md.ndims(),
                    data_type::f32, ndim_normalizer);
        } else {
            b.src1_desc.dt = data_type::f32;
        }

        b.alg = alg_kind::binary_prelu;
        return status::success;
    }

#if __cplusplus >= 202002L
    bool operator==(const binary_t &) const = default;
#endif

    relative_md_t src1_desc;
    alg_kind_t alg;
    uint8_t pad[4] = {};
};

} // namespace post_op

// The representation of post_ops_t is not compatible with reuse due to the
// inclusion of memory descriptors and runtime arguments. A separate
// representation is added here for use in reusable GPU kernels.
struct gpu_post_ops_t {

    gpu_post_ops_t() = default;

    static status_t make(gpu_post_ops_t &gpu_post_ops,
            const post_ops_t &post_ops, const memory_desc_wrapper &dst_md,
            post_op::specializations_t opts = {},
            post_op::ndim_normalizer_t ndim_normalizer = {}) {
        auto &ops = gpu_post_ops.ops_;
        ops.clear();
        ops.reserve(into<size_t>(post_ops.len()));
        using namespace post_op;
        for (auto &entry : post_ops.entry_) {
            switch (entry.kind) {
                case (primitive_kind::sum):
                    ops.emplace_back(sum_t(entry.sum, opts.sum));
                    break;
                case (primitive_kind::eltwise):
                    ops.emplace_back(eltwise_t(entry.eltwise, opts.eltwise));
                    break;
                case (primitive_kind::convolution):
                    ops.emplace_back(depthwise_conv_t(entry.depthwise_conv));
                    break;
                case (primitive_kind::binary): {
                    binary_t b;
                    CHECK(binary_t::make(
                            b, entry.binary, opts.binary, ndim_normalizer));
                    ops.emplace_back(b);
                    break;
                }
                case (primitive_kind::prelu): {
                    binary_t b;
                    CHECK(binary_t::make(b, entry.prelu, dst_md, opts.binary,
                            ndim_normalizer));
                    ops.emplace_back(b);
                    break;
                }
                default: gpu_error_not_expected(); return status::runtime_error;
            }
        }
        return status::success;
    }

    struct entry_t {
        entry_t() : kind_(post_op::kind_t::undef) {}
        entry_t(post_op::sum_t e) : kind_(post_op::kind_t::sum), sum_(e) {}
        entry_t(post_op::eltwise_t e)
            : kind_(post_op::kind_t::eltwise), eltwise_(e) {}
        entry_t(post_op::depthwise_conv_t e)
            : kind_(post_op::kind_t::conv), depthwise_conv_(e) {}
        entry_t(post_op::binary_t e)
            : kind_(post_op::kind_t::binary), binary_(e) {}

        ~entry_t() {
            switch (kind_) {
                case (post_op::kind_t::sum): sum_.~sum_t(); break;
                case (post_op::kind_t::eltwise): eltwise_.~eltwise_t(); break;
                case (post_op::kind_t::conv):
                    depthwise_conv_.~depthwise_conv_t();
                    break;
                case (post_op::kind_t::binary): binary_.~binary_t(); break;
                default: gpu_error_not_expected();
            }
        }

        entry_t(const entry_t &other) = default;
        entry_t &operator=(const entry_t &) = default;

        post_op::kind_t kind() const { return kind_; }

        // Only const ref accessors are allowed to ensure specializations are
        // not put in an inconsistent state.
        bool is_sum() const { return kind_ == post_op::kind_t::sum; }
        const post_op::sum_t &as_sum() const {
            gpu_assert(is_sum());
            return sum_;
        }

        bool is_eltwise() const { return kind_ == post_op::kind_t::eltwise; }
        const post_op::eltwise_t &as_eltwise() const {
            gpu_assert(is_eltwise());
            return eltwise_;
        }

        bool is_depthwise_conv() const {
            return kind_ == post_op::kind_t::conv;
        }
        const post_op::depthwise_conv_t &as_depthwise_conv() const {
            gpu_assert(is_depthwise_conv());
            return depthwise_conv_;
        }

        bool is_binary() const { return kind_ == post_op::kind_t::binary; }
        const post_op::binary_t &as_binary() const {
            gpu_assert(is_binary());
            return binary_;
        }

        void set_scale(float scale) {
            switch (kind_) {
                case (post_op::kind_t::sum):
                    sum_.inline_scale = true;
                    sum_.scale = scale;
                    break;
                case (post_op::kind_t::eltwise):
                    sum_.inline_scale = true;
                    eltwise_.scale = scale;
                    break;
                default: gpu_error_not_expected(); break;
            }
        }

#if __cplusplus >= 202002L
        bool operator==(const entry_t &other) const {
            if (kind_ != other.kind_) return false;
            switch (kind_) {
                case (post_op::kind_t::sum): return sum_ == other.sum_;
                case (post_op::kind_t::eltwise):
                    return eltwise_ == other.eltwise_;
                case (post_op::kind_t::conv):
                    return depthwise_conv_ == other.depthwise_conv_;
                case (post_op::kind_t::binary): return binary_ == other.binary_;
                case (post_op::kind_t::undef): return true;
            }
            gpu_error_not_expected();
            return false;
        }
#endif

        void serialize(serialization_stream_t &s) const {
            s.append(kind_);
            switch (kind_) {
                case (post_op::kind_t::sum): s.append(sum_); break;
                case (post_op::kind_t::eltwise): s.append(eltwise_); break;
                case (post_op::kind_t::conv): s.append(depthwise_conv_); break;
                case (post_op::kind_t::binary): s.append(binary_); break;
                default: gpu_error_not_expected(); break;
            }
        }

        static entry_t deserialize(deserializer_t &d) {
            auto kind = d.pop<post_op::kind_t>();
            switch (kind) {
                case (post_op::kind_t::sum): return d.pop<post_op::sum_t>();
                case (post_op::kind_t::eltwise):
                    return d.pop<post_op::eltwise_t>();
                case (post_op::kind_t::conv):
                    return d.pop<post_op::depthwise_conv_t>();
                case (post_op::kind_t::binary):
                    return d.pop<post_op::binary_t>();
                default: gpu_error_not_expected(); return entry_t();
            }
        }

    private:
        post_op::kind_t kind_;
        union {
            post_op::sum_t sum_;
            post_op::eltwise_t eltwise_;
            post_op::depthwise_conv_t depthwise_conv_;
            post_op::binary_t binary_;
        };
    };

    static_assert(sizeof(entry_t) < 64,
            "Avoid unnecessary growth of this structure to limit the size of "
            "gpu_post_ops");

    // Enable container operations on std::vector
    using container_type = std::vector<entry_t>;
    using iterator = container_type::iterator;
    using reverse_iterator = container_type::reverse_iterator;
    using const_iterator = container_type::const_iterator;
    using const_reverse_iterator = container_type::const_reverse_iterator;

    iterator begin() { return ops_.begin(); }
    const_iterator begin() const { return ops_.begin(); }
    reverse_iterator rbegin() { return ops_.rbegin(); }
    const_reverse_iterator rbegin() const { return ops_.rbegin(); }
    iterator end() { return ops_.end(); }
    const_iterator end() const { return ops_.end(); }
    reverse_iterator rend() { return ops_.rend(); }
    const_reverse_iterator rend() const { return ops_.rend(); }
    bool empty() const { return ops_.empty(); }
    const entry_t &back() const { return ops_.back(); }
    entry_t &back() { return ops_.back(); }
    const entry_t &operator[](size_t idx) const { return ops_[idx]; }
    entry_t &operator[](size_t idx) { return ops_[idx]; }
    void pop_back() { return ops_.pop_back(); }

    void serialize(serialization_stream_t &s) const { s.append(ops_); }

    static gpu_post_ops_t deserialize(deserializer_t &d) {
        gpu_post_ops_t po;
        d.pop(po.ops_);
        return po;
    }

#if __cplusplus >= 202002L
    bool operator==(const gpu_post_ops_t &) const = default;
#else
    bool operator==(const gpu_post_ops_t &other) const {
        return serialization_stream_t(*this) == serialization_stream_t(other);
    }
#endif

    size_t len() const { return ops_.size(); }

private:
    std::vector<entry_t> ops_;
};

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
