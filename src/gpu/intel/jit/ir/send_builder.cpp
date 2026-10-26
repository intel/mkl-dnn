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

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "gpu/intel/jit/ir/send_builder.hpp"

#include "gpu/intel/jit/ir/block_2d_utils.hpp"
#include "gpu/intel/jit/ir/legacy.hpp"
#include "gpu/intel/jit/ir/send_plan.hpp"
#include "gpu/intel/logging.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Helper class to iterate through global memory offsets, provides queries to
// check whether given blocks are dense, properly aligned, etc.
class memory_walker_t {
public:
    memory_walker_t(const constraint_set_t &cset, const view_t &view)
        : view_(view)
        , mask_tensor_(view.create_mask_tensor(cset).reinterpret(view.type()))
        , full_size_(into<int>(
                  utils::div_up(view.velems() * view.type().bitsize(), 8))) {
        init_dense_blocks(cset);
        reset();
    }

    void reset() {
        cur_off_ = 0;
        remaining_size_ = full_size_;
    }

    const mask_tensor_t &mask_tensor() const { return mask_tensor_; }

    bool has_next() const { return cur_off_ < full_size_; }

    int remaining_size() const { return remaining_size_; }

    int remaining_elems() const {
        return remaining_size_ * type().packing() / type().size();
    }

    bool is_dense_and_aligned(int off, int size, int alignment) const {
        if (off + size > remaining_size_) return false;
        if (size == 0) return true;
        int beg = cur_off_ + off;
        int end = cur_off_ + off + size;
        if (get_block_index(beg) != get_block_index(end - 1)) return false;
        if (alignment != 0 && get_alignment(beg) < alignment) return false;
        return true;
    }

    // Returns true if each of the given slot regions is dense and aligned.
    bool check_region(int off, int slots, int slot_size, int alignment) const {
        for (int i = 0; i < slots; i++) {
            int off = i * slot_size;
            // Overflow is fine, expect it to be handled by proper masking.
            if (off >= remaining_size_) return true;
            if ((slot_size * slots * type().packing()) % type().size() != 0)
                return false;
            if (!is_dense_and_aligned(off, slot_size, alignment)) return false;
        }
        return true;
    }

    // Returns true if the given region can be masked with `mask_size`
    // granularity and `nmasks` number of masks.
    bool check_mask_size(int off, int size, int mask_size, int nmasks) const {
        auto mask = get_mask(off, size, mask_size, nmasks, /*allow_fail=*/true);
        return bool(mask);
    }

    expr_t get_offset(int off, expr_t &base, int &off_const) const {
        if (off >= remaining_size_) {
            base = expr_t(0);
            off_const = 0;
            return base;
        }
        int block_idx = get_block_index(cur_off_ + off);
        gpu_assert(block_idx >= 0 && block_idx < int(block_offs_.size()));
        base = block_offs_[block_idx];
        auto prev_base = block_offs_[block_idx == 0 ? 0 : block_idx - 1];
        auto get_const_summand = [&](const expr_t &expr) -> int {
            if (!expr.type().is_int()) return 0;
            auto binary_op = expr.as_ptr<binary_op_t>();
            if (binary_op && binary_op->op_kind == op_kind_t::_add
                    && is_const(binary_op->b))
                return to_cpp<int>(binary_op->b);
            return 0;
        };

        auto const_summand = get_const_summand(base);
        auto base1 = simplify(base - const_summand);
        auto base2 = simplify(prev_base - get_const_summand(prev_base));
        bool same_base = base1.is_equal(base2);
        off_const = (cur_off_ + off) % dense_block_size_;
        if (!same_base || const_summand == 0) return base + off_const;
        base = base1;
        off_const += const_summand;
        return base + off_const;
    }

    // Returns a boolean mask expression for the given region to access.
    expr_t get_mask(int off, int size, int mask_size, int nmasks,
            bool allow_fail = false) const {
        gpu_assert(size % mask_size == 0) << "Incompatible mask size.";
        auto sub_mask_tensor = create_sub_mask_tensor(off, size);
        sub_mask_tensor
                = sub_mask_tensor.reinterpret(dsl::type_t::u8(mask_size));
        if (sub_mask_tensor.is_empty()) {
            if (allow_fail) return expr_t();
            gpu_error_not_expected();
        }
        auto ret = sub_mask_tensor.to_expr(nmasks);
        if (ret.is_empty()) {
            if (allow_fail) return expr_t();
            gpu_error_not_expected() << "Can't create mask.";
        }
        return ret;
    }

    // Moves the current position `size` bytes ahead.
    void advance(int size) {
        gpu_assert((size * type().packing()) % type().size() == 0);
        size = std::min(size, remaining_size_);
        cur_off_ += size;
        remaining_size_ -= size;
    }

private:
    const dsl::type_t &type() const { return view_.type(); }

    void init_dense_blocks(const constraint_set_t &cset) {
        auto l = view_.create_pseudo_vlayout();
        // Find the maximum innermost dense tile.
        stride_t stride = 1;
        tile_t tile;
        for (auto &b : l.blocks()) {
            if (b.stride != stride) break;
            tile[b.idx] *= b.size;
            stride = b.size * b.stride;
        }
        dense_block_size_
                = into<int>(tile.elems() * type().size() / type().packing());
        // Split the memory view into dense blocks and precompute block offsets
        // and alignments.
        view_.for_each_tile(tile, [&](const icoord_t &start) {
            auto off = view_.offset_bytes(start);
            off = simplify(off, cset);

            const int base_alignment = 128;
            int f = into<int>(get_max_const_factor(off, cset));
            int alignment = f ? ir_utils::max_pow2_divisor(f) : base_alignment;

            block_offs_.push_back(std::move(off));
            block_alignments_.push_back(alignment);
        });
    }

    mask_tensor_t create_sub_mask_tensor(int off, int size) const {
        gpu_assert((off * type().packing()) % type().size() == 0);
        gpu_assert((size * type().packing()) % type().size() == 0);

        std::vector<dim_t> sub_dims
                = {(dim_t)size * type().packing() / type().size()};
        layout_t sub_layout(type(), sub_dims);
        mask_tensor_t sub_mask_tensor(sub_layout);
        int beg = (cur_off_ + off) * type().packing() / type().size();
        int end = (cur_off_ + off + size) * type().packing() / type().size();
        for (int i = beg; i < end; i++) {
            auto mask = (i < mask_tensor_.elems()) ? mask_tensor_.mask(i)
                                                   : expr_t(false);
            sub_mask_tensor.set_mask(i - beg, mask);
        }
        return sub_mask_tensor;
    }

    int get_block_index(int off) const { return off / dense_block_size_; }

    int get_alignment(int off) const {
        int block_alignment = block_alignments_[off / dense_block_size_];
        return ir_utils::max_pow2_divisor(
                block_alignment + off % dense_block_size_);
    }

    view_t view_;
    mask_tensor_t mask_tensor_;
    std::vector<expr_t> block_offs_;
    std::vector<int> block_alignments_;
    int cur_off_ = 0;
    int full_size_ = 0;
    int remaining_size_ = 0;
    int dense_block_size_ = 0;
};

class layout_walker_t {
public:
    layout_walker_t() = default;
    layout_walker_t(const layout_t &layout, int grf_size)
        : layout_(layout), grf_size_(grf_size), idxs_(layout.blocks().size()) {}

    int offset_bytes() const { return off_bytes_; }

    bool can_access(int size) const {
        int off = off_bytes_ + size;
        return off <= max_offset_bytes();
    }

    // Returns true if the next `elems` elements can be stored in the layout
    // given the following requirements:
    // - They must be uniformly strided with `stride` (specified in elements)
    // - The last element must be GRF boundary aligned (unless `is_last_region`
    //   is true)
    // - The last element must not cross the layout boundary
    bool can_advance(int stride, int elems, bool is_last_region = false) {
        if (is_last_region) elems = std::min(elems, remaining_elems());
        const auto stride_bytes
                = utils::div_up(stride * type().size(), type().packing());
        auto cur_idxs = idxs_;
        int cur_off_bytes = off_bytes_;
        for (int i = 0; i < elems - 1; i++) {
            int next_off_bytes = advance(cur_idxs, cur_off_bytes);
            if (next_off_bytes - cur_off_bytes != stride_bytes) return false;
            cur_off_bytes = next_off_bytes;
        }
        cur_off_bytes = advance(cur_idxs, cur_off_bytes);
        if (cur_off_bytes > max_offset_bytes()) return false;
        if (!is_last_region && cur_off_bytes % grf_size_ != 0) return false;
        return true;
    }

    // Moves the current position `elems` elements ahead.
    void advance(int elems) {
        elems = std::min(elems, remaining_elems());
        for (int i = 0; i < elems; i++) {
            off_bytes_ = advance(idxs_, off_bytes_);
            elems_++;
        }
    }

private:
    const dsl::type_t &type() const { return layout_.type(); }

    int max_offset_bytes() const { return (int)size_bytes(layout_, grf_size_); }

    int remaining_elems() const { return into<int>(layout_.elems() - elems_); }

    int advance(std::vector<int> &idxs, int off_bytes) const {
        for (size_t i = 0; i < idxs.size(); i++) {
            if (++idxs[i] < layout_[i].size) break;
            idxs[i] = 0;
        }
        int off = 0;
        for (size_t i = 0; i < idxs.size(); i++) {
            int stride = (int)layout_[i].stride;
            off += idxs[i] * stride;
        }
        return utils::div_up(off * type().size(), type().packing());
    }

    layout_t layout_;
    int grf_size_;

    std::vector<int> idxs_;
    int elems_ = 0;
    int off_bytes_ = 0;
};

access_builder_t::access_builder_t(ir_context_t &ir_ctx, const view_t &mem_view,
        const expr_t &mem_buf, const expr_t &reg_buf, send_op_t send_op,
        send_address_t send_address, send_cache_hint_t send_cache_hint,
        send_params_t &send_params, bool zero_out)
    : ir_ctx_(&ir_ctx)
    , mem_view_(mem_view)
    , mem_buf_(mem_buf)
    , reg_buf_(reg_buf)
    , send_op_(send_op)
    , send_address_(send_address)
    , send_cache_hint_(send_cache_hint)
    , mem_type_(mem_view.type())
    , zero_out_(zero_out) {
    if (send_params.use_send_plan) {
        auto sp = create_send_plan(
                ir_ctx.options(), mem_view, send_params, zero_out);
        if (sp && !sp.is_2d()) send_params.hint_2d = send_2d_hint_t();
        if (!sp) return;
        reg_layout_ = sp.reg_layout();
        reg_buf_size_ = sp.reg_buf_size();
        stmt_ = sp.create_stmt(mem_buf, reg_buf);
        return;
    }
    if (send_params.hint_2d.enable) {
        if (try_build_2d(send_params)) return;
    }
    send_params.hint_2d = send_2d_hint_t();
    build();
}

access_builder_t::access_builder_t(access_builder_t &&) = default;
access_builder_t::~access_builder_t() = default;

void access_builder_t::build() {
    bool ok = false;
    memory_walker_t mem_walker(ir_ctx_->cset(), mem_view_);
    for (auto &l : candidate_payload_layouts()) {
        // Try to find send decomposition with the given GRF payload layout.
        if (try_build(l, mem_walker)) {
            ok = true;
            break;
        }
    }
    if (!ok && send_op_ == send_op_t::prefetch) {
        // Do not treat as an error, skip prefetch messages during generation.
        gpu_warning() << "Can't generate send decomposition for prefetch.";
        return;
    }
    gpu_assert(ok) << "Can't generate send decomposition.";
}

static bool stride_dimension_ok(const view_t &view, size_t stride_tidx,
        size_t stride_vidx, const coord_t &vstart) {
    auto &tdim = view.tdim(stride_tidx);
    auto e = tdim.expr();
    for (dim_idx_t i = 0; i < tdim.nvargs(); i++) {
        dim_idx_t vidx = tdim.vidx(i);
        auto &vvar = view.vvars()[vidx];
        if (vidx == stride_vidx) {
            e = substitute(e, vvar, expr_t(0));
        } else {
            e = substitute(e, vvar, vstart[vidx]);
        }
    }
    e = simplify(e);
    return e.is(0);
}

static expr_t try_scalarize(const expr_t &e) {
    if (e.type().is_scalar()) return e;

    if (auto *shuffle = e.as_ptr<shuffle_t>()) {
        if (shuffle->is_broadcast()) return try_scalarize(shuffle->vec[0]);
        return expr_t();
    }

    if (auto *binary = e.as_ptr<binary_op_t>()) {
        auto a = try_scalarize(binary->a);
        auto b = try_scalarize(binary->b);
        if (a.is_empty() || b.is_empty()) return expr_t();
        return binary_op_t::make(binary->op_kind, a, b);
    }

    gpu_error_not_expected() << e;
    return expr_t();
}

static stmt_t try_promote_to_lsc(const stmt_t &_call) {
    if (_call.is_empty()) return _call;
    auto &call = _call.as<func_call_t>();
    auto &send = call.func.as<send_t>();
    if (send.is_lsc || send.is_2d()) return call;
    if (send.hw < ngen::HW::XeHPG) return call;
    if (send.is_slm()) return call;
    if (!send.is_block()) return call;

    auto mask = try_scalarize(send_t::arg_mask(call));
    if (mask.is_empty()) return call;

    auto new_args = call.args;
    send_t::arg_mask(new_args) = std::move(mask);

    auto lsc_send = send_t::make(send.hw, send.op, send.address, send.type,
            send.slots, /*is_lsc=*/true, send.fill_buf, send.cache_hint);
    return lsc_send.call(new_args);
}

bool access_builder_t::try_build_2d(send_params_t &send_params) {
    auto vlayout = mem_view_.create_pseudo_vlayout();
    auto &hint = send_params.hint_2d;
    // The data may be loaded in a wider data type to get a proper GRF layout.
    if (!hint.type.is_undef()) vlayout = ir::reinterpret(vlayout, hint.type);

    bool is_store = (send_op_ == send_op_t::store);
    bool is_prefetch = (send_op_ == send_op_t::prefetch);
    auto send_type = dsl::type_t::u(vlayout.type().size() * 8);
    auto blocks = vlayout.blocks();
    if (blocks.size() < 2) return false;

    auto &b0 = blocks[0];
    auto &b1 = blocks[1];
    gpu_assert(b0.idx != b1.idx);
    if (b0.stride != stride_t(1)) return false;
    if (!b1.stride.is_fixed()) return false;

    auto get_tdim_idx = [&](size_t vdim_idx, int &stride) {
        size_t ret = dim_idx::invalid;
        for (dim_idx_t i = 0; i < mem_view_.ntdims(); i++) {
            auto &tdim = mem_view_.tdim(i);
            for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                if (tdim.vidx(j) == vdim_idx) {
                    gpu_assert(ret == dim_idx::invalid);
                    stride = (int)tdim.vstride(j);
                    ret = i;
                }
            }
        }
        return ret;
    };

    int w_tstride = 0;
    int h_tstride = 0;
    size_t w_dim_idx = get_tdim_idx(b0.idx, w_tstride);
    size_t h_dim_idx = get_tdim_idx(b1.idx, h_tstride);

    if (w_tstride != 1) return false;

    auto &tlayout = mem_view_.tlayout();
    auto get_2d_dim = [&](size_t tidx) {
        return inner_block(tlayout, tidx, /*skip_outer=*/false);
    };

    int surface_width = 0;
    int surface_height = 0;
    int surface_pitch = int(b1.stride);
    bool is_w_blocked = (get_2d_dim(w_dim_idx) != tlayout.elems(w_dim_idx));
    bool is_h_blocked = (get_2d_dim(h_dim_idx) != tlayout.elems(h_dim_idx));
    // Virtual surface means loading from the innermost block of a block layout
    // which implies no bound checks embedded into 2D block message.
    bool use_virtual_surface = is_w_blocked || is_h_blocked;
    if (use_virtual_surface) {
        if (h_tstride != 1) return false;
        surface_width = into<int>(b0.size);
        surface_height = into<int>(b1.size);
    } else {
        surface_width = into<int>(tlayout.elems(w_dim_idx));
        surface_height = into<int>(tlayout.elems(h_dim_idx));
        if (surface_height % h_tstride != 0) return false;
        surface_height = surface_height / h_tstride;
    }
    int type_factor = ir_utils::safe_divide(send_type.size(), mem_type_.size());
    surface_width /= type_factor;

    int width = hint.width;
    int height = hint.height;
    int count = 1;
    bool vnni = hint.vnni;
    bool transpose = hint.transpose;

    // Try to reduce the number of messages by increasing count per message.
    // Select the highest valid count.
    std::vector<int> counts = block_2d_counts(ir_ctx_->hw(), is_prefetch,
            is_store, transpose, width, mem_type_.size());
    for (auto i : counts)
        if (b0.size % (i * width) == 0) {
            count = i;
            break;
        }

    int W = surface_width;
    int H = surface_height;
    int P = surface_pitch;
    int w = width;
    int h = height;
    int c = count;
    if (!fixup_send_2d_params(send_type, vnni, transpose,
                /*use_xy=*/!use_virtual_surface, W, H, P, w, h, c,
                hint.vnni_permute_factor))
        return false;

    tile_t tile;
    tile[b0.idx] = count * width;
    tile[b1.idx] = height;

    reg_layout_ = layout_t(type_factor == 1 ? mem_type_ : send_type,
            std::vector<dim_t>(vlayout.ndims(), 1));
    int h_inner = vnni ? 4 / send_type.size() : 1;
    int h_outer = ir_utils::safe_divide(height, h_inner);
    reg_layout_ = reg_layout_.with_block({b1.idx, h_inner});
    if (transpose) {
        reg_layout_ = reg_layout_.with_block({b1.idx, h_outer});
        reg_layout_ = reg_layout_.with_block({b0.idx, width});
    } else {
        reg_layout_ = reg_layout_.with_block({b0.idx, width});
        reg_layout_ = reg_layout_.with_block({b1.idx, h_outer});
    }
    reg_layout_ = reg_layout_.with_block({b0.idx, count});

    auto maybe_add_outer = [](layout_t &l, const layout_t::block_t &b, int &i) {
        if ((b.size > 1) && (b.size > i))
            l = l.with_block({b.idx, ir_utils::safe_divide(b.size, i)});
        i = utils::div_up(i, b.size);
    };
    int b0_inner = count * width;
    int b1_inner = height;
    for (auto &iter : blocks) {
        if (iter.idx == b0.idx)
            maybe_add_outer(reg_layout_, iter, b0_inner);
        else if (iter.idx == b1.idx)
            maybe_add_outer(reg_layout_, iter, b1_inner);
    }

    if (type_factor != 1) {
        auto blocks = reg_layout_.blocks();
        reg_layout_
                = layout_t(mem_type_, std::vector<dim_t>(vlayout.ndims(), 1));
        reg_layout_ = reg_layout_.with_block({b0.idx, type_factor});
        for (auto &b : blocks)
            reg_layout_ = reg_layout_.with_block({b.idx, b.size});
    }

    for (auto &b : blocks) {
        if (utils::one_of(b.idx, b0.idx, b1.idx)) continue;
        reg_layout_ = reg_layout_.with_block({b.idx, b.size});
    }

    reg_layout_walker_
            = utils::make_unique<layout_walker_t>(reg_layout_, grf_size());

    // Update user hint.
    hint.type = send_type;
    hint.enable = true;
    hint.vnni = vnni;
    hint.transpose = transpose;
    hint.width = w;
    hint.height = h;
    auto _send = send_t::make_2d(ir_ctx_->hw(), send_params.convert(send_op_),
            send_type, W, H, P, w, h, c, vnni, transpose, zero_out_,
            send_cache_hint_);
    auto &send = _send.as<send_t>();

    stmt_ = stmt_t();
    const auto &vstart0 = mem_view_.vstart();
    for (auto &start : vlayout.iter(tile)) {
        int access_size = send.access_size();
        int access_elems = access_size / mem_type_.size();

        // Check mask requirements.
        expr_t mask;
        if (!check_2d_mask(tile, start, use_virtual_surface, w_dim_idx,
                    h_dim_idx, mask)) {
            return false;
        }

        if (!send.is_prefetch_2d()) {
            if (!reg_layout_walker_->can_advance(1, access_elems)) {
                return false;
            }

            if (!reg_layout_walker_->can_access(send.payload_size())) {
                return false;
            }
        }

        auto vstart = vstart0;
        for (dim_idx_t i = 0; i < into<dim_idx_t>(vlayout.ndims()); i++) {
            if (start.get(i) == 0) continue;
            int factor = (i == b0.idx.index() ? type_factor : 1);
            vstart[i] += factor * start[i];
        }
        auto tstart
                = mem_view_.cvt_vargs_to_targs(vstart, /*ignore_vstart=*/true);

        auto &_x = tstart[w_dim_idx];
        auto &_y = tstart[h_dim_idx];

        expr_t x(0);
        expr_t y(0);

        bool skip_send = false;
        if (!use_virtual_surface) {
            std::swap(x, _x);
            std::swap(y, _y);
            if (type_factor != 1) x /= type_factor;

            if (h_tstride != 1) {
                if (!stride_dimension_ok(
                            mem_view_, h_dim_idx, b1.idx, vstart)) {
                    if (send.is_prefetch_2d()) {
                        skip_send = true;
                    } else {
                        break;
                    }
                }
                y /= h_tstride;
            }
        }

        auto off = simplify(
                offset_bytes(mem_view_.tlayout(), tstart), ir_ctx_->cset());

        // Check alignment requirements.
        int64_t align = get_max_const_factor(off, ir_ctx_->cset());
        if (align % block_2d_base_alignment(ir_ctx_->hw()) != 0) {
            return false;
        }

        if (!skip_send) {
            if (!ir_ctx_->cset().can_prove(
                        x % block_2d_x_alignment(send_type.size()) == 0)) {
                break;
            }
            auto reg_buf = (send.is_prefetch_2d()
                            ? expr_t()
                            : reg_buf_ + reg_layout_walker_->offset_bytes());
            auto send_stmt = send(mem_buf_, off, reg_buf, mask, x, y);
            stmt_ = stmt_.append(send_stmt);
        }

        reg_layout_walker_->advance(send.access_size() / mem_type_.size());
    }

    return true;
}

bool access_builder_t::fixup_send_2d_params(const dsl::type_t &send_type,
        bool vnni, bool transpose, bool use_xy, int &W, int &H, int &P, int &w,
        int &h, int &c, int &vnni_permute_factor) {
    int surface_width_size = W * send_type.size();
    auto whp_ok = [&]() {
        return block_2d_width_ok(W, send_type.size()) && block_2d_height_ok(H)
                && block_2d_pitch_ok(
                        ir_ctx_->hw(), P, send_type.size(), use_xy);
    };

    // No VNNI permute by default.
    vnni_permute_factor = 0;

    // Surface width must be >= 64 bytes. For smaller width we can apply
    // reshape, e.g. [16a] x [16b] -> [8a] x [2a16b] to have block with larger
    // width. Such reshape impacts width/height handling with the following
    // implications:
    // - Reshape is applied only for VNNI and no transpose case. This allows to
    //   get the same GRF layout but with permuted height elements:
    //     - Layout without reshape: 8a16b2a
    //     - Layout with    reshape: 8a16b2a (a/height dimension is permuted)
    // - Permutation is safe when it's done for the reduction dimension
    //   (doesn't matter in which order elements are accumulated).
    // - Permutation pattern must be the same between A and B tensors
    if (surface_width_size >= 64) return whp_ok();

    // Reshape is only expected/supported with VNNI.
    if (!vnni || transpose) return false;

    if (64 % surface_width_size != 0) return false;

    int factor = 64 / surface_width_size;
    if (h % factor != 0) return false;

    int max_count = block_2d_counts(ir_ctx_->hw(),
            send_op_ == send_op_t::prefetch, send_op_ == send_op_t::store,
            transpose, w, send_type.size())[0];

    if (factor > max_count) return false;

    vnni_permute_factor = factor;
    W *= factor;
    P *= factor;
    H /= factor;
    h /= factor;
    c = factor;
    return whp_ok();
}

bool access_builder_t::check_2d_mask(const tile_t &tile, const coord_t &coord,
        bool use_virtual_surface, size_t w_dim_idx, size_t h_dim_idx,
        expr_t &mask) const {
    auto sub_view = mem_view_.create_sub_view(tile, coord);
    auto mask_tensor = sub_view.create_mask_tensor(ir_ctx_->cset());
    mask = mask_tensor.to_expr(1);
    if (mask) return true;

    // Virtual surface implies no out-of-bound send checks.
    if (use_virtual_surface) return false;

    // Remove bound conditions that are covered by out-of-bound send checks.
    uint32_t tmask = 0xFFFFFFFF;
    for (dim_idx_t i = 0; i < sub_view.nvdims(); i++) {
        if (!utils::one_of(i, w_dim_idx, h_dim_idx)) continue;
        for (dim_idx_t j = 0; j < sub_view.ntdims(); j++) {
            auto &tdim = sub_view.tdim(j);
            for (dim_idx_t k = 0; k < tdim.nvargs(); k++) {
                if (tdim.vidx(k) == i) {
                    // TODO: Check if tdim mask is a bound mask.
                    tmask &= ~(1U << i);
                }
            }
        }
    }
    mask_tensor = sub_view.create_mask_tensor(ir_ctx_->cset(), tmask);
    mask = mask_tensor.to_expr(1);
    if (mask) return true;

    return false;
}

bool access_builder_t::try_build(
        const layout_t &try_layout, memory_walker_t &mem_walker) {
    auto &try_layout_blocks = try_layout.blocks();
    const int type_size = mem_type_.size();
    const int type_packing = mem_type_.packing();
    int reg_stride
            = (try_layout_blocks.empty() ? 0
                                         : (int)try_layout_blocks[0].stride);
    auto send_list = send_t::get_all(ir_ctx_->hw(), send_op_, send_address_,
            mem_type_, zero_out_, send_cache_hint_);
    reg_layout_walker_
            = utils::make_unique<layout_walker_t>(try_layout, grf_size());
    stmt_ = stmt_t();
    mem_walker.reset();
    // Iterate through the memory view, greedily select messages according to
    // the sorted message list.
    while (mem_walker.has_next()) {
        func_t _send;
        for (auto &_s : send_list) {
            auto &s = _s.as<send_t>();

            int slot_size = s.type.size();
            int alignment = s.alignment();
            int nmasks = s.nmasks();
            int payload_stride = s.payload_type_stride();
            int access_size = s.access_size();
            int access_elems = access_size * type_packing / type_size;
            bool is_last_chunk = mem_walker.remaining_size() <= access_size;

            if (reg_stride != 1 || payload_stride != slot_size) {
                // Detected strided GRF layout or strided payload. In this
                // case require full data type and stride match.
                if (reg_stride != 0
                        && payload_stride * type_packing
                                != reg_stride * type_size)
                    continue;
                if (type_packing * s.type.size() != type_size) continue;
            }
            // Prefetches don't have payload so skip these conditions for
            // prefetch.
            if (!s.is_prefetch()) {
                if (!reg_layout_walker_->can_advance(
                            reg_stride, access_elems, is_last_chunk))
                    continue;

                if (!reg_layout_walker_->can_access(s.payload_size())) continue;
            }

            // Check if slots are contiguous and aligned.
            if (!mem_walker.check_region(0, s.slots, slot_size, alignment))
                continue;

            // Check mask requirements.
            // XXX: Postpone mask check for prefetch until during send call
            // generation. If the mask cannot be generated, skip the prefetch.
            if (!s.is_prefetch()
                    && !mem_walker.check_mask_size(
                            0, access_size, s.mask_size(), nmasks))
                continue;

            _send = _s;
            break;
        }
        // Can't find a message - try another GRF layout for payload.
        if (_send.is_empty()) return false;

        auto &send = _send.as<send_t>();
        auto send_stmt = create_send_stmt(send, mem_walker);
        send_stmt = try_promote_to_lsc(send_stmt);
        stmt_ = stmt_.append(send_stmt);

        reg_layout_walker_->advance(
                send.access_size() * type_packing / type_size);
        mem_walker.advance(send.access_size());
    }
    reg_layout_ = try_layout;
    return true;
}

std::vector<layout_t> access_builder_t::candidate_payload_layouts() const {
    int type_size = mem_type_.size();
    auto vlayout = mem_view_.create_dense_vlayout();

    std::vector<layout_t> ret;

    // Dense payload layout directly mapping to the memory view.
    ret.push_back(vlayout);

    // These payload layouts are to match payload for byte x {1,2} scattered
    // messages (they are dword-strided).
    if (type_size == 2) ret.push_back(make_strided(vlayout, 2));
    if (type_size == 1 && mem_type_.bitsize() == 8)
        ret.push_back(make_strided(vlayout, 4));

    return ret;
}

stmt_t access_builder_t::create_send_stmt(
        const send_t &send, const memory_walker_t &mem_walker) {
    std::vector<expr_t> off_vec;
    // Try to detect a common base and const vector offset to reduce further
    // arithmetic.
    expr_t off_base0;
    int off_const0 = -1;
    bool is_same_base = true;
    std::vector<expr_t> off_const_vec;
    for (int i = 0; i < send.slots; i++) {
        expr_t off_base;
        int off_const;
        auto off = mem_walker.get_offset(
                i * send.type.size(), off_base, off_const);
        if (off_base0.is_empty()) {
            off_base0 = std::move(off_base);
            off_const0 = off_const;
        } else if (!off_base.is_equal(off_base0)) {
            is_same_base = false;
        }
        off_vec.push_back(std::move(off));
        off_const_vec.emplace_back(off_const - off_const0);
    }
    expr_t off;
    if (send.slots == 1 || !is_same_base) {
        off = shuffle_t::make(off_vec);
    } else {
        off = shuffle_t::make_broadcast(off_base0, send.slots)
                + shuffle_t::make_broadcast(off_const0, send.slots)
                + shuffle_t::make(off_const_vec);
    }
    bool allow_fail = send.is_prefetch();
    auto _mask = mem_walker.get_mask(
            0, send.access_size(), send.mask_size(), send.nmasks(), allow_fail);
    if (_mask.is_empty()) return stmt_t();

    auto _reg_buf = (send.is_prefetch()
                    ? expr_t()
                    : reg_buf_ + reg_layout_walker_->offset_bytes());
    auto ret = send(mem_buf_, off, _reg_buf, _mask);
    return ret;
}

static const int any_block = 0;

send_2d_hint_t get_send_2d_hint(send_op_t send_op, const dsl::type_t &type,
        bool vnni, bool transpose, int w_tile, int h_tile,
        int w_blk = any_block, int h_blk = any_block) {
    gpu_assert(!(vnni && transpose)) << "VNNI with transpose is not supported.";

    // Disable sub-byte types for now.
    if (type.packing() > 1) return send_2d_hint_t();

    // XXX: Convert transpose to VNNI when transpose is not
    // supported. This will require additional reorder but
    // reorder from "partially transposed" VNNI transformed
    // layout is cheaper.
    if (transpose && type.size() != 4) {
        vnni = true;
        transpose = false;
    }

    bool is_load_or_prefetch
            = utils::one_of(send_op, send_op_t::load, send_op_t::prefetch);
    bool is_store = (send_op == send_op_t::store);

    // Only D8, D16 and D32 are implemented.
    if (!utils::one_of(type.size(), 1, 2, 4)) return send_2d_hint_t();

    // VNNI and transpose are mutually exclusive.
    if (vnni && transpose) return send_2d_hint_t();

    // VNNI and transpose are supported with load only.
    if (is_store && (vnni || transpose)) return send_2d_hint_t();

    // VNNI is supported with D8 and D16 only.
    if (vnni && !utils::one_of(type.size(), 1, 2)) return send_2d_hint_t();

    // Transpose is supported with D32 only.
    if (transpose && type.size() != 4) return send_2d_hint_t();

    int w_min = (transpose ? 1 : 4 / type.size());
    int w_max = (transpose ? 8 : (vnni ? 16 : 64 / type.size()));
    int h_min = (vnni ? (4 / type.size()) : 1);
    int h_max = (is_load_or_prefetch ? 32 : 8);

    if (w_blk != any_block && (w_blk < w_min || w_blk > w_max))
        return send_2d_hint_t();
    if (h_blk != any_block && (h_blk < h_min || h_blk > h_max))
        return send_2d_hint_t();

    auto find_block = [&](int dim, int min, int max) {
        for (int b = max; b >= min; b--) {
            if (dim % b == 0) return b;
        }
        return -1;
    };

    if (w_blk == any_block) w_blk = find_block(w_tile, w_min, w_max);
    if (h_blk == any_block) h_blk = find_block(h_tile, h_min, h_max);
    if (w_blk == -1 || h_blk == -1) return send_2d_hint_t();

    if (vnni) {
        // TODO: Remove.
        gpu_assert(h_blk > 0);
        h_blk = find_block(h_tile, h_blk, h_max);
    }
    if (transpose && w_blk > 0) {
        // TODO: Remove.
        gpu_assert(w_blk > 0);
        w_blk = find_block(w_tile, w_blk, w_max);
    }

    send_2d_hint_t hint;
    hint.type = type;
    hint.enable = true;
    hint.width = w_blk;
    hint.height = h_blk;
    hint.vnni = vnni;
    hint.transpose = transpose;
    return hint;
}

bool send_2d_params_ok(
        const dsl::kernel::options_t &options, send_op_t send_op) {
    if (options.hw() < ngen::HW::XeHPC) return false;
    if (!utils::one_of(send_op, send_op_t::load, send_op_t::prefetch,
                send_op_t::store))
        return false;
    return true;
}

bool send_2d_vlayout_ok(const layout_t &vlayout) {
    const auto &blocks = vlayout.blocks();
    if (blocks.size() < 2) return false;

    const auto &b0 = blocks[0];
    const auto &b1 = blocks[1];
    if (b0.idx == b1.idx) return false;
    if (b0.stride != stride_t(1)) return false;
    if (b1.stride.is_unknown()) return false;
    return true;
}

send_2d_hint_t get_send_2d_hint(const dsl::kernel::options_t &options,
        send_op_t send_op, const view_t &view, bool allow_2d,
        bool use_send_plan) {
    send_2d_hint_t hint;
    if (!allow_2d || !send_2d_params_ok(options, send_op)) return hint;
    auto vlayout = view.create_pseudo_vlayout();
    if (!send_2d_vlayout_ok(vlayout)) return hint;
    auto blocks = vlayout.blocks();
    auto &b0 = blocks[0];
    auto &b1 = blocks[1];

    if (b0.size >= 128) return hint;
    return get_send_2d_hint(send_op, view.type(), false, false,
            into<int>(b0.size), into<int>(b1.size));
}

send_2d_hint_t get_send_2d_hint(const dsl::kernel::options_t &options,
        send_op_t send_op, fma_kind_t fma_kind, abc_kind_t abc_kind,
        const view_t &view, const gemm_schedule_t &gemm_schedule, bool allow_2d,
        bool use_send_plan) {
    send_2d_hint_t hint;
    if (!allow_2d || !send_2d_params_ok(options, send_op)) return hint;
    auto vlayout = view.create_pseudo_vlayout();
    if (!send_2d_vlayout_ok(vlayout)) return hint;
    auto blocks = vlayout.blocks();
    auto &b0 = blocks[0];
    auto &b1 = blocks[1];

    if (send_op == send_op_t::load && fma_kind == fma_kind_t::dpas
            && utils::one_of(abc_kind, abc_kind_t::a, abc_kind_t::b)) {
        // Handle 4 cases (consider bf16):
        // src1, MxK: 16a16b -> 8a16b2a           (VNNI)
        // src1, KxM: 16a16b -> 16b16a -> 8b16a2b (transpose + VNNI)
        // src2, KxN: 16a16b -> 16b16a            (transpose)
        // src2, NxK: 16a16b -> 16a16b            ()
        bool is_dpas_src1 = (abc_kind == abc_kind_t::b);
        int m_blk = options.simd();
        int n_blk = any_block;
        int mn_blk = (is_dpas_src1 ? m_blk : n_blk);
        int k_blk = 32 / view.type().size();
        auto &bmnk_mapper = gemm_schedule.bmnk_mapper();
        bool is_b0_k
                = (bmnk_mapper.bmnk_kind(abc_kind, b0.idx) == bmnk_kind_t::k);
        bool transpose = (is_dpas_src1 == is_b0_k);
        int b0_blk = is_b0_k ? k_blk : mn_blk;
        int b1_blk = !is_b0_k ? k_blk : mn_blk;
        if (b0_blk != any_block && b0.size % b0_blk != 0) return hint;
        if (b1_blk != any_block && b1.size % b1_blk != 0) return hint;
        bool vnni = is_dpas_src1 && !transpose;
        hint = get_send_2d_hint(send_op, view.type(), vnni, transpose,
                into<int>(b0.size), into<int>(b1.size), b0_blk, b1_blk);
    } else {
        if (b0.size >= 128) return hint;
        hint = get_send_2d_hint(send_op, view.type(), false, false,
                into<int>(b0.size), into<int>(b1.size));
    }

    // XXX: Special VNNI permute hint to use with Xa16b:bf16 layout which can't
    // be loaded as is due to 2D send width limitations.
    // Surface width must be >= 64 bytes. For smaller width we can apply
    // reshape, e.g. [16a] x [16b] -> [8a] x [2a16b] to have block with larger
    // width. Such reshape impacts width/height handling with the following
    // implications:
    // - Reshape is applied only for VNNI and no transpose case. This allows to
    //   get the same GRF layout but with permuted height elements:
    //     - Layout without reshape: 8a16b2a
    //     - Layout with    reshape: 8a16b2a (a/height dimension is permuted)
    // - Permutation is safe when it's done for the reduction dimension
    //   (doesn't matter in which order elements are accumulated).
    // - Permutation pattern must be the same between A and B tensors
    if (use_send_plan && send_op == send_op_t::load && hint.vnni
            && !hint.transpose && view.type().size() == 2
            && utils::one_of(abc_kind, abc_kind_t::a, abc_kind_t::b)
            && b0.size == 16 && (dim_t)b1.stride == 16
            && utils::one_of(b1.size, 8, 16, 32)) {
        hint.vnni_permute_factor = 2;
    }

    return hint;
}

send_params_t get_send_params(const dsl::kernel::options_t &options,
        send_op_t send_op, send_address_t send_address, const view_t &view,
        send_cache_hint_t cache_hint, fma_kind_t fma_kind, abc_kind_t abc_kind,
        bool allow_2d) {
    send_params_t params;
    params.hw = options.hw();
    params.mem_type = view.type();
    params.send_op = send_op;
    params.send_address = send_address;
    params.use_send_plan = can_use_send_plan(view);
    params.cache_hint = cache_hint;
    params.prefer_dense
            = (fma_kind == fma_kind_t::dpas && abc_kind == abc_kind_t::a);
    params.hint_2d = get_send_2d_hint(
            options, send_op, view, allow_2d, params.use_send_plan);
    return params;
}

send_params_t get_send_params(const dsl::kernel::options_t &options,
        send_op_t send_op, send_address_t send_address, fma_kind_t fma_kind,
        abc_kind_t abc_kind, const view_t &view,
        const gemm_schedule_t &gemm_schedule, bool allow_2d) {
    auto params = get_send_params(options, send_op, send_address, view,
            send_cache_hint_t::undef, fma_kind, abc_kind, false);
    params.hint_2d = get_send_2d_hint(options, send_op, fma_kind, abc_kind,
            view, gemm_schedule, allow_2d, params.use_send_plan);
    return params;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif
