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

#include "gpu/intel/jit/ir/send_plan.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "common/utils.hpp"
#include "gemmstone/../../dsl/ir/pass/simplify.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"
#include "gpu/intel/jit/ir/legacy.hpp"
#include "gpu/intel/jit/ir/send_builder.hpp"
#include "gpu/intel/logging.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class send_plan_impl_t {
public:
    virtual ~send_plan_impl_t() = default;
    virtual const send_params_t &send_params() const = 0;
    virtual bool is_2d() const = 0;
    virtual bool is_scattered() const = 0;
    virtual const layout_t &reg_layout() const = 0;
    virtual int reg_buf_size() const = 0;
    virtual bool can_split(int factor) const = 0;
    virtual void set_split(int factor) = 0;
    virtual int split_factor() const = 0;
    virtual int estimate_regs(
            bool with_buffer, bool with_headers, bool reuse_headers) const
            = 0;
    virtual std::string str(const std::string &tag) const = 0;

    int estimate_regs(bool with_buffer, bool with_headers) const {
        return estimate_regs(
                with_buffer, with_headers, /*reuse_headers=*/false);
    }
    int estimate_regs(bool with_buffer) const {
        return estimate_regs(with_buffer, /*with_headers=*/true);
    }
    int estimate_regs() const { return estimate_regs(/*with_buffer=*/true); }

    stmt_t create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx, const expr_t &pattern) const {
        const auto &op = send_params().send_op;
        auto stmt = do_create_stmt(mem_buf, reg_buf, subtile_idx, pattern);

        if (!is_scattered()) return stmt;
        const auto &reg = reg_layout();
        const auto &msg = message_layout();
        if (reg.is_equal_normalized(msg)) return stmt;

        if (op == send_op_t::load) {
            auto reorder = create_reorder_stmt(msg, reg, reg_buf, reg_buf);
            return stmt_seq_t::make({stmt, reorder});
        }
        if (op == send_op_t::store) {
            auto reorder = create_reorder_stmt(reg, msg, reg_buf, reg_buf);
            return stmt_seq_t::make({reorder, stmt});
        }
        return stmt;
    }

protected:
    bool try_restride_layout(layout_t &layout) const {
        if (!is_scattered()) return false;
        const auto &op = send_params().send_op;
        if (!utils::one_of(op, send_op_t::load, send_op_t::store)) return false;
        auto &type = layout.type();
        auto blocks = layout.blocks();
        if (type.size() >= 2) return false;
        if (blocks.size() < 2) return false;
        auto &front = blocks[0];
        auto &second = blocks[1];
        if ((dim_t)front.stride > 1) return false;
        if (front.size * type.size() >= 4 * type.packing()) return false;
        if ((dim_t)second.stride * type.size() < 4 * type.packing())
            return false;

        front.stride = 4 * type.packing() / (front.size * type.size());
        layout = layout.with(blocks);
        return true;
    }

private:
    virtual const layout_t &message_layout() const { return reg_layout(); }
    virtual stmt_t do_create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx, const expr_t &pattern) const
            = 0;
};

send_op_t to_2d(send_op_t op) {
    switch (op) {
        case send_op_t::prefetch: return send_op_t::prefetch_2d;
        case send_op_t::load: return send_op_t::load_2d;
        case send_op_t::store: return send_op_t::store_2d;
        case send_op_t::prefetch_2d:
        case send_op_t::load_2d:
        case send_op_t::store_2d: return op;
        default: gpu_error_not_expected();
    }
    return send_op_t::undef;
}

// Offset vector.
class vec_off_t {
public:
    vec_off_t() = default;
    explicit vec_off_t(int64_t value) : vec_({value}) {}
    vec_off_t(size_t n, int64_t value) : vec_(n, value) {}

    void push_back(int64_t value) { vec_.push_back(value); }

    int size() const { return (int)vec_.size(); }

    bool is_empty() const { return vec_.empty(); }

    int64_t &operator[](int i) { return vec_[i]; }
    const int64_t &operator[](int i) const { return vec_[i]; }

    bool operator==(const vec_off_t &other) const {
        return ir_utils::is_equal(vec_, other.vec_);
    }

    bool operator!=(const vec_off_t &other) const { return !operator==(other); }

    bool all_of(int64_t other) const {
        for (int64_t v : vec_)
            if (v != other) return false;
        return true;
    }

    void merge(const vec_off_t &other) {
        vec_.insert(vec_.end(), other.vec_.begin(), other.vec_.end());
    }

    vec_off_t &operator+=(int64_t shift) {
        for (auto &v : vec_)
            v += shift;
        return *this;
    }

    vec_off_t &operator+=(const vec_off_t &shift) {
        if (shift.vec_.size() == 1) return operator+=(shift.vec_[0]);
        gpu_assert(vec_.size() == shift.vec_.size());
        for (int i = 0; i < size(); i++)
            vec_[i] += shift[i];
        return *this;
    }

    std::vector<int64_t>::const_iterator begin() const { return vec_.begin(); }
    std::vector<int64_t>::const_iterator end() const { return vec_.end(); }

    std::string str() const {
        using namespace ir_utils;
        ostringstream_t oss;
        oss << vec_;
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    std::vector<int64_t> vec_;
};

vec_off_t operator+(vec_off_t a, int64_t b) {
    return a += b;
}

// Vector of offset vectors.
class vec_vec_off_t {
public:
    vec_vec_off_t() = default;
    vec_vec_off_t(size_t n, const vec_off_t &value) : vec_(n, value) {}

    void push_back(const vec_off_t &value) { vec_.push_back(value); }

    int size() const { return (int)vec_.size(); }

    bool is_empty() const { return vec_.empty(); }

    vec_off_t &operator[](int i) { return vec_[i]; }
    const vec_off_t &operator[](int i) const { return vec_[i]; }

    void merge(const vec_vec_off_t &other) {
        vec_.insert(vec_.end(), other.vec_.begin(), other.vec_.end());
    }

    vec_off_t slice(int idx) const {
        gpu_assert(!is_empty());
        gpu_assert(idx >= 0 && idx < vec_[0].size());
        vec_off_t ret;
        for (auto &o : vec_) {
            ret.push_back(o[idx]);
        }
        return ret;
    }

    vec_vec_off_t &operator+=(const vec_off_t &other) {
        if (other.is_empty()) return *this;
        for (auto &o : vec_)
            o += other;
        return *this;
    }

    bool operator==(const vec_vec_off_t &other) const {
        return ir_utils::is_equal(vec_, other.vec_);
    }

    bool operator!=(const vec_vec_off_t &other) const {
        return !operator==(other);
    }

    std::string str(const std::string &indent = {}) const {
        using namespace ir_utils;
        ostringstream_t oss;
        oss << indent << vec_;
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    std::vector<vec_off_t> vec_;
};

expr_t to_vec(const expr_t &e, int elems) {
    if (e.type().elems() == elems) return e;
    gpu_assert(e.type().is_scalar());
    return shuffle_t::make_broadcast(e, elems);
}

expr_t to_vec(const vec_off_t &off, int elems) {
    gpu_assert(off.size() == elems);
    if (off.size() == 1) return off[0];
    std::vector<expr_t> e_off;
    e_off.reserve(off.size());
    for (auto &o : off)
        e_off.emplace_back(o);
    return shuffle_t::make(e_off);
}

expr_t add(const expr_t &a, const vec_off_t &b, int elems) {
    auto ret = to_vec(a, elems);
    if (b.all_of(0)) return ret;
    ret += to_vec(b, elems);
    return ret;
}

expr_t add(const expr_t &a, int64_t b, int elems) {
    auto ret = to_vec(a, elems);
    if (b == 0) return ret;
    ret += to_vec(b, elems);
    return ret;
}

expr_t slice(const expr_t &e, int off, int elems) {
    if (e.is_empty()) return expr_t();

    if (is_const(e) || is_var(e)) {
        gpu_assert(off == 0 && elems == 1);
        return e;
    }

    if (auto *binary = e.as_ptr<binary_op_t>()) {
        auto a = slice(binary->a, off, elems);
        auto b = slice(binary->b, off, elems);
        return binary_op_t::make(binary->op_kind, a, b);
    }

    if (auto *shuffle = e.as_ptr<shuffle_t>()) {
        if (shuffle->is_broadcast())
            return shuffle_t::make_broadcast(shuffle->vec[0], elems);
        if (off + elems <= e.type().elems())
            return shuffle_t::make(*shuffle, off, off + elems);
        std::vector<expr_t> vec;
        for (int i = off; i < e.type().elems(); i++)
            vec.push_back(e[i]);
        int rem = elems - (int)vec.size();
        for (int i = 0; i < rem; i++)
            vec.push_back(e.type().is_bool() ? expr_t(false) : expr_t(0));
        return shuffle_t::make(vec);
    }

    gpu_error_not_expected();
    return expr_t();
}

// Modular arithmetic over powers of two. Used to check alignment requirements.
class modulus_t {
public:
    modulus_t() : lg2_(0) {}

    explicit modulus_t(int64_t value) : lg2_(to_lg2(std::abs(value))) {}

    int64_t n() const { return is_zero() ? 0 : (int64_t)1 << lg2_; }

    bool is_divisible(int64_t div) const { return n() % div == 0; }

    bool is_zero() const { return lg2_ == lg2_zero_; }

    modulus_t &set_zero() {
        lg2_ = lg2_zero_;
        return *this;
    }

    modulus_t &operator+=(const modulus_t &b) {
        if (is_zero() && b.is_zero()) return set_zero();
        lg2_ = std::min(lg2_, b.lg2_);
        return *this;
    }

    modulus_t &operator*=(const modulus_t &b) {
        if (is_zero() || b.is_zero()) return set_zero();
        lg2_ += b.lg2_;
        lg2_ = std::min((int)max_lg2_, lg2_);
        return *this;
    }

    modulus_t &operator%=(int64_t b) {
        if (is_zero()) return set_zero();
        auto b_lg2 = to_lg2(b);
        if (math::is_pow2(b) && lg2_ >= b_lg2) return set_zero();
        lg2_ = std::min(lg2_, b_lg2);
        return *this;
    }

    modulus_t &operator/=(int64_t b) {
        if (is_zero()) return set_zero();
        auto b_lg2 = to_lg2(b);
        if (math::is_pow2(b) && lg2_ >= b_lg2) {
            lg2_ -= b_lg2;
            return *this;
        }
        lg2_ = 0;
        return *this;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << "modulus(" << n() << ")";
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    static int to_lg2(int64_t v) {
        gpu_assert(v >= 0);
        if (v == 0) return lg2_zero_;
        return std::min((int)max_lg2_, ngen::utils::bsf(v));
    }

    int lg2_;
    // 2^20 is enough for all alignment restrictions.
    static const int max_lg2_ = 20;
    static const int lg2_zero_ = max_lg2_ + 1;
};

modulus_t operator+(modulus_t a, const modulus_t &b) {
    return a += b;
}
modulus_t operator*(modulus_t a, const modulus_t &b) {
    return a *= b;
}
modulus_t operator*(modulus_t a, int64_t b) {
    return a *= modulus_t(b);
}
modulus_t operator%(modulus_t a, int64_t b) {
    return a %= b;
}

class tdim_info_t {
public:
    tdim_info_t() = default;
    tdim_info_t(size_t tidx, const tdim_t &tdim, const view_t &view,
            int64_t block = 1)
        : tidx_(tidx)
        , size_(view.tlayout().elems(tidx))
        , base_mod_(to_base(tdim, view.vvars()))
        , block_(block)
        , dim_(&tdim) {
        for (dim_idx_t i = 0; i < tdim.nvargs(); i++) {
            vidxs_[i] = tdim.vidx(i);
            vstrides_[i] = int64_t(tdim.vstride(i));
        }
    }

    size_t tidx() const { return tidx_; }

    int64_t size() const { return size_; }

    size_t vidx(int i) const { return vidxs_[i]; }

    dim_t vstride(int i) const { return vstrides_[i]; }

    int64_t block() const { return block_; }

    const modulus_t &base_mod() const { return base_mod_; }

    bool is_identity() const { return dim_->is_identity(); }

    tdim_info_t with_block(int64_t block) const {
        auto ret = *this;
        ret.block_ = block;
        return ret;
    }

    const expr_t &mask() const { return dim_->mask(); }

    bool has_mask() const {
        auto &mask = dim_->mask();
        return mask && !mask.is_equal(expr_t(true));
    }

    bool has_vidx(size_t vidx) const {
        return utils::one_of(vidx, vidxs_[0], vidxs_[1]);
    }

    int64_t vstride_by_vidx(size_t vidx) const {
        for (int i = 0; i < 2; i++) {
            if (vidxs_[i] == vidx) return vstrides_[i];
        }
        gpu_error_not_expected();
        return 0;
    }

    template <typename T,
            typename R = typename std::conditional<std::is_integral<T>::value,
                    int64_t, T>::type>
    R offset(const std::vector<T> &voff, const T &base = T()) const {
        R ret = base;
        for (int i = 0; i < 2; i++) {
            if (vidxs_[i] == dim_idx::invalid) continue;
            ret += voff[vidxs_[i]] * vstrides_[i];
        }
        if (block_ != 1) ret /= block_;
        return ret;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << "tdim(idx = " << tidx_;
        oss << ", size = " << size_;
        oss << ", vidxs = [" << vidxs_[0] << ", " << vidxs_[1] << "]";
        oss << ", vstrides = [" << vstrides_[0] << ", " << vstrides_[1] << "]";
        oss << ", block = " << block_ << ")";
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    static modulus_t to_base(const tdim_t &tdim,
            const std::vector<expr_t> &vvars, const expr_t &e) {
        if (is_const(e)) return modulus_t(to_cpp<int64_t>(e));
        for (dim_idx_t i = 0; i < tdim.nvargs(); i++) {
            if (e.is_same(vvars[tdim.vidx(i)])) return modulus_t(0);
        }
        auto *binary = e.as_ptr<binary_op_t>();
        if (binary) {
            auto a = to_base(tdim, vvars, binary->a);
            auto b = to_base(tdim, vvars, binary->b);
            switch (binary->op_kind) {
                case op_kind_t::_mul: return a * b;
                case op_kind_t::_add: return a + b;
                default: return modulus_t();
            }
        }
        return modulus_t();
    }

    static modulus_t to_base(
            const tdim_t &tdim, const std::vector<expr_t> &vvars) {
        if (tdim.is_identity()) return modulus_t(0);
        return to_base(tdim, vvars, tdim.expr());
    }

    size_t tidx_ = dim_idx::invalid;
    int64_t size_ = 0;
    modulus_t base_mod_;
    size_t vidxs_[2] = {dim_idx::invalid, dim_idx::invalid};
    int64_t vstrides_[2] = {0, 0};
    int64_t block_ = 1;
    const tdim_t *dim_ = nullptr;
};

enum class mask_kind_t {
    undef,
    ab,
    b,
};

class mask_desc_t {
public:
    mask_desc_t(const expr_t &base, const tdim_info_t &tdim, mask_kind_t kind,
            int64_t a, int64_t b)
        : base_(base), tdim_(tdim), kind_(kind), a_(a), b_(b) {
        switch (kind_) {
            case mask_kind_t::ab:
                is_bound_ = (a_ == 0) && (b_ * tdim.block() == tdim.size());
                break;
            case mask_kind_t::b:
                is_bound_ = (b_ * tdim.block() == tdim.size());
                break;
            default: is_bound_ = false; break;
        }
    }

    const tdim_info_t &tdim() const { return tdim_; }

    size_t tidx() const { return tdim_.tidx(); }

    bool is_bound() const { return is_bound_; }

    void set_base(const expr_t &base) {
        base_ = base;
        dim_t factor = 1;
        if (tdim_.vidx(1) == dim_idx::invalid) {
            factor = get_max_const_factor(base_, constraint_set_t());
            factor = math::gcd(factor, a_ * tdim_.block());
            factor = math::gcd(factor, b_ * tdim_.block());
            if (factor % tdim_.block() != 0)
                factor = math::gcd(factor, tdim_.block());
        }
        if (factor != tdim_.block()) {
            a_ = a_ * tdim_.block() / factor;
            b_ = b_ * tdim_.block() / factor;
            tdim_ = tdim_.with_block(factor);
        }

        if (tdim_.block() != 1) base_ /= tdim_.block();
    }

    bool is_const_base() const { return is_const(base_); }

    bool const_fold(int64_t inc) const {
        gpu_assert(is_const_base());
        int64_t base_const = to_cpp<int64_t>(base_);
        int64_t v = base_const + inc;
        switch (kind_) {
            case mask_kind_t::ab: return (a_ <= v) && (v < b_);
            case mask_kind_t::b: return (v < b_);
            default: gpu_error_not_expected();
        }
        return false;
    }

    expr_t to_expr(const vec_off_t &off) const {
        int slots = off.size();
        if (slots == 0) return expr_t();
        auto x = add(base_, off, slots);
        switch (kind_) {
            case mask_kind_t::ab:
                return (x >= to_vec(a_, slots)) & (x < to_vec(b_, slots));
            case mask_kind_t::b: return (x < to_vec(b_, slots));
            default: gpu_error_not_expected();
        }
        return expr_t();
    }

    std::string str(const std::string &indent = {}) const {
        ostringstream_t oss;
        oss << indent << "mask#" << tidx() << std::endl;
        oss << indent << "  " << "base = " << base_ << std::endl;
        oss << indent << "  " << "block = " << tdim_.block() << std::endl;
        switch (kind_) {
            case mask_kind_t::ab:
                oss << indent << "  " << a_ << " <= x < " << b_;
                break;
            case mask_kind_t::b: oss << indent << "  " << "x < " << b_; break;
            default: gpu_error_not_expected();
        }
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    // Mask in the general form:
    //   x = (base + S_i * v_i + S_j * v_j) / block
    //   a <= x < b
    expr_t base_;
    tdim_info_t tdim_;
    mask_kind_t kind_ = mask_kind_t::undef;
    int64_t a_ = 0;
    int64_t b_ = 0;
    bool is_bound_ = false;
};

bool has_vidx_mask(const std::vector<mask_desc_t> &mask_descs, size_t idx,
        dim_t dim, dim_t block, dim_t &factor) {
    factor = 1;
    for (auto &md : mask_descs) {
        auto &tdim = md.tdim();
        if (!tdim.has_vidx(idx)) continue;
        if (tdim.vidx(1) != dim_idx::invalid) return true;
        if (dim >= tdim.block()) {
            gpu_assert(dim % tdim.block() == 0);
            return true;
        }
        if (dim * block >= tdim.block()) {
            factor = ir_utils::safe_divide(tdim.block(), dim);
            factor = math::gcd(factor, block);
            return true;
        }
    }
    return false;
}

template <typename T>
std::vector<T> slice(const std::vector<T> &v, int start, int stop) {
    return std::vector<T>(v.begin() + start, v.begin() + stop);
}

const char *fail_2d_header() {
    return "Cannot use 2D send. ";
}

template <typename T0>
bool fail_2d(const T0 &t0) {
    gpu_trace() << fail_2d_header() << t0;
    return false;
}

template <typename T0, typename T1>
bool fail_2d(const T0 &t0, const T1 &t1) {
    gpu_trace() << fail_2d_header() << t0 << t1;
    return false;
}

template <typename T0, typename T1, typename T2>
bool fail_2d(const T0 &t0, const T1 &t1, const T2 &t2) {
    gpu_trace() << fail_2d_header() << t0 << t1 << t2;
    return false;
}

struct send_2d_params_t {
    operator bool() const { return !is_empty(); }

    bool is_empty() const { return !is_valid; }

    bool is_store() const { return send_op == send_op_t::store; }

    bool is_prefetch() const { return send_op == send_op_t::prefetch; }

    std::vector<int> get_counts(const dsl::hw_t &hw) const {
        return block_2d_counts(hw.ngen_hw(), is_prefetch(), is_store(),
                transpose, w, type.size());
    }

    // Reduce the number of messages by increasing count per
    // message.
    void try_promote_count(const dsl::hw_t &hw) {
        if (vnni_factor != 1) return;
        std::vector<int> counts = get_counts(hw);
        for (auto i : counts)
            if (w_rcount % i == 0) {
                c = i;
                w_rcount /= i;
                break;
            }
    }

    bool apply_vnni_factor(int factor, const dsl::hw_t &hw) {
        if (factor == 0) return true;
        if (use_xy)
            return fail_2d(
                    "Can't apply VNNI factor: incompatible with XY mode.");
        // Reshape is only expected/supported with VNNI.
        if (!vnni || transpose)
            return fail_2d(
                    "Can't apply VNNI factor: unsupported vnni/transpose.");
        if (64 % (W * type.size()) != 0)
            return fail_2d("Can't apply VNNI factor: invalid surface width.");
        //if (factor != W * type.size()) return false;
        if (H % factor != 0)
            return fail_2d("Can't apply VNNI factor: invalid surface height.");
        if (c != 1) return fail_2d("Can't apply VNNI factor: invalid count.");
        if (factor > get_counts(hw)[0])
            return fail_2d(
                    "Can't apply VNNI factor: factor exceeds max_count.");
        W *= factor;
        H /= factor;
        P *= factor;
        h /= factor;
        c = factor;
        vnni_factor = factor;
        return true;
    }

    bool is_supported(const dsl::hw_t &hw) const {
        if (!block_2d_width_ok(W, type.size()))
            return fail_2d("Width is not supported.");
        if (!block_2d_height_ok(H)) return fail_2d("Height is not supported.");
        if (!block_2d_pitch_ok(hw, P, type.size(), false))
            return fail_2d("Pitch is not supported.");
        return true;
    }

    expr_t to_base(const layout_t &tlayout, const coord_t &targs) const {
        auto t = targs;
        if (use_xy) {
            t[w_tidx] = expr_t(0);
            t[h_tidx] = expr_t(0);
        }
        auto ret = offset_bytes(tlayout, t);
        if (h_vstride != 1) {
            coord_t t(targs.size());
            t[h_tidx] = targs[h_tidx] % h_vstride;
            ret += offset_bytes(tlayout, t);
        }
        return ret;
    }

    expr_t to_x(const coord_t &targs) const {
        if (!use_xy) return expr_t(0);
        auto ret = targs[w_tidx];
        return ret;
    }

    expr_t to_y(const coord_t &targs) const {
        if (!use_xy) return expr_t(0);
        auto ret = targs[h_tidx];
        if (h_vstride != 1) ret /= h_vstride;
        return ret;
    }

    int64_t to_addr_inc(const layout_t &vlayout,
            const std::vector<int> &vblock_off, const tdim_info_t &w_tdim,
            const tdim_info_t &h_tdim) const {
        auto &blocks = vlayout.blocks();
        int nblocks = (int)blocks.size();
        gpu_assert((int)vblock_off.size() == nblocks);
        int64_t ret = 0;
        for (int i = 0; i < nblocks; i++) {
            auto &b = blocks[i];
            if (use_xy) {
                if (w_tdim.has_vidx(b.idx.index())) continue;
                if (h_tdim.has_vidx(b.idx.index())) continue;
            }
            ret += (int64_t)b.stride * vblock_off[i];
        }
        return ret * vlayout.type().size() / vlayout.type().packing();
    }

    int64_t to_x_inc(
            const tdim_info_t &w_tdim, const std::vector<int> &voff) const {
        if (!use_xy) return 0;
        return w_tdim.offset(voff);
    }

    int64_t to_y_inc(
            const tdim_info_t &h_tdim, const std::vector<int> &voff) const {
        if (!use_xy) return 0;
        return h_tdim.offset(voff);
    }

    int x_off(int linear_wh) const {
        int w_send_idx = linear_wh % w_rcount;
        return w_send_idx * w * c;
    }

    int y_off(int linear_wh) const {
        int h_send_idx = linear_wh / w_rcount;
        return h_send_idx * h;
    }

    layout_t reg_layout(
            int grf_size, size_t ndims, const dsl::type_t &mem_type) const {
        layout_t l(type, std::vector<dim_t>(ndims, 1));
        dim_t cur_stride = 1;
        enum class pad_kind_t {
            none,
            dim_pow2,
            stride_grf,
        };
        auto add_block = [&](size_t dim_idx, dim_t block,
                                 pad_kind_t pad = pad_kind_t::none) {
            l = l.with_block({dim_idx, block, cur_stride});
            dim_t stride = cur_stride * block;
            switch (pad) {
                case pad_kind_t::dim_pow2:
                    stride = cur_stride * utils::rnd_up_pow2(block);
                    break;
                case pad_kind_t::stride_grf:
                    stride = utils::rnd_up(stride, grf_size / type.size());
                    break;
                case pad_kind_t::none: break;
                default: gpu_error_not_expected();
            }
            cur_stride = stride;
        };

        auto add_padded_block = [&](size_t dim_idx, dim_t block, dim_t pad) {
            int type_size = l.type().size();
            gpu_assert(pad % type_size == 0);
            if (l.blocks().empty())
                return l.with_block({dim_idx, block, pad / type_size});
            auto &last = l.blocks().back();
            auto stride = utils::rnd_up(
                    (dim_t)last.stride * last.size, (dim_t)(pad / type_size));
            return l.with_block({dim_idx, block, stride});
        };

        if (transpose) {
            add_block(h_vidx, h, pad_kind_t::dim_pow2);
            add_block(w_vidx, w, pad_kind_t::stride_grf);
        } else if (vnni) {
            int h_inner = 4 / type.size();
            int h_outer = ir_utils::safe_divide(h, h_inner);
            add_block(h_vidx, h_inner);
            add_block(w_vidx, w, pad_kind_t::dim_pow2);
            add_block(h_vidx, h_outer, pad_kind_t::stride_grf);
        } else {
            add_block(w_vidx, w, pad_kind_t::dim_pow2);
            add_block(h_vidx, h, pad_kind_t::stride_grf);
        }
        add_block(vnni_factor > 1 ? h_vidx : w_vidx, c);
        l = add_padded_block(w_vidx, w_rcount, grf_size);
        l = add_padded_block(h_vidx, h_rcount, grf_size);
        if (type != mem_type) l = ir::reinterpret(l, mem_type);
        return l;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << c << "x" << h << "x" << w;
        if (vnni || transpose) {
            oss << ".";
            if (vnni) oss << "v";
            if (transpose) oss << "t";
        }
        return oss.str();
    }

    XE_DEFINE_DUMP()

    bool is_valid = false;
    send_op_t send_op = send_op_t::undef;
    dsl::type_t type;
    bool use_xy = true;
    bool transpose = false;
    bool vnni = false;
    int vnni_factor = 1;
    int64_t W = 0; // Surface width.
    int64_t H = 0; // Surface height.
    int64_t P = 0; // Pitch.
    int w = 0; // Block width.
    int h = 0; // Block height.
    int c = 0; // Batch count.
    int w_rcount = 0;
    int h_rcount = 0;
    size_t w_vidx = dim_idx::invalid;
    size_t h_vidx = dim_idx::invalid;
    size_t w_tidx = dim_idx::invalid;
    size_t h_tidx = dim_idx::invalid;
    int h_vstride = 0;
};

struct send_block_t {
    std::string str(const std::string &indent = {}) const {
        ostringstream_t oss;
        oss << "mem[" << addr_inc << "]";
        oss << " reg[" << reg_off << "]";
        if (!mask_inc.is_empty()) oss << " mask: " << mask_inc;
        return oss.str();
    }

    XE_DEFINE_DUMP()

    int64_t addr_inc = 0;
    int64_t x_inc = 0;
    int64_t y_inc = 0;
    vec_off_t mask_inc; // nmasks
    int reg_off = 0;
};

int rounded_slots(int slots, int max_slots) {
    if (max_slots == 1) {
        gpu_assert(slots == 1);
        return 1;
    }

    int ret = 0;
    int cur_slots = max_slots;
    for (int i = 0; i < slots; i += cur_slots) {
        cur_slots = std::min(cur_slots, slots - i);
        cur_slots = utils::rnd_up_pow2(cur_slots);
        ret += cur_slots;
    }
    return ret;
}

int get_max_slots(const dsl::hw_t &hw, const send_params_t &send_params) {
    if (hw >= ngen::HW::XeHPC) return 32;
    if (send_params.send_op == send_op_t::atomic_fadd) return 8;
    return 16;
}

int get_max_block_size(const dsl::hw_t &hw, const send_params_t &params) {
    if (hw >= ngen::HW::XeHPC) return 512;
    return params.send_address == send_address_t::slm && hw <= ngen::HW::XeLP
            ? 128
            : 256;
}

class split_bounds_t {
public:
    split_bounds_t(const layout_t &layout, int factor) {
        gpu_assert(layout.offset().is(0)) << layout;
        auto tile_coord = split_exact(layout, factor);
        if (tile_coord.is_invalid()) return;

        for (auto &start : layout.iter(tile_coord.tile))
            offs_.push_back(offset_bytes<int>(layout, start));
    }

    int factor() const { return (int)offs_.size(); }

    bool is_empty() const { return offs_.empty(); }

    bool within(dim_t beg, dim_t end) const {
        if (beg >= offs_.back()) return true;
        for (int i = 0; i < factor() - 1; i++) {
            if (offs_[i] <= beg && end <= offs_[i + 1]) return true;
        }
        return false;
    }

    bool contains(int subtile_idx, dim_t off) const {
        int o0 = offs_[subtile_idx];
        int o1 = subtile_idx + 1 < factor() ? offs_[subtile_idx + 1]
                                            : std::numeric_limits<int>::max();
        return off >= o0 && off < o1;
    }

    int normalize_reg_off(int subtile_idx, int reg_off) const {
        return reg_off - offs_[subtile_idx];
    }

private:
    std::vector<int> offs_;
};

struct send_group_t {
    bool is_empty() const { return type_size == 0; }
    bool is_2d() const { return !send_2d_params.is_empty(); }
    bool is_block() const { return slots == 1 && type_size >= 16; }
    bool is_scattered() const { return !is_2d() && !is_block(); }

    int rounded_slots() const { return jit::rounded_slots(slots, max_slots); }
    int payload_size() const {
        int grf_size = hw.grf_size();
        if (is_block()) return utils::rnd_up(type_size, grf_size);
        if (is_scattered()) {
            return utils::rnd_up(slots * slot_stride, grf_size);
        }
        if (is_2d()) {
            auto &p2d = send_2d_params;
            int size = p2d.type.size() * p2d.w * p2d.h * p2d.c;
            size = utils::rnd_up(size, grf_size);
            size *= p2d.w_rcount;
            size *= p2d.h_rcount;
            return size;
        }
        gpu_error_not_expected();
        return 0;
    }

    bool has_mask(const mask_desc_t &md) const { return has_mask(md.tidx()); }
    bool has_mask(size_t tidx) const { return (mask_bits & (1 << tidx)) != 0; }

    int nmasks() const {
        int ret = 0;
        for (int i = 0; i < 31; i++) {
            if (has_mask(i)) ret++;
        }
        return ret;
    }

    send_group_t slice(int start, int stop, bool fuse, bool is_last) const {
        gpu_assert(slots == 1);
        gpu_assert(start < stop);
        int len = (fuse ? stop - start : 1);
        auto ret = *this;
        if (!is_last) ret.pad_bytes = 1;
        if (fuse) ret.type_size *= len;
        ret.blocks.clear();
        ret.add(*this, start, stop, fuse);
        return ret;
    }

    void add(const send_group_t &g, int start, int stop, bool fuse) {
        gpu_assert(g.addr_inc == addr_inc);
        gpu_assert(g.mask_inc == mask_inc);
        int len = (fuse ? stop - start : 1);
        gpu_assert(len * g.type_size == type_size);
        if (fuse) {
            blocks.push_back(g.blocks[start]);
        } else {
            for (int i = start; i < stop; i++) {
                blocks.push_back(g.blocks[i]);
            }
        }
    }

    void add_block(int64_t addr_inc, const vec_off_t &mask, int reg_off,
            int64_t x_inc = 0, int64_t y_inc = 0) {
        blocks.emplace_back();
        auto &b = blocks.back();
        b.addr_inc = addr_inc;
        b.x_inc = x_inc;
        b.y_inc = y_inc;
        b.mask_inc = mask;
        b.reg_off = reg_off;
    }

    expr_t create_mask(const std::vector<mask_desc_t> &mask_descs,
            const vec_off_t &inc) const {
        if (nmasks() == 0) return expr_t();
        expr_t ret;
        int idx = 0;
        for (auto &md : mask_descs) {
            if (!has_mask(md.tidx())) continue;
            auto md_mask = md.to_expr(mask_inc.slice(idx) + inc[idx]);
            if (ret.is_empty()) {
                ret = std::move(md_mask);
            } else {
                ret &= md_mask;
            }
            idx++;
        }
        return ret;
    }

    std::vector<func_t> create_send_funcs(
            const send_params_t &send_params) const {
        std::vector<func_t> ret;
        bool is_lsc = (hw >= ngen::HW::XeHPG);
        if (is_block()) {
            int cur_size = get_max_block_size(hw, send_params);
            for (int i = 0; i < type_size; i += cur_size) {
                cur_size = std::min(cur_size, type_size - i);
                cur_size = utils::rnd_down_pow2(cur_size);
                gpu_assert(cur_size >= 16);
                auto type = dsl::type_t::oword(cur_size / 16);
                type = fixup_type(type, send_params);
                auto f = send_t::make(hw, send_params.send_op,
                        send_params.send_address, type, 1,
                        send_t::default_slot_mask, is_lsc, fill_buf,
                        send_params.cache_hint);
                ret.push_back(f);
            }
        } else if (is_scattered()) {
            int cur_slots = max_slots;
            auto type = dsl::type_t::u(type_size * 8);
            for (int i = 0; i < slots; i += cur_slots) {
                cur_slots = std::min(cur_slots, slots - i);
                uint32_t slot_mask = send_t::default_slot_mask;
                if (!math::is_pow2(cur_slots)) {
                    slot_mask = (1u << cur_slots) - 1;
                    cur_slots = utils::rnd_up_pow2(cur_slots);
                }
                type = fixup_type(type, send_params);
                auto f = send_t::make(hw, send_params.send_op,
                        send_params.send_address, type, cur_slots, slot_mask,
                        is_lsc, fill_buf, send_params.cache_hint);
                ret.push_back(f);
            }
        } else if (is_2d()) {
            auto &p = send_2d_params;
            int rcount = p.w_rcount * p.h_rcount;
            for (int i = 0; i < rcount; i++) {
                auto type = fixup_type(p.type, send_params);
                auto f = send_t::make_2d(hw, to_2d(send_params.send_op), type,
                        into<int>(p.W), into<int>(p.H), into<int>(p.P), p.w,
                        p.h, p.c, p.vnni, p.transpose, fill_buf,
                        send_params.cache_hint);
                ret.push_back(f);
            }
        } else {
            gpu_error_not_expected();
        }
        return ret;
    }

    std::string str(const std::string &indent = {}) const {
        if (is_empty()) return indent + "(nil)";
        ostringstream_t oss;
        if (is_2d()) {
            oss << indent << "send_2d." << send_2d_params;
        } else if (is_block()) {
            oss << indent << "send.b" << type_size;
        } else if (is_scattered()) {
            oss << indent << "send.b" << type_size << "x" << slots;
        } else {
            gpu_error_not_expected();
        }
        if (!fill_buf) oss << ".nofill";
        oss << "(" << addr_inc << ")";
        if (mask_bits != 0)
            oss << std::endl << indent << "  mask_base: " << mask_inc;
        int nblocks = (int)blocks.size();
        for (int i = 0; i < nblocks; i++) {
            oss << std::endl << indent << "   #" << i << " " << blocks[i];
            if (is_2d()) {
                oss << " x = " << blocks[i].x_inc << " y = " << blocks[i].y_inc;
            }
        }

        return oss.str();
    }

    XE_DEFINE_DUMP()

    dsl::type_t fixup_type(
            const dsl::type_t &type, const send_params_t &send_params) const {
        if (hw >= ngen::HW::XeHPC) return type;

        bool is_slm = (send_params.send_address == send_address_t::slm);
        bool is_atomic = (send_params.send_op == send_op_t::atomic_fadd);
        if (!is_slm && type == dsl::type_t::oword(16))
            return dsl::type_t::hword(8);
        if (is_atomic && type.size() == 4) return dsl::type_t::dword();
        if (type.size() <= 4) return dsl::type_t::byte(type.size());
        if (type.size() == 8) return dsl::type_t::qword();

        return type;
    }

    send_group_t split(
            const split_bounds_t &bounds, int subtile_idx, bool is_g1b1) const {
        if (!is_block() || bounds.is_empty()) return send_group_t();

        int factor = bounds.factor();
        if (is_g1b1) {
            if (type_size % factor != 0) return send_group_t();
            int new_type_size = type_size / factor;
            int grf_size = hw.grf_size();
            if (new_type_size % grf_size != 0) return send_group_t();
            auto ret = *this;
            ret.addr_inc[0] = addr_inc[0] + new_type_size * subtile_idx;
            ret.type_size = new_type_size;
            return ret;
        }

        // Assume that subtile parts do not cross blocks (verified in can_split()).
        std::vector<send_block_t> new_blocks;
        for (auto &b : blocks) {
            if (bounds.contains(subtile_idx, b.reg_off)) {
                auto bb = b;
                bb.reg_off = bounds.normalize_reg_off(subtile_idx, b.reg_off);
                new_blocks.push_back(std::move(bb));
            }
        }

        auto ret = *this;
        ret.blocks = std::move(new_blocks);
        return ret;
    }

    dsl::hw_t hw;
    int max_slots = 1;
    int type_size = 0;
    int slots = 0;
    int slot_stride = 0;
    int mask_bits = 0;
    int pad_bytes = 0;
    bool fill_buf = true;
    send_2d_params_t send_2d_params;

    vec_off_t addr_inc; // slots
    int64_t x_inc = 0;
    int64_t y_inc = 0;
    vec_vec_off_t mask_inc; // slots x nmasks

    std::vector<send_block_t> blocks;
};

class mod_info_t {
public:
    mod_info_t() = default;
    mod_info_t(const view_t &view, const std::vector<tdim_info_t> &tdims)
        : vmods_(view.nvdims()), tdims_(tdims) {}

    const std::vector<modulus_t> &vmods() const { return vmods_; }

    template <typename T>
    static modulus_t get_modulus(
            const layout_t &layout, const std::vector<T> &off, const T &base) {
        auto ndims = layout.ndims();
        gpu_assert(off.size() == ndims);
        std::vector<modulus_t> mods(layout.ndims());
        for (size_t i = 0; i < ndims; i++)
            mods[i] = off[i];
        modulus_t ret = base;
        std::vector<bool> ok(layout.ndims(), true);
        for (auto &b : layout.blocks()) {
            if (b.size == 1) continue;
            auto &m = mods[b.idx];
            ret += (m % b.size) * (int64_t)b.stride;
            if (!layout.is_outermost(b)) m /= (int64_t)b.size;
        }
        return ret * layout.type().size();
    }

    modulus_t get_modulus(const layout_t &tlayout,
            const std::vector<modulus_t> &vmods) const {
        std::vector<modulus_t> tmods;
        for (int i = 0; i < (int)tdims_.size(); i++) {
            auto &tdim = tdims_[i];
            tmods.push_back(tdim.offset(vmods, tdim.base_mod()));
        }
        return mod_info_t::get_modulus(tlayout, tmods, modulus_t(0));
    }

    void set_vmods(const std::vector<modulus_t> &vmods) { vmods_ = vmods; }

private:
    std::vector<modulus_t> vmods_;
    std::vector<tdim_info_t> tdims_;
};

send_kind_t get_send_kind(const send_t &send) {
    if (send.is_block()) return send_kind_t::block;
    if (send.is_scattered()) return send_kind_t::scattered;
    if (send.is_2d()) return send_kind_t::_2d;
    return send_kind_t::undef;
}

send_kind_t get_send_kind(const stmt_t &s) {
    auto &send = s.as<func_call_t>().func.as<send_t>();
    return get_send_kind(send);
}

struct layout_2d_wrapper_t {
    layout_2d_wrapper_t(const layout_t &l) : l(l) {}

    int nblocks(size_t idx = dim_idx::invalid) const {
        int ret = 0;
        for (auto &b : l.blocks()) {
            if (b.size == 1) continue;
            if (idx == dim_idx::invalid || b.idx.index() == idx) ret++;
        }
        return ret;
    }
    const layout_block_t &w_block() const {
        gpu_assert(nblocks() >= 2);
        return l[0];
    }
    const layout_block_t &h_block() const {
        gpu_assert(nblocks() >= 2);
        return l[1];
    }
    int64_t w_stride() const { return int64_t(w_block().stride); }
    int64_t h_stride() const { return int64_t(h_block().stride); }
    dim_t w_dim() const { return w_block().size; }
    dim_t h_dim() const { return h_block().size; }
    size_t w_idx() const { return w_block().idx.index(); }
    size_t h_idx() const { return h_block().idx.index(); }

    const layout_t &l;
};

class view_info_t {
public:
    view_info_t(const dsl::hw_t &hw, const view_t &view,
            const send_params_t &send_params)
        : hw_(hw), view_(view), send_params_(send_params) {
        vlayout_ = view.create_pseudo_vlayout(/*init_offset=*/true);

        init_tdims();
        init_mask_descs();
        init_mod_info();
        init_send_kind();
        init_base();
    }

    const dsl::hw_t &hw() const { return hw_; }
    const view_t &view() const { return view_; }
    const send_params_t &send_params() const { return send_params_; }
    const layout_t &vlayout() const { return vlayout_; }
    const tdim_info_t &tdim(size_t tidx) const { return tdims_[tidx]; }
    size_t inner_idx() const { return inner_idx_; }
    size_t outer_idx() const { return outer_idx_; }
    int reg_bytes_per_elem() const { return reg_bytes_per_elem_; }
    int reg_bits_per_elem() const { return reg_bits_per_elem_; }
    send_kind_t send_kind() const { return send_kind_; }
    const send_2d_params_t &send_2d_params() const { return send_2d_params_; }
    const expr_t &addr_base() const { return addr_base_; }
    const expr_t &x_base() const { return x_base_; }
    const expr_t &y_base() const { return y_base_; }
    const std::vector<mask_desc_t> &mask_descs() const { return mask_descs_; }
    const mod_info_t &mod_info() const { return mod_info_; }
    int grf_size() const { return hw_.grf_size(); }

    int mask_bits() const {
        int ret = 0;
        auto &p2d = send_2d_params_;
        for (auto &md : mask_descs_) {
            if (!p2d.is_empty() && p2d.use_xy
                    && utils::one_of(md.tidx(), send_2d_params_.w_tidx,
                            send_2d_params_.h_tidx))
                continue;
            ret |= (1 << md.tidx());
        }
        return ret;
    }

    const tdim_info_t &vidx_to_tdim(size_t vidx) const {
        for (dim_idx_t i = 0; i < view_.ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (utils::one_of(vidx, tdim.vidx(0), tdim.vidx(1))) return tdim;
        }
        gpu_error_not_expected();
        return tdims_[0];
    }

    int init_scattered_params(const send_params_t &send_params, int inner_bytes,
            int total_bytes) const {
        // atomic_fadd messages imply direct type match
        if (send_params.send_op == send_op_t::atomic_fadd)
            return send_params.mem_type.size();

        const bool is_hw_xelp_or_below = (hw() <= ngen::HW::XeLP);
        const bool is_slm = (send_params.send_address == send_address_t::slm);
        const bool is_store = (send_params.send_op == send_op_t::store);
        const bool is_dangling = (inner_bytes >= 8 && total_bytes % 64 != 0);
        int slot_size;

        //SLM qword not supported; issue with qword store if slots < 8
        if (is_hw_xelp_or_below && (is_slm || (is_store && is_dangling)))
            slot_size = ir_utils::max_divisor(inner_bytes, {1, 2, 4});
        else
            slot_size = ir_utils::max_divisor(inner_bytes, {1, 2, 4, 8});

        // GPUs <= XeLP requires qword alignment for qword scattered messages,
        // downgrade to byte scattered (x1, x2 or x4) when alignment is
        // sub-qword.
        if (is_hw_xelp_or_below && slot_size == 8) {
            const int align = into<int>(get_block_alignment_bytes(inner_idx()));
            slot_size = std::min(
                    slot_size, ir_utils::max_divisor(align, {1, 2, 4, 8}));
        }
        return slot_size;
    }

private:
    dim_t get_block_alignment_bytes(size_t inner_idx) const {
        // Get base address.
        const auto &tlayout = view().tlayout();
        const auto &type = vlayout().type();
        dim_t align = mod_info().get_modulus(tlayout, mod_info().vmods()).n();
        // Get outer strides.
        for (size_t i = inner_idx; i < vlayout().nblocks(); i++) {
            auto &b = vlayout()[i];
            dim_t stride_bytes = dim_t(b.stride) * type.size() / type.packing();
            align = math::gcd(align, stride_bytes);
        }
        return align;
    }

    void init_tdims() {
        for (dim_idx_t i = 0; i < view_.ntdims(); i++) {
            tdims_.emplace_back(i, view_.tdim(i), view_);
        }
    }

    void init_mask_descs() {
        for (dim_idx_t i = 0; i < view_.ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (tdim.has_mask()) mask_descs_.push_back(create_mask_desc(tdim));
        }
        auto tstart = view_.cvt_vargs_to_targs(
                view_.vstart(), /*ignore_vstart=*/true);
        for (auto &md : mask_descs_)
            md.set_base(tstart[md.tidx()]);
    }

    void init_base() {
        if (send_kind_ == send_kind_t::_2d) {
            const auto &vstart = view_.vstart();
            auto tstart
                    = view_.cvt_vargs_to_targs(vstart, /*ignore_vstart=*/true);
            auto &p2d = send_2d_params_;
            addr_base_ = p2d.to_base(view_.tlayout(), tstart);
            x_base_ = p2d.to_x(tstart);
            y_base_ = p2d.to_y(tstart);
        } else {
            addr_base_ = offset_bytes(vlayout_);
        }
        addr_base_ = simplify(addr_base_);
    }

    void init_mod_info() {
        mod_info_ = mod_info_t(view_, tdims_);
        std::vector<modulus_t> vmods(view_.nvdims());
        for (dim_idx_t i = 0; i < view_.nvdims(); i++)
            vmods[i]
                    = modulus_t(view_.vstart()[i].is(0) ? 0 : view_.vdims()[i]);
        mod_info_.set_vmods(vmods);
    }

    send_2d_params_t try_init_2d() const;

    bool can_use_block(size_t inner_idx, int inner_bytes, int total_bytes,
            const send_params_t &send_params) const {
        if (send_params.send_op == send_op_t::atomic_fadd) return false;

        const auto align = (hw_ < ngen::HW::XeHPC)
                ? std::min(32, ir_utils::max_pow2_divisor(inner_bytes))
                : 8;
        if (get_block_alignment_bytes(inner_idx) % align != 0) return false;

        if (inner_bytes % hw_.grf_size() == 0) return true;

        int oword_size = 16;
        if (inner_bytes % oword_size == 0 && inner_bytes == total_bytes) {
            int grf_size = hw_.grf_size();
            uint32_t owords = inner_bytes / oword_size;
            uint32_t owords_per_grf = grf_size / oword_size;
            uint32_t sub_grf_mask = (owords_per_grf - 1);
            // At most one sub-GRF tail message can be supported.
            if (ngen::utils::popcnt(owords & sub_grf_mask) > 1) return false;
            return true;
        }
        return false;
    }

    void init_send_kind() {
        send_2d_params_ = try_init_2d();
        if (!send_2d_params_.is_empty()) {
            reg_bytes_per_elem_ = vlayout_.type().size();
            reg_bits_per_elem_ = vlayout_.type().bitsize();
            send_kind_ = send_kind_t::_2d;
            outer_idx_ = inner_idx_ = 2;
            return;
        }
        vlayout_ = split_layout_inner(vlayout_, inner_idx_);
        const dsl::type_t &type = vlayout_.type();
        int inner_elems = 1;
        int total_elems = into<int>(vlayout_.elems());
        auto &blocks = vlayout_.blocks();
        for (size_t i = 0; i < inner_idx_; i++) {
            inner_elems *= into<int>(blocks[i].size);
        }
        int inner_bytes = type.size() * inner_elems / type.packing();
        int total_bytes = type.size() * total_elems / type.packing();
        if (can_use_block(inner_idx_, inner_bytes, total_bytes, send_params_)) {
            send_kind_ = send_kind_t::block;
        } else {
            send_kind_ = send_kind_t::scattered;
        }
        vlayout_ = split_layout_outer(vlayout_, outer_idx_, reg_bits_per_elem_);
        reg_bytes_per_elem_ = utils::div_up(reg_bits_per_elem_, 8);
    }

    layout_t split_layout_inner(
            const layout_t &layout, size_t &inner_idx) const {
        stride_t stride = 1;
        std::vector<dim_t> dims(layout.ndims(), 1);
        inner_idx = layout.blocks().size();
        for (auto &b : layout.blocks()) {
            if (b.stride != stride) {
                inner_idx = layout.get_idx(b);
                break;
            }
            dim_t factor;
            if (has_vidx_mask(mask_descs_, b.idx, dims[b.idx.index()], b.size,
                        factor)) {
                inner_idx = layout.get_idx(b);
                if (factor == 1) return layout;
                inner_idx++;
                if (factor != b.size)
                    return layout.split_block(b, factor, b.size / factor);
            }
            stride *= b.size;
            dims[b.idx] *= b.size;
        }
        return layout;
    }

    double outer_split_score(
            int slot_size, int slots, int max_slots, int total_bytes) const {
        dim_t r_slots = rounded_slots(slots, max_slots);
        dim_t r_size = r_slots * slot_size;
        dim_t nmsgs = utils::div_up(r_slots, max_slots)
                * ir_utils::safe_divide(total_bytes, slots * slot_size);

        double score = total_bytes / (double)nmsgs;
        if (send_params_.prefer_dense) {
            bool is_dense = (r_size % grf_size() == 0)
                    || (total_bytes == slots * slot_size);
            score += 100 * is_dense;
        }
        return score;
    }

    layout_t split_layout_outer(const layout_t &layout, size_t &outer_idx,
            int &reg_bits_per_elem) const {
        outer_idx = inner_idx_;
        if (send_kind_ == send_kind_t::block) {
            reg_bits_per_elem = layout.type().bitsize();
            return layout;
        }
        gpu_assert(send_kind_ == send_kind_t::scattered);

        const dsl::type_t &type = layout.type();
        int inner_elems = 1;
        int total_elems = into<int>(vlayout_.elems());

        auto &blocks = layout.blocks();
        size_t nblocks = blocks.size();
        for (size_t i = 0; i < inner_idx_; i++) {
            inner_elems *= (int)blocks[i].size;
        }
        gpu_assert(total_elems * type.size() % type.packing() == 0);
        gpu_assert(inner_elems * type.size() % type.packing() == 0);

        int inner_bytes = inner_elems * type.size() / type.packing();
        int total_bytes = total_elems * type.size() / type.packing();
        int slot_size
                = init_scattered_params(send_params_, inner_bytes, total_bytes);
        reg_bits_per_elem = std::max(1, 4 / slot_size) * type.bitsize();

        int max_slots = get_max_slots(hw_, send_params_);
        int inner_slots = ir_utils::safe_divide(inner_bytes, slot_size);
        int slots = inner_slots;
        size_t best_idx = layout.nblocks() - 1;
        dim_t best_factor = blocks.empty() ? 1 : blocks.back().size;
        double best_score = 0;
        for (size_t i = inner_idx_; i < nblocks; i++) {
            auto &b = blocks[i];
            for (dim_t j = b.size; j > 1; j--) {
                if (b.size % j == 0) {
                    double score = outer_split_score(slot_size,
                            into<int>(slots * j), max_slots, total_bytes);
                    if (score > best_score) {
                        best_score = score;
                        best_idx = i;
                        best_factor = j;
                    }
                }
            }
            slots *= (int)b.size;
        }
        outer_idx = best_idx + 1;
        if (!blocks.empty() && best_factor != blocks[best_idx].size) {
            auto &b = blocks[best_idx];
            return layout.split_block(b, best_factor, b.size / best_factor);
        }
        return layout;
    }

    static mask_desc_t create_mask_desc(const tdim_info_t &tdim) {
        auto &x = view_t::placeholder_var();
        int64_t block = 1;
        mask_kind_t kind;
        int64_t a = 0;
        int64_t b = 0;
        init_mask_kind(tdim.mask(), x, a, b, block, kind);
        return mask_desc_t(expr_t(), tdim.with_block(block), kind, a, b);
    }

    static void init_mask_kind(const expr_t &e, const expr_t &x, int64_t &a,
            int64_t &b, int64_t &block, mask_kind_t &kind) {
        if (is_x_lt_y(e, x, b, block)) {
            kind = mask_kind_t::b;
        } else if (is_x_ge_a_and_x_lt_b(e, x, a, b, block)) {
            kind = mask_kind_t::ab;
        } else {
            gpu_error_not_expected() << e;
        }
    }

    static bool is_x_div_c(expr_t e, const expr_t &x, int64_t &c) {
        e = fast_simplify(e);
        if (e.is_same(x)) {
            c = 1;
            return true;
        }
        auto *binary = e.as_ptr<binary_op_t>();
        if (!binary || binary->op_kind != op_kind_t::_div) return false;
        if (!binary->a.is_same(x)) return false;
        if (!is_const(binary->b) || !binary->b.type().is_int()) return false;
        c = to_cpp<int64_t>(binary->b);
        return true;
    }

    static bool is_x_op_y(expr_t e, const expr_t &x, op_kind_t op_kind,
            int64_t &b, int64_t &c) {
        e = fast_simplify(e);
        auto *binary = e.as_ptr<binary_op_t>();
        if (!binary || binary->op_kind != op_kind) return false;
        if (!is_x_div_c(binary->a, x, c)) return false;
        if (!is_const(binary->b) || !binary->b.type().is_int()) return false;
        b = to_cpp<int64_t>(binary->b);
        return true;
    }

    static bool is_x_lt_y(
            const expr_t &e, const expr_t &x, int64_t &y, int64_t &c) {
        return is_x_op_y(e, x, op_kind_t::_lt, y, c);
    }

    static bool is_x_ge_a_and_x_lt_b(
            expr_t e, const expr_t &x, int64_t &a, int64_t &b, int64_t &c) {
        e = fast_simplify(e);
        auto *binary = e.as_ptr<binary_op_t>();
        if (!binary || binary->op_kind != op_kind_t::_and) return false;
        int64_t a_c;
        int64_t b_c;
        if (!is_x_op_y(binary->a, x, op_kind_t::_ge, a, a_c)) return false;
        if (!is_x_op_y(binary->b, x, op_kind_t::_lt, b, b_c)) return false;
        if (a_c != b_c) return false;
        c = a_c;
        return true;
    }

    static expr_t fast_simplify(const expr_t &e) {
        auto *binary = e.as_ptr<binary_op_t>();
        if (binary && binary->op_kind == op_kind_t::_and) {
            if (binary->a.is_equal(expr_t(true))) return binary->b;
            if (binary->b.is_equal(expr_t(true))) return binary->a;
        }
        return e;
    }

    dsl::hw_t hw_;
    view_t view_;
    send_params_t send_params_;
    std::vector<tdim_info_t> tdims_;
    layout_t vlayout_; // Virtual layout.
    size_t inner_idx_ = 0;
    size_t outer_idx_ = 0;
    int reg_bits_per_elem_ = 0;
    int reg_bytes_per_elem_ = 0;
    send_kind_t send_kind_ = send_kind_t::undef;
    send_2d_params_t send_2d_params_;
    expr_t addr_base_; // Base offset.
    expr_t x_base_; // Base X offset (specific to 2D send).
    expr_t y_base_; // Base Y offset (specific to 2D send).
    std::vector<mask_desc_t> mask_descs_;
    mod_info_t mod_info_;
};

class send_2d_helper_t {
public:
    send_2d_helper_t(const view_info_t &info, const send_params_t &send_params)
        : info_(info) {
        if (!try_enable(send_params.send_op, send_params.hint_2d)) return;
    }

    bool try_enable(send_op_t send_op, const send_2d_hint_t &hint) {
        if (!hint.enable) return fail_2d("2D hint is not set.");

        auto &vlayout = info_.vlayout();
        auto &mod_info = info_.mod_info();
        params_.send_op = send_op;
        params_.type = vlayout.type();
        if (hint.type != params_.type)
            return fail_2d("Retyping is not supported.");

        layout_2d_wrapper_t lw(vlayout);

        if (lw.nblocks() < 2) return fail_2d("Too few blocks: ", vlayout);
        if (lw.w_stride() != 1)
            return fail_2d("Non-unit w stride: ", lw.w_stride());

        auto &w_tdim = info_.vidx_to_tdim(lw.w_idx());
        auto &h_tdim = info_.vidx_to_tdim(lw.h_idx());

        auto w_vidx = lw.w_idx();
        auto h_vidx = lw.h_idx();
        size_t w_tidx = w_tdim.tidx();
        size_t h_tidx = h_tdim.tidx();
        bool use_xy = true;

        int w_tcount = 0;
        int h_tcount = 0;
        for (auto &b : info_.view().tlayout().blocks()) {
            w_tcount += (b.idx.index() == w_tidx);
            h_tcount += (b.idx.index() == h_tidx);
        }

        if (w_tcount > 1 || h_tcount > 1) use_xy = false;
        if (lw.nblocks(w_vidx) != 1) use_xy = false;
        if (lw.nblocks(h_vidx) != 1) use_xy = false;

        if (!use_xy) {
            if (w_tcount <= 1 || h_tcount <= 1) {
                return fail_2d("No-XY mode implies both w/h-blocked layout: ",
                        info_.view().tlayout());
            }
        }

        dim_t W = use_xy ? w_tdim.size() : lw.w_dim();
        dim_t H = use_xy ? h_tdim.size() : lw.h_dim();
        dim_t P = lw.h_stride();
        int w = hint.width;
        int h = hint.height;
        int c = 1;
        dim_t w_rcount = ir_utils::safe_divide(lw.w_dim(), w);
        dim_t h_rcount = ir_utils::safe_divide(lw.h_dim(), h);

        // block is 1D; fallback to block/scattered message
        if (w == 1 || h == 1) return fail_2d("No benefit from 2D message");

        // Check v -> t strides.
        dim_t w_vstride = w_tdim.vstride_by_vidx(w_vidx);
        if (w_vstride != 1)
            return fail_2d("Non-unit w (v -> t) stride: ", w_vstride);

        dim_t h_vstride = h_tdim.vstride_by_vidx(h_vidx);
        if (h_vstride != 1) {
            int h_nvblocks = 0;
            h_nvblocks += lw.nblocks(h_tdim.vidx(0));
            h_nvblocks += lw.nblocks(h_tdim.vidx(1));
            if (h_nvblocks > 1)
                return fail_2d("Can't handle multi h dimension with stride.");
            gpu_assert(use_xy) << "Unexpected combination.";
            if (H % h_vstride != 0)
                return fail_2d(
                        "Can't apply non-unit h (v -> t) stride: ", h_tdim);
            H /= h_vstride;
        }

        params_.use_xy = use_xy;
        params_.transpose = hint.transpose;
        params_.vnni = hint.vnni;
        params_.W = W;
        params_.H = H;
        params_.P = P;
        params_.w = w;
        params_.h = h;
        params_.c = c;
        params_.w_rcount = into<int>(w_rcount);
        params_.h_rcount = into<int>(h_rcount);
        params_.w_vidx = w_vidx;
        params_.h_vidx = h_vidx;
        params_.w_tidx = w_tidx;
        params_.h_tidx = h_tidx;
        params_.h_vstride = into<int>(h_vstride);

        if (!params_.apply_vnni_factor(hint.vnni_permute_factor, info_.hw()))
            return false;
        if (!params_.is_supported(info_.hw())) return false;
        if (!base_alignment_ok(vlayout, mod_info, h_tdim, h_vstride))
            return false;
        if (!x_alignment_ok(w_tdim, mod_info)) return false;
        if (!masks_ok()) return false;

        params_.try_promote_count(info_.hw());
        params_.is_valid = true;
        return true;
    }

    const send_2d_params_t &params() const { return params_; }

private:
    bool base_alignment_ok(const layout_t &vlayout, const mod_info_t &mod_info,
            const tdim_info_t &h_tdim, int64_t h_vstride) const {
        auto vmods = mod_info.vmods();
        vmods[params_.w_vidx] = modulus_t(0);
        vmods[params_.h_vidx] = modulus_t(0);
        auto base_mod = mod_info.get_modulus(info_.view().tlayout(), vmods);
        int base_align = block_2d_base_alignment(info_.hw());

        // TODO: move unaligned portion of offset to block start x
        auto offset = offset_bytes(info_.view().tlayout());
        if (!is_const(offset) || to_cpp<int64_t>(offset) % base_align)
            return fail_2d("Unsupported base alignment: ", base_align);

        if (!base_mod.is_divisible(base_align) != 0)
            return fail_2d("Unsupported base alignment: ", base_align);

        for (size_t i = 2; i < vlayout.nblocks(); i++) {
            int64_t stride = (int64_t)vlayout[i].stride * vlayout.type().size();
            if (stride % base_align != 0)
                return fail_2d(
                        "Outer stride results in unsupported base alignment: ",
                        stride);
        }

        if (h_vstride != 1) {
            vmods[params_.h_vidx] = modulus_t();
            auto base_mod = mod_info.get_modulus(info_.view().tlayout(), vmods);
            if (!base_mod.is_divisible(base_align))
                return fail_2d(
                        "Unsupported base alignment for h-strided access: ",
                        base_mod);
        }

        return true;
    }

    bool x_alignment_ok(
            const tdim_info_t &w_tdim, const mod_info_t &mod_info) const {
        if (!params_.use_xy) return true;
        auto x_mod = w_tdim.offset(mod_info.vmods(), w_tdim.base_mod());
        int align = block_2d_x_alignment(params_.type.size());
        int x_align = align;
        if (!x_mod.is_divisible(x_align) != 0)
            return fail_2d("Unsupported x alignment: ", x_mod);
        if (params_.w % align != 0)
            return fail_2d(
                    "Unsupported width/alignment combination: ", params_.w);

        return true;
    }

    // Checks that both w/h masks are bound masks.
    bool masks_ok() const {
        for (auto &md : info_.mask_descs()) {
            if (utils::one_of(md.tidx(), params_.w_tidx, params_.h_tidx)) {
                if (params_.use_xy) {
                    if (!md.is_bound())
                        return fail_2d("w/h is not a bound dimension: ", md);
                } else {
                    int64_t dim = (md.tidx() == params_.w_tidx ? params_.W
                                                               : params_.H);
                    if (md.tdim().block() % dim != 0) {
                        return fail_2d(
                                "Unsupported w/h mask with non-XY mode: ", md);
                    }
                }
            }
        }
        return true;
    }

    const view_info_t &info_;
    send_2d_params_t params_;
};

send_2d_params_t view_info_t::try_init_2d() const {
    send_2d_helper_t h(*this, send_params_);
    return h.params();
}

void advance(std::vector<int> &idxs, const layout_t &l, int inc) {
    for (size_t i = 0; i < idxs.size(); i++) {
        int block = (int)l[i].size;
        int inc_idx = (idxs[i] + inc) % block;
        inc = (idxs[i] + inc) / block;
        idxs[i] = inc_idx;
        if (inc == 0) break;
    }
}

class view_iterator_t {
public:
    view_iterator_t(const view_info_t &info)
        : info_(info)
        , inner_elems_(1)
        , block_off_(nblocks())
        , block_dims_(nblocks())
        , off_(info.vlayout().ndims()) {
        for (size_t i = 0; i < info_.inner_idx(); i++) {
            inner_elems_ *= (int)blocks()[i].size;
        }
        std::vector<int> dims(info_.vlayout().ndims(), 1);
        for (size_t i = 0; i < nblocks(); i++) {
            auto &b = blocks()[i];
            block_dims_[i] = dims[b.idx.index()];
            dims[b.idx.index()] *= (int)b.size;
        }
    }

    int type_size() const { return info_.vlayout().type().size(); }
    int type_packing() const { return info_.vlayout().type().packing(); }
    int inner_elems() const { return inner_elems_; }
    int inner_bytes() const {
        return inner_elems_ * type_size() / type_packing();
    }
    int reg_off() const { return reg_off_; }

    int middle_blocks() const {
        int ret = 1;
        for (size_t i = info_.inner_idx(); i < info_.outer_idx(); i++)
            ret *= (int)blocks()[i].size;
        return ret;
    }

    dim_t total_bytes() const {
        return info_.vlayout().elems() * type_size() / type_packing();
    }

    size_t nblocks() const { return blocks().size(); }

    const std::vector<layout_block_t> &blocks() const {
        return info_.vlayout().blocks();
    }

    bool has_next(int elems) const {
        if (linear_off_ == 0 && info_.inner_idx() == nblocks()) return false;
        return linear_off_ + elems < info_.vlayout().elems();
    }

    void next(vec_vec_off_t &mask, vec_off_t &addr, int elems, int slots,
            int slot_size, int mask_bits) {
        gpu_assert(has_next(elems));
        advance(block_off_, info_.vlayout(), elems);
        linear_off_ += elems;
        reg_off_ += utils::div_up(elems * info_.reg_bits_per_elem(), 8);
        off_.assign(info_.vlayout().ndims(), 0);
        for (size_t i = 0; i < nblocks(); i++) {
            auto &b = blocks()[i];
            off_[b.idx.index()] += block_off_[i] * block_dims_[i];
        }
        mask.merge(get_mask(mask_bits, slots));
        addr.merge(get_addr(slots, slot_size));
    }

    void next(vec_off_t &mask, int64_t &addr, int64_t &x, int64_t &y, int elems,
            int mask_bits) {
        vec_vec_off_t _mask;
        vec_off_t _addr;
        next(_mask, _addr, elems, 1, 0, mask_bits);
        mask = _mask[0];
        auto &p2d = info_.send_2d_params();
        if (p2d.is_empty()) {
            addr = _addr[0];
            x = 0;
            y = 0;
        } else {
            auto &w_tdim = info_.tdim(p2d.w_tidx);
            auto &h_tdim = info_.tdim(p2d.h_tidx);
            addr = p2d.to_addr_inc(info_.vlayout(), block_off_, w_tdim, h_tdim);
            x = p2d.to_x_inc(w_tdim, off_);
            y = p2d.to_y_inc(h_tdim, off_);
        }
    }

    void pad_reg_off(int bytes) { reg_off_ = utils::rnd_up(reg_off_, bytes); }

    vec_off_t get_addr(int slots = 1, int slot_size = 0) const {
        int64_t ret = 0;
        for (size_t i = 0; i < nblocks(); i++) {
            auto &b = blocks()[i];
            ret += (int64_t)b.stride * block_off_[i];
        }
        ret *= type_size();
        ret /= type_packing();
        vec_off_t vec(slots, ret);
        for (int i = 0; i < slots; i++)
            vec[i] += i * slot_size;
        return vec;
    }

    vec_vec_off_t get_mask(int mask_bits, int slots = 1) const {
        vec_off_t ret;
        for (auto &md : info_.mask_descs()) {
            if ((mask_bits & (1 << md.tidx())) == 0) continue;
            ret.push_back(md.tdim().offset(off_));
        }
        return vec_vec_off_t(slots, ret);
    }

private:
    const view_info_t &info_;
    int inner_elems_ = 0;
    int linear_off_ = 0;
    int reg_off_ = 0;
    std::vector<int> block_off_;
    std::vector<int> block_dims_;
    std::vector<int> off_;
};

// Assigns tokens based on can_fuse flags.
std::vector<int> get_tokens(const std::vector<bool> &can_fuse) {
    int n = (int)can_fuse.size() - 1;
    int token = 1;
    std::vector<int> ret(n);
    for (int i = 0; i < n;) {
        if (i + 1 < n && !can_fuse[i] && can_fuse[i + 1]) {
            ret[i] = token;
            ret[i + 1] = token;
            i += 2;
            while (i < n && can_fuse[i])
                ret[i++] = token;
            token++;
            continue;
        }
        ret[i++] = token++;
    }
    return ret;
}

class fast_send_plan_t final : public send_plan_impl_t {
public:
    fast_send_plan_t(const view_info_t &info, const layout_t &reg_layout,
            int reg_buf_size)
        : send_params_(info.send_params())
        , addr_base_(info.addr_base())
        , x_base_(info.x_base())
        , y_base_(info.y_base())
        , message_layout_(reg_layout)
        , reg_layout_(reg_layout)
        , reg_buf_size_(reg_buf_size)
        , mask_descs_(info.mask_descs()) {}

    const send_params_t &send_params() const override { return send_params_; }
    const layout_t &reg_layout() const override { return reg_layout_; }
    int reg_buf_size() const override {
        return utils::div_up(reg_buf_size_, split_factor_);
    }
    const std::vector<mask_desc_t> &mask_descs() const { return mask_descs_; }

    bool is_empty() const { return send_groups_.empty(); }
    bool is_2d() const override {
        return !is_empty() && send_groups_[0].is_2d();
    }
    bool is_scattered() const override {
        return !is_empty() && send_groups_[0].is_scattered();
    }
    bool is_block() const { return !is_empty() && send_groups_[0].is_block(); }

    void set_send_groups(const std::vector<send_group_t> &send_groups) {
        send_groups_ = send_groups;
    }
    void fixup_params() {
        if (!is_2d()) send_params_.hint_2d.enable = false;
        try_restride_layout(reg_layout_);
    }

    std::string str(const std::string &tag) const override {
        ostringstream_t oss;
        oss << tag << ":" << std::endl;
        oss << "  base = " << addr_base_ << std::endl;
        if (x_base_) oss << "  x = " << x_base_ << std::endl;
        if (y_base_) oss << "  y = " << y_base_ << std::endl;
        oss << "  layout = " << reg_layout_ << " (size = " << reg_buf_size_
            << ")" << std::endl;
        if (split_factor_ != 1)
            oss << " split_factor = " << split_factor_ << std::endl;
        const std::string indent = "  ";
        for (auto &md : mask_descs_)
            oss << md.str(indent) << std::endl;
        int ndescs = (int)send_groups_.size();
        for (int i = 0; i < ndescs; i++) {
            oss << send_groups_[i].str(indent);
            if (i != ndescs - 1) oss << std::endl;
        }
        return oss.str();
    }
    std::string str() const { return str("send_plan"); }

    bool can_split(int factor) const override {
        if (factor == 1) return true;
        // XXX: For now handle block messages only.
        if (!is_block()) return false;
        bool is_g1b1 = (send_groups_.size() == 1)
                && (send_groups_[0].blocks.size() == 1);
        if (is_g1b1) {
            // Try split.
            auto g = send_groups_[0].split(
                    split_bounds_t(reg_layout(), factor), 0, is_g1b1);
            if (!g.is_empty()) return true;
        }

        split_bounds_t bounds(reg_layout(), factor);
        if (bounds.is_empty()) return false;

        for (auto &g : send_groups_) {
            for (auto &b : g.blocks) {
                int beg = b.reg_off;
                int end = beg + g.payload_size();
                if (!bounds.within(beg, end)) return false;
            }
        }

        return true;
    }

    void set_split(int factor) override {
        gpu_assert(can_split(factor));
        split_factor_ = factor;
    }

    int split_factor() const override { return split_factor_; }

    int estimate_regs(bool with_buffer, bool with_headers,
            bool reuse_headers) const override {
        int header_size = 0;
        for (auto &g : send_groups_) {
            int g_header_size = 0;
            auto funcs = g.create_send_funcs(send_params_);
            for (int i = 0; i < (int)funcs.size(); i++) {
                auto &send = funcs[i].as<send_t>();
                if (reuse_headers) {
                    g_header_size = std::max(g_header_size, send.header_size());
                } else {
                    g_header_size += send.header_size();
                }
            }
            if (reuse_headers) {
                header_size = std::max(header_size, g_header_size);
            } else {
                header_size += g_header_size * (int)g.blocks.size();
            }
        }
        int ret = 0;
        if (with_headers) ret += header_size;
        if (with_buffer) ret += reg_buf_size();
        int grf_size = send_params_.hw.grf_size();
        return utils::div_up(ret, grf_size);
    }

    expr_t get_mem_off(const send_group_t &g, const expr_t &g_mem_off,
            int slots, int slot_off, int byte_off) const {
        if (g.is_2d()) return g_mem_off;
        if (g.is_block()) return g_mem_off + byte_off;
        if (g.is_scattered()) return slice(g_mem_off, slot_off, slots);
        gpu_error_not_expected();
        return expr_t();
    }

    expr_t get_mask(const send_group_t &g, const expr_t &g_mask,
            const send_t &send, int slot_off) const {
        if (g_mask.is_empty()) return expr_t();
        if (g.is_2d() || g.is_block())
            return shuffle_t::make_broadcast(g_mask, send.nmasks());
        if (g.is_scattered()) return slice(g_mask, slot_off, send.slots);
        gpu_error_not_expected();
        return expr_t();
    }

    XE_DEFINE_DUMP()

private:
    const layout_t &message_layout() const override { return message_layout_; }

    stmt_t do_create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx, const expr_t &pattern) const override {
        stmt_t ret;
        bool is_g1b1 = (send_groups_.size() == 1)
                && (send_groups_[0].blocks.size() == 1);
        for (auto &_g : send_groups_) {
            auto g = (split_factor_ == 1)
                    ? _g
                    : _g.split(split_bounds_t(message_layout_, split_factor_),
                              subtile_idx, is_g1b1);
            gpu_assert(!g.is_empty());
            bool try_legacy = send_params().try_legacy
                    && (g.hw < ngen::HW::XeHPC) && g.is_block();
            std::vector<stmt_t> calls;
            std::vector<send_info_t> send_infos;
            auto base_mem_off = add(addr_base_, g.addr_inc, g.slots);
            auto base_x = g.is_2d() ? add(x_base_, g.x_inc, 1) : expr_t();
            auto base_y = g.is_2d() ? add(y_base_, g.y_inc, 1) : expr_t();
            auto funcs = g.create_send_funcs(send_params_);
            for (auto &b : g.blocks) {
                auto b_mem_off = add(base_mem_off, b.addr_inc, g.slots);
                auto b_x_off = g.is_2d() ? add(base_x, b.x_inc, 1) : expr_t();
                auto b_y_off = g.is_2d() ? add(base_y, b.y_inc, 1) : expr_t();
                auto b_mask = g.create_mask(mask_descs_, b.mask_inc);
                int byte_off = 0;
                int slot_off = 0;
                int reg_off = b.reg_off;
                auto &p2d = g.send_2d_params;
                for (int i = 0; i < (int)funcs.size(); i++) {
                    auto &send = funcs[i].as<send_t>();
                    auto mem_off = get_mem_off(
                            g, b_mem_off, send.slots, slot_off, byte_off);
                    auto mask = get_mask(g, b_mask, send, slot_off);
                    auto x = g.is_2d() ? add(b_x_off, p2d.x_off(i), 1)
                                       : expr_t();
                    auto y = g.is_2d() ? add(b_y_off, p2d.y_off(i), 1)
                                       : expr_t();
                    auto call = send(mem_buf, mem_off,
                            reg_buf.is_empty() ? expr_t() : reg_buf + reg_off,
                            mask, x, y, pattern);
                    if (try_legacy) {
                        send_infos.emplace_back(
                                g.addr_inc[0] + b.addr_inc + byte_off, reg_off,
                                send.payload_size());
                    }
                    calls.push_back(call);
                    byte_off += send.access_size();
                    slot_off += send.slots;
                    reg_off += send.payload_size();
                }
            }
            if (try_legacy) calls = try_legacy_send(calls, send_infos);
            for (auto &call : calls)
                ret = ret.append(call);
        }
        return ret;
    }

    struct send_info_t {
        send_info_t(int64_t mem_off, int reg_off, int size)
            : mem_off(mem_off), reg_off(reg_off), size(size) {}

        bool can_fuse(const send_info_t &prev) const {
            if (prev.mem_off + prev.size != mem_off) return false;
            if (prev.reg_off + prev.size != reg_off) return false;
            return true;
        }

        int64_t mem_off;
        int reg_off;
        int size;
    };

    static std::vector<stmt_t> try_legacy_send(const std::vector<stmt_t> &calls,
            const std::vector<send_info_t> &infos) {
        if (calls.empty()) return calls;
        gpu_assert(calls.size() == infos.size());
        int nmsgs = (int)calls.size();
        std::vector<bool> can_fuse(nmsgs + 1, true);
        can_fuse.front() = false;
        can_fuse.back() = false;
        for (int i = 1; i < nmsgs; i++) {
            auto &prev = calls[i - 1].as<func_call_t>();
            auto &cur = calls[i].as<func_call_t>();
            if (!cur.func.is_equal(prev.func)) {
                can_fuse[i] = false;
                continue;
            }
            if (!infos[i].can_fuse(infos[i - 1])) {
                can_fuse[i] = false;
                continue;
            }
            if (send_t::arg_mask(prev).is_empty()
                    || send_t::arg_mask(cur).is_empty()) {
                can_fuse[i] = false;
                continue;
            }
        }
        // Fuse blocks with the same token.
        auto tokens = get_tokens(can_fuse);
        int beg = 0;
        int cur_token = tokens[0];
        int cur_size = infos[0].size;
        int max_masked_bytes = 64;
        std::vector<stmt_t> ret;
        for (int i = 1; i < nmsgs + 1; i++) {
            if (i == nmsgs || tokens[i] != cur_token
                    || cur_size + infos[i].size > max_masked_bytes) {
                ret.push_back(merge(calls, beg, i));
                beg = i;
                cur_size = 0;
            }
            if (i < nmsgs) {
                cur_token = tokens[i];
                cur_size += infos[i].size;
            }
        }
        return ret;
    }

    static expr_t remove_bcast(const expr_t &e) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) return shuffle->vec[0];
        return e;
    }

    static stmt_t merge(const std::vector<stmt_t> &calls, int start, int stop) {
        gpu_assert(start < stop);
        int len = stop - start;
        if (len == 1) return calls[start];

        auto &c0 = calls[start].as<func_call_t>();
        auto &s0 = c0.func.as<send_t>();
        int size = s0.payload_size();
        int new_size = size * len;
        bool ok = math::is_pow2(new_size);
        stmt_t ret;
        if (ok) {
            auto &s0 = c0.func.as<send_t>();
            auto type = s0.type;
            type = type.with_elems(
                    type.elems() * (new_size / s0.payload_size()));
            auto func = send_t::make(s0.hw, s0.op, s0.address, type, s0.slots,
                    /*is_lsc=*/false, s0.fill_buf, s0.cache_hint);
            auto new_args = c0.args;
            auto &mask = send_t::arg_mask(new_args);
            std::vector<expr_t> vec_mask;
            for (int i = start; i < stop; i++) {
                auto i_mask = remove_bcast(send_t::arg_mask(calls[i]));
                gpu_assert(i_mask);
                gpu_assert(i_mask.type().is_scalar());
                for (int i = 0; i < size / 4; i++) {
                    vec_mask.push_back(i_mask);
                }
            }
            mask = shuffle_t::make(vec_mask);
            mask = simplify_propagate_shuffle(mask);
            gpu_assert(mask.type().elems() <= 16);
            ret = ret.append(func.call(new_args));
        } else {
            for (int i = start; i < stop; i++) {
                ret = ret.append(calls[i]);
            }
        }
        return ret;
    }

    send_params_t send_params_;
    expr_t addr_base_;
    expr_t x_base_;
    expr_t y_base_;
    layout_t message_layout_, reg_layout_;
    int reg_buf_size_;
    std::vector<mask_desc_t> mask_descs_;
    std::vector<send_group_t> send_groups_;
    int split_factor_ = 1;
};

class ir_send_plan_t final : public send_plan_impl_t {
public:
    ir_send_plan_t(const dsl::kernel::options_t &options, const view_t &view,
            send_params_t &send_params)
        : send_params_(send_params)
        , ir_ctx_(options, cset_)
        , dummy_mem_buf_(
                  var_t::make(dsl::type_t::byte(dsl::type::attr_t::ptr), "mem"))
        , dummy_reg_buf_(
                  var_t::make(dsl::type_t::byte(dsl::type::attr_t::ptr), "reg"))
        , access_(make_access_builder(
                  ir_ctx_, view, dummy_mem_buf_, dummy_reg_buf_, send_params))
        , reg_layout_(access_.reg_layout()) {
        auto calls = find_objects<func_call_t>(access_.stmt());
        for (auto &c : calls) {
            switch (get_send_kind(c)) {
                case send_kind_t::_2d: is_2d_ = true; break;
                case send_kind_t::block: break;
                case send_kind_t::scattered: is_scattered_ = true; break;
                default: gpu_error_not_expected();
            }
        }
        try_restride_layout(reg_layout_);
    }

    ir_send_plan_t(const ir_send_plan_t &) = delete;
    ir_send_plan_t &operator=(const ir_send_plan_t &) = delete;
    ~ir_send_plan_t() override = default;

    const send_params_t &send_params() const override { return send_params_; }

    bool is_2d() const override { return is_2d_; }

    bool is_scattered() const override { return is_scattered_; }

    const layout_t &reg_layout() const override { return reg_layout_; }

    int reg_buf_size() const override {
        return utils::div_up(access_.reg_buf_size(), split_factor_);
    }

    bool can_split(int factor) const override {
        if (factor == 1) return true;
        split_bounds_t bounds(reg_layout(), factor);
        if (bounds.is_empty()) return false;
        auto calls = find_objects<func_call_t>(access_.stmt());
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            auto &send = c.func.as<send_t>();
            auto &reg_buf = send_t::arg_reg_buf(c);
            dim_t beg = get_offset(reg_buf);
            dim_t end = beg + send.payload_size();
            if (!bounds.within(beg, end)) return false;
        }
        return true;
    }

    void set_split(int factor) override {
        gpu_assert(can_split(factor));
        split_factor_ = factor;
    }

    int split_factor() const override { return split_factor_; }

    int estimate_regs(bool with_buffer, bool with_headers,
            bool reuse_headers) const override {
        auto calls = find_objects<func_call_t>(access_.stmt());
        int header_size = 0;
        for (auto &c : calls) {
            auto &send = c.as<func_call_t>().func.as<send_t>();
            if (reuse_headers) {
                header_size = std::max(header_size, send.header_size());
            } else {
                header_size += send.header_size();
            }
        }
        int ret = 0;
        if (with_headers) ret += header_size;
        if (with_buffer) ret += reg_buf_size();
        int grf_size = send_params_.hw.grf_size();
        return utils::div_up(ret, grf_size);
    }

    std::string str(const std::string &tag) const override {
        ostringstream_t oss;
        oss << tag << ":" << std::endl;
        oss << access_.stmt();
        return oss.str();
    }

private:
    const layout_t &message_layout() const override {
        return access_.reg_layout();
    }

    stmt_t do_create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx, const expr_t &pattern) const override {
        auto stmt = access_.stmt();
        if (stmt.is_empty()) return stmt_t();
        stmt = substitute(stmt, dummy_mem_buf_, mem_buf);
        stmt = substitute(stmt, dummy_reg_buf_, reg_buf);
        split_bounds_t bounds(reg_layout(), split_factor_);
        return split(stmt, bounds, subtile_idx);
    }

    static expr_t get_base(const expr_t &e) {
        auto *ptr = e.as_ptr<ptr_t>();
        if (ptr) return ptr->base;
        gpu_assert(e.is<var_t>()) << e;
        return e;
    }

    static int64_t get_offset(const expr_t &e) {
        auto *ptr = e.as_ptr<ptr_t>();
        if (ptr) return to_cpp<int64_t>(ptr->off);
        gpu_assert(e.is<var_t>()) << e;
        return 0;
    }

    static stmt_t split(
            const stmt_t &stmt, const split_bounds_t &bounds, int subtile_idx) {
        if (bounds.factor() == 1) return stmt;
        auto ret = stmt;
        auto calls = find_objects<func_call_t>(stmt);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            auto &send = c.func.as<send_t>();
            auto &reg_buf = send_t::arg_reg_buf(c);
            auto reg_base = get_base(reg_buf);
            int reg_off = into<int>(get_offset(reg_buf));
            if (!bounds.contains(subtile_idx, reg_off)) {
                ret = substitute(ret, c, stmt_t());
                continue;
            }
            int new_reg_off = bounds.normalize_reg_off(subtile_idx, reg_off);
            auto new_args = c.as<func_call_t>().args;
            send_t::arg_reg_buf(new_args) = reg_base + new_reg_off;
            ret = substitute(ret, c, send.call(new_args));
        }
        return ret;
    }

    send_params_t send_params_;
    constraint_set_t cset_;
    ir_context_t ir_ctx_;
    expr_t dummy_mem_buf_;
    expr_t dummy_reg_buf_;
    access_builder_t access_;
    layout_t reg_layout_;
    bool is_2d_ = false;
    bool is_scattered_ = false;
    int split_factor_ = 1;
};

send_plan_t::send_plan_t() = default;
send_plan_t::send_plan_t(std::unique_ptr<send_plan_impl_t> impl)
    : impl_(std::move(impl)) {}
send_plan_t::send_plan_t(send_plan_t &&other) = default;
send_plan_t::~send_plan_t() = default;
send_plan_t &send_plan_t::operator=(send_plan_t &&other) {
    impl_ = std::move(other.impl_);
    return *this;
}

const send_params_t &send_plan_t::send_params() const {
    return impl_->send_params();
}
bool send_plan_t::is_2d() const {
    return impl_->is_2d();
}
bool send_plan_t::is_scattered() const {
    return impl_->is_scattered();
}
const layout_t &send_plan_t::reg_layout() const {
    return impl_->reg_layout();
}
int send_plan_t::reg_buf_size() const {
    if (!impl_) return 0;
    return impl_->reg_buf_size();
}

stmt_t send_plan_t::create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
        int subtile_idx, const expr_t &pattern) const {
    return impl_->create_stmt(mem_buf, reg_buf, subtile_idx, pattern);
}

int send_plan_t::estimate_regs(
        bool with_buffer, bool with_headers, bool reuse_headers) const {
    if (!impl_) return 0;
    return impl_->estimate_regs(with_buffer, with_headers, reuse_headers);
}

bool send_plan_t::can_split(int factor) const {
    return impl_->can_split(factor);
}

void send_plan_t::set_split(int factor) {
    impl_->set_split(factor);
}

int send_plan_t::split_factor() const {
    return impl_->split_factor();
}

std::string send_plan_t::str(const std::string &tag) const {
    if (!impl_) return tag + ": (nil)";
    return impl_->str(tag);
}

send_group_t init_2d(const view_info_t &info, view_iterator_t &it,
        layout_t &reg_layout, bool fill_buf) {
    auto &params = info.send_2d_params();
    auto &vlayout = info.vlayout();
    send_group_t ret;
    ret.hw = info.hw();
    ret.type_size = params.type.size();
    ret.slots = 1;
    ret.fill_buf = fill_buf;
    ret.mask_bits = info.mask_bits();
    ret.send_2d_params = params;
    ret.addr_inc = vec_off_t(0);
    ret.mask_inc = it.get_mask(ret.mask_bits);
    int grf_size = info.grf_size();
    reg_layout = params.reg_layout(grf_size, vlayout.ndims(), vlayout.type());
    ret.pad_bytes = into<int>(size_bytes(reg_layout, grf_size));
    return ret;
}

send_group_t init_block(const view_info_t &info, view_iterator_t &it,
        layout_t &reg_layout, bool fill_buf) {
    send_group_t ret;
    ret.hw = info.hw();
    ret.type_size = it.inner_bytes();
    ret.slots = 1;
    ret.fill_buf = fill_buf;
    ret.mask_bits = info.mask_bits();
    ret.addr_inc = vec_off_t(0);
    ret.mask_inc = it.get_mask(ret.mask_bits);
    ret.pad_bytes = info.hw().grf_size();
    auto &vlayout = info.vlayout();
    auto &blocks = vlayout.blocks();
    reg_layout = layout_t(vlayout.type(),
            std::vector<layout_block_t>(
                    blocks.begin(), blocks.begin() + info.outer_idx()),
            0, vlayout.ndims());
    return ret;
}

send_group_t init_scattered(const view_info_t &info,
        const send_params_t &send_params, view_iterator_t &it,
        layout_t &reg_layout, bool fill_buf) {
    auto &vlayout = info.vlayout();
    auto &blocks = vlayout.blocks();
    int type_size = vlayout.type().size();
    int type_packing = vlayout.type().packing();
    int slot_size = info.init_scattered_params(send_params, it.inner_bytes(),
            into<int>(vlayout.elems() * type_size / type_packing));
    // If not dword aligned, ensure that packing elements from the inner block
    // results in a valid layout. Otherwise, pack fewer elements per slot.
    if ((slot_size & 0x3) && !blocks.empty()) {
        auto inner_block = blocks.front().size * type_size / type_packing;
        auto masked = inner_block & (slot_size - 1);
        if (masked) slot_size = into<int>(masked & ~(masked - 1));
    }
    int slot_stride = std::max(4, slot_size);
    int inner_slots = ir_utils::safe_divide(it.inner_bytes(), slot_size);

    gpu_assert((slot_size * type_packing % type_size == 0)
            || (slot_stride == slot_size));

    send_group_t ret;
    ret.hw = info.hw();
    ret.max_slots = get_max_slots(info.hw(), send_params);
    ret.type_size = slot_size;
    ret.slot_stride = slot_stride;
    ret.slots = inner_slots * it.middle_blocks();
    ret.fill_buf = fill_buf;
    ret.mask_bits = info.mask_bits();
    auto mask_base = it.get_mask(ret.mask_bits, inner_slots);
    auto addr_base = it.get_addr(inner_slots, slot_size);
    for (int i = 0; i < it.middle_blocks() - 1; i++) {
        it.next(mask_base, addr_base, it.inner_elems(), inner_slots, slot_size,
                ret.mask_bits);
    }
    ret.addr_inc = std::move(addr_base);
    ret.mask_inc = std::move(mask_base);
    reg_layout = layout_t(vlayout.type(),
            std::vector<layout_block_t>(
                    blocks.begin(), blocks.begin() + info.outer_idx()),
            0, vlayout.ndims());
    reg_layout = reg_layout.make_dense();
    if (slot_stride != slot_size) {
        if (slot_size * type_packing == type_size) {
            reg_layout = make_strided(reg_layout, slot_stride / slot_size);
        } else {
            gpu_assert(reg_layout.nblocks() > 0);
            auto &b0 = reg_layout[0];
            int inner = slot_size * type_packing / type_size;
            reg_layout = reg_layout.split_block(b0, inner, b0.size / inner);
            int stride1 = ir_utils::safe_divide(
                    slot_stride * type_packing, type_size);
            reg_layout = make_strided(reg_layout, stride1, 1);
        }
    }

    int grf_size = info.grf_size();
    ret.pad_bytes = ret.rounded_slots() * slot_size;
    ret.pad_bytes = utils::rnd_up(ret.pad_bytes, grf_size);
    return ret;
}

std::vector<send_group_t> fuse_blocks(
        const std::vector<mask_desc_t> &mask_descs,
        const send_group_t &send_group) {
    if (!send_group.is_block()) return {send_group};
    int nblocks = (int)send_group.blocks.size();
    std::vector<bool> can_fuse(nblocks + 1, true);
    can_fuse[0] = false;
    can_fuse[nblocks] = false;

    // Check if block's mask matches the previous mask.
    for (int i = 0, idx = 0; i < 31; i++) {
        if (!send_group.has_mask(i)) continue;
        auto &md = mask_descs[idx];
        if (md.is_const_base()) {
            bool prev_value = false;
            for (int j = 0; j < nblocks; j++) {
                auto &b = send_group.blocks[j];
                auto &mask_inc = b.mask_inc;
                auto value = md.const_fold(
                        send_group.mask_inc[0][idx] + mask_inc[idx]);
                if (j > 0 && !ir_utils::is_equal(value, prev_value)) {
                    can_fuse[j] = false;
                }
                prev_value = value;
            }
        } else {
            for (int j = 1; j < nblocks; j++) {
                auto &b0 = send_group.blocks[j - 1];
                auto &b1 = send_group.blocks[j];
                if (b0.mask_inc[idx] != b1.mask_inc[idx]) can_fuse[j] = false;
            }
        }
        idx++;
    }
    // Check if block's address is contiguous.
    for (int i = 1; i < nblocks; i++) {
        auto &cur = send_group.blocks[i];
        auto &prev = send_group.blocks[i - 1];
        if (cur.addr_inc - prev.addr_inc != send_group.type_size) {
            can_fuse[i] = false;
        }
        if (cur.reg_off - prev.reg_off != send_group.type_size) {
            can_fuse[i] = false;
        }
    }

    int fuse_count = 0;
    for (int i = 0; i < nblocks; i++) {
        fuse_count += can_fuse[i];
    }
    // No blocks to fuse, return.
    if (fuse_count == 0) return {send_group};

    // Fuse blocks with the same token.
    auto tokens = get_tokens(can_fuse);
    std::vector<send_group_t> ret;
    int beg = 0;
    int cur_token = tokens[0];
    for (int i = 1; i < nblocks + 1; i++) {
        if (i == nblocks || tokens[i] != cur_token) {
            int len = i - beg;
            bool found = false;
            for (auto &g : ret) {
                if (g.type_size == send_group.type_size * len) {
                    g.add(send_group, beg, i, /*fuse=*/true);
                    found = true;
                    break;
                }
            }
            if (!found)
                ret.push_back(send_group.slice(
                        beg, i, /*fuse=*/true, /*is_last=*/i == nblocks));
            beg = i;
        }
        if (i < nblocks) cur_token = tokens[i];
    }

    // Verify that the total size is the same.
    int bytes = 0;
    for (auto &sg : ret)
        bytes += sg.type_size * (int)sg.blocks.size();
    gpu_assert(bytes == nblocks * send_group.type_size);

    return ret;
}

bool can_use_send_plan(const view_t &view) {
    for (dim_idx_t i = 0; i < view.ntdims(); i++) {
        auto &tdim = view.tdim(i);
        for (dim_idx_t j = 0; j < tdim.nvargs(); j++)
            if (tdim.vstride(j).is_unknown()) return false;
    }
    return true;
}

send_plan_t create_ir_send_plan(const dsl::kernel::options_t &options,
        const view_t &view, const send_params_t &_send_params) {
    auto send_params = _send_params;
    auto send_plan
            = utils::make_unique<ir_send_plan_t>(options, view, send_params);
    return send_plan_t(std::move(send_plan));
}

send_plan_t create_send_plan(const dsl::kernel::options_t &options,
        const view_t &view, const send_params_t &send_params, bool fill_buf) {
    if (!send_params.use_send_plan)
        return create_ir_send_plan(options, view, send_params);
    auto &hw = options.hw();
    view_info_t info(hw, view, send_params);
    view_iterator_t it(info);

    send_group_t base_group;
    layout_t reg_layout;
    switch (info.send_kind()) {
        case send_kind_t::_2d:
            base_group = init_2d(info, it, reg_layout, fill_buf);
            break;
        case send_kind_t::block:
            base_group = init_block(info, it, reg_layout, fill_buf);
            break;
        case send_kind_t::scattered:
            base_group = init_scattered(
                    info, send_params, it, reg_layout, fill_buf);
            break;
        default: gpu_error_not_expected();
    }

    // Add outer blocks to GRF layout.
    auto &blocks = info.vlayout().blocks();
    size_t outer_idx = info.outer_idx();
    dim_t stride = 1;
    if (!reg_layout.blocks().empty()) {
        auto &last = reg_layout.blocks().back();
        stride = (dim_t)last.stride * last.size;
    }
    const dsl::type_t &type = reg_layout.type();
    stride = utils::rnd_up(
            stride, base_group.pad_bytes * type.packing() / type.size());
    for (size_t i = outer_idx; i < blocks.size(); i++) {
        auto &b = blocks[i];
        reg_layout = reg_layout.with_block({b.idx, b.size, stride});
        stride *= b.size;
    }

    int reg_buf_size = send_params.is_prefetch()
            ? 0
            : into<int>(size_bytes(reg_layout, base_group.pad_bytes));
    auto ret = utils::make_unique<fast_send_plan_t>(
            info, reg_layout, reg_buf_size);
    base_group.add_block(0, vec_off_t(base_group.nmasks(), 0), 0);

    bool is_first = true;
    int step_elems = it.inner_elems();
    while (it.has_next(step_elems)) {
        vec_off_t mask;
        int64_t addr;
        int64_t x;
        int64_t y;
        it.next(mask, addr, x, y, step_elems, base_group.mask_bits);
        it.pad_reg_off(base_group.pad_bytes);
        base_group.add_block(addr, mask, it.reg_off(), x, y);
        if (is_first) {
            step_elems = it.inner_elems() * it.middle_blocks();
            is_first = false;
        }
    }

    ret->set_send_groups(fuse_blocks(ret->mask_descs(), base_group));

    if (base_group.is_scattered()
            && send_params.send_op == send_op_t::prefetch) {
        return send_plan_t();
    }

    ret->fixup_params();
    return send_plan_t(std::move(ret));
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
