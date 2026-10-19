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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_KERNEL_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_KERNEL_HPP

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "gemmstone/../../dsl/ir/codegen/operand.hpp"
#include "gemmstone/../../dsl/ir/codegen/register_allocator.hpp"
#include "gemmstone/../../dsl/ir/codegen/register_scope.hpp"
#include "gemmstone/../../dsl/ir/codegen/reorder.hpp"
#include "gemmstone/../../dsl/ir/ir.hpp"
#include "gemmstone/../../dsl/ir/walk_order.hpp"
#include "gemmstone/../../dsl/utils/logging.hpp"
#include "ngen.hpp"
#include "ngen_emulation.hpp"
#include "ngen_register_allocator.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class expr_binding_t {
public:
    expr_binding_t(ngen::HW hw) : hw_(hw) {}

    ~expr_binding_t() {
        dsl_assert(expr2dst_.empty()) << "Detected missing unbind_dst().";
    }

    bool is_dst_bound(const expr_t &expr) const {
        return expr2dst_.count(expr) == 1;
    }

    ngen_operand_t get_dst(const expr_t &expr) const {
        dsl_assert(is_dst_bound(expr)) << "Destination is not bound: " << expr;
        return expr2dst_.at(expr);
    }

    void bind_dst(const expr_t &expr, const ngen_operand_t &operand) {
        dsl_assert(expr);
        auto ret = expr2dst_.insert({expr, operand});
        dsl_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind_dst(const expr_t &expr) {
        dsl_assert(expr);
        auto it = expr2dst_.find(expr);
        dsl_assert(it != expr2dst_.end());
        expr2dst_.erase(it);
    }

    bool is_bound(const expr_t &expr) const {
        return expr2operand_.count(expr) == 1;
    }

    ngen_operand_t get(const expr_t &expr, bool allow_empty = false) const {
        if (expr.is_empty()) return ngen_operand_t();
        if (!is_bound(expr)) {
            if (!allow_empty)
                dsl_assert(false) << "Operand is not bound: " << expr;
            return ngen_operand_t();
        }
        return expr2operand_.at(expr);
    }

    void bind(const expr_t &expr, const ngen::Subregister &sub) {
        bind(expr, ngen_operand_t(reg_buf_data_t(hw_, sub)));
    }

    void bind(const expr_t &expr, const ngen_operand_t &operand) {
        if (is_dst_bound(expr)) unbind_dst(expr);

        auto op_to_bind = operand;

        // Operand is with predicate - can't bind.
        if (operand.mod().getPredCtrl() != ngen::PredCtrl::None) return;

        int esize = operand.mod().getExecSize();
        if (esize == 0) esize = 1;
        if (esize != expr.type().elems() && !expr.type().is_bool()) {
            dsl_assert(expr.type().is_scalar() || esize == 1)
                    << "Expected broadcast.";
            if (operand.is_reg_buf_data() && esize != 1) {
                // Bind scalar expression to the first vector element.
                op_to_bind = operand.reg_buf_data().format(0, 1);
            }
        }

        auto ret = expr2operand_.insert({expr, op_to_bind});
        dsl_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind(const expr_t &expr) {
        dsl_assert(expr);

        auto it = expr2operand_.find(expr);
        dsl_assert(it != expr2operand_.end());
        expr2operand_.erase(it);
    }

    std::string str() const {
        ostringstream_t oss;
        oss << "operands: [";
        const char *prefix = "";
        for (auto &kv : expr2operand_) {
            oss << prefix << kv.first << ": " << kv.second;
            prefix = ", ";
        }
        oss << "], dsts: [";
        prefix = "";
        for (auto &kv : expr2dst_) {
            oss << prefix << kv.first << ": " << kv.second;
            prefix = ", ";
        }
        oss << "]";
        return oss.str();
    }

private:
    ngen::HW hw_;
    object_map_t<expr_t, ngen_operand_t> expr2dst_;
    object_map_t<expr_t, ngen_operand_t> expr2operand_;
};

template <typename GeneratorT>
class expr_evaluator_t;

template <typename GeneratorT>
class ir_to_ngen_t;

struct kernel_attr_t {
    kernel_attr_t(const stmt_t &s);
    bool has_send_atomics = false;
    bool has_dpas = false;
    bool has_signal_header = false;
};

template <typename BaseGeneratorT>
class ir_to_ngen_generator_t : public BaseGeneratorT {
public:
    NGEN_FORWARD_SCOPE(BaseGeneratorT)

    ir_to_ngen_generator_t(const kernel::iface_t &kernel_iface,
            const kernel::options_t &options,
            const ngen::DebugConfig &debug_config)
        : BaseGeneratorT(options.hw().product(), debug_config)
        , kernel_iface_(kernel_iface)
        , options_(options)
        , ra_(getHardware())
        , expr_binding_(getHardware())
        , emu_strategy_(getHardware(), options_.hw().stepping()) {
        ra_.setRegisterCount(options_.regs());
    }

    void force_emulate64() { emu_strategy_.emulate64 = true; }

    reg_allocator_t &ra() { return ra_; }
    const reg_allocator_t &ra() const { return ra_; }

    expr_binding_t &expr_binding() { return expr_binding_; }
    const expr_binding_t &expr_binding() const { return expr_binding_; }

    ngen::Subregister grid_ids[3] = {r0.ud(1), r0.ud(6), r0.ud(7)};

    const kernel::iface_t &kernel_iface() const { return kernel_iface_; }
    const kernel::options_t &options() const { return options_; }
    const hw_t &hw_info() const { return options_.hw(); }

    void generate_prologue() {
        const bool no_mask = true;
        if (no_mask) BaseGeneratorT::setDefaultNoMask();
        BaseGeneratorT::setDefaultAutoSWSB(true);

        BaseGeneratorT::prologue();

        // Enable IEEE f32 -> s32 rounding and f64/f32/f16 denormals.
        if (getHardware() <= ngen::HW::Xe3p) {
            or_(1, BaseGeneratorT::cr0, BaseGeneratorT::cr0, uint16_t(0x14C0));
        }
    }

    void bind_external_vars(
            const stmt_t &kernel_body, const walk_order_t *walk_order) {
        alloc_manager_t alloc_mgr(kernel_body);

        // Data in r0 is necessary for epilogue generation
        ra_.claim(BaseGeneratorT::r0);

        // Bind local IDs.
        for (int i = 0; i < 3; i++) {
            auto local_id = alloc_mgr.find_var(ir::local_id_name(i), true);
            if (!local_id.is_empty()) {
                if (local_id.type().elems() == 1) {
                    // Used by legacy oneDNN applications
                    auto local_id_reg = BaseGeneratorT::getLocalID(i).uw(0);
                    ra_.claim(local_id_reg);
                    reg_buf_data_t local_reg_buf(getHardware(), local_id_reg);
                    expr_binding().bind(local_id, local_reg_buf);
                } else {
                    auto local_id_reg = BaseGeneratorT::getLocalID(i);
                    ra_.claim(local_id_reg);
                    reg_buf_data_t local_reg_buf(getHardware(), local_id_reg);
                    expr_binding().bind(local_id, local_reg_buf);
                }
            }
            auto local_size = alloc_mgr.find_var(ir::local_size_name(i), true);
            if (!local_size.is_empty()) {
                auto local_size_reg = BaseGeneratorT::getLocalSize(i).uw(0);
                ra_.claim(local_size_reg);
                expr_binding().bind(local_size, local_size_reg);
            }
        }

        // Bind arguments.
        for (size_t i = 0; i < kernel_iface_.nargs(); i++) {
            auto arg_var = kernel_iface_[i];
            auto &name = arg_var.as<var_t>().name;
            if (arg_var.type().is_gm()) {
                // Pointer arithmetic lower may change the variable type.
                auto body_var = alloc_mgr.find_var(name, /*allow_empty=*/true);
                if (body_var) arg_var = body_var;
            }
            auto arg_reg = BaseGeneratorT::getArgument(name);
            ra_.claim(arg_reg);
            expr_binding().bind(arg_var, arg_reg);
        }

        // Bind SLM buffer (SLM loads/stores use 0-based offsets).
        auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
        if (slm_buf) expr_binding().bind(slm_buf, to_ngen(expr_t(0)));

        // Workaround a hardware bug on MTL and ARL. In some scenarios, a read
        // suppression bug results in incorrect results when using r0. This
        // disables the use of r0 to avoid the issue.
        ngen::GRF r0_info = BaseGeneratorT::r0;
        if (one_of(getProductFamily(),
                    {ngen::ProductFamily::MTL, ngen::ProductFamily::ARL})) {
            r0_info = ra_.alloc();
            int grf_size = ngen::GRF::bytes(getHardware());
            ra_.claim(r0_info);
            mov(grf_size / 4, r0_info.ud(), r0.ud());

            // grid ids to be reclaimed below to reduce register usage
            ra_.release(r0_info);
        }

        // Bind grid indices.
        int r0_sub_idxs[] = {1, 6, 7};
        for (int i = 0; i < 3; i++) {
            auto tg_idx = alloc_mgr.find_var(ir::group_id_name(i), true);
            if (tg_idx) {
                ngen::Subregister tg_reg = r0_info.ud(r0_sub_idxs[i]);
                expr_binding().bind(tg_idx, tg_reg);
                ra_.claim(tg_reg);
                grid_ids[i] = tg_reg;
            } else if (walk_order) {
                for (auto &b : walk_order->blocks()) {
                    if (b.grid_id == i) {
                        ngen::Subregister tg_reg = r0_info.ud(r0_sub_idxs[i]);
                        ra_.claim(tg_reg);
                        grid_ids[i] = tg_reg;
                        break;
                    }
                }
            }
        }

        if (emu_strategy_.emulate64) {
            emu_state_.temp[0] = ra_.alloc();
            emu_state_.temp[1] = ra_.alloc();
        }

        auto kernel_attrs = kernel_attr_t(kernel_body);
        // Allocate and initialize signal header for future use.
        if (kernel_attrs.has_signal_header) {
            signal_header_ = ra_.alloc();
            BaseGeneratorT::barrierheader(signal_header_);
        }
    }

    void bind_kernel_grid_walk_order_blocked(const ngen::Subregister &id,
            const std::vector<std::pair<int, int>> &blocks,
            const std::vector<int> &dims,
            const std::vector<expr_t> &grid_vars) {
        int ndims = (int)dims.size();
        int nblocks = (int)blocks.size();
        std::vector<ngen::Subregister> rem_dims(ndims);
        std::vector<ngen::Subregister> dim_idxs(ndims);
        for (int i = 0; i < ndims; i++) {
            rem_dims[i] = ra_.alloc_sub<int32_t>();
            dim_idxs[i] = ra_.alloc_sub<int32_t>();
            emov(1, rem_dims[i], dims[i]);
            emov(1, dim_idxs[i], 0);
        }

        auto mul_add = [&](const ngen::Subregister &dst,
                               const ngen::Subregister &src0,
                               const ngen::Subregister &src1, uint32_t src2) {
            bool is_src2_16_bit
                    = (src2 <= std::numeric_limits<uint16_t>::max());
            if (getHardware() >= ngen::HW::XeLP && is_src2_16_bit && false) {
                mad(1, dst, src0, src1, src2);
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp.d(0), src1, src2 & 0xFFFF);
                mul(1, tmp.d(1), src1, src2 >> 16);
                shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
                add(1, tmp.d(0), tmp.d(1), tmp.d(0));
                add(1, dst, src0, tmp.d(0));
                ra_.safeRelease(tmp);
            }
        };

        auto _id = ra_.alloc_sub<int32_t>();
        auto qot = ra_.alloc_sub<int32_t>();
        auto rem = ra_.alloc_sub<int32_t>();
        auto rem_size = ra_.alloc_sub<uint32_t>();
        auto rounded = ra_.alloc_sub<int32_t>();
        emov(1, _id, id);
        for (int i = nblocks - 1; i >= 0; i--) {
            int dim_idx = blocks[i].first;
            int inner_block_size = 1;
            for (int j = 0; j < i; j++) {
                if (blocks[j].first == dim_idx)
                    inner_block_size *= blocks[j].second;
            }
            emov(1, rem_size, inner_block_size);
            for (int j = 0; j < ndims; j++) {
                if (j == dim_idx) continue;
                emul(1, rem_size, rem_size, rem_dims[j]);
            }
            eidiv(1, qot, rem, _id, rem_size);
            emov(1, _id, rem);
            mul_add(dim_idxs[dim_idx], qot, dim_idxs[dim_idx],
                    blocks[i].second);
            emul(1, rounded, qot, inner_block_size);
            eadd(1, rounded, rem_dims[dim_idx], -rounded);
            min_(1, rem_dims[dim_idx], rounded, inner_block_size);
        }
        ra_.safeRelease(_id);
        ra_.safeRelease(qot);
        ra_.safeRelease(rem);
        ra_.safeRelease(rem_size);
        ra_.safeRelease(rounded);

        for (int i = 0; i < ndims; i++)
            ra_.safeRelease(rem_dims[i]);

        for (int i = 0; i < ndims; i++) {
            expr_binding().bind(grid_vars[i], dim_idxs[i]);
        }
    }

    void bind_kernel_grid_walk_order_non_blocked(const ngen::Subregister &id,
            const std::vector<std::pair<int, int>> &blocks,
            const std::vector<expr_t> &grid_vars) {
        int nblocks = (int)blocks.size();
        dsl_assert((int)grid_vars.size() == nblocks);
        if (nblocks == 1) {
            expr_binding().bind(grid_vars[0], id);
            return;
        }
        auto _id = ra_.alloc_sub<int32_t>();
        emov(1, _id, id);
        for (int i = 0; i < nblocks; i++) {
            int dim_idx = blocks[i].first;
            auto idx = ra_.alloc_sub<int32_t>();
            eidiv(1, _id, idx, _id, (uint32_t)blocks[i].second);
            expr_binding().bind(grid_vars[dim_idx], idx);
        }
        ra_.safeRelease(_id);
    }

    void bind_kernel_grid_walk_order(const walk_order_t &walk_order) {
        const int grid_ndims = 3;
        for (int i = 0; i < grid_ndims; i++) {
            std::vector<std::pair<int, int>> blocks;
            dsl::idx_map_t<int> dim_map;
            auto to_dim_idx = [&](const dsl::idx_t &dim) {
                if (dim_map.has(dim)) return dim_map.at(dim);
                int idx = (int)dim_map.size();
                dim_map.set(dim, idx);
                return idx;
            };
            for (auto &b : walk_order.blocks()) {
                if (b.grid_id != i) continue;
                blocks.emplace_back(to_dim_idx(b.dim), b.size);
            }
            if (dim_map.is_empty()) continue;
            std::vector<int> dims;
            std::vector<expr_t> grid_vars;
            dims.resize(dim_map.size());
            grid_vars.resize(dim_map.size());
            for (auto it = dim_map.begin(); it != dim_map.end(); it++) {
                dims[it.value()] = walk_order.dim_size(*it);
                grid_vars[it.value()] = walk_order.grid_var(*it);
            }
            if (walk_order.is_blocked(i)) {
                bind_kernel_grid_walk_order_blocked(
                        grid_ids[i], blocks, dims, grid_vars);
            } else {
                bind_kernel_grid_walk_order_non_blocked(
                        grid_ids[i], blocks, grid_vars);
            }
        }
    }

    void generate_epilogue() {
        BaseGeneratorT::epilogue(r0);
        pad_kernel();
    }

    // Kernel padding for instruction prefetch.
    void pad_kernel() {
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    const ngen::GRF &signal_header() { return signal_header_; }

    void emov(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0) {
        if (dst.is_reg_data()) {
            if (src0.is_reg_data()) {
                emov(mod, dst.reg_data(), src0.reg_data());
            } else if (src0.is_reg_buf_data()) {
                emov(mod, dst.reg_data(), src0.reg_buf_data().reg_data());
            } else if (src0.is_immediate()) {
                emov(mod, dst.reg_data(), src0.immediate());
            } else if (dst.type() == ngen::DataType::uw
                    || dst.type() == ngen::DataType::ud) {
                emov(mod, dst.reg_data(), src0.flag_register());
                if (src0.is_negated()) {
                    not_(mod, dst.reg_data(), dst.reg_data());
                }
            } else {
                emov(mod | src0.flag_register_mod(), dst.reg_data(), 1);
                emov(mod | ~src0.flag_register_mod(), dst.reg_data(), 0);
            }
        } else {
            // dst is a flag register.
            dsl_assert(!dst.is_negated());
            auto _mod = mod;
            _mod.setExecSize(1);
            if (src0.is_reg_data()) {
                emov(_mod, dst.flag_register(), src0.reg_data());
            } else {
                emov(_mod, dst.flag_register(), src0.immediate());
            }
        }
    }

    void eadd(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src0.is_immediate()) {
            dsl_assert(src1.is_reg_data());
            eadd(mod, dst, src1, src0);
            return;
        }
        if (src1.is_reg_data()) {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            if (ngen_is_qw(src1.type())) {
                auto tmp = ra_.alloc_sub(src1.type());
                emov(1, tmp, src1.immediate());
                eadd(mod, dst.reg_data(), src0.reg_data(), tmp);
                ra_.safeRelease(tmp);
            } else {
                eadd(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
            }
        }
    }

    void emul(const ngen::InstructionModifier &mod_, const ngen_operand_t &dst_,
            const ngen_operand_t &src0_, const ngen_operand_t &src1_) {
        int width = mod_.getExecSize();
        int esize = ngen_is_dw(src0_.type()) && ngen_is_dw(src1_.type())
                ? 8
                : width;
        int step = esize;
        auto src0 = src0_;
        auto src1 = src1_;
        auto dst = dst_;
        auto mod = mod_;
        if (src0.is_immediate()) {
            dsl_assert(src1.is_reg_data());
            emul(mod, dst, src1, src0);
            return;
        }
        if (src1.is_reg_data()) {
            for (int i = 0; i < width; i += step) {
                step = std::min(step, width - i);
                step = rounddown_pow2(step);
                esize = step;
                mod.setExecSize(esize);
                auto subreg = [&](const ngen_operand_t &src) {
                    auto hs = src.reg_buf_data().hs();
                    int stride = hs == 0 ? 1 : esize;
                    return src.sub_reg_data(i, stride);
                };
                src0 = subreg(src0_);
                src1 = subreg(src1_);
                dst = dst_.sub_reg_data(i, esize);
                if (ngen_is_dw(src1.type()) && ngen_is_w(src0.type())) {
                    emul(mod, dst.reg_data(), src1.reg_data(), src0.reg_data());
                } else {
                    emul(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
                }
            }
        } else {
            emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eadd3(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &_src0, const ngen_operand_t &_src1,
            const ngen_operand_t &_src2) {
        auto src0 = _src0;
        auto src1 = _src1;
        auto src2 = _src2;
        ngen_register_scope_t scope(ra_);
        align_src_dst_offset(this, scope, mod, dst, src0);
        align_src_dst_offset(this, scope, mod, dst, src1, true);
        if (getHardware() >= ngen::HW::XeHP) {
            if (src2.is_reg_data()) {
                align_src_dst_offset(this, scope, mod, dst, src2);
                add3(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                        fixup_ternary_rgn(src1.reg_data()), src2.reg_data());
            } else {
                add3(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                        fixup_ternary_rgn(src1.reg_data()), src2.immediate());
            }
            return;
        }
        add(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        if (src2.is_reg_data()) {
            align_src_dst_offset(this, scope, mod, dst, src2, true);
            add(mod, dst.reg_data(), dst.reg_data(), src2.reg_data());
        } else {
            add(mod, dst.reg_data(), dst.reg_data(), src2.immediate());
        }
    }

    void emad(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &_src0, const ngen_operand_t &_src1,
            const ngen_operand_t &_src2) {
        auto src0 = _src0;
        auto src1 = _src1;
        auto src2 = _src2;
        ngen_register_scope_t scope(ra_);
        align_src_dst_offset(this, scope, mod, dst, src1, true);
        if (src2.is_reg_data()) {
            align_src_dst_offset(this, scope, mod, dst, src0);
            align_src_dst_offset(this, scope, mod, dst, src2);
            mad(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                    fixup_ternary_rgn(src1.reg_data()), src2.reg_data());
        } else if (src0.is_immediate()
                && (ngen_is_dw(src0.type())
                        || src0.type() == ngen::DataType::uw)) {
            // dword immediate src0 is not supported, move to a register.
            auto tmp_src0 = scope.alloc_sub(src0.type());
            mov(1, tmp_src0, src0.immediate());
            mad(mod, dst.reg_data(), tmp_src0,
                    fixup_ternary_rgn(src1.reg_data()), src2.immediate());
        } else {
            align_src_dst_offset(this, scope, mod, dst, src0);
            mad(mod, dst.reg_data(), fixup_ternary_rgn(src0.reg_data()),
                    fixup_ternary_rgn(src1.reg_data()), src2.immediate());
        }
    }

    void ediv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (!src1.is_immediate()) {
            // Immediate src0 is not supported with fdiv_ieee.
            if (src0.is_immediate() && getHardware() >= ngen::HW::XeHPC) {
                auto tmp_src0 = ra_.alloc_sub(src0.type());
                mov(mod, tmp_src0, src0.immediate());
                efdiv(mod, dst,
                        ngen_operand_t(reg_buf_data_t(getHardware(), tmp_src0)),
                        src1);
                ra_.safeRelease(tmp_src0);
            } else {
                efdiv(mod, dst, src0, src1);
            }
        } else {
            auto &src1_imm = src1.immediate();
            if (to_ir(src0.type()).is_fp()) {
                constexpr float inf = std::numeric_limits<float>::infinity();
                float f = to_cpp<float>(src1_imm);
                float f_inv = f ? 1.f / f : std::signbit(f) ? -inf : inf;
                ngen::Immediate src1_inv_value(f_inv);
                emul(mod, dst, src0, src1_inv_value);
            } else {
                int32_t src1_value = to_cpp<int32_t>(src1_imm);
                dsl_assert(0 < src1_value && src1_value <= INT32_MAX)
                        << src1_value;
                eidiv(mod, dst.reg_data(), ngen::Subregister(), src0.reg_data(),
                        src1_value);
            }
        }
    }

    void efdiv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        int esize = mod.getExecSize();
        int grf_size = ngen::GRF::bytes(getHardware());
        int div_esize = std::min(esize, grf_size / int(sizeof(float)));

        dsl_assert(dst.type() == ngen::DataType::f);
        dsl_assert(src0.type() == ngen::DataType::f);
        dsl_assert(src1.type() == ngen::DataType::f);

        if (src1.reg_data().getHS() != 0) {
            int nregs = std::max(1, (mod.getExecSize() * 4) / grf_size);
            auto s1 = src1.reg_data();
            auto tmp_range_ = ra_.alloc_range(nregs);
            auto tmp = tmp_range_[0].retype(s1.getType());
            auto t1 = tmp.f(s1.getOffset())
                              .setRegion(s1.getVS(), s1.getWidth(), s1.getHS());
            inv(mod, t1, s1);
            emul(mod, dst.reg_data(), src0.reg_data(), t1);
            ra_.safeRelease(tmp);
            return;
        }

        // fdiv_ieee() is not supported in XeHPG so we use a less precise, inv-based sequence.
        if (getHardware() < ngen::HW::XeHPC) {
            auto tmp = ra_.alloc_sub<float>();
            inv(1, tmp, src1.reg_data());
            emul(mod, dst, src0,
                    ngen_operand_t(reg_buf_data_t(getHardware(), tmp)));
            ra_.safeRelease(tmp);
            return;
        }

        auto one = ra_.alloc().f();
        auto zero = ra_.alloc().f();
        auto tmp = ra_.alloc_range(4);

        auto div_mod = ngen::InstructionModifier(mod);
        div_mod.setExecSize(div_esize);

        mov(div_mod, one, ngen::Immediate(1));
        mov(div_mod, zero, ngen::Immediate(0));

        for (int i = 0; i < mod.getExecSize(); i += div_esize) {
            // Copy to temporary registers to ensure dst, num and denom are
            // distinct as required for fdiv_ieee.
            auto d = dst.sub_reg_data(i, div_esize).reg_data();
            auto s0 = src0.sub_reg_data(i, div_esize).reg_data();
            auto s1 = src1.sub_reg_data(i, 1).reg_data();
            bool force_spill = overlaps(div_esize, d, s0)
                    || overlaps(div_esize, d, s1)
                    || overlaps(div_esize, s0, s1);
            bool d_spill = force_spill || (d.getHS() != 1);
            bool s0_spill = force_spill || (s0.getHS() != 1);
            bool s1_spill = force_spill || (s1.getHS() != 1);
            auto dst_rd = w_spill(d, div_esize, d_spill);
            auto src0_rd = r_spill(s0, div_esize, s0_spill);
            auto src1_rd = r_spill(s1, div_esize, s1_spill);
            // Enable mask as fdiv_ieee relies on masked if/endif flow.
            BaseGeneratorT::setDefaultNoMask(false);
            fdiv_ieee(div_mod, f0[0], dst_rd(), src0_rd(), src1_rd(), zero, one,
                    tmp, div_mod);
            BaseGeneratorT::setDefaultNoMask(true);
        }

        ra_.safeRelease(one);
        ra_.safeRelease(zero);
        ra_.safeRelease(tmp);
    }

    void emod(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        dsl_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        dsl_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
        eidiv(mod, ngen::Subregister(), dst.reg_data(), src0.reg_data(),
                src1_value);
    }

    void eshl(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eshr(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emin(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emax(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, src0.reg_data(), src1.reg_data());
        } else if (one_of(src1.immediate().getType(),
                           {ngen::DataType::q, ngen::DataType::uq})) {
            auto tmp = src1.immediate().getType() == ngen::DataType::uq
                    ? ra_.alloc().uq()
                    : ra_.alloc().q();
            mov(1, tmp, src1.immediate());
            cmp(mod, src0.reg_data(), tmp);
            ra_.safeRelease(tmp);
        } else {
            cmp(mod, src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else if (one_of(src1.immediate().getType(),
                           {ngen::DataType::q, ngen::DataType::uq})) {
            auto tmp = src1.immediate().getType() == ngen::DataType::uq
                    ? ra_.alloc().uq()
                    : ra_.alloc().q();
            mov(1, tmp, src1.immediate());
            cmp(mod, dst.reg_data(), src0.reg_data(), tmp);
            ra_.safeRelease(tmp);
        } else {
            cmp(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eand(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src0.is_reg_data() && src1.is_reg_data()) {
            and_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            if (src0.is_reg_data())
                and_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
            else
                and_(mod, dst.reg_data(), src1.reg_data(), src0.immediate());
        }
    }

    // Emulates integer division by a non-constant (rounding towards negative
    // infinity).
    // Requirements:
    //     INT32_MIN <= x <= UINT32_MAX
    //     0         <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    // See utils::idiv_magicgu_packed() for information
    // about magic calculation.
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &qot,
            const ngen::RegData &rem, const ngen::RegData &x,
            const ngen::RegData &_y, const ngen::RegData &_magic) {
        dsl_assert(x.getHS() == 0);
        dsl_assert(_y.getType() == ngen::DataType::ud);
        dsl_assert(_magic.getHS() == 0);
        dsl_assert(_magic.getType() == ngen::DataType::uq);

        bool x_signed = one_of(x.getType(),
                {ngen::DataType::b, ngen::DataType::w, ngen::DataType::d});
        auto div_type = (x_signed ? ngen::DataType::d : ngen::DataType::ud);
        auto magic = ngen::Subregister(
                _magic, _magic.getOffset(), _magic.getType());
        auto y = ngen::Subregister(_y, _y.getOffset(), _y.getType());
        auto m = magic.ud(0);
        auto p = magic.ud(1);

        auto x_tmp = ra_.alloc().retype(div_type);
        auto qot_tmp = ra_.alloc().retype(div_type);
        auto p_tmp = ra_.alloc_sub<uint32_t>();
        auto _x = x_tmp[0];
        auto _qot = qot_tmp[0];
        mov(1, _x, x);

        auto acc = acc0.retype(div_type);
        mul(1, acc[0], _x, m.uw(0));
        mach(1, _qot, _x, m);
        add(1, p_tmp, p, -32);
        cmp(1 | ge | f0[0], p, 32);
        eshr(1 | f0[0], _qot, _qot, p_tmp);
        eshr(1 | ~f0[0], _qot, _x, p);
        if (!qot.isInvalid()) mov(mod, qot, _qot);

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            auto tmp = ra_.alloc_sub<uint64_t>();
            mul(1, tmp.ud(0), _qot, y.uw(0));
            mul(1, tmp.ud(1), _qot, y.uw(1));
            shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
            add(1, tmp.ud(0), tmp.ud(1), tmp.ud(0));
            add(mod, rem, x, -tmp.ud(0));
            ra_.safeRelease(tmp);
        }

        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
        ra_.safeRelease(p_tmp);
    }

    // Emulates integer division by a non-constant (rounding towards negative
    // infinity). This version is based on FP inverse and does not require a
    // pre-computed "magic" value. Note, that cr0 register is updated/restored
    // to use RTZ mode when converting float -> int.
    // Requirements (validated range):
    //    -2^20 <= x <= 2^20
    //     0    <  y <= 2^20
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &_qot,
            const ngen::RegData &rem, const ngen::RegData &x,
            const ngen::RegData &_y, bool update_cr0_fp_to_int_rtz = true) {
        dsl_assert(mod.getExecSize() == 1);
        dsl_assert(_y.getType() == ngen::DataType::ud);
        auto cr0_save = ra_.alloc_sub<uint32_t>();
        auto f_tmp = ra_.alloc_sub<float>();
        auto x_tmp = ra_.alloc_sub<float>();
        auto qot_tmp = ra_.alloc_sub<int32_t>();
        auto y = ngen::Subregister(_y, _y.getOffset(), _y.getType());
        mov(1, cr0_save, cr0);
        // Set RTZ rounding mode when converting float to int.
        and_(1, cr0, cr0, ~0x1000);
        mov(1, f_tmp, y);
        mov(1, x_tmp, x);
        inv(1, f_tmp, f_tmp);
        add(1, f_tmp.ud(0), f_tmp.ud(0), 1);
        mul(1, f_tmp, x_tmp, f_tmp);
        mov(mod, qot_tmp, f_tmp);
        if (!rem.isInvalid()) {
            auto tmp = ra_.alloc_sub<int64_t>();
            mul(1, tmp.d(0), qot_tmp, y.uw(0));
            mul(1, tmp.d(1), qot_tmp, y.uw(1));
            shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
            add(1, tmp.d(0), tmp.d(1), tmp.d(0));
            add(mod, rem, x, -tmp.d(0));
            ra_.safeRelease(tmp);
        }
        if (!_qot.isInvalid()) mov(mod, _qot, qot_tmp);
        mov(1, cr0, cr0_save);
        ra_.safeRelease(cr0_save);
        ra_.safeRelease(f_tmp);
        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
    }

    // Emulates integer division by a constant (rounding towards negative
    // infinity)
    // Requirements:
    //     INT32_MIN <= x <= UINT32_MAX
    //     0         <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::InstructionModifier &mod, const ngen::RegData &qot,
            const ngen::RegData &rem, const ngen::RegData &x, uint32_t y) {
        bool x_signed = one_of(x.getType(),
                {ngen::DataType::b, ngen::DataType::w, ngen::DataType::d});
        auto div_type = (x_signed ? ngen::DataType::d : ngen::DataType::ud);
        dsl_assert(x.getHS() == 0);
        if (ngen::utils::is_zero_or_pow2(y)) {
            auto _x = get_subregister(x);
            if (x.getNeg() || (x == qot) || (x == rem)) {
                // Negation modifier has bitwise semantics with shr/and so x
                // needs to be arithmetically negated first.
                _x = ra_.alloc_sub(div_type);
                mov(1, _x, x);
            }
            if (!qot.isInvalid()) eshr(mod, qot, _x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(mod, rem, _x, y - 1);
            if (_x != x) ra_.safeRelease(_x);
            return;
        }

        uint32_t m = 0, p = 0;
        utils::idiv_magicgu(y, m, p);

        auto x_tmp = ra_.alloc().retype(div_type);
        auto qot_tmp = ra_.alloc().retype(div_type);
        auto _x = x_tmp[0];
        auto _qot = qot_tmp[0];
        mov(1, _x, x);

        // qot = (x * m) >> p
        bool use_mach = true;
        if (hw_info() == ngen::HW::Xe3p) use_mach = false;
        if (use_mach) {
            auto acc = acc0.retype(div_type);
            mul(1, acc[0], _x, m & 0xFFFF);
            mach(1, _qot, _x, m);
            eshr(1, _qot, _qot, p - 32);
        } else {
            auto q_tmp = qot_tmp.retype(ngen::DataType::q);
            emul(1, q_tmp[0], _x, m);
            eshr(1, q_tmp, q_tmp, p);
        }

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            bool y_is_16_bit = (y <= static_cast<uint32_t>(
                                        std::numeric_limits<int16_t>::max()));
            if (getHardware() >= ngen::HW::XeLP && y_is_16_bit) {
                mad(mod, rem, x, _qot, -int16_t(y));
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp.ud(0), _qot, y & 0xFFFF);
                mul(1, tmp.ud(1), _qot, y >> 16);
                shl<uint32_t>(1, tmp.ud(1), tmp.ud(1), 16);
                add(1, tmp.ud(0), tmp.ud(1), tmp.ud(0));
                add(mod, rem, x, -tmp.ud(0));
                ra_.safeRelease(tmp);
            }
        }
        if (!qot.isInvalid()) mov(mod, qot, _qot);

        ra_.safeRelease(x_tmp);
        ra_.safeRelease(qot_tmp);
    }

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        ngen::EmulationImplementation::emov<DT>(
                *this, mod, dst, src0, emu_strategy_);
    }
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0) {
        ngen::EmulationImplementation::emov<DT>(
                *this, mod, dst, src0, emu_strategy_);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        ngen::EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        ngen::EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        ngen::EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        ngen::EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        ngen::EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        ngen::EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, emu_strategy_, emu_state_);
    }

    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, ngen::RegData src1) {
        const bool is_q = ngen_is_qw(src0.getType());
        if (is_q) {
            dsl_error() << "eshr(q, q, reg_data) case not implemented";
        } else {
            if (ngen::isSigned(src0.getType())) {
                asr<DT>(mod, dst, src0, src1);
            } else {
                shr<DT>(mod, dst, src0, src1);
            }
        }
    }

    void esel(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (ngen_is_qw(dst.type())) {
            auto neg_mod = mod;
            neg_mod.setPredInv(!mod.isPredInv());
            emov(mod, dst, src0);
            emov(neg_mod, dst, src1);
        } else if (src1.is_reg_data()) {
            sel(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            sel(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

protected:
    // Helper RAII class allocating a temporary GRF buffer aligned at a
    // register boundary for instructions that require aligned operands.
    class spiller_t {
    public:
        // rd - register region to align
        // esize - execution size used with the register region
        // read - whether operand is to be used as input (needs pre-copy)
        // write - whether operand is to be used as output (needs post-copy)
        // force_copy - always copy the region (even if it's aligned)
        spiller_t(ir_to_ngen_generator_t *host, const ngen::RegData &rd,
                int esize, bool read, bool write, bool force_copy)
            : host_(host), rd_(rd), esize_(esize), read_(read), write_(write) {
            if (rd.getOffset() == 0 && !force_copy) return;

            int w = rd.getWidth();
            int hs = rd.getHS();
            int vs = rd.getVS();
            int grf_size = ngen::GRF::bytes(host->getHardware());
            int regs
                    = div_up(std::max(esize * hs, 1) * rd.getBytes(), grf_size);
            tmp_range_ = host_->ra_.alloc_range(regs);
            auto tmp = tmp_range_[0].retype(rd_.getType());
            tmp_ = ngen::RegisterRegion(tmp, vs, w, hs);
            if (read_) host_->mov(esize_, to_xd(tmp_), to_xd(rd_));
        }

        spiller_t(spiller_t &&other) : spiller_t(other) {
            other.tmp_range_ = ngen::GRFRange();
        }

        ngen::RegData operator()() const {
            return tmp_.isInvalid() ? rd_ : tmp_;
        }

        ~spiller_t() {
            if (tmp_range_.isInvalid()) return;
            if (write_) host_->mov(esize_, to_xd(rd_), to_xd(tmp_));
            host_->ra_.safeRelease(tmp_range_);
        }

    private:
        spiller_t(const spiller_t &) = default;

        static ngen::RegData to_xd(const ngen::RegData &rd) {
            auto ret = rd;
            switch (rd.getBytes()) {
                case 1: ret.setType(ngen::DataType::ub); break;
                case 2: ret.setType(ngen::DataType::uw); break;
                case 4: ret.setType(ngen::DataType::ud); break;
                default: stub();
            }
            return ret;
        }

        ir_to_ngen_generator_t *host_ = nullptr;
        ngen::RegData rd_;
        int esize_;
        bool read_ = false;
        bool write_ = false;
        ngen::GRFRange tmp_range_;
        ngen::RegData tmp_;
    };

    spiller_t spill(const ngen::RegData &rd, int esize, bool read, bool write,
            bool force_copy) {
        return spiller_t(this, rd, esize, read, write, force_copy);
    }

    spiller_t r_spill(
            const ngen::RegData &rd, int esize, bool force_copy = false) {
        return spill(rd, esize, true, false, force_copy);
    }

    spiller_t w_spill(
            const ngen::RegData &rd, int esize, bool force_copy = false) {
        return spill(rd, esize, false, true, force_copy);
    }

    bool overlaps(
            int esize, const ngen::RegData &a, const ngen::RegData &b) const {
        int grf_size = ngen::GRF::bytes(getHardware());
        int a_beg = a.getBase() * grf_size + a.getByteOffset();
        int b_beg = b.getBase() * grf_size + b.getByteOffset();
        int a_end = a_beg + std::max(esize * a.getHS(), 1) * a.getBytes() - 1;
        int b_end = b_beg + std::max(esize * b.getHS(), 1) * b.getBytes() - 1;
        a_beg /= grf_size;
        b_beg /= grf_size;
        a_end /= grf_size;
        b_end /= grf_size;
        if (a_beg <= b_beg && b_beg <= a_end) return true;
        if (a_beg <= b_end && b_end <= a_end) return true;
        return false;
    }

    static ngen::RegData fixup_ternary_rgn(const ngen::RegData &r) {
        ngen::RegData retn = r;
        return ((retn.getHS() == 1) && (retn.getVS() == retn.getWidth()))
                ? retn.setRegion(1, 1, 0)
                : retn;
    }

    kernel::iface_t kernel_iface_;
    kernel::options_t options_;
    reg_allocator_t ra_;
    expr_binding_t expr_binding_;
    ngen::GRF signal_header_;

    ngen::EmulationStrategy emu_strategy_;
    ngen::EmulationState emu_state_;
};

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
class ngen_asm_code_generator_with_interface_t : public ngen::AsmCodeGenerator {
public:
    ngen_asm_code_generator_with_interface_t(
            const ngen::Product &product, const ngen::DebugConfig &)
        : ngen::AsmCodeGenerator(product)
        , interface_(ngen::getCore(product.family)) {}

    NGEN_FORWARD_SCOPE(ngen::AsmCodeGenerator)

    const ngen::NEOInterfaceHandler &getInterface() const { return interface_; }
    int getSIMD() const { return interface_.getSIMD(); }
    ngen::Subregister getGroupID(int dim) const {
        return interface_.getGroupID(dim);
    }
    void prologue() { interface_.generatePrologue(*this); }
    void epilogue(ngen::RegData r0_info) {
        int GRFCount = interface_.getGRFCount();
        bool hasSLM = (interface_.getSLMSize() > 0);
        epilogue(GRFCount, hasSLM, r0_info);
    }

    void set_interface(const ngen::NEOInterfaceHandler &interface) {
        interface_ = interface;
    }

    ngen::Subregister getArgument(const std::string &name) const {
        return interface_.getArgument(name);
    }
    ngen::GRF getLocalID(int dim) const { return interface_.getLocalID(dim); }
    ngen::Subregister getLocalSize(int dim) const {
        return interface_.getLocalSize(dim);
    }
    std::string str() {
        ostringstream_t oss;
        getCode(oss);
        return oss.str();
    }

protected:
    ngen::NEOInterfaceHandler interface_;
};
using ir_asm_generator_t
        = ir_to_ngen_generator_t<ngen_asm_code_generator_with_interface_t>;
#endif

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif
