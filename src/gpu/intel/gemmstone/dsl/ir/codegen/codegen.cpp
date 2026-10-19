/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <typeinfo>

#include "dsl/ir/codegen/bank_conflict_allocation.hpp"
#include "dsl/ir/codegen/extensions.hpp"
#include "dsl/ir/codegen/kernel.hpp"
#include "dsl/ir/codegen/reduce.hpp"
#include "dsl/ir/codegen/register_scope.hpp"
#include "dsl/ir/codegen/reorder.hpp"
#include "dsl/ir/codegen/send.hpp"
#include "dsl/ir/fma.hpp"

#include "gemmstone/runtime.hpp"
#include "ngen.hpp"
#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
#include "ngen_elf.hpp"
#endif
#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
#include <optional>
#include "ngen_sycl.hpp"
#endif
#ifdef GEMMSTONE_WITH_L0_RUNTIME
#include "ngen_level_zero.hpp"
#endif
#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
#include "ngen_opencl.hpp"
#endif

// clang-format off
// unifdef is sensitive to newlines in preprocessor directives, so we need to
// keep the formatting here.
// clang-format on

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

inline ngen::ConditionModifier cmp_op_to_ngen(op_kind_t op_kind) {
    dsl_assert(is_cmp_op(op_kind));
    switch (op_kind) {
        case op_kind_t::_eq: return ngen::ConditionModifier::eq;
        case op_kind_t::_ne: return ngen::ConditionModifier::ne;
        case op_kind_t::_ge: return ngen::ConditionModifier::ge;
        case op_kind_t::_gt: return ngen::ConditionModifier::gt;
        case op_kind_t::_le: return ngen::ConditionModifier::le;
        case op_kind_t::_lt: return ngen::ConditionModifier::lt;
        default: stub();
    }
    return ngen::ConditionModifier::none;
}

// Lowers IR to nGEN.
template <typename ngen_generator_t>
class ir_to_ngen_t final : public codegen_extension_iface_t,
                           public ir_visitor_t {
public:
    ir_to_ngen_t(ngen_generator_t *host) : host_(host) {}

    ~ir_to_ngen_t() override
#if GEMMSTONE_ASSERTIONS
    {
        if (bank_conflicts_ > 0)
            dsl_warning() << "Found bank conflicts: " << bank_conflicts_;
        if (bundle_conflicts_ > 0)
            dsl_warning() << "Found bundle conflicts: " << bundle_conflicts_;
    }
#else
            = default;
#endif

    host_t root_code_generator() const override {
        return {static_cast<typename ngen_generator_t::RootCodeGenerator *>(
                        host_),
                typeid(typename ngen_generator_t::RootCodeGenerator),
                host_->getHardware()};
    }
    const kernel::options_t &options() const override {
        return host_->options();
    }
    reg_allocator_t &allocator() override { return host_->ra(); }

    std::vector<ngen_operand_t> evaluate(const std::vector<expr_t> &exprs,
            ngen_register_scope_t &scope) override {
        expr_evaluator_t<ngen_generator_t> expr_evaluator(host_, scope);
        return expr_evaluator.eval(exprs);
    }

    // Not currently provided via codegen_extension_iface_t due to no users
    ngen_operand_t evaluate(const expr_t &e, ngen_register_scope_t &scope,
            const ngen_operand_t &dst_operand = ngen_operand_t(),
            bool fill_mask0 = false) const {
        expr_evaluator_t<ngen_generator_t> expr_evaluator(host_, scope);
        return expr_evaluator.eval(e, dst_operand, fill_mask0);
    }

    ngen::HW hw() const { return host_->getHardware(); }

    void _visit(const alloc_t &obj) override {
        ngen_register_scope_t scope(host_->ra());
        bool do_alloc = (obj.kind == alloc_kind_t::grf);
        bool use_bc_alloc = false;
        if (do_alloc) {
            const int max_ngen_type_bits = 64;
            reg_buf_data_t rbd;
            if (obj.has_attr<bank_conflict_attr_t>()) {
                rbd = create_bank_conflict_allocation(obj);
                use_bc_alloc = true;
            } else if (obj.size * 8 <= max_ngen_type_bits) {
                rbd = scope.alloc_reg_data(type_t::u(obj.size * 8));
            } else {
                const int regs = div_up(obj.size, ngen::GRF::bytes(hw()));
                if (is_header(obj.buf)) {
                    rbd = alloc_header(scope, regs);
                } else {
                    rbd = scope.alloc_reg_buf(regs);
                }
            }
            if (obj.has_attr<grf_permute_attr_t>()) {
                auto &attr = obj.get_attr<grf_permute_attr_t>();
                rbd.set_grf_permutation(*attr.grf_perm);
            }
            host_->expr_binding().bind(obj.buf, rbd);
        }
        host_->comment(obj.line_str() + " -> "
                + host_->expr_binding().get(obj.buf).str());
        visit(obj.body);
        if (do_alloc) host_->expr_binding().unbind(obj.buf);
        if (use_bc_alloc) release_bank_conflict_allocation(obj);
    }

    void _visit(const assign_t &obj) override {
        host_->comment(obj.str());
        ngen_register_scope_t scope(host_->ra());
        auto var = evaluate(obj.var, scope);
        evaluate(obj.value, scope, var);
    }

    void _visit(const for_t &obj) override {
        host_->comment(obj.line_str());
        ngen_register_scope_t scope(host_->ra());
        auto var_op = scope.alloc_reg_data(obj.var.type());
        bool dynamic_loop = !is_const(obj.init) || !is_const(obj.bound);
        auto init_op = evaluate(obj.init, scope);
        auto bound_op = evaluate(obj.bound, scope);
        auto step_op = evaluate(obj.step, scope);

        host_->expr_binding().bind(obj.var, var_op);
        host_->comment(obj.var.str() + " -> "
                + host_->expr_binding().get(obj.var).str());

        host_->emov(1, var_op, init_op);

        // For dynamic loops use standard format otherwise
        // use do-while format.
        if (dynamic_loop) {
            ngen::Label loop_end_label;
            ngen::Label loop_begin_label;
            host_->mark(loop_begin_label);
            host_->ecmp(1 | host_->ge | host_->f0[0], var_op, bound_op);
            host_->jmpi(1 | host_->f0[0], loop_end_label);
            visit(obj.body);

            host_->eadd(1, var_op, var_op, step_op);
            host_->jmpi(1, loop_begin_label);
            host_->mark(loop_end_label);
        } else {
            ngen::Label loop_label;
            host_->mark(loop_label);
            visit(obj.body);

            host_->eadd(1, var_op, var_op, step_op);
            host_->ecmp(1 | host_->lt | host_->f0[0], var_op, bound_op);
            host_->jmpi(1 | host_->f0[0], loop_label);
        }

        host_->expr_binding().unbind(obj.var);
        host_->comment("end " + obj.line_str());
    }

    void _visit(const func_call_t &obj) override {
        if (!obj.func.as_ptr<builtin_t>()) host_->comment(obj.line_str());
        ngen_register_scope_t scope(host_->ra());

        auto &func = obj.func;
        if (func.is<dpas_t>()) {
            auto arg_ops = evaluate(obj.args, scope);
            dpas(func.as<dpas_t>(), arg_ops, obj.attr);
        } else if (func.is<mad_t>()) {
            auto arg_ops = evaluate(obj.args, scope);
            mad(scope, func.as<mad_t>(), arg_ops, obj.attr);
        } else if (func.is<send_t>()) {
            auto &send_func = func.as<send_t>();
            auto args = obj.args;
            auto &mask = send_t::arg_mask(args);
            // If all channels are disabled for writing, quick return.
            if (all_of(mask, expr_t(false))) {
                if (send_func.is_load() || send_func.is_load_2d()) {
                    auto reg_buf_op
                            = evaluate(send_t::arg_reg_buf(args), scope);
                    auto pattern_op
                            = evaluate(send_t::arg_fill_pattern(args), scope);
                    fill_buf(reg_buf_op, send_func.payload_size(), pattern_op);
                }
                return;
            }
            // If all channels are enabled, do not use mask.
            if (all_of(mask, expr_t(true))) mask = expr_t();
            auto arg_ops = evaluate(args, scope);
            send(scope, func.as<send_t>(), arg_ops, obj.attr);
        } else if (func.is<reorder_t>()) {
            auto arg_ops = evaluate(obj.args, scope);
            dsl_assert(obj.attr.is_empty()) << "Unexpected attribute.";
            reorder(scope, func.as<reorder_t>(), arg_ops);
        } else if (func.is<reduce_t>()) {
            auto arg_ops = evaluate(obj.args, scope);
            dsl_assert(obj.attr.is_empty()) << "Unexpected attribute.";
            reduce(scope, func.as<reduce_t>(), arg_ops);
        } else if (func.is_same(funcs::barrier_func())) {
            barrier(obj.attr);
        } else if (func.is_same(funcs::barrier_wait_func())) {
            barrier_wait();
        } else if (func.is_same(funcs::signal_func())) {
            signal(obj.attr);
        } else if (func.is_same(funcs::slm_fence_func())) {
            slm_fence(obj.attr);
        } else if (func.is_same(funcs::zero_out_func())) {
            auto buf_op = evaluate(obj.args[0], scope);
            fill_buf(buf_op.reg_buf_data(), to_cpp<int>(obj.args[1]));
        } else if (auto *builtin = obj.func.as_ptr<builtin_t>()) {
            if (builtin->name == "alloc") {
                auto &var = obj.args[0].as<var_t>();
                if (!var.type.is_slm())
                    host_->expr_binding().bind(var, [&]() {
                        auto type = var.type;
                        if (type.is_scalar()) {
                            auto sub = host_->ra().alloc_sub(to_ngen(type));
                            return ngen_operand_t(reg_buf_data_t(hw(), sub));
                        }

                        int grf_size = ngen::GRF::bytes(hw());
                        int regs = div_up(
                                type.with_elems(type.elems()).size(), grf_size);
                        reg_buf_t buf(hw(), host_->ra().alloc_range(regs));
                        return ngen_operand_t(
                                reg_buf_data_t(buf).format(0, type.elems(), 1,
                                        to_ngen(type.scalar())),
                                type.elems());
                    }());
                host_->comment(obj.line_str() + " -> "
                        + host_->expr_binding().get(var).str());
            } else if (builtin->name == "free") {
                host_->comment(obj.str());
                auto &var = obj.args[0].as<var_t>();
                if (!var.type.is_slm()) host_->expr_binding().unbind(var);
            } else {
                dsl_error() << "Unknown builtin: " << obj.str();
            }
        } else if (options().extension_handler()) {
            options().extension_handler()(obj, *this);
        } else {
            dsl_error() << object_t(obj);
        }
    }

    void _visit(const if_t &obj) override {
        dsl_assert(obj.cond.type().elems() == host_->getSIMD());
        host_->comment(obj.line_str());

        bool has_else = bool(obj.else_body);
        ngen_register_scope_t scope(host_->ra());
        auto cond_op = evaluate(obj.cond, scope);

        ngen::Label l_else;
        ngen::Label l_endif;
        host_->if_(host_->getSIMD() | cond_op.flag_register(),
                has_else ? l_else : l_endif, l_endif);
        visit(obj.body);
        if (has_else) {
            host_->comment("else // " + obj.line_str());
            host_->else_(host_->getSIMD(), l_endif, l_endif);
            host_->mark(l_else);
            visit(obj.else_body);
        }
        host_->mark(l_endif);
        host_->endif(host_->getSIMD());
        host_->comment("end " + obj.line_str());
    }

    void _visit(const let_t &obj) override {
        if (obj.value.is_empty()) {
            auto var_op = host_->expr_binding().get(obj.var);
            host_->comment(obj.line_str() + " -> " + var_op.str());
            // External variable, must be already bound.
            dsl_assert(host_->expr_binding().is_bound(obj.var))
                    << "Variable is not defined: " << obj.var;
            visit(obj.body);
            return;
        }

        ngen_register_scope_t scope(host_->ra());
        host_->comment(obj.line_str());
        if (is_const(obj.value) || is_shuffle_const(obj.value)
                || obj.var.type() != obj.value.type()) {
            auto &var_type = obj.var.type();
            auto var_op = (var_type.is_bool())
                    ? ngen_operand_t(scope.alloc_flag(var_type.elems()))
                    : ngen_operand_t(scope.alloc_reg_data(var_type));
            evaluate(
                    obj.value, scope, ngen_operand_t(var_op, var_type.elems()));
            host_->expr_binding().bind(obj.var, var_op);
        } else {
            auto value_op = evaluate(obj.value, scope);
            host_->expr_binding().bind(obj.var, value_op);
        }

        auto var_op = host_->expr_binding().get(obj.var);
        host_->comment(obj.var.str() + " -> " + var_op.str());

        // At this point the scope contains allocations for temporary
        // expressions. We need to 1) query and later re-claim the allocation
        // for the let variable in a new scope and 2) release the current scope
        // allocations to reduce GRF consumption.
        ngen::GRFRange var_grf_range;
        ngen::Subregister var_sub;

        if (var_op.is_reg_data()) {
            auto var_rd = var_op.reg_data();
            var_grf_range = scope.find_grf_range(
                    var_rd.getBase(), var_rd.getByteOffset());
            var_sub = scope.find_sub(var_rd.getBase(), var_rd.getByteOffset());
        }

        // Release the current scope allocations.
        scope.clear();

        // Claim the let variable allocation.
        ngen_register_scope_t var_scope(host_->ra());
        if (!var_grf_range.isInvalid()) {
            var_scope.claim(var_grf_range);
        } else if (!var_sub.isInvalid()) {
            var_scope.claim(var_sub);
        }

        visit(obj.body);
        host_->expr_binding().unbind(obj.var);
    }

    void _visit(const store_t &obj) override {
        host_->comment(obj.line_str());
        ngen_register_scope_t scope(host_->ra());
        auto buf_op = evaluate(obj.buf, scope);
        auto off = to_cpp<int>(obj.off);
        auto mask_op = evaluate(obj.mask, scope);

        auto &type = obj.value.type();
        auto stype = type.scalar();

        int stride;
        if (obj.has_default_stride()) {
            stride = 1;
        } else {
            dsl_assert(obj.stride % stype.size() == 0);
            stride = obj.stride / stype.size();
        }

        ngen::InstructionModifier mod = type.elems();
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto dst_rbd = buf_op.reg_buf_data().format(
                off / stype.size(), type.elems(), stride, to_ngen(stype));
        ngen_operand_t dst(dst_rbd, mod);
        evaluate(
                obj.value, scope, dst, obj.fill_mask0 && !mask_op.is_invalid());
    }

    void _visit(const while_t &obj) override {
        host_->comment(obj.line_str());
        ngen_register_scope_t scope(host_->ra());

        ngen::Label loop_end_label;
        ngen::Label loop_begin_label;

        host_->mark(loop_begin_label);
        auto cond_op = evaluate(obj.cond, scope);
        host_->jmpi(1 | ~cond_op.flag_register_mod(), loop_end_label);
        visit(obj.body);
        host_->jmpi(1, loop_begin_label);
        host_->mark(loop_end_label);
        host_->comment("end " + obj.line_str());
    }

private:
    bool is_header(const expr_t &buf) const {
        auto &name = buf.as<var_t>().name;
        return name == "h" || name.find("h_") == 0;
    }

    // Allocates headers using heuristics to reduce back-to-back header reuse -
    // this helps to eliminate potential stalls caused by SWSB dependencies.
    reg_buf_t alloc_header(ngen_register_scope_t &scope, int regs) {
        auto is_used_recently = [&](const ngen::GRFRange &range) {
            if (range.isInvalid()) return false;
            for (int i = range.getBase(); i < range.getBase() + range.getLen();
                    i++) {
                for (auto &r : last_used_header_regs_)
                    if (i == r) return true;
            }
            return false;
        };
        auto record = [&](const ngen::GRFRange &range) {
            for (int i = range.getBase(); i < range.getBase() + range.getLen();
                    i++) {
                last_used_header_regs_.push_back(i);
            }
            // Remove old header registers from tracking.
            size_t cur_size = last_used_header_regs_.size();
            if (cur_size > max_tracked_header_regs) {
                last_used_header_regs_.erase(last_used_header_regs_.begin(),
                        last_used_header_regs_.begin() + cur_size
                                - max_tracked_header_regs);
            }
        };
        // Try to allocate/claim registers until we find two GRF ranges that
        // were not used recently. Registers are usually allocated
        // sequentially, and the first range may still be in use in SWSB
        // analysis: e.g. when a SIMD16 load instruction accesses one register
        // while SWSB analysis assumes it's a full SIMD32 accessing two
        // registers.
        std::vector<ngen::GRFRange> ranges;
        for (int found = 0; found < 2;) {
            auto r = scope.register_allocator().try_alloc_range(regs);
            ranges.push_back(r);
            if (!is_used_recently(r)) found++;
        }
        auto range = ranges.back();
        for (auto &r : ranges)
            scope.register_allocator().safeRelease(r);
        // If there no range found, fall back to regular allocation, without
        // any heuristics.
        if (range.isInvalid())
            range = scope.alloc_range(regs);
        else
            scope.claim(range);
        record(range);
        return reg_buf_t(scope.hw(), range);
    }

#if GEMMSTONE_ASSERTIONS
    void check_bank_conflicts(const ngen::InstructionModifier &mod,
            const ngen::RegData &_src0, const ngen::RegData &_src1,
            const ngen::RegData &_src2, bool is_dpas = false) {
        int esize = mod.getExecSize();
        int hw_simd = (hw() >= ngen::HW::XeHPC ? 16 : 8);
        auto shift = [this](const ngen::RegData &rd, int exec_off) {
            if (exec_off == 0 || rd.isNull()) return rd;
            int type_size = ngen::getBytes(rd.getType());
            int w = (exec_off % rd.getWidth());
            int h = (exec_off / rd.getWidth());
            int off = rd.getByteOffset()
                    + (w * rd.getHS() + h * rd.getVS()) * type_size;
            int grf_size = ngen::GRF::bytes(hw());
            int shifted_base = rd.getBase() + off / grf_size;
            int shifted_off = off % grf_size;
            auto ret = rd;
            ret.setBase(shifted_base);
            ret.setOffset(utils::safe_divide(shifted_off, type_size));
            return ret;
        };
        for (int i = 0; i < esize; i += hw_simd) {
            auto src0 = shift(_src0, i);
            auto src1 = shift(_src1, i);
            auto src2 = shift(_src2, i);
            bool same_bank01 = ngen::Bundle::same_bank(hw(), src0, src1);
            bool same_bank02 = ngen::Bundle::same_bank(hw(), src0, src2);
            if (is_dpas) {
                if (same_bank02) bank_conflicts_++;
            } else {
                if (same_bank01 && same_bank02) bank_conflicts_++;
                if (ngen::Bundle::conflicts(hw(), src0, src1)
                        || ngen::Bundle::conflicts(hw(), src0, src2)
                        || ngen::Bundle::conflicts(hw(), src1, src2)) {
                    bundle_conflicts_++;
                }
            }
        }
    }
#else
    template <typename... ArgsT>
    void check_bank_conflicts(const ArgsT &...) {}
#endif

    reg_buf_t create_bank_conflict_allocation(const alloc_t &alloc) {
        auto &bc_attr = alloc.get_attr<bank_conflict_attr_t>();
        auto it = bc_allocations_.find(bc_attr);
        if (it != bc_allocations_.end()) {
            it->second.retain();
            return it->second.get_reg_buf(alloc.buf);
        }
        auto bca = bank_conflict_allocation_t::create(host_->ra(), bc_attr);
        if (bca.is_empty()) return {};

        auto ret = bc_allocations_.emplace(bc_attr, std::move(bca));
        return ret.first->second.get_reg_buf(alloc.buf);
    }

    void release_bank_conflict_allocation(const alloc_t &alloc) {
        auto &bc_attr = alloc.get_attr<bank_conflict_attr_t>();
        auto it = bc_allocations_.find(bc_attr);
        dsl_assert(it != bc_allocations_.end());
        it->second.release(alloc.buf);
        if (it->second.refs() == 0) bc_allocations_.erase(bc_attr);
    }

    void signal(const func_call_attr_t &attr) {
        ngen::InstructionModifier mod;
        if (attr) mod = mod | attr.as<instruction_modifier_attr_t>().mod;
        host_->barriermsg(mod, host_->signal_header());
    }

    void barrier_wait() { host_->barrierwait(); }

    void slm_fence(const func_call_attr_t &attr) {
        ngen_register_scope_t scope(host_->ra());
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (attr) mod = mod | attr.as<instruction_modifier_attr_t>().mod;

        host_->slmfence(mod, tmp, host_->r0);
        host_->fencewait();
    }

    void barrier(const func_call_attr_t &attr) {
        ngen_register_scope_t scope(host_->ra());
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (attr) mod = mod | attr.as<instruction_modifier_attr_t>().mod;

        host_->slmfence(mod, tmp, host_->r0);
        host_->fencewait();
        host_->barriermsg(mod, host_->signal_header());
        host_->barrierwait();
    }

    void dpas(const dpas_t &dpas_func, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = dpas_t::arg_dst(args).reg_buf_data();
        auto src1 = dpas_t::arg_src1(args).reg_buf_data();
        auto src2 = dpas_t::arg_src2(args).reg_buf_data();

        if (dpas_func.is_dpasw) dst = dst.unpermute();

        int esize = dpas_func.exec_size;

        ngen::RegData src0;
        auto &src0_op = dpas_t::arg_src0(args);
        if (!src0_op.is_immediate()) {
            auto src0_rbd = src0_op.reg_buf_data().format(
                    0, esize, 1, to_ngen(dpas_func.dst_type));
            if (dpas_func.is_dpasw) src0_rbd = src0_rbd.unpermute();
            src0 = src0_rbd;
        } else {
            dsl_assert(src0_op.is_immediate());
            dsl_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null.retype(to_ngen(dpas_func.dst_type));
        }

        dst = dst.format(0, esize, 1, to_ngen(dpas_func.dst_type));
        src1 = src1.format(0, esize, 1, to_ngen(dpas_func.src1_type));
        int src2_width = (dpas_func.is_dp4a() ? 1 : esize);
        int src2_stride = (dpas_func.is_dp4a() ? 0 : 1);
        src2 = src2.format(
                0, src2_width, src2_stride, to_ngen(dpas_func.src2_type));

        ngen::InstructionModifier mod = esize;
        if (attr) mod = mod | attr.as<instruction_modifier_attr_t>().mod;
        check_bank_conflicts(mod, src0, src1, src2, /*is_dpas=*/true);
        if (dpas_func.is_dpasw) {
            host_->dpasw(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        } else if (dpas_func.is_dp4a()) {
            if (src0.isNull()) {
                host_->dp4a(mod, dst, 0, src1, src2);
            } else {
                host_->dp4a(mod, dst, src0, src1, src2);
            }
        } else {
            host_->dpas(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        }
    }

    void mad(ngen_register_scope_t &scope, const mad_t &mad_func,
            const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = mad_t::arg_dst(args).reg_buf_data();
        auto src1 = mad_t::arg_src1(args).reg_buf_data();
        auto src2 = mad_t::arg_src2(args).reg_buf_data();

        ngen::RegData src0;
        auto &src0_op = mad_t::arg_src0(args);
        if (!src0_op.is_immediate()) {
            src0 = src0_op.reg_buf_data()
                           .format(0, mad_func.exec_size, 1,
                                   to_ngen(mad_func.dst_type))
                           .reg_data();
        } else {
            dsl_assert(src0_op.is_immediate());
            dsl_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
            src0.setType(to_ngen(mad_func.dst_type));
        }

        dst = dst.format(0, mad_func.exec_size, 1, to_ngen(mad_func.dst_type));

        int src1_width = (mad_func.src1_stride == 0 ? 1 : mad_func.exec_size);
        int src2_width = (mad_func.src2_stride == 0 ? 1 : mad_func.exec_size);
        src1 = src1.format(0, src1_width, mad_func.src1_stride,
                to_ngen(mad_func.src1_type));
        src2 = src2.format(0, src2_width, mad_func.src2_stride,
                to_ngen(mad_func.src2_type));

        ngen::InstructionModifier mod = mad_func.exec_size;
        if (attr) mod = mod | attr.as<instruction_modifier_attr_t>().mod;

        check_bank_conflicts(mod, src0, src1, src2, /*is_dpas=*/false);
        if (src0.isNull()) {
            host_->mul(mod, dst, src1, src2);
        } else {
            dsl_assert(dst.byte_offset() == src0.getByteOffset())
                    << "dst/src0 must be aligned to the same GRF offset.";
            if (mad_func.dst_type == type_t::f64()
                    && src1.reg_data().getHS() == 0
                    && src1.reg_data().getVS() == 0) {
                align_src_dst_offset(host_, scope, mod, dst, src1);
                align_src_dst_offset(host_, scope, mod, dst, src2, true);
                // Workaround for sporadic f64 mad errors with broadcast src1 on XeHPC.
                host_->mad(mod, dst, src0, src2, src1);
            } else {
                align_src_dst_offset(host_, scope, mod, dst, src1, true);
                align_src_dst_offset(host_, scope, mod, dst, src2);
                host_->mad(mod, dst, src0, src1, src2);
            }
        }
    }

    void fill_buf(const ngen_operand_t &buf_op, int size,
            const ngen_operand_t &pattern = {}) const {
        auto &rd = buf_op.reg_buf_data();
        type_t type = (pattern.is_invalid() ? type_t::f32() : type_t::u32());
        int grf_size = ngen::GRF::bytes(hw());
        int step = 2 * grf_size / type.size();
        int elems = size / type.size();
        for (int i = 0; i < elems; i += step) {
            step = std::min(step, elems - i);
            step = rounddown_pow2(step);
            auto sub_rd_mov = rd.format(i, step, 1, to_ngen(type)).reg_data();
            if (pattern.is_invalid()) {
                host_->emov(step, sub_rd_mov, ngen::Immediate(0));
            } else if (pattern.is_immediate()) {
                host_->emov(step, sub_rd_mov, pattern.immediate());
            } else if (pattern.is_reg_data()) {
                host_->emov(step, sub_rd_mov, pattern.reg_data());
            } else {
                stub();
            }
        }
    }

    ngen::RegData send_maybe_make_dense_payload(ngen_register_scope_t &scope,
            const send_t &send_func, const ngen_operand_t &op_buf) const {
        if (send_func.is_prefetch() || send_func.is_prefetch_2d())
            return ngen::RegData(host_->null);

        auto &buf = op_buf.reg_buf_data();
        int size = send_func.payload_size();
        bool is_dense = buf.is_dense(size);
        if (is_dense) return ngen::GRF(buf.base());

        if (send_func.is_load() || send_func.is_load_2d()) {
            dsl_error() << "Expected dense GRF region for load message.";
            return ngen::RegData();
        }

        dsl_assert(send_func.is_store() || send_func.is_store_2d()
                || send_func.is_atomic());

        // Reorder buffer to a dense buffer for store.
        int grf_size = ngen::GRF::bytes(hw());
        int grf_elems = grf_size / ngen::getBytes(buf.type());
        int regs = div_up(size, grf_size);

        auto tmp = scope.alloc_range(regs);

        int dwords = ngen::GRF::bytes(hw()) / sizeof(int32_t);
        int max_step = 2;
        for (int i = 0; i < regs;) {
            auto sub_buf = buf.format(i * grf_elems);
            int step = std::min(max_step, regs - i);
            if (step > 1 && !sub_buf.is_dense(step * grf_size)) step = 1;
            int esize = step * dwords;
            auto src = sub_buf.subregister(ngen::DataType::ud)(1);
            auto dst = tmp[i].ud(0)(1);
            host_->emov(esize, dst, src);
            i += step;
        }
        return tmp[0];
    }

    void send_atomic_add_emu(ngen_register_scope_t &scope,
            const send_t &send_func, const ngen_operand_t &mask_op,
            ngen::InstructionModifier &mod, const ngen::RegData &mem_off_op,
            ngen::RegData &rd) const {
        int size = send_func.payload_size();
        dsl_assert((send_func.type.is_dword() || send_func.type.is_qword())
                && (size == 32 || size == 64))
                << "expected atomic message dwordx8 or qwordx8";
        auto load_func = send_t::make(send_func.hw, send_op_t::load,
                send_func.address, send_func.type, send_func.slots,
                send_func.fill_buf, send_func.cache_hint);
        auto &load_send = load_func.as<send_t>();
        send_impl_t load(load_send);
        auto cmpwr_func = send_t::make(send_func.hw, send_op_t::atomic_cmpwr,
                send_func.address, send_func.type, send_func.slots,
                send_func.fill_buf, send_func.cache_hint);
        auto &cmpwr_send = cmpwr_func.as<send_t>();
        send_impl_t cmpwr(cmpwr_send);
        bool is_df = send_func.type.is_qword();

        int grf_size = ngen::GRF::bytes(hw());
        int regs = div_up(size, grf_size);

        auto new_val = scope.alloc_range(2 * regs);
        auto old_save = scope.alloc_range(regs);
        auto flag = scope.alloc_flag(send_func.slots);
        ngen::Label atomic_label;
        rd.setType(is_df ? ngen::DataType::df : ngen::DataType::f);

        load.emit(host_, scope, mod, mem_off_op,
                is_df ? new_val[0].df(0) : new_val[0].f(0));

        if (mask_op.is_invalid())
            host_->emov(1, flag, ngen::Immediate(uint16_t((1 << 8) - 1)));
        else
            host_->and_(1, flag, mod.getFlagReg(),
                    ngen::Immediate(uint16_t((1 << 8) - 1)));

        auto region
                = is_df ? new_val[2].df(0)(4, 4, 1) : new_val[1].f(0)(8, 8, 1);
        auto old_region
                = is_df ? new_val[0].df(0)(4, 4, 1) : new_val[0].f(0)(8, 8, 1);
        auto old_save_region = is_df ? old_save[0].df(0)(4, 4, 1)
                                     : old_save[0].f(0)(8, 8, 1);
        int esize = (is_df && size < 64) ? 4 : 8;
        host_->mark(atomic_label);
        host_->emov(esize, old_save_region, old_region);
        auto ne_mod = esize | flag | host_->ne | flag;
        auto eq_mod = esize | flag | host_->eq | flag;
        host_->add(esize, region, old_region, rd.setRegion(4, 4, 1));
        auto cmpwr_mod = mod;
        cmpwr_mod.setPredCtrl(ngen::PredCtrl::None);
        cmpwr_mod |= flag;
        cmpwr.emit(host_, scope, cmpwr_mod, old_region, mem_off_op, old_region);
        host_->cmp(ne_mod, old_save_region, old_region);
        // The previous comparison always fails for NaNs so check for NaNs
        // explictly to prevent an infinite loop.
        host_->cmp(eq_mod, old_region, old_region);
        host_->while_(esize | flag, atomic_label);
    }

    void send(ngen_register_scope_t &scope, const send_t &send_func,
            const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) const {
        send_impl_t spec_impl(send_func);
        auto &mem_off_op = send_t::arg_mem_off(args);
        auto &reg_buf_op = send_t::arg_reg_buf(args);
        auto &mask_op = send_t::arg_mask(args);
        auto &fill_pattern = send_t::arg_fill_pattern(args);

        ngen::InstructionModifier mod = send_func.nmasks();
        dsl_assert(is_pow2(mod.getExecSize()));
        if (attr) mod |= attr.as<instruction_modifier_attr_t>().mod;
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();

        // Zero-out inactive channels unless told not to.
        if (send_func.fill_buf
                && (send_func.is_load() || send_func.is_load_2d())
                && mod.getPredCtrl() != ngen::PredCtrl::None) {
            fill_buf(reg_buf_op, send_func.payload_size(), fill_pattern);
        }

        // Emit send instruction.
        auto rd = send_maybe_make_dense_payload(scope, send_func, reg_buf_op);
        if (!send_func.has_default_slot_mask()) {
            if (mod.getPredCtrl() != ngen::PredCtrl::None) {
                auto flag = mod.getFlagReg();
                if (send_func.slots > 16)
                    flag = ngen::FlagRegister(flag.index() >> 1);
                host_->and_(1, flag, flag, send_func.slot_mask);
            } else {
                auto flag = scope.alloc_flag(send_func.slots);
                host_->emov(1, flag, send_func.slot_mask);
                mod |= flag;
            }
        }
        if ((hw() <= ngen::HW::XeLP && send_func.is_atomic())
                || (hw() == ngen::HW::XeHPG && send_func.is_atomic()
                        && send_func.type.is_qword()
                        && !(host_->hw_info().has_fp64_atomic_support()))) {
            send_atomic_add_emu(
                    scope, send_func, mask_op, mod, mem_off_op.reg_data(), rd);
        } else {
            spec_impl.emit(host_, scope, mod, mem_off_op.reg_data(), rd);
        }
    }

    void reorder(ngen_register_scope_t &scope, const reorder_t &reorder_func,
            const std::vector<ngen_operand_t> &args) const {
        auto &src_op = reorder_t::arg_src_buf(args);
        auto &dst_op = reorder_t::arg_dst_buf(args);

        reorder_impl_t reorder_impl(hw(), reorder_func);
        reorder_impl.emit(
                host_, scope, src_op.reg_buf_data(), dst_op.reg_buf_data());
    }

    void reduce(ngen_register_scope_t &scope, const reduce_t &reduce_func,
            const std::vector<ngen_operand_t> &args) const {
        auto &src_op = reduce_t::arg_src_buf(args);
        auto &dst_op = reduce_t::arg_dst_buf(args);

        reduce_impl_t reduce_impl(hw(), reduce_func, host_->getSIMD());
        reduce_impl.emit(
                host_, scope, src_op.reg_buf_data(), dst_op.reg_buf_data());
    }

    ngen_generator_t *host_;

#if GEMMSTONE_ASSERTIONS
    int bank_conflicts_ = 0;
    int bundle_conflicts_ = 0;
#endif

    object_map_t<alloc_attr_t, bank_conflict_allocation_t> bc_allocations_;

    const size_t max_tracked_header_regs = 8;
    std::vector<int> last_used_header_regs_;
};

// Evaluates expression by emitting instructions with nGEN.
template <typename ngen_generator_t>
class expr_evaluator_t : public ir_visitor_t {
public:
    expr_evaluator_t(ngen_generator_t *host, ngen_register_scope_t &scope)
        : host_(host), expr_binding_(host->expr_binding()), scope_(scope) {}

    constexpr ngen::HW hw() const { return host_->getHardware(); }

    bool is_int_up_convert(const expr_t &e, type_t &type) const {
        auto it = int_up_converts_.find(e);
        if (it == int_up_converts_.end()) return false;
        type = it->second;
        return true;
    }

    // If `dst_operand` is not empty, use its pre-allocated location for the
    // result.
    ngen_operand_t eval(const expr_t &e,
            const ngen_operand_t &dst_operand = ngen_operand_t(),
            bool fill_mask0 = false) {
        if (!dst_operand.is_invalid()) {
            dsl_assert(dst_operand.mod().getExecSize() != 0);
        }
        if (expr_binding_.is_bound(e)) {
            if (!dst_operand.is_invalid()) {
                auto bind = expr_binding_.get(e);
                if (fill_mask0) {
                    dsl_assert(!bind.is_immediate());
                    host_->sel(dst_operand.mod(), dst_operand.reg_data(),
                            bind.reg_data(), 0);
                } else {
                    const auto grf_size = ngen::GRF::bytes(hw());
                    if (hw() >= ngen::HW::Xe3p && bind.is_reg_buf_data()) {
                        auto mod = dst_operand.mod();
                        auto dst = dst_operand.reg_data();
                        auto src = bind.reg_data();
                        auto exec_size = mod.getExecSize();
                        const auto dst_stride = dst.getHS();
                        const auto dst_byte_offset = dst.getByteOffset();
                        const auto dst_type_size = dst.getBytes();
                        const auto dst_bytes = dst_type_size
                                * ((exec_size - 1) * dst_stride + 1);
                        const auto dst_end_byte = dst_byte_offset + dst_bytes;
                        if (dst_end_byte > grf_size) { // Compressed instruction
                            const auto tail_bytes = grf_size - dst_byte_offset;
                            // Index where we cross the grf boundary.
                            const auto cidx = 1
                                    + (tail_bytes / dst_type_size - 1)
                                            / dst_stride;
                            const auto src_width = src.getWidth();
                            const auto src_hs = src.getHS();
                            const auto src_vs = src.getVS();
                            const auto x = cidx % src_width;
                            const auto y = cidx / src_width;
                            const auto src_base = src.getBase()
                                    + (src_hs * x + src_vs * y) / grf_size;
                            const auto dst_base = dst.getBase();
                            if (src_base == dst_base) {
                                auto &ra = scope_.register_allocator();
                                const auto src_type = src.getType();
                                const int nregs = div_up(dst_bytes, grf_size);
                                auto tmp = ra.alloc_range(nregs);
                                auto t = tmp.sub(hw(), 0, src_type)(dst_stride);
                                host_->emov(exec_size, t, src);
                                host_->emov(mod, dst, t);
                                ra.safeRelease(tmp);
                                return dst_operand;
                            }
                        }
                    }
                    host_->emov(dst_operand.mod(), dst_operand, bind);
                }
                return dst_operand;
            }
        } else {
            if (dst_operand.is_invalid()) {
                visit(e);
            } else if (!fill_mask0) {
                expr_binding_.bind_dst(e, dst_operand);
                visit(e);
            } else {
                auto op = eval(e);
                dsl_assert(!op.is_immediate());
                host_->sel(dst_operand.mod(), dst_operand.reg_data(),
                        op.reg_data(), 0);
            }
        }

        return expr_binding_.get(e, /*allow_empty=*/true);
    }

    std::vector<ngen_operand_t> eval(const std::vector<expr_t> &exprs) {
        std::vector<ngen_operand_t> ret;
        for (auto &e : exprs) {
            if (!expr_binding_.is_bound(e)) visit(e);
            ret.push_back(expr_binding_.get(e));
        }
        return ret;
    }

    void _visit(const binary_op_t &obj) override {
        auto dst_op = alloc_dst_op(obj);
        auto mod = dst_op.mod();

        switch (obj.op_kind) {
            case op_kind_t::_and: {
                if (obj.type.is_bool()) {
                    auto has_and_only = [](const expr_t &bin_obj) {
                        auto bin_ops = find_objects<binary_op_t>(bin_obj);
                        for (auto &op : bin_ops) {
                            auto &bin = op.as<binary_op_t>();
                            if (is_cmp_op(bin.op_kind)
                                    && bin.op_kind != op_kind_t::_and)
                                return false;
                        }
                        return true;
                    };

                    auto a_is_var = has_and_only(obj.a);
                    auto b_is_var = has_and_only(obj.b);
                    auto a = b_is_var ? obj.b : obj.a;
                    auto b = b_is_var ? obj.a : obj.b;
                    auto flag_type = obj.type.elems() == 16
                            ? ngen::DataType::uw
                            : ngen::DataType::ud;
                    if (a_is_var && b_is_var) {
                        auto tmp0 = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);
                        auto tmp1 = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);

                        auto tmp_dst = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);
                        auto src0_op = eval(obj.a, tmp0);

                        auto src1_op = eval(obj.b, tmp1);

                        host_->eand(1, tmp_dst, src0_op, src1_op);
                        host_->emov(1, dst_op, tmp_dst);
                    } else if (a_is_var || b_is_var) {
                        auto tmp1 = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);

                        auto tmp0 = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);
                        auto tmp_dst = ngen_operand_t(
                                scope_.alloc_reg_data(to_ir(flag_type)), 1);
                        auto src0_op = eval(a, tmp0);
                        eval(b, ngen_operand_t(dst_op, mod));

                        host_->emov(1, tmp1, dst_op);
                        host_->eand(1, tmp_dst, src0_op, tmp1);
                        host_->emov(1, dst_op, tmp_dst);
                    } else {
                        eval(a, dst_op);
                        eval(b,
                                ngen_operand_t(dst_op,
                                        mod | dst_op.flag_register_mod()));
                    }
                    break;
                }
                // else fall through to the default label.
            }
            default: {
                // Some cases require pre-allocated register regions with
                // special strides for a/b.
                ngen_register_scope_t scope(host_->ra());
                auto a_out_op = maybe_alloc_strided_op(obj.type, obj.a, scope);
                auto b_out_op = maybe_alloc_strided_op(obj.type, obj.b, scope);
                bool is_mul = obj.op_kind == op_kind_t::_mul;
                flag_setter_t no_vs(&allow_vert_stride_region_, !is_mul);
                auto src0_op = eval(obj.a, a_out_op);
                auto src1_op = eval(obj.b, b_out_op);

                if ((src0_op.is_reg_buf_data()
                            && src0_op.reg_buf_data().hs() != 0)
                        || (src1_op.is_reg_buf_data()
                                && src1_op.reg_buf_data().hs() != 0))
                    mod.setExecSize(obj.type.elems());

                ebinary(obj, mod, dst_op, src0_op, src1_op);
                break;
            }
        }

        bind(obj, dst_op);
    }

    void _visit(const bool_imm_t &obj) override {
        // Scalar booleans must never be directly lowered:
        // - Booleans are mapped to flag registers
        // - Flag register stores vector of boolean vectors
        // - All boolean values in IR must be expressed by shuffle_t objects
        // - _visit(shuffle_t *) must properly handle vector of booleans -> flag
        //   register lowering
        stub();
    }

    void _visit(const cast_t &obj) override {
        auto &from_type = obj.expr.type();
        auto &to_type = obj.type;

        dsl_assert(from_type != to_type) << "Equal types are not expected.";

        if (is_const(obj.expr) && !to_type.is_bool()) {
            if (obj.expr.type().is_bool()) {
                bind(obj,
                        to_ngen(expr_t(to_cpp<bool>(obj.expr) ? 1 : 0),
                                to_type));
            } else {
                bind(obj, to_ngen(obj.expr, to_type));
            }
            return;
        }

        auto dst_op = alloc_dst_op(obj);

        // Handle ptr -> u64 and u64 -> ptr casts.
        if (one_of(obj.type, {type_t::u64(), type_t::byte(type::attr_t::ptr)})
                && one_of(obj.expr.type(),
                        {type_t::u64(), type_t::byte(type::attr_t::ptr)})) {
            eval(obj.expr, dst_op);
            bind(obj, dst_op);
            return;
        }

        // Handle integer (down-)conversion, assume bitwise equality in this
        // case. Examples: d <-> ud, d -> w, q -> d.
        bool is_int_convert = from_type.is_scalar() && to_type.is_scalar()
                && from_type.is_int() && to_type.is_int();
        bool is_int_down_convert
                = is_int_convert && from_type.size() >= to_type.size();
        bool is_int_up_convert
                = is_int_convert && from_type.size() < to_type.size();
        if (is_int_down_convert) {
            eval(obj.expr, dst_op.reinterpret(from_type));
            bind(obj, dst_op);
            return;
        }

        auto expr_op = eval(obj.expr);
        auto mod = dst_op.mod();
        if (obj.saturate) mod |= host_->sat;
        host_->emov(mod, dst_op, expr_op);
        if (is_int_up_convert) int_up_converts_.emplace(obj, from_type);
        bind(obj, dst_op);
    }

    void _visit(const float_imm_t &obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const iif_t &obj) override {
        auto dst_op = alloc_dst_op(obj);
        auto cond_op = eval(obj.cond);
        auto true_expr_op = eval(obj.true_expr);
        auto false_expr_op = eval(obj.false_expr);
        auto mod = dst_op.mod();
        host_->esel(mod | cond_op.flag_register_mod(), dst_op, true_expr_op,
                false_expr_op);
        bind(obj, dst_op);
    }

    void _visit(const int_imm_t &obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const load_t &obj) override {
        auto &type = obj.type;
        auto stype = type.scalar();
        auto buf_op = eval(obj.buf);
        auto off_op = eval(obj.off);
        int stride;
        if (obj.has_default_stride()) {
            stride = 1;
        } else {
            dsl_assert(obj.stride % stype.size() == 0);
            stride = obj.stride / stype.size();
        }
        int off = to_cpp<int>(off_op.immediate());
        auto load_rbd = buf_op.reg_buf_data().format(
                off / stype.size(), type.elems(), stride, to_ngen(stype));
        bind(obj, load_rbd);
    }
    void _visit(const ref_t &obj) override {
        auto &type = obj.type;
        auto stype = type.scalar();
        auto buf_op = eval(obj.var);
        int off = obj.off;
        auto load_rbd = buf_op.reg_buf_data().format(
                off, type.elems(), /*stride=*/1, to_ngen(stype));
        bind(obj, load_rbd);
    }

    void _visit(const ptr_t &obj) override {
        auto base_op = eval(obj.base);

        if (obj.off.is(0)) {
            bind(obj, base_op);
            return;
        }

        dsl_assert(base_op.is_reg_buf_data());

        int elem_off = to_cpp<int>(obj.off);
        int byte_off
                = utils::safe_divide(elem_off * obj.type.scalar().bitsize(), 8);
        bind(obj, base_op.reg_buf_data().format(byte_off, ngen::DataType::ub));
    }

    void _visit(const shuffle_t &obj) override {
        int elems = obj.elems();
        if (obj.type.is_bool() && is_shuffle_const(obj)) {
            auto dst_op = alloc_dst_op(obj);
            auto e_shuffle = expr_t(obj);
            dsl_assert(dst_op.is_flag_register()
                    || dst_op.type() == ngen::DataType::uw
                    || dst_op.type() == ngen::DataType::ud)
                    << e_shuffle;
            dsl_assert(!dst_op.is_negated()) << e_shuffle;
            uint32_t flag_mask = 0;
            for (int i = elems - 1; i >= 0; i--) {
                flag_mask <<= 1;
                flag_mask |= (to_cpp<bool>(e_shuffle[i]) ? 1 : 0);
            }
            if (dst_op.mod().getPredCtrl() == ngen::PredCtrl::None) {
                host_->emov(1, dst_op, ngen::Immediate(flag_mask));
            } else {
                dsl_assert(dst_op.mod().getFlagReg().getARFBase()
                        == dst_op.flag_register().getARFBase());
                host_->and_(1, dst_op.flag_register(), dst_op.flag_register(),
                        ngen::Immediate(flag_mask));
            }
            bind(obj, dst_op);
            return;
        }

        if (obj.is_broadcast()) {
            if (obj.type.is_bool()) {
                auto dst_op = alloc_dst_op(obj);
                eval(obj.vec[0], dst_op);
                bind(obj, dst_op);
            } else {
                auto scalar_op = eval(obj.vec[0]);
                bind(obj, scalar_op);
            }
            return;
        }

        if (try_region_peephole(obj)) return;
        if (try_packed_int_peephole(obj)) return;

        // tuples: <offset, length, idx>
        std::vector<std::tuple<int, int, int>> chunks;
        for (int i = 0; i < elems; i++) {
            int idx = obj.idx[i];
            if (chunks.empty() || std::get<2>(chunks.back()) != idx) {
                chunks.emplace_back(i, 1, idx);
            } else {
                std::get<1>(chunks.back())++;
            }
        }

        auto dst_op = alloc_dst_op(obj);
        auto op = ngen_operand_t(scope_.alloc_reg_data(type_t::u16(1)), 1);
        for (auto &chunk : chunks) {
            int off = std::get<0>(chunk);
            int length = std::get<1>(chunk);
            int idx = std::get<2>(chunk);
            // Split length into powers of two.
            while (length > 0) {
                int exec_size = (1 << ilog2(length));
                if (obj.type.is_bool()) {
                    dsl_assert(off % 8 == 0)
                            << "expected mask offset to be multiple of 8";
                    auto chunk_op = op.reg_buf_data().subregister(
                            off / 8, ngen::DataType::b)(1);
                    eval(obj.vec[idx], ngen_operand_t(dst_op, exec_size));
                    host_->emov(1, chunk_op, dst_op.flag_register().b(0));
                } else {
                    auto chunk_op = dst_op.sub_reg_data(off, exec_size);
                    eval(obj.vec[idx], ngen_operand_t(chunk_op, exec_size));
                }
                length -= exec_size;
                off += exec_size;
            }
        }
        if (obj.type.is_bool()) host_->emov(1, dst_op, op);
        bind(obj, dst_op);
    }

    void _visit(const ternary_op_t &obj) override {
        flag_setter_t no_vs(&allow_vert_stride_region_, false);
        auto dst_op = alloc_dst_op(obj);
        auto mod = dst_op.mod();
        auto src0_op = eval(obj.a);
        auto src1_op = eval(obj.b);
        auto src2_op = eval(obj.c);
        switch (obj.op_kind) {
            case op_kind_t::_add3:
                host_->eadd3(mod, dst_op, src0_op, src1_op, src2_op);
                break;
            case op_kind_t::_mad:
                host_->emad(mod, dst_op, src0_op, src1_op, src2_op);
                break;
            case op_kind_t::_idiv:
                host_->eidiv(mod, dst_op.reg_data(), ngen::Subregister(),
                        src0_op.reg_data(), src1_op.reg_data(),
                        src2_op.reg_data());
                break;
            case op_kind_t::_imod:
                host_->eidiv(mod, ngen::Subregister(), dst_op.reg_data(),
                        src0_op.reg_data(), src1_op.reg_data(),
                        src2_op.reg_data());
                break;
            default: stub();
        }
        bind(obj, dst_op);
    }

    void _visit(const unary_op_t &obj) override {
        dsl_assert(obj.op_kind == op_kind_t::_minus);
        ngen_operand_t a_op;
        a_op = eval(obj.a);
        bind(obj, -a_op);
    }

    void _visit(const var_t &obj) override {
        dsl_assert(expr_binding_.is_bound(obj))
                << "Variable is not defined: " << expr_t(obj);
    }

private:
    struct flag_setter_t {
        flag_setter_t(bool *flag, bool value) : flag(flag), old(*flag) {
            *flag = value;
        }
        ~flag_setter_t() { *flag = old; }

        bool *flag;
        bool old;
    };

    ngen_operand_t alloc_dst_op(const expr_t &e) {
        dsl_assert(!expr_binding_.is_bound(e)) << "Already evaluated: " << e;
        if (expr_binding_.is_dst_bound(e)) return expr_binding_.get_dst(e);

        // Expression is not bound yet, allocate new storage and bind.
        ngen_operand_t op;
        if (e.type().is_bool()) {
            int elems = std::max(
                    e.type().elems(), std::max(16, host_->getSIMD()));
            op = ngen_operand_t(scope_.alloc_flag(elems), elems);
        } else {
            op = ngen_operand_t(
                    scope_.alloc_reg_data(e.type()), e.type().elems());
        }
        expr_binding_.bind_dst(e, op);
        return op;
    }

    // Pre-allocates a strided register region for expression `e` if needed.
    ngen_operand_t maybe_alloc_strided_op(const type_t &res_type,
            const expr_t &e, ngen_register_scope_t &scope) {
        // Need q-strided region for `e` if res_type is q/uq and `e` is of a
        // sub-q data type and not a scalar.
        if (e.type().is_scalar()) return ngen_operand_t();
        if (!res_type.scalar().is_x64()) return ngen_operand_t();
        if (e.type().scalar().is_x64()) return ngen_operand_t();

        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) return ngen_operand_t();

        int stride = res_type.bitsize() / e.type().bitsize();
        return ngen_operand_t(
                scope.alloc_reg_data(e.type(), stride), e.type().elems());
    }

    void bind(const expr_t &e, const ngen_operand_t &op) {
        if (!expr_binding_.is_dst_bound(e)) {
            expr_binding_.bind(e, op);
            return;
        }
        auto dst_op = expr_binding_.get_dst(e);
        if (dst_op == op) {
            expr_binding_.bind(e, op);
            return;
        }
        // Expression is already bound, move to the location it was bound to.
        // This is required for immediate values - they are bound as is but
        // sometimes we need them to be moved to registers.
        host_->emov(dst_op.mod(), dst_op, op);
        expr_binding_.bind(e, dst_op);
    }

    void ebinary(const binary_op_t &obj, const ngen::InstructionModifier &mod,
            const ngen_operand_t &_dst, const ngen_operand_t &_src0,
            const ngen_operand_t &_src1) {
        auto &dst = _dst;
        auto src0 = _src0;
        auto src1 = _src1;
        align_src_dst_offset(host_, scope_, mod, dst, src0);
        align_src_dst_offset(host_, scope_, mod, dst, src1, true);
        switch (obj.op_kind) {
            case op_kind_t::_add: host_->eadd(mod, dst, src0, src1); break;
            case op_kind_t::_sub: host_->eadd(mod, dst, src0, -src1); break;
            case op_kind_t::_mul: host_->emul(mod, dst, src0, src1); break;
            case op_kind_t::_div: host_->ediv(mod, dst, src0, src1); break;
            case op_kind_t::_mod: host_->emod(mod, dst, src0, src1); break;
            case op_kind_t::_shl: host_->eshl(mod, dst, src0, src1); break;
            case op_kind_t::_shr: host_->eshr(mod, dst, src0, src1); break;
            case op_kind_t::_min: host_->emin(mod, dst, src0, src1); break;
            case op_kind_t::_max: host_->emax(mod, dst, src0, src1); break;
            case op_kind_t::_ge:
            case op_kind_t::_gt:
            case op_kind_t::_le:
            case op_kind_t::_lt:
            case op_kind_t::_eq:
            case op_kind_t::_ne: {
                dsl_assert(!dst.is_negated())
                        << "Destination can't be negated.";
                ngen::InstructionModifier cmp_mod = mod;
                if (!src0.is_reg_data()) {
                    cmp_mod |= cmp_op_to_ngen(negate_cmp_op(obj.op_kind));
                    cmp_mod |= dst.flag_register();
                    host_->ecmp(cmp_mod, src1, src0);
                } else {
                    cmp_mod |= cmp_op_to_ngen(obj.op_kind);
                    cmp_mod |= dst.flag_register();
                    host_->ecmp(cmp_mod, src0, src1);
                }
                break;
            }
            case op_kind_t::_and: host_->eand(mod, dst, src0, src1); break;
            case op_kind_t::_prelu: {
                int grf_size = ngen::GRF::bytes(hw());
                int esize = mod.getExecSize();
                int off = src0.reg_data().getOffset();
                int regs = div_up(esize * int(sizeof(float)) + off, grf_size);
                auto temp = scope_.alloc_reg_buf_data(regs).format(
                        off, esize, 1, ngen::DataType::f);
                host_->emul(mod, temp, dst, src1);
                // Workaround for regioning restriction.
                if (esize == 2) {
                    host_->csel(mod | host_->le, dst.reg_data(),
                            temp.subregister(0)(1),
                            dst.reg_buf_data().subregister(0)(1),
                            dst.reg_buf_data().subregister(0)(1));
                } else {
                    host_->csel(mod | host_->le, dst.reg_data(), temp,
                            dst.reg_data(), dst.reg_data());
                }

                break;
            }
            default: dsl_error() << "Unknown kind: " << to_string(obj.op_kind);
        }
    }

    struct conjunct_t {
        conjunct_t(op_kind_t op, ngen_operand_t a, ngen_operand_t b)
            : op_(op), a_(std::move(a)), b_(std::move(b)) {}
        op_kind_t op_;
        ngen_operand_t a_, b_;
    };

    void split_by_and(const expr_t &e, std::vector<conjunct_t> &cv, type_t ty) {
        if (auto bin = e.as_ptr<binary_op_t>()) {
            if (bin->op_kind == op_kind_t::_and) {
                split_by_and(bin->a, cv, ty);
                split_by_and(bin->b, cv, ty);
            } else
                cv.emplace_back(bin->op_kind, eval(bin->a), eval(bin->b));
        } else {
            auto cast = cast_t::make(ty, e);
            cv.emplace_back(op_kind_t::undef, eval(cast), ngen_operand_t());
        }
    }

    ngen_operand_t try_process_negated_flags(const expr_t &e) {
        ngen_operand_t retn;
        auto cast = e.as<unary_op_t>().a.as_ptr<cast_t>();
        if (cast && cast->expr.type().is_bool()) {
            int elems = cast->expr.type().elems();
            ngen_operand_t flags(scope_.alloc_flag(elems), e.type().elems());
            retn = alloc_dst_op(e);
            auto mod = retn.mod();
            auto ar_op = [&](ngen::InstructionModifier m, const conjunct_t &c) {
                if (c.op_ != op_kind_t::undef)
                    host_->ecmp(m | cmp_op_to_ngen(c.op_), retn, c.a_, c.b_);
                else
                    host_->emov(m, retn, -c.a_);
            };
            std::vector<conjunct_t> cv;
            split_by_and(cast->expr, cv, cast->type);
            for (size_t i = 0; i < cv.size(); i++) {
                if (cv[i].op_ == op_kind_t::undef) {
                    dsl_assert(i == cv.size() - 1);
                }
            }

            ar_op(mod, cv[0]);
            mod |= flags.flag_register();
            for (size_t i = 1; i < cv.size(); i++)
                ar_op(mod, cv[i]);
            retn = -retn;
        }
        return retn;
    }

    bool try_region_peephole(const shuffle_t &obj) {
        int elems = obj.elems();
        if (elems % 2 != 0) return false;

        std::vector<ngen_operand_t> vec(obj.vec.size());
        ngen::DataType data_type = ngen::DataType::invalid;
        for (size_t i = 0; i < vec.size(); i++) {
            if (!obj.vec[i].is<load_t>()) return false;
            vec[i] = eval(obj.vec[i]);
            dsl_assert(vec[i].is_reg_buf_data()) << obj.vec[i];
            auto &rbd = vec[i].reg_buf_data();
            if (data_type == ngen::DataType::invalid) {
                data_type = rbd.type();
                continue;
            }
            if (data_type != rbd.type()) return false;
        }

        int grf_size = ngen::GRF::bytes(hw());
        auto diff_bytes
                = [&](const ngen_operand_t &a, const ngen_operand_t &b) {
            auto a_rd = a.reg_data();
            auto b_rd = b.reg_data();
            int a_off = a_rd.getBase() * grf_size + a_rd.getByteOffset();
            int b_off = b_rd.getBase() * grf_size + b_rd.getByteOffset();
            return b_off - a_off;
        };

        int type_size = ngen::getBytes(data_type);
        int stride_bytes = diff_bytes(vec[0], vec[1]);
        if (stride_bytes < 0 || stride_bytes % type_size != 0) return false;

        // Pattern 1: [xxyy]
        auto is_xxyy = [&]() {
            if (!allow_vert_stride_region_) return false;
            for (int i = 0; i < elems / 2; i++) {
                if (obj.idx[i] != 0) return false;
                if (obj.idx[i + elems / 2] != 1) return false;
            }
            return true;
        };
        if (is_xxyy()) {
            auto &rbd = vec[0].reg_buf_data();
            auto rd = rbd.reg_data();
            int regs = div_up(stride_bytes * 2, grf_size);
            if (regs > 2) return false;
            rd.setRegion(stride_bytes / type_size, elems / 2, 0);
            reg_buf_t rb(hw(), ngen::GRFRange(rd.getBase(), regs));
            bind(obj, reg_buf_data_t(rb, rd));
            return true;
        }

        // Pattern 2: [xyxy]
        auto is_xyxy = [&]() {
            for (int i = 0; i < elems / 2; i++) {
                if (obj.idx[i] != i) return false;
                if (obj.idx[i] != obj.idx[i + elems / 2]) return false;
                if (i > 0 && diff_bytes(vec[i - 1], vec[i]) != stride_bytes)
                    return false;
            }
            return true;
        };
        if (is_xyxy()) {
            auto &rbd = vec[0].reg_buf_data();
            auto rd = rbd.reg_data();
            int regs = div_up(stride_bytes * elems / 2, grf_size);
            if (regs > 2) return false;
            rd.setRegion(0, elems / 2, stride_bytes / type_size);
            reg_buf_t rb(hw(), ngen::GRFRange(rd.getBase(), regs));
            bind(obj, reg_buf_data_t(rb, rd));
            return true;
        }

        return false;
    }

    bool try_packed_int_peephole(const shuffle_t &obj) {
        if (!obj.type.is_x32()) return false;
        if (!one_of(obj.elems(), {8, 16})) return false;

        int64_t int_min = std::numeric_limits<int>::min();
        int64_t int_max = std::numeric_limits<int>::max();
        int vec_size = (int)obj.vec.size();
        std::vector<int> vec(vec_size);
        for (int i = 0; i < vec_size; i++) {
            if (!is_const(obj.vec[i])) return false;
            int64_t value = to_cpp<int64_t>(obj.vec[i]);
            if (value < int_min || value > int_max) return false;
            vec[i] = (int)value;
        }

        const int esize = 8;

        auto half_same = [&](int off) {
            return std::equal(obj.idx.begin() + off + 1,
                    obj.idx.begin() + off + esize, obj.idx.begin() + off);
        };
        // If true, the case is too trivial for :v/:uv to justify the overhead
        if (half_same(0) && half_same(esize % obj.elems())) return false;

        int vec_min = *std::min_element(vec.begin(), vec.end());
        int vec_max = *std::max_element(vec.begin(), vec.end());

        int factor = vec_max - vec_min;
        for (int i = 0; i < vec_size; i++)
            factor = gcd(vec[i] - vec_min, factor);

        // XXX: Disabled due to an emulation limitation: vector multiplication
        // by dword constant is not implemented yet.
        int64_t s16_min = std::numeric_limits<int16_t>::min();
        int64_t s16_max = std::numeric_limits<int16_t>::max();
        if (factor < s16_min || factor > s16_max) return false;

        auto check_range = [&](int f, int m, int a, int b) {
            for (int i = 0; i < vec_size; i++) {
                int d = (vec[i] - m) / f;
                if (d < a || d > b) return false;
            }
            return true;
        };

        bool use_uv = false, use_v = false;
        for (int f : {1, factor, -factor}) {
            use_uv = check_range(f, vec_min, 0, 15);
            use_v = check_range(f, vec_min, -8, 7);
            if (use_uv || use_v) {
                factor = f;
                break;
            }
        }
        if (!use_uv && !use_v) return false;
        if (vec_min % factor == 0) {
            bool new_use_uv = check_range(factor, 0, 0, 15);
            bool new_use_v = check_range(factor, 0, -8, 7);
            if (new_use_uv || new_use_v) {
                vec_min = 0;
                use_uv = new_use_uv;
                use_v = new_use_v;
            }
        }

        auto set_packed = [](uint32_t &packed, int8_t value, int idx) {
            uint32_t v = (value >= 0 ? value : ((value & 0x7) | 0x8));
            packed = packed | (v << idx * 4);
        };

        auto dst = alloc_dst_op(obj);
        auto &dst_rbd = dst.reg_buf_data();
        int dst_stride = dst_rbd.hs();
        int w_size = sizeof(uint16_t);
        int grf_size = ngen::GRF::bytes(hw());
        auto tmp = scope_.alloc_reg_buf_data(1);
        auto w_type = (use_uv) ? ngen::DataType::uw : ngen::DataType::w;
        for (int i = 0; i < obj.elems(); i += esize) {
            uint32_t packed = 0;
            for (int j = 0; j < esize; j++)
                set_packed(packed, (vec[obj.idx[i + j]] - vec_min) / factor, j);
            auto t = tmp.format(i, esize, 1, w_type);
            host_->emov(esize, t,
                    (use_uv) ? ngen::Immediate::uv(packed)
                             : ngen::Immediate::v(packed));
        }
        auto d = dst_rbd.format(0, obj.elems(), dst_stride);
        auto t = tmp.format(0, obj.elems(), 1, w_type);
        reg_buf_data_t t_strided;
        bool align_with_dst = false;
        if (hw() == ngen::HW::Xe3p) align_with_dst = true;
        if (align_with_dst) {
            int w_stride = dst_stride * (ngen::getBytes(dst.type()) / w_size);
            int tmp_strided_regs
                    = div_up(obj.elems() * w_size * w_stride, grf_size);
            auto tmp_strided = scope_.alloc_reg_buf_data(tmp_strided_regs);
            t_strided = tmp_strided.format(0, obj.elems(), w_stride, w_type);
            host_->emov(obj.elems(), t_strided, t);
        } else {
            t_strided = std::move(t);
        }
        if (factor != 1) {
            host_->emul(obj.elems(), d, t_strided, ngen::Immediate(factor));
        }
        if (factor == 1 || vec_min != 0) {
            host_->eadd(obj.elems(), d, (factor == 1) ? t_strided : d,
                    ngen::Immediate(vec_min));
        }
        bind(obj, dst);
        return true;
    }

    ngen_generator_t *host_;
    expr_binding_t expr_binding_;
    ngen_register_scope_t &scope_;
    bool allow_vert_stride_region_ = true;

    object_eq_map_t<expr_t, type_t> int_up_converts_;
};

// This interface is relied on by the oneDNN ir_kernel_t extensions. Do not
// modify without propagating this change.
template <typename GeneratorT>
void convert_ir_to_ngen_impl(const stmt_t &body, GeneratorT &host,
        const walk_order_t *kernel_grid_walk_order = nullptr) {
    host.comment("Prologue");
    host.generate_prologue();

    host.bind_external_vars(body, kernel_grid_walk_order);
    if (kernel_grid_walk_order)
        host.bind_kernel_grid_walk_order(*kernel_grid_walk_order);

    host.comment("IR");
    ir_to_ngen_t<GeneratorT> visitor(&host);
    visitor.visit(body);

    host.comment("Epilogue");
    host.generate_epilogue();
}

template <typename GeneratorT>
void convert_ir_to_ngen(const stmt_t &body, GeneratorT &host,
        const walk_order_t *kernel_grid_walk_order = nullptr) {
    return convert_ir_to_ngen_impl(body, host, kernel_grid_walk_order);
}

#ifdef NGEN_ASM
template <>
void convert_ir_to_ngen<ir_asm_generator_t>(const stmt_t &body,
        ir_asm_generator_t &host, const walk_order_t *kernel_grid_walk_order) {

    return convert_ir_to_ngen_impl(body, host, kernel_grid_walk_order);
}
#endif

kernel_attr_t::kernel_attr_t(const stmt_t &s) {
    class visitor_t : public ir_visitor_t {
    public:
        visitor_t(kernel_attr_t &attrs) : attrs(attrs) {};

        void _visit(const func_call_t &obj) override {
            auto &func = obj.func;
            auto *dpas = func.as_ptr<dpas_t>();
            auto *send = func.as_ptr<send_t>();
            if (dpas)
                attrs.has_dpas = true;
            else if (send && send->is_atomic())
                attrs.has_send_atomics = true;
            else if (func.is_same(funcs::signal_func()))
                attrs.has_signal_header = true;
            else if (func.is_same(funcs::barrier_func()))
                attrs.has_signal_header = true;
        }

        kernel_attr_t &attrs;
    };

    visitor_t(*this).visit(s);
}

// This interface is relied on by the oneDNN ir_kernel_t extensions. Do not
// modify without propagating this change.
ngen::NEOInterfaceHandler generate_ngen_interface(
        const kernel::iface_t &kernel_iface, const kernel::options_t &options,
        const stmt_t &kernel_body) {

    ngen::NEOInterfaceHandler interface(options.hw());
    interface.externalName(kernel_iface.kernel_name());
    interface.requireLocalID(3);
    interface.requireLocalSize();
    interface.requireGRF(options.regs());
    interface.requireSIMD(options.simd());
    interface.requireBarrier();
    auto kernel_attrs = kernel_attr_t(kernel_body);

    // Allow dpas override to avoid context switch overhead on XeHPG
    if (kernel_attrs.has_dpas || options.require_dpas())
        interface.requireDPAS();
    if (kernel_attrs.has_send_atomics) interface.requireGlobalAtomics();

    if (options.hw() == ngen::HW::Xe3p && !options.hw().efficient_64_bit())
        interface.setEfficient64Bit(false);

    for (size_t i = 0; i < kernel_iface.nargs(); i++) {
        auto &name = kernel_iface[i].as<var_t>().name;
        auto &type = kernel_iface[i].type();
        if (type.is_ptr() || type.is_gm()) {
            interface.newArgument(name, ngen::ExternalArgumentType::GlobalPtr,
                    ngen::GlobalAccessType::Stateless);
        } else {
            interface.newArgument(name, to_ngen(type));
        }
    }

    int slm_size = alloc_manager_t(kernel_body).slm_size();
    interface.requireSLM(slm_size);

    interface.finalize();
    return interface;
}
} // namespace ir

#ifdef GEMMSTONE_BUILD_12LP
#define GEMMSTONE_XELP_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XELP_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_12HP
#define GEMMSTONE_XEHP_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XEHP_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_12P7
#define GEMMSTONE_XEHPG_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XEHPG_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_12P8
#define GEMMSTONE_XEHPC_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XEHPC_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_XE2
#define GEMMSTONE_XE2_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XE2_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_XE3
#define GEMMSTONE_XE3_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XE3_ISA(...)
#endif

#ifdef GEMMSTONE_BUILD_XE3P
#define GEMMSTONE_XE3P_ISA(...) __VA_ARGS__
#else
#define GEMMSTONE_XE3P_ISA(...)
#endif

#define GEMMSTONE_OTHER_CASES

#define GPU_HW_CASE_(hw) \
    case ngen::HW::hw: { \
        GPU_HW_CASE(ngen::HW::hw); \
        break; \
    }

#define GEMMSTONE_HW_SWITCH(hw) \
    switch (hw) { \
        GEMMSTONE_XELP_ISA(GPU_HW_CASE_(XeLP)); \
        GEMMSTONE_XEHP_ISA(GPU_HW_CASE_(XeHP)); \
        GEMMSTONE_XEHPG_ISA(GPU_HW_CASE_(XeHPG)); \
        GEMMSTONE_XEHPC_ISA(GPU_HW_CASE_(XeHPC)); \
        GEMMSTONE_XE2_ISA(GPU_HW_CASE_(Xe2)); \
        GEMMSTONE_XE3_ISA(GPU_HW_CASE_(Xe3)); \
        GEMMSTONE_XE3P_ISA(GPU_HW_CASE_(Xe3p)); \
        GEMMSTONE_OTHER_CASES; \
        default: dsl_assert(false) << "Unexpected GPU architecture"; \
    }

#ifdef GEMMSTONE_WITH_ASM_RUNTIME
std::string make_asm(const kernel_t &ir_kernel) {
    auto &iface = ir_kernel.iface;
    auto &options = ir_kernel.options;
    auto &body = ir_kernel.body;
    auto &debug_cfg = ir_kernel.debug_cfg;

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            iface, options, body);

    ir::ir_to_ngen_generator_t<ir::ngen_asm_code_generator_with_interface_t> g(
            iface, options, debug_cfg);
    g.set_interface(interface);
    ir::convert_ir_to_ngen(body, g);
    return g.str();
}
#endif

#ifdef GEMMSTONE_WITH_BINARY_RUNTIME
template <ngen::HW hw>
using binary_gen_t = ir::ir_to_ngen_generator_t<ngen::ELFCodeGenerator<hw>>;

std::vector<uint8_t> make_binary(const kernel_t &ir_kernel) {
    auto &iface = ir_kernel.iface;
    auto &options = ir_kernel.options;
    auto &body = ir_kernel.body;
    auto &debug_cfg = ir_kernel.debug_cfg;
    maybe_unused(debug_cfg);

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            iface, options, body);

#define GPU_HW_CASE(hw) \
    binary_gen_t<(hw)> g(iface, options, debug_cfg); \
    g.setInterface(interface); \
    ir::convert_ir_to_ngen(body, g); \
    return g.getBinary(); \
    break;

    GEMMSTONE_HW_SWITCH(options.hw().ngen_hw());
#undef GPU_HW_CASE
    return {};
}
#endif

#ifdef GEMMSTONE_WITH_SYCL_RUNTIME
template <ngen::HW hw>
using sycl_gen_t = ir::ir_to_ngen_generator_t<ngen::SYCLCodeGenerator<hw>>;

GEMMSTONE_XELP_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::XeLP>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::XeLP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHP_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::XeHP>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::XeHP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPG_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::XeHPG>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::XeHPG> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPC_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::XeHPC>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::XeHPC> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE2_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::Xe2>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::Xe2> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::Xe3>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::Xe3> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3P_ISA(
        template void ir::convert_ir_to_ngen<sycl_gen_t<ngen::HW::Xe3p>>(
                const stmt_t &body, sycl_gen_t<ngen::HW::Xe3p> &host,
                const walk_order_t *kernel_grid_walk_order));

::sycl::kernel make_kernel(
        const kernel_t &ir_kernel, ::sycl::context ctx, ::sycl::device dev) {
    auto &iface = ir_kernel.iface;
    auto &options = ir_kernel.options;
    auto &body = ir_kernel.body;
    auto &debug_cfg = ir_kernel.debug_cfg;
    maybe_unused(debug_cfg);

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            iface, options, body);
    std::optional<::sycl::kernel> kernel;

#define GPU_HW_CASE(hw) \
    sycl_gen_t<(hw)> g(iface, options, debug_cfg); \
    g.setInterface(interface); \
    ir::convert_ir_to_ngen(body, g); \
    kernel = g.getKernel(ctx, dev); \
    break;

    GEMMSTONE_HW_SWITCH(options.hw().ngen_hw());
#undef GPU_HW_CASE
    return kernel.value();
}
#endif

#ifdef GEMMSTONE_WITH_L0_RUNTIME
template <ngen::HW hw>
using l0_gen_t = ir::ir_to_ngen_generator_t<ngen::LevelZeroCodeGenerator<hw>>;

GEMMSTONE_XELP_ISA(
        template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::XeLP>>(
                const stmt_t &body, l0_gen_t<ngen::HW::XeLP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHP_ISA(
        template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::XeHP>>(
                const stmt_t &body, l0_gen_t<ngen::HW::XeHP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPG_ISA(
        template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::XeHPG>>(
                const stmt_t &body, l0_gen_t<ngen::HW::XeHPG> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPC_ISA(
        template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::XeHPC>>(
                const stmt_t &body, l0_gen_t<ngen::HW::XeHPC> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE2_ISA(template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::Xe2>>(
        const stmt_t &body, l0_gen_t<ngen::HW::Xe2> &host,
        const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3_ISA(template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::Xe3>>(
        const stmt_t &body, l0_gen_t<ngen::HW::Xe3> &host,
        const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3P_ISA(
        template void ir::convert_ir_to_ngen<l0_gen_t<ngen::HW::Xe3p>>(
                const stmt_t &body, l0_gen_t<ngen::HW::Xe3p> &host,
                const walk_order_t *kernel_grid_walk_order));

LevelZeroKernelAndModule make_kernel(const kernel_t &ir_kernel,
        ze_context_handle_t ctx, ze_device_handle_t dev) {
    auto &iface = ir_kernel.iface;
    auto &options = ir_kernel.options;
    auto &body = ir_kernel.body;
    auto &debug_cfg = ir_kernel.debug_cfg;
    maybe_unused(debug_cfg);

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            iface, options, body);

#define GPU_HW_CASE(hw) \
    l0_gen_t<(hw)> g(iface, options, debug_cfg); \
    g.setInterface(interface); \
    ir::convert_ir_to_ngen(body, g); \
    auto km = g.getModuleAndKernel(ctx, dev); \
    return {km.second, km.first};

    GEMMSTONE_HW_SWITCH(options.hw().ngen_hw());
#undef GPU_HW_CASE
    return {};
}
#endif

#ifdef GEMMSTONE_WITH_OPENCL_RUNTIME
template <ngen::HW hw>
using ocl_gen_t = ir::ir_to_ngen_generator_t<ngen::OpenCLCodeGenerator<hw>>;

GEMMSTONE_XELP_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::XeLP>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::XeLP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHP_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::XeHP>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::XeHP> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPG_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::XeHPG>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::XeHPG> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XEHPC_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::XeHPC>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::XeHPC> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE2_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::Xe2>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::Xe2> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::Xe3>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::Xe3> &host,
                const walk_order_t *kernel_grid_walk_order));
GEMMSTONE_XE3P_ISA(
        template void ir::convert_ir_to_ngen<ocl_gen_t<ngen::HW::Xe3p>>(
                const stmt_t &body, ocl_gen_t<ngen::HW::Xe3p> &host,
                const walk_order_t *kernel_grid_walk_order));

cl_kernel make_kernel(
        const kernel_t &ir_kernel, cl_context ctx, cl_device_id dev) {
    auto &iface = ir_kernel.iface;
    auto &options = ir_kernel.options;
    auto &body = ir_kernel.body;
    auto &debug_cfg = ir_kernel.debug_cfg;
    maybe_unused(debug_cfg);

    ngen::NEOInterfaceHandler interface = generate_ngen_interface(
            iface, options, body);

#define GPU_HW_CASE(hw) \
    ocl_gen_t<(hw)> g(iface, options, debug_cfg); \
    g.setInterface(std::move(interface)); \
    ir::convert_ir_to_ngen(body, g); \
    return g.getKernel(ctx, dev);

    GEMMSTONE_HW_SWITCH(options.hw().ngen_hw());
#undef GPU_HW_CASE
    return {};
}
#endif

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif
