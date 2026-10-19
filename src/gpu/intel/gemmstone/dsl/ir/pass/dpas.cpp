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

#include "dsl/ir/pass/dpas.hpp"
#include "dsl/ir/fma.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/ir/pass/trace.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class mul_mutator_t : public ir_mutator_t {
public:
    struct entry_t {
        stmt_t stmt;
        bool is_dpas = false;
        int dpas_sdepth = 0;
        int dpas_rcount = 0;

        bool is_dpas_8x8() const {
            return is_dpas && dpas_sdepth == 8 && dpas_rcount == 8;
        }
    };

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.label != stmt_label_t::mul()) return ir_mutator_t::_mutate(obj);
        auto body = mutate_mul(obj.body);
        return stmt_group_t::make(obj.label, body);
    }

    stmt_t mutate_mul(const stmt_t &stmt) const {
        auto stmt_vec = flatten_statements(stmt);
        std::vector<entry_t> entries;
        for (auto &s : stmt_vec) {
            entries.emplace_back();
            auto &e = entries.back();
            e.stmt = s;
            e.is_dpas = is_func_call<dpas_t>(s) && !dpas_t::is_dp4a_call(s);
            if (e.is_dpas) {
                auto &dpas = s.as<func_call_t>().func.as<dpas_t>();
                e.dpas_sdepth = dpas.sdepth;
                e.dpas_rcount = dpas.rcount;
            }
        }
        return mutate_mul_impl(entries);
    }

    virtual stmt_t mutate_mul_impl(const std::vector<entry_t> &entries) const
            = 0;
};

class dpas_atomic_mutator_t : public mul_mutator_t {
public:
    stmt_t mutate_mul_impl(const std::vector<entry_t> &entries) const override {
        stmt_t ret;
        auto is_dpas_fwd = [&](size_t i) {
            auto &e = entries[i];
            if (!e.is_dpas) return false;
            auto &call = e.stmt.as<func_call_t>();
            auto *attr = call.attr.as_ptr<instruction_modifier_attr_t>();
            return attr && attr->mod.isFwd();
        };
        auto can_chain
                = [&](size_t i, const expr_t &src1, const expr_t &src2_base) {
            if (is_dpas_fwd(i)) return true;
            if (i + 1 >= entries.size()) return false;
            auto &next = entries[i + 1];
            if (!next.is_dpas) return false;
            auto &next_src1 = dpas_t::arg_src1(next.stmt);
            auto &next_src2 = dpas_t::arg_src2(next.stmt);
            auto &next_src2_base = next_src2.as<ptr_t>().base;
            if (next_src1.is_equal(src1) && next_src2_base.is_equal(src2_base))
                return true;
            // If the current dpas is Fwd-chained with the previous one then
            // use previous/next buffers to decide if it's Atomic-chain
            // compatible.
            // Example:
            //     dpas8x8.x16(c[0], c[0], b[0], a[0]) {Atomic, Fwd}        <- previous dpas
            //     dpas8x8.x16(c[0], c[0], b[1024], a[1024]) {Atomic}
            //     dpas8x8.x16(c[512], c[512], b[0], a[256]) {Atomic, Fwd}  <- next dpas (src1 is the same, src2_base is the same)
            if (i > 0 && is_dpas_fwd(i - 1)) {
                auto &prev = entries[i - 1];
                auto &prev_src1 = dpas_t::arg_src1(prev.stmt);
                auto &prev_src2 = dpas_t::arg_src2(prev.stmt);
                auto &prev_src2_base = prev_src2.as<ptr_t>().base;
                if (prev_src1.is_equal(next_src1)
                        && prev_src2_base.is_equal(next_src2_base))
                    return true;
            }
            return false;
        };
        for (size_t i = 0; i < entries.size(); i++) {
            auto s = entries[i].stmt;
            if (entries[i].is_dpas) {
                auto &src1 = dpas_t::arg_src1(entries[i].stmt);
                auto &src2 = dpas_t::arg_src2(entries[i].stmt);
                auto &src2_base = src2.as<ptr_t>().base;
                if (can_chain(i, src1, src2_base)) {
                    auto atomic_attr = instruction_modifier_attr_t::make(
                            ngen::InstructionModifier(
                                    ngen::ThreadCtrl::Atomic));
                    s = atomic_attr.apply_to(s);
                }
            }
            ret = ret.append(s);
        }
        return ret;
    }
};

class dpas_fwd_mutator_t : public mul_mutator_t {
public:
    stmt_t mutate_mul_impl(
            const std::vector<entry_t> &_entries) const override {
        auto entries = _entries;
        int nentries = (int)entries.size();
        stmt_t ret;
        for (int i = 0; i < nentries; i++) {
            auto &ei = entries[i];
            if (ei.stmt.is_empty()) continue;
            if (!ei.is_dpas_8x8()) {
                ret = ret.append(ei.stmt);
                continue;
            }
            auto &cur_dst = dpas_t::arg_dst(ei.stmt);
            int fwd_idx = -1;
            for (int j = i + 1; j < nentries; j++) {
                auto &ej = entries[j];
                if (ej.stmt.is_empty() || !ej.is_dpas_8x8()) continue;
                auto &dst = dpas_t::arg_dst(ej.stmt);
                auto &src0 = dpas_t::arg_src0(ej.stmt);
                if (dst.is_equal(cur_dst) && src0.is_equal(cur_dst)) {
                    fwd_idx = j;
                    break;
                }
            }
            if (fwd_idx != -1) {
                auto tmp_mod = ngen::InstructionModifier();
                tmp_mod.setBranchCtrl(true);
                auto fwd_attr = instruction_modifier_attr_t::make(tmp_mod);
                auto s = ei.stmt;
                s = fwd_attr.apply_to(s);
                ret = ret.append(s);
                ret = ret.append(entries[fwd_idx].stmt);
                entries[fwd_idx].stmt = stmt_t();
                continue;
            }
            ret = ret.append(ei.stmt);
        }
        return ret;
    }
};

stmt_t inject_dpas_fwd(const stmt_t &stmt) {
    return dpas_fwd_mutator_t().mutate(stmt);
}

stmt_t inject_dpas_atomic(const stmt_t &stmt, bool filter_by_label) {
    if (filter_by_label) return dpas_atomic_mutator_t().mutate(stmt);
    auto ret = dpas_atomic_mutator_t().mutate_mul(stmt);
    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
