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

#include <functional>

#include "dsl/ir/fma.hpp"
#include "dsl/ir/grf_permutation.hpp"
#include "dsl/ir/pass/dpasw.hpp"
#include "dsl/ir/pass/simplify.hpp"
#include "dsl/ir/pass/trace.hpp"
#include "dsl/ir/send.hpp"
#include "dsl/utils/logging.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class dpasw_injector_t {
public:
    dpasw_injector_t(const dsl::hw_t &hw, const stmt_t &load_mul_stmt,
            const expr_t &c_buf, const stmt_t &c_store_stmt,
            alloc_updater_t &alloc_updater, const expr_t &tg_idx0)
        : hw_(hw)
        , load_mul_stmt_(load_mul_stmt)
        , c_buf_(c_buf)
        , c_store_stmt_(c_store_stmt)
        , alloc_updater_(alloc_updater)
        , tg_idx0_(tg_idx0) {}

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }

    void inject() {
        expr_t src2_base;
        if (!extract_dpas_calls(src2_base)) return;

        grf_permutation_t grf_perm;

        bool was_injected = false;
        int dpas_count = int(dpas_infos_.size());
        for (int i = 0; i < dpas_count;) {
            if (i + 1 < dpas_count) {
                auto &a = dpas_infos_[i];
                auto &b = dpas_infos_[i + 1];
                if (try_convert_to_dpasw(a, b, grf_perm)) {
                    was_injected = true;
                    i += 2;
                    continue;
                }
            }
            if (try_convert_to_dpasw(dpas_infos_[i], grf_perm)) {
                was_injected = true;
            }
            ++i;
        }
        // Nothing to update, no dpas -> dpasw transformation.
        if (!was_injected) return;

        int src2_size = 0;
        object_map_t<stmt_t, int> send2off;
        std::function<int(const stmt_t &)> get_src2_off;
        get_src2_off = [&](const stmt_t &s) {
            auto &si = find_send_info(s);
            if (si.base_call) return get_src2_off(si.base_call);
            if (si.prev_send) return get_src2_off(si.prev_send);

            auto it = send2off.find(s);
            if (it != send2off.end()) return it->second;

            auto ret = send2off.insert({s, src2_size});
            if (!ret.second) return ret.first->second;

            int new_size = si.new_reg_buf_size();
            src2_size += new_size;
            return ret.first->second;
        };
        for (auto &si : send_infos_) {
            if (!si.reg_buf_base().is_equal(src2_base)) continue;

            int src2_off = get_src2_off(si.call);
            auto src2_sub = src2_base[src2_off];
            auto new_call = si.new_call;
            if (new_call) {
                new_call = substitute(
                        new_call, send_t::arg_reg_buf(new_call), src2_sub, 1);
            }

            load_mul_stmt_ = substitute(load_mul_stmt_, si.call, new_call, 1);
            for (auto &d : si.dpas_consumers) {
                auto &di = find_dpas_info(d);
                dsl_assert(si.promote_to_dpasw == di.promote_to_dpasw)
                        << "Both send and dpas must be updated.";
                if (di.update_applied) {
                    stub("Can it happen?");
                    continue;
                }
                auto new_call = di.new_call;
                new_call = substitute(new_call, dpas_t::arg_src2(new_call),
                        src2_sub[di.src2_relative_off], 1);
                load_mul_stmt_
                        = substitute(load_mul_stmt_, di.call, new_call, 1);
                di.update_applied = true;
            }
        }

        // Update src2 size after applying send updates.
        alloc_updater_.resize(src2_base, src2_size);

        // Apply permutation to C buffer.
        alloc_updater_.add_attr(c_buf_,
                grf_permute_attr_t::make(
                        std::make_shared<grf_permutation_t>(grf_perm)));
    }

private:
    struct send_info_t {
        send_info_t() = default;

        send_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const send_t &send() const {
            return call.as<func_call_t>().func.as<send_t>();
        }

        const send_t &new_send() const {
            dsl_assert(!new_call.is_same(call));
            return new_call.as<func_call_t>().func.as<send_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &reg_buf() const { return send_t::arg_reg_buf(call); }

        const expr_t &reg_buf_base() const {
            return reg_buf().as<ptr_t>().base;
        }

        int reg_buf_size() const { return send().payload_size(); }

        int new_reg_buf_size() const {
            if (new_call.is_same(call)) return reg_buf_size();
            return new_send().payload_size();
        }

        void set_new_call(const stmt_t &s, const stmt_t &base = stmt_t()) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                new_call = s;
                base_call = base;
                return;
            }
            dsl_assert(new_call.is_equal(s));
            dsl_assert(base_call.is_equal(base));
        }

        void set_prev_send(const stmt_t &s) {
            int prev_size
                    = s.as<func_call_t>().func.as<send_t>().payload_size();
            if (reg_buf_size() != prev_size) return;
            prev_send = s;
        }

        stmt_t call;
        std::vector<stmt_t> dpas_consumers;

        bool promote_to_dpasw = false;
        stmt_t new_call;
        stmt_t base_call;
        stmt_t prev_send;
    };

    struct dpas_info_t {
        dpas_info_t() = default;

        dpas_info_t(const stmt_t &call) : call(call), new_call(call) {}

        const dpas_t &dpas() const {
            return call.as<func_call_t>().func.as<dpas_t>();
        }

        const std::vector<expr_t> &args() const {
            return call.as<func_call_t>().args;
        }

        const expr_t &src1_buf() const { return dpas_t::arg_src1(call); }

        const expr_t &src2_buf() const { return dpas_t::arg_src2(call); }

        int src2_size() const { return dpas().src2_size(); }

        void set_new_call(const stmt_t &s, int src2_relative_off) {
            if (!promote_to_dpasw) {
                promote_to_dpasw = true;
                this->src2_relative_off = src2_relative_off;
                new_call = s;
                return;
            }
            dsl_assert(this->src2_relative_off == src2_relative_off);
            dsl_assert(new_call.is_equal(s));
        }

        stmt_t call;
        stmt_t send_producer;

        bool promote_to_dpasw = false;
        bool update_applied = false;
        int src2_relative_off = 0;
        stmt_t new_call;
    };

    send_info_t &find_send_info(const stmt_t &s) {
        for (auto &si : send_infos_)
            if (si.call.is_same(s)) return si;
        stub();
        return send_infos_.front();
    }

    dpas_info_t &find_dpas_info(const stmt_t &s) {
        for (auto &si : dpas_infos_)
            if (si.call.is_same(s)) return si;
        stub();
        return dpas_infos_.front();
    }
    static bool is_send(const stmt_t &s, send_info_t &info) {
        if (!is_func_call<send_t>(s)) return false;
        info = send_info_t(s);
        return true;
    }

    static bool is_dpas(const stmt_t &s, dpas_info_t &info) {
        if (!is_func_call<dpas_t>(s)) return false;
        if (dpas_t::is_dp4a_call(s)) return false;
        info = dpas_info_t(s);
        return true;
    }

    static bool has_constant_mask(const stmt_t &s) {
        dsl_assert(is_func_call<send_t>(s));
        auto &mask = send_t::arg_mask(s);
        if (mask.is_empty()) return true;
        if (is_const(mask)) return true;
        if (is_shuffle_const(mask)) return true;
        return false;
    }

    bool extract_dpas_calls(expr_t &src2_base) {
        object_eq_map_t<expr_t, stmt_t> buf2send;

        auto set_src2_base = [&](const expr_t &ptr) {
            auto &ptr_base = ptr.as<ptr_t>().base;
            if (src2_base.is_empty()) {
                src2_base = ptr_base;
                return;
            }
            dsl_assert(src2_base.is_same(ptr_base));
        };

        // Iterate through dpas and send calls.
        auto stmt_vec = flatten_statements(load_mul_stmt_);
        for (auto &s : stmt_vec) {
            send_info_t send_info;
            if (is_send(s, send_info)) {
                auto &buf = send_info.reg_buf();
                stmt_t prev_send;
                auto it = buf2send.find(buf);
                if (it != buf2send.end()) prev_send = it->second;
                buf2send[buf] = s;
                send_infos_.push_back(std::move(send_info));
                if (prev_send) { send_infos_.back().set_prev_send(prev_send); }
                continue;
            }
            dpas_info_t dpas_info;
            if (is_dpas(s, dpas_info)) {
                set_src2_base(dpas_info.src2_buf());
                auto &buf = dpas_info.src2_buf();
                auto it = buf2send.find(buf);
                if (it == buf2send.end()) continue;
                auto &send_info = find_send_info(it->second);
                // For simplicity require full size match between load and dpas
                // instructions. That is dpas src2 buffer should be fully
                // loaded by the corresponding send message.
                if (send_info.reg_buf_size() != dpas_info.src2_size()) {
                    dsl_warning() << "Can't inject dpasw: different register "
                                     "sizes in send and dpas.";
                    return false;
                }
                dpas_info.send_producer = send_info.call;
                send_info.dpas_consumers.push_back(s);
                dpas_infos_.push_back(dpas_info);
            }
        }
        return true;
    }

    // Checks for the following pattern:
    //    dpas.sxr(a_dst, a_src0, src1, src2)
    //    dpas.sxr(b_dst, b_src0, src1, src2 + s * r * 4)
    static bool can_convert_to_dpasw(
            const dpas_info_t &a, const dpas_info_t &b) {
        if (!a.dpas().is_equal(b.dpas())) return false;
        if (!a.src1_buf().is_equal(b.src1_buf())) return false;
        if (!has_constant_mask(a.send_producer)) return false;
        if (!has_constant_mask(b.send_producer)) return false;

        auto src2_off0 = to_cpp<int>(a.src2_buf().as<ptr_t>().off);
        auto src2_off1 = to_cpp<int>(b.src2_buf().as<ptr_t>().off);

        if (src2_off1 - src2_off0 != a.src2_size()) return false;

        return true;
    }

    bool try_convert_to_dpasw(
            dpas_info_t &a, dpas_info_t &b, grf_permutation_t &grf_perm) {
        if (hw_ >= ngen::HW::XeHPC) return false;

        // Check if DPAS -> DPASW transformation is possible.
        if (!can_convert_to_dpasw(a, b)) return false;

        // Perform the transformation:
        // Before:
        //   send(mem, a_off, src2[0])
        //   send(mem, b_off, src2[s * r * 4])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        //   dpas.sxr(b_dst, b_src0, src1, src2[s * r * 4])
        // After:
        //   send(mem, a_off + (tg_idx0 % 2) * (b_off - a_off), src2)
        //   dpasw.sxr(p_a_dst, p_a_src0, src1, src2[0])
        //   dpasw.sxr(p_b_dst, p_b_src0, src1, src2[s * r * 4 / 2])
        // Where:
        //   p_a_dst[:] = a_dst[0:rcount / 2] + b_dst[0:rcount / 2]
        //   p_b_dst[:] = a_dst[rcount / 2:rcount] + b_dst[rcount / 2:rcount]
        dsl_assert(a.dpas().is_equal(b.dpas()));
        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        auto a_args = a.args();
        auto b_args = b.args();
        dpas_t::arg_src2(b_args) -= dpasw.src2_size();

        a.set_new_call(dpasw.call(a.args()), 0);
        b.set_new_call(dpasw.call(b_args), dpasw.src2_size());

        // Record permutation for registers to apply it for the destination
        // store later.
        const auto grf_size = hw_.grf_size();
        const auto rcount = a.dpas().rcount;
        for (int j = 0; j < rcount; j++) {
            int k = j % (rcount / 2);
            auto a_old = dpas_t::arg_dst(a_args) + grf_size * j;
            auto b_old = dpas_t::arg_dst(b_args) + grf_size * j;
            expr_t grf_new;
            if (j < rcount / 2) {
                grf_new = dpas_t::arg_dst(a_args)[grf_size * k];
            } else {
                grf_new = dpas_t::arg_dst(b_args)[grf_size * k];
            }
            set_grf_permute(grf_perm, a_old, grf_new);
            set_grf_permute(grf_perm, b_old, grf_new + grf_size * rcount / 2);
        }

        auto &a_send = find_send_info(a.send_producer);
        auto &b_send = find_send_info(b.send_producer);

        auto &a_mem_off = send_t::arg_mem_off(a_send.call);
        auto &b_mem_off = send_t::arg_mem_off(b_send.call);
        auto ab_addr_diff = simplify(b_mem_off - a_mem_off);
        dsl_assert(is_const(ab_addr_diff));

        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * to_cpp<int64_t>(ab_addr_diff);

        a_send.set_new_call(a_send.send().call(new_send_args));
        b_send.set_new_call(stmt_t(), a_send.call);

        return true;
    }

    void set_grf_permute(grf_permutation_t &grf_perm, const expr_t &old_grf,
            const expr_t &new_grf) {
        int old_off = to_cpp<int>(old_grf.as<ptr_t>().off);
        int new_off = to_cpp<int>(new_grf.as<ptr_t>().off);

        const int grf_size = hw_.grf_size();

        dsl_assert(old_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";
        dsl_assert(new_off % grf_size == 0)
                << "Must be aligned to GRF boundary.";

        old_off /= grf_size;
        new_off /= grf_size;

        grf_perm.set_permute(old_off, new_off);
    }

    static bool can_convert_to_dpasw(const dpas_info_t &a_dpas,
            const send_info_t &a_send, const expr_t &tg_idx0) {
        if (contains_object(a_send.call, tg_idx0)) return false;
        if (!has_constant_mask(a_send.call)) return false;
        return a_dpas.dpas().rcount % 2 == 0;
    }

    static func_t create_half_send(const send_t &send) {
        dsl_assert(send.type.elems() % 2 == 0) << "Can't create half-send.";
        auto _s = send_t::make(send.hw, send.op, send.address,
                send.type.with_elems(send.type.elems() / 2), send.slots,
                send.is_lsc, send.fill_buf, send.cache_hint);
        auto &s = _s.as<send_t>();
        dsl_assert(s.is_supported())
                << "Can't find send reading half of the original send.";
        maybe_unused(s);
        return _s;
    }

    bool try_convert_to_dpasw(dpas_info_t &a, grf_permutation_t &grf_perm) {
        if (hw_ >= ngen::HW::XeHPC) return false;
        if (!can_convert_to_dpasw(a, find_send_info(a.send_producer), tg_idx0_))
            return false;

        // Perform the transformation:
        // Before:
        //   send(mem, a_off, src2[0])
        //   dpas.sxr(a_dst, a_src0, src1, src2[0])
        // After:
        //   send(mem, a_off + (tg_idx0 % 2) * (s * r * 4 / 2), src2)
        //   dpasw.sxr(a_dst, a_src0, src1, src2[0])

        auto _dpasw = dpas_t::make_dpasw(a.dpas());
        auto &dpasw = _dpasw.as<dpas_t>();

        a.set_new_call(dpasw.call(a.args()), 0);

        auto &a_send = find_send_info(a.send_producer);
        auto new_send_args = a_send.args();
        send_t::arg_mem_off(new_send_args)
                += (tg_idx0_ % 2) * (a.src2_size() / 2);
        a_send.set_new_call(
                create_half_send(a_send.send()).call(new_send_args));

        return true;
    }

    dsl::hw_t hw_;
    stmt_t load_mul_stmt_;
    expr_t c_buf_;
    stmt_t c_store_stmt_;
    alloc_updater_t &alloc_updater_;
    expr_t tg_idx0_;

    std::vector<dpas_info_t> dpas_infos_;
    std::vector<send_info_t> send_infos_;
};

void inject_dpasw(const dsl::hw_t &hw, stmt_t &load_mul_stmt,
        const expr_t &c_buf, stmt_t &c_store_stmt,
        alloc_updater_t &alloc_updater, const expr_t &tg_idx0) {
    dpasw_injector_t injector(
            hw, load_mul_stmt, c_buf, c_store_stmt, alloc_updater, tg_idx0);

    injector.inject();
    load_mul_stmt = injector.load_mul_stmt();
    c_store_stmt = injector.c_store_stmt();
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
