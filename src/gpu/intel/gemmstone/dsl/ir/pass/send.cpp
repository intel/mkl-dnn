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

#include "dsl/ir/pass/send.hpp"

#include "dsl/ir/pass/simplify.hpp"
#include "dsl/ir/pass/trace.hpp"
#include "dsl/ir/send.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class buffer_offset_lifter_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t &obj) override {
        if (!obj.func.is<send_t>()) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        if (!mem_buf.is<ptr_t>()) return ir_mutator_t::_mutate(obj);

        auto &base = mem_buf.as<ptr_t>().base;
        auto &off = mem_buf.as<ptr_t>().off;

        std::vector<expr_t> new_args = obj.args;
        send_t::arg_mem_buf(new_args) = base;
        auto &mem_off = send_t::arg_mem_off(new_args);
        mem_off += shuffle_t::make_broadcast(off, mem_off.type().elems());
        return obj.func.call(new_args, obj.attr);
    }
};

stmt_t lift_buffer_offsets_in_send(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    buffer_offset_lifter_t lifter;
    auto ret = lifter.mutate(s);
    trace_pass("lift_buffer_offsets_in_send", ret, ir_ctx);
    return ret;
}

class send_injector_t : public ir_mutator_t {
public:
    send_injector_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const func_call_t &obj) override {
        auto *send = obj.func.as_ptr<send_t>();
        if (!send) return ir_mutator_t::_mutate(obj);

        auto &mem_buf = send_t::arg_mem_buf(obj);
        auto &mem_off = send_t::arg_mem_off(obj);
        // If header/offset is a pointer, then assume this send already has a
        // proper header.
        if (mem_off.type().is_ptr()) return obj;

        auto &reg_buf = send_t::arg_reg_buf(obj);
        auto &mask = send_t::arg_mask(obj);
        expr_t pattern;
        if (obj.args.size() > 6) pattern = send_t::arg_fill_pattern(obj);

        dsl_assert(is_var(mem_buf)) << mem_buf;

        auto header_buf
                = ir_ctx_.create_tmp_var(type_t::byte(type::attr_t::ptr), "h");
        auto off_store = simplify_store(
                send->create_offset_store(header_buf, mem_buf, mem_off));

        if (send->is_2d()) {
            auto emit_store = [&](const expr_t &e, int off) {
                auto store = store_t::make(header_buf, off, e);
                off_store = off_store.append(store);
            };
            auto emit_store_s32 = [&](const expr_t &value, int off) {
                emit_store(cast(value, type_t::s32()), off);
            };
            auto &info = send->block_2d_info;
            int type_size = send->type.size();
            emit_store_s32(info.surface_width * type_size - 1,
                    send_t::header_2d_off_surface_width());
            emit_store_s32(info.surface_height - 1,
                    send_t::header_2d_off_surface_height());
            emit_store_s32(info.surface_pitch * type_size - 1,
                    send_t::header_2d_off_surface_pitch());
            emit_store(send_t::arg_x(obj), send_t::header_2d_off_x());
            emit_store(send_t::arg_y(obj), send_t::header_2d_off_y());
            uint32_t w_enc = info.width - 1;
            uint32_t h_enc = info.height - 1;
            uint32_t count_enc = info.count - 1;
            emit_store_s32((count_enc << 16) + (h_enc << 8) + w_enc,
                    send_t::header_2d_off_whc());
        }

        auto new_call = func_call_t::make(obj.func,
                {mem_buf, header_buf, reg_buf, mask, expr_t(), expr_t(),
                        std::move(pattern)},
                obj.attr);
        auto body = stmt_seq_t::make(off_store, new_call);

        // Allocate header.
        return alloc_t::make(
                header_buf, send->header_size(), alloc_kind_t::grf, body);
    }

private:
    stmt_t simplify_store(const stmt_t &_store) const {
        auto &store = _store.as<store_t>();

        auto value = store.value;
        value = simplify(value, ir_ctx_.cset());

        // Convert to N-ary form and back to expand multiplications. This
        // helps to find more common subexpressions during the pass.
        value = nary_op_canonicalize(value);
        value = nary_op_back_transform(value);

        return store_t::make(store.buf, store.off, value, store.stride);
    }

    ir_context_t &ir_ctx_;
};

void inject_send(kernel_t &k) {
    ir_context_t ir_ctx(k);
    k.body = inject_send(k.body, ir_ctx);
}

stmt_t inject_send(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = send_injector_t(ir_ctx).mutate(s);
    trace_pass("inject_send", ret, ir_ctx);
    return ret;
}

class send_2d_header_store_lifter_t : public ir_mutator_t {
public:
    send_2d_header_store_lifter_t(const stmt_t &root) {
        auto calls = find_objects<func_call_t>(root);
        for (auto &c : calls) {
            if (!is_func_call<send_t>(c)) continue;
            if (!c.as<func_call_t>().func.as<send_t>().is_2d()) continue;
            auto header_buf = send_t::arg_mem_off(c);
            dsl_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(std::move(header_buf));
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto it = stores_.find(obj.buf);
        if (it == stores_.end()) return new_obj;

        auto &alloc = new_obj.as<alloc_t>();
        stmt_t header_store;
        for (auto &s : it->second)
            header_store = header_store.append(s);
        it->second.clear();

        auto new_body = header_store.append(alloc.body);
        return alloc_t::make(
                alloc.buf, alloc.size, alloc.kind, alloc.attrs, new_body);
    }

    object_t _mutate(const store_t &obj) override {
        if (header_bufs_.count(obj.buf) == 0) return obj;
        // Do not lift address assignments and non-const x and y.
        int off = to_cpp<int>(obj.off);
        if (off == 0) return obj;
        if (one_of(off, {send_t::header_2d_off_x(), send_t::header_2d_off_y()})
                && !is_const(obj.value))
            return obj;
        stores_[obj.buf].emplace_back(obj);
        return stmt_t();
    }

private:
    object_set_t<expr_t> header_bufs_;
    object_map_t<expr_t, std::vector<stmt_t>> stores_;
};

stmt_t lift_send_2d_header_store(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = send_2d_header_store_lifter_t(s).mutate(s);
    trace_pass("lift_send_2d_header_store", ret, ir_ctx);
    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
