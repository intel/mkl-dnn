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

#include "dsl/ir/send.hpp"
#include "dsl/ir/ir.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

stmt_t send_t::create_offset_store(const expr_t &header_buf,
        const expr_t &mem_buf, const expr_t &_mem_off,
        bool is_signed_offset) const {
    dsl_assert(is_var(mem_buf));
    int header_off = 0;
    int unit_size = 1;
    if (!is_lsc && is_block() && is_slm()) {
        header_off = 2 * address_type().size();
        // Convert byte offset to dwords/owords/hwords offset.
        unit_size = type.scalar().size();
    }

    expr_t mem_off = _mem_off;
    if (unit_size != 1) mem_off /= unit_size;

    expr_t header_sub_buf = header_buf[header_off];

    expr_t off;
    if (is_a64()) {
        off = cast(mem_buf, address_type());
        if (mem_off.type().elems() > 1) {
            off = shuffle_t::make_broadcast(off, mem_off.type().elems());
        }
        off += mem_off;
    } else {
        off = std::move(mem_off);
    }
    off = cast(off, address_type(is_signed_offset, off.type().elems()));
    return store_t::make(header_sub_buf, 0, off);
}

bool send_t::is_supported() const {
    int max_access_size
            = (is_2d() && !is_store_2d()) ? 32 * grf_size() : 8 * grf_size();
    if (access_size() > max_access_size) return false;

    // Block messages imply one slot.
    if (is_block() && slots != 1) return false;

    if (is_block() && !one_of(type.elems(), {1, 2, 4, 8, 16})) return false;

    // owordx8 is max supported unless accessing SLM.
    if (type.is_oword() && !is_slm() && type.elems() > 8) return false;

    // hword is not supported with SLM.
    if (is_slm() && type.is_hword()) return false;

    // Allow only block messages for SLM to reduce offset-related arithmetic.
    if (is_slm() && !is_block()) return false;

    // Only load/store with SLM.
    if (is_slm() && !is_load() && !is_store()) return false;

    // No hword stores before XeHPC.
    if (is_store() && type.is_hword() && !is_xe_hpc_plus()) return false;

    // XXX: Half-GRF stores result in correctness issues on XeHPC.
    if (is_store() && is_block() && is_xe_hpc_plus()
            && type.size() % grf_size() != 0)
        return false;

    // Skip transposing messages, they need additional logic in message
    // decomposition to handle layouts.
    if (type.is_dword() && type.elems() != 1) return false;
    if (type.is_qword() && type.elems() != 1) return false;

    // XXX: Allow only hword x {1,2,4,8} prefetch for now.
    if (is_prefetch() && !type.is_hword()) return false;
    if (is_prefetch() && type.elems() > 8) return false;

    // Expect only float atomics.
    if (is_atomic() && !(type.is_dword() || type.is_qword())) return false;

    if (is_atomic() && !is_xe_hpc_plus() && is_a64() && slots > 8) return false;

    // XXX: Tested only byte scattered messages.
    if (is_scattered() && !is_atomic() && !type.is_byte() && !type.is_qword())
        return false;

    if (type.is_byte() && type.elems() > 4) return false;

    if (is_scattered() && !is_atomic() && !one_of(type.elems(), {1, 2, 4, 8}))
        return false;

    return true;
}

std::vector<func_t> send_t::get_all(const hw_t &hw, send_op_t op,
        send_address_t address, const type_t &mem_type, bool zero_out,
        send_cache_hint_t cache_hint) {
    std::vector<func_t> filtered;
    for (int slots : {1, 2, 4, 8, 16}) {
        for (int elems : {1, 2, 4, 8, 16}) {
            for (auto &type : {type_t::byte(), type_t::dword(), type_t::qword(),
                         type_t::oword(), type_t::hword()}) {
                // Require data type size exact match for atomic messages.
                if (op == send_op_t::atomic_fadd
                        && type.size() != mem_type.size())
                    continue;

                auto f = send_t::make(hw, op, address, type.with_elems(elems),
                        slots, zero_out, cache_hint);
                if (!f.as<send_t>().is_supported()) continue;
                filtered.push_back(f);
            }
        }
    }

    // Sort by total size in descending order.
    std::sort(filtered.begin(), filtered.end(),
            [](const func_t &_a, const func_t &_b) {
        auto &a = _a.as<send_t>();
        auto &b = _b.as<send_t>();
        size_t a_sz = a.access_size();
        size_t b_sz = b.access_size();
        // Put block messages first.
        if (a.is_block() != b.is_block()) return a.is_block();
        // Prefer messages with a smaller type as they have less strict
        // alignment requirements.
        if (a_sz == b_sz)
            return a.type.scalar().size() < b.type.scalar().size();
        return a_sz > b_sz;
    });

    // Remove block messages with the same size (e.g. owordx4 and hwordx2).
    std::vector<func_t> ret;
    for (size_t i = 0; i < filtered.size(); i++) {
        if (i > 0) {
            auto &s_prev = filtered[i - 1].as<send_t>();
            auto &s_cur = filtered[i].as<send_t>();
            if (s_prev.is_block() && s_cur.is_block()
                    && (s_prev.type.size() == s_cur.type.size()))
                continue;
        }
        ret.push_back(filtered[i]);
    }

    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
