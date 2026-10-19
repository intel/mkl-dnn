/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_DSL_IR_SEND_HPP
#define GEMMSTONE_DSL_IR_SEND_HPP

#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/dsl/hw.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

enum class send_kind_t {
    undef,
    _2d,
    block,
    scattered,
};

// Send operation kind.
enum class send_op_t {
    undef,
    atomic_add,
    atomic_fadd,
    atomic_bfadd,
    atomic_cmpwr,
    load,
    load_2d,
    prefetch,
    prefetch_2d,
    store,
    store_2d,
};

// Send address model.
enum class send_address_t {
    a64,
    slm,
};

} // namespace ir

template <>
inline const name_map_t<ir::send_cache_hint_t> &get_name_map() {
    static const name_map_t<ir::send_cache_hint_t> names {
            {ir::send_cache_hint_t::undef, "cache:undef"},
            {ir::send_cache_hint_t::load_once, "cache:load_once"},
    };
    return names;
}

template <>
inline const name_map_t<ir::send_kind_t> &get_name_map() {
    static const name_map_t<ir::send_kind_t> names {
            {ir::send_kind_t::undef, "undef"},
            {ir::send_kind_t::_2d, "2d"},
            {ir::send_kind_t::block, "block"},
            {ir::send_kind_t::scattered, "scattered"},
    };
    return names;
}

template <>
inline const name_map_t<ir::send_op_t> &get_name_map() {
    static const name_map_t<ir::send_op_t> names {
            {ir::send_op_t::undef, "undef"},
            {ir::send_op_t::atomic_add, "atomic_add"},
            {ir::send_op_t::atomic_fadd, "atomic_fadd"},
            {ir::send_op_t::atomic_cmpwr, "atomic_cmpwr"},
            {ir::send_op_t::load, "load"},
            {ir::send_op_t::load_2d, "load_2d"},
            {ir::send_op_t::prefetch, "prefetch"},
            {ir::send_op_t::prefetch_2d, "prefetch_2d"},
            {ir::send_op_t::store, "store"},
            {ir::send_op_t::store_2d, "store_2d"},
    };
    return names;
}

template <>
inline const name_map_t<ir::send_address_t> &get_name_map() {
    static const name_map_t<ir::send_address_t> names {
            {ir::send_address_t::a64, "a64"},
            {ir::send_address_t::slm, "slm"},
    };
    return names;
}

namespace ir {

struct block_2d_info_t {
    bool is_empty() const { return surface_width.is_empty(); }

    bool operator==(const block_2d_info_t &other) const {
        if (is_empty() != other.is_empty()) return false;
        if (is_empty()) return true;
        return (surface_width.is_equal(other.surface_width))
                && (surface_height.is_equal(other.surface_height))
                && (surface_pitch.is_equal(other.surface_pitch))
                && (width == other.width) && (height == other.height)
                && (count == other.count) && (vnni == other.vnni)
                && (transpose == other.transpose);
    }

    std::string str() const {
        ostringstream_t oss;
        oss << count << "x";
        oss << height << "x";
        oss << width;
        if (vnni || transpose) {
            oss << ".";
            if (vnni) oss << "v";
            if (transpose) oss << "t";
        }
        return oss.str();
    }

    // Encoded in header.
    expr_t surface_width;
    expr_t surface_height;
    expr_t surface_pitch;
    int width = 0;
    int height = 0;
    int count = 0;
    // Part of descriptor.
    bool vnni = false;
    bool transpose = false;
};

// Function representing send messages.
class send_t : public func_impl_t, public object::info_t<send_t> {
public:
    static func_t make(const hw_t &hw, send_op_t op, send_address_t address,
            const type_t &type, int slots, bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        return make(hw, op, address, type, slots, default_slot_mask,
                hw >= ngen::HW::XeHPC, zero_out, cache_hint);
    }

    static func_t make(const hw_t &hw, send_op_t op, send_address_t address,
            const type_t &type, int slots, bool is_lsc, bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        return make(hw, op, address, type, slots, default_slot_mask, is_lsc,
                zero_out, cache_hint);
    }

    static func_t make(const hw_t &hw, send_op_t op, send_address_t address,
            const type_t &type, int slots, uint32_t slot_mask, bool is_lsc,
            bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        return func_t(new send_t(hw, op, address, type, slots, slot_mask,
                is_lsc, zero_out, cache_hint));
    }

    static func_t make(const hw_t &hw, send_op_t op, send_address_t address,
            const type_t &type, int slots, uint32_t slot_mask, bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        return make(hw, op, address, type, slots, slot_mask,
                hw >= ngen::HW::XeHPC, zero_out, cache_hint);
    }

    static func_t make_2d(const hw_t &hw, send_op_t op, const type_t &type,
            expr_t surface_width, expr_t surface_height, expr_t surface_pitch,
            int width, int height, int count, bool vnni, bool transpose,
            bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        block_2d_info_t info;
        info.surface_width = std::move(surface_width);
        info.surface_height = std::move(surface_height);
        info.surface_pitch = std::move(surface_pitch);
        info.width = width;
        info.height = height;
        info.count = count;
        info.vnni = vnni;
        info.transpose = transpose;
        return func_t(new send_t(hw, op, type, zero_out, info, cache_hint));
    }

    static func_t make_2d(const hw_t &hw, send_op_t op, const type_t &type,
            int width, int height, int count, bool vnni, bool transpose,
            bool zero_out,
            send_cache_hint_t cache_hint = send_cache_hint_t::undef) {
        return make_2d(hw, op, type, /*surface_width=*/ {},
                /*surface_height=*/ {},
                /*surface_pitch=*/ {}, width, height, count, vnni, transpose,
                zero_out, cache_hint);
    }

    bool is_equal(const impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        // hw is not compared as cross-hardware IR operations are not expected.
        return (op == other.op) && (address == other.address)
                && (type == other.type) && (slots == other.slots)
                && (slot_mask == other.slot_mask) && (is_lsc == other.is_lsc)
                && (fill_buf == other.fill_buf)
                && (block_2d_info == other.block_2d_info)
                && (cache_hint == other.cache_hint);
    }
    std::string str() const override {
        ostringstream_t oss;
        oss << to_string(op);
        oss << ".";
        oss << type.str();
        if (is_scattered()) oss << "x" << slots;
        if (is_2d()) oss << "." << block_2d_info.str();
        if (!fill_buf) oss << ".nofill";
        if (cache_hint != send_cache_hint_t::undef)
            oss << "." << to_string(cache_hint);
        return oss.str();
    }

    IR_DEFINE_ARG_GET(mem_buf, 0)
    IR_DEFINE_ARG_GET(mem_off, 1)
    IR_DEFINE_ARG_GET(header_buf, 1)
    IR_DEFINE_ARG_GET(reg_buf, 2)
    IR_DEFINE_ARG_GET(mask, 3)
    IR_DEFINE_ARG_GET(x, 4)
    IR_DEFINE_ARG_GET(y, 5)
    IR_DEFINE_ARG_GET(fill_pattern, 6)

    // Header offsets in bytes for 2D block messages.
    static int header_2d_off_base() { return 0; }
    static int header_2d_off_surface_width() { return 8; }
    static int header_2d_off_surface_height() { return 12; }
    static int header_2d_off_surface_pitch() { return 16; }
    static int header_2d_off_x() { return 20; }
    static int header_2d_off_y() { return 24; }
    static int header_2d_off_whc() { return 28; }

    stmt_t operator()(const expr_t &mem_buf, const expr_t &mem_off,
            const expr_t &reg_buf, const expr_t &mask,
            const expr_t &x = expr_t(), const expr_t &y = expr_t(),
            const expr_t &pattern = expr_t()) const {
        return call({mem_buf, mem_off, reg_buf, mask, x, y, pattern});
    }

    bool is_atomic() const {
        return one_of(op,
                {send_op_t::atomic_add, send_op_t::atomic_fadd,
                        send_op_t::atomic_cmpwr});
    }
    bool is_load() const { return op == send_op_t::load; }
    bool is_load_2d() const { return op == send_op_t::load_2d; }
    bool is_prefetch() const { return op == send_op_t::prefetch; }
    bool is_prefetch_2d() const { return op == send_op_t::prefetch_2d; }
    bool is_store() const { return op == send_op_t::store; }
    bool is_store_2d() const { return op == send_op_t::store_2d; }
    bool is_2d() const {
        return is_load_2d() || is_store_2d() || is_prefetch_2d();
    }
    bool is_a64() const { return address == send_address_t::a64; }
    bool is_slm() const { return address == send_address_t::slm; }

    bool is_block() const {
        return one_of(type.scalar(), {type_t::oword(), type_t::hword()});
    }

    bool is_scattered() const { return !is_block() && !is_2d(); }

    // Size of memory (global memory or SLM) to access.
    int access_size() const {
        if (is_2d()) {
            auto &info = block_2d_info;
            return type.size() * info.width * info.height * info.count;
        }
        return type.size() * slots;
    }

    int payload_type_stride() const {
        dsl_assert(!is_2d());
        return std::max(4, type.size());
    }

    // Full size of payload GRF buffer for this message. Buffer may be strided
    // and/or require GRF boundary round-up.
    int payload_size() const {
        if (is_2d()) {
            auto &info = block_2d_info;
            int w = info.width;
            int h = info.height;
            int c = info.count;
            if (info.transpose) {
                h = roundup_pow2(h);
            } else {
                w = roundup_pow2(w);
            }
            return round_up(type.size() * w * h, grf_size()) * c;
        }
        int sz = payload_type_stride() * slots;
        return round_up(sz, grf_size());
    }

    int alignment() const {
        if (is_2d()) return 128;
        if (is_block()) return type.scalar().size();
        return 1;
    }

    int mask_size() const {
        if (is_2d()) return access_size();
        if (is_block()) {
            // LSC messages use SIMT1 execution mask (one mask per message).
            if (is_lsc) return type.size();
            return 4;
        }

        if (is_scattered()) return type.size();

        stub();
        return 0;
    }

    int nmasks() const {
        if (is_2d()) return 1;
        int masks = utils::safe_divide(type.size() * slots, mask_size());
        if (hw < ngen::HW::XeHPC && is_block() && masks > 16) {
            // Round-robin masking, 16 bits are reused with dword granularity.
            dsl_assert(masks % 16 == 0);
            masks = 16;
        }
        return masks;
    }

    int address_size() const { return is_a64() ? 8 : 4; }

    type_t address_type(bool is_signed = false, int elems = 1) const {
        int bits = address_size() * 8;
        return is_signed ? type_t::s(bits, elems) : type_t::u(bits, elems);
    }

    // Size of header in bytes.
    int header_size() const {
        if (is_2d()) return grf_size();
        return round_up(address_size() * slots, grf_size());
    }

    // Generates a statement to store (and maybe convert) the offset to the
    // message header according to the message description.
    stmt_t create_offset_store(const expr_t &header_buf, const expr_t &mem_buf,
            const expr_t &mem_off, bool is_signed_offset = false) const;

    bool is_supported() const;

    bool has_default_slot_mask() const {
        uint32_t all_slots_mask = (slots == 32 ? 0xFFFFFFFF : (1 << slots) - 1);
        return (slot_mask & all_slots_mask) == all_slots_mask;
    }

    static std::vector<func_t> get_all(const hw_t &hw, send_op_t op,
            send_address_t address, const type_t &mem_type, bool zero_out,
            send_cache_hint_t cache_hint);

    hw_t hw;
    send_op_t op;
    send_address_t address;
    type_t type;
    int slots;
    uint32_t slot_mask;
    bool is_lsc;
    bool fill_buf;

    block_2d_info_t block_2d_info;
    send_cache_hint_t cache_hint;

    static const uint32_t default_slot_mask = 0xFFFFFFFF;

private:
    int grf_size() const { return hw.grf_size(); }

    bool is_xe_hp_plus() const { return hw >= ngen::HW::XeHP; }

    bool is_xe_hpc_plus() const { return hw >= ngen::HW::XeHPC; }

    send_t(const hw_t &hw, send_op_t op, send_address_t address,
            const type_t &type, int slots, uint32_t slot_mask, bool is_lsc,
            bool zero_out, send_cache_hint_t cache_hint)
        : func_impl_t(get_info())
        , hw(hw)
        , op(op)
        , address(address)
        , type(type)
        , slots(slots)
        , slot_mask(slot_mask)
        , is_lsc(is_lsc)
        , fill_buf(zero_out)
        , cache_hint(cache_hint) {}

    send_t(const hw_t &hw, send_op_t op, const type_t &type, bool zero_out,
            const block_2d_info_t &block_2d_info, send_cache_hint_t cache_hint)
        : func_impl_t(get_info())
        , hw(hw)
        , op(op)
        , address(send_address_t::a64)
        , type(type)
        , slots(1)
        , slot_mask(default_slot_mask)
        , is_lsc(true)
        , fill_buf(zero_out)
        , block_2d_info(block_2d_info)
        , cache_hint(cache_hint) {
        dsl_assert(one_of(op,
                {send_op_t::load_2d, send_op_t::store_2d,
                        send_op_t::prefetch_2d}));
        if (is_store_2d()) {
            dsl_assert(!block_2d_info.vnni);
            dsl_assert(!block_2d_info.transpose);
        }
    }
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
