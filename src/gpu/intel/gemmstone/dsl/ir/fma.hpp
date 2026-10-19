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

#ifndef GEMMSTONE_DSL_IR_FMA_HPP
#define GEMMSTONE_DSL_IR_FMA_HPP

#include <sstream>
#include <string>

#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/dsl/hw.hpp"
#include "gemmstone/dsl/tensor.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Possible backend instruction sets
enum class fma_kind_t {
    undef = 0,
    mad,
    dp4a,
    dpas,
    dpasw,
};

inline bool is_dp_fma(fma_kind_t kind) {
    switch (kind) {
        case fma_kind_t::dp4a:
        case fma_kind_t::dpas:
        case fma_kind_t::dpasw: return true;
        default: return false;
    }
}

} // namespace ir

template <>
inline const name_map_t<ir::fma_kind_t> &get_name_map() {
    static const name_map_t<ir::fma_kind_t> names {
            {ir::fma_kind_t::undef, "undef"},
            {ir::fma_kind_t::mad, "mad"},
            {ir::fma_kind_t::dp4a, "dp4a"},
            {ir::fma_kind_t::dpas, "dpas"},
            {ir::fma_kind_t::dpasw, "dpasw"},
    };
    return names;
}

namespace ir {

fma_kind_t get_supported_fma_kind(
        const hw_t &hw, const type_t &a, const type_t &b, const type_t &c);

int get_simd_size(const hw_t &hw, fma_kind_t kind, const type_t &a,
        const type_t &b, const type_t &c);

// Function representing DPAS instruction.
class dpas_t : public func_impl_t, public object::info_t<dpas_t> {
public:
    static func_t make(bool is_dpasw, int exec_size, uint8_t sdepth,
            uint8_t rcount, const type_t &dst_type, const type_t &src1_type,
            const type_t &src2_type) {
        return func_t(new dpas_t(is_dpasw, exec_size, sdepth, rcount, dst_type,
                src1_type, src2_type));
    }

    static func_t make_dpasw(const dpas_t &dpas) {
        return func_t(new dpas_t(true, dpas.exec_size, dpas.sdepth, dpas.rcount,
                dpas.dst_type, dpas.src1_type, dpas.src2_type));
    }

    static func_t make_dp4a(int exec_size, const type_t &dst_type,
            const type_t &src1_type, const type_t &src2_type) {
        return make(/*is_dpasw=*/false, exec_size, /*sdepth=*/1, /*rcount=*/1,
                dst_type, src1_type, src2_type);
    }

    static bool is_dp4a_call(const stmt_t &s) {
        auto call = s.as_ptr<func_call_t>();
        return call && call->func.as<dpas_t>().is_dp4a();
    }

    bool is_dp4a() const { return rcount == 1 && sdepth == 1; }

    bool is_equal(const impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (is_dpasw == other.is_dpasw) && (exec_size == other.exec_size)
                && (sdepth == other.sdepth) && (rcount == other.rcount)
                && (dst_type == other.dst_type)
                && (src1_type == other.src1_type)
                && (src2_type == other.src2_type);
    }
    std::string str() const override {
        ostringstream_t oss;
        oss << (is_dpasw ? "dpasw" : is_dp4a() ? "dp4a" : "dpas");
        if (!is_dp4a()) {
            oss << std::to_string(sdepth) << "x" << std::to_string(rcount);
        }
        oss << ".x" << std::to_string(exec_size);
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst, 0)
    IR_DEFINE_ARG_GET(src0, 1)
    IR_DEFINE_ARG_GET(src1, 2)
    IR_DEFINE_ARG_GET(src2, 3)

    stmt_t operator()(const expr_t &dst, const expr_t &src0, const expr_t &src1,
            const expr_t &src2) const {
        return call({dst, src0, src1, src2});
    }

    int dst_size() const {
        return exec_size * (int)(rcount * sizeof(uint32_t));
    }
    int src0_size() const { return dst_size(); }
    int src1_size() const {
        return exec_size * (int)(sdepth * sizeof(uint32_t));
    }
    int src2_size() const {
        const int dpas_size = sdepth * rcount * sizeof(uint32_t);
        return is_dpasw ? dpas_size / 2 : dpas_size;
    }

    layout_t a_layout(std::array<idx_t, 2> dims = {0, 1}) const;
    layout_t b_layout(std::array<idx_t, 2> dims = {0, 1}) const;
    layout_t c_layout(std::array<idx_t, 2> dims = {0, 1}) const;

    static bool matches_types(
            const hw_t &hw, const type_t &a, const type_t &b, const type_t &c);
    static bool is_src_type(type_t type);

    bool is_dpasw;

    int exec_size;
    uint8_t sdepth;
    uint8_t rcount;

    type_t dst_type; // src0 type is same as dst_type.
    type_t src1_type;
    type_t src2_type;

private:
    dpas_t(bool is_dpasw, int exec_size, uint8_t sdepth, uint8_t rcount,
            const type_t &dst_type, const type_t &src1_type,
            const type_t &src2_type)
        : func_impl_t(get_info())
        , is_dpasw(is_dpasw)
        , exec_size(exec_size)
        , sdepth(sdepth)
        , rcount(rcount)
        , dst_type(dst_type)
        , src1_type(src1_type)
        , src2_type(src2_type) {}
};

// Function representing MAD instruction.
class mad_t : public func_impl_t, public object::info_t<mad_t> {
public:
    static func_t make(const hw_t &hw, const type_t &dst_type, int exec_size,
            const type_t &src1_type, int src1_stride, const type_t src2_type,
            int src2_stride) {
        return func_t(new mad_t(hw, dst_type, exec_size, src1_type, src1_stride,
                src2_type, src2_stride));
    }

    std::string str() const override {
        ostringstream_t oss;
        oss << "madx" << exec_size;
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst, 0)
    IR_DEFINE_ARG_GET(src0, 1)
    IR_DEFINE_ARG_GET(src1, 2)
    IR_DEFINE_ARG_GET(src2, 3)

    stmt_t operator()(const expr_t &dst, const expr_t &src0, const expr_t &src1,
            const expr_t &src2) const {
        return call({dst, src0, src1, src2});
    }

    int dst_size() const { return exec_size * dst_type.size(); }
    int src0_size() const { return dst_size(); }
    int src1_size() const {
        return std::max(
                src1_type.size(), src1_stride * src1_type.size() * exec_size);
    }
    int src2_size() const {
        return std::max(
                src2_type.size(), src2_stride * src2_type.size() * exec_size);
    }

    static bool matches_types(
            const hw_t &hw, const type_t &a, const type_t &b, const type_t &c);

    static const int max_exec_size = 32;
    static int get_max_exec_size_bytes(const hw_t &hw) {
        return hw >= ngen::HW::XeHPC ? 128 : 64;
    }
    static int get_simd_size(
            const hw_t &hw, const type_t &a, const type_t &b, const type_t &c) {
        int max_size = max_exec_size;
        int max_exec_size_bytes = get_max_exec_size_bytes(hw);
        int max_type_size = (one_of(type_t::bf8(), {a, b, c})
                                    || one_of(type_t::hf8(), {a, b, c}))
                ? 2
                : std::max(a.size(), std::max(b.size(), c.size()));
        return std::min(max_size, max_exec_size_bytes / max_type_size);
    }
    int get_exec_size() const { return exec_size; }

    type_t dst_type;
    type_t src1_type;
    type_t src2_type;

    int exec_size;
    int src1_stride;
    int src2_stride;

private:
    mad_t(const hw_t &hw, const type_t &dst_type, int exec_size,
            const type_t &src1_type, int src1_stride, const type_t &src2_type,
            int src2_stride)
        : func_impl_t(get_info())
        , dst_type(dst_type)
        , src1_type(src1_type)
        , src2_type(src2_type)
        , exec_size(exec_size)
        , src1_stride(src1_stride)
        , src2_stride(src2_stride) {
        int max_exec_size_bytes = get_max_exec_size_bytes(hw);
        dsl_assert(is_pow2(exec_size));

        dsl_assert(exec_size <= max_exec_size);
        dsl_assert(dst_size() <= max_exec_size_bytes);
        dsl_assert(src1_size() <= max_exec_size_bytes);
        dsl_assert(src2_size() <= max_exec_size_bytes);
    }
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
