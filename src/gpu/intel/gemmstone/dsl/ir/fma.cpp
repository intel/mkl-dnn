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

#include "dsl/ir/fma.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

fma_kind_t get_supported_fma_kind(
        const hw_t &hw, const type_t &a, const type_t &b, const type_t &c) {
    if (hw >= ngen::HW::XeHP && hw.systolic_support()
            && dpas_t::matches_types(hw, a, b, c)) {
        if (hw >= ngen::HW::XeHPC)
            return fma_kind_t::dpas;
        else
            return fma_kind_t::dpasw;
    }
    if ((hw == ngen::HW::XeLP
                || (hw >= ngen::HW::XeHP && !hw.systolic_support()))
            && (a.is_x8() && b.is_x8() && c.is_s32()))
        return fma_kind_t::dp4a;
    if (mad_t::matches_types(hw, a, b, c)) return fma_kind_t::mad;
    return fma_kind_t::undef;
}

int get_simd_size(const hw_t &hw, const fma_kind_t kind, const type_t &a,
        const type_t &b, const type_t &c) {
    int ret = 0;
    switch (kind) {
        case fma_kind_t::dp4a:
        case fma_kind_t::dpas:
        case fma_kind_t::dpasw: ret = hw >= ngen::HW::XeHPC ? 16 : 8; break;
        case fma_kind_t::mad: ret = mad_t::get_simd_size(hw, a, b, c); break;
        default: break;
    }
    dsl_assert(ret != 0);
    return ret;
}

bool dpas_t::is_src_type(type_t type) {
    if (type.is_bf8() || type.is_hf8() || type.is_fp4()) return true;
    return type.is_x8() || type.is_bf16() || type.is_f16() || type.is_tf32();
}

layout_t dpas_t::a_layout(std::array<idx_t, 2> dims) const {
    if (!is_src_type(src1_type)) stub();

    int m_blk = exec_size;
    int inner_blk = src1_type.bytes_to_elems(4);
    int outer_blk = sdepth;
    std::vector<layout::block_t> blocks
            = {{dims[1], inner_blk}, {dims[0], m_blk}, {dims[1], outer_blk}};
    return layout_t(src1_type, blocks);
}

layout_t dpas_t::b_layout(std::array<idx_t, 2> dims) const {
    if (!is_src_type(src2_type)) stub();

    int n_blk = rcount;
    int k_blk = src2_type.bytes_to_elems(sdepth * 4);
    std::vector<layout::block_t> blocks = {{dims[0], k_blk}, {dims[1], n_blk}};
    return layout_t(src2_type, blocks);
}

layout_t dpas_t::c_layout(std::array<idx_t, 2> dims) const {
    int m_blk = exec_size;
    int n_blk = rcount;
    std::vector<layout::block_t> blocks = {{dims[0], m_blk}, {dims[1], n_blk}};
    return layout_t(dst_type, blocks);
}

bool dpas_t::matches_types(
        const hw_t &hw, const type_t &a, const type_t &b, const type_t &c) {
    if (a.is_x8() && b.is_x8() && c.is_s32()) return true;
    if (a.is_fp8() && b.is_fp8() && (c.is_f32() || c.is_bf16())) return true;
    if (a.is_fp4() && b.is_fp4() && (c.is_f32() || c.is_bf16())) return true;
    if (a.is_f16() && b.is_f16() && c.is_f32()) return true;
    if (a.is_bf16() && b.is_bf16() && c.is_f32()) return true;
    if (a.is_tf32() && b.is_tf32() && c.is_f32() && hw >= ngen::HW::XeHPC)
        return true;

    return false;
}

bool mad_t::matches_types(
        const hw_t &hw, const type_t &a, const type_t &b, const type_t &c) {
    if (a != b && !(a.is_x8() && b.is_x8())) return false;

    if (a.is_fp8() && b.is_fp8()) return true;
    if (a.is_fp4() && b.is_fp4()) return true;
    if (a.is_f64() && c.is_f64()) return true;
    if (a.is_f32() && c.is_f32()) return true;
    if (a.is_f16() && c.is_f16()) return true;
    if (a.is_f16() && c.is_f32()) return true;
    if (a.is_bf16() && c.is_f32()) return true;
    if (a.is_f32() && c.is_bf16()) return true;
    if (a.is_x8() && (c.is_x16() || c.is_x32())) return true;
    if ((a.is_x16() || a.is_x32()) && (c.is_x16() || c.is_x32())) return true;

    return false;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
