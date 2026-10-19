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

#ifndef GEMMSTONE_DSL_IR_REORDER_HPP
#define GEMMSTONE_DSL_IR_REORDER_HPP

#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/dsl/tensor.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Implements reorder between GRF buffers in given layouts. Conversion between
// data types is supported.
class reorder_t : public func_impl_t, public object::info_t<reorder_t> {
public:
    static func_t make(dsl::layout_t src_layout, dsl::layout_t dst_layout,
            bool do_normalize = true) {
        if (do_normalize) normalize(src_layout, dst_layout);
        return func_t(
                new reorder_t(std::move(src_layout), std::move(dst_layout)));
    }

    static void normalize(dsl::layout_t &a, dsl::layout_t &b);

    std::string str() const override {
        ostringstream_t oss;
        oss << "reorder[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    dsl::layout_t src_layout;
    dsl::layout_t dst_layout;

private:
    reorder_t(dsl::layout_t src_layout, dsl::layout_t dst_layout)
        : func_impl_t(get_info())
        , src_layout(std::move(src_layout))
        , dst_layout(std::move(dst_layout)) {}
};

inline stmt_t create_reorder_stmt(const dsl::layout_t &src,
        const dsl::layout_t &dst, const expr_t &src_buf,
        const expr_t &dst_buf) {
    dsl_assert(src.ndims() == dst.ndims()) << "Layouts are incompatible.";
    dsl_assert(src.elems() == dst.elems()) << "Layouts are incompatible.";
    auto func = reorder_t::make(src, dst);
    return func.call({dst_buf, src_buf});
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
