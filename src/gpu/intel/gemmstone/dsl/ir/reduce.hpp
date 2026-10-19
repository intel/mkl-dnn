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

#ifndef GEMMSTONE_DSL_IR_REDUCE_HPP
#define GEMMSTONE_DSL_IR_REDUCE_HPP

#include "gemmstone/../../dsl/ir/ir.hpp"
#include "gemmstone/../../dsl/ir/reorder.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Implements reduction of GRF buffer for given layout.
class reduce_t : public func_impl_t, public object::info_t<reduce_t> {
public:
    static func_t make(
            const dsl::layout_t &src_layout, const dsl::layout_t &dst_layout) {
        return func_t(new reduce_t(src_layout, dst_layout));
    }

    std::string str() const override {
        ostringstream_t oss;
        oss << "reduce[" << src_layout << ", " << dst_layout << "]";
        return oss.str();
    }

    IR_DEFINE_ARG_GET(dst_buf, 0)
    IR_DEFINE_ARG_GET(src_buf, 1)

    dsl::layout_t src_layout;
    dsl::layout_t dst_layout;

private:
    reduce_t(const dsl::layout_t &src_layout, const dsl::layout_t &dst_layout)
        : func_impl_t(get_info())
        , src_layout(src_layout)
        , dst_layout(dst_layout) {}
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
