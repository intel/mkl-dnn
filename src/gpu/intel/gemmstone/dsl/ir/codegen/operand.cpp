/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "dsl/ir/codegen/operand.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

static std::string to_string(ngen::DataType type) {
    switch (type) {
#define CASE(name) \
    case ngen::DataType::name: return #name
        CASE(ud);
        CASE(d);
        CASE(uw);
        CASE(w);
        CASE(ub);
        CASE(b);
        CASE(df);
        CASE(f);
        CASE(uq);
        CASE(q);
        CASE(hf);
        CASE(bf);
        CASE(uv);
        CASE(v);
        CASE(vf);
        CASE(bf8);
        CASE(hf8);
        CASE(tf32);
        CASE(u4);
        CASE(s4);
        CASE(u2);
        CASE(s2);
#undef CASE
        default: return "unknown";
    }
}

std::string ngen_operand_t::str() const {
    if (!is_reg_buf_data()) return "ngen_operand";

    auto &rbd = reg_buf_data();
    auto &rb = rbd.reg_buf();
    ostringstream_t oss;
    if (rbd.type() != ngen::DataType::invalid) {
        dsl_assert(rb.blocks() == 1);
        dsl_assert(!rb.with_permute());
        oss << "r" << rbd.base() << ".";
        oss << rbd.offset() << ":";
        oss << to_string(rbd.type());
    } else {
        oss << rb.str();
    }
    return oss.str();
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
