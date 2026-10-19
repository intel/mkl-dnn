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

#include "gemmstone/dsl/kernel.hpp"
#include "dsl/ir/core.hpp"
#include "dsl/ir/ir.hpp"
#include "ngen_interface.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

kernel::iface_t::iface_t(const ngen::InterfaceHandler &iface)
    : kernel_name_(iface.getExternalName()) {
    for (unsigned int i = 0; i < iface.numAssignments(); i++) {
        auto &a = iface.getAssignment(i);
        if (a.exttype == ngen::ExternalArgumentType::Scalar) {
            register_arg(a.name, type_t(a.type));
        } else if (a.exttype == ngen::ExternalArgumentType::GlobalPtr) {
            register_arg(a.name, type_t::byte(type::attr_t::gm));
        } else if (a.exttype == ngen::ExternalArgumentType::LocalPtr) {
            register_arg(a.name, type_t::byte(type::attr_t::slm));
        } else {
            stub();
        }
    }
}

const ir::expr_t &kernel::iface_t::operator[](size_t idx) const {
    dsl_assert(idx < nargs());
    return args_[idx].var;
}
ir::expr_t kernel::iface_t::find_arg(
        const std::string &name, bool allow_empty) const {
    auto *arg = find_arg_impl(name);
    if (arg) return arg->var;
    if (!allow_empty) dsl_error() << "Argument not found: " << name;
    return ir::expr_t();
}

size_t kernel::iface_t::index(const std::string &name) const {
    for (size_t i = 0; i < nargs(); i++) {
        if (args_[i].name() == name) return i;
    }
    return -1;
}

ir::expr_t kernel::iface_t::register_arg(
        const std::string &name, const type_t &type) {
    auto ret = ir::var_t::make(type, name);
    register_arg(ret);
    return ret;
}

ir::expr_t kernel::iface_t::register_global(
        const std::string &name, const type_t &type) {
    auto ret = ir::var_t::make(type.with_gm(), name);
    register_arg(ret);
    return ret;
}

const std::string &kernel::iface_t::arg_t::name() const {
    return var.as<ir::var_t>().name;
}

std::string kernel_t::str() const {
    ostringstream_t oss;
    const char *prefix = "";
    oss << iface.kernel_name() << "(";
    for (size_t i = 0; i < iface.nargs(); i++) {
        oss << prefix << iface[i].str();
        prefix = ", ";
    }
    oss << "):\n" << to_string(body, 1);
    return oss.str();
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
