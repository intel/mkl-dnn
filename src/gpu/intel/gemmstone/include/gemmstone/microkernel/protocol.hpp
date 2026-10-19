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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_PROTOCOL_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_PROTOCOL_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

// Describes the type of a microkernel argument (scalar/pointer/tensor).
struct StructuredType {
    enum Type { // Element data type
        u64,
        s64,
        u32,
        s32,
        u16,
        s16,
        u8,
        s8,
        u4,
        s4,
        u3, //    integral
        f64,
        f32,
        f16,
        bf16,
        bf8,
        hf8,
        f8_e8m0,
        f4_e2m1, //    floating-point
        any, //    unspecified
    } type
            = Type::any;
    enum Format { Scalar, GlobalPointer, LocalPointer, Tensor } format = Scalar;
    int ndims = 1;

    StructuredType() = default;
    StructuredType(Type type_) : type(type_) {}
    StructuredType(Format format_) : format(format_) {}
    StructuredType(int ndims_) : format(Tensor), ndims(ndims_) {}
};

// A protocol describes a class of microkernels that provide the same functionality
//  and share a high-level interface.
class Protocol {
public:
    // Description of a single argument from a protocol's prototype.
    struct Argument {
        const char *name;
        enum { In = 0b01, Out = 0b10, InOut = In | Out } direction;
        StructuredType stype;

        bool in() const { return direction & In; }
        bool out() const { return direction & Out; }
    };

    // Description of a single protocol setting.
    struct Setting {
        const char *name;
    };

    Protocol() = default;
    Protocol(std::string name, std::vector<Argument> arguments,
            std::vector<Setting> settings)
        : kernelBaseName_(std::move(name)), arguments_(std::move(arguments)), settings_(std::move(settings)) {}
    const std::string &kernelBaseName() const { return kernelBaseName_; }
    const std::vector<Argument> &arguments() const {
        return arguments_;
    }
    const std::vector<Setting> &settings() const { return settings_; }

private:
    std::string kernelBaseName_;
    std::vector<Argument> arguments_;
    std::vector<Setting> settings_;
};

}
GEMMSTONE_NAMESPACE_END

#endif /* header guard */
