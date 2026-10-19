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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_PACKAGE_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_PACKAGE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include "gemmstone/microkernel/protocol.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

struct Argument;
struct RegisterRange;
struct Setting;

struct ClobberSet {
    static constexpr int maxRegs = 512;
    static constexpr int maxAcc = 16;
    static constexpr int maxFlag = 16;
    std::array<bool, maxRegs> clobbered = {};
    std::array<bool, maxAcc> accClobbered = {};
    std::array<bool, maxFlag> flagClobbered = {};

    void clear() { clobbered.fill(false); accClobbered.fill(false); flagClobbered.fill(false); }
    bool empty() const { return std::none_of(clobbered.begin(), clobbered.end(), [](bool b) { return b; }); }
    size_t size() const { return clobbered.size(); }
    bool operator[](uint32_t reg) const { return clobbered[reg]; }
    bool &operator[](uint32_t reg) { return clobbered[reg]; }

    bool acc(uint32_t idx) const { return idx < maxAcc && accClobbered[idx]; }
    bool flag(uint32_t idx) const { return idx < maxFlag && flagClobbered[idx]; }

    void addAcc(uint32_t start, uint32_t len) {
        for (uint32_t i = start; i < start + len && i < maxAcc; i++)
            accClobbered[i] = true;
    }
    void addFlag(uint32_t start, uint32_t len) {
        for (uint32_t i = start; i < start + len && i < maxFlag; i++)
            flagClobbered[i] = true;
    }

    std::string str() const {
        std::string out = "{";
        int i = 0;
        while (i < maxRegs) {
            int start = i;
            while (i < maxRegs && clobbered[i]) i++;
            i++;
            if(i == start + 1) continue;
            if (out.size() > 1) out += ", ";
            out += "r" + std::to_string(start);
            if (i - 2 > start) out += "-r" + std::to_string(i - 2);
        }
        out += "}";
        return out;
    }

    void add(uint32_t regStart, uint32_t regLen) {
        for (uint32_t i = regStart; i < regStart + regLen && i < maxRegs; i++)
            clobbered[i] = true;
    }
};

// Microkernel package.
// Fields marked [*] are automatically filled in by finalize().
struct Package {

    // Information on a single configuration setting.
    struct Setting {
        std::string name; // Setting name
        int value; // Setting numeric value
    };

    /* Identifiers */
    Protocol protocol; // Protocol implemented by microkernel
    uint64_t luid; // Unique package ID for use in catalog [*]
    std::vector<uint8_t> providerID; // Optional free-form identifier for use by microkernel provider

    /* Code */
    std::vector<uint8_t> binary; // Raw binary blob

    /* Register usage */
    std::vector<Argument> arguments; // Input and output arguments for microkernel
    std::vector<RegisterRange> clobbers; // Registers clobbered by microkernel (includes arguments) [*]
    uint32_t argumentBase = 0; // First GRF byte used by microkernel; host thread payload lies below

    /* Requirements */
    uint32_t gmdidCompat; // Compatible GMDID
    int grfMin = 0; // Minimum GRF size [*]
    int barrierCount = 0; // Number of barriers used by microkernel
    bool systolic = false; // Does microkernel use systolic array? [*]

    /* Configuration */
    std::vector<Setting> settings; // Description of this microkernel's configuration (WG size, tile size, etc.) for host kernel to interpret

    int getSetting(const char *name) const {
        for (auto &setting : settings)
            if (setting.name == name) return setting.value;

        throw std::runtime_error(
                std::string("Microkernel package does not provide requested setting: ")
                                 + name);
    }

    enum class Status {
        Pending,
        Success,
        UncertainClobbers,
        UnsupportedHW,
    };

    Status status = Status::Pending;

    // Analyzes the package and deduces information from the raw microkernel binary.
    // Note: if any ARF registers are used, they must be declared in knownClobbers, which
    // indicates that the microkernel has save/restore code for those registers. Only supports
    // acc/flag ARF registers so far
    Status finalize(const ClobberSet &knownClobbers = {});
};

// Contiguous span of register space.
struct RegisterRange {
    uint32_t boffset = 0; // Byte offset into GRF
    uint32_t blen = 0; // Length of range in bytes

    RegisterRange() = default;
    RegisterRange(uint32_t boffset_, uint32_t blen_)
        : boffset(boffset_), blen(blen_) {}
};

// Encapsulation of tensor size information.
struct TensorConfig {
    static constexpr int maxDims = 4;
    std::array<int, maxDims> dims
            = {1, 1, 1, 1}; // Tensor tile size (elements per dimension)
    std::array<int, maxDims> block = {1, 1, 1,
            1}; // Block sizes within tile (equal to dims if only one block)

    int elements() const {
        int result = 1;
        for (auto d : dims)
            result *= d;
        return result;
    }

    int blockElements() const {
        int result = 1;
        for (auto d : block)
            result *= d;
        return result;
    }

    bool blocked() const {
        for (int i = 0; i < maxDims; i++)
            if (block[i] < dims[i]) return true;
        return false;
    }

    int blocks() const {
        int result = 1;
        for (int i = 0; i < maxDims; i++)
            result *= dims[i] / block[i];
        return result;
    }
};

// Information on a single argument (input/output).
struct Argument {
    std::string name; // Argument name
    std::vector<RegisterRange> location; // Register location(s)
    StructuredType::Type actualType
            = StructuredType::any; // Type, if not specified by protocol
    TensorConfig sizes; // Tensor size, for tensor arguments
};

}
GEMMSTONE_NAMESPACE_END

#endif /* header guard */
