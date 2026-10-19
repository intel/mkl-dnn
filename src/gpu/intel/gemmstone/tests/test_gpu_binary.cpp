/*******************************************************************************
* Copyright 2026 Intel Corporation
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


#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "generator/microkernel/elf.hpp"
#include "gemmstone/microkernel/fuser.hpp"
#include "gemmstone/microkernel/package.hpp"
#include "gemmstone/microkernel/shim.hpp"
#include "ngen_dynamic.hpp"
#include "ngen_elf.hpp"
#include "test_common.hpp"

using namespace gemmstone::microkernel;

#define REQUIRE(cond) \
    do { \
        bool tests_pass = true; \
        EXPECT(cond); \
        if (!tests_pass) return false; \
    } while (0)

namespace {

constexpr uint32_t microkernelID = 7;
using Binary = std::vector<uint8_t>;

struct Architecture {
    ngen::HW hw;
    uint32_t gmdid;
    const char *device;
};

constexpr Architecture architectures[] = {
        {ngen::HW::XeHPC, 0x030F0000, "pvc"},
        {ngen::HW::XeHPG, 0x030DC008, "dg2"},
        {ngen::HW::Xe2, 0x05004000, "bmg"},
        {ngen::HW::Xe3, 0x07800004, "ptl"},
        {ngen::HW::Xe3p, 0x08C2C000, "cri"},
};

Symbol named_symbol(const Binary &binary, const char *name) {
    auto headers = sectionHeaders(binary);
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i].type != SectionHeader::SymbolTable) continue;
        auto symbols = sectionEntries<Symbol>(binary, i);
        for (auto &symbol : symbols)
            if (elfString(binary, headers[i].link, symbol.name)
                    == std::string(name))
                return symbol;
    }
    return {};
}

bool check_debug_relocations(const Binary &before, const Binary &after,
        const std::string &textName, uint64_t start, uint64_t removed,
        uint64_t inserted, bool debug) {
    bool tests_pass = true;
    auto map = [&](uint64_t addr) -> uint64_t {
        if (addr <= start) return addr;
        if (addr < start + removed) return start;
        return addr - removed + inserted;
    };
    bool checked = false;
    auto headers = sectionHeaders(before);
    for (size_t s = 0; s < headers.size(); s++) {
        if (headers[s].type != SectionHeader::RelocationWithAddend) continue;
        auto symtab = headers[s].link;
        if (headers[symtab].type != SectionHeader::SymbolTable) continue;
        auto syms = sectionEntries<Symbol>(before, symtab);
        auto entries = sectionEntries<RelocationWithAddend>(before, s);
        auto afterEntries = sectionEntries<RelocationWithAddend>(after, s);
        REQUIRE(afterEntries.size() == entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            auto &symbol = syms.at(entries[i].info >> 32);
            auto section = sectionName(before, symbol.section);
            auto name = elfString(before, headers[symtab].link, symbol.name);
            if (std::strncmp(section, ".debug", 6) != 0 || name != textName)
                continue;
            checked = true;
            EXPECT(uint64_t(afterEntries[i].addend)
                    == map(uint64_t(entries[i].addend)));
        }
    }
    if (debug) EXPECT(checked);
    return tests_pass;
}

// Text offset of the instruction whose 32-bit immediate is `imm`, or `size`
// if absent. The immediate is the last dword of an uncompacted instruction.
uint64_t find_immediate(const Binary &binary, const SectionHeader &text,
        uint32_t imm) {
    for (uint64_t offset = 0; offset + 16 <= text.size; offset += 8) {
        uint32_t value;
        std::memcpy(&value, &binary[text.offset + offset + 12], sizeof(value));
        if (value == imm) return offset;
    }
    return text.size;
}

// Microkernel body: a short variant to shrink the host, or a forward jump
// plus filler to grow it.
template <ngen::HW hw>
class Fragment : public ngen::ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)
public:
    explicit Fragment(bool shortKernel)
        : ngen::ELFCodeGenerator<hw>(ngen::DebugConfig()) {
        setDefaultNoMask();
        setDefaultAutoSWSB();
        if (!shortKernel) {
            ngen::Label skip;
            jmpi(1, skip);
            mov<uint32_t>(1, r65, 42);
            mark(skip);
            for (int i = 0; i < 32; i++)
                nop();
        }
        add<uint32_t>(1, r65, r64, 1000);
    }
};

// Shim (with embedded microkernel) generated as for production kernels.
template <ngen::HW hw>
std::string microkernel_shim(uint32_t gmdid, bool shortKernel) {
    Fragment<hw> fragment(shortKernel);
    auto grf = ngen::GRF::bytes(hw);
    Package package;
    package.protocol = Protocol("gpu_binary_ukernel",
            {{"value", Protocol::Argument::In, StructuredType::u32}}, {});
    package.binary = fragment.getCode();

    constexpr int argumentGRF = 64;
    package.arguments = {{"value", {RegisterRange(argumentGRF * grf, 4)},
            StructuredType::any, {}}};
    package.argumentBase = argumentGRF * grf;
    package.gmdidCompat = gmdid;
    if (package.finalize() != Package::Status::Success)
        throw std::runtime_error("Cannot finalize microkernel package");
    ShimOptions options;
    options.subgroupSize = 32;
    options.microkernelID = microkernelID;
    return generateShim(package, HostLanguage::OpenCL_C, options);
}

std::string microkernel_shim(const Architecture &arch, bool shortKernel) {
    using ngen::HW;
    auto g = arch.gmdid;
    switch (arch.hw) {
        case HW::XeHPG: return microkernel_shim<HW::XeHPG>(g, shortKernel);
        case HW::XeHPC: return microkernel_shim<HW::XeHPC>(g, shortKernel);
        case HW::Xe2: return microkernel_shim<HW::Xe2>(g, shortKernel);
        case HW::Xe3: return microkernel_shim<HW::Xe3>(g, shortKernel);
        case HW::Xe3p: return microkernel_shim<HW::Xe3p>(g, shortKernel);
        default: throw std::runtime_error("Unsupported architecture");
    }
}

// The host calls the microkernel within a branch.
constexpr const char *hostName = "host";
const char *hostKernel = R"CL(
__attribute__((intel_reqd_sub_group_size(32)))
__kernel void host(__global uint *out, uint k) {
    if (out[0]) gpu_binary_ukernel(k);
}
)CL";

// libocloc, loaded at runtime so the test builds without it and skips when it
// is unavailable. Signatures from compute-runtime's ocloc_api.h.
class Ocloc {
public:
    Ocloc()
        : invoke_(symbol<Invoke>("oclocInvoke"))
        , free_(symbol<FreeOutput>("oclocFreeOutput")) {}
    bool available() const { return invoke_ && free_; }

    // Compiles OpenCL C `source` for `device`; empty on failure.
    Binary compile(const std::string &source, const char *device,
            bool debug) const {
        std::string options = "-cl-std=CL2.0";
        if (debug) options += " -g";
        Binary binary;
        run("compile", "gpu_binary.cl",
                Binary(source.begin(), source.end() + 1),
                {"-device", device, "-options", options.c_str()}, ".bin",
                &binary);
        return binary;
    }

    bool validate(const Binary &binary) const {
        return run("validate", "gpu_binary.bin", binary);
    }

    // Disassembles exactly `symbol`'s code with `ocloc disasm`. On IGA failure
    // ocloc still succeeds, but dumps raw bytes instead of `<section>.asm`:
    bool disassemble(const Binary &binary, const Symbol &symbol) const {
        auto copy = binary;
        auto headers = sectionHeaders(copy);
        auto &text = headers[symbol.section];
        text.offset += symbol.value - text.addr;
        text.size = symbol.size;
        storeSectionHeaders(copy, headers);
        return run("disasm", "gpu_binary.bin", copy,
                {"-dump", "gpu_binary_disasm/"},
                sectionName(binary, symbol.section) + std::string(".asm"));
    }

private:
    using Invoke = int (*)(unsigned, const char **, uint32_t, const uint8_t **,
            const uint64_t *, const char **, uint32_t, const uint8_t **,
            const uint64_t *, const char **, uint32_t *, uint8_t ***,
            uint64_t **, char ***);
    using FreeOutput = int (*)(uint32_t *, uint8_t ***, uint64_t **, char ***);

    template <typename F>
    static F symbol(const char *name) {
        for (auto *lib : {"libocloc.so", "libocloc.so.1"})
            if (auto *f = ngen::dynamic::findSymbol(lib, name))
                return reinterpret_cast<F>(f);
        return nullptr;
    }

    // Runs `ocloc <command> -file <file> <args...>` on in-memory `input`.
    // Succeeds if ocloc does and produces an output whose name ends with
    // `suffix`, which is stored in `output`. On failure, prints the log.
    bool run(const char *command, const char *file, const Binary &input,
            std::vector<const char *> args = {}, const std::string &suffix = "",
            Binary *output = nullptr) const {
        args.insert(args.begin(), {"ocloc", command, "-file", file});
        const uint8_t *data = input.data();
        uint64_t length = input.size();
        uint32_t count = 0;
        uint8_t **outData = nullptr;
        uint64_t *outLength = nullptr;
        char **outName = nullptr;
        bool success = invoke_(unsigned(args.size()), args.data(), 1, &data,
                               &length, &file, 0, nullptr, nullptr, nullptr,
                               &count, &outData, &outLength, &outName)
                == 0;
        bool found = suffix.empty();
        std::string log;
        for (uint32_t i = 0; i < count; i++) {
            std::string name = outName[i];
            auto *d = outData[i];
            if (name == "stdout.log") log.assign(d, d + outLength[i]);
            if (!found && name.size() >= suffix.size()
                    && name.compare(name.size() - suffix.size(), suffix.size(),
                               suffix)
                            == 0) {
                found = true;
                if (output) output->assign(d, d + outLength[i]);
            }
        }
        free_(&count, &outData, &outLength, &outName);
        if (success && found) return true;
        printf("ocloc %s:\n%s\n", command, log.c_str());
        return false;
    }

    Invoke invoke_;
    FreeOutput free_;
};

// Checks that apply to any GPU binary containing `kernel`.
bool check_binary(
        const Ocloc &ocloc, const Binary &binary, const char *kernel) {
    bool tests_pass = true;
    EXPECT(ocloc.validate(binary));
    auto symbol = named_symbol(binary, kernel);
    REQUIRE(symbol.size > 0);
    EXPECT(ocloc.disassemble(binary, symbol));
    return tests_pass;
}

// Compiles the host kernel with a microkernel shim, fuses the microkernel into
// it, then checks the result against the original.
bool test_fusion(const Ocloc &ocloc, const Architecture &arch, bool debug,
        bool shortKernel) {
    bool tests_pass = true;
    auto source = microkernel_shim(arch, shortKernel) + hostKernel;
    auto binary = ocloc.compile(source, arch.device, debug);
    REQUIRE(!binary.empty());
    auto original = binary;
    auto host = named_symbol(binary, hostName);
    REQUIRE(host.size > 0);

    // Locate the splice independently of the fuser.
    auto text = sectionHeader(binary, host.section);
    auto start = find_immediate(binary, text, sigilStart ^ microkernelID);
    auto end = find_immediate(binary, text, sigilEnd ^ microkernelID) + 16;
    REQUIRE(start < end && end <= text.size);
    auto marker = source.find(sigilBinary);
    auto hex = source.find(' ', marker) + 1;
    auto codeBytes = (source.find('\n', hex) - hex) / 2;
    auto inserted = codeBytes + (start & 8);
    if (shortKernel) REQUIRE(inserted < end - start);

    fuse(binary, source.c_str(), ngen::GRF::bytes(arch.hw));
    EXPECT(check_debug_relocations(original, binary,
            sectionName(original, host.section), start, end - start, inserted,
            debug));

    auto newText = sectionHeader(binary, host.section);
    auto newHost = named_symbol(binary, hostName);
    EXPECT(newText.size == text.size - (end - start) + inserted);
    EXPECT(newHost.value - newText.addr == host.value - text.addr);
    EXPECT(newHost.size + (end - start) == host.size + inserted);
    EXPECT(check_binary(ocloc, binary, hostName));

    return tests_pass;
}

} // namespace

int main() {
    const Ocloc ocloc;
    if (!ocloc.available()) {
        printf("SKIP: libocloc not found\n");
        return 0;
    }
    bool pass = true;
    for (const auto &arch : architectures)
        for (bool debug : {false, true})
            for (bool shortKernel : {false, true}) {
                std::string name = std::string("test ") + arch.device
                        + " microkernel fusion"
                        + (shortKernel ? " shrinking host" : "")
                        + (debug ? " with debug info" : "");
                bool ok = false;
                try {
                    ok = test_fusion(ocloc, arch, debug, shortKernel);
                } catch (const std::exception &e) {
                    printf("exception: %s\n", e.what());
                }
                // Tagged so CI can identify the failing case.
                printf("%s: %s\n", ok ? "PASSED" : "FAILED", name.c_str());
                pass &= ok;
            }
    return pass ? 0 : 1;
}
