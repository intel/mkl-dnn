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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "generator/microkernel/elf.hpp"
#include "generator/microkernel/payload.hpp"
#include "gemmstone/microkernel/fuser.hpp"
#include "ngen_elf.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

static void fixupJumpTargets(uint8_t *start, size_t len, ptrdiff_t adjust);

/* Host payloads, checked against the microkernel's register placement. */
struct PayloadCheck {
    const std::vector<KernelInfo> *kernels;
    uint32_t argumentBase;
    int grfBytes;
    bool *validated;
};

static void checkPayload(const PayloadCheck &check, const char *kernelName);

namespace {

struct TextSplice {
    int section;
    uint64_t start;
    uint64_t removed;
    uint64_t inserted;

    uint64_t map(uint64_t address) const {
        if (address <= start) return address;
        if (address < start + removed) return start;
        return address - removed + inserted;
    }
};

// Debug sections refer to text through symbols named after the text section.
bool isTextAlias(const std::vector<uint8_t> &binary, int symtab,
        const Symbol &symbol, int text) {
    if (symbol.section >= fileHeader(binary).sectionCount) return false;
    auto link = sectionHeader(binary, symtab).link;
    return std::strncmp(
                   sectionName(binary, symbol.section), ".debug", 6)
            == 0
            && std::strcmp(elfString(binary, link, symbol.name),
                       sectionName(binary, text))
            == 0;
}

template <typename Map>
void fixupAddend(std::vector<uint8_t> &, int, RelocationWithAddend &rel,
        const TextSplice &, Map map) {
    rel.addend = int64_t(map(uint64_t(rel.addend)));
}

// REL addends are stored in the relocated field.
template <typename Map>
void fixupAddend(std::vector<uint8_t> &binary, int target,
        const Relocation &rel, const TextSplice &splice, Map map) {
    if (target == splice.section) return;
    enum { SymAddr = 1, SymAddr32 = 2 };
    uint64_t bytes = 0;
    switch (uint32_t(rel.info)) {
        case SymAddr: bytes = 8; break;
        case SymAddr32: bytes = 4; break;
        default:
            if (std::strncmp(
                        sectionName(binary, target), ".debug", 6)
                    == 0)
                throw std::runtime_error(
                        "Unsupported debug relocation of microkernel text");
            throw std::runtime_error(
                    "Unsupported relocation of microkernel host code");
    }
    auto t = sectionHeader(binary, target);
    if (!sectionInBounds(binary, t) || rel.offset > t.size
            || bytes > t.size - rel.offset)
        throw std::runtime_error("Invalid zebin relocation");
    auto *field = binary.data() + t.offset + rel.offset;
    uint64_t addend = 0;
    std::memcpy(&addend, field, bytes);
    addend = map(addend);
    if (bytes < 8 && (addend >> 32) != 0)
        throw std::runtime_error("Relocation addend out of range");
    std::memcpy(field, &addend, bytes);
}

template <typename R>
void fixupRelocations(
        std::vector<uint8_t> &binary, int s, const TextSplice &splice) {
    auto rels = sectionEntries<R>(binary, s);
    if (rels.empty()) return;
    auto table = sectionHeader(binary, s);
    if (sectionHeader(binary, table.link).type
            != SectionHeader::SymbolTable)
        throw std::runtime_error("Invalid zebin relocation table");
    auto symbols = sectionEntries<Symbol>(binary, table.link);
    auto target = int(table.info);
    auto textAddr = sectionHeader(binary, splice.section).addr;
    auto spliceEnd = splice.start + splice.removed;

    for (auto &rel : rels) {
        if (target == splice.section) {
            if (rel.offset >= spliceEnd)
                rel.offset = rel.offset - splice.removed + splice.inserted;
            else if (rel.offset >= splice.start)
                throw std::runtime_error(
                        "Relocation inside replaced microkernel shim");
        }

        auto symbolIndex = rel.info >> 32;
        if (symbolIndex >= symbols.size())
            throw std::runtime_error("Invalid zebin relocation");
        const auto &symbol = symbols[symbolIndex];

        // Text offsets of the symbol before and after the splice. Text aliases
        // are not moved, so their addends absorb the splice.
        uint64_t oldSymbol = 0, newSymbol = 0;
        if (symbol.section == splice.section) {
            if (symbol.value < textAddr)
                throw std::runtime_error("Invalid zebin text symbol");
            oldSymbol = symbol.value - textAddr;
            newSymbol = oldSymbol >= spliceEnd
                    ? oldSymbol - splice.removed + splice.inserted
                    : oldSymbol;
        } else if (isTextAlias(binary, int(table.link), symbol, splice.section))
            oldSymbol = newSymbol = symbol.value;
        else
            continue;

        fixupAddend(binary, target, rel, splice, [&](uint64_t addend) {
            return splice.map(oldSymbol + addend) - newSymbol;
        });
    }
    storeSectionEntries(binary, s, rels);
}

void fixupElf(std::vector<uint8_t> &binary, const TextSplice &splice) {
    auto sheaders = sectionHeaders(binary);
    auto textAddr = sectionHeader(binary, splice.section).addr;
    auto spliceEnd = splice.start + splice.removed;

    for (int s = 0; s < int(sheaders.size()); s++) {
        if (sheaders[s].type == SectionHeader::Relocation)
            fixupRelocations<Relocation>(binary, s, splice);
        if (sheaders[s].type == SectionHeader::RelocationWithAddend)
            fixupRelocations<RelocationWithAddend>(binary, s, splice);
    }

    // Relocations above use the original symbol values.
    for (int s = 0; s < int(sheaders.size()); s++) {
        if (sheaders[s].type != SectionHeader::SymbolTable) continue;
        auto symbols = sectionEntries<Symbol>(binary, s);
        for (auto &symbol : symbols) {
            if (symbol.section != splice.section) continue;
            if (symbol.value < textAddr)
                throw std::runtime_error("Invalid zebin text symbol");
            auto start = symbol.value - textAddr;
            if (start >= spliceEnd) {
                symbol.value = symbol.value - splice.removed + splice.inserted;
            } else if (start <= splice.start
                    && symbol.size >= spliceEnd - start) {
                symbol.size = symbol.size - splice.removed + splice.inserted;
            } else if (symbol.size > 0
                    && (start > splice.start
                            || symbol.size > splice.start - start)) {
                throw std::runtime_error(
                        "Zebin symbol overlaps replaced microkernel shim");
            }
        }
        storeSectionEntries(binary, s, symbols);
    }
}

} // namespace

/* Compacted no-op, used to keep microkernel instructions 16-byte aligned.
   Encodings are per-arch compaction table entries; verify additions with
   iga64 (sync.nop {Compacted} / mov (1|M0) null:ud r0.0:ud {Compacted}). */
static const uint8_t *alignmentFiller(ngen::HW hw) {
    static const uint8_t syncNop[8] = {0x01, 0, 0, 0xE8, 0x01, 0, 0x11, 0};
    static const uint8_t movNull[8] = {0x61, 0, 0x84, 0xBC, 0, 0, 0, 0};
    static const uint8_t movNullXe2[8] = {0x61, 0, 0x84, 0xFC, 0, 0, 0x10, 0};
    static const uint8_t movNullXe3p[8] = {0x61, 0, 0x84, 0xA4, 0, 0, 0, 0};
    switch (hw) {
        case ngen::HW::Unknown:
        case ngen::HW::Gen9:
        case ngen::HW::Gen10:
        case ngen::HW::Gen11:
        case ngen::HW::XeLP:
        case ngen::HW::XeHP: return nullptr;
        case ngen::HW::XeHPG: return syncNop;
        case ngen::HW::XeHPC: return movNull;
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return movNullXe2;
        case ngen::HW::Xe3p: return movNullXe3p;
    }
    return nullptr;
}

static void fuse(std::vector<uint8_t> &binary,
        const std::vector<uint8_t> &microkernel, long id,
        const PayloadCheck *check, const uint8_t *filler) {
    auto base = binary.data();
    auto bytes = binary.size();

    auto fheader = fileHeader(binary);

    bool foundZeInfo = false;
    std::vector<std::pair<SectionHeader, int>> textSections;

    auto sheaders = sectionHeaders(binary);

    for (int s = 0; s < fheader.sectionCount; s++) {
        switch (sheaders[s].type) {
            case SectionHeader::Type::ZeInfo: foundZeInfo = true; break;
            case SectionHeader::Type::Program: {
                std::string sname = sectionName(binary, s);
                if (sname == ".text.Intel_Symbol_Table_Void_Program")
                    continue;
                if (sname.substr(0, 6) != ".text.") continue;
                textSections.emplace_back(sheaders[s], s);
                break;
            }
            default: break;
        }
    }

    if (!foundZeInfo || textSections.empty())
        throw std::runtime_error(
                "IGC did not generate a valid zebin program binary");

    for (auto &entry : textSections) {
        auto text = entry.first;
        int textSectionID = entry.second;
        if (!sectionInBounds(binary, text)) continue;

        auto *insn = reinterpret_cast<const uint32_t *>(base + text.offset);
        auto *iend = reinterpret_cast<const uint32_t *>(
                base + text.offset + text.size);

        const uint8_t *spliceStart = nullptr;
        const uint8_t *spliceEnd = nullptr;

        for (; insn < iend; insn += 4) {
            if (insn[0] & (1u << 29))
                insn -= 2;
            else if (insn[3] == (sigilStart ^ id))
                spliceStart = reinterpret_cast<const uint8_t *>(insn);
            else if (insn[3] == (sigilEnd ^ id)) {
                spliceEnd = reinterpret_cast<const uint8_t *>(insn);
                break;
            }
        }

        if (!spliceStart || !spliceEnd) continue;

        // Validate the argument placement against the selected base.
        if (check)
            checkPayload(*check,
                    sectionName(binary, textSectionID) + 6);

        auto removeBytes = spliceEnd - spliceStart + 16;

        size_t before = spliceStart - base;
        auto after = bytes - before - removeBytes;

        auto kbefore = before - text.offset;

        /* Keep the microkernel's (uncompacted) instructions 16-byte aligned
           so they do not straddle instruction cache lines. */
        std::vector<uint8_t> blob;
        if (filler && (kbefore & 8)) blob.assign(filler, filler + 8);
        blob.insert(blob.end(), microkernel.begin(), microkernel.end());

        ptrdiff_t sizeAdjust = blob.size() - removeBytes;

        auto kafter = text.size - kbefore - removeBytes;

        std::vector<uint8_t> newBinary(bytes + sizeAdjust);
        auto newBase = newBinary.data();

        memmove(newBase, base, before);
        memmove(newBase + before, blob.data(), blob.size());
        memmove(newBase + before + blob.size(),
                spliceStart + removeBytes, after);

        fixupJumpTargets(newBase + text.offset, kbefore, +sizeAdjust);
        fixupJumpTargets(
                newBase + before + blob.size(), kafter, -sizeAdjust);

        if (fheader.sectionTableOff > before) {
            fheader.sectionTableOff += sizeAdjust;
            storeFileHeader(newBinary, fheader);
        }

        sheaders = sectionHeaders(newBinary);
        sheaders[textSectionID].size += sizeAdjust;
        for (int s = 0; s < fheader.sectionCount; s++)
            if (sheaders[s].offset > before) sheaders[s].offset += sizeAdjust;
        storeSectionHeaders(newBinary, sheaders);

        TextSplice splice {
                textSectionID, kbefore, uint64_t(removeBytes), blob.size()};
        fixupElf(newBinary, splice);

#ifdef SPLICE_DEBUG
        std::ofstream dump0("original." + std::to_string(id) + ".bin");
        dump0.write((const char *)binary.data(), binary.size());

        std::ofstream dump("patched." + std::to_string(id) + ".bin");
        dump.write((const char *)newBinary.data(), newBinary.size());
#endif

        std::swap(binary, newBinary);

        // Tail-recurse to handle any further instances of this microkernel
        fuse(binary, microkernel, id, check, filler);
        return;
    }
}

// Drop the zebin SPIR-V section (ZebinSpirv -> Null) so the runtime can't
// rebuild the program from stale IR and lose the spliced-in microkernels.
static void stripIntermediateRepresentation(std::vector<uint8_t> &binary) {
    if (!isZebin(binary)) return;
    auto fheader = fileHeader(binary);
    auto sheaders = sectionHeaders(binary);
    for (int s = 0; s < fheader.sectionCount; s++)
        if (sheaders[s].type == SectionHeader::ZebinSpirv)
            sheaders[s].type = SectionHeader::Null;
    storeSectionHeaders(binary, sheaders);
}

static bool findZeInfo(
        const std::vector<uint8_t> &binary, const char *&text, size_t &length) {
    if (!isZebin(binary)) return false;
    auto fheader = fileHeader(binary);

    auto sheaders = sectionHeaders(binary);
    for (int s = 0; s < fheader.sectionCount; s++) {
        if (sheaders[s].type != SectionHeader::Type::ZeInfo) continue;
        if (!sectionInBounds(binary, sheaders[s])) return false;
        text = reinterpret_cast<const char *>(
                binary.data() + sheaders[s].offset);
        length = size_t(sheaders[s].size);
        return true;
    }
    return false;
}

// vISA does not reserve registers against IGC's thread payload, so a microkernel
// placed too low silently corrupts the host kernel's arguments.
static void checkPayload(const PayloadCheck &check, const char *kernelName) {
    for (auto &kernel : *check.kernels) {
        if (kernel.name != kernelName) continue;
        *check.validated = true;

        auto payloadEnd = payloadEndBytes(kernel, check.grfBytes);
        if (payloadEnd <= check.argumentBase) return;

        auto argumentBytes = payloadEnd - crossthreadBase(kernel, check.grfBytes);
        throw std::runtime_error("Microkernel registers (r"
                + std::to_string(check.argumentBase / check.grfBytes)
                + " and up) overlap the thread payload of kernel "
                + kernel.name + ", which ends at r"
                + std::to_string((payloadEnd - 1) / check.grfBytes)
                + ". Raise HostPayload::argumentBytes to "
                + std::to_string(argumentBytes) + " or more");
    }
}

bool fuse(std::vector<uint8_t> &binary, const char *source, int grfBytes) {
    std::vector<uint8_t> microkernel;
    const auto sigilLen = strlen(sigilBinary);

    auto filler = alignmentFiller(
            ngen::ELFCodeGenerator<ngen::HW::Unknown>::getBinaryArch(binary));

    auto toNybble = [](char c) {
        return ((c >= 'A') ? (c - 'A' + 10) : (c - '0')) & 0xF;
    };

    /* Read the payload layout before fusing moves the .ze_info section. */
    std::vector<KernelInfo> kernels;
    const char *zeInfo = nullptr;
    size_t zeInfoLength = 0;
    bool haveLayout = grfBytes > 0 && findZeInfo(binary, zeInfo, zeInfoLength)
            && parseZeInfo(zeInfo, zeInfoLength, kernels);
    bool validated = false;

    for (const char *s = std::strstr(source, sigilBinary); s;
            s = std::strstr(s, sigilBinary)) {
        s += sigilLen;
        char *after;
        long id = strtol(s, &after, 10);
        uint32_t argumentBase = 0;
        if (*after == ':')
            argumentBase = uint32_t(strtoul(after + 1, &after, 10));
        microkernel.clear();
        for (s = after + 1; *s != '\n'; s += 2) {
            if (!s[0] || !s[1]) break;
            microkernel.push_back(static_cast<uint8_t>(
                    (toNybble(s[0]) << 4) | toNybble(s[1])));
        }
        PayloadCheck check {&kernels, argumentBase, grfBytes, &validated};
        fuse(binary, microkernel, id,
                (haveLayout && argumentBase > 0) ? &check : nullptr, filler);
    }
    stripIntermediateRepresentation(binary);
    return validated;
}

static void fixupJumpTargets(uint8_t *start, size_t len, ptrdiff_t adjust) {
    auto istart = reinterpret_cast<int32_t *>(start);
    auto iend = reinterpret_cast<int32_t *>(start + len);

    for (auto insn = istart; insn < iend; insn += 4) {
        if (insn[0] & (1u << 29)) {
            insn -= 2; /* skip compacted instructions */
            continue;
        }
        uint8_t op = insn[0] & 0xFF;
        if ((op & 0xF0) != 0x20) continue; /* skip non-jumps */
        if (op == 0x2B || op == 0x2D) continue; /* skip ret/calla */
        bool hasUIP = (op == 0x22 || op == 0x23 || op == 0x24 || op == 0x28
                || op == 0x2A || op == 0x2E);

        auto jumpFixup = [=](int32_t &ip) {
            auto target = ((insn - istart) << 2) + ip;
            if (target < 0 || target >= ptrdiff_t(len))
                ip += static_cast<int32_t>(adjust);
        };

        if (hasUIP) jumpFixup(insn[2]);
        jumpFixup(insn[3]);
    }
}

bool hasMicrokernels(const char *source) {
    return std::strstr(source, sigilBinary);
}

}
GEMMSTONE_NAMESPACE_END
