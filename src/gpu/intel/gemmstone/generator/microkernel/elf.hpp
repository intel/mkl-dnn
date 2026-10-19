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

#ifndef GEMMSTONE_GENERATOR_MICROKERNEL_ELF_HPP
#define GEMMSTONE_GENERATOR_MICROKERNEL_ELF_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gemmstone/config.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

enum {
    ELFMagic = 0x464C457F, // '\x7FELF'
    ELFClass64 = 2,
    ELFLittleEndian = 1,
    ELFVersion1 = 1,
    ELFRelocatable = 1,
};
enum { MachineIntelGT = 205, ZebinExec = 0xFF12 };

struct FileHeader {
    uint32_t magic;
    uint8_t elfClass;
    uint8_t endian;
    uint8_t version;
    uint8_t osABI;
    uint64_t pad;
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint64_t entrypoint;
    uint64_t programHeaderOff;
    uint64_t sectionTableOff;
    uint32_t flags;
    uint16_t size;
    uint16_t programHeaderSize;
    uint16_t programTableEntries;
    uint16_t sectionHeaderSize;
    uint16_t sectionCount;
    uint16_t strTableIndex;
};

struct Relocation {
    uint64_t offset;
    uint64_t info;
};

struct RelocationWithAddend {
    uint64_t offset;
    uint64_t info;
    int64_t addend;
};

struct Symbol {
    uint32_t name;
    uint8_t info;
    uint8_t other;
    uint16_t section;
    uint64_t value;
    uint64_t size;
};

struct SectionHeader {
    uint32_t name;
    enum Type : uint32_t {
        Null,
        Program,
        SymbolTable = 2,
        StringTable = 3,
        RelocationWithAddend = 4,
        Note = 7,
        NoBits = 8,
        Relocation = 9,
        ZebinSpirv = 0xFF000009,
        ZeInfo = 0xFF000011
    } type;
    uint64_t flags;
    uint64_t addr;
    uint64_t offset;
    uint64_t size;
    uint32_t link;
    uint32_t info;
    uint64_t alignx10;
    uint64_t entrySize;
};

inline bool isZebin(const std::vector<uint8_t> &binary) {
    if (binary.size() < sizeof(FileHeader)) return false;
    FileHeader header;
    std::memcpy(&header, binary.data(), sizeof(header));
    return header.magic == ELFMagic && header.elfClass == ELFClass64
            && header.endian == ELFLittleEndian
            && (header.version == 0 || header.version == ELFVersion1)
            && (header.type == ZebinExec || header.type == ELFRelocatable)
            && header.sectionHeaderSize == sizeof(SectionHeader)
            && header.sectionTableOff <= binary.size()
            && (binary.size() - header.sectionTableOff) / sizeof(SectionHeader)
            >= header.sectionCount;
}

inline FileHeader fileHeader(const std::vector<uint8_t> &binary) {
    if (!isZebin(binary)) throw std::runtime_error("Invalid zebin file header");
    FileHeader header;
    std::memcpy(&header, binary.data(), sizeof(header));
    return header;
}

inline void storeFileHeader(
        std::vector<uint8_t> &binary, const FileHeader &header) {
    if (binary.size() < sizeof(header))
        throw std::runtime_error("Invalid zebin file header");
    std::memcpy(binary.data(), &header, sizeof(header));
}

inline std::vector<SectionHeader> sectionHeaders(
        const std::vector<uint8_t> &binary) {
    auto header = fileHeader(binary);
    std::vector<SectionHeader> sections(header.sectionCount);
    if (!sections.empty())
        std::memcpy(sections.data(), binary.data() + header.sectionTableOff,
                sections.size() * sizeof(SectionHeader));
    return sections;
}

inline SectionHeader sectionHeader(
        const std::vector<uint8_t> &binary, std::size_t index) {
    auto header = fileHeader(binary);
    if (index >= header.sectionCount)
        throw std::runtime_error("Invalid zebin section index");
    SectionHeader section;
    std::memcpy(&section,
            binary.data() + header.sectionTableOff
                    + index * sizeof(SectionHeader),
            sizeof(section));
    return section;
}

inline void storeSectionHeaders(std::vector<uint8_t> &binary,
        const std::vector<SectionHeader> &sections) {
    auto header = fileHeader(binary);
    if (sections.size() != header.sectionCount)
        throw std::runtime_error("Invalid zebin section table");
    if (!sections.empty())
        std::memcpy(binary.data() + header.sectionTableOff, sections.data(),
                sections.size() * sizeof(SectionHeader));
}

inline bool sectionInBounds(
        const std::vector<uint8_t> &binary, const SectionHeader &section) {
    return section.offset <= binary.size()
            && section.size <= binary.size() - section.offset;
}

inline const char *elfString(const std::vector<uint8_t> &binary,
        std::size_t table, uint64_t offset) {
    auto strings = sectionHeader(binary, table);
    if (!sectionInBounds(binary, strings) || offset >= strings.size
            || !std::memchr(binary.data() + strings.offset + offset, 0,
                    strings.size - offset))
        throw std::runtime_error("Invalid zebin string table entry");
    return reinterpret_cast<const char *>(
            binary.data() + strings.offset + offset);
}

inline const char *sectionName(
        const std::vector<uint8_t> &binary, std::size_t index) {
    return elfString(binary, fileHeader(binary).strTableIndex,
            sectionHeader(binary, index).name);
}

template <typename T>
std::vector<T> sectionEntries(
        const std::vector<uint8_t> &binary, std::size_t index) {
    auto section = sectionHeader(binary, index);
    if (section.entrySize != sizeof(T) || section.size % sizeof(T) != 0
            || !sectionInBounds(binary, section))
        throw std::runtime_error("Invalid zebin section entries");
    std::vector<T> entries(section.size / sizeof(T));
    if (!entries.empty())
        std::memcpy(entries.data(), binary.data() + section.offset,
                section.size);
    return entries;
}

template <typename T>
void storeSectionEntries(std::vector<uint8_t> &binary, std::size_t index,
        const std::vector<T> &entries) {
    auto section = sectionHeader(binary, index);
    if (section.size != entries.size() * sizeof(T)
            || !sectionInBounds(binary, section))
        throw std::runtime_error("Invalid zebin section entries");
    if (!entries.empty())
        std::memcpy(binary.data() + section.offset, entries.data(),
                section.size);
}

}
GEMMSTONE_NAMESPACE_END

#endif
