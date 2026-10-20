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

#include "gemmstone/microkernel/shim.hpp"
#include "gemmstone/microkernel/fuser.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

GEMMSTONE_NAMESPACE_START
namespace microkernel {

template <typename T, typename U>
static inline T divideUp(T num, U denom) {
    return (num + denom - 1) / denom;
}

int grfWidth(uint32_t gmdid) {
    union {
        uint32_t raw;
        struct {
            uint32_t revision : 6;
            uint32_t reserved : 8;
            uint32_t release : 8;
            uint32_t architecture : 10;
        };
    } decode;

    decode.raw = gmdid;
    if (decode.architecture == 12 && decode.release >= 60
            && decode.release < 70) /* XeHPC */
        return 64;
    if (decode.architecture >= 20) /* Xe2/Xe3 */
        return 64;

    return 32;
}

bool isSIMT(HostLanguage language) {
    switch (language) {
        case HostLanguage::OpenCL_C:
        case HostLanguage::SYCL: return true;
        default: return false;
    }
}

const char *globalAddrSpaceDecorator(HostLanguage language) {
    if (language == HostLanguage::OpenCL_C) return "global ";
    return "";
}

const char *localAddrSpaceDecorator(HostLanguage language) {
    if (language == HostLanguage::OpenCL_C) return "local ";
    return "";
}

int typeSize(StructuredType::Type type) {
    switch (type) {
        case StructuredType::s64:
        case StructuredType::u64:
        case StructuredType::f64: return 8;
        case StructuredType::s32:
        case StructuredType::u32:
        case StructuredType::f32: return 4;
        case StructuredType::s16:
        case StructuredType::u16:
        case StructuredType::f16:
        case StructuredType::bf16: return 2;
        case StructuredType::s8:
        case StructuredType::u8:
        case StructuredType::bf8:
        case StructuredType::hf8:
        case StructuredType::f8_e8m0:
        case StructuredType::f4_e2m1: return 1;
        default: throw std::runtime_error("Unknown type");
    }
}

int typeSize(StructuredType stype) {
    switch (stype.format) {
        case StructuredType::GlobalPointer: return 8;
        case StructuredType::LocalPointer: return 4;
        case StructuredType::Scalar: return typeSize(stype.type);
        default: throw std::runtime_error("Unexpected format");
    }
}

const char *typeName(
        StructuredType::Type type, HostLanguage language = HostLanguage::None) {
    if (language == HostLanguage::vISA) switch (type) {
            case StructuredType::s64: return "q";
            case StructuredType::s32: return "d";
            case StructuredType::s16: return "w";
            case StructuredType::s8: return "b";
            case StructuredType::u64: return "uq";
            case StructuredType::u32: return "ud";
            case StructuredType::u16: return "uw";
            case StructuredType::u8: return "ub";
            case StructuredType::f64: return "df";
            case StructuredType::f32: return "f";
            case StructuredType::f16: return "hf";
            case StructuredType::bf16: return "bf";
            case StructuredType::bf8: return "bf8";
            case StructuredType::hf8: return "hf8";
            case StructuredType::f8_e8m0: return "e8m0";
            case StructuredType::f4_e2m1: return "e2m1";
            default: throw std::runtime_error("Unknown type");
        }
    else
        switch (type) {
            case StructuredType::s64: return "long";
            case StructuredType::s32: return "int";
            case StructuredType::s16: return "short";
            case StructuredType::s8: return "char";
            case StructuredType::u64: return "ulong";
            case StructuredType::u32: return "uint";
            case StructuredType::u16: return "ushort";
            case StructuredType::u8: return "uchar";
            case StructuredType::s4: return "s4";
            case StructuredType::u4: return "u4";
            case StructuredType::u3: return "u3";
            case StructuredType::f64: return "double";
            case StructuredType::f32: return "float";
            case StructuredType::f16: return "half";
            case StructuredType::bf16:
                return (language == HostLanguage::None) ? "bfloat16" : "bf16";
            case StructuredType::bf8:
                return (language == HostLanguage::None) ? "bfloat8" : "f8_e5m2";
            case StructuredType::hf8:
                return (language == HostLanguage::None) ? "hfloat8" : "f8_e4m3";
            case StructuredType::f8_e8m0: return "e8m0";
            case StructuredType::f4_e2m1:
                return (language == HostLanguage::None) ? "e2m1" : "f4_e2m1";
            default: return "char";
        }
}

std::string typeName(StructuredType stype, HostLanguage language,
        const TensorConfig *sizes = nullptr) {
    switch (stype.format) {
        case StructuredType::Scalar: return typeName(stype.type, language);
        case StructuredType::GlobalPointer:
            return std::string(globalAddrSpaceDecorator(language))
                    + typeName(stype.type, language) + '*';
        case StructuredType::LocalPointer:
            return std::string(localAddrSpaceDecorator(language))
                    + typeName(stype.type, language) + '*';
        case StructuredType::Tensor: {
            auto name = std::string(typeName(stype.type)) + "_tile_";
            assert(sizes && "Sizes not provided");

            for (int i = 0; i < stype.ndims; i++) {
                if (i > 0) name += 'x';
                name += std::to_string(sizes->dims[i]);
            }

            if (sizes->blocked()) {
                name += "_blocked_";
                for (int i = 0; i < stype.ndims; i++) {
                    if (i > 0) name += 'x';
                    name += std::to_string(sizes->block[i]);
                }
            }

            return name;
        }
        default: throw std::runtime_error("Unknown format");
    }
}

template <typename ProtocolT, typename ActualT>
std::vector<const ActualT *> matchProtocol(
        const std::vector<ProtocolT> &plist, const std::vector<ActualT> &list) {
    int n = int(plist.size());

    std::vector<const ActualT *> result(n);

    for (int i = 0; i < n; i++) {
        for (auto &item : list) {
            if (item.name == plist[i].name) {
                if (result[i])
                    throw std::runtime_error(
                            "Microkernel has a duplicate argument/setting: "
                            + item.name);
                result[i] = &item;
            }
        }
        if (!result[i])
            throw std::runtime_error(
                    std::string("Microkernel missing a required "
                                "argument/setting for its protocol: ")
                    + plist[i].name);
    }

    return result;
}

std::string generateShim(const Package &package, HostLanguage language,
        const ShimOptions &options) {
    std::stringstream shim;
    shim.imbue(std::locale::classic());

    bool cpp = (language == HostLanguage::SYCL);

    /* Match up protocol args with microkernel args */
    auto pargs = package.protocol.arguments();
    auto args = matchProtocol(pargs, package.arguments);
    auto nargs = int(pargs.size());

    /* Match up protocol settings with microkernel settings */
    const auto &psettings = package.protocol.settings();
    auto settings = matchProtocol(psettings, package.settings);

    /* Collect actual argument types */
    std::vector<StructuredType> stypes(pargs.size());
    for (size_t i = 0; i < pargs.size(); i++) {
        stypes[i] = pargs[i].stype;
        if (stypes[i].type == StructuredType::any)
            stypes[i].type = args[i]->actualType;
        else if (args[i]->actualType != StructuredType::any
                && stypes[i].type != args[i]->actualType)
            throw std::runtime_error(
                    "Microkernel argument type does not match its protocol");
    }

    /* Get decorated kernel name */
    std::string kname = package.protocol.kernelBaseName();
    if (!options.decorator.empty()) {
        kname += '_';
        kname += options.decorator;
    }

    /* Helper to construct "nice" argument type names for tensor types (e.g. gemm_c_type) */
    auto argTypeName = [&](int i) {
        if (stypes[i].format == StructuredType::Tensor)
            return kname + '_' + pargs[i].name + "_type";
        else
            return typeName(stypes[i], language, &args[i]->sizes);
    };

    if (options.subgroupSize == 0 && isSIMT(language))
        throw std::runtime_error("Subgroup size must be specified.");

    /* OpenCL C: Round up tensor structures and generate type definitions for them */
    if (language == HostLanguage::OpenCL_C) {
        /* to do: de-duplicate identical tensor structures */
        for (int i = 0; i < nargs; i++) {
            auto &sizes = args[i]->sizes;

            if (stypes[i].format != StructuredType::Tensor) continue;

            auto sname = typeName(stypes[i], language, &sizes);
            auto ename = typeName(stypes[i].type, language);
            auto typedef_name = "_e_" + sname;
            int vlen = divideUp(sizes.blockElements(), options.subgroupSize);

            shim << "#ifndef MICRO_DECL_" << sname
                 << "\n"
                    "#define MICRO_DECL_"
                 << sname
                 << "\n"
                    "typedef " << ename <<
                    " __attribute__((ext_vector_type(" << vlen << "))) "
                 << typedef_name << ";\n"
                    "typedef struct {\n"
                    "    "
                 << typedef_name << ' ';
            shim << " x[" << sizes.blocks()
                 << "];\n"
                    "} "
                 << sname << ";\n";

            if (options.useTileOps) {
                int ndims = stypes[i].ndims;
                shim << "DECLARE_" << ndims << "D_TILE_OPS(";
                shim << sname << ',' << ename << ',' << options.subgroupSize;
                for (int d = 0; d < ndims; d++)
                    shim << ',' << sizes.block[d];
                for (int d = 0; d < ndims; d++)
                    shim << ',' << sizes.dims[d] / sizes.block[d];
                shim << ")\n";
            }

            shim << "#endif\n";
        }
    }

    std::string returnType;
    int returnArg = -1;

    if (language != HostLanguage::None) {
        /* Create a definition for each setting */
        auto sintro = cpp ? "static constexpr int " : "#define ";
        auto ssep = cpp ? " = " : " ";

        for (auto &setting : settings)
            shim << sintro << kname << '_' << setting->name << ssep
                 << setting->value << '\n';

        /* Create definitions for some additional package flags */
        shim << sintro << kname << "_barrier_count " << ssep
             << package.barrierCount << '\n';
        shim << sintro << kname << "_systolic " << ssep << int(package.systolic)
             << '\n';

        /* Generate typedefs for the tensor argument types, and #defines for blocking sizes */
        for (int i = 0; i < nargs; i++) {
            if (stypes[i].format != StructuredType::Tensor) continue;

            auto &sizes = args[i]->sizes;
            auto tname = typeName(stypes[i], language, &sizes);
            auto aname = argTypeName(i);
            shim << "typedef " << tname << ' ' << aname << ";\n";
            for (int d = 0; d < stypes[i].ndims; d++) {
                shim << sintro << aname << "_block" << d << ssep
                     << sizes.block[d] << '\n';
                shim << sintro << aname << "_nblock" << d << ssep
                     << (sizes.dims[d] / sizes.block[d]) << '\n';
            }
        }

        /* Locate return type. Return types are used in the case of a single output. */
        for (int i = 0; i < nargs; i++) {
            if (pargs[i].direction == Protocol::Argument::Out) {
                if (!returnType.empty()) {
                    returnArg = -1;
                    returnType = "";
                    break;
                }
                returnArg = i;
                returnType = argTypeName(i);
            }
        }

        if (returnType.empty()) returnType = "void";

        /* Synthesize wrapper function declaration */
        bool firstArg = true;
        shim << returnType << ' ' << kname << '(';
        for (int i = 0; i < nargs; i++) {
            if (i == returnArg) continue;
            if (!firstArg) shim << ", ";
            if (!pargs[i].out()
                    && (pargs[i].stype.format == StructuredType::GlobalPointer
                            || pargs[i].stype.format
                                    == StructuredType::LocalPointer)) {
                shim << "const ";
            }
            shim << argTypeName(i) << ' ';
            if (pargs[i].out()) shim << (cpp ? '&' : '*');
            shim << pargs[i].name;
            firstArg = false;
        }
        shim << ") {\n";

        if (returnArg >= 0)
            shim << "    " << returnType << ' ' << pargs[returnArg].name
                 << ";\n";
    }

    /* Gather underlying vISA shim args, one for each vISA variable.                     */
    /* There will be one vISA shim arg for each scalar/pointer argument or tensor block. */
    struct v_shim_argument_t {
        bool in, out, uniform, copy;
        RegisterRange location;
        std::string name;
        StructuredType::Type type;
    };

    std::vector<v_shim_argument_t> vargs, vargsIn;

    for (int i = 0; i < nargs; i++) {
        v_shim_argument_t varg;

        varg.in = pargs[i].in();
        varg.out = pargs[i].out();

        varg.type = stypes[i].type;
        if (stypes[i].format == StructuredType::GlobalPointer)
            varg.type = StructuredType::u64;
        else if (stypes[i].format == StructuredType::LocalPointer)
            varg.type = StructuredType::u32;

        auto &vargList = varg.out ? vargs : vargsIn;

        bool byPtr = varg.out && !cpp && (i != returnArg);

        if (stypes[i].format == StructuredType::Tensor) {
            int rangeIdx = 0, rangeOffset = 0;
            int blockBytes
                    = args[i]->sizes.blockElements() * typeSize(stypes[i].type);

            varg.uniform = false;
            varg.copy = options.copyTensorArgs;
            varg.location.blen = blockBytes;

            /* Create vISA variable for each block */
            for (int iblock = 0; iblock < args[i]->sizes.blocks(); iblock++) {
                auto &range = args[i]->location[rangeIdx];
                uint32_t noffset = rangeOffset + blockBytes;
                if (range.blen < noffset)
                    throw std::runtime_error(
                            "Tensor block not contiguous in registers");

                varg.location.boffset = range.boffset + rangeOffset;
                varg.name = pargs[i].name;
                varg.name += byPtr ? "->x[" : ".x[";
                varg.name += std::to_string(iblock);
                varg.name += ']';

                vargList.push_back(varg);

                if (range.blen > noffset)
                    rangeOffset = noffset;
                else {
                    rangeOffset = 0;
                    rangeIdx++;
                }

                bool bdone = (iblock + 1 >= args[i]->sizes.blocks());
                bool rdone = (rangeIdx >= int(args[i]->location.size()));
                if (rdone && !bdone)
                    throw std::runtime_error(
                            "Not enough registers allocated for declared "
                            "tensor size");
                else if (bdone && !rdone)
                    throw std::runtime_error(
                            "Too many registers allocated for declared tensor "
                            "size");
            }
        } else {
            /* Scalars are uniform, but IGC accepts only "=rw" (per-lane)
               output constraints. */
            if (pargs[i].out())
                throw std::runtime_error(
                        "Microkernel scalar output arguments are not "
                        "supported");
            if (args[i]->location.size() != 1)
                throw std::runtime_error(
                        "Microkernel scalar argument is not contiguous in "
                        "registers");
            if (int(args[i]->location[0].blen) != typeSize(pargs[i].stype))
                throw std::runtime_error(
                        "Microkernel argument does not have expected size");
            varg.location = args[i]->location[0];
            varg.uniform = true;
            varg.copy = options.copyScalarArgs;
            varg.name = pargs[i].name;
            if (byPtr) varg.name = '*' + varg.name;
            vargList.push_back(std::move(varg));
        }
    }

    /* Concatenate output and input arguments */
    vargs.insert(vargs.end(), vargsIn.begin(), vargsIn.end());

    /* Start vISA shim */
    shim << "    __asm__ volatile(\"{\\n\"\n";

    /* Tie arguments to physical registers */
    int gwidth = grfWidth(package.gmdidCompat);
    std::vector<std::string> copyNames(vargs.size());

    for (int i = 0; i < int(vargs.size()); i++) {
        auto &range = vargs[i].location;
        auto goffset = range.boffset % gwidth;

        /* Check that arg can be covered by a vISA variable */
        if (goffset != 0 && int(goffset + range.blen) > gwidth)
            throw std::runtime_error(
                    "Microkernel tensor argument misaligned in registers");

        if (vargs[i].copy) {
            copyNames[i] = "COPY" + std::to_string(i) + '_'
                    + std::to_string(range.boffset) + '_'
                    + std::to_string(range.blen);
            shim << "            \".decl " << copyNames[i] << " v_type=G type="
                 << typeName(vargs[i].type, HostLanguage::vISA)
                 << " num_elts=" << (range.blen / typeSize(vargs[i].type))
                 << "\\n\"\n";
        } else
            copyNames[i] = '%' + std::to_string(i);

        shim << "            \".implicit_PSEUDO_INPUT " << copyNames[i]
             << " offset=" << range.boffset << " size=" << range.blen
             << "\\n\"\n";
    }

    /* Check whether any inputs/outputs need copying */
    bool anyCopyIn = false, anyCopyOut = false;
    for (auto &varg : vargs) {
        if (varg.in) anyCopyIn |= varg.copy;
        if (varg.out) anyCopyOut |= varg.copy;
    }

    /* Protect input copies from preceding code */
    if (anyCopyIn) shim << "            \"fence_sw\\n\"\n";

    /* Copy inputs as needed */
    enum CopyArgType { Argument, Copy, Null };

    auto copyArgName = [&](CopyArgType type, int i) {
        switch (type) {
            case Argument: return '%' + std::to_string(i);
            case Copy: return copyNames[i];
            case Null: return std::string("V0");
            default: throw std::runtime_error("Invalid argument class");
        }
    };

    auto copyArg = [&](int i, CopyArgType from, CopyArgType to) {
        int remaining = vargs[i].location.blen;
        int tsize = typeSize(vargs[i].type);
        int offset = 0;
        while (offset < (int)vargs[i].location.blen) {
            int chunk = std::min(remaining, gwidth * 2);
            int esize = std::min(chunk / tsize, 32);
            chunk = esize * tsize;
            int r = offset / gwidth;
            int c = (offset % gwidth) / tsize;
            shim << "            \"mov (M1_NM, " << esize << ") "
                 << copyArgName(to, i) << '(' << r << ',' << c << ")<1> "
                 << copyArgName(from, i) << '(' << r << ',' << c
                 << ")<1;1,0>\\n\"\n";
            offset += chunk;
        }
    };

    for (int i = 0; i < int(vargs.size()); i++)
        if (vargs[i].copy && vargs[i].in) copyArg(i, Argument, Copy);

    /* Wrangle clobber regions. */
    struct clobber_t {
        RegisterRange location;
        std::string name;
        bool arg;
        bool preclobbered = false;
    };

    /* Temporarily, we need to subtract argument ranges from clobber ranges due to vISA restrictions.
       Sort arguments by location in preparation for that. */
    std::vector<clobber_t> vargClobbers;
    for (int i = 0; i < int(vargs.size()); i++) {
        clobber_t clobber;
        clobber.location = vargs[i].location;
        clobber.name = (vargs[i].copy ? "COPY" : "%") + std::to_string(i);
        clobber.arg = false; // Reuse 'arg' field as flag
        clobber.preclobbered = vargs[i].copy && vargs[i].in;
        vargClobbers.push_back(std::move(clobber));
    }

    std::sort(vargClobbers.begin(), vargClobbers.end(),
            [](const clobber_t &vc1, const clobber_t &vc2) {
        return (vc1.location.boffset < vc2.location.boffset);
    });

    /* Expand clobber ranges to legal vISA variables */
    std::vector<clobber_t> clobbers;

    if (package.clobbers.empty())
        throw std::runtime_error(
                "Microkernel does not define any clobber regions");

    auto vargIter = vargClobbers.begin();
    for (auto &range : package.clobbers) {
        uint32_t offset = range.boffset;
        int remaining = range.blen;

        while (remaining > 0) {
            /* Subtract argument ranges */
            while (vargIter != vargClobbers.end()
                    && vargIter->location.boffset + vargIter->location.blen
                            <= offset)
                ++vargIter;

            int nextOffset = 0, nextRemaining = 0;
            if (vargIter != vargClobbers.end()
                    && vargIter->location.boffset < offset + remaining) {
                nextOffset
                        = vargIter->location.boffset + vargIter->location.blen;
                nextRemaining = offset + remaining - nextOffset;
                remaining = std::min<int>(
                        remaining, vargIter->location.boffset - offset);
                vargIter->arg = true;
            }

            while (remaining > 0) {
                /* DWord align for convenience */
                remaining += offset & 3;
                offset &= ~3;
                remaining = (remaining + 3) & ~3;

                /* Carve off an aligned power-of-2 or GRF-aligned chunk */
                int chunk = remaining;
                if (offset % gwidth) {
                    chunk = std::min(remaining, gwidth >> 1);
                    while (offset & (std::min(chunk, gwidth) - 1))
                        chunk >>= 1;
                }

                clobber_t clobber;
                clobber.location = RegisterRange(offset, chunk);
                clobber.name = "CLOBBER" + std::to_string(clobbers.size()) + '_'
                        + std::to_string(offset) + '_' + std::to_string(chunk);
                clobber.arg = false;
                clobbers.push_back(std::move(clobber));

                remaining -= chunk;
                offset += chunk;
            }

            if (nextRemaining > 0)
                offset = nextOffset, remaining = nextRemaining;
        }
    }

    /* Add clobbered arguments to list */
    for (auto &vargClobber : vargClobbers)
        if (vargClobber.arg) clobbers.push_back(std::move(vargClobber));

    /* Declare clobbers and tie them to physical registers */
    for (int i = 0; i < int(clobbers.size()); i++) {
        if (clobbers[i].arg) continue;
        auto &cname = clobbers[i].name;
        auto &range = clobbers[i].location;

        shim << "            \".decl " << cname
             << " v_type=G type=ud num_elts=" << (range.blen >> 2) << "\\n\"\n";
        shim << "            \".implicit_PSEUDO_INPUT " << cname
             << " offset=" << range.boffset << " size=" << range.blen
             << "\\n\"\n";
    }

    /* Mark beginning of patch region */
    const auto &clobber0Name = clobbers[0].name;
    shim << "            \"fence_sw\\n\"\n"
            "            \"add (M1,1) "
         << clobber0Name << "(0,0)<1> " << clobber0Name << "(0,0)<0;1,0> 0x"
         << std::hex << (sigilStart ^ options.microkernelID) << std::dec
         << ":ud\\n\"\n"
            "            \"fence_sw\\n\"\n";

    /* Use inputs to ensure vISA considers their values live */
    for (int i = 0; i < int(vargs.size()); i++)
        if (vargs[i].in) copyArg(i, vargs[i].copy ? Copy : Argument, Null);

    /* Overwrite clobbers to ensure vISA considers their ranges live */
    for (int i = 0; i < int(clobbers.size()); i++) {
        if (clobbers[i].preclobbered) continue;
        auto &cname = clobbers[i].name;
        auto &range = clobbers[i].location;
        uint32_t offset = 0;
        while (offset < range.blen) {
            int chunk = std::min<int>(range.blen - offset, gwidth * 2);
            if (chunk & (chunk - 1)) {
                int chunk2;
                for (chunk2 = 2; chunk2 <= chunk; chunk2 <<= 1)
                    ;
                chunk = chunk2 >> 1;
            }

            int r = offset / gwidth;
            int c = offset % gwidth;
            shim << "            \"mov (M1," << (chunk >> 2) << ") " << cname
                 << '(' << r << ',' << (c >> 2) << ")<1> 0xAAAAAAAA:ud\\n\"\n";
            offset += chunk;
        }
    }

    /* Add dummy instructions to enable kernel features as needed */
    if (package.systolic) {
        // Find 8 consecutive clobber registers for dummy DPAS.
        uint32_t dlen = gwidth * 8;
        int idst = -1;
        for (int i = 0; i < int(clobbers.size()); i++) {
            if (clobbers[i].location.blen >= dlen) {
                idst = i;
                break;
            }
        }
        shim << "            \".decl DUMMY_DPAS_SRC v_type=G type=ud num_elts="
             << (dlen >> 2);
        if (idst >= 0) shim << " alias=<" << clobbers[idst].name << ",0>";
        shim << "\\n\"\n"
                "            \".decl DUMMY_DPAS_DST v_type=G type=f num_elts="
             << (dlen >> 2)
             << " alias=<DUMMY_DPAS_SRC,0>\\n\"\n"
                "            \"dpas.bf.bf.8.1 (M1,"
             << (gwidth >> 2)
             << ") DUMMY_DPAS_DST.0 V0.0 DUMMY_DPAS_SRC.0 "
                "DUMMY_DPAS_SRC(0,0)\\n\"\n";
    }

    if (package.barrierCount == 1) {
        shim << "            \"barrier\\n\"\n";
    } else if (package.barrierCount > 1) {
        // Named barriers -- TBD.
        // .kernel_attr NBarrierCnt=<...>
        throw std::runtime_error("Named barriers not yet implemented");
    }

    /* Mark end of patch region */
    shim << "            \"fence_sw\\n\"\n"
            "            \"add (M1,1) "
         << clobber0Name << "(0,0)<1> " << clobber0Name << "(0,0)<0;1,0> 0x"
         << std::hex << (sigilEnd ^ options.microkernelID) << std::dec
         << ":ud\\n\"\n"
            "            \"fence_sw\\n\"\n";

    /* Copy output arguments as needed */
    for (int i = 0; i < int(vargs.size()); i++)
        if (vargs[i].copy && vargs[i].out) copyArg(i, Copy, Argument);

    /* Protect output copies from preceding code */
    if (anyCopyOut) shim << "            \"fence_sw\\n\"\n";

    /* End of inline vISA string */
    shim << "        \"}\\n\"\n"
            "   ";

    /* Enumerate inline vISA arguments */
    for (bool doOutputs : {true, false}) {
        bool first = true;
        shim << " : ";
        for (auto &varg : vargs) {
            if (doOutputs != varg.out) continue;
            if (!first) shim << ", ";
            shim << '\"';
            if (varg.out) shim << (varg.in ? '+' : '=');
            shim << (varg.uniform ? "rw.u" : "rw");
            shim << "\"(" << varg.name << ')';
            first = false;
        }
    }

    shim << ");\n";

    /* Insert binary code in comment for fuser */
    const char hexChars[] = "0123456789abcdef";
    shim << "    // " << sigilBinary << options.microkernelID << ':'
         << package.argumentBase << ' ';
    for (auto b : package.binary)
        shim << hexChars[(b >> 4) & 0xF] << hexChars[b & 0xF];
    shim << '\n';

    /* End function declaration */
    if (language != HostLanguage::None) {
        if (returnArg >= 0)
            shim << "    return " << pargs[returnArg].name << ";\n";
        shim << "}\n";
    }

    return shim.str();
}

}
GEMMSTONE_NAMESPACE_END
