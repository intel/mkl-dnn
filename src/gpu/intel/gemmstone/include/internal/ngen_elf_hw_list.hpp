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

// Shared list of nGEN base-class instantiations: explicitly instantiated by
// ngen_elf.cpp, declared extern template elsewhere (default).

#ifndef GEMMSTONE_INCLUDE_INTERNAL_NGEN_ELF_HW_LIST_HPP
#define GEMMSTONE_INCLUDE_INTERNAL_NGEN_ELF_HW_LIST_HPP

#include "ngen_elf.hpp"

#ifndef GEMMSTONE_NGEN_ELF_KEYWORD
#define GEMMSTONE_NGEN_ELF_KEYWORD extern template class
#endif

#define GEMMSTONE_NGEN_ELF_INSTANTIATE(HW_ENUM) \
    GEMMSTONE_NGEN_ELF_KEYWORD ngen::BinaryCodeGenerator<ngen::HW::HW_ENUM>; \
    GEMMSTONE_NGEN_ELF_KEYWORD ngen::ELFCodeGenerator<ngen::HW::HW_ENUM>;

#ifdef INSTANTIATE_HW_12LP
GEMMSTONE_NGEN_ELF_INSTANTIATE(Gen12LP)
#endif
#ifdef INSTANTIATE_HW_12HP
GEMMSTONE_NGEN_ELF_INSTANTIATE(XeHP)
#endif
#ifdef INSTANTIATE_HW_12p7
GEMMSTONE_NGEN_ELF_INSTANTIATE(XeHPG)
#endif
#ifdef INSTANTIATE_HW_12p8
GEMMSTONE_NGEN_ELF_INSTANTIATE(XeHPC)
#endif
#ifdef INSTANTIATE_HW_Xe2
GEMMSTONE_NGEN_ELF_INSTANTIATE(Xe2)
#endif
#ifdef INSTANTIATE_HW_Xe3
GEMMSTONE_NGEN_ELF_INSTANTIATE(Xe3)
#endif
#ifdef INSTANTIATE_HW_Xe3p
GEMMSTONE_NGEN_ELF_INSTANTIATE(Xe3p)
#endif

#undef GEMMSTONE_NGEN_ELF_INSTANTIATE
#undef GEMMSTONE_NGEN_ELF_KEYWORD

#endif
