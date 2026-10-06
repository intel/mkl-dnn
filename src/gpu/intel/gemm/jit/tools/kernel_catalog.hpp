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

#ifndef GEMMSTONE_TOOLS_KERNEL_CATALOG_HPP
#define GEMMSTONE_TOOLS_KERNEL_CATALOG_HPP

#include "gemmstone/kernel_catalog.hpp"
#include "internal/ngen_includes.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace catalog {

inline constexpr char Sentinel[] = "/*@kcatalog@*/";

struct Catalog {
    std::vector<gemmstone::kcatalog::Entry> entries;
    std::string header;
};

const char *intern(const std::string &value);
void validate(const gemmstone::kcatalog::Entry &entry);
char tag(ngen::HW hw);
void addDefaultRestrictions(gemmstone::kcatalog::Entry &entry);
std::string modifierTags(const std::string &modifiers);

void serialize(std::ostream &s, const std::vector<gemmstone::kcatalog::Entry> &entries, const std::string &header = "");
std::vector<gemmstone::kcatalog::Entry> deserialize(std::istream &s, std::string &header);

} // namespace catalog

#endif
