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

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "generator/microkernel/payload.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

static std::string unquote(const std::string &s) {
    if (s.size() >= 2 && (s.front() == '\'' || s.front() == '"')
            && s.back() == s.front())
        return s.substr(1, s.size() - 2);
    return s;
}

bool parseZeInfo(
        const char *text, size_t length, std::vector<KernelInfo> &kernels) {
    kernels.clear();

    bool inKernels = false;
    enum { None, ExecEnv, Payload, PerThread } section = None;
    PayloadArgument perThread;
    auto *end = text + length;

    for (auto *l = text; l < end;) {
        auto *eol = static_cast<const char *>(
                std::memchr(l, '\n', size_t(end - l)));
        if (!eol) eol = end;
        int indent = 0;
        while (l + indent < eol && l[indent] == ' ')
            indent++;
        std::string line(l + indent, size_t(eol - l - indent));
        l = (eol < end) ? eol + 1 : end;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        if (indent == 0) {
            inKernels = (line == "kernels:");
            section = None;
            continue;
        }
        if (!inKernels) continue;

        bool item = (line.compare(0, 2, "- ") == 0);
        if (item) line = line.substr(2);

        /* A list item at kernel level starts a kernel; its first key is inline. */
        if (indent == 2 && item) {
            kernels.emplace_back();
            section = None;
            indent = 4;
            item = false;
        }
        if (kernels.empty()) return false;
        auto &kernel = kernels.back();

        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon), val;
        auto v = line.find_first_not_of(' ', colon + 1);
        if (v != std::string::npos) val = unquote(line.substr(v));

        auto number = [&] {
            return uint32_t(std::strtoul(val.c_str(), nullptr, 10));
        };

        if (indent == 4 && !item) {
            if (key == "name") {
                kernel.name = val;
                continue;
            }
            if (val.empty()) {
                section = (key == "execution_env")                ? ExecEnv
                        : (key == "payload_arguments")            ? Payload
                        : (key == "per_thread_payload_arguments") ? PerThread
                                                                  : None;
                perThread = {};
                continue;
            }
        }
        if (section == None) continue;

        if (section == ExecEnv) {
            if (key == "inline_data_payload_size")
                kernel.inlineDataBytes = number();
            continue;
        }

        if (section == PerThread) {
            /* Only the extent matters; local IDs need no padding fixup. */
            if (item) perThread = {};
            if (key == "offset") perThread.offset = number();
            else if (key == "size") perThread.size = number();
            kernel.perThreadBytes = std::max(
                    kernel.perThreadBytes, perThread.offset + perThread.size);
            continue;
        }

        if (item) kernel.payload.emplace_back();
        if (kernel.payload.empty()) { kernels.clear(); return false; }
        auto &arg = kernel.payload.back();
        if (key == "offset") arg.offset = number();
        else if (key == "size") arg.size = number();
    }

    return !kernels.empty();
}

static uint32_t alignUp(uint32_t x, uint32_t align) {
    return (x + align - 1) & ~(align - 1);
}

uint32_t crossthreadBase(const KernelInfo &kernel, int grfBytes) {
    auto grf = uint32_t(grfBytes);
    return grf + alignUp(kernel.perThreadBytes, grf);
}

uint32_t payloadEndBytes(const KernelInfo &kernel, int grfBytes) {
    // Inline data occupies a whole GRF regardless of inline_data_payload_size,
    // so offsets at or past it shift by the padding.
    auto inlineData = kernel.inlineDataBytes;
    uint32_t pad = inlineData
            ? alignUp(inlineData, uint32_t(grfBytes)) - inlineData
            : 0;

    auto base = crossthreadBase(kernel, grfBytes);
    auto end = base;
    for (auto &arg : kernel.payload) {
        auto offset = base + arg.offset + ((arg.offset >= inlineData) ? pad : 0);
        end = std::max(end, offset + arg.size);
    }
    return end;
}

}
GEMMSTONE_NAMESPACE_END
