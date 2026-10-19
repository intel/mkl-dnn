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

#include "derby_utils.hpp"

#include <algorithm>

#ifndef _WIN32
#include <fcntl.h>
#endif

using namespace ngen;

static Product currentProduct() {
    static Product product = detectDevice(false);
    return product;
}

HW currentHW() {
    static HW hw = getCore(currentProduct().family);
    return hw;
}

int eusPerSS()
{
    switch (currentHW()) {
        case HW::XeHPC:
        case HW::Xe2:
        case HW::Xe3:
        case HW::Xe3p:
                        return 8;

        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG: return 16;
        default: return 0;
    }
}

int euCount()
{
    static int count = 0;
    static bool checked = false;

    if (!checked) {
        auto env = getenv("euCount");
        if (env) {
            count = atoi(env);
            checked = true;
        }
    }

    if (!checked) {
        count = getDeviceEUCount();
        if (currentHW() == HW::XeHPC && getenv("PVC_EU_2X"))
            count >>= 1;
        checked = true;
    }

    return count;
}

int ssCount()
{
    return euCount() / eusPerSS();
}

bool gen12()
{
    return currentHW() >= HW::Gen12LP;
}

bool isColMajor(char c)
{
    return (c == 'A') || (c == 'N');
}

bool isSubByte(char c)
{
    return (std::toupper(c) == 'F');
}

bool isComplex(char c)
{
    return (c == 'J' || c == 'K' || c == 'U' || c == 'C' || c == 'Z');
}

bool isInteger(char c)
{
    c = std::toupper(c);
    return (c == 'F' || c == 'O' || c == 'W' || c == 'I');
}

bool isBlock2D(char c)
{
    c = std::toupper(c);
    return (c == 'M' || c == 'V' || c == 'T');
}

int grfBytes()
{
    return (currentHW() >= HW::XeHPC) ? 64 : 32;
}

int GRFPerEU()
{
    switch (currentHW()) {
        case HW::XeLP:   return 896; // Equivalent for 128 GRF/thread, 7 threads/EU.
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:
        case HW::Xe2:
        case HW::Xe3:
        case HW::Xe3p:     return (currentProduct().family == ProductFamily::CRI) ? 2048 : 1024;
        default: fail("Unknown architecture.");
    }
}

int threadsPerEU(int GRFPerThread)
{
    if (currentHW() <= ngen::HW::XeLP)
        return 7;

    if (currentProduct().family == ProductFamily::CRI && GRFPerThread == 256)
        return 4;
    int maxThreads = (currentProduct().family == ProductFamily::NVLP) ? 10 : 8;
    return std::min(maxThreads, GRFPerEU() / GRFPerThread);
}

int threadsPerSS(int grfs)
{
    return threadsPerEU(grfs) * eusPerSS();
}

int maxWG()
{
    switch (currentHW()) {
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:
        case HW::Xe2:
        case HW::Xe3:
        case HW::Xe3p:
                            return 64;
        default: fail("Unknown architecture.");
    }
}

int maxSLM()
{
    switch (currentHW()) {
        case HW::Gen12LP:
        case HW::XeHP:
        case HW::XeHPG:
        case HW::XeHPC:     return 131072;
        case HW::Xe2:       return 163840;
        case HW::Xe3:       return 196608;
        case HW::Xe3p:      return 393216;
        default: fail("Unknown architecture");
    }
}

[[noreturn]] void fail(const char *message)
{
    fputs(message, stderr);
    fputc('\n', stderr);
    exit(2);
}

[[noreturn]] void fail(const std::string &message)
{
    fail(message.c_str());
}

void testPipes()
{
#ifndef _WIN32
    std::array<int, 2> pipeFDs;
    const int checkSz = 1 << 22;    // 4MB

    if (pipe(&pipeFDs[0]) < 0)
        return;

    if (fcntl(pipeFDs[1], F_SETPIPE_SZ, checkSz) < checkSz) {
        fprintf(stderr, "Warning: pipe size limit is < 4MB. Kernels larger than this size will be skipped.\n"
                        "Use 'sudo sysctl fs.pipe-max-size=4194304' to increase the limit\n");
    }

    for (auto fd: pipeFDs) close(fd);
#endif
}
