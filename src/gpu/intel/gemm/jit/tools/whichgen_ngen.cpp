/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <iostream>

#include "device_utils.hpp"

int main(int argc, char **argv)
{
    using namespace ngen;

    auto hw = getCore(detectDevice(true).family);

    int code = 0;
    const char *gen = "";

    switch (hw) {
        case HW::Gen12LP: gen = "12LP"; break;
        case HW::XeHP:    gen = "12HP"; break;
        case HW::XeHPG:   gen = "12p7"; break;
        case HW::XeHPC:   gen = "12p8"; break;
        case HW::Xe2:     gen = "Xe2";  break;
        case HW::Xe3:     gen = "Xe3";  break;
        case HW::Xe3p:    gen = "Xe3p"; break;
        default:                        break;
    }

    if (argc > 1 && !strcmp(argv[1], "-c"))
        exit(static_cast<int>(hw));
    else
        std::cout << gen << std::endl;
}
