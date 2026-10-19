/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_EXIT_CODES_HPP
#define GEMMSTONE_TOOLS_EXIT_CODES_HPP

// Exit codes shared between tools.
enum {
    EXIT_CODE_STUB = 64,
    EXIT_CODE_OUT_OF_REGISTERS = 65,
    EXIT_CODE_HW_UNSUPPORTED = 66,
    EXIT_CODE_MISC_ERROR = 67,
    EXIT_CODE_INACCURATE = 68,
    EXIT_CODE_OVERRUN = 69,
};

#endif /* header guard */
