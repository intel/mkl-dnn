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

#ifndef GEMMSTONE_TESTS_TEST_COMMON_HPP
#define GEMMSTONE_TESTS_TEST_COMMON_HPP

#include <cstdio>
#include <exception>
#include <string>

inline const char *relative(const char *full_name) {
    return full_name + std::string(full_name).rfind("tests/");
}

#define EXPECT(cond) \
    do { \
        try { \
            if (cond) break; \
        } catch (const std::exception &e) { \
            printf("EXCEPTION in " #cond ": %s\t(%s:%d)\n", e.what(), \
                    relative(__FILE__), __LINE__); \
            tests_pass = false; \
            break; \
        } catch (...) { \
            printf("UNKNOWN EXCEPTION in " #cond "\t(%s:%d)\n", \
                    relative(__FILE__), __LINE__); \
            tests_pass = false; \
            break; \
        } \
        tests_pass = false; \
        printf("FAILED: " #cond "\t(%s:%d)\n", \
                relative(__FILE__), __LINE__); \
    } while (0)

#endif
