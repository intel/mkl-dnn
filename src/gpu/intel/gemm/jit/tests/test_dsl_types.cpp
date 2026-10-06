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

#include "test_common.hpp"

#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/dsl/type.hpp"

using namespace gemmstone::dsl;

bool test_cast_type() {
    bool tests_pass = true;
    EXPECT(cast(u8, 0).type() == u8);
    EXPECT(cast(s8, 0).type() == s8);
    EXPECT(cast(u16, 0).type() == u16);
    EXPECT(cast(s16, 0).type() == s16);
    EXPECT(cast(u32, 0).type() == u32);
    EXPECT(cast(s32, 0).type() == s32);
    EXPECT(cast(u64, 0).type() == u64);
    EXPECT(cast(s64, 0).type() == s64);
    return tests_pass;
}

int main() {
    bool tests_pass = test_cast_type();
    return tests_pass ? 0 : -1;
}
