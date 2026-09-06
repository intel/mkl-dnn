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

#ifndef UTILS_BENCH_MODE_HPP
#define UTILS_BENCH_MODE_HPP

#include <sstream>

// Mode bit is a base enum class to build `bench_mode_t` abstraction on top.
// Bits represent a basic responsibility of a complete flow.
// Under `test object` both `primitive` or `graph` objects are assumed.
enum class mode_bit_t : unsigned {
    undefined = 0x0,
    // `list` bit is for a `prb_t` object creation.
    list = 0x1,
    // `init` bit is for a test object creation/initialization.
    init = 0x2,
    // `exec` bit is for a test object execution.
    exec = 0x4,
    // `corr` bit is for a test object correctness validation.
    corr = 0x8,
    // `perf` bit is for a test object performance validation.
    perf = 0x10,
    // `fast` bit is for a modified performance validation flow.
    fast = 0x20,
    // `bitwise` bit is for a numerical determinism validation flow.
    bitwise = 0x40,
    // `sim` bit is for a single-run simulation-friendly performance flow.
    sim = 0x80,
};

// Mode modifiers is an extension of `bench_mode_t` abstraction which specifies
// how exactly the selected flow will be changed based on user's choice.
enum class mode_modifier_t : unsigned {
    // No modification to the final flow.
    none = 0x0,
    // Enable parallel test object creation. Uses as many threads as identified
    // by `dnnl_get_max_threads()`.
    par_create = 0x1,
    // Disable usage of reference memories in the flow. It removes mapping,
    // unmapping and filling functionality.
    no_ref_memory = 0x2,
    // Accelerate the native correctness reference by filling inputs so the
    // result is periodic along M and N, computing a single representative
    // panel and broadcasting it. Currently honored by the matmul driver only;
    // other drivers ignore it. Requires the native reference (`--fast-ref=false`).
    ref_periodic_fill = 0x4,
};

mode_modifier_t operator|(mode_modifier_t lhs, mode_modifier_t rhs);
mode_modifier_t &operator|=(mode_modifier_t &lhs, mode_modifier_t rhs);
mode_modifier_t operator&(mode_modifier_t lhs, mode_modifier_t rhs);

// Mode is built of mode bits to represent an action requested by the user.
// Note: `mode_t` has collision with type from `/usr/include/sys/types.h`
enum class bench_mode_t : unsigned {
    undefined = 0x0,
    list = static_cast<unsigned>(mode_bit_t::list),
    init = list | static_cast<unsigned>(mode_bit_t::init),
    exec = init | static_cast<unsigned>(mode_bit_t::exec),
    corr = exec | static_cast<unsigned>(mode_bit_t::corr),
    perf = exec | static_cast<unsigned>(mode_bit_t::perf),
    perf_fast = perf | static_cast<unsigned>(mode_bit_t::fast),
    perf_sim = perf | static_cast<unsigned>(mode_bit_t::sim),
    bitwise = exec | static_cast<unsigned>(mode_bit_t::bitwise),
};

mode_bit_t operator&(bench_mode_t lhs, mode_bit_t rhs);

extern bench_mode_t bench_mode; // user mode
extern bench_mode_t default_bench_mode; // correctness default mode
extern mode_modifier_t bench_mode_modifier; // user mode modifier
extern mode_modifier_t default_bench_mode_modifier; // empty default modifier

// In certain scenarios the logic should check for a specific bit. But sometimes
// it's required to check for exact mode, for this purposes use operator== and
// other similar operators.
bool has_bench_mode_bit(mode_bit_t mode_bit);
bool has_bench_mode_modifier(mode_modifier_t modifier);

std::ostream &operator<<(std::ostream &s, bench_mode_t mode);
std::ostream &operator<<(std::ostream &s, mode_modifier_t modifier);

#endif
