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

#include "bench_mode.hpp"

bench_mode_t default_bench_mode {bench_mode_t::corr};
bench_mode_t bench_mode {default_bench_mode};
mode_modifier_t default_bench_mode_modifier {mode_modifier_t::none};
mode_modifier_t bench_mode_modifier {default_bench_mode_modifier};

bool has_bench_mode_bit(mode_bit_t mode_bit) {
    return static_cast<bool>(bench_mode & mode_bit);
}

bool has_bench_mode_modifier(mode_modifier_t modifier) {
    return static_cast<bool>(bench_mode_modifier & modifier);
}

std::ostream &operator<<(std::ostream &s, bench_mode_t mode) {
    if (mode == bench_mode_t::list) s << "L";
    if (mode == bench_mode_t::init) s << "I";
    if (mode == bench_mode_t::exec) s << "R";
    if (has_bench_mode_bit(mode_bit_t::corr)) s << "C";
    if (has_bench_mode_bit(mode_bit_t::perf)
            && !has_bench_mode_bit(mode_bit_t::fast)
            && !has_bench_mode_bit(mode_bit_t::sim))
        s << "P";
    if (has_bench_mode_bit(mode_bit_t::fast)) s << "F";
    if (has_bench_mode_bit(mode_bit_t::sim)) s << "S";
    if (has_bench_mode_bit(mode_bit_t::bitwise)) s << "B";
    return s;
}

std::ostream &operator<<(std::ostream &s, mode_modifier_t modifier) {
    if (modifier == mode_modifier_t::none) s << "";
    if (has_bench_mode_modifier(mode_modifier_t::par_create)) s << "P";
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) s << "M";
    if (has_bench_mode_modifier(mode_modifier_t::ref_periodic_fill)) s << "A";
    return s;
}

mode_bit_t operator&(bench_mode_t lhs, mode_bit_t rhs) {
    return static_cast<mode_bit_t>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

mode_modifier_t operator|(mode_modifier_t lhs, mode_modifier_t rhs) {
    return static_cast<mode_modifier_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

mode_modifier_t &operator|=(mode_modifier_t &lhs, mode_modifier_t rhs) {
    lhs = lhs | rhs;
    return lhs;
}

mode_modifier_t operator&(mode_modifier_t lhs, mode_modifier_t rhs) {
    return static_cast<mode_modifier_t>(
            static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}
