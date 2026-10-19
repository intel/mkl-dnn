/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef INTERNAL_GENERATOR_INLINE_HXX
#define INTERNAL_GENERATOR_INLINE_HXX

static inline int r0DWords(ngen::HW hw)
{
    if (hw >= ngen::HW::Xe3p) return 16;
    return 8;
}

// Call a functor needing the r0 header in a GRF.
template <ngen::HW hw>
template <typename F>
void Generator<hw>::useR0(CommonState &state, F f)
{
    if (state.r0_info.isARF()) {
        auto r0_info = state.ra.alloc();
        mov<uint32_t>(r0DWords(hw), r0_info, state.r0_info);
        f(r0_info);
        state.ra.safeRelease(r0_info);
    } else
        f(ngen::GRF{state.r0_info.getBase()});
}

// Call a functor needing a GRF temporary and the r0 header in a GRF.
template <ngen::HW hw>
template <typename F>
void Generator<hw>::useTempAndR0(CommonState &state, F f)
{
    auto temp = state.ra.alloc();
    if (state.r0_info.isARF()) {
        auto r0_info = state.ra.alloc();
        mov<uint32_t>(r0DWords(hw), r0_info, state.r0_info);
        f(temp, r0_info);
        state.ra.safeRelease(r0_info);
    } else
        f(temp, ngen::GRF{state.r0_info.getBase()});
    state.ra.safeRelease(temp);
}

#endif /* header guard */
