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


#ifndef GEMMSTONE_GENERATOR_PIECES_MAP_HPP
#define GEMMSTONE_GENERATOR_PIECES_MAP_HPP

#include "internal/ngen_includes.hpp"

#include "gemmstone/type.hpp"
#include "gemmstone/strategy.hpp"

#include "grf_multirange.hpp"
#include "hw_utils.hpp"
#include "register_layout.hpp"


GEMMSTONE_NAMESPACE_START

static inline bool canDualGRF(ngen::HW hw, ngen::DataType dt, const CommonStrategy &strategy);

// The map(...) family of template functions apply a functor
//   to a range of registers.

// Perform a binary register-wise operation.
template <typename F>
static inline void map(ngen::HW hw, ngen::DataType dt, const GRFMultirange &r1, const GRFMultirange &r2, const CommonStrategy &strategy, F f)
{
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr))
            nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt));
        rr += nr;
    }
}

// Perform a ternary register-wise operation.
template <typename F>
static inline void map(ngen::HW hw, ngen::DataType dt, const GRFMultirange &r1, const GRFMultirange &r2, const GRFMultirange &r3, const CommonStrategy &strategy, F f)
{
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr) || !r3.contiguous(rr, nr))
            nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt), r3[rr].retype(dt));
        rr += nr;
    }
}

// Perform a quaternary register-wise operation.
template <typename F>
static inline void map(ngen::HW hw, ngen::DataType dt, const GRFMultirange &r1, const GRFMultirange &r2, const GRFMultirange &r3, const GRFMultirange &r4, const CommonStrategy &strategy, F f)
{
    int ne = elementsPerGRF(hw, dt);
    int rstride = canDualGRF(hw, dt, strategy) ? 2 : 1;
    int len = r1.getLen();

    for (int rr = 0; rr < len;) {
        int nr = std::min<int>(len - rr, rstride);
        if (!r1.contiguous(rr, nr) || !r2.contiguous(rr, nr) || !r3.contiguous(rr, nr) || !r4.contiguous(rr, nr))
            nr = 1;
        f(nr * ne, r1[rr].retype(dt), r2[rr].retype(dt), r3[rr].retype(dt), r4[rr].retype(dt));
        rr += nr;
    }
}

// Perform a unary register-wise operation on a register block.
template <typename F>
static inline void map(ngen::HW hw, ngen::DataType dt, const GRFMultirange &regs,
                       const RegisterLayout &layout, const CommonStrategy &strategy, F f,
                       int cxComponent = -1)
{
    using namespace ngen;

    int curReg = 0, curOff = 0, curBytes = 0;
    auto ebytes = getBytes(dt);

    auto map1 = [&]() {
        curOff &= -ebytes;
        curBytes &= -ebytes;
        while (curBytes) {
            int maxBytes;
            int regOff = curOff & (GRF::bytes(hw) - 1);
            if (regOff != 0)
                maxBytes = GRF::bytes(hw) - regOff;
            else {
                int nr = 1;
                if (canDualGRF(hw, dt, strategy) && regs.contiguous(curOff >> GRF::log2Bytes(hw), 2))
                    nr = 2;
                maxBytes = nr * GRF::bytes(hw);
            }

            auto nbytes = ngen::utils::rounddown_pow2(std::min(maxBytes, curBytes));
            auto ne = std::min<int>(32, nbytes / ebytes);
            nbytes = ne * ebytes;

            auto reg = regs[curOff >> GRF::log2Bytes(hw)].sub((curOff & (GRF::bytes(hw) - 1)) / ebytes, dt)(1);

            f(ne, reg);

            curBytes -= nbytes;
            curOff += nbytes;
        }
    };

    for (auto &block : layout) {
        int endReg = (curOff + curBytes + block.bytes - 1) >> GRF::log2Bytes(hw);
        if ((block.offsetBytes == curOff + curBytes) && regs.contiguous(curReg, endReg - curReg + 1))
            curBytes += block.bytes;
        else {
            map1();
            curOff = block.offsetBytes;
            curReg = curOff >> GRF::log2Bytes(hw);
            curBytes = block.bytes;
        }
    }

    map1();
}

// Variants that allow the type to be specified as a (C++ type) template parameter.
template <typename T, typename F>
static inline void map(ngen::HW hw, const GRFMultirange &r1, const GRFMultirange &r2,
                       const CommonStrategy &strategy, F f) {
    map(hw, ngen::getDataType<T>(), r1, r2, strategy, f);
}

template <typename T, typename F>
static inline void map(ngen::HW hw, const GRFMultirange &r1, const GRFMultirange &r2, const GRFMultirange &r3,
                       const CommonStrategy &strategy, F f) {
    map(hw, ngen::getDataType<T>(), r1, r2, r3, strategy, f);
}

template <typename T, typename F>
static inline void map(ngen::HW hw, const GRFMultirange &regs, const RegisterLayout &layout,
                       const CommonStrategy &strategy, F f) {
    map(hw, ngen::getDataType<T>(), regs, layout, strategy, f);
}

// Variant that allow the type to be specified as a native Type, rather than an nGEN type.
template <typename... Targs>
static inline void map(ngen::HW hw, Type T, Targs &&...args) {
    map(hw, T.ngen(), std::forward<Targs>(args)...);
}

static inline bool canDualGRF(ngen::HW hw, ngen::DataType dt, const CommonStrategy &strategy)
{
    return (strategy.dualGRF && (elementsPerGRF(hw, dt) < 32));
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
