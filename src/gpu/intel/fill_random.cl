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

#include "gpu/intel/include/philox.h"

// Fills a buffer with pseudo-random data using Philox RNG.
// Each work-item generates 4 × uint32 (16 bytes) via vstore4.
// The mask clears exponent LSBs to prevent NaN/Inf for FP types.
// Output is (random & mask) | value; mask = 0 gives a constant fill.
__kernel void fill_random(__global uint *buf, uint seed, ulong byte_count,
        uint mask, ulong value) {
    const ulong start = (ulong)get_global_id(0) * 16;
    if (start >= byte_count) return;

    uint b = (uint)(start >> 2);
    uint2 v = as_uint2(value);
    uint4 rnd = (philox_4x32_vec4(b, b ^ seed) & (uint4)(mask)) | (uint4)(v, v);

    if (start + 16 <= byte_count) {
        vstore4(rnd, get_global_id(0), buf);
    } else {
        __global uchar *p = (__global uchar *)buf;
        uchar16 bytes = as_uchar16(rnd);
        for (ulong i = 0; i < byte_count - start; i++)
            p[start + i] = bytes[i];
    }
}
