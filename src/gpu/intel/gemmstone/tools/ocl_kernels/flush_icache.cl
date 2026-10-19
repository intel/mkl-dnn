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

uint get_tm0_0() {
    uint val;
    __asm volatile("mov (M1_NM,1) %0(0,0)<1> V6(0,0)<0;1,0>" : "=rw"(val));
    return val;
}


#define REPEAT_10(s) s; s; s; s; s; s; s; s; s; s;

#define REPEAT_100(s) \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s); \
    REPEAT_10(s);

#define REPEAT_1000(s) \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s); \
    REPEAT_100(s);

#define REPEAT_10000(s) \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s); \
    REPEAT_1000(s);

__attribute__((intel_reqd_sub_group_size(16)))
kernel void flush_icache_kernel()
{
    uint x = get_tm0_0();

    REPEAT_10000((x ^= 0x89abcdef, x *= 0x01234567));

    volatile uint a = x;
}
