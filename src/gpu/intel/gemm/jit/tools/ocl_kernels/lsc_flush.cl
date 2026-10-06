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

__attribute__((intel_reqd_sub_group_size(16)))
kernel void lsc_flush_kernel(global uint *ptr, ulong bytes)
{
    uint loaded = 0;
    ulong off = get_global_id(0) * 64;
    if (off < bytes) {
        ptr += off / sizeof(uint);
        __asm__ volatile("lsc_load.ugm.uc.uc (M1,16) %0:d32 flat[%1]:a64" : "=rw"(loaded) : "rw"(ptr));
    }
}