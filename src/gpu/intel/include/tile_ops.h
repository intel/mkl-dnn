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

#ifndef GPU_INTEL_INCLUDE_TILE_OPS_H
#define GPU_INTEL_INCLUDE_TILE_OPS_H

#include "gpu/intel/include/generic_vector_ops.h"
#include "gpu/intel/include/types.h"

#define def_as_native_layout(dt) \
    NATIVE_LAYOUT_TYPE(dt) \
    __attribute__((overloadable)) as_native_layout(dt val) { \
        return val.data; \
    }
#define def_identity_native_layout_into(dt) \
    dt __attribute__((overloadable)) as_native_layout(dt val) { \
        return val; \
    }

def_as_native_layout(bf16);
def_as_native_layout(f8_e5m2);
def_as_native_layout(f8_e4m3);
def_as_native_layout(e8m0);
def_as_native_layout(f4_e2m1);
def_as_native_layout(s4);
def_as_native_layout(u4);

def_identity_native_layout_into(float);
def_identity_native_layout_into(char);
def_identity_native_layout_into(uchar);
def_identity_native_layout_into(short);
def_identity_native_layout_into(ushort);
def_identity_native_layout_into(int);
def_identity_native_layout_into(uint);
IF_DOUBLE_SUPPORTED(def_identity_native_layout_into(double));
IF_HALF_SUPPORTED(def_identity_native_layout_into(half));

#undef def_as_native_layout
#undef def_identity_native_layout_into

#define IS_POWER_OF_2(x) (((x) != 0) && (((x) & ((x) - 1)) == 0))

float __builtin_IB_atomic_max_local_f32(__local float *, float);
float __builtin_IB_atomic_add_local_f32(__local float *, float);
float __builtin_IB_atomic_add_global_f32(__global float *, float);
half __builtin_IB_atomic_add_global_f16(__global half *, half);

__attribute__((overloadable)) float local_atomic_max(local float *p, float v) {
    return __builtin_IB_atomic_max_local_f32(p, v);
}

/* not implemented */
__attribute__((overloadable)) half local_atomic_max(local half *p, half v);
__attribute__((overloadable)) ushort local_atomic_max(
        local ushort *p, ushort v);

__attribute__((overloadable)) uint local_atomic_max(local uint *p, uint v) {
    return atomic_max(p, v);
}

__attribute__((overloadable)) int local_atomic_max(local int *p, int v) {
    return atomic_max(p, v);
}

__attribute__((overloadable)) float local_atomic_add(local float *p, float v) {
    return __builtin_IB_atomic_add_local_f32(p, v);
}

/* not implemented */
__attribute__((overloadable)) half local_atomic_add(local half *p, half v);
__attribute__((overloadable)) ushort local_atomic_add(
        local ushort *p, ushort v);

__attribute__((overloadable)) uint local_atomic_add(local uint *p, uint v) {
    return atomic_add(p, v);
}

__attribute__((overloadable)) int local_atomic_add(local int *p, int v) {
    return atomic_add(p, v);
}

__attribute__((overloadable)) float global_atomic_add(
        global float *p, float v) {
    return __builtin_IB_atomic_add_global_f32(p, v);
}

__attribute__((overloadable)) half global_atomic_add(global half *p, half v) {
    return __builtin_IB_atomic_add_global_f16(p, v);
}

/* not implemented */
__attribute__((overloadable)) ushort global_atomic_add(
        global ushort *p, ushort v);

__attribute__((overloadable)) uint global_atomic_add(global uint *p, uint v) {
    return atomic_add(p, v);
}

__attribute__((overloadable)) int global_atomic_add(global int *p, int v) {
    return atomic_add(p, v);
}

#define DEF_BLOCK_LOAD_STORE(type, itype, suffix, n) \
    __attribute__((overloadable)) type##n block_load( \
            const global type *p, int vlen) \
            __attribute__((enable_if(vlen == n, "wrong vector length"))) { \
        return as_##type##n( \
                intel_sub_group_block_read##suffix##n((global void *)p)); \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##n v) { \
        intel_sub_group_block_write##suffix##n( \
                (global itype *)p, as_##itype##n(v)); \
    } \
    __attribute__((overloadable)) void block_store(local type *p, type##n v) { \
        intel_sub_group_block_write##suffix##n( \
                (local itype *)p, as_##itype##n(v)); \
    }

#define DEF_BLOCK_LOAD_STORE1(type, itype, suffix) \
    __attribute__((overloadable)) type##1 block_load( \
            const global type *p, int vlen) \
            __attribute__((enable_if(vlen == 1, "wrong vector length"))) { \
        type##1 x; \
        x[0] = as_##type( \
                intel_sub_group_block_read##suffix((global void *)p)); \
        return x; \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##1 v) { \
        intel_sub_group_block_write##suffix( \
                (global itype *)p, as_##itype(v[0])); \
    } \
    __attribute__((overloadable)) void block_store(local type *p, type##1 v) { \
        int id = get_sub_group_local_id(); \
        ((local itype *)p)[id] = as_##itype(v[0]); \
    }

#define DEF_BLOCK_LOAD_STORE16(type, itype, suffix) \
    __attribute__((overloadable)) type##16 block_load( \
            const global type *p, int vlen) \
            __attribute__((enable_if(vlen == 16, "wrong vector length"))) { \
        type##16 x; \
        x.s01234567 = as_##type##8( \
                intel_sub_group_block_read##suffix##8((global void *)p)); \
        x.s89abcdef = as_##type##8(intel_sub_group_block_read##suffix##8( \
                (global void *)(p + 8 * get_sub_group_size()))); \
        return x; \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##16 v) { \
        intel_sub_group_block_write##suffix##8( \
                (global itype *)p, as_##itype##8(v.s01234567)); \
        intel_sub_group_block_write##suffix##8( \
                (global itype *)(p + 8 * get_sub_group_size()), \
                as_##itype##8(v.s89abcdef)); \
    } \
    __attribute__((overloadable)) void block_store( \
            local type *p, type##16 v) { \
        intel_sub_group_block_write##suffix##8( \
                (local itype *)p, as_##itype##8(v.s01234567)); \
        intel_sub_group_block_write##suffix##8( \
                (local itype *)(p + 8 * get_sub_group_size()), \
                as_##itype##8(v.s89abcdef)); \
    }

#define DEF_BLOCK_LOAD_STORE32(type, itype, suffix) \
    __attribute__((overloadable)) type##32 block_load( \
            const global type *p, int vlen) \
            __attribute__((enable_if(vlen == 32, "wrong vector length"))) { \
        type##32 x; \
        x = (type##32)(as_##type##8(intel_sub_group_block_read##suffix##8( \
                               (global void *)p)), \
                as_##type##8(intel_sub_group_block_read##suffix##8( \
                        (global void *)(p + 8 * get_sub_group_size()))), \
                as_##type##8(intel_sub_group_block_read##suffix##8( \
                        (global void *)(p + 16 * get_sub_group_size()))), \
                as_##type##8(intel_sub_group_block_read##suffix##8( \
                        (global void *)(p + 24 * get_sub_group_size())))); \
        return x; \
    } \
    __attribute__((overloadable)) void block_store( \
            global type *p, type##32 v) { \
        intel_sub_group_block_write##suffix##8((global itype *)p, \
                (itype##8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])); \
        intel_sub_group_block_write##suffix##8( \
                (global itype *)(p + 8 * get_sub_group_size()), \
                (itype##8)(v[8], v[9], v[10], v[11], v[12], v[13], v[14], \
                        v[15])); \
        intel_sub_group_block_write##suffix##8( \
                (global itype *)(p + 16 * get_sub_group_size()), \
                (itype##8)(v[16], v[17], v[18], v[19], v[20], v[21], v[22], \
                        v[23])); \
        intel_sub_group_block_write##suffix##8( \
                (global itype *)(p + 24 * get_sub_group_size()), \
                (itype##8)(v[24], v[25], v[26], v[27], v[28], v[29], v[30], \
                        v[31])); \
    } \
    __attribute__((overloadable)) void block_store( \
            local type *p, type##32 v) { \
        intel_sub_group_block_write##suffix##8((local itype *)p, \
                (itype##8)(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])); \
        intel_sub_group_block_write##suffix##8( \
                (local itype *)(p + 8 * get_sub_group_size()), \
                (itype##8)(v[8], v[9], v[10], v[11], v[12], v[13], v[14], \
                        v[15])); \
        intel_sub_group_block_write##suffix##8( \
                (local itype *)(p + 16 * get_sub_group_size()), \
                (itype##8)(v[16], v[17], v[18], v[19], v[20], v[21], v[22], \
                        v[23])); \
        intel_sub_group_block_write##suffix##8( \
                (local itype *)(p + 24 * get_sub_group_size()), \
                (itype##8)(v[24], v[25], v[26], v[27], v[28], v[29], v[30], \
                        v[31])); \
    }

DEF_BLOCK_LOAD_STORE1(half, ushort, _us)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 2)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 4)
DEF_BLOCK_LOAD_STORE(half, ushort, _us, 8)
DEF_BLOCK_LOAD_STORE16(half, ushort, _us)
typedef ushort ushort32 __attribute__((ext_vector_type(32)));
typedef half half32 __attribute__((ext_vector_type(32)));
DEF_BLOCK_LOAD_STORE32(half, ushort, _us)

typedef ushort ushort1 __attribute__((ext_vector_type(1)));
DEF_BLOCK_LOAD_STORE1(ushort, ushort, _us)
DEF_BLOCK_LOAD_STORE(ushort, ushort, _us, 2)
DEF_BLOCK_LOAD_STORE(ushort, ushort, _us, 4)
DEF_BLOCK_LOAD_STORE(ushort, ushort, _us, 8)
DEF_BLOCK_LOAD_STORE16(ushort, ushort, _us)
DEF_BLOCK_LOAD_STORE32(ushort, ushort, _us)

DEF_BLOCK_LOAD_STORE1(uint, uint, )
DEF_BLOCK_LOAD_STORE(uint, uint, , 2)
DEF_BLOCK_LOAD_STORE(uint, uint, , 4)
DEF_BLOCK_LOAD_STORE(uint, uint, , 8)
DEF_BLOCK_LOAD_STORE16(uint, uint, )
typedef uint uint32 __attribute__((ext_vector_type(32)));
DEF_BLOCK_LOAD_STORE32(uint, uint, )

DEF_BLOCK_LOAD_STORE1(float, uint, )
DEF_BLOCK_LOAD_STORE(float, uint, , 2)
DEF_BLOCK_LOAD_STORE(float, uint, , 4)
DEF_BLOCK_LOAD_STORE(float, uint, , 8)
DEF_BLOCK_LOAD_STORE16(float, uint, )
typedef float float32 __attribute__((ext_vector_type(32)));
DEF_BLOCK_LOAD_STORE32(float, uint, )

#define DEF_BLOCK2D_LOAD_STORE(type, itype, vl, SG, suffix, BR, BC) \
    itype##vl __builtin_IB_subgroup_block_read_flat_##suffix( \
            long, int, int, int, int2); \
    void __builtin_IB_subgroup_block_write_flat_##suffix( \
            long, int, int, int, int2, itype##vl); \
    __attribute__((overloadable)) type##vl block2d_load(const global type *p, \
            int w, int h, int ld, int x, int y, int br, int bc, int sg) \
            __attribute__((enable_if(br == BR, "wrong #rows"))) \
            __attribute__((enable_if(bc == BC, "wrong #columns"))) \
            __attribute__((enable_if(sg == SG, "wrong subgroup size"))) { \
        ulong pp = as_long(p); \
        ulong prem = pp & 0x3F; \
        pp &= ~0x3F; \
        x += (prem / sizeof(type)); \
        w += prem; \
        int2 coord = {x, y}; \
        return as_##type##vl(__builtin_IB_subgroup_block_read_flat_##suffix( \
                pp, w - 1, h - 1, ld - 1, coord)); \
    } \
    __attribute__((overloadable)) void block2d_store(type##vl v, \
            const global type *p, int w, int h, int ld, int x, int y, int br, \
            int bc, int sg) \
            __attribute__((enable_if(br == BR, "wrong #rows"))) \
            __attribute__((enable_if(bc == BC, "wrong #columns"))) \
            __attribute__((enable_if(sg == SG, "wrong subgroup size"))) { \
        ulong pp = as_long(p); \
        ulong prem = pp & 0x3F; \
        pp &= ~0x3F; \
        x += (prem / sizeof(type)); \
        w += prem; \
        int2 coord = {x, y}; \
        __builtin_IB_subgroup_block_write_flat_##suffix( \
                pp, w - 1, h - 1, ld - 1, coord, as_##itype##vl(v)); \
    }

DEF_BLOCK2D_LOAD_STORE(half, ushort, 8, 16, u16_m8k16v1, 16, 8)
DEF_BLOCK2D_LOAD_STORE(half, ushort, 8, 16, u16_m4k32v1, 32, 4)
DEF_BLOCK2D_LOAD_STORE(half, ushort, 16, 16, u16_m8k32v1, 32, 8)

DEF_BLOCK2D_LOAD_STORE(ushort, ushort, 8, 16, u16_m8k16v1, 16, 8)
DEF_BLOCK2D_LOAD_STORE(ushort, ushort, 8, 16, u16_m4k32v1, 32, 4)
DEF_BLOCK2D_LOAD_STORE(ushort, ushort, 16, 16, u16_m8k32v1, 32, 8)

DEF_BLOCK2D_LOAD_STORE(float, uint, 8, 16, u32_m8k16v1, 16, 8)

/* Native 2D block ops for u32 with br=32 (u32_m4k32v1, u32_m8k32v1) are not valid
   Compose from valid u32_m8k16v1 (br=16, bc=8) block2d ops where possible,
   falling back to subgroup-wide element access otherwise */
__attribute__((overloadable)) float8 block2d_load(const global float *p, int w,
        int h, int ld, int x, int y, int br, int bc, int sg)
        __attribute__((enable_if(br == 32, "wrong #rows")))
        __attribute__((enable_if(bc == 4, "wrong #columns")))
        __attribute__((enable_if(sg == 16, "wrong subgroup size"))) {
    /* Load two 16x8 blocks via valid u32_m8k16v1 and extract first 4 cols */
    float8 lo = block2d_load(p, w, h, ld, x, y, 16, 8, 16);
    float8 hi = block2d_load(p, w, h, ld, x + 16, y, 16, 8, 16);
    return (float8)(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]);
}

__attribute__((overloadable)) void block2d_store(float8 v,
        const global float *p, int w, int h, int ld, int x, int y, int br,
        int bc, int sg) __attribute__((enable_if(br == 32, "wrong #rows")))
__attribute__((enable_if(bc == 4, "wrong #columns")))
__attribute__((enable_if(sg == 16, "wrong subgroup size"))) {
    /* Cannot use bc=8 block2d_store (would write extra columns)
       Fall back to subgroup-wide element stores */
    int lid = get_sub_group_local_id();
    int ld_f = ld / (int)sizeof(float);
    _Pragma("unroll") for (int c = 0; c < 4; c++) {
        global float *col = (global float *)p + (long)(y + c) * ld_f + x;
        col[lid] = v[2 * c];
        col[16 + lid] = v[2 * c + 1];
    }
}

__attribute__((overloadable)) float16 block2d_load(const global float *p, int w,
        int h, int ld, int x, int y, int br, int bc, int sg)
        __attribute__((enable_if(br == 32, "wrong #rows")))
        __attribute__((enable_if(bc == 8, "wrong #columns")))
        __attribute__((enable_if(sg == 16, "wrong subgroup size"))) {
    /* Compose from two valid u32_m8k16v1 (br=16, bc=8) block2d loads */
    float8 lo = block2d_load(p, w, h, ld, x, y, 16, 8, 16);
    float8 hi = block2d_load(p, w, h, ld, x + 16, y, 16, 8, 16);
    return (float16)(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3],
            lo[4], hi[4], lo[5], hi[5], lo[6], hi[6], lo[7], hi[7]);
}

__attribute__((overloadable)) void block2d_store(float16 v,
        const global float *p, int w, int h, int ld, int x, int y, int br,
        int bc, int sg) __attribute__((enable_if(br == 32, "wrong #rows")))
__attribute__((enable_if(bc == 8, "wrong #columns")))
__attribute__((enable_if(sg == 16, "wrong subgroup size"))) {
    /* Decompose into two u32_m8k16v1 (br=16, bc=8) block2d stores.
       Even vector indices hold rows 0-15, odd hold rows 16-31 */
    float8 lo = (float8)(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);
    float8 hi = (float8)(v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]);
    block2d_store(lo, p, w, h, ld, x, y, 16, 8, 16);
    block2d_store(hi, p, w, h, ld, x + 16, y, 16, 8, 16);
}

#define tile_fill(t, v) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = v; \
    } while (0)

#define tile_elementwise(t, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = f(t.x[i]); \
    } while (0)

#define tile_elementwise_s(t, f) \
    do { \
        _Pragma("unroll") for (int i = 0; \
                               i < sizeof((t).x) / sizeof((t).x[0]); i++) { \
            _Pragma("unroll") for (int s = 0; \
                                   s < sizeof((t).x[0]) / sizeof((t).x[0][0]); \
                                   s++)(t) \
                    .x[i][s] \
                    = f((t).x[i][s]); \
        } \
    } while (0)

#define tile_binary(t, t2, f) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t.x[i] \
                = f(t.x[i], t2.x[i]); \
    } while (0)

#define tile_store_global_bounds_cvt(_t, _ptr, _ld, _off_r, _off_c, _max_r, \
        _max_c, _cvt, _sg, _br, _bc, _nbr, _nbc) \
    do { \
        for (int _j = 0; _j < (_bc * _nbc); _j++) { \
            for (int _i0 = 0; _i0 < (_br * _nbr); _i0 += _sg) { \
                int _i = _i0 + get_sub_group_local_id(); \
                int _gr = (_off_r) + _j; \
                int _gc = (_off_c) + _i; \
                if (_gr < (_max_r) && _gc < (_max_c)) \
                    (_ptr)[(ulong)_gc * (_ld) + _gr] = _cvt( \
                            tile_access(_t, _i0, _j, _sg, _br, _bc, _nbr)); \
            } \
        } \
    } while (0)

#define tile_copy(t, t_new) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) t_new.x[i] \
                = __builtin_convertvector(t.x[i], __typeof__(t_new.x[i])); \
    } while (0)

#define tile_convert(t, t_new, conversion_func) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) { \
            _Pragma("unroll") for (int s = 0; \
                                   s < sizeof(t.x[0]) / sizeof(t.x[0][0]); \
                                   s++) t_new.x[i][s] \
                    = conversion_func(t.x[i][s]); \
        } \
    } while (0)

#define tile_copy_to_vec2_cvt(t, t_new, type, cvt) \
    do { \
        _Pragma("unroll") for (int i = 0; i < sizeof(t.x) / sizeof(t.x[0]); \
                               i++) { \
            _Pragma("unroll") for (int s = 0; \
                                   s < sizeof(t.x[0]) / sizeof(t.x[0][0]) / 2; \
                                   s++) { \
                type v = {cvt(t.x[i][2 * s]), cvt(t.x[i][2 * s + 1])}; \
                t_new.x[i][s] = as_uint(v); \
            } \
        } \
    } while (0)

#define tile_access(t, i0, j, sg, br, bc, nbr) \
    (t).x[(i0) / (br) + (nbr) * ((j) / (bc))] \
         [((i0) % (br)) / (sg) + ((j) % (bc)) * ((br) / (sg))]

#define xlane_tile_access(t, i, j, sg, br, bc, nbr) \
    sub_group_broadcast(tile_access(t, i, j, sg, br, bc, nbr), i % sg)

#define tile_predicated_assignment_t( \
        t, sg_offset_r, sg_offset_c, predicate, value, sg, br, bc, nbr, nbc) \
    do { \
        for (int j = 0; j < (bc * nbc); j++) { \
            for (int i0 = 0; i0 < (br * nbr); i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                int offset_r = sg_offset_r + j; \
                int offset_c = sg_offset_c + i; \
                if (predicate(offset_r, offset_c)) { \
                    tile_access(t, i0, j, sg, br, bc, nbr) = value; \
                } \
            } \
        } \
    } while (0)

#define tile_predicated_assignment( \
        t, sg_offset_r, sg_offset_c, predicate, value, sg, br, bc, nbr, nbc) \
    do { \
        for (int j = 0; j < (bc * nbc); j++) { \
            for (int i0 = 0; i0 < (br * nbr); i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                int offset_r = sg_offset_r + i; \
                int offset_c = sg_offset_c + j; \
                if (predicate(offset_r, offset_c)) { \
                    tile_access(t, i0, j, sg, br, bc, nbr) = value; \
                } \
            } \
        } \
    } while (0)

#define tile_predicated_select_t(t, sg_offset_r, sg_offset_c, predicate, \
        true_value, false_value, sg, br, bc, nbr, nbc) \
    do { \
        for (int j = 0; j < (bc * nbc); j++) { \
            for (int i0 = 0; i0 < (br * nbr); i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                int offset_r = sg_offset_r + j; \
                int offset_c = sg_offset_c + i; \
                tile_access(t, i0, j, sg, br, bc, nbr) \
                        = predicate(offset_r, offset_c) ? true_value \
                                                        : false_value; \
            } \
        } \
    } while (0)

#define tile_predicated_select(t, sg_offset_r, sg_offset_c, predicate, \
        true_value, false_value, sg, br, bc, nbr, nbc) \
    do { \
        for (int j = 0; j < (bc * nbc); j++) { \
            for (int i0 = 0; i0 < (br * nbr); i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                int offset_r = sg_offset_r + i; \
                int offset_c = sg_offset_c + j; \
                tile_access(t, i0, j, sg, br, bc, nbr) \
                        = predicate(offset_r, offset_c) ? true_value \
                                                        : false_value; \
            } \
        } \
    } while (0)

#define DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_full(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_full(tile_type *t, \
            const local element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load(tile_type *t, \
            const local element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[i]; \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_load_t_full(tile_type *t, \
            const local element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; \
                               i0 += sg, ptr += ld * sg) { \
            _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        = ptr[get_sub_group_local_id() * ld + j]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t(tile_type *t, \
            const local element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_t_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; \
                               i0 += sg, ptr += ld * sg) { \
            int i = i0 + get_sub_group_local_id(); \
            if (offset_r + i < m) \
                _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                    if (offset_c + j < n) { \
                        tile_access(*t, i0, j, sg, br, bc, nbr) \
                                = ptr[get_sub_group_local_id() * ld + j]; \
                    } \
                } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t_full(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += (off_t)ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; \
                               i0 += sg, ptr += ld * sg) { \
            _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        = ptr[get_sub_group_local_id() * ld + j]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_load_t_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += (off_t)ld * offset_r + offset_c; \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; \
                               i0 += sg, ptr += ld * sg) { \
            int i = i0 + get_sub_group_local_id(); \
            if (offset_r + i < m) \
                _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                    if (offset_c + j < n) { \
                        tile_access(*t, i0, j, sg, br, bc, nbr) \
                                = ptr[get_sub_group_local_id() * ld + j]; \
                    } \
                } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_t(t, ptr, m, n, n, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_t_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = ld * (i0 + get_sub_group_local_id()); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t(tile_type t, \
            local element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_store_t_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr++) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = ld * (i0 + get_sub_group_local_id()); \
                    if ((offset_r + i0 + get_sub_group_local_id()) < m) \
                        ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            local element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_store_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_full(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_store_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store(tile_type t, \
            global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_store(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_t_full(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += (off_t)ld * offset_r + offset_c; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = ld * (i0 + get_sub_group_local_id()); \
                ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t(tile_type t, \
            global element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= offset_r + br * nbr && n >= offset_c + bc * nbc) { \
            tile_store_t_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += (off_t)ld * offset_r + offset_c; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr++) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = ld * (i0 + get_sub_group_local_id()); \
                    if ((offset_r + i0 + get_sub_group_local_id()) < m) \
                        ptr[i] = tile_access(t, i0, j, sg, br, bc, nbr); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t_packed_src1(tile_type *t, \
            local element_type *ptr, int panel, int ld, int offset_r, \
            int offset_c) { \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r % panel; \
        int offset_r1 = offset_r - offset_r0; \
        ptr += offset_r0 + panel * offset_c + ld * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, ptr += sg * panel) { \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) \
                    tile_access(*(t), j0, i, sg, br, bc, nbr) \
                    = ptr[i]; \
        } \
    } \
    __attribute__((overloadable)) void tile_load_packed_src1(tile_type *t, \
            local element_type *ptr, int panel, int ld, int offset_r, \
            int offset_c) { \
        ptr += offset_c * panel; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += panel) \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
            int i = i0 + get_sub_group_local_id(); \
            int offset_r0 = (offset_r + i) % panel; \
            int offset_r1 = (offset_r + i) - offset_r0; \
            tile_access(*(t), i0, j, sg, br, bc, nbr) \
                    = ptr[offset_r0 + offset_r1 * ld]; \
        } \
    } \
    __attribute__((overloadable)) void tile_store_packed_src1(tile_type t, \
            local element_type *ptr, int panel, int ld, int offset_r, \
            int offset_c) { \
        ptr += offset_c * panel; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += panel) \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
            int i = i0 + get_sub_group_local_id(); \
            int offset_r0 = (offset_r + i) % panel; \
            int offset_r1 = (offset_r + i) - offset_r0; \
            ptr[offset_r0 + offset_r1 * ld] \
                    = tile_access(t, i0, j, sg, br, bc, nbr); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_packed_src1(tile_type t, \
            local element_type *ptr, int panel, int ld, int offset_r, \
            int offset_c) { \
        /* Assumption: block fits in a single panel */ \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r % panel; \
        int offset_r1 = offset_r - offset_r0; \
        ptr += offset_r0 + panel * offset_c + ld * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, ptr += sg * panel) { \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) ptr[i] \
                    = tile_access(t, j0, i, sg, br, bc, nbr); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_sys_src1(tile_type t, \
            local element_type *ptr, int tileR, int tileC, int wg_tile_m, \
            int wg_tile_n, int offset_r, int offset_c) { \
        const int crosspack = 4 / sizeof(element_type); \
        const int tile_panel_size = tileR * tileC; \
        const int num_row_panels = wg_tile_m / tileR; \
        const int num_col_panels = wg_tile_n / tileC; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                const int in_r = offset_r + i0 + get_sub_group_local_id(); \
                const int in_c = offset_c + j; \
                /* Compute 2D panel grid position: */ \
                const int row_panel = in_r \
                        / tileR; /* Which vertical panel (every sg rows) */ \
                const int col_panel = in_c \
                        / tileC; /* Which horizontal panel (every tile_n columns) */ \
                const int panel_base \
                        = (col_panel * num_row_panels + row_panel) \
                        * tile_panel_size; \
                /*const int panel_base = (row_panel * num_col_panels + col_panel) * tile_panel_size;*/ \
                /* Within-panel offsets using crosspack layout: */ \
                const int in_panel_row = in_r \
                        & (tileR - 1); /* Row within panel (in_r % sg) */ \
                const int in_panel_col = in_c \
                        & (tileC \
                                - 1); /* Column within panel (in_c % tile_n) */ \
                const int col_group_offset = (in_panel_col / crosspack) \
                        * (crosspack * tileR); /* Column pair group */ \
                const int sg_lane_offset = in_panel_row \
                        * crosspack; /* Subgroup lane position */ \
                const int crosspack_offset = (in_panel_col \
                        & (crosspack - 1)); /* Position within column pair */ \
                const int out_idx = panel_base + col_group_offset \
                        + sg_lane_offset + crosspack_offset; \
                ptr[out_idx] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_sys_src1(tile_type *t, \
            local element_type *ptr, int tileR, int tileC, int wg_tile_m, \
            int wg_tile_n, int offset_r, int offset_c) { \
        const int crosspack = 4 / sizeof(element_type); \
        const int tile_panel_size = tileR * tileC; \
        const int num_row_panels = wg_tile_m / tileR; \
        const int num_col_panels = wg_tile_n / tileC; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                const int in_r = offset_r + i0 + get_sub_group_local_id(); \
                const int in_c = offset_c + j; \
                /* Compute 2D panel grid position: */ \
                const int row_panel = in_r \
                        / tileR; /* Which vertical panel (every sg rows) */ \
                const int col_panel = in_c \
                        / tileC; /* Which horizontal panel (every tile_n columns) */ \
                const int panel_base \
                        = (col_panel * num_row_panels + row_panel) \
                        * tile_panel_size; \
                /*const int panel_base = (row_panel * num_col_panels + col_panel) * tile_panel_size;*/ \
                /* Within-panel offsets using crosspack layout: */ \
                const int in_panel_row = in_r \
                        & (tileR - 1); /* Row within panel (in_r % sg) */ \
                const int in_panel_col = in_c \
                        & (tileC \
                                - 1); /* Column within panel (in_c % tile_n) */ \
                const int col_group_offset = (in_panel_col / crosspack) \
                        * (crosspack * tileR); /* Column pair group */ \
                const int sg_lane_offset = in_panel_row \
                        * crosspack; /* Subgroup lane position */ \
                const int crosspack_offset = (in_panel_col \
                        & (crosspack - 1)); /* Position within column pair */ \
                const int out_idx = panel_base + col_group_offset \
                        + sg_lane_offset + crosspack_offset; \
                tile_access(*t, i0, j, sg, br, bc, nbr) = ptr[out_idx]; \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src1(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r & (sg - 1); \
        int offset_r1 = offset_r & ~(sg - 1); \
        ptr += offset_r0 + sg * offset_c + ld * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, ptr += sg * sg) { \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) ptr[i] \
                    = tile_access(t, j0, i, sg, br, bc, nbr); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src11(tile_type t, \
            local element_type *ptr, int tileR, int tileC, int wg_tile_m, \
            int wg_tile_n, int offset_r, int offset_c) { \
        const int crosspack = 4 / sizeof(element_type); \
        const int tile_panel_size = tileR * tileC; \
        const int num_row_panels \
                = wg_tile_m / tileR; /* is correct when _t? */ \
        const int num_col_panels = wg_tile_n / tileC; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                const int in_r = offset_r + i0 + get_sub_group_local_id(); \
                const int in_c = offset_c + j; \
                /* Compute 2D panel grid position: */ \
                const int row_panel = in_c / tileR; \
                const int col_panel = in_r / tileC; \
                const int panel_base \
                        = (col_panel * num_row_panels + row_panel) \
                        * tile_panel_size; \
                /* Within-panel offsets using crosspack layout: */ \
                const int in_panel_row = in_c \
                        & (tileR - 1); /* Row within panel (in_c % tileR) */ \
                const int in_panel_col = in_r \
                        & (tileC \
                                - 1); /* Column within panel (in_r % tileC) */ \
                const int col_group_offset = (in_panel_col / crosspack) \
                        * (crosspack * tileR); /* Column pair group */ \
                const int sg_lane_offset = in_panel_row \
                        * crosspack; /* Subgroup lane position */ \
                const int crosspack_offset = (in_panel_col \
                        & (crosspack - 1)); /* Position within column pair */ \
                const int out_idx = panel_base + col_group_offset \
                        + sg_lane_offset + crosspack_offset; \
                ptr[out_idx] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_sys_src22(tile_type t, \
            local element_type *ptr, int panel_n, int wg_tile_m, \
            int wg_tile_n, int offset_r, int offset_c) { \
        const int crosspack = 16; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                const int in_r = offset_r + i0 + get_sub_group_local_id(); \
                const int in_c = offset_c + j; \
                /* Panel-based addressing: panel_n cols per panel */ \
                const int col_panel = in_c / panel_n; \
                const int in_panel_c = in_c & (panel_n - 1); \
                /* Within-panel offsets using crosspack layout: */ \
                const int col_group_offset = (in_r / crosspack) \
                        * (crosspack * panel_n); /* Column pair group */ \
                const int sg_lane_offset \
                        = in_panel_c * crosspack; /* Subgroup lane position */ \
                const int crosspack_offset = (in_r & (crosspack - 1)); \
                const int out_idx = col_panel * (panel_n * wg_tile_m) \
                        + col_group_offset + sg_lane_offset \
                        + crosspack_offset; \
                ptr[out_idx] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src22(tile_type t, \
            local element_type *ptr, int panel_n, int wg_tile_m, \
            int wg_tile_n, int offset_r, int offset_c) { \
        const int crosspack = 16; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                const int in_r = offset_r + i0 + get_sub_group_local_id(); \
                const int in_c = offset_c + j; \
                /* Panel-based addressing: panel_n cols per panel */ \
                const int col_panel = in_r / panel_n; \
                const int in_panel_c = in_r & (panel_n - 1); \
                /* Within-panel offsets using crosspack layout: */ \
                const int col_group_offset = (in_c / crosspack) \
                        * (crosspack * panel_n); /* Column pair group */ \
                const int sg_lane_offset \
                        = in_panel_c * crosspack; /* Subgroup lane position */ \
                const int crosspack_offset = (in_c \
                        & (crosspack - 1)); /* Position within column pair */ \
                const int out_idx = col_panel * (panel_n * wg_tile_n) \
                        + col_group_offset + sg_lane_offset \
                        + crosspack_offset; \
                ptr[out_idx] = tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_t_sys_src2(tile_type t, \
            local element_type *ptr, int tile_n, int ld, int offset_r, \
            int offset_c) { \
        const int cp = 32 / sizeof(element_type); \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r & (cp - 1); \
        int offset_r1 = offset_r & ~(cp - 1); \
        ptr += offset_r0 + tile_n * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, offset_c += sg) { \
            int offset_c0 = offset_c & (tile_n - 1); \
            int offset_c1 = offset_c & ~(tile_n - 1); \
            local element_type *ptr_j = ptr + cp * offset_c0 + ld * offset_c1; \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) { \
                *ptr_j = tile_access(t, j0, i, sg, br, bc, nbr); \
                ptr_j++; \
                if ((~i & (cp - 1)) == 0) ptr_j += cp * (tile_n - 1); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_t_sys_src2(tile_type *t, \
            local element_type *ptr, int tile_n, int ld, int offset_r, \
            int offset_c) { \
        const int cp = 32 / sizeof(element_type); \
        offset_c += get_sub_group_local_id(); \
        int offset_r0 = offset_r & (cp - 1); \
        int offset_r1 = offset_r & ~(cp - 1); \
        ptr += offset_r0 + tile_n * offset_r1; \
        _Pragma("unroll") for (int j0 = 0; j0 < br * nbr; \
                               j0 += sg, offset_c += sg) { \
            int offset_c0 = offset_c & (tile_n - 1); \
            int offset_c1 = offset_c & ~(tile_n - 1); \
            local element_type *ptr_j = ptr + cp * offset_c0 + ld * offset_c1; \
            _Pragma("unroll") for (int i = 0; i < bc * nbc; i++) { \
                tile_access(*t, j0, i, sg, br, bc, nbr) = *ptr_j; \
                ptr_j++; \
                if ((~i & (cp - 1)) == 0) ptr_j += cp * (tile_n - 1); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_atomic_max_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                (void)local_atomic_max( \
                        ptr + i, tile_access(t, i0, j, sg, br, bc, nbr)); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_atomic_add_full(tile_type t, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                (void)local_atomic_add( \
                        ptr + i, tile_access(t, i0, j, sg, br, bc, nbr)); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_atomic_add_full(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                int i = i0 + get_sub_group_local_id(); \
                (void)global_atomic_add( \
                        ptr + i, tile_access(t, i0, j, sg, br, bc, nbr)); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_atomic_add(tile_type t, \
            global element_type *ptr, int m, int n, int ld, int offset_r, \
            int offset_c) { \
        if (m >= (offset_r + (br * nbr)) && n >= (offset_c + (bc * nbc))) { \
            tile_atomic_add_full(t, ptr, ld, offset_r, offset_c); \
            return; \
        } \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = i0 + get_sub_group_local_id(); \
                    if (offset_r + i < m) \
                        (void)global_atomic_add(ptr + i, \
                                tile_access(t, i0, j, sg, br, bc, nbr)); \
                } \
            } \
        } \
    }

#define DECLARE_2D_TILE_VREDUCE(tile_type, sg, br, bc, nbr, nbc, rtile_type, \
        rsg, rbr, rbc, rnbr, rnbc) \
    __attribute__((overloadable)) void tile_vreduce_add( \
            tile_type t, rtile_type *tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr) \
                        += tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vreduce_max( \
            tile_type t, rtile_type *tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr) \
                        = max(tile_access(t, i0, j, sg, br, bc, nbr), \
                                tile_access(*tr, i0, 0, rsg, rbr, rbc, rnbr)); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_add( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        += tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_sub( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        -= tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_mul( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        *= tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_vbroadcast_min( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        = min(tile_access(*t, i0, j, sg, br, bc, nbr), \
                                tile_access(tr, i0, 0, rsg, rbr, rbc, rnbr)); \
            } \
        } \
    }

#define DECLARE_2D_TILE_HREDUCE(tile_type, sg, br, bc, nbr, nbc, rtile_type, \
        rsg, rbr, rbc, rnbr, rnbc) \
    __attribute__((overloadable)) void tile_hreduce_add( \
            tile_type t, rtile_type *tr) { \
        _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
            _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
                tile_access(*tr, i0, j, rsg, rbr, rbc, rnbr) \
                        += tile_access(t, i0, j, sg, br, bc, nbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_add( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        += xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_sub( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        -= xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_mul( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) \
                        *= xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_hbroadcast_min( \
            tile_type *t, rtile_type tr) { \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                tile_access(*t, i0, j, sg, br, bc, nbr) = min( \
                        tile_access(*t, i0, j, sg, br, bc, nbr), \
                        xlane_tile_access(tr, j, 0, rsg, rbr, rbc, rnbr)); \
            } \
        } \
    }

#define DECLARE_2D_TILE_RSELECT(tile_type0, sg0, br0, bc0, nbr0, nbc0, \
        tile_type1, sg1, br1, bc1, nbr1, nbc1) \
    __attribute__((overloadable)) void tile_rselect( \
            tile_type0 *t0, tile_type1 t1, int idx) { \
        _Pragma("unroll") for (int j = 0; j < bc0 * nbc0; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br0 * nbr0; i0 += sg0) { \
                tile_access(*t0, i0, j, sg0, br0, bc0, nbr0) \
                        = tile_access(t1, i0, j, sg1, br1, bc1, nbr1); \
                _Pragma("unroll") for (int z = 1; \
                                       z < (br1 * nbr1 / br0 * nbr0); \
                                       z++) if (z == idx) { \
                    tile_access(*t0, i0, j, sg0, br0, bc0, nbr0) \
                            = tile_access(t1, i0 + z * br0 * nbr0, j, sg1, \
                                    br1, bc1, nbr1); \
                } \
            } \
        } \
    }

#define DECLARE_2D_TILE_COPY_REBLOCK(tile_type0, sg0, br0, bc0, nbr0, nbc0, \
        tile_type1, sg1, br1, bc1, nbr1, nbc1, conversion_func) \
    __attribute__((overloadable)) void tile_copy_reblock( \
            tile_type0 t0, tile_type1 *t1) { \
        _Pragma("unroll") for (int j = 0; j < bc0 * nbc0; j++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < br0 * nbr0; i0 += sg0) { \
                tile_access(*t1, i0, j, sg1, br1, bc1, nbr1) \
                        = conversion_func( \
                                tile_access(t0, i0, j, sg0, br0, bc0, nbr0)); \
            } \
        } \
    }

#define DECLARE_2D_TILE_PRINT(tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void print_tile(tile_type t, \
            const __constant char *format, int wg_x, int wg_y, int wg_z, \
            int sg_per_wg_m, int sg_per_wg_n) { \
        if (get_group_id(0) == wg_x && get_group_id(1) == wg_y \
                && get_group_id(2) == wg_z) { \
            uint sg_ij = sub_group_broadcast(get_local_id(1), 0); \
            int sg_i = sg_ij % sg_per_wg_m; \
            int sg_j = sg_ij / sg_per_wg_m; \
            if (get_local_id(0) == 0 && get_local_id(1) == 0 \
                    && get_local_id(2) == 0) \
                printf(#tile_type "(%lu,%lu):\n", get_group_id(0), \
                        get_group_id(1)); \
            barrier(CLK_LOCAL_MEM_FENCE); \
            for (int sgr = 0; sgr < sg_per_wg_n; sgr++) { \
                for (int rr = 0; rr < nbr * br; rr++) { \
                    barrier(CLK_LOCAL_MEM_FENCE); \
                    if (get_local_id(0) == 0 && get_local_id(1) == 0) { \
                        printf("%d: ", sgr *nbr *br + rr); \
                    } \
                    barrier(CLK_LOCAL_MEM_FENCE); \
                    for (int sgc = 0; sgc < sg_per_wg_m; sgc++) { \
                        if (sg_i == sgc && sg_j == sgr) { \
                            for (int cc = 0; cc < nbc * bc; cc++) { \
                                element_type value; \
                                value = xlane_tile_access( \
                                        t, rr, cc, sg, br, bc, nbr); \
                                if (get_sub_group_local_id() == 0) \
                                    printf(format, value); \
                            } \
                        } \
                        barrier(CLK_LOCAL_MEM_FENCE); \
                    } \
                    if (get_local_id(0) == 0 && get_local_id(1) == 0) \
                        printf("\n"); \
                } \
            } \
        } \
    }

#define DECLARE_2D_TILE(tile_type, element_type, sg, br, bc, nbr, nbc) \
    typedef element_type \
            __attribute__((ext_vector_type(br * bc / sg))) _e_##tile_type; \
    typedef struct { \
        _Static_assert(IS_POWER_OF_2(br * bc / sg), \
                "Tile vector length (" #br " * " #bc " / " #sg \
                ") must be a power of 2"); \
        _e_##tile_type x[nbr * nbc]; \
    } tile_type; \
    DECLARE_2D_TILE_OPS(tile_type, element_type, sg, br, bc, nbr, nbc)

/* Requires bc == 1 currently */
#define DECLARE_2D_TILE_BLOCK_OPS( \
        tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_block(tile_type *t, \
            const global element_type *ptr, int ld, int offset_r, \
            int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                    ->x[ii + nbr * jj] \
                    = block_load(ptr + ii * br, br / sg); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block(tile_type t, \
            global element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) \
                    block_store(ptr + ii * br, (t).x[ii + nbr * jj]); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block_packed(tile_type t, \
            local element_type *ptr, int panel, int ld, int offset_r, \
            int offset_c) { \
        /* Assumes each block fits in a single panel */ \
        ptr += offset_c * panel; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += panel) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) { \
                int offset_r0 = (offset_r + ii * br) % panel; \
                int offset_r1 = (offset_r + ii * br) - offset_r0; \
                block_store(ptr + offset_r0 + offset_r1 * ld, \
                        (t).x[ii + nbr * jj]); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_block(tile_type *t, \
            const global element_type *ptr, int n, int ld, int offset_r, \
            int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        n -= offset_c; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            if (jj < n) { \
                _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                        ->x[ii + nbr * jj] \
                        = block_load(ptr + ii * br, br / sg); \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block(tile_type t, \
            global element_type *ptr, int n, int ld, int offset_r, \
            int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        n -= offset_c; \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++, ptr += ld * bc) { \
            if (jj < n) { \
                _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) \
                        block_store(ptr + ii * br, (t).x[ii + nbr * jj]); \
            } \
        } \
    }

#define DECLARE_2D_TILE_BLOCK2D_OPS( \
        tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_block2d(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        const int e = sizeof(element_type); \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++)(t) \
                    ->x[ii + nbr * jj] \
                    = block2d_load(ptr, m * e, n, ld * e, offset_r + ii * br, \
                            offset_c + jj * bc, br, bc, sg); \
        } \
    } \
    __attribute__((overloadable)) void tile_load_block2d(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_block2d(t, ptr, m, n, m, offset_r, offset_c); \
    } \
    __attribute__((overloadable)) void tile_store_block2d(tile_type t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        const int e = sizeof(element_type); \
        _Pragma("unroll") for (int jj = 0; jj < nbc; jj++) { \
            _Pragma("unroll") for (int ii = 0; ii < nbr; ii++) block2d_store( \
                    (t).x[ii + nbr * jj], ptr, m *e, n, ld *e, \
                    offset_r + ii * br, offset_c + jj * bc, br, bc, sg); \
        } \
    } \
    __attribute__((overloadable)) void tile_store_block2d(tile_type t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_store_block2d(t, ptr, m, n, m, offset_r, offset_c); \
    }

#define DECLARE_2D_TILE_LOAD_PACKED_VEC( \
        tile_type, element_type, vec_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_packed_vec2(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        ptr += (off_t)ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = 2 * (i0 + get_sub_group_local_id()); \
                    vec_type loaded = 0; \
                    if (offset_r + i < m) loaded.s0 = ptr[i]; \
                    if (offset_r + i + 1 < m) loaded.s1 = ptr[i + 1]; \
                    tile_access(*t, i0, j, sg, br, bc, nbr) = as_uint(loaded); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_packed_vec2(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_packed_vec2(t, ptr, m, n, m, offset_r, offset_c); \
    }

/* As DECLARE_2D_TILE_LOAD_PACKED_VEC, but applies a scalar conversion function
 * to each loaded element (e.g. fp8 -> half) before packing. */
#define DECLARE_2D_TILE_LOAD_PACKED_VEC_CVT( \
        tile_type, element_type, vec_type, cvt, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) void tile_load_packed_vec2_cvt(tile_type *t, \
            const global element_type *ptr, int m, int n, int ld, \
            int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < bc * nbc; j++, ptr += ld) { \
            if (offset_c + j < n) { \
                _Pragma("unroll") for (int i0 = 0; i0 < br * nbr; i0 += sg) { \
                    int i = 2 * (i0 + get_sub_group_local_id()); \
                    vec_type loaded = 0; \
                    if (offset_r + i < m) loaded.s0 = cvt(ptr[i]); \
                    if (offset_r + i + 1 < m) loaded.s1 = cvt(ptr[i + 1]); \
                    tile_access(*t, i0, j, sg, br, bc, nbr) = as_uint(loaded); \
                } \
            } \
        } \
    } \
    __attribute__((overloadable)) void tile_load_packed_vec2_cvt(tile_type *t, \
            const global element_type *ptr, int m, int n, int offset_r, \
            int offset_c) { \
        tile_load_packed_vec2_cvt(t, ptr, m, n, m, offset_r, offset_c); \
    }

#define cooperative_prefetch_2d(ptr, r, c, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_internal((const global uchar *)ptr, \
            (r) * sizeof(*(ptr)), c, (ld) * sizeof(*(ptr)), sg_id, n_sg, \
            sg_size, caching)

#define cooperative_prefetch_2d_rem( \
        ptr, r, c, rmax, cmax, ld, sg_id, n_sg, sg_size, caching) \
    cooperative_prefetch_2d_internal((const global uchar *)ptr, \
            (r) * sizeof(*(ptr)), c, (rmax) * sizeof(*(ptr)), cmax, \
            (ld) * sizeof(*(ptr)), sg_id, n_sg, sg_size, caching)

/* IGC prefetch intrinsics */
enum LSC_LDCC {
    LSC_LDCC_DEFAULT = 0,
    LSC_LDCC_L1UC_L3UC = 1,
    LSC_LDCC_L1UC_L3C = 2,
    LSC_LDCC_L1C_L3UC = 3,
    LSC_LDCC_L1C_L3C = 4,
    LSC_LDCC_L1S_L3UC = 5,
    LSC_LDCC_L1S_L3C = 6,
    LSC_LDCC_L1IAR_L3C = 7,
};

extern void __builtin_IB_lsc_prefetch_global_uchar(
        const __global uchar *base, int immElemOff, enum LSC_LDCC cacheOpt);

extern void __builtin_IB_lsc_prefetch_global_uint(
        const __global uint *base, int immElemOff, enum LSC_LDCC cacheOpt);

__attribute__((overloadable)) void cooperative_prefetch_2d_internal(
        const global uchar *ptr, uint rbytes, uint c, uint ld_bytes, uint sg_id,
        uint n_sg, uint sg_size, enum LSC_LDCC caching) {
    const uint cl_per_col = (rbytes + 63) >> 6;
    const uint cl = cl_per_col * c;

    const uint cl_per_sg = (cl + n_sg - 1) / n_sg;
    const uint cl_iters = (cl_per_sg + sg_size - 1) / sg_size;
#pragma unroll
    for (uint ii_cl = 0; ii_cl < cl_iters; ii_cl++) {
        uint i_cl = (ii_cl * cl_per_sg + sg_id) * sg_size
                + get_sub_group_local_id();
        if (i_cl < cl) {
            uint r_cl = i_cl % cl_per_col;
            uint c_cl = i_cl / cl_per_col;
            uint pf_off = r_cl * 64 + c_cl * ld_bytes;
            const global uint *p = (const global uint *)(ptr + pf_off);
            __builtin_IB_lsc_prefetch_global_uint(p, 0, caching);
        }
    }
}

__attribute__((overloadable)) void cooperative_prefetch_2d_internal(
        const global uchar *ptr, uint rbytes, uint c, uint rbytes_max,
        uint c_max, uint ld_bytes, uint sg_id, uint n_sg, uint sg_size,
        enum LSC_LDCC caching) {
    const uint cl_per_col = (rbytes_max + 63) >> 6;
    const uint cl = cl_per_col * c_max;

    const uint cl_per_sg = (cl + n_sg - 1) / n_sg;
    const uint cl_iters = (cl_per_sg + sg_size - 1) / sg_size;
    const uint max_off = rbytes - 1 + (c - 1) * ld_bytes;
#pragma unroll
    for (uint ii_cl = 0; ii_cl < cl_iters; ii_cl++) {
        uint i_cl = (ii_cl * cl_per_sg + sg_id) * sg_size
                + get_sub_group_local_id();
        if (i_cl < cl) {
            uint r_cl = i_cl % cl_per_col;
            uint c_cl = i_cl / cl_per_col;
            uint pf_off = min(r_cl * 64 + c_cl * ld_bytes, max_off);
            const global uchar *pp = ptr + pf_off;
            __builtin_IB_lsc_prefetch_global_uchar(pp, 0, caching);
        }
    }
}

// inplace load-add-store to SLM, avoids allocating a full intermediate
// accumulator tile.
#define DECLARE_2D_TILE_SLM_ADD(tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) inline void tile_slm_add(tile_type addend, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_c + offset_r; \
        _Pragma("unroll") for (int j = 0; j < (bc) * (nbc); j++, ptr += ld) { \
            _Pragma("unroll") for (int i0 = 0; i0 < (br) * (nbr); \
                                   i0 += (sg)) { \
                int i = i0 + get_sub_group_local_id(); \
                ptr[i] += tile_access(addend, i0, j, sg, br, bc, nbr); \
            } \
        } \
    }

#define DECLARE_2D_TILE_SLM_ADD_T( \
        tile_type, element_type, sg, br, bc, nbr, nbc) \
    __attribute__((overloadable)) inline void tile_slm_add_t(tile_type addend, \
            local element_type *ptr, int ld, int offset_r, int offset_c) { \
        ptr += ld * offset_r + offset_c; \
        _Pragma("unroll") for (int j = 0; j < (bc) * (nbc); j++, ptr++) { \
            _Pragma("unroll") for (int i0 = 0; i0 < (br) * (nbr); \
                                   i0 += (sg)) { \
                int i = ld * (i0 + get_sub_group_local_id()); \
                ptr[i] += tile_access(addend, i0, j, sg, br, bc, nbr); \
            } \
        } \
    }

#endif
