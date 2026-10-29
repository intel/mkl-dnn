/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_X64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/primitive_attr.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// NOLINTBEGIN(modernize-use-using)
// GCC treats using and typedef differently for enums and structs
// https://stackoverflow.com/questions/48613758
// The type defines organization of batch of matrices
typedef enum {
    // Undefined brgemm batch kind
    brgemm_batch_kind_undef = 0,
    // A and B arrays of pointers
    brgemm_addr = 1,
    // Base address and array of offsets from base address.
    brgemm_offs = 2,
    // Base address and fixed stride between matrices.
    brgemm_strd = 3,
    // Base address and static array of fixed offsets.
    brgemm_static_offs = 4,
} brgemm_batch_kind_t;

// The type defines the storage format of matrix
typedef enum {
    brgemm_layout_undef = 0,
    brgemm_col_major = 1,
    brgemm_row_major = 2,
} brgemm_layout_t;

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} brgemm_broadcast_t;

typedef enum {
    brgemm_lo_default = 0,
    brgemm_lo_bl_1load,
    brgemm_lo_bl_1bcst,
} brgemm_kernel_loop_order_t;

typedef enum {
    brgemm_prf_default = 1,
    brgemm_prf0,
    brgemm_prf1,
    brgemm_prf2,
    brgemm_prfNTA,
} brgemm_kernel_prefetching_t;

typedef enum {
    brgemm_prfw_default = 0,
    brgemm_prfw_store,
    brgemm_prfw_loop_store,
} brgemm_kernel_prefetchw_t;

typedef enum {
    brgemm_innermost_undef = 0,
    brgemm_bd_loop_innermost,
    brgemm_ld_loop_innermost,
} brgemm_kernel_innermost_loop_t;

typedef enum {
    brgemm_hint_nt_undef = 0,
    brgemm_hint_nt_false,
    brgemm_hint_nt_true,
} brgemm_kernel_hint_nt_t;
// NOLINTEND(modernize-use-using)

struct brgemm_strides_t {
    // Stride between A matrices
    dim_t stride_a;
    // Stride between B matrices
    dim_t stride_b;
};

// NOLINTBEGIN(modernize-use-using)
// memory advice feature heuristic is based on the performance tests done
// on simulator and lets the tile loading snoop for other cores caches if
// the A/B matrices are shared. thus, if already shared, no need to fetch
// from lower level memories the assumption is that if we don't divide
// the C matrix evenly on row chunks per thread, then it worth checking
// mem advice as there will be sharing
typedef enum {
    brgemm_hint_mem_advice_undef = 0,

    // only matrix A is read shared between threads. selected when
    // there is sharing of data between threads over the A matrix
    // when chunks are processed horizontally and the threads don't
    // divide the A buffer w/o remainder in chunks. Thus, it is worth
    // sharing between threads
    // or when chunks are processed vertically and split between threads
    brgemm_hint_mem_advice_A,

    // only matrix B is read shared between threads. selected when
    // there is sharing of data between threads over the A matrix
    // when chunks are processed vertically and the threads don't
    // divide the A buffer w/o remainder in chunks. Thus, it is worth
    // sharing between threads.
    // or when chunks are processed horizontally and split between threads
    brgemm_hint_mem_advice_B,

    // if both conditions above apply, it worth sharing on both A and B buffer
    // between threads
    brgemm_hint_mem_advice_A_B,
} brgemm_kernel_hint_mem_advice_t;
// NOLINTEND(modernize-use-using)

struct brgemm_prf_t {
    int dist0 {-1};
    int dist1 {-1};
    int dist2 {-1};
    int distNTA {-1};
    bool sprinkled {false};
};

struct brgemm_batch_element_t {
    brgemm_batch_element_t() { ptr.A = ptr.B = nullptr; }
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
    struct {
        dim_t top = 0;
        dim_t bottom = 0;
    } vvpad; // w.r.t. M dimension

    // Used to calculate compensation when batch padding is present.
    // Note: batch_pad represent the overlap between weights and the height
    // dimension w.r.t. convolution dimensions.
    dim_t has_s8s8_comp_batch_pad = 0;
};

struct DNNL_API brgemm_attr_t {
    brgemm_attr_t();
    // if unrolled kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on kernel call
    // else "max_bs" is the maximum batch size that can be used
    dim_t max_bs;
    int max_top_vpad, max_bottom_vpad;
    int max_top_bpad, max_bottom_bpad;
    dim_t hint_expected_A_size, hint_expected_B_size, hint_expected_C_size;
    brgemm_kernel_innermost_loop_t hint_innermost_loop
            = brgemm_ld_loop_innermost;
    brgemm_kernel_loop_order_t hint_loop_order;
    brgemm_kernel_prefetching_t hint_prefetching
            = brgemm_kernel_prefetching_t::brgemm_prf_default;
    brgemm_kernel_prefetchw_t hint_prefetchw
            = brgemm_kernel_prefetchw_t::brgemm_prfw_default;
    brgemm_prf_t hint_prfA, hint_prfB, hint_prfC;

    // This parameter determines how we will read the tail by K dimension from
    // matrix A. For AMX if the parameter is true then the brgemm will first
    // copy the data to the intermediate buffer and only then use the tileload.
    // For non-AMX the A data are loaded byte by byte if flag is set
    bool wary_A_k_tail_read {false};
    // For AMX the K dimension given to the brgemm is required to be divisible
    // by vnni granularity. In addition blocking by K dimension may not be
    // optimal if K greater than tile size and divisible by it.
    // The parameter 'extendable_k' enables the brgemm to use the optimal K
    // block size assuming that the following requirements for the matrix B are
    // fulfilled:
    //  - It is ​​properly blocked (64 bytes block by K dimension).
    //  - The dimension K is padded by zeros.
    // For K tail handling in this case the brgemm behavior is determined by the
    // 'wary_A_k_tail_read' parameter.
    bool extendable_k {false};
    bool generate_skip_accumulation;
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 – bd_mask is not used
    // 1 – bd_mask is used on storing stage only
    // 2 – bd_mask used both on reading and storing stages
    int bd_mask_level;
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    bool use_uker;
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    bool use_interleave_stores;
    impl::fpmath_mode_t fpmath_mode = fpmath_mode::strict;
    bool b_is_vnni {false};
    // Second level leading dimension describing distance between 16-line
    // blocks in case of blocked layout. Used to calculate address of next
    // bd block. By default are equal to regular leading dimension parameters
    // specified on brgemm creation.
    // Supported by brgemm unrolled kernel for now.
    dim_t LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    // If "true" then batchsize is allowed to change on each kernel call
    // and there is no unrolling by batchsize in kernel
    bool var_bs {false};
    bool postops_only {false};
    // Force the kernel to populate the C buffer (reg_C) on non-postops
    // calls, even when dt_c == dt_d and no post-ops. Set when an external
    // C buffer must be filled (e.g. cross-K partial reduction in dst dtype).
    bool use_intermediate_c_buffer {false};
    // Hint for bs_group value in brgemm_desc_t
    int hint_bs_group {0};

    int hint_bd_block {0};
    int hint_ld_block {0};
    int hint_bd_block2 {0};
    int hint_ld_block2 {0};
    bool hint_ununroll_bd_loop {false};

    brgemm_kernel_hint_mem_advice_t mem_advice {brgemm_hint_mem_advice_undef};

    brgemm_kernel_hint_nt_t hint_load_nt_A {brgemm_hint_nt_undef};
    brgemm_kernel_hint_nt_t hint_load_nt_B {brgemm_hint_nt_undef};
    // this variable is used in tile decomposition heuristics
    // to calculate "effective" K value which may be different from just
    // K * batch_size for non-1x1 convolutions due to data overlap
    float K_koef = 1.f;
    // this flag may be used to omit some actions on calling from blocking
    // in convolutions init_conf
    bool test_call = false;
    // bd_mask is char array in which each element is a boolean value that
    // determines whether to write this row to the result matrix or skip
    const char *bd_mask;
    // static_offsets is a static array of fixed offsets used for
    // brgemm_static_offs batch kind
    const brgemm_batch_element_t *static_offsets;
    // hint whether to use on the fly copy of A due to unaligned accesses
    bool hint_fused_copy_a = false;
    // request the ACE compute path; honored only when the descriptor resolves
    // to an ACE ISA, see brgemm_desc_t::is_ace()
    bool use_ace = false;
};

struct brgemm_desc_t {
    brgemm_desc_t() = default;
    brgemm_desc_t(const brgemm_desc_t &other);
    DNNL_API ~brgemm_desc_t();

    // Note: new added parameters must be taken into account in the brgemm
    // comparison function
    dim_t bcast_dim = 0; // M;
    dim_t load_dim = 0; // N;
    dim_t reduce_dim = 0; // K;
    dim_t LDA = 0;
    dim_t LDB = 0;
    dim_t LDC = 0;
    dim_t LDD = 0;

    bool fused_copy_a = false;
    // we use two isa_ variables
    // isa_user to store the user provided isa value
    // isa_impl to store actual implementation. This can change until the kernel
    // is created, as calls to set_attr can affect this variable. Ex: bf32
    impl::cpu::x64::cpu_isa_t isa_user = isa_undef;
    impl::cpu::x64::cpu_isa_t isa_impl = isa_undef;
    float alpha = 0.0f;
    float beta = 0.0f;

    impl::data_type_t dt_a = data_type::undef;
    impl::data_type_t dt_c = data_type::undef;
    impl::data_type_t dt_b = data_type::undef;
    impl::data_type_t dt_d = data_type::undef;
    impl::data_type_t dt_bias = data_type::undef;

    dim_t stride_a = 0; // Offset in bytes
    dim_t stride_b = 0;

    brgemm_layout_t layout = brgemm_layout_undef;
    brgemm_batch_kind_t type = brgemm_batch_kind_t::brgemm_addr;
    bool is_dgmm = false; // set to true in brdgmm_desc_init
    bool with_sum = false;
    bool req_cal_comp_pads = false;
    bool req_comp_pads_with_bcast = false;

    float sum_scale = 0.0f;
    int32_t sum_zp = 0;
    impl::data_type_t sum_dt = data_type::undef;
    bool with_eltwise = false;
    bool with_binary = false;
    bool skip_zp_b_compensation = false;
    bool skip_zp_a_compensation = false;
    bool n_bcast_1_load = false;
    bool fp8_with_f16_vnni_block = false;

    brgemm_broadcast_t zp_type_a = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_b = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_c = brgemm_broadcast_t::none;

    bool skip_wei_scales = false;
    bool is_single_wei_scale = false;
    bool is_per_n_wei_scales = false;
    bool is_per_k_wei_scales = false;
    bool with_src_scales = false;
    bool is_per_k_src_scales = false;
    dim_t src_scale_m_stride = 0;
    data_type_t dt_src_scales = data_type::undef;

    bool with_per_mn_compensation = false;

    bool has_per_k_scales() const {
        return is_per_k_wei_scales || is_per_k_src_scales;
    }

    bool with_wei_scales = false;
    // `dst_scales` passed as a bare pointer making kernel change multiplication
    // to division was proved to be significantly slower, both for pure divps
    // instruction, or for emulation through rcpps. Therefore, each
    // implementation dealing with `dst_scales` must prepare a per thread
    // scratchpad memory and inverse values (usually, just one) inside a
    // parallel task.
    bool with_dst_scales = false;
    data_type_t dt_wei_scales = data_type::undef;
    // Grouping in batch used by brdgmm kernel
    int bs_group {0};

    brgemm_attr_t brgattr;

    // Derived  parameters
    dim_t LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    bool is_blocked = false;

    dim_t bdb = 0;
    int bd_block = 0, bdb_tail = 0;
    dim_t bdb2 = 0;
    int bd_block2 = 0, bdb2_tail = 0;
    dim_t ldb = 0;
    int ld_block = 0, ldb_tail = 0;
    dim_t ldb2 = 0;
    int ld_block2 = 0, ldb2_tail = 0;
    dim_t rdb = 0;
    int rd_block = 0, rdb_tail = 0;
    int rd_step = 0, ld_step = 0;

    int typesize_A = 0;
    int typesize_B = 0;
    int typesize_C = 0;
    int typesize_D = 0;
    int typesize_bias = 0;

    bool is_ymm = false;
    bool is_zmm = false;
    bool is_tmm = false;
    bool is_int8 = false, is_int8_tmm = false;
    bool is_bf16 = false, is_bf16_tmm = false, is_bf16_emu = false;
    bool is_fp8 = false, is_fp8_tmm = false;
    bool is_f16 = false, is_f16_tmm = false;
    bool is_f32 = false;
    bool is_bf32 = false;

    bool has_int8_vnni = false;

    bool load_nt_A = false;
    bool load_nt_B = false;

    bool embd_bcst = false;
    bool with_bias = false;
    bool req_s8s8_compensation = false;
    bool req_src_s8_shift = false;
    // `with_weights_scale_adjust` is controlled by the implementation and not
    // by kernel API.
    bool with_weights_scale_adjust = false;
    brgemm_kernel_innermost_loop_t innermost_loop = brgemm_ld_loop_innermost;
    bool is_M_tail = false;
    bool interleave_tilestores_ = false;
    brgemm_prf_t prfA, prfB, prfC;
    bool is_runtime_lda = false;
    bool is_runtime_ldb = false;
    bool is_runtime_ldc = false;
    bool is_runtime_ldd = false;

    bool transA = false;

    // GEMV-specific parameters.
    //
    // GEMV kernel path is enabled.
    bool is_gemv = false;
    // Controls whether y is treated as a row in GEMV-specific code paths.
    bool treat_y_as_row = false;
    // GEMV transA register-blocking unroll factor.
    int gemv_transa_bd_unroll = 0;
    // non-transA: tail along the reduction dimension.
    // transA:     number of valid elements in the final vector accumulator.
    // Bounded: always < simd width.
    int gemv_tail = 0;

    static constexpr int MAX_VPAD = 100;
    static constexpr int AMX_TILES_NUM = 8;
    static constexpr int tilesize = 1024;
    // AMX accumulator tile element size (always f32, 4 bytes, independent of dt_c)
    static constexpr int amx_c_tile_elem_size = static_cast<int>(sizeof(float));

    void set_attr(const primitive_attr_t *ppdattr);
    void set_dst_md(const memory_desc_t *pdst_md);
    const primitive_attr_t *attr() const { return attr_; }
    const memory_desc_t *dst_md() const { return dst_md_; }

    // GEMV logical bd block used by GEMV-specific kernel code.
    //
    // non-transA:
    //   - same as `bd_block`
    //
    // transA:
    //  - number of full vector accumulators updated per reduction step
    //  - equal to `gemv_transa_bd_unroll`
    int gemv_bd_block() const {
        assert(is_gemv);
        if (!is_gemv) return 0;
        return transA ? gemv_transa_bd_unroll : bd_block;
    }

    // GEMV bdb tail used by GEMV-specific kernel code.
    //
    // non-transA:
    //   - same as `bdb_tail`
    //
    // transA:
    //   - number of full vector accumulators in the tail block
    //   - the remaining first-level register tail is stored separately in
    //     `gemv_tail`
    int gemv_bdb_tail() const {
        assert(is_gemv);
        if (!is_gemv) return 0;
        const int gemv_transa_simd_w
                = transA ? bd_block / gemv_transa_bd_unroll : 1;
        assert(IMPLICATION(transA, gemv_transa_simd_w != 1));
        return transA ? bdb_tail / gemv_transa_simd_w : bdb_tail;
    }

    // GEMV accumulator layout.
    //
    // non-transA:
    //   - each accumulator register holds a single output element
    //
    // transA:
    //   - each accumulator register holds a SIMD-width vector of output
    //     elements
    bool gemv_acc_is_vector() const {
        assert(is_gemv);
        return transA;
    }

    // Number of accumulator indices (`bd`) processed in the current GEMV block.
    //
    // non-transA:
    //   - same as the number of output elements in the current block
    //   - regular block: defined by `gemv_bd_block()`
    //   - tail block:    defined by `gemv_bdb_tail()`
    //
    // transA:
    //   - number of vector accumulators in the current block
    //   - regular block: number of full vector accumulators,
    //                    defined by `gemv_bd_block()`
    //   - tail block:    number of full vector accumulators in the tail block,
    //                    defined by `gemv_bdb_tail()`
    //   - if `gemv_tail > 0`, one additional accumulator is included for the
    //     final partial vector
    int gemv_num_acc_blocks(bool is_bdb_tail) const {
        assert(is_gemv);
        if (!gemv_acc_is_vector())
            return is_bdb_tail ? gemv_bdb_tail() : gemv_bd_block();

        const bool has_tail_acc = is_bdb_tail && gemv_tail > 0;
        const int full_accs = is_bdb_tail ? gemv_bdb_tail() : gemv_bd_block();
        return full_accs + has_tail_acc;
    }

    // Returns true when GEMV can use native `vdpbf16ps` instead of
    // bf16->f32 conversion + `vfmadd231ps`.
    bool gemv_use_vdpbf16ps() const {
        assert(is_gemv);
        return !transA && is_bf16;
    }

    // Returns true if GEMV uses a single scale value.
    bool gemv_single_wei_scale() const {
        assert(is_gemv);
        // Per-N scales collapse to a single scalar unless `y` is treated as a
        // row.
        return !is_per_n_wei_scales || !treat_y_as_row;
    }

    // Determines if the accumulator dt is integer.
    //
    // When the per-k scales or per-MN compensation are present, need to
    // accumulate in f32 due to precision loss.
    bool is_integer_acc() const {
        return is_int8 && !has_per_k_scales() && !with_per_mn_compensation;
    }

    // Determines if the DQ2PS conversion is needed for the accumulator.
    bool do_dq2ps_cvt() const {
        return is_integer_acc() && (alpha != 1.f || beta != 1.f);
    }

    // return 'true' when FP8 MAC is not natively supported by the CPU ISA
    bool is_fp8_via_convert() const {
        return is_fp8 && utils::one_of(isa_impl, avx10_1_512_amx_fp16, avx10_2);
    }

    bool is_fp8_via_convert_non_amx() const {
        return is_fp8_via_convert() && isa_impl == avx10_2;
    }

    // return 'true' when fp8 dst, bias, or binary post-op data are converted
    // by emulation because the CPU ISA has no native fp8 conversion
    bool is_fp8_post_ops_emu() const {
        using namespace data_type;
        if (is_superset(isa_impl, avx10_2)) return false;
        if (utils::one_of(f8_e5m2, dt_d, dt_bias)
                || utils::one_of(f8_e4m3, dt_d, dt_bias))
            return true;
        if (!with_binary) return false;
        const auto &post_ops = attr()->post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &entry = post_ops.entry_[i];
            if (entry.is_binary()
                    && utils::one_of(
                            entry.binary.src1_desc.data_type, f8_e5m2, f8_e4m3))
                return true;
        }
        return false;
    }

    bool is_input_convert() const { return is_bf32 || is_fp8_via_convert(); }

    bool is_row_major() const {
        assert(layout != brgemm_layout_undef);
        return layout == brgemm_row_major;
    }

    // Tile register decomposition
    // TMUL tails need their own tile because the tail geometry is baked into
    // the palette. ACE tiles are fixed-size, so a tail reuses the last tile.
    int get_bd_block2() const noexcept {
        return static_cast<int>((bdb <= bd_block2)
                        ? bdb
                        : (bd_block2 + ((bdb_tail && !is_ace()) ? 1 : 0)));
    }

    int get_ld_block2() const noexcept {
        return static_cast<int>((ldb <= ld_block2)
                        ? ldb
                        : (ld_block2 + ((ldb_tail && !is_ace()) ? 1 : 0)));
    }

    int get_num_C_tiles() const noexcept {
        return get_bd_block2() * get_ld_block2();
    }
    int get_C_tensor(int m, int n, bool m_tail = false,
            bool n_tail = false) const noexcept {
        auto M = m_tail ? get_bd_block2() - 1 : m;
        auto N = n_tail ? get_ld_block2() - 1 : n;
        return (M * get_ld_block2() + N);
    }

    int get_num_A_tiles() const noexcept {
        if (is_ace()) return 0;
        const auto req_tiles = (bdb_tail && bdb > 1) ? 2 : 1;
        const auto max_tiles = AMX_TILES_NUM - get_num_C_tiles() - 1;
        const auto n_tiles = nstl::min(get_bd_block2(), max_tiles);
        assert(n_tiles >= req_tiles);
        return nstl::max(req_tiles, n_tiles);
    }

    int get_A_tensor(int m, bool m_tail = false) const noexcept {
        const auto full_A_tiles = get_num_A_tiles() - (bdb_tail ? 1 : 0);
        auto M = (m_tail || full_A_tiles == 0) ? get_num_A_tiles() - 1
                                               : m % full_A_tiles;
        return (get_num_C_tiles() + M);
    }

    int get_num_B_tiles() const noexcept {
        if (is_ace()) return 0;
        const auto req_tiles = (ldb_tail && ldb > 1) ? 2 : 1;
        const auto max_tiles
                = AMX_TILES_NUM - get_num_C_tiles() - get_num_A_tiles();
        const auto n_tiles = nstl::min(get_ld_block2(), max_tiles);
        assert(n_tiles >= req_tiles);
        return nstl::max(req_tiles, n_tiles);
    }

    int get_B_tensor(int n, bool n_tail = false) const noexcept {
        const auto full_B_tiles = get_num_B_tiles() - (ldb_tail ? 1 : 0);
        auto N = (n_tail || full_B_tiles == 0) ? get_num_B_tiles() - 1
                                               : n % full_B_tiles;
        return (get_num_C_tiles() + get_num_A_tiles() + N);
    }

    bool can_reuse_input_transform() const noexcept {
        // FP8 conversion and transform caching share temporary workspace, so
        // caching would allow converted data to overwrite a saved transform.
        return !is_fp8_via_convert();
    }

    bool save_transform_A() const noexcept {
        return can_reuse_input_transform() && ldb2 > 1;
    }

    bool save_transform_B() const noexcept {
        // ACE never caches B: it loads B straight from memory with vmovups,
        // so no transform buffer is reserved for it.
        return !is_ace() && can_reuse_input_transform() && bdb2 > 1;
    }

    dim_t get_convert_wsp_buffer_size() const noexcept {
        if (!is_input_convert()) return 0;
        const dim_t n_bdb = bd_block2;
        const dim_t n_rdb = rdb + (rdb_tail != 0);
        const dim_t n_ldb = ldb + (ldb_tail != 0);
        const dim_t downcvt_tiles = brgattr.max_bs * n_rdb * (n_bdb + n_ldb);
        return downcvt_tiles * tilesize;
    }

    dim_t get_fused_copy_a_wsp_buffer_size() const noexcept {
        if (fused_copy_a)
            return brgattr.max_bs * bcast_dim
                    * dnnl::impl::utils::rnd_up(reduce_dim, max_rd_block())
                    * typesize_A;
        return 0;
    }

    int all_rdb() const noexcept {
        return static_cast<int>(rdb + (rdb_tail != 0));
    }

    int rd_block_A_size() const noexcept { return rd_block * typesize_A; }
    int rd_block_B_size() const noexcept { return rd_block * typesize_B; }

    // ACE A transform stores each bd_block in 4 ZMM registers.
    // It packs 16 rows into 4 ZMMs and transposes to one ZMM per rd_step.
    static constexpr int ace_zmms_per_bd_block = 4;
    int ace_transformed_A_bd_block_size() const noexcept {
        return ace_zmms_per_bd_block
                * static_cast<int>(cpu_isa_traits_t<avx512_core>::vlen);
    }
    int ace_transformed_A_bd_block2_size() const noexcept {
        return bd_block2 * ace_transformed_A_bd_block_size();
    }

    dim_t get_wsp_buffer_size() const noexcept {
        dim_t sz = 0;
        if (is_ace()) {
            // ACE transforms and caches A only; B is consumed directly from
            // its source matrix by the ZMM load path. No C tile area is
            // reserved either, ACE reads accumulators out with tilemovrow.
            if (save_transform_A())
                sz = static_cast<dim_t>(ace_transformed_A_bd_block2_size())
                        * brgattr.max_bs * all_rdb();
        } else if (is_tmm) {
            sz = get_num_C_tiles() * tilesize; // postops buffer
            sz += get_convert_wsp_buffer_size();
            if (amx_wary_k_tail()) sz += tilesize;
            sz += get_fused_copy_a_wsp_buffer_size();
        }
        return sz;
    }

    // A class version of the `static` version of the function.
    // Note: used in benchdnn only, not used inside the library.
    bool is_b_data_layout_vnni() const {
        return is_b_data_layout_vnni(dt_a, dt_b, brgattr.b_is_vnni, isa_impl);
    }

    // This function indicates when VNNI granularity packing is expected by the
    // kernel.
    //
    // Note: used as the `static` function in ukernel only, not anywhere else.
    //   `static`-ness is required to identify if the transform routine must be
    //   used for the ukernel to work properly. This information is critical
    //   because the transform routine accepts only 4 `ldb` values which affects
    //   ukernel creation. Otherwise, the user must create the ukernel object,
    //   query the packing info, and if it's required, likely re-create the
    //   object with a different `ldb` value, which may not work because
    //   creation stage for user's application may not provide all the info to
    //   create a ukernel object.
    // Note: for `bf32` (or brgattr.fpmath_mode_ == bf16) the function returns
    //   `true` because the data transformation to vnni layout is internal and
    //   transparent to the user.
    // Note: the library MUST NOT break the ability to provide this information
    //   without brgemm_desc_t object creation.
    static bool is_b_data_layout_vnni(data_type_t dt_a, data_type_t dt_b,
            bool attr_b_is_vnni, cpu_isa_t isa) {
        using namespace data_type;
        switch (dt_b) {
            case f32: return false;
            // Note: `dt_a == f32` means implicit up-conversion of B to f32.
            case f16:
                return dt_a != f32
                        && (is_f16_b_non_amx_vnni(dt_b, attr_b_is_vnni, isa)
                                || is_superset(isa, avx512_core_amx_fp16)
                                || is_superset(isa, avx2_vnni_2));
            // Note: `dt_a == f32` means implicit up-conversion of B to f32.
            case bf16: return dt_a != f32;
            default: return true;
        }
    }

    // This function indicates when the kernel would operate with the D pointer
    // (`true`) and when not (`false`). It's important to distinguish these two
    // cases due to the fact that kernel would ignore D pointer completely if
    // no post-accumulation work is identified.
    //
    // Correspondent decisions are done in `store_accumulators` function.
    // The function is used inside kernel generation and ukernel API.
    // TODO: extend usage to primitives (each of them utilize their own copy
    // of this definition).
    bool are_post_ops_applicable() const {
        const bool has_zero_points = !utils::everyone_is(
                brgemm_broadcast_t::none, zp_type_a, zp_type_b, zp_type_c);
        return dt_c != dt_d || with_eltwise || with_binary || with_bias
                || with_sum || req_s8s8_compensation || has_zero_points
                || with_src_scales || with_wei_scales || with_dst_scales
                || brgattr.use_intermediate_c_buffer;
    }

    bool can_dispatch_uker() const noexcept {
        using namespace utils;
        return is_tmm
                && one_of(type, brgemm_addr, brgemm_offs, brgemm_static_offs)
                && brgattr.use_uker
                && everyone_is(false, is_runtime_lda, is_runtime_ldb,
                        is_runtime_ldc, is_runtime_ldd);
    }

    bool is_ace() const noexcept {
        return brgattr.use_ace && is_superset(isa_impl, cpu_isa_t::avx10_2_ace);
    }

    bool is_xf16() const noexcept { return is_bf16 || is_f16; }

    bool is_f16_b_non_amx_vnni() const {
        return is_f16_b_non_amx_vnni(dt_b, brgattr.b_is_vnni, isa_impl);
    }

    // Note: `static` version appears because of `static is_b_data_layout_vnni`.
    static bool is_f16_b_non_amx_vnni(
            data_type_t dt_b, bool attr_b_is_vnni, cpu_isa_t isa) {
        // This function controls the code section which relies on
        // `avx512_core_fp16` instructions directly.
        return dt_b == data_type::f16 && attr_b_is_vnni
                && isa == avx512_core_fp16;
    }

    bool reduce_by_words() const {
        return is_bf16_tmm || is_f16_tmm || is_input_convert();
    }
    int max_rd_block() const { return reduce_by_words() ? 32 : 64; }
    int rd_block_step() const { return (reduce_by_words() && !is_fp8) ? 2 : 4; }

    bool amx_may_extend_k() const {
        return (is_superset(isa_impl, avx512_core_amx) && brgattr.extendable_k
                && (reduce_dim % data_type_vnni_granularity(dt_a)
                        || (reduce_dim > max_rd_block()
                                && reduce_dim % max_rd_block())));
    }
    bool amx_wary_k_tail() const {
        return amx_may_extend_k() && brgattr.wary_A_k_tail_read;
    }

    bool operator==(const brgemm_desc_t &rhs) const;
    bool operator<(const brgemm_desc_t &rhs) const;

private:
    primitive_attr_t *attr_ {nullptr};
    memory_desc_t *dst_md_ {nullptr};
    void set_attr_null() { attr_ = nullptr; }
    void set_dst_md_null() { dst_md_ = nullptr; }

    void cleanup_attr();
    void cleanup_dst_md();

    // The default assignment operator is intended to be used in custom copy
    // constructor only to avoid copying field-by-field
    brgemm_desc_t &operator=(const brgemm_desc_t &) = default;
};

struct brgemm_dynamic_values_t {
    dim_t dynamic_LDA = 0;
    dim_t dynamic_LDB = 0;
    dim_t dynamic_LDC = 0;
    dim_t dynamic_LDD = 0;
    brgemm_dynamic_values_t(dim_t LDA, dim_t LDB, dim_t LDC, dim_t LDD)
        : dynamic_LDA(LDA)
        , dynamic_LDB(LDB)
        , dynamic_LDC(LDC)
        , dynamic_LDD(LDD) {}
};

struct brgemm_kernel_params_t {
    const void *ptr_A = nullptr;
    const void *ptr_B = nullptr;
    const brgemm_batch_element_t *batch = nullptr;
    void *ptr_C = nullptr;

    const void *ptr_bias = nullptr;
    void *ptr_D = nullptr;

    const void *ptr_src_scales = nullptr;
    const void *ptr_wei_scales = nullptr;
    const void *ptr_dst_scales = nullptr;
    void *ptr_buf = nullptr;

    size_t do_post_ops = 0;
    size_t do_apply_comp = 0;
    size_t BS = 0;

    /*
     * ptr to table of void * elements that are pointers to post_op binary
     * src1 tensors
     */
    const void *post_ops_binary_rhs_arg_vec = nullptr;
    size_t oc_logical_off = 0;
    size_t first_mb_matrix_addr_off = 0;
    size_t dst_row_logical_off = 0;

    const char *data_C_ptr_ = nullptr;

    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *a_zp_values = nullptr;
    const void *c_zp_values = nullptr;
    const void *ptr_per_mn_compensation = nullptr;
    size_t skip_accm = 0;
    int32_t zp_a_val = 1;
    dim_t dynamic_LDA = 0;
    dim_t dynamic_LDB = 0;
    dim_t dynamic_LDC = 0;
    dim_t dynamic_LDD = 0;
};

// Base class for all BRGEMM kernels.
//
// A BRGEMM kernel is both the code generator that creates the kernel and the
// handle returned by the `brgemm_kernel_*()` functions in `brgemm.hpp`.
//
// Each kernel is created by its own `create_*()` factory, keeping the kernel
// definitions out of this header.
struct brgemm_kernel_t : public jit_generator_t {
    brgemm_kernel_t(const char *impl_name, cpu_isa_t isa_impl)
        : jit_generator_t(impl_name, isa_impl) {}
    ~brgemm_kernel_t() override = default;

    void operator()(const brgemm_kernel_params_t *params) const {
        jit_generator_t::operator()(params);
    }
};

/// @param bias Vector of bias (vector length is N)
/// @param scales - Vector of scale factor values which represents combination
///     scale factors for matrixes A and B. If brgemm_desc_t::is_per_n_scale = true
///     vector length is N otherwise it must be broadcasted to vector of simd
///     width length
/// @param binary_post_ops_rhs - Ptr to table of pointers to tensors used as rhs
///     in binary post-operation { void* binary_op_tensor1, ...,
///      void* binary_op_tensor_n}
/// @param oc_logical_off - Used in binary postops in per_oc bcast strategy.
///     Offset to start oc processed by given thread in elements.
/// @param dst_row_logical_off - Used in binary postops in per_oc bcast
///     strategy. Offset to start oc processed by given thread in elements.
/// @param a_zp_compensations - Pre-computed compensations for A matrix zero
///     point values.
/// @param b_zp_compensations - Pre-computed compensations for B matrix zero
///     point values.
/// @param c_zp_values - C matrix zero point values.
/// @param skip_accumulation - specifies whether to skip accumulation when
///    computing post-ops. `Beta` value from descriptor affects final
///    accumulator values taken.
/// @param do_only_comp - specifies whether to perform accumulation only and skip
///    post-ops.
/// @param do_only_zp_a_val - specifies to apply pre-calculated compensation for
///     A zero point only and skip the rest post-ops.
/// @param zp_a_val - zero point value for A, required to adjust compensation
///     values if do_only_zp_a_val = true.
/// @param dst_scales - Vector of inverted scale factor values for matix C,
///     common scale vector type only is supported, it must be broadcasted to
///     vector of simd width length.
///
struct brgemm_post_ops_data_t {
    brgemm_post_ops_data_t() = default;
    brgemm_post_ops_data_t(const void *bias, const void *binary_post_ops_rhs,
            size_t oc_logical_off, const size_t dst_row_logical_off = 0,
            const char *data_C_ptr_ = nullptr,
            const size_t first_mb_matrix_addr_off = 0,
            const void *a_zp_compensations = nullptr,
            const void *b_zp_compensations = nullptr,
            const void *c_zp_values = nullptr, bool skip_accumulation = false,
            int32_t zp_a_val = 1, bool do_only_comp = false,
            bool do_only_zp_a_val = false, const void *src_scales = nullptr,
            const void *wei_scales = nullptr, const void *dst_scales = nullptr,
            const void *a_zp_values = nullptr,
            const void *per_mn_compensation = nullptr)
        : bias(bias)
        , binary_post_ops_rhs(binary_post_ops_rhs)
        , oc_logical_off(oc_logical_off)
        , dst_row_logical_off(dst_row_logical_off)
        , data_C_ptr_(data_C_ptr_)
        , first_mb_matrix_addr_off(first_mb_matrix_addr_off)
        , a_zp_compensations(a_zp_compensations)
        , b_zp_compensations(b_zp_compensations)
        , c_zp_values(c_zp_values)
        , skip_accumulation(skip_accumulation)
        , zp_a_val {zp_a_val}
        , do_only_comp {do_only_comp}
        , do_only_zp_a_val {do_only_zp_a_val}
        , src_scales(src_scales)
        , wei_scales(wei_scales)
        , dst_scales(dst_scales)
        , a_zp_values(a_zp_values)
        , per_mn_compensation(per_mn_compensation) {}

    const void *bias = nullptr;
    const void *binary_post_ops_rhs = nullptr;
    size_t oc_logical_off = 0;
    size_t dst_row_logical_off = 0;
    const char *data_C_ptr_ = nullptr;
    size_t first_mb_matrix_addr_off = 0;
    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    const bool skip_accumulation = false;
    int32_t zp_a_val = 1;
    const bool do_only_comp = false;
    const bool do_only_zp_a_val = false;
    const void *src_scales = nullptr;
    const void *wei_scales = nullptr;
    const void *dst_scales = nullptr;
    const void *a_zp_values = nullptr;
    const void *per_mn_compensation = nullptr;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
