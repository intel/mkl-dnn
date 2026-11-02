.. index:: pair: page; BRGeMM ukernel example
.. _doxid-cpu_brgemm_example_cpp:

BRGeMM ukernel example
======================

This C++ API example demonstrates how to create and execute a BRGeMM ukernel.

This C++ API example demonstrates how to create and execute a BRGeMM ukernel.

.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2024-2025 Intel Corporation
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
	
	
	#include <algorithm>
	#include <cmath>
	#include <iostream>
	#include <string>
	#include <utility>
	#include <vector>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl_ukernel.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	using namespace :ref:`dnnl::ukernel <doxid-namespacednnl_1_1ukernel>`;
	
	void brgemm_example() {
	
	    // Create execution dnnl::engine. Needed for reorders to operate over input
	    // data.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0);
	
	    // Create dnnl::stream. Needed for reorders for the same reason.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // ukernel dimensions.
	    // K is for a whole tensor, K_blk is for a single ukernel.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` M = 8, K = 128, K_blk = 64, N = 48;
	    if (K % K_blk != 0) {
	        printf("K_blk must divide K.\n");
	        return;
	    }
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` n_calls = K / K_blk;
	
	    :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a_dt = :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`;
	    :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b_dt = :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`;
	    :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` c_dt = :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`; // Accumulator data type.
	    :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` d_dt = :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`; // Output data type.
	
	    // Query the packing requirement from the ukernel. It's enough to query
	    // packing requirements once for multiple ukernel objects.
	    const auto pack = :ref:`brgemm::get_B_pack_type <doxid-structdnnl_1_1ukernel_1_1brgemm_1aa2e3310da8dd905e3f7ee81060f1ad19>`(a_dt, b_dt);
	
	    // If the value is `pack_type::undef`, ukernel API is not supported on the
	    // target system.
	    if (pack == :ref:`pack_type::undef <doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54af31ee5e3824f1f5e5d206bdf3029f22b>`) {
	        printf("Kernel is not supported on this platform.\n");
	        return;
	    }
	
	    // Packing is required if the returned value is different from
	    // `pack_type::no_pack`.
	    // If packing is required, specific `ldb` value can be used ahead, since
	    // transform has a limited set of supported values.
	    bool need_pack = pack != :ref:`pack_type::no_trans <doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a76659c0424cb9f2555bc14e7d947db13>`;
	
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` lda = K;
	    // `ldb` for `need_pack = true` must be one of 16, 32, 48, or 64. This
	    // example doesn't explore options for dividing N into blocks which would
	    // likely happen for N > 64.
	    // const memory::dim ldb = need_pack ? N_block : N;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` ldb = N;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` ldc = N; // Leading dimension for accumulator.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` ldd = N; // Leading dimension for an actual output.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` batch_size = n_calls - 1;
	
	    // A, B, and C tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` A_dims = {M, K};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` B_dims = {K, N};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` C_dims = {M, N};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` D_dims = {M, N};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` binary_add_dims = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` B_scales_dims = {1, N};
	
	    // Allocate buffers with user data.
	    std::vector<float> A_user_data(product(A_dims));
	    std::vector<float> B_user_data(product(B_dims));
	    std::vector<float> binary_add_user_data(product(binary_add_dims));
	    std::vector<float> B_scales_user_data(product(B_scales_dims));
	    std::vector<float> D_data(product(D_dims)); // For reference comparison
	    std::vector<float> D_user_data(product(D_dims)); // For reference comparison
	
	    // Initialize A.
	    std::generate(A_user_data.begin(), A_user_data.end(), []() {
	        static int i = 0;
	        return i++ % 4;
	    });
	    // Initialize B.
	    std::generate(B_user_data.begin(), B_user_data.end(), []() {
	        static int i = 6;
	        static int sign_gen = 0;
	        int sign = (sign_gen++ % 2) ? -1 : 1;
	        float val = sign * (i++ % 5);
	        return val;
	    });
	    // Initialize binary_add.
	    std::generate(
	            binary_add_user_data.begin(), binary_add_user_data.end(), []() {
	                static int i = 3;
	                return i++ % 6;
	            });
	    // Initialize B scales.
	    std::generate(B_scales_user_data.begin(), B_scales_user_data.end(), []() {
	        static int i = 4;
	        return (float)(i++ % 16) / 8.f;
	    });
	
	    // Create f32 memories. They are used as data holders and reorder into
	    // memories passed to the ukernel.
	    auto A_f32_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            A_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto B_f32_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            B_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto binary_add_f32_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            binary_add_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto B_scales_f32_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            B_scales_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto D_f32_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            D_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	
	    auto A_f32_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(A_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, A_user_data.data());
	    auto B_f32_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(B_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, B_user_data.data());
	    auto binary_add_f32_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(binary_add_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, binary_add_user_data.data());
	    auto B_scales_f32_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(B_scales_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, B_scales_user_data.data());
	    auto D_f32_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(D_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, D_user_data.data());
	
	    // Create ukernel memories in requested data types.
	    // Note that all formats are `ab`.
	    auto A_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(A_dims, a_dt, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto B_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(B_dims, b_dt, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto binary_add_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            binary_add_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto B_scales_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            B_scales_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto C_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(C_dims, c_dt, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto D_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(D_dims, d_dt, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	
	    auto A_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(A_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto B_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(B_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto binary_add_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(binary_add_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto B_scales_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(B_scales_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto C_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(C_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto D_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(D_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    const auto *A_ptr = reinterpret_cast<uint8_t *>(A_mem.get_data_handle());
	    auto *B_ptr = reinterpret_cast<uint8_t *>(B_mem.get_data_handle());
	
	    const size_t a_dt_size
	            = :ref:`memory::data_type_size <doxid-structdnnl_1_1memory_1ac4064e92cc225fbb6a0431b90004511c>`(A_mem.get_desc().get_data_type());
	    const size_t b_dt_size
	            = :ref:`memory::data_type_size <doxid-structdnnl_1_1memory_1ac4064e92cc225fbb6a0431b90004511c>`(B_mem.get_desc().get_data_type());
	
	    // Reorder user data into buffers passed to ukernels in target data types.
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(A_f32_mem, A_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, A_f32_mem, A_mem);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(B_f32_mem, B_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, B_f32_mem, B_mem);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(binary_add_f32_mem, binary_add_mem)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, binary_add_f32_mem, binary_add_mem);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(B_scales_f32_mem, B_scales_mem)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, B_scales_f32_mem, B_scales_mem);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(D_f32_mem, D_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, D_f32_mem, D_mem);
	    // Prepare C buffer. Needed to use a single ukernel in the example with
	    // `set_add_C(true)`.
	    // Note: to avoid this step, the first ukernel should run
	    // `set_add_C(false)`, and it will initialize C buffer with intermediate
	    // values.
	    float *C_ptr = reinterpret_cast<float *>(C_mem.get_data_handle());
	    for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` i = 0; i < M * N; i++) {
	        C_ptr[i] = 0;
	    }
	
	    // Create ukernel post-ops (ReLU + Add).
	    // It reuses `primitive_attr` abstraction.
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` brgemm_ops;
	    brgemm_ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	            :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, /* alpha = */ 0.f, /* beta = */ 0.f);
	    brgemm_ops.:ref:`append_binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`(:ref:`algorithm::binary_add <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2c3faf084cf82b5603946995f637b35>`, binary_add_md);
	
	    // Create BRGeMM ukernel objects.
	    // There are two objects:
	    // * `brg` is the basic one which operates over K dimension divided into
	    //   blocks. It utilizes `set_add_C(true)` to accumulate into the same
	    //   buffer. It also uses `batch_size` to process as much as the number of
	    //   blocks over K minus one.
	    // * `brg_po` is the ukernel that would be called the last in the chain
	    //   since it has attributes attached to the object and those will execute
	    //   after all accumulation over K dimension is done.
	    :ref:`brgemm <doxid-structdnnl_1_1ukernel_1_1brgemm>` brg, brg_po;
	    if (batch_size > 0) {
	        // Construct a basic brgemm object.
	        // `allow_empty` makes the interface to return an empty `brg` object
	        // in case of the critical error.
	        brg = :ref:`brgemm <doxid-structdnnl_1_1ukernel_1_1brgemm>`(M, N, K_blk, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt,
	                /* allow_empty = */ true);
	        if (!brg) {
	            printf("Error: brg object was not constructed.\n");
	            return;
	        }
	
	        // Instruct the ukernel to append the result to the C tensor.
	        brg.:ref:`set_add_C <doxid-structdnnl_1_1ukernel_1_1brgemm_1a4546a4aad9b1e3769ce1b5c51b7f746c>`(true);
	
	        // Finalize the initialization.
	        // Successful completion returns `true`. Otherwise, `brg` object can't
	        // be used due to lack of support or non-compatible settings. The
	        // specific reason may be found by using `ONEDNN_VERBOSE=all` env var.
	        const bool ok = brg.:ref:`finalize <doxid-structdnnl_1_1ukernel_1_1brgemm_1a5e42fe0936faab76c675edb97860de0e>`();
	        if (!ok) {
	            printf("Kernel is not supported on this platform.\n");
	            return;
	        }
	
	        // Generate the executable code.
	        brg.:ref:`generate <doxid-structdnnl_1_1ukernel_1_1brgemm_1ae7c33dba7d829ced8d6b2de161159f69>`();
	    }
	
	    // Construct a brgemm object with post-ops.
	    brg_po = :ref:`brgemm <doxid-structdnnl_1_1ukernel_1_1brgemm>`(M, N, K_blk, 1, lda, ldb, ldc, a_dt, b_dt, c_dt,
	            /* allow_empty = */ true);
	    if (!brg_po) {
	        printf("Error: brg_po object was not constructed.\n");
	        return;
	    }
	
	    // Instruct the kernel to append the result to the C tensor computed by
	    // `brg` ukernel.
	    brg_po.:ref:`set_add_C <doxid-structdnnl_1_1ukernel_1_1brgemm_1a4546a4aad9b1e3769ce1b5c51b7f746c>`(true);
	    // Specify post-ops.
	    brg_po.:ref:`set_post_ops <doxid-structdnnl_1_1ukernel_1_1brgemm_1a99c44446d24cb50e8c1c20c11c4d7e4e>`(ldd, d_dt, brgemm_ops);
	    // Specify quantization scales for B.
	    if (b_dt == :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>` || b_dt == :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`) {
	        brg_po.:ref:`set_B_scales <doxid-structdnnl_1_1ukernel_1_1brgemm_1a0b17ef5afc621818865d6e41dba66ccc>`(/* mask = */ 2);
	    }
	
	    // Finalize the initialization.
	    const bool ok = brg_po.:ref:`finalize <doxid-structdnnl_1_1ukernel_1_1brgemm_1a5e42fe0936faab76c675edb97860de0e>`();
	    if (!ok) {
	        printf("Kernel is not supported on this platform.\n");
	        return;
	    }
	
	    // Generate the executable code.
	    brg_po.:ref:`generate <doxid-structdnnl_1_1ukernel_1_1brgemm_1ae7c33dba7d829ced8d6b2de161159f69>`();
	
	    // Query a scratchpad size and initialize a scratchpad buffer if the ukernel
	    // is expecting it. This is a service space needed, has nothing in common
	    // with accumulation buffer.
	    size_t scratchpad_size = brg_po.:ref:`get_scratchpad_size <doxid-structdnnl_1_1ukernel_1_1brgemm_1ada0b6984b8b9253cba9756c680c07d16>`();
	    std::vector<uint8_t> scratchpad(scratchpad_size);
	
	    uint8_t *B_blocked = nullptr;
	    void *B_base_ptr = B_ptr;
	    size_t blocked_B_size = 0;
	
	    // If packing is needed, create a dedicated object for data transformation.
	    if (need_pack) {
	        // Transform kernel for tensor B. The ukernel expects B passed in a
	        // special VNNI format for low precision data types, e.g., bfloat16_t
	        // or int8.
	        // Note: the routine doesn't provide a `batch_size` argument in the
	        // constructor as it can be either incorporated into `K` dimension, or
	        // manually iterated over in a for-loop on the user side.
	        :ref:`transform <doxid-structdnnl_1_1ukernel_1_1transform>` pack_B(/* K = */ K_blk * n_calls, /* N = */ N,
	                /* in_pack_type = */ :ref:`pack_type::no_trans <doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a76659c0424cb9f2555bc14e7d947db13>`, /* in_ld = */ N,
	                /* out_ld = */ ldb, /* in_dt = */ b_dt, /* out_dt = */ b_dt);
	
	        // Size of the packed tensor.
	        blocked_B_size = ldb * K_blk * :ref:`memory::data_type_size <doxid-structdnnl_1_1memory_1ac4064e92cc225fbb6a0431b90004511c>`(b_dt);
	
	        B_blocked = new uint8_t[blocked_B_size * n_calls];
	        B_base_ptr = B_blocked;
	
	        // Generate the executable code.
	        pack_B.generate();
	
	        // Pack B routine execution.
	        // Note: usually should be split to process only a part of B that the
	        // ukernel will execute.
	        pack_B.execute(B_ptr, B_blocked);
	    }
	
	    // ukernel execution section.
	    //
	    // Prepare buffers for execution.
	    std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(batch_size);
	    for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` i = 0; i < batch_size; i++) {
	        const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` A_offset_i = i * K_blk * a_dt_size;
	        const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` B_offset_i
	                = need_pack ? i * blocked_B_size : i * N * K_blk * b_dt_size;
	        A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
	    }
	
	    if (brg) {
	        // A call to initialize hardware features. For example, prepare AMX
	        // unit.
	        brg.:ref:`set_hw_context <doxid-structdnnl_1_1ukernel_1_1brgemm_1ac273853c939803d7c0f20fe1b8c41f48>`();
	
	        // An execute call. `A_B_offsets` is a vector of pairs of offsets to A
	        // and packed B tensors. `C_ptr` is a pointer to an accumulator buffer.
	        brg.:ref:`execute <doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b>`(A_ptr, B_base_ptr, A_B_offsets, C_ptr, scratchpad.data());
	    }
	
	    // Same set of operations for a ukernel with post-ops.
	    std::vector<std::pair<memory::dim, memory::dim>> A_B_po_offsets;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` A_offset_po = batch_size * K_blk * a_dt_size;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` B_offset_po = need_pack
	            ? batch_size * blocked_B_size
	            : batch_size * N * K_blk * b_dt_size;
	    A_B_po_offsets.emplace_back(A_offset_po, B_offset_po);
	
	    // This object also requires this call since ukernel with post-ops may
	    // require differently initialized internals underneath. If basic ukernel
	    // was used and they share the same internals, this call will be optimized.
	    brg_po.:ref:`set_hw_context <doxid-structdnnl_1_1ukernel_1_1brgemm_1ac273853c939803d7c0f20fe1b8c41f48>`();
	
	    // Prepare post-ops arguments and put them in a vector to make sure pointers
	    // are sitting side by side.
	    std::vector<const void *> bin_po_ptrs;
	    bin_po_ptrs.push_back(binary_add_mem.get_data_handle());
	
	    // Setting post-ops arguments into an attributes arguments storage.
	    :ref:`attr_params <doxid-structdnnl_1_1ukernel_1_1attr__params>` params;
	    params.:ref:`set_post_ops_args <doxid-structdnnl_1_1ukernel_1_1attr__params_1af991f15932b7c0fef737cdc61dd56de0>`(bin_po_ptrs.data());
	    params.:ref:`set_B_scales <doxid-structdnnl_1_1ukernel_1_1attr__params_1a9e2c17ea304a349479bc36124b08e200>`(B_scales_mem.get_data_handle());
	
	    // An execute call. The difference here is when post operations are
	    // requested, an additional D tensor pointer to store final output result
	    // after finishing accumulation and post-ops application is required.
	    // Additionally, a special `params` object with post operations handles
	    // is required.
	    //
	    // If post operations are not defined, the call is invalid, and a special
	    // API checks its validity.
	    if (brg_po.:ref:`is_execute_postops_valid <doxid-structdnnl_1_1ukernel_1_1brgemm_1a2636a460ecb30c8c9535d8c18858c1ef>`()) {
	        brg_po.:ref:`execute <doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b>`(A_ptr, B_base_ptr, A_B_po_offsets, C_ptr,
	                D_mem.get_data_handle(), scratchpad.data(), params);
	    } else {
	        brg_po.:ref:`execute <doxid-structdnnl_1_1ukernel_1_1brgemm_1a89e2b117573de5ac4be161c7294af55b>`(
	                A_ptr, B_base_ptr, A_B_po_offsets, C_ptr, scratchpad.data());
	    }
	
	    // Once all computations are done and there are no more calls to ukernels
	    // until they delegate control to the application, need to release the
	    // hardware context.
	    :ref:`brgemm::release_hw_context <doxid-structdnnl_1_1ukernel_1_1brgemm_1a4cdc1e8b77991a2da8a69ae5f4ce267a>`();
	
	    // Clean up an extra buffer.
	    delete B_blocked;
	
	    // Used for verification results, need unconditional reorder.
	    auto user_D_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(D_f32_md, :ref:`engine <doxid-structdnnl_1_1engine>`, D_data.data());
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(D_mem, user_D_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, D_mem, user_D_mem);
	
	    // Skip the check by default as data filling doesn't help with proper
	    // verification of the result. Negative result doesn't necessarily mean
	    // the functionality is broken. This is just a general sanity check.
	    if (true) return;
	
	    // A simplified fast verification that ukernel returned expected results.
	    // Note: potential off-by-1 or 2 errors may pop up. This could be solved
	    // with more sparse filling.
	    bool to_throw = false;
	    for (int m = 0; m < M; m++) {
	        for (int n = 0; n < N; n++) {
	            D_user_data[m * N + n] = 0;
	            for (int k = 0; k < K; k++) {
	                D_user_data[m * N + n]
	                        += A_user_data[m * K + k] * B_user_data[k * N + n];
	            }
	            // B scales ref
	            D_user_data[m * N + n] *= B_scales_user_data[n];
	            // Relu post-op ref
	            D_user_data[m * N + n] = std::max(D_user_data[m * N + n], 0.f);
	            // Binary post-op ref
	            D_user_data[m * N + n] += binary_add_user_data[0];
	
	            const float diff
	                    = fabsf(D_user_data[m * N + n] - D_data[m * N + n]);
	            if (diff > 1.19e-7) {
	                to_throw = true;
	                if (true) {
	                    printf("Error: [%3d:%3d] Ref:%12g Got:%12g Diff:%12g\n", m,
	                            n, D_user_data[m * N + n], D_data[m * N + n], diff);
	                }
	            }
	        }
	    }
	    if (to_throw) { throw :ref:`status::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>`; }
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`}, brgemm_example);
	}

