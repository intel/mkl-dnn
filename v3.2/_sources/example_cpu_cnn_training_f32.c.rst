.. index:: pair: example; cpu_cnn_training_f32.c
.. _doxid-cpu_cnn_training_f32_8c-example:

cpu_cnn_training_f32.c
======================

This C API example demonstrates how to build an AlexNet model training.

This C API example demonstrates how to build an AlexNet model training.

.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2016-2022 Intel Corporation
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
	
	
	
	// Required for posix_memalign
	#define _POSIX_C_SOURCE 200112L
	
	#include <stdio.h>
	#include <stdlib.h>
	#include <string.h>
	
	#include "oneapi/dnnl/dnnl.h"
	
	#include "example_utils.h"
	
	#define BATCH 8
	#define IC 3
	#define OC 96
	#define CONV_IH 227
	#define CONV_IW 227
	#define CONV_OH 55
	#define CONV_OW 55
	#define CONV_STRIDE 4
	#define CONV_PAD 0
	#define POOL_OH 27
	#define POOL_OW 27
	#define POOL_STRIDE 2
	#define POOL_PAD 0
	
	static size_t product(:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` *arr, size_t size) {
	    size_t prod = 1;
	    for (size_t i = 0; i < size; ++i)
	        prod *= arr[i];
	    return prod;
	}
	
	static void init_net_data(float *data, uint32_t dim, const :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` *dims) {
	    if (dim == 1) {
	        for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` i = 0; i < :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[0]; ++i) {
	            data[i] = (float)(i % 1637);
	        }
	    } else if (dim == 4) {
	        for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in = 0; in < :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[0]; ++in)
	            for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ic = 0; ic < :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[1]; ++ic)
	                for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ih = 0; ih < :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[2]; ++ih)
	                    for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` iw = 0; iw < :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[3]; ++iw) {
	                        :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` indx = in * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[1] * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[2] * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[3]
	                                + ic * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[2] * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[3] + ih * :ref:`dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>`[3] + iw;
	                        data[indx] = (float)(indx % 1637);
	                    }
	    }
	}
	
	typedef struct {
	    int nargs;
	    :ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` *args;
	} args_t;
	
	static void prepare_arg_node(args_t *node, int nargs) {
	    node->args = (:ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` *)malloc(sizeof(:ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>`) * nargs);
	    node->nargs = nargs;
	}
	static void free_arg_node(args_t *node) {
	    free(node->args);
	}
	
	static void set_arg(:ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` *arg, int arg_idx, :ref:`dnnl_memory_t <doxid-structdnnl__memory>` memory) {
	    arg->:ref:`arg <doxid-structdnnl__exec__arg__t_1a46c7f77870713b8af3fd37dc66e9690b>` = arg_idx;
	    arg->:ref:`memory <doxid-structdnnl__exec__arg__t_1a048f23e80b923636267c4dece912cd0d>` = memory;
	}
	
	static void init_data_memory(uint32_t dim, const :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` *dims,
	        :ref:`dnnl_format_tag_t <doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` user_tag, :ref:`dnnl_engine_t <doxid-structdnnl__engine>` engine, float *data,
	        :ref:`dnnl_memory_t <doxid-structdnnl__memory>` *memory) {
	    :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` user_md;
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	            &user_md, dim, dims, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, user_tag));
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(memory, user_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	    CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(user_md));
	    write_to_dnnl_memory(data, *memory);
	}
	
	:ref:`dnnl_status_t <doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` prepare_reorder(:ref:`dnnl_memory_t <doxid-structdnnl__memory>` *user_memory, // in
	        :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` prim_memory_md, // in
	        :ref:`dnnl_engine_t <doxid-structdnnl__engine>` prim_engine, // in: primitive's engine
	        int dir_is_user_to_prim, // in: user -> prim or prim -> user
	        :ref:`dnnl_memory_t <doxid-structdnnl__memory>` *prim_memory, // out: primitive's memory created
	        :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` *reorder, // out: reorder primitive created
	        uint32_t *net_index, // primitive index in net (inc if reorder created)
	        :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` *net, args_t *net_args) { // net params
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` user_memory_md;
	    :ref:`dnnl_memory_get_memory_desc <doxid-group__dnnl__api__memory_1ga82045853279cc76f52672b8172afdaee>`(*user_memory, &user_memory_md);
	
	    :ref:`dnnl_engine_t <doxid-structdnnl__engine>` user_mem_engine;
	    :ref:`dnnl_memory_get_engine <doxid-group__dnnl__api__memory_1ga583a4a06428de7d6c4251313e57ad814>`(*user_memory, &user_mem_engine);
	
	    if (!:ref:`dnnl_memory_desc_equal <doxid-group__dnnl__api__memory_1gad722c21c9af227ac7dc25c3ab649aae5>`(user_memory_md, prim_memory_md)) {
	        CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(prim_memory, prim_memory_md, prim_engine,
	                :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	        :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` reorder_pd;
	        if (dir_is_user_to_prim) {
	            CHECK(:ref:`dnnl_reorder_primitive_desc_create <doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(&reorder_pd,
	                    user_memory_md, user_mem_engine, prim_memory_md,
	                    prim_engine, NULL));
	        } else {
	            CHECK(:ref:`dnnl_reorder_primitive_desc_create <doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(&reorder_pd,
	                    prim_memory_md, prim_engine, user_memory_md,
	                    user_mem_engine, NULL));
	        }
	        CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(reorder, reorder_pd));
	        CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(reorder_pd));
	
	        net[*net_index] = *reorder;
	        prepare_arg_node(&net_args[*net_index], 2);
	        set_arg(&net_args[*net_index].args[0], :ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`,
	                dir_is_user_to_prim ? *user_memory : *prim_memory);
	        set_arg(&net_args[*net_index].args[1], :ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`,
	                dir_is_user_to_prim ? *prim_memory : *user_memory);
	        (*net_index)++;
	    } else {
	        *prim_memory = NULL;
	        *reorder = NULL;
	    }
	
	    return :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>`;
	}
	
	void simple_net() {
	    :ref:`dnnl_engine_t <doxid-structdnnl__engine>` :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`;
	    CHECK(:ref:`dnnl_engine_create <doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(&engine, :ref:`dnnl_cpu <doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2abde7b942413dd36f8285dd360fc0c797>`, 0)); // idx
	
	    // build a simple net
	    uint32_t n_fwd = 0, n_bwd = 0;
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` net_fwd[10], net_bwd[10];
	    args_t net_fwd_args[10], net_bwd_args[10];
	
	    const int ndims = 4;
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` net_dst_sizes = {BATCH, OC, POOL_OH, POOL_OW};
	
	    float *net_src
	            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
	    float *net_dst
	            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));
	
	    init_net_data(net_src, ndims, net_src_sizes);
	    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));
	
	    //----------------------------------------------------------------------
	    //----------------- Forward Stream -------------------------------------
	    // AlexNet: conv
	    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
	    // {BATCH, OC, CONV_OH, CONV_OW}
	    // strides: {CONV_STRIDE, CONV_STRIDE}
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_user_src_sizes;
	    for (int i = 0; i < ndims; i++)
	        conv_user_src_sizes[i] = net_src_sizes[i];
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_user_weights_sizes = {OC, IC, 11, 11};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_bias_sizes = {OC};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_user_dst_sizes = {BATCH, OC, CONV_OH, CONV_OW};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_strides = {CONV_STRIDE, CONV_STRIDE};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_dilation = {0, 0};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` conv_padding = {CONV_PAD, CONV_PAD};
	
	    float *conv_src = net_src;
	    float *conv_weights = (float *)malloc(
	            product(conv_user_weights_sizes, ndims) * sizeof(float));
	    float *conv_bias
	            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));
	
	    init_net_data(conv_weights, ndims, conv_user_weights_sizes);
	    init_net_data(conv_bias, 1, conv_bias_sizes);
	
	    // create memory for user data
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_user_src_memory, conv_user_weights_memory,
	            conv_user_bias_memory;
	    init_data_memory(ndims, conv_user_src_sizes, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`, engine, conv_src,
	            &conv_user_src_memory);
	    init_data_memory(ndims, conv_user_weights_sizes, :ref:`dnnl_oihw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11176ff202375dcd0d06e2fba5f8a8e0>`, engine,
	            conv_weights, &conv_user_weights_memory);
	    init_data_memory(1, conv_bias_sizes, :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>`, engine, conv_bias,
	            &conv_user_bias_memory);
	
	    // create a convolution
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` conv_pd;
	
	    {
	        // create data descriptors for convolution w/ no specified format
	        :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_src_md, conv_weights_md, conv_bias_md,
	                conv_dst_md;
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_src_md, ndims,
	                conv_user_src_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_weights_md, ndims,
	                conv_user_weights_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	                &conv_bias_md, 1, conv_bias_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_dst_md, ndims,
	                conv_user_dst_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	
	        CHECK(:ref:`dnnl_convolution_forward_primitive_desc_create <doxid-group__dnnl__api__convolution_1gab5d114c896caa5c32e0035eaafbd5f40>`(&conv_pd, engine,
	                :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_convolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8258635c519746dbf543ac13054acb5a>`, conv_src_md,
	                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
	                conv_dilation, conv_padding, conv_padding, NULL));
	
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_src_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_weights_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_bias_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_dst_md));
	    }
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_internal_src_memory, conv_internal_weights_memory,
	            conv_internal_dst_memory;
	
	    // create memory for dst data, we don't need to reorder it to user data
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_dst_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(&conv_internal_dst_memory, conv_dst_md, engine,
	            :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // create reorder primitives between user data and convolution srcs
	    // if required
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_reorder_src, conv_reorder_weights;
	
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_src_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_src_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5>`, 0);
	    CHECK(prepare_reorder(&conv_user_src_memory, conv_src_md, engine, 1,
	            &conv_internal_src_memory, &conv_reorder_src, &n_fwd, net_fwd,
	            net_fwd_args));
	
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_weights_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_weights_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a12ea0b4858b84889acab34e498323355>`, 0);
	    CHECK(prepare_reorder(&conv_user_weights_memory, conv_weights_md, engine, 1,
	            &conv_internal_weights_memory, &conv_reorder_weights, &n_fwd,
	            net_fwd, net_fwd_args));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_src_memory = conv_internal_src_memory
	            ? conv_internal_src_memory
	            : conv_user_src_memory;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_weights_memory = conv_internal_weights_memory
	            ? conv_internal_weights_memory
	            : conv_user_weights_memory;
	
	    // finally create a convolution primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&conv, conv_pd));
	    net_fwd[n_fwd] = conv;
	    prepare_arg_node(&net_fwd_args[n_fwd], 4);
	    set_arg(&net_fwd_args[n_fwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory);
	    set_arg(&net_fwd_args[n_fwd].args[1], :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`,
	            conv_weights_memory);
	    set_arg(&net_fwd_args[n_fwd].args[2], :ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv_user_bias_memory);
	    set_arg(&net_fwd_args[n_fwd].args[3], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`,
	            conv_internal_dst_memory);
	    n_fwd++;
	
	    // AlexNet: relu
	    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
	
	    float negative_slope = 0.0f;
	
	    // keep memory format of source same as the format of convolution
	    // output in order to avoid reorder
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_src_md = conv_dst_md;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_dst_md = relu_src_md;
	
	    // create a relu primitive descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` relu_pd;
	    CHECK(:ref:`dnnl_eltwise_forward_primitive_desc_create <doxid-group__dnnl__api__eltwise_1gaf5ae8472e1a364502103dea646ccb5bf>`(&relu_pd, engine,
	            :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, relu_src_md, relu_dst_md,
	            negative_slope, 0, NULL));
	
	    // create relu dst memory
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` relu_dst_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &relu_dst_memory, relu_dst_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create a relu primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` relu;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&relu, relu_pd));
	    net_fwd[n_fwd] = relu;
	    prepare_arg_node(&net_fwd_args[n_fwd], 2);
	    set_arg(&net_fwd_args[n_fwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`,
	            conv_internal_dst_memory);
	    set_arg(&net_fwd_args[n_fwd].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, relu_dst_memory);
	    n_fwd++;
	
	    // AlexNet: lrn
	    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
	    // local size: 5
	    // alpha: 0.0001
	    // beta: 0.75
	    // k: 1.0
	    uint32_t local_size = 5;
	    float alpha = 0.0001f;
	    float beta = 0.75f;
	    float k = 1.0f;
	
	    // create lrn src memory descriptor using dst memory descriptor
	    //  from previous primitive
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_src_md = relu_dst_md;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_dst_md = lrn_src_md;
	
	    // create a lrn primitive descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` lrn_pd;
	    CHECK(:ref:`dnnl_lrn_forward_primitive_desc_create <doxid-group__dnnl__api__lrn_1ga7d2550452cd5858747686b338cfde252>`(&lrn_pd, engine, :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`,
	            :ref:`dnnl_lrn_across_channels <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd>`, lrn_src_md, lrn_dst_md, local_size, alpha,
	            beta, k, NULL));
	
	    // create primitives for lrn dst and workspace memory
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` lrn_dst_memory, lrn_ws_memory;
	
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &lrn_dst_memory, lrn_dst_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // create workspace only in training and only for forward primitive
	    // query lrn_pd for workspace, this memory will be shared with forward lrn
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_ws_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(lrn_pd, :ref:`dnnl_query_workspace_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &lrn_ws_memory, lrn_ws_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create a lrn primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` lrn;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&lrn, lrn_pd));
	    net_fwd[n_fwd] = lrn;
	    prepare_arg_node(&net_fwd_args[n_fwd], 3);
	    set_arg(&net_fwd_args[n_fwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, relu_dst_memory);
	    set_arg(&net_fwd_args[n_fwd].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, lrn_dst_memory);
	    set_arg(&net_fwd_args[n_fwd].args[2], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, lrn_ws_memory);
	    n_fwd++;
	
	    // AlexNet: pool
	    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
	    // kernel: {3, 3}
	    // strides: {POOL_STRIDE, POOL_STRIDE}
	    // dilation: {0, 0}
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` pool_dst_sizes;
	    for (int i = 0; i < ndims; i++)
	        pool_dst_sizes[i] = net_dst_sizes[i];
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` pool_kernel = {3, 3};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` pool_strides = {POOL_STRIDE, POOL_STRIDE};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` pool_padding = {POOL_PAD, POOL_PAD};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` pool_dilation = {0, 0};
	
	    // create memory for user dst data
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_user_dst_memory;
	    init_data_memory(4, pool_dst_sizes, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`, engine, net_dst,
	            &pool_user_dst_memory);
	
	    // create a pooling primitive descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` pool_pd;
	
	    {
	        // create pooling src memory descriptor using dst descriptor
	        //  from previous primitive
	        :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_src_md = lrn_dst_md;
	
	        // create descriptors for dst pooling data
	        :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_dst_md;
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&pool_dst_md, 4, pool_dst_sizes,
	                :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	
	        CHECK(:ref:`dnnl_pooling_forward_primitive_desc_create <doxid-group__dnnl__api__pooling_1ga4921dcd2653e2046ef8de99c354957fe>`(&pool_pd, engine,
	                :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_pooling_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7>`, pool_src_md, pool_dst_md,
	                pool_strides, pool_kernel, pool_dilation, pool_padding,
	                pool_padding, NULL));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(pool_dst_md));
	    }
	
	    // create memory for workspace
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_ws_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_ws_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(pool_pd, :ref:`dnnl_query_workspace_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &pool_ws_memory, pool_ws_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // create reorder primitives between pooling dsts and user format dst
	    // if required
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool_reorder_dst;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_internal_dst_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_dst_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(pool_pd, :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`, 0);
	    n_fwd += 1; // tentative workaround: preserve space for pooling that should
	            // happen before the reorder
	    CHECK(prepare_reorder(&pool_user_dst_memory, pool_dst_md, engine, 0,
	            &pool_internal_dst_memory, &pool_reorder_dst, &n_fwd, net_fwd,
	            net_fwd_args));
	    n_fwd -= pool_reorder_dst ? 2 : 1;
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_dst_memory = pool_internal_dst_memory
	            ? pool_internal_dst_memory
	            : pool_user_dst_memory;
	
	    // finally create a pooling primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&pool, pool_pd));
	    net_fwd[n_fwd] = pool;
	    prepare_arg_node(&net_fwd_args[n_fwd], 3);
	    set_arg(&net_fwd_args[n_fwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn_dst_memory);
	    set_arg(&net_fwd_args[n_fwd].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_memory);
	    set_arg(&net_fwd_args[n_fwd].args[2], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, pool_ws_memory);
	    n_fwd++;
	
	    if (pool_reorder_dst) n_fwd += 1;
	
	    //-----------------------------------------------------------------------
	    //----------------- Backward Stream -------------------------------------
	    //-----------------------------------------------------------------------
	
	    // ... user diff_data ...
	    float *net_diff_dst
	            = (float *)malloc(product(pool_dst_sizes, 4) * sizeof(float));
	
	    init_net_data(net_diff_dst, 4, pool_dst_sizes);
	
	    // create memory for user diff dst data
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_user_diff_dst_memory;
	    init_data_memory(4, pool_dst_sizes, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`, engine, net_diff_dst,
	            &pool_user_diff_dst_memory);
	
	    // Pooling Backward
	    // pooling diff src memory descriptor
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_diff_src_md = lrn_dst_md;
	
	    // pooling diff dst memory descriptor
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_diff_dst_md = pool_dst_md;
	
	    // backward primitive descriptor needs to hint forward descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` pool_bwd_pd;
	    CHECK(:ref:`dnnl_pooling_backward_primitive_desc_create <doxid-group__dnnl__api__pooling_1ga0f1637d5ab52a8b784e642d6afac9fec>`(&pool_bwd_pd, engine,
	            :ref:`dnnl_pooling_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7>`, pool_diff_src_md, pool_diff_dst_md, pool_strides,
	            pool_kernel, pool_dilation, pool_padding, pool_padding, pool_pd,
	            NULL));
	
	    // create reorder primitive between user diff dst and pool diff dst
	    // if required
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_diff_dst_memory, pool_internal_diff_dst_memory;
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool_reorder_diff_dst;
	    CHECK(prepare_reorder(&pool_user_diff_dst_memory, pool_diff_dst_md, engine,
	            1, &pool_internal_diff_dst_memory, &pool_reorder_diff_dst, &n_bwd,
	            net_bwd, net_bwd_args));
	
	    pool_diff_dst_memory = pool_internal_diff_dst_memory
	            ? pool_internal_diff_dst_memory
	            : pool_user_diff_dst_memory;
	
	    // create memory for pool diff src data
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_diff_src_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(&pool_diff_src_memory, pool_diff_src_md, engine,
	            :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create backward pooling primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool_bwd;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&pool_bwd, pool_bwd_pd));
	    net_bwd[n_bwd] = pool_bwd;
	    prepare_arg_node(&net_bwd_args[n_bwd], 3);
	    set_arg(&net_bwd_args[n_bwd].args[0], :ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`,
	            pool_diff_dst_memory);
	    set_arg(&net_bwd_args[n_bwd].args[1], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, pool_ws_memory);
	    set_arg(&net_bwd_args[n_bwd].args[2], :ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`,
	            pool_diff_src_memory);
	    n_bwd++;
	
	    // Backward lrn
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_diff_dst_md = pool_diff_src_md;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_diff_src_md = lrn_diff_dst_md;
	
	    // create backward lrn descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` lrn_bwd_pd;
	    CHECK(:ref:`dnnl_lrn_backward_primitive_desc_create <doxid-group__dnnl__api__lrn_1gafc38999581f962346f08517ef3383617>`(&lrn_bwd_pd, engine,
	            :ref:`dnnl_lrn_across_channels <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd>`, lrn_diff_src_md, lrn_diff_dst_md,
	            lrn_src_md, local_size, alpha, beta, k, lrn_pd, NULL));
	
	    // create memory for lrn diff src
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` lrn_diff_src_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(&lrn_diff_src_memory, lrn_diff_src_md, engine,
	            :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create backward lrn primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` lrn_bwd;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&lrn_bwd, lrn_bwd_pd));
	    net_bwd[n_bwd] = lrn_bwd;
	    prepare_arg_node(&net_bwd_args[n_bwd], 4);
	    set_arg(&net_bwd_args[n_bwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, relu_dst_memory);
	    set_arg(&net_bwd_args[n_bwd].args[1], :ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`,
	            pool_diff_src_memory);
	    set_arg(&net_bwd_args[n_bwd].args[2], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, lrn_ws_memory);
	    set_arg(&net_bwd_args[n_bwd].args[3], :ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`,
	            lrn_diff_src_memory);
	    n_bwd++;
	
	    // Backward relu
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_diff_src_md = lrn_diff_src_md;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_diff_dst_md = lrn_diff_src_md;
	
	    // create backward relu descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` relu_bwd_pd;
	    CHECK(:ref:`dnnl_eltwise_backward_primitive_desc_create <doxid-group__dnnl__api__eltwise_1gaba11ca62016a1c23d997db47bcd6c27d>`(&relu_bwd_pd, engine,
	            :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, relu_diff_src_md, relu_diff_dst_md, relu_src_md,
	            negative_slope, 0, relu_pd, NULL));
	
	    // create memory for relu diff src
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` relu_diff_src_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(&relu_diff_src_memory, relu_diff_src_md, engine,
	            :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create backward relu primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` relu_bwd;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&relu_bwd, relu_bwd_pd));
	    net_bwd[n_bwd] = relu_bwd;
	    prepare_arg_node(&net_bwd_args[n_bwd], 3);
	    set_arg(&net_bwd_args[n_bwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`,
	            conv_internal_dst_memory);
	    set_arg(&net_bwd_args[n_bwd].args[1], :ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`,
	            lrn_diff_src_memory);
	    set_arg(&net_bwd_args[n_bwd].args[2], :ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`,
	            relu_diff_src_memory);
	    n_bwd++;
	
	    // Backward convolution with respect to weights
	    float *conv_diff_bias_buffer
	            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));
	    float *conv_user_diff_weights_buffer = (float *)malloc(
	            product(conv_user_weights_sizes, 4) * sizeof(float));
	
	    // initialize memory for diff weights in user format
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_user_diff_weights_memory;
	    init_data_memory(4, conv_user_weights_sizes, :ref:`dnnl_oihw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da11176ff202375dcd0d06e2fba5f8a8e0>`, engine,
	            conv_user_diff_weights_buffer, &conv_user_diff_weights_memory);
	
	    // create backward convolution primitive descriptor
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` conv_bwd_weights_pd;
	
	    {
	        // memory descriptors should be in format `any` to allow backward
	        // convolution for
	        // weights to chose the format it prefers for best performance
	        :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_diff_src_md, conv_diff_weights_md,
	                conv_diff_bias_md, conv_diff_dst_md;
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_diff_src_md, 4,
	                conv_user_src_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_diff_weights_md, 4,
	                conv_user_weights_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	                &conv_diff_bias_md, 1, conv_bias_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>`));
	        CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_diff_dst_md, 4,
	                conv_user_dst_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	
	        // create backward convolution descriptor
	        CHECK(:ref:`dnnl_convolution_backward_weights_primitive_desc_create <doxid-group__dnnl__api__convolution_1gadfb6988120ff24a0b62d9e8a7443ba09>`(
	                &conv_bwd_weights_pd, engine, :ref:`dnnl_convolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8258635c519746dbf543ac13054acb5a>`,
	                conv_diff_src_md, conv_diff_weights_md, conv_diff_bias_md,
	                conv_diff_dst_md, conv_strides, conv_dilation, conv_padding,
	                conv_padding, conv_pd, NULL));
	
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_diff_src_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_diff_weights_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_diff_bias_md));
	        CHECK(:ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_diff_dst_md));
	    }
	
	    // for best performance convolution backward might chose
	    // different memory format for src and diff_dst
	    // than the memory formats preferred by forward convolution
	    // for src and dst respectively
	    // create reorder primitives for src from forward convolution to the
	    // format chosen by backward convolution
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_bwd_reorder_src;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_bwd_internal_src_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_diff_src_md = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(
	            conv_bwd_weights_pd, :ref:`dnnl_query_src_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5>`, 0);
	    CHECK(prepare_reorder(&conv_src_memory, conv_diff_src_md, engine, 1,
	            &conv_bwd_internal_src_memory, &conv_bwd_reorder_src, &n_bwd,
	            net_bwd, net_bwd_args));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_bwd_weights_src_memory = conv_bwd_internal_src_memory
	            ? conv_bwd_internal_src_memory
	            : conv_src_memory;
	
	    // create reorder primitives for diff_dst between diff_src from relu_bwd
	    // and format preferred by conv_diff_weights
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_reorder_diff_dst;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_internal_diff_dst_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_diff_dst_md = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(
	            conv_bwd_weights_pd, :ref:`dnnl_query_diff_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae28e33688bf6c55edcf108bd24eb90de>`, 0);
	
	    CHECK(prepare_reorder(&relu_diff_src_memory, conv_diff_dst_md, engine, 1,
	            &conv_internal_diff_dst_memory, &conv_reorder_diff_dst, &n_bwd,
	            net_bwd, net_bwd_args));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_diff_dst_memory = conv_internal_diff_dst_memory
	            ? conv_internal_diff_dst_memory
	            : relu_diff_src_memory;
	
	    // create reorder primitives for conv diff weights memory
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_reorder_diff_weights;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_internal_diff_weights_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_diff_weights_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(
	                    conv_bwd_weights_pd, :ref:`dnnl_query_diff_weights_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8551246c3e70fa1e420411507dbdfe32>`, 0);
	    n_bwd += 1; // tentative workaround: preserve space for conv_bwd_weights
	            // that should happen before the reorder
	
	    CHECK(prepare_reorder(&conv_user_diff_weights_memory, conv_diff_weights_md,
	            engine, 0, &conv_internal_diff_weights_memory,
	            &conv_reorder_diff_weights, &n_bwd, net_bwd, net_bwd_args));
	    n_bwd -= conv_reorder_diff_weights ? 2 : 1;
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_diff_weights_memory = conv_internal_diff_weights_memory
	            ? conv_internal_diff_weights_memory
	            : conv_user_diff_weights_memory;
	
	    // create memory for diff bias memory
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_diff_bias_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_diff_bias_md = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(
	            conv_bwd_weights_pd, :ref:`dnnl_query_diff_weights_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8551246c3e70fa1e420411507dbdfe32>`, 1);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(&conv_diff_bias_memory, conv_diff_bias_md, engine,
	            :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	    CHECK(:ref:`dnnl_memory_set_data_handle <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>`(
	            conv_diff_bias_memory, conv_diff_bias_buffer));
	
	    // finally created backward convolution weights primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_bwd_weights;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&conv_bwd_weights, conv_bwd_weights_pd));
	    net_bwd[n_bwd] = conv_bwd_weights;
	    prepare_arg_node(&net_bwd_args[n_bwd], 4);
	    set_arg(&net_bwd_args[n_bwd].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`,
	            conv_bwd_weights_src_memory);
	    set_arg(&net_bwd_args[n_bwd].args[1], :ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`,
	            conv_diff_dst_memory);
	    set_arg(&net_bwd_args[n_bwd].args[2], :ref:`DNNL_ARG_DIFF_WEIGHTS <doxid-group__dnnl__api__primitives__common_1ga3324092ef421f77aebee83b0117cac60>`,
	            conv_diff_weights_memory);
	    set_arg(&net_bwd_args[n_bwd].args[3], :ref:`DNNL_ARG_DIFF_BIAS <doxid-group__dnnl__api__primitives__common_1ga1cd79979dda6df65ec45eef32a839901>`,
	            conv_diff_bias_memory);
	    n_bwd++;
	
	    if (conv_reorder_diff_weights) n_bwd += 1;
	
	    // output from backward stream
	    void *net_diff_weights = NULL;
	    void *net_diff_bias = NULL;
	
	    int n_iter = 10; // number of iterations for training.
	    :ref:`dnnl_stream_t <doxid-structdnnl__stream>` stream;
	    CHECK(:ref:`dnnl_stream_create <doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354>`(&stream, engine, :ref:`dnnl_stream_default_flags <doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547acf05c543bccebd58e6d4e0db7137fb92>`));
	    // Execute the net
	    for (int i = 0; i < n_iter; i++) {
	        for (uint32_t i = 0; i < n_fwd; ++i)
	            CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(net_fwd[i], stream,
	                    net_fwd_args[i].nargs, net_fwd_args[i].args));
	
	        // Update net_diff_dst
	        void *net_output = NULL; // output from forward stream:
	        CHECK(:ref:`dnnl_memory_get_data_handle <doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5>`(pool_user_dst_memory, &net_output));
	        // ...user updates net_diff_dst using net_output...
	        // some user defined func update_diff_dst(net_diff_dst, net_output)
	
	        // Backward pass
	        for (uint32_t i = 0; i < n_bwd; ++i)
	            CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(net_bwd[i], stream,
	                    net_bwd_args[i].nargs, net_bwd_args[i].args));
	
	        // ... update weights ...
	        CHECK(:ref:`dnnl_memory_get_data_handle <doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5>`(
	                conv_user_diff_weights_memory, &net_diff_weights));
	        CHECK(:ref:`dnnl_memory_get_data_handle <doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5>`(
	                conv_diff_bias_memory, &net_diff_bias));
	        // ...user updates weights and bias using diff weights and bias...
	        // some user defined func update_weights(conv_user_weights_memory,
	        // conv_bias_memory,
	        //      net_diff_weights, net_diff_bias);
	    }
	    CHECK(:ref:`dnnl_stream_wait <doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>`(stream));
	
	    :ref:`dnnl_stream_destroy <doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44>`(stream);
	
	    // clean up nets
	    for (uint32_t i = 0; i < n_fwd; ++i)
	        free_arg_node(&net_fwd_args[i]);
	    for (uint32_t i = 0; i < n_bwd; ++i)
	        free_arg_node(&net_bwd_args[i]);
	
	    // Cleanup forward
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(pool_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(lrn_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(relu_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(conv_pd));
	
	    free(net_src);
	    free(net_dst);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_user_src_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_user_weights_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_user_bias_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_internal_src_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_internal_weights_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_internal_dst_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_reorder_src);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_reorder_weights);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv);
	
	    free(conv_weights);
	    free(conv_bias);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(relu_dst_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(relu);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(lrn_ws_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(lrn_dst_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(lrn);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_user_dst_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_internal_dst_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_ws_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(pool_reorder_dst);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(pool);
	
	    // Cleanup backward
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(pool_bwd_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(lrn_bwd_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(relu_bwd_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(conv_bwd_weights_pd));
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_user_diff_dst_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_diff_src_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(pool_internal_diff_dst_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(pool_reorder_diff_dst);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(pool_bwd);
	
	    free(net_diff_dst);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(lrn_diff_src_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(lrn_bwd);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(relu_diff_src_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(relu_bwd);
	
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_user_diff_weights_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_diff_bias_memory);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_bwd_internal_src_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_bwd_reorder_src);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_internal_diff_dst_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_reorder_diff_dst);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(conv_internal_diff_weights_memory);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_reorder_diff_weights);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(conv_bwd_weights);
	
	    free(conv_diff_bias_buffer);
	    free(conv_user_diff_weights_buffer);
	
	    :ref:`dnnl_engine_destroy <doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(engine);
	}
	
	int main(int argc, char **argv) {
	    simple_net();
	    printf("Example passed on CPU.\n");
	    return 0;
	}
