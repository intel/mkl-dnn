.. index:: pair: page; CNN f32 inference example
.. _doxid-cnn_inference_f32_c:

CNN f32 inference example
=========================

This C API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.

This C API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.

Some key take-aways include:

* How tensors are implemented and submitted to primitives.

* How primitives are created.

* How primitives are sequentially submitted to the network, where the output from primitives is passed as input to the next primitive. The latter specifies a dependency between the primitive input and output data.

* Specific 'inference-only' configurations.

* Limiting the number of reorders performed that are detrimental to performance.

The example implements the AlexNet layers as numbered primitives (for example, conv1, pool1, conv2).

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
	        for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` i = 0; i < dims[0]; ++i) {
	            data[i] = (float)(i % 1637);
	        }
	    } else if (dim == 4) {
	        for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in = 0; in < dims[0]; ++in)
	            for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ic = 0; ic < dims[1]; ++ic)
	                for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ih = 0; ih < dims[2]; ++ih)
	                    for (:ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` iw = 0; iw < dims[3]; ++iw) {
	                        :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` indx = in * dims[1] * dims[2] * dims[3]
	                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
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
	
	void simple_net(:ref:`dnnl_engine_kind_t <doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind) {
	    :ref:`dnnl_engine_t <doxid-structdnnl__engine>` :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`;
	    CHECK(:ref:`dnnl_engine_create <doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(&engine, engine_kind, 0));
	
	    // build a simple net
	    uint32_t n = 0;
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` net[10];
	    args_t net_args[10];
	
	    const int ndims = 4;
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
	    :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` net_dst_sizes = {BATCH, OC, POOL_OH, POOL_OW};
	
	    float *net_src
	            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
	    float *net_dst
	            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));
	
	    init_net_data(net_src, ndims, net_src_sizes);
	    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));
	
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
	
	    // create data descriptors for convolution w/ no specified format
	
	    :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md;
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_src_md, ndims,
	            conv_user_src_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_weights_md, ndims,
	            conv_user_weights_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	            &conv_bias_md, 1, conv_bias_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_x <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9ccb37bb1a788f0245efbffbaf81e145>`));
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&conv_dst_md, ndims,
	            conv_user_dst_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	
	    // create a convolution
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` conv_pd;
	    CHECK(:ref:`dnnl_convolution_forward_primitive_desc_create <doxid-group__dnnl__api__convolution_1gab5d114c896caa5c32e0035eaafbd5f40>`(&conv_pd, engine,
	            :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_convolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8258635c519746dbf543ac13054acb5a>`, conv_src_md, conv_weights_md,
	            conv_bias_md, conv_dst_md, conv_strides, conv_dilation,
	            conv_padding, conv_padding, NULL));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_internal_src_memory, conv_internal_weights_memory,
	            conv_internal_dst_memory;
	
	    // create memory for dst data, we don't need reorder it to user data
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>`
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &conv_internal_dst_memory, dst_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // create reorder primitives between user data and convolution srcs
	    // if required
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv_reorder_src, conv_reorder_weights;
	
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>`
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_src_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a14a86faee7b85eeb60d0d7886756ffa5>`, 0);
	    CHECK(prepare_reorder(&conv_user_src_memory, src_md, engine, 1,
	            &conv_internal_src_memory, &conv_reorder_src, &n, net, net_args));
	
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` :ref:`weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>`
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_weights_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a12ea0b4858b84889acab34e498323355>`, 0);
	    CHECK(prepare_reorder(&conv_user_weights_memory, weights_md, engine, 1,
	            &conv_internal_weights_memory, &conv_reorder_weights, &n, net,
	            net_args));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_src_memory = conv_internal_src_memory
	            ? conv_internal_src_memory
	            : conv_user_src_memory;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` conv_weights_memory = conv_internal_weights_memory
	            ? conv_internal_weights_memory
	            : conv_user_weights_memory;
	
	    // finally create a convolution primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` conv;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&conv, conv_pd));
	    net[n] = conv;
	    prepare_arg_node(&net_args[n], 4);
	    set_arg(&net_args[n].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory);
	    set_arg(&net_args[n].args[1], :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_memory);
	    set_arg(&net_args[n].args[2], :ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv_user_bias_memory);
	    set_arg(&net_args[n].args[3], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_internal_dst_memory);
	    n++;
	
	    // AlexNet: relu
	    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
	    float negative_slope = 0.0f;
	
	    // create relu memory descriptor on dst memory descriptor
	    // from previous primitive
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_src_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(conv_pd, :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`, 0);
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` relu_dst_md = relu_src_md;
	
	    // create a relu
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` relu_pd;
	    CHECK(:ref:`dnnl_eltwise_forward_primitive_desc_create <doxid-group__dnnl__api__eltwise_1gaf5ae8472e1a364502103dea646ccb5bf>`(&relu_pd, engine,
	            :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, relu_src_md, relu_dst_md,
	            negative_slope, 0, NULL));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` relu_dst_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &relu_dst_memory, relu_dst_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create a relu primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` relu;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&relu, relu_pd));
	    net[n] = relu;
	    prepare_arg_node(&net_args[n], 2);
	    set_arg(&net_args[n].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_internal_dst_memory);
	    set_arg(&net_args[n].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, relu_dst_memory);
	    n++;
	
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
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` lrn_dst_memory;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &lrn_dst_memory, lrn_dst_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` lrn_ws_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` lrn_ws_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(lrn_pd, :ref:`dnnl_query_workspace_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &lrn_ws_memory, lrn_ws_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    // finally create a lrn primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` lrn;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&lrn, lrn_pd));
	    net[n] = lrn;
	    prepare_arg_node(&net_args[n], 3);
	    set_arg(&net_args[n].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, relu_dst_memory);
	    set_arg(&net_args[n].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, lrn_dst_memory);
	    set_arg(&net_args[n].args[2], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, lrn_ws_memory);
	    n++;
	
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
	
	    // create pooling memory descriptor on dst descriptor
	    //  from previous primitive
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_src_md = lrn_dst_md;
	
	    // create descriptors for dst pooling data
	    :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_dst_any_md;
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(&pool_dst_any_md, ndims,
	            pool_dst_sizes, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`));
	
	    // create memory for user data
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_user_dst_memory;
	    init_data_memory(ndims, pool_dst_sizes, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`, engine, net_dst,
	            &pool_user_dst_memory);
	
	    // create a pooling
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` pool_pd;
	    CHECK(:ref:`dnnl_pooling_forward_primitive_desc_create <doxid-group__dnnl__api__pooling_1ga4921dcd2653e2046ef8de99c354957fe>`(&pool_pd, engine,
	            :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_pooling_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acf3529ba1c4761c0da90eb6750def6c7>`, pool_src_md, pool_dst_any_md,
	            pool_strides, pool_kernel, pool_dilation, pool_padding,
	            pool_padding, NULL));
	
	    // create memory for workspace
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_ws_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_ws_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(pool_pd, :ref:`dnnl_query_workspace_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a1c465006660aabe46e644e6df7d36e8a>`, 0);
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &pool_ws_memory, pool_ws_md, engine, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_dst_memory;
	
	    // create reorder primitives between user data and pooling dsts
	    // if required
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool_reorder_dst;
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` pool_internal_dst_memory;
	    :ref:`const_dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` pool_dst_md
	            = :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>`(pool_pd, :ref:`dnnl_query_dst_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059add5c338ad7ae0c296509e54d22130598>`, 0);
	    n += 1; // tentative workaround: preserve space for pooling that should
	            // happen before the reorder
	    CHECK(prepare_reorder(&pool_user_dst_memory, pool_dst_md, engine, 0,
	            &pool_internal_dst_memory, &pool_reorder_dst, &n, net, net_args));
	    n -= pool_reorder_dst ? 2 : 1;
	
	    pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory
	                                               : pool_user_dst_memory;
	
	    // finally create a pooling primitive
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` pool;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&pool, pool_pd));
	    net[n] = pool;
	    prepare_arg_node(&net_args[n], 3);
	    set_arg(&net_args[n].args[0], :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn_dst_memory);
	    set_arg(&net_args[n].args[1], :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_memory);
	    set_arg(&net_args[n].args[2], :ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, pool_ws_memory);
	    n++;
	
	    if (pool_reorder_dst) n += 1;
	
	    :ref:`dnnl_stream_t <doxid-structdnnl__stream>` stream;
	    CHECK(:ref:`dnnl_stream_create <doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354>`(&stream, engine, :ref:`dnnl_stream_default_flags <doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547acf05c543bccebd58e6d4e0db7137fb92>`));
	    for (uint32_t i = 0; i < n; ++i) {
	        CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(
	                net[i], stream, net_args[i].nargs, net_args[i].args));
	    }
	
	    CHECK(:ref:`dnnl_stream_wait <doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>`(stream));
	
	    // clean-up
	    for (uint32_t i = 0; i < n; ++i)
	        free_arg_node(&net_args[i]);
	
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(conv_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(relu_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(lrn_pd));
	    CHECK(:ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(pool_pd));
	
	    :ref:`dnnl_stream_destroy <doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44>`(stream);
	
	    free(net_src);
	    free(net_dst);
	
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_src_md);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_weights_md);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_bias_md);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(conv_dst_md);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(pool_dst_any_md);
	
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
	
	    :ref:`dnnl_engine_destroy <doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(engine);
	}
	
	int main(int argc, char **argv) {
	    :ref:`dnnl_engine_kind_t <doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind = parse_engine_kind(argc, argv);
	    simple_net(engine_kind);
	    printf("Example passed on %s.\n", engine_kind2str_upper(engine_kind));
	    return 0;
	}

