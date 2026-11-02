.. index:: pair: page; Reorder between CPU and GPU engines
.. _doxid-cross_engine_reorder_c:

Reorder between CPU and GPU engines
===================================

This C API example demonstrates programming flow when reordering memory between CPU and GPU engines.

This C API example demonstrates programming flow when reordering memory between CPU and GPU engines.

.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2022 Intel Corporation
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
	
	
	
	#include <stdio.h>
	#include <stdlib.h>
	
	#include "oneapi/dnnl/dnnl.h"
	
	#include "example_utils.h"
	
	size_t product(int n_dims, const :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]) {
	    size_t n_elems = 1;
	    for (int d = 0; d < n_dims; ++d) {
	        n_elems *= (size_t)dims[d];
	    }
	    return n_elems;
	}
	
	void fill(:ref:`dnnl_memory_t <doxid-structdnnl__memory>` mem, int n_dims, const :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]) {
	    const size_t n_elems = product(n_dims, dims);
	    float *array = (float *)malloc(n_elems * sizeof(float));
	    if (!array) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "malloc returned NULL");
	
	    for (size_t e = 0; e < n_elems; ++e) {
	        array[e] = e % 7 ? 1.0f : -1.0f;
	    }
	
	    write_to_dnnl_memory(array, mem);
	    free(array);
	}
	
	int find_negative(:ref:`dnnl_memory_t <doxid-structdnnl__memory>` mem, int n_dims, const :ref:`dnnl_dim_t <doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dims[]) {
	    const size_t n_elems = product(n_dims, dims);
	    float *array = (float *)malloc(n_elems * sizeof(float));
	    if (!array) COMPLAIN_EXAMPLE_ERROR_AND_EXIT("%s", "malloc returned NULL");
	    read_from_dnnl_memory(array, mem);
	
	    int negs = 0;
	    for (size_t e = 0; e < n_elems; ++e) {
	        negs += array[e] < 0.0f;
	    }
	
	    free(array);
	    return negs;
	}
	
	void cross_engine_reorder() {
	    :ref:`dnnl_engine_t <doxid-structdnnl__engine>` engine_cpu, engine_gpu;
	    CHECK(:ref:`dnnl_engine_create <doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(&engine_cpu, validate_engine_kind(:ref:`dnnl_cpu <doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2abde7b942413dd36f8285dd360fc0c797>`), 0));
	    CHECK(:ref:`dnnl_engine_create <doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(&engine_gpu, validate_engine_kind(:ref:`dnnl_gpu <doxid-group__dnnl__api__engine_1gga04b3dd9eba628ea02218a52c4c4363a2a6ab900ae0fc26be67742c0e59a015438>`), 0));
	
	    const :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` tz = {2, 16, 1, 1};
	
	    :ref:`dnnl_memory_desc_t <doxid-structdnnl__memory__desc>` m_cpu_md, m_gpu_md;
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	            &m_cpu_md, 4, tz, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`));
	    CHECK(:ref:`dnnl_memory_desc_create_with_tag <doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
	            &m_gpu_md, 4, tz, :ref:`dnnl_f32 <doxid-group__dnnl__api__data__types_1gga012ba1c84ff24bdd068f9d2f9b26a130a6b33889946b183311c39cc1bd0656ae9>`, :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>`));
	
	    :ref:`dnnl_memory_t <doxid-structdnnl__memory>` m_cpu, m_gpu;
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &m_cpu, m_cpu_md, engine_cpu, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	    CHECK(:ref:`dnnl_memory_create <doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
	            &m_gpu, m_gpu_md, engine_gpu, :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`));
	
	    fill(m_cpu, 4, tz);
	    if (find_negative(m_cpu, 4, tz) == 0)
	        COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
	                "%s", "incorrect data fill, no negative values found");
	
	    /* reorder cpu -> gpu */
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` r1_pd;
	    CHECK(:ref:`dnnl_reorder_primitive_desc_create <doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(
	            &r1_pd, m_cpu_md, engine_cpu, m_gpu_md, engine_gpu, NULL));
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` r1;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&r1, r1_pd));
	
	    /* relu gpu */
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` relu_pd;
	    CHECK(:ref:`dnnl_eltwise_forward_primitive_desc_create <doxid-group__dnnl__api__eltwise_1gaf5ae8472e1a364502103dea646ccb5bf>`(&relu_pd, engine_gpu,
	            :ref:`dnnl_forward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`, :ref:`dnnl_eltwise_relu <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5e37643fec6531331e2e38df68d4c65a>`, m_gpu_md, m_gpu_md, 0.0f, 0.0f,
	            NULL));
	
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` relu;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&relu, relu_pd));
	
	    /* reorder gpu -> cpu */
	    :ref:`dnnl_primitive_desc_t <doxid-structdnnl__primitive__desc>` r2_pd;
	    CHECK(:ref:`dnnl_reorder_primitive_desc_create <doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(
	            &r2_pd, m_gpu_md, engine_gpu, m_cpu_md, engine_cpu, NULL));
	    :ref:`dnnl_primitive_t <doxid-structdnnl__primitive>` r2;
	    CHECK(:ref:`dnnl_primitive_create <doxid-group__dnnl__api__primitives__common_1gad07540a0074d9cd3a6970b49897e57d3>`(&r2, r2_pd));
	
	    :ref:`dnnl_stream_t <doxid-structdnnl__stream>` stream_gpu;
	    CHECK(:ref:`dnnl_stream_create <doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354>`(
	            &stream_gpu, engine_gpu, :ref:`dnnl_stream_default_flags <doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547acf05c543bccebd58e6d4e0db7137fb92>`));
	
	    :ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` r1_args[] = {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, m_cpu}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, m_gpu}};
	    CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(r1, stream_gpu, 2, r1_args));
	
	    :ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` relu_args[]
	            = {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, m_gpu}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, m_gpu}};
	    CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(relu, stream_gpu, 2, relu_args));
	
	    :ref:`dnnl_exec_arg_t <doxid-structdnnl__exec__arg__t>` r2_args[] = {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, m_gpu}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, m_cpu}};
	    CHECK(:ref:`dnnl_primitive_execute <doxid-group__dnnl__api__primitives__common_1ga57f8ec3a6e5b33a1068cf2236028935c>`(r2, stream_gpu, 2, r2_args));
	
	    CHECK(:ref:`dnnl_stream_wait <doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>`(stream_gpu));
	
	    if (find_negative(m_cpu, 4, tz) != 0)
	        COMPLAIN_EXAMPLE_ERROR_AND_EXIT(
	                "%s", "found negative values after ReLU applied");
	
	    /* clean up */
	    :ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(relu_pd);
	    :ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(r1_pd);
	    :ref:`dnnl_primitive_desc_destroy <doxid-group__dnnl__api__primitives__common_1ga643938c7c73d200ac1fd3866204e7285>`(r2_pd);
	
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(relu);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(r1);
	    :ref:`dnnl_primitive_destroy <doxid-group__dnnl__api__primitives__common_1gaba605c4591c2054a6ee80ec1b581659f>`(r2);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(m_cpu);
	    :ref:`dnnl_memory_destroy <doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(m_gpu);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(m_cpu_md);
	    :ref:`dnnl_memory_desc_destroy <doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(m_gpu_md);
	
	    :ref:`dnnl_stream_destroy <doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44>`(stream_gpu);
	
	    :ref:`dnnl_engine_destroy <doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(engine_cpu);
	    :ref:`dnnl_engine_destroy <doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(engine_gpu);
	}
	
	int main() {
	    cross_engine_reorder();
	    printf("Example passed on CPU/GPU.\n");
	    return 0;
	}

