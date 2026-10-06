/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "dnnl_memory.hpp"

#include "self/self.hpp"

namespace self {

static int check_bool_operator() {
    dnnl_dim_t dims {1};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
    DNN_SAFE(dnn_mem_t::init_md(md, 1, &dims, dnnl_f32, tag::abx), WARN);
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md0;
    DNN_SAFE(dnn_mem_t::init_md(md0, 0, &dims, dnnl_f32, tag::abx), WARN);
    {
        dnn_mem_t m;
        SELF_CHECK_EQ(bool(m), false);
    }
    {
        dnn_mem_t m(md, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, get_test_engine(), /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(1, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(0, &dims, dnnl_f32, &dims /* strides */, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    {
        dnn_mem_t m(md, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(m), true);
        dnn_mem_t n(md0, dnnl_f32, tag::abx, get_test_engine(),
                /* prefill = */ false);
        SELF_CHECK_EQ(bool(n), false);
    }
    return OK;
}

static int check_init_md_status() {
    dnnl_dims_t dims {2, 16};
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> md;
    SELF_CHECK_EQ(
            dnn_mem_t::init_md(md, 2, dims, dnnl_f32, "ab"), dnnl_success);
    // A tag with more letters than `ndims` must return an error instead of
    // terminating the run.
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bad_md;
    SELF_CHECK_EQ(dnn_mem_t::init_md(bad_md, 2, dims, dnnl_f32, "abcd"),
            dnnl_invalid_arguments);
    return OK;
}

void memory() {
    RUN(check_bool_operator());
    RUN(check_init_md_status());
}

} // namespace self
