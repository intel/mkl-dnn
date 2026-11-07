/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <cinttypes>
#include <stdio.h>

#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;

const char *dnnl_runtime2str(unsigned runtime) {
    switch (runtime) {
        case DNNL_RUNTIME_NONE: return "none";
        case DNNL_RUNTIME_SEQ: return "sequential";
        case DNNL_RUNTIME_OMP: return "OpenMP";
        case DNNL_RUNTIME_TBB: return "TBB";
        case DNNL_RUNTIME_OCL: return "OpenCL";
        case DNNL_RUNTIME_THREADPOOL: return "threadpool";
        case DNNL_RUNTIME_SYCL: return "SYCL";
        case DNNL_RUNTIME_ZE: return "Level Zero";
        default: return "unknown";
    }
}

const char *dnnl_fmt_kind2str(dnnl_format_kind_t v) {
    if (v == dnnl_format_kind_undef) return "undef";
    if (v == dnnl_format_kind_any) return "any";
    if (v == dnnl_blocked) return "blocked";
    if (v == dnnl_format_kind_sparse) return "sparse";
    if (v == format_kind::wino || v == format_kind::rnn_packed
            || v == format_kind::cublaslt_blocked
            || v == format_kind::zen_packed)
        return "opaque";
    if (v == dnnl_format_kind_host_scalar) return "host_scalar";
    if (v == dnnl_format_kind_max) return "max";
    assert(!"unknown fmt_kind");
    return "unknown fmt_kind";
}
