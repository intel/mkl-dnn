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

#include <atomic>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "utils/parallel.hpp"

#include "zeropad/zeropad.hpp"

extern "C" {
dnnl_status_t dnnl_impl_zero_pad(
        const dnnl_memory *memory, dnnl_stream *stream);
}

namespace zeropad {

static int compare(const dnn_mem_t &test_mem, res_t *res) {
    if (test_mem.ndims() == 0) return OK;
    if (test_mem.format_kind() != dnnl_blocked) return OK;

    std::atomic<int> ok(true);

    const uint8_t *mem = (const uint8_t *)test_mem;
    size_t type_size = test_mem.sizeof_dt();

    const auto increment
            = [&](dnnl_dims_t &pos, dnnl_dim_t &idx, bool &done, int stop_dim) {
        for (int i = test_mem.ndims() - 1; i >= stop_dim; i--) {
            pos[i]++;
            if (pos[i] < test_mem.dims()[i]) {
                break;
            } else {
                pos[i] = 0;
                if (i == stop_dim) done = true;
            }
        }
        idx = md_off_v(test_mem, pos);
    };

    benchdnn_parallel_nd(test_mem.dims()[0], [&](dnnl_dim_t dim0) {
        dnnl_dims_t pos = {0};
        pos[0] = dim0;
        dnnl_dim_t idx = md_off_v(test_mem, pos);
        bool done = false;

        while (!done && ok) {
            for (size_t i = 0; i < type_size; i++) {
                uint8_t mem_value = mem[type_size * idx + i];
                if (mem_value != dnnl_mem_default_value) ok = false;
            }
            increment(pos, idx, done, 1);
        }
    });

    // Serially check for errors for data dumping purposes
    if (!ok) {
        int errors = 0;
        dnnl_dims_t pos = {0};
        dnnl_dim_t idx = md_off_v(test_mem, pos);
        bool done = false;
        while (!done) {
            for (size_t i = 0; i < type_size; i++) {
                uint8_t mem_value = mem[type_size * idx + i];
                bool idx_ok = (mem_value == dnnl_mem_default_value);
                if (!idx_ok) errors++;
                const bool dump = (!idx_ok && (errors < 10 || verbose >= 10))
                        || (verbose >= 99);
                if (dump) {
                    BENCHDNN_PRINT(0,
                            "[%4ld][" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] dt:% 9.6g \n",
                            (long)idx, pos[0], pos[1], pos[2], pos[3], pos[4],
                            pos[5], test_mem.get_elem(idx));
                    break;
                }
            }
            increment(pos, idx, done, 0);
        }

        BENCHDNN_PRINT(0, "%s\n", "@@@ check_non_zeroed_elements failed");
        res->errors += errors;
    }

    int errors = 0;
    auto status = check_zero_padding(test_mem, test_mem.dt(), res, &errors);
    res->errors += errors;

    bool passed = ok && (status == OK);
    if (passed) res->state = PASSED;
    return passed ? OK : FAIL;
}

static dnnl_status_t perf_func(
        const dnnl_stream_t &stream, const std::vector<dnnl_exec_arg_t> &args) {
    return dnnl_impl_zero_pad(args[0].memory, stream);
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type({prb->dt}, FWD_D, res);
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == bench_mode_t::list) return res->state = LISTED, OK;

    prb->skip_unimplemented(res);
    if (res->state == SKIPPED) return OK;

    auto data_md = dnn_mem_t::init_md(
            prb->ndims, prb->dims.data(), prb->dt, prb->tag);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    if (bench_mode == bench_mode_t::init) return res->state = INITIALIZED, OK;

    SAFE(check_mem_size(data_md, res), WARN);
    if (res->state == SKIPPED) return OK;

    const auto &test_engine = get_test_engine();

    // `NaN` prefilling is essential for zero-padding.
    dnn_mem_t test_mem(data_md, test_engine, /* prefill = */ true);

    args_t args;
    args.set(0, test_mem);
    perf_function_t perf_func_ = &perf_func;

    SAFE(run_execution(perf_func_, test_engine, args, res), WARN);

    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(compare(test_mem, res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        // Get plain memory desc size to have a proper padded area size.
        auto plain_data_md = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->dt, tag::abx);
        // Fill output bytes for perf_report.
        res->ibytes = 0; // Since we don't read any data from padding.
        res->obytes = dnnl_memory_desc_get_size(data_md)
                - dnnl_memory_desc_get_size(plain_data_md);
    }
    return measure_perf(get_default_thr_ctx(), res, perf_func_, args);
}

} // namespace zeropad
