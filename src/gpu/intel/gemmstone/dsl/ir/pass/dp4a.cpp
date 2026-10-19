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

#include "dsl/ir/pass/dp4a.hpp"
#include "dsl/ir/core.hpp"
#include "dsl/ir/fma.hpp"
#include "dsl/ir/pass/trace.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class dp4a_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const func_call_t &obj) override {
        auto *dpas = obj.func.as_ptr<dpas_t>();
        if (!dpas) return obj;

        int M = dpas->exec_size;
        int N = dpas->rcount;
        int K = dpas->sdepth * 4;

        auto &dst = dpas_t::arg_dst(obj);
        auto &src0 = dpas_t::arg_src0(obj);
        auto &src1 = dpas_t::arg_src1(obj);
        auto &src2 = dpas_t::arg_src2(obj);
        int dst_size = dpas->dst_type.size();
        int src0_size = dpas->dst_type.size();
        int src1_size = dpas->src1_type.size();
        int src2_size = dpas->src2_type.size();
        auto dst_type = to_dp4a_type(dpas->dst_type);
        auto src1_type = to_dp4a_type(dpas->src1_type);
        auto src2_type = to_dp4a_type(dpas->src2_type);
        bool is_src0_zero = src0.is(0);

        stmt_t stmt;
        auto _dp4a = dpas_t::make(
                /*is_dpasw=*/false, M, 1, 1, dst_type, src1_type, src2_type);
        auto &dp4a = _dp4a.as<dpas_t>();
        auto zero = shuffle_t::make_broadcast(0, M);
        int k0 = (is_src0_zero ? -4 : 0);
        for (int k = k0; k < K; k += 4) {
            for (int n = 0; n < N; n++) {
                int dst_off = n * M * dst_size;
                int src0_off = n * M * src0_size;
                int src1_off = k * M * src1_size;
                int src2_off = (n * K + k) * src2_size;
                auto _dst = dst + dst_off;
                auto _src0 = is_src0_zero ? _dst : (src0 + src0_off);
                auto _src1 = src1 + src1_off;
                auto _src2 = src2 + src2_off;
                if (k < 0) {
                    stmt = stmt.append(store_t::make(_dst, 0, zero));
                } else {
                    stmt = stmt.append(dp4a(_dst, _src0, _src1, _src2));
                }
            }
        }
        return std::move(stmt);
    }

private:
    static dsl::type_t to_dp4a_type(const dsl::type_t &type) {
        if (type.is_x32()) return type;
        if (type.is_s8()) return dsl::type_t::s32();
        if (type.is_u8()) return dsl::type_t::u32();
        stub();
        return dsl::type_t();
    }
};

stmt_t inject_dp4a(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = dp4a_injector_t().mutate(s);
    trace_pass("inject_dp4a", ret, ir_ctx);
    return ret;
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
