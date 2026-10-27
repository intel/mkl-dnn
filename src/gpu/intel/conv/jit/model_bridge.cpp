/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/intel/conv/jit/model_bridge.hpp"

#include <mutex>

#include "gpu/intel/conv/jit/config.hpp"
#include "gpu/intel/conv/jit/model.hpp"
#include "gpu/intel/conv/jit/model_data.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {
namespace model {

type_t to_type(data_type_t dt) {
    switch (static_cast<int>(dt)) {
        case data_type::s8:
        case data_type::f4_e2m1:
        case data_type::f8_e5m2:
        case data_type::f8_e4m3:
        case data_type::u8: return type_t::d8;
        case data_type::f16:
        case data_type::bf16: return type_t::d16;
        case data_type::tf32:
        case data_type::f32:
        case data_type::s32: return type_t::d32;
        case data_type::f64: return type_t::d64;
        default: gpu_error_not_expected() << "Unknown type: " << dt;
    }
    return type_t::undef;
}

hw_t to_hw(ngen::HW hw) {
    using intel::jit::to_string;

    switch (hw) {
        case ngen::HW::XeLP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG: return hw_t::xehpg;
        case ngen::HW::XeHPC: return hw_t::xehpc;
        case ngen::HW::Xe2: return hw_t::xehpc;
        case ngen::HW::Xe3: return hw_t::xehpc;
        case ngen::HW::Xe3p: return hw_t::xehpc;
        default: gpu_error_not_expected() << "Unknown HW: " << to_string(hw);
    }
    return hw_t::undef;
}

fma_t to_fma(fma_kind_t fma) {
    switch (fma) {
        case fma_kind_t::mad: return fma_t::mad;
        case fma_kind_t::dp4a:
        case fma_kind_t::dpas:
        case fma_kind_t::dpasw: return fma_t::dpas;
        default:
            gpu_error_not_expected() << "Unknown FMA kind: " << to_string(fma);
    }
    return fma_t::undef;
}

hw_config_t to_hw_config(const config_t &cfg) {
    auto &prb = cfg.prb();
    auto &hw = cfg.hw();
    return hw_config_t(to_hw(hw), to_fma(cfg.fma_kind()),
            to_type(prb.a_data_type), hw.eu_count());
}

conv_sample_t to_sample(const config_t &cfg, const blocking_params_t &params) {
    auto &prb = cfg.prb();
    conv_sample_t ret;
    ret.prop = (prb.is_fwd ? prop_t::fwd
                           : (prb.is_bwd_d ? prop_t::bwd_d : prop_t::bwd_w));
    ret.src_type = to_type(prb.a_data_type);
    ret.dst_type = to_type(prb.c_data_type);
    ret.hw_cfg = to_hw_config(cfg);
    ret.transpose = prb.ab_swap_transpose;

    auto &blk = params.blocking();
    auto shape = cfg.shape(/*pad=*/false);
#define HANDLE(name) \
    do { \
        ret.shape.name = -1; \
        ret.loop.name = -1; \
        ret.tg.name = -1; \
        ret.iter.name = -1; \
        if (!shape.has(pvars::name)) break; \
        ret.shape.name = shape.get(pvars::name); \
        ret.loop.name = blk.loop().get(pvars::name, 1); \
        ret.tg.name = blk.thread_group().get(pvars::name, 1); \
        ret.iter.name = blk.iter().get(pvars::name, 1); \
    } while (false)
    HANDLE(g);
    HANDLE(mb);
    HANDLE(oc);
    HANDLE(ic);
    HANDLE(id);
    HANDLE(ih);
    HANDLE(iw);
    HANDLE(od);
    HANDLE(oh);
    HANDLE(ow);
    HANDLE(kd);
    HANDLE(kh);
    HANDLE(kw);
#undef HANDLE
    ret.pad();
    return ret;
}

conv_sample_t fixup(const conv_sample_t &sample) {
    auto ret = sample;
    if (sample.prop == prop_t::bwd_w && sample.dst_type < type_t::d32)
        ret.dst_type = type_t::d32;
    if (sample.prop == prop_t::fwd && sample.src_type == type_t::d8)
        ret.dst_type = type_t::d8;
    return ret;
}

enum class gbr_kind_t {
    all_common,
    xehpc_common,
    xehpc_dw,
    xehpg_common,
    xehpg_dw,
    _max
};

using gbr_kind_hash_t = ir_utils::enum_hash_t<gbr_kind_t>;

gbr_kind_t get_gbr_kind(const config_t &cfg) {
    auto &prb = cfg.prb();
    if (cfg.hw() >= ngen::HW::XeHPC) {
        if (prb.is_dw) return gbr_kind_t::xehpc_dw;
        return gbr_kind_t::xehpc_common;
    }
    if (prb.is_dw) return gbr_kind_t::xehpg_dw;
    return gbr_kind_t::xehpg_common;
}

inline bool is_big_endian() {
    uint32_t u = 0x01020304;
    uint8_t a[4] = {};
    std::memcpy(a, &u, sizeof(u));
    return a[0] == 0x01;
}

std::vector<uint8_t> unpack_data(const uint64_t *data_u64, size_t n) {
    size_t elem_len = sizeof(data_u64[0]);
    size_t size = n * elem_len;
    std::vector<uint8_t> data_u8(size);
    std::memcpy(data_u8.data(), data_u64, size);
    if (is_big_endian()) {
        for (size_t i = 0; i < size; i += elem_len) {
            for (size_t j = 0; j < elem_len / 2; j++) {
                std::swap(data_u8[i + j], data_u8[i + elem_len - 1 - j]);
            }
        }
    }
    return data_u8;
}

gradient_boost_regressor_t &get_gbr(const config_t &cfg) {
    using data_func_t = const uint64_t *(*)(size_t &);
    // clang-format off
    static const std::unordered_map<gbr_kind_t, data_func_t, gbr_kind_hash_t>
            kind2data = {
                    {gbr_kind_t::xehpc_common, get_model_xehpc_common_data},
                    {gbr_kind_t::xehpg_common, get_model_xehpg_common_data},
                    {gbr_kind_t::xehpc_dw, get_model_xehpc_dw_data},
                    {gbr_kind_t::xehpg_dw, get_model_xehpg_dw_data}
            };
    // clang-format on
    static const int nkinds = (int)gbr_kind_t::_max;
    static gradient_boost_regressor_t gbrs[nkinds];
    static std::once_flag flags[nkinds];
    auto kind = get_gbr_kind(cfg);
    auto &gbr = gbrs[(int)kind];
    // Deserialize lazily, on first use.
    std::call_once(flags[(int)kind], [&] {
        size_t n = 0;
        auto *data = kind2data.at(kind)(n);
        auto s = serialization_stream_t::from_data(unpack_data(data, n));
        deserializer_t d(s);
        gbr = gradient_boost_regressor_t::deserialize(d);
    });
    return gbr;
}

float get_score(const config_t &cfg, const blocking_params_t &params) {
    auto sample = to_sample(cfg, params);
    sample = fixup(sample);
    auto bmnk_sample = sample.to_bmnk_conv_sample();
    return get_gbr(cfg).predict(bmnk_sample.to_x());
}

} // namespace model
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
