/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <stdlib.h>
#include <string.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/impl_filter.hpp"
#include "utils/parser.hpp"

#include "self/self.hpp"

using namespace parser;

namespace self {

using pk_t = attr_t::post_ops_t::kind_t;

static int check_simple_enums() {
    using p = attr_t::post_ops_t;
    SELF_CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::SUM), "sum");
    SELF_CHECK_CASE_STR_EQ(p::kind2str(p::kind_t::RELU), "relu");

    SELF_CHECK_EQ(p::str2kind("sum"), p::kind_t::SUM);
    SELF_CHECK_EQ(p::str2kind("SuM"), p::kind_t::SUM);

    SELF_CHECK_EQ(p::str2kind("relu"), p::kind_t::RELU);
    SELF_CHECK_EQ(p::str2kind("ReLU"), p::kind_t::RELU);

    return OK;
}

static int check_attr2str() {
    attr_t attr;
    SELF_CHECK_EQ(attr.is_def(), true);

    SELF_CHECK_PRINT_EQ(attr, "");

    attr = attr_t();
    attr.zero_points.set(DNNL_ARG_SRC, policy_t::COMMON, 1);
    SELF_CHECK_PRINT_EQ(attr, "--attr-zero-points=src:common:1 ");

    attr.zero_points.set(DNNL_ARG_SRC, policy_t::PER_DIM_0, 3);
    attr.zero_points.set(DNNL_ARG_WEIGHTS, {policy_t::PER_DIM_1, 2});
    SELF_CHECK_PRINT_EQ2(attr,
            "--attr-zero-points=src:per_dim_0+wei:per_dim_1 ",
            "--attr-zero-points=wei:per_dim_1+src:per_dim_0 ");

    attr = attr_t();
    attr.scales.set(DNNL_ARG_SRC_0,
            attr_t::arg_scales_t::entry_t(policy_t::COMMON, 2.3f));
    SELF_CHECK_PRINT_EQ(attr, "--attr-scales=src:common:2.3 ");

    attr.scales.set(DNNL_ARG_SRC_0,
            attr_t::arg_scales_t::entry_t(policy_t::COMMON, 2.2f));
    attr.scales.set(DNNL_ARG_SRC_1,
            attr_t::arg_scales_t::entry_t(policy_t::COMMON, 3.f));
    SELF_CHECK_PRINT_EQ(attr, "--attr-scales=src:common:2.2+src1:common:3 ");

    attr = attr_t();
    attr.scales.set(DNNL_ARG_SRC_0,
            attr_t::arg_scales_t::entry_t(
                    policy_t::PER_OCIC, 2.3f, dnnl_bf16, {2, 1}));
    attr.zero_points.set(
            DNNL_ARG_WEIGHTS, {policy_t::PER_DIM_1, 2, dnnl_s8, {}});
    SELF_CHECK_PRINT_EQ(attr,
            "--attr-scales=src:per_ocic:bf16:2x1 "
            "--attr-zero-points=wei:per_dim_1:s8 ");

    attr = attr_t();
    attr.scales.set(DNNL_ARG_SRC_0,
            attr_t::arg_scales_t::entry_t(
                    policy_t::PER_OC, 2.3f, dnnl_bf16, {}));
    attr.zero_points.set(
            DNNL_ARG_WEIGHTS, {policy_t::PER_OCIC, 2, dnnl_s8, {4, 1}});
    SELF_CHECK_PRINT_EQ(attr,
            "--attr-scales=src:per_oc:bf16 "
            "--attr-zero-points=wei:per_ocic:s8:4x1 ");

    attr = attr_t();
    attr.fpmath_mode.set(dnnl_fpmath_mode_strict, true);
    SELF_CHECK_PRINT_EQ(attr, "--attr-fpmath=strict:true ");
    attr.fpmath_mode.set(dnnl_fpmath_mode_bf16, false);
    SELF_CHECK_PRINT_EQ(attr, "--attr-fpmath=bf16 ");

    return OK;
}

static int check_attr() {
#define SELF_CHECK_ATTR_ZP(zp, arg, zero_points_policy, zero_points_value, \
        zero_points_data_type, zero_points_groups, zero_points_mask_input) \
    do { \
        const auto &entry = (zp).get(arg); \
        SELF_CHECK_EQ(entry.policy, zero_points_policy); \
        SELF_CHECK_EQ(entry.value, zero_points_value); \
        SELF_CHECK_EQ(entry.dt, zero_points_data_type); \
        for (size_t i = 0; i < (zero_points_groups).size(); ++i) \
            SELF_CHECK_EQ(entry.groups[i], (zero_points_groups)[i]); \
        SELF_CHECK_EQ(entry.mask_input, zero_points_mask_input); \
    } while (0)

    static base_settings_t def;
    {
        base_settings_t s;
        std::vector<attr_t::zero_points_t> &zp = s.zero_points;
        std::string content_to_parse;

        content_to_parse = std::string(
                "--attr-zero-points=src:common:0+wei:per_oc+dst:common:-2,"
                "src:4");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(zp.size(), 2);
        const std::vector<dnnl_dim_t> def_g {};
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_SRC, policy_t::COMMON, 0, dnnl_s32,
                def_g, attr_t::mask_input_t::policy);
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_WEIGHTS, policy_t::PER_OC, 0,
                dnnl_s32, def_g, attr_t::mask_input_t::policy);
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_DST, policy_t::COMMON, -2, dnnl_s32,
                def_g, attr_t::mask_input_t::policy);

        SELF_CHECK_EQ(zp[1].get(DNNL_ARG_SRC).mask, 4);
        SELF_CHECK_EQ(
                zp[1].get(DNNL_ARG_SRC).mask_input, attr_t::mask_input_t::mask);
    }

    {
        base_settings_t s;
        std::vector<attr_t::arg_scales_t> &sc = s.scales;
        std::string content_to_parse;

        content_to_parse = std::string(
                "--attr-scales=src:common:1.5+wei:2+src:common:0.5");
        // `src` scale is overridden with the latter value.
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(sc.size(), 1);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC).policy, policy_t::COMMON);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC).scale, 0.5f);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC).mask_input,
                attr_t::mask_input_t::policy);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_WEIGHTS).mask, 2);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_WEIGHTS).scale, 1.f);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_WEIGHTS).mask_input,
                attr_t::mask_input_t::mask);

        content_to_parse
                = std::string("--attr-scales=src:common:2.5+src1:common:1.5");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(sc.size(), 1);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC_0).policy, policy_t::COMMON);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC_0).scale, 2.5);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC_1).policy, policy_t::COMMON);
        SELF_CHECK_EQ(sc[0].get(DNNL_ARG_SRC_1).scale, 1.5);
    }

    {
        base_settings_t s;
        std::vector<attr_t::zero_points_t> &zp = s.zero_points;
        std::string content_to_parse("--attr-zero-points=wei:per_ocic:s8:2x1");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(zp.size(), 1);
        std::vector<dnnl_dim_t> groups = {2, 1};
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_WEIGHTS, policy_t::PER_OCIC, 0,
                dnnl_s8, groups, attr_t::mask_input_t::policy);
    }

    // sbzp quantization mode is encoded as the trailing token of a scale entry,
    // following a (possibly empty) groups field. This is parser validation, so
    // scales and zero-points are exercised independently.
    {
        base_settings_t s;
        std::vector<attr_t::arg_scales_t> &sc = s.scales;

        // With groups.
        SELF_CHECK_EQ(parse_attributes(s, def,
                              "--attr-scales=wei:per_ocic:f16:32x1:sbzp"),
                true);
        SELF_CHECK_EQ(sc.size(), 1);
        std::vector<dnnl_dim_t> groups = {32, 1};
        const auto &sc_g = sc[0].get(DNNL_ARG_WEIGHTS);
        SELF_CHECK_EQ(sc_g.policy, policy_t::PER_OCIC);
        SELF_CHECK_EQ(sc_g.dt, dnnl_f16);
        for (size_t i = 0; i < groups.size(); ++i)
            SELF_CHECK_EQ(sc_g.groups[i], groups[i]);
        SELF_CHECK_EQ(sc_g.qmode, dnnl_quantization_mode_static_sbzp);

        // Without groups: the qmode token still parses in its own slot and must
        // not be mistaken for the groups field.
        SELF_CHECK_EQ(parse_attributes(
                              s, def, "--attr-scales=wei:per_tensor:f16::sbzp"),
                true);
        SELF_CHECK_EQ(sc.size(), 1);
        const auto &sc_ng = sc[0].get(DNNL_ARG_WEIGHTS);
        SELF_CHECK_EQ(sc_ng.policy, policy_t::PER_TENSOR);
        SELF_CHECK_EQ(sc_ng.dt, dnnl_f16);
        SELF_CHECK_EQ(sc_ng.groups.empty(), true);
        SELF_CHECK_EQ(sc_ng.qmode, dnnl_quantization_mode_static_sbzp);
    }

    {
        base_settings_t s;
        std::vector<attr_t::zero_points_t> &zp = s.zero_points;

        // With groups.
        SELF_CHECK_EQ(parse_attributes(s, def,
                              "--attr-zero-points=wei:per_ocic:f16:32x1:sbzp"),
                true);
        SELF_CHECK_EQ(zp.size(), 1);
        std::vector<dnnl_dim_t> groups = {32, 1};
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_WEIGHTS, policy_t::PER_OCIC, 0,
                dnnl_f16, groups, attr_t::mask_input_t::policy);
        SELF_CHECK_EQ(zp[0].get(DNNL_ARG_WEIGHTS).qmode,
                dnnl_quantization_mode_static_sbzp);

        // Without groups: the qmode token still parses in its own slot and must
        // not be mistaken for the groups field.
        SELF_CHECK_EQ(parse_attributes(s, def,
                              "--attr-zero-points=wei:per_tensor:f16::sbzp"),
                true);
        SELF_CHECK_EQ(zp.size(), 1);
        const std::vector<dnnl_dim_t> def_g {};
        SELF_CHECK_ATTR_ZP(zp[0], DNNL_ARG_WEIGHTS, policy_t::PER_TENSOR, 0,
                dnnl_f16, def_g, attr_t::mask_input_t::policy);
        SELF_CHECK_EQ(zp[0].get(DNNL_ARG_WEIGHTS).qmode,
                dnnl_quantization_mode_static_sbzp);
    }

    {
        base_settings_t s;
        std::vector<attr_t::arg_scales_t> &sc = s.scales;
        std::string content_to_parse(
                "--attr-scales=attr_post_op_dw_wei:common:2");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(sc.size(), 1);
        const auto arg = DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS;
        SELF_CHECK_EQ(sc[0].get(arg).policy, policy_t::COMMON);
        SELF_CHECK_EQ(sc[0].get(arg).scale, 2.f);
    }

    // depthwise conv section
    {
        base_settings_t s;
        std::vector<attr_t::post_ops_t> &po = s.post_ops;
        std::string content_to_parse("--attr-post-ops=dw:k3s1p1");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(po[0].len(), 1);
        const auto &e = po[0].entry[0];
        SELF_CHECK_EQ(e.kind, pk_t::DW);
        const auto &ce = e.convolution;
        SELF_CHECK_EQ(ce.kernel, 3);
        SELF_CHECK_EQ(ce.stride, 1);
        SELF_CHECK_EQ(ce.padding, 1);
        SELF_CHECK_EQ(ce.dst_dt, dnnl_f32);
    }

    {
        base_settings_t s;
        std::vector<attr_t::post_ops_t> &po = s.post_ops;
        std::string content_to_parse(
                "--attr-post-ops=relu:0.5+dw:k3s2p1:s8+linear:2:1");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(po[0].len(), 3);
        auto &e = po[0].entry[0];
        SELF_CHECK_EQ(e.kind, pk_t::RELU);
        auto &ee = e.eltwise;
        SELF_CHECK_EQ(ee.alg, dnnl_eltwise_relu);
        SELF_CHECK_EQ(ee.alpha, 0.5f);
        SELF_CHECK_EQ(ee.beta, 0.f);

        e = po[0].entry[1];
        SELF_CHECK_EQ(e.kind, pk_t::DW);
        const auto &ce = e.convolution;
        SELF_CHECK_EQ(ce.kernel, 3);
        SELF_CHECK_EQ(ce.stride, 2);
        SELF_CHECK_EQ(ce.padding, 1);
        SELF_CHECK_EQ(ce.dst_dt, dnnl_s8);

        e = po[0].entry[2];
        SELF_CHECK_EQ(e.kind, pk_t::LINEAR);
        ee = e.eltwise;
        SELF_CHECK_EQ(ee.alg, dnnl_eltwise_linear);
        SELF_CHECK_EQ(ee.alpha, 2.f);
        SELF_CHECK_EQ(ee.beta, 1.f);
    }

    {
        base_settings_t s;
        std::vector<attr_t::fpmath_mode_t> &fm = s.fpmath_mode;
        std::string content_to_parse("--attr-fpmath=strict:true");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(fm[0].mode, dnnl_fpmath_mode_strict);
        SELF_CHECK_EQ(fm[0].apply_to_int, true);
    }

    {
        base_settings_t s;
        std::vector<attr_t::fpmath_mode_t> &fm = s.fpmath_mode;
        std::string content_to_parse("--attr-fpmath=bf16");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(fm[0].mode, dnnl_fpmath_mode_bf16);
        SELF_CHECK_EQ(fm[0].apply_to_int, false);
    }

    {
        base_settings_t s;
        // Updating the default values and expect them to be returned.
        std::vector<attr_t::fpmath_mode_t> &fm = s.fpmath_mode;
        def.fpmath_mode.emplace_back();
        def.fpmath_mode[0].set(dnnl_fpmath_mode_bf16, true);
        std::string content_to_parse("--attr-fpmath=");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(fm[0].mode, dnnl_fpmath_mode_bf16);
        SELF_CHECK_EQ(fm[0].apply_to_int, true);
        // Reset default settings
        def = base_settings_t();
    }

    {
        base_settings_t s;
        std::vector<attr_t::dropout_t> &d = s.dropout;
        std::string content_to_parse("--attr-dropout=0.5:12345:axb:16:true");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(d[0].p, 0.5f);
        SELF_CHECK_EQ(d[0].seed, 12345);
        SELF_CHECK_CASE_STR_EQ(d[0].tag.c_str(), tag::axb);
        SELF_CHECK_EQ(d[0].offset, 16);
        SELF_CHECK_EQ(d[0].use_host_scalars, true);
    }

    {
        base_settings_t s;
        std::vector<attr_t::dropout_t> &d = s.dropout;
        std::string content_to_parse("--attr-dropout=0.5:12345:axb");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(d[0].p, 0.5f);
        SELF_CHECK_EQ(d[0].seed, 12345);
        SELF_CHECK_CASE_STR_EQ(d[0].tag.c_str(), tag::axb);
        SELF_CHECK_EQ(d[0].offset, 0);
        SELF_CHECK_EQ(d[0].use_host_scalars, false);
    }

    {
        base_settings_t s;
        std::vector<attr_t::dropout_t> &d = s.dropout;
        std::string content_to_parse(
                "--attr-dropout=0.25:123456789:undef:987654321:false");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(d[0].p, 0.25f);
        SELF_CHECK_EQ(d[0].seed, 123456789);
        SELF_CHECK_CASE_STR_EQ(d[0].tag.c_str(), tag::undef);
        SELF_CHECK_EQ(d[0].offset, 987654321);
        SELF_CHECK_EQ(d[0].use_host_scalars, false);
    }

    {
        base_settings_t s;
        std::vector<attr_t::dropout_t> &d = s.dropout;
        std::string content_to_parse("--attr-dropout=0.75");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(d[0].p, 0.75f);
        SELF_CHECK_EQ(d[0].seed, 0);
        SELF_CHECK_CASE_STR_EQ(d[0].tag.c_str(), tag::any);
        SELF_CHECK_EQ(d[0].offset, 0);
        SELF_CHECK_EQ(d[0].use_host_scalars, false);
    }

    {
        base_settings_t s;
        std::vector<attr_t::dropout_t> &d = s.dropout;
        std::string content_to_parse("--attr-dropout=");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(d[0].p, 0.f);
        SELF_CHECK_EQ(d[0].seed, 0);
        SELF_CHECK_CASE_STR_EQ(d[0].tag.c_str(), tag::any);
        SELF_CHECK_EQ(d[0].offset, 0);
        SELF_CHECK_EQ(d[0].use_host_scalars, false);
    }

    {
        base_settings_t s;
        std::vector<attr_t::rounding_mode_t> &rm = s.rounding_mode;
        std::string content_to_parse("--attr-rounding-mode=dst:stochastic");
        auto st = parse_attributes(s, def, content_to_parse.c_str());
        SELF_CHECK_EQ(st, true);
        SELF_CHECK_EQ(rm[0].get(DNNL_ARG_DST), dnnl_rounding_mode_stochastic);
        SELF_CHECK_EQ(rm[0].get(DNNL_ARG_SRC), dnnl_rounding_mode_environment);
    }

    {
        base_settings_t s;
        std::vector<attr_t::precomputed_reductions_t> &pr
                = s.precomputed_reductions;
        std::string content_to_parse;
        content_to_parse = std::string(
                "--attr-precomputed-reductions=src:per_tensor:s32:1x1024");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).policy, policy_t::PER_TENSOR);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).dt, dnnl_s32);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).groups[0], 1);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).groups[1], 1024);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).mask_input,
                attr_t::mask_input_t::policy);

        content_to_parse
                = std::string("--attr-precomputed-reductions=src:15:s32:1x32");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).mask, 15);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).dt, dnnl_s32);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).groups[0], 1);
        SELF_CHECK_EQ(pr[0].get(DNNL_ARG_SRC).groups[1], 32);
        SELF_CHECK_EQ(
                pr[0].get(DNNL_ARG_SRC).mask_input, attr_t::mask_input_t::mask);

        content_to_parse = std::string("--attr-precomputed-reductions=");
        SELF_CHECK_EQ(parse_attributes(s, def, content_to_parse.c_str()), true);
        SELF_CHECK_EQ(
                pr[0].get(DNNL_ARG_SRC).mask_input, attr_t::mask_input_t::none);
    }

#undef SELF_CHECK_ATTR_ZP

    return OK;
}

void append_sum(attr_t::post_ops_t &po, float ascale = 1.f,
        int32_t zero_point = 0, dnnl_data_type_t adt = dnnl_data_type_undef) {
    po.entry.emplace_back(pk_t::SUM);
    auto &e = po.entry.back();

    e.sum.scale = ascale;
    e.sum.zero_point = zero_point;
    e.sum.dt = adt;
}

void append_convolution(attr_t::post_ops_t &po, pk_t akind, int kernel,
        int stride, int padding, dnnl_data_type_t adst_dt = dnnl_f32) {
    po.entry.emplace_back(akind);
    auto &e = po.entry.back();

    e.convolution.kernel = kernel;
    e.convolution.stride = stride;
    e.convolution.padding = padding;
    e.convolution.dst_dt = adst_dt;
}

void append_eltwise(attr_t::post_ops_t &po, pk_t akind, float aalpha = 0.f,
        float abeta = 0.f) {
    po.entry.emplace_back(akind);
    auto &e = po.entry.back();

    e.eltwise.alg = attr_t::post_ops_t::kind2dnnl_kind(akind);
    e.eltwise.alpha = aalpha;
    e.eltwise.beta = abeta;
}

void append_binary(attr_t::post_ops_t &po, pk_t akind, dnnl_data_type_t src_dt1,
        attr_t::mask_input_t mask_input, int64_t mask, attr_t::policy_t policy,
        const std::string &tag) {
    po.entry.emplace_back(akind);
    auto &e = po.entry.back();

    e.binary.alg = attr_t::post_ops_t::kind2dnnl_kind(akind);
    e.binary.src1_dt = src_dt1;
    e.binary.mask_input = mask_input;
    e.binary.mask = mask;
    e.binary.policy = policy;
    e.binary.tag = tag;
}

static int check_post_ops2str() {
    attr_t::post_ops_t po;
    SELF_CHECK_EQ(po.is_def(), true);
    SELF_CHECK_PRINT_EQ(po, "");

    append_sum(po);
    SELF_CHECK_EQ(po.len(), 1);
    SELF_CHECK_PRINT_EQ(po, "sum");

    append_eltwise(po, pk_t::RELU);
    SELF_CHECK_EQ(po.len(), 2);
    SELF_CHECK_PRINT_EQ(po, "sum+relu");

    append_sum(po, 2.f, 1, dnnl_s8);
    SELF_CHECK_EQ(po.len(), 3);
    SELF_CHECK_PRINT_EQ(po, "sum+relu+sum:2:1:s8");

    append_eltwise(po, pk_t::LINEAR, 5.f, 10.f);
    SELF_CHECK_EQ(po.len(), 4);
    SELF_CHECK_PRINT_EQ(po, "sum+relu+sum:2:1:s8+linear:5:10");

    append_convolution(po, pk_t::DW, 3, 1, 1);
    SELF_CHECK_EQ(po.len(), 5);
    SELF_CHECK_PRINT_EQ(po, "sum+relu+sum:2:1:s8+linear:5:10+dw:k3s1p1");

    append_convolution(po, pk_t::DW, 3, 2, 1, dnnl_s32);
    SELF_CHECK_EQ(po.len(), 6);
    SELF_CHECK_PRINT_EQ(
            po, "sum+relu+sum:2:1:s8+linear:5:10+dw:k3s1p1+dw:k3s2p1:s32");

    {
        using mi_t = attr_t::mask_input_t;
        attr_t::post_ops_t bin_po_int_mask;
        append_binary(bin_po_int_mask, pk_t::ADD, dnnl_f32, mi_t::mask, 13,
                policy_t::COMMON, tag::abx);
        SELF_CHECK_PRINT_EQ(bin_po_int_mask, "add:f32:13:abx");
    }

    {
        using mi_t = attr_t::mask_input_t;
        attr_t::post_ops_t bin_po_policy;
        append_binary(bin_po_policy, pk_t::ADD, dnnl_f32, mi_t::policy, 13,
                policy_t::COMMON, tag::any);
        SELF_CHECK_PRINT_EQ(bin_po_policy, "add:f32:common");
    }

    return OK;
}

static int check_str2post_ops() {
    attr_t::post_ops_t ops;

    SELF_CHECK_EQ(ops.is_def(), true);

    auto quick = [&](int len) -> int {
        for (int i = 0; i < 2; ++i) {
            if (2 * i + 0 >= len) return OK;
            SELF_CHECK_EQ(ops.entry[2 * i + 0].kind, attr_t::post_ops_t::SUM);
            SELF_CHECK_EQ(ops.entry[2 * i + 0].sum.scale, 2. + i);
            if (2 * i + 1 >= len) return OK;
            SELF_CHECK_EQ(ops.entry[2 * i + 1].kind, attr_t::post_ops_t::RELU);
            SELF_CHECK_EQ(ops.entry[2 * i + 1].eltwise.alpha, 0.);
            SELF_CHECK_EQ(ops.entry[2 * i + 1].eltwise.beta, 0.);
        }
        return OK;
    };

    using namespace parser::parsers;

    ops = str2attr_post_ops("");
    SELF_CHECK_EQ(ops.is_def(), true);

    ops = str2attr_post_ops("sum:2");
    SELF_CHECK_EQ(quick(1), OK);

    ops = str2attr_post_ops("sum:2+relu");
    SELF_CHECK_EQ(quick(2), OK);

    ops = str2attr_post_ops("sum:2+relu+sum:3");
    SELF_CHECK_EQ(quick(3), OK);

    ops = str2attr_post_ops("sum:2+relu+sum:3+relu");
    SELF_CHECK_EQ(quick(4), OK);

    return OK;
}

static int check_tags() {
    for (int tag_ = dnnl_format_tag_undef; tag_ != dnnl_format_tag_last;
            tag_++) {
        dnnl_format_tag_t format_tag = (dnnl_format_tag_t)tag_;
        const char *str_tag = fmt_tag2str(format_tag);
        int ndims = 1;
        for (char c = (char)('a' + DNNL_MAX_NDIMS - 1); c >= 'a'; c--) {
            if (strchr(str_tag, c)) {
                ndims = c - 'a' + 1;
                break;
            }
        }
        const dnnl_dims_t dims
                = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
        auto md_from_str = dnn_mem_t::init_md(ndims, dims, dnnl_f32, str_tag);

        dnnl_memory_desc_t md_from_tag;
        DNN_SAFE(dnnl_memory_desc_create_with_tag(
                         &md_from_tag, ndims, dims, dnnl_f32, format_tag),
                CRIT);
        int eq = dnnl_memory_desc_equal(md_from_tag, md_from_str);
        SELF_CHECK_EQ(eq, 1);

        DNN_SAFE(dnnl_memory_desc_destroy(md_from_tag), CRIT);
    }

    return OK;
}

static int check_trim_tags() {
    {
        std::string tag = "BA16a16b4a";
        std::string ndims_trimmed_tag = trim_tag(tag, 1);
        SELF_CHECK_EQ(true, ndims_trimmed_tag == "A16a4a");
    }
    {
        std::string tag = "BA16a16b4a";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 2);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "A16a");
    }
    {
        std::string tag = "abcd";
        std::string ndims_trimmed_tag = trim_tag(tag, 2);
        SELF_CHECK_EQ(true, ndims_trimmed_tag == "ab");
    }
    {
        std::string tag = "abcd";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 10);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "ab");
    }
    {
        std::string tag = "abcd";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 12);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "ab");
    }
    {
        std::string tag = "aBcd16b";
        std::string ndims_trimmed_tag = trim_tag(tag, 2);
        SELF_CHECK_EQ(true, ndims_trimmed_tag == "aB16b");
    }
    {
        std::string tag = "aBcd16b";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 2);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "A16a");
    }
    {
        std::string tag = "BA16a16b4a";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 1);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "A16a4a");
    }
    {
        std::string tag = "BADC2c16a8d16b4a";
        std::string masked_trimmed_tag = trim_tag_by_mask(tag, 14);
        SELF_CHECK_EQ(true, masked_trimmed_tag == "ACB2b8c16a");
    }

    return OK;
}

static int check_skip_impl() {
    impl_filter_t impl_filter({"gemm"}, /* use_impl = */ false,
            /* respect_global_filter = */ true);
    SELF_CHECK_EQ(true, need_next_impl("x64:gemm:jit", impl_filter));

    impl_filter = impl_filter_t({"ref", "x64:gemm"}, /* use_impl = */ false,
            /* respect_global_filter = */ true);
    SELF_CHECK_EQ(true, need_next_impl("x64:gemm:jit", impl_filter));

    impl_filter = impl_filter_t({"this_finds_nothing"}, /* use_impl = */ false,
            /* respect_global_filter = */ true);
    SELF_CHECK_EQ(false, need_next_impl("x64:gemm:jit", impl_filter));

    return OK;
}

void common() {
    RUN(check_simple_enums());
    RUN(check_attr2str());
    RUN(check_attr());
    RUN(check_post_ops2str());
    RUN(check_str2post_ops());
    RUN(check_tags());
    RUN(check_trim_tags());
    RUN(check_skip_impl());
}

} // namespace self
