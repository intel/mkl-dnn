/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/post_ops.hpp"
#include "oneapi/dnnl/dnnl_types.h"

#include "gpu/intel/primitive_conf.hpp"

// Included to enable compatibility tests
#include "gpu/intel/include/dnnl_interop.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

bool memory_desc_ndims_ok(const memory_desc_t *md) {
    return md->ndims <= MAX_NDIMS;
}

memory_desc_info_t memory_desc_info_t::create(const memory_desc_wrapper &mdw) {
    using namespace format_tag;

    auto md_info = memory_desc_info_t();

    md_info.nlevels = 2;

    md_info.ndims = mdw.ndims();
    md_info.data_type = mdw.data_type();
    md_info.size = mdw.size();
    md_info.offset0 = mdw.offset0();

    auto &blk = mdw.blocking_desc();
    dim_t blk_stride = utils::array_product(blk.inner_blks, blk.inner_nblks);

    for (int d = 0; d < mdw.ndims(); ++d) {
        utils::array_set(md_info.blocks[d], 1, md_info.nlevels + 1);
        utils::array_set(md_info.strides[d], 0, md_info.nlevels + 1);
    }

    for (int d = 0; d < mdw.ndims(); ++d) {
        md_info.dims[d] = mdw.dims()[d];
        md_info.padded_dims[d] = mdw.padded_dims()[d];
        md_info.strides[d][0] = md_info.dims[d] == 1 ? 0 : blk.strides[d];
    }

    int levels[MAX_NDIMS] = {0};
    for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
        dim_t d = blk.inner_idxs[iblk];
        ++levels[d];

        md_info.blocks[d][levels[d]] = blk.inner_blks[iblk];
        blk_stride /= blk.inner_blks[iblk];
        md_info.strides[d][levels[d]] = blk_stride;
    }
    return md_info;
}

attr_info_t attr_info_t::create(const primitive_attr_t *attr) {
    const auto &po = attr->post_ops_;

    attr_info_t attr_info;

    attr_info.binary_idx = po.find(primitive_kind::binary);
    attr_info.with_binary = (attr_info.binary_idx != -1);

    // Eltwise
    attr_info.eltwise_idx = po.find(primitive_kind::eltwise);
    attr_info.with_eltwise = (attr_info.eltwise_idx != -1);

    if (attr_info.with_eltwise) {
        auto &eltwise = po.entry_[attr_info.eltwise_idx].eltwise;
        attr_info.eltwise_alg = eltwise.alg;
        attr_info.eltwise_scale = eltwise.scale;
        attr_info.eltwise_alpha = eltwise.alpha;
        attr_info.eltwise_beta = eltwise.beta;
    } else {
        attr_info.eltwise_alg = alg_kind::undef;
        attr_info.eltwise_scale = 1.0f;
        attr_info.eltwise_alpha = 1.0f;
        attr_info.eltwise_beta = 0.0f;
    }

    // Sum
    attr_info.sum_idx = po.find(primitive_kind::sum);
    attr_info.sum_scale
            = (attr_info.sum_idx != -1 ? po.entry_[attr_info.sum_idx].sum.scale
                                       : 0.0f);
    attr_info.sum_data_type = (attr_info.sum_idx != -1)
            ? po.entry_[attr_info.sum_idx].sum.dt
            : dnnl_data_type_undef;
    attr_info.with_sum
            = (attr_info.sum_idx != -1) && (attr_info.sum_scale != 0.0f);

    const auto &src_scales = attr->scales_.get(DNNL_ARG_SRC);
    attr_info.with_src_scales = !src_scales.has_default_values();
    attr_info.with_src0_scale = !src_scales.has_default_values();
    attr_info.src_scales_data_type = src_scales.get_data_type();

    const auto &src1_scales = attr->scales_.get(DNNL_ARG_SRC_1);
    attr_info.with_src1_scale = !src1_scales.has_default_values();
    if (attr_info.with_src1_scale) { gpu_assert(src1_scales.get_mask() == 0); }

    const auto &wei_scales = attr->scales_.get(DNNL_ARG_WEIGHTS);
    attr_info.with_wei_scales = !wei_scales.has_default_values();
    // TODO: remove the default `0` value.
    attr_info.wei_scales_mask
            = attr_info.with_wei_scales ? wei_scales.get_mask() : 0;
    attr_info.wei_scales_data_type = wei_scales.get_data_type();

    const auto &dst_scales = attr->scales_.get(DNNL_ARG_DST);
    attr_info.with_dst_scales = !dst_scales.has_default_values();
    attr_info.dst_scales_mask = dst_scales.get_mask();
    attr_info.dst_scales_data_type = dst_scales.get_data_type();

    // zero points
    const auto &zp = attr->zero_points_;
    attr_info.with_src_zpoints = !zp.has_default_values(DNNL_ARG_SRC);
    attr_info.with_wei_zpoints = !zp.has_default_values(DNNL_ARG_WEIGHTS);
    attr_info.with_dst_zpoints = !zp.has_default_values(DNNL_ARG_DST);
    attr_info.src_zpoints_data_type = zp.get_data_type(DNNL_ARG_SRC);
    attr_info.wei_zpoints_data_type = zp.get_data_type(DNNL_ARG_WEIGHTS);
    attr_info.dst_zpoints_data_type = zp.get_data_type(DNNL_ARG_DST);

    // host-side scalars for scales or zero-points
    attr_info.with_host_src_scale = src_scales.is_host_scalar();
    attr_info.with_host_wei_scale = wei_scales.is_host_scalar();
    attr_info.with_host_dst_scale = dst_scales.is_host_scalar();
    attr_info.with_dyn_dst_scale = dst_scales.is_dynamic();
    attr_info.with_host_src_zp = zp.get(DNNL_ARG_SRC).is_host_scalar();
    attr_info.with_host_wei_zp = zp.get(DNNL_ARG_WEIGHTS).is_host_scalar();
    attr_info.with_host_dst_zp = zp.get(DNNL_ARG_DST).is_host_scalar();

    attr_info.with_per_ic_src_zpoints = attr_info.with_src_zpoints
            && !zp.has_default_values(DNNL_ARG_SRC)
            && zp.get_mask(DNNL_ARG_SRC) > 0;

    attr_info.with_per_oc_dst_zpoints = attr_info.with_dst_zpoints
            && !zp.has_default_values(DNNL_ARG_DST)
            && zp.get_mask(DNNL_ARG_DST) > 0;

    // Rounding mode.
    attr_info.with_dst_sround = attr->rounding_mode_.get(DNNL_ARG_DST)
            == rounding_mode::stochastic;

    attr_info.initialized = true;
    return attr_info;
}

void quantization_t::define_macros(
        compute::kernel_ctx_t &kernel_ctx, const std::string &name) const {
    if (with_scale()) {
        kernel_ctx.define_int("WITH_" + name + "_SCALE", 1);
        kernel_ctx.define_int(name + "_SCALE_MASK", scale_mask());
        kernel_ctx.define_int(name + "_NUM_SCALES", num_scales());
        kernel_ctx.define_int(name + "_SCALE_GROUP", scale_group());
        kernel_ctx.define_int(name + "_SCALE_GROUP_DIM", scale_group_dim());
    }
    // Unconditionally as this defines types in kernels.
    // Note: consistent with ocl_types.hpp
    def_data_type(kernel_ctx, scale_dt(), name + "_SCALES");

    if (with_zp()) {
        kernel_ctx.define_int("WITH_" + name + "_ZPOINT", 1);
        kernel_ctx.define_int(name + "_ZPOINT_MASK", zp_mask());
        kernel_ctx.define_int(name + "_NUM_ZPOINTS", num_zps());
        kernel_ctx.define_int(name + "_ZPOINT_GROUP", zp_group());
        kernel_ctx.define_int(name + "_ZPOINT_GROUP_DIM", zp_group_dim());
    }
    // Unconditionally as this defines types in kernels.
    // Note: consistent with ocl_types.hpp
    def_data_type(kernel_ctx, zp_dt(), name + "_ZP");
}

void sum_quantization_t::define_macros(
        compute::kernel_ctx_t &kernel_ctx, const std::string &name) const {
    if (with_scale()) kernel_ctx.define_int("WITH_" + name + "_SCALE", 1);
    if (with_zp()) kernel_ctx.define_int("WITH_" + name + "_ZPOINT", 1);
}

void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        const dim_t block = block_dims[d];

        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < md.ndims()) ? block : 1);
        kernel_ctx.define_int(utils::format("%s_S%d", str, d),
                (d < md.ndims()) ? strides_compat[0][d] : 0);
        kernel_ctx.define_int(utils::format("%s_SB%d", str, d),
                (d < md.ndims()) ? strides_compat[1][d] : 0);
    }

    kernel_ctx.define_int(utils::format("%s_OFFSET_PAD", str), md.md_->offset0);
}

void set_offsets(const memory_desc_wrapper &md, dim_t offs[4][MAX_NDIMS]) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);
    const dims_t &dims = md.dims();

    for (int d = 0; d < md.ndims(); ++d) {
        const dim_t block = block_dims[d];

        offs[0][d] = block;
        offs[1][d] = strides_compat[0][d];
        offs[2][d] = strides_compat[1][d];
        offs[3][d] = dims[d];
    }
}

outer_strides_getter_t get_outer_strides(const memory_desc_wrapper &md) {
    return {md};
}

void def_offsets(const dim_t offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str,
        const dim_idx_t ndims) {

    for (dim_idx_t d = 0; d < MAX_NDIMS; d++) {
        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < ndims) ? offs[0][d] : 1);
        kernel_ctx.define_int(
                utils::format("%s_S%d", str, d), (d < ndims) ? offs[1][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_SB%d", str, d), (d < ndims) ? offs[2][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_D%d", str, d), (d < ndims) ? offs[3][d] : 1);
    }
}

void def_block_offsets(const block_layout_t &layout,
        compute::kernel_ctx_t &kernel_ctx, const char *str) {

    for (const block_t &b : layout) {
        kernel_ctx.define_int(utils::format("%s_B%d", str, b.dim_idx), b.block);
        kernel_ctx.define_int(
                utils::format("%s_SB%d", str, b.dim_idx), int64_t(b.stride));
    }
}

const char *get_type_name(data_type_t dt) {
    switch (dt) {
        case data_type::undef: return "undef_data";
        case data_type::bf16: return "bf16";
        case data_type::f16: return "half";
        case data_type::f32: return "float";
        case data_type::f64: return "double";
        case data_type::s8: return "char";
        case data_type::u8: return "uchar";
        case data_type::f8_e4m3: return "f8_e4m3";
        case data_type::f8_e5m2: return "f8_e5m2";
        case data_type::f4_e2m1: return "f4_e2m1";
        case data_type::e8m0: return "e8m0";
        case data_type::s4: return "s4";
        case data_type::u4: return "u4";
        case data_type::u2: return "u2";
        case data_type::u3: return "u3";
        case data_type::s32: return "int";
        case data_type::s64: return "long";
        default:
            gpu_error_not_expected()
                    << "Unexpected data type " << dnnl_dt2str(dt);
            return "invalid";
    }
}

void def_data_type(
        compute::kernel_ctx_t &kernel_ctx, data_type_t dt, const char *str) {
    const char *name = get_type_name(dt);

    switch (dt) {
        case data_type::undef:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=undef_data -D%s_DT_UNDEF", str, str));
            break;
        case data_type::bf16:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_BF16", str, name, str));
            break;
        case data_type::f16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=half -D%s_DT_F16", str, str));
            break;
        case data_type::f32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=float -D%s_DT_F32", str, str));
            break;
        case data_type::f64:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=double -D%s_DT_F64", str, str));
            break;
        case data_type::s8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=char -D%s_DT_S8", str, str));
            break;
        case data_type::u8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=uchar -D%s_DT_U8", str, str));
            break;
        case data_type::f8_e4m3:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_HF8", str, name, str));
            break;
        case data_type::f8_e5m2:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_BF8", str, name, str));
            break;
        case data_type::f4_e2m1:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_F4_E2M1", str, name, str));
            break;
        case data_type::e8m0:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_E8M0", str, name, str));
            break;
        case data_type::s4:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=%s -D%s_DT_S4", str, name, str));
            break;
        case data_type::u4:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=%s -D%s_DT_U4", str, name, str));
            break;
        case data_type::u2:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=%s -D%s_DT_U2", str, name, str));
            break;
        case data_type::u3:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=%s -D%s_DT_U3", str, name, str));
            break;
        case data_type::s32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=int -D%s_DT_S32", str, str));
            break;
        case data_type::s64:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=int -D%s_DT_S64", str, str));
            break;
        default:
            gpu_error_not_expected()
                    << "Unexpected data type " << dnnl_dt2str(dt);
            break;
    }
}
void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const std::string &str) {
    return def_data_type(kernel_ctx, dt, str.c_str());
}

void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_info_t &md_info, const char *prefix) {
    def_data_type(kernel_ctx, md_info.data_type, prefix);
    kernel_ctx.register_buffer_size(md_info);

    kernel_ctx.define_int(utils::format("%s_OFFSET0", prefix), md_info.offset0);
    kernel_ctx.define_int(utils::format("%s_NDIMS", prefix), md_info.ndims);

    kernel_ctx.define_int(utils::format("%s_NLEVELS", prefix), md_info.nlevels);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        dim_t dim = (d < md_info.ndims) ? md_info.dims[d] : 1;
        dim_t padded_dim = (d < md_info.ndims) ? md_info.padded_dims[d] : 1;
        kernel_ctx.define_int(utils::format("%s_D%d", prefix, d), dim);
        kernel_ctx.define_int(utils::format("%s_PD%d", prefix, d), padded_dim);

        for (int l = 0; l < md_info.nlevels + 1; ++l) {
            dim_t block = (d < md_info.ndims) ? md_info.blocks[d][l] : 1;
            dim_t stride = (d < md_info.ndims) ? md_info.strides[d][l] : 0;
            kernel_ctx.define_int(
                    utils::format("%s_B%d_%d", prefix, d, l), block);
            if (stride != DNNL_RUNTIME_DIM_VAL)
                kernel_ctx.define_int(
                        utils::format("%s_S%d_%d", prefix, d, l), stride);
            else
                kernel_ctx.add_option(utils::format(
                        "%s_S%d_%d=invalid_stride", prefix, d, l));
        }
    }
}

bool post_ops_with_binary_ok(const primitive_attr_t *attr,
        const memory_desc_t &dst_md, const int max_ndims_supported) {
    const auto &p = attr->post_ops_;
    const auto dst_dt = dst_md.data_type;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false, false); };
    auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };
    auto is_prelu = [&](int idx) { return p.entry_[idx].is_prelu(); };

    bool is_po_ok = true;
    for (int po_idx = 0; po_idx < p.len(); ++po_idx) {
        is_po_ok = is_po_ok
                && (is_eltwise(po_idx) || is_sum(po_idx) || is_binary(po_idx)
                        || is_prelu(po_idx));
        if (is_binary(po_idx)) {
            const auto &bin_desc = p.entry_[po_idx].binary.src1_desc;
            bool has_runtime_dims = false;
            int num_size_one_dims = 0;
            for (int dim_idx = 0; dim_idx < bin_desc.ndims; dim_idx++) {
                if (dim_idx < max_ndims_supported) {
                    if (bin_desc.dims[dim_idx] == DNNL_RUNTIME_DIM_VAL)
                        has_runtime_dims = true;
                    else if (bin_desc.dims[dim_idx] == 1)
                        num_size_one_dims++;
                } else {
                    // accept descriptor if unsupported dims are equal to 1.
                    if (bin_desc.dims[dim_idx] != 1) is_po_ok = false;
                }
            }

            // Only 1D runtime dimensions are supported
            if (has_runtime_dims && num_size_one_dims != dst_md.ndims - 1)
                is_po_ok = false;
        }
        if (is_sum(po_idx)) {
            if (p.entry_[po_idx].sum.dt != dnnl_data_type_undef
                    && types::data_type_size(p.entry_[po_idx].sum.dt)
                            != types::data_type_size(dst_dt))
                return false;
        }
    }

    if (p.len() > MAX_POST_OPS_SUPPORTED) is_po_ok = false;

    return is_po_ok;
}

status_t get_prelu_md(int prelu_mask, const dim_t *dst_dims,
        memory_desc_t &weight_mem_desc, int weight_ndims) {
    format_tag_t weights_tag;
    dims_t weight_dims {};
    for (int d = 0; d < weight_ndims; ++d) {
        if (((prelu_mask >> d) & 0x1) == 1) {
            weight_dims[d] = dst_dims[d];
        } else {
            weight_dims[d] = 1;
        }
    }
    switch (weight_ndims) {
        case 1: weights_tag = format_tag_t::dnnl_a; break;
        case 2: weights_tag = format_tag_t::dnnl_ab; break;
        case 3: weights_tag = format_tag_t::dnnl_acb; break;
        case 4: weights_tag = format_tag_t::dnnl_acdb; break;
        case 5: weights_tag = format_tag_t::dnnl_acdeb; break;
        default: weights_tag = format_tag_t::dnnl_format_tag_undef; break;
    }
    CHECK(memory_desc_init_by_tag(weight_mem_desc, weight_ndims, weight_dims,
            data_type_t::dnnl_f32, weights_tag));
    return status::success;
}

status_t def_post_ops_cfg(compute::kernel_ctx_t &kernel_ctx,
        const post_ops_t &post_ops, const memory_desc_t &dst_md) {
    std::string po_kernel_args = "-DPOST_OP_ARGS=\"";

    bool post_op_uses_bf16 = false;
    bool post_op_uses_bf8 = false;
    bool post_op_uses_hf8 = false;

    auto set_post_op_uses = [&](data_type_t type) {
        post_op_uses_bf16 |= (type == data_type::bf16);
        post_op_uses_bf8 |= (type == data_type::f8_e5m2);
        post_op_uses_hf8 |= (type == data_type::f8_e4m3);
    };

    auto define_float = [&](const std::string &name, float value,
                                std::initializer_list<float> inlines = {}) {
        for (float v : inlines) {
            if (v == value) {
                kernel_ctx.define_float(name.c_str(), value);
                return;
            }
        }
        po_kernel_args += std::string(", float " + name);
    };
    auto define_int = [&](const std::string &name, int value,
                              std::initializer_list<int> inlines = {}) {
        for (int v : inlines) {
            if (v == value) {
                kernel_ctx.define_int(name.c_str(), value);
                return;
            }
        }
        po_kernel_args += std::string(", int " + name);
    };

    auto add_po_defines = [&](const std::string &bin_arg_name,
                                  const post_ops_t::entry_t &e, int idx_) {
        std::string idx = std::to_string(idx_);
        if (e.is_binary() || e.is_prelu()) {
            post_op::relative_md_t src_rmd;
            if (e.is_binary()) {
                kernel_ctx.register_buffer_size(e.binary.src1_desc);
                kernel_ctx.define_int("PO_" + idx + "_ALG", e.binary.alg);
                CHECK(post_op::relative_md_t::make(
                        src_rmd, e.binary.src1_desc, {}));
            } else {
                kernel_ctx.define_int(
                        "PO_" + idx + "_ALG", alg_kind_t::dnnl_eltwise_relu);
                memory_desc_t weight_mem_desc;
                CHECK(get_prelu_md(e.prelu.mask, dst_md.dims, weight_mem_desc,
                        dst_md.ndims));
                kernel_ctx.register_buffer_size(weight_mem_desc);
                CHECK(post_op::relative_md_t::make(
                        src_rmd, weight_mem_desc, {}));
            }

            std::array<std::string, MAX_NDIMS> stride_vars;
            for (int i = 0; i < dst_md.ndims; i++) {
                stride_vars[i] = "po" + idx + "_stride" + std::to_string(i);
            }
            kernel_ctx.add_option(src_rmd.ocl_defines(
                    "PO_" + idx, stride_vars, dst_md.ndims));
            set_post_op_uses(src_rmd.dt);

            po_kernel_args += std::string(", const __global ")
                    + get_type_name(src_rmd.dt) + " *po" + idx + "_binary_arg";
            for (int i = 0; i < dst_md.ndims; i++) {
                if (!src_rmd.is_broadcast(i, dst_md.ndims)
                        && !src_rmd.is_inner_dim(i, dst_md.ndims))
                    po_kernel_args += std::string(", dim_t " + stride_vars[i]);
            }

            if (!e.is_binary_with_ternary_op()) {
                kernel_ctx.add_option("-DAPPLY_PO_" + idx + "=APPLY_PO_BINARY");
            } else {
                kernel_ctx.add_option(
                        "-DAPPLY_PO_" + idx + "=APPLY_PO_TERNARY");

                post_op::relative_md_t src2_rmd;
                kernel_ctx.register_buffer_size(e.binary.src2_desc);
                CHECK(post_op::relative_md_t::make(
                        src2_rmd, e.binary.src2_desc, {}));

                std::array<std::string, MAX_NDIMS> stride2_vars;
                for (int i = 0; i < dst_md.ndims; i++) {
                    stride2_vars[i]
                            = "po" + idx + "_stride2_" + std::to_string(i);
                }
                kernel_ctx.add_option(src2_rmd.ocl_defines(
                        "PO2_" + idx, stride2_vars, dst_md.ndims));
                set_post_op_uses(src2_rmd.dt);

                po_kernel_args += std::string(", const __global ")
                        + get_type_name(src2_rmd.dt) + " *po" + idx
                        + "_binary_arg2";
                for (int i = 0; i < dst_md.ndims; i++) {
                    if (!src2_rmd.is_broadcast(i, dst_md.ndims)
                            && !src2_rmd.is_inner_dim(i, dst_md.ndims))
                        po_kernel_args
                                += std::string(", dim_t " + stride2_vars[i]);
                }
            }
        } else if (e.is_eltwise()) {
            define_float("po" + idx + "_alpha", e.eltwise.alpha);
            define_float("po" + idx + "_beta", e.eltwise.beta);
            define_float("po" + idx + "_scale", e.eltwise.scale, {0, 1});

            kernel_ctx.add_option("-DAPPLY_PO_" + idx + "=APPLY_PO_ELTWISE");
            kernel_ctx.define_int("PO_" + idx + "_ALG", e.eltwise.alg);
        } else if (e.is_sum(false, false)) {
            define_int("po" + idx + "_zp", e.sum.zero_point, {0});
            define_float("po" + idx + "_scale", e.sum.scale, {0, 1});

            kernel_ctx.add_option("-DAPPLY_PO_" + idx + "=APPLY_PO_SUM");
            kernel_ctx.define_int("PO_" + idx + "_ALG", alg_kind::undef);
            set_post_op_uses(e.sum.dt);
        } else {
            return status::runtime_error;
        }
        return status::success;
    };

    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const std::string bin_arg_name
                = "PO_" + std::to_string(idx) + "_BIN_ARG";
        CHECK(add_po_defines(bin_arg_name, post_ops.entry_[idx], idx));
    }

    kernel_ctx.define_int("POST_OP_CHAIN_LENGTH", post_ops.len());
    if (post_op_uses_bf16) kernel_ctx.define_int("POST_OP_USING_BF16", 1);
    if (post_op_uses_bf8) kernel_ctx.define_int("POST_OP_USING_BF8", 1);
    if (post_op_uses_hf8) kernel_ctx.define_int("POST_OP_USING_HF8", 1);

    po_kernel_args += "\"";
    kernel_ctx.add_option(po_kernel_args);
    return status::success;
}

int append_post_ops_to_arg_list_base(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw) {
    auto set_arg_entry = [&](const post_ops_t::entry_t &e, int po_idx) {
        if (e.is_binary() || e.is_prelu()) {
            auto arg = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx)
                    | (e.is_binary() ? DNNL_ARG_SRC_1 : DNNL_ARG_WEIGHTS));
            gpu_assert(arg.is_const());

            auto &binary_arg = arg.mem()
                    ? *(arg.mem()->memory_storage())
                    : dnnl::impl::memory_storage_t::empty_storage();
            arg_list.set(post_op_idx++, binary_arg);

            post_op::relative_md_t src_rmd;
            memory_desc_t src;
            if (e.is_binary()) {
                src = e.binary.src1_desc;
                status_t status = post_op::relative_md_t::make(
                        src_rmd, e.binary.src1_desc, {});
                gpu_assert(status == status::success);
            } else {
                status_t status = get_prelu_md(
                        e.prelu.mask, dst_mdw.dims(), src, dst_mdw.ndims());
                gpu_assert(status == status::success);
                status = post_op::relative_md_t::make(src_rmd, src, {});
                gpu_assert(status == status::success);
            }

            memory_desc_wrapper src_mdw = src;
            for (int i = 0; i < src_mdw.ndims(); i++) {
                if (!src_rmd.is_broadcast(i, src_mdw.ndims())
                        && !src_rmd.is_inner_dim(i, src_mdw.ndims()))
                    arg_list.set(post_op_idx++, src_mdw.strides()[i]);
            }

            // src2 arg for ternary ops (binary_select).
            if (e.is_binary_with_ternary_op()) {
                auto arg2 = args.at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx)
                        | DNNL_ARG_SRC_2);
                gpu_assert(arg2.is_const());
                auto &binary_arg2 = arg2.mem()
                        ? *(arg2.mem()->memory_storage())
                        : dnnl::impl::memory_storage_t::empty_storage();
                arg_list.set(post_op_idx++, binary_arg2);

                post_op::relative_md_t src2_rmd;
                status_t status = post_op::relative_md_t::make(
                        src2_rmd, e.binary.src2_desc, {});
                gpu_assert(status == status::success);
                memory_desc_wrapper src2_mdw = e.binary.src2_desc;
                for (int i = 0; i < src2_mdw.ndims(); i++) {
                    if (!src2_rmd.is_broadcast(i, src2_mdw.ndims())
                            && !src2_rmd.is_inner_dim(i, src2_mdw.ndims()))
                        arg_list.set(post_op_idx++, src2_mdw.strides()[i]);
                }
            }
        } else if (e.is_eltwise()) {
            arg_list.set(post_op_idx++, e.eltwise.alpha);
            arg_list.set(post_op_idx++, e.eltwise.beta);
            if (!utils::one_of(e.eltwise.scale, 0, 1))
                arg_list.set(post_op_idx++, e.eltwise.scale);
        } else if (e.is_sum(false, false)) {
            if (e.sum.zero_point != 0) {
                arg_list.set(post_op_idx++, e.sum.zero_point);
            }
            if (!utils::one_of(e.sum.scale, 0, 1)) {
                arg_list.set(post_op_idx++, e.sum.scale);
            }
        }
    };

    for (int idx = 0; idx < post_ops.len(); ++idx) {
        set_arg_entry(post_ops.entry_[idx], idx);
    }
    return post_op_idx;
}
int append_post_ops_to_arg_list(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw) {
    exec_args_t args;
    return append_post_ops_to_arg_list_base(
            ctx.args(), arg_list, post_op_idx, post_ops, dst_mdw);
}

bool post_ops_preserves_zeroes(
        const exec_ctx_t &ctx, const post_ops_t &post_ops) {
    bool preserve_zeroes = true;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const post_ops_t::entry_t &po_entry = post_ops.entry_[idx];
        if (po_entry.is_binary()) {
            // only binary mul is preserving zeroes
            preserve_zeroes &= utils::one_of(po_entry.binary.alg,
                    dnnl::impl::alg_kind_t::dnnl_binary_mul,
                    dnnl::impl::alg_kind_t::dnnl_binary_mul_inplace);
        }
        if (po_entry.is_eltwise(false)) {
            preserve_zeroes &= gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                    po_entry.eltwise.alg, po_entry.eltwise.alpha,
                    po_entry.eltwise.beta);
        }
    }
    return preserve_zeroes;
}

status_t def_attr_info_impl(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md) {
    gpu_assert(attr_info.initialized);

    kernel_ctx.define_int("WITH_POST_OP", post_ops.len() > 0);

    if (!kernel_ctx.has_macro("ELTWISE_ALG"))
        kernel_ctx.define_int("ELTWISE_ALG", attr_info.eltwise_alg);

    kernel_ctx.define_int("WITH_SUM", attr_info.with_sum);
    kernel_ctx.define_int("SUM_IDX", attr_info.sum_idx);
    kernel_ctx.define_float("SUM_SCALE", attr_info.sum_scale);
    kernel_ctx.define_int("SUM_SCALE1", attr_info.sum_scale == 1.0f);

    kernel_ctx.define_int("WITH_SRC0_SCALE", attr_info.with_src0_scale);
    kernel_ctx.define_int("WITH_SRC1_SCALE", attr_info.with_src1_scale);

    kernel_ctx.define_int("WITH_SRC_SCALES", attr_info.with_src_scales);
    kernel_ctx.define_int("WITH_WEI_SCALES", attr_info.with_wei_scales);
    kernel_ctx.define_int("WITH_DST_SCALES", attr_info.with_dst_scales);
    kernel_ctx.define_int("WEI_SCALES_MASK", attr_info.wei_scales_mask);
    kernel_ctx.define_int("DST_SCALES_MASK", attr_info.dst_scales_mask);
    def_data_type(kernel_ctx, attr_info.src_scales_data_type, "SRC_SCALES");
    def_data_type(kernel_ctx, attr_info.wei_scales_data_type, "WEI_SCALES");
    def_data_type(kernel_ctx, attr_info.dst_scales_data_type, "DST_SCALES");

    kernel_ctx.define_int("WITH_SRC_ZPOINTS", attr_info.with_src_zpoints);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS", attr_info.with_wei_zpoints);
    kernel_ctx.define_int("WITH_DST_ZPOINTS", attr_info.with_dst_zpoints);
    kernel_ctx.define_int(
            "WITH_SRC_ZPOINTS_PER_IC", attr_info.with_per_ic_src_zpoints);
    kernel_ctx.define_int(
            "WITH_DST_ZPOINTS_PER_OC", attr_info.with_per_oc_dst_zpoints);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS_DT_S8",
            attr_info.wei_zpoints_data_type == dnnl_s8);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS_DT_U8",
            attr_info.wei_zpoints_data_type == dnnl_u8);

    kernel_ctx.define_int("WITH_HOST_SRC_ZP", attr_info.with_host_src_zp);
    kernel_ctx.define_int("WITH_HOST_WEI_ZP", attr_info.with_host_wei_zp);
    kernel_ctx.define_int("WITH_HOST_DST_ZP", attr_info.with_host_dst_zp);
    kernel_ctx.define_int("WITH_HOST_SRC_SCALE", attr_info.with_host_src_scale);
    kernel_ctx.define_int("WITH_HOST_WEI_SCALE", attr_info.with_host_wei_scale);
    kernel_ctx.define_int("WITH_HOST_DST_SCALE", attr_info.with_host_dst_scale);
    kernel_ctx.define_int("WITH_DYN_DST_SCALE", attr_info.with_dyn_dst_scale);
    kernel_ctx.define_int("WITH_MX_DST_SCALE", attr_info.with_dyn_dst_scale);

    return def_post_ops_cfg(kernel_ctx, post_ops, dst_md);
}

status_t def_attr_info(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md) {
    return def_attr_info_impl(kernel_ctx, attr_info, post_ops, dst_md);
}

void def_dispatch(compute::kernel_ctx_t &kernel_ctx,
        const compute::dispatch_t &dispatch) {
    dispatch.def_kernel_macros(kernel_ctx);
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
