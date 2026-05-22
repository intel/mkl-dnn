/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_LNORM_CONFIG_HPP
#define GPU_INTEL_LNORM_CONFIG_HPP

#include "gpu/gpu_layer_normalization_pd.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace lnorm {

using pd_t = layer_normalization_pd_t;
using fwd_pd_t = gpu_layer_normalization_fwd_pd_t;
using bwd_pd_t = gpu_layer_normalization_bwd_pd_t;

struct conf_t {
    data_type_t src_dt, dst_dt;
    data_type_t weights_data_type = data_type::f32;

    bool is_fwd;
    dim_idx_t ndims;
    dim_t norm_axis;
    dim_t across_axis;
    int norm_block;
    int num_norm_blocks;
    dim_t norm_block_fused;
    int num_norm_blocks_fused;
    int across_block;
    int num_across_blocks;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t stat_md_info;

    bool use_scale;
    bool use_shift;
    bool use_fused;
    bool calculate_stats;
    bool save_stats;
    bool vectorize_calc_stats;
    bool vectorize_bwd;
    bool vectorize_bwd_scaleshift;
    float eps;
    int sub_group_size;
    int vect_dt_n;
    int vect_size_fused;
    int shift_off;
    int n_chunk_size;
    dim_t finalize_n_chunks;
    dim_t n_chunks;
    int vector_size_scaleshift;
    bool use_src_buffer;
    bool skip_mean;
    bool require_stateless_addressing;

    attr_info_t attr_info;

    compute::dispatch_t dispatch_scaleshift;
    compute::dispatch_t dispatch_scaleshift_finalize;
    compute::dispatch_t dispatch;
    compute::dispatch_t dispatch_fused;
};

} // namespace lnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
