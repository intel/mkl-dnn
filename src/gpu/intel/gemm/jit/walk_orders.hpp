/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_JIT_WALK_ORDERS_HPP
#define GPU_INTEL_GEMM_JIT_WALK_ORDERS_HPP

#include "common/utils.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

inline uint32_t uint32_reciprocal(uint32_t x) {
    if (x == 0) return 0;
    return (uint32_t)utils::div_up(uint64_t(0x100000000) << math::ilog2q(x), x);
}

inline void linear_order_args(compute::kernel_arg_list_t &arg_list, int &argn,
        const compute::range_t &lws, compute::range_t &gws, int32_t m,
        int32_t n, int32_t k, bool disable_hilbert,
        const gemmstone::CommonDriverInfo &info,
        const gemmstone::EvaluateAuxOutput *aux,
        const compute::device_info_t *dev_info) {
    using namespace gemmstone;

    if (info.kParallel() && info.fusedBeta()) {
        auto groups_k = uint32_t(gws[2] / lws[2]);
        arg_list.set(argn++, groups_k);
    }

    if (!info.isLinearOrder()) return;

    int m_index = info.isNMK() ? 1 : 0;
    int n_index = info.isNMK() ? 0 : 1;
    auto groups_m = uint32_t(gws[m_index] / lws[m_index]);
    auto groups_n = uint32_t(gws[n_index] / lws[n_index]);
    auto group_count = groups_m * groups_n;

    uint32_t ss_count = dev_info->eu_count() / dev_info->max_eus_per_wg();
    uint32_t thread_per_ss = dev_info->hw_threads(info.grfCount) / ss_count;
    uint32_t thread_per_tg = into<uint32_t>(lws.nelems());
    uint32_t tg_per_ss = thread_per_tg ? thread_per_ss / thread_per_tg : 0;
    uint32_t concurrent_tg = std::max(1u, tg_per_ss * ss_count);

    arg_list.set(argn++, groups_m);
    arg_list.set(argn++, groups_n);

    if (info.isSimpleLinear()) {
        uint32_t gcmn_recip
                = uint32_reciprocal(info.isMNK() ? groups_m : groups_n);
        arg_list.set(argn++, gcmn_recip);
    } else if (info.isHilbert()) {
        uint32_t vd = 0, uvd = 0;
        double ratio = double(groups_n) / double(groups_m);
        if (ratio >= 1) {
            vd = (uint32_t)std::ceil(groups_n / std::round(2 * ratio));
            uvd = groups_m * vd;
        } else {
            vd = (uint32_t)std::ceil(groups_m / std::round(2 / ratio));
            uvd = groups_n * vd;
            vd |= 0xFFFF0000u;
        }

        uint32_t uvd_recip = uint32_reciprocal(uvd);
        uint32_t bail = disable_hilbert ? 512 : 1;

        arg_list.set(argn++, vd);
        arg_list.set(argn++, uvd_recip);
        arg_list.set(argn++, bail);
    } else if (info.isBoustrophedon()) {
        double bias = double(info.wg[0] * info.unroll[0])
                / double(info.wg[1] * info.unroll[1]);
        double sm = std::sqrt(concurrent_tg / bias);
        double sn = std::sqrt(concurrent_tg * bias);

        int32_t slice = 0, thresh = 0;
        bool ok = false;

        for (bool nslice : {groups_m > groups_n, groups_m <= groups_n}) {
            double s = nslice ? sn : sm;
            auto sf = int(std::floor(s));
            auto sc = int(std::ceil(s));
            if (concurrent_tg % sc == 0) s = sf = sc;
            if (concurrent_tg % (sc + 1) == 0) s = sf = sc = sc + 1;

            int gc = nslice ? groups_n : groups_m;
            int gco = nslice ? groups_m : groups_n;

            for (int srange = 0; srange <= 2 && !ok; srange++) {
                int s0 = (srange < 2) ? sc : sf;
                bool up = (srange == 1);
                int s1 = s0 + (up ? 1 : -1);
                if (s1 <= 0) continue;

                auto rem = gc % s0;
                if (!rem || up)
                    thresh = gc / s0 - rem;
                else
                    thresh = utils::div_up(gc, s0) - (s0 - rem);

                ok = (thresh >= 0) && (gco >= 2 * nstl::max(s0, s1));
                slice = s0;
                if (!up) {
                    if (thresh > 0)
                        thresh = -thresh;
                    else {
                        slice--;
                        thresh = gc;
                    }
                }
                if (nslice) slice *= -1;
            }

            if (ok) break;
        }

        if (!ok) {
            // Fallback slicing.
            bool nslice = (groups_m > groups_n);
            double s = nslice ? sn : sm;
            int gc = nslice ? groups_n : groups_m;

            if (gc < s * 1.5)
                slice = gc;
            else
                slice = gc / utils::div_up(gc, int(std::round(s)));

            thresh = nstl::max(0, (gc / slice) - (gc % slice));
            if (nslice) slice *= -1;
        }

        if (slice == 0) {
            slice = 1;
            thresh = groups_m;
        }

        arg_list.set(argn++, slice);
        arg_list.set(argn++, thresh);
    }

    if (info.kParallelVariable()) {
        uint32_t k_parallel_start = utils::rnd_dn(group_count, concurrent_tg);
        if (aux && !aux->kParallelVariable)
            k_parallel_start
                    = group_count; /* disable variable k-slicing if indicated by kernel selector */
        if (k_parallel_start > 0 && k_parallel_start != group_count)
            k_parallel_start -= concurrent_tg;
        uint32_t k_sliced_tiles = group_count - k_parallel_start;
        uint32_t k_sliced_phases = 1; /* todo: use 2 phases where beneficial */
        uint32_t tiles_per_phase
                = utils::div_up(k_sliced_tiles, k_sliced_phases);

        int k_padding = info.kPadding(), old_k_padding = k_padding;
        auto k_padded = k;
        int64_t k_total = 0;
        uint32_t k0 = k;

        do {
            k_padded = utils::rnd_up(k + k_padding, info.unroll[LoopK]);
            k_total = int64_t(k_padded) * tiles_per_phase;
            if (k_total == 0) break;

            k0 = into<int32_t>(utils::div_up(k_total, concurrent_tg));
            k0 = utils::rnd_up(k0, info.unroll[LoopK]);

            old_k_padding = k_padding;
            k_padding = std::min<int>(k_padding, 2 * k0);
        } while (k_padding != old_k_padding);

        group_count = k_parallel_start;
        uint32_t k_parallel_groups = 0;
        uint32_t k_sync_slabs = 0;

        if (k0 > 0) {
            k_parallel_groups = uint32_t(utils::div_up(k_total, k0));
            if (k_sliced_phases > 1) k_parallel_groups = concurrent_tg;
            group_count += k_parallel_groups * k_sliced_phases;

            if (tiles_per_phase > 0) {
                k_sync_slabs = (k_parallel_groups + (tiles_per_phase >> 1))
                        / tiles_per_phase;
                if (k_sync_slabs > 0) k_sync_slabs--;
                k_sync_slabs = std::min(k_sync_slabs, (k_padded - 1) / k0);
            }
        }

        uint32_t k_unsynced_padded = k_padded - k_sync_slabs * k0;
        uint32_t k_recip = uint32_reciprocal(k_unsynced_padded);

        uint32_t kv_config = k_sliced_tiles | (k_sync_slabs << 16);
        if (k_sliced_phases > 1) kv_config |= 0x80000000u;

        arg_list.set(argn++, k0);
        arg_list.set(argn++, kv_config);
        arg_list.set(argn++, k_recip);
    }

    if (info.isPersistent()) {
        group_count = nstl::min(group_count, concurrent_tg);
        arg_list.set(argn++, group_count);
    }

    gws[0] = lws[0] * (group_count + info.extraWGs());
    gws[1] = lws[1];
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
