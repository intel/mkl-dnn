/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include <algorithm>

#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

reg_config_t make_reg_config(cpu_isa_t isa, int param_reg, int rsp_reg,
        const std::vector<int> &reserved_masks, const tile_config_t &tiles) {
    reg_config_t rc;
    rc.param_reg = param_reg;

    // Intel APX adds `r16` to `r31` where the machine has it enabled. AVX2* is
    // excluded, since VEX cannot address memory through them.
    const bool with_apx = is_superset(isa, avx512_core) && mayiuse(avx10_2);
    const int n_gpr = with_apx ? 32 : 16;
    const int n_vec = isa_num_vregs(isa);

    auto contains = [](const std::vector<int> &v, int i) {
        return std::find(v.begin(), v.end(), i) != v.end();
    };

    // GPR file includes every gpr except the stack pointer (`rsp`), the
    // argument pointer and the frame pointer (`rbp`) if requested.
    // The spill slot size is 8 bytes.
    const bool with_rbp = jit_generator_t::may_use_rbp();
    reg_file_t gpr_file;
    gpr_file.slot_size = 8;
    for (int i = 0; i < n_gpr; i++) {
        if (i == rsp_reg || i == param_reg) continue;
        if (i == Xbyak::Operand::RBP && !with_rbp) continue;
        gpr_file.regs.push_back(i);
    }

    // Vector file includes every vector register. On AVX2* a mask is a vector
    // register, so masks allocate from this same file (see `kind_to_file`
    // below) and spill like vectors. Spill slot is the vector size 32 for ymm
    // and 64 for zmm.
    reg_file_t vec_file;
    vec_file.slot_size = isa_max_vlen(isa);
    for (int i = 0; i < n_vec; i++)
        vec_file.regs.push_back(i);

    rc.pools.files = {gpr_file, vec_file};

    // Map each register kind to its files. `reg_kind_t` order is
    // { gpr, vec, mask, tile }.
    //  * gpr: file 0
    //  * vec: file 1
    //  * mask: file 1 on AVX2*, where a mask is a vector register, and file 2
    //    on AVX-512, which has a dedicated k-register file.
    //  * tile: the tile files, which follow the others (see below).
    if (is_superset(isa, avx512_core)) {
        // Mask file holds `k1` to `k7` minus `reserved_masks`. `k0` is excluded
        // because it cannot encode a write mask. A k-register is never spilled,
        // so the file has no spill slot.
        reg_file_t mask_file;
        mask_file.spillable = false;
        for (int i = 1; i < 8; i++) {
            if (!contains(reserved_masks, i)) mask_file.regs.push_back(i);
        }

        rc.pools.files.push_back(mask_file);
        rc.pools.kind_to_file = {{0, 1}, {1, 1}, {2, 1}};
    } else {
        rc.pools.kind_to_file = {{0, 1}, {1, 1}, {1, 1}};
    }

    // Tile files follow the others, one per shape, in the order the shapes
    // first appear. Each holds the tile registers configured with its shape,
    // and unused tiles are left out. A tile is never spilled, so the files
    // have no spill slot.
    std::vector<reg_file_t> &files = rc.pools.files;
    file_range_t tile_files = {(int)files.size(), 0};
    for (int t = 0; t < (int)tiles.shapes.size(); t++) {
        const tile_shape_t &shape = tiles.shapes[t];
        if (shape == tile_shape_t {}) continue;

        auto file = std::find_if(files.begin() + tile_files.first, files.end(),
                [&](const reg_file_t &f) { return f.tile_shape == shape; });
        if (file == files.end()) {
            reg_file_t tile_file;
            tile_file.spillable = false;
            tile_file.tile_shape = shape;
            file = files.insert(files.end(), tile_file);
            tile_files.n++;
        }
        file->regs.push_back(t);
    }
    rc.pools.kind_to_file.push_back(tile_files);

    return rc;
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
