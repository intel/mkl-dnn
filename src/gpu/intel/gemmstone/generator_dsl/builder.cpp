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

#include "dsl/ir/fma.hpp"
#include "dsl/ir/pass/dpas.hpp"
#include "dsl/ir/pass/simplify.hpp"
#include "dsl/utils/logging.hpp"
#include "dsl/utils/utils.hpp"
#include "gemmstone/config.hpp"
#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/strategy.hpp"
#include "generator_dsl/kernel_desc.hpp"

GEMMSTONE_NAMESPACE_START

using namespace dsl;

struct transform_t {
    // Sample transforms on bf16 data with pack_size 16:
    // none:           64a64b -> 64a64b
    // block:          64a64b -> 4b64a16b
    // vnni:           64a64b -> 4b32a16b2a
    // transpose_vnni: 64a64b -> 4a32b16a2b
    enum class kind_t { none, block, vnni, transpose_vnni };

    transform_t() = default;

    transform_t(kind_t t_kind, int pack_size, ngen::CacheSettingsLSC cache_hint,
            std::array<idx_t, 2> dims)
        : kind(t_kind)
        , pack_size(pack_size)
        , cache_hint(to_ir(cache_hint))
        , dims(std::move(dims)) {}

    layout_t get_layout(const tile_t &sizes, type_t type) const {

        auto col_var = dims[0];
        auto col = sizes[dims[0]];
        auto row_var = dims[1];
        auto row = sizes[dims[1]];
        auto t = type.size();

        auto normalized = kind;
        if (normalized == kind_t::transpose_vnni) {
            std::swap(col_var, row_var);
            std::swap(col, row);
            normalized = kind_t::vnni;
        }

        if (normalized == kind_t::vnni && t >= 4) normalized = kind_t::block;

        int col_inner = pack_size ? pack_size : grf_size();
        if (normalized == kind_t::block && col <= col_inner)
            normalized = kind_t::none;

        switch (normalized) {
            case kind_t::none:
                return layout_t(type, {{col_var, col, 1}, {row_var, row, col}});

            case kind_t::block: {
                int col_outer = (int)(col / col_inner);
                return layout_t(type,
                        {{col_var, col_inner, 1}, {row_var, row, col_inner},
                                {col_var, col_outer, row * col_inner}});
            }

            case kind_t::vnni: {
                int row_inner = 4 / t;
                int row_outer = (int)(row / row_inner);
                int col_outer = (int)(col / col_inner);
                return layout_t(type,
                        {{row_var, row_inner, 1},
                                {col_var, col_inner, row_inner},
                                {row_var, row_outer, col_inner * row_inner},
                                {col_var, col_outer,
                                        row_outer * col_inner * row_inner}});
            }

            // Impossible to hit due to normalization
            case kind_t::transpose_vnni:
            default: stub(); return {};
        }
    }

    static send_cache_hint_t to_ir(ngen::CacheSettingsLSC hint) {
        switch (hint) {
            case ngen::CacheSettingsLSC::L1C_L3C:
                return send_cache_hint_t::load_once;
            case ngen::CacheSettingsLSC::Default:
                return send_cache_hint_t::hw_default;
            default: stub(); return send_cache_hint_t::undef;
        }
    }

    kind_t kind = kind_t::none;
    int pack_size = 0;
    send_cache_hint_t cache_hint = send_cache_hint_t::undef;
    std::array<idx_t, 2> dims = {};
};

static const idx_t m_var("m");
static const idx_t n_var("n");
static const idx_t k_var("k");

struct kloop_iterator_t {

    virtual const global_tensor_t &A_prefetch() const = 0;
    virtual const global_tensor_t &A_load() const = 0;
    virtual const global_tensor_t &B_prefetch() const = 0;
    virtual const global_tensor_t &B_load() const = 0;
    virtual const global_tensor_t &C_store() const = 0;

    virtual void A_prefetch_inc(int64_t k_block) = 0;
    virtual void A_load_inc(int64_t k_block) = 0;

    virtual void B_prefetch_inc(int64_t k_block) = 0;
    virtual void B_load_inc(int64_t k_block) = 0;

    virtual void kloop_inc(int64_t k_block) = 0;

    virtual expr_t update_C() const = 0;

    // Returns whether the given increment is in bounds
    virtual expr_t is_inbounds(int64_t increment) const = 0;
};

transform_t get_transform(const MatrixAddressingStrategy &matrix_strategy,
        std::array<idx_t, 2> dims, bool is_prefetch = false) {
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            // TODO: Remove workaround unimplemented scattered->vnni support.
            if (is_prefetch)
                return transform_t(transform_t::kind_t::none, 0,
                        matrix_strategy.cachingR, dims);

            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);

        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block2D: {
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        };
        case AccessType::Block2DVNNI: {
            return transform_t(transform_t::kind_t::vnni, matrix_strategy.tileR,
                    matrix_strategy.cachingR, dims);
        }
        default: stub(); return {};
    }
}
static void multiply(const tensor_t &C, const tensor_t &A, const tensor_t &B,
        const tile_t &tile, const icoord_t &base, bool is_systolic) {
    using ir::dpas_t;
    if (is_systolic) {
        int64_t simd = 16;
        int64_t sdepth = 8;
        int64_t max_rcount = 8;

        auto C_layout = C.layout();
        auto simd_idx = C_layout[0].idx;
        auto sdepth_idx = A.layout()[0].idx == C_layout[0].idx
                ? A.layout()[1].idx
                : A.layout()[0].idx;
        auto rcount_idx = C_layout[1].idx;
        auto sdepth_pack = 4 / A.type().size();

        tile_t inst_tile {{simd_idx, simd}, {sdepth_idx, sdepth * sdepth_pack},
                {rcount_idx, max_rcount}};

        dsl_assert(tile[simd_idx] % simd == 0);
        dsl_assert(tile[sdepth_idx] % (sdepth_pack * sdepth) == 0);
        dsl_assert(C_layout[0].size == simd);

        std::vector<layout::block_t> dpas_blocks;
        dpas_blocks.emplace_back(sdepth_idx, tile[sdepth_idx]);
        for (auto &b : C_layout.blocks()) {
            dpas_blocks.emplace_back(b.idx, b.size);
        }
        auto dpas_layout = C_layout.with(dpas_blocks);

        std::vector<stmt_t> dpas_stmts;
        for (auto &coord : dpas_layout.iter(inst_tile)) {
            int simd = (int)inst_tile[simd_idx];
            auto sdepth = inst_tile[sdepth_idx] / sdepth_pack;
            auto rcount = std::min(inst_tile[rcount_idx],
                    tile[rcount_idx] - coord[rcount_idx]);

            auto dpas = dpas_t::make(false, simd, into<uint8_t>(sdepth),
                    into<uint8_t>(rcount), C.type(), B.type(), A.type());
            // FIXME: This code can access out-of-bounds coordinates, adding
            // modulus to keep the old behavior with v2 layout.
            auto get_coord = [](const tensor_t &t, icoord_t coord) {
                for (auto &d : coord) {
                    coord[d] = coord[d] % t.tile().get(d, coord[d] + 1);
                }
                return coord;
            };
            auto dst = C.subvec(base + coord, 1);
            auto src1 = A.subvec(get_coord(A, base + coord), 1);
            auto src2 = B.subvec(get_coord(B, base + coord), 1);
            dpas_stmts.emplace_back(dpas.as<dpas_t>()(dst, dst, src1, src2));
        }
        append(inject_dpas_atomic(ir::stmt_seq_t::make(dpas_stmts),
                /*filter_by_label=*/false));
    } else {
        auto max_simd = 32;

        auto C_layout = C.layout();
        auto simd_idx = C_layout[0].idx;
        auto rcount_idx = C_layout[1].idx;
        const auto &m_idx = simd_idx;
        const auto &n_idx = rcount_idx;
        auto k_idx = one_of(A.layout()[1].idx, {simd_idx, rcount_idx})
                ? A.layout()[0].idx
                : A.layout()[1].idx;

        tile_t inst_tile {{{simd_idx, max_simd}, {rcount_idx, 1}, {k_idx, 1}}};

        int M = (int)inst_tile.get(m_idx, 1);
        int N = (int)inst_tile.get(n_idx, 1);
        int K = (int)inst_tile.get(k_idx, 1);
        bool is_a_bcast = (M * K == 1);
        bool is_b_bcast = (K * N == 1);
        int a_stride = is_a_bcast ? 0 : int(A.layout().stride(m_idx));
        int b_stride = is_b_bcast ? 0 : int(B.layout().stride(n_idx));

        std::vector<layout::block_t> mad_blocks;
        mad_blocks.emplace_back(k_idx, tile[k_idx]);
        for (auto &b : C_layout.blocks()) {
            mad_blocks.emplace_back(b.idx, b.size);
        }
        auto mad_layout = C_layout.with(mad_blocks);

        dsl_assert(tile[simd_idx] * C.type().size() % grf_size() == 0);
        for (auto &coord : mad_layout.iter(inst_tile)) {
            int simd = (int)std::min(
                    inst_tile[simd_idx], tile[simd_idx] - coord[simd_idx]);

            auto mad = ir::mad_t::make(get_hw(), C.type(), simd, A.type(),
                    a_stride, B.type(), b_stride);

            auto dst = C.subvec(base + coord, 1);
            auto src1 = A.subvec(base + coord, 1);
            auto src2 = B.subvec(base + coord, 1);

            append(mad.as<ir::mad_t>()(dst, dst, src1, src2));
        }
    }
}

idx_map_t<expr_t> get_strides(
        MatrixLayout layout, std::array<idx_t, 2> pvars, expr_t ld) {
    switch (layout) {
        case MatrixLayout::N: return {{pvars[0], 1}, {pvars[1], ld}};
        case MatrixLayout::T: return {{pvars[0], ld}, {pvars[1], 1}};
        default: stub(); return {};
    };
}

struct tensor_config_t {
    tensor_config_t(const global_tensor_t &g, transform_t t, int copies)
        : transform(t) {
        tile = g.tile();
        layout = t.get_layout(g.tile(), g.scalar_type());
        layout = layout.with_block({k_var, copies});
    }

    tile_t tile;
    layout_t layout;

    transform_t transform;
};

#ifndef STANDALONE
void apply_post_ops(const dnnl::impl::gpu::intel::gpu_post_ops_t &ops,
        const tensor_t &C, const std::vector<expr_t> &idxs,
        const std::vector<idx_t> &dims) {
    for (size_t i = 0; i < ops.len(); i++) {
        if (ops[i].is_eltwise()) {
            stub();
        } else if (ops[i].is_sum()) {
            stub();
        } else if (ops[i].is_binary()) {
            auto &e = ops[i].as_binary();
            std::string i_s = std::to_string(i);
            std::string stride_prefix = "binary" + i_s + "_stride";

            int ndims = (int)dims.size();
            idx_map_t<int> dim_to_md;
            for (int i = 0; i < ndims; i++) {
                dim_to_md[dims[i]] = i;
            };

            std::vector<expr_t> strides;
            strides.reserve(idxs.size());
            for (unsigned int j = 0; j < idxs.size(); j++) {
                if (e.src1_desc.is_inner_dim(j, ndims)) {
                    strides.emplace_back(1);
                } else if (e.src1_desc.is_broadcast(j, ndims)) {
                    strides.emplace_back(0);
                } else {
                    strides.emplace_back(
                            arg(stride_prefix + std::to_string(j), true));
                }
            }

            auto src_g = [&]() -> global_tensor_t {
                expr_t src_g_offset = ir::simplify(arg("offset_binary" + i_s)
                        + e.src1_desc.get_offset(idxs, strides));

                idx_map_t<expr_t> g_strides;
                idx_map_t<expr_t> g_sizes;
                for (int i = 0; i < ndims; i++) {
                    g_strides[dims[i]] = strides[i];
                    g_sizes[dims[i]] = e.src1_desc.is_broadcast(i, ndims)
                            ? expr_t(1)
                            : expr_t(0); //TODO: Get actual size
                }

                return {arg("binary" + i_s), src_g_offset, coord_t(), g_sizes,
                        g_strides, {}};
            }();

            layout_t src_layout = {src_g.scalar_type()};
            auto C_layout = C.layout();
            for (auto &b : C_layout.blocks()) {
                if (!e.src1_desc.is_broadcast(dim_to_md[b.idx], ndims)) {
                    src_layout = src_layout.with_block({b.idx, b.size});
                } else {
                    src_layout = src_layout.with_block({b.idx, 1});
                }
            }

            tensor_t src = def("binary" + i_s + "_blk", src_layout);
            std::cout << "src_g: " << src_g.str() << "\n";
            std::cout << "src: " << src.str() << "\n";
            load(src, src_g);

            switch (e.alg) {
                case dnnl::impl::alg_kind::binary_add: C += src; break;
                default: stub();
            }

        } else {
            stub();
        }
    }
}
#endif

// Basic iterator with no iteration over m and n.
struct basic_iterator_t : kloop_iterator_t {
    basic_iterator_t(const global_tensor_t &A, int A_prefetch_k_blk,
            int A_load_k_blk, const global_tensor_t &B, int B_prefetch_k_blk,
            int B_load_k_blk, const global_tensor_t &C)
        : m_idx_ {C.coord()[m_var]}
        , m_(C.sizes()[m_var])
        , n_idx_ {C.coord()[n_var]}
        , n_(C.sizes()[n_var])
        , k_idx_ {A.coord()[k_var]}
        , k_ {A.sizes()[k_var]}
        , A_prefetch_ {A.buf(), A.base_offset(), A.coord(), A.sizes(),
                  A.strides(),
                  tile_t {{m_var, C.tile()[m_var]}, {k_var, A_prefetch_k_blk}}}
        , A_load_ {A.buf(), A.base_offset(), A.coord(), A.sizes(), A.strides(),
                  tile_t {{m_var, C.tile()[m_var]}, {k_var, A_load_k_blk}}}
        , B_prefetch_ {B.buf(), B.base_offset(), B.coord(), B.sizes(),
                  B.strides(),
                  tile_t {{k_var, B_prefetch_k_blk}, {n_var, C.tile()[n_var]}}}
        , B_load_ {B.buf(), B.base_offset(), B.coord(), B.sizes(), B.strides(),
                  tile_t {{k_var, B_load_k_blk}, {n_var, C.tile()[n_var]}}}
        , C_store_ {C}

    {
        assume(m_idx_ % C.tile()[m_var] == 0);
        assume(n_idx_ % C.tile()[n_var] == 0);

        assume(m_idx_ >= 0);
        assume(n_idx_ >= 0);
        assume(k_idx_ >= 0);
    }

    const global_tensor_t &A_prefetch() const override { return A_prefetch_; }
    const global_tensor_t &A_load() const override { return A_load_; }
    const global_tensor_t &B_prefetch() const override { return B_prefetch_; }
    const global_tensor_t &B_load() const override { return B_load_; }
    const global_tensor_t &C_store() const override { return C_store_; }

    void A_prefetch_inc(int64_t k_block) override {
        A_prefetch_off += k_block;
        A_prefetch_.set_coord(k_var, k_idx_ + A_prefetch_off);
    }

    void A_load_inc(int64_t k_block) override {
        A_load_off += k_block;
        A_load_.set_coord(k_var, k_idx_ + A_load_off);
    }

    void B_prefetch_inc(int64_t k_block) override {
        B_prefetch_off += k_block;
        B_prefetch_.set_coord(k_var, k_idx_ + B_prefetch_off);
    }

    void B_load_inc(int64_t k_block) override {
        B_load_off += k_block;
        B_load_.set_coord(k_var, k_idx_ + B_load_off);
    }

    void kloop_inc(int64_t k_block) override {
        // Prefetch/load computation is relative to k_idx
        A_prefetch_inc(-k_block);
        B_prefetch_inc(-k_block);
        A_load_inc(-k_block);
        B_load_inc(-k_block);

        assign(k_idx_, k_idx_ + k_block);
    }

    expr_t update_C() const override { return false; }

    expr_t is_inbounds(int64_t increment) const override {
        return (m_idx_ < m_) & (n_idx_ < n_) & (k_idx_ < k_ - increment);
    }

private:
    expr_t m_idx_;
    expr_t m_;
    expr_t n_idx_;
    expr_t n_;
    expr_t k_idx_;
    expr_t k_;

    int64_t A_prefetch_off = 0;
    int64_t A_load_off = 0;
    int64_t B_prefetch_off = 0;
    int64_t B_load_off = 0;

    global_tensor_t A_prefetch_;
    global_tensor_t A_load_;
    global_tensor_t B_prefetch_;
    global_tensor_t B_load_;
    global_tensor_t C_store_;
};

struct generator_dsl_t {
    generator_dsl_t(const generator_dsl_desc_t &desc)
        : problem(desc.problem)
        , strategy(desc.strategy)
        , iface(desc.kernel_iface())
        , options(desc.options) {}

    kernel_t build() {
        if (strategy.kParallel || strategy.kParallelLocal) {
            dsl_warning() << "kParallel support is unimplemented";
            return {};
        }
        if (strategy.persistentLoop()) {
            dsl_warning() << "persistentLoop support is unimplemented";
            return {};
        }
        if (strategy.slmA || strategy.slmB) {
            dsl_warning() << "slm copy support is unimplemented, disabling "
                             "slm copy";
        }

        if (strategy.wgPadFactor > 1) {
            dsl_warning() << "work group padding is unimplemented";
            return {};
        }

        if (strategy.cWalkOrder != WalkOrder::HW2D) {
            dsl_warning() << "Unsupported walk order";
            return {};
        }

        if (problem.Ta != problem.Ta_ext || problem.Tb != problem.Tb_ext
                || problem.Tc != problem.Tc_ext) {
            dsl_warning() << "Type conversion support is unimplemented";
            return {};
        }

        if (problem.batch != BatchMode::None
                && problem.batch != BatchMode::Strided) {
            dsl_warning() << "Batch mode is unimplemented";
            return {};
        }

        declare_kernel(iface, options);

        const auto m = arg("m");
        const auto n = arg("n");
        const auto k = arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];

        std::array<idx_t, 2> A_vars = {m_var, k_var};
        std::array<idx_t, 2> B_vars = {k_var, n_var};
        std::array<idx_t, 2> C_vars = {m_var, n_var};

        auto A_prefetch_transform
                = get_transform(strategy.A_prefetch, A_vars, true);
        auto A_load_transform = get_transform(strategy.A, A_vars);

        auto B_prefetch_transform
                = get_transform(strategy.B_prefetch, B_vars, true);
        auto B_load_transform = get_transform(strategy.B, B_vars);

        tile_t C_dims {{{m_var, m_blk}, {n_var, n_blk}}};
        auto C_store_transform = get_transform(strategy.C, C_vars);

        tensor_t C = def(
                "C_blk", C_store_transform.get_layout(C_dims, problem.Tc), 0);

        idx_t subgroup_dim = C.layout()[0].idx;
        int m_group_idx = strategy.loopOrder[0] == LoopM ? 0 : 1;
        auto thr_local_id = [](int idx) { return extract(local_id(idx), 0); };

        auto m_idx = let("m_idx",
                (group_id(m_group_idx) * local_size(m_group_idx)
                        + thr_local_id(m_group_idx))
                        * (subgroup_dim == m_var ? m_blk / strategy.subgroupSize
                                                 : m_blk));
        int n_group_idx = strategy.loopOrder[0] == LoopN ? 0 : 1;
        auto n_idx = let("n_idx",
                (group_id(n_group_idx) * local_size(n_group_idx)
                        + thr_local_id(n_group_idx))
                        * (subgroup_dim == n_var ? n_blk / strategy.subgroupSize
                                                 : n_blk));
        auto k_idx = def("k_idx", k.type(), 0);

        auto offset_A = arg("offset_A");
        auto offset_B = arg("offset_B");
        auto offset_C = arg("offset_C");

        std::vector<expr_t> C_idxs = {m_idx, n_idx};
        if (problem.batch == BatchMode::Strided) {
            struct info_t {
                info_t(expr_t size, expr_t idiv_magic)
                    : size(std::move(size))
                    , idiv_magic(std::move(idiv_magic)) {}
                expr_t size;
                expr_t idiv_magic;
            };

            auto info = [&]() {
                std::vector<info_t> ret;
                ret.reserve(problem.batchDims - 1);
                for (int i = 0; i < problem.batchDims - 1; i++) {
                    std::string i_s = std::to_string(i);
                    ret.emplace_back(
                            arg("batch_size" + i_s), arg("batch_magic" + i_s));
                }
                return ret;
            }();

            auto id = let("batch_id" + std::to_string(problem.batchDims - 1),
                    group_id(2) * local_size(2) + thr_local_id(2));
            for (int i = problem.batchDims - 1; i >= 0; i--) {
                std::string i_s = std::to_string(i);

                auto idx = let("batch_idx" + i_s, [&]() {
                    if (i == 0) return expr_t(id);
                    auto id_next = let("batch_id" + std::to_string(i - 1),
                            ternary_idiv(id, info[i - 1].size,
                                    info[i - 1].idiv_magic));
                    auto ret = id - info[i - 1].size * id_next;
                    id.assign(id_next);
                    return ret;
                }());
                C_idxs.emplace_back(idx);

                offset_A = offset_A + idx * arg("stride_A" + i_s);
                offset_B = offset_B + idx * arg("stride_B" + i_s);
                offset_C = offset_C + idx * arg("stride_C" + i_s);
            }
        }

        global_tensor_t A_base {arg("A"), offset_A,
                {{m_var, m_idx}, {k_var, k_idx}}, {{m_var, m}, {k_var, k}},
                get_strides(problem.A.layout, A_vars, arg("lda")), {}};
        global_tensor_t B_base {arg("B"), offset_B,
                {{k_var, k_idx}, {n_var, n_idx}}, {{k_var, k}, {n_var, n}},
                get_strides(problem.B.layout, B_vars, arg("ldb")), {}};
        global_tensor_t C_base {arg("C"), offset_B,
                {{m_var, m_idx}, {n_var, n_idx}}, {{m_var, m}, {n_var, n}},
                get_strides(problem.C.layout, C_vars, arg("ldc")),
                {{m_var, m_blk}, {n_var, n_blk}}};

        basic_iterator_t kloop_it(A_base, strategy.ka_prefetch,
                strategy.ka_load, B_base, strategy.kb_prefetch,
                strategy.kb_load, C_base);

        auto store_C = [&]() {
#ifndef STANDALONE
            apply_post_ops(problem.postOps.ops, C, C_idxs, {m_var, n_var});
#endif
            store(kloop_it.C_store(), C, {}, {C_store_transform.cache_hint});
        };

        tensor_config_t A_load(
                kloop_it.A_load(), A_load_transform, strategy.A_copies);
        tensor_config_t B_load(
                kloop_it.B_load(), B_load_transform, strategy.B_copies);

        auto prefetchA = strategy.prefetchA
                ? round_down(strategy.prefetchA, strategy.ka_prefetch)
                : 0;
        if (prefetchA != strategy.prefetchA)
            dsl_warning() << "Unimplemented partial A tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchA << " -> " << prefetchA;
        auto prefetchB = strategy.prefetchB
                ? round_down(strategy.prefetchB, strategy.kb_prefetch)
                : 0;
        if (prefetchB != strategy.prefetchB)
            dsl_warning() << "Unimplemented partial B tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchB << " -> " << prefetchB;

        k_loop_config_t k_loop_main {k_blk, prefetchA, prefetchB, kloop_it,
                std::move(A_load), std::move(B_load), A_prefetch_transform,
                B_prefetch_transform, C};

        dsl_assert(k_loop_main.A_load_warmup() % kloop_it.A_load().tile()[k_var]
                == 0);
        dsl_assert(k_loop_main.B_load_warmup() % kloop_it.B_load().tile()[k_var]
                == 0);

        tensor_config_t A_load_short(kloop_it.A_load(), A_load_transform, 1);
        tensor_config_t B_load_short(kloop_it.B_load(), B_load_transform, 1);

        auto k_blk_short
                = (int)lcm(A_load_short.tile[k_var], B_load_short.tile[k_var]);
        k_loop_config_t k_loop_short {k_blk_short, 0, 0, kloop_it,
                std::move(A_load_short), std::move(B_load_short),
                A_prefetch_transform, B_prefetch_transform, C};
        dsl_assert(k_loop_short.k_warmup() == 0);

        if (problem.A.alignment) {
            assume(arg("lda") % (problem.A.alignment / problem.Ta_ext) == 0);
        }
        if (problem.B.alignment) {
            assume(arg("ldb") % (problem.B.alignment / problem.Tb_ext) == 0);
        }
        if (problem.C.alignment) {
            assume(arg("ldc") % (problem.C.alignment / problem.Tc_ext) == 0);
        }

        _if(kloop_it.is_inbounds(0), [&]() {
            _if(k >= k_loop_main.k_warmup(), [&]() {
                build_k_loop(k_loop_main);
            }, [&]() { build_k_loop(k_loop_short); });
            store_C();
        });

        return end_kernel();
    }

    struct k_loop_config_t {
        int64_t k_blk;
        int64_t A_prefetch_warmup; // Offset to A prefetch
        int64_t B_prefetch_warmup; // Offset to B prefetch
        basic_iterator_t kloop_it;
        tensor_config_t A_load;
        tensor_config_t B_load;
        transform_t A_prefetch_transform;
        transform_t B_prefetch_transform;
        tensor_t C;

        int64_t A_load_warmup() const {
            return A_load.layout.elems(k_var) - A_load.tile[k_var];
        }
        int64_t B_load_warmup() const {
            return B_load.layout.elems(k_var) - B_load.tile[k_var];
        }
        int64_t k_warmup() const {
            return std::max({A_load_warmup(), B_load_warmup(),
                    A_prefetch_warmup, B_prefetch_warmup});
        }
    };

    void build_k_loop(const k_loop_config_t &cfg) {
        auto k_blk = cfg.k_blk;
        auto kloop_it = cfg.kloop_it;
        auto &C = cfg.C;

        tensor_t A = def("A_blk", cfg.A_load.layout);
        tensor_t B = def("B_blk", cfg.B_load.layout);

        auto mma_k_blk
                = std::min(cfg.A_load.tile[k_var], cfg.B_load.tile[k_var]);

        auto pipeline_idx
                = [&](int64_t loop_idx, int64_t warmup_size, int64_t period) {
            return (loop_idx + warmup_size) % period;
        };

        auto A_prefetch_blk = cfg.A_prefetch_warmup
                ? kloop_it.A_prefetch().tile()[k_var]
                : 0;
        auto A_prefetch = [&](int64_t k_unroll_idx) {
            if (cfg.A_prefetch_warmup == 0) return;
            auto idx = pipeline_idx(
                    k_unroll_idx, cfg.A_prefetch_warmup, A_prefetch_blk);
            if (idx % A_prefetch_blk != 0) return;
            prefetch(kloop_it.A_prefetch(), {{k_var, 0}},
                    {cfg.A_prefetch_transform.cache_hint});
            kloop_it.A_prefetch_inc(A_prefetch_blk);
        };

        auto A_load_blk = cfg.A_load.tile[k_var];
        auto A_load = [&](int64_t k_unroll_idx) {
            auto idx = pipeline_idx(k_unroll_idx, cfg.A_load_warmup(),
                    cfg.A_load.layout.elems(k_var));
            if (idx % A_load_blk != 0) return;
            load(A.sub(cfg.A_load.tile, {{k_var, idx}}), kloop_it.A_load(),
                    {{k_var, 0}}, {cfg.A_load.transform.cache_hint});
            kloop_it.A_load_inc(A_load_blk);
        };

        auto B_prefetch_blk = cfg.B_prefetch_warmup
                ? kloop_it.B_prefetch().tile()[k_var]
                : 0;
        auto B_prefetch = [&](int64_t k_unroll_idx) {
            if (cfg.B_prefetch_warmup == 0) return;
            int64_t idx = pipeline_idx(
                    k_unroll_idx, cfg.B_prefetch_warmup, B_prefetch_blk);
            if (idx % B_prefetch_blk != 0) return;
            prefetch(kloop_it.B_prefetch(), {{k_var, 0}},
                    {cfg.B_prefetch_transform.cache_hint});
            kloop_it.B_prefetch_inc(B_prefetch_blk);
        };

        auto B_load_blk = cfg.B_load.tile[k_var];
        auto B_load = [&](int64_t k_unroll_idx) {
            auto idx = pipeline_idx(k_unroll_idx, cfg.B_load_warmup(),
                    cfg.B_load.layout.elems(k_var));
            if (idx % B_load_blk != 0) return;
            load(B.sub(cfg.B_load.tile, {{k_var, idx}}), kloop_it.B_load(),
                    {{k_var, 0}}, {cfg.B_load.transform.cache_hint});
            kloop_it.B_load_inc(B_load_blk);
        };

        int64_t k_unroll_blk = [&]() {
            auto ret = k_blk;
            for (auto v :
                    {A_prefetch_blk, A_load_blk, B_prefetch_blk, B_load_blk}) {
                ret = gcd(ret, v);
            }
            return ret;
        }();

        auto k_body
                = [&](int64_t k_offset, bool do_A_prefetch, bool do_B_prefetch,
                          bool do_A_load, bool do_B_load, bool do_mma) {
            if (do_A_prefetch) { A_prefetch(k_offset); }

            if (do_B_prefetch) { B_prefetch(k_offset); }

            if (do_A_load) { A_load(k_offset); }

            if (do_B_load) { B_load(k_offset); }

            if (do_mma) {
                if (k_offset % mma_k_blk == 0) {
                    tile_t tile = C.tile();
                    tile[k_var] = mma_k_blk;
                    multiply(C, A, B, tile, {{k_var, k_offset}},
                            strategy.systolic);
                }
            }
        };

        // Pipeline controls
        auto warmup = cfg.k_warmup();

        for (auto k_unroll_idx = -warmup; k_unroll_idx < 0;
                k_unroll_idx += k_unroll_blk) {
            bool A_prefetch = k_unroll_idx + cfg.A_prefetch_warmup >= 0;
            bool B_prefetch = k_unroll_idx + cfg.B_prefetch_warmup >= 0;
            bool A_load = k_unroll_idx + cfg.A_load_warmup() >= 0;
            bool B_load = k_unroll_idx + cfg.B_load_warmup() >= 0;
            bool do_mma = false;
            k_body(k_unroll_idx, A_prefetch, B_prefetch, A_load, B_load,
                    do_mma);
        }

        _while(kloop_it.is_inbounds(warmup), [&]() {
            for (int64_t k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                k_body(k_unroll_idx, cfg.A_prefetch_warmup,
                        cfg.B_prefetch_warmup, true, true, true);
            }
            kloop_it.kloop_inc(k_blk);
        });

        auto tail_end = round_up(warmup, k_blk);
        for (int64_t k_unroll_idx = 0; k_unroll_idx < tail_end;
                k_unroll_idx += k_unroll_blk) {
            bool A_prefetch = k_unroll_idx + cfg.A_prefetch_warmup < tail_end;
            bool B_prefetch = k_unroll_idx + cfg.B_prefetch_warmup < tail_end;
            bool A_load = k_unroll_idx + cfg.A_load_warmup() < tail_end;
            bool B_load = k_unroll_idx + cfg.B_load_warmup() < tail_end;
            k_body(k_unroll_idx, A_prefetch, B_prefetch, A_load, B_load, true);
        }
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
    const kernel::iface_t &iface;
    const kernel::options_t &options;
};

kernel_t make_kernel(const generator_dsl_desc_t &desc) {
    return generator_dsl_t(desc).build();
}

GEMMSTONE_NAMESPACE_END
