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

#include <cstring>
#include <random>
#include <unordered_map>

#include "dnnl_debug.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/numeric.hpp"
#include "utils/parallel.hpp"
#include "utils/stringstream.hpp"

fill_cfg_t::fill_cfg_t(dnnl_data_type_t dt, float range_min_val,
        float range_max_val, bool only_integer, attr_t::post_ops_t::kind_t alg,
        const std::string &name)
    : dt_(dt)
    , range_min_val_(MAX2(lowest_dt(dt_), range_min_val))
    , range_max_val_(MIN2(max_dt(dt_), range_max_val))
    , density_(1.f)
    , only_integer_(is_integral_dt(dt_) || only_integer)
    , name_(name) {
    if (alg == attr_t::post_ops_t::kind_t::SUB) {
        // Apply range inversion if `alg` is `sub`. This helps to keep output
        // data positive if it was intended to be positive. In rest cases act
        // like for binary `add` algorithm. If `attr` is unavailable in the
        // code, use `attr_t::post_ops_t::kind_t::ADD` as a defulat value.
        float sub_range_min_val_ = -range_min_val_;
        float sub_range_max_val_ = -range_max_val_;
        range_min_val_ = MIN2(sub_range_min_val_, sub_range_max_val_);
        range_max_val_ = MAX2(sub_range_min_val_, sub_range_max_val_);
    } else if (alg == attr_t::post_ops_t::kind_t::MUL) {
        // Reduce the range for multiplication to decrease a computational
        // error magnitute which can lead to rounding to a different output
        // value for low-precision data types.
        // TODO: replace with using specific values instead.
        range_min_val_ /= 8.f;
        range_max_val_ /= 8.f;
    }
}

fill_cfg_t::fill_cfg_t(
        const std::vector<float> &user_set, const std::string &name)
    : fill_cfg_t(user_set, 1.f, name) {}

fill_cfg_t::fill_cfg_t(const std::vector<float> &user_set, float density,
        const std::string &name)
    : dt_(dnnl_data_type_undef)
    , range_min_val_(-FLT_MAX)
    , range_max_val_(FLT_MAX)
    , predefined_set_(user_set)
    , density_(density)
    , only_integer_(false)
    , name_(name) {
    assert(!predefined_set_.empty());
}

std::string fill_cfg_t::print_verbose() const {
    stringstream_t ss;

    ss << "[FILL_CFG]";
    if (!name_.empty()) ss << " name:\'" << name_ << "\';";

    // Predefined set is mutually excluded with a range setting.
    if (!predefined_set_.empty()) {
        ss << " set:[";
        for (const auto &e : predefined_set_) {
            ss << e << ";";
        }
        ss << "]";
    } else {
        ss << " dt:" << dt_;
        ss << " range:[" << range_min_val_ << ";" << range_max_val_ << "]";
        if (only_integer_) ss << " only_integer:true";
    }

    if (density_ < 1.f) { ss << " density:" << density_; }

    return ss.str();
}

const fill_cfg_t &get_default_fill_cfg() {
    static const fill_cfg_t fill_cfg;
    return fill_cfg;
}

const fill_cfg_t &get_perf_fill_cfg(dnnl_data_type_t dt) {
    assert(has_bench_mode_bit(mode_bit_t::perf));

#define CASE(dt, low_end, high_end) \
    case dt: { \
        static const fill_cfg_t fill_cfg(dt, MAX2((low_end), lowest_dt(dt)), \
                MIN2((high_end), max_dt(dt)), /* only_int = */ false, \
                attr_t::post_ops_t::kind_t::ADD, "perf_mode_fill"); \
        return fill_cfg; \
    }

    switch (dt) {
        CASE(dnnl_f4_e2m1, -2.f, 2.f);
        CASE(dnnl_e8m0, -2.f, 2.f);
        CASE(dnnl_f8_e5m2, -2.f, 2.f);
        CASE(dnnl_f8_e4m3, -2.f, 2.f);
        CASE(dnnl_bf16, -32.f, 32.f);
        CASE(dnnl_f16, -8.f, 8.f);
        CASE(dnnl_f32, -1024.f, 1024.f);
        CASE(dnnl_f64, -1024.f, 1024.f);
        CASE(dnnl_s32, -1024.f, 1024.f);
        CASE(dnnl_s8, -32, 32);
        CASE(dnnl_u8, 0, 64);
        CASE(dnnl_s4, -8, 7);
        CASE(dnnl_u4, 0, 15);
        CASE(dnnl_u2, 0, 3);
        CASE(dnnl_u3, 0, 7);
        default: {
            assert(!"bad data_type");
            SAFE_V(FAIL);
            static const fill_cfg_t dummy;
            return dummy;
        }
    }
#undef CASE
}

int fill_scales(const attr_t &attr, int arg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto &e = attr.scales.get(arg);
    return fill_scales(e, mem_dt, mem_fp, res);
}

int fill_scales(const attr_t::arg_scales_t::entry_t &e, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    // Dynamic scales must not be filled.
    if (e.is_dynamic()) return OK;

    if (mem_dt) { assert(mem_dt.nelems() == mem_fp.nelems()); }

    if (e.has_single_element()) {
        assert(nelems == 1);
        mem_fp.set_f32_elem(0, e.scale);
        // TODO: replace reorder with `fill` that takes any pattern unlike
        // memset.
        if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);
    } else {
        // 1/16, 1/8 and 1/4 to properly work with grouped scaling.
        // Full density as zero scales are prohibited.
        static const std::vector<float> scales_set {0.0625f, 0.125f, 0.25f};
        fill_cfg_t fill_cfg(scales_set, /* density = */ 1.f, "scales");
        SAFE(fill_random_real(mem_dt, mem_fp, res, fill_cfg), WARN);
    }

    return OK;
}

int fill_zero_points(const attr_t &attr, int arg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    assert(mem_dt.nelems() == mem_fp.nelems());

    const auto &e = attr.zero_points.get(arg);
    if (e.has_single_element()) {
        assert(nelems == 1);
        mem_fp.set_f32_elem(0, e.value);
        // TODO: replace reorder with `fill` that takes any pattern unlike
        // memset.
        if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);
    } else {
        // Negative values are valid only for signed data types.
        const int min_val = MAX2(-2, static_cast<int>(lowest_dt(mem_dt.dt())));
        std::vector<float> zero_points_set {0, 1, 2};
        if (min_val < 0) {
            zero_points_set.insert(zero_points_set.end(), {-2.f, -1.f});
        }
        // Fractional half and quarter steps stay exact in floating-point data
        // types.
        if (!is_integral_dt(mem_dt.dt())) {
            zero_points_set.insert(
                    zero_points_set.end(), {-0.5f, -0.25f, 0.25f, 0.5f});
        }
        fill_cfg_t fill_cfg(
                zero_points_set, /* density = */ 1.f, "zero_points");
        SAFE(fill_random_real(mem_dt, mem_fp, res, fill_cfg), WARN);
    }

    return OK;
}

int fill_random_real_dense(dnn_mem_t &mem, dnn_mem_t &mem_ref, res_t *res,
        const fill_cfg_t &fill_cfg) {
    auto nelems = mem_ref.nelems();
    if (nelems == 0) return OK;

    BENCHDNN_PRINT(6, "%s\n", fill_cfg.print_verbose().c_str());

    // This function doesn't handle density yet.
    assert(fill_cfg.density_ == 1.f);

    // The `nelems()` function returns a product of dims/pdims regardless of
    // whether the tensor is dense or sparse (this is by design). Because of
    // that we need to adjust the `nelems` value for the sparse tensor as the
    // number of elements to fill is equal to `nnz`.
    if (mem_ref.format_kind() == dnnl_format_kind_sparse)
        nelems = query_md_nnz(mem_ref.md_);

    // Note: fill_cfg_t drives value distribution, but the final rounding is
    // in compliance with the memory object the values are inserted. Depending
    // on a case, it may or may not benefit to force same data type for filling
    // and final memory object data type.
    const dnnl_data_type_t round_dt = mem ? mem.dt() : mem_ref.dt();

    /* Do fixed partitioning to have same filling for any number of threads */
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(nelems + idx_start + 1);
        int_seed.discard(1);

        std::uniform_real_distribution<float> gen_real;
        if (fill_cfg.predefined_set_.empty()) {
            // Keep `gen_real` empty for predefined set as unneeded.
            // Note: Windows also throws an assert as it can't operate on
            // default range_min and range_max values.
            gen_real = std::uniform_real_distribution<float>(
                    fill_cfg.range_min_val_, fill_cfg.range_max_val_);
        }
        // For `predefined_set_` use indices for uniform distribution.
        std::uniform_int_distribution<> gen_int
                = !fill_cfg.predefined_set_.empty()
                ? std::uniform_int_distribution<>(0,
                          static_cast<int>(fill_cfg.predefined_set_.size()) - 1)
                : std::uniform_int_distribution<>(
                          fill_cfg.range_min_val_, fill_cfg.range_max_val_);

        const auto get_val = [&]() {
            if (!fill_cfg.predefined_set_.empty()) {
                return fill_cfg.predefined_set_[gen_int(int_seed)];
            }
            return fill_cfg.only_integer_
                    ? static_cast<float>(gen_int(int_seed))
                    : gen_real(int_seed);
        };

        if (mem_ref.dt() == dnnl_f32) {
            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                float val = get_val();
                mem_ref.set_f32_elem(
                        idx, round_to_nearest_representable(round_dt, val));
            }
        } else {
            // There are some rare scenarios when mem_ref is not f32.
            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                float val = get_val();
                mem_ref.set_elem(
                        idx, round_to_nearest_representable(round_dt, val));
            }
        }
    });

    // Note: `only_integer_` option is tricky. While it allows to avoid
    // cancellation effect triggering, it doesn't allow to validate loads
    // properly due to the values used. To ensure the library works properly,
    // basically, add fractional part of 0.5f for all floating types regardless
    // the setting, and for all integral types use values not available in other
    // data types to trigger potential overflow.
    if (fill_cfg.only_integer_) {
        const auto adjust_val = [&](float orig_val) {
            if (!is_integral_dt(round_dt)) {
                // Catch faulty integer loads instead fp type.
                return orig_val + 0.5f >= fill_cfg.range_max_val_
                        ? orig_val - 0.5f
                        : orig_val + 0.5f;
            } else if (round_dt == dnnl_s8) {
                ; // Using s8 val of -128 leads to a binary_mul alg magnifying
                        // the diff (at least for eltwise) when it used to fit.
                        // Need a general solution for the problem.
            } else if (round_dt == dnnl_u8) {
                return 128.f; // catch faulty s8 loads instead of u8.
            } else if (round_dt == dnnl_s32) {
                return 256.f; // catch faulty int8 loads instead of s32.
            } else if (round_dt == dnnl_u4) {
                return 15.f;
            } else if (round_dt == dnnl_s4) {
                return -8.f;
            } else if (round_dt == dnnl_u2) {
                return 3.f;
            } else if (round_dt == dnnl_u3) {
                return 7.f;
            } else {
                assert(!"unexpected data type");
            }
            return orig_val;
        };

        // There are some rare scenarios when mem_ref is not f32. Since it's a
        // single element per tensor, can call a regular interface.
        const float elem_first_val = adjust_val(mem_ref.get_elem(0));
        mem_ref.set_elem(
                0, round_to_nearest_representable(round_dt, elem_first_val));
    }

    if (mem) SAFE(mem.reorder(mem_ref, res), WARN);

    return OK;
}

// Since a sparsity pattern affects performance, it's crucial to keep the
// pattern intact and only randomize tensor values. Thus, the function relies on
// an assumption that in every sparse format the first handle holds values and
// the rest (two for CSR/COO/packed, one - offsets - for grouped) are
// responsible for a sparsity pattern, and are **already filled**.
int fill_random_real_sparse(const_dnnl_memory_t dnnl_memory, dnn_mem_t &mem,
        dnn_mem_t &mem_ref, res_t *res, const fill_cfg_t &fill_cfg) {
    auto orig_cc_mem_md = query_md(dnnl_memory);
    const int nhandles = query_md_num_handles(orig_cc_mem_md);
    assert(nhandles == 2 || nhandles == 3);
    // Copy-exact the content of metadata buffers. Let data handle go further.
    for (int idx = 1; idx < nhandles; idx++) {
        void *dst_ptr = mem_ref.get_mapped_pointer<void>(idx);
        void *src_ptr = nullptr;
        dnnl_memory_get_data_handle_v2(dnnl_memory, &src_ptr, idx);

        const size_t size = dnnl_memory_desc_get_size_v2(orig_cc_mem_md, idx);
        std::memcpy(dst_ptr, src_ptr, size);
    }

    return fill_random_real_dense(mem, mem_ref, res, fill_cfg);
}

int fill_random_real(dnn_mem_t &mem, dnn_mem_t &mem_ref, res_t *res,
        const fill_cfg_t &fill_cfg, const_dnnl_memory_t dnnl_memory) {
    if (mem_ref.format_kind() == dnnl_format_kind_sparse) {
        assert(dnnl_memory != nullptr);
        return fill_random_real_sparse(
                dnnl_memory, mem, mem_ref, res, fill_cfg);
    }
    return fill_random_real_dense(mem, mem_ref, res, fill_cfg);
}

int fill_random_real(dnn_mem_t &mem_ref, const fill_cfg_t &fill_cfg,
        const_dnnl_memory_t dnnl_memory) {
    dnn_mem_t dummy;
    return fill_random_real(dummy, mem_ref, nullptr, fill_cfg, dnnl_memory);
}

std::string execarg2str(int exec_arg) {
    using arg_map_t = std::unordered_map<int, std::string>;
#define ARG(id) \
    { id, #id }
    static const arg_map_t ids = {
            ARG(DNNL_ARG_UNDEF),
            ARG(DNNL_ARG_SRC_0),
            ARG(DNNL_ARG_SRC_1),
            ARG(DNNL_ARG_SRC_2),
            ARG(DNNL_ARG_SRC_3),
            ARG(DNNL_ARG_DST_0),
            ARG(DNNL_ARG_DST_1),
            ARG(DNNL_ARG_DST_2),
            ARG(DNNL_ARG_WEIGHTS_0),
            ARG(DNNL_ARG_WEIGHTS_1),
            ARG(DNNL_ARG_WEIGHTS_2),
            ARG(DNNL_ARG_WEIGHTS_3),
            ARG(DNNL_ARG_BIAS),
            ARG(DNNL_ARG_REDUCE),
            ARG(DNNL_ARG_MEAN),
            ARG(DNNL_ARG_VARIANCE),
            ARG(DNNL_ARG_SCALE),
            ARG(DNNL_ARG_SHIFT),
            ARG(DNNL_ARG_WORKSPACE),
            ARG(DNNL_ARG_SCRATCHPAD),
            ARG(DNNL_ARG_DIFF_SRC_0),
            ARG(DNNL_ARG_DIFF_SRC_1),
            ARG(DNNL_ARG_DIFF_SRC_2),
            ARG(DNNL_ARG_DIFF_SRC_3),
            ARG(DNNL_ARG_DIFF_DST_0),
            ARG(DNNL_ARG_DIFF_DST_1),
            ARG(DNNL_ARG_DIFF_DST_2),
            ARG(DNNL_ARG_DIFF_WEIGHTS_0),
            ARG(DNNL_ARG_DIFF_WEIGHTS_1),
            ARG(DNNL_ARG_DIFF_WEIGHTS_2),
            ARG(DNNL_ARG_DIFF_WEIGHTS_3),
            ARG(DNNL_ARG_DIFF_BIAS),
            ARG(DNNL_ARG_DIFF_SCALE),
            ARG(DNNL_ARG_DIFF_SHIFT),
            ARG(DNNL_ARG_ATTR_ROUNDING_SEED),
            ARG(DNNL_ARG_ATTR_DROPOUT_MASK),
            ARG(DNNL_ARG_ATTR_DROPOUT_PROBABILITY),
            ARG(DNNL_ARG_ATTR_DROPOUT_SEED),
            ARG(DNNL_ARG_ATTR_DROPOUT_OFFSET),
    };
    static const arg_map_t flags = {
            ARG(DNNL_ARG_ATTR_PRECOMPUTED_REDUCTIONS),
            ARG(DNNL_ARG_ATTR_SCALES),
            ARG(DNNL_ARG_ATTR_ZERO_POINTS),
            ARG(DNNL_ARG_ATTR_POST_OP_DW),
    };
    static const arg_map_t post_ops = {
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(1)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(2)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(3)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(4)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(5)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(6)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(7)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(8)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(9)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(10)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(11)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(12)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(13)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(14)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(15)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(16)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(17)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(18)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(19)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(20)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(21)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(22)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(23)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(24)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(25)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(26)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(27)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(28)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(29)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(30)),
            ARG(DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)),
    };
#undef ARG
#define SPACE " | "
    auto search = [](const arg_map_t &map, int i) {
        auto it = map.find(i);
        if (it != map.end()) return SPACE + it->second;
        if (!i) return std::string();
        ostringstream_t oss;
        oss << std::hex << i;
        return SPACE "0x" + oss.str();
    };
    std::string retn = search(
            post_ops, exec_arg & ~(DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1));
    exec_arg &= DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
    for (auto &flag : flags) {
        assert(((flag.first - 1) & flag.first) == 0);
        if (exec_arg & flag.first) {
            retn += SPACE + flag.second;
            exec_arg &= ~flag.first;
        }
    }
    if (exec_arg >= DNNL_ARG_MULTIPLE_DST) {
        retn += SPACE "DNNL_ARG_MULTIPLE_DST+"
                + std::to_string(exec_arg - DNNL_ARG_MULTIPLE_DST);
    } else if (exec_arg >= DNNL_ARG_MULTIPLE_SRC) {
        retn += SPACE "DNNL_ARG_MULTIPLE_SRC+"
                + std::to_string(exec_arg - DNNL_ARG_MULTIPLE_SRC);
    } else {
        retn += search(ids, exec_arg); // including DNNL_ARG_UNDEF
    }
    return retn.erase(0, sizeof(SPACE) - 1);
#undef SPACE
}

std::string buffer_prefix;

bool fill_from_file(
        int exec_arg, dnn_mem_t &mem, dnn_mem_t &ref_mem, res_t *res) {
    static const char format[] = "[FILL]: File %s %s; buffer not imported.\n";
    auto prefix = buffer_prefix;
    if (prefix.empty()) return false;

    // `mem` is not supposed to be filled.
    if (!mem || (mem.nelems() == 0)) return false;

    prefix += "." + std::to_string(exec_arg) + ".bin";

    FILE *file = nullptr;
#ifdef _WIN32
    if (fopen_s(&file, prefix.c_str(), "rb")) file = nullptr;
#else
    file = fopen(prefix.c_str(), "rb");
#endif
    prefix = "'" + prefix + "' [" + execarg2str(exec_arg) + "]";
    if (!file) {
        BENCHDNN_PRINT(2, format, prefix.c_str(), "not found");
        return false;
    }
    fseek(file, 0, SEEK_END);
    size_t total = 0, size = ftell(file);
    if (mem.size() != size) {
        fclose(file);
        BENCHDNN_PRINT(0, format, prefix.c_str(),
                "differs in size from the buffer's memory descriptor");
        BENCHDNN_PRINT(0, "Actual size: %zu\nNeeded size: %zu (%zu x %s)\n",
                size, mem.size(), mem.size() / mem.sizeof_dt(),
                dt2str(mem.dt()));
        SAFE_V(FAIL);
        return false;
    }
    fseek(file, 0, SEEK_SET);
    for (size_t read = ~0; read && (total < size); total += read)
        read = fread(
                static_cast<uint8_t *>(mem) + total, 1, size - total, file);
    fclose(file);
    if (total != size) {
        BENCHDNN_PRINT(0, format, prefix.c_str(), "cannot be read correctly");
        SAFE_V(FAIL);
        return false;
    }
    if (ref_mem && (ref_mem.reorder(mem, res) != OK)) {
        BENCHDNN_PRINT(0, format, prefix.c_str(), "cannot be reordered");
        SAFE_V(FAIL);
        return false;
    }
    BENCHDNN_PRINT(2, "File %s successfully processed; buffer imported.\n",
            prefix.c_str());
    return true;
}
