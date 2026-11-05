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

#ifndef DNN_TYPES_HPP
#define DNN_TYPES_HPP

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "common.hpp"
#include "oneapi/dnnl/dnnl_types.h"
#include "utils/data_kind.hpp"
#include "utils/dims.hpp"
#include "utils/wrapper.hpp"

namespace tag {
extern const char *x;
extern const char *abx;
extern const char *axb;
extern const char *any;
extern const char *undef;
} // namespace tag

/* TODO: merge prop and dir_t (in favor of prop) */
const char *prop2str(dnnl_prop_kind_t prop);
dnnl_prop_kind_t prop2prop_kind(dir_t dir);
int str2arg(const std::string &str);
std::string arg2str(int arg);

std::ostream &operator<<(std::ostream &s, dir_t dir);
std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt);
std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek);
template <typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &v) {
    s << v[0];
    for (size_t d = 1; d < v.size(); ++d)
        s << ":" << v[d];
    return s;
}

struct attr_t {
    // policy_t defines the way entity values will be applied to a tensor
    enum policy_t {
        COMMON = 0, // single value for each point in a tensor
        // apply a single value per...
        PER_OC, // channel (dims[1]) point
        PER_OCIC, // channel (dims[0] and dims[1]) point
        PER_DIM_0, // ... dims[0] point.
        PER_DIM_1, // ... dims[1] point.
        PER_DIM_01, // ... unique combination of dims[0] and dims[1] points.
        PER_DIM_2, // ... dims[2] point.
        PER_DIM_3, // ... dims[3] point.
        PER_TENSOR, // ... point in the tensor.
        HOST_SCALAR, // same as COMMON, but uses host-side scalar memory
        POLICY_TOTAL // guard
    };

    enum class mask_input_t {
        none,
        mask,
        policy,
    };

    static policy_t str2policy(const std::string &str);
    static const char *policy2str(policy_t policy);
    static int policy2mask(int arg, policy_t policy, int ndims = -1,
            dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
            bool has_groups = false);

    struct zero_points_t {
        struct entry_t {
            entry_t(policy_t apolicy = POLICY_TOTAL, int avalue = 0,
                    dnnl_data_type_t adt = dnnl_s32,
                    const std::vector<dnnl_dim_t> &agroups = {},
                    dnnl_quantization_mode_t aqmode
                    = dnnl_quantization_mode_static_sazp)
                : policy(apolicy)
                , value(avalue)
                , dt(adt)
                , groups(agroups)
                , mask_input(policy == POLICY_TOTAL ? mask_input_t::none
                                                    : mask_input_t::policy)
                , qmode(aqmode) {}

            int from_str(const std::string &s);

            bool is_def() const { return mask_input == mask_input_t::none; }

            bool is_host_scalar() const { return policy == HOST_SCALAR; }
            bool has_single_element() const {
                if (mask_input == mask_input_t::mask) return mask == 0;
                assert(mask_input == mask_input_t::policy);
                return policy == policy_t::COMMON || policy == HOST_SCALAR;
            }

            policy_t policy = POLICY_TOTAL;
            int value = 0;
            dnnl_data_type_t dt = dnnl_s32;
            std::vector<dnnl_dim_t> groups;
            int64_t mask = INT_MIN;
            mask_input_t mask_input = mask_input_t::none;
            dnnl_quantization_mode_t qmode = dnnl_quantization_mode_static_sazp;
        };

        int from_str(const std::string &s);

        int operator[](int arg) const { return get(arg).value; }

        bool is_def(int arg) const {
            return points.empty() || get(arg).is_def();
        }
        bool is_def() const {
            if (points.empty()) return true;

            bool def = true;
            for (const auto &e : points) {
                def = def && is_def(e.first);
            }
            return def;
        }

        bool has_host_scalars() const {
            for (const auto &e : points) {
                if (e.second.is_host_scalar()) return true;
            }
            return false;
        }

        void set(int arg, policy_t policy, int value) {
            set(arg, entry_t(policy, value));
        }
        void set(int arg, const entry_t &entry) {
            if (!entry.is_def()) points[arg] = entry;
        }
        void set(int arg, policy_t policy, int value,
                dnnl_data_type_t data_type, std::vector<dnnl_dim_t> &groups) {
            set(arg, entry_t(policy, value, data_type, groups));
        }
        entry_t get(int arg) const {
            const auto it = points.find(arg);
            return it == points.end() ? entry_t() : it->second;
        }

        int get_mask(int arg,
                dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
                int ndims = -1, bool has_groups = false) const;

        std::map<int, entry_t> points;
    };

    struct precomputed_reductions_t {
        struct entry_t {
            entry_t(policy_t apolicy = POLICY_TOTAL,
                    dnnl_data_type_t adt = dnnl_s32,
                    const std::vector<dnnl_dim_t> &agroups = {})
                : policy(apolicy)
                , dt(adt)
                , groups(agroups)
                , mask_input(policy == POLICY_TOTAL ? mask_input_t::none
                                                    : mask_input_t::policy) {}

            int from_str(const std::string &s);

            bool is_def() const { return mask_input == mask_input_t::none; }

            policy_t policy = POLICY_TOTAL;
            dnnl_data_type_t dt = dnnl_s32;
            std::vector<dnnl_dim_t> groups;
            int64_t mask = INT_MIN;
            mask_input_t mask_input = mask_input_t::none;
        };

        int from_str(const std::string &s);

        bool is_def(int arg) const {
            return entries.empty() || get(arg).is_def();
        }
        bool is_def() const {
            if (entries.empty()) return true;

            bool def = true;
            for (const auto &e : entries) {
                def = def && is_def(e.first);
            }
            return def;
        }

        void set(int arg, const entry_t &entry) {
            if (!entry.is_def()) entries[arg] = entry;
        }
        void set(int arg, policy_t policy, dnnl_data_type_t data_type,
                std::vector<dnnl_dim_t> &groups) {
            set(arg, entry_t(policy, data_type, groups));
        }
        entry_t get(int arg) const {
            const auto it = entries.find(arg);
            return it == entries.end() ? entry_t() : it->second;
        }

        int get_mask(int arg,
                dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
                int ndims = -1, bool has_groups = false) const;

        std::map<int, entry_t> entries;
    };

    struct arg_scales_t {
        struct entry_t {
            entry_t(policy_t apolicy = POLICY_TOTAL, float ascale = 1.f,
                    dnnl_data_type_t adt = dnnl_f32,
                    const std::vector<dnnl_dim_t> &agroups = {},
                    dnnl_quantization_mode_t aqmode
                    = dnnl_quantization_mode_static_sazp)
                : policy(apolicy)
                , scale(ascale)
                , dt(adt)
                , groups(agroups)
                , mask_input(policy == POLICY_TOTAL ? mask_input_t::none
                                                    : mask_input_t::policy)
                , qmode(aqmode) {}

            int from_str(const std::string &s);

            bool is_def() const { return mask_input == mask_input_t::none; }

            bool is_host_scalar() const { return policy == HOST_SCALAR; }
            bool is_mx() const {
                return qmode == dnnl_quantization_mode_dynamic_mx;
            }
            bool is_dynamic_fp() const {
                return qmode == dnnl_quantization_mode_dynamic_fp;
            }
            bool is_dynamic() const { return is_mx() || is_dynamic_fp(); }
            bool is_sbzp() const {
                return qmode == dnnl_quantization_mode_static_sbzp;
            }
            bool has_single_element() const {
                if (mask_input == mask_input_t::mask) return mask == 0;
                assert(mask_input == mask_input_t::policy);
                return policy == policy_t::COMMON || policy == HOST_SCALAR;
            }

            policy_t policy = POLICY_TOTAL;
            float scale = 1.f;
            dnnl_data_type_t dt = dnnl_f32;
            std::vector<dnnl_dim_t> groups;
            int64_t mask = INT_MIN;
            mask_input_t mask_input = mask_input_t::none;
            dnnl_quantization_mode_t qmode = dnnl_quantization_mode_static_sazp;
        };

        void set(int arg, const entry_t &scale) { scales[arg] = scale; }

        entry_t get(int arg) const {
            const auto &s = scales.find(arg);
            return s == scales.end() ? entry_t() : s->second;
        }

        int get_mask(int arg,
                dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
                int ndims = -1, bool has_groups = false) const;

        bool is_def(int arg) const {
            return scales.empty() || get(arg).is_def();
        }
        bool is_def() const {
            if (scales.empty()) return true;

            bool def = true;
            for (const auto &e : scales) {
                def = def && is_def(e.first);
            }
            return def;
        }
        bool has_host_scalars() const {
            for (const auto &e : scales) {
                if (e.second.is_host_scalar()) return true;
            }
            return false;
        }
        int from_str(const std::string &s);

        std::map<int, entry_t> scales;
    };

    struct rounding_mode_t {
        dnnl_rounding_mode_t get(int arg) const {
            const auto &r = rounding_modes_.find(arg);
            return r == rounding_modes_.end() ? dnnl_rounding_mode_environment
                                              : r->second;
        }
        void set(int arg, dnnl_rounding_mode_t rm) {
            rounding_modes_[arg] = rm;
        }
        void set_seed(uint32_t s) {
            if (is_set_seed && seed != s)
                BENCHDNN_PRINT(0, "%s\n",
                        "WARN: rounding seed has to be the same for all "
                        "arguments");
            is_set_seed = true;
            seed = s;
        }
        bool is_def() const { return rounding_modes_.empty(); }

        std::map<int, dnnl_rounding_mode_t> rounding_modes_;
        // Setting seed to all ones for better entropy
        uint32_t seed = -1;
        bool is_set_seed = false;
    };

    struct post_ops_t {
        enum kind_t {
            // sum
            SUM,
            // depthwise convolution
            DW,
            // eltwise
            ELTWISE_START, // a guard to check kind is eltwise
            ABS,
            CLIP,
            CLIP_V2,
            CLIP_V2_DST,
            ELU,
            ELU_DST,
            EXP,
            EXP_DST,
            GELU_ERF,
            GELU_TANH,
            HARDSIGMOID,
            HARDSWISH,
            LINEAR,
            LOG,
            LOGISTIC,
            LOGISTIC_DST,
            MISH,
            POW,
            RELU,
            RELU_DST,
            ROUND,
            SQRT,
            SQRT_DST,
            SQUARE,
            SRELU,
            SWISH,
            TANH,
            TANH_DST,
            ELTWISE_END, // a guard to check kind is eltwise
            // binary
            BINARY_START, // a guard to check kind is binary
            ADD,
            DIV,
            EQ,
            GE,
            GT,
            LE,
            LT,
            MAX,
            MIN,
            MUL,
            MUL_INPLACE,
            NE,
            SELECT,
            SUB,
            BINARY_END, // a guard to check kind is binary
            // prelu
            PRELU,
            // guard entry
            KIND_TOTAL
        };
        static kind_t str2kind(const std::string &str);
        static const char *kind2str(kind_t kind);
        static dnnl_alg_kind_t kind2dnnl_kind(kind_t kind);

        struct entry_t {
            entry_t(kind_t akind) : kind(akind) {
                if (is_sum_kind()) {
                } else if (is_eltwise_kind()) {
                    eltwise.alg = kind2dnnl_kind(kind);
                } else if (is_convolution_kind()) {
                } else if (is_binary_kind()) {
                    binary.alg = kind2dnnl_kind(kind);
                }
            }

            kind_t kind;
            struct {
                float scale = 1.f;
                int32_t zero_point = 0;
                dnnl_data_type_t dt = dnnl_data_type_undef;
            } sum;
            struct {
                dnnl_alg_kind_t alg = dnnl_alg_kind_undef;
                float alpha = 0.f;
                float beta = 0.f;
            } eltwise;
            struct {
                int kernel = 0;
                int stride = 0;
                int padding = 0;
                dnnl_data_type_t dst_dt = dnnl_f32;
            } convolution;
            struct binary_t {
                dnnl_alg_kind_t alg = dnnl_alg_kind_undef;
                dnnl_data_type_t src1_dt = dnnl_data_type_undef;
                policy_t policy = policy_t::POLICY_TOTAL;
                int64_t mask = INT_MIN;
                mask_input_t mask_input = mask_input_t::none;
                std::string tag = tag::any;
                dims_t strides;
                bool grouped = false;

                // For the src2 tensor when the algorithm takes ternary inputs
                dnnl_data_type_t src2_dt = dnnl_data_type_undef;
                policy_t src2_policy = policy_t::POLICY_TOTAL;
                int64_t src2_mask = INT_MIN;
                mask_input_t src2_mask_input = mask_input_t::none;
                std::string src2_tag = tag::any;
                dims_t strides2;
            } binary;
            struct {
                policy_t policy = policy_t::POLICY_TOTAL;
                int64_t mask = INT_MIN;
                mask_input_t mask_input = mask_input_t::none;
            } prelu;

            bool is_sum_kind() const;
            bool is_convolution_kind() const;
            bool is_eltwise_kind() const;
            bool is_binary_kind() const;
            bool is_binary_kind_with_ternary_op() const;
            bool is_binary_kind_inplace() const;
            bool is_prelu_kind() const;
        };

        post_ops_t() = default;

        int len() const { return (int)entry.size(); }
        bool is_def() const { return len() == 0; }

        int find(kind_t kind, int start = 0, int stop = -1) const;
        int eltwise_index() const;
        int convolution_index() const;
        int binary_index() const;
        int prelu_index() const;

        bool has_inplace_binary() const;

        // Note: this function is used only in reference compute paths.
        //
        // Note: `ndims` must be provided for primitives that have post-ops mask
        // that depends on them. Currently, it affects only lnorm with binary
        // post-op with policy PER_OC.
        //
        // Note: some primitives might need a special handling for a given
        // policy. For such primitives `prim_kind` must be set to generate a
        // correct mask value. Currently, this behavior depends on
        // `policy2mask(...)` implementation.
        std::vector<std::pair<int, int>> get_po_masks(int ndims,
                dnnl_primitive_kind_t prim_kind
                = dnnl_undefined_primitive) const;

        std::vector<entry_t> entry;
    };

    struct deterministic_t {
        // The default value is changed for a bitwise mode. To properly work,
        // `--mode=B` must be specified before the driver name, otherwise,
        // the driver settings will use `false` as the mode bit was not set
        // before entering the driver parsing section.
        deterministic_t() : enabled(has_bench_mode_bit(mode_bit_t::bitwise)) {}

        bool is_def() const {
            static deterministic_t def;
            return enabled == def.enabled;
        }

        bool enabled;
    };

    struct fpmath_mode_t {
        fpmath_mode_t() = default;

        void set(dnnl_fpmath_mode_t mode, bool apply_to_int = false) {
            this->mode = mode;
            this->apply_to_int = apply_to_int;
        }
        bool is_def() const {
            return mode == dnnl_fpmath_mode_strict && apply_to_int == false;
        }

        dnnl_fpmath_mode_t mode = dnnl_fpmath_mode_strict;
        bool apply_to_int = false;
    };

    struct dropout_t {
        float p = 0.f;
        int64_t seed = 0;
        int64_t offset = 0;
        bool use_host_scalars = false;
        std::string tag = tag::any;

        bool is_def() const { return p == 0.f; }

        bool has_output_mask() const { return tag != tag::undef; }
    };

    attr_t()
        : scratchpad_mode(get_default_scratchpad_mode())
        , acc_mode(dnnl_accumulation_mode_strict) {}

    template <typename First, typename... Rest>
    void insert(const First &first, const Rest &...rest) {
        this->insert(first);
        if (sizeof...(rest) > 0) this->insert(rest...);
    }

    void insert(const arg_scales_t &as) { this->scales = as; }
    void insert(const zero_points_t &zp) { this->zero_points = zp; }
    void insert(const precomputed_reductions_t &pr) {
        this->precomputed_reductions = pr;
    }
    void insert(const post_ops_t &po) { this->post_ops = po; }
    void insert(dnnl_scratchpad_mode_t sm) { this->scratchpad_mode = sm; }
    void insert(const fpmath_mode_t &fpm) { this->fpmath_mode = fpm; }
    void insert(dnnl_accumulation_mode_t am) { this->acc_mode = am; }
    void insert(const deterministic_t &d) { this->deterministic = d; }
    void insert(const dropout_t &d) { this->dropout = d; }
    void insert(const rounding_mode_t &rm) { this->rounding_mode = rm; }

    // When parallel creation modifier is enabled, the library scratchpad mode
    // can't be used unless "-DDNNL_ENABLE_CONCURRENT_EXEC=ON" is enabled at the
    // build time, otherwise scratchpad pointers are invalidated (as were
    // created inside threads that no longer exist when execution time comes).
    // Relevant for both engines since GPU uses CPU for faster validation.
    //
    // Same situation with native_graph execution mode.
    static dnnl_scratchpad_mode_t get_default_scratchpad_mode();

    arg_scales_t scales;
    zero_points_t zero_points;
    precomputed_reductions_t precomputed_reductions;
    post_ops_t post_ops;
    dnnl_scratchpad_mode_t scratchpad_mode;
    fpmath_mode_t fpmath_mode;
    dnnl_accumulation_mode_t acc_mode;
    deterministic_t deterministic;
    dropout_t dropout;
    rounding_mode_t rounding_mode;

    bool is_def(bool skip_fpmath = false, bool skip_acc_mode = false) const;
};

struct isa_hints_t {
    enum cpu_hints_t {
        // If DNNL_CPU_ISA_HINTS is set then use hints from there
        // Otherwise no hints
        none = 0x0,
        // No CPU ISA specific hints
        // Will override DNNL_CPU_ISA_HINTS if that is available too
        no_hints = 0x1,
        // Use prefer_ymm CPU ISA hint
        // Will override DNNL_CPU_ISA_HINTS if that is available too
        prefer_ymm = 0x2,
    };

    cpu_hints_t hints_;
    isa_hints_t(cpu_hints_t hints) : hints_(hints) {}

    cpu_hints_t get() const { return hints_; }

    static std::string hints2str(const isa_hints_t &isa_hints) {
        switch (isa_hints.hints_) {
            case none: return "none";
            case no_hints: return "no_hints";
            case prefer_ymm: return "prefer_ymm";
            default: assert(!"unknown hint"); return "unknown_hint";
        }
    }

    static isa_hints_t str2hints(const char *str) {
        cpu_hints_t hints = none;

        if (strcasecmp(str, "prefer_ymm") == 0)
            hints = prefer_ymm;
        else if (strcasecmp(str, "no_hints") == 0)
            hints = no_hints;

        return isa_hints_t(hints);
    }
};

using policy_t = attr_t::policy_t;

struct sparse_options_t {
    static constexpr dnnl_sparse_encoding_t def_encoding
            = dnnl_sparse_encoding_undef;
    static constexpr float def_sparsity = 0.9f;

    struct grouped_data_t {
        // Group sizes distribution profiles:
        // - profile_none means group_sizes are given explicitly
        // - for the rest of the profiles (deferred, with generated offsets)
        //   see --grouped in MatMul Driver documentation
        enum profile_t {
            profile_none = 0,
            profile_balanced,
            profile_hot,
            profile_decode
        };

        // kind: see above
        // param: seed for balanced, and percentage for hot; both have default
        struct profile_spec_t {
            profile_t kind = profile_none;
            dnnl_dim_t param = -1;

            // per-profile default
            static constexpr dnnl_dim_t def_balanced_seed = 0;
            static constexpr dnnl_dim_t def_hot_percent = 50;

            // param with the profile's named default filled in
            dnnl_dim_t get_param_val() const {
                if (param >= 0) return param;
                switch (kind) {
                    case profile_balanced: return def_balanced_seed;
                    case profile_hot: return def_hot_percent;
                    default: return param;
                }
            }
        };

        // Grouped Data Specification
        // buffer indices for multi-handle grouped memory
        static constexpr int grouped_values_idx = 0;
        static constexpr int grouped_offsets_idx = 1;
        // index of the dimension with variable size (0 for M)
        int variable_dim_idx = -1;
        // total number of grouped blocks
        dnnl_dim_t group_count = 0;
        // sizes for each group along the variable dimension
        std::vector<dnnl_dim_t> group_sizes;
        // resolved dispatch hint that goes to the library (0 = auto/unused);
        // set from the explicit :hint or profile HINT, else by resolve_profile
        dnnl_dim_t max_variable_dim = 0;

        // Offsets Generation Specification
        profile_spec_t profile;
    };

    sparse_options_t() = default;
    sparse_options_t(int arg, dnnl_sparse_encoding_t encoding, float sparsity) {
        add(arg, encoding, sparsity);
    }

    void add(int arg, dnnl_sparse_encoding_t encoding, float sparsity) {
        options_.insert({arg, {encoding, sparsity}});
    }

    void set_grouped(int arg, const grouped_data_t &gd) {
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
        add(arg, dnnl_grouped, 0.0f);
#endif
        grouped_data_[arg] = gd;
    }

    grouped_data_t::profile_spec_t get_profile(int arg) const {
        const auto it = grouped_data_.find(arg);
        return it == grouped_data_.end() ? grouped_data_t::profile_spec_t {}
                                         : it->second.profile;
    }

    bool has_deferred_profile() const {
        return get_profile(DNNL_ARG_SRC).kind != grouped_data_t::profile_none;
    }

    // Generate group_sizes for deferred profiles
    int resolve_profile(dnnl_dim_t total_size);

    int get_variable_dim_idx(int arg) const {
        const auto it = grouped_data_.find(arg);
        return it == grouped_data_.end() ? -1 : it->second.variable_dim_idx;
    }

    // Get group count - the count is the same across all grouped arguments
    dnnl_dim_t get_group_count() const {
        if (grouped_data_.empty()) return 0;
        return grouped_data_.begin()->second.group_count;
    }

    const std::vector<dnnl_dim_t> &get_group_sizes(int arg) const {
        static const std::vector<dnnl_dim_t> empty;
        const auto it = grouped_data_.find(arg);
        return it == grouped_data_.end() ? empty : it->second.group_sizes;
    }

    dnnl_dim_t get_max_variable_dim() const {
        if (grouped_data_.empty()) return 0;
        return grouped_data_.begin()->second.max_variable_dim;
    }

    dnnl_sparse_encoding_t get_encoding(int arg) const {
        if (options_.count(arg) == 0) return dnnl_sparse_encoding_undef;
        return options_.at(arg).first;
    }
    dnnl_sparse_encoding_t get_encoding(data_kind_t kind) const {
        // Note: the commented code doesn't work as `arg` returned is a
        // backward exec_arg. See the function comment.
        // const auto arg = data_kind2exec_arg(kind);
        // return get_encoding(arg);
        switch (kind) {
            case SRC: return get_encoding(DNNL_ARG_SRC);
            case WEI: return get_encoding(DNNL_ARG_WEIGHTS);
            case DST: return get_encoding(DNNL_ARG_DST);
            default: return def_encoding;
        }
    }

    float get_sparsity(int arg) const {
        if (options_.count(arg) == 0) return 0.0f;
        return options_.at(arg).second;
    }

    bool is_encoding_def(int arg) const {
        return get_encoding(arg) == def_encoding;
    }

    bool is_grouped(int arg) const {
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
        return get_encoding(arg) == dnnl_grouped;
#else
        return false;
#endif
    }

    // Check for grouped SRC by grouped WEI into dense DST
    bool is_2dby2d() const {
        return is_grouped(DNNL_ARG_SRC) && is_grouped(DNNL_ARG_WEIGHTS)
                && get_variable_dim_idx(DNNL_ARG_SRC) == 1;
    }

    bool is_sparsity_def(int arg) const {
        return get_sparsity(arg) == def_sparsity;
    }

    bool is_def() const {
        bool ret = true;
        for (const auto &opt : options_) {
            ret = ret && is_encoding_def(opt.first)
                    && is_sparsity_def(opt.first);
        }
        return ret;
    }

    std::vector<int> get_args() const {
        std::vector<int> args;
        args.reserve(options_.size());
        for (const auto &opt : options_) {
            args.push_back(opt.first);
        }
        return args;
    }

    int from_str(const std::string &s);

private:
    std::unordered_map<int, std::pair<dnnl_sparse_encoding_t, float>> options_;
    std::unordered_map<int, grouped_data_t> grouped_data_;
};

std::ostream &operator<<(
        std::ostream &s, const sparse_options_t &sparse_options);
std::ostream &operator<<(std::ostream &s, const policy_t &policy);
std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points);
std::ostream &operator<<(std::ostream &s,
        const attr_t::precomputed_reductions_t &precomputed_reductions);
std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k);
std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops);
std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm);
std::ostream &operator<<(std::ostream &s, const attr_t::fpmath_mode_t &fm);
std::ostream &operator<<(std::ostream &s, dnnl_accumulation_mode_t am);
std::ostream &operator<<(std::ostream &s, dnnl_rounding_mode_t rm);
std::ostream &operator<<(std::ostream &s, const attr_t::dropout_t &drop);
std::ostream &operator<<(std::ostream &s, const attr_t &attr);

// A container for additional data and info, not available from user's input at
// parse time, but which are required to create the library attributes.
struct attr_args_t {
    struct dw_t {
        dnnl_data_type_t wei_dt = dnnl_data_type_undef;
        dnnl_data_type_t bia_dt = dnnl_data_type_undef;
    };

    attr_args_t() = default;

    void prepare_quant(const attr_t &attr, int arg, int mask = INT_MIN) {
        entries.insert(std::make_pair(arg, mask));
    }

    dnnl_status_t prepare_post_ops_mds(const attr_t &attr, int ndims,
            const dnnl_dims_t prb_dims,
            dnnl_primitive_kind_t prim_kind = dnnl_undefined_primitive,
            const sparse_options_t *sparse_options = nullptr);

    void prepare_dw_post_op(const attr_t &attr, dnnl_data_type_t wei_dt,
            dnnl_data_type_t bia_dt);

    // Returns mask set for correspondent `arg`. The default value is `-1`.
    int get_mask(int arg) const {
        const auto it = entries.find(arg);
        return it == entries.end() ? undefined_mask : it->second;
    }

    dnnl_memory_desc_t get_md(int arg) const {
        const auto it = mds.find(arg);
        return it == mds.end() ? nullptr : (dnnl_memory_desc_t)it->second;
    }

    dnnl_data_type_t get_dw_arg(int arg) const {
        if (arg == DNNL_ARG_WEIGHTS)
            return dw_entry.wei_dt;
        else if (arg == DNNL_ARG_BIAS)
            return dw_entry.bia_dt;
        else {
            assert(!"unsupported_argument");
            return dnnl_data_type_undef;
        }
    }

    static constexpr int undefined_mask = INT_MIN;

private:
    std::map<int, int /* mask*/> entries;
    std::map<int, benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>> mds;
    dw_t dw_entry; // only single dw fusion is supported
};

std::ostream &dump_global_params(std::ostream &s);

// Validates a tag/meta-tag.
int check_tag(const std::string &tag_, bool check_enum_tags_only = false);

// Checks if the tag is present in the dnnl_format_tag_t enum.
bool is_enum_tag(const std::string &tag);

// Validates a tag in abc notation.
int check_abc_tag(const std::string &tag, bool check_enum_tags_only = false);

// Removes extra dimensions from a tag according to ndims.
std::string trim_tag(const std::string &tag, int ndims);
// Removes extra dimensions from a tag according to mask. `ndims` version is a
// custom case of `mask` version, assuming that first `ndims` dimensions of mask
// are non-zero.
std::string trim_tag_by_mask(const std::string &tag, int mask);

// Converts a tag/meta-tag to abc notation.
std::string normalize_tag(const std::string &tag, int ndims = -1);

dnnl_status_t create_dnnl_attr(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> &dnnl_attr,
        const attr_t &attr, const attr_args_t &attr_args, int ndims);

dnnl_engine_kind_t str2engine_kind(const std::string &s);
dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str);
dnnl_fpmath_mode_t str2fpmath_mode(const char *str);
dnnl_accumulation_mode_t str2accumulation_mode(const char *str);
dnnl_rounding_mode_t str2rounding_mode(const std::string &str);

struct dnn_mem_t;

void maybe_scale(const attr_t &attr, float &d, const float *scales, int64_t c,
        int arg, bool opposite_scale = false);
void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point = false);
float compute_eltwise_fwd(
        attr_t::post_ops_t::kind_t kind, float src, float alpha, float beta);
float compute_eltwise_bwd(attr_t::post_ops_t::kind_t kind, float d_dst,
        float src, float alpha, float beta);
float compute_binary(
        attr_t::post_ops_t::kind_t kind, float src0, float src1, bool src2);
void maybe_dropout(const attr_t &attr, float &val, int64_t idx,
        const dnn_mem_t &dropout_mask);
void maybe_round(const attr_t &attr, int arg, float &val, int64_t offset,
        dnnl_data_type_t dst_dt);
void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_po_vals);
inline void maybe_post_ops(
        const attr_t &attr, float &val, float sum_val = 0.f) {
    maybe_post_ops(attr, val, sum_val, std::vector<float>());
}

void update_cpu_ref_attrs(attr_t &attr, dnnl_data_type_t dst_dt);
#endif
