/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef UTILS_SETTINGS_HPP
#define UTILS_SETTINGS_HPP

#include "utils/impl_filter.hpp"

#include "tests/thread_context.hpp"

struct base_settings_t {
    struct settings_attributes_t {
        void clear() { attrs_.clear(); }

        void init(const std::vector<attr_t::arg_scales_t> &scales,
                const std::vector<attr_t::zero_points_t> &zero_points,
                const std::vector<attr_t::precomputed_reductions_t>
                        &precomputed_reductions,
                const std::vector<attr_t::post_ops_t> &post_ops,
                const std::vector<dnnl_scratchpad_mode_t> &scratchpad_mode,
                const std::vector<attr_t::fpmath_mode_t> &fpmath_mode,
                const std::vector<dnnl_accumulation_mode_t> &acc_mode,
                const std::vector<attr_t::deterministic_t> &deterministic,
                const std::vector<attr_t::dropout_t> &dropout,
                const std::vector<attr_t::rounding_mode_t> &rounding_mode) {
            for_(const auto &s : scales)
            for_(const auto &zp : zero_points)
            for_(const auto &pr : precomputed_reductions)
            for_(const auto &po : post_ops)
            for_(const auto &sm : scratchpad_mode)
            for_(const auto &fm : fpmath_mode)
            for_(const auto &am : acc_mode)
            for_(const auto &d : deterministic)
            for_(const auto &dr : dropout)
            for (const auto &rm : rounding_mode)
                attrs_.push_back(
                        get_attr(s, zp, pr, po, sm, fm, am, d, dr, rm));
        }

        using vector_type = std::vector<attr_t>;
        using iterator = vector_type::iterator;
        using const_iterator = vector_type::const_iterator;
        using const_reference = const attr_t &;

        iterator begin() noexcept { return attrs_.begin(); }
        const_iterator begin() const noexcept { return attrs_.begin(); }
        iterator end() noexcept { return attrs_.end(); }
        const_iterator end() const noexcept { return attrs_.end(); }
        size_t size() const noexcept { return attrs_.size(); }
        const_reference front() const {
            // Used in prb_t(const settings_t &) constructor which requires
            // settings to have a single setup.
            assert(size() == 1);
            return *attrs_.begin();
        }

    private:
        std::vector<attr_t> attrs_;

        template <typename... ArgsT>
        static attr_t get_attr(const ArgsT &...args) {
            attr_t attr;
            attr.insert(args...);
            return attr;
        }
    };

    base_settings_t(const char *perf_template = perf_template_def)
        : perf_template(perf_template) {
        dnnl_get_default_fpmath_mode(&(this->fpmath_mode[0].mode));
    }

    // Parsed members
    std::vector<int64_t> mb {0};
    std::vector<bool> inplace {false};
    std::vector<attr_t::arg_scales_t> scales {attr_t::arg_scales_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};
    std::vector<attr_t::precomputed_reductions_t> precomputed_reductions {
            attr_t::precomputed_reductions_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            attr_t::get_default_scratchpad_mode()};
    std::vector<attr_t::fpmath_mode_t> fpmath_mode {attr_t::fpmath_mode_t()};
    std::vector<dnnl_accumulation_mode_t> acc_mode {
            dnnl_accumulation_mode_strict};
    std::vector<attr_t::deterministic_t> deterministic {
            attr_t::deterministic_t()};
    std::vector<attr_t::dropout_t> dropout {attr_t::dropout_t()};
    std::vector<attr_t::rounding_mode_t> rounding_mode {
            attr_t::rounding_mode_t()};
    std::vector<thr_ctx_t> ctx_init {get_default_thr_ctx()};
    std::vector<thr_ctx_t> ctx_exe {get_default_thr_ctx()};
    impl_filter_t impl_filter;
    const char *pattern = nullptr;
    // Non-parsed members
    settings_attributes_t attributes;

    // The leading \1 (byte 0x01) marks built-in templates for CSV escaping.
    // base_perf_report_t consumes this marker before printing the template.
    const char *perf_template_csv_base(const std::string &driver_args) const {
        static const std::string csv_pre
                = std::string("\1perf,%engine%,%impl%,%name%,");
        static const std::string csv_post = std::string(
                ",%attr%,%desc%,%Gops%,%+ctime%,%-time%,%-Gflops%,%0time%,%"
                "0Gflops%");
        static const std::string csv = csv_pre + driver_args + csv_post;
        return csv.c_str();
    }

    static constexpr const char *perf_template_def
            = "\1perf,%engine%,%impl%,%name%,%prb%,%Gops%,%+ctime%,%-time%,%-"
              "Gflops%,%0time%,%0Gflops%";
    const char *perf_template = perf_template_def;

    // Returns `true` if all vector members in this class have capacity of one.
    virtual bool has_single_setup() const {
        return mb.size() == 1 && inplace.size() == 1 && scales.size() == 1
                && zero_points.size() == 1 && precomputed_reductions.size() == 1
                && post_ops.size() == 1 && scratchpad_mode.size() == 1
                && fpmath_mode.size() == 1 && acc_mode.size() == 1
                && deterministic.size() == 1 && ctx_init.size() == 1
                && ctx_exe.size() == 1;
    }

    virtual void finalize() {
        attributes.clear();
        attributes.init(scales, zero_points, precomputed_reductions, post_ops,
                scratchpad_mode, fpmath_mode, acc_mode, deterministic, dropout,
                rounding_mode);
    }
};

#endif
