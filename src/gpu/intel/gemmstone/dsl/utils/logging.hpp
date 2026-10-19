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

#ifndef GEMMSTONE_DSL_UTILS_LOGGING_HPP
#define GEMMSTONE_DSL_UTILS_LOGGING_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

enum class log_level_t {
    off = 0,
    warning = 100,
    suggestion = 120,
    info = 150,
    debug = 160,
    perf = 170,
    trace = 200,
};

template <typename T, typename = void>
struct print_helper_t {
    static void call(std::ostream &out, const T &t) { out << t; }
};

template <typename T>
struct print_helper_t<T, decltype(std::declval<T>().str(), void())> {
    static void call(std::ostream &out, const T &t) { out << t.str(); }
};

inline std::vector<std::string> split(const std::string &s,
        const std::string &delimiter = std::string(1, ' ')) {
    size_t beg = 0;
    size_t end = 0;
    std::vector<std::string> ret;
    do {
        end = s.find(delimiter, beg);
        size_t len
                = (end == std::string::npos) ? std::string::npos : (end - beg);
        ret.push_back(s.substr(beg, len));
        beg = end + delimiter.size();
    } while (end != std::string::npos);
    return ret;
}

template <log_level_t level, bool value = true>
class logger_t {
public:
    logger_t(const char *file_name, int line, std::ostream &out = std::cout)
        : file_path_(file_name + std::string(":") + std::to_string(line))
        , out_(out) {}

    logger_t(const logger_t &) = delete;
    logger_t &operator=(const logger_t &) = delete;

    ~logger_t() {
        add_header(true);
        if (lines_.size() == 1) {
            out_ << " " << lines_[0] << std::endl;
        } else {
            out_ << std::endl;
            for (auto &l : lines_) {
                add_header(/*with_file=*/false);
                out_ << "  " << l << std::endl;
            }
        }
    }

    static bool is_enabled() {
        return getVerbose(GEMMVerbose::DebugInfo) >= static_cast<int>(level);
    }

    log_level_t get_level() const { return level; }

    operator bool() const { return value; }

    template <typename T>
    logger_t &operator<<(const T &obj) {
        ostringstream_t oss;
        print_helper_t<T>::call(oss, obj);
        auto lines = split(oss.str(), "\n");
        if (lines_.empty() || lines.empty()) {
            lines_ = std::move(lines);
            return *this;
        }
        lines_.back() += lines[0];
        lines_.insert(lines_.end(), lines.begin() + 1, lines.end());
        return *this;
    }

private:
    void add_header(bool with_file) {
        switch (level) {
            case log_level_t::warning: out_ << "[ WARN]"; break;
            case log_level_t::suggestion: out_ << "[SUGGESTION]"; break;
            case log_level_t::info: out_ << "[ INFO]"; break;
            case log_level_t::debug: out_ << "[DEBUG]"; break;
            case log_level_t::perf: out_ << "[ PERF]"; break;
            case log_level_t::trace: out_ << "[TRACE]"; break;
            default: stub();
        }
        if (with_file) out_ << "[" << file_path_ << "]";
    }

    std::string file_path_;
    std::ostream &out_;
    std::vector<std::string> lines_;
};

#define dsl_perf() \
    gemmstone::logger_t<gemmstone::log_level_t::perf>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::perf>( \
                    __FILE__, __LINE__)

// Trace can result in overhead making measurement meaningless
#define dsl_perf_no_trace() \
    gemmstone::logger_t<gemmstone::log_level_t::perf>::is_enabled() \
            && !gemmstone::logger_t< \
                    gemmstone::log_level_t::trace>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::perf>( \
                    __FILE__, __LINE__)

#define dsl_info() \
    gemmstone::logger_t<gemmstone::log_level_t::info>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::info>( \
                    __FILE__, __LINE__)

#define dsl_debug() \
    gemmstone::logger_t<gemmstone::log_level_t::debug>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::debug>( \
                    __FILE__, __LINE__)

#define dsl_warning() \
    gemmstone::logger_t<gemmstone::log_level_t::warning>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::warning>( \
                    __FILE__, __LINE__)

#define dsl_suggestion() \
    gemmstone::logger_t<gemmstone::log_level_t::suggestion>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::suggestion>( \
                    __FILE__, __LINE__)

#define dsl_trace() \
    gemmstone::logger_t<gemmstone::log_level_t::trace>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::trace>( \
                    __FILE__, __LINE__)

#define dsl_check(cond) \
    if (!(cond)) \
    return gemmstone::logger_t<gemmstone::log_level_t::trace>::is_enabled() \
            && gemmstone::logger_t<gemmstone::log_level_t::trace, false>( \
                    __FILE__, __LINE__)

GEMMSTONE_NAMESPACE_END

#endif
