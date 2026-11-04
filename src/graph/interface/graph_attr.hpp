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

#ifndef GRAPH_INTERFACE_GRAPH_ATTR_HPP
#define GRAPH_INTERFACE_GRAPH_ATTR_HPP

#include "common/utils.hpp"

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {

struct fpmath_t {
    fpmath_t() = default;

    fpmath_t(fpmath_mode_t mode, bool apply_to_int)
        : mode_(mode), apply_to_int_(apply_to_int) {}

    bool operator==(const fpmath_t &rhs) const {
        return mode_ == rhs.mode_ && apply_to_int_ == rhs.apply_to_int_;
    }

    fpmath_mode_t mode_ = fpmath_mode::strict;
    bool apply_to_int_ = false;
};

struct graph_attr_t {
    graph_attr_t() = default;

    explicit graph_attr_t(const fpmath_t &fpmath) : fpmath_(fpmath) {}

    const fpmath_t &get_fpmath() const { return fpmath_; }
    void set_fpmath(const fpmath_t &fpmath) { fpmath_ = fpmath; }

    bool operator==(const graph_attr_t &rhs) const {
        return fpmath_ == rhs.fpmath_;
    }

    size_t hash() const {
        size_t seed = 0;
        seed = hash_combine(seed, static_cast<size_t>(fpmath_.mode_));
        seed = hash_combine(seed, fpmath_.apply_to_int_);
        return seed;
    }

private:
    fpmath_t fpmath_;
};

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
