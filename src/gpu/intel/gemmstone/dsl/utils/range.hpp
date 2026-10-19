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

#ifndef GEMMSTONE_DSL_UTILS_RANGE_HPP
#define GEMMSTONE_DSL_UTILS_RANGE_HPP

#include "dsl/utils/iterator.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

// Filter range
//
// E.g.
//
//   auto filtered_results = all_results
//         | filter([](const result_t &res) { ... });
//   for (auto result : filtered_results)
//       ...
template <typename Fn>
struct filter_range_t {
    Fn fn;
};

template <typename Fn>
filter_range_t<Fn> filter(Fn fn) {
    return {std::move(fn)};
}

template <typename IterT, typename Fn>
auto operator|(const IterT &iter,
        const filter_range_t<Fn> &fn) -> decltype(filter(iter, fn.fn)) {
    return filter(iter, fn.fn);
}

// Transform range
//
// E.g.
//
//   std::vector<int> numbers = {0, 1, 2, 3}
//   for (auto number : numbers | transform([] (int a) { return a + 1; }))
//       std::cout << number << '\n';
template <typename Fn>
struct transform_range_t {
    Fn fn;
};

template <typename Fn>
transform_range_t<Fn> transform(Fn fn) {
    return {fn};
}

template <typename IterT, typename Fn>
auto operator|(const IterT &iter,
        const transform_range_t<Fn> &fn) -> decltype(transform(iter, fn.fn)) {
    return transform(iter, fn.fn);
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
