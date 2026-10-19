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

#ifndef GEMMSTONE_DSL_UTILS_ITERATOR_HPP
#define GEMMSTONE_DSL_UTILS_ITERATOR_HPP

#include <functional>
#include <utility>

#include "gemmstone/dsl/tensor.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

// Iterator filter
//
// E.g.
//
//   std::vector<layout_block_t> blocks = {...};
//   auto block_filter = [](const layout_block_t &blk) { return blk.block != 1; };
//   filter_t<std::vector<layout_block_t>> non_size_1_blocks(blocks, block_filter);
//   for (const auto &blk : non_size_1_blocks) {
//       // blk is a value from blocks, skipping those with blk.block == 1
//       ...
//   }
template <typename IterT>
class filter_t {
    using inner_iter_t = decltype(std::declval<const IterT>().begin());
    using iter_value_t = decltype(*std::declval<inner_iter_t>());
    using predicate_t = std::function<bool(const iter_value_t &)>;

public:
    class iterator_t {
    public:
        bool operator==(const iterator_t &it) const { return it_ == it.it_; }
        bool operator!=(const iterator_t &it) const { return !operator==(it); }
        iter_value_t operator*() const { return *it_; }
        iterator_t &operator++() {
            if (it_ == end_) return *this;
            while (++it_ != end_ && !predicate_(*it_))
                ;
            return *this;
        }

        iterator_t(const inner_iter_t &it, const inner_iter_t &end,
                predicate_t predicate)
            : it_(it), end_(end), predicate_(std::move(predicate)) {
            if (it_ != end_ && !predicate_(*it_)) operator++();
        }

    private:
        inner_iter_t it_, end_;
        std::function<bool(const iter_value_t &)> predicate_;
    };

    iterator_t begin() const { return {begin_, end_, predicate_}; }
    iterator_t end() const { return {end_, end_, predicate_}; }

    filter_t(const IterT &it, predicate_t predicate)
        : begin_(it.begin())
        , end_(it.end())
        , predicate_(std::move(predicate)) {}

private:
    inner_iter_t begin_, end_;
    predicate_t predicate_;
};

template <typename IterT, typename FnT>
filter_t<IterT> filter(const IterT &iter, FnT predicate) {
    return {iter, std::move(predicate)};
}

// Given two iterators of the same type, creates an iterator whose value is a
// pair made from the values of the given iterators. The iterators are
// incremented according to the provided ordering function, unless the chosen
// iterator is on its last element. In such cases, the other iterator will be
// incremented.
template <typename IterT>
struct merge_t {
    using inner_iter_t = decltype(std::declval<const IterT>().begin());
    using iter_value_t = decltype(*std::declval<inner_iter_t>());
    using cmp_t
            = std::function<bool(const iter_value_t &, const iter_value_t &)>;

public:
    using value_t = std::array<iter_value_t, 2>;
    struct iterator_t {
        bool operator==(const iterator_t &o) const {
            return a_it_ == o.a_it_ || b_it_ == o.b_it_;
        }
        bool operator!=(const iterator_t &o) const { return !operator==(o); }
        value_t operator*() const { return {*a_it_, *b_it_}; }
        iterator_t &operator++() {
            if (a_it_ == a_end_ || b_it_ == b_end_) return *this;
            auto a_tmp = a_it_, b_tmp = b_it_;
            bool a_less = cmp_(*a_it_, *b_it_);
            bool b_less = cmp_(*b_it_, *a_it_);
            bool a_final = (++a_tmp == a_end_);
            bool b_final = (++b_tmp == b_end_);
            bool a_adv = (a_less && !a_final) || b_final;
            bool b_adv = (b_less && !b_final) || a_final;
            if (a_adv) ++a_it_;
            if (b_adv) ++b_it_;
            if (!a_adv && !b_adv) ++a_it_; // tie-break for strong orders
            return *this;
        }

        iterator_t(const inner_iter_t &a_it, const inner_iter_t &a_end,
                const inner_iter_t &b_it, const inner_iter_t &b_end, cmp_t cmp)
            : a_it_(a_it)
            , a_end_(a_end)
            , b_it_(b_it)
            , b_end_(b_end)
            , cmp_(std::move(cmp)) {}

    private:
        inner_iter_t a_it_, a_end_;
        inner_iter_t b_it_, b_end_;
        cmp_t cmp_;
    };

    iterator_t begin() const {
        return {a_begin_, a_end_, b_begin_, b_end_, cmp_};
    }
    iterator_t end() const { return {a_end_, a_end_, b_end_, b_end_, cmp_}; }

    merge_t(const IterT &a, const IterT &b, cmp_t cmp)
        : a_begin_(a.begin())
        , a_end_(a.end())
        , b_begin_(b.begin())
        , b_end_(b.end())
        , cmp_(std::move(cmp)) {}

private:
    inner_iter_t a_begin_, a_end_;
    inner_iter_t b_begin_, b_end_;
    cmp_t cmp_;
};

template <typename IterT, typename FnT>
merge_t<IterT> merge(const IterT &a, const IterT &b, FnT cmp) {
    return {a, b, std::move(cmp)};
}

template <typename ResultT, typename IterT>
class transform_t {
    using inner_iter_t = decltype(std::declval<IterT>().begin());
    using iter_value_t = decltype(*std::declval<inner_iter_t>());
    using transform_op_t = std::function<ResultT(const iter_value_t &)>;

public:
    class iterator_t {
    public:
        bool operator==(const iterator_t &it) const { return it_ == it.it_; }
        bool operator!=(const iterator_t &it) const { return !operator==(it); }
        ResultT operator*() const { return transform_(*it_); }
        iterator_t &operator++() { return (++it_, *this); }

        iterator_t(const inner_iter_t &it, transform_op_t transform)
            : it_(it), transform_(std::move(transform)) {}

    private:
        inner_iter_t it_;
        transform_op_t transform_;
    };

    iterator_t begin() const { return {begin_, transform_}; }
    iterator_t end() const { return {end_, transform_}; }

    transform_t(const IterT &iterable, transform_op_t transform)
        : begin_(iterable.begin())
        , end_(iterable.end())
        , transform_(std::move(transform)) {}

private:
    inner_iter_t begin_, end_;
    transform_op_t transform_;
};

template <typename IterT, typename FnT,
        typename ResultT
        = decltype(std::declval<FnT>()(*(std::declval<const IterT>().begin())))>
transform_t<ResultT, IterT> transform(IterT &&iter, FnT transform) {
    return {iter, std::move(transform)};
}

template <typename IterT>
class inner_tiles_t {
    using inner_iter_t = decltype(std::declval<const IterT>().begin());
    using iter_value_t = decltype(*std::declval<inner_iter_t>());
    using decayed_iter_value_t = typename std::decay<iter_value_t>::type;
    static_assert(
            std::is_same<decayed_iter_value_t, dsl::layout::block_t>::value,
            "inner_tiles_t only accepts iterables with layout_block_t values");

public:
    class iterator_t {
    public:
        bool operator==(const iterator_t &it) const { return it_ == it.it_; }
        bool operator!=(const iterator_t &it) const { return !operator==(it); }

        iterator_t &operator++() {
            if (it_ == end_) return *this;

            auto block = (*it_).size;
            auto size = block / scale();
            if ((size & 1) == 0) {
                log2scale_++;
                return *this;
            }
            while (++factor_ <= size) {
                if (size % factor_ == 0) return *this;
            }

            dims_[(*it_).idx] *= block;
            ++it_;
            factor_ = 1;
            log2scale_ = 0;
            return operator++();
        }

        dsl::tile_t operator*() const {
            auto dims = dims_;
            dims[(*it_).idx] *= factor();
            return dsl::tile_t(dims);
        }

        iterator_t(
                const inner_iter_t &it, const inner_iter_t &end, size_t ndims)
            : it_(it), end_(end), dims_(ndims, 1), factor_(1) {}

    private:
        int64_t factor() const { return factor_ * scale(); }
        int64_t scale() const { return 1ll << log2scale_; }

        inner_iter_t it_, end_;
        std::vector<int64_t> dims_;
        int64_t factor_;
        int64_t log2scale_ = 0;
    };

    iterator_t begin() const { return {begin_, end_, ndims_}; }
    iterator_t end() const { return {end_, end_, ndims_}; }

    inner_tiles_t(const IterT &iterable, size_t ndims)
        : begin_(iterable.begin()), end_(iterable.end()), ndims_(ndims) {}

private:
    inner_iter_t begin_, end_;
    size_t ndims_;
};

template <typename IterT>
inner_tiles_t<IterT> inner_tiles(const IterT &iter, size_t ndims) {
    return {iter, ndims};
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
