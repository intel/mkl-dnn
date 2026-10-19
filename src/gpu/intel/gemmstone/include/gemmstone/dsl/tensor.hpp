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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_TENSOR_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_TENSOR_HPP

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>

#include "gemmstone/dsl/decl.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

// The idx_t class represents an index into a tensors basis with functionality
// for logging. This class has two modes of operation: as a key in a hash map or
// as an index into contiguous containers. When used as a key, idx_t can encode
// a custom name which is especially useful when the basis dimension is shared
// among tensor objects. When used for contiguous containers, the dimension name
// is implicitly determined from the integer index value.
class idx_t : public stringify_t<idx_t> {
public:
    idx_t() = default;
    idx_t(size_t idx) : name_(idx) {}
    explicit idx_t(const std::string &name) : name_(name) {}
    bool is_undef() const { return name_.is_empty(); }
    bool operator==(const idx_t &other) const { return name_ == other.name_; }
    bool operator!=(const idx_t &other) const { return name_ != other.name_; }
    bool operator<(const idx_t &other) const { return name_ < other.name_; }
    size_t index() const { return name_.index(); }
    operator size_t() const { return index(); }
    size_t get_hash() const { return name_.get_hash(); }
    std::string str() const { return name_.str(); }

private:
    class name_t {
    public:
        name_t() = default;
        explicit name_t(size_t idx) {
            gemm_assert(idx < 26);
            data_[0] = static_cast<char>('a' + idx);
        }
        explicit name_t(const std::string &s) {
            gemm_assert(!s.empty() && s.length() <= max_len);
            s.copy(data_, s.length());
        }

        bool operator==(const name_t &other) const {
            return numeric_value() == other.numeric_value();
        }

        bool operator!=(const name_t &other) const {
            return !operator==(other);
        }

        // NUL-padded names: byte-swapped compare matches strncmp() ordering.
        bool operator<(const name_t &other) const {
            return bswap(numeric_value()) < bswap(other.numeric_value());
        }

        size_t index() const {
            gemm_assert(length() == 1);
            gemm_assert('a' <= data_[0] && data_[0] <= 'z');
            return data_[0] - 'a';
        }

        size_t length() const { return std::strlen(data_); }
        bool is_empty() const { return data_[0] == 0; }
        size_t get_hash() const {
            return std::hash<uint64_t> {}(numeric_value());
        }
        std::string str() const { return data_; }

    private:
        static uint64_t bswap(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
            return __builtin_bswap64(v);
#else
            uint64_t ret = 0;
            for (int i = 0; i < 8; i++)
                ret |= ((v >> (8 * i)) & 0xFF) << (8 * (7 - i));
            return ret;
#endif
        }
        uint64_t numeric_value() const {
            uint64_t ret;
            static_assert(sizeof(*this) == sizeof(ret),
                    "name_t is intended to be small enough to fit in a single "
                    "register");
            memcpy(&ret, this, sizeof(ret));
            return ret;
        }
        static constexpr uint32_t max_len = 7;
        char data_[max_len + 1] = {};
    };

    name_t name_;
};

class stride_t : public stringify_t<stride_t> {
public:
    constexpr stride_t(int64_t stride = undefined_stride) : stride_(stride) {}

    constexpr bool operator==(const stride_t &other) const {
        return stride_ == other.stride_;
    }

    constexpr bool operator!=(const stride_t &other) const {
        return !operator==(other);
    }

    stride_t &operator*=(const stride_t &other) {
        gemm_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ *= other.stride_;
        return *this;
    }
    stride_t &operator/=(const stride_t &other) {
        gemm_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ /= other.stride_;
        return *this;
    }
    stride_t &operator%=(const stride_t &other) {
        gemm_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ %= other.stride_;
        return *this;
    }

    size_t get_hash() const { return std::hash<int64_t> {}(stride_); }

    explicit operator int64_t() const {
        gemm_assert(is_fixed());
        return stride_;
    }
    explicit operator int() const {
        gemm_assert(is_fixed());
        return into<int>(stride_);
    }

    constexpr bool is_fixed() const { return !is_unknown() && !is_undefined(); }
    constexpr bool is_unknown() const { return stride_ == unknown_stride; }
    constexpr bool is_undefined() const { return stride_ == undefined_stride; }

    static constexpr stride_t unknown() { return stride_t(unknown_stride); }
    static constexpr stride_t undefined() { return stride_t(undefined_stride); }
    static constexpr stride_t max() { return stride_t(max_stride); }

    std::string str() const {
        if (is_undefined()) return "(invalid)";
        if (is_unknown()) return "(unknown)";
        return std::to_string(stride_);
    }

private:
    // Both negative sentinels: won't interfere with valid strides
    static constexpr int64_t unknown_stride
            = std::numeric_limits<int64_t>::min();
    static constexpr int64_t undefined_stride = unknown_stride + 1;
    static constexpr int64_t max_stride = std::numeric_limits<int64_t>::max();

    int64_t stride_ = undefined_stride;
};

static_assert(sizeof(stride_t) == 8, "stride_t is unexpectedly large");

inline stride_t operator*(stride_t a, stride_t b) {
    return a *= b;
}
inline stride_t operator/(stride_t a, stride_t b) {
    return a /= b;
}
inline stride_t operator%(stride_t a, stride_t b) {
    return a %= b;
}
inline bool operator<(stride_t a, stride_t b) {
    gemm_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) < int64_t(b);
}
inline bool operator<=(stride_t a, stride_t b) {
    gemm_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) <= int64_t(b);
}
inline bool operator>(stride_t a, stride_t b) {
    gemm_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) > int64_t(b);
}
inline bool operator>=(stride_t a, stride_t b) {
    gemm_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) >= int64_t(b);
}

// The idx_map_t is a helper class to simplify constructing hash maps using
// idx_t as a key. On top of the normal hash map interfaces, this helper class
// includes features for logging and hashing.
template <typename ValueT>
class idx_map_t : public stringify_t<idx_map_t<ValueT>> {
public:
    class iterator_t {
    public:
        iterator_t(typename std::map<idx_t, ValueT>::const_iterator it)
            : it_(it) {}

        iterator_t &operator++() {
            it_++;
            return *this;
        }
        iterator_t operator++(int) { return it_++; }

        bool operator==(const iterator_t &other) const {
            return it_ == other.it_;
        }

        bool operator!=(const iterator_t &other) const {
            return it_ != other.it_;
        }

        const idx_t &operator*() const { return it_->first; }

        const idx_t &idx() const { return it_->first; }
        const ValueT &value() const { return it_->second; }

    private:
        typename std::map<idx_t, ValueT>::const_iterator it_;
    };

    idx_map_t() = default;
    idx_map_t(const std::initializer_list<idx_t> &keys, const ValueT &value) {
        for (auto &k : keys)
            operator[](k) = value;
    }

    idx_map_t(const std::initializer_list<std::pair<idx_t, ValueT>> &keys) {
        for (auto &k : keys)
            operator[](k.first) = k.second;
    }

    explicit idx_map_t(const std::string &s) {
        for (auto &kv : to_string_int_pairs(s)) {
            operator[](idx_t(kv.first)) = ValueT(kv.second);
        }
    }

    virtual ~idx_map_t() = default;

    virtual ValueT default_value() const { return ValueT(); }

    bool has(const idx_t &key) const { return map_.count(key) != 0; }
    iterator_t begin() const { return iterator_t(map_.begin()); }
    iterator_t end() const { return iterator_t(map_.end()); }
    size_t size() const { return map_.size(); }
    bool is_empty() const { return map_.empty(); }
    explicit operator bool() const { return !is_empty(); }
    void set(const idx_t &key, const ValueT &value) { map_[key] = value; }

    void unset(const idx_t &key) {
        if (!has(key)) return;
        map_.erase(key);
    }

    std::vector<idx_t> keys() const {
        std::vector<idx_t> ret;
        for (auto &key : *this)
            ret.push_back(key);
        return ret;
    }

    const ValueT &operator[](const idx_t &key) const {
        gemm_assert(has(key), "Key not found: " + key.str());
        return map_.at(key);
    }

    ValueT &operator[](const idx_t &key) {
        if (!has(key)) set(key, default_value());
        return map_[key];
    }

    template <typename DerivedT>
    DerivedT with_impl(const idx_t &key, const ValueT &value) const {
        DerivedT ret = static_cast<const DerivedT &>(*this);
        ret[key] = value;
        return ret;
    }

    template <typename DerivedT>
    DerivedT with_impl(const idx_map_t &other) const {
        DerivedT ret = static_cast<const DerivedT &>(*this);
        for (auto &key : other) {
            ret.set(key, other.at(key));
        }
        return ret;
    }

    idx_map_t with(const idx_t &key, const ValueT &value) const {
        return with_impl<idx_map_t>(key, value);
    }

    const ValueT &at(const idx_t &key) const { return operator[](key); }
    ValueT get(const idx_t &key, const ValueT &_default) const {
        if (!has(key)) return _default;
        return at(key);
    }
    ValueT get(const idx_t &key) const { return get(key, default_value()); }
    void erase(const idx_t &key) { map_.erase(key); }

    std::vector<ValueT> values() const {
        std::vector<ValueT> ret(size());
        for (size_t i = 0; i < size(); i++)
            ret[i] = at(i);
        return ret;
    }

    std::unordered_map<std::string, int64_t> to_string_map() const {
        std::unordered_map<std::string, int64_t> ret;
        for (auto &kv : map_)
            ret[kv.first.str()] = kv.second;
        return ret;
    }

    idx_map_t operator|(const idx_map_t &other) const {
        idx_map_t ret = *this;
        for (auto &kv : other.map_) {
            auto it = map_.find(kv.first);
            if (it != map_.end()) {
                gemm_assert(it->second == kv.second);
                continue;
            }
            ret[kv.first] = kv.second;
        }
        return ret;
    }

    bool operator==(const idx_map_t &other) const {
        if (size() != other.size()) return false;
        auto it1 = map_.begin();
        auto it2 = other.map_.begin();
        for (size_t i = 0; i < size(); i++) {
            if (it1->first != it2->first) return false;
            if (!is_equal_t<ValueT, ValueT>::call(it1->second, it2->second))
                return false;
            it1++;
            it2++;
        }
        return true;
    }

    bool operator!=(const idx_map_t &other) const { return !operator==(other); }

    idx_map_t drop_defaults() const {
        idx_map_t ret;
        for (auto &d : *this) {
            if (is_equal_t<ValueT, ValueT>::call(at(d), default_value()))
                continue;
            ret[d] = at(d);
        }
        return ret;
    }

    size_t get_hash() const { return hash(map_); }

    void parse(std::istream &in) {
        std::string s;
        in >> s;
        if (s == "x") return;
        for (auto &kv : to_string_int_pairs(s)) {
            operator[](idx_t(kv.first)) = ValueT(kv.second);
        }
    }

    enum class str_format_t {
        dense,
        list,
        multiline,
    };

    std::string str_impl(str_format_t format) const {
        if (is_empty()) return "x";
        ostringstream_t oss;
        bool is_first = true;
        for (auto &kv : map_) {
            auto &p = kv.first;
            auto &value = kv.second;
            if (format == str_format_t::dense) {
                oss << p;
                out_stream_t<ValueT>::call(oss, value);
            } else if (format == str_format_t::list) {
                if (!is_first) oss << ", ";
                oss << p << ": ";
                out_stream_t<ValueT>::call(oss, value);
                is_first = false;
            } else if (format == str_format_t::multiline) {
                if (!is_first) oss << "\n";
                oss << std::setw(4) << p << ": ";
                out_stream_t<ValueT>::call(oss, value);
                is_first = false;
            }
        }
        return oss.str();
    }

    virtual str_format_t str_format() const {
        if (std::is_integral<ValueT>::value) return str_format_t::dense;
        return str_format_t::list;
    }

    virtual std::string str() const { return str_impl(str_format()); }

private:
    static std::vector<std::pair<std::string, int>> to_string_int_pairs(
            const std::string &s) {
        std::vector<std::pair<std::string, int>> ret;
        int name_beg = -1;
        int value_beg = -1;
        for (int pos = 0; pos < (int)s.size() + 1; pos++) {
            bool prev_digit = pos > 0 && std::isdigit(s[pos - 1]);
            bool cur_digit = pos < (int)s.size() && std::isdigit(s[pos]);
            if ((pos == 0 || prev_digit) && !cur_digit) {
                if (name_beg != -1 && value_beg != -1) {
                    auto key = s.substr(name_beg, value_beg - name_beg);
                    auto value
                            = std::stoi(s.substr(value_beg, pos - value_beg));
                    ret.emplace_back(key, value);
                }
                name_beg = pos;
                value_beg = -1;
            }
            if (!prev_digit && cur_digit) value_beg = pos;
        }
        return ret;
    }

    template <typename T, typename U, typename = void>
    struct is_equal_t {
        static bool call(const T &t, const U &u) { return t == u; }
    };

    template <typename T, typename U>
    struct is_equal_t<T, U,
            decltype(std::declval<T>().is_equal(std::declval<U>()), void())> {
        static bool call(const T &t, const U &u) { return t.is_equal(u); }
    };

    template <typename T, typename = void>
    struct out_stream_t {
        static std::ostream &call(std::ostream &out, const T &t) {
            return out << t;
        }
    };

    template <typename T>
    struct out_stream_t<T, decltype(std::declval<T>().str(), void())> {
        static std::ostream &call(std::ostream &out, const T &t) {
            return out << t.str();
        }
    };

    template <typename T>
    struct out_stream_t<std::vector<T>> {
        static std::ostream &call(std::ostream &out, const std::vector<T> &v) {
            out << "[";
            for (auto it = v.begin(); it != v.end(); it++) {
                if (it != v.begin()) out << ", ";
                out_stream_t<T>::call(out, *it);
            }
            out << "]";
            return out;
        }
    };

    std::map<idx_t, ValueT> map_;
};

class tile_t : public idx_map_t<int64_t> {
public:
    using idx_map_t<int64_t>::idx_map_t;

    tile_t() = default;
    tile_t(const std::vector<int64_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(idx_t(std::string(1, into<char>('a' + i))), values[i]);
    }

    int64_t default_value() const override { return 1; }

    int64_t elems() const {
        int64_t ret = 1;
        for (auto &d : *this)
            ret *= at(d);
        return ret;
    }

    bool try_factor(const idx_t &key, int64_t factor) {
        if (factor == 1) return true;
        if (!has(key)) return false;
        int64_t &value = operator[](key);
        if (value % factor != 0) return false;
        value /= factor;
        return true;
    }

    bool is_divisible(const tile_t &other) const {
        for (auto &i : other) {
            if (at(i) % other.at(i) != 0) return false;
        }
        return true;
    }

    tile_t with(const idx_t &key, int64_t value) const {
        return idx_map_t<int64_t>::with_impl<tile_t>(key, value);
    }

    tile_t with(const tile_t &other) const {
        return idx_map_t<int64_t>::with_impl<tile_t>(other);
    }

    tile_t operator/(const tile_t &divisor) const {
        gemm_assert(is_divisible(divisor));
        tile_t result = *this;
        for (auto &dim : divisor) {
            result[dim] = result.at(dim) / divisor.at(dim);
        }
        return result;
    }

#if __cplusplus >= 202002L
    bool operator==(const tile_t &other) const = default;
#endif
};

// Coordinate with integer values.
class icoord_t : public idx_map_t<int64_t> {
public:
    using idx_map_t<int64_t>::idx_map_t;

    icoord_t() = default;
    icoord_t(size_t size) : icoord_t(std::vector<int64_t>(size, 0)) {}
    icoord_t(const std::vector<int64_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(i, values[i]);
    }
    int64_t default_value() const override { return 0; }
};

// Coordinate with expression values.
class coord_t : public idx_map_t<expr_t> {
public:
    using idx_map_t<expr_t>::idx_map_t;

    coord_t() = default;
    coord_t(size_t size) : coord_t(std::vector<expr_t>(size, 0)) {}
    coord_t(const std::vector<expr_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(i, values[i]);
    }
    coord_t(const icoord_t &icoord) {
        for (auto &d : icoord)
            set(d, icoord[d]);
    }
    expr_t default_value() const override { return expr_t(0); }
    coord_t with(const idx_t &key, const expr_t &value) const {
        return idx_map_t<expr_t>::with_impl<coord_t>(key, value);
    }
    coord_t with(const coord_t &other) const {
        return idx_map_t<expr_t>::with_impl<coord_t>(other);
    }
};

inline coord_t operator+(const coord_t &a, const coord_t &b) {
    coord_t ret;
    for (auto &d : a) {
        ret[d] = a.get(d, expr_t(0)) + b.get(d, expr_t(0));
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, expr_t(0)) + b.get(d, expr_t(0));
    }
    return ret;
}

inline icoord_t operator+(const icoord_t &a, const icoord_t &b) {
    icoord_t ret;
    for (auto &d : a) {
        ret[d] = a.get(d, 0) + b.get(d, 0);
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, 0) + b.get(d, 0);
    }
    return ret;
}

struct layout_t;

namespace layout {
struct block_t : public stringify_t<block_t> {
    block_t() = default;
    block_t(const idx_t &idx, int64_t size, stride_t stride = stride_t())
        : idx(idx), size(size), stride(stride) {}

    bool operator==(const block_t &other) const {
        return (idx == other.idx) && (size == other.size)
                && (stride == other.stride);
    }
    bool operator!=(const block_t &other) const { return !(*this == other); }

    bool operator<(const block_t &other) const {
        if (stride != other.stride) return stride < other.stride;
        if (idx != other.idx) return idx < other.idx;
        return size < other.size;
    }

    size_t get_hash() const { return hash(idx, size, stride); }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(idx = " << idx;
        oss << ", size = " << size;
        oss << ", stride = " << stride.str();
        oss << ")";
        return oss.str();
    }

    idx_t idx;
    int64_t size = 1;
    stride_t stride;
};

std::vector<block_t> merge_blocks(const std::vector<block_t> &blocks);
std::vector<block_t> normalize_blocks(const std::vector<block_t> &blocks);

// Iterates through subtiles of the layout by returning the coordinates for
// each tile. The iteration order is defined by the layout blocks.
struct tile_iterator_t {
    tile_iterator_t &operator++();
    const icoord_t &operator*() const { return coord_; }
    tile_iterator_t begin() const { return *this; }
    tile_iterator_t end() const { return {}; }
    bool operator==(const tile_iterator_t &o) const { return d_ == o.d_; }
    bool operator!=(const tile_iterator_t &o) const { return d_ != o.d_; }

protected:
    friend struct gemmstone::dsl::layout_t;
    tile_iterator_t() = default;
    tile_iterator_t(const layout_t &layout, const tile_t &tile);

private:
    struct index_data_t {
        index_data_t(idx_t idx, int64_t size, int64_t stride_, int64_t tile)
            : i(0)
            , idx(idx)
            , end(std::min(size, div_up(stride_ * size, tile)))
            , stride(stride_ * div_up(size, end)) {}
        bool operator==(const index_data_t &other) const {
            return i == other.i && idx == other.idx && end == other.end
                    && stride == other.stride;
        }
        int64_t i;
        idx_t idx;
        int64_t end;
        int64_t stride;
    };
    std::vector<index_data_t> d_;
    icoord_t coord_;
};

} // namespace layout

struct layout_t : public stringify_t<layout_t> {
    using block_t = layout::block_t;

    layout_t() : type_(type_t::undef()), ndims_(0), offset_(0) {
        sanity_check();
    }
    layout_t(const type_t &type, const std::vector<int64_t> &dims,
            const expr_t &offset = 0, bool do_normalize = true);
    layout_t(const type_t &type, const std::vector<block_t> &blocks = {},
            const expr_t &offset = 0, size_t ndims = max_ndims,
            bool do_normalize = true);

    layout_t with(const std::vector<block_t> &blocks,
            bool do_normalize = true) const {
        return {type_, blocks, offset_, ndims_, do_normalize};
    }

    layout_t with(const type_t &new_type) const {
        return {new_type, blocks_, offset_, ndims_, false};
    }

    // Unknown/undefined strides are assumed to be the outermost block.
    // Furthermore, undefined strides are assumed to be dense with respect to
    // the previous block.
    layout_t with_block(block_t block) const;

    layout_t with_offset(const expr_t &offset) const {
        return {type_, blocks_, offset, ndims_, false};
    }

    bool is_empty() const {
        if (type_.is_undef()) gemm_assert(*this == layout_t());
        return type_.is_undef();
    }

    // Use of this interface is deprecated.
    size_t ndims(bool check_invalid = true) const {
        if (check_invalid) gemm_assert(has_ndims());
        return ndims_;
    }

    // Number of elements in the layout
    int64_t elems(const idx_t &idx = {}) const {
        int64_t ret = 1;
        for (auto &b : blocks_)
            if (idx.is_undef() || b.idx == idx) ret *= b.size;
        return ret;
    }

    template <typename T = expr_t>
    T offset(const coord_t &args = {}, bool ignore_offset = false) const;

    const type_t &type() const { return type_; }

    tile_t tile() const {
        tile_t tile;
        for (auto &b : blocks_)
            tile[b.idx] = tile.get(b.idx, 1) * b.size;
        return tile;
    }

    // Returns the maximum inner subtile containing less than max elements. This
    // tile is dense within the layout when is_dense is true and perfectly
    // subdivides the layout when perfectly_divides is true.
    tile_t max_subtile(int64_t max, bool is_dense = true,
            bool perfectly_divides = true) const;

    stride_t stride(const idx_t &idx, int block_idx = 0) const {
        int i = 0;
        for (auto &b : blocks_) {
            if (b.idx != idx) continue;
            if (i == block_idx) { return b.stride; }
            i++;
        }
        return stride_t();
    }

    size_t nblocks() const { return blocks().size(); }

    const std::vector<block_t> &blocks() const { return blocks_; }

    const block_t &operator[](size_t idx) const { return blocks_[idx]; }
    block_t &operator[](size_t idx) { return blocks_[idx]; }

    void set_offset(const expr_t &offset) { offset_ = offset; }

    bool is_strictly_equal(const layout_t &other, bool compare_offset = true,
            bool compare_strides = true) const;

    bool operator==(const layout_t &other) const {
        return type_ == other.type_ && ndims_ == other.ndims_
                && offset_.is_equal(other.offset_) && blocks_ == other.blocks_;
    }
    bool operator!=(const layout_t &other) const { return !operator==(other); }
    bool operator<=(const layout_t &other) const;

    bool is_equal_normalized(
            const layout_t &other, bool compare_offset = true) const {
        return normalize().is_strictly_equal(other.normalize(), compare_offset);
    }

    size_t get_hash() const { return hash(type_, ndims_, offset_, blocks_); }

    expr_t operator()(const coord_t &coord) const { return offset(coord); }

    std::string str() const {
        if (offset().is(0)) return desc_str();
        return desc_str() + " offset: " + offset_.str();
    }

    // Returns a canonical representation of the layout:
    // - Size one blocks are removed
    // - Consecutive dense blocks are merged
    layout_t normalize() const { return with(blocks_); }

    // Returns a new (sub-)layout that fully contains the passed sub-tensor.
    // Strides are kept unchanged.
    // Assumption: the original layout can be tiled by the passed sub-tensor.
    // For example: XaYb4a2b can be tiled into 2x2 sub-tensors but it's not
    // possible to tile it into 3x2 sub-tensors.
    layout_t sub(const tile_t &tile, const coord_t &start = {}) const;

    bool is_dense() const {
        stride_t stride = 1;
        for (auto &b : blocks_) {
            if (b.stride != stride) return false;
            stride *= b.size;
        }
        return true;
    }

    // Returns a packed layout where all blocks are contiguous, without gaps.
    layout_t make_dense() const {
        int64_t stride = 1;
        auto new_blocks = blocks_;
        for (auto &b : new_blocks) {
            b.stride = stride;
            stride *= b.size;
        }
        return with(new_blocks);
    }

    // Returns an equivalent layout where the specified block is split into two.
    // size0 - inner block size.
    // size1 - outer block size.
    layout_t split_block(const block_t &b, int64_t size0, int64_t size1) const;

    using tile_iterator_t = layout::tile_iterator_t;
    tile_iterator_t iter(const tile_t &tile) const {
        return tile_iterator_t(*this, tile);
    }

    template <typename F>
    void for_each_tile(const tile_t &tile, const F &f) const {
        for (auto &coord : iter(tile)) {
            f(coord);
        }
    }

    size_t get_idx(const block_t &b) const {
        gemm_assert(&blocks().front() <= &b && &b <= &blocks().back());
        return &b - &blocks().front();
    }

    bool is_outermost(const block_t &block) const {
        for (size_t i = get_idx(block) + 1; i < blocks().size(); i++) {
            if (blocks()[i].idx == block.idx) return false;
        }
        return true;
    }

private:
    static constexpr size_t max_ndims = 16;
    bool has_ndims() const { return ndims_ < max_ndims; }
    std::string desc_str(bool dnnl_style = false) const;
    void sanity_check() const;

    type_t type_; // Data type of the layout.
    size_t ndims_ = max_ndims; //(Deprecated) Number of dimensions.
    expr_t offset_; // Offset to the start of the layout.
    std::vector<block_t> blocks_; // Blocks ordered from innermost to outermost.
};

extern template expr_t layout_t::offset<expr_t>(
        const coord_t &args, bool ignore_offset) const;
extern template int layout_t::offset<int>(
        const coord_t &args, bool ignore_offset) const;
extern template int64_t layout_t::offset<int64_t>(
        const coord_t &args, bool ignore_offset) const;

struct tensor_t : public stringify_t<tensor_t> {
    tensor_t() = default;
    tensor_t(const expr_t &buf, const layout_t &layout)
        : buf_(buf)
        , base_layout_(layout)
        , layout_(layout)
        , tile_(layout.tile()) {}

    const expr_t &buf() const { return buf_; }
    const layout_t &layout() const { return layout_; }
    const type_t &type() const { return buf_.type(); }
    type_t scalar_type() const { return type().scalar(); }
    const icoord_t &coord() const { return coord_; }
    const tile_t &tile() const { return tile_; }
    bool is_empty() const { return buf_.is_empty(); }
    bool is_simd() const { return type().is_simd(); }
    int64_t elems() const {
        if (tile_.is_empty()) return base_layout_.elems();
        return tile_.elems();
    }

    tensor_t sub(const tile_t &tile, const icoord_t &coord) const {
        // coord is not measured relative to tile size
        for (auto &var : coord)
            gemm_assert(coord[var] % tile.get(var) == 0);
        tensor_t ret = *this;
        ret.coord_ = coord_ + coord;
        ret.tile_ = tile;
        ret.layout_ = base_layout_.sub(tile, ret.coord_);
        return ret;
    }

    // Returns a reference (`ir::ref_t`) to a contiguous vector of `elems` elements starting at `coord`.
    expr_t subvec(const icoord_t &coord, int elems) const;

    std::string str() const {
        std::ostringstream oss;
        oss << "buf:     " << buf_.str() << "\n";
        oss << "layout:  " << layout_.str();
        return oss.str();
    }

private:
    const layout_t &base_layout() const { return base_layout_; }

    expr_t buf_;
    layout_t base_layout_;
    layout_t layout_;
    icoord_t coord_;
    tile_t tile_;
};

struct global_tensor_t : public stringify_t<global_tensor_t> {
    global_tensor_t() = default;
    global_tensor_t(const expr_t &buf, const expr_t &base_offset,
            const idx_map_t<expr_t> &sizes, const idx_map_t<expr_t> &strides)
        : buf_(buf)
        , base_offset_(base_offset)
        , sizes_(sizes)
        , strides_(strides) {
        gemm_assert(
                buf.type().is_gm(), "Buffer must be of a global memory type.");
    }
    global_tensor_t(const expr_t &buf, const idx_map_t<expr_t> &sizes,
            const idx_map_t<expr_t> &strides)
        : global_tensor_t(buf, {}, sizes, strides) {}
    global_tensor_t(const expr_t &buf, const expr_t &base_offset,
            const coord_t &coord, const idx_map_t<expr_t> &sizes,
            const idx_map_t<expr_t> &strides, const tile_t &tile)
        : buf_(buf)
        , base_offset_(base_offset)
        , coord_(coord)
        , sizes_(sizes)
        , strides_(strides)
        , tile_(tile) {
        gemm_assert(
                buf.type().is_gm(), "Buffer must be of a global memory type.");
    }

    const expr_t &buf() const { return buf_; }
    expr_t buf_u64(const expr_t &off = {}) const;
    const type_t &type() const { return buf_.type(); }
    type_t scalar_type() const { return type().scalar(); }
    const expr_t &base_offset() const { return base_offset_; }
    const coord_t &coord() const { return coord_; }
    const idx_map_t<expr_t> &sizes() const { return sizes_; }
    const idx_map_t<expr_t> &strides() const { return strides_; }
    const tile_t &tile() const { return tile_; }

    void set_coord(const coord_t &coord) { coord_ = coord; }
    void set_coord(const idx_t &key, const expr_t &value) {
        coord_[key] = value;
    }
    void set_base_offset(const expr_t &offset) { base_offset_ = offset; }

    expr_t offset(const coord_t &sub_coord) const;

    global_tensor_t sub(
            const tile_t &sub_tile, const coord_t &sub_coord) const {
        global_tensor_t ret = *this;
        for (auto &d : sub_coord) {
            if (sizes_.has(d)) ret.coord_[d] = coord_.get(d) + sub_coord[d];
        }
        for (auto &d : sub_tile) {
            if (sizes_.has(d)) ret.tile_[d] = sub_tile[d];
        }
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "buf:           " << buf_;
        if (base_offset_) oss << "\nbase_offset: " << base_offset_ << "\n";
        if (coord_) oss << "\ncoord:         " << coord_.str();
        if (tile_) oss << "\ntile:          " << tile_.str();
        return oss.str();
    }

private:
    expr_t buf_;
    expr_t base_offset_;
    coord_t coord_;
    idx_map_t<expr_t> sizes_;
    idx_map_t<expr_t> strides_;
    tile_t tile_;
};

} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
