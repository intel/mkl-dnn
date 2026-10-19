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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_SERIALIZATION_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_SERIALIZATION_HPP

#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace multiopt {

class Serialized
{
    std::vector<std::byte> raw;
    size_t readPtr = 0;

public:
    Serialized() {}
    Serialized(std::vector<std::byte> &&raw_) : raw(raw_) {}

    std::vector<std::byte> &&data() { return std::move(raw); }

    template <typename T>
    Serialized &operator<<(const T &t) {
        static_assert(std::is_trivially_copyable_v<T>);
        serialize(&t, sizeof(T));
        return *this;
    }

    template <typename T>
    Serialized &operator<<(const std::vector<T> &v) {
        operator<<(v.size());
        if constexpr (std::is_trivially_copyable_v<T>)
            serialize(v.data(), sizeof(T) * v.size());
        else for (auto &x: v)
            *this << x;
        return *this;
    }

    Serialized &operator<<(const std::vector<bool> &v) {
        std::vector<uint8_t> vi(v.begin(), v.end());
        return *this << vi;
    }

    template <typename T1, typename T2>
    Serialized &operator<<(const std::pair<T1, T2> &p) {
        return *this << p.first << p.second;
    }

    template <typename T>             Serialized &operator<<(const std::deque<T> &c)            { return serializeContainer(c); }
    template <typename T>             Serialized &operator<<(const std::unordered_set<T> &c)    { return serializeContainer(c); }
    template <typename K, typename V> Serialized &operator<<(const std::unordered_map<K, V> &c) { return serializeContainer(c); }

    template <typename T>
    Serialized &operator>>(T &t) {
        static_assert(std::is_trivially_copyable_v<T>);
        deserialize(&t, sizeof(T));
        return *this;
    }

    template <typename T>
    Serialized &operator>>(std::vector<T> &v) {
        size_t size;
        operator>>(size);
        v.resize(size);
        if constexpr (std::is_trivially_copyable_v<T>)
            deserialize(v.data(), sizeof(T) * size);
        else for (auto &x: v)
            *this >> x;
        return *this;
    }

    Serialized &operator>>(std::vector<bool> &v) {
        std::vector<uint8_t> vi;
        operator>>(vi);
        v = {vi.begin(), vi.end()};
        return *this;
    }

    template <typename T1, typename T2>
    std::enable_if_t<!std::is_const_v<T1> && !std::is_const_v<T2>, Serialized&>
    operator>>(std::pair<T1, T2> &p) {
        return *this >> p.first >> p.second;
    }

    template <typename T> Serialized &operator>>(std::deque<T> &c) {
        size_t size;
        operator>>(size);
        c.clear();
        for (; size > 0; size--) {
            T t;
            *this >> t;
            c.push_back(std::move(t));
        }
        return *this;
    }

    template <typename T>             Serialized &operator>>(std::unordered_set<T> &c)    { return deserializeUnorderedContainer(c); }
    template <typename K, typename V> Serialized &operator>>(std::unordered_map<K, V> &c) { return deserializeUnorderedContainer(c); }

private:
    void serialize(const void *data, size_t bytes) {
        if (bytes == 0) return;

        auto osize = raw.size();
        raw.resize(raw.size() + bytes);
        std::memcpy(&raw[osize], data, bytes);
    }

    template <typename C>
    Serialized &serializeContainer(const C &c) {
        operator<<(c.size());
        for (auto &x: c)
            *this << x;
        return *this;
    }

    void deserialize(void *data, size_t bytes) {
        if (raw.size() < bytes || readPtr > (raw.size() - bytes))
            throw std::runtime_error("Unexpected end-of-stream during deserialization");
        std::memcpy(data, &raw[readPtr], bytes);
        readPtr += bytes;
    }

    template <typename C>
    Serialized &deserializeUnorderedContainer(C &c) {
        using T = typename C::value_type;
        size_t size;
        operator>>(size);
        c.clear();
        for (; size > 0; size--) {
            T t;
            *this >> t;
            c.insert(std::move(t));
        }
        return *this;
    }
};

template <typename T>
std::vector<std::byte> serialize(const T &t) {
    Serialized s;
    s << t;
    return s.data();
}

} /* namespace multiopt */

#endif
