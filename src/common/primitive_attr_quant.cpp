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

#include "common/primitive_attr_quant.hpp"
#include "common/primitive_hashing.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {

const quant_entry_t &default_quant_entry() {
    static const quant_entry_t default_quant_entry;
    return default_quant_entry;
}

size_t quant_entry_t::get_hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, mask_);
    seed = hash_combine(seed, static_cast<size_t>(data_type_));
    seed = hash_combine(seed, group_ndims_);
    if (group_ndims_ > 0)
        seed = primitive_hashing::get_array_hash(
                seed, group_dims_, group_ndims_);
    seed = hash_combine(seed, is_host_scalar_);
    seed = hash_combine(seed, qmode_);
    return seed;
}

void quant_entry_t::serialize(serialization_stream_t &sstream) const {
    sstream.append(mask_);
    sstream.append(data_type_);
    sstream.append_array(group_ndims_, group_dims_);
    sstream.append(is_host_scalar_);
    sstream.append(qmode_);
}

quant_entry_t quant_entry_t::deserialize(deserializer_t &d) {
    quant_entry_t e;
    d.pop(e.mask_);
    d.pop(e.data_type_);
    size_t group_ndims;
    d.pop_array(group_ndims, e.group_dims_);
    e.group_ndims_ = static_cast<int>(group_ndims);
    d.pop(e.is_host_scalar_);
    d.pop(e.qmode_);
    return e;
}

std::string quant_entry_t::get_verbose() const {
    std::string s;
    s.append(std::to_string(mask_));
    s.append(":").append(dnnl_dt2str(data_type_));
    if (is_host_scalar_) { s.append(":host_scalar"); }

    // Dump the optional trailing fields (groups, quantization mode) in backward
    // direction so that a present later field forces earlier slots to be
    // emitted (empty when unset). This keeps the quantization mode in its own
    // slot instead of being misread as the groups field, and avoids nesting a
    // condition for each next dumping part.
    std::string str, str_tmp;
    if (!has_default_qmode()) {
        str_tmp = std::string(":") + dnnl_quantization_mode2str(qmode_);
        str = str_tmp + str;
    }
    if (!str.empty() || !has_default_groups()) {
        str_tmp = std::string(":")
                + (!has_default_groups() ? std::to_string(group_dims_[0]) + "x"
                                        + std::to_string(group_dims_[1])
                                         : std::string());
        str = str_tmp + str;
    }
    s.append(str);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const quant_entry_t &e) {
    ss << e.get_verbose();
    return ss;
}

size_t quant_entries_t::get_hash() const {
    size_t seed = 0;
    // Go through scales for all arguments.
    for (const auto &e : entries_) {
        seed = hash_combine(seed, e.first);
        seed = hash_combine(seed, e.second.get_hash());
    }
    return seed;
}

void quant_entries_t::serialize(serialization_stream_t &sstream) const {
    sstream.append(entries_.size());
    for (const auto &e : entries_) {
        sstream.append(e.first);
        sstream.append(e.second);
    }
}

template <typename T>
T deserialize_entries(deserializer_t &d) {
    T entries;
    size_t size = d.pop<size_t>();
    for (size_t i = 0; i < size; i++) {
        int arg = d.pop<int>();
        entries.set(arg, d.pop<quant_entry_t>());
    }
    return entries;
}

std::string quant_entries_t::get_verbose() const {
    std::string s;
    std::string empty_delim, attr_delim = "+";
    std::string delim = empty_delim;
    for (const auto &scale : entries_) {
        const auto &q = scale.second;
        if (q.has_default_values()) continue;

        int arg = scale.first;
        s.append(delim)
                .append(arg2str(arg))
                .append(":")
                .append(q.get_verbose());
        delim = attr_delim;
    }
    return s;
}

scales_t scales_t::deserialize(deserializer_t &d) {
    return deserialize_entries<scales_t>(d);
}

zero_points_t zero_points_t::deserialize(deserializer_t &d) {
    return deserialize_entries<zero_points_t>(d);
}

precomputed_reductions_t precomputed_reductions_t::deserialize(
        deserializer_t &d) {
    return deserialize_entries<precomputed_reductions_t>(d);
}

} // namespace impl
} // namespace dnnl
