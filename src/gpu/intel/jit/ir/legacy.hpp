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

#ifndef GPU_INTEL_JIT_IR_LEGACY_HPP
#define GPU_INTEL_JIT_IR_LEGACY_HPP

#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/../../dsl/ir/fma.hpp"
#include "gemmstone/../../dsl/ir/ir.hpp"
#include "gemmstone/../../dsl/ir/reduce.hpp"
#include "gemmstone/../../dsl/ir/reorder.hpp"
#include "gemmstone/../../dsl/ir/send.hpp"
#include "gemmstone/../../dsl/ir/walk_order.hpp"

// TODO: The interfaces in this file are for legacy compatibility with oneDNN.
// In general, these interfaces should be deleted and replaced with public
// gemmstone interfaces or moved out of gemmstone entirely.

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

namespace dsl = gemmstone::dsl;
namespace ir = gemmstone::dsl::ir;

// Legacy interfaces from gemmstone/dsl/ir/core.hpp
template <typename KeyT>
using object_set_t = ir::object_set_t<KeyT>;
template <typename KeyT>
using object_eq_set_t = ir::object_eq_set_t<KeyT>;
template <typename KeyT, typename ValueT>
using object_map_t = ir::object_map_t<KeyT, ValueT>;
template <typename KeyT, typename ValueT>
using object_eq_map_t = ir::object_eq_map_t<KeyT, ValueT>;

using ir_mutator_t = ir::ir_mutator_t;
using ir_visitor_t = ir::ir_visitor_t;

namespace object {
template <typename T>
using info_t = ir::object::info_t<T>;
using impl_t = ir::object::impl_t;
} // namespace object

using object_t = ir::object_t;
using expr_t = ir::expr_t;
using stmt_t = ir::stmt_t;

using op_kind_t = ir::op_kind_t;
using binary_op_t = ir::binary_op_t;
using bool_imm_t = ir::bool_imm_t;
using cast_t = ir::cast_t;
using const_var_t = ir::const_var_t;
using float_imm_t = ir::float_imm_t;
using int_imm_t = ir::int_imm_t;
using iif_t = ir::iif_t;
using linear_t = ir::linear_t;
using load_t = ir::load_t;
using ptr_t = ir::ptr_t;
using shuffle_t = ir::shuffle_t;
using ternary_op_t = ir::ternary_op_t;
using unary_op_t = ir::unary_op_t;
using var_t = ir::var_t;
using ref_t = ir::ref_t;
using var_t = ir::var_t;

using alloc_kind_t = ir::alloc_kind_t;
using alloc_attr_impl_t = ir::alloc_attr_impl_t;
using alloc_attr_t = ir::alloc_attr_t;
using bank_conflict_attr_t = ir::bank_conflict_attr_t;
using alloc_t = ir::alloc_t;
using assign_t = ir::assign_t;
using store_t = ir::store_t;
using for_t = ir::for_t;
using if_t = ir::if_t;
using let_t = ir::let_t;
using stmt_label_t = ir::stmt_label_t;
using stmt_group_t = ir::stmt_group_t;
using stmt_seq_t = ir::stmt_seq_t;
using while_t = ir::while_t;
using func_call_attr_t = ir::func_call_attr_t;
using instruction_modifier_attr_t = ir::instruction_modifier_attr_t;
using func_impl_t = ir::func_impl_t;
using func_t = ir::func_t;
using func_call_t = ir::func_call_t;
using builtin_t = ir::builtin_t;

using ir::is_const;
using ir::is_cpp;
using ir::is_func_call;
using ir::to_cpp;
using ir::to_expr;

// Legacy interfaces from gemmstone/dsl/ir/ir.hpp
namespace funcs = gemmstone::dsl::ir::funcs;

using alloc_manager_t = ir::alloc_manager_t;
using alloc_updater_t = ir::alloc_updater_t;
using bound_finder_base_t = ir::bound_finder_base_t;
using buffer_manager_t = ir::buffer_manager_t;
using constraint_set_t = ir::constraint_set_t;
using ir_context_t = ir::ir_context_t;
using scope_visitor_t = ir::scope_visitor_t;

using ir::cast;
using ir::count_objects;
using ir::expr_cast;
using ir::find_objects;
using ir::find_unique_objects;
using ir::make_buffer;

// Legacy interfaces from gemmstone/dsl/ir/send.hpp
using send_address_t = ir::send_address_t;
using send_cache_hint_t = ir::send_cache_hint_t;
using send_kind_t = ir::send_kind_t;
using send_op_t = ir::send_op_t;
using send_t = ir::send_t;

// Legacy interfaces from gemmstone/dsl/ir/fma.hpp
using dpas_t = ir::dpas_t;
using fma_kind_t = ir::fma_kind_t;
using mad_t = ir::mad_t;
using ir::get_supported_fma_kind;
using ir::is_dp_fma;

template <>
inline ir::fma_kind_t to_enum(const std::string &s) {
    return dsl::from_string<ir::fma_kind_t>(s);
}

// Legacy interfaces from gemmstone/dsl/reorder.hpp
using reorder_t = ir::reorder_t;
using gemmstone::dsl::ir::move_to_integer_pipe;
using gemmstone::dsl::ir::try_reinterpret_to_wider_type;

// Legacy interfaces from gemmstone/dsl/reduce.hpp
using reduce_t = ir::reduce_t;

// Legacy interfaces from gemmstone/dsl/ir/walk_order.hpp
using walk_order_t = ir::walk_order_t;
inline walk_order_t make_walk_order(const std::string &s) {
    walk_order_t walk_order;
    auto parts = gpu_utils::split(s, ",");
    gpu_assert(parts.size() <= 3);
    for (int i = 0; i < (int)parts.size(); i++) {
        for (auto &kv : ir_utils::to_string_int_pairs(parts[i])) {
            walk_order.add(dsl::idx_t(kv.first), kv.second, i);
        }
    }
    return walk_order;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
