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

#include "ref_partition.hpp"
#include "cpu/platform.hpp"
#include "dnnl_common.hpp"
#include "setting_handler.hpp"

#include "utils/compare.hpp"
#include "utils/system.hpp"

namespace graph {

namespace {

int check_memory_fit(bool fits_ram, size_t mem_req, size_t mem_limit,
        mem_platform_t memory_kind, res_t *res) {
    const bool is_device = memory_kind == GPU_REQ;
    const char *memory_kind_str = is_device ? "GPU" : "CPU";
    if (!fits_ram) {
        BENCHDNN_PRINT(1, "[CHECK_MEM]: Not enough %s RAM for a problem.\n",
                memory_kind_str);
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_enough_ram;
    }

    size_t memory_capacity = get_cpu_ram_size();
    size_t max_alloc_capacity = 0;
    if (is_device) {
        SAFE(get_gpu_ram_sizes(memory_capacity, max_alloc_capacity), WARN);
        // verbose level is aligned with check_total_size() in dnnl_common.cpp.
        BENCHDNN_PRINT((!fits_ram ? 1 : 6),
                "[CHECK_MEM]: Requested: %s; graph_device_limit: %s; "
                "device_RAM_capacity: %s; gpu_max_alloc: %s;\n",
                smart_bytes(mem_req).c_str(), smart_bytes(mem_limit).c_str(),
                smart_bytes(memory_capacity).c_str(),
                smart_bytes(max_alloc_capacity).c_str());
    } else {
        BENCHDNN_PRINT((!fits_ram ? 1 : 6),
                "[CHECK_MEM]: Requested: %s; graph_CPU_limit: %s; "
                "CPU_RAM_capacity: %s;\n",
                smart_bytes(mem_req).c_str(), smart_bytes(mem_limit).c_str(),
                smart_bytes(memory_capacity).c_str());
    }
    return OK;
}

} // namespace

ref_partition_t::ref_partition_t(const deserialized_graph_t &dg,
        const dnnl::graph::partition &par,
        const std::vector<dnnl::graph::logical_tensor> &ins,
        const std::vector<dnnl::graph::logical_tensor> &outs)
    : dg_(&dg), data_displacer(dg, par) {
    const auto &op_ids = par.get_ops();
    const std::unordered_set<size_t> op_ids_set(op_ids.begin(), op_ids.end());

    // dg.ops_ needs make sure its Topo order to first idx, first executed.
    for (const auto &aop : dg.ops_) {
        if (op_ids_set.find(aop.id_) == op_ids_set.end()) continue;

        auto aop_ref = std::ref(aop);
        partition_ops_ref_.emplace_back(aop_ref);
        for (const auto &in_lt : aop.in_lts_) {
            in_lt_2_ops_[in_lt.id_].emplace_back(aop_ref);
            lt_id_2_lt_.emplace(in_lt.id_, in_lt);
        }
        for (const auto &out_lt : aop.out_lts_) {
            out_lt_2_op_.emplace(out_lt.id_, aop_ref);
            lt_id_2_lt_.emplace(out_lt.id_, out_lt);
        }
    }

    for (const auto &in : ins) {
        partition_in_ids_.emplace_back(in.get_id());
    }
    for (const auto &out : outs) {
        partition_out_ids_.emplace_back(out.get_id());
    }
}

int ref_partition_t::init_ref(
        const std::vector<size_t> &graph_in_ports, res_t *res) {

    // Not create reference primitives and filling data with pre-designed
    // strategies if reference memories are not required.
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    for (const auto &par_op_ref : partition_ops_ref_) {
        // res should be independent from op to op
        res->state = UNTESTED;

        // End op doesn't need to create ref primitive
        if (par_op_ref.get().kind_ == "End") continue;

        auto ref_prim = ::std::make_shared<ref_primitive_t>(par_op_ref.get());

        ref_prims_.emplace(par_op_ref.get().id_, ref_prim);
        SAFE(ref_prim->init_prb(res), WARN);

        SAFE_V(ref_prim->init_prim(::get_test_engine(), res));

        // Check whether the op has any output logical tensor that is the
        // output of the partition. If so, the driver need to allocate memory
        // for correctness check.
        const auto &check_mem_sizes_args = res->mem_size_args;
        const auto is_output = is_output_op(par_op_ref.get());
        SAFE_V(check_partition_total_size(
                check_mem_sizes_args, is_output, res));
        if (res->state == SKIPPED) return OK;

        SAFE_V(check_partition_total_size(par_op_ref.get(), res));
        if (res->state == SKIPPED) return OK;

        ref_prim->init_memory_args(::get_test_engine(), res);
        SAFE_V(ref_prim->init_ref_memory_args(::get_test_engine(), res));

        // store the memory for each logical tensor
        // op `emplace` will keep the first memory it met for each id
        bool use_dst = ::graph::eltwise::get_flag_use_dst_for_bwd_compute(
                par_op_ref);
        for (size_t i = 0; i < par_op_ref.get().in_lts_.size(); i++) {
            const auto &lt = par_op_ref.get().in_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_input_offset(
                    ref_prim->get_kind(), i, use_dst);
            lt_id_2_mems_.emplace(lt.id_, ref_prim->get_arg(arg));
        }
        for (size_t i = 0; i < par_op_ref.get().out_lts_.size(); i++) {
            const auto &lt = par_op_ref.get().out_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_output_offset(
                    ref_prim->get_kind(), i);
            if (arg == 0) {
                fake_lt_ids_.insert(lt.id_);
            } else if (arg > 0) {
                lt_id_2_mems_.emplace(lt.id_, ref_prim->get_arg(arg));
            }
        }

        // Displace the data generated by the driver filling functions with
        // values supplied from the dg object. Otherwise, the values for
        // reference would diverge from the values passed to the Graph API.
        SAFE(ref_prim->displace_scales(), WARN);

        // Initialze the rest ops if current status is UNTESTED or EXECUTED or DEFERRED.
        // otherwise there is no need to init memory for the rest ops.
        if (res->state != UNTESTED && res->state != EXECUTED
                && res->state != DEFERRED) {
            // But for perf mode, when the tensors in the current op is not
            // the graph in/out, continue, otherwise return.
            if (has_bench_mode_bit(mode_bit_t::perf)) {
                for (const auto &d_lt : par_op_ref.get().in_lts_) {
                    auto iter_find = std::find(graph_in_ports.begin(),
                            graph_in_ports.end(), d_lt.id_);
                    if (iter_find != graph_in_ports.end()) { return FAIL; }
                }
                // If all op ids are not graph inputs, the op failure doesn't
                // affect the perf mode.
                continue;
            } else {
                return FAIL;
            }
        }
    }

    return OK;
}

int ref_partition_t::init_graph_mem(
        partition_mem_map_t &partition_mem_map, res_t *res) {

    // init graph input/oputput memory from lt_id_2_mems_
    for (const auto &id : partition_in_ids_) {
        if (!has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
            if (lt_id_2_mems_.find(id) == lt_id_2_mems_.end()) {
                BENCHDNN_PRINT(0, "Fail: cannot find memory for %zu\n", id);
                res->state = FAILED;
                return FAIL;
            }
            partition_mem_map.emplace(id,
                    dnn_graph_mem_t(lt_id_2_mems_.at(id), lt_id_2_lt_.at(id),
                            /*is_op_input=*/true, res));
        } else
            partition_mem_map.emplace(id,
                    dnn_graph_mem_t({}, lt_id_2_lt_.at(id),
                            /*is_op_input=*/true, res));
    }

    for (const auto &id : partition_out_ids_) {

        if (fake_lt_ids_.find(id) != fake_lt_ids_.end()
                || has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
            partition_mem_map.emplace(id,
                    dnn_graph_mem_t({}, lt_id_2_lt_.at(id),
                            /*is_op_input=*/false, res,
                            /*use_graph_layout=*/true));
        } else if (lt_id_2_mems_.find(id) != lt_id_2_mems_.end()) {
            // For output memories of graph, they need to be in compliance with
            // the reference memories regarding the shapes and memory tags, as
            // the memories of both paths will be reordered to abx for
            // comparison.
            partition_mem_map.emplace(id,
                    dnn_graph_mem_t(lt_id_2_mems_.at(id), lt_id_2_lt_.at(id),
                            /*is_op_input=*/false, res));
        } else {
            BENCHDNN_PRINT(0, "Fail: cannot find memory for %zu\n", id);
            res->state = FAILED;
            return FAIL;
        }
    }

    return OK;
}

void ref_partition_t::exec_ops(res_t *res) {
    for (const auto &par_op_ref : partition_ops_ref_) {
        const auto &op = par_op_ref.get();
        // skip End op's execution
        if (op.kind_ == "End") continue;
        // displace data if needed, before executing the ref_prim
        for (size_t i = 0; i < op.in_lts_.size(); i++) {
            size_t lt_id = op.in_lts_[i].id_;
            SAFE_V(data_displacer.displace_input_data(
                    lt_id, lt_id_2_mems_, res));
        }

        auto ref_prim = ref_prims_.at(op.id_);
        // check if the condition input of Select op is from the parent op.
        bool select_op_cond_has_parent
                = ref_prim->get_kind() == dnnl::graph::op::kind::Select
                && get_parent_op(op.in_lts_[0].id_);

        // link args && replace the memory before execution
        bool use_dst = ::graph::eltwise::get_flag_use_dst_for_bwd_compute(
                par_op_ref);
        for (size_t i = 0; i < op.in_lts_.size(); i++) {
            const auto &lt = op.in_lts_[i];
            int arg = get_prim_arg_name_from_graph_op_input_offset(
                    ref_prim->get_kind(), i, use_dst);
            if (select_op_cond_has_parent && i == 0) {
                // Since select primitive implementation only support s8 data
                // type for the condition input, need to transfer the f32
                // results from the previous op to s8.
                dnn_mem_t &dst_i
                        = const_cast<dnn_mem_t &>(ref_prim->get_arg(arg));
                const auto &src_i = lt_id_2_mems_.at(lt.id_);
                SAFE_V(dst_i.reorder(src_i, res));
                continue;
            }
            ref_prim->replace_arg(arg, lt_id_2_mems_.at(lt.id_));
        }
        for (size_t i = 0; i < op.out_lts_.size(); i++) {
            const auto &lt = op.out_lts_[i];
            // skip replace for fake output tensor
            if (fake_lt_ids_.find(lt.id_) != fake_lt_ids_.end()) continue;
            int arg = get_prim_arg_name_from_graph_op_output_offset(
                    ref_prim->get_kind(), i);
            ref_prim->replace_arg(arg, lt_id_2_mems_.at(lt.id_));
        }

        // There are unfusable operations inside complex fusion partitions
        // (such as Softmax in SDPA or chains of MatMuls in MLP) that are
        // executed with user-requested data type. To have correctness
        // validation working as expected, the data for such operations should
        // be adjusted accordingly in case of low precision data types. E.g.,
        // if pattern is bfloat16 only, the output of a matmul op is bfloat16.
        // Having a float reference implies that is should use "same" bfloat16
        // data, otherwise, the output from bfloat16 softmax inside the library
        // and float softmax inside the reference will mismatch, which happens
        // due to the property of softmax, and exponent part in particular.
        //
        // However, this practice of data conversion to a lower precision and
        // back must be limited to the cases when it's necessary.
        //
        // For SDPA, it is limited for a Softmax with a parent op presented, as
        // it's assumed Softmax is unfusable.
        const bool is_softmax_in_sdpa_pattern
                = ref_prim->get_kind() == dnnl::graph::op::kind::SoftMax
                && dg_->get_recognized_pattern()
                        == graph_recognized_pattern_t::sdpa_fwd;

        // For SDPA training backward, it is limited for MatMuls used to compute
        // dQ, dK, dV (which has no child ops or child ops = ReduceSum for GQA).
        const deserialized_op_t *child_op = nullptr;
        const bool is_matmul_in_sdpa_bwd_pattern
                = ref_prim->get_kind() == dnnl::graph::op::kind::MatMul
                && dg_->get_recognized_pattern()
                        == graph_recognized_pattern_t::sdpa_bwd
                && (!has_child_op(op, &child_op)
                        || child_op->kind_ == "ReduceSum");

        // gMLP can have two Multiply ops - one as a part of SwigLU, the other
        // is a Gate-Up connection, which is a target of reorder through a
        // potential TypeCast after it.
        const bool is_multiply_in_gmlp_pattern
                = ref_prim->get_kind() == dnnl::graph::op::kind::Multiply
                && dg_->get_recognized_pattern()
                        == graph_recognized_pattern_t::gmlp
                && has_child_op(op, &child_op) && child_op->kind_ == "MatMul";

        if (is_softmax_in_sdpa_pattern || is_matmul_in_sdpa_bwd_pattern
                || is_multiply_in_gmlp_pattern) {
            for (size_t i = 0; i < op.in_lts_.size(); i++) {
                const auto dt = ref_prim->get_lt_dt(op.in_lts_[i].id_);
                // There's no need to reorder data for f32 tensors.
                if (dt == dnnl_f32 || dt == dnnl_data_type_undef) continue;

                // MLP pattern requires reorder only for an input coming from
                // MatMul0 directly, not from Swish.
                if (is_multiply_in_gmlp_pattern) {
                    const auto parent_op = get_parent_op(op.in_lts_[i].id_);
                    if (!parent_op) continue;
                    if (parent_op->kind_ != "MatMul") continue;
                }

                int arg = get_prim_arg_name_from_graph_op_input_offset(
                        ref_prim->get_kind(), i, use_dst);
                dnn_mem_t &src_i
                        = const_cast<dnn_mem_t &>(ref_prim->get_arg(arg));
                dnn_mem_t src_low_dt(src_i, dt, tag::abx, src_i.engine());
                SAFE_V(src_i.reorder(src_low_dt, res));
            }
        }

        SAFE_V(ref_prim->execute_prim(res));

        // For an output, because of various graph compositions, there's a more
        // detailed guide when data adjustment should happen. It's covered by
        // `need_unfusable_output_crop` function.
        //
        // A data type to where transform the data will also be provided by the
        // same function since there are corner cases.
        if (is_softmax_in_sdpa_pattern || is_matmul_in_sdpa_bwd_pattern
                || is_multiply_in_gmlp_pattern) {
            for (size_t i = 0; i < op.out_lts_.size(); i++) {
                dnnl_data_type_t dt = dnnl_data_type_undef;
                if (!need_unfusable_output_crop(op, i, dt)) continue;
                // There's no need to reorder data for f32 tensors.
                if (dt == dnnl_f32) continue;

                int arg = get_prim_arg_name_from_graph_op_output_offset(
                        ref_prim->get_kind(), i);
                dnn_mem_t &dst_i
                        = const_cast<dnn_mem_t &>(ref_prim->get_arg(arg));
                dnn_mem_t dst_low_dt(dst_i, dt, tag::abx, dst_i.engine());
                SAFE_V(dst_i.reorder(dst_low_dt, res));
            }
        }
    }
}

int ref_partition_t::check_partition_correctness(
        partition_mem_map_t &partition_mem_map, res_t *res) {

    bool mistrusted = false, has_eltwise = false, output_has_nans = false;
    const auto &map_kind_to_alg = eltwise::get_eltwise_kind_map();

    // Fused SDPA rounds P before normalizing, the reference after: allow one
    // e4m3 rounding on each side.
    bool has_f8_probs_quant = false;
    for (const auto &op : partition_ops_ref_) {
        const auto &aop = op.get();
        if (aop.kind_ != "Quantize" && aop.kind_ != "DynamicQuantize") continue;
        const auto out_dt = aop.out_lts_[0].get_data_type();
        if (out_dt != logical_tensor::data_type::f8_e4m3
                && out_dt != logical_tensor::data_type::f8_e5m2)
            continue;
        const auto *parent_op = get_parent_op(aop.in_lts_[0].id_);
        if (parent_op && parent_op->kind_ == "SoftMax") {
            has_f8_probs_quant = true;
            break;
        }
    }
    const float p2p_threshold = has_f8_probs_quant ? 2 * 0.0625f : 0.f;

    for (const auto &op : partition_ops_ref_) {
        size_t op_id = op.get().id_;
        const auto &op_kind = op.get().kind_;
        // End op's correctness check is performed during its producer.
        if (op_kind == "End") continue;
        const auto &ref_prim = ref_prims_.at(op_id);

        // if there is eltwise post-ops or binary div post-ops (GPU test), need
        // to relax compare critria.
        // Currently, both cases use set_has_eltwise_post_op flag in benchdnn
        // compare function.
        // The flag name is not very accurate, add this note to avoid confusion
        const auto op_driver = op.get().opkind2driver();
        has_eltwise = has_eltwise
                || (op_driver == dnnl_driver_t::eltwise
                        || ((opstr2kind(op_kind)
                                            == dnnl::graph::op::kind::Divide
                                    || op_driver == dnnl_driver_t::softmax)
                                && engine_tgt_kind == dnnl_gpu));
        output_has_nans = output_has_nans
                || ((map_kind_to_alg.find(op_kind) != map_kind_to_alg.end())
                        && ::eltwise::eltwise_alg_returns_nan_or_inf(
                                map_kind_to_alg.at(op_kind)))
                // `f8_e4m3` range is very short which makes inputs convert
                // into NaNs.
                || (op_driver == dnnl_driver_t::reorder
                        && op.get().in_lts_.front().get_data_type()
                                == logical_tensor::data_type::f8_e4m3);

        // get the args that need comparing
        args_t output_args;
        for (size_t out_idx = 0; out_idx < op.get().out_lts_.size();
                ++out_idx) {
            int out_arg = get_prim_arg_name_from_graph_op_output_offset(
                    opstr2kind(op_kind), out_idx);
            if (out_arg == 0) continue; // unsupported case

            size_t out_lt_id = op.get().out_lts_[out_idx].id_;
            for (size_t i = 0; i < partition_out_ids_.size(); i++) {
                if (out_lt_id != partition_out_ids_[i]) continue;

                auto &graph_mem = partition_mem_map.at(out_lt_id);
                const auto &par_out_mem = graph_mem.get_mem();
                output_args.set(out_arg, par_out_mem);
                break;
            }
        }
        if (output_args.size() == 0) continue;

        // reset the state
        res->state = EXECUTED;

        // TODO(zhitao): need to check whether the operation that produces the
        // output args is the children of the operations that affect
        // output_has_nans, such as:
        //
        //             |
        //       _____MatMul_______
        //      |                  |
        //      |                  |
        //     SQRT              ReLU
        //      |                  |
        // The graph driver allows nans from the branch of Sqrt, but for the
        // other branch, the driver should not tolerate that.
        ref_prim->check_correctness(
                output_args, has_eltwise, output_has_nans, res, p2p_threshold);
        if (res->state == FAILED) {
            if (is_nvidia_gpu()) {
                res->state = SKIPPED;
                res->reason = reason_t::skip_not_supported;
                return OK;
            }
            return FAIL;
        }

        mistrusted = mistrusted || (res->state == MISTRUSTED);
    }
    if (res->errors > 0) {
        res->state = FAILED;
    } else if (mistrusted) {
        res->state = MISTRUSTED;
    } else {
        res->state = PASSED;
    }

    return OK;
}

bool ref_partition_t::has_parent_op(
        const deserialized_op_t &op, bool check_all_in_lts) const {
    if (partition_ops_ref_.size() < 2) return false;

    for (const auto &in_lt : op.in_lts_) {
        const auto *parent_op = get_parent_op(in_lt.id_);
        if (!parent_op) {
            if (check_all_in_lts) return false;
            continue;
        } else {
            if (check_all_in_lts) continue;
            return true;
        }
    }

    // The logic for `check_all_in_lts=true` is exclusive along the
    // verification. If it made till the end, all lts had a parent. The logic
    // for `check_all_in_lts=false` would return during the verification, and if
    // reached the end, it means no parent was met.
    return check_all_in_lts;
}

// TODO: add get_child and remove the second arg.
bool ref_partition_t::has_child_op(const deserialized_op_t &op,
        const deserialized_op_t **child_op_ptr) const {
    if (partition_ops_ref_.size() < 2) return false;

    for (const auto &out_lt : op.out_lts_) {
        // Check if child op exist for an `op`.
        const auto &child_op = dg_->get_op_by_in_lt(out_lt.id_);
        if (child_op.empty()) continue;

        // If it does, check its ID presents in a partition.
        for (const auto &op_ref : partition_ops_ref_) {
            const auto &cur_op = op_ref.get();
            if (child_op.id_ == cur_op.id_) {
                if (child_op_ptr) *child_op_ptr = &child_op;
                return true;
            }
        }
    }

    return false;
}

const deserialized_op_t *ref_partition_t::get_parent_op(size_t in_lt_id) const {
    if (partition_ops_ref_.size() < 2) return nullptr;

    // Check if a parent op exists for an `op`.
    const auto &parent_op = dg_->get_op_by_out_lt(in_lt_id);
    if (parent_op.empty()) return nullptr;

    // If it does, check its ID presents in a partition.
    for (const auto &op_ref : partition_ops_ref_) {
        const auto &cur_op = op_ref.get();
        if (parent_op.id_ == cur_op.id_) { return &parent_op; }
    }

    return nullptr;
}

// This function decides when unfusable transcendental op output should be
// reordered to lower data type and back to f32 for a reference path.
bool ref_partition_t::need_unfusable_output_crop(const deserialized_op_t &op,
        size_t output_offset, dnnl_data_type_t &dt) const {
    const deserialized_op_t *child_op = nullptr;
    // First of all, the output should have a child op...
    if (!has_child_op(op, &child_op)) return false;
    // If the child op is not a TypeCast, it's safe to crop.
    if (child_op->kind_ != "TypeCast") {
        // Target dt in this case is the output dt of input `op`.
        dt = convert_dt(op.out_lts_[output_offset].get_data_type());
        return true;
    }
    // When it is a TypeCast (it always changes `cur_dt` <-> f32, both ways are
    // possible), there are options:
    // * If it's the last one, no crop, as f32 will happen on the other end.
    const deserialized_op_t *next_child_op = nullptr;
    if (!has_child_op(*child_op, &next_child_op)) return false;
    // * If there's a child Quantize, no crop either, since output would
    //   perform a reorder with a proper scale value to match the other end.
    if (next_child_op->kind_ == "Quantize") return false;
    // * However, a second TypeCast would negate an effect of the previous...
    if (next_child_op->kind_ == "TypeCast") {
        // Target dt in this case is the output dt of the last TypeCast.
        dt = convert_dt(next_child_op->out_lts_[output_offset].get_data_type());
        return true;
    }

    // Rest potential outcomes are default to make a crop. The target dt in
    // this case is the output dt of the child op.
    dt = convert_dt(child_op->out_lts_[output_offset].get_data_type());
    return true;
}

bool ref_partition_t::is_output_op(const deserialized_op_t &op) const {
    return std::any_of(op.out_lts_.begin(), op.out_lts_.end(),
            [this](const deserialized_lt_t &lt) {
        return std::find(partition_out_ids_.begin(), partition_out_ids_.end(),
                       lt.id_)
                != partition_out_ids_.end();
    });
}

// check the partition memory footprint of the graph path
int ref_partition_t::check_partition_total_size(
        const deserialized_op_t &op, res_t *res) {

    // Prepare the memory limit for benchdnn graph
    static size_t benchdnn_cpu_limit = get_benchdnn_cpu_limit();
    static size_t benchdnn_device_limit = get_benchdnn_device_limit();
    auto &graph_mem_req = graph_memory_req_args_t::get_instance();

    size_t new_mem_req = 0;
    // Step 1. Add input/output tensors if they are partition input/outputs.
    const auto partition_in_out_lts = get_in_out_lt_ids(op);
    for (const auto &lt_id : partition_in_out_lts) {
        if (lt_id_2_lt_.find(lt_id) == lt_id_2_lt_.end()) return FAIL;
        if (!accounted_graph_lt_ids_.insert(lt_id).second) continue;
        new_mem_req += lt_id_2_lt_.at(lt_id).create().get_mem_size();
    }

    // Step 2. Check whether the memory is enough.
    if (is_gpu()) {
        size_t total_gpu_req = graph_mem_req.get_mem_req(GPU_REQ) + new_mem_req;
        const size_t new_mapped_mem_req
                = has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
                ? 0
                : new_mem_req;
        const bool fits_device_ram = total_gpu_req <= benchdnn_device_limit;
        SAFE(check_memory_fit(fits_device_ram, total_gpu_req,
                     benchdnn_device_limit, GPU_REQ, res),
                WARN);

        const size_t combined_cpu_req = graph_mem_req.get_mem_req(CPU_REQ)
                + new_mapped_mem_req + total_gpu_req;
        const bool fits_cpu_ram = combined_cpu_req <= benchdnn_cpu_limit;
        SAFE(check_memory_fit(fits_cpu_ram, combined_cpu_req,
                     benchdnn_cpu_limit, CPU_REQ, res),
                WARN);

        graph_mem_req.increase_mem_req(GPU_REQ, GRAPH_USER, new_mem_req);
        graph_mem_req.increase_mem_req(CPU_REQ, GRAPH_USER, new_mapped_mem_req);
        graph_mem_req.increase_mapped_mem_req(GRAPH_USER, new_mapped_mem_req);
    } else {
        size_t total_cpu_req = graph_mem_req.get_mem_req(CPU_REQ) + new_mem_req;
        bool fits_cpu_ram = total_cpu_req <= benchdnn_cpu_limit;
        SAFE(check_memory_fit(fits_cpu_ram, total_cpu_req, benchdnn_cpu_limit,
                     CPU_REQ, res),
                WARN);

        graph_mem_req.increase_mem_req(CPU_REQ, GRAPH_USER, new_mem_req);
    }

    return res->state == FAILED ? FAIL : OK;
}

// check the partition memory footprint of the reference path
int ref_partition_t::check_partition_total_size(
        const check_mem_size_args_t &check_mem_size_args, bool is_output_op,
        res_t *res) {

    // Prepare the memory limit for benchdnn graph
    static size_t benchdnn_cpu_limit = get_benchdnn_cpu_limit();
    static size_t benchdnn_device_limit = get_benchdnn_device_limit();
    auto &graph_mem_req = graph_memory_req_args_t::get_instance();

    const bool is_corr = has_bench_mode_bit(mode_bit_t::corr);
    const bool is_bitwise = has_bench_mode_bit(mode_bit_t::bitwise);

    // The size of output reference memory with tag abx and f32.
    const size_t output_ref_mem_size = (is_corr || is_bitwise)
            ? check_mem_size_args.total_ref_md_size[1]
            : 0;

    // total size cpu includes:
    // 1. Memory allocated for a test obj( such as the memory for input and outputs, saved in total_size_device )
    // 2. Memory allocated for reference computation (`C` mode only)
    // 3. Memory to be allocated for comparing results(`C` mode only)
    // 4. Memory to be allocated for mapping device memory(GPU backend only)
    size_t new_cpu_req = check_mem_size_args.total_size_ref
            + check_mem_size_args.total_size_compare
            + check_mem_size_args.total_size_mapped;
    size_t new_gpu_req = check_mem_size_args.total_size_device;

    // STEP 1: Memory allocation stage for the reference path
    if (is_cpu()) new_cpu_req += check_mem_size_args.total_size_device;
    if (is_corr) {
        // If the op is not output, no need to allocate memory for correctness
        // check.
        if (!is_output_op) {
            new_cpu_req -= output_ref_mem_size;
            if (is_bitwise) new_cpu_req -= output_ref_mem_size;
        }
    }

    // STEP 2: Check whether the memory is enough
    size_t total_cpu_req = graph_mem_req.get_mem_req(CPU_REQ) + new_cpu_req;

    // GPU mem size check.
    if (is_gpu()) {
        size_t total_gpu_req = graph_mem_req.get_mem_req(GPU_REQ) + new_gpu_req;
        const bool fits_device_ram = total_gpu_req <= benchdnn_device_limit;
        SAFE(check_memory_fit(fits_device_ram, total_gpu_req,
                     benchdnn_device_limit, GPU_REQ, res),
                WARN);

        const size_t combined_cpu_req = total_cpu_req + total_gpu_req;
        const bool fits_cpu_ram = combined_cpu_req <= benchdnn_cpu_limit;
        SAFE(check_memory_fit(fits_cpu_ram, combined_cpu_req,
                     benchdnn_cpu_limit, CPU_REQ, res),
                WARN);
        graph_mem_req.increase_mem_req(GPU_REQ, REF, new_gpu_req);
        graph_mem_req.increase_mapped_mem_req(
                REF, check_mem_size_args.total_size_mapped);
    } else {
        const bool fits_cpu_ram = total_cpu_req <= benchdnn_cpu_limit;
        SAFE(check_memory_fit(fits_cpu_ram, total_cpu_req, benchdnn_cpu_limit,
                     CPU_REQ, res),
                WARN);
    }

    // STEP 3: Update the required memory size
    graph_mem_req.increase_mem_req(CPU_REQ, REF, new_cpu_req);

    return res->state == FAILED ? FAIL : OK;
}

// Return the logical tensor ids of the given op which is the input/output of
// the partition.
std::vector<size_t> ref_partition_t::get_in_out_lt_ids(
        const deserialized_op_t &op) const {
    std::vector<size_t> in_out_lt_ids;
    std::for_each(op.in_lts_.begin(), op.in_lts_.end(),
            [&in_out_lt_ids, this](const deserialized_lt_t &lt) {
        if (std::find(
                    partition_in_ids_.begin(), partition_in_ids_.end(), lt.id_)
                != partition_in_ids_.end())
            in_out_lt_ids.emplace_back(lt.id_);
    });
    std::for_each(op.out_lts_.begin(), op.out_lts_.end(),
            [&in_out_lt_ids, this](const deserialized_lt_t &lt) {
        if (std::find(partition_out_ids_.begin(), partition_out_ids_.end(),
                    lt.id_)
                != partition_out_ids_.end())
            in_out_lt_ids.emplace_back(lt.id_);
    });
    return in_out_lt_ids;
}

} // namespace graph
