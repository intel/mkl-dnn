/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_SEND_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_SEND_HPP

#include "dsl/ir/codegen/kernel.hpp"
#include "dsl/ir/codegen/register_scope.hpp"
#include "dsl/ir/send.hpp"
#include "ngen.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

inline ngen::CacheSettingsLSC get_cache_settings(
        const send_t &send, const hw_t &hw) {
    auto ret = ngen::CacheSettingsLSC::Default;
    bool is_load = send.is_load() || send.is_load_2d();
    bool is_store = send.is_store() || send.is_store_2d();
    bool is_prefetch = send.is_prefetch() || send.is_prefetch_2d();
    switch (send.cache_hint) {
        case send_cache_hint_t::undef:
            switch (send.hw.ngen_hw()) {
                case ngen::HW::XeHPG:
                    // Use default cache policy on xelpg to avoid suspected driver issue.
                    if (is_store && hw.systolic_support())
                        ret = ngen::CacheSettingsLSC::L1WB_L3WB;
                    break;
                case ngen::HW::XeHPC:
                    if (is_store) {
                        ret = ngen::CacheSettingsLSC::L1UC_L3WB;
                    } else if (is_load || is_prefetch) {
                        ret = ngen::CacheSettingsLSC::L1C_L3C;
                    }
                    break;
                case ngen::HW::Xe3p:
                    if (is_store) {
                        ret = ngen::CacheSettingsLSC::L1WB_L2WB_L3UC;
                    } else if (is_load || is_prefetch) {
                        ret = ngen::CacheSettingsLSC::L1C_L2C_L3UC;
                    }
                    break;
                default: break;
            }
            break;
        case send_cache_hint_t::load_once:
            ret = ngen::CacheSettingsLSC::L1C_L3C;
            break;
        case send_cache_hint_t::hw_default:
            ret = ngen::CacheSettingsLSC::Default;
            break;
    }
    return ret;
}

template <typename DataSpecT, typename = void>
struct atomic_helper_t {
    template <typename GeneratorT>
    static void call(GeneratorT *, ngen::AtomicOp,
            const ngen::InstructionModifier &, const DataSpecT &,
            ngen::AddressBase, const ngen::RegData &, const ngen::RegData &) {
        dsl_error() << "Unknown DataSpec: atomics are not supported.";
    }
};

template <typename DataSpecT>
struct atomic_helper_t<DataSpecT,
        typename std::enable_if<
                std::is_same<DataSpecT, ngen::scattered_qword>::value
                || std::is_same<DataSpecT,
                        ngen::scattered_dword>::value>::type> {
    template <typename GeneratorT>
    static void call(GeneratorT *host, ngen::AtomicOp atomic_op,
            const ngen::InstructionModifier &mod, const DataSpecT &spec,
            ngen::AddressBase base, const ngen::RegData &addr,
            const ngen::RegData &data) {
        host->atomic(atomic_op, mod, spec, base, addr, data);
    }

    template <typename GeneratorT>
    static void call(GeneratorT *host, ngen::AtomicOp atomic_op,
            const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const DataSpecT &spec, ngen::AddressBase base,
            const ngen::RegData &addr, const ngen::RegData &data) {
        host->atomic(atomic_op, mod, dst, spec, base, addr, data);
    }
};

// Helper to emit send instructions.
class send_impl_t {
public:
    send_impl_t(const send_t &send) : send_(send) {}

    template <typename GeneratorT, typename T>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::InstructionModifier &mod, const ngen::RegData &header,
            const T &data) {
        if (send_.is_2d()) {
            emit_2d(host, mod, data, header);
            return;
        }

        if (send_.is_lsc) {
            emit_lsc(host, mod, data, header);
            return;
        }

        auto address_base = to_address_base(send_.address);

        int elems = send_.type.elems();
        auto &t = send_.type;
        if (t.is_byte())
            emit_load_or_store(host, mod, ngen::scattered_byte(elems),
                    address_base, header, data);
        else if (t.is_dword())
            emit_load_or_store(host, mod, ngen::scattered_dword(elems),
                    address_base, header, data);
        else if (t.is_qword())
            emit_load_or_store(host, mod, ngen::scattered_qword(elems),
                    address_base, header, data);
        else if (t.is_oword())
            emit_load_or_store(host, mod, ngen::block_oword(elems),
                    address_base, header, data);
        else if (t.is_hword())
            emit_load_or_store(host, mod, ngen::block_hword(elems),
                    address_base, header, data);
        else
            dsl_error() << t;
    }

    template <typename GeneratorT, typename T>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::InstructionModifier &mod, const T &dst,
            const ngen::RegData &header, const T &data) {
        if (send_.type.is_dword())
            emit_atomic_cmpwr(host, mod, dst,
                    ngen::scattered_dword(send_.type.elems()),
                    to_address_base(send_.address), header, data);
        else if (send_.type.is_qword())
            emit_atomic_cmpwr(host, mod, dst,
                    ngen::scattered_qword(send_.type.elems()),
                    to_address_base(send_.address), header, data);
        else
            dsl_error() << send_.type;
    }

private:
    template <typename GeneratorT, typename DataSpecT>
    void emit_load_or_store(GeneratorT *host,
            const ngen::InstructionModifier &mod, const DataSpecT &spec,
            ngen::AddressBase base, const ngen::RegData &addr,
            const ngen::RegData &data) {
        if (send_.is_load()) {
            host->load(mod, data, spec, base, addr);
        } else if (send_.is_atomic()) {
            atomic_helper_t<DataSpecT>::call(
                    host, to_atomic_op(send_.op), mod, spec, base, addr, data);
        } else if (send_.is_store()) {
            host->store(mod, spec, base, addr, data);
        } else {
            dsl_error() << "Can't emit send: " << send_;
        }
    }
    template <typename GeneratorT, typename DataSpecT>
    void emit_atomic_cmpwr(GeneratorT *host,
            const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const DataSpecT &spec, ngen::AddressBase base,
            const ngen::RegData &addr, const ngen::RegData &data) {
        atomic_helper_t<DataSpecT>::call(
                host, ngen::AtomicOp::cmpwr, mod, dst, spec, base, addr, data);
    }

    template <typename GeneratorT>
    void emit_lsc(GeneratorT *host, const ngen::InstructionModifier &mod,
            const ngen::RegData &data, const ngen::RegData &header) {

        auto get_lsc_type = [&](const type_t &type, bool is_block) {
            if (!send_.is_block()) return type;
            for (auto &t : {type_t::qword(), type_t::dword()}) {
                if (type.size() % t.size() == 0) {
                    int elems = type.size() / t.size();
                    dsl_assert(is_pow2(elems));
                    dsl_assert(elems >= 1 && elems <= 64);
                    return t.with_elems(elems);
                }
            }
            stub();
            return type;
        };

        std::unique_ptr<ngen::DataSpecLSC> lsc_spec;
        auto lsc_type = to_data_lsc(get_lsc_type(send_.type, send_.is_block()));
        if (send_.is_scattered()) {
            lsc_spec = make_unique<ngen::DataSpecLSC>(
                    ngen::scattered(lsc_type.first, lsc_type.second));
        } else if (send_.is_block()) {
            lsc_spec = make_unique<ngen::DataSpecLSC>(
                    ngen::block(lsc_type.first, lsc_type.second));
        } else {
            stub();
        }

        if (send_.is_slm()) {
            if (send_.is_load()) {
                host->load.slm(mod, data, *lsc_spec, host->SLM, header);
            } else if (send_.is_store()) {
                host->store.slm(mod, *lsc_spec, host->SLM, header, data);
            } else {
                stub();
            }
        } else if (send_.is_a64()) {
            *lsc_spec |= get_cache_settings(send_, host->hw_info());
            if (send_.is_load() || send_.is_prefetch()) {
                if (host->hw_info() >= ngen::HW::Xe3p)
                    *lsc_spec |= ngen::DataSpecLSC::createOverfetch();
                host->load.ugm(mod, data, *lsc_spec, host->A64, header);
            } else if (send_.is_store()) {
                host->store.ugm(mod, *lsc_spec, host->A64, header, data);
            } else if (send_.is_atomic()) {
                host->atomic.ugm(to_atomic_op(send_.op), mod, *lsc_spec,
                        to_address_base(send_.address), header, data);
            }
        } else {
            stub();
        }
    }

    template <typename GeneratorT>
    void emit_2d(GeneratorT *host, const ngen::InstructionModifier &mod,
            const ngen::RegData &data, const ngen::RegData &header) {
        auto &info = send_.block_2d_info;
        ngen::DataSizeLSC data_size = ngen::DataSizeLSC::D8;
        switch (send_.type.size()) {
            case 1: data_size = ngen::DataSizeLSC::D8; break;
            case 2: data_size = ngen::DataSizeLSC::D16; break;
            case 4: data_size = ngen::DataSizeLSC::D32; break;
            default: stub();
        }
        ngen::DataSpecLSC data_spec(data_size);
        if (info.vnni) data_spec |= host->vnni;
        if (info.transpose) data_spec |= host->transpose;
        ngen::block_2d spec(data_spec, info.width, info.height, info.count);
        spec |= get_cache_settings(send_, host->hw_info());
        if (send_.is_load_2d() || send_.is_prefetch_2d()) {
            host->load(mod, data, spec, host->A64, header);
        } else if (send_.is_store_2d()) {
            host->store(mod, spec, host->A64, header, data);
        } else {
            stub();
        }
    }

    static std::pair<ngen::DataSizeLSC, int> to_data_lsc(const type_t &type) {
        switch (type.scalar().size()) {
            case 1: {
                if (type.elems() == 1)
                    return std::make_pair(ngen::DataSizeLSC::D8U32, 1);
                if (type.elems() == 2)
                    return std::make_pair(ngen::DataSizeLSC::D16U32, 1);
                if (type.elems() == 4)
                    return std::make_pair(ngen::DataSizeLSC::D32, 1);
                if (type.elems() == 8)
                    return std::make_pair(ngen::DataSizeLSC::D64, 1);
                break;
            }
            case 2: {
                if (type.elems() == 1)
                    return std::make_pair(ngen::DataSizeLSC::D16U32, 1);
                if (type.elems() == 2)
                    return std::make_pair(ngen::DataSizeLSC::D32, 1);
                if (type.elems() == 4)
                    return std::make_pair(ngen::DataSizeLSC::D64, 1);
                break;
            }
            case 4: return std::make_pair(ngen::DataSizeLSC::D32, type.elems());
            case 8: return std::make_pair(ngen::DataSizeLSC::D64, type.elems());
            default: break;
        }
        stub();
        return std::make_pair(ngen::DataSizeLSC::D8, 1);
    }

    static ngen::AddressBase to_address_base(send_address_t address) {
        switch (address) {
            case send_address_t::a64: return ngen::AddressBase::createA64(true);
            case send_address_t::slm: return ngen::AddressBase::createSLM();
            default: stub();
        }
        return ngen::AddressBase();
    }

    static ngen::AtomicOp to_atomic_op(send_op_t op) {
        switch (op) {
            case send_op_t::atomic_add: return ngen::AtomicOp::add;
            case send_op_t::atomic_fadd: return ngen::AtomicOp::fadd;
            case send_op_t::atomic_bfadd: return ngen::AtomicOp::bfadd;
            case send_op_t::atomic_cmpwr: return ngen::AtomicOp::cmpwr;
            default: stub();
        }
        return ngen::AtomicOp();
    }

    const send_t &send_;
};

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
