/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_IMPLEMENTATION_RVV_RVV_HPP
#define RVJIT_IMPLEMENTATION_RVV_RVV_HPP

#include <algorithm>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {

/// The SEW constant associated to an operand type
///
/// @param dt The operand type
inline SEW sew_for(const type_t &dt) {
    switch (rvjit::sewb(dt)) {
        case 8: return SEW::e64;
        case 4: return SEW::e32;
        case 2: return SEW::e16;
        default: return SEW::e8;
    }
}

/// Obtains an LMUL representation for an integer group size parameter
///
/// @param group_size The number of registers in a group
///
/// @note Defaults to m1 for group sizes other than 2, 4, and 8
inline LMUL lmul_for(const int group_size) {
    switch (group_size) {
        case 1: return LMUL::m1;
        case 2: return LMUL::m2;
        case 4: return LMUL::m4;
        case 8: return LMUL::m8;
        default: return LMUL::m1;
    }
}

/// Obtain the number of vector register groups for a given LMUL setting
///
/// @param m The LMUL setting
inline int get_nvgroups(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 32;
        case LMUL::m2: return 16;
        case LMUL::m4: return 8;
        case LMUL::m8: return 4;
        default: return 1;
    }
}

/// Checks if LMUL A generates vectors larger than or equal to B
///
/// @param A The LMUL setting compared
/// @param B The LMUL setting compared against
inline bool is_lmul_gte(const LMUL &A, const LMUL &B) {
    return vgroup_size(A) >= vgroup_size(B);
}

/// Get the single-element-width size, in bytes, associated to a SEW setting
///
/// @param s The SEW setting
inline int sewb(const SEW &s) {
    switch (s) {
        case SEW::e8: return 1;
        case SEW::e16: return 2;
        case SEW::e32: return 4;
        case SEW::e64: return 8;
        default: return 0;
    }
}

/// Obtains the LMUL representation for a base LMUL and in/out SEW settings
///
/// @param base The LMUL setting of the input
/// @param in The SEW setting of the input
/// @param out The SEW setting of the output
inline LMUL lmul_for(LMUL base, SEW in, SEW out) {
    return lmul_for(vgroup_size(base) * (sewb(out) / sewb(in)));
}

/// Computes the maximum vector length given a certain vector type settings
///
/// @param vlenb The vector register length in bytes
/// @param sew The SEW setting
/// @param lmul The LMUL setting
inline int get_maximum_vector_length(const int vlenb, const SEW &sew = SEW::e8,
        const LMUL &lmul = LMUL::m1) {
    return vlenb * vgroup_size(lmul) / sewb(sew);
}

/// Hardware model driving arithmetic micro-kernel unrolling decisions
struct vpu_t {
    int vlenb = 0;
    int vlaneb = 0;
    LMUL max_lmul = LMUL::m8;
    LMUL lmul_preference = LMUL::m1;

    // Cache level feeding the vector loads/stores (not necessarily L1)
    int cache_size = 0;
    int cache_line_size = 0;
    int cache_latency = 0;

    vpu_t() = default;
    virtual ~vpu_t() = default;

    /// Creates a model from its derived fields
    ///
    /// @param vlenb The vector register length in bytes
    /// @param vlaneb The vector lane width in bytes
    /// @param max_lmul The largest LMUL setting worth using
    /// @param lmul_preference The LMUL setting used unless forced
    /// @param cache_size The size of the cache level feeding the vector unit
    /// @param cache_line_size The line size of that cache level
    /// @param cache_latency The latency of that cache level
    vpu_t(int vlenb, int vlaneb, LMUL max_lmul, LMUL lmul_preference,
            int cache_size, int cache_line_size, int cache_latency)
        : vlenb(vlenb)
        , vlaneb(vlaneb)
        , max_lmul(max_lmul)
        , lmul_preference(lmul_preference)
        , cache_size(cache_size)
        , cache_line_size(cache_line_size)
        , cache_latency(cache_latency) {}

    /// The latency of a vector instruction, in lane passes
    int vector_latency() const { return vlaneb > 0 ? vlenb / vlaneb : 1; }

    /// The latency of a vector memory instruction
    int memory_latency() const { return cache_latency; }

    /// Fills the model in from explicit hardware parameters
    ///
    /// @param vlen The vector register length in bits
    /// @param cache_size The size of the cache level feeding the vector unit
    /// @param cache_line_size The line size of that cache level
    /// @param cache_latency The latency of that cache level
    ///
    /// @note The parameters are plain numbers the caller has already
    ///     obtained; this method performs no hardware querying of its own
    virtual void from_params(
            int vlen, int cache_size, int cache_line_size, int cache_latency) {
        // Long vector architecture vlen
        static constexpr int vlong = 2048;

        // Vector architectural storage
        vlenb = vlen / 8;

        // Guess the vector lane width from vlen: short vectors use a
        // pipelined 64-bit SIMD unit, mid ones widen to control latency,
        // and long ones match the cache-line size.
        vlaneb = vlen <= 256 ? 8 : vlen <= 1024 ? 16 : cache_line_size;

        // Empirical observation: limit vector grouping to reach 2048 bits
        max_lmul = LMUL::m8;
        for (LMUL m : {LMUL::m1, LMUL::m2, LMUL::m4, LMUL::m8}) {
            if (vlen * vgroup_size(m) >= vlong) {
                max_lmul = m;
                break;
            }
        }

        // Set a preferred LMUL setting with enough registers to mask latency
        lmul_preference = LMUL::m1;
        const int lat = vlaneb > 0 ? vlenb / vlaneb : 0;
        for (LMUL m : {LMUL::m4, LMUL::m2, LMUL::m1}) {
            if (is_lmul_gte(max_lmul, m) && lat <= get_nvgroups(m)) {
                lmul_preference = m;
                break;
            }
        }

        this->cache_size = cache_size;
        this->cache_line_size = cache_line_size;
        this->cache_latency = cache_latency;
    }
};

// Internal linkage avoiding an out-of-line definition
static const Reg ignore_out(0); // Ignore `set_vtype`'s own output

/// Vector type settings (SEW, LMUL, policies) of a micro-kernel
///
/// @details Input and output views of SEW/LMUL support mixed precision
struct vtype_t {
    /// Selects which SEW/LMUL pair to apply on a vtype setter operation
    ///
    /// @note `output` is the accumulators'
    enum class cfg_t { input, output };

    SEW sewi = SEW::e8; // ESEW for input operands
    SEW sewo = SEW::e8; // ESEW for output operands
    LMUL lmuli = LMUL::m1; // ELMUL for input operands
    LMUL lmulo = LMUL::m1; // ELMUL for output operands
    VTA vt = VTA::ta; // Vector tail policy, uses agnostic as default
    VMA vm = VMA::ma; // Vector mask policy, uses agnostic as default
    Reg avl {}; // AVL register consumed directly by set_vtype()'s vsetvli

    /// Configures vtype for an uniform-precision code segment
    ///
    /// @param sew The SEW setting of inputs and outputs
    /// @param lmul The LMUL setting of inputs and outputs
    ///
    /// @note Does not validate `lmul` against any hardware ceiling; the
    ///     caller is responsible for passing an `lmul` its target supports
    void for_uniform(SEW sew, LMUL lmul) {
        sewi = sewo = sew;
        lmuli = lmulo = lmul;
    }

    /// Configures vtype for a widening mixed-precision code segment
    ///
    /// @param sewi The SEW setting of the inputs
    /// @param sewo The SEW setting of the outputs
    /// @param lmul The LMUL setting of the inputs, from which the LMUL of the
    ///     outputs is derived
    ///
    /// @note Does not validate the sewi/sewo ratio or lmul/the derived
    ///     lmulo against any hardware ceiling
    void for_mixed(SEW sewi, SEW sewo, LMUL lmul) {
        this->sewi = sewi;
        this->sewo = sewo;
        lmuli = lmul;
        lmulo = lmul_for(lmul, sewi, sewo);
    }

    /// Whether input and output share the same SEW/LMUL settings
    bool is_uniform() const { return sewi == sewo && lmuli == lmulo; }

    /// Whether input and output have different SEW/LMUL settings
    bool is_mixed() const { return !is_uniform(); }
};

} // namespace rvv
} // namespace rvjit

#endif
