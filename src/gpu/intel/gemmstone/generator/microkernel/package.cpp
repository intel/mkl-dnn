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

#include "gemmstone/microkernel/package.hpp"
#include "ngen_decoder.hpp"
#include "npack/neo_packager.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

using namespace ngen;

Package::Status Package::finalize(const ClobberSet &knownClobbers) {
    using namespace ngen;

    auto &status = this->status;
    status = Status::Success;

    auto product = npack::decodeHWIPVersion(gmdidCompat);
    auto hw = getCore(product.family);

    if (hw == HW::Unknown) {
        status = Status::UnsupportedHW;
        return status;
    }

    Decoder decoder(hw, binary);

    auto clobbered = knownClobbers;
    int maxAccIdx = -1;

    for (; !decoder.done(); decoder.advance()) {
        // Check for systolic usage.
        auto op = decoder.opcode();
        systolic |= (op == Opcode::dpas || op == Opcode::dpasw);

        // Get destination region and add to clobbers. This indeterminate for
        // indirect or variable sized destinations. In this case, rely on
        // knownClobbers.
        DependencyRegion dstRegion;
        if (!decoder.getOperandRegion(dstRegion, -1)) continue;

        if (dstRegion.rf == RegFileGRF) {
            if (dstRegion.unspecified
                && !(dstRegion.isValid() && knownClobbers[dstRegion.base])) {
                    status = Status::UncertainClobbers;
            } else
                for (int j = 0; j < dstRegion.size; j++)
                    clobbered[dstRegion.base + j] = true;
        }

        // Allow acc/flag ARF destinations only if declared in knownClobbers.
        if (dstRegion.rf == RegFileARF) {
            ARFType dstType = static_cast<ARFType>(dstRegion.base >> 4);
            int arfIdx = dstRegion.base & 0xF;
            int arfLen = dstRegion.size;
            bool known = false;
            if (dstType == ARFType::acc) {
                known = true;
                for (int j = arfIdx; j < arfIdx + arfLen; j++)
                    known &= knownClobbers.acc(j);
                maxAccIdx = std::max(maxAccIdx, arfIdx + arfLen - 1);
            } else if (dstType == ARFType::f) {
                known = true;
                for (int j = arfIdx; j < arfIdx + arfLen; j++)
                    known &= knownClobbers.flag(j);
            }
            if (!known)
                status = Status::UncertainClobbers;
        }
    }

    // Group clobber array into consecutive ranges.
    clobbers.clear();

    int regBytes = GRF::bytes(hw);
    int base = 0, len = 0;
    for (int j = 0; j < int(clobbered.size()); j++) {
        if (clobbered[j]) {
            if (len > 0)
                len++;
            else
                base = j, len = 1;
        } else if (len > 0) {
            clobbers.emplace_back(
                    RegisterRange(base * regBytes, len * regBytes));
            len = 0;
        }
    }
    if (len > 0)
        clobbers.emplace_back(RegisterRange(base * regBytes, len * regBytes));

    // Capture GRF usage from clobbers and arguments.
    uint32_t last = 0;
    if (!clobbers.empty()) {
        auto &final = clobbers.back();
        last = final.boffset + final.blen;
    }
    for (const auto &argument : arguments)
        for (auto &range : argument.location)
            last = std::max(last, range.boffset + range.blen);

    grfMin = (last + regBytes - 1) / regBytes;

    int nacc = AccumulatorRegister::count(hw, grfMin);
    if (nacc <= maxAccIdx) {
        // Upgrade to 8 accumulators
        grfMin = 256;
    }

    // Generate LUID from hash of kernel. Later, the cataloguer can update it in case of collisions.
    uint32_t luid = 0;
    uint32_t multiplier = 1357;

    auto *u32ptr = (const uint32_t *)binary.data();
    for (size_t i = 0; i < (binary.size() >> 2); i++) {
        luid ^= u32ptr[i] * multiplier;
        multiplier += 2;
        luid = (luid << 3) | (luid >> 29);
    }

    this->luid = luid;

    return status;
}

}
GEMMSTONE_NAMESPACE_END
