/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_KERNEL_FIT_HPP
#define GEMMSTONE_TOOLS_KERNEL_FIT_HPP

#include "gemmstone/kernel_catalog.hpp"
#include "tools/benchstone/benchstone.hpp"
#include "tools/frontend.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace fit {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct DataPoint {
    int64_t m = 1, n = 1, k = 1, batch = 1;
    int64_t lda = -1, ldb = -1, ldc = -1;
    double nominal = 0., effective = 0.;
};

class Log {
public:
    Log() = default;
    explicit Log(std::ostream &stream, int verboseLevel = 1, bool showData = false)
        : verboseLevel_(verboseLevel), showData_(showData), stream_(&stream) {}

    int verboseLevel() const { return verboseLevel_; }
    bool showData() const { return showData_; }
    std::ostream *stream() const { return stream_; }

private:
    int verboseLevel_ = 0;
    bool showData_ = false;
    std::ostream *stream_ = nullptr;
};

struct Options {
    bool tightLD = false;
    bool retainMeasurements = false;
    int euCount = 0;
    double nominalFrequency = 0.;
    size_t llcSize = 0;
    Log log;
};

struct Measurements {
    bool batched = false;
    std::vector<DataPoint> points;
};

struct Result {
    gemmstone::kcatalog::Entry entry;
    std::vector<Measurements> measurements;
};

void collectData(benchstone::Driver &driver, const frontend::Result &res,
                    std::vector<DataPoint> &data, bool batchActive,
                    bool noBlock, bool needEffective, bool evalMode,
                    const Options &options = Options());

// Model W priority defaults to the product of the kernel's M and N unroll factors.
Result fit(benchstone::Driver *driver, const std::string &gemm_desc, char modelId,
                    const ngen::Product &product, double priorityOverride = 0.,
                    const Options &options = Options());
Result fit(benchstone::Driver *driver, const gemmstone::kcatalog::Entry &entry,
                    char modelId, double priorityOverride = 0.,
                    const Options &options = Options());

} // namespace fit

#endif
