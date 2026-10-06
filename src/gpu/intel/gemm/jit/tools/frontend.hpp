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

#ifndef GEMMSTONE_TOOLS_FRONTEND_HPP
#define GEMMSTONE_TOOLS_FRONTEND_HPP

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "gemmstone/config.hpp"

#include "ngen.hpp"

namespace frontend {

struct Result {
    Result() = default;

    // Shortcut initializer for a failure state with an error message
    Result(std::string &&msg) {
        *this = Result{};
        info = msg;
    }
    
     static Result fromTmpFile();

    std::string info, asmStr;
    std::vector<uint8_t> binary;
    bool success = false;
};

struct Generator {
    // Without a product, the device is detected (GEN/FAMILY, else OpenCL).
    static std::unique_ptr<Generator> Create();
    static std::unique_ptr<Generator> Create(ngen::Product product);
    static bool IsSupported(ngen::HW hw);
    virtual ~Generator() noexcept(false) = default;
    virtual void cancelAutoSWSB() = 0;
    virtual Result generate(const std::string &kernelArgs, bool infoOnly = false) = 0;

    std::string failureMessage = "";
protected:
    Generator() {};
};

namespace async {
struct Job {
    Job();
    Job(const std::string &kernel_) : Job() {
        kernel = kernel_;
        launch();
    }
    ~Job();

    void launch();
    Result finish();

    bool empty() const { return kernel.empty(); }
    void clear() { kernel.clear(); }

    // Non-blocking (or bounded-wait) readiness check.
    bool ready(std::chrono::duration<double> maxWait = std::chrono::duration<double>(0)) const {
        if (!future.valid()) return true;
        return future.wait_for(maxWait) == std::future_status::ready;
    }

    std::string kernel;
    std::future<Result> future;
    std::chrono::steady_clock::time_point launchTime;
private:
    ngen::Product product;
    std::unique_ptr<Generator> generator;
};

} /* namespace async */

} /* namespace frontend */

#endif
