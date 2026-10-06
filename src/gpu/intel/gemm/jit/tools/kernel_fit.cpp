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

#include "tools/kernel_fit.hpp"
#include "tools/kernel_catalog.hpp"

#include "gemmstone/config.hpp"
#include "tools/benchstone/benchstone.hpp"
#include "tools/benchstone/kernel_info.hpp"
#include "tools/device_utils.hpp"
#include "tools/frontend.hpp"
#include "tools/utils.hpp"

#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/driver_info.hpp"
#include "internal/utils.hpp"

#ifndef GEMMSTONE_WITH_MKL
#include "mini_mkl.hpp"
#elif __has_include("mkl/mkl.h")
#include "mkl/mkl.h"
#else
#include "mkl.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

using namespace gemmstone;

namespace fit {

namespace {

class StreamState {
public:
    explicit StreamState(std::ostream &stream)
        : stream_(stream), flags_(stream.flags()), precision_(stream.precision()), fill_(stream.fill())
    {}

    ~StreamState()
    {
        stream_.flags(flags_);
        stream_.precision(precision_);
        stream_.fill(fill_);
    }

private:
    std::ostream &stream_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
};

/*******************/
/* Data Collection */
/*******************/

using Evaluator = std::function<double(int64_t, int64_t, int64_t)>;

using utils::contains;
using utils::divUp;
using utils::alignUp;
using utils::extractWords;

void remove(std::string &haystack, const std::string &needle)
{
    if (needle.empty()) return;
    auto needleChars = needle.length();
    while (true) {
        auto pos = haystack.rfind(needle);
        if (pos == std::string::npos) break;
        haystack.erase(pos, needleChars);
    }
}

bool isPackedChar(const kcatalog::string &layout)
{
    return one_of(layout[0], {'A', 'B'});
}

double extPrecisionBytes(const kcatalog::string &precision, bool isC)
{
    std::string_view view(precision);
    char ext = view.at(0);
    if (ext == '[')
        ext = view.at(isC ? 2 : 1);
    auto T = charPrecision(ext);
    return double(T.paddedSize()) / double(T.perByte());
}

size_t llcSize(int eus, size_t cacheSize, const ngen::Product &product)
{
    const size_t MB = 1048576;
    if (cacheSize >= MB)
        return cacheSize;

    switch (catalog::tag(ngen::getCore(product.family))) {
        case kcatalog::HWTagGen12LP: return 12 * MB;
        case kcatalog::HWTagXeHPG:   return (eus > 256) ? 16 * MB : (eus > 128) ? 8 * MB : 4 * MB;
        case kcatalog::HWTagXeHPC:   return 256 * MB;
        case kcatalog::HWTagXe2:     return 8 * MB;
        case kcatalog::HWTagXe3:     return 8 * MB;
        case kcatalog::HWTagXe3p:    return 192 * MB;
        default: throw Error("unknown architecture; could not estimate LLC size");
    }
}

void collectData1(benchstone::Driver &driver, const frontend::Result &res,
                     std::vector<DataPoint> &data, bool batchActive, bool noBlock, bool evalMode)
{
    if (driver.isKernelLoaded())
        driver.finishKernel();

    driver.Progress = false;
    driver.TimingOnly = true;
    driver.NoBlocking = noBlock;
    driver.TestBeta0 = false;
    driver.FixWGK = false;
    if (!driver.isSimulation()) {
        driver.TimeMaxReps = 100;
        driver.RelativeMeanStdDevGoal = 0.001;
    }
    if (evalMode) {
        driver.TestBeta0 = true;
        driver.FixWGK = true;
    }

    if (res.binary.empty())
        throw Error("could not load kernel");

    KernelInfo info(res.info);
    if (!driver.initKernel(info, res.binary))
        throw Error("could not initialize kernel");

    int64_t maxM = 1, maxN = 1, maxK = 1, maxBatch = 1;
    int64_t single_elems = 0;
    for (const auto &dp: data) {
        maxM = std::max(maxM, dp.m);
        maxN = std::max(maxN, dp.n);
        maxK = std::max(maxK, dp.k);
        int64_t batch = (batchActive && info.isBatched()) ? dp.batch : 1;
        maxBatch = std::max(maxBatch, batch);
        single_elems += batch * (dp.m * dp.n + dp.m * dp.k + dp.n * dp.k);
    }

    int64_t series_elems = maxBatch * (maxM * maxN + maxM * maxK + maxN * maxK);
    bool series = series_elems < single_elems;
    auto runTest = [&](TestMode mode, uint32_t batch, int32_t m, int32_t n, int32_t k,
                       int32_t lda, int32_t ldb, int32_t ldc) {
        auto result = driver.test(info, mode, batch, m, n, k, lda, ldb, ldc,
                                  false, false, false, 'F', false, false);
        if (result.accuracy == TestAccuracy::RuntimeError)
            throw Error(std::string("benchstone runtime error: ") + result.failingCase);
        return result;
    };

    if (series)
        (void) runTest(TestMode::BeginSeries, maxBatch, maxM, maxN, maxK, -1, -1, -1);

    for (auto &dp: data) {
        auto mode = series ? TestMode::TimeOneOfSeries : TestMode::Time;
        auto result = runTest(mode, uint32_t(dp.batch), int32_t(dp.m), int32_t(dp.n), int32_t(dp.k),
                              int32_t(dp.lda), int32_t(dp.ldb), int32_t(dp.ldc));

        dp.nominal = result.time;
        dp.effective = result.effTime;
    }

    if (series)
        (void) runTest(TestMode::EndSeries, maxBatch, maxM, maxN, maxK, -1, -1, -1);

    driver.finishKernel();
}

bool cleanUpEffectiveData(std::vector<DataPoint> &data, double minGoodFraction = 0.8)
{
    double maxFreqDrop = 1.0;
    int total = int(data.size()), good = 0;
    for (const auto &dp: data) {
        if (dp.effective > 0.) {
            good++;
            maxFreqDrop = std::max(maxFreqDrop, dp.nominal / dp.effective);
        }
    }

    if (good == 0)
        throw Error("unable to measure GPU frequency");
    else if (good == total)
        return true;

    for (auto &dp: data)
        if (dp.effective == 0.)
            dp.effective = dp.nominal;

    return (good >= total * minGoodFraction) && (maxFreqDrop <= 1.05);
}

} // namespace

void collectData(benchstone::Driver &driver, const frontend::Result &res,
                    std::vector<DataPoint> &data, bool batchActive,
                    bool noBlock, bool needEffective, bool evalMode, const Options &options)
{
    if (data.empty()) return;
    if (aubCaptureEnabled()) throw Error("cannot run benchmarks, aub capture is enabled");

    if (options.tightLD)
        for (auto &dp: data)
            if (dp.lda < 0 && dp.ldb < 0 && dp.ldc < 0)
                dp.lda = dp.ldb = dp.ldc = 0;

    collectData1(driver, res, data, batchActive, noBlock, evalMode);

    if (!needEffective) return;

    if (simulation()) {
        for (auto &dp: data)
            dp.effective = dp.nominal;
        return;
    }

    auto ok = cleanUpEffectiveData(data);
    if (!ok) {
        collectData1(driver, res, data, batchActive, noBlock, evalMode);
        ok = cleanUpEffectiveData(data);
        if (!ok)
            throw Error("unable to collect enough frequency data");
    }
}

namespace {

struct Fitter {
    const Options &options;
    Result &result;

    std::string modifiers;

    Fitter(const Options &options_, Result &result_, const std::string &modifiers_)
        : options(options_), result(result_), modifiers(modifiers_) {}

    std::ostream &fitLog() const { return *options.log.stream(); }

    void collectData(benchstone::Driver &driver, const frontend::Result &res, std::vector<DataPoint> &data,
                     bool batchActive, bool noBlock = false, bool needEffective = false, bool evalMode = false);
    void verboseFit(const char *params, const char *method, const double *values, int nvalues, double fitQuality = -1.);
    void verboseFit(const char *params, const char *method, double v0, double fitQuality = -1.);
    void verboseFit(const char *params, const char *method, double v0, double v1, double fitQuality);
    double relativeRMSE(const std::vector<DataPoint> &data, const std::function<double(const DataPoint &)> &predict,
                        bool effective, const Evaluator &eval = nullptr);
    double linearRegression(const std::vector<DataPoint> &data, const Evaluator &eval,
                            double &c0, double &c1, bool effective = false);
    double clampedLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval,
                                   double &c0, double &c1, double low, double high, bool effective = false);
    double scaledMinimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c1, bool effective = false);
    double minimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c, bool effective = false);
    double mixCoeffFit(const std::vector<DataPoint> &data, const Evaluator &eval0, const Evaluator &eval1,
                       double &c, bool effective = false);
    double fixedMinLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval,
                                    double &c0, double &c1, bool effective = false);
    frontend::Result prepareKernel(const kcatalog::Entry &e, bool &batchActive,
                                   const std::string &alpha = "1", const std::string &beta = "0",
                                   const std::string &additions = "") const;
    void fitModelE(kcatalog::Entry &e, benchstone::Driver &driver);
    void fitModelS(kcatalog::Entry &e, benchstone::Driver &driver);
};

void Fitter::collectData(benchstone::Driver &driver, const frontend::Result &res, std::vector<DataPoint> &data,
                         bool batchActive, bool noBlock, bool needEffective, bool evalMode)
{
    fit::collectData(driver, res, data, batchActive, noBlock, needEffective, evalMode, options);
    if (options.retainMeasurements && !data.empty()) {
        Measurements measurements;
        measurements.batched = batchActive;
        measurements.points = data;
        result.measurements.push_back(std::move(measurements));
    }
}

/*****************/
/* Fit Utilities */
/*****************/

void Fitter::verboseFit(const char *params, const char *method, const double *values, int nvalues, double fitQuality)
{
    if (options.log.verboseLevel() <= 0) return;

    auto &log = fitLog();
    StreamState state(log);
    log << "  " << std::left << std::setw(12) << params << " " << std::setw(26) << method << " =";
    for (int i = 0; i < nvalues; i++)
        log << " " << std::right << std::setw(12) << std::setprecision(6) << values[i];
    if (fitQuality >= 0.)
        log << "  (relerr=" << std::fixed << std::setprecision(1) << fitQuality << "%)" << std::defaultfloat;
    log << "\n";
}

void Fitter::verboseFit(const char *params, const char *method, double v0, double fitQuality)
{
    verboseFit(params, method, &v0, 1, fitQuality);
}

void Fitter::verboseFit(const char *params, const char *method, double v0, double v1, double fitQuality)
{
    double vals[2] = {v0, v1};
    verboseFit(params, method, vals, 2, fitQuality);
}

// Compute relative RMSE (%) of a fit: 100 * sqrt(mean(((y - yhat) / y)^2)).
// Skips data points where y is near zero.
double Fitter::relativeRMSE(const std::vector<DataPoint> &data, const std::function<double(const DataPoint &)> &predict,
                            bool effective, const Evaluator &eval)
{
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;
    double sumRelSq = 0.;
    int count = 0;

    if (options.log.showData()) {
        auto &log = fitLog();
        StreamState state(log);
        if (eval)
            log << "    " << std::setw(8) << "m" << " " << std::setw(8) << "n" << " " << std::setw(8) << "k"
                << " " << std::setw(12) << "x" << " " << std::setw(12) << "y" << " "
                << std::setw(12) << "predicted" << " " << std::setw(11) << "relerr%" << "\n";
        else
            log << "    " << std::setw(8) << "m" << " " << std::setw(8) << "n" << " " << std::setw(8) << "k"
                << " " << std::setw(12) << "y" << " " << std::setw(12) << "predicted" << " "
                << std::setw(11) << "relerr%" << "\n";
    }

    for (auto &dp: data) {
        double y = dp.*fit;
        double yhat = predict(dp);
        double x = eval ? eval(dp.m, dp.n, dp.k) : 0.;
        bool ignore = (x == 0.) && (std::abs(yhat) < 1e-20);
        double rel = (std::abs(y) < 1e-20 || ignore) ? 0. : (y - yhat) / y;

        if (options.log.showData()) {
            auto &log = fitLog();
            StreamState state(log);
            if (eval)
                log << "    " << std::setw(8) << dp.m << " " << std::setw(8) << dp.n << " " << std::setw(8) << dp.k
                    << " " << std::setw(12) << std::setprecision(6) << x << " " << std::setw(12) << y << " "
                    << std::setw(12) << yhat << " " << std::showpos << std::fixed << std::setprecision(1)
                    << std::setw(10) << 100. * rel << "%" << std::noshowpos << std::defaultfloat << "\n";
            else
                log << "    " << std::setw(8) << dp.m << " " << std::setw(8) << dp.n << " " << std::setw(8) << dp.k
                    << " " << std::setw(12) << std::setprecision(6) << y << " " << std::setw(12) << yhat << " "
                    << std::showpos << std::fixed << std::setprecision(1) << std::setw(10) << 100. * rel << "%"
                    << std::noshowpos << std::defaultfloat << "\n";
        }

        if (std::abs(y) < 1e-20) continue;
        sumRelSq += rel * rel;
        count++;
    }
    return count > 0 ? 100. * std::sqrt(sumRelSq / count) : -1.;
}

double Fitter::linearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1, bool effective)
{
    size_t n = data.size();
    std::vector<double> A(2 * n), y(n);
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    for (size_t i = 0; i < n; i++) {
        A[i] = 1.;
        A[i + n] = eval(data[i].m, data[i].n, data[i].k);
        y[i] = data[i].*fit;
    }

    auto result = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, 2, 1, A.data(), n, y.data(), n);
    if (result != 0)
        throw Error("linear regression failed. Collected performance data may be invalid.");

    c0 = y[0];
    c1 = y[1];

    return relativeRMSE(data, [&](const DataPoint &dp) { return c0 + c1 * eval(dp.m, dp.n, dp.k); }, effective, eval);
}

double Fitter::clampedLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1,
                                       double low, double high, bool effective)
{
    size_t n = data.size();
    std::vector<double> x(n), y(n);
    std::vector<double> A(2 * n), b(n);
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double bestC0 = 0., bestC1 = 0.;
    double bestMSE = std::numeric_limits<double>::infinity();

    auto consider = [&](double tc0, double tc1) {
        double mse = 0.;
        for (size_t i = 0; i < n; i++) {
            double ypred = std::max(low, std::min(high, tc0 + tc1 * x[i]));
            mse += (y[i] - ypred) * (y[i] - ypred);
        }
        if (mse < bestMSE) { bestC0 = tc0; bestC1 = tc1; bestMSE = mse; }
    };

    for (size_t i = 0; i < n; i++) {
        x[i] = eval(data[i].m, data[i].n, data[i].k);
        y[i] = data[i].*fit;
    }

    for (size_t i0 = 0; i0 < n - 1; i0++) {
        for (size_t i1 = i0 + 1; i1 < n; i1++) {
            if (x[i0] == x[i1]) continue;

            double tc1 = (y[i0] - y[i1]) / (x[i0] - x[i1]);
            double tc0 = y[i0] - (tc1 * x[i0]);
            // Check case where best linear segment goes through two data points.
            consider(tc0, tc1);

            int nn = i1 - i0 + 1;
            if (nn > 2) {
                for (int ii = 0; ii < nn; ii++) {
                    A[ii] = 1.;
                    A[ii + nn] = x[i0 + ii];
                    b[ii] = y[i0 + ii];
                }
                auto result = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', nn, 2, 1, A.data(), nn, b.data(), nn);
                if (result != 0)
                    throw Error("linear regression failed. Collected performance data may be invalid.");
                // Check case where best linear segment is inside (x(i0), x(i1)).
                consider(b[0], b[1]);
            }
            // To be rigorous, should also check cases where one endpoint of the linear segment
            //  goes through a data point and the other does not. Probably this is good enough though.
        }
    }

    if (!std::isfinite(bestMSE))
        throw Error("clamped linear regression failed. Not enough data was collected.");
    c0 = bestC0;
    c1 = bestC1;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return std::max(low, std::min(high, c0 + c1 * eval(dp.m, dp.n, dp.k)));
    }, effective, eval);
}

double Fitter::scaledMinimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c1, bool effective)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    bool ok = false;
    c1 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val != 0.) {
            c1 = std::min(c1, data[i].*fit / val);
            ok = true;
        }
    }
    if (!ok)
        throw Error("max-fit failed. Collected performance data may be invalid.");

    return relativeRMSE(data, [&](const DataPoint &dp) { return c1 * eval(dp.m, dp.n, dp.k); }, effective, eval);
}

// Fits a constant to the data points where eval(...) attains its minimum value.
double Fitter::minimumFit(const std::vector<DataPoint> &data, const Evaluator &eval, double &c, bool effective)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double minVal = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; i++)
        minVal = std::min(minVal, eval(data[i].m, data[i].n, data[i].k));

    double sum = 0.;
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        if (eval(data[i].m, data[i].n, data[i].k) == minVal) {
            sum += data[i].*fit;
            count++;
        }
    }

    c = (count > 0) ? (sum / count) : -std::numeric_limits<double>::infinity();

    return relativeRMSE(data, [&](const DataPoint &) { return c; }, effective, eval);
}

double Fitter::mixCoeffFit(const std::vector<DataPoint> &data, const Evaluator &eval0, const Evaluator &eval1, double &c, bool effective)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    int samples = 0;
    double sum = 0.;
    for (size_t i = 0; i < n; i++) {
        auto val0 = eval0(data[i].m, data[i].n, data[i].k);
        auto val1 = eval1(data[i].m, data[i].n, data[i].k);
        if (val1 != val0) {
            auto mix = (data[i].*fit - val0) / (val1 - val0);
            sum += std::max(0., std::min(1., mix));
            samples++;
        }
    }

    c = (samples > 0) ? (sum / samples) : 0.;

    return relativeRMSE(data, [&](const DataPoint &dp) {
        return (1. - c) * eval0(dp.m, dp.n, dp.k) + c * eval1(dp.m, dp.n, dp.k);
    }, effective);
}

double Fitter::fixedMinLinearRegression(const std::vector<DataPoint> &data, const Evaluator &eval, double &c0, double &c1, bool effective)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    double minVal = std::numeric_limits<double>::infinity();
    double minY = 0.;
    int minCount = 0;
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val < minVal) {
            minVal = val;
            minY = data[i].*fit;
            minCount = 1;
        } else if (val == minVal) {
            minY += data[i].*fit;
            minCount++;
        }
    }

    c0 = minY / minCount;
    double c1Total = 0.;
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        auto val = eval(data[i].m, data[i].n, data[i].k);
        if (val == minVal) continue;
        c1Total += (data[i].*fit - c0) / (val - minVal);
        count++;
    }

    if (count == 0)
        throw Error("linear fit failed. Collected performance data may be invalid.");

    c1 = c1Total / count;
    c0 -= c1 * minVal;

    return relativeRMSE(data, [&](const DataPoint &dp) { return c0 + c1 * eval(dp.m, dp.n, dp.k); }, effective, eval);
}

double percentile(const std::vector<DataPoint> &data, double frac, bool effective = false)
{
    size_t n = data.size();
    auto fit = effective ? &DataPoint::effective : &DataPoint::nominal;

    std::vector<double> all;
    all.reserve(n);
    for (size_t i = 0; i < n; i++)
        all.push_back(data[i].*fit);
    std::sort(all.begin(), all.end());

    auto idx = std::min(size_t(std::round(n * frac)), n - 1);
    return all[idx];
}

/**********************/
/* Kernel Preparation */
/**********************/

bool mustUseBeta1(const std::string &strategy)
{
    return (contains(strategy, "kb") || contains(strategy, "kv")) && !contains(strategy, "fb");
}

std::string frontendInputString(const kcatalog::Entry &e, const std::string &modifiers,
                                const std::string &alpha, const std::string &beta, const std::string &additions)
{
    std::string result;
    if (!modifiers.empty()) result = modifiers + ' ';
    result += e.selector.kernelType;
    result += ' ';
    for (int i = 0; i < 3; i++)
        result += e.selector.precisions[i];
    result += ' ';
    for (int i = 0; i < 3; i++) {
        result += e.selector.layouts[i];
        if (e.restrictions.alignment[i] > 1) {
            result += '@';
            result += std::to_string(e.restrictions.alignment[i]);
        }
    }
    result += ' ';
    result += std::to_string(e.driverInfo.unroll[gemmstone::LoopM]);
    result += ' ';
    result += std::to_string(e.driverInfo.unroll[gemmstone::LoopN]);
    result += ' ';
    result += alpha;
    result += ' ';
    result += beta;

    auto appendStrategy = [&](const std::string &s) {
        std::istringstream ss(s);
        std::string word;
        while (ss >> word) { result += ' '; result += word; }
    };
    appendStrategy(e.strategy);
    appendStrategy(additions);

    return result;
}

frontend::Result Fitter::prepareKernel(const kcatalog::Entry &e, bool &batchActive,
                                       const std::string &alpha, const std::string &beta,
                                       const std::string &additions) const
{
    auto input = frontendInputString(e, modifiers, alpha, beta, additions);
    auto generator = frontend::Generator::Create();
    auto res = generator->generate(input);

    if (!res.success)
        throw Error("frontend: " + res.info + " (kernel: " + input + ")");

    batchActive = KernelInfo(res.info).l3.at(0).batch;
    return res;
}

std::string mock(const std::string &s)
{
    return "-- mock " + s;
}

void extractStrategyParameters(kcatalog::Entry &e, const std::string &modifiers, const ngen::Product &product)
{
    bool beta1 = mustUseBeta1(e.strategy);
    auto input = frontendInputString(e, modifiers, "1", beta1 ? "1" : "0", "");

    auto generator = frontend::Generator::Create(product);
    auto res = generator->generate(input, true);
    if (!res.success)
        throw Error("frontend: " + res.info + " (kernel: " + input + ")");

    std::stringstream params(res.info);
    e.driverInfo = gemmstone::CommonDriverInfo::undump(params);
    if (contains(e.strategy, "fp") || contains(e.strategy, "kv"))
        e.driverInfo.flags |= gemmstone::FlagFusedPostOps;
}

kcatalog::Entry parseDescriptor(const std::string &desc, const ngen::Product &product, std::string &modifiers)
{
    auto words = extractWords(desc);
    size_t idx = 0;
    auto next = [&]() -> const std::string & {
        if (idx >= words.size())
            throw Error("unexpected or ill-formed kernel descriptor: \"" + desc + "\"");
        return words[idx++];
    };

    kcatalog::Entry entry = {};
    entry.restrictions = kcatalog::defaultRestrictions();
    entry.restrictions.tags = catalog::intern(entry.restrictions.tags);
    entry.selector.hw = catalog::tag(ngen::getCore(product.family));

    auto kernelType = utils::readKernelType(next);
    modifiers = std::move(kernelType.modifiers);
    entry.selector.kernelType = catalog::intern(kernelType.base);
    entry.restrictions.tags = catalog::intern(std::string(entry.restrictions.tags) + catalog::modifierTags(modifiers));

    {
        std::string precisions[3];
        const auto &precStr = next();
        int pidx = -1;
        bool inBracket = false;
        for (char c: precStr) {
            if (!inBracket) pidx++;
            if (pidx < 0 || pidx >= 3)
                throw Error("could not parse precisions in descriptor: \"" + desc + "\"");
            if (c == '[' || c == ']') {
                bool newInBracket = (c == '[');
                if (inBracket == newInBracket)
                    throw Error("could not parse precisions in descriptor: \"" + desc + "\"");
                inBracket = newInBracket;
            }
            precisions[pidx] += c;
        }
        for (int i = 0; i < 3; i++)
            entry.selector.precisions[i] = catalog::intern(precisions[i]);
    }

    {
        std::string layouts[3];
        const auto &layoutStr = next();
        int lidx = -1;
        const char *c = layoutStr.c_str();
        for (; *c; c++) {
            if (std::isalpha(static_cast<unsigned char>(*c))) lidx++;
            if (lidx < 0 || lidx >= 3)
                throw Error("could not parse layouts in descriptor: \"" + desc + "\"");
            if (*c == '@') {
                char *end;
                entry.restrictions.alignment[lidx] = strtol(c + 1, &end, 10);
                c = end - 1;
                continue;
            } else
                entry.restrictions.alignment[lidx] = 1;
            layouts[lidx] += *c;
        }
        for (int i = 0; i < 3; i++)
            entry.selector.layouts[i] = catalog::intern(layouts[i]);
    }

    entry.driverInfo.unroll[gemmstone::LoopM] = std::atoi(next().c_str());
    entry.driverInfo.unroll[gemmstone::LoopN] = std::atoi(next().c_str());

    if (idx < words.size() && !std::isalpha(static_cast<unsigned char>(words[idx][0]))) {
        idx++;
        if (idx < words.size() && !std::isalpha(static_cast<unsigned char>(words[idx][0])))
            idx++;
    }

    std::string strategy;
    while (idx < words.size()) {
        if (!strategy.empty()) strategy += ' ';
        strategy += words[idx++];
    }
    entry.strategy = catalog::intern(strategy);

    extractStrategyParameters(entry, modifiers, product);
    return entry;
}

int64_t mnPerformanceSize(char)
{
    return simulation() ? 128 : 4096;
}

void findSquarishSize(int64_t &outM, int64_t &outN, int64_t wgTileM, int64_t wgTileN, int wgPerWave)
{
    outM = -1; outN = -1;
    for (int nwgM = 1; nwgM <= wgPerWave; nwgM++) {
        if (wgPerWave % nwgM) continue;
        int64_t candidateM = nwgM * wgTileM;
        int64_t candidateN = (wgPerWave / nwgM) * wgTileN;
        if (outM == -1 || (std::abs(candidateM - candidateN) < std::abs(outM - outN))) {
            outM = candidateM;
            outN = candidateN;
        }
    }
}

/**********************/
/* Model 'E' Fitting  */
/**********************/

#define PARAM(p) e.model.params[kcatalog::ParamE_ ## p]

void Fitter::fitModelE(kcatalog::Entry &e, benchstone::Driver &driver)
{
    using namespace gemmstone;

    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    int unrollK = e.driverInfo.unroll[LoopK];
    int wgTileM = e.driverInfo.wgTile(LoopM);
    int wgTileN = e.driverInfo.wgTile(LoopN);
    int wgM = e.driverInfo.wg[LoopM];
    int wgN = e.driverInfo.wg[LoopN];
    int wgK = e.driverInfo.wg[LoopK];
    if (e.driverInfo.kParallelVariable())
        wgK = 1;
    int threadsPerWG = wgM * wgN * wgK * e.driverInfo.wgExpand;

    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);

    int eus = options.euCount > 0 ? options.euCount : driver.info.EUs;
    double freq = options.nominalFrequency > 0. ? options.nominalFrequency : driver.info.ClockFrequency;
    if (eus <= 0)
        throw Error("could not detect GPU EU count");
    if (freq <= 0.)
        throw Error("could not detect GPU frequency");
    size_t cacheSize = llcSize(eus, options.llcSize > 0 ? options.llcSize : driver.info.LLCSize,
                               driver.HardwareProduct);
    bool sim = simulation();

    bool largeL3 = (e.selector.hw == kcatalog::HWTagXeHPC);

    EvaluateParams p = {{1, 65536, 65536, 65536}, 1., 0., eus};
    p.product = driver.HardwareProduct;
    auto dp = getDerivedParams(e, p);

    auto threadCount = [&](int64_t m, int64_t n, int64_t) {
        return divUp(m, wgTileM) * divUp(n, wgTileN) * threadsPerWG;
    };
    auto matrixAElems = [](int64_t m, int64_t, int64_t k) { return static_cast<double>(m * k); };
    auto matrixBElems = [](int64_t, int64_t n, int64_t k) { return static_cast<double>(n * k); };
    auto fmaCount     = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * n * k); };
    auto partialWaveCount = [&](int64_t m, int64_t n, int64_t k) {
        return static_cast<double>(divUp(threadCount(m, n, k), dp.hwThreadsPartialWave));
    };

    int wgPerWave = dp.hwThreadCapacity / threadsPerWG;

    size_t memMin, memMax;
    bool gemvM = (unrollM == 1 && unrollN > 1);
    bool gemvN = (unrollM > 1 && unrollN == 1);
    if (largeL3) {
        memMin = 1048576;
        memMax = cacheSize / 3;
    } else {
        memMin = cacheSize * 2;
        memMax = cacheSize * 8;
    }

    // SYMM/HEMM: fit the corresponding GEMM kernel instead.
    auto e_bm = e;
    if (contains(e.selector.kernelType, "symm") || contains(e.selector.kernelType, "hemm"))
        e_bm.selector.kernelType = "gemm";

    // Global k-parallel kernels: remove global k-parallelization for most testing.
    bool globalKParallel = e.driverInfo.kParallel() || e.driverInfo.kParallelVariable();
    if (globalKParallel) {
        std::string strategy = e_bm.strategy;
        remove(strategy, " kb");
        remove(strategy, " kv");
        if (e.driverInfo.kParallelVariable())
            remove(strategy, " kr");
        e_bm.strategy = catalog::intern(strategy);
    }

    std::vector<DataPoint> k0Sizes;
    double wstep  = sim ? 4 : 1./4;
    double wlimit = sim ? 8 : 32;
    for (double nwave = wstep; nwave <= wlimit; nwave += wstep) {
        DataPoint pt;
        pt.m = wgTileM;
        pt.n = wgTileN * std::round((nwave * dp.hwThreadCapacity) / threadsPerWG);
        pt.k = 0;
        if (pt.n <= 0) continue;
        if (largeL3) {
            auto cBytes = pt.m * pt.n * cElemBytes;
            if (cBytes > (cacheSize / 2)) continue;
        }
        k0Sizes.push_back(pt);
    }

    if (k0Sizes.empty())
        throw Error("could not generate any k=0 benchmark sizes for fitting");

    if (k0Sizes.back().m * k0Sizes.back().n * cElemBytes <= memMin) {
        for (double scale = 1; scale <= 2; scale += 1./8) {
            DataPoint pt;
            pt.m = wgTileM;
            pt.n = alignUp(int64_t(std::ceil((scale * memMin) / (pt.m * cElemBytes))), wgTileN);
            pt.k = 0;
            k0Sizes.push_back(pt);
            if (sim) break;
        }
    }

    std::vector<DataPoint> afbSizes;
    auto mnStep = std::min(wgTileM, wgTileN);
    for (int64_t mn = mnStep; mn < 4096; mn += mnStep) {
        DataPoint pt;
        pt.m = divUp(mn, wgTileM);
        pt.n = divUp(mn, wgTileN);
        if (pt.m * pt.n >= wgPerWave) break;
        pt.m *= wgTileM;
        pt.n *= wgTileN;
        pt.k = 128;
        if (sim)
            afbSizes.push_back(pt);
        else for (int64_t k: {64, 96, 128, 192}) {
            pt.k = k;
            afbSizes.push_back(pt);
        }
    }
    if (sim) {
        std::vector<DataPoint> selected;
        auto n = int(afbSizes.size());
        auto stride = std::max(1, n / 4);
        for (int i = 0; i < 4 && i * stride < n; i++)
            selected.push_back(afbSizes[i * stride]);
        std::swap(selected, afbSizes);
    }

    std::vector<DataPoint> maSizes, mbSizes;
    auto memMinAB = memMin / std::max(aElemBytes, bElemBytes);
    auto memMaxAB = memMax / std::max(aElemBytes, bElemBytes);
    for (int64_t k0: {256, 1024}) {
        DataPoint pt;
        pt.k = k0 * wgK;
        for (double mn = memMinAB / pt.k; mn <= memMaxAB / pt.k; mn *= 1.25) {
            int mnInt = std::max(mn, 1.);
            pt.m = mnInt; pt.n = 1; maSizes.push_back(pt);
            pt.m = 1; pt.n = mnInt; mbSizes.push_back(pt);
        }
    }

    auto wgCapacity = dp.hwThreadCapacity / threadsPerWG;
    auto gemvUnrollM = wgTileM * wgCapacity;
    auto gemvUnrollN = wgTileN * wgCapacity;
    {
        DataPoint pt;
        pt.m = gemvUnrollM; pt.n = 1;
        for (double k = memMinAB / gemvUnrollM; k <= memMaxAB / gemvUnrollM; k *= 2) {
            pt.k = std::max(k, 1.);
            maSizes.push_back(pt);
        }
        pt.n = gemvUnrollN; pt.m = 1;
        for (double k = memMinAB / gemvUnrollN; k <= memMaxAB / gemvUnrollN; k *= 2) {
            pt.k = std::max(k, 1.);
            mbSizes.push_back(pt);
        }
    }
    if (sim) { maSizes.resize(1); mbSizes.resize(1); }

    std::vector<DataPoint> efSizes;
    int64_t bestM, bestN;
    findSquarishSize(bestM, bestN, wgTileM, wgTileN, wgPerWave);
    int64_t perThreadK = sim ? 1024 : 2048;
    int64_t tileK = alignUp(perThreadK, unrollK) * wgK;
    int64_t minTileMN = mnPerformanceSize(e.selector.precisions[0][0]) / std::sqrt(double(wgK));
    int64_t tileM = std::max(2 * bestM, alignUp(minTileMN, bestM));
    int64_t tileN = std::max(2 * bestN, alignUp(minTileMN, bestN));

    for (int expandM: {1, 2}) for (int expandN: {1, 2}) for (int expandK: {1, 2}) {
        DataPoint pt;
        pt.m = tileM * expandM; pt.n = tileN * expandN; pt.k = tileK * expandK;
        efSizes.push_back(pt);
    }
    if (sim) efSizes.resize(1);

    std::vector<DataPoint> epSizes;
    if (dp.hwThreadsPartialWave < dp.hwThreadCapacity) {
        for (int pw = 1; pw * dp.hwThreadsPartialWave <= dp.hwThreadCapacity; pw++) {
            DataPoint pt;
            findSquarishSize(pt.m, pt.n, wgTileM, wgTileN, (dp.hwThreadsPartialWave * pw) / threadsPerWG);
            pt.k = tileK;
            epSizes.push_back(pt);
        }
    }
    if (sim && epSizes.size() > 4) {
        std::vector<DataPoint> epSizes1;
        for (int i = 1; i < 4; i++) {
            auto epIdx = size_t(std::round(epSizes.size() * i * 0.25) - 1);
            epSizes1.push_back(epSizes[epIdx]);
        }
        std::swap(epSizes, epSizes1);
    }

    std::vector<DataPoint> emSizes;
    for (int expandM: {1, 2}) for (int expandN: {1, 2}) for (bool adjustM: {true, false}) {
        DataPoint pt;
        pt.m = bestM * expandM; pt.n = bestN * expandN; pt.k = tileK;
        if (adjustM) pt.m += wgTileM; else pt.n += wgTileN;
        emSizes.push_back(pt);
    }
    if (sim) emSizes.resize(1);

    std::vector<DataPoint> frSizes;
    for (double k = 64.; k < 2 * tileK; k *= 1.1) {
        DataPoint pt;
        pt.m = 2 * tileM; pt.n = 2 * tileN; pt.k = alignUp(int64_t(k), unrollK);
        frSizes.push_back(pt);
    }

    bool batchActive = false;

    double c0, c1;
    auto res = prepareKernel(e_bm, batchActive, "1", "1", mock("c kkr"));
    collectData(driver, res, k0Sizes, batchActive, true);
    double q = fixedMinLinearRegression(k0Sizes, partialWaveCount, c0, c1);
    verboseFit("C0, C1", "fixedMinLinearRegression", c0, c1, q);
    PARAM(C0) = c0;
    PARAM(C1) = c1;

    auto subtractConstantCosts = [&](std::vector<DataPoint> &data) {
        for (auto &pt: data) {
            auto overhead = std::max(0., c0 + partialWaveCount(pt.m, pt.n, pt.k) * c1);
            pt.nominal -= overhead;
            pt.effective -= overhead;
        }
    };

    auto matrixCElemsAboveMin = [=](int64_t m, int64_t n, int64_t) {
        return (m * n * cElemBytes >= memMin) ? static_cast<double>(m * n) : 0.;
    };

    // Now time C updates: first load + store, using beta = 1 and atomics disabled.
    // Use effective efficiency to help stabilize timing.
    // Only consider data points above memMin bytes, for greater stability.
    res = prepareKernel(e_bm, batchActive, "1", "1", "nau");
    collectData(driver, res, k0Sizes, batchActive, true);
    subtractConstantCosts(k0Sizes);
    q = scaledMinimumFit(k0Sizes, matrixCElemsAboveMin, PARAM(Mcu));
    verboseFit("Mcu", "scaledMinimumFit", PARAM(Mcu), q);

    // Same for beta = 0 kernel, to time only C store.
    res = prepareKernel(e_bm, batchActive, "1", "0");
    collectData(driver, res, k0Sizes, batchActive, true);
    subtractConstantCosts(k0Sizes);
    q = scaledMinimumFit(k0Sizes, matrixCElemsAboveMin, PARAM(Mc));
    verboseFit("Mc", "scaledMinimumFit", PARAM(Mc), q);

    // Global k-parallel kernels with fused beta: test beta = 0 again
    //  with fused beta scaling.
    PARAM(Cb0) = PARAM(Cb1) = 0;
    if (e.driverInfo.fusedBeta() && !e.driverInfo.altFusedBeta()) {
        res = prepareKernel(e_bm, batchActive, "1", "0", "kb fb " + mock("c kkr"));
        collectData(driver, res, k0Sizes, batchActive, true);
        q = fixedMinLinearRegression(k0Sizes, partialWaveCount, PARAM(Cb0), PARAM(Cb1));
        verboseFit("Cb0, Cb1", "fixedMinLinearRegression", PARAM(Cb0), PARAM(Cb1), q);
    }

    // Alternate fused beta: measure difference of beta = 0 vs. beta = 1,
    //   and take the 90th percentile.
    if (e.driverInfo.altFusedBeta()) {
        auto afbSizes1 = afbSizes;
        res = prepareKernel(e, batchActive, "1", "0");
        collectData(driver, res, afbSizes, batchActive, true);
        res = prepareKernel(e, batchActive, "1", "1");
        collectData(driver, res, afbSizes1, batchActive, true);
        for (size_t i = 0; i < afbSizes.size(); i++) {
            afbSizes1[i].nominal   = afbSizes[i].nominal   - afbSizes1[i].nominal;
            afbSizes1[i].effective = afbSizes[i].effective - afbSizes1[i].effective;
        }
        PARAM(Cb0) = std::max(0., percentile(afbSizes1, 0.9));

        if (e.driverInfo.fusedPostOps()) {
            auto afpSizes = afbSizes;
            res = prepareKernel(e, batchActive, "1", "0", "ffp");
            collectData(driver, res, afpSizes, batchActive, true);
            for (size_t i = 0; i < afbSizes.size(); i++) {
                afpSizes[i].nominal   -= afbSizes[i].nominal;
                afpSizes[i].effective -= afbSizes[i].effective;
            }
            PARAM(Cp0) = std::max(0., percentile(afpSizes, 0.9));
        }
    }

    // For kernels with local k reduction, re-measure constant overheads without local k reduction
    //   ... for variable k-sliced kernels, measure _with_ local k reduction (it was disabled).
    PARAM(Ck0) = PARAM(Ck1) = 0.;
    if (contains(e.strategy, "kr")) {
        bool withKR = e.driverInfo.kParallelVariable();
        res = prepareKernel(e_bm, batchActive, "1", "1", "kr " + mock(withKR ? "c kkr" : "c"));
        collectData(driver, res, k0Sizes, batchActive, true);
        q = fixedMinLinearRegression(k0Sizes, partialWaveCount, c0, c1);
        verboseFit("C0, C1 [kr]", "fixedMinLinearRegression", c0, c1, q);

        int maxWGK = e_bm.driverInfo.wg[LoopK];
        if (maxWGK > 1) {
            if (withKR) {
                std::swap(PARAM(C0), c0);
                std::swap(PARAM(C1), c1);
            }
            PARAM(Ck0) = (PARAM(C0) - c0) / (maxWGK - 1);
            PARAM(Ck1) = (PARAM(C1) - c1) / (maxWGK - 1);
            PARAM(C0) = c0 - PARAM(Ck0);
            PARAM(C1) = c1 - PARAM(Ck1);
        }
    }

    // Estimate memory loading costs for A/B separately. Skip for packed kernels.
    // Remove computation for this part.
    bool doMemA = !isPackedChar(e.selector.layouts[0]);
    bool doMemB = !isPackedChar(e.selector.layouts[1]);

    if (doMemA || doMemB)
        res = prepareKernel(e_bm, batchActive, "1", "0", mock("c m"));

    if (doMemA) {
        collectData(driver, res, maSizes, batchActive, true);
        subtractConstantCosts(maSizes);
        q = scaledMinimumFit(maSizes, matrixAElems, PARAM(Ma));
        PARAM(Ma) = std::max(PARAM(Ma), 0.);
        verboseFit("Ma", "scaledMinimumFit", PARAM(Ma), q);
    } else
        PARAM(Ma) = 0.;

    if (doMemB) {
        collectData(driver, res, mbSizes, batchActive, true);
        subtractConstantCosts(mbSizes);
        q = scaledMinimumFit(mbSizes, matrixBElems, PARAM(Mb));
        PARAM(Mb) = std::max(PARAM(Mb), 0.);
        verboseFit("Mb", "scaledMinimumFit", PARAM(Mb), q);
    } else
        PARAM(Mb) = 0.;

    // Compute parameters. Use beta = 1 so benchstone can enable k blocking.
    res = prepareKernel(e_bm, batchActive, "1", "1", mock("c"));
    collectData(driver, res, efSizes, batchActive, false, true);
    subtractConstantCosts(efSizes);
    q = scaledMinimumFit(efSizes, fmaCount, PARAM(Ef), true);
    verboseFit("Ef", "scaledMinimumFit", PARAM(Ef), q);

    double maxFreqDrop = 1.0;
    for (const auto &pt: efSizes) {
        auto drop = pt.nominal / pt.effective;
        if (drop > 4.) continue;
        maxFreqDrop = std::max(maxFreqDrop, drop);
    }
    PARAM(Fp) = maxFreqDrop;
    verboseFit("Fp", "max(nom/eff)", PARAM(Fp));

    if (gemvM || gemvN || sim)
        PARAM(Fr0) = PARAM(Fr1) = 0;
    else {
        collectData(driver, res, frSizes, batchActive, false, true);
        for (auto &pt: frSizes)
            pt.nominal /= pt.effective;
        q = clampedLinearRegression(frSizes, fmaCount, PARAM(Fr0), PARAM(Fr1), 1, maxFreqDrop, false);
        verboseFit("Fr0, Fr1", "clampedLinearRegression", PARAM(Fr0), PARAM(Fr1), q);
    }

    if (epSizes.size() > 1) {
        collectData(driver, res, epSizes, batchActive, false, true);
        subtractConstantCosts(epSizes);
        for (auto &pt: epSizes)
            pt.effective /= fmaCount(pt.m, pt.n, pt.k);
        auto partialWaveQuotient = [&](int64_t m, int64_t n, int64_t k) {
            return double(dp.partialWaveCount) / partialWaveCount(m, n, k);
        };
        q = linearRegression(epSizes, partialWaveQuotient, PARAM(Ep0), PARAM(Ep1), true);
        PARAM(Ep1) = std::max(0., PARAM(Ep1));
        verboseFit("Ep0, Ep1", "linearRegression", PARAM(Ep0), PARAM(Ep1), q);
    } else {
        PARAM(Ep0) = PARAM(Ef);
        PARAM(Ep1) = 0.;
    }

    auto tPredictNoLB = [=](int64_t m, int64_t n, int64_t k) { return PARAM(Ef) * fmaCount(m, n, k); };
    auto tPredictLB = [=](int64_t m, int64_t n, int64_t k) {
        int64_t threads = threadCount(m, n, k);
        int64_t kthread = divUp(k, wgK);
        int partialWaveThreads = threads % dp.hwThreadCapacity;
        int partialWaves = divUp(partialWaveThreads, dp.hwThreadsPartialWave);
        double Ef = PARAM(Ef);
        double Ep = std::max(Ef, PARAM(Ep0) + (PARAM(Ep1) * dp.partialWaveCount) / partialWaves);
        double Tf = PARAM(Ef) * (threads - partialWaveThreads);
        double Tp = Ep * partialWaves * dp.hwThreadsPartialWave;
        return (Tf + Tp) * unrollM * unrollN * kthread;
    };

    collectData(driver, res, emSizes, batchActive, false, true);
    subtractConstantCosts(emSizes);
    q = mixCoeffFit(emSizes, tPredictNoLB, tPredictLB, PARAM(Em), true);
    verboseFit("Em", "mixCoeffFit", PARAM(Em), q);

    for (int i = 0; i < kcatalog::ParamE_Em; i++)
        e.model.params[i] *= freq * eus;
    e.model.paramCount = kcatalog::ParamECount;
}

#undef PARAM

/**********************/
/* Model 'S' Fitting  */
/**********************/
// Model 'S': predecessor of model 'E', with simpler load-balancing terms.

#define PARAM(p) e.model.params[kcatalog::ParamS_ ## p]

void Fitter::fitModelS(kcatalog::Entry &e, benchstone::Driver &driver)
{
    using namespace gemmstone;

    int unrollM = e.driverInfo.unroll[LoopM];
    int unrollN = e.driverInfo.unroll[LoopN];
    int unrollK = e.driverInfo.unroll[LoopK];
    int wgTileM = e.driverInfo.wgTile(LoopM);
    int wgTileN = e.driverInfo.wgTile(LoopN);
    int wgK = e.driverInfo.wg[LoopK];
    int threadsPerWG = e.driverInfo.threadsPerWG();

    auto aElemBytes = extPrecisionBytes(e.selector.precisions[0], false);
    auto bElemBytes = extPrecisionBytes(e.selector.precisions[1], false);
    auto cElemBytes = extPrecisionBytes(e.selector.precisions[2], true);

    int eus = options.euCount > 0 ? options.euCount : driver.info.EUs;
    double freq = options.nominalFrequency > 0. ? options.nominalFrequency : driver.info.ClockFrequency;
    if (eus <= 0)
        throw Error("could not detect GPU EU count");
    if (freq <= 0.)
        throw Error("could not detect GPU frequency");
    size_t cacheSize = llcSize(eus, options.llcSize > 0 ? options.llcSize : driver.info.LLCSize,
                               driver.HardwareProduct);
    bool largeL3 = (e.selector.hw == kcatalog::HWTagXeHPC);

    EvaluateParams p = {{1, 65536, 65536, 65536}, 1., 0., eus};
    p.product = driver.HardwareProduct;
    auto dp = getDerivedParams(e, p);

    auto threadCount = [&](int64_t m, int64_t n, int64_t) {
        return divUp(m, wgTileM) * divUp(n, wgTileN) * threadsPerWG;
    };
    auto matrixAElems = [](int64_t m, int64_t, int64_t k) { return static_cast<double>(m * k); };
    auto matrixBElems = [](int64_t, int64_t n, int64_t k) { return static_cast<double>(n * k); };
    auto fmaCount     = [](int64_t m, int64_t n, int64_t k) { return static_cast<double>(m * n * k); };
    auto partialWaveCount    = [&](int64_t m, int64_t n, int64_t k) { return static_cast<double>(divUp(threadCount(m, n, k), dp.hwThreadsPartialWave)); };
    auto partialWaveQuotient = [&](int64_t m, int64_t n, int64_t k) {
        return double(dp.partialWaveCount) / partialWaveCount(m, n, k);
    };

    // SYMM/HEMM: fit the corresponding GEMM kernel instead.
    auto e_bm = e;
    if (contains(e.selector.kernelType, "symm") || contains(e.selector.kernelType, "hemm"))
        e_bm.selector.kernelType = "gemm";

    // Test cases:
    //  - Constant costs: k = 0 cases (fitting in L3 for PVC)
    std::vector<DataPoint> k0Sizes;
    for (double nwave = 1./4; nwave <= 32; nwave += 1./4) {
        DataPoint pt;
        pt.m = wgTileM;
        pt.n = wgTileN * std::round((nwave * dp.hwThreadCapacity) / threadsPerWG);
        pt.k = 0;

        if (pt.n <= 0) continue;
        if (largeL3) {
            auto cBytes = pt.m * pt.n * cElemBytes;
            if (cBytes > (cacheSize / 2))
                continue;
        }

        k0Sizes.push_back(pt);
    }

    //  - Memory costs: gemv cases fitting in L3 for PVC, outside L3 for others.
    std::vector<DataPoint> maSizes, mbSizes;
    size_t memMin, memMax;
    bool gemvM = (unrollM == 1 && unrollN > 1);
    bool gemvN = (unrollM > 1 && unrollN == 1);
    if (largeL3) {
        memMin = 1048576;
        memMax = cacheSize / 3;
    } else {
        memMin = cacheSize * 2;
        memMax = cacheSize * 8;
    }
    memMin /= std::max(aElemBytes, bElemBytes);
    memMax /= std::max(aElemBytes, bElemBytes);

    for (int64_t k0: {256, 1024}) {
        DataPoint pt;
        pt.k = k0 * wgK;
        for (double mn = memMin / pt.k; mn <= memMax / pt.k; mn *= 1.25) {
            int mnInt = std::max(mn, 1.);
            pt.m = mnInt;
            pt.n = 1;
            maSizes.push_back(pt);
            pt.m = 1;
            pt.n = mnInt;
            mbSizes.push_back(pt);
        }
    }

    auto wgCapacity = dp.hwThreadCapacity / threadsPerWG;
    auto gemvUnrollM = wgTileM * wgCapacity;
    auto gemvUnrollN = wgTileN * wgCapacity;
    /* Add some cases with nice m/n. */
    {
        DataPoint pt;
        pt.m = gemvUnrollM;
        pt.n = 1;
        for (double k = memMin / gemvUnrollM; k <= memMax / gemvUnrollM; k *= 2) {
            pt.k = std::max(k, 1.);
            maSizes.push_back(pt);
        }

        pt.n = gemvUnrollN;
        pt.m = 1;
        for (double k = memMin / gemvUnrollN; k <= memMax / gemvUnrollN; k *= 2) {
            pt.k = std::max(k, 1.);
            mbSizes.push_back(pt);
        }
    }

    //  - Compute costs: peak performance: large near-square C, good k
    std::vector<DataPoint> efSizes;
    int wgPerWave = dp.hwThreadCapacity / threadsPerWG;
    int64_t bestM, bestN;
    findSquarishSize(bestM, bestN, wgTileM, wgTileN, wgPerWave);
    int64_t tileK = alignUp(2048, unrollK) * wgK;
    int64_t minTileMN = mnPerformanceSize(e.selector.precisions[0][0]) / std::sqrt(double(wgK));
    int64_t tileM = std::max(2 * bestM, alignUp(minTileMN, bestM));
    int64_t tileN = std::max(2 * bestN, alignUp(minTileMN, bestN));

    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (int expandK: {1, 2}) {
                DataPoint pt;
                pt.m = tileM * expandM;
                pt.n = tileN * expandN;
                pt.k = tileK * expandK;
                efSizes.push_back(pt);
            }
        }
    }

    //  - Compute costs: partial waves: integral numbers of partial waves, up to 1 full wave
    std::vector<DataPoint> epSizes;
    if (dp.hwThreadsPartialWave < dp.hwThreadCapacity) {
        for (int pw = 1; pw * dp.hwThreadsPartialWave <= dp.hwThreadCapacity; pw++) {
            DataPoint pt;
            findSquarishSize(pt.m, pt.n, wgTileM, wgTileN, (dp.hwThreadsPartialWave * pw) / threadsPerWG);
            pt.k = tileK;
            epSizes.push_back(pt);
        }
    }

    //  - Compute costs: mixing factor: 1-4 full waves + a little extra
    std::vector<DataPoint> emSizes;
    for (int expandM: {1, 2}) {
        for (int expandN: {1, 2}) {
            for (bool adjustM: {true, false}) {
                DataPoint pt;
                pt.m = bestM * expandM;
                pt.n = bestN * expandN;
                pt.k = tileK;
                if (adjustM)
                    pt.m += wgTileM;
                else
                    pt.n += wgTileN;
                emSizes.push_back(pt);
            }
        }
    }

    // Frequency throttling: varying k
    std::vector<DataPoint> frSizes;
    for (double k = 64.; k < 2 * tileK; k *= 1.1) {
        DataPoint pt;
        pt.m = 2 * tileM;
        pt.n = 2 * tileN;
        pt.k = alignUp(int64_t(k), unrollK);
        frSizes.push_back(pt);
    }

    bool batchActive = false;

    // Time beta = 1 kernel with k = 0 to measure constant terms C01, C11.
    // Use effective efficiency to help stabilize timing.
    auto res = prepareKernel(e_bm, batchActive, "1", "1");
    collectData(driver, res, k0Sizes, batchActive, true, true);
    double q = linearRegression(k0Sizes, partialWaveCount, PARAM(C01), PARAM(C11), true);
    verboseFit("C01, C11", "linearRegression", PARAM(C01), PARAM(C11), q);
    q = minimumFit(k0Sizes, partialWaveCount, PARAM(Cm1), true);
    verboseFit("Cm1", "minimumFit", PARAM(Cm1), q);

    // Same for beta = 0 kernel.
    // Use variable beta for kb kernels to enable the non-atomic beta = 0 path
    //   (benchstone will test with beta = 0).
    auto beta0 = e.driverInfo.kParallel() ? "-" : "0";
    res = prepareKernel(e_bm, batchActive, "1", beta0);
    collectData(driver, res, k0Sizes, batchActive, true, true);
    q = linearRegression(k0Sizes, partialWaveCount, PARAM(C00), PARAM(C10), true);
    verboseFit("C00, C10", "linearRegression", PARAM(C00), PARAM(C10), q);
    q = minimumFit(k0Sizes, partialWaveCount, PARAM(Cm0), true);
    verboseFit("Cm0", "minimumFit", PARAM(Cm0), q);

    // Baseline for the remaining fitting: remove C update to reduce variance.
    double c0, c1;
    res = prepareKernel(e_bm, batchActive, "1", beta0, mock("c"));
    collectData(driver, res, k0Sizes, batchActive, true, true);
    q = linearRegression(k0Sizes, partialWaveCount, c0, c1, true);
    verboseFit("c0, c1", "linearRegression", c0, c1, q);

    auto subtractConstantCosts = [&](std::vector<DataPoint> &data) {
        for (auto &pt: data) {
            auto overhead = std::max(0., c0 + partialWaveCount(pt.m, pt.n, pt.k) * c1);
            pt.nominal -= overhead;
            pt.effective -= overhead;
        }
    };

    // Estimate memory loading costs for A/B separately. Skip for packed kernels.
    // Remove computation for this part.
    bool doMemA = !isPackedChar(e.selector.layouts[0]);
    bool doMemB = !isPackedChar(e.selector.layouts[1]);

    if (doMemA || doMemB)
        res = prepareKernel(e_bm, batchActive, "1", beta0, mock("c m"));

    if (doMemA) {
        collectData(driver, res, maSizes, batchActive, true, true);
        subtractConstantCosts(maSizes);
        q = scaledMinimumFit(maSizes, matrixAElems, PARAM(Ma), true);
        PARAM(Ma) = std::max(PARAM(Ma), 0.);
        verboseFit("Ma", "scaledMinimumFit", PARAM(Ma), q);
    } else
        PARAM(Ma) = 0.;

    if (doMemB) {
        collectData(driver, res, mbSizes, batchActive, true, true);
        subtractConstantCosts(mbSizes);
        q = scaledMinimumFit(mbSizes, matrixBElems, PARAM(Mb), true);
        PARAM(Mb) = std::max(PARAM(Mb), 0.);
        verboseFit("Mb", "scaledMinimumFit", PARAM(Mb), q);
    } else
        PARAM(Mb) = 0.;

    e.model.paramCount = kcatalog::ParamSCount;

    // Compute parameters. Use beta = 1 so benchstone can enable k blocking.
    res = prepareKernel(e_bm, batchActive, "1", "1", mock("c"));
    collectData(driver, res, efSizes, batchActive, false, true);
    subtractConstantCosts(efSizes);
    q = scaledMinimumFit(efSizes, fmaCount, PARAM(Ef), true);
    verboseFit("Ef", "scaledMinimumFit", PARAM(Ef), q);

    double maxFreqDrop = 1.0;
    for (const auto &pt: efSizes) {
        auto drop = pt.nominal / pt.effective;
        if (drop > 4.) continue;    /* discard unreasonably low measured frequencies (likely MDAPI bug) */
        maxFreqDrop = std::max(maxFreqDrop, drop);
    }
    PARAM(Fp) = maxFreqDrop;
    verboseFit("Fp", "max(nom/eff)", PARAM(Fp));

    if (gemvM || gemvN)
        PARAM(Fr0) = PARAM(Fr1) = 0;
    else {
        collectData(driver, res, frSizes, batchActive, false, true);
        for (auto &pt: frSizes)
            pt.nominal /= pt.effective;
        q = clampedLinearRegression(frSizes, fmaCount, PARAM(Fr0), PARAM(Fr1), 1, maxFreqDrop, false);
        verboseFit("Fr0, Fr1", "clampedLinearRegression", PARAM(Fr0), PARAM(Fr1), q);
    }

    if (epSizes.size() > 1) {
        collectData(driver, res, epSizes, batchActive, false, true);
        subtractConstantCosts(epSizes);
        for (auto &pt: epSizes)
            pt.effective /= fmaCount(pt.m, pt.n, pt.k);
        q = linearRegression(epSizes, partialWaveQuotient, PARAM(Ep0), PARAM(Ep1), true);
        PARAM(Ep1) = std::max(0., PARAM(Ep1));
        verboseFit("Ep0, Ep1", "linearRegression", PARAM(Ep0), PARAM(Ep1), q);
    } else {
        PARAM(Ep0) = PARAM(Ef);
        PARAM(Ep1) = 0.;
    }

    auto tPredictNoLB = [=](int64_t m, int64_t n, int64_t k) { return PARAM(Ef) * fmaCount(m, n, k); };
    auto tPredictLB   = [=](int64_t m, int64_t n, int64_t k) {
        int64_t threads = threadCount(m, n, k);
        int64_t kthread = divUp(k, wgK);
        int partialWaveThreads = threads % dp.hwThreadCapacity;
        int partialWaves = divUp(partialWaveThreads, dp.hwThreadsPartialWave);
        double Ef = PARAM(Ef);
        double Ep = std::max(Ef, PARAM(Ep0) + (PARAM(Ep1) * dp.partialWaveCount) / partialWaves);
        double Tf = PARAM(Ef) * (threads - partialWaveThreads);
        double Tp = Ep * partialWaves * dp.hwThreadsPartialWave;
        return (Tf + Tp) * unrollM * unrollN * kthread;
    };

    collectData(driver, res, emSizes, batchActive, false, true);
    subtractConstantCosts(emSizes);
    q = mixCoeffFit(emSizes, tPredictNoLB, tPredictLB, PARAM(Em), true);
    verboseFit("Em", "mixCoeffFit", PARAM(Em), q);

    // Normalize data to remove frequency/EU count dependency.
    // Model outputs total EU cycles.
    for (int i = 0; i < kcatalog::ParamS_Em; i++)
        e.model.params[i] *= freq * eus;
}

#undef PARAM

// Model 'W': old kernel selection logic, based on simple heuristic that chooses the
//            smallest unrolls that fit in 1 wave on the GPU, if possible.
void fitModelW(kcatalog::Entry &e, double priorityOverride)
{
    using namespace gemmstone;
    e.model.paramCount = kcatalog::ParamWCount;
    e.model.params[kcatalog::ParamWPriority] = (priorityOverride != 0.)
            ? priorityOverride
            : e.driverInfo.unroll[LoopM] * e.driverInfo.unroll[LoopN];
}

// Benchmarked models are fit on, and for, the driver's device.
benchstone::Driver &benchmarkDriver(benchstone::Driver *driver, const kcatalog::Entry &e)
{
    std::string model = std::string("model ") + e.model.id;
    if (!driver)
        throw Error(model + " fitting requires a benchmark driver");
    if (driver->HardwareProduct.family == ngen::ProductFamily::Unknown)
        throw Error(model + " fitting requires a known GPU");
    if (detectDevice(false).family != driver->HardwareProduct.family)
        throw Error(model + " fitting requires an executable GPU; GEN/FAMILY overrides are not supported");
    if (e.selector.hw != catalog::tag(driver->Hardware))
        throw Error(model + " fitting requires a kernel for the benchmark GPU (kernel HW tag '"
                    + e.selector.hw + "', GPU HW tag '" + catalog::tag(driver->Hardware) + "')");
    return *driver;
}

Result fitEntry(benchstone::Driver *driver, kcatalog::Entry entry, char modelId,
                double priorityOverride, const Options &options, const std::string &modifiers = "")
{
    Result result;
    Fitter fitter(options, result, modifiers);
    result.entry = std::move(entry);
    result.entry.model.id = modelId;

    struct NoPerfBail {
        benchstone::Driver *driver;
        double saved;
        NoPerfBail(benchstone::Driver *driver) : driver(driver), saved(driver ? driver->PerfBailThreshold : 0.) {
            if (driver) driver->PerfBailThreshold = 0.;
        }
        ~NoPerfBail() {
            if (driver) driver->PerfBailThreshold = saved;
        }
    } noPerfBail(driver);

    switch (modelId) {
        case 'E':
            fitter.fitModelE(result.entry, benchmarkDriver(driver, result.entry));
            break;
        case 'S':
            fitter.fitModelS(result.entry, benchmarkDriver(driver, result.entry));
            break;
        case 'W': fitModelW(result.entry, priorityOverride); break;
        default:  throw Error(std::string("fit() does not yet support model '") + modelId + "'");
    }

    catalog::validate(result.entry);
    return result;
}

} // namespace

Result fit(benchstone::Driver *driver, const std::string &gemm_desc, char modelId,
                    const ngen::Product &product, double priorityOverride, const Options &options)
{
    std::string modifiers;
    auto entry = parseDescriptor(gemm_desc, product, modifiers);
    return fitEntry(driver, std::move(entry), modelId, priorityOverride, options, modifiers);
}

Result fit(benchstone::Driver *driver, const kcatalog::Entry &entry, char modelId,
                    double priorityOverride, const Options &options)
{
    return fitEntry(driver, entry, modelId, priorityOverride, options);
}

} // namespace fit
