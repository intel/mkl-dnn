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

#ifndef GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_COMMON_HPP
#define GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <ostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "gemmstone/driver_info.hpp"
#include "gemmstone/type.hpp"
#include "kernel_info.hpp"

#include "tools/types/scale_e8m0.hpp"
#include "tools/types/fp4.hpp"
#include "tools/types/nf4.hpp"
#include "tools/types/bfloat8.hpp"
#include "tools/types/bfloat16.hpp"
#include "tools/types/hfloat8.hpp"
#include "tools/types/fp16.hpp"
#include "tools/types/int4.hpp"
#include "tools/types/int2.hpp"
#include "tools/types/tf32.hpp"
#include "tools/types/complex_helpers.hpp"
#include "tools/hw_queries.hpp"
#include "tools/utils.hpp"

#ifdef BENCHSTONE_USE_MMAP
#include <sys/mman.h>
#endif

#define MKL_Complex8  complex<float>
#define MKL_Complex16 complex<double>


typedef int64_t integer;

template <typename T>
static inline auto div_up(T value, int divisor) {
    return (value + divisor - 1) / divisor;
}

template <typename T>
static inline auto align_down(T value, int factor) {
    return factor * (value / factor);
}

template <typename T>
static inline auto align_up(T value, int factor) {
    return factor * div_up(value, factor);
}

// Type conversions for printing with ostreams.
template <typename T> struct printable { using type = T; };
template <> struct printable<uint8_t> { using type = int; };
template <> struct printable<int8_t>  { using type = int; };

template <typename T> using printable_t = typename printable<T>::type;

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif

using utils::rprintf;
using utils::yprintf;
using utils::gprintf;

template <typename... Args>
std::string qsprintf(const char *format, const Args& ...args)
{
    constexpr size_t nmax = 256;
    std::string result;
    result.resize(nmax);
    result.resize(snprintf(result.data(), nmax, format, args...));
    return result;
}

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

/* Runtime to compile time type templating helpers */
template <typename T> static constexpr inline gemmstone::Type typeEnum = gemmstone::Type::invalid;
#define TYPE_MAP(TT, E) \
    template <> constexpr inline gemmstone::Type typeEnum<TT> = gemmstone::Type::E;

TYPE_MAP(bfloat8, bf8)
TYPE_MAP(hfloat8, hf8)
TYPE_MAP(bfloat16, bf16)
TYPE_MAP(half, f16)
TYPE_MAP(float, f32)
TYPE_MAP(tf32, tf32)
TYPE_MAP(double, f64)
TYPE_MAP(f4_e2m1, f4_e2m1)
TYPE_MAP(nf4, nf4)
TYPE_MAP(uint2, u2)
TYPE_MAP(int2, s2)
TYPE_MAP(uint4, u4)
TYPE_MAP(int4, s4)
TYPE_MAP(uint8_t, u8)
TYPE_MAP(int8_t, s8)
TYPE_MAP(scale_e8m0, f8_e8m0)
TYPE_MAP(int16_t, s16)
TYPE_MAP(uint32_t, u32)
TYPE_MAP(int32_t, s32)

#undef TYPE_MAP

using AllTypes = std::tuple<
    bfloat16, half, float, tf32, double,
    complex<bfloat16>, complex<half>, complex<float>, complex<tf32>, complex<double>,
    bfloat8, hfloat8, f4_e2m1, nf4, scale_e8m0,
    uint2, uint4, uint8_t, uint16_t, uint32_t,
    int2, int4, int8_t, int16_t, int32_t
>;

#define TYPE_TEMPLATIFY_LOOKUP(f)                                               \
template <typename Fptr, std::size_t... _seq>                                   \
static inline auto lookup_##f(gemmstone::Type key, std::index_sequence<_seq...>) {         \
    Fptr f_ptr = nullptr;                                                       \
    (void) ((key == typeEnum<std::tuple_element_t<_seq, AllTypes>>              \
                && (f_ptr = &_##f<std::tuple_element_t<_seq, AllTypes>>, true)) \
            || ... || (throw std::runtime_error("Unknown type"), false));       \
    return f_ptr;                                                               \
}

#define TYPE_TEMPLATIFY_FILTERED(f, filter) TYPE_TEMPLATIFY_LOOKUP(f)   \
template <typename... Args>                                             \
auto f(gemmstone::Type key, Args&& ...args) {                                      \
    using Seq = std::make_index_sequence<std::tuple_size_v<AllTypes>>;  \
    return filter(lookup_##f<decltype(&_##f<int>)>(key, Seq{}),         \
                  std::forward<Args>(args)...);                         \
}

#define TYPE_TEMPLATIFY(f) TYPE_TEMPLATIFY_FILTERED(f, std::invoke)

/* Type helpers. */
template <typename T>
constexpr bool isIntegral()                        { return std::is_integral_v<T>; }
template <> constexpr bool isIntegral<int2>()      { return true; }
template <> constexpr bool isIntegral<uint2>()     { return true; }
template <> constexpr bool isIntegral<int4>()      { return true; }
template <> constexpr bool isIntegral<uint4>()     { return true; }

template <typename T> static constexpr inline T qnan()
{
    if constexpr (!isIntegral<T>())
        return std::numeric_limits<float>::quiet_NaN();
    else
        return ~0;
}

template <typename T> static constexpr inline T inf()
{
    if constexpr (!isIntegral<T>())
        return std::numeric_limits<float>::infinity();
    else
        return std::numeric_limits<T>::max();
}

template <typename T> static constexpr inline T epsilon()
{
    if constexpr (!isIntegral<T>())
        return std::numeric_limits<T>::epsilon();
    else
        return 0;
}

template <typename T> static constexpr inline double _epsilon() { return epsilon<real_t<T>>(); }

TYPE_TEMPLATIFY(epsilon)

static inline double epsilon(char precision) { return epsilon(gemmstone::charToType(precision)); }

static inline int precisionLoss(char precision, char precisionInt, int components)
{
    double eps = epsilon(precision);
    double epsInt = std::pow(epsilon(precisionInt), components);

    if (eps >= epsInt) return 1;
    if (eps == 0) return 1;

    return int(std::round(epsInt / eps));
}

template <typename T> static inline bool identicalBits(T a, T b)
{
    return memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T> static inline T conj(T x)
{
    if constexpr (isComplex<T>())
        return std::conj(x);
    else
        return x;
}

/* Type conversions */
template <typename T> struct sumHelper { using type = T; };
template <> struct sumHelper<int2>     { using type = int32_t; };
template <> struct sumHelper<uint2>    { using type = int32_t; };
template <> struct sumHelper<int4>     { using type = int32_t; };
template <> struct sumHelper<uint4>    { using type = int32_t; };
template <> struct sumHelper<int8_t>   { using type = int32_t; };
template <> struct sumHelper<uint8_t>  { using type = int32_t; };
template <> struct sumHelper<int16_t>  { using type = int32_t; };
template <> struct sumHelper<uint16_t> { using type = int32_t; };

template <typename T> using sum_t = typename sumHelper<T>::type;

static inline gemmstone::Type sumType(gemmstone::Type T) {
    if (T.isInteger() && T.paddedSize() < 4)
        return gemmstone::Type::s32;
    return T;
}

template <typename T> static inline auto diffHelper() {
    if constexpr (std::is_integral_v<T>)
        return std::make_signed_t<T>{};
    else
        return T{};
}

template <typename T> using diff_t = decltype(diffHelper<T>());

template <typename T> static inline auto storageHelper() {
    if constexpr (sizeof(T) == 1) return uint8_t{};
    else if constexpr (sizeof(T) == 2) return uint16_t{};
    else if constexpr (sizeof(T) == 4) return uint32_t{};
    else if constexpr (sizeof(T) == 8) return uint64_t{};
    else return T{};
}

template <typename T> using storage_t = decltype(storageHelper<T>());

/* Check if two values are within a tolerance of each other, or are identical NaNs. */
template <typename T, typename U>
static inline bool checkNearby(T val, T ref, U maxAbsErr)
{
    return (std::abs(diff_t<T>(val - ref)) <= maxAbsErr) || identicalBits(val, ref);
}

template <typename T>
static inline bool checkULPNearby(T val, T ref, int maxULPErr)
{
    if constexpr (isComplex<T>()) {
        return checkULPNearby(val.real(), ref.real(), maxULPErr)
            && checkULPNearby(val.imag(), ref.imag(), maxULPErr);
    } else {
        storage_t<T> valBits, refBits;
        std::memcpy(&valBits, &val, sizeof(val));
        std::memcpy(&refBits, &ref, sizeof(ref));
        return checkNearby(valBits, refBits, maxULPErr);
    }
}

/* Type conversion, with saturation and rounding. */
template <typename From, typename To> To static inline convert(From v) {
    if constexpr (isComplex<From>()) {
        if constexpr (isComplex<To>())
            return To{convert<real_t<From>, real_t<To>>(v.real()),
                      convert<real_t<From>, real_t<To>>(v.imag())};
        else
            return convert<real_t<From>, To>(v.real());
    } else {
        if constexpr (isIntegral<To>() && !isIntegral<From>()) {
            if (v > 0)
                v = From(std::floor(v + 0.5f));
            else
                v = From(std::ceil(v - 0.5f));
        }
        if constexpr (isIntegral<To>()) {
            if (v < std::numeric_limits<To>::min()) v = From(std::numeric_limits<To>::min());
            if (v > std::numeric_limits<To>::max()) v = From(std::numeric_limits<To>::max());
        }
        return To(v);
    }
}

/* Reference types. In many places, to reduce templating,
   operations are performed in a containing "supertype", which
   is one of f32/f64/cf32/cf64. Scalar data is stored as a
   "superscalar", namely cf64. */
using superscalar = complex<double>;

static inline constexpr gemmstone::Type supertype(gemmstone::Type T) {
    using namespace gemmstone;
    switch (T) {
        case Type::u64:
        case Type::s64:
        case Type::u32:
        case Type::s32:
        case Type::f64: return Type::f64;
        default:        return Type::f32;
    }
}

template <typename... Ts>
static inline constexpr gemmstone::Type supertype(gemmstone::Type T1, gemmstone::Type T2, Ts... T)
{
    using namespace gemmstone;
    auto ST1 = supertype(T1);
    auto ST2 = supertype(T2, T...);
    bool cx = ST1.isComplex() || ST2.isComplex();
    bool dp = ST1.real() == Type::f64 || ST2.real() == Type::f64;

    return dp ? Type::f64 : Type::f32;
}

static inline constexpr bool isSupertype(gemmstone::Type T) { return T == supertype(T); }

template <typename T>
static inline auto stHelper() {
    if constexpr (std::is_same_v<T, complex<double>>
               || std::is_same_v<T, complex<float>>
               || std::is_same_v<T, double>)
        return T{};
    else if constexpr (isIntegral<T>() && sizeof(T) >= 4)
        return double{};
    else
        return float{};
}

template <typename T> using supertype_t = decltype(stHelper<T>());

/* Typed memory storage */
template <typename fp> inline fp *alloc(size_t nelem);
template <typename fp> inline void dealloc(fp *mem, size_t nelem);

class TypedData
{
public:
    TypedData() noexcept {}

    /* Non-owning constructor */
    TypedData(gemmstone::Type T_, void *ptr_, size_t bytes_) : T(T_), ptr((uint8_t *) ptr_), bytes(bytes_) {}

    /* Owning constructor */
    TypedData(gemmstone::Type T_, size_t elements) : T(T_), bytes(elements * T_.paddedSize()), own(true) {
        ptr = alloc<uint8_t>(bytes);
    }

    TypedData(const TypedData &other) : T(other.T), bytes(other.bytes), own(other.own) {
        if (own) {
            ptr = alloc<uint8_t>(bytes);
            std::memmove(ptr, other.ptr, bytes);
        } else
            ptr = other.ptr;
    }

    TypedData(TypedData &&other) noexcept { *this = std::move(other); }

    ~TypedData() {
        if (own) dealloc(ptr, bytes);
    }

    TypedData &operator=(const TypedData &other) {
        TypedData temp{other};
        swap(*this, temp);
        return *this;
    }

    TypedData &operator=(TypedData &&other) noexcept {
        swap(*this, other);
        return *this;
    }

    friend void swap(TypedData &d1, TypedData &d2) {
        using std::swap;
        swap(d1.T, d2.T);
        swap(d1.ptr, d2.ptr);
        swap(d1.bytes, d2.bytes);
        swap(d1.own, d2.own);
    }

    void reset() { *this = TypedData(); }

    size_t size() const  { return bytes / T; }

    void zero() { std::memset(ptr, 0, bytes); }

    /* Non-owning views */
    TypedData view()                 { return subview(bytes); }
    TypedData subview(size_t bytes_) {
        if (bytes_ > bytes) throw std::runtime_error("Out-of-bounds subview");
        return TypedData(T, ptr, bytes_);
    }

    gemmstone::Type type() const { return T; }

    /* Typed pointer access */
    template <typename U> U *as() {
        if (typeEnum<U> != T)
            throw std::runtime_error("Data does not have expected type");
        else
            return reinterpret_cast<U*>(ptr);
    }

    template <typename U> const U *as() const {
        return const_cast<TypedData*>(this)->as<U>();
    }

    template <typename U> U *asUnchecked() {
        return reinterpret_cast<U*>(ptr);
    }

    template <typename U> const U *asUnchecked() const {
        return const_cast<TypedData*>(this)->asUnchecked<U>();
    }

    /* Pointer arithmetic */
    TypedData operator+(ptrdiff_t x) { return TypedData(T, ptr + x * T.paddedSize(), bytes - x * T.paddedSize()); }
    TypedData operator-(ptrdiff_t x) { return TypedData(T, ptr - x * T.paddedSize(), bytes + x * T.paddedSize()); }

    /* Conversion between packed ("smashed") and unpacked sub-byte types.
       Benchstone maintains one sub-byte element per byte while manipulating data. */
    inline void smash();
    inline void unsmash();

    size_t smashedBytes() { return (T.bits() < 8) ? (bytes * T) : bytes; }

    gemmstone::Type T = gemmstone::Type::invalid;
    uint8_t *ptr = nullptr;
    size_t bytes = 0;
    bool own = false;
};


template <typename fp>
inline fp *alloc(size_t nelem)
{
    if (nelem == 0) nelem = 1;

#ifdef BENCHSTONE_USE_MMAP
    void *mem = mmap(NULL, nelem * sizeof(fp), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (mem == MAP_FAILED) {
        fprintf(stderr, "mmap failed: could not allocate memory (%zu bytes)\n", nelem * sizeof(fp));
        exit(3);
    }

    bzero(mem, nelem * sizeof(fp));
#else
    void *mem = calloc(nelem * sizeof(fp), 1);
#endif

    return (fp *) mem;
}

template <typename fp>
inline void dealloc(fp *mem, size_t nelem)
{
    if (!mem) return;

#ifdef BENCHSTONE_USE_MMAP
    munmap(mem, nelem * sizeof(fp));
#else
    free(mem);
#endif
}

void TypedData::smash()
{
    if (T.perByte() == 1) return;
    if (T.perByte() == 2) {
        for (size_t i = 0; i < bytes / 2; i++)
            ptr[i] = ptr[2 * i] | (ptr[2 * i + 1] << 4);
        if (bytes & 1)
            ptr[bytes >> 1] = ptr[bytes - 1];
        return;
    }
    if (T.perByte() == 4) {
        for (size_t i = 0; i < bytes / 4; i++)
            ptr[i] = ptr[4 * i] | (ptr[4 * i + 1] << 2) | (ptr[4 * i + 2] << 4) | (ptr[4 * i + 3] << 6);
        uint8_t val = 0;
        switch (bytes % 4) {
        case 3: val |= ptr[bytes - 1] << 4;
        case 2: val |= ptr[bytes - (bytes % 4) + 1] << 2;
        case 1: val |= ptr[bytes - (bytes % 4)];
        default: break;
        }
        ptr[bytes >> 2] = val;
        return;
    }
    gemmstone::stub();
}

void TypedData::unsmash()
{
    if (T.perByte() == 1) return;
    if (T.perByte() == 2) {
        if (bytes & 1)
            ptr[bytes - 1] = ptr[bytes >> 1] & 0xF;
        for (size_t i = bytes / 2 - 1; i >= 1; i--) {
            ptr[2 * i] = ptr[i] & 0xF;
            ptr[2 * i + 1] = ptr[i] >> 4;
        }
        if (bytes >= 2) {
            ptr[1] = ptr[0] >> 4;
            ptr[0] &= 0xF;
        }
    }
    if (T.perByte() == 4) {
        switch (bytes % 4) {
        case 3: ptr[bytes - 1] = ptr[bytes >> 2] >> 4;
        case 2: ptr[bytes - (bytes % 4) + 1] = (ptr[bytes >> 2] >> 2) & 0x3;
        case 1: ptr[bytes - (bytes % 4)] = ptr[bytes >> 2] & 0x3;
        default: break;
        }
        for (size_t i = bytes / 4 - 1; i >= 0; i--) {
            ptr[4 * i + 3] = ptr[i] >> 6;
            ptr[4 * i + 2] = (ptr[i] >> 4) & 0x3;
            ptr[4 * i + 1] = (ptr[i] >> 2) & 0x3;
            ptr[4 * i] = ptr[i] & 0x3;
        }
    }
    gemmstone::stub();
}

enum Metric {
    L1R,
    L1W,
    L3R,
    L3W,
    L3RW,
    L3Hit,
    L3Miss,
    L3CLR,
    L3CLW,
    L3Busy,
    L3InputAvailable,
    L3OutputReady,
    L3SuperQFull,
    L3Stall,
    MainR,
    MainW,
    MainR32B,
    MainW32B,
    MainR64B,
    MainW64B,
    MainRL3,
    MainWL3,
    MainRRate,
    MainWRate,
    MainQFull,
    TLBMiss,
    ICacheMiss,
    ComputeStall,
    SharedFunctionStall,
    ALUWriteBackStall,
    BarrierStall,
    ControlStall,
    InstrFetchStall,
    OtherStall,
    PipeStall,
    SbidStall,
    SendStall,
    MetricCount
};

static const std::unordered_map<std::string, Metric> MetricNames = {
    {"LOAD_STORE_CACHE_BYTE_READ", L1R},
    {"LOAD_STORE_CACHE_BYTE_WRITE", L1W},
    {"L3_BYTE_READ", L3R},
    {"L3_BYTE_WRITE", L3W},
    {"GPU_MEMORY_BYTE_READ", MainR},
    {"GPU_MEMORY_BYTE_WRITE", MainW},
    {"GPU_MEMORY_32B_TRANSACTION_READ", MainR32B},
    {"GPU_MEMORY_32B_TRANSACTION_WRITE", MainW32B},
    {"GPU_MEMORY_64B_TRANSACTION_READ", MainR64B},
    {"GPU_MEMORY_64B_TRANSACTION_WRITE", MainW64B},
    {"GPU_MEMORY_L3_READ", MainRL3},
    {"GPU_MEMORY_L3_WRITE", MainWL3},
    {"GPU_MEMORY_BYTE_READ_RATE", MainRRate},
    {"GPU_MEMORY_BYTE_WRITE_RATE", MainWRate},
    {"GPU_MEMORY_REQUEST_QUEUE_FULL", MainQFull},
    {"GtiReadThroughput", MainR},
    {"GtiWriteThroughput", MainW},
    {"L3ShaderThroughput", L3RW},
    {"L3_HIT", L3Hit},
    {"L3_MISS", L3Miss},
    {"L3_READ", L3CLR},
    {"L3_WRITE", L3CLW},
    {"L3_BUSY", L3Busy},
    {"L3_INPUT_AVAILABLE", L3InputAvailable},
    {"L3_OUTPUT_READY", L3OutputReady},
    {"L3_SUPERQ_FULL", L3SuperQFull},
    {"L3_STALL", L3Stall},
    {"TLB_MISS_L3NODE0", TLBMiss},
    {"TLB_MISS_L3NODE1", TLBMiss},
    {"TLB_MISS_L3NODE2", TLBMiss},
    {"TLB_MISS_L3NODE3", TLBMiss},
    {"TLB_MISS_L3NODE4", TLBMiss},
    {"TLB_MISS_L3NODE5", TLBMiss},
    {"TLB_MISS_L3NODE6", TLBMiss},
    {"TLB_MISS_L3NODE7", TLBMiss},
    {"TLB_MISS", TLBMiss},
    {"ICACHE_MISS_BANK0_XECORE0", ICacheMiss},
    {"XVE_STALL", ComputeStall},
    {"XVE_SHARED_FUNCTION_ACCESS_HOLD", SharedFunctionStall},
    {"XVE_STALL_ALUWR", ALUWriteBackStall},
    {"XVE_STALL_BARRIER", BarrierStall},
    {"XVE_STALL_CONTROL", ControlStall},
    {"XVE_STALL_INSTFETCH", InstrFetchStall},
    {"XVE_STALL_OTHER", OtherStall},
    {"XVE_STALL_PIPESTALL", PipeStall},
    {"XVE_STALL_SBID", SbidStall},
    {"XVE_STALL_SENDWR", SendStall},
};

using BufferFlags = uint32_t;
enum : uint32_t {
    BUF_CROSS_4GB = 1,
    BUF_COPY = 2,
    BUF_RW = 4,
    BUF_READ = 8,   BUF_R = 8,
    BUF_WRITE = 16, BUF_W = 16,
    BUF_UNINIT = 32,
};

struct BlockingInfo
{
    gemmstone::LoopType var;
    integer val;
};

struct FusedDriverParams
{
    int majorWalkDim;           // Dimension for largest-scale walk pattern.
    integer blockMN;            // m/n block size.
    integer blockK;             // k block size.
    integer initialZ0, stdZ0;   // z0 choices for initial copies, and other copies.
    int footprint[2];           // # of blocks in m/n dimensions for footprint.
    int tidThresh;              // Minimum EU thread ID that prefers copies.
    bool reuse;                 // Reuse temporary buffers?

    int slots() const { return footprint[0] + footprint[1]; }

    FusedDriverParams() : majorWalkDim(0), blockMN(512), blockK(1024), initialZ0(128), stdZ0(1024), footprint{8, 4}, tidThresh(6), reuse(true) {}
};

enum class DriverPhase {
    InitMultiple,     /* Create buffers for series of tests.    */
    Init,             /* Create buffers, etc.                   */
    PrepareList,      /* Prepare command list(s). (L0)          */
    Run,              /* Run.                                   */
    RunList,          /* Run command list(s). (L0)              */
    Finalize,         /* Copy back data and release resources.  */
    FinalizeMultiple  /* Release resources for series of tests. */
};

enum class TestAccuracy {
    Unchecked = -1, OK = 0, Overrun = 1, Inaccurate = 2, RuntimeError = 3
};

enum class TestMode {
    Accuracy, Time, BeginSeries, EndSeries, TimeOneOfSeries
};

struct TestResult {
    TestResult() = default;
    TestResult(double freq) : clockFrequency(freq) {}
    double clockFrequency;
    TestAccuracy accuracy = TestAccuracy::Unchecked;
    double hostTime = 0.;      // seconds
    double deviceTime = 0.;    // seconds
    double avgFreq = 0.;
    double time = 0.;          // seconds
    double effTime = 0.;       // seconds (accounts for frequency throttling)
    double bw = 0.;            // bytes/s
    double flops = 0.;         // ops/s
    double efficiency = 0.;    // roofline model, percentage
    double effEfficiency = 0.; // efficiency (accounts for frequency throttling)
    int iterations = 0;
    std::vector<double> tileFreqs;
    std::array<double, MetricCount> metrics;
    std::string failingCase;

    std::vector<double> allHTimes, allDTimes, allFreqs;

    bool failed() const {
        return (static_cast<int>(accuracy) > 0);
    }
    void markAccuracy(TestAccuracy accuracyIn) {
        accuracy = static_cast<TestAccuracy>(std::max(static_cast<int>(accuracy), static_cast<int>(accuracyIn)));
    }
    TestResult &operator+=(const TestResult &other) {
        int totalIters = iterations + other.iterations;
        markAccuracy(other.accuracy);

        if (iterations > 0 && avgFreq > 0 && other.avgFreq == 0)
            return *this;   /* Skip this result if average frequency could not be measured. */
        else if (other.iterations > 0 && avgFreq == 0 && other.avgFreq > 0) {
            *this = other;  /* Overwrite past results if average frequency is now measurable. */
            return *this;
        }

        if (totalIters > 0) {
            hostTime   = (iterations * hostTime   + other.iterations * other.hostTime)   / totalIters;
            deviceTime = (iterations * deviceTime + other.iterations * other.deviceTime) / totalIters;
            avgFreq    = (iterations * avgFreq    + other.iterations * other.avgFreq)    / totalIters;
            for (int m = 0; m < MetricCount; m++)
                metrics[m] = (iterations * metrics[m] + other.iterations * other.metrics[m]) / totalIters;

            if (other.iterations == 0)
                /* no op */;
            else if (iterations == 0)
                tileFreqs = other.tileFreqs;
            else if (other.tileFreqs.size() == tileFreqs.size()) for (size_t i = 0; i < tileFreqs.size(); i++)
                tileFreqs[i] = (iterations * tileFreqs[i] + other.iterations * other.tileFreqs[i]) / totalIters;
            else
                tileFreqs.clear();
        }
        iterations = totalIters;
        allHTimes.insert(allHTimes.end(), other.allHTimes.begin(), other.allHTimes.end());
        allDTimes.insert(allDTimes.end(), other.allDTimes.begin(), other.allDTimes.end());
        allFreqs.insert(allFreqs.end(), other.allFreqs.begin(), other.allFreqs.end());
        return *this;
    }
    double deviceTimeIfAvailable() const {
        return (deviceTime != 0.) ? deviceTime : hostTime;
    }
    void recordIteration() {
        if (iterations != 0) throw std::runtime_error("TestResult is in invalid state.");
        iterations = 1;
        allHTimes.push_back(hostTime);
        allDTimes.push_back(deviceTime);
        allFreqs.push_back(avgFreq);
    }
    double effectiveHostTime()   const { return effectiveTime(allHTimes); }
    double effectiveDeviceTime() const { return effectiveTime(allDTimes); }
    double effectiveDeviceTimeIfAvailable() const {
        return (deviceTime != 0.) ? effectiveDeviceTime() : effectiveHostTime();
    }
    double medianHostTime()   const { return medianTime(allHTimes); }
    double medianDeviceTime() const { return medianTime(allDTimes); }
    double medianDeviceTimeIfAvailable() const {
        return (deviceTime != 0.) ? medianDeviceTime() : medianHostTime();
    }
    double medianEffectiveDeviceTimeIfAvailable() const {
        return (deviceTime != 0.) ? medianEffectiveTime(allDTimes) : medianEffectiveTime(allHTimes);
    }
    double stddevHostTime(bool effective = false)      const { return stddevTime(allHTimes, effective); }
    double stddevDeviceTime(bool effective = false)    const { return stddevTime(allDTimes, effective); }
    double stddevDeviceTimeIfAvailable(bool effective = false) const {
        return (deviceTime != 0.) ? stddevDeviceTime(effective) : stddevHostTime(effective);
    }
    double stddevBestAvailable() const {
        return stddevDeviceTimeIfAvailable(avgFreq > 0.);
    }
    double meanStddevBestAvailable() const {
        return stddevBestAvailable() / std::sqrt(double(iterations));
    }

    double bestTimeAvailable() const {
        return (avgFreq > 0) ? reportedEffectiveDeviceTimeIfAvailable() : reportedDeviceTimeIfAvailable();
    }

    static bool useMedian() {
        static bool avg = getenv("USE_AVG_TIME");
        return !avg;
    }
    static const char *timeLabel() { return useMedian() ? "Median" : "Average"; }

    double reportedDeviceTimeIfAvailable() const {
        return useMedian() ? medianDeviceTimeIfAvailable() : deviceTimeIfAvailable();
    }
    double reportedEffectiveDeviceTimeIfAvailable() const {
        return useMedian() ? medianEffectiveDeviceTimeIfAvailable() : effectiveDeviceTimeIfAvailable();
    }

private:
    double effectiveTime(const std::vector<double> &times) const {
        double sum = 0.;
        for (int i = 0; i < iterations; i++)
            sum += times[i] * allFreqs[i] / clockFrequency;
        return sum / iterations;
    }

    static double medianOf(std::vector<double> v) {
        if (v.empty()) return 0.;
        size_t n = v.size();
        size_t mid = n / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        if (n & 1) return v[mid];
        double upper = v[mid];
        double lower = *std::max_element(v.begin(), v.begin() + mid);
        return 0.5 * (lower + upper);
    }

    double medianTime(const std::vector<double> &times) const {
        return medianOf(std::vector<double>(times.begin(), times.begin() + iterations));
    }

    double medianEffectiveTime(const std::vector<double> &times) const {
        std::vector<double> etimes(iterations);
        for (int i = 0; i < iterations; i++)
            etimes[i] = times[i] * allFreqs[i] / clockFrequency;
        return medianOf(std::move(etimes));
    }

    double stddevTime(const std::vector<double> &times, bool effective) const {
        auto etime = [&](int i) {
            return effective ? times[i] * allFreqs[i] / clockFrequency
                             : times[i];
        };
        if (iterations == 1)
            return std::numeric_limits<double>::quiet_NaN();
        double sum = 0.;
        for (int i = 0; i < iterations; i++)
            sum += etime(i);
        double mean = sum / iterations;
        double stddev = 0.;
        for (int i = 0; i < iterations; i++) {
            double diff = etime(i) - mean;
            stddev += diff * diff;
        }
        stddev = sqrt(stddev / (iterations - 1));
        return stddev;
    }
};


/***********\
|* Helpers *|
\***********/
struct Duration {
    double seconds;
};

inline std::ostream &operator<<(std::ostream &s, Duration dur)
{
    if (!std::isfinite(dur.seconds))
        s << "n/a";
    else if (dur.seconds >= 1.)
        s << dur.seconds << 's';
    else if (dur.seconds >= 1e-3)
        s << dur.seconds * 1e3 << "ms";
    else if (dur.seconds >= 1e-6)
        s << dur.seconds * 1e6 << "us";
    else
        s << dur.seconds * 1e9 << "ns";

    return s;
}

inline uint32_t irecip(uint32_t x)
{
    auto shift = int(floor(log2(double(x))));
    return uint32_t(((uint64_t(0x100000000) << shift) + x - 1) / x);
}

inline char loopChar(gemmstone::LoopType l, int loopNum = -1)
{
    if (loopNum == 0 || loopNum == 1) switch (l) {
        case gemmstone::LoopMNHilbertMNK:
        case gemmstone::LoopMNHilbertNMK:
        case gemmstone::LoopMNBoustrophedonMNK:
        case gemmstone::LoopMNBoustrophedonNMK: return 'M' + loopNum;
        default:                                break;
    }

    switch (l) {
        case gemmstone::LoopM:                  return 'M';
        case gemmstone::LoopN:                  return 'N';
        case gemmstone::LoopK:                  return 'K';
        case gemmstone::LoopNone:               return '-';
        case gemmstone::LoopMNHilbertMNK:       return 'H';
        case gemmstone::LoopMNHilbertNMK:       return 'h';
        case gemmstone::LoopMNBoustrophedonMNK: return 'B';
        case gemmstone::LoopMNBoustrophedonNMK: return 'b';
        default:                                return '?';
    }
}

inline int typeSize(char type)
{
    return gemmstone::charToType(type).paddedSize();
}

inline int elementsPerByte(char type)
{
    return gemmstone::charToType(type).perByte();
}

inline integer offset(bool trans, integer ld, integer i, integer j)
{
    return trans ? (i * ld + j) : (j * ld + i);
}

template <typename fp>
fp *offset(fp *src, bool trans, integer ld, integer i, integer j)
{
    return src + offset(trans, ld, i, j);
}

// Choose "nice" leading dimension at least as large as given ld.
//  If upperBound == true, then return an upper bound for niceLD(a) over all a <= ld.
template <typename I>
inline I niceLD(ngen::HW hw, Type T, I ld, bool upperBound = false)
{
    ld *= T;

    I align = (hw >= ngen::HW::Xe2) ? 512 : 128;
    I noalign = 4096;
    I newld = (ld + align - 1) & -align;
    if (!(newld & (noalign - 1)) || upperBound)
        newld += align;

    return newld / T;
}

inline const char *hwName(ngen::HW hw, bool shortName = false)
{
    using ngen::HW;
    switch (hw) {
        case HW::Gen12LP: return shortName ? "12LP" : "Gen12LP";
        case HW::XeHP:    return shortName ? "12HP" : "XeHP";
        case HW::XeHPG:   return shortName ? "12p7" : "XeHPG";
        case HW::XeHPC:   return shortName ? "12p8" : "XeHPC";
        case HW::Xe2:     return "Xe2";
        case HW::Xe3:     return "Xe3";
        case HW::Xe3p:    return "Xe3p";
        default:               return "unknown";
    }
}

inline int threadsPerEU(ngen::HW hw, const gemmstone::CommonDriverInfo &info)
{
    int threads;
        {
        threads = (hw >= ngen::HW::XeHP ? 8 : 7);
        if (info.grfCount >= 256)
            threads >>= 1;
    }
    return threads;
}

inline int hwThreads(ngen::HW hw, int EUs, const gemmstone::CommonDriverInfo &info)
{
    return EUs * threadsPerEU(hw, info);
}

inline int maxThreadsPerTG(ngen::HW hw, const gemmstone::CommonDriverInfo &info)
{
    int maxThreads = 0;
    switch (hw) {
        case ngen::HW::Gen12LP: maxThreads = 64; break;
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG:   maxThreads = 128; break;
        case ngen::HW::XeHPC:
        case ngen::HW::Xe2:
        case ngen::HW::Xe3:
        case ngen::HW::Xe3p:    maxThreads = 64; break;
        default: gemmstone::stub();
    }
    if (info.grfCount >= 256)
        maxThreads >>= 1;
    return maxThreads;
}

inline int eusPerSS(ngen::HW hw)
{
    switch (hw) {
        case ngen::HW::Gen12LP:
        case ngen::HW::XeHP:
        case ngen::HW::XeHPG: return 16;
        default:        return 8;
    }
}

inline double throughput(const ngen::Product &product, const Level3KernelInfo &info)
{
    return throughput(product, info.typeAInt(), info.typeBInt(), info.typeCInt(), info.systolic);
}

inline double triangleEntries(integer m, integer n, bool lower)
{
    auto mn = double(std::min<integer>(m, n));
    auto x = double(std::max<integer>(0, lower ? (m - n) : (n - m)));

    return (mn * (mn + 1) * 0.5) + (mn * x);
}

inline double ops(integer m, integer n, integer k, const Level3KernelInfo &info, int wgLimit)
{
    int fmaFlops = 2 * (info.typeA().complexComponents())
                     * (info.typeB().complexComponents());

    double ops;
    if (info.trxm() && info.left)
        ops = triangleEntries(m, k, info.lower) * n;
    else if (info.trxm() && !info.left)
        ops = triangleEntries(k, n, info.lower) * m;
    else if (info.triangular)
        ops = triangleEntries(m, n, info.lower) * k;
    else
        ops = double(m) * double(n) * double(k);

    if (info.isLinearOrder() && wgLimit > 0) {
        if (info.trxm()) gemmstone::stub();
        auto maxOps = double(info.wgTile(LoopM) * info.wgTile(LoopN)) * double(wgLimit) * double(k);
        ops = std::min(ops, maxOps);
    }

    return fmaFlops * ops;
}

struct Traffic {
    double read, write, l3Read;
};

inline Traffic traffic(integer m, integer n, integer k, integer batch, const Level3KernelInfo &info, bool updating, int wgLimit)
{
    double A_traffic, B_traffic, C_traffic, C_trafficW;

    A_traffic = (info.trxm() && info.left)  ? triangleEntries(m, k, info.lower)
                                            : double(m) * double(k);
    B_traffic = (info.trxm() && !info.left) ? triangleEntries(k, n, info.lower)
                                            : double(n) * double(k);
    C_traffic = info.triangular             ? triangleEntries(m, n, info.lower)
                                            : double(m) * double(n);
    if (info.isLinearOrder() && wgLimit > 0) {
        if (info.trxm()) gemmstone::stub();
        auto C_trafficMax = double(info.wgTile(LoopM) * info.wgTile(LoopN)) * double(wgLimit);
        C_traffic = std::min(C_traffic, C_trafficMax);
    }

    A_traffic *= batch;
    B_traffic *= batch;
    C_traffic *= batch;

    C_trafficW = C_traffic;
    if (!info.trxm() && !updating)
        C_traffic = 0;

    int mWGTiles = int(div_up(m, info.wgTile(LoopM)));
    int nWGTiles = int(div_up(n, info.wgTile(LoopN)));

    double aSize = info.typeA().bits() * 0.125;
    double bSize = info.typeB().bits() * 0.125;
    double cSize = info.typeC().bits() * 0.125;

    Traffic result;
    result.read = A_traffic * aSize + B_traffic * bSize + C_traffic * cSize;
    result.l3Read = A_traffic * nWGTiles * aSize + B_traffic * mWGTiles * bSize + C_traffic * cSize;
    result.write = C_trafficW * cSize;

    return result;
}

inline std::tuple<integer, integer, integer> getBlocking(ngen::HW hw, int EUs, integer m, integer n, integer k, const Level3KernelInfo &info,
    bool noBlocking, bool doSKParallelOverride, std::vector<BlockingInfo> skParallelOverride,
    bool fixedBlocking, const integer *NBlockOverrides, bool testNonKParallelGlobal)
    {
        integer BlockM, BlockN, BlockK;
        bool overrideK = false;

        if (k <= 0) k = 1;

        BlockM = info.blocking[LoopM];
        BlockN = info.blocking[LoopN];
        BlockK = info.blocking[LoopK];

        if (noBlocking) {
            BlockM = m;
            BlockN = n;
            BlockK = k;
        }

        integer BlockKBeta0 = k;
        
        int FillMultiplier = 1;
        auto fillMult = getenv("FILL_MULTIPLIER");
        if (fillMult) FillMultiplier = atoi(fillMult);

        // Try to fill available HW threads for k-parallel kernels.
        if ((info.kParallel() || info.shrinkWGK()) && (hwThreads(hw, EUs, info) > 0)) {
            int nm = int(div_up(std::min(m, BlockM), info.unroll[LoopM]));
            int nn = int(div_up(std::min(n, BlockN), info.unroll[LoopN]));
            if (info.fixedWG()) {
                nm = align_up(nm, info.wg[LoopM]);
                nn = align_up(nn, info.wg[LoopN]);
            }
            int nk = hwThreads(hw, EUs, info) / (nm * nn) * FillMultiplier;
            if (info.kPadding()) nk--;
            nk = std::max(1, nk);
            if (info.kParallelLocal() && !info.kParallel()) {
                nk = std::min(nk, info.wg[LoopK]);
                BlockKBeta0 = BlockK = div_up(k, nk);
            } else
                BlockK = div_up(k, nk);
        } else if (info.kParallelLocal())
            BlockKBeta0 = BlockK = div_up(k, info.wg[LoopK]);

        // Process overrides.
        if (doSKParallelOverride) {
            for (auto &o: skParallelOverride) {
                switch (o.var) {
                    case LoopM: BlockM = o.val; break;
                    case LoopN: BlockN = o.val; break;
                    case LoopK: BlockK = o.val; overrideK = true; break;
                    default: break;
                }
            }
        }

        // Even out the blocks.
        if (!fixedBlocking) {
            if (m > BlockM || NBlockOverrides[LoopM]) {
                auto mdiv = m / BlockM;
                if (NBlockOverrides[LoopM]) mdiv = NBlockOverrides[LoopM];
                BlockM = div_up(m, mdiv);
            } else if (m < BlockM)
                BlockM = m;
            if (n > BlockN || NBlockOverrides[LoopN]) {
                auto ndiv = (n + (BlockN >> 1)) / BlockN;
                if (NBlockOverrides[LoopN]) ndiv = NBlockOverrides[LoopN];
                BlockN = div_up(n, ndiv);
            } else if (n < BlockN)
                BlockN = n;
            if ((k > BlockK || NBlockOverrides[LoopK]) && !overrideK) {
                auto kdiv = (k + BlockK - 1) / BlockK;
                if (NBlockOverrides[LoopK]) kdiv = NBlockOverrides[LoopK];
                BlockK = div_up(k, kdiv);
            } else if (k < BlockK)
                BlockK = k;
        }

        if ((info.fixedBeta() || testNonKParallelGlobal) && (info.beta != 1) && !overrideK && !info.fusedBeta())
            BlockK = BlockKBeta0;

        BlockM = align_up(BlockM, info.wgTile(LoopM));
        BlockN = align_up(BlockN, info.wgTile(LoopN));
        BlockK = align_up(BlockK, info.unroll[LoopK]);

        return std::make_tuple(BlockM, BlockN, BlockK);
    }

inline integer getBlocking(ngen::HW hw, int EUs, integer mn, integer k, CopyKernelInfo &info, bool noBlocking, integer NBlockOverrides[3])
{
    if (k <= 0) k = 1;

    integer Block = 2048;

    integer z = info.xLoop() ? mn : k; /* Dimension of kernel's inner loop */
    integer w = info.xLoop() ? k : mn; /* Other dimension */
    auto unrollW = info.xLoop() ? info.unrollY() : info.unrollX();
    auto unrollZ = info.xLoop() ? info.unrollX() : info.unrollY();

    if (noBlocking)
        Block = info.zParallel() ? z : k;
    else if (info.zParallel() && !NBlockOverrides[2]) {
        // Block in z dimension.
        if (info.zBlockFixed())
            Block = unrollZ;
        else {
            int threadsW = int(div_up(w, unrollW));
            threadsW = align_up(threadsW, info.wg[0]);
            int threadsZ = int(div_up(hwThreads(hw, EUs, info), threadsW));
            Block = div_up(z, threadsZ);
            Block = std::max<integer>(Block, 32);
            Block = align_up(Block, unrollZ);
        }
    } else if (k > Block || NBlockOverrides[2]) {
        // Block in k dimension.
        auto kdiv = div_up(k, Block);
        if (NBlockOverrides[2]) kdiv = NBlockOverrides[2];
        Block = div_up(k, kdiv);
        Block = align_up(Block, 32);
        Block = align_up(Block, info.unrollY());
    }

    return Block;
}

inline std::tuple<int, int> getDefaultWGSize(ngen::HW hw, bool copyA, bool copyB, bool xParallel = false)
{
    std::tuple<int, int> result;

    result = xParallel ? std::make_tuple(4, div_up(eusPerSS(hw), 4))
                       : std::make_tuple(2, 8);

    return result;
}

inline void applyWGOverrides(const gemmstone::CommonDriverInfo &info, int WGOverride[3], int *wg0, int *wg1 = nullptr, int *wg2 = nullptr)
{
    bool ok = true;
    static bool warned = false;

    auto apply1 = [&](int idx, int *wg) {
        if (WGOverride[idx] && wg) {
            if (info.fixedWG() && *wg != WGOverride[idx] && idx < 2)   // wg2 can always be overridden...
                ok = false;
            else
                *wg = WGOverride[idx];
        }
    };

    apply1(0, wg0);
    apply1(1, wg1);
    apply1(2, wg2);

    if (!ok && !warned) {
        warned = true;
        yprintf("Kernel's workgroup size is fixed and cannot be overridden");
    }
}

inline std::vector<BlockingInfo> getFlexBlocking(ngen::HW hw, int EUs, integer m, integer n, integer k, const KernelInfo &info,
    bool noBlocking, bool doSKParallelOverride, std::vector<BlockingInfo> skParallelOverride,
    bool fixedBlocking, integer NBlockOverrides[3], bool testNonKParallelGlobal)
{
    int InnerBlockM, InnerBlockN;
    std::tie(InnerBlockM, InnerBlockN) = getDefaultWGSize(hw, false, false);

    InnerBlockM *= info.l3[0].unroll[LoopM];
    InnerBlockN *= info.l3[0].unroll[LoopN];

    integer BlockM, BlockN, BlockK;
    std::tie(BlockM, BlockN, BlockK) = getBlocking(hw, EUs, m, n, k, info.l3[0], noBlocking, doSKParallelOverride, skParallelOverride, fixedBlocking, NBlockOverrides, testNonKParallelGlobal);

    return {{LoopK, BlockK},
            {LoopN, BlockN},
            {LoopM, BlockM},
            {LoopN, InnerBlockN},
            {LoopM, InnerBlockM}};
}

inline void printFlexBlocking(const std::vector<BlockingInfo> &blocking)
{
    for (auto &b : blocking)
        std::cout << "mnk"[int(b.var)] << b.val << ' ';
}

inline FusedDriverParams getFusedParams(ngen::HW hw, integer m, integer n, integer k, const KernelInfo &info, bool doFusedParamsOverride, const FusedDriverParams &fusedParamsOverride)
{
    FusedDriverParams params;

    if (doFusedParamsOverride)
        params = fusedParamsOverride;
    else {
        params.tidThresh = threadsPerEU(hw, info.l3[0]) - 1;
        params.reuse = (hw <= ngen::HW::Gen12LP);
    }

    params.blockMN = align_up(params.blockMN, info.l3[0].unroll[LoopM]);
    params.blockMN = align_up(params.blockMN, info.l3[0].unroll[LoopN]);

    return params;
}

/* Adjust work sizes for fused EU architectures. */
inline void adjustWSFusedEUs(const size_t *local, size_t *global, const gemmstone::CommonDriverInfo &info)
{
    if (!info.fusedEUs()) return;

    auto fusedIdx = int(info.fusedLoop);
    if (info.isNMK() && (fusedIdx == LoopM || fusedIdx == LoopN))
        fusedIdx = (LoopM + LoopN) - fusedIdx;

    if (local[fusedIdx] & 1) throw std::runtime_error("Thread group size must be a multiple of 2 in the fused dimension.");
    global[fusedIdx] = (global[fusedIdx] + 1) & ~1;

    if (fusedIdx != 0 && info.loopOrder[0] != gemmstone::LoopNone) {
        if (local[0] & 1) throw std::runtime_error("Thread group size must be a multiple of 2 in the dimension of the innermost loop.");
        global[0] = (global[0] + 1) & ~1;
    }
}

/* Adjust work sizes to ensure uniform workgroups (local divides global evenly). */
inline void adjustWSUniform(ngen::HW hw, int dims, size_t *localWS, size_t *globalWS, const gemmstone::CommonDriverInfo &info, bool uniformWG)
{
    bool fixedUniformWG = info.fixedWG();
    uniformWG |= fixedUniformWG;

    if (!uniformWG) return;

    int lastNon1 = dims - 1;
    for (; lastNon1 >= 0 && (globalWS[lastNon1] == 1 || localWS[lastNon1] == 1); lastNon1--) ;

    for (int d = 0; d < dims; d++) {
        if (!fixedUniformWG) {
            localWS[d] = std::min(localWS[d], globalWS[d]);
            if (hw >= ngen::HW::XeHP && d < lastNon1)
                localWS[d] = ngen::utils::roundup_pow2(localWS[d]);
        }
        globalWS[d] = align_up(globalWS[d], localWS[d]);
    }
}

#endif
