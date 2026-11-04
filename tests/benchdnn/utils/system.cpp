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

#include <algorithm> // for std::find, std::any_of
#include <climits> // for PATH_MAX
#include <cstdio> // for snprintf
#include <cstdlib> // for realpath
#include <fstream> // for std::ifstream
#include <string>
#include <utility> // for std::move
#include <vector>

#include <assert.h>

#ifdef __linux__
#include <dirent.h> // for opendir, readdir, closedir
#include <sys/stat.h> // for stat
#endif

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#include <CL/cl_ext.h>
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
#include "oneapi/dnnl/dnnl_ze.hpp"
#endif

#include "cpu/platform.hpp"

#include "common.hpp"
#include "utils/engine.hpp"
#include "utils/parser.hpp"
#include "utils/summary.hpp"
#include "utils/system.hpp"

#if defined(_WIN32)
#include "windows.h"

size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
#include <unistd.h>
#include <sys/sysctl.h>

size_t get_cpu_ram_size() {
#ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
#else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
#endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

int get_gpu_ram_sizes(size_t &ram_size, size_t &max_alloc_size) {
    if (!is_gpu()) return OK;
    if (ram_size > 0 && max_alloc_size > 0) return OK;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_device_id ocl_dev = dnnl::ocl_interop::get_device(eng);

    cl_ulong ram_sz = 0;
    cl_int status = clGetDeviceInfo(ocl_dev, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    ram_size = (size_t)ram_sz;
    // For OCL runtime we allow allocation of buffers up to VRAM size,
    // with the usage of CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL flag.
    max_alloc_size = (size_t)ram_sz;
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    ram_size = (size_t)sycl_dev
                       .get_info<::sycl::info::device::global_mem_size>();
    max_alloc_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::max_mem_alloc_size>();
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;
    if (count > 1) {
        assert(!"Found more than a single entry for memory.");
        return FAIL;
    }

    ze_device_memory_properties_t ze_dev_mem_props {};
    ze_dev_mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
    status = zeDeviceGetMemoryProperties(ze_dev, &count, &ze_dev_mem_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    ram_size = (size_t)ze_dev_mem_props.totalSize;

    ze_device_properties_t ze_dev_props {};
    ze_dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ze_dev_props.pNext = nullptr;
    status = zeDeviceGetProperties(ze_dev, &ze_dev_props);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    max_alloc_size = (size_t)ze_dev_props.maxMemAllocSize;
    return OK;
#else
    assert(!"unsupported GPU runtime");
#endif
    ram_size = 0;
    max_alloc_size = 0;
    return OK;
}

int get_cpu_cache_size(cpu_cache_args_t &cache_args) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    cache_args.L2_size = get_per_core_cache_size(2);
    cache_args.L3_size = get_per_core_cache_size(3);
    cache_args.num_cores = get_num_cores();
    cache_args.total_socket_size
            = (cache_args.L2_size + cache_args.L3_size) * cache_args.num_cores;
#else
    // If functions are not available, just use 150 MiB.
    cache_args.total_socket_size = 150 * 1024 * 1024;
#endif
    return OK;
}

int get_gpu_cache_size(size_t &cache_size) {
    if (!is_gpu()) return OK;

    static size_t _cache_size = 0;
    if (_cache_size > 0) {
        cache_size = _cache_size;
        return OK;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_int status = CL_SUCCESS;
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong cache_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            sizeof(cl_ulong), &cache_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    _cache_size = (size_t)cache_sz;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    _cache_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::global_mem_cache_size>();
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    auto ze_dev = dnnl::ze_interop::get_device(eng);

    ze_result_t status = ZE_RESULT_SUCCESS;

    uint32_t count = 0;
    status = zeDeviceGetCacheProperties(ze_dev, &count, nullptr);
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    std::vector<ze_device_cache_properties_t> ze_cache_props(count);
    for (auto &e : ze_cache_props) {
        e.stype = ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
    }

    status = zeDeviceGetCacheProperties(ze_dev, &count, ze_cache_props.data());
    if (status != ZE_RESULT_SUCCESS) return FAIL;

    for (uint32_t i = 0; i < count; i++) {
        if (ze_cache_props[i].flags == 0) {
            _cache_size = ze_cache_props[i].cacheSize;
            break;
        }
    }
#else
    assert(!"unsupported GPU runtime");
#endif
    cache_size = _cache_size;
    return OK;
}

#ifdef __linux__
// A missing or unparsable file leaves `value` untouched and returns `FAIL` so
// the caller skips a setting that can't be verified rather than acting on a
// bogus reading.
static int read_sysfs_int(const std::string &path, int64_t &value) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return FAIL;
    ifs >> value;
    if (ifs.fail()) return FAIL;
    return OK;
}

static std::string read_sysfs_str(const std::string &path) {
    std::ifstream ifs(path);
    std::string value;
    if (!(ifs >> value)) return std::string();
    return value;
}

// TODO: the sysfs traversal helpers below (path_exists, list_dir, and the
// manual path concatenation in find_gpu_gts_*) reimplement functionality
// available in `std::filesystem` (exists, directory_iterator, path). Replace
// them with `<filesystem>` once benchdnn moves to the C++17 standard.
static bool path_exists(const std::string &path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

static std::vector<std::string> list_dir(const std::string &path) {
    std::vector<std::string> names;
    DIR *dir = ::opendir(path.c_str());
    if (!dir) return names;
    for (dirent *e = ::readdir(dir); e != nullptr; e = ::readdir(dir)) {
        std::string n = e->d_name;
        if (n == "." || n == "..") continue;
        names.push_back(std::move(n));
    }
    ::closedir(dir);
    return names;
}

struct gpu_gt_t {
    std::string name;
    std::string min_freq_path;
    std::string max_freq_path;
};

// Path components encode the xe driver's sysfs ABI. Media GTs (those without
// rcs/ccs engines) don't run compute workloads, so their frequency is
// irrelevant to performance measurements and they are skipped.
static std::vector<gpu_gt_t> find_gpu_gts_xe(const std::string &card) {
    std::vector<gpu_gt_t> gts;
    const std::string device = card + "/device";
    for (const auto &tile : list_dir(device)) {
        if (tile.rfind("tile", 0) != 0) continue;
        const std::string tile_path
                = parser::utils::str_join(device, tile, '/');
        for (const auto &gt : list_dir(tile_path)) {
            const std::string gt_path
                    = parser::utils::str_join(tile_path, gt, '/');
            const std::string freq = gt_path + "/freq0";
            if (!path_exists(freq + "/min_freq")) continue;
            const std::string eng = gt_path + "/engines";
            if (path_exists(eng) && !path_exists(eng + "/rcs")
                    && !path_exists(eng + "/ccs"))
                continue;
            gts.push_back({parser::utils::str_join(tile, gt, '/'),
                    freq + "/min_freq", freq + "/max_freq"});
        }
    }
    return gts;
}

// Per-GT path is the i915 driver's sysfs ABI; the single-GT path at the card
// root covers kernels that don't expose a `gt/` hierarchy.
static std::vector<gpu_gt_t> find_gpu_gts_i915(const std::string &card) {
    std::vector<gpu_gt_t> gts;
    const std::string gt_root = card + "/gt";
    for (const auto &gt : list_dir(gt_root)) {
        const std::string gt_path = parser::utils::str_join(gt_root, gt, '/');
        const std::string min = gt_path + "/rps_min_freq_mhz";
        if (path_exists(min))
            gts.push_back({gt, min, gt_path + "/rps_max_freq_mhz"});
    }
    if (gts.empty() && path_exists(card + "/gt_min_freq_mhz"))
        gts.push_back(
                {"gt", card + "/gt_min_freq_mhz", card + "/gt_max_freq_mhz"});
    return gts;
}

// Parses the kernel's cpulist format, e.g. "0-3,8,10-11".
static std::vector<int> parse_cpulist(const std::string &list) {
    std::vector<int> cpus;
    size_t pos = list.empty() ? std::string::npos : 0;
    while (pos != std::string::npos) {
        const std::string range = parser::get_substr(list, pos, ',');
        size_t range_pos = 0;
        const int first = std::stoi(parser::get_substr(range, range_pos, '-'));
        const int last = range_pos == std::string::npos
                ? first
                : std::stoi(parser::get_substr(range, range_pos, '-'));
        for (int cpu = first; cpu <= last; cpu++)
            cpus.push_back(cpu);
    }
    return cpus;
}

static std::string get_cpu_sysfs_path(int cpu) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
}

struct cpu_group_t {
    std::string name;
    int cpu;
};

// Hybrid CPUs have core types with distinct frequency and idle-state settings,
// so each type must be verified, but cores of the same type are configured
// alike and a single representative per type keeps the check cheap.
//
// Intel hybrid systems expose a PMU per core type listing its CPUs. Atom cores
// without L3 are LP E-cores, which reside on a separate low-power island with
// own frequency limits, so the L3 presence tells them from regular E-cores.
// Without policy or PMU groups, use a single unnamed group.
static std::vector<cpu_group_t> get_cpu_groups() {
    std::vector<cpu_group_t> groups;
#if (defined(DNNL_AARCH64) && DNNL_AARCH64 == 1) \
        || (defined(DNNL_RV64) && DNNL_RV64 == 1)
    // CPUs in a cpufreq policy share frequency controls. Check one online
    // representative per policy on AArch64 and RV64.
    const std::string policy_root = "/sys/devices/system/cpu/cpufreq";
    for (const auto &policy : list_dir(policy_root)) {
        if (policy.rfind("policy", 0) != 0) continue;
        const std::string policy_path
                = parser::utils::str_join(policy_root, policy, '/');
        std::ifstream cpus(policy_path + "/affected_cpus");
        int cpu;
        if (cpus >> cpu) groups.push_back({policy, cpu});
    }
    if (!groups.empty()) {
        std::sort(groups.begin(), groups.end(),
                [](const cpu_group_t &a, const cpu_group_t &b) {
            return a.cpu < b.cpu;
        });
        return groups;
    }
#endif
    const auto p_cpus
            = parse_cpulist(read_sysfs_str("/sys/devices/cpu_core/cpus"));
    const auto e_cpus
            = parse_cpulist(read_sysfs_str("/sys/devices/cpu_atom/cpus"));
    const auto has = [](const std::vector<int> &cpus, int cpu) {
        return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
    };

    for (int cpu :
            parse_cpulist(read_sysfs_str("/sys/devices/system/cpu/online"))) {
        std::string name;
        if (has(p_cpus, cpu))
            name = "P-core";
        else if (has(e_cpus, cpu))
            name = path_exists(get_cpu_sysfs_path(cpu) + "/cache/index3")
                    ? "E-core"
                    : "LP E-core";
        const bool seen = std::any_of(groups.begin(), groups.end(),
                [&](const cpu_group_t &g) { return g.name == name; });
        if (!seen) groups.push_back({name, cpu});
    }
    return groups;
}

static bool report_cpu_perf_hints_impl() {
    bool any_hint = false;

    for (const auto &group : get_cpu_groups()) {
        const std::string cpu_path = get_cpu_sysfs_path(group.cpu);
        const std::string cpu_name = "cpu" + std::to_string(group.cpu);
        const std::string label = group.name.empty()
                ? "CPU (" + cpu_name + ")"
                : "CPU " + group.name + " (" + cpu_name + ")";

        // The effective (scaling) min and max frequency are expected to
        // coincide. When they don't, the frequency may float during the run
        // and skew performance measurements. Note: `cpuinfo_*_freq` report
        // the hardware limits and always differ, so `scaling_*_freq` (the
        // governor's allowed range) is inspected instead as it reflects
        // whether the frequency is pinned.
        int64_t min_freq = 0, max_freq = 0;
        if (read_sysfs_int(cpu_path + "/cpufreq/scaling_min_freq", min_freq)
                        == OK
                && read_sysfs_int(
                           cpu_path + "/cpufreq/scaling_max_freq", max_freq)
                        == OK) {
            if (min_freq != max_freq) {
                BENCHDNN_PRINT(0,
                        "[PERF_HINTS] Info: %s min (%ld kHz) and max (%ld "
                        "kHz) frequencies differ. Pin the frequency for "
                        "stable performance measurements.\n",
                        label.c_str(), (long)min_freq, (long)max_freq);
                any_hint = true;
            }
        }

        // Enabled cpuidle states are inspected rather than the
        // `intel_idle.max_cstate` boot parameter because states can also be
        // disabled at runtime and the check must not depend on the idle
        // driver. State 0 is polling and state 1 is C1, both wake up fast.
        const std::string idle_path = cpu_path + "/cpuidle";
        int deepest_idx = 1;
        std::string deepest_name;
        for (const auto &state : list_dir(idle_path)) {
            if (state.rfind("state", 0) != 0) continue;
            const int idx = std::stoi(state.substr(5));
            if (idx <= deepest_idx) continue;
            const std::string state_path
                    = parser::utils::str_join(idle_path, state, '/');
            int64_t disabled = 0;
            if (read_sysfs_int(state_path + "/disable", disabled) != OK
                    || disabled)
                continue;
            deepest_idx = idx;
            deepest_name = read_sysfs_str(state_path + "/name");
        }
        if (!deepest_name.empty()) {
            BENCHDNN_PRINT(0,
                    "[PERF_HINTS] Info: %s idle states up to %s are enabled. "
                    "Disable states deeper than C1 for stable performance "
                    "measurements.\n",
                    label.c_str(), deepest_name.c_str());
            any_hint = true;
        }
    }

    constexpr const char *no_turbo_path
            = "/sys/devices/system/cpu/intel_pstate/no_turbo";

    // Turbo mode is supposed to be disabled (`no_turbo` equals to `1`).
    int64_t no_turbo = 0;
    if (read_sysfs_int(no_turbo_path, no_turbo) == OK) {
        if (no_turbo != 1) {
            BENCHDNN_PRINT(0, "%s",
                    "[PERF_HINTS] Info: CPU Turbo mode is enabled. Disable "
                    "it for stable performance measurements.\n");
            any_hint = true;
        }
    }

    return any_hint;
}

// Returns "dddd:bb:dd.f" of the test engine's GPU, or "" if unknown.
static std::string get_gpu_pci_bdf() {
    char bdf[16] = {};
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_device_pci_bus_info_khr p {};
    if (clGetDeviceInfo(dnnl::ocl_interop::get_device(eng),
                CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(p), &p, nullptr)
            != CL_SUCCESS)
        return {};
    snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", p.pci_domain, p.pci_bus,
            p.pci_device, p.pci_function);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_ZE
    auto eng = dnnl::engine(get_test_engine(), true);
    ze_pci_ext_properties_t p {};
    p.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
    if (zeDevicePciGetPropertiesExt(dnnl::ze_interop::get_device(eng), &p)
            != ZE_RESULT_SUCCESS)
        return {};
    snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", p.address.domain,
            p.address.bus, p.address.device, p.address.function);
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto dev = dnnl::sycl_interop::get_device(eng);
    if (!dev.has(::sycl::aspect::ext_intel_pci_address)) return {};
    return dev.get_info<::sycl::ext::intel::info::device::pci_address>();
#endif
    return bdf;
}

// An unpinned GPU frequency (min != max) may float during the run and skew
// performance measurements. The sysfs layout is probed (xe, then i915) because
// the driver name alone doesn't determine which frequency-control files exist.
static bool report_gpu_perf_hints_impl() {
    bool any_hint = false;

    const std::string bdf = get_gpu_pci_bdf();
    const std::string drm_root = "/sys/class/drm";
    for (const auto &card : list_dir(drm_root)) {
        // A `-` marks a connector node (e.g. `card0-HDMI-A-1`), not a device.
        if (card.rfind("card", 0) != 0 || card.find('-') != std::string::npos)
            continue;
        const std::string card_path
                = parser::utils::str_join(drm_root, card, '/');
        // Only Intel GPUs expose the frequency controls inspected below.
        if (read_sysfs_str(card_path + "/device/vendor") != "0x8086") continue;
        // Skip cards other than the test engine's one; `device` links to the
        // PCI node named by its BDF.
        char real[PATH_MAX];
        if (!bdf.empty() && realpath((card_path + "/device").c_str(), real)) {
            const std::string node = real;
            if (node.substr(node.rfind('/') + 1) != bdf) continue;
        }

        auto gts = find_gpu_gts_xe(card_path);
        if (gts.empty()) gts = find_gpu_gts_i915(card_path);

        if (gts.empty()) {
            BENCHDNN_PRINT(0,
                    "[PERF_HINTS] Warning: GPU device \"%s\" was not "
                    "identified. The device search logic needs adjustment.\n",
                    card.c_str());
            continue;
        }

        for (const auto &gt : gts) {
            int64_t min_freq = 0, max_freq = 0;
            if (read_sysfs_int(gt.min_freq_path, min_freq) != OK
                    || read_sysfs_int(gt.max_freq_path, max_freq) != OK)
                continue;
            if (min_freq != max_freq) {
                BENCHDNN_PRINT(0,
                        "[PERF_HINTS] Info: GPU %s %s min (%ld MHz) and max "
                        "(%ld MHz) frequencies differ. Pin the frequency for "
                        "stable performance measurements.\n",
                        card.c_str(), gt.name.c_str(), (long)min_freq,
                        (long)max_freq);
                any_hint = true;
            }
        }
    }

    return any_hint;
}
#endif

void report_perf_hints() {
    if (!summary.perf_hints) return;

    // Report at most once per run.
    static bool reported = false;
    if (reported) return;
    reported = true;

#ifdef __linux__
    // Diagnostics are engine-specific and kept separate so that other CPU
    // architectures (e.g. AArch64, RISC-V) or GPU vendors can extend them
    // independently.
    bool any_hint = false;
    if (is_cpu())
        any_hint = report_cpu_perf_hints_impl();
    else if (is_gpu())
        any_hint = report_gpu_perf_hints_impl();

    if (any_hint) {
        BENCHDNN_PRINT(0, "%s",
                "[PERF_HINTS] Use `--summary=no-perf-hints` to disable these "
                "messages.\n");
    }
#endif
}
