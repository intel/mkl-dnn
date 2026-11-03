#pragma once
/*******************************************************************************
 * Copyright 2020-2023 FUJITSU LIMITED
 * Copyright 2026 Arm Ltd. and affiliates
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
#ifndef XBYAK_AARCH64_UTIL_H_
#define XBYAK_AARCH64_UTIL_H_

#if !(defined(__linux__) || defined(__APPLE__) || defined(_M_ARM64))
#error "Unsupported OS"
#endif

#include <stdint.h>
#include <memory>

namespace Xbyak_aarch64 {
namespace util {
typedef uint64_t Type;

constexpr uint32_t maxCacheLevel = 7; // Specification of Armv9
constexpr uint32_t maxTopologyLevel = 2;
constexpr uint32_t max_path_len = 1024;

enum Arm64CpuTopologyLevel { SmtLevel = 1, CoreLevel = 2 };
enum Arm64CacheType { NoCache = 0, InstCacheOnly = 1, DataCacheOnly = 2, SeparateCache = 3, UnifiedCache = 4, OtherCache = 5 };
enum Arm64CacheLevel { L1 = 1, L2, L3, L4, L5, L6, L7 };
enum cacheType_t { ICache = 0, DCache = 1, UCache = 2 };

enum sveLen_t {
  SVE_NONE = 0,
  SVE_128 = 16 * 1,
  SVE_256 = 16 * 2,
  SVE_384 = 16 * 3,
  SVE_512 = 16 * 4,
  SVE_640 = 16 * 5,
  SVE_768 = 16 * 6,
  SVE_896 = 16 * 7,
  SVE_1024 = 16 * 8,
  SVE_1152 = 16 * 9,
  SVE_1280 = 16 * 10,
  SVE_1408 = 16 * 11,
  SVE_1536 = 16 * 12,
  SVE_1664 = 16 * 13,
  SVE_1792 = 16 * 14,
  SVE_1920 = 16 * 15,
  SVE_2048 = 16 * 16,
};

enum hwCap_t {
  XBYAK_AARCH64_HWCAP_NONE = 0,
  XBYAK_AARCH64_HWCAP_ADVSIMD = 1 << 1,
  XBYAK_AARCH64_HWCAP_FP = 1 << 2,
  XBYAK_AARCH64_HWCAP_SVE = 1 << 3,
  XBYAK_AARCH64_HWCAP_ATOMIC = 1 << 4,
  XBYAK_AARCH64_HWCAP_BF16 = 1 << 5,
  XBYAK_AARCH64_HWCAP_CRC = 1 << 6,
  XBYAK_AARCH64_HWCAP_JSCVT = 1 << 7,
  XBYAK_AARCH64_HWCAP_SME = 1 << 8,
  XBYAK_AARCH64_HWCAP_SME_I16I64 = 1 << 9,
  XBYAK_AARCH64_HWCAP_SME_F16F16 = 1 << 10,
  XBYAK_AARCH64_HWCAP_SME_F64F64 = 1 << 11,
  XBYAK_AARCH64_HWCAP_FPHP = 1 << 12,
  XBYAK_AARCH64_HWCAP_FHM = 1 << 13,
  XBYAK_AARCH64_HWCAP_SVEI8MM = 1 << 14,
};

struct implementer_t {
  uint32_t id;
  const char *implementer;
};

/**
   CPU detection class
*/
class CpuInfo;
class Cpu {
private:
  std::unique_ptr<CpuInfo> info;

public:
  Cpu();
  ~Cpu();

  void dumpCacheInfo() const;
  Arm64CacheType getCacheType(const Arm64CacheLevel i) const;
  uint32_t getCoresSharingDataCache(const Arm64CacheLevel i) const;
  uint32_t getDataCacheSize(const Arm64CacheLevel i) const;
  const char *getImplementer() const;
  uint32_t getLastDataCacheLevel() const;
  uint32_t getNumCores(Arm64CpuTopologyLevel level) const;
  Type getType() const;
  uint64_t getSveLen() const;
  uint64_t getSmeLen() const;
  bool has(Type type) const;
  bool isAtomicSupported() const;
  bool isBf16Supported() const;
  bool isF16Supported() const;
  bool isFhmSupported() const;
};

} // namespace util
} // namespace Xbyak_aarch64
#endif
