/*******************************************************************************
 * Copyright 2022-2023 FUJITSU LIMITED
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
#ifndef _M_ARM64
#error "Something wrong"
#endif

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_util.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <malloc.h>
#include <windows.h>

#ifndef PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE (68)
#endif

namespace Xbyak_aarch64 {
namespace util {

class CpuInfoWindows : public CpuInfo {
public:
  CpuInfoWindows() {
    init();
    setHwCap();
    setCacheHierarchy();
    setImplementer();
    setSmeLen();
  }

private:
  void setCacheHierarchy() {
    DWORD bufSize = 0;
    GetLogicalProcessorInformation(NULL, &bufSize);
    auto *ptr = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)_alloca(bufSize);
    if (GetLogicalProcessorInformation(ptr, &bufSize) == FALSE)
      return;

    DWORD offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= bufSize) {
      switch (ptr->Relationship) {
      case RelationProcessorCore:
        numCores_[1]++;
        break;

      case RelationCache: {
        const auto cache = &ptr->Cache;
        auto levelCache = &cacheInfo_.levelCache[cache->Level - 1];
        ULONG_PTR mask = ptr->ProcessorMask;
        int count = 0;
        while (mask) {
          count += (mask & 0x1) ? 1 : 0;
          mask = mask >> 1;
        }

        switch (cache->Type) {
        case CacheUnified:
          levelCache->type = UnifiedCache;
          levelCache->size[2] = cache->Size;
          levelCache->sharingCores[2] = count;
          break;
        case CacheInstruction:
          levelCache->type = levelCache->type == DataCacheOnly ? SeparateCache : InstCacheOnly;
          levelCache->size[0] = cache->Size;
          levelCache->sharingCores[0] = count;
          break;
        case CacheData:
          levelCache->type = levelCache->type == InstCacheOnly ? SeparateCache : DataCacheOnly;
          levelCache->size[1] = cache->Size;
          levelCache->sharingCores[1] = count;
          break;
        default:
          break;
        }
      }
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }
    numCores_[0] = numCores_[1];

    setLastDataCacheLevel();
  }

  void setHwCap() {
    if (IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_ADVSIMD;
    if (IsProcessorFeaturePresent(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_ATOMIC;
    if (IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_CRC;
    if (IsProcessorFeaturePresent(PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_JSCVT;
    if (IsProcessorFeaturePresent(PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_BF16;
#ifdef PF_ARM_SME_INSTRUCTIONS_AVAILABLE
    if (IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME;
#endif
#ifdef PF_ARM_SME_I16I64_INSTRUCTIONS_AVAILABLE
    if (IsProcessorFeaturePresent(PF_ARM_SME_I16I64_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_I16I64;
#endif
#ifdef PF_ARM_SME_F16F16_INSTRUCTIONS_AVAILABLE
    if (IsProcessorFeaturePresent(PF_ARM_SME_F16F16_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_F16F16;
#endif
#ifdef PF_ARM_SME_F64F64_INSTRUCTIONS_AVAILABLE
    if (IsProcessorFeaturePresent(PF_ARM_SME_F64F64_INSTRUCTIONS_AVAILABLE))
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_F64F64;
#endif
  }
};

} // namespace util
} // namespace Xbyak_aarch64
