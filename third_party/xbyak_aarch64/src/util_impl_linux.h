/*******************************************************************************
 * Copyright 2020-2023 FUJITSU LIMITED
 * Copyright 2025-2026 Arm Ltd. and affiliates
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
#ifndef __linux__
#error "Something wrong"
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dirent.h>
#include <regex.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_util.h"

/* In old Linux such as Ubuntu 16.04, HWCAP_ATOMICS, HWCAP_FP, HWCAP_ASIMD
   can not be found in <bits/hwcap.h> which is included from <sys/auxv.h>.
   Xbyak_aarch64 uses <asm/hwcap.h> as an alternative.
 */
#ifndef HWCAP_FP
#include <asm/hwcap.h>
#endif

/* Linux kernel used in Ubuntu 20.04 does not have HWCAP2_BF16 definition. */
#ifdef AT_HWCAP2
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1UL << 14)
#endif
#ifndef HWCAP2_SVEI8MM
#define HWCAP2_SVEI8MM (1UL << 9)
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL << 23)
#endif
#ifndef HWCAP2_SME_I16I64
#define HWCAP2_SME_I16I64	(1UL << 24)
#endif
#ifndef HWCAP2_SME_F64F64
#define HWCAP2_SME_F64F64 (1UL << 25)
#endif
#ifndef HWCAP2_SME_F16F16
#define HWCAP2_SME_F16F16 (1UL << 42)
#endif
#endif

namespace Xbyak_aarch64 {
namespace util {
#define XBYAK_AARCH64_ERROR_ fprintf(stderr, "%s, %d, Error occured while reading cache infomation.\n", __FILE__, __LINE__);
#define XBYAK_AARCH64_PATH_NODES "/sys/devices/system/node/node"
#define XBYAK_AARCH64_PATH_CORES "/sys/devices/system/node/node0/cpu"
#define XBYAK_AARCH64_PATH_CACHE_DIR "/sys/devices/system/cpu/cpu0/cache"
#define XBYAK_AARCH64_PATH_CACHE_LEVEL "/sys/devices/system/cpu/cpu0/cache/index0/level"
#define XBYAK_AARCH64_PATH_CACHE_SIZE "/sys/devices/system/cpu/cpu0/cache/index0/size"
#define XBYAK_AARCH64_PATH_CACHE_TYPE "/sys/devices/system/cpu/cpu0/cache/index0/type"
#define XBYAK_AARCH64_PATH_CACHE_LIST "/sys/devices/system/cpu/cpu0/cache/index0/shared_cpu_list"
#define XBYAK_AARCH64_MIDR_EL1(I, V, A, P, R) ((I << 24) | (V << 20) | (A << 16) | (P << 4) | (R << 0))
#define XBYAK_AARCH64_READ_SYSREG(var, ID) asm("mrs %0, " #ID : "=r"(var));

class CpuInfoLinux : public CpuInfo {
public:
  CpuInfoLinux() {
    init();
    setSysRegVal(); // Read MIDR_EL1 before setCacheHierarchy().
    setNumCores();
    setHwCap();
    setCacheHierarchy();
    setImplementer();
    setSmeLen();
  }

private:
  static constexpr int max_path_len = 1024;
  static constexpr int buf_size = 1024;
  const char *path_midr_el1 = "/sys/devices/system/cpu/cpu0/regs/identification/midr_el1";
  const struct cacheInfo_v2_t cacheInfoDict[2] = {{/* A64FX */ XBYAK_AARCH64_MIDR_EL1(0x46, 0x1, 0xf, 0x1, 0x0),
                                                   {{/* L1 */ SeparateCache, {1024 * 64, 1024 * 64, 0}, {1, 1, 0}},
                                                    {/* L2 */ UnifiedCache, {0, 0, 1024 * 1024 * 8}, {0, 0, 12}},
                                                    {/* L3 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L4 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L5 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L6 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L7 */ NoCache, {0, 0, 0}, {0, 0, 0}}}},
                                                  {/* A64FX */ XBYAK_AARCH64_MIDR_EL1(0x46, 0x2, 0xf, 0x1, 0x0),
                                                   {{/* L1 */ SeparateCache, {1024 * 64, 1024 * 64, 0}, {1, 1, 0}},
                                                    {/* L2 */ UnifiedCache, {0, 0, 1024 * 1024 * 8}, {0, 0, 12}},
                                                    {/* L3 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L4 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L5 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L6 */ NoCache, {0, 0, 0}, {0, 0, 0}},
                                                    {/* L7 */ NoCache, {0, 0, 0}, {0, 0, 0}}}}};

  int getFilePathMaxTailNumPlus1(const char *path) {
    char dir_path[max_path_len];
    char file_pattern[max_path_len];
    int retVal = 0;

    getRegEx(dir_path, path, "/([^/]+/)+");
    /* Remove last '/'. */
    dir_path[strlen(dir_path) - 1] = '\0';
    getRegEx(file_pattern, path, "[^/]+$");
    strncat(file_pattern, "[0-9]+", 16);

    DIR *dir = opendir(dir_path);
    if (dir == NULL)
      return retVal;
    struct dirent *dp;

    dp = readdir(dir);
    while (dp != NULL) {
      if (getRegEx(dir_path, dp->d_name, file_pattern) == 0)
        retVal++;
      dp = readdir(dir);
    }

    if (dir != NULL)
      closedir(dir);

    return retVal;
  }

  void getLineInFile(char *buf, const char *path, const int num) {
    auto chomp = [](char *ptr, const int num) {
      for (int i = 0; i < num; i++) {
        if ('\n' == *(ptr + i))
          *(ptr + i) = '\0';
        else if ('\0' == *(ptr + i))
          break;
      }
    };

    FILE *fp = fopen(path, "r");
    if (!(fp && fread(buf, sizeof(char), num, fp)))
      buf[0] = '\0';

    chomp(buf, buf_size);

    if (fp) {
      fclose(fp);
    }
  }

  /**
   * Return directory path
   * @param[in] path ex. /sys/devices/system/node/node
   * @param[out] buf ex. /sys/devices/system/node
   */
  int getRegEx(char *buf, const char *path, const char *regex) {
    regex_t regexBuf;
    regmatch_t match[1];

    if (regcomp(&regexBuf, regex, REG_EXTENDED) != 0)
      throw ERR_INTERNAL;

    const int retVal = regexec(&regexBuf, path, 1, match, 0);
    regfree(&regexBuf);

    if (retVal != 0)
      return -1;

    const int startIdx = match[0].rm_so;
    const int endIdx = match[0].rm_eo;

    /* Something wrong (multiple match or not match) */
    if (startIdx == -1 || endIdx == -1 || (endIdx - startIdx - 1) < 1)
      return -1;

    strncpy(buf, path + startIdx, endIdx - startIdx);
    buf[endIdx - startIdx] = '\0';

    return 0;
  }

  /* Read the following files and set cacheInfo_.
     If an error occurs, halfway result are cleared and return false.
     /sys/devices/system/cpu/cpu0/cache/index[0-9]+/level "1", "2", ...
     /sys/devices/system/cpu/cpu0/cache/index[0-9]+/size  "32K", "1M"
     /sys/devices/system/cpu/cpu0/cache/index[0-9]+/type  "Instruction", "Data", "Unified"
     /sys/devices/system/cpu/cpu0/cache/index[0-9]+/shared_cpu_list "0", "0-1"
  */
  bool readCacheInfoFromSysDevice() {
    char buf0[buf_size] = {0};
    char buf1[buf_size] = {0};
    char buf2[buf_size] = {0};

    struct dirent *dp;
    DIR *dir = opendir(XBYAK_AARCH64_PATH_CACHE_DIR);
    if (dir == NULL)
      goto init_and_return_false;

    dp = readdir(dir);
    while (dp != NULL) {
      regex_t regexBuf;
      regmatch_t match[2];

      if (regcomp(&regexBuf, "index[0-9]*$", REG_EXTENDED) != 0)
        throw ERR_INTERNAL;

      if (regexec(&regexBuf, dp->d_name, 1, match, 0) == 0) { // Found index[1-9][0-9]. directory
        char *dir_name = buf0;
        char *file_name = buf1;
        char *buf = buf2;
        char *end_ptr;
        strncpy(dir_name, XBYAK_AARCH64_PATH_CACHE_DIR, buf_size);
        strncat(dir_name, "/", 2);
        strncat(dir_name, dp->d_name + match[0].rm_so, match[0].rm_eo - match[0].rm_so);
        strncat(dir_name, "/", 2);

        // Get cache level
        strncpy(file_name, dir_name, buf_size);
        strncat(file_name, "level", buf_size);
        getLineInFile(buf, file_name, buf_size);
        const long int level = strtol(buf, &end_ptr, 10);
        if ('\0' != *end_ptr) { // Non-numeric characters exist.
          XBYAK_AARCH64_ERROR_;
          goto init_and_return_false;
        }

        // Get cache size
        strncpy(file_name, dir_name, buf_size);
        strncat(file_name, "size", buf_size);
        getLineInFile(buf, file_name, buf_size);
        long int size = strtol(buf, &end_ptr, 10);
        if ('\0' != *end_ptr) {
          if (strncmp(end_ptr, "K", 2) == 0) {
            size = size * 1024;
          } else if (strncmp(end_ptr, "M", 2) == 0) {
            size = size * 1024 * 1024;
          } else {
            XBYAK_AARCH64_ERROR_;
            goto init_and_return_false;
          }
        }

        // Get cache type
        Arm64CacheType type;
        strncpy(file_name, dir_name, buf_size);
        strncat(file_name, "type", buf_size);
        getLineInFile(buf, file_name, buf_size);
        if (strncmp(buf, "Instruction", buf_size) == 0) {
          type = InstCacheOnly;
        } else if (strncmp(buf, "Data", buf_size) == 0) {
          type = DataCacheOnly;
        } else if (strncmp(buf, "Unified", buf_size) == 0) {
          type = UnifiedCache;
        } else { // Unconsidered text exists.
          XBYAK_AARCH64_ERROR_;
          goto init_and_return_false;
        }

        /* Get cache-sharing cpu list
           Example0: "0"
           Example1: "0-7"
           Example2: "0,64"
           Example3: "0-31,64-95" */
        long int start, end;
        int sharing_cores = 0;
        strncpy(file_name, dir_name, buf_size);
        strncat(file_name, "shared_cpu_list", buf_size);
        getLineInFile(buf, file_name, buf_size);
        /* Debug:
           strncpy(buf, "0", buf_size);
           strncpy(buf, "4-8", buf_size);
           strncpy(buf, "2,34,111", buf_size);
           strncpy(buf, "12-23,48-60", buf_size);
        */
        end_ptr = buf;
        while ('\0' != *buf) {
          start = strtol(buf, &end_ptr, 10);
          if ('\0' == *end_ptr) {
            sharing_cores += 1;
            // No more core exists.
            break;
          } else if ('-' == *end_ptr) {
            buf = end_ptr + 1;
            end = strtol(buf, &end_ptr, 10);
            sharing_cores += end - start + 1;
            buf = end_ptr;
            while (',' == *buf || ' ' == *buf)
              buf++;
          } else if (',' == *end_ptr) {
            buf = end_ptr + 1;
            sharing_cores += 1;
          } else {
            XBYAK_AARCH64_ERROR_;
            goto init_and_return_false;
          }
        }

        auto cache = &cacheInfo_.levelCache[level - 1];

        switch (type) {
        case UnifiedCache:
          cache->type = UnifiedCache;
          cache->size[2] = size;
          cache->sharingCores[2] = sharing_cores;
          break;
        case InstCacheOnly:
          cache->type = cache->type == DataCacheOnly ? SeparateCache : InstCacheOnly;
          cache->size[0] = size;
          cache->sharingCores[0] = sharing_cores;
          break;
        case DataCacheOnly:
          cache->type = cache->type == InstCacheOnly ? SeparateCache : DataCacheOnly;
          cache->size[1] = size;
          cache->sharingCores[1] = sharing_cores;
          break;
        default:
          XBYAK_AARCH64_ERROR_;
          goto init_and_return_false;
        }
      }

      regfree(&regexBuf);
      dp = readdir(dir); // Try next
    }

    if (dir != NULL)
      closedir(dir);

    setLastDataCacheLevel();
    return true;

  init_and_return_false:
    init(); // Clear halfway result
    return false;
  }

  void setCacheHierarchy() {
    /* Cache size of AArch64 CPUs are described in the system registers,
       which can't be read from user-space applications.
       Linux provides `sysconf` API and `/sys/devices/system/cpu/`
       device files to get cache size, but they dosen't always return
       correct values. It may depend on Linux kernel version and
       support status of CPUs. To avoid this situation, cahche size is
       firstly read from `cacheInfoDict`, secondly get by `sysconf`.

       `sysconf` example
       #include <unistd.h>
       int main() {
         reutrn sysconf(_SC_LEVEL1_DCACHE_SIZE);
       }
     */
    const cacheInfo_v2_t *c = nullptr;
    const uint64_t midr_el1 = cacheInfo_.midr_el1;

    for (size_t j = 0; j < sizeof(cacheInfoDict) / sizeof(cacheInfo_v2_t); j++) {
      if (cacheInfoDict[j].midr_el1 == midr_el1) {
        c = cacheInfoDict + j;
        break;
      }
    }

    if (c != nullptr) {
      for (size_t i = 0; i < maxCacheLevel; i++) {
        auto dict = &c->levelCache[i];
        auto cache = &cacheInfo_.levelCache[i];
        cache->type = dict->type;

        switch (dict->type) {
        case InstCacheOnly:
          cache->size[0] = dict->size[0];
          cache->sharingCores[0] = dict->sharingCores[0];
          break;
        case DataCacheOnly:
          cache->size[1] = dict->size[1];
          cache->sharingCores[1] = dict->sharingCores[1];
          break;
        case SeparateCache:
          cache->size[0] = dict->size[0];
          cache->size[1] = dict->size[1];
          cache->sharingCores[0] = dict->sharingCores[0];
          cache->sharingCores[1] = dict->sharingCores[1];
          break;
        case UnifiedCache:
          cache->size[2] = dict->size[2];
          cache->sharingCores[2] = dict->sharingCores[2];
          break;
        default:
          // Do nothing
          break;
        }
        lastDataCacheLevel_ = (dict->size[1] || dict->size[2]) ? i + 1 : lastDataCacheLevel_;
      }
    } else if (!readCacheInfoFromSysDevice()) {
      /**
       * @ToDo Get chache information by `sysconf`
       * for the case thd dictionary is unavailable.
       */
      lastDataCacheLevel_ = 2; // It is assumed L1 and L2 cache exist.

      cacheInfo_.levelCache[0].size[0] = sysconf(_SC_LEVEL1_ICACHE_SIZE); // L1, ICache
      cacheInfo_.levelCache[0].size[1] = sysconf(_SC_LEVEL1_DCACHE_SIZE); // L1, DCache
      cacheInfo_.levelCache[1].size[2] = sysconf(_SC_LEVEL2_CACHE_SIZE);  // L2, UCache
      cacheInfo_.levelCache[2].size[2] = sysconf(_SC_LEVEL3_CACHE_SIZE);  // L3, UCache
    }
  }

  void setHwCap() {
    const unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ATOMICS)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_ATOMIC;

    if (hwcap & HWCAP_FP)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_FP;
    if (hwcap & HWCAP_FPHP)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_FPHP;
    if (hwcap & HWCAP_ASIMDFHM)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_FHM;
    if (hwcap & HWCAP_ASIMD)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_ADVSIMD;
    if (hwcap & HWCAP_CRC32)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_CRC;
    if (hwcap & HWCAP_JSCVT)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_JSCVT;

#ifdef AT_HWCAP2
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & HWCAP2_BF16)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_BF16;
    if (hwcap2 & HWCAP2_SVEI8MM)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SVEI8MM;
    if (hwcap2 & HWCAP2_SME)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME;
    if (hwcap2 & HWCAP2_SME_I16I64)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_I16I64;
    if (hwcap2 & HWCAP2_SME_F16F16)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_F16F16;
    if (hwcap2 & HWCAP2_SME_F64F64)
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SME_F64F64;
#endif

#ifdef HWCAP_SVE
    /* Some old <sys/auxv.h> may not define HWCAP_SVE.
       In that case, SVE is treated as if it were not supported. */
    if (hwcap & HWCAP_SVE) {
      type_ |= (Type)XBYAK_AARCH64_HWCAP_SVE;
      // svcntb(); if arm_sve.h is available
      sveLen_ = (sveLen_t)prctl(51); // PR_SVE_GET_VL
    }
#endif
  }

  void setNumCores() {
    /**
     * @ToDo There are some methods to get # of cores.
     Considering various kernel versions and CPUs, a combination of
     multiple methods may be required.
     1) sysconf(_SC_NPROCESSORS_ONLN)
     2) /sys/devices/system/cpu/online
     3) std::thread::hardware_concurrency()
    */
    numCores_[0] = numCores_[1] = sysconf(_SC_NPROCESSORS_ONLN);
  }

  void setSysRegVal() {
    /* Some Linux does not permit to access MIDR_EL1 register
       from user program. The register value is disclosed
       in /sys/devices/system/cpu/cpu?/regs/identification/midr_el1 */
    char buf[64];
    char *stop;

    FILE *file = fopen(path_midr_el1, "r");
    if (file == nullptr) {
      // sysfs is unaccessible for AWS Lambdas
      fprintf(stderr, "%s, %d: Can't open MIDR_EL1 sysfs entry\n", __FILE__, __LINE__);
      cacheInfo_.midr_el1 = 0xFF << 24;
      return;
    }

    if (fread(buf, sizeof(char), 64, file) == 0) {
      // File can be empty if invoked inside docker container
      fprintf(stderr, "%s, %d: Can't read MIDR_EL1 sysfs entry\n", __FILE__, __LINE__);
      cacheInfo_.midr_el1 = 0xFF << 24;
      return;
    }

    cacheInfo_.midr_el1 = strtoull(buf, &stop, 16);

    fclose(file);
  }
};

#undef XBYAK_AARCH64_ERROR_
} // namespace util
} // namespace Xbyak_aarch64
