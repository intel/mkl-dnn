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

#include "perf_lock.hpp"

#include <cerrno>
#include <cstdio>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
#include <sys/stat.h>
#endif

namespace {

#if defined(_WIN32)
constexpr const char *filename = "C:\\ProgramData\\onednn_perf.lock";
#else
constexpr const char *filename = "/tmp/onednn_perf.lock";
#endif

struct perf_lock_t {
    bool acquire();
    void release();

private:
    bool fail(const char *action) {
        failed_ = true;
        std::fprintf(stderr,
                "Warning: couldn't %s the performance lock. Continuing "
                "without it; GPU kernels may run unsynchronized with other "
                "processes.\n",
                action);
        return false;
    }

    int depth_ = 0;
    bool failed_ = false;
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

perf_lock_t &state() {
    thread_local perf_lock_t s;
    return s;
}

#if defined(_WIN32)

bool perf_lock_t::acquire() {
    if (failed_) return false;
    if (depth_ > 0) {
        depth_++;
        return true;
    }

    HANDLE h = CreateFileA(filename, GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return fail("open");

    OVERLAPPED overlapped = {};
    // Blocking, exclusive lock over the whole file.
    if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD,
                &overlapped)) {
        CloseHandle(h);
        return fail("acquire");
    }
    handle_ = h;
    depth_ = 1;
    return true;
}

void perf_lock_t::release() {
    if (--depth_ > 0) return;

    OVERLAPPED overlapped = {};
    UnlockFileEx(handle_, 0, MAXDWORD, MAXDWORD, &overlapped);
    CloseHandle(handle_);
    handle_ = INVALID_HANDLE_VALUE;
}

#else

bool perf_lock_t::acquire() {
    if (failed_) return false;
    if (depth_ > 0) {
        depth_++;
        return true;
    }

    int fd = open(filename, O_RDONLY | O_CLOEXEC);
    if (fd < 0 && errno == ENOENT) {
        fd = open(filename, O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0666);
        // `umask` may have stripped permissions needed by other users.
        if (fd >= 0)
            (void)fchmod(fd, 0666);
        else if (errno == EEXIST)
            fd = open(filename, O_RDONLY | O_CLOEXEC);
    }
    if (fd < 0) return fail("open");

    // Blocking, exclusive lock, released automatically once the last
    // descriptor referring to it (held by this or any other process) is
    // closed, including on abnormal process termination.
    if (flock(fd, LOCK_EX) != 0) {
        close(fd);
        return fail("acquire");
    }
    fd_ = fd;
    depth_ = 1;
    return true;
}

void perf_lock_t::release() {
    if (--depth_ > 0) return;

    flock(fd_, LOCK_UN);
    close(fd_);
    fd_ = -1;
}

#endif
} // namespace

perf_lock_guard_t::perf_lock_guard_t(bool enabled)
    : active_(enabled && state().acquire()) {}

perf_lock_guard_t::~perf_lock_guard_t() {
    if (active_) state().release();
}
