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

#include "dsl/utils/logging.hpp"

#ifdef __unix__
#include <unistd.h>
#endif

#include <climits>
#include <cstring>

GEMMSTONE_NAMESPACE_START
#ifdef STANDALONE

#ifdef __unix__
namespace {
int getEnvInt(const char *name, int default_value) {
    const char *env_value = ::getenv(name);
    if (env_value) return atoi(env_value);
    return default_value;
}
} // namespace

int getVerbose(GEMMVerbose v) {
    switch (v) {
        case GEMMVerbose::DebugInfo: {
            static int debug_level = getEnvInt("GEMMSTONE_DEBUG", 0);
            return debug_level;
        }
    }
    return 0;
}
#else
int getVerbose(GEMMVerbose v) {
    return 0;
}
#endif

#endif
GEMMSTONE_NAMESPACE_END
