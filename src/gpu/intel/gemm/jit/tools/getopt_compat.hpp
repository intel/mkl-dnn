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

#ifndef GEMMSTONE_TOOLS_GETOPT_COMPAT_HPP
#define GEMMSTONE_TOOLS_GETOPT_COMPAT_HPP

// Cross-platform drop-in replacement for <getopt.h>.
// On POSIX systems, defers to the system header.
// On Windows (MSVC), provides a self-contained implementation of
//   getopt_long, optarg, optind, opterr, and struct option.

#ifndef _WIN32

#include <getopt.h>

#else // _WIN32

#include <cstdio>
#include <cstring>

#define no_argument       0
#define required_argument 1
#define optional_argument 2

struct option {
    const char *name;
    int         has_arg;
    int        *flag;
    int         val;
};

inline const char *optarg = nullptr;
inline int         optind = 1;
inline int         opterr = 1;
inline int         optopt = 0;

// Internal state for option parsing.
namespace getopt_detail {
    inline int    pos = 0;    // position within a short-option cluster
}

inline int getopt_long(int argc, char *const argv[], const char *optstring,
                       const struct option *longopts, int *longindex)
{
    // Reset internal state when optind is reset to 1.
    if (optind <= 1) {
        getopt_detail::pos = 0;
        if (optind < 1) optind = 1;
    }

    optarg = nullptr;

    while (optind < argc) {
        const char *arg = argv[optind];

        // If we're not in the middle of a short-option cluster:
        if (getopt_detail::pos == 0) {
            // Stop at first non-option or "--".
            if (arg[0] != '-' || arg[1] == '\0')
                return -1;
            if (arg[1] == '-' && arg[2] == '\0') {
                optind++;
                return -1;
            }

            // Try long option.
            if (arg[1] == '-') {
                const char *name = arg + 2;
                const char *eq = std::strchr(name, '=');
                size_t namelen = eq ? (size_t)(eq - name) : std::strlen(name);

                for (int i = 0; longopts && longopts[i].name; i++) {
                    if (std::strncmp(longopts[i].name, name, namelen) == 0
                        && longopts[i].name[namelen] == '\0') {
                        // Found match.
                        if (longindex) *longindex = i;
                        optind++;

                        if (longopts[i].has_arg == required_argument) {
                            if (eq) {
                                optarg = eq + 1;
                            } else if (optind < argc) {
                                optarg = argv[optind++];
                            } else {
                                if (opterr)
                                    fprintf(stderr, "%s: option '--%s' requires an argument\n",
                                            argv[0], longopts[i].name);
                                optopt = longopts[i].val;
                                return '?';
                            }
                        } else if (longopts[i].has_arg == optional_argument) {
                            if (eq) optarg = eq + 1;
                        } else {
                            if (eq) {
                                if (opterr)
                                    fprintf(stderr, "%s: option '--%s' doesn't allow an argument\n",
                                            argv[0], longopts[i].name);
                                optopt = longopts[i].val;
                                return '?';
                            }
                        }

                        if (longopts[i].flag) {
                            *longopts[i].flag = longopts[i].val;
                            return 0;
                        }
                        return longopts[i].val;
                    }
                }

                // Unknown long option.
                if (opterr)
                    fprintf(stderr, "%s: unrecognized option '%s'\n", argv[0], arg);
                optind++;
                return '?';
            }

            // Short option: start at position 1 (after '-').
            getopt_detail::pos = 1;
        }

        // Process short option at current position.
        char c = arg[getopt_detail::pos];
        const char *spec = std::strchr(optstring, c);

        if (!spec || c == ':') {
            optopt = c;
            if (opterr)
                fprintf(stderr, "%s: invalid option -- '%c'\n", argv[0], c);
            getopt_detail::pos++;
            if (arg[getopt_detail::pos] == '\0') {
                optind++;
                getopt_detail::pos = 0;
            }
            return '?';
        }

        getopt_detail::pos++;
        if (spec[1] == ':') {
            if (arg[getopt_detail::pos] != '\0') {
                // Rest of current argv element is the argument.
                optarg = arg + getopt_detail::pos;
                optind++;
            } else if (spec[2] != ':') {
                // Required argument: next argv element is the argument.
                optind++;
                if (optind < argc) {
                    optarg = argv[optind++];
                } else {
                    if (opterr)
                        fprintf(stderr, "%s: option requires an argument -- '%c'\n", argv[0], c);
                    optopt = c;
                    getopt_detail::pos = 0;
                    return (optstring[0] == ':') ? ':' : '?';
                }
            } else {
                // Optional argument (::) with no inline value: do not
                // consume the next argv element.
                optind++;
            }
            getopt_detail::pos = 0;
        } else {
            // No argument needed.
            if (arg[getopt_detail::pos] == '\0') {
                optind++;
                getopt_detail::pos = 0;
            }
        }

        return c;
    }

    return -1;
}

#endif // _WIN32

#endif // GEMMSTONE_TOOLS_GETOPT_COMPAT_HPP
