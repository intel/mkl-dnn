# *******************************************************************************
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# Sets up the environment to build and run applications with this oneDNN
# package. Source it from bash or zsh:
#
#     source <package>/env/vars.sh

if [ -n "${BASH_VERSION:-}" ]; then
    _dnnl_vars_script="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_dnnl_vars_script="${(%):-%x}"'
else
    echo "vars.sh: only bash and zsh are supported" >&2
    return 1
fi

DNNLROOT="$(cd "$(dirname "$_dnnl_vars_script")/.." && pwd)"
export DNNLROOT
unset _dnnl_vars_script

export CMAKE_PREFIX_PATH="$DNNLROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$DNNLROOT/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$DNNLROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$DNNLROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
