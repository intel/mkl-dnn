#! /bin/bash

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

# Installs Intel oneAPI DPC++/C++ Compiler from the Intel YUM repository.
#
# Usage: install_dpcpp.sh

set -o errexit -o pipefail -o nounset

# Users of the SYCL packages need the same or a newer version of the compiler
# runtime. Keep in sync with install_dpcpp.bat.
VERSION="2026.0"

cat > /etc/yum.repos.d/oneAPI.repo << 'EOF'
[oneAPI]
name=Intel oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
EOF

dnf install -y "intel-oneapi-compiler-dpcpp-cpp-${VERSION}"
