#===============================================================================
# Copyright 2022 Intel Corporation
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
#===============================================================================

# workaround for icx cannot init constexpr with dllimport symbols(detail::handle)
# also, suppose it's unnecessary for users to link the library with the flag.
if (DNNL_LIBRARY_TYPE STREQUAL "SHARED")
    remove_definitions_with_host_compiler(-DDNNL_DLL)
endif()

include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(TEST_ALLOCATOR ${CMAKE_CURRENT_SOURCE_DIR}/test_allocator.cpp)

add_subdirectory(api)
add_subdirectory(unit)
