#===============================================================================
# Copyright 2026 Intel Corporation
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

include(CTest)

set(GENERATOR_TESTS
    test_dsl_types
    test_dsl_tensor
    test_gpu_binary
)

foreach(TEST ${GENERATOR_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} gemmstone_dsl)
    target_link_libraries(${TEST} generator)
    target_link_libraries(${TEST} generator_dsl)
    if(OpenCL_FOUND)
      target_link_libraries(${TEST} OpenCL::OpenCL)
    endif()
    if(LevelZero_FOUND)
      target_link_libraries(${TEST} LevelZero::LevelZero)
    endif()
    add_test(${TEST} ${TEST})
endforeach()

# libocloc is loaded at runtime.
target_link_libraries(test_gpu_binary ${CMAKE_DL_LIBS})
