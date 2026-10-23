@echo off

rem *******************************************************************************
rem Copyright 2026 Intel Corporation
rem SPDX-License-Identifier: Apache-2.0
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem *******************************************************************************

rem Builds, smoke tests, and installs a oneDNN release package on Windows. Run
rem from the oneDNN source directory.
rem
rem Usage: build.bat <install-dir>
rem
rem Environment:
rem   COMPILER     cl or icx
rem   CPU_RUNTIME  value for ONEDNN_CPU_RUNTIME
rem   GPU_RUNTIME  value for ONEDNN_GPU_RUNTIME

setlocal

if "%~1"=="" (
    echo Usage: %~nx0 ^<install-dir^>
    exit /b 1
)
set "INSTALL_DIR=%~1"
set "CMAKE_EXTRA_ARGS="

if "%COMPILER%"=="cl" goto :setup_msvc
if "%COMPILER%"=="icx" goto :setup_icx
echo Unsupported compiler: %COMPILER%
exit /b 1

:setup_msvc
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
goto :build

:setup_icx
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
set "CMAKE_EXTRA_ARGS=-DONEDNN_DPCPP_HOST_COMPILER=cl"
goto :build

:build
where %COMPILER%
if errorlevel 1 exit /b 1

cmake -S . -B build -G Ninja ^
    -DCMAKE_C_COMPILER=%COMPILER% ^
    -DCMAKE_CXX_COMPILER=%COMPILER% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" ^
    -DONEDNN_CPU_RUNTIME=%CPU_RUNTIME% ^
    -DONEDNN_GPU_RUNTIME=%GPU_RUNTIME% ^
    -DONEDNN_BUILD_EXAMPLES=ON ^
    -DONEDNN_BUILD_TESTS=OFF ^
    %CMAKE_EXTRA_ARGS%
if errorlevel 1 exit /b 1
cmake --build build --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1
rem Smoke test the CPU engine only, GPU is not expected to be available.
ctest --test-dir build --output-on-failure -E "gpu|cross-engine"
if errorlevel 1 exit /b 1
cmake --install build
if errorlevel 1 exit /b 1
