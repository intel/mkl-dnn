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

rem Installs Intel oneAPI DPC++/C++ Compiler from the standalone offline
rem installer.
rem
rem Usage: install_dpcpp.bat

setlocal

rem Users of the SYCL packages need the same or a newer version of the compiler
rem runtime. Keep in sync with install_dpcpp.sh.
set "VERSION=2026.0"
set "URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/baec913e-84a0-439d-becc-6fb62f2d0c83/intel-dpcpp-cpp-compiler-2026.0.0.560_offline.exe"
set "WORK_DIR=%TEMP%\dpcpp-installer"

curl.exe --fail --location --retry 5 --retry-delay 5 --output "%WORK_DIR%.exe" "%URL%"
if errorlevel 1 exit /b 1
start /b /wait "" "%WORK_DIR%.exe" -s -x -f "%WORK_DIR%" --log "%WORK_DIR%-extract.log"
if errorlevel 1 exit /b 1
del "%WORK_DIR%.exe"
"%WORK_DIR%\bootstrapper.exe" -s --action install --eula=accept ^
    -p=NEED_VS2017_INTEGRATION=0 ^
    -p=NEED_VS2019_INTEGRATION=0 ^
    -p=NEED_VS2022_INTEGRATION=0 ^
    --log-dir="%WORK_DIR%-logs"
if errorlevel 1 exit /b 1
rmdir /s /q "%WORK_DIR%"

set "ICX=%ProgramFiles(x86)%\Intel\oneAPI\compiler\%VERSION%\bin\icx.exe"
if not exist "%ICX%" (
    echo Intel oneAPI DPC++/C++ Compiler %VERSION% is not installed
    exit /b 1
)
