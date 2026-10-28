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

rem Sets up the environment to build and run applications with this oneDNN
rem package. Run it from cmd:
rem
rem     call <package>\env\vars.bat

for %%i in ("%~dp0..") do set "DNNLROOT=%%~fi"

rem Separators are added only to variables that are already set, as an empty
rem entry may be interpreted as the current directory. Values are quoted, so
rem parentheses in paths like "Program Files (x86)" do not end the blocks.
if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%DNNLROOT%;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%DNNLROOT%"
)
if defined INCLUDE (
    set "INCLUDE=%DNNLROOT%\include;%INCLUDE%"
) else (
    set "INCLUDE=%DNNLROOT%\include"
)
if defined LIB (
    set "LIB=%DNNLROOT%\lib;%LIB%"
) else (
    set "LIB=%DNNLROOT%\lib"
)
set "PATH=%DNNLROOT%\bin;%PATH%"
