#!/usr/bin/perl
#===============================================================================
# Copyright 2020 Intel Corporation
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


use strict;
use warnings;

my $USER=`whoami`;
chomp $USER;

my $benchmarkid = "";
if (defined $ENV{BENCHMARK_ID}) {
    $benchmarkid = $ENV{BENCHMARK_ID};
}

my $PFX="/tmp/gemm.$USER$benchmarkid";

my $quiet = 0;
if (@ARGV > 0 && $ARGV[0] eq "-q") {
    shift;
    $quiet = 1;
}

my $kerneldesc = join(' ', @ARGV);

my $suffix = "";
if (defined $ENV{SUFFIX}) {
    $suffix = "-" . $ENV{SUFFIX};
}

my $frontend_exec = "ofrontend$suffix";
(-x "./$frontend_exec") || die("Could not find generator executable $frontend_exec in your working directory");

if (system("./$frontend_exec -o $PFX $kerneldesc")) {
    if ($quiet) {
        exit ($? >> 8);
    } else {
        my $err = "$PFX.note";
        system('sed -ne \'/terminate called/,$ p\' ' . "$err > /dev/stderr");
        system('sed -ne \'/Exception occurred/,$ p\' ' . "$err > /dev/stderr");
        die("Failed to create kernel -- full error output: $err");
    }
}

if ($benchmarkid eq "") {
    system("cp $PFX.elf gemm.elf; cp $PFX.note gemm.note");
}
