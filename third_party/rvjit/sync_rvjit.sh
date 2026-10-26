#!/bin/bash

#===============================================================================
# Copyright 2026 Barcelona Supercomputing Center
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

# Vendors the rvjit headers into third_party/rvjit.
#
# Usage: sync_rvjit.sh <rvjit repository path or url> [ref | --worktree]
#
#   ref         the release tag to vendor (e.g. v0.1.0), or a commit or
#               branch (default: HEAD)
#   --worktree  vendor the working tree of a local checkout, uncommitted
#               changes included; the recorded revision is marked "-dirty"

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    sed -n '/^# Usage/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

src=$1
ref=${2:-HEAD}
if [ -d "$src" ]; then src=$(cd "$src" && pwd); fi

dest=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ "$ref" = "--worktree" ]; then
    [ -d "$src/.git" ] || { echo "error: --worktree needs a local checkout"; exit 1; }
    from=$(cd "$src" && pwd)
    rev=$(git -C "$from" rev-parse HEAD)
    if [ -n "$(git -C "$from" status --porcelain)" ]; then rev="$rev-dirty"; fi
    version=$(git -C "$from" describe --tags --exact-match 2>/dev/null || true)
else
    git clone --quiet --no-checkout "$src" "$tmp/rvjit"
    git -C "$tmp/rvjit" checkout --quiet "$ref"
    from="$tmp/rvjit"
    rev=$(git -C "$from" rev-parse HEAD)
    version=$(git -C "$from" describe --tags --exact-match 2>/dev/null || true)
fi

[ -d "$from/rvjit" ] || { echo "error: $from has no rvjit/ directory"; exit 1; }

rm -rf "$dest/rvjit"
mkdir -p "$dest"
cp -r "$from/rvjit" "$dest/rvjit"
cp "$from/LICENSE" "$dest/LICENSE"

cat > "$dest/README.md" <<README
This code is from rvjit, a component library for RV64 JIT code generation
maintained by the Barcelona Supercomputing Center.

source: $src
version: ${version:-none (not a release)}
revision: $rev

Do not edit the files here: update them with third_party/rvjit/sync_rvjit.sh.
README

echo "rvjit ${version:-$rev} ($rev) vendored into $dest"
