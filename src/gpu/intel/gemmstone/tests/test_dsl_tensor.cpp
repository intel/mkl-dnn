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

#include "test_common.hpp"

#include "gemmstone/dsl/tensor.hpp"

#include <algorithm>
#include <random>
#include <vector>

using namespace gemmstone::dsl;
using block_t = layout::block_t;

using blocks_t = std::vector<block_t>;

blocks_t split_random_block(const blocks_t &blocks, std::mt19937 &rng) {
    if (blocks.empty()) return blocks;
    // Collect {block_index, divisor} pairs for all possible splits.
    std::vector<std::pair<int, int64_t>> splits;
    for (int i = 0; i < (int)blocks.size(); i++) {
        if (!blocks[i].stride.is_fixed()) continue;
        for (int64_t d = 2; d < blocks[i].size; d++)
            if (blocks[i].size % d == 0) splits.push_back({i, d});
    }
    if (splits.empty()) return blocks;
    auto [idx, inner] = splits[rng() % splits.size()];
    auto &b = blocks[idx];
    int64_t outer = b.size / inner;
    int64_t stride = (int64_t)b.stride;
    blocks_t res;
    for (int i = 0; i < (int)blocks.size(); i++) {
        if (i == idx) {
            res.push_back({b.idx, inner, stride});
            res.push_back({b.idx, outer, stride * inner});
        } else {
            res.push_back(blocks[i]);
        }
    }
    return res;
}

blocks_t shuffle_blocks(const blocks_t &blocks, std::mt19937 &rng) {
    blocks_t res = blocks;
    std::shuffle(res.begin(), res.end(), rng);
    return res;
}

blocks_t add_size_1_blocks(const blocks_t &blocks, std::mt19937 &rng) {
    blocks_t res = blocks;
    int n = 1 + rng() % 3;
    for (int i = 0; i < n; i++) {
        size_t pos = rng() % (res.size() + 1);
        idx_t idx(rng() % 3);
        int64_t stride = 1 + rng() % 64;
        res.insert(res.begin() + pos, {idx, 1, stride});
    }
    return res;
}

blocks_t gen_layout(std::mt19937 &rng) {
    int ndims = 1 + rng() % 3;
    int nblocks = ndims + rng() % 3;
    blocks_t blocks;
    int64_t stride = 1;
    for (int i = 0; i < nblocks; i++) {
        idx_t idx(rng() % ndims);
        int64_t size = 2 + rng() % 15;
        blocks.push_back({idx, size, stride});
        stride *= size;
    }
    return blocks;
}

bool test_normalize_blocks() {
    bool tests_pass = true;
    std::mt19937 rng(42);

    for (int tc = 0; tc < 50; tc++) {
        auto original = gen_layout(rng);
        auto expected = layout::normalize_blocks(original);
        for (int iter = 0; iter < 20; iter++) {
            auto modified = original;
            int n_splits = 1 + rng() % 3;
            for (int s = 0; s < n_splits; s++)
                modified = split_random_block(modified, rng);
            modified = shuffle_blocks(modified, rng);
            modified = add_size_1_blocks(modified, rng);
            modified = shuffle_blocks(modified, rng);
            auto got = layout::normalize_blocks(modified);
            EXPECT(got == expected);
        }
    }
    return tests_pass;
}

int main() {
    bool tests_pass = test_normalize_blocks();
    printf(tests_pass ? "PASSED\n" : "FAILED\n");
    return tests_pass ? 0 : 1;
}
