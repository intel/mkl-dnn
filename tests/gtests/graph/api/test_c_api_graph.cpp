/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

#include "test_api_common.h"

TEST(CAPI, CreateGraphWithEngine) {
    dnnl_graph_graph_t agraph = nullptr;
    dnnl_engine_kind_t engine = dnnl_cpu;

#define CREATE_GRAPH_WITH_ENGINE_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_graph_create(&agraph, engine), dnnl_success,
            CREATE_GRAPH_WITH_ENGINE_DESTROY);
    CREATE_GRAPH_WITH_ENGINE_DESTROY;
#undef CREATE_GRAPH_WITH_ENGINE_DESTROY
}

TEST(CAPI, SetAndGetDeterministic) {
    dnnl_graph_graph_t agraph = nullptr;

#define SET_AND_GET_DETERMINISTIC_DESTROY \
    do { \
        dnnl_graph_graph_destroy(agraph); \
        agraph = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_graph_create(&agraph, dnnl_cpu), dnnl_success,
            SET_AND_GET_DETERMINISTIC_DESTROY);

    int deterministic = -1;
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(agraph, &deterministic),
            dnnl_success, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(deterministic, 0, SET_AND_GET_DETERMINISTIC_DESTROY);

    // Any non-zero value enables the mode.
    ASSERT_EQ_SAFE(dnnl_graph_graph_set_deterministic(agraph, 2), dnnl_success,
            SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(agraph, &deterministic),
            dnnl_success, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(deterministic, 1, SET_AND_GET_DETERMINISTIC_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_graph_set_deterministic(agraph, 0), dnnl_success,
            SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(agraph, &deterministic),
            dnnl_success, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(deterministic, 0, SET_AND_GET_DETERMINISTIC_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_graph_set_deterministic(nullptr, 1),
            dnnl_invalid_arguments, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(nullptr, &deterministic),
            dnnl_invalid_arguments, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(agraph, nullptr),
            dnnl_invalid_arguments, SET_AND_GET_DETERMINISTIC_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_graph_finalize(agraph), dnnl_success,
            SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_set_deterministic(agraph, 1),
            dnnl_invalid_graph, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_graph_get_deterministic(agraph, &deterministic),
            dnnl_success, SET_AND_GET_DETERMINISTIC_DESTROY);
    ASSERT_EQ_SAFE(deterministic, 0, SET_AND_GET_DETERMINISTIC_DESTROY);

    SET_AND_GET_DETERMINISTIC_DESTROY;
#undef SET_AND_GET_DETERMINISTIC_DESTROY
}
