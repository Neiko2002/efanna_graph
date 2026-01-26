#pragma once

#include <algorithm>
#include <vector>
#include <unordered_set>

#include "kgraph.h"
#include "kgraph-data.h"
#include "stopwatch.h"
#include "logging.h"

namespace kgraph::benchmark {

static void test_graph_anns(kgraph::KGraph* graph,
                            const kgraph::Matrix<float>& queries,
                            kgraph::MatrixOracle<float, kgraph::metric::l2sqr>& oracle,
                            const std::vector<std::vector<uint32_t>>& ground_truth,
                            const uint32_t repeat,
                            const uint32_t k,
                            const std::vector<unsigned int>& P_parameter,
                            const float recall_target = 0.995f) {
    std::vector<unsigned int> P_sorted = P_parameter;
    std::sort(P_sorted.begin(), P_sorted.end());

    // Create result matrix ONCE outside the loop (like index_evaluation.cpp)
    const size_t query_count = queries.size();
    std::vector<unsigned int> result(query_count * k);

    kgraph::KGraph::SearchParams params;
    params.K = k;
    params.M = kgraph::default_M;
    params.S = kgraph::default_S;
    params.T = kgraph::default_T;
    params.init = 0;

    for (unsigned int P_search : P_sorted) {
        params.P = std::max(P_search, k);

        StopW stopw;
        size_t correct = 0;

        for (uint32_t t = 0; t < repeat; t++) {
            std::fill(result.begin(), result.end(), 0u);

            for (size_t i = 0; i < query_count; ++i) {
                graph->search(oracle.query(queries[i]), params, &result[i * k], nullptr);
            }

            for (size_t i = 0; i < query_count; i++) {
                const auto& gt = ground_truth[i];
                unsigned int* predictions = &result[i * k];
                for (size_t r = 0; r < k; r++) {
                    if (std::binary_search(gt.begin(), gt.end(), predictions[r])) {
                        correct++;
                    }
                }
            }
        }

        float recall = static_cast<float>(correct) / static_cast<float>(repeat) / (static_cast<float>(query_count) * static_cast<float>(k));
        auto time_us_per_query = stopw.getElapsedTimeMicro() / (query_count * repeat);

        log("P_search %5u, recall %.4f, time_us_per_query %8lld\n", P_search, recall, static_cast<long long>(time_us_per_query));

        // Early exit if recall target reached
        if (recall >= recall_target) {
            log("Recall target %.3f reached, stopping P sweep\n", recall_target);
            break;
        }
    }
}

static void test_graph_explore(kgraph::KGraph* graph,
                         const kgraph::Matrix<float>& queries,
                         kgraph::MatrixOracle<float, kgraph::metric::l2sqr>& oracle,
                         const std::vector<std::vector<uint32_t>>& ground_truth,
                         const std::vector<std::vector<uint32_t>>& entry_node_indices,
                         const uint32_t k,
                        const float recall_target = 0.995f) {
    log("Testing Exploration (k=%u)...\n", k);

    if (entry_node_indices.size() != queries.size()) {
        log("Exploration Test aborted: entry_node_indices size (%zu) != queries size (%u)\n",
            entry_node_indices.size(), queries.size());
        return;
    }

    for (size_t q = 0; q < entry_node_indices.size(); ++q) {
        if (entry_node_indices[q].empty()) {
            log("Exploration Test aborted: entry_node_indices[%zu] is empty\n", q);
            return;
        }
    }

    kgraph::KGraph::SearchParams params;
    params.M = kgraph::default_M;
    params.P = k;
    params.S = 1000;
    params.T = kgraph::default_T;
    params.init = 1;
    params.K = k;

    const size_t query_count = queries.size();
    std::vector<unsigned int> result(query_count * k);

    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 2; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            std::fill(result.begin(), result.end(), 0u);

            StopW stopw;
            size_t correct = 0;

            for (size_t q = 0; q < query_count; ++q) {
                unsigned int* prediction = &result[q * k];
                prediction[0] = entry_node_indices[q][0];

                const auto len = graph->explore(oracle.query(queries[q]), params, prediction, max_distance_count, nullptr);

                if (q < ground_truth.size()) {
                    const auto& gt = ground_truth[q];
                    for (size_t r = 0; r < len; r++) {
                        if (std::binary_search(gt.begin(), gt.end(), prediction[r])) {
                            correct++;
                        }
                    }
                }
            }

            const float denom = static_cast<float>(query_count) * static_cast<float>(k);
            const float recall = denom > 0.0f ? static_cast<float>(correct) / denom : 0.0f;
            const uint64_t time_us_per_query = queries.size() > 0 ? (stopw.getElapsedTimeMicro() / queries.size()) : 0;

            log("k %5u, max_distance_count %6u, recall %.4f, time_us_per_query %6llu\n",
                k, max_distance_count, recall, static_cast<unsigned long long>(time_us_per_query));

            // Early exit if recall target reached
            if (recall >= recall_target) {
                log("Recall target %.3f reached, stopping P sweep\n", recall_target);
                break;
            }
        }
    }
}

} // namespace kgraph::benchmark
