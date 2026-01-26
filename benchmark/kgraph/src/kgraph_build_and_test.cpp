#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "build.h"
#include "dataset.h"
#include "logging.h"
#include "stopwatch.h"

using namespace kgraph;
using namespace kgraph::benchmark;

struct DatasetConfig {
    DatasetName dataset_name = DatasetName::SIFT1M;
    Metric metric = Metric::L2;

    CreateGraphTest create_graph;
};

static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M) {
        conf.create_graph.K = 90;
        conf.create_graph.L = 130;
        conf.create_graph.iterations = 12;
        conf.create_graph.S = 20;
        conf.create_graph.R = 50;
        conf.create_graph.recall = 1.00f; // our addition
        conf.create_graph.P_parameter = {100, 200, 300, 500};
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.create_graph.K = 90;
        conf.create_graph.L = 130;
        conf.create_graph.iterations = 12;
        conf.create_graph.S = 20;
        conf.create_graph.R = 50;
        conf.create_graph.recall = 1.00f; // our addition
        conf.create_graph.P_parameter = {100, 300, 500, 700};
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.create_graph.K = 100;
        conf.create_graph.L = 150;
        conf.create_graph.iterations = 12;
        conf.create_graph.S = 35;
        conf.create_graph.R = 150;
        conf.create_graph.P_parameter = {100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000};
    } else if (dataset_name == DatasetName::ENRON) {
        conf.create_graph.K = 50;
        conf.create_graph.L = 80;
        conf.create_graph.iterations = 7;
        conf.create_graph.S = 15;
        conf.create_graph.R = 100;
        conf.create_graph.P_parameter = {100, 200, 300, 400, 500, 1000, 2000, 4000, 8000, 16000};
    } else if (dataset_name == DatasetName::AUDIO) {
        conf.create_graph.K = 40;
        conf.create_graph.L = 60;
        conf.create_graph.iterations = 5;
        conf.create_graph.S = 20;
        conf.create_graph.R = 100;
        conf.create_graph.P_parameter = {100, 200, 300, 400, 600, 700, 1000, 3000, 8000};
    }

    return conf;
}

struct GraphPaths {
    std::filesystem::path graph_dir;

    GraphPaths(const Dataset& ds) : graph_dir(ds.data_root() / ds.name() / "kgraph") {}

    std::string base_name(unsigned dims, const CreateGraphTest& cg) const {
        return string_format("K%u_L%u_It%u_S%u_R%u", cg.K, cg.L, cg.iterations, cg.S, cg.R);
    }

    std::string graph_directory() const {
        return graph_dir.string();
    }

    std::string graph_file(unsigned dims, const CreateGraphTest& cg) const {
        return (graph_dir / (base_name(dims, cg) + ".kg")).string();
    }

    std::string graph_log_file(unsigned dims, const CreateGraphTest& cg) const {
        return (graph_dir / (base_name(dims, cg) + ".log")).string();
    }
};

static void run_create_graph_test(const Dataset& ds,
                                  const DatasetConfig& config,
                                  const GraphPaths& paths,
                                  const kgraph::Matrix<float>& base_matrix,
                                  const kgraph::Matrix<float>& query_matrix) {
    const auto& cg = config.create_graph;
    const uint32_t dims = static_cast<uint32_t>(base_matrix.dim());

    std::string graph_path = paths.graph_file(dims, cg);
    std::string log_path = paths.graph_log_file(dims, cg);

    log("\n=== CREATE_GRAPH Test ===\n");
    log("Settings: K=%u, L=%u, It=%u, S=%u, R=%u, recall=%.3f\n", cg.K, cg.L, cg.iterations, cg.S, cg.R, cg.recall);
    log("Graph: %s\n", graph_path.c_str());
    log("Log: %s\n", log_path.c_str());

    std::filesystem::create_directories(paths.graph_directory());
    if (std::filesystem::exists(log_path)) {
        log("CREATE_GRAPH: Skipping - log file already exists: %s\n", log_path.c_str());
        return;
    }
    set_log_file(log_path, true);
    attach_cerr_to_log();

    kgraph::MatrixOracle<float, kgraph::metric::l2sqr> oracle(base_matrix);

    log("Base matrix: size=%u, dim=%u\n", base_matrix.size(), base_matrix.dim());
    log("Query matrix: size=%u, dim=%u\n", query_matrix.size(), query_matrix.dim());

    std::unique_ptr<kgraph::KGraph> index;

    if (std::filesystem::exists(graph_path)) {
        log("Graph already exists, loading: %s\n", graph_path.c_str());
        index.reset(kgraph::KGraph::create());
        index->load(graph_path.c_str());
    } else {
        log("\n--- Building graph ---\n");
        kgraph::KGraph::IndexInfo info{};
        index = build_index(oracle, cg, &info);
        index->save(graph_path.c_str());
        log("Saved graph: %s\n", graph_path.c_str());
        log("Stop condition: %u, iterations=%u, recall=%.4f, delta=%.6f, accuracy=%.4f\n",
            static_cast<unsigned>(info.stop_condition), info.iterations, info.recall, info.delta, info.accuracy);
    }

    if (index) {
        run_common_tests(index.get(), ds, query_matrix, oracle, cg, false);
    }

    reset_log_to_console();
    log("CREATE_GRAPH: Log written to: %s\n", log_path.c_str());
}

int main(int argc, char** argv) {
    log("Testing ...\n");

    #if defined(__AVX__)
        std::cout << "use AVX2  ..." << std::endl;
    #elif defined(__SSE2__)
        std::cout << "use SSE  ..." << std::endl;
    #else
        std::cout << "use arch  ..." << std::endl;
    #endif
    std::cout << "KGRAPH_MATRIX_ALIGN " << KGRAPH_MATRIX_ALIGN << std::endl;

    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    const auto data_path = std::filesystem::path(DATA_PATH);
    log("data_path %s\n", data_path.string().c_str());

    DatasetName ds_name = DatasetName::ALL;
    std::string test_type_arg = "create_graph";
    std::string data_root = data_path.string();
    bool do_run = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "help" || arg == "--help") {
            log("Usage: kgraph_benchmark <dataset> [test_type] [data_root] [--run|--dry-run]\n");
            log("Datasets: sift1m, deep1m, audio, glove, enron, all\n");
            log("Test types:\n");
            log("  create_graph    - Build graph, run stats, ANNS, explore\n");
            log("  all             - Same as create_graph\n");
            log("Options: [data_root] path (default: DATA_PATH), --run or --dry-run\n");
            return 0;
        }

        if (arg == "--run") {
            do_run = true;
            continue;
        }
        if (arg == "--dry-run") {
            do_run = false;
            continue;
        }

        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid()) {
            ds_name = parsed_ds;
            continue;
        }

        if (arg == "create_graph" || arg == "all") {
            test_type_arg = arg;
            continue;
        }

        data_root = arg;
    }

    auto run_for_dataset = [&](const DatasetName& name) -> int {
        try {
            Dataset ds(name, data_root);
            auto config = get_dataset_config(name);
            config.metric = ds.info().metric;
            GraphPaths graph_paths(ds);

            log("\n=== Dataset: %s ===\n", ds.name());
            log("Repository file: %s\n", ds.base_file().c_str());
            log("Query file: %s\n", ds.query_file().c_str());
            log("Graph directory: %s\n", graph_paths.graph_directory().c_str());
            log("Ground truth (full): %s\n", ds.groundtruth_file_full().c_str());
            log("Ground truth (half): %s\n", ds.groundtruth_file_half().c_str());
            log("Build settings: K=%u, L=%u, It=%u, S=%u, R=%u, recall=%.3f\n",
                config.create_graph.K,
                config.create_graph.L,
                config.create_graph.iterations,
                config.create_graph.S,
                config.create_graph.R,
                config.create_graph.recall);

            if (do_run) {
                if (!std::filesystem::exists(ds.base_file())) {
                    log("Missing base file: %s\n", ds.base_file().c_str());
                    return 1;
                }
                if (!std::filesystem::exists(ds.query_file())) {
                    log("Missing query file: %s\n", ds.query_file().c_str());
                    return 1;
                }
                // Avoid std::abort later: check groundtruth upfront.
                if (!std::filesystem::exists(ds.groundtruth_file_full())) {
                    log("Missing groundtruth (full): %s\n", ds.groundtruth_file_full().c_str());
                    return 1;
                }
                if (!std::filesystem::exists(ds.groundtruth_file_half())) {
                    log("Missing groundtruth (half): %s\n", ds.groundtruth_file_half().c_str());
                    return 1;
                }

                log("\nLoading data...\n");
                auto base_matrix = ds.load_base();
                auto query_matrix = ds.load_query();

                log("Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

                if (test_type_arg == "create_graph" || test_type_arg == "all") {
                    run_create_graph_test(ds, config, graph_paths, base_matrix, query_matrix);
                }
            }

            return 0;
        } catch (const std::exception& e) {
            log("ERROR: Dataset '%s' failed with exception: %s\n", name.name(), e.what());
            return 1;
        } catch (...) {
            log("ERROR: Dataset '%s' failed with unknown exception\n", name.name());
            return 1;
        }
    };

    if (ds_name == DatasetName::ALL) {
        for (const auto& name : DatasetName::all()) {
            log("\n--- ALL: starting dataset %s ---\n", name.name());
            int rc = run_for_dataset(name);
            log("--- ALL: finished dataset %s with rc=%d ---\n", name.name(), rc);
        }
    } else {
        return run_for_dataset(ds_name);
    }

    return 0;
}
