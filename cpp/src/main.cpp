#include "fuse_engine.hpp"
#include "fuse_csv.hpp"
#include "fuse_models.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i + 1 < argc; i += 2) {
        args[argv[i]] = argv[i + 1];
    }
    return args;
}

}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    try {
        std::string tasks_path = args.at("--tasks");
        std::string micro_path = args.at("--microfragments");
        std::string paths_path = args.at("--paths");
        std::string mode = args.at("--mode");
        std::string metrics_out = args.at("--metrics-out");
        std::string events_out = args.at("--events-out");
        auto tasks = fuse::load_tasks(tasks_path);
        auto microfragments = fuse::load_microfragments(micro_path);
        auto paths = fuse::load_paths(paths_path);
        auto result = fuse::run_simulation(tasks, microfragments, paths, mode);
        std::vector<std::vector<std::string>> metric_rows;
        for (const auto& metric : result.metrics) {
            if (metric.key == "mode") {
                metric_rows.push_back({metric.key, mode});
            } else {
                metric_rows.push_back({metric.key, std::to_string(metric.value)});
            }
        }
        fuse::write_csv(metrics_out, {"key", "value"}, metric_rows);
        std::vector<std::string> event_header = {
            "event", "cycle", "finish_cycle", "runtime_micro_id", "task_runtime_id", "task_type",
            "template_family", "reused_result", "structural_reuse", "template_prefetched",
            "switch_cycles", "path_id", "path_reuse_count", "rfu_id"
        };
        std::vector<std::vector<std::string>> event_rows;
        for (const auto& event : result.events) {
            std::vector<std::string> row;
            for (const auto& key : event_header) {
                auto it = event.values.find(key);
                row.push_back(it == event.values.end() ? "" : it->second);
            }
            event_rows.push_back(std::move(row));
        }
        fuse::write_csv(events_out, event_header, event_rows);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
