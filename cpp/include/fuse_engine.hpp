#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "fuse_models.hpp"

namespace fuse {

struct MetricRow {
    std::string key;
    double value;
};

struct EventRow {
    std::unordered_map<std::string, std::string> values;
};

struct SimulationResult {
    std::vector<MetricRow> metrics;
    std::vector<EventRow> events;
};

SimulationResult run_simulation(
    const std::vector<TaskRow>& tasks,
    const std::vector<MicroRow>& microfragments,
    const std::vector<PathRow>& paths,
    const std::string& mode
);

}
