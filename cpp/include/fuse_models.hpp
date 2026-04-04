#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "fuse_csv.hpp"

namespace fuse {

struct TaskRow {
    std::string runtime_id;
    std::string task_id;
    std::string task_type;
    std::string domain;
    long long arrival_cycle;
};

struct MicroRow {
    std::string runtime_micro_id;
    std::string runtime_id;
    std::string task_id;
    std::string task_type;
    std::string stage;
    std::string parent_fragment_id;
    std::string base_micro_id;
    std::string op;
    std::string template_family;
    std::string structural_hash;
    std::string semantic_hash;
    std::string semantic_group;
    std::string domain;
    std::string dtype;
    std::string layout;
    std::string tile;
    std::string quant;
    long long delta_t;
    long long estimated_ops;
    long long estimated_bytes;
    std::vector<std::string> deps;
};

struct PathRow {
    std::string path_id;
    std::vector<std::string> members;
    std::vector<std::string> signature;
    std::vector<std::string> template_families;
    long long reuse_count = 0;
};

TaskRow parse_task_row(const Row& row);
MicroRow parse_micro_row(const Row& row);
PathRow parse_path_row(const Row& row);
std::vector<TaskRow> load_tasks(const std::string& path);
std::vector<MicroRow> load_microfragments(const std::string& path);
std::vector<PathRow> load_paths(const std::string& path);
long long base_latency_cycles(const MicroRow& micro);
double dedicated_speedup(const std::string& mode, const std::string& task_type);
double energy_scale(const std::string& mode);

}
