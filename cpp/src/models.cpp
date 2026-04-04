#include "fuse_models.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace fuse {

namespace {

long long as_i64(const Row& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
        return 0;
    }
    return std::stoll(it->second);
}

std::string as_string(const Row& row, const std::string& key) {
    auto it = row.find(key);
    if (it == row.end()) {
        return "";
    }
    return it->second;
}

const std::unordered_map<std::string, long long> kBaseLatency = {
    {"field_mul", 2},
    {"field_add", 1},
    {"field_reduce", 1},
    {"fft_butterfly", 2},
    {"poseidon_sbox", 2},
    {"poseidon_mix", 1},
    {"msm_bucket", 2},
    {"reduce_tree", 1},
    {"im2col", 1},
    {"matmul_tile", 2},
    {"relu", 1},
    {"maxpool", 1},
    {"gelu", 2},
    {"layernorm", 2},
    {"subbytes", 1},
    {"shift_rows", 1},
    {"mix_columns", 2},
    {"add_round_key", 1},
    {"key_expand", 1},
    {"linear_map", 2},
    {"permutation", 1},
    {"nonlinear_lookup", 2},
    {"generic", 1},
};

}

TaskRow parse_task_row(const Row& row) {
    TaskRow task;
    task.runtime_id = as_string(row, "runtime_id");
    task.task_id = as_string(row, "task_id");
    task.task_type = as_string(row, "task_type");
    task.domain = as_string(row, "domain");
    task.arrival_cycle = as_i64(row, "arrival_cycle");
    return task;
}

MicroRow parse_micro_row(const Row& row) {
    MicroRow micro;
    micro.runtime_micro_id = as_string(row, "runtime_micro_id");
    micro.runtime_id = as_string(row, "runtime_id");
    micro.task_id = as_string(row, "task_id");
    micro.task_type = as_string(row, "task_type");
    micro.stage = as_string(row, "stage");
    micro.parent_fragment_id = as_string(row, "parent_fragment_id");
    micro.base_micro_id = as_string(row, "base_micro_id");
    micro.op = as_string(row, "op");
    micro.template_family = as_string(row, "template_family");
    micro.structural_hash = as_string(row, "structural_hash");
    micro.semantic_hash = as_string(row, "semantic_hash");
    micro.semantic_group = as_string(row, "semantic_group");
    micro.domain = as_string(row, "domain");
    micro.dtype = as_string(row, "dtype");
    micro.layout = as_string(row, "layout");
    micro.tile = as_string(row, "tile");
    micro.quant = as_string(row, "quant");
    micro.delta_t = as_i64(row, "delta_t");
    micro.estimated_ops = as_i64(row, "estimated_ops");
    micro.estimated_bytes = as_i64(row, "estimated_bytes");
    std::string deps = as_string(row, "deps");
    if (!deps.empty()) {
        micro.deps = split(deps, ';');
    }
    return micro;
}

PathRow parse_path_row(const Row& row) {
    PathRow path;
    path.path_id = as_string(row, "path_id");
    std::string members = as_string(row, "members");
    if (!members.empty()) {
        path.members = split(members, ';');
    }
    std::string signature = as_string(row, "signature");
    if (!signature.empty()) {
        path.signature = split(signature, ';');
    }
    std::string families = as_string(row, "template_families");
    if (!families.empty()) {
        path.template_families = split(families, ';');
    }
    path.reuse_count = as_i64(row, "reuse_count");
    return path;
}

std::vector<TaskRow> load_tasks(const std::string& path) {
    std::vector<TaskRow> out;
    for (const auto& row : read_csv(path)) {
        out.push_back(parse_task_row(row));
    }
    return out;
}

std::vector<MicroRow> load_microfragments(const std::string& path) {
    std::vector<MicroRow> out;
    for (const auto& row : read_csv(path)) {
        out.push_back(parse_micro_row(row));
    }
    return out;
}

std::vector<PathRow> load_paths(const std::string& path) {
    std::vector<PathRow> out;
    for (const auto& row : read_csv(path)) {
        out.push_back(parse_path_row(row));
    }
    return out;
}

long long base_latency_cycles(const MicroRow& micro) {
    auto it = kBaseLatency.find(micro.op);
    long long base = 1;
    if (it != kBaseLatency.end()) {
        base = it->second;
    } else {
        auto jt = kBaseLatency.find(micro.template_family);
        if (jt != kBaseLatency.end()) {
            base = jt->second;
        }
    }
    long long scale = std::max(1LL, micro.estimated_ops / 64LL);
    long long memory_penalty = micro.estimated_bytes > 0 ? 1LL : 0LL;
    return std::max(1LL, base * scale + memory_penalty);
}

double dedicated_speedup(const std::string& mode, const std::string& task_type) {
    if (mode == "static_zkp" && task_type == "ZKP") {
        return 1.0;
    }
    if (mode == "eyeriss" && task_type == "CNN") {
        return 1.11;
    }
    if (mode == "openaes" && task_type == "AES") {
        return 1.03;
    }
    if (mode == "multicore") {
        return 0.94;
    }
    if (mode == "gpu_tensor") {
        if (task_type == "LLM" || task_type == "CNN") {
            return 1.18;
        }
        if (task_type == "AES") {
            return 1.04;
        }
        return 0.98;
    }
    if (mode == "transformer_asic") {
        if (task_type == "LLM") {
            return 1.28;
        }
        return 0.97;
    }
    return 1.0;
}

double energy_scale(const std::string& mode) {
    if (mode == "fuse") {
        return 1.0;
    }
    if (mode == "degraded") {
        return 1.12;
    }
    if (mode == "static_zkp") {
        return 1.03;
    }
    if (mode == "eyeriss") {
        return 0.96;
    }
    if (mode == "openaes") {
        return 0.98;
    }
    if (mode == "multicore") {
        return 1.08;
    }
    if (mode == "secure_ct") {
        return 1.15;
    }
    if (mode == "secure_randomized") {
        return 1.08;
    }
    if (mode == "no_fvm") {
        return 0.96;
    }
    if (mode == "no_prefetch") {
        return 1.02;
    }
    if (mode == "fixed_schedule") {
        return 1.04;
    }
    if (mode == "gpu_tensor") {
        return 1.18;
    }
    if (mode == "transformer_asic") {
        return 0.97;
    }
    return 1.0;
}

}
