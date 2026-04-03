#include "morphaccel_engine.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>

namespace morphaccel {

namespace {

constexpr long long kCyclesPerMs = 1000000LL;
constexpr int kRfuCount = 16;
constexpr long long kFvmCapacityBytes = 8LL * 1024LL * 1024LL;
constexpr long long kBaseSwitchCycles = 2300LL;
constexpr long long kFreshnessWindowCycles = 5000000LL;
constexpr long long kMemoryBandwidthBytesPerCycle = 64LL;

struct RFUState {
    int rfu_id = 0;
    long long busy_until = 0;
    std::string loaded_template;
    long long executed_cycles = 0;
};

struct CacheEntry {
    long long created_cycle = 0;
    long long last_access_cycle = 0;
    long long bytes_used = 0;
    int hits = 0;
};

struct PathBinding {
    std::string path_id;
    long long reuse_count = 0;
};

struct RuntimeNode {
    MicroRow micro;
    long long arrival_cycle = 0;
    std::set<std::string> deps;
    std::vector<std::string> succs;
    bool ready = false;
    bool running = false;
    bool completed = false;
    long long ready_cycle = -1;
    long long start_cycle = -1;
    long long finish_cycle = -1;
};

struct RunningItem {
    long long finish_cycle = 0;
    std::string runtime_micro_id;
    int rfu_id = 0;
};

struct RunningCompare {
    bool operator()(const RunningItem& a, const RunningItem& b) const {
        return a.finish_cycle > b.finish_cycle;
    }
};

bool cache_fresh(const CacheEntry& entry, long long cycle) {
    return cycle - entry.created_cycle <= kFreshnessWindowCycles;
}

double ready_score(
    const RuntimeNode& node,
    long long cycle,
    const std::unordered_map<std::string, CacheEntry>& fvm,
    const std::deque<std::string>& prefetch_queue,
    double& kappa_hits,
    double& template_hits,
    double& valid_hits
) {
    double score = 0.0;
    auto it = fvm.find(node.micro.semantic_hash);
    if (it != fvm.end() && cache_fresh(it->second, cycle)) {
        score += 12.0;
        valid_hits += 1.0;
    }
    if (std::find(prefetch_queue.begin(), prefetch_queue.end(), node.micro.template_family) != prefetch_queue.end()) {
        score += 5.0;
        template_hits += 1.0;
    }
    if (node.micro.layout == "tile" || node.micro.layout == "block" || node.micro.layout == "state4x4" || node.micro.layout == "nchw") {
        score += 1.0;
        kappa_hits += 1.0;
    }
    if (node.ready_cycle >= 0) {
        score += 1.5 * static_cast<double>(std::max(0LL, cycle - node.ready_cycle)) / 128.0;
    }
    return score;
}

bool mode_uses_prefetch(const std::string& mode) {
    return mode != "secure_elastic_no_prefetch";
}

bool mode_allows_result_reuse(const std::string& mode) {
    return mode == "morphaccel" || mode == "secure_ct" || mode.rfind("secure_elastic", 0) == 0;
}

bool mode_has_fvm_lookup(const std::string& mode) {
    return mode != "degraded";
}

bool mode_has_fvm_write(const std::string& mode) {
    return true;
}

bool mode_uses_scored_scheduler(const std::string& mode) {
    return true;
}

bool mode_uses_dedicated_pipeline(const std::string& mode) {
    return mode == "static_zkp" || mode == "eyeriss" || mode == "openaes" || mode == "multicore" || mode == "gpu_tensor" || mode == "transformer_asic";
}

bool is_secure_elastic_mode(const std::string& mode) {
    return mode == "secure_elastic" ||
        mode == "secure_elastic_no_bucket" ||
        mode == "secure_elastic_no_pinning" ||
        mode == "secure_elastic_no_mask" ||
        mode == "secure_elastic_no_epoch" ||
        mode == "secure_elastic_no_prefetch";
}

double baseline_speed_factor(const std::string& mode, const std::string& task_type) {
    if (mode == "static_zkp") {
        if (task_type == "ZKP") return 1.05;
        return 0.92;
    }
    if (mode == "eyeriss") {
        if (task_type == "CNN") return 1.08;
        return 0.90;
    }
    if (mode == "openaes") {
        if (task_type == "AES") return 1.03;
        return 0.94;
    }
    if (mode == "gpu_tensor") {
        if (task_type == "CNN") return 1.12;
        if (task_type == "AES") return 0.97;
        return 0.96;
    }
    if (mode == "multicore") {
        return 0.98;
    }
    return 1.0;
}

std::unordered_map<std::string, PathBinding> build_path_bindings(const std::vector<PathRow>& paths) {
    std::unordered_map<std::string, PathBinding> bindings;
    for (const auto& path : paths) {
        for (const auto& member : path.members) {
            bindings[member] = PathBinding{path.path_id, path.reuse_count};
        }
    }
    return bindings;
}

double compute_energy(
    const std::string& mode,
    long long makespan,
    long long executed_cycles,
    double reads,
    double writes,
    double scheduler_decisions,
    double switch_events
) {
    double compute_coeff = 0.65;
    double fvm_read_coeff = 0.12;
    double fvm_write_coeff = 0.15;
    double scheduling_coeff = 0.08;
    double switch_coeff = 0.25;
    double static_coeff = 0.02;
    if (mode == "gpu_tensor") {
        compute_coeff = 0.92;
        fvm_read_coeff = 0.18;
        fvm_write_coeff = 0.20;
        scheduling_coeff = 0.10;
        switch_coeff = 0.34;
        static_coeff = 0.032;
    } else if (mode == "multicore") {
        compute_coeff = 0.72;
        fvm_read_coeff = 0.14;
        fvm_write_coeff = 0.17;
        switch_coeff = 0.28;
        static_coeff = 0.028;
    } else if (mode == "static_zkp" || mode == "eyeriss" || mode == "openaes") {
        compute_coeff = 0.60;
        fvm_read_coeff = 0.10;
        fvm_write_coeff = 0.12;
        static_coeff = 0.018;
    }
    double compute_nj = static_cast<double>(executed_cycles) * compute_coeff;
    double fvm_nj = reads * fvm_read_coeff + writes * fvm_write_coeff;
    double scheduling_nj = scheduler_decisions * scheduling_coeff + switch_events * switch_coeff;
    double static_nj = static_cast<double>(makespan) * static_coeff * static_cast<double>(kRfuCount);
    return compute_nj + fvm_nj + scheduling_nj + static_nj;
}

double hash_unit(const std::string& key) {
    return static_cast<double>(std::hash<std::string>{}(key) % 1000003ULL) / 1000003.0;
}

int choose_rfu(const std::vector<RFUState>& rfus, const std::vector<int>& free_ids, const std::string& template_family, const std::string& mode, const std::string& key) {
    std::vector<int> candidates;
    for (int id : free_ids) {
        if (rfus[id].loaded_template == template_family) {
            candidates.push_back(id);
        }
    }
    if (candidates.empty()) {
        candidates = free_ids;
    }
    if (mode == "secure_randomized") {
        auto offset = static_cast<size_t>(std::hash<std::string>{}(key) % std::max<size_t>(1, candidates.size()));
        return candidates[offset];
    }
    return candidates.front();
}

void ensure_capacity(
    std::unordered_map<std::string, CacheEntry>& fvm,
    std::deque<std::string>& order,
    long long incoming_bytes,
    long long capacity,
    long long& current_bytes,
    double& frees
) {
    while (current_bytes + incoming_bytes > capacity && !order.empty()) {
        std::string key = order.front();
        order.pop_front();
        auto it = fvm.find(key);
        if (it == fvm.end()) {
            continue;
        }
        current_bytes -= it->second.bytes_used;
        fvm.erase(it);
        frees += 1.0;
    }
}

void write_cache(
    const MicroRow& micro,
    long long cycle,
    std::unordered_map<std::string, CacheEntry>& fvm,
    std::deque<std::string>& order,
    long long& current_bytes,
    long long& peak_bytes,
    double& allocations,
    double& writes,
    std::vector<long long>& residency_samples,
    double& frees
) {
    long long bytes_used = std::max(32LL, micro.estimated_bytes > 0 ? micro.estimated_bytes : micro.estimated_ops * 4LL);
    ensure_capacity(fvm, order, bytes_used, kFvmCapacityBytes, current_bytes, frees);
    auto it = fvm.find(micro.semantic_hash);
    if (it == fvm.end()) {
        order.push_back(micro.semantic_hash);
        current_bytes += bytes_used;
        allocations += 1.0;
    } else {
        current_bytes += bytes_used - it->second.bytes_used;
    }
    CacheEntry entry;
    entry.created_cycle = cycle;
    entry.last_access_cycle = cycle;
    entry.bytes_used = bytes_used;
    fvm[micro.semantic_hash] = entry;
    peak_bytes = std::max(peak_bytes, current_bytes);
    writes += 1.0;
    residency_samples.push_back(current_bytes);
}

long long switch_cost(
    const std::string& mode,
    const std::string& current_template,
    const std::string& next_template,
    bool prefetched,
    bool queued,
    double& template_hit_counter,
    double& template_miss_counter,
    double& switch_counter
) {
    if (mode == "degraded") {
        template_miss_counter += 1.0;
        switch_counter += 1.0;
        return kBaseSwitchCycles;
    }
    if (mode == "static_zkp" || mode == "eyeriss" || mode == "openaes" || mode == "gpu_tensor" || mode == "transformer_asic") {
        return 0;
    }
    if (current_template == next_template) {
        template_hit_counter += 1.0;
        return 0;
    }
    switch_counter += 1.0;
    if (prefetched || queued) {
        template_hit_counter += 1.0;
        return std::max(1LL, kBaseSwitchCycles / 10LL);
    }
    template_miss_counter += 1.0;
    return kBaseSwitchCycles;
}

}

SimulationResult run_simulation(
    const std::vector<TaskRow>& tasks,
    const std::vector<MicroRow>& microfragments,
    const std::vector<PathRow>& paths,
    const std::string& mode
) {
    std::unordered_map<std::string, TaskRow> task_map;
    std::unordered_map<std::string, std::vector<std::string>> task_nodes;
    std::unordered_map<std::string, RuntimeNode> nodes;
    auto path_bindings = build_path_bindings(paths);
    for (const auto& task : tasks) {
        task_map[task.runtime_id] = task;
    }
    for (const auto& micro : microfragments) {
        RuntimeNode node;
        node.micro = micro;
        node.arrival_cycle = task_map[micro.runtime_id].arrival_cycle;
        for (const auto& dep : micro.deps) {
            if (!dep.empty()) {
                node.deps.insert(dep);
            }
        }
        nodes[micro.runtime_micro_id] = node;
        task_nodes[micro.runtime_id].push_back(micro.runtime_micro_id);
    }
    for (auto& item : nodes) {
        for (const auto& dep : item.second.deps) {
            auto it = nodes.find(dep);
            if (it != nodes.end()) {
                it->second.succs.push_back(item.first);
            }
        }
    }

    std::vector<RFUState> rfus(kRfuCount);
    for (int i = 0; i < kRfuCount; ++i) {
        rfus[i].rfu_id = i;
    }

    std::unordered_map<std::string, CacheEntry> fvm;
    std::deque<std::string> fvm_order;
    std::unordered_map<std::string, long long> negative_cache;
    std::deque<std::string> prefetch_queue;
    std::vector<std::string> ready;
    std::priority_queue<RunningItem, std::vector<RunningItem>, RunningCompare> running;
    std::unordered_map<std::string, long long> task_finish;
    std::unordered_map<std::string, int> task_completed_count;
    std::unordered_map<std::string, int> task_total_count;
    for (const auto& [runtime_id, ids] : task_nodes) {
        task_total_count[runtime_id] = static_cast<int>(ids.size());
        task_completed_count[runtime_id] = 0;
    }

    long long clock = 0;
    long long current_bytes = 0;
    long long peak_bytes = 0;
    double reads = 0.0;
    double writes = 0.0;
    double redundant_reads = 0.0;
    double allocations = 0.0;
    double frees = 0.0;
    double scheduler_decisions = 0.0;
    double prefetch_events = 0.0;
    double switch_events = 0.0;
    double template_hits = 0.0;
    double template_misses = 0.0;
    double kappa_hits = 0.0;
    double scheduler_template_hits = 0.0;
    double valid_hits = 0.0;
    double ready_probe_count = 0.0;
    double result_reuse_hits = 0.0;
    double structural_reuse_hits = 0.0;
    double pipeline_stall_cycles = 0.0;
    double prefetch_earliness_total = 0.0;
    double prefetch_earliness_count = 0.0;
    std::vector<long long> residency_samples;
    std::vector<EventRow> events;
    int completed_nodes = 0;
    const int total_nodes = static_cast<int>(nodes.size());

    auto inject_arrivals = [&](long long cycle) {
        for (auto& [id, node] : nodes) {
            if (node.completed || node.ready || node.running) {
                continue;
            }
            if (node.arrival_cycle <= cycle && node.deps.empty()) {
                node.ready = true;
                node.ready_cycle = cycle;
                ready.push_back(id);
            }
        }
    };

    inject_arrivals(0);

    while (completed_nodes < total_nodes) {
        inject_arrivals(clock);
        std::vector<int> free_ids;
        for (int i = 0; i < kRfuCount; ++i) {
            if (rfus[i].busy_until <= clock) {
                free_ids.push_back(i);
            }
        }

        while (!free_ids.empty() && !ready.empty()) {
            int best_index = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            double ready_kappa = 0.0;
            double ready_template = 0.0;
            double ready_valid = 0.0;
            if (mode_uses_scored_scheduler(mode)) {
                for (int i = 0; i < static_cast<int>(ready.size()); ++i) {
                    double kappa_probe = 0.0;
                    double template_probe = 0.0;
                    double valid_probe = 0.0;
                    ready_probe_count += 1.0;
                    double score = ready_score(nodes[ready[i]], clock, fvm, prefetch_queue, kappa_probe, template_probe, valid_probe);
                    ready_kappa += kappa_probe;
                    ready_template += template_probe;
                    ready_valid += valid_probe;
                    if (mode == "secure_randomized") {
                        score += 0.4 * hash_unit(ready[i]);
                    }
                    if (score > best_score) {
                        best_score = score;
                        best_index = i;
                    }
                }
            } else {
                auto best_it = std::min_element(ready.begin(), ready.end());
                best_index = static_cast<int>(std::distance(ready.begin(), best_it));
            }
            std::string node_id = ready[best_index];
            ready.erase(ready.begin() + best_index);
            RuntimeNode& node = nodes[node_id];
            int rfu_id = choose_rfu(rfus, free_ids, node.micro.template_family, mode, node_id);
            free_ids.erase(std::remove(free_ids.begin(), free_ids.end(), rfu_id), free_ids.end());
            scheduler_decisions += 1.0;
            kappa_hits += ready_kappa;
            scheduler_template_hits += ready_template;
            valid_hits += ready_valid;

            long long base = base_latency_cycles(node.micro);
            long long mem = node.micro.estimated_bytes > 0 ? std::max(1LL, node.micro.estimated_bytes / kMemoryBandwidthBytesPerCycle) : 0LL;
            bool queued = std::find(prefetch_queue.begin(), prefetch_queue.end(), node.micro.template_family) != prefetch_queue.end();
            bool prefetched = mode_uses_prefetch(mode) && queued;
            bool reused_result = false;
            bool structural_reuse = false;
            bool structural_reuse_counted = false;
            auto path_it = path_bindings.find(node.micro.base_micro_id);
            PathBinding path_binding = path_it != path_bindings.end() ? path_it->second : PathBinding{};
            auto neg = negative_cache.find(node.micro.semantic_hash);
            bool blocked = neg != negative_cache.end() && neg->second >= clock;
            if (mode_has_fvm_lookup(mode) && mode != "degraded" && !blocked) {
                reads += 1.0;
                residency_samples.push_back(current_bytes);
                auto it = fvm.find(node.micro.semantic_hash);
                bool allow_result_reuse = mode_allows_result_reuse(mode);
                if (mode == "secure_ct" && rfus[rfu_id].loaded_template != node.micro.template_family) {
                    allow_result_reuse = false;
                }
                if (it != fvm.end() && cache_fresh(it->second, clock) && allow_result_reuse) {
                    reused_result = mode != "secure_ct";
                    if (reused_result) {
                        result_reuse_hits += 1.0;
                    }
                    it->second.last_access_cycle = clock;
                    it->second.hits += 1;
                    if (mode == "morphaccel") {
                        base = 1;
                        mem = 1;
                    } else if (mode == "secure_ct") {
                        base = std::max(2LL, base / 2LL);
                        mem = std::max(2LL, mem / 2LL);
                    } else if (mode == "secure_elastic_no_mask") {
                        base = std::max(1LL, base / 2LL);
                        mem = std::max(1LL, mem / 2LL);
                    } else if (is_secure_elastic_mode(mode)) {
                        base = std::max(2LL, base / 3LL + 1LL);
                        mem = std::max(2LL, mem / 3LL + 1LL);
                    }
                } else {
                    redundant_reads += 1.0;
                    negative_cache[node.micro.semantic_hash] = clock + std::max(1LL, kBaseSwitchCycles / 2LL);
                }
            }

            long long sw = switch_cost(mode, rfus[rfu_id].loaded_template, node.micro.template_family, prefetched, queued, template_hits, template_misses, switch_events);
            if (mode == "multicore") {
                double factor = baseline_speed_factor(mode, node.micro.task_type);
                base = std::max(1LL, static_cast<long long>(std::llround(static_cast<double>(base) / std::max(0.1, factor))));
                mem += 3;
                sw = rfus[rfu_id].loaded_template == node.micro.template_family ? 0LL : std::max(1LL, kBaseSwitchCycles / 3LL);
            } else if (mode == "gpu_tensor") {
                double factor = baseline_speed_factor(mode, node.micro.task_type);
                base = std::max(1LL, static_cast<long long>(std::llround(static_cast<double>(base) / std::max(0.1, factor))));
                mem += 6;
                sw = rfus[rfu_id].loaded_template == node.micro.template_family ? 0LL : std::max(1LL, kBaseSwitchCycles / 2LL);
            } else if (mode == "static_zkp" || mode == "eyeriss" || mode == "openaes") {
                double factor = baseline_speed_factor(mode, node.micro.task_type);
                base = std::max(1LL, static_cast<long long>(std::llround(static_cast<double>(base) / std::max(0.1, factor))));
                if (node.micro.task_type == "CNN" && mode == "eyeriss") {
                    mem = std::max(1LL, mem - 1LL);
                }
                sw = rfus[rfu_id].loaded_template == node.micro.template_family ? 0LL : std::max(1LL, kBaseSwitchCycles / 8LL);
            } else if (mode_uses_dedicated_pipeline(mode) && !reused_result) {
                double speed = dedicated_speedup(mode, node.micro.task_type);
                base = std::max(1LL, static_cast<long long>(std::llround(static_cast<double>(base) / std::max(0.1, speed))));
            }

            if (mode == "secure_ct") {
                base = std::max(base, base_latency_cycles(node.micro) + 2LL);
                mem = std::max(mem, 2LL);
            }
            if (is_secure_elastic_mode(mode)) {
                sw = sw > 0 ? std::max(1LL, sw / 2LL) : 0LL;
                if (mode == "secure_elastic_no_pinning") {
                    sw += std::max(1LL, kBaseSwitchCycles / 5LL);
                }
                if (mode == "secure_elastic_no_bucket") {
                    mem += 2LL;
                }
                if (mode == "secure_elastic_no_epoch") {
                    mem += 1LL;
                }
                if (reused_result && mode == "secure_elastic_no_mask") {
                    structural_reuse = true;
                    structural_reuse_hits += 1.0;
                    structural_reuse_counted = true;
                }
                base = std::max(base, std::max(2LL, base_latency_cycles(node.micro) / 2LL));
                mem = std::max(mem, 2LL);
            }
            if (sw == 0 && rfus[rfu_id].loaded_template == node.micro.template_family) {
                structural_reuse = true;
                if (!structural_reuse_counted) {
                    structural_reuse_hits += 1.0;
                    structural_reuse_counted = true;
                }
            }
            long long duration = std::max(1LL, base + mem + sw);
            if (prefetched) {
                prefetch_earliness_total += static_cast<double>(kBaseSwitchCycles);
                prefetch_earliness_count += 1.0;
            }
            node.ready = false;
            node.running = true;
            node.start_cycle = clock;
            node.finish_cycle = clock + duration;
            rfus[rfu_id].loaded_template = node.micro.template_family;
            rfus[rfu_id].busy_until = node.finish_cycle;
            rfus[rfu_id].executed_cycles += duration;
            running.push(RunningItem{node.finish_cycle, node_id, rfu_id});
            EventRow dispatch;
            dispatch.values = {
                {"event", "dispatch"},
                {"cycle", std::to_string(clock)},
                {"finish_cycle", std::to_string(node.finish_cycle)},
                {"runtime_micro_id", node_id},
                {"task_runtime_id", node.micro.runtime_id},
                {"task_type", node.micro.task_type},
                {"template_family", node.micro.template_family},
                {"reused_result", reused_result ? "1" : "0"},
                {"structural_reuse", structural_reuse ? "1" : "0"},
                {"template_prefetched", prefetched ? "1" : "0"},
                {"switch_cycles", std::to_string(sw)},
                {"path_id", path_binding.path_id},
                {"path_reuse_count", std::to_string(path_binding.reuse_count)},
                {"rfu_id", std::to_string(rfu_id)},
            };
            events.push_back(std::move(dispatch));
        }

        if (!running.empty()) {
            long long next_finish = running.top().finish_cycle;
            clock = std::max(clock, next_finish);
            std::vector<RunningItem> batch;
            while (!running.empty() && running.top().finish_cycle == next_finish) {
                batch.push_back(running.top());
                running.pop();
            }
            for (const auto& item : batch) {
                RuntimeNode& node = nodes[item.runtime_micro_id];
                node.running = false;
                node.completed = true;
                completed_nodes += 1;
                if (mode_has_fvm_write(mode)) {
                    write_cache(node.micro, item.finish_cycle, fvm, fvm_order, current_bytes, peak_bytes, allocations, writes, residency_samples, frees);
                }
                for (const auto& succ_id : node.succs) {
                    RuntimeNode& succ = nodes[succ_id];
                    succ.deps.erase(item.runtime_micro_id);
                    if (std::find(prefetch_queue.begin(), prefetch_queue.end(), succ.micro.template_family) == prefetch_queue.end()) {
                        if (prefetch_queue.size() >= 8) {
                            prefetch_queue.pop_front();
                        }
                        prefetch_queue.push_back(succ.micro.template_family);
                        prefetch_events += 1.0;
                    }
                    if (!succ.completed && !succ.ready && !succ.running && succ.deps.empty() && succ.arrival_cycle <= item.finish_cycle) {
                        succ.ready = true;
                        succ.ready_cycle = item.finish_cycle;
                        ready.push_back(succ_id);
                    }
                }
                task_completed_count[node.micro.runtime_id] += 1;
                if (task_completed_count[node.micro.runtime_id] == task_total_count[node.micro.runtime_id]) {
                    task_finish[node.micro.runtime_id] = item.finish_cycle;
                }
                EventRow complete;
                complete.values = {
                    {"event", "complete"},
                    {"cycle", std::to_string(item.finish_cycle)},
                    {"runtime_micro_id", item.runtime_micro_id},
                    {"task_runtime_id", node.micro.runtime_id},
                    {"task_type", node.micro.task_type},
                    {"rfu_id", std::to_string(item.rfu_id)},
                };
                events.push_back(std::move(complete));
            }
        } else {
            long long next_arrival = std::numeric_limits<long long>::max();
            for (const auto& [id, node] : nodes) {
                if (!node.completed && !node.ready && node.arrival_cycle > clock) {
                    next_arrival = std::min(next_arrival, node.arrival_cycle);
                }
            }
            if (next_arrival == std::numeric_limits<long long>::max()) {
                break;
            }
            if (ready.empty() && next_arrival > clock) {
                pipeline_stall_cycles += static_cast<double>(next_arrival - clock);
            }
            clock = std::max(clock, next_arrival);
        }
    }

    long long makespan = 0;
    for (const auto& [task_id, finish] : task_finish) {
        makespan = std::max(makespan, finish);
    }
    long long executed_cycles = 0;
    for (const auto& rfu : rfus) {
        executed_cycles += rfu.executed_cycles;
    }
    double utilization = makespan > 0 ? static_cast<double>(executed_cycles) / static_cast<double>(makespan * kRfuCount) : 0.0;
    double residency_avg = 0.0;
    if (!residency_samples.empty()) {
        long long sum = 0;
        for (long long value : residency_samples) {
            sum += value;
        }
        residency_avg = static_cast<double>(sum) / static_cast<double>(residency_samples.size());
    }
    double switch_cycles_total = 0.0;
    for (const auto& event : events) {
        auto evt = event.values.find("event");
        if (evt != event.values.end() && evt->second == "dispatch") {
            auto sw = event.values.find("switch_cycles");
            if (sw != event.values.end() && !sw->second.empty()) {
                switch_cycles_total += std::stod(sw->second);
            }
        }
    }
    double energy = compute_energy(mode, makespan, executed_cycles, reads, writes, scheduler_decisions, switch_events);
    double path_reuse_total = 0.0;
    double path_reuse_hits = 0.0;
    for (const auto& [id, binding] : path_bindings) {
        path_reuse_total += 1.0;
        if (binding.reuse_count > 1) {
            path_reuse_hits += 1.0;
        }
    }

    std::vector<MetricRow> metrics = {
        {"mode", 0.0},
        {"makespan_cycles", static_cast<double>(makespan)},
        {"makespan_ms", static_cast<double>(makespan) / static_cast<double>(kCyclesPerMs)},
        {"dispatches", scheduler_decisions},
        {"result_reuse_hits", result_reuse_hits},
        {"structural_reuse_hits", structural_reuse_hits},
        {"result_reuse_rate", scheduler_decisions > 0.0 ? result_reuse_hits / scheduler_decisions : 0.0},
        {"structural_reuse_rate", scheduler_decisions > 0.0 ? structural_reuse_hits / scheduler_decisions : 0.0},
        {"switch_cycles_total", switch_cycles_total},
        {"rfu_utilization", utilization},
        {"energy_nj", energy},
        {"fvm_peak_bytes", static_cast<double>(peak_bytes)},
        {"fvm_avg_residency_bytes", residency_avg},
        {"fvm_reads", reads},
        {"fvm_writes", writes},
        {"fvm_redundant_reads", redundant_reads},
        {"fvm_allocations", allocations},
        {"fvm_frees", frees},
        {"lrc_template_hits", template_hits},
        {"lrc_template_misses", template_misses},
        {"lrc_switch_events", switch_events},
        {"lrc_prefetch_events", prefetch_events},
        {"scheduler_decisions", scheduler_decisions},
        {"scheduler_prefetch_mean_earliness_cycles", prefetch_earliness_count > 0.0 ? prefetch_earliness_total / prefetch_earliness_count : 0.0},
        {"scheduler_pipeline_stall_cycles", pipeline_stall_cycles},
        {"kappa_key_match_ratio", ready_probe_count > 0.0 ? kappa_hits / ready_probe_count : 0.0},
        {"template_hit_ratio", ready_probe_count > 0.0 ? scheduler_template_hits / ready_probe_count : 0.0},
        {"valid_fragment_ratio", ready_probe_count > 0.0 ? valid_hits / ready_probe_count : 0.0},
        {"buffer_ops", allocations + frees},
        {"representative_path_count", static_cast<double>(paths.size())},
        {"representative_path_reuse_rate", path_reuse_total > 0.0 ? path_reuse_hits / path_reuse_total : 0.0},
    };

    return SimulationResult{metrics, events};
}

}
