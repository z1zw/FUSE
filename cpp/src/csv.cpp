#include "morphaccel_csv.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace morphaccel {

std::vector<std::string> split(const std::string& value, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delim)) {
        out.push_back(item);
    }
    if (!value.empty() && value.back() == delim) {
        out.push_back("");
    }
    return out;
}

std::string trim(const std::string& value) {
    std::size_t left = 0;
    std::size_t right = value.size();
    while (left < right && (value[left] == ' ' || value[left] == '\t' || value[left] == '\r' || value[left] == '\n')) {
        ++left;
    }
    while (right > left && (value[right - 1] == ' ' || value[right - 1] == '\t' || value[right - 1] == '\r' || value[right - 1] == '\n')) {
        --right;
    }
    return value.substr(left, right - left);
}

std::vector<Row> read_csv(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("failed to open csv: " + path);
    }
    std::string line;
    if (!std::getline(fin, line)) {
        return {};
    }
    std::vector<std::string> header = split(trim(line), ',');
    std::vector<Row> rows;
    while (std::getline(fin, line)) {
        if (trim(line).empty()) {
            continue;
        }
        std::vector<std::string> parts = split(line, ',');
        Row row;
        for (std::size_t i = 0; i < header.size(); ++i) {
            std::string value = i < parts.size() ? trim(parts[i]) : "";
            if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            row[header[i]] = value;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

void write_csv(const std::string& path, const std::vector<std::string>& header, const std::vector<std::vector<std::string>>& rows) {
    std::ofstream fout(path);
    if (!fout) {
        throw std::runtime_error("failed to open output csv: " + path);
    }
    for (std::size_t i = 0; i < header.size(); ++i) {
        fout << header[i];
        if (i + 1 < header.size()) {
            fout << ',';
        }
    }
    fout << '\n';
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            fout << row[i];
            if (i + 1 < row.size()) {
                fout << ',';
            }
        }
        fout << '\n';
    }
}

}
