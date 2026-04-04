#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace fuse {

using Row = std::unordered_map<std::string, std::string>;

std::vector<std::string> split(const std::string& value, char delim);
std::string trim(const std::string& value);
std::vector<Row> read_csv(const std::string& path);
void write_csv(const std::string& path, const std::vector<std::string>& header, const std::vector<std::vector<std::string>>& rows);

}
