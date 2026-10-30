#pragma once
#include "types.hpp"
#include <vector>
#include <string>

class Exporter {
public:
    static void print_terminal_report(const std::vector<AnomalyRecord>& anomalies, int time_step, double cpu_ms, double gpu_ms);
    static bool save_csv_report(const std::string& filename, const std::vector<AnomalyRecord>& anomalies);
};