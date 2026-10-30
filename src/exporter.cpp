#include "exporter.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

void Exporter::print_terminal_report(const std::vector<AnomalyRecord>& anomalies, int time_step, double cpu_ms, double gpu_ms) {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "         GEOTHERMAL-PARALLEL :: ANALISIS DE ANOMALIAS CLIMATICAS                " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << " [OpenMP] 4 Hilos Fisicos (AVX2)  : " << std::fixed << std::setprecision(2) << cpu_ms << " ms" << std::endl;
    std::cout << " [CUDA]   GTX 1650 Ti (Full HD)   : " << gpu_ms << " ms" << std::endl;
    std::cout << " [Paso Temporal Evaluado]         : T = " << time_step << " (Horas)" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << " ID_CELDA   LAT       LON      T_OBS(C)   MEDIA(C)   SIGMA(C)   Z-SCORE   ESTADO" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    int printed = 0;
    for (const auto& a : anomalies) {
        if (a.time_idx == time_step) {
            std::string estado = (a.z_score >= 3.0f) ? "CRITICO" : (a.z_score <= -3.0f ? "HELADA" : "MODERADO");
            std::cout << " " << std::setw(8) << a.cell_id
                      << "  " << std::setw(7) << a.lat
                      << "  " << std::setw(8) << a.lon
                      << "   " << std::setw(7) << a.t_obs
                      << "    " << std::setw(7) << a.t_mean
                      << "    " << std::setw(7) << a.sigma
                      << "   " << std::setw(7) << a.z_score
                      << "   " << estado << std::endl;
            if (++printed >= 10) break; // Mostrar las primeras 10 en terminal
        }
    }
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << " Total anomalias registradas en el dataset: " << anomalies.size() << std::endl;
    std::cout << "================================================================================\n" << std::endl;
}

bool Exporter::save_csv_report(const std::string& filename, const std::vector<AnomalyRecord>& anomalies) {
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "CELL_ID,LAT,LON,TIME_STEP,T_OBS,T_MEAN,SIGMA,Z_SCORE\n";
    for (const auto& a : anomalies) {
        out << a.cell_id << "," << a.lat << "," << a.lon << ","
            << a.time_idx << "," << a.t_obs << "," << a.t_mean << ","
            << a.sigma << "," << a.z_score << "\n";
    }
    return true;
}