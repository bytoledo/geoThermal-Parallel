#pragma once

#include <string>
#include <vector>
#include "types.hpp"

// =============================================================================
// MOTOR DE COMPUTO ANALITICO EN CPU (OPENMP + AVX2)
// =============================================================================

class CPUEngine {
public:
    // Configura la prioridad de proceso y asigna afinidad a núcleos físicos de CPU
    static void init_system_affinity();

    // Reserva memoria Host alineada a 32 bytes para tensores continuos en formato SoA
    static bool allocate_host_memory(ClimateTensorHost& host);

    // Libera los búfers alineados en RAM
    static void free_host_memory(ClimateTensorHost& host);

    // Lee el dataset CSV plano y puebla los arreglos espaciales y temporales
    static bool load_csv_dataset(const std::string& filepath, ClimateTensorHost& host);

    // Lee la matriz topográfica binaria (.bin) de 100x100 floats a host.elevations
    static bool load_elevation_binary(const std::string& filepath, ClimateTensorHost& host);

    // Computa media móvil temporal (SMA), desviación estándar y Z-Scores paralelizados con OpenMP
    static void compute_time_series(ClimateTensorHost& host, int window_size, float z_threshold,
                                    std::vector<AnomalyRecord>& anomalies_out);
};