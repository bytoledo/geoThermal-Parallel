/**
 * GEOTHERMAL // CPU ANALYTICS & MEMORY CONTROLLER
 * Asignación alineada AVX2, carga SoA y cálculo paralelo de medias móviles y Z-Scores.
 */

#include "cpu_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <windows.h>

// =============================================================================
// 1. CONFIGURACIÓN DEL SISTEMA Y GESTIÓN DE HILOS
// =============================================================================

// Establece alta prioridad de proceso y fija afinidad a núcleos físicos (paso de 2 cores)
void CPUEngine::init_system_affinity() {
    omp_set_num_threads(4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        DWORD_PTR mask = (1ULL << (tid * 2));
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}

// =============================================================================
// 2. GESTIÓN DE MEMORIA ALINEADA HOST (ESTRUCTURA DE ARREGLOS - SoA)
// =============================================================================

// Reserva búfers alineados a 32 bytes en RAM para permitir vectorización AVX2
bool CPUEngine::allocate_host_memory(ClimateTensorHost& host) {
    size_t spatial_bytes     = TOTAL_CELLS * sizeof(float);
    size_t full_tensor_bytes = static_cast<size_t>(TOTAL_CELLS) * TIME_STEPS * sizeof(float);

    host.lats       = static_cast<float*>(_aligned_malloc(spatial_bytes, 32));
    host.lons       = static_cast<float*>(_aligned_malloc(spatial_bytes, 32));
    host.elevations = static_cast<float*>(_aligned_malloc(spatial_bytes, 32));
    host.temps      = static_cast<float*>(_aligned_malloc(full_tensor_bytes, 32));
    host.moving_avg = static_cast<float*>(_aligned_malloc(full_tensor_bytes, 32));
    host.z_scores   = static_cast<float*>(_aligned_malloc(full_tensor_bytes, 32));
    host.wind_u     = static_cast<float*>(_aligned_malloc(full_tensor_bytes, 32));
    host.wind_v     = static_cast<float*>(_aligned_malloc(full_tensor_bytes, 32));

    return (host.lats && host.lons && host.elevations && host.temps && host.moving_avg && 
            host.z_scores && host.wind_u && host.wind_v);
}

// Libera los arreglos alineados reservados en RAM
void CPUEngine::free_host_memory(ClimateTensorHost& host) {
    if (host.lats)       { _aligned_free(host.lats);       host.lats = nullptr; }
    if (host.lons)       { _aligned_free(host.lons);       host.lons = nullptr; }
    if (host.elevations) { _aligned_free(host.elevations); host.elevations = nullptr; }
    if (host.temps)      { _aligned_free(host.temps);      host.temps = nullptr; }
    if (host.moving_avg) { _aligned_free(host.moving_avg); host.moving_avg = nullptr; }
    if (host.z_scores)   { _aligned_free(host.z_scores);   host.z_scores = nullptr; }
    if (host.wind_u)     { _aligned_free(host.wind_u);     host.wind_u = nullptr; }
    if (host.wind_v)     { _aligned_free(host.wind_v);     host.wind_v = nullptr; }
}

// =============================================================================
// 3. CARGA Y SERIALIZACIÓN DE DATASETS
// =============================================================================

// Lee el archivo CSV plano y puebla los tensores climáticos en memoria
bool CPUEngine::load_csv_dataset(const std::string& filepath, ClimateTensorHost& host) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[-] Error: No se pudo abrir el archivo de dataset: " << filepath << "\n";
        return false;
    }

    std::string line;
    if (!std::getline(file, line)) return false; // Descartar fila de encabezados

    int time_idx = 0;
    int cell_id = 0;
    float lat = 0.0f, lon = 0.0f, temp = 0.0f, u_val = 0.0f, v_val = 0.0f;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Formato esperado: time_idx,cell_id,lat,lon,temperature,wind_u,wind_v
        int parsed = sscanf(line.c_str(), "%d,%d,%f,%f,%f,%f,%f", 
                            &time_idx, &cell_id, &lat, &lon, &temp, &u_val, &v_val);

        if (parsed >= 5) {
            if (cell_id >= 0 && cell_id < TOTAL_CELLS && time_idx >= 0 && time_idx < TIME_STEPS) {
                if (time_idx == 0) {
                    host.lats[cell_id] = lat;
                    host.lons[cell_id] = lon;
                }
                
                size_t offset = static_cast<size_t>(time_idx) * TOTAL_CELLS + cell_id;
                host.temps[offset] = temp;

                if (parsed == 7) {
                    host.wind_u[offset] = u_val;
                    host.wind_v[offset] = v_val;
                } else {
                    host.wind_u[offset] = 1.5f;
                    host.wind_v[offset] = 1.0f;
                }
            }
        }
    }
    return true;
}

// Lee la matriz topográfica binaria cruda (10,000 floats IEEE 754 = 40 KB)
bool CPUEngine::load_elevation_binary(const std::string& filepath, ClimateTensorHost& host) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[-] Advertencia: No se pudo abrir el relieve " << filepath << ". Rellenando elevacion con 0.\n";
        std::fill_n(host.elevations, TOTAL_CELLS, 0.0f);
        return false;
    }

    file.read(reinterpret_cast<char*>(host.elevations), TOTAL_CELLS * sizeof(float));
    if (!file) {
        std::cerr << "[-] Error: Se leyeron solo " << file.gcount() << " bytes de " << filepath << "\n";
        return false;
    }
    return true;
}

// =============================================================================
// 4. COMPUTO ANALÍTICO TEMPORAL (OPENMP + SIMD)
// =============================================================================

// Calcula media móvil temporal, desviación típica y puntuación Z sobre la ventana deslizante
void CPUEngine::compute_time_series(ClimateTensorHost& host, int window_size, float z_threshold,
                                    std::vector<AnomalyRecord>& anomalies_out) {
    anomalies_out.clear();

    // Contenedores privados por hilo para evitar contención de bloqueos críticos
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<AnomalyRecord>> thread_anomalies(max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int c = 0; c < TOTAL_CELLS; ++c) {
            float cell_lat = host.lats[c];
            float cell_lon = host.lons[c];

            for (int t = 0; t < TIME_STEPS; ++t) {
                size_t idx = static_cast<size_t>(t) * TOTAL_CELLS + c;

                // Paso inicial de calentamiento para índices inferiores al tamaño de ventana
                if (t < window_size - 1) {
                    host.moving_avg[idx] = host.temps[idx];
                    host.z_scores[idx] = 0.0f;
                    continue;
                }

                // Cálculo de media móvil
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int w = 0; w < window_size; ++w) {
                    sum += host.temps[(static_cast<size_t>(t) - w) * TOTAL_CELLS + c];
                }
                float mean = sum / static_cast<float>(window_size);

                // Cálculo de varianza y desviación típica
                float var_sum = 0.0f;
                #pragma omp simd reduction(+:var_sum)
                for (int w = 0; w < window_size; ++w) {
                    float diff = host.temps[(static_cast<size_t>(t) - w) * TOTAL_CELLS + c] - mean;
                    var_sum += diff * diff;
                }
                float sigma = std::sqrt(var_sum / static_cast<float>(window_size));

                // Normalización de Z-Score
                float z = (sigma > 1e-4f) ? ((host.temps[idx] - mean) / sigma) : 0.0f;

                host.moving_avg[idx] = mean;
                host.z_scores[idx] = z;

                // Filtrado de eventos fuera de umbral estadístico
                if (std::abs(z) >= z_threshold) {
                    thread_anomalies[tid].push_back({
                        c, cell_lat, cell_lon, t,
                        host.temps[idx], mean, sigma, z
                    });
                }
            }
        }
    }

    // Fusión lineal de los resultados recopilados por cada hilo
    for (const auto& local_list : thread_anomalies) {
        anomalies_out.insert(anomalies_out.end(), local_list.begin(), local_list.end());
    }
}