/**
 * GEOTHERMAL // CPU ANALYTICS & MEMORY CONTROLLER
 * Asignación alineada AVX2, carga SoA y cálculo paralelo de medias móviles y Z-Scores.
 */

#include "cpu_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <omp.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// =============================================================================
// 1. CONFIGURACIÓN DEL SISTEMA Y GESTIÓN DE HILOS
// =============================================================================

// Establece alta prioridad de proceso y escala a todos los núcleos lógicos disponibles
void CPUEngine::init_system_affinity() {
    omp_set_num_threads(omp_get_num_procs());
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
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

// Lee el archivo CSV completo en RAM y parsea en paralelo con OpenMP + std::from_chars
bool CPUEngine::load_csv_dataset(const std::string& filepath, ClimateTensorHost& host) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[-] Error: No se pudo abrir el archivo de dataset: " << filepath << "\n";
        return false;
    }

    std::streamsize file_size = file.tellg();
    if (file_size <= 0) return false;
    file.seekg(0, std::ios::beg);

    std::string buffer(static_cast<size_t>(file_size), '\0');
    if (!file.read(buffer.data(), file_size)) {
        std::cerr << "[-] Error al leer el archivo: " << filepath << "\n";
        return false;
    }

    const char* buf_start = buffer.data();
    const char* buf_end   = buf_start + buffer.size();

    // Descartar fila de encabezados
    const char* data_start = static_cast<const char*>(std::memchr(buf_start, '\n', static_cast<size_t>(buf_end - buf_start)));
    if (!data_start) return false;
    ++data_start;

    size_t data_len = static_cast<size_t>(buf_end - data_start);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        const char* p = data_start + (data_len * tid) / nthreads;
        const char* end = data_start + (data_len * (tid + 1)) / nthreads;

        if (tid > 0) {
            while (p < buf_end && *p != '\n') ++p;
            if (p < buf_end) ++p;
        }
        if (tid < nthreads - 1) {
            while (end < buf_end && *end != '\n') ++end;
            if (end < buf_end) ++end;
        } else {
            end = buf_end;
        }

        while (p < end) {
            const char* line_end = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
            const char* next_line = line_end ? (line_end + 1) : end;
            const char* le = line_end ? line_end : end;
            while (le > p && (le[-1] == '\r' || le[-1] == ' ')) --le;

            if (p < le) {
                int time_idx = 0, cell_id = 0;
                float lat = 0.0f, lon = 0.0f, temp = 0.0f, u_val = 1.5f, v_val = 1.0f;

                auto r1 = std::from_chars(p, le, time_idx);
                if (r1.ec == std::errc{} && r1.ptr < le && *r1.ptr == ',') {
                    auto r2 = std::from_chars(r1.ptr + 1, le, cell_id);
                    if (r2.ec == std::errc{} && r2.ptr < le && *r2.ptr == ',') {
                        auto r3 = std::from_chars(r2.ptr + 1, le, lat);
                        if (r3.ec == std::errc{} && r3.ptr < le && *r3.ptr == ',') {
                            auto r4 = std::from_chars(r3.ptr + 1, le, lon);
                            if (r4.ec == std::errc{} && r4.ptr < le && *r4.ptr == ',') {
                                auto r5 = std::from_chars(r4.ptr + 1, le, temp);
                                if (r5.ec == std::errc{}) {
                                    if (r5.ptr < le && *r5.ptr == ',') {
                                        auto r6 = std::from_chars(r5.ptr + 1, le, u_val);
                                        if (r6.ec == std::errc{} && r6.ptr < le && *r6.ptr == ',') {
                                            std::from_chars(r6.ptr + 1, le, v_val);
                                        }
                                    }

                                    if (cell_id >= 0 && cell_id < TOTAL_CELLS && time_idx >= 0 && time_idx < TIME_STEPS) {
                                        if (time_idx == 0) {
                                            host.lats[cell_id] = lat;
                                            host.lons[cell_id] = lon;
                                        }
                                        size_t offset = static_cast<size_t>(time_idx) * TOTAL_CELLS + cell_id;
                                        host.temps[offset] = temp;
                                        host.wind_u[offset] = u_val;
                                        host.wind_v[offset] = v_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            p = next_line;
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

// Calcula media móvil temporal, desviación típica y puntuación Z con ventana deslizante O(1)
void CPUEngine::compute_time_series(ClimateTensorHost& host, int window_size, float z_threshold,
                                    std::vector<AnomalyRecord>& anomalies_out) {
    anomalies_out.clear();

    // Contenedores privados por hilo para evitar contención de bloqueos críticos
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<AnomalyRecord>> thread_anomalies(max_threads);
    const double inv_window = 1.0 / static_cast<double>(window_size);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int c = 0; c < TOTAL_CELLS; ++c) {
            float cell_lat = host.lats[c];
            float cell_lon = host.lons[c];
            double sum = 0.0;
            double sum_sq = 0.0;

            for (int t = 0; t < TIME_STEPS; ++t) {
                size_t idx = static_cast<size_t>(t) * TOTAL_CELLS + c;
                double val = static_cast<double>(host.temps[idx]);

                sum += val;
                sum_sq += val * val;

                if (t >= window_size) {
                    double old_val = static_cast<double>(host.temps[static_cast<size_t>(t - window_size) * TOTAL_CELLS + c]);
                    sum -= old_val;
                    sum_sq -= old_val * old_val;
                }

                // Paso inicial de calentamiento para índices inferiores al tamaño de ventana
                if (t < window_size - 1) {
                    host.moving_avg[idx] = static_cast<float>(val);
                    host.z_scores[idx] = 0.0f;
                    continue;
                }

                double mean = sum * inv_window;
                double var = std::max(0.0, (sum_sq * inv_window) - (mean * mean));
                float sigma = static_cast<float>(std::sqrt(var));
                float mean_f = static_cast<float>(mean);

                // Normalización de Z-Score
                float z = (sigma > 1e-4f) ? ((host.temps[idx] - mean_f) / sigma) : 0.0f;

                host.moving_avg[idx] = mean_f;
                host.z_scores[idx] = z;

                // Filtrado de eventos fuera de umbral estadístico
                if (std::abs(z) >= z_threshold) {
                    thread_anomalies[tid].push_back({
                        c, cell_lat, cell_lon, t,
                        host.temps[idx], mean_f, sigma, z
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