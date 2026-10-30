/**
 * GEOTHERMAL-PARALLEL // MOTOR DE TELEMETRIA HPC
 * Servidor HTTP, gestión de memoria Host/Device, cómputo OpenMP y renderizado CUDA.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include "types.hpp"
#include "cpu_engine.hpp"
#include "cuda_engine.cuh"

using json = nlohmann::json;
namespace fs = std::filesystem;

// =============================================================================
// 1. ESTRUCTURAS DE CONFIGURACIÓN Y PARSEO
// =============================================================================

struct RegionConfig {
    std::string region_id    = "colima_copernicus";
    std::string region_name  = "Occidente de Mexico // Colima";
    std::string dataset_path = "data/weather_colima.csv";
    float lat_min            = 18.0f;
    float lat_max            = 22.0f;
    float lon_min            = -105.0f;
    float lon_max            = -101.0f;
    int grid_width           = 100;
    int grid_height          = 100;
    int time_steps           = 720;
};

RegionConfig load_configuration(const std::string& config_path) {
    RegionConfig cfg;
    std::ifstream file(config_path);
    if (!file.is_open()) {
        std::cout << "[!] config.json no encontrado. Usando defaults.\n";
        return cfg;
    }

    try {
        json j;
        file >> j;

        if (j.contains("active_region") && j.contains("regions")) {
            std::string active_id = j["active_region"].get<std::string>();
            if (j["regions"].contains(active_id)) {
                auto reg = j["regions"][active_id];
                cfg.region_id    = active_id;
                cfg.region_name  = reg.value("name", "Occidente de Mexico // Colima");
                cfg.dataset_path = reg.value("dataset_path", "data/weather_colima.csv");

                if (reg.contains("bounds")) {
                    cfg.lat_min = reg["bounds"].value("lat_min", 18.0f);
                    cfg.lat_max = reg["bounds"].value("lat_max", 22.0f);
                    cfg.lon_min = reg["bounds"].value("lon_min", -105.0f);
                    cfg.lon_max = reg["bounds"].value("lon_max", -101.0f);
                }
                if (reg.contains("grid")) {
                    cfg.grid_width  = reg["grid"].value("width", 100);
                    cfg.grid_height = reg["grid"].value("height", 100);
                }
                cfg.time_steps = reg.value("time_steps", 720);
                return cfg;
            }
        }

        cfg.region_name  = j.value("region_name", "Occidente de Mexico // Colima");
        cfg.dataset_path = j.value("dataset_path", "data/weather_colima.csv");
        if (j.contains("bounds")) {
            cfg.lat_min = j["bounds"].value("lat_min", 18.0f);
            cfg.lat_max = j["bounds"].value("lat_max", 22.0f);
            cfg.lon_min = j["bounds"].value("lon_min", -105.0f);
            cfg.lon_max = j["bounds"].value("lon_max", -101.0f);
        }
        if (j.contains("grid")) {
            cfg.grid_width  = j["grid"].value("width", 100);
            cfg.grid_height = j["grid"].value("height", 100);
        }
        cfg.time_steps = j.value("time_steps", 720);

    } catch (const std::exception& e) {
        std::cerr << "[-] Error al parsear config.json: " << e.what() << "\n";
    }
    return cfg;
}

// =============================================================================
// 2. REPORTES Y TELEMETRÍA DE RENDIMIENTO
// =============================================================================

void print_telemetry_report(const RegionConfig& config,
                            std::uintmax_t file_size_bytes,
                            double ram_mb,
                            double vram_mb,
                            double load_ms,
                            double compute_ms,
                            float gpu_render_ms,
                            size_t anomaly_count) {
    double file_size_mb = static_cast<double>(file_size_bytes) / (1024.0 * 1024.0);

    std::cout << "\n----------------------------------------------------------------------\n";
    std::cout << "                       METRICAS DEL SISTEMA                           \n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  [DATASET EN DISCO]       : " << config.dataset_path << "\n";
    std::cout << "  * Tamano en almacenamiento: " << file_size_mb << " MB (" << file_size_bytes << " bytes)\n";
    std::cout << "  * Dimension temporal     : " << config.time_steps << " pasos horarios\n";
    std::cout << "  * Elementos totales      : " << (TOTAL_CELLS * config.time_steps) << " registros (celda x hora)\n\n";

    std::cout << "  [MEMORIA EN OPERACION]   :\n";
    std::cout << "  * Huella en RAM (Host)   : " << ram_mb << " MB (Espacio SoA alineado AVX2)\n";
    std::cout << "  * Datos a procesar       : " << ram_mb << " MB (Equivale a la ocupacion en RAM)\n";
    std::cout << "  * Huella en VRAM (GPU)   : " << vram_mb << " MB (Framebuffers 1080p + Texturas)\n\n";

    std::cout << "  [RENDIMIENTO Y TIEMPOS]  :\n";
    std::cout << "  * Carga y parseo CSV     : " << load_ms << " ms (" << (load_ms / 1000.0) << " s)\n";
    std::cout << "  * Computo CPU (OpenMP)   : " << compute_ms << " ms (" << (compute_ms / 1000.0) << " s)\n";
    std::cout << "  * Render GPU por frame   : " << gpu_render_ms << " ms (~" 
              << static_cast<int>(1000.0f / std::max(0.001f, gpu_render_ms)) << " FPS teoricos en CUDA)\n";
    std::cout << "  * Anomalias detectadas   : " << anomaly_count << " eventos criticos (|Z| >= 2.5)\n";
    std::cout << "----------------------------------------------------------------------\n\n";
}

// =============================================================================
// 3. PUNTO DE ENTRADA Y SERVICIOS REST
// =============================================================================

int main() {
    std::cout << "\n======================================================================\n";
    std::cout << "          GEOTHERMAL-PARALLEL // MOTOR DE TELEMETRIA HPC              \n";
    std::cout << "======================================================================\n\n";

    CPUEngine::init_system_affinity();

    RegionConfig config = load_configuration("config.json");
    std::cout << "[CONFIG] Region: " << config.region_name << "\n";
    std::cout << "[CONFIG] Grid: " << config.grid_width << "x" << config.grid_height 
              << " (" << TOTAL_CELLS << " celdas) | Pasos Temporales: " << config.time_steps << "h\n";

    std::uintmax_t file_size_bytes = 0;
    try {
        if (fs::exists(config.dataset_path)) {
            file_size_bytes = fs::file_size(config.dataset_path);
        }
    } catch (...) {}

    ClimateTensorHost host;
    if (!CPUEngine::allocate_host_memory(host)) {
        std::cerr << "[-] Error critico: fallo al reservar memoria host alineada para AVX2.\n";
        return 1;
    }

    size_t spatial_bytes   = static_cast<size_t>(TOTAL_CELLS) * sizeof(float);
    size_t tensor_bytes    = static_cast<size_t>(TOTAL_CELLS) * config.time_steps * sizeof(float);
    size_t total_ram_bytes = (spatial_bytes * 3) + (tensor_bytes * 5); // 3 tensores espaciales (lats, lons, elevations)
    double ram_mb          = static_cast<double>(total_ram_bytes) / (1024.0 * 1024.0);

    std::cout << ">>> Leyendo dataset en memoria...\n";
    auto t0_load = std::chrono::high_resolution_clock::now();
    if (!CPUEngine::load_csv_dataset(config.dataset_path, host)) {
        std::cerr << "[-] Error al cargar dataset: " << config.dataset_path << "\n";
        CPUEngine::free_host_memory(host);
        return 1;
    }
    auto t1_load = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1_load - t0_load).count();

    // Carga de la matriz de elevación inicial para la región activa
    std::string init_elev_path = "data/elevations/elevation_" + config.region_id + ".bin";
    CPUEngine::load_elevation_binary(init_elev_path, host);

    std::cout << ">>> Procesando analisis temporal y Z-Scores (OpenMP)...\n";
    std::vector<AnomalyRecord> global_anomalies;
    auto t0_compute = std::chrono::high_resolution_clock::now();
    CPUEngine::compute_time_series(host, 24, 2.5f, global_anomalies);
    auto t1_compute = std::chrono::high_resolution_clock::now();
    double compute_ms = std::chrono::duration<double, std::milli>(t1_compute - t0_compute).count();

    ClimateTensorDevice dev;
    if (!CUDAEngine::init_device_memory(dev)) {
        std::cerr << "[-] Error al reservar VRAM en GPU.\n";
        CPUEngine::free_host_memory(host);
        return 1;
    }

    size_t vram_bytes = (static_cast<size_t>(TOTAL_CELLS) * 3 * sizeof(float)) + 
                        (static_cast<size_t>(RENDER_WIDTH) * RENDER_HEIGHT * sizeof(uchar4_pixel));
    double vram_mb    = static_cast<double>(vram_bytes) / (1024.0 * 1024.0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<uchar4_pixel> h_framebuffer(RENDER_WIDTH * RENDER_HEIGHT);

    // Subida inicial de la matriz de elevación a VRAM
    CUDAEngine::upload_elevations(dev, host.elevations, stream);
    cudaStreamSynchronize(stream);

    // Medición de rendimiento en GPU para un fotograma 1080p
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    CUDAEngine::upload_temperatures(dev, &host.temps[0], stream);
    cudaEventRecord(start_gpu, stream);
    CUDAEngine::render_temperature_field(dev, stream, -25.0f, 50.0f);
    cudaEventRecord(stop_gpu, stream);
    cudaEventSynchronize(stop_gpu);

    float gpu_render_ms = 0.0f;
    cudaEventElapsedTime(&gpu_render_ms, start_gpu, stop_gpu);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    print_telemetry_report(config, file_size_bytes, ram_mb, vram_mb, load_ms, compute_ms, gpu_render_ms, global_anomalies.size());

    httplib::Server svr;
    svr.set_mount_point("/", "./web");

    svr.Get("/api/metadata", [&](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"region_id", config.region_id},
            {"region_name", config.region_name},
            {"dataset_path", config.dataset_path},
            {"lat_min", config.lat_min},
            {"lat_max", config.lat_max},
            {"lon_min", config.lon_min},
            {"lon_max", config.lon_max},
            {"grid_width", config.grid_width},
            {"grid_height", config.grid_height},
            {"time_steps", config.time_steps}
        };
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(j.dump(), "application/json");
    });

    svr.Get("/api/heatmap/temperature", [&](const httplib::Request& req, httplib::Response& res) {
        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        CUDAEngine::upload_temperatures(dev, &host.temps[t * TOTAL_CELLS], stream);
        CUDAEngine::render_temperature_field(dev, stream, -25.0f, 50.0f);

        cudaMemcpyAsync(h_framebuffer.data(), dev.d_heatmap_out,
                        RENDER_WIDTH * RENDER_HEIGHT * sizeof(uchar4_pixel),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(reinterpret_cast<const char*>(h_framebuffer.data()),
                        h_framebuffer.size() * sizeof(uchar4_pixel),
                        "application/octet-stream");
    });

    // Endpoint: Búfer RGBA binario renderizado en CUDA para mapa de topografía
    svr.Get("/api/heatmap/topography", [&](const httplib::Request&, httplib::Response& res) {
        float min_e = 1e9f, max_e = -1e9f;
        for (int i = 0; i < TOTAL_CELLS; ++i) {
            float val = host.elevations[i];
            if (val < min_e) min_e = val;
            if (val > max_e) max_e = val;
        }

        CUDAEngine::render_topography_field(dev, stream, min_e, max_e);

        cudaMemcpyAsync(h_framebuffer.data(), dev.d_heatmap_out,
                        RENDER_WIDTH * RENDER_HEIGHT * sizeof(uchar4_pixel),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(reinterpret_cast<const char*>(h_framebuffer.data()),
                        h_framebuffer.size() * sizeof(uchar4_pixel),
                        "application/octet-stream");
    });

    svr.Get("/api/heatmap/anomalies", [&](const httplib::Request& req, httplib::Response& res) {
        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        CUDAEngine::upload_zscores(dev, &host.z_scores[t * TOTAL_CELLS], stream);
        CUDAEngine::render_anomaly_field(dev, stream);

        cudaMemcpyAsync(h_framebuffer.data(), dev.d_heatmap_out,
                        RENDER_WIDTH * RENDER_HEIGHT * sizeof(uchar4_pixel),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(reinterpret_cast<const char*>(h_framebuffer.data()),
                        h_framebuffer.size() * sizeof(uchar4_pixel),
                        "application/octet-stream");
    });

    svr.Get("/api/probe", [&](const httplib::Request& req, httplib::Response& res) {
        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        if (req.has_param("cell_id")) {
            int cid = std::stoi(req.get_param_value("cell_id"));
            if (cid >= 0 && cid < TOTAL_CELLS) {
                int flat_idx = t * TOTAL_CELLS + cid;
                json j = {
                    {"cell_id", cid},
                    {"lat", host.lats[cid]},
                    {"lon", host.lons[cid]},
                    {"temp", host.temps[flat_idx]},
                    {"moving_avg", host.moving_avg[flat_idx]},
                    {"z_score", host.z_scores[flat_idx]},
                    {"wind_u", host.wind_u[flat_idx]},
                    {"wind_v", host.wind_v[flat_idx]},
                    {"elevation", host.elevations[cid]}
                };
                res.set_header("Access-Control-Allow-Origin", "*");
                res.set_content(j.dump(), "application/json");
                return;
            }
        }
        res.status = 400;
        res.set_content("{\"error\":\"cell_id no valido\"}", "application/json");
    });

    svr.Get("/api/anomalies", [&](const httplib::Request& req, httplib::Response& res) {
        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        json arr = json::array();
        for (int i = 0; i < TOTAL_CELLS; ++i) {
            int flat_idx = t * TOTAL_CELLS + i;
            float z = host.z_scores[flat_idx];
            if (std::abs(z) >= 2.0f) {
                arr.push_back({
                    {"cell_id", i},
                    {"lat", host.lats[i]},
                    {"lon", host.lons[i]},
                    {"t_obs", host.temps[flat_idx]},
                    {"t_mean", host.moving_avg[flat_idx]},
                    {"z_score", z}
                });
            }
        }
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(arr.dump(), "application/json");
    });

    svr.Get("/api/timeline", [&](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();
        for (int t = 0; t < config.time_steps; ++t) {
            int count = 0;
            float max_abs_z = 0.0f;
            for (int i = 0; i < TOTAL_CELLS; ++i) {
                float z = host.z_scores[t * TOTAL_CELLS + i];
                if (std::abs(z) >= 2.0f) {
                    count++;
                    if (std::abs(z) > max_abs_z) max_abs_z = std::abs(z);
                }
            }
            arr.push_back({{"time", t}, {"count", count}, {"max_z", max_abs_z}});
        }
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(arr.dump(), "application/json");
    });

    svr.Get("/api/wind", [&](const httplib::Request& req, httplib::Response& res) {
        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        std::vector<float> wind_buffer(TOTAL_CELLS * 2);
        size_t offset = static_cast<size_t>(t) * TOTAL_CELLS;
        
        for (int i = 0; i < TOTAL_CELLS; ++i) {
            wind_buffer[i * 2 + 0] = host.wind_u[offset + i];
            wind_buffer[i * 2 + 1] = host.wind_v[offset + i];
        }

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(reinterpret_cast<const char*>(wind_buffer.data()),
                        wind_buffer.size() * sizeof(float),
                        "application/octet-stream");
    });

    auto haversine_km = [](float lat1, float lon1, float lat2, float lon2) -> float {
        constexpr float R = 6371.0f;
        constexpr float DEG2RAD = 3.14159265358979323846f / 180.0f;
        float dlat = (lat2 - lat1) * DEG2RAD;
        float dlon = (lon2 - lon1) * DEG2RAD;
        float a = std::sin(dlat * 0.5f) * std::sin(dlat * 0.5f) +
                  std::cos(lat1 * DEG2RAD) * std::cos(lat2 * DEG2RAD) *
                  std::sin(dlon * 0.5f) * std::sin(dlon * 0.5f);
        return 2.0f * R * std::asin(std::sqrt(a));
    };

    svr.Get("/api/slice", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("lat1") || !req.has_param("lon1") || 
            !req.has_param("lat2") || !req.has_param("lon2")) {
            res.status = 400;
            res.set_content("{\"error\":\"Faltan coordenadas de origen o destino\"}", "application/json");
            return;
        }

        float lat1 = std::stof(req.get_param_value("lat1"));
        float lon1 = std::stof(req.get_param_value("lon1"));
        float lat2 = std::stof(req.get_param_value("lat2"));
        float lon2 = std::stof(req.get_param_value("lon2"));
        int samples = req.has_param("samples") ? std::clamp(std::stoi(req.get_param_value("samples")), 20, 200) : 100;

        int t = req.has_param("time") ? std::stoi(req.get_param_value("time")) : 0;
        t = std::clamp(t, 0, config.time_steps - 1);

        size_t t_offset = static_cast<size_t>(t) * TOTAL_CELLS;
        float total_dist = haversine_km(lat1, lon1, lat2, lon2);

        json points = json::array();

        for (int s = 0; s < samples; ++s) {
            float alpha = static_cast<float>(s) / static_cast<float>(samples - 1);
            float cur_lat = lat1 + alpha * (lat2 - lat1);
            float cur_lon = lon1 + alpha * (lon2 - lon1);
            float cur_dist = alpha * total_dist;

            float gx = ((cur_lon - config.lon_min) / (config.lon_max - config.lon_min)) * (config.grid_width - 1);
            float gy = ((cur_lat - config.lat_min) / (config.lat_max - config.lat_min)) * (config.grid_height - 1);

            float interp_temp = 0.0f;
            float interp_z    = 0.0f;
            float interp_elev = 0.0f;

            if (gx >= 0.0f && gx <= (config.grid_width - 1) && gy >= 0.0f && gy <= (config.grid_height - 1)) {
                int x0 = static_cast<int>(gx);
                int y0 = static_cast<int>(gy);
                int x1 = std::min(x0 + 1, config.grid_width - 1);
                int y1 = std::min(y0 + 1, config.grid_height - 1);

                float dx = gx - static_cast<float>(x0);
                float dy = gy - static_cast<float>(y0);

                int idx00 = y0 * config.grid_width + x0;
                int idx10 = y0 * config.grid_width + x1;
                int idx01 = y1 * config.grid_width + x0;
                int idx11 = y1 * config.grid_width + x1;

                interp_temp = (1.0f - dx) * (1.0f - dy) * host.temps[t_offset + idx00] +
                                     dx  * (1.0f - dy) * host.temps[t_offset + idx10] +
                              (1.0f - dx) *        dy  * host.temps[t_offset + idx01] +
                                     dx  *        dy  * host.temps[t_offset + idx11];

                interp_z    = (1.0f - dx) * (1.0f - dy) * host.z_scores[t_offset + idx00] +
                                     dx  * (1.0f - dy) * host.z_scores[t_offset + idx10] +
                              (1.0f - dx) *        dy  * host.z_scores[t_offset + idx01] +
                                     dx  *        dy  * host.z_scores[t_offset + idx11];

                if (host.elevations != nullptr) {
                    interp_elev = (1.0f - dx) * (1.0f - dy) * host.elevations[idx00] +
                                         dx  * (1.0f - dy) * host.elevations[idx10] +
                                  (1.0f - dx) *        dy  * host.elevations[idx01] +
                                         dx  *        dy  * host.elevations[idx11];
                }
            }

            points.push_back({
                {"dist_km",   std::round(cur_dist * 10.0f) / 10.0f},
                {"lat",       cur_lat},
                {"lon",       cur_lon},
                {"temp",      std::round(interp_temp * 10.0f) / 10.0f},
                {"z_score",   std::round(interp_z * 100.0f) / 100.0f},
                {"elevation", std::round(interp_elev)}
            });
        }

        json res_json = {
            {"total_dist_km", std::round(total_dist * 10.0f) / 10.0f},
            {"points", points}
        };

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(res_json.dump(), "application/json");
    });

    svr.Get("/api/regions", [&](const httplib::Request&, httplib::Response& res) {
        std::ifstream file("config.json");
        json j;
        if (file.is_open()) file >> j;
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_content(j.dump(), "application/json");
    });

    svr.Post("/api/regions/switch", [&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (!req.has_param("id")) {
            res.status = 400;
            res.set_content("{\"error\":\"Falta parametro id\"}", "application/json");
            return;
        }
        std::string target_id = req.get_param_value("id");

        std::ifstream file("config.json");
        if (!file.is_open()) {
            res.status = 500;
            res.set_content("{\"error\":\"No se pudo abrir config.json\"}", "application/json");
            return;
        }
        json j;
        file >> j;

        if (!j.contains("regions") || !j["regions"].contains(target_id)) {
            res.status = 404;
            res.set_content("{\"error\":\"Region no encontrada\"}", "application/json");
            return;
        }

        auto reg = j["regions"][target_id];
        config.region_id    = target_id;
        config.region_name  = reg["name"].get<std::string>();
        config.dataset_path = reg["dataset_path"].get<std::string>();
        config.lat_min      = reg["bounds"]["lat_min"];
        config.lat_max      = reg["bounds"]["lat_max"];
        config.lon_min      = reg["bounds"]["lon_min"];
        config.lon_max      = reg["bounds"]["lon_max"];

        std::cout << "\n======================================================\n";
        std::cout << ">>> [CAMBIO EN CALIENTE] Region: " << config.region_name << "\n";
        std::cout << ">>> Dataset: " << config.dataset_path << "\n";

        global_anomalies.clear();
        CPUEngine::load_csv_dataset(config.dataset_path, host);

        // Recarga de elevación para la nueva región y sincronización con VRAM
        std::string elev_path = "data/elevations/elevation_" + target_id + ".bin";
        CPUEngine::load_elevation_binary(elev_path, host);
        CUDAEngine::upload_elevations(dev, host.elevations, stream);
        cudaStreamSynchronize(stream);

        CPUEngine::compute_time_series(host, 24, 2.5f, global_anomalies);
        std::cout << "[✓] CPU (OpenMP): " << global_anomalies.size() << " anomalias detectadas.\n";

        CUDAEngine::upload_temperatures(dev, host.temps, stream);
        CUDAEngine::upload_zscores(dev, host.z_scores, stream);
        cudaStreamSynchronize(stream);
        std::cout << "[✓] CUDA: Frame inicial transferido a VRAM.\n";
        std::cout << "======================================================\n";

        j["active_region"] = target_id;
        std::ofstream out_file("config.json");
        out_file << j.dump(2);

        res.set_content("{\"status\":\"ok\",\"region\":\"" + config.region_name + "\"}", "application/json");
    });

    std::cout << ">>> Servidor HTTP corriendo en http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);

    cudaStreamDestroy(stream);
    CUDAEngine::free_device_memory(dev);
    CPUEngine::free_host_memory(host);

    return 0;
}