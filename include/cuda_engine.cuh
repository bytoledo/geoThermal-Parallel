#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "types.hpp"

// =============================================================================
// 1. ESTRUCTURAS DE MEMORIA EN DISPOSITIVO (VRAM)
// =============================================================================

// Búfers de trabajo y tensores asignados en la GPU
struct ClimateTensorDevice {
    // Tensores volumétricos globales en VRAM [TIME_STEPS * TOTAL_CELLS]
    float* d_temperature  = nullptr;
    float* d_zscores      = nullptr;
    float* d_u            = nullptr;
    float* d_v            = nullptr;

    // Búfers de fotograma activo (Slice 2D: GRID_WIDTH * GRID_HEIGHT)
    float* d_slice_temp   = nullptr;
    float* d_slice_zscore = nullptr;
    float* d_elevation    = nullptr;

    // Framebuffer RGBA de salida a resolución completa (RENDER_WIDTH * RENDER_HEIGHT)
    uchar4_pixel* d_heatmap     = nullptr;
    uchar4_pixel* d_heatmap_out = nullptr;

    size_t size = 0;
};

// =============================================================================
// 2. INTERFAZ DE CONTROL DEL MOTOR CUDA (HOST API)
// =============================================================================

namespace CUDAEngine {
    // Reserva de búfers y framebuffers en VRAM
    bool init_device_memory(ClimateTensorDevice& dev);

    // Liberación segura de punteros de memoria de dispositivo
    void free_device_memory(ClimateTensorDevice& dev);
    
    // Transferencia asíncrona de un slice temporal de temperaturas a VRAM
    void upload_temperatures(ClimateTensorDevice& dev, const float* h_temps, cudaStream_t stream);

    // Transferencia asíncrona de un slice temporal de Z-Scores a VRAM
    void upload_zscores(ClimateTensorDevice& dev, const float* h_zscores, cudaStream_t stream);

    // Despacho del kernel de renderizado para el mapa térmico interpolado (8 rangos)
    void render_temperature_field(const ClimateTensorDevice& dev, cudaStream_t stream, float min_temp, float max_temp);

    // Despacho del kernel de renderizado para el mapa de severidad de anomalías
    void render_anomaly_field(const ClimateTensorDevice& dev, cudaStream_t stream);

    // Transferencia asíncrona de un slice temporal de elevaciones a VRAM
    void upload_elevations(ClimateTensorDevice& dev, const float* h_elevations, cudaStream_t stream);

    // Despacho del kernel de renderizado para el mapa de topografía
    void render_topography_field(const ClimateTensorDevice& dev, cudaStream_t stream, float min_elev, float max_elev);
}