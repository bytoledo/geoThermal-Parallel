#pragma once

#include <cstdint>
#include <cstddef>

// =============================================================================
// 1. CONSTANTES DIMENSIONALES DEL MOTOR HPC
// =============================================================================

// Resolución espacial de la cuadrícula geográfica
constexpr int GRID_WIDTH    = 100;
constexpr int GRID_HEIGHT   = 100;
constexpr int TOTAL_CELLS   = GRID_WIDTH * GRID_HEIGHT;

// Dimensión temporal (30 días en intervalos horarios continuos)
constexpr int TIME_STEPS    = 720;

// Resolución de salida del framebuffer acelerado por CUDA (1080p)
constexpr int RENDER_WIDTH  = 1920;
constexpr int RENDER_HEIGHT = 1080;

// =============================================================================
// 2. ESTRUCTURAS DE DATOS Y BÚFERS DE PÍXELES
// =============================================================================

// Representación binaria RGBA de 32 bits compatible con uchar4 de CUDA
struct uchar4_pixel {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

// Registro analítico de eventos anómalos (|Z| >= umbral)
struct AnomalyRecord {
    int cell_id;
    float lat;
    float lon;
    int time_idx;
    float t_obs;
    float t_mean;
    float sigma;
    float z_score;
};

// =============================================================================
// 3. ESTRUCTURA DE ARREGLOS (SoA) PARA VECTORIZACIÓN AVX2
// =============================================================================

// Tensores continuos en memoria Host alineados a 32/64 bytes
struct ClimateTensorHost {
    float* lats       = nullptr; // [TOTAL_CELLS]
    float* lons       = nullptr; // [TOTAL_CELLS]
    float* elevations = nullptr; // [TOTAL_CELLS]
    float* temps      = nullptr; // [TIME_STEPS * TOTAL_CELLS]
    float* moving_avg = nullptr; // [TIME_STEPS * TOTAL_CELLS]
    float* z_scores   = nullptr; // [TIME_STEPS * TOTAL_CELLS]
    float* wind_u     = nullptr; // [TIME_STEPS * TOTAL_CELLS]
    float* wind_v     = nullptr; // [TIME_STEPS * TOTAL_CELLS]
};