/**
 * GEOTHERMAL // CUDA COMPUTE & RENDERING PIPELINE
 * Kernels de interpolación bilineal 1080p y mapeo cromático en VRAM.
 */

#include "cuda_engine.cuh"
#include <device_launch_parameters.h>
#include <cmath>

// =============================================================================
// 1. HELPERS DISPOSITIVO Y PALETAS CROMÁTICAS
// =============================================================================

// Constructor inline de píxel RGBA de 32 bits
__device__ inline uchar4_pixel make_pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uchar4_pixel p;
    p.r = r; p.g = g; p.b = b; p.a = a;
    return p;
}

// Mapeo continuo de temperatura física a paleta térmica de 8 nodos (Violeta a Granate)
__device__ inline uchar4_pixel evaluate_direct_temperature(float temp, float min_temp, float max_temp) {
    if (isnan(temp) || isinf(temp)) {
        return { 0, 0, 0, 0 };
    }

    float norm = (temp - min_temp) / (max_temp - min_temp);
    norm = fminf(fmaxf(norm, 0.0f), 1.0f);

    uchar4_pixel color;
    color.a = 210;

    const float STEP = 1.0f / 7.0f;

    if (norm < STEP) {
        // Tramo 1: Magenta helado -> Violeta oscuro
        float f = norm / STEP;
        color.r = (uint8_t)((1.0f - f) * 235.0f + f * 147.0f);
        color.g = (uint8_t)((1.0f - f) * 200.0f + f * 51.0f);
        color.b = (uint8_t)((1.0f - f) * 254.0f + f * 234.0f);
    } else if (norm < 2.0f * STEP) {
        // Tramo 2: Violeta oscuro -> Azul ultramar
        float f = (norm - STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 147.0f + f * 30.0f);
        color.g = (uint8_t)((1.0f - f) * 51.0f  + f * 64.0f);
        color.b = (uint8_t)((1.0f - f) * 234.0f + f * 235.0f);
    } else if (norm < 3.0f * STEP) {
        // Tramo 3: Azul ultramar -> Cian brillante
        float f = (norm - 2.0f * STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 30.0f  + f * 6.0f);
        color.g = (uint8_t)((1.0f - f) * 64.0f  + f * 182.0f);
        color.b = (uint8_t)((1.0f - f) * 235.0f + f * 212.0f);
    } else if (norm < 4.0f * STEP) {
        // Tramo 4: Cian brillante -> Verde esmeralda
        float f = (norm - 3.0f * STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 6.0f   + f * 16.0f);
        color.g = (uint8_t)((1.0f - f) * 182.0f + f * 185.0f);
        color.b = (uint8_t)((1.0f - f) * 212.0f + f * 129.0f);
    } else if (norm < 5.0f * STEP) {
        // Tramo 5: Verde esmeralda -> Amarillo dorado
        float f = (norm - 4.0f * STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 16.0f  + f * 234.0f);
        color.g = (uint8_t)((1.0f - f) * 185.0f + f * 179.0f);
        color.b = (uint8_t)((1.0f - f) * 129.0f + f * 8.0f);
    } else if (norm < 6.0f * STEP) {
        // Tramo 6: Amarillo dorado -> Rojo vivo
        float f = (norm - 5.0f * STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 234.0f + f * 239.0f);
        color.g = (uint8_t)((1.0f - f) * 179.0f + f * 68.0f);
        color.b = (uint8_t)((1.0f - f) * 8.0f   + f * 68.0f);
    } else {
        // Tramo 7: Rojo vivo -> Granate profundo
        float f = (norm - 6.0f * STEP) / STEP;
        color.r = (uint8_t)((1.0f - f) * 239.0f + f * 127.0f);
        color.g = (uint8_t)((1.0f - f) * 68.0f  + f * 29.0f);
        color.b = (uint8_t)((1.0f - f) * 68.0f  + f * 29.0f);
    }

    return color;
}

// Mapeo divergente de severidad estadística para Z-Scores (|Z| >= 2.0)
__device__ inline uchar4_pixel evaluate_anomaly_zscore(float z) {
    if (z >= 2.0f) {
        // Anomalía térmica positiva (ola de calor / actividad geotérmica)
        float t = fminf((z - 2.0f) / 1.6f, 1.0f);
        float r = 1.0f;
        float g = 0.1f * (1.0f - t);
        float b = 0.35f * (1.0f - t);
        uint8_t alpha = (uint8_t)(100 + t * 140);
        return make_pixel((uint8_t)(r * 255.0f), (uint8_t)(g * 255.0f), (uint8_t)(b * 255.0f), alpha);
    } else if (z <= -2.0f) {
        // Anomalía térmica negativa (ola de frío)
        float t = fminf((-z - 2.0f) / 1.6f, 1.0f);
        float r = 0.0f;
        float g = 0.85f;
        float b = 1.0f;
        uint8_t alpha = (uint8_t)(100 + t * 140);
        return make_pixel((uint8_t)(r * 255.0f), (uint8_t)(g * 255.0f), (uint8_t)(b * 255.0f), alpha);
    }

    // Régimen normal: píxel transparente
    return { 0, 0, 0, 0 };
}

// =============================================================================
// 2. KERNELS DE INTERPOLACIÓN ESPACIAL 2D
// =============================================================================

// Reescalado bilineal de la malla de temperatura a resolución de pantalla
__global__ void kernel_direct_temperature(
    const float* __restrict__ d_temps,
    uchar4_pixel* __restrict__ d_out,
    int in_w, int in_h,
    int out_w, int out_h,
    float min_temp, float max_temp)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= out_w || py >= out_h) return;

    // Mapeo de coordenadas con corrección vertical Sur -> Norte
    float gx = ((float)px / (float)(out_w - 1)) * (float)(in_w - 1);
    float gy = (1.0f - ((float)py / (float)(out_h - 1))) * (float)(in_h - 1);

    int x0 = (int)gx;
    int y0 = (int)gy;
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float dx = gx - (float)x0;
    float dy = gy - (float)y0;

    int idx00 = y0 * in_w + x0;
    int idx10 = y0 * in_w + x1;
    int idx01 = y1 * in_w + x0;
    int idx11 = y1 * in_w + x1;

    float t00 = d_temps[idx00];
    float t10 = d_temps[idx10];
    float t01 = d_temps[idx01];
    float t11 = d_temps[idx11];

    float interp_temp = (1.0f - dx) * (1.0f - dy) * t00 +
                               dx  * (1.0f - dy) * t10 +
                        (1.0f - dx) *        dy  * t01 +
                               dx  *        dy  * t11;

    d_out[py * out_w + px] = evaluate_direct_temperature(interp_temp, min_temp, max_temp);
}

// Reescalado bilineal del campo Z-Score a resolución de pantalla
__global__ void kernel_anomaly_zscore(
    const float* __restrict__ d_zscores,
    uchar4_pixel* __restrict__ d_out,
    int in_w, int in_h,
    int out_w, int out_h)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= out_w || py >= out_h) return;

    float gx = ((float)px / (float)(out_w - 1)) * (float)(in_w - 1);
    float gy = (1.0f - ((float)py / (float)(out_h - 1))) * (float)(in_h - 1);

    int x0 = (int)gx;
    int y0 = (int)gy;
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float dx = gx - (float)x0;
    float dy = gy - (float)y0;

    int idx00 = y0 * in_w + x0;
    int idx10 = y0 * in_w + x1;
    int idx01 = y1 * in_w + x0;
    int idx11 = y1 * in_w + x1;

    float interp_z = (1.0f - dx) * (1.0f - dy) * d_zscores[idx00] +
                            dx  * (1.0f - dy) * d_zscores[idx10] +
                     (1.0f - dx) *        dy  * d_zscores[idx01] +
                            dx  *        dy  * d_zscores[idx11];

    d_out[py * out_w + px] = evaluate_anomaly_zscore(interp_z);
}

// =============================================================================
// 3. KERNEL HIPSOMÉTRICO CON ILUMINACIÓN DIFUSA (HILLSHADE)
// =============================================================================

__device__ inline uchar4_pixel evaluate_hypsometry(float norm) {
    if (isnan(norm) || isinf(norm)) {
        return { 0, 0, 0, 0 };
    }
    norm = fminf(fmaxf(norm, 0.0f), 1.0f);
    uchar4_pixel c;
    c.a = 240;

    if (norm < 0.25f) {
        float f = norm / 0.25f;
        c.r = (uint8_t)((1.0f - f) * 46.0f + f * 115.0f);
        c.g = (uint8_t)((1.0f - f) * 117.0f + f * 144.0f);
        c.b = (uint8_t)((1.0f - f) * 89.0f  + f * 82.0f);
    } else if (norm < 0.55f) {
        float f = (norm - 0.25f) / 0.30f;
        c.r = (uint8_t)((1.0f - f) * 115.0f + f * 194.0f);
        c.g = (uint8_t)((1.0f - f) * 144.0f + f * 163.0f);
        c.b = (uint8_t)((1.0f - f) * 82.0f  + f * 94.0f);
    } else if (norm < 0.85f) {
        float f = (norm - 0.55f) / 0.30f;
        c.r = (uint8_t)((1.0f - f) * 194.0f + f * 120.0f);
        c.g = (uint8_t)((1.0f - f) * 163.0f + f * 80.0f);
        c.b = (uint8_t)((1.0f - f) * 94.0f  + f * 55.0f);
    } else {
        float f = (norm - 0.85f) / 0.15f;
        c.r = (uint8_t)((1.0f - f) * 120.0f + f * 240.0f);
        c.g = (uint8_t)((1.0f - f) * 80.0f  + f * 245.0f);
        c.b = (uint8_t)((1.0f - f) * 55.0f  + f * 255.0f);
    }
    return c;
}

__global__ void kernel_topography_hillshade(
    const float* __restrict__ d_elevation,
    uchar4_pixel* __restrict__ d_out,
    int in_w, int in_h,
    int out_w, int out_h,
    float min_elev, float max_elev)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= out_w || py >= out_h) return;

    float gx = ((float)px / (float)(out_w - 1)) * (float)(in_w - 1);
    float gy = (1.0f - ((float)py / (float)(out_h - 1))) * (float)(in_h - 1);

    int x0 = (int)gx;
    int y0 = (int)gy;
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float dx = gx - (float)x0;
    float dy = gy - (float)y0;

    float e00 = d_elevation[y0 * in_w + x0];
    float e10 = d_elevation[y0 * in_w + x1];
    float e01 = d_elevation[y1 * in_w + x0];
    float e11 = d_elevation[y1 * in_w + x1];

    float interp_elev = (1.0f - dx) * (1.0f - dy) * e00 +
                               dx  * (1.0f - dy) * e10 +
                        (1.0f - dx) *        dy  * e01 +
                               dx  *        dy  * e11;

    // Derivadas espaciales parciales
    float dz_dx = (e10 - e00) * (1.0f - dy) + (e11 - e01) * dy;
    float dz_dy = (e01 - e00) * (1.0f - dx) + (e11 - e10) * dx;

    // Vector normal a la superficie con factor de escala Z
    constexpr float Z_FACTOR = 0.007f;
    float nx = -dz_dx * Z_FACTOR;
    float ny = -dz_dy * Z_FACTOR;
    float nz = 1.0f;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= len; ny /= len; nz /= len;

    // Dirección solar (Azimut 315° NW, Altitud 45°)
    constexpr float lx = -0.5f;
    constexpr float ly = 0.5f;
    constexpr float lz = 0.707106f;

    float shade = fmaxf(0.0f, nx * lx + ny * ly + nz * lz);
    shade = 0.35f + 0.65f * shade;

    float norm = (interp_elev - min_elev) / fmaxf(1.0f, (max_elev - min_elev));
    uchar4_pixel col = evaluate_hypsometry(norm);

    col.r = (uint8_t)fminf(255.0f, col.r * shade);
    col.g = (uint8_t)fminf(255.0f, col.g * shade);
    col.b = (uint8_t)fminf(255.0f, col.b * shade);

    d_out[py * out_w + px] = col;
}

// =============================================================================
// 4. GESTIÓN DE VRAM Y EJECUCIÓN (HOST API)
// =============================================================================

// Reserva de tensores y framebuffers en memoria de dispositivo
bool CUDAEngine::init_device_memory(ClimateTensorDevice& dev) {
    size_t slice_bytes = TOTAL_CELLS * sizeof(float);
    size_t out_bytes   = RENDER_WIDTH * RENDER_HEIGHT * sizeof(uchar4);

    cudaError_t e1 = cudaMalloc((void**)&dev.d_slice_temp, slice_bytes);
    cudaError_t e2 = cudaMalloc((void**)&dev.d_slice_zscore, slice_bytes);
    cudaError_t e4 = cudaMalloc((void**)&dev.d_elevation, slice_bytes);
    cudaError_t e3 = cudaMalloc((void**)&dev.d_heatmap_out, out_bytes);
    dev.d_heatmap  = dev.d_heatmap_out;

    return (e1 == cudaSuccess && e2 == cudaSuccess && e3 == cudaSuccess && e4 == cudaSuccess);
}

// Liberación explícita de buffers en GPU
void CUDAEngine::free_device_memory(ClimateTensorDevice& dev) {
    if (dev.d_slice_temp)   { cudaFree(dev.d_slice_temp);   dev.d_slice_temp = nullptr; }
    if (dev.d_slice_zscore) { cudaFree(dev.d_slice_zscore); dev.d_slice_zscore = nullptr; }
    if (dev.d_elevation)    { cudaFree(dev.d_elevation);    dev.d_elevation = nullptr; }
    if (dev.d_heatmap_out)  { cudaFree(dev.d_heatmap_out);  dev.d_heatmap_out = nullptr; dev.d_heatmap = nullptr; }
}

// Transferencia asíncrona de datos escalares de temperatura a GPU
void CUDAEngine::upload_temperatures(ClimateTensorDevice& dev, const float* h_temps, cudaStream_t stream) {
    cudaMemcpyAsync(dev.d_slice_temp, h_temps, TOTAL_CELLS * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// Transferencia asíncrona de datos escalares de anomalías a GPU
void CUDAEngine::upload_zscores(ClimateTensorDevice& dev, const float* h_zscores, cudaStream_t stream) {
    cudaMemcpyAsync(dev.d_slice_zscore, h_zscores, TOTAL_CELLS * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// Despacho del kernel de renderizado para el mapa de temperatura
void CUDAEngine::render_temperature_field(const ClimateTensorDevice& dev, cudaStream_t stream, float min_temp, float max_temp) {
    dim3 threads(16, 16);
    dim3 blocks((RENDER_WIDTH + threads.x - 1) / threads.x,
                (RENDER_HEIGHT + threads.y - 1) / threads.y);

    kernel_direct_temperature<<<blocks, threads, 0, stream>>>(
        dev.d_slice_temp, dev.d_heatmap_out,
        GRID_WIDTH, GRID_HEIGHT,
        RENDER_WIDTH, RENDER_HEIGHT,
        min_temp, max_temp
    );
}

// Despacho del kernel de renderizado para el mapa de anomalías
void CUDAEngine::render_anomaly_field(const ClimateTensorDevice& dev, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((RENDER_WIDTH + threads.x - 1) / threads.x,
                (RENDER_HEIGHT + threads.y - 1) / threads.y);

    kernel_anomaly_zscore<<<blocks, threads, 0, stream>>>(
        dev.d_slice_zscore, dev.d_heatmap_out,
        GRID_WIDTH, GRID_HEIGHT,
        RENDER_WIDTH, RENDER_HEIGHT
    );
}

// Transferencia asíncrona de un slice temporal de elevaciones a VRAM
void CUDAEngine::upload_elevations(ClimateTensorDevice& dev, const float* h_elevations, cudaStream_t stream) {
    cudaMemcpyAsync(dev.d_elevation, h_elevations, TOTAL_CELLS * sizeof(float), cudaMemcpyHostToDevice, stream);
}

// Despacho del kernel de renderizado para el mapa de topografía
void CUDAEngine::render_topography_field(const ClimateTensorDevice& dev, cudaStream_t stream, float min_elev, float max_elev) {
    dim3 threads(16, 16);
    dim3 blocks((RENDER_WIDTH + threads.x - 1) / threads.x,
                (RENDER_HEIGHT + threads.y - 1) / threads.y);

    kernel_topography_hillshade<<<blocks, threads, 0, stream>>>(
        dev.d_elevation, dev.d_heatmap_out,
        GRID_WIDTH, GRID_HEIGHT,
        RENDER_WIDTH, RENDER_HEIGHT,
        min_elev, max_elev
    );
}