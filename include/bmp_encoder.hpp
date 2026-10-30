#pragma once
#include "types.hpp"
#include <vector>
#include <cstdint>

class BMPEncoder {
public:
    static std::vector<uint8_t> encode_rgba_to_bmp(const uchar4_pixel* pixels, int width, int height) {
        int row_size = width * 4;
        int image_size = row_size * height;
        int file_size = 54 + image_size;

        std::vector<uint8_t> bmp(file_size);

        // Cabecera BMP (14 bytes)
        bmp[0] = 'B';
        bmp[1] = 'M';
        *reinterpret_cast<uint32_t*>(&bmp[2])  = file_size;
        *reinterpret_cast<uint32_t*>(&bmp[10]) = 54; // Offset hacia los píxeles

        // Cabecera DIB (BITMAPINFOHEADER - 40 bytes)
        *reinterpret_cast<uint32_t*>(&bmp[14]) = 40;
        *reinterpret_cast<int32_t*>(&bmp[18])  = width;
        *reinterpret_cast<int32_t*>(&bmp[22])  = -height; // Altura negativa = orientación top-down
        *reinterpret_cast<uint16_t*>(&bmp[26]) = 1;       // Planos de color
        *reinterpret_cast<uint16_t*>(&bmp[28]) = 32;      // 32 bits (RGBA)
        *reinterpret_cast<uint32_t*>(&bmp[30]) = 0;       // Sin compresión (BI_RGB)
        *reinterpret_cast<uint32_t*>(&bmp[34]) = image_size;

        // Copiar los píxeles convirtiendo RGBA a BGRA (orden nativo de BMP en Windows)
        uint8_t* dst = bmp.data() + 54;
        for (int i = 0; i < width * height; ++i) {
            dst[i * 4 + 0] = pixels[i].b;
            dst[i * 4 + 1] = pixels[i].g;
            dst[i * 4 + 2] = pixels[i].r;
            dst[i * 4 + 3] = pixels[i].a;
        }

        return bmp;
    }
};