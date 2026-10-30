"""
GEOTHERMAL // PROCESADOR DE ELEVACIONES BATCH (COPERNICUS DEM)
Itera por todas las regiones de config.json y convierte sus .tif a matrices .bin de 100x100.
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import tifffile
from scipy.ndimage import map_coordinates

def process_dem(tif_path, bounds, output_bin_path):
    GRID_W, GRID_H = 100, 100

    if not os.path.exists(tif_path):
        return False

    print(f"\n[+] Procesando: {tif_path}")

    # 1. Cargar matriz de datos
    try:
        with Image.open(tif_path) as img:
            data = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"[-] Error al decodificar con Pillow: {e}")
        return False

    data = np.where((data < -1000) | (data > 9000), 0.0, data).astype(np.float32)

    # 2. Extraer límites del ráster
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        tags = page.tags

        if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
            scale = tags["ModelPixelScaleTag"].value
            tiepoint = tags["ModelTiepointTag"].value
            dx, dy = scale[0], scale[1]
            origin_x, origin_y = tiepoint[3], tiepoint[4]
            h_img, w_img = data.shape

            tif_lon_min = origin_x
            tif_lon_max = origin_x + w_img * dx
            tif_lat_max = origin_y
            tif_lat_min = origin_y - h_img * dy
        else:
            h_img, w_img = data.shape
            tif_lat_min, tif_lat_max = bounds["lat_min"], bounds["lat_max"]
            tif_lon_min, tif_lon_max = bounds["lon_min"], bounds["lon_max"]

    # 3. Malla uniforme de interpolación
    target_lats = np.linspace(bounds["lat_min"], bounds["lat_max"], GRID_H)
    target_lons = np.linspace(bounds["lon_min"], bounds["lon_max"], GRID_W)

    pixel_c = (target_lons - tif_lon_min) / (tif_lon_max - tif_lon_min) * (w_img - 1)
    pixel_r = (tif_lat_max - target_lats) / (tif_lat_max - tif_lat_min) * (h_img - 1)

    mesh_r, mesh_c = np.meshgrid(pixel_r, pixel_c, indexing='ij')

    # 4. Interpolación bilineal
    sampled_elev = map_coordinates(data, [mesh_r, mesh_c], order=1, mode='nearest')
    sampled_elev = np.ascontiguousarray(sampled_elev, dtype=np.float32)

    # 5. Guardar binario crudo (40 KB)
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    with open(output_bin_path, "wb") as f:
        f.write(sampled_elev.tobytes())

    print(f"  [✓] Salida generada: {output_bin_path} ({os.path.getsize(output_bin_path)} bytes)")
    print(f"  * Rango: {sampled_elev.min():.1f}m a {sampled_elev.max():.1f}m")
    return True

def main():
    config_path = "config.json"
    if not os.path.exists(config_path):
        print("[-] Error: config.json no encontrado.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    regions = cfg.get("regions", {})
    processed_count = 0

    for reg_id, reg in regions.items():
        bounds = reg.get("bounds")
        if not bounds:
            continue

        bin_path = f"data/elevations/elevation_{reg_id}.bin"
        
        # Buscar .tif o .tiff
        tif_path = None
        for ext in (".tif", ".tiff"):
            candidate = f"data/elevations/dem_{reg_id}_copernicus{ext}"
            if os.path.exists(candidate):
                tif_path = candidate
                break

        if tif_path:
            if process_dem(tif_path, bounds, bin_path):
                processed_count += 1
        else:
            print(f"[!] Pendiente DEM para '{reg_id}': Colocar 'data/elevations/dem_{reg_id}.tif'")

    print(f"\n[=] Finalizado: {processed_count} regiones procesadas.")

if __name__ == "__main__":
    main()