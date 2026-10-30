"""
GEOTHERMAL // GENERADOR DE DATASET SINTÉTICO DE PRUEBA
Genera una serie de tiempo meteorológica y una cuadrícula de relieve
para permitir la compilación y ejecución inmediata del motor HPC al clonar el repo.
"""
import os
import numpy as np

GRID_W, GRID_H = 100, 100
TOTAL_CELLS = GRID_W * GRID_H
TIME_STEPS = 720  # 30 días * 24 horas

LAT_MIN, LAT_MAX = 18.0, 22.0
LON_MIN, LON_MAX = -105.0, -101.0

os.makedirs("data", exist_ok=True)
os.makedirs("data/elevations", exist_ok=True)

csv_path = os.path.join("data", "weather_colima.csv")
bin_path = os.path.join("data", "elevations", "elevation_colima_copernicus.bin")

# 1. Generar CSV de clima
print(f">>> Generando dataset climatico sintetico ({csv_path})...")
lats = np.linspace(LAT_MIN, LAT_MAX, GRID_H)
lons = np.linspace(LON_MIN, LON_MAX, GRID_W)
grid_lats, grid_lons = np.meshgrid(lats, lons, indexing="ij")
flat_lats = grid_lats.ravel()
flat_lons = grid_lons.ravel()

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("cell_id,lat,lon,time_idx,temp,wind_u,wind_v\n")
    for t in range(TIME_STEPS):
        diurnal = 5.5 * np.sin(2.0 * np.pi * t / 24.0)
        temps = 25.0 + diurnal + (flat_lats - LAT_MIN) * 1.5 - (flat_lons - LON_MIN) * 0.8
        u_wind = 2.0 + 0.6 * np.sin(t / 12.0)
        v_wind = 1.0 + 0.4 * np.cos(t / 12.0)

        for cid in range(TOTAL_CELLS):
            f.write(
                f"{cid},{flat_lats[cid]:.4f},{flat_lons[cid]:.4f},{t},"
                f"{temps[cid]:.2f},{u_wind:.2f},{v_wind:.2f}\n"
            )

print(f"[✓] CSV generado: {csv_path} ({os.path.getsize(csv_path) / (1024*1024):.1f} MB)")

# 2. Generar relieve si no existe (40,000 bytes)
if not os.path.exists(bin_path):
    print(f">>> Generando relieve sintetico ({bin_path})...")
    elev = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for y in range(GRID_H):
        for x in range(GRID_W):
            base = (x * 12.0) + (y * 20.0)
            dist_v = np.sqrt((x - 60)**2 + (y - 75)**2)
            volcan = 3800.0 * np.exp(-0.5 * (dist_v / 8.0)**2)
            elev[y, x] = max(0.0, base + volcan)

    with open(bin_path, "wb") as f:
        f.write(elev.tobytes())
    print(f"[✓] Relieve generado: {bin_path} ({os.path.getsize(bin_path)} bytes)")