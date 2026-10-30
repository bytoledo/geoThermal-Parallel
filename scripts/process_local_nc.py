"""
GEOTHERMAL // NETCDF INGESTION & INTERPOLATION PIPELINE
Procesa datasets climáticos NetCDF (ERA5), interpola a malla regular 100x100
y exporta el tensor temporal serializado en formato CSV para el backend C++/CUDA.
"""

import os
import glob
import json
import numpy as np
import xarray as xr
import pandas as pd

# =============================================================================
# 1. CONSTANTES GLOBALES Y CONFIGURACIÓN DEL MOTOR
# =============================================================================

GRID_RES = 100
TIME_STEPS = 720  # 30 días x 24 horas continuas
CENTROID_TOLERANCE_DEG = 2.5


# =============================================================================
# 2. RESOLUCIÓN ESPACIAL Y ASIGNACIÓN GEOGRÁFICA
# =============================================================================

def match_region(nc_lats, nc_lons, config_regions):
    """
    Asigna el archivo NetCDF a una región de config.json comparando
    la distancia euclidiana entre centroides geográficos.
    """
    file_lat_center = (float(nc_lats.min()) + float(nc_lats.max())) / 2.0
    file_lon_center = (float(nc_lons.min()) + float(nc_lons.max())) / 2.0

    best_match_key = None
    min_dist = float("inf")

    for r_key, r_info in config_regions.items():
        bounds = r_info["bounds"]
        cfg_lat_center = (bounds["lat_min"] + bounds["lat_max"]) / 2.0
        cfg_lon_center = (bounds["lon_min"] + bounds["lon_max"]) / 2.0

        dist = np.hypot(file_lat_center - cfg_lat_center, file_lon_center - cfg_lon_center)
        if dist < min_dist:
            min_dist = dist
            best_match_key = r_key

    return best_match_key if min_dist < CENTROID_TOLERANCE_DEG else None


# =============================================================================
# 3. PROCESAMIENTO, INTERPOLACIÓN Y EXPORTACIÓN DE TENSORES
# =============================================================================

def process_single_nc(nc_path, r_key, r_info):
    """
    Estandariza coordenadas, rellena valores faltantes marinos,
    remuestrea bilinealmente a 100x100 y exporta el archivo CSV estructurado.
    """
    bounds = r_info["bounds"]
    out_csv = r_info["dataset_path"]
    name = r_info["name"]

    print("\n" + "=" * 56)
    print(f"[+] Archivo:  {os.path.basename(nc_path)}")
    print(f"[+] Region:   {name} (ID: {r_key})")
    print(f"[+] Destino:  {out_csv}")
    print("=" * 56)

    # 1. Carga del dataset multidimensional
    ds = xr.open_dataset(nc_path)

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    # Conversión de coordenadas de longitud 0..360 a estándar internacional -180..180
    if float(ds[lon_name].max()) > 180.0:
        ds = ds.assign_coords({lon_name: (((ds[lon_name] + 180) % 360) - 180)})
        ds = ds.sortby(lon_name)

    # Normalización del orden de ejes a formato ascendente Sur -> Norte y Oeste -> Este
    if float(ds[lat_name].values[0]) > float(ds[lat_name].values[-1]):
        ds = ds.reindex({lat_name: ds[lat_name][::-1]})

    if float(ds[lon_name].values[0]) > float(ds[lon_name].values[-1]):
        ds = ds.reindex({lon_name: ds[lon_name][::-1]})

    # 2. Detección de variables térmicas y vectoriales de viento
    temp_var = "t2m" if "t2m" in ds else ("temperature" if "temperature" in ds else list(ds.data_vars)[0])
    u_var = "u10" if "u10" in ds else ("wind_u" if "wind_u" in ds else None)
    v_var = "v10" if "v10" in ds else ("wind_v" if "wind_v" in ds else None)

    # Conversión de Kelvin a Celsius
    if float(ds[temp_var].mean(skipna=True)) > 100.0:
        ds[temp_var] = ds[temp_var] - 273.15

    # Difusión de valores terrestres hacia zonas marinas para evitar discontinuidades
    ds[temp_var] = ds[temp_var].interpolate_na(dim=lon_name, method="nearest", fill_value="extrapolate")
    ds[temp_var] = ds[temp_var].interpolate_na(dim=lat_name, method="nearest", fill_value="extrapolate")

    # 3. Interpolación espacial a la cuadrícula fija 100x100
    target_lats = np.linspace(bounds["lat_min"], bounds["lat_max"], GRID_RES)
    target_lons = np.linspace(bounds["lon_min"], bounds["lon_max"], GRID_RES)

    ds_interp = ds.interp(
        {lat_name: target_lats, lon_name: target_lons},
        method="linear",
        kwargs={"fill_value": "extrapolate"}
    )

    # Relleno de seguridad para celdas residuales
    mean_temp = float(ds_interp[temp_var].mean(skipna=True))
    ds_interp[temp_var] = ds_interp[temp_var].fillna(mean_temp)

    if u_var:
        ds_interp[u_var] = ds_interp[u_var].fillna(0.0)
    if v_var:
        ds_interp[v_var] = ds_interp[v_var].fillna(0.0)

    t_celsius = ds_interp[temp_var].values
    u_vals = ds_interp[u_var].values if u_var else np.zeros_like(t_celsius)
    v_vals = ds_interp[v_var].values if v_var else np.zeros_like(t_celsius)

    steps_available = min(t_celsius.shape[0], TIME_STEPS)

    # 4. Construcción vectorizada del DataFrame para optimizar I/O y uso de CPU
    total_spatial = GRID_RES * GRID_RES
    total_rows = steps_available * total_spatial

    # Índices y coordenadas 2D repetidas temporalmente
    lat_grid, lon_grid = np.meshgrid(target_lats, target_lons, indexing="ij")
    lats_flat = np.tile(lat_grid.flatten(), steps_available)
    lons_flat = np.tile(lon_grid.flatten(), steps_available)
    cell_ids = np.tile(np.arange(total_spatial, dtype=np.int32), steps_available)
    time_indices = np.repeat(np.arange(steps_available, dtype=np.int32), total_spatial)

    t_flat = t_celsius[:steps_available].reshape(-1)
    u_flat = u_vals[:steps_available].reshape(-1)
    v_flat = v_vals[:steps_available].reshape(-1)

    df = pd.DataFrame({
        "time_idx": time_indices,
        "cell_id": cell_ids,
        "lat": np.round(lats_flat, 4),
        "lon": np.round(lons_flat, 4),
        "temperature": np.round(t_flat, 2),
        "wind_u": np.round(u_flat, 2),
        "wind_v": np.round(v_flat, 2)
    })

    # 5. Escritura a disco
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    ds.close()

    print(f"[✓] Guardado: {out_csv} ({total_rows:,} registros, {steps_available} pasos)")


# =============================================================================
# 4. CICLO PRINCIPAL DE DESCUBRIMIENTO Y EJECUCIÓN
# =============================================================================

def main():
    if not os.path.exists("config.json"):
        print("[-] Error: config.json no encontrado.")
        return

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    regions = config.get("regions", {})

    # Búsqueda recursiva de archivos NetCDF en la carpeta de datasets
    candidate_files = sorted(list(set(
        glob.glob("data/**/*.nc", recursive=True) + glob.glob("data/*.nc")
    )))

    if not candidate_files:
        print("[!] No se encontraron archivos .nc en la carpeta data/.")
        return

    processed = set()
    for nc_file in candidate_files:
        try:
            ds = xr.open_dataset(nc_file)
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lats = ds[lat_name].values
            lons = ds[lon_name].values
            ds.close()

            matched_id = match_region(lats, lons, regions)
            if matched_id and matched_id not in processed:
                process_single_nc(nc_file, matched_id, regions[matched_id])
                processed.add(matched_id)
            elif not matched_id:
                print(f"[-] Sin coincidencia regional: {nc_file} "
                      f"(Lat: {lats.min():.1f} a {lats.max():.1f}, "
                      f"Lon: {lons.min():.1f} a {lons.max():.1f})")
        except Exception as e:
            print(f"[-] Error al procesar {nc_file}: {e}")


if __name__ == "__main__":
    main()