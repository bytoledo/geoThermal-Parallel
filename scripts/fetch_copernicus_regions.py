import os
import json
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

GRID_RES = 100
TIME_STEPS = 720  # 30 días x 24 horas

def fetch_region_era5(client, region_key, reg_info):
    name = reg_info["name"]
    bounds = reg_info["bounds"]
    out_csv = reg_info["dataset_path"]
    
    # Copernicus ERA5 usa el formato de bounding box: [Norte, Oeste, Sur, Este]
    # lat_max (N), lon_min (W), lat_min (S), lon_max (E)
    area = [
        bounds["lat_max"],
        bounds["lon_min"],
        bounds["lat_min"],
        bounds["lon_max"]
    ]
    
    nc_temp = f"data/raw_{region_key}.nc"
    print(f"\n========================================================")
    print(f"[+] Descargando Copernicus ERA5 para: {name}")
    print(f"[+] Área [N, W, S, E]: {area}")
    print(f"========================================================")

    # Petición a CDS API para 1 mes continuo (720 horas)
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature',
                '10m_u_component_of_wind',
                '10m_v_component_of_wind'
            ],
            'year': '2024',
            'month': '01',
            'day': [f"{d:02d}" for d in range(1, 31)],  # 30 días = 720 horas
            'time': [f"{h:02d}:00" for h in range(24)],
            'area': area,
            'format': 'netcdf',
        },
        nc_temp
    )

    print(f"[✓] NetCDF descargado. Remuestreando a malla {GRID_RES}x{GRID_RES}...")
    
    # Abrir NetCDF con xarray
    ds = xr.open_dataset(nc_temp)
    
    # Definir la cuadrícula uniforme exacta de 100x100
    target_lats = np.linspace(bounds["lat_min"], bounds["lat_max"], GRID_RES)
    target_lons = np.linspace(bounds["lon_min"], bounds["lon_max"], GRID_RES)
    
    # Interpolar bilinealmente al grid del motor
    ds_interp = ds.interp(latitude=target_lats, longitude=target_lons, method="linear")
    
    # Convertir temperatura de Kelvin a Celsius (T_c = T_k - 273.15)
    t2m_celsius = ds_interp['t2m'].values - 273.15
    u10 = ds_interp['u10'].values
    v10 = ds_interp['v10'].values
    
    # Estructurar filas para el CSV del motor
    rows = []
    for t_idx in range(TIME_STEPS):
        for r_idx, lat in enumerate(target_lats):
            for c_idx, lon in enumerate(target_lons):
                cell_id = r_idx * GRID_RES + c_idx
                temp = float(t2m_celsius[t_idx, r_idx, c_idx])
                u = float(u10[t_idx, r_idx, c_idx])
                v = float(v10[t_idx, r_idx, c_idx])
                rows.append([t_idx, cell_id, round(lat, 4), round(lon, 4), round(temp, 2), round(u, 2), round(v, 2)])
    
    df = pd.DataFrame(rows, columns=["time_idx", "cell_id", "lat", "lon", "temperature", "wind_u", "wind_v"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"[✓] Dataset generado: {out_csv} ({len(df)} registros)")
    
    # Limpiar archivo NetCDF temporal
    ds.close()
    if os.path.exists(nc_temp):
        os.remove(nc_temp)

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    c = cdsapi.Client()
    regions = config["regions"]

    for r_id, r_data in regions.items():
        # Saltar si es Colima y ya tienes el archivo original
        if r_id == "colima_copernicus" and os.path.exists(r_data["dataset_path"]):
            print(f"[i] Saltando {r_id}, dataset ya existente.")
            continue
        fetch_region_era5(c, r_id, r_data)

if __name__ == "__main__":
    main()