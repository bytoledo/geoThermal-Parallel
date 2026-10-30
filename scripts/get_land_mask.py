"""
GEOTHERMAL // VECTOR LAND MASK INGESTION
Descarga y estructura geometrías continentales simplificadas (GeoJSON)
para aplicar recortes vectoriales dinámicos (clip-path) en el frontend.
"""

import os
import json
import urllib.request
import urllib.error

# =============================================================================
# 1. CONFIGURACIÓN DE FUENTES Y DESTINO
# =============================================================================

# Capa global de polígonos terrestres simplificados a escala 1:110m
GEOJSON_URL = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
OUTPUT_FILEPATH = os.path.join("web", "land_mask.geojson")
HTTP_TIMEOUT_SECONDS = 15


# =============================================================================
# 2. DESCARGA Y SERIALIZACIÓN DE GEOMETRÍAS
# =============================================================================

def fetch_and_save_land_mask(url: str, output_path: str) -> bool:
    """
    Descarga la colección de límites continentales y la exporta
    a disco como FeatureCollection GeoJSON estandarizada.
    """
    print(">>> Solicitando polígonos terrestres globales...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "GeoThermal-HPC/1.0 (Windows NT 10.0; Win64; x64)"}
    )

    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = response.read().decode("utf-8")
            data = json.loads(payload)

        # Empaquetado estricto bajo estándar GeoJSON FeatureCollection
        features = data.get("features", [])
        output_payload = {
            "type": "FeatureCollection",
            "features": features
        }

        # Creación de directorio destino y persistencia en disco
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(output_payload, out_file, separators=(",", ":"))

        file_size_kb = os.path.getsize(output_path) / 1024.0
        print(f"[✓] Mascara global persistida en: {output_path}")
        print(f"    * Poligonos procesados: {len(features)}")
        print(f"    * Tamano en disco:      {file_size_kb:.2f} KB")
        return True

    except urllib.error.URLError as err:
        print(f"[-] Error de red al descargar GeoJSON: {err.reason}")
        return False
    except json.JSONDecodeError as err:
        print(f"[-] Error al decodificar respuesta JSON: {err}")
        return False
    except OSError as err:
        print(f"[-] Error de I/O al escribir archivo: {err}")
        return False


# =============================================================================
# 3. PUNTO DE ENTRADA
# =============================================================================

def main():
    success = fetch_and_save_land_mask(GEOJSON_URL, OUTPUT_FILEPATH)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()