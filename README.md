# GeoThermal HPC Engine

Pipeline analítico y de visualización espaciotemporal de microclimas con aceleración paralela híbrida:
* **CPU:** Procesamiento analítico en memoria compartida y detección de anomalías ($Z\text{-Score}$) mediante **OpenMP** y vectorización **AVX2**.
* **GPU:** Interpolación espacial continua a resolución 1080p y sombreado de relieve oblicuo (*Hillshade*) mediante **NVIDIA CUDA**.
* **Frontend Web:** Interfaz geográfica interactiva con **Leaflet.js**, telemetría en tiempo real, corte transversal topográfico y soporte de máscaras vectoriales.

---

## Estructura del Repositorio

* `include/` y `src/`: Backend de cálculo en C++17, kernels CUDA (`.cu`) y servidor HTTP REST.
* `web/`: Interfaz web (Leaflet, canvas dinámico, estilos y máscara costera GeoJSON).
* `scripts/`: Herramientas para generación sintética y procesamiento de rásteres satelitales / NetCDF.
* `data/elevations/`: Matrices topográficas binarias optimizadas (100×100 floats IEEE 754, 40 KB c/u).

---

## Requisitos del Entorno

* Compilador C++17 compatible con OpenMP (MSVC v143+ o GCC 11+)
* NVIDIA CUDA Toolkit (>= 12.0) y GPU con arquitectura compatible
* CMake (>= 3.20)
* Python 3.9+ con `numpy`

---

## Puesta en Marcha Rápida (Prototipo Funcional)

### 1. Clonar el repositorio
```bash
git clone [https://github.com/TU_USUARIO/geothermal-parallel.git](https://github.com/TU_USUARIO/geothermal-parallel.git)
cd geothermal-parallel
```

### 2. Generar el dataset de prueba (Mock Data)
Genera la serie temporal de 720 pasos horarios y la malla de elevación en segundos sin descargar archivos externos:
```bash
python scripts/generate_sample_data.py
```

### 3. Compilar el proyecto con CMake
```powershell
cmake -B build -S .
cmake --build build --config Release
```

### 4. Ejecutar el Servidor
```powershell
.\build\Release\GeoThermalParallel.exe
```

Abre tu navegador en `http://localhost:8080`. Podrás explorar:
* **Temperatura:** Capa continua interpolada por CUDA.
* **Anomalías:** Eventos críticos filtrados con OpenMP.
* **Topografía:** Sombreado de laderas por iluminación solar.
* **Corte Transversal:** Perfil geodésico de relieve en metros sobre el nivel del mar y temperatura en tiempo real.