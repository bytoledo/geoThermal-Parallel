@echo off
echo ========================================================
echo   GEOTHERMAL HPC ENGINE // COMPILACION Y EJECUCION
echo ========================================================

:: 1. Verificar si existen datos
if not exist "data\weather_colima.csv" (
    echo [*] Generando dataset sintético de prueba...
    python scripts\generate_sample_data.py
)

:: 2. Configurar y compilar con CMake
echo [*] Configurando CMake...
cmake -B build -S .
if %ERRORLEVEL% neq 0 (
    echo [-] Error al configurar CMake.
    pause
    exit /b %ERRORLEVEL%
)

echo [*] Compilando binario Release...
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo [-] Error durante la compilacion.
    pause
    exit /b %ERRORLEVEL%
)

:: 3. Lanzar ejecutable
echo [✓] Compilacion completada con exito.
echo [*] Iniciando servidor en http://localhost:8080...
.\build\Release\GeoThermalParallel.exe
pause