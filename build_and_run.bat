@echo off
echo ========================================================
echo   GEOTHERMAL HPC ENGINE // COMPILACION Y EJECUCION
echo ========================================================

:: 0. Cargar entorno MSVC x64 (necesario para Ninja / link.exe)
if not defined VCINSTALLDIR (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
    )
)

:: 1. Verificar si existen datos
if not exist "data\weather_colima.csv" (
    echo [*] Generando dataset sintético de prueba...
    python scripts\generate_sample_data.py
)

:: 2. Configurar y compilar con CMake
echo [*] Configurando CMake...
if exist "build\CMakeCache.txt" (
    cmake -B build -S .
) else (
    cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
)
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
if exist ".\build\Release\GeoThermalParallel.exe" (
    .\build\Release\GeoThermalParallel.exe
) else if exist ".\build\GeoThermalParallel.exe" (
    .\build\GeoThermalParallel.exe
) else (
    echo [-] No se encontro el ejecutable GeoThermalParallel.exe en build\ ni en build\Release\.
    pause
    exit /b 1
)
pause