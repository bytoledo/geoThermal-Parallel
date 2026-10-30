/**
 * GEOTHERMAL // CLIENT CONTROLLER
 * Orquestación de capas Leaflet, visualización GPU, cinemática de viento y telemetría.
 */

// =============================================================================
// 1. ESTADO GLOBAL Y CONFIGURACIÓN ESPACIO-TEMPORAL
// =============================================================================

let LAT_MIN = 18.0, LAT_MAX = 22.0;
let LON_MIN = -105.0, LON_MAX = -101.0;
let GRID_RES = 100;
let TOTAL_TIME_STEPS = 720;
let currentRegionId = "colima_copernicus";

let currentMode = 'temperature'; // 'temperature' | 'anomalies' | 'topography'
let currentTime = 120;
let isPlaying = false;
let playInterval = null;
let playbackSpeedMs = 600;

let timelineData = [];
let currentAnomaliesCache = [];
let showMarkers = true;
let landMaskData = null;

// =============================================================================
// 2. INICIALIZACIÓN DEL MAPA Y CAPAS BASE
// =============================================================================

const map = L.map('map', {
    zoomSnap: 0.1,
    zoomDelta: 0.25,
    wheelPxPerZoomLevel: 100,
    minZoom: 2,
    maxZoom: 18
}).setView([19.24, -103.72], 6.5);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap'
}).addTo(map);

// Capa contenedora para la textura térmica generada en CUDA
const cudaOverlay = L.imageOverlay('', [[18, -105], [22, -101]], { 
    opacity: 0.78, 
    interactive: false 
}).addTo(map);

// Grupos de capas vectoriales para interacción puntual
const anomalyMarkers = L.layerGroup().addTo(map);
const probeMarker = L.layerGroup().addTo(map);

// Canvas fuera de pantalla para volcar el búfer de píxeles crudo de GPU
const directCanvas = document.createElement('canvas');
directCanvas.width = 1920;
directCanvas.height = 1080;
const ctx = directCanvas.getContext('2d');

// =============================================================================
// 3. MÁSCARA COSTERA VECTORIAL (GEOJSON / SVG CLIP-PATH)
// =============================================================================

fetch('land_mask.geojson')
    .then(r => {
        if (!r.ok) throw new Error("No se encontró land_mask.geojson");
        return r.json();
    })
    .then(geoJson => {
        landMaskData = geoJson;
        applyOverlayMask();
    })
    .catch(err => console.warn("Aviso:", err.message));

function applyOverlayMask() {
    const imgEl = cudaOverlay.getElement();
    if (!imgEl || !landMaskData) return;

    const bounds = cudaOverlay.getBounds();
    const originPoint = map.latLngToLayerPoint(bounds.getNorthWest());

    let svgPath = "";
    landMaskData.features.forEach(feature => {
        const geom = feature.geometry;
        const coords = geom.coordinates;
        
        const processRing = (ring) => {
            let path = "";
            let hasPointsNear = false;

            for (let idx = 0; idx < ring.length; idx++) {
                const pt = ring[idx]; // [lon, lat]
                const lat = pt[1];
                const lon = pt[0];

                if (lat >= bounds.getSouth() - 2.0 && lat <= bounds.getNorth() + 2.0 &&
                    lon >= bounds.getWest() - 2.0 && lon <= bounds.getEast() + 2.0) {
                    hasPointsNear = true;
                }

                const pixel = map.latLngToLayerPoint([lat, lon]);
                const relX = (pixel.x - originPoint.x).toFixed(1);
                const relY = (pixel.y - originPoint.y).toFixed(1);
                path += (idx === 0 ? `M ${relX} ${relY} ` : `L ${relX} ${relY} `);
            }
            return hasPointsNear ? path + "Z " : "";
        };

        if (geom.type === "Polygon") {
            coords.forEach(r => { svgPath += processRing(r); });
        } else if (geom.type === "MultiPolygon") {
            coords.forEach(poly => poly.forEach(r => { svgPath += processRing(r); }));
        }
    });

    if (svgPath.trim().length > 0) {
        imgEl.style.clipPath = `path('${svgPath}')`;
        imgEl.style.webkitClipPath = `path('${svgPath}')`;
    } else {
        imgEl.style.clipPath = 'none';
        imgEl.style.webkitClipPath = 'none';
    }
}

map.on('zoom animzoom viewreset move moveend', applyOverlayMask);

// =============================================================================
// 4. MOTOR DE CINEMÁTICA Y PARTÍCULAS DE VIENTO
// =============================================================================

let showWind = false;
let windData = null;
const PARTICLE_COUNT = 1800;
const particles = [];

const windCanvas = document.createElement('canvas');
windCanvas.style.position = 'absolute';
windCanvas.style.top = '0';
windCanvas.style.left = '0';
windCanvas.style.width = '100%';
windCanvas.style.height = '100%';
windCanvas.style.pointerEvents = 'none';
windCanvas.style.zIndex = '500';
map.getContainer().appendChild(windCanvas);
const wCtx = windCanvas.getContext('2d');

function resizeWindCanvas() {
    const size = map.getSize();
    windCanvas.width = size.x;
    windCanvas.height = size.y;
}

function initParticles() {
    particles.length = 0;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            lat: LAT_MIN + Math.random() * (LAT_MAX - LAT_MIN),
            lon: LON_MIN + Math.random() * (LON_MAX - LON_MIN),
            age: Math.floor(Math.random() * 80),
            maxAge: 40 + Math.floor(Math.random() * 40)
        });
    }
}

function fetchWindData(t) {
    if (!showWind) return;
    fetch(`/api/wind?time=${t}`)
        .then(res => {
            if (!res.ok) throw new Error("Fallo HTTP /api/wind");
            return res.arrayBuffer();
        })
        .then(buffer => {
            windData = new Float32Array(buffer);
        })
        .catch(err => console.error("Error al cargar vectores de viento:", err));
}

function getInterpolatedVector(lat, lon) {
    if (!windData || windData.length === 0) return null;

    const gx = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * (GRID_RES - 1);
    const gy = ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * (GRID_RES - 1);

    if (gx < 0 || gx >= GRID_RES - 1 || gy < 0 || gy >= GRID_RES - 1) return null;

    const x0 = Math.floor(gx), y0 = Math.floor(gy);
    const x1 = x0 + 1, y1 = y0 + 1;
    const dx = gx - x0, dy = gy - y0;

    const idx00 = (y0 * GRID_RES + x0) * 2;
    const idx10 = (y0 * GRID_RES + x1) * 2;
    const idx01 = (y1 * GRID_RES + x0) * 2;
    const idx11 = (y1 * GRID_RES + x1) * 2;

    const u = (1 - dx) * (1 - dy) * windData[idx00] +
                   dx  * (1 - dy) * windData[idx10] +
              (1 - dx) *      dy  * windData[idx01] +
                   dx  *      dy  * windData[idx11];

    const v = (1 - dx) * (1 - dy) * windData[idx00 + 1] +
                   dx  * (1 - dy) * windData[idx10 + 1] +
              (1 - dx) *      dy  * windData[idx01 + 1] +
                   dx  *      dy  * windData[idx11 + 1];

    return [u, v];
}

function stepWind() {
    if (!showWind || !windData) {
        wCtx.clearRect(0, 0, windCanvas.width, windCanvas.height);
        requestAnimationFrame(stepWind);
        return;
    }

    wCtx.fillStyle = 'rgba(15, 17, 26, 0.88)';
    wCtx.globalCompositeOperation = 'destination-in';
    wCtx.fillRect(0, 0, windCanvas.width, windCanvas.height);
    wCtx.globalCompositeOperation = 'source-over';

    wCtx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    wCtx.lineWidth = 1.4;
    wCtx.beginPath();

    const dt = 0.0035;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const p = particles[i];
        const vec = getInterpolatedVector(p.lat, p.lon);

        if (!vec || p.age >= p.maxAge) {
            p.lat = LAT_MIN + Math.random() * (LAT_MAX - LAT_MIN);
            p.lon = LON_MIN + Math.random() * (LON_MAX - LON_MIN);
            p.age = 0;
            continue;
        }

        const pt1 = map.latLngToContainerPoint([p.lat, p.lon]);
        p.lon += vec[0] * dt;
        p.lat += vec[1] * dt;
        p.age++;
        const pt2 = map.latLngToContainerPoint([p.lat, p.lon]);

        wCtx.moveTo(pt1.x, pt1.y);
        wCtx.lineTo(pt2.x, pt2.y);
    }

    wCtx.stroke();
    requestAnimationFrame(stepWind);
}

map.on('move zoom resize', resizeWindCanvas);
resizeWindCanvas();
initParticles();

// =============================================================================
// 5. CARGA DE DATOS, HEATMAP GPU Y ANOMALÍAS
// =============================================================================

function loadData(t) {
    currentTime = t;
    const timeInputEl = document.getElementById('timeInput');
    if (timeInputEl) timeInputEl.value = t;

    // 1. Resolver URL según el modo analítico activo
    let heatmapUrl = '';
    if (currentMode === 'temperature') {
        heatmapUrl = `/api/heatmap/temperature?time=${t}`;
    } else if (currentMode === 'anomalies') {
        heatmapUrl = `/api/heatmap/anomalies?time=${t}`;
    } else if (currentMode === 'topography') {
        heatmapUrl = `/api/heatmap/topography?_ts=${currentRegionId}`;
    }
    
    // 2. Carga y decodificación del buffer crudo RGBA (1920x1080) desde CUDA
    if (heatmapUrl) {
        fetch(heatmapUrl)
            .then(res => res.arrayBuffer())
            .then(buffer => {
                const pixelData = new Uint8ClampedArray(buffer);
                const imgData = new ImageData(pixelData, 1920, 1080);
                ctx.clearRect(0, 0, 1920, 1080);
                ctx.putImageData(imgData, 0, 0);

                cudaOverlay.once('load', applyOverlayMask);
                cudaOverlay.setUrl(directCanvas.toDataURL());
            })
            .catch(err => console.error("Error al cargar heatmap:", err));
    }

    // 3. Capa vectorial de anomalías (solo activa en modo anomalies)
    if (currentMode === 'anomalies') {
        fetch(`/api/anomalies?time=${t}`)
            .then(r => r.json())
            .then(data => {
                currentAnomaliesCache = data;
                table.setData(data);
                const badge = document.getElementById('anomalyBadge');
                if (badge) badge.innerText = `${data.length} alertas`;
                renderAnomalyMarkers();
            })
            .catch(err => console.error("Error al cargar anomalias:", err));
    } else {
        anomalyMarkers.clearLayers();
    }

    // 4. Refresco dinámico del corte transversal si está abierto con 2 puntos
    if (sliceMode && slicePoints.length === 2) {
        fetchSliceProfile();
    }

    fetchWindData(t);
    drawTimeline();
}

function renderAnomalyMarkers() {
    anomalyMarkers.clearLayers();
    if (!showMarkers || currentMode !== 'anomalies') return;

    currentAnomaliesCache.forEach(pt => {
        const isHeat = pt.z_score >= 0;
        const marker = L.circleMarker([pt.lat, pt.lon], {
            radius: 5,
            color: isHeat ? '#ff0055' : '#00ddff',
            fillColor: isHeat ? '#ff3366' : '#33ccff',
            fillOpacity: 0.85,
            weight: 1.5
        });
        marker.bindTooltip(`Celda #${pt.cell_id} | Z: ${pt.z_score.toFixed(2)}&sigma;`);
        anomalyMarkers.addLayer(marker);
    });
}

// =============================================================================
// 6. TIMELINE TEMPORAL (SPARKLINE CANVAS)
// =============================================================================

const tCanvas = document.getElementById('timeline-canvas');
const tCtx = tCanvas.getContext('2d');

function initTimeline() {
    fetch('/api/timeline')
        .then(r => r.json())
        .then(data => {
            timelineData = data;
            drawTimeline();
        })
        .catch(err => console.error("Error al cargar timeline:", err));
}

function drawTimeline() {
    const w = tCanvas.clientWidth;
    const h = tCanvas.clientHeight;
    tCanvas.width = w;
    tCanvas.height = h;

    tCtx.fillStyle = '#090a0f';
    tCtx.fillRect(0, 0, w, h);

    if (timelineData.length === 0) return;

    const maxCount = Math.max(...timelineData.map(d => d.count), 1);
    const barWidth = w / TOTAL_TIME_STEPS;

    for (let i = 0; i < TOTAL_TIME_STEPS; i++) {
        const d = timelineData[i];
        if (d && d.count > 0) {
            const barHeight = (d.count / maxCount) * (h - 4);
            tCtx.fillStyle = d.max_z >= 3.0 ? '#f43f5e' : '#3b82f6';
            tCtx.fillRect(i * barWidth, h - barHeight, Math.max(barWidth, 1.2), barHeight);
        }
    }

    const cursorX = (currentTime / TOTAL_TIME_STEPS) * w;
    tCtx.fillStyle = '#ffffff';
    tCtx.fillRect(cursorX - 1, 0, 2, h);
}

tCanvas.addEventListener('click', (e) => {
    const rect = tCanvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const step = Math.min(TOTAL_TIME_STEPS - 1, Math.max(0, Math.floor((clickX / rect.width) * TOTAL_TIME_STEPS)));
    setTime(step);
});

// =============================================================================
// 7. TABLA DE ANOMALÍAS (TABULATOR) E INSPECCIÓN PUNTUAL (PROBE)
// =============================================================================

const table = new Tabulator("#table-container", {
    layout: "fitColumns",
    height: "100%",
    placeholder: "No hay anomalías críticas registradas en este paso temporal.",
    columns: [
        { title: "ID", field: "cell_id", width: 65, hozAlign: "center" },
        { title: "Lat", field: "lat", width: 75, formatter: cell => Number(cell.getValue()).toFixed(2) },
        { title: "Lon", field: "lon", width: 75, formatter: cell => Number(cell.getValue()).toFixed(2) },
        { title: "T. Obs", field: "t_obs", formatter: cell => `${Number(cell.getValue()).toFixed(1)} °C` },
        { title: "Media", field: "t_mean", formatter: cell => `${Number(cell.getValue()).toFixed(1)} °C` },
        { title: "Z-Score", field: "z_score", formatter: cell => {
            const val = Number(cell.getValue());
            const color = val >= 2.5 ? '#ff1a75' : (val <= 2.0 ? '#00e5ff' : '#ffaa00');
            return `<b style="color:${color};">${val.toFixed(2)} &sigma;</b>`;
        }}
    ]
});

table.on("rowClick", (e, row) => {
    const d = row.getData();
    map.flyTo([d.lat, d.lon], 10, { duration: 0.8 });
    L.popup()
        .setLatLng([d.lat, d.lon])
        .setContent(`<b>Celda #${d.cell_id}</b><br>T. Obs: ${d.t_obs.toFixed(1)}°C<br>Z-Score: ${d.z_score.toFixed(2)}&sigma;`)
        .openOn(map);
});

map.on('click', (e) => {
    if (sliceMode) return;
    const { lat, lng } = e.latlng;

    if (lat >= LAT_MIN && lat <= LAT_MAX && lng >= LON_MIN && lng <= LON_MAX) {
        const row = Math.min(GRID_RES - 1, Math.max(0, Math.round(((lat - LAT_MIN) / (LAT_MAX - LAT_MIN)) * (GRID_RES - 1))));
        const col = Math.min(GRID_RES - 1, Math.max(0, Math.round(((lng - LON_MIN) / (LON_MAX - LON_MIN)) * (GRID_RES - 1))));
        const cellId = row * GRID_RES + col;

        fetch(`/api/probe?time=${currentTime}&cell_id=${cellId}`)
            .then(res => res.json())
            .then(data => {
                const tempVal = Number(data.temp).toFixed(1);
                const zVal = Number(data.z_score).toFixed(2);
                const elevVal = data.elevation !== undefined ? Math.round(data.elevation) : 'N/D';
                
                let badgeColor = '#00c3ff';
                if (data.temp >= 34.0) badgeColor = '#d50000';
                else if (data.temp >= 28.0) badgeColor = '#ff5722';
                else if (data.temp >= 22.0) badgeColor = '#ffeb3b';
                else if (data.temp >= 16.0) badgeColor = '#00e676';

                probeMarker.clearLayers();
                const marker = L.circleMarker([data.lat, data.lon], {
                    radius: 6,
                    color: '#ffffff',
                    fillColor: badgeColor,
                    fillOpacity: 1.0,
                    weight: 2
                });
                probeMarker.addLayer(marker);

                const popupContent = `
                    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; color: #1e293b; min-width: 140px;">
                        <div style="font-weight: 700; color: #475569; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 6px;">
                            ESTACIÓN // CELDA #${data.cell_id}
                        </div>
                        <div style="font-size: 20px; font-weight: 800; color: ${badgeColor}; margin-bottom: 4px;">
                            ${tempVal} <span style="font-size: 13px; font-weight: 600;">°C</span>
                        </div>
                        <div style="color: #64748b; font-size: 11px; line-height: 1.4;">
                            <b>Elevación:</b> ${elevVal} msnm<br>
                            <b>Lat:</b> ${data.lat.toFixed(3)}°<br>
                            <b>Lon:</b> ${data.lon.toFixed(3)}°<br>
                            <b>Z-Score:</b> ${zVal}&sigma;
                        </div>
                    </div>
                `;

                L.popup({ offset: [0, -6] })
                    .setLatLng([data.lat, data.lon])
                    .setContent(popupContent)
                    .openOn(map);
            });
    }
});

map.on('popupclose', () => probeMarker.clearLayers());

// =============================================================================
// 8. INTERFAZ DE USUARIO: TÍTULO, LEYENDA Y MODOS
// =============================================================================

function updateBrandTitle(regionName) {
    const brandEl = document.querySelector('.brand');
    if (!brandEl || !regionName) return;

    const parts = regionName.split('//').map(s => s.trim().toUpperCase());
    
    if (parts.length >= 2) {
        brandEl.innerHTML = `
            <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.8px; color: #38bdf8;">
                GEOTHERMAL // ${parts[0]}
            </div>
            <div style="font-size: 10px; font-weight: 600; letter-spacing: 0.5px; color: #94a3b8; margin-top: 1px;">
                ${parts.slice(1).join(' // ')}
            </div>
        `;
    } else {
        brandEl.innerHTML = `
            <div style="font-size: 12px; font-weight: 800; letter-spacing: 0.8px; color: #38bdf8;">
                GEOTHERMAL // ${regionName.toUpperCase()}
            </div>
        `;
    }
}

function getLegendHTML() {
    if (currentMode === 'temperature') {
        return `
            <div class="tv-weather-legend">
                <div class="tv-legend-title">
                    <span>Temperatura Superficial</span>
                    <span style="color: #38bdf8;">°C</span>
                </div>
                <div class="tv-gradient-track" style="background: linear-gradient(to right, 
                    #f5d0fe 0%, #9333ea 14.3%, #1e40af 28.6%, #06b6d4 42.8%, 
                    #10b981 57.1%, #eab308 71.4%, #ef4444 85.7%, #7f1d1d 100%);">
                </div>
                <div class="tv-scale-marks">
                    <span>-20°</span><span>-10°</span><span>0°</span><span>10°</span>
                    <span>20°</span><span>30°</span><span>40°</span><span>50°</span>
                </div>
            </div>
        `;
    } else if (currentMode === 'topography') {
        return `
            <div class="tv-weather-legend">
                <div class="tv-legend-title">
                    <span>Relieve Hipsométrico</span>
                    <span style="color: #38bdf8;">msnm</span>
                </div>
                <div class="tv-gradient-track" style="background: linear-gradient(to right, 
                    rgb(46,117,89) 0%, rgb(115,144,82) 25%, rgb(194,163,94) 55%, rgb(120,80,55) 85%, rgb(240,245,255) 100%);">
                </div>
                <div class="tv-scale-marks">
                    <span>Valle</span><span>Meseta</span><span>Sierras</span><span>Cumbres</span>
                </div>
            </div>
        `;
    }
    return `
        <div class="map-legend">
            <b>SEVERIDAD DE ANOMALÍA</b><br>
            <span style="color:#00e5ff;">Frío (-3σ)</span> | <span style="color:#ffaa00;">Normal</span> | <span style="color:#ff1a75;">Calor (+3σ)</span>
            <div class="gradient-bar" style="background: linear-gradient(to right, #00e5ff, #10b981, #ffaa00, #ff1a75);"></div>
        </div>
    `;
}

const legendControl = L.control({ position: 'bottomleft' });
let legendContainer = null;

legendControl.onAdd = function() {
    legendContainer = L.DomUtil.create('div', 'legend-wrapper');
    legendContainer.innerHTML = getLegendHTML();
    return legendContainer;
};
legendControl.addTo(map);

function updateLegendUI() {
    if (legendContainer) legendContainer.innerHTML = getLegendHTML();
}

function switchMode(mode) {
    currentMode = mode;

    const btnTemp = document.getElementById('tabHeatmap');
    const btnAnom = document.getElementById('tabAnomalies');
    const btnTopo = document.getElementById('tabTopography');

    if (btnTemp) btnTemp.classList.toggle('active', mode === 'temperature');
    if (btnAnom) btnAnom.classList.toggle('active', mode === 'anomalies');
    if (btnTopo) btnTopo.classList.toggle('active', mode === 'topography');

    const leftPanel = document.getElementById('left-panel');
    const toggleWrapper = document.getElementById('markersToggleWrapper');

    if (mode === 'anomalies') {
        if (leftPanel) leftPanel.style.display = 'flex';
        if (toggleWrapper) toggleWrapper.style.display = 'flex';
    } else {
        anomalyMarkers.clearLayers();
        if (leftPanel) leftPanel.style.display = 'none';
        if (toggleWrapper) toggleWrapper.style.display = 'none';
    }

    if (sliceMode) {
        closeSliceTray();
    }

    map.invalidateSize();
    updateLegendUI();
    loadData(currentTime);
}

// =============================================================================
// 9. CONTROLES DE TIEMPO Y REPRODUCCIÓN
// =============================================================================

function setTime(val) {
    currentTime = parseInt(val, 10);
    if (isNaN(currentTime)) currentTime = 0;
    currentTime = Math.max(0, Math.min(TOTAL_TIME_STEPS - 1, currentTime));
    loadData(currentTime);
}

function stepTime(delta) {
    setTime(currentTime + delta);
}

function togglePlay() {
    isPlaying = !isPlaying;
    const btn = document.getElementById('btnPlay');
    if (isPlaying) {
        btn.innerText = "⏸ Pausa";
        playInterval = setInterval(() => {
            currentTime = (currentTime + 1) % TOTAL_TIME_STEPS;
            loadData(currentTime);
        }, playbackSpeedMs);
    } else {
        btn.innerText = "▶ Play";
        clearInterval(playInterval);
    }
}

// =============================================================================
// 10. LISTENERS DEL DOM Y CAMBIO DE REGIÓN
// =============================================================================

const speedSelector = document.getElementById('playSpeed');
if (speedSelector) {
    speedSelector.addEventListener('change', (e) => {
        playbackSpeedMs = parseInt(e.target.value, 10);
        if (isPlaying) {
            clearInterval(playInterval);
            playInterval = setInterval(() => {
                currentTime = (currentTime + 1) % TOTAL_TIME_STEPS;
                loadData(currentTime);
            }, playbackSpeedMs);
        }
    });
}

const toggleMarkersEl = document.getElementById('toggleMarkers');
if (toggleMarkersEl) {
    toggleMarkersEl.addEventListener('change', (e) => {
        showMarkers = e.target.checked;
        renderAnomalyMarkers();
    });
}

const toggleWindEl = document.getElementById('toggleWind');
if (toggleWindEl) {
    showWind = toggleWindEl.checked;
    toggleWindEl.addEventListener('change', (e) => {
        showWind = e.target.checked;
        if (!showWind) {
            wCtx.clearRect(0, 0, windCanvas.width, windCanvas.height);
        } else {
            fetchWindData(currentTime);
        }
    });
}

const regionSelector = document.getElementById('regionSelect');
if (regionSelector) {
    regionSelector.addEventListener('change', async (e) => {
        const regionId = e.target.value;
        try {
            const res = await fetch(`/api/regions/switch?id=${regionId}`, { method: 'POST' });
            if (!res.ok) throw new Error("Fallo al cambiar de región en backend");

            const metaRes = await fetch('/api/metadata');
            const meta = await metaRes.json();

            currentRegionId = meta.region_id || regionId;
            LAT_MIN = meta.lat_min;
            LAT_MAX = meta.lat_max;
            LON_MIN = meta.lon_min;
            LON_MAX = meta.lon_max;
            GRID_RES = meta.grid_width;
            TOTAL_TIME_STEPS = meta.time_steps;

            updateBrandTitle(meta.region_name);

            const dLatHalf = (LAT_MAX - LAT_MIN) / (GRID_RES - 1) / 2.0;
            const dLonHalf = (LON_MAX - LON_MIN) / (GRID_RES - 1) / 2.0;
            const bounds = [
                [LAT_MIN - dLatHalf, LON_MIN - dLonHalf],
                [LAT_MAX + dLatHalf, LON_MAX + dLonHalf]
            ];

            cudaOverlay.setBounds(bounds);
            map.fitBounds(bounds, {
                padding: [20, 20],
                animate: false
            });

            for (let i = 0; i < PARTICLE_COUNT; i++) {
                particles[i].lat = LAT_MIN + Math.random() * (LAT_MAX - LAT_MIN);
                particles[i].lon = LON_MIN + Math.random() * (LON_MAX - LON_MIN);
                particles[i].age = 0;
            }

            currentTime = 0;
            initTimeline();
            loadData(0);
        } catch (err) {
            console.error("[-] Error en cambio de región:", err);
        }
    });
}

// =============================================================================
// 11. CROSS-SECTION SLICE (RELIEVE REAL ASCENDENTE Y HUD)
// =============================================================================

let sliceMode = false;
let slicePoints = [];
let sliceLineLayer = L.layerGroup().addTo(map);
let sliceDataCache = null;
let hoverIndex = -1;

let sliceHoverMarker = L.circleMarker([0, 0], {
    radius: 6,
    color: '#ffffff',
    fillColor: '#38bdf8',
    fillOpacity: 1,
    weight: 2
});

const sliceCanvas = document.getElementById('slice-canvas');
const sCtx = sliceCanvas ? sliceCanvas.getContext('2d') : null;

function toggleSliceMode() {
    sliceMode = !sliceMode;
    const btn = document.getElementById('tabSlice');
    if (btn) btn.classList.toggle('active', sliceMode);

    if (sliceMode) {
        slicePoints = [];
        sliceLineLayer.clearLayers();
        document.getElementById('slice-tray').style.display = 'block';
        document.getElementById('sliceStats').innerText = "Haz clic en el Punto A";
        map.getContainer().style.cursor = 'crosshair';
    } else {
        closeSliceTray();
    }
}

function closeSliceTray() {
    sliceMode = false;
    slicePoints = [];
    sliceDataCache = null;
    hoverIndex = -1;
    sliceLineLayer.clearLayers();
    if (map.hasLayer(sliceHoverMarker)) map.removeLayer(sliceHoverMarker);
    document.getElementById('slice-tray').style.display = 'none';
    const btn = document.getElementById('tabSlice');
    if (btn) btn.classList.remove('active');
    map.getContainer().style.cursor = '';
}

map.on('click', (e) => {
    if (!sliceMode) return;

    if (slicePoints.length === 0) {
        slicePoints.push(e.latlng);
        sliceLineLayer.clearLayers();
        if (map.hasLayer(sliceHoverMarker)) map.removeLayer(sliceHoverMarker);
        
        const markerA = L.circleMarker(e.latlng, {
            radius: 5, color: '#38bdf8', fillColor: '#0284c7', fillOpacity: 1
        }).bindTooltip("A", { permanent: true, direction: "top" });
        sliceLineLayer.addLayer(markerA);

        document.getElementById('sliceStats').innerText = "Haz clic en el Punto B";
    } else if (slicePoints.length === 1) {
        slicePoints.push(e.latlng);

        const markerB = L.circleMarker(e.latlng, {
            radius: 5, color: '#f43f5e', fillColor: '#e11d48', fillOpacity: 1
        }).bindTooltip("B", { permanent: true, direction: "top" });
        sliceLineLayer.addLayer(markerB);

        const polyline = L.polyline([slicePoints[0], slicePoints[1]], {
            color: '#38bdf8', weight: 2, dashArray: '4, 4'
        });
        sliceLineLayer.addLayer(polyline);

        fetchSliceProfile();
    } else {
        slicePoints = [e.latlng];
        sliceLineLayer.clearLayers();
        if (map.hasLayer(sliceHoverMarker)) map.removeLayer(sliceHoverMarker);

        const markerA = L.circleMarker(e.latlng, {
            radius: 5, color: '#38bdf8', fillColor: '#0284c7', fillOpacity: 1
        }).bindTooltip("A", { permanent: true, direction: "top" });
        sliceLineLayer.addLayer(markerA);
        document.getElementById('sliceStats').innerText = "Haz clic en el Punto B";
    }
});

function fetchSliceProfile() {
    if (slicePoints.length < 2) return;
    const [pA, pB] = slicePoints;

    fetch(`/api/slice?lat1=${pA.lat}&lon1=${pA.lng}&lat2=${pB.lat}&lon2=${pB.lng}&time=${currentTime}&samples=120`)
        .then(r => r.json())
        .then(data => {
            sliceDataCache = data;
            document.getElementById('sliceStats').innerText = `Distancia: ${data.total_dist_km} km | 120 muestras`;
            drawSliceProfile();
        })
        .catch(err => console.error("Error al calcular corte transversal:", err));
}

function drawSliceProfile() {
    if (!sliceDataCache || !sliceCanvas) return;

    const w = sliceCanvas.parentElement.clientWidth;
    const h = sliceCanvas.parentElement.clientHeight;
    sliceCanvas.width = w;
    sliceCanvas.height = h;

    sCtx.clearRect(0, 0, w, h);

    const pts = sliceDataCache.points;
    if (pts.length < 2) return;

    // Escala basada en la elevación real sobre el nivel del mar (msnm)
    const elevs = pts.map(p => p.elevation ?? 0);
    const minElev = Math.max(0, Math.floor(Math.min(...elevs)));
    const maxElev = Math.max(minElev + 100, Math.ceil(Math.max(...elevs)));
    const totalDist = sliceDataCache.total_dist_km;

    // 1. Rejilla de referencia
    sCtx.strokeStyle = '#1e293b';
    sCtx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
        let y = (h / 4) * i;
        sCtx.beginPath();
        sCtx.moveTo(0, y);
        sCtx.lineTo(w, y);
        sCtx.stroke();
    }

    // 2. Silueta montañosa sólida (bloque ascendente)
    const terrainGrad = sCtx.createLinearGradient(0, 0, 0, h);
    terrainGrad.addColorStop(0, '#334155');
    terrainGrad.addColorStop(1, '#0f172a');

    sCtx.beginPath();
    pts.forEach((p, i) => {
        const x = (p.dist_km / totalDist) * w;
        const normH = ((p.elevation ?? 0) - minElev) / (maxElev - minElev);
        const y = h - (normH * (h - 28) + 8);

        if (i === 0) sCtx.moveTo(x, y);
        else sCtx.lineTo(x, y);
    });
    sCtx.lineTo(w, h);
    sCtx.lineTo(0, h);
    sCtx.closePath();
    sCtx.fillStyle = terrainGrad;
    sCtx.fill();

    // 3. Cresta del relieve iluminada
    sCtx.beginPath();
    pts.forEach((p, i) => {
        const x = (p.dist_km / totalDist) * w;
        const normH = ((p.elevation ?? 0) - minElev) / (maxElev - minElev);
        const y = h - (normH * (h - 28) + 8);

        if (i === 0) sCtx.moveTo(x, y);
        else sCtx.lineTo(x, y);
    });
    sCtx.strokeStyle = '#38bdf8';
    sCtx.lineWidth = 2.5;
    sCtx.stroke();

    // 4. Cursor interactivo y HUD
    if (hoverIndex >= 0 && hoverIndex < pts.length) {
        const activePt = pts[hoverIndex];
        const hX = (activePt.dist_km / totalDist) * w;
        const normH = ((activePt.elevation ?? 0) - minElev) / (maxElev - minElev);
        const hY = h - (normH * (h - 28) + 8);

        sCtx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        sCtx.setLineDash([3, 3]);
        sCtx.beginPath();
        sCtx.moveTo(hX, 0);
        sCtx.lineTo(hX, h);
        sCtx.stroke();
        sCtx.setLineDash([]);

        sCtx.fillStyle = '#ffffff';
        sCtx.beginPath();
        sCtx.arc(hX, hY, 4, 0, Math.PI * 2);
        sCtx.fill();

        const label = `${Math.round(activePt.elevation ?? 0)} msnm | ${activePt.temp.toFixed(1)}°C | ${activePt.dist_km.toFixed(1)} km`;
        sCtx.font = 'bold 10px monospace';
        const textWidth = sCtx.measureText(label).width;
        let boxX = hX + 8;
        if (boxX + textWidth + 8 > w) boxX = hX - textWidth - 16;

        sCtx.fillStyle = 'rgba(15, 23, 42, 0.95)';
        sCtx.fillRect(boxX, 6, textWidth + 10, 18);
        sCtx.strokeStyle = '#38bdf8';
        sCtx.lineWidth = 1;
        sCtx.strokeRect(boxX, 6, textWidth + 10, 18);

        sCtx.fillStyle = '#38bdf8';
        sCtx.fillText(label, boxX + 5, 19);
    }

    // 5. Escala en metros
    sCtx.fillStyle = '#94a3b8';
    sCtx.font = '10px monospace';
    sCtx.fillText(`${minElev}m`, 4, h - 4);
    sCtx.fillText(`${maxElev}m`, 4, 12);
}

if (sliceCanvas) {
    sliceCanvas.addEventListener('mousemove', (e) => {
        if (!sliceDataCache) return;
        const rect = sliceCanvas.getBoundingClientRect();
        const mouseX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const pts = sliceDataCache.points;

        const targetDist = (mouseX / rect.width) * sliceDataCache.total_dist_km;

        let closestIdx = 0;
        let minDiff = Infinity;
        for (let i = 0; i < pts.length; i++) {
            const diff = Math.abs(pts[i].dist_km - targetDist);
            if (diff < minDiff) {
                minDiff = diff;
                closestIdx = i;
            }
        }

        hoverIndex = closestIdx;
        const pt = pts[closestIdx];

        sliceHoverMarker.setLatLng([pt.lat, pt.lon]);
        if (!map.hasLayer(sliceHoverMarker)) {
            sliceHoverMarker.addTo(map);
        }

        drawSliceProfile();
    });

    sliceCanvas.addEventListener('mouseleave', () => {
        hoverIndex = -1;
        if (map.hasLayer(sliceHoverMarker)) {
            map.removeLayer(sliceHoverMarker);
        }
        drawSliceProfile();
    });
}

// =============================================================================
// 12. INICIALIZACIÓN GLOBAL (BOOTSTRAP)
// =============================================================================

function bootstrapRegion() {
    fetch('/api/metadata')
        .then(r => r.json())
        .then(meta => {
            currentRegionId = meta.region_id || "colima_copernicus";
            LAT_MIN = meta.lat_min;
            LAT_MAX = meta.lat_max;
            LON_MIN = meta.lon_min;
            LON_MAX = meta.lon_max;
            GRID_RES = meta.grid_width;
            TOTAL_TIME_STEPS = meta.time_steps;

            if (regionSelector) {
                regionSelector.value = currentRegionId;
            }

            updateBrandTitle(meta.region_name);

            const dLatHalf = (LAT_MAX - LAT_MIN) / (GRID_RES - 1) / 2.0;
            const dLonHalf = (LON_MAX - LON_MIN) / (GRID_RES - 1) / 2.0;
            const bounds = [
                [LAT_MIN - dLatHalf, LON_MIN - dLonHalf],
                [LAT_MAX + dLatHalf, LON_MAX + dLonHalf]
            ];

            cudaOverlay.setBounds(bounds);
            map.fitBounds(bounds, {
                padding: [20, 20],
                animate: false
            });

            resizeWindCanvas();
            fetchWindData(currentTime);
            stepWind();

            switchMode('temperature');
            initTimeline();
            loadData(currentTime);
        })
        .catch(err => {
            console.warn("Aviso metadata:", err);
            resizeWindCanvas();
            fetchWindData(currentTime);
            stepWind();
            switchMode('temperature');
            initTimeline();
            loadData(currentTime);
        });
}

bootstrapRegion();

window.addEventListener('resize', () => {
    resizeWindCanvas();
    drawTimeline();
    applyOverlayMask();
    if (showWind) fetchWindData(currentTime);
});