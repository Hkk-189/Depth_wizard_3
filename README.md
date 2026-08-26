# DepthWizard 3

**Unified 3D Terrain & DEM Elevation Analyzer with Extremes Location Mapping & WASD Navigation**

Transform any standard 2D image (monocular depth via **Depth Anything V2**) or **GeoTIFF DEM** (true metric elevation via **rasterio**) into an interactive 3D WebGL terrain model and extract comprehensive elevation/depth statistics, including **approximate peak and valley locations**, from a **single unified input**.

---

## Features

- **Unified Single-Input Workflow**: Drop either a standard image (`.jpg`, `.png`, `.webp`, `.bmp`) or a GeoTIFF (`.tif`, `.tiff`), and DepthWizard will automatically generate both the interactive 3D terrain mesh and complete elevation/depth statistics.
- **Min/Max Approximate Location Extraction**:
  - **Peak (Max Elevation)** & **Valley (Min Elevation)** locations reported in geographic coordinates (e.g. Lat/Lon), pixel space $(X, Y)$, relative percentage offsets $(\sim X\%, \sim Y\%)$, and quadrant/sector direction (e.g. *North-West*, *South-Central*).
  - **Uniform Hovering 3D Location Dots**: Both the peak (🔴 Red) and valley (🔵 Blue) glowing beacon dots hover on the **exact same horizontal elevation plane** above the 3D model, with adaptive guide lines reaching down to their ground surface points.
- **Navigation & Controls**:
  - **WASD / Arrow Keys**: Smooth 60 FPS flight/glide movement across the terrain in horizontal perspective.
  - **Zoom In / Out**: Dedicated UI zoom buttons (`➕` / `➖`), keyboard hotkeys (`+` / `-`), and mouse scroll wheel.
  - **Mouse Orbit & Pan**: Left-click drag to rotate/orbit, right-click drag to pan.
  - **Interactive Sliders**: Real-time **Height Scale** vertical exaggeration slider, **Wireframe Mode**, **Auto-Rotate turntable**, and **Camera Reset**.
- **AI Monocular Depth Estimation**: Utilizes `depth-anything/Depth-Anything-V2-Small-hf` via PyTorch for fast, relative depth inference.
- **GeoTIFF DEM Elevation Processing**: Reads georeferenced rasters, masks nodata & sentinel values, calculates min/max/range elevation in meters, spatial coordinates, and coordinate reference system (CRS).
- **Non-blocking Concurrency**: FastAPI threadpool execution for CPU/GPU-bound tasks to maintain a responsive event loop.

---

## Tech Stack

- **Backend**: FastAPI, PyTorch (CUDA / MPS / CPU), Hugging Face Transformers, Rasterio, OpenCV, NumPy, Pillow
- **Frontend**: Three.js (r128), OrbitControls, HTML5 / CSS3 Glassmorphism UI

---

## Setup & Installation

```bash
# 1. Activate virtual environment
source myenv/bin/activate   # or create your own: python3 -m venv myenv

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# Run server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** directly in your browser.

---

## Keyboard & Navigation Controls

| Control | Action |
| :--- | :--- |
| **`W` / `↑`** | Move / Glide forward across the terrain |
| **`S` / `↓`** | Move / Glide backward |
| **`A` / `←`** | Move / Strafe left |
| **`D` / `→`** | Move / Strafe right |
| **`+` / `=` / `➕` Button** | Zoom in towards center |
| **`-` / `_` / `➖` Button** | Zoom out away from center |
| **`Home` / `🏠` Button** | Reset camera view to default 3D isometric angle |
| **Left Click + Drag** | Orbit / Rotate camera |
| **Right Click + Drag** | Pan camera |
| **Mouse Wheel** | Scroll zoom |

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the Three.js frontend web app (`index.html`) |
| `POST` | `/process` | **Unified Endpoint**: Accepts multipart `file` (Image or GeoTIFF). Returns 3D mesh data (`depth_data`, `rgb_b64`), elevation extremes (`min_val`, `max_val`), and approximate locations (`min_location`, `max_location` with pixel, percent, sector, geo coords, and 3D grid). |
| `POST` | `/generate-3d` | Backwards-compatible 3D generation endpoint. |
| `POST` | `/elevation-extremes` | Backwards-compatible DEM elevation extraction endpoint. |
| `GET` | `/health` | Healthcheck returning device status (CUDA/MPS/CPU), model name, and limits. |

*Upload Limit*: 200 MB per file.
