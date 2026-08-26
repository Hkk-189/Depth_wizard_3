import io
import base64
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
import rasterio
from rasterio.io import MemoryFile
from rasterio.errors import NotGeoreferencedWarning
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/tif"
}
ALLOWED_ALL_MIMES = ALLOWED_IMAGE_MIMES | {"application/octet-stream"}
ALL_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

GRID_SIZE = 256
TEXTURE_SIZE = 512
HEIGHT_SCALE = 30.0

image_processor = None
model = None


def load_model_if_needed():
    global image_processor, model
    if image_processor is None or model is None:
        print(f"Loading Depth Anything V2 on {DEVICE}...")
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
        print("Model loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_if_needed()
    yield


app = FastAPI(title="DepthWizard 3", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _suffix_ok(filename: str | None, suffixes: tuple) -> bool:
    if not filename:
        return False
    return filename.lower().endswith(suffixes)


def read_file_capped(file: UploadFile, allowed_mimes: set, suffixes: tuple, label: str = "file") -> bytes:
    contents = file.file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{label} too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")
    mime = (file.content_type or "").lower()
    if mime not in allowed_mimes and not _suffix_ok(file.filename, suffixes):
        raise HTTPException(415, f"unsupported {label} type: mime={mime!r}, filename={file.filename!r}")
    if len(contents) == 0:
        raise HTTPException(400, f"empty {label}")
    return contents


def _describe_location(row: int, col: int, height: int, width: int, geo_xy: list[float] | None = None) -> dict:
    """Calculates human-readable, pixel, percentage, and 3D grid approximate location for extremes."""
    x_pct = round(float((col / max(1, width)) * 100.0), 1)
    y_pct = round(float((row / max(1, height)) * 100.0), 1)

    v_dir = "North (Top)" if y_pct < 33.3 else ("South (Bottom)" if y_pct > 66.7 else "Center")
    h_dir = "West (Left)" if x_pct < 33.3 else ("East (Right)" if x_pct > 66.7 else "Center")

    if v_dir == "Center" and h_dir == "Center":
        sector = "Center"
    elif v_dir == "Center":
        sector = h_dir
    elif h_dir == "Center":
        sector = v_dir
    else:
        sector = f"{v_dir.split(' ')[0]}-{h_dir.split(' ')[0]}"

    grid_r = int(round(row * (GRID_SIZE - 1) / max(1, height - 1)))
    grid_c = int(round(col * (GRID_SIZE - 1) / max(1, width - 1)))

    desc = f"{sector} (~{x_pct}% X, ~{y_pct}% Y)"
    if geo_xy:
        desc += f" @ ({geo_xy[0]:.4f}, {geo_xy[1]:.4f})"
    else:
        desc += f" [Pixel: {col}x, {row}y]"

    return {
        "pixel": [int(col), int(row)],
        "percentage": [x_pct, y_pct],
        "sector": sector,
        "grid_3d": [grid_c, grid_r],
        "geo_coords": [round(float(geo_xy[0]), 4), round(float(geo_xy[1]), 4)] if geo_xy else None,
        "description": desc,
    }


def _infer_depth_ai(rgb_image: np.ndarray) -> np.ndarray:
    """Runs Depth Anything V2 inference on an RGB NumPy image array (H, W, 3)."""
    load_model_if_needed()

    pil_img = Image.fromarray(rgb_image)
    inputs = image_processor(images=pil_img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth = prediction.squeeze().cpu().numpy()

    denom = float(depth.max() - depth.min())
    if denom > 1e-6:
        depth_norm = (depth - depth.min()) / denom
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)

    return depth_norm


def process_dem_geotiff(contents: bytes, filename: str | None = None) -> dict | None:
    """Attempts to read contents as a GeoTIFF / DEM using rasterio."""
    try:
        with MemoryFile(contents) as memfile:
            with memfile.open() as ds:
                is_gtiff_driver = (ds.driver == "GTiff")
                has_crs = (ds.crs is not None)
                is_tif_extension = bool(filename and filename.lower().endswith((".tif", ".tiff")))

                if not (is_gtiff_driver or has_crs or is_tif_extension):
                    return None

                if ds.count < 1:
                    return None

                if ds.count == 1:
                    band = ds.read(1).astype(np.float32)
                    nodata = ds.nodata
                    mask = np.zeros(band.shape, dtype=bool)

                    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
                        mask |= (band == nodata)
                    mask |= np.isnan(band)

                    if nodata is None:
                        for sentinel in (-9999.0, -32768.0, -99999.0):
                            mask |= (band == sentinel)

                    valid = np.where(mask, np.nan, band)
                    if np.all(np.isnan(valid)):
                        return None

                    min_v = float(np.nanmin(valid))
                    max_v = float(np.nanmax(valid))
                    min_idx = np.unravel_index(int(np.nanargmin(valid)), valid.shape)
                    max_idx = np.unravel_index(int(np.nanargmax(valid)), valid.shape)

                    h, w = band.shape
                    try:
                        min_xy = [float(x) for x in ds.xy(min_idx[0], min_idx[1])]
                        max_xy = [float(x) for x in ds.xy(max_idx[0], max_idx[1])]
                    except Exception:
                        min_xy = [float(min_idx[1]), float(min_idx[0])]
                        max_xy = [float(max_idx[1]), float(max_idx[0])]

                    min_loc = _describe_location(int(min_idx[0]), int(min_idx[1]), h, w, min_xy)
                    max_loc = _describe_location(int(max_idx[0]), int(max_idx[1]), h, w, max_xy)

                    crs_str = str(ds.crs) if ds.crs else "Non-projected / Local"
                    valid_count = int(np.count_nonzero(~np.isnan(valid)))

                    denom = max_v - min_v if max_v > min_v else 1.0
                    normalized = np.where(mask, min_v, band)
                    normalized = np.clip((normalized - min_v) / denom, 0.0, 1.0)

                    depth_resized = cv2.resize(normalized, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

                    # Colored DEM texture
                    tex_uint8 = cv2.resize((normalized * 255.0).astype(np.uint8), (TEXTURE_SIZE, TEXTURE_SIZE), interpolation=cv2.INTER_AREA)
                    tex_colored = cv2.applyColorMap(tex_uint8, cv2.COLORMAP_TURBO)
                    ok, buf = cv2.imencode(".jpg", tex_colored)
                    if not ok:
                        raise HTTPException(500, "texture encoding failed")
                    rgb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                    return {
                        "grid_size": GRID_SIZE,
                        "height_scale": HEIGHT_SCALE,
                        "depth_data": depth_resized.tolist(),
                        "rgb_b64": rgb_b64,
                        "texture_mime": "image/jpeg",
                        "is_geotiff": True,
                        "min_location": min_loc,
                        "max_location": max_loc,
                        "elevation": {
                            "is_metric": True,
                            "min_val": min_v,
                            "max_val": max_v,
                            "range_val": max_v - min_v,
                            "unit": "m",
                            "min_xy": min_xy,
                            "max_xy": max_xy,
                            "min_location": min_loc,
                            "max_location": max_loc,
                            "crs": crs_str,
                            "valid_pixel_count": valid_count,
                            "nodata_value": nodata,
                            "shape": list(band.shape),
                            "source_type": "GeoTIFF DEM (Metric Elevation)",
                        }
                    }

                elif ds.count >= 3 and (has_crs or is_gtiff_driver):
                    r = ds.read(1)
                    g = ds.read(2)
                    b = ds.read(3)
                    rgb = np.stack([r, g, b], axis=-1)
                    if rgb.dtype != np.uint8:
                        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)

                    depth_norm = _infer_depth_ai(rgb)
                    depth_resized = cv2.resize(depth_norm, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

                    texture = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (TEXTURE_SIZE, TEXTURE_SIZE))
                    ok, buf = cv2.imencode(".jpg", texture)
                    if not ok:
                        raise HTTPException(500, "texture encoding failed")
                    rgb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                    min_idx = np.unravel_index(int(np.argmin(depth_norm)), depth_norm.shape)
                    max_idx = np.unravel_index(int(np.argmax(depth_norm)), depth_norm.shape)
                    h, w = depth_norm.shape

                    min_loc = _describe_location(int(min_idx[0]), int(min_idx[1]), h, w)
                    max_loc = _describe_location(int(max_idx[0]), int(max_idx[1]), h, w)

                    crs_str = str(ds.crs) if ds.crs else "Georeferenced RGB"
                    return {
                        "grid_size": GRID_SIZE,
                        "height_scale": HEIGHT_SCALE,
                        "depth_data": depth_resized.tolist(),
                        "rgb_b64": rgb_b64,
                        "texture_mime": "image/jpeg",
                        "is_geotiff": True,
                        "min_location": min_loc,
                        "max_location": max_loc,
                        "elevation": {
                            "is_metric": False,
                            "min_val": 0.0,
                            "max_val": 1.0,
                            "range_val": 1.0,
                            "unit": "norm",
                            "min_xy": [float(ds.bounds.left), float(ds.bounds.bottom)] if ds.bounds else [0.0, 0.0],
                            "max_xy": [float(ds.bounds.right), float(ds.bounds.top)] if ds.bounds else [float(rgb.shape[1]), float(rgb.shape[0])],
                            "min_location": min_loc,
                            "max_location": max_loc,
                            "crs": crs_str,
                            "valid_pixel_count": int(rgb.shape[0] * rgb.shape[1]),
                            "nodata_value": ds.nodata,
                            "shape": [int(rgb.shape[0]), int(rgb.shape[1])],
                            "source_type": "Multi-band GeoTIFF + Depth Anything V2",
                        }
                    }
    except Exception:
        return None
    return None


def process_standard_image(contents: bytes) -> dict:
    """Decodes standard image formats and runs Depth Anything V2 depth estimation."""
    raw = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise HTTPException(400, "could not decode image file")

    h, w = raw.shape[:2]
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    depth_norm = _infer_depth_ai(rgb)
    depth_resized = cv2.resize(depth_norm, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

    texture = cv2.resize(raw, (TEXTURE_SIZE, TEXTURE_SIZE))
    ok, buf = cv2.imencode(".jpg", texture)
    if not ok:
        raise HTTPException(500, "jpeg texture encoding failed")
    rgb_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    # Calculate min and max locations in relative depth map
    min_idx = np.unravel_index(int(np.argmin(depth_norm)), depth_norm.shape)
    max_idx = np.unravel_index(int(np.argmax(depth_norm)), depth_norm.shape)

    min_loc = _describe_location(int(min_idx[0]), int(min_idx[1]), h, w)
    max_loc = _describe_location(int(max_idx[0]), int(max_idx[1]), h, w)

    return {
        "grid_size": GRID_SIZE,
        "height_scale": HEIGHT_SCALE,
        "depth_data": depth_resized.tolist(),
        "rgb_b64": rgb_b64,
        "texture_mime": "image/jpeg",
        "is_geotiff": False,
        "min_location": min_loc,
        "max_location": max_loc,
        "elevation": {
            "is_metric": False,
            "min_val": 0.0,
            "max_val": 1.0,
            "range_val": 1.0,
            "unit": "norm",
            "min_xy": [float(min_idx[1]), float(min_idx[0])],
            "max_xy": [float(max_idx[1]), float(max_idx[0])],
            "min_location": min_loc,
            "max_location": max_loc,
            "crs": "Relative Depth (AI Estimated)",
            "valid_pixel_count": int(h * w),
            "nodata_value": None,
            "shape": [int(h), int(w)],
            "source_type": "Monocular Image (Depth Anything V2)",
        }
    }


def process_unified_file(contents: bytes, filename: str | None = None) -> dict:
    """Unified entrypoint: intelligently handles GeoTIFFs or standard 2D images."""
    dem_res = process_dem_geotiff(contents, filename)
    if dem_res is not None:
        return dem_res
    return process_standard_image(contents)


@app.get("/")
def serve_index():
    index_file = Path(__file__).parent / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(404, "index.html not found")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "model": MODEL_NAME,
        "grid_size": GRID_SIZE,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.post("/process")
def process_endpoint(file: UploadFile = File(...)):
    """Unified endpoint accepting either a GeoTIFF (.tif) or any 2D image (JPG, PNG, WebP, etc.)."""
    contents = read_file_capped(file, ALLOWED_ALL_MIMES, ALL_SUFFIXES, "input file")
    result = process_unified_file(contents, file.filename)
    return JSONResponse(result)


@app.post("/generate-3d")
def generate_3d_endpoint(file: UploadFile = File(...)):
    """Backwards-compatible endpoint for 3D generation."""
    contents = read_file_capped(file, ALLOWED_ALL_MIMES, ALL_SUFFIXES, "image")
    result = process_unified_file(contents, file.filename)
    return JSONResponse(result)


@app.post("/elevation-extremes")
def elevation_extremes_endpoint(file: UploadFile = File(...)):
    """Backwards-compatible endpoint for DEM elevation calculation."""
    contents = read_file_capped(file, ALLOWED_ALL_MIMES, ALL_SUFFIXES, "tif")
    result = process_unified_file(contents, file.filename)
    if "elevation" in result:
        elev = result["elevation"]
        return JSONResponse({
            "min_m": elev["min_val"],
            "max_m": elev["max_val"],
            "min_xy": elev["min_xy"],
            "max_xy": elev["max_xy"],
            "min_location": elev.get("min_location"),
            "max_location": elev.get("max_location"),
            "crs": elev["crs"],
            "valid_pixel_count": elev["valid_pixel_count"],
            "nodata_value": elev["nodata_value"],
            "shape": elev["shape"],
            "is_metric": elev["is_metric"],
        })
    raise HTTPException(400, "Could not extract elevation extremes from file")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
