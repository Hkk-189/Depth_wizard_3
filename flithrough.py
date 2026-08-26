import io
import numpy as np
import cv2
from PIL import Image
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Hugging Face imports
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Choose model scale: 'Depth-Anything-V2-Small-hf', 'Medium', or 'Large'
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Depth Anything V2 model on {DEVICE}...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model loaded successfully!")

@app.post("/generate-3d")
async def generate_3d(file: UploadFile = File(...)):
    # Read incoming image file
    contents = await file.read()
    raw_img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_img)

    # Preprocess image for model
    inputs = image_processor(images=pil_image, return_tensors="pt").to(DEVICE)

    # Run PyTorch inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate output back to image shape
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=pil_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Normalize depth map to [0, 1] range for WebGL height displacement
    depth_array = prediction.squeeze().cpu().numpy()
    depth_norm = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

    # Resize array (e.g. 256x256 grid) for real-time 60 FPS WebGL mesh rendering
    GRID_SIZE = 256
    depth_resized = cv2.resize(depth_norm, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)

    # Prepare base64 RGB string for texture mapping
    _, buffer = cv2.imencode('.jpg', cv2.resize(raw_img, (512, 512)))
    rgb_hex = buffer.tobytes().hex()

    return JSONResponse({
        "grid_size": GRID_SIZE,
        "depth_data": depth_resized.tolist(),
        "rgb_hex": rgb_hex
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
