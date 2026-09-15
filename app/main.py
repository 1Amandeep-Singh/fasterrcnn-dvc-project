"""
FastAPI inference service for the DVC-trained Faster R-CNN model.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Then either open http://localhost:8000/docs for the interactive Swagger UI,
or use the Postman collection in postman/FasterRCNN_API.postman_collection.json.
"""
import io
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.transforms import functional as F

# allow `import model` from src/ without turning src into an installed package
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from model import build_model  # noqa: E402

from app.schemas import Detection, HealthResponse, PredictionResponse

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

with open(ROOT_DIR / "params.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() and CFG["train"]["device"] == "cuda" else "cpu")
NUM_CLASSES = CFG["train"]["num_classes"]
MODEL_PATH = ROOT_DIR / CFG["paths"]["model_out"]
LABEL_MAP_PATH = ROOT_DIR / CFG["paths"]["label_map"]

app = FastAPI(
    title="Faster R-CNN Object Detection API",
    description="Serves a DVC-trained torchvision Faster R-CNN model for object detection inference.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
label_map: dict[int, str] = {}


@app.on_event("startup")
def load_model():
    """Loads model weights once at startup rather than per-request."""
    global model, label_map

    if not MODEL_PATH.exists():
        print(f"[startup] WARNING: model weights not found at {MODEL_PATH}. "
              f"Run `dvc repro` to train first. /predict will 503 until then.")
        return

    m = build_model(NUM_CLASSES, pretrained_backbone=False)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.eval()
    model = m

    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

    print(f"[startup] model loaded on {DEVICE}, {len(label_map)} classes")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        model_loaded=model is not None,
        num_classes=NUM_CLASSES,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image file (jpg/png) to run detection on"),
    score_threshold: float = 0.5,
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model with `dvc repro` first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Expected an image file, got {file.content_type}")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    tensor = F.to_tensor(image).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad():
        output = model([tensor])[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if score.item() < score_threshold:
            continue
        label_id = int(label.item())
        detections.append(Detection(
            label_id=label_id,
            label_name=label_map.get(label_id, f"class_{label_id}"),
            score=round(float(score.item()), 4),
            box=[round(v, 2) for v in box.tolist()],
        ))

    return PredictionResponse(
        filename=file.filename,
        image_width=image.width,
        image_height=image.height,
        inference_time_ms=round(elapsed_ms, 2),
        detections=detections,
    )
