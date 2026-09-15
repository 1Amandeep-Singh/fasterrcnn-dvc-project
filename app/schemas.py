from typing import List

from pydantic import BaseModel, Field


class Detection(BaseModel):
    label_id: int
    label_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    box: List[float] = Field(..., description="[x1, y1, x2, y2] in pixel coordinates")


class PredictionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    inference_time_ms: float
    detections: List[Detection]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    num_classes: int
