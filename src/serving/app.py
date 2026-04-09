"""
FastAPI model serving endpoint for anomaly detection.

Endpoints:
  POST /predict        — score a single image, returns anomaly score + verdict
  GET  /health         — liveness check
  GET  /model/info     — current loaded model version
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow.pytorch
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configurable via env vars — makes swapping models in prod trivial
MODEL_NAME = os.getenv("MODEL_NAME", "anomaly-detector")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_PATH = os.getenv("MODEL_PATH", "")  # If set, load weights directly (portable, no MLflow paths)
LATENT_DIM = int(os.getenv("LATENT_DIM", "32"))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.034"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from src.training.model import ConvAutoencoder
    if MODEL_PATH:
        # Direct file load — portable across environments, no MLflow path resolution
        log.info(f"Loading model weights from {MODEL_PATH}")
        m = ConvAutoencoder(latent_dim=LATENT_DIM)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model = m
    else:
        log.info(f"Loading model {MODEL_NAME} v{MODEL_VERSION} from MLflow registry...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}", map_location=device)
    model.eval()
    log.info("Model loaded successfully")
    yield
    model = None


app = FastAPI(
    title="Anomaly Detection API",
    description="Detects anomalies in images using a convolutional autoencoder",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_name: str
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": ANOMALY_THRESHOLD,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    tensor = transform(image).unsqueeze(0).to(device)
    score = model.reconstruction_error(tensor).item()

    return PredictionResponse(
        anomaly_score=round(score, 6),
        is_anomaly=score > ANOMALY_THRESHOLD,
        threshold=ANOMALY_THRESHOLD,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
