# ml-anomaly-pipeline

An end-to-end ML pipeline for image anomaly detection. Trained on normal images, flags anything that looks off at inference time.

Built as a hands-on way to work through the full MLOps stack — not just training a model, but actually shipping it.

---

## What's inside

```
src/training/    — model, dataset loader, training script, evaluation
src/serving/     — FastAPI app that loads the model and serves predictions
src/monitoring/  — drift detection using Evidently
infra/terraform/ — AWS infra (S3 + ECR)
.github/workflows/ — CI/CD: train → quality gate → build → deploy
```

## How it works

The model is a convolutional autoencoder. It trains only on "normal" samples, so it gets good at reconstructing them. At inference, anything with high reconstruction error gets flagged as an anomaly.

AUROC: **0.992** on the test set (normal class vs. all others).

---

## Running locally

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python -m src.training.train --epochs 15

# Check experiment runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000

# Serve
uvicorn src.serving.app:app --reload
# → http://localhost:8000
```

## Running with Docker

```bash
docker build -t anomaly-detector .
docker run -d -p 8000:8000 anomaly-detector

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@your_image.png"
```

Response:
```json
{
  "anomaly_score": 0.069,
  "is_anomaly": true,
  "threshold": 0.034,
  "model_name": "anomaly-detector",
  "model_version": "1"
}
```

---

## Infra (AWS)

S3 and ECR are provisioned with Terraform. No manual console clicks.

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

Creates:
- S3 bucket for model artifacts (versioned, AES256 encrypted)
- ECR repository for the Docker image

## CI/CD

GitHub Actions runs on every push to `main`:

1. Retrain the model
2. Check AUROC — if it drops below 0.90, the deploy is blocked
3. Build Docker image and push to ECR
4. Deploy to EC2

## Drift monitoring

Evidently checks whether incoming image distributions have shifted from the training set. Runs daily via cron. Opens a GitHub issue automatically if drift is detected.

```bash
python -m src.monitoring.drift
```

---

## Stack

PyTorch · MLflow · FastAPI · Docker · Terraform · AWS (S3, ECR) · GitHub Actions · Evidently
