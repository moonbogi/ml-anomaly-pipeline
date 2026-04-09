"""
Training entrypoint with MLflow experiment tracking.

Usage:
    python -m src.training.train
    python -m src.training.train --latent-dim 64 --epochs 20 --lr 0.0005

Everything is logged to MLflow:
  - hyperparameters
  - per-epoch train loss
  - final evaluation metrics (AUROC)
  - the trained model artifact
"""

import argparse
import logging
import os
import sys

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.dataset import get_normal_dataloader, get_test_dataloader
from src.training.evaluate import evaluate_auroc
from src.training.model import ConvAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "anomaly-detection"


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection autoencoder")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--normal-class", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    train_loader = get_normal_dataloader(
        data_dir=args.data_dir,
        normal_class=args.normal_class,
        img_size=args.img_size,
        batch_size=args.batch_size,
        train=True,
    )
    test_loader = get_test_dataloader(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )

    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params({
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "normal_class": args.normal_class,
            "img_size": args.img_size,
            "device": str(device),
        })

        log.info(f"Starting training: {args.epochs} epochs")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            log.info(f"Epoch {epoch}/{args.epochs} — loss: {train_loss:.6f}")

        # Evaluate
        log.info("Evaluating on test set...")
        metrics = evaluate_auroc(model, test_loader, args.normal_class, device)
        mlflow.log_metrics(metrics)
        log.info(f"Evaluation: {metrics}")

        # Log model to MLflow Model Registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="anomaly-detector",
        )
        log.info("Model logged to MLflow registry as 'anomaly-detector'")

        run_id = mlflow.active_run().info.run_id
        log.info(f"MLflow run ID: {run_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
