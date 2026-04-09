"""
Evaluate the trained autoencoder as an anomaly detector.
Computes AUROC — standard metric for anomaly detection.

Normal class → low reconstruction error (label=1 for ROC)
Anomaly class → high reconstruction error (label=0 for ROC)
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.training.model import ConvAutoencoder


def evaluate_auroc(
    model: ConvAutoencoder,
    test_loader: DataLoader,
    normal_class: int,
    device: torch.device,
) -> dict:
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            errors = model.reconstruction_error(images)
            all_scores.extend(errors.cpu().numpy())
            # 1 = normal, 0 = anomaly
            all_labels.extend((labels.numpy() == normal_class).astype(int))

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # Higher error = more anomalous, so negate for AUROC
    auroc = roc_auc_score(labels, -scores)

    # Threshold at 95th percentile of training errors
    threshold = float(np.percentile(scores[labels == 1], 95))
    predictions = (scores < threshold).astype(int)
    accuracy = float((predictions == labels).mean())

    return {
        "auroc": float(auroc),
        "accuracy_at_95pct_threshold": accuracy,
        "anomaly_score_mean_normal": float(scores[labels == 1].mean()),
        "anomaly_score_mean_anomaly": float(scores[labels == 0].mean()),
        "threshold_95pct": threshold,
    }
