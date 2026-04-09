"""
Data and model drift monitoring using Evidently.

Two checks:
  1. Data drift  — are incoming image features drifting from training distribution?
  2. Model drift — is the anomaly score distribution shifting over time?

Run this on a schedule (e.g. daily cron or GitHub Actions) against a window
of recent prediction logs vs the training reference set.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_image_features(images: np.ndarray) -> pd.DataFrame:
    """
    Flatten images into pixel statistics per image.
    In production, use encoder embeddings instead — richer signal.
    """
    n = images.shape[0]
    flat = images.reshape(n, -1)
    return pd.DataFrame({
        "mean_pixel": flat.mean(axis=1),
        "std_pixel": flat.std(axis=1),
        "min_pixel": flat.min(axis=1),
        "max_pixel": flat.max(axis=1),
        "p25_pixel": np.percentile(flat, 25, axis=1),
        "p75_pixel": np.percentile(flat, 75, axis=1),
    })


def run_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "./monitoring_reports",
) -> dict:
    """
    Compares current window to reference (training distribution).
    Returns a summary dict and saves an HTML report.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(report_path)
    log.info(f"Drift report saved: {report_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    n_drifted = result["metrics"][0]["result"]["number_of_drifted_columns"]
    n_total = result["metrics"][0]["result"]["number_of_columns"]

    summary = {
        "timestamp": timestamp,
        "drift_detected": drift_detected,
        "drifted_columns": n_drifted,
        "total_columns": n_total,
        "drift_share": n_drifted / n_total,
        "report_path": report_path,
    }

    log.info(f"Drift summary: {summary}")

    if drift_detected:
        log.warning(
            f"DATA DRIFT DETECTED: {n_drifted}/{n_total} features drifted. "
            "Consider retraining."
        )

    return summary


def check_score_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold_shift: float = 0.005,
) -> dict:
    """
    Detect model output drift by comparing anomaly score distributions.
    A significant mean shift may indicate model degradation or data shift.
    """
    ref_mean = float(reference_scores.mean())
    cur_mean = float(current_scores.mean())
    shift = abs(cur_mean - ref_mean)

    result = {
        "reference_mean_score": ref_mean,
        "current_mean_score": cur_mean,
        "absolute_shift": shift,
        "drift_detected": shift > threshold_shift,
    }

    if result["drift_detected"]:
        log.warning(
            f"SCORE DRIFT DETECTED: mean shifted by {shift:.4f} "
            f"(ref={ref_mean:.4f}, cur={cur_mean:.4f})"
        )

    return result


if __name__ == "__main__":
    # Simulate a drift check with synthetic data
    log.info("Running simulated drift check...")

    rng = np.random.default_rng(42)
    ref_images = rng.uniform(0, 1, (500, 32 * 32)).reshape(500, 1, 32, 32)
    cur_images_no_drift = rng.uniform(0, 1, (100, 32 * 32)).reshape(100, 1, 32, 32)
    cur_images_drifted = rng.uniform(0.4, 1.4, (100, 32 * 32)).reshape(100, 1, 32, 32)

    ref_df = extract_image_features(ref_images)
    cur_df_ok = extract_image_features(cur_images_no_drift)
    cur_df_drift = extract_image_features(cur_images_drifted)

    print("\n--- No drift scenario ---")
    run_drift_report(ref_df, cur_df_ok)

    print("\n--- Drift scenario ---")
    run_drift_report(ref_df, cur_df_drift)
