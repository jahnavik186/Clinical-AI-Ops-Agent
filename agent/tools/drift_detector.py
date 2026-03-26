"""
Drift Detector
==============
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov tests
to detect data drift in incoming clinical model predictions vs training baseline.

PSI interpretation:
  < 0.10  → No significant change
  0.10–0.20 → Moderate change, monitor
  > 0.20  → Significant drift, retrain recommended
  > 0.25  → Severe drift, page on-call
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("DriftDetector")

# Features considered clinically critical (lower PSI threshold)
CRITICAL_FEATURES = {
    "lab_glucose", "lab_creatinine", "vitals_spo2",
    "vitals_heart_rate", "age", "bmi"
}

PSI_THRESHOLD_WARN = float(os.environ.get("PSI_THRESHOLD_WARN", "0.10"))
PSI_THRESHOLD_RETRAIN = float(os.environ.get("PSI_THRESHOLD_RETRAIN", "0.20"))
PSI_THRESHOLD_CRITICAL = float(os.environ.get("PSI_THRESHOLD_CRITICAL", "0.25"))


class DriftDetector:
    """Detects feature distribution drift for deployed clinical ML models."""

    def __init__(self):
        self.s3 = boto3.client("s3")
        self.bucket = os.environ.get("DATA_BUCKET", "clinical-ai-ops-data")
        self._baseline_cache: dict = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, endpoint_name: str, lookback_hours: int = 6) -> dict:
        """
        Run drift detection for the given endpoint.

        Args:
            endpoint_name: The SageMaker endpoint to check
            lookback_hours: Window of recent prediction logs to analyze

        Returns:
            dict with per-feature PSI scores, KS p-values, verdict, and
            list of drifted features
        """
        logger.info(f"Loading baseline for {endpoint_name}")
        baseline_df = self._load_baseline(endpoint_name)

        logger.info(f"Loading recent predictions (last {lookback_hours}h)")
        recent_df = self._load_recent_predictions(endpoint_name, lookback_hours)

        if recent_df.empty:
            logger.warning("No recent predictions found. Skipping drift check.")
            return self._empty_report(endpoint_name, "no_recent_data")

        common_features = [
            c for c in baseline_df.columns
            if c in recent_df.columns and c != "label"
        ]

        psi_scores = {}
        ks_results = {}
        drifted_features = []
        critical_drift = False

        for feature in common_features:
            psi = self._compute_psi(
                baseline_df[feature].dropna().values,
                recent_df[feature].dropna().values,
            )
            ks_stat, ks_pval = stats.ks_2samp(
                baseline_df[feature].dropna().values,
                recent_df[feature].dropna().values,
            )

            psi_scores[feature] = round(float(psi), 4)
            ks_results[feature] = {
                "statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_pval), 4),
            }

            threshold = (
                PSI_THRESHOLD_RETRAIN * 0.8  # tighter for critical features
                if feature in CRITICAL_FEATURES
                else PSI_THRESHOLD_RETRAIN
            )

            if psi > threshold:
                drifted_features.append(feature)
                logger.warning(f"Drift on '{feature}': PSI={psi:.4f}")
                if psi > PSI_THRESHOLD_CRITICAL:
                    critical_drift = True

        verdict = self._determine_verdict(drifted_features, critical_drift)

        report = {
            "endpoint_name": endpoint_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "lookback_hours": lookback_hours,
            "n_baseline_samples": len(baseline_df),
            "n_recent_samples": len(recent_df),
            "features_checked": len(common_features),
            "psi_scores": psi_scores,
            "ks_results": ks_results,
            "drifted_features": drifted_features,
            "critical_drift": critical_drift,
            "verdict": verdict,
            "thresholds": {
                "warn": PSI_THRESHOLD_WARN,
                "retrain": PSI_THRESHOLD_RETRAIN,
                "critical": PSI_THRESHOLD_CRITICAL,
            },
        }

        logger.info(f"Drift report: verdict={verdict}, drifted={drifted_features}")
        return report

    # ── PSI Computation ────────────────────────────────────────────────────────

    def _compute_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index.
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)
        """
        # Build bins from baseline
        _, bin_edges = np.histogram(baseline, bins=n_bins)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions, clip to avoid log(0)
        baseline_pct = np.clip(baseline_counts / len(baseline), 1e-6, None)
        current_pct = np.clip(current_counts / len(current), 1e-6, None)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)

    # ── Data Loading ───────────────────────────────────────────────────────────

    def _load_baseline(self, endpoint_name: str) -> pd.DataFrame:
        """Load training baseline feature distribution from S3."""
        if endpoint_name in self._baseline_cache:
            return self._baseline_cache[endpoint_name]

        key = f"baselines/{endpoint_name}/feature_baseline.parquet"
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pd.read_parquet(obj["Body"])
            self._baseline_cache[endpoint_name] = df
            return df
        except self.s3.exceptions.NoSuchKey:
            logger.warning(f"No baseline found at s3://{self.bucket}/{key}. Using synthetic.")
            return self._synthetic_baseline()

    def _load_recent_predictions(
        self, endpoint_name: str, lookback_hours: int
    ) -> pd.DataFrame:
        """Load recent prediction request logs from S3."""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        prefix = f"prediction-logs/{endpoint_name}/"

        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            if "Contents" not in response:
                return pd.DataFrame()

            dfs = []
            for obj in response["Contents"]:
                last_modified = obj["LastModified"].replace(tzinfo=None)
                if last_modified >= cutoff:
                    data = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])
                    dfs.append(pd.read_parquet(data["Body"]))

            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        except Exception as e:
            logger.warning(f"Could not load recent predictions: {e}. Using synthetic.")
            return self._synthetic_recent(drifted=True)

    # ── Verdict ────────────────────────────────────────────────────────────────

    def _determine_verdict(self, drifted_features: list, critical_drift: bool) -> str:
        if critical_drift:
            return "critical_drift_retrain_and_alert"
        if drifted_features:
            return "drift_detected_retrain"
        return "stable"

    # ── Synthetic Data for Demo/Tests ──────────────────────────────────────────

    def _synthetic_baseline(self) -> pd.DataFrame:
        """Generate realistic synthetic baseline for local demo."""
        np.random.seed(42)
        n = 5000
        return pd.DataFrame({
            "age": np.random.normal(58, 15, n).clip(18, 95),
            "bmi": np.random.normal(27, 5, n).clip(15, 55),
            "lab_glucose": np.random.normal(105, 25, n).clip(60, 400),
            "lab_creatinine": np.random.normal(1.1, 0.4, n).clip(0.4, 8.0),
            "vitals_spo2": np.random.normal(97, 2, n).clip(85, 100),
            "vitals_heart_rate": np.random.normal(78, 12, n).clip(40, 150),
        })

    def _synthetic_recent(self, drifted: bool = True) -> pd.DataFrame:
        """Generate synthetic recent data — optionally with drift injected."""
        np.random.seed(99)
        n = 800
        df = pd.DataFrame({
            "age": np.random.normal(58, 15, n).clip(18, 95),
            "bmi": np.random.normal(27, 5, n).clip(15, 55),
            "lab_glucose": np.random.normal(105, 25, n).clip(60, 400),
            "lab_creatinine": np.random.normal(1.1, 0.4, n).clip(0.4, 8.0),
            "vitals_spo2": np.random.normal(97, 2, n).clip(85, 100),
            "vitals_heart_rate": np.random.normal(78, 12, n).clip(40, 150),
        })

        if drifted:
            # Simulate a seasonal glucose spike (e.g. holiday diet effect)
            df["lab_glucose"] = np.random.normal(145, 40, n).clip(60, 500)
            logger.info("Synthetic drift injected on 'lab_glucose'")

        return df

    def _empty_report(self, endpoint_name: str, reason: str) -> dict:
        return {
            "endpoint_name": endpoint_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "verdict": "skipped",
            "reason": reason,
            "drifted_features": [],
            "psi_scores": {},
        }
