"""
Tests for Clinical AI Ops Agent
================================
Unit tests for drift detection, deployment safety gate, and alert publishing.
All tests run without AWS credentials using mocks.
"""

from unittest.mock import patch

import numpy as np

# ── Drift Detector Tests ───────────────────────────────────────────────────────

class TestDriftDetector:

    def setup_method(self):
        """Set up DriftDetector with mocked AWS clients."""
        with patch("boto3.client"):
            from agent.tools.drift_detector import DriftDetector
            self.detector = DriftDetector()
            # Use synthetic data (no S3 required)
            self.baseline = self.detector._synthetic_baseline()
            self.recent_stable = self.detector._synthetic_recent(drifted=False)
            self.recent_drifted = self.detector._synthetic_recent(drifted=True)

    def test_psi_stable_distribution(self):
        """PSI should be low when distributions are identical."""
        psi = self.detector._compute_psi(
            self.baseline["lab_glucose"].values,
            self.baseline["lab_glucose"].values,  # Same data
        )
        assert psi < 0.05, f"Expected PSI < 0.05 for identical distributions, got {psi:.4f}"

    def test_psi_drifted_distribution(self):
        """PSI should be high when distributions shift significantly."""
        baseline = np.random.normal(100, 15, 5000)
        drifted = np.random.normal(150, 30, 800)   # Mean shifted by 50%
        psi = self.detector._compute_psi(baseline, drifted)
        assert psi > 0.20, f"Expected PSI > 0.20 for shifted distribution, got {psi:.4f}"

    def test_psi_clip_avoids_division_by_zero(self):
        """PSI computation should not raise even with edge case distributions."""
        baseline = np.ones(100)          # All same value
        current = np.zeros(100)
        psi = self.detector._compute_psi(baseline, current)
        assert np.isfinite(psi), "PSI should be finite even with edge case data"

    def test_run_stable_returns_stable_verdict(self):
        """Full run on stable data should return stable verdict."""
        with patch.object(self.detector, "_load_baseline", return_value=self.baseline), \
             patch.object(self.detector, "_load_recent_predictions", return_value=self.recent_stable):
            report = self.detector.run("test-endpoint", lookback_hours=6)

        assert report["verdict"] == "stable"
        assert report["drifted_features"] == []
        assert "psi_scores" in report
        assert report["n_baseline_samples"] == len(self.baseline)

    def test_run_drifted_returns_retrain_verdict(self):
        """Full run on drifted data should return drift verdict and identify feature."""
        with patch.object(self.detector, "_load_baseline", return_value=self.baseline), \
             patch.object(self.detector, "_load_recent_predictions", return_value=self.recent_drifted):
            report = self.detector.run("test-endpoint", lookback_hours=6)

        assert "drift" in report["verdict"]
        assert "lab_glucose" in report["drifted_features"]
        assert report["psi_scores"]["lab_glucose"] > 0.20

    def test_run_empty_predictions_returns_skipped(self):
        """If no recent predictions exist, agent should skip gracefully."""
        import pandas as pd
        with patch.object(self.detector, "_load_baseline", return_value=self.baseline), \
             patch.object(self.detector, "_load_recent_predictions", return_value=pd.DataFrame()):
            report = self.detector.run("test-endpoint", lookback_hours=6)

        assert report["verdict"] == "skipped"
        assert report["reason"] == "no_recent_data"

    def test_critical_drift_sets_flag(self):
        """PSI > 0.25 should set critical_drift=True."""
        import pandas as pd
        # Inject severe drift
        severe_drift = self.baseline.copy()
        severe_drift["lab_glucose"] = np.random.normal(250, 60, len(self.baseline))

        with patch.object(self.detector, "_load_baseline", return_value=self.baseline), \
             patch.object(self.detector, "_load_recent_predictions", return_value=severe_drift):
            report = self.detector.run("test-endpoint", lookback_hours=6)

        assert report["critical_drift"] is True


# ── Deployment Manager Tests ───────────────────────────────────────────────────

class TestDeploymentManager:

    def setup_method(self):
        with patch("boto3.client"):
            from agent.tools.deployment_manager import DeploymentManager
            self.manager = DeploymentManager()

    def test_safety_gate_blocks_low_f1(self):
        """Deployment should be halted when F1 < threshold."""
        result = self.manager.deploy(
            endpoint_name="test-endpoint",
            model_artifact_s3="s3://bucket/model.tar.gz",
            validation_f1=0.65,   # Below 0.80 threshold
        )
        assert result["status"] == "halted"
        assert result["production_unchanged"] is True
        assert "0.65" in result["reason"]

    def test_safety_gate_passes_sufficient_f1(self):
        """Deployment should proceed when F1 >= threshold."""
        with patch.object(
            self.manager, "_create_model"
        ), patch.object(
            self.manager, "_create_endpoint_config"
        ), patch.object(
            self.manager, "_update_endpoint"
        ), patch.object(
            self.manager, "_shift_traffic"
        ):
            result = self.manager.deploy(
                endpoint_name="test-endpoint",
                model_artifact_s3="s3://bucket/model.tar.gz",
                validation_f1=0.85,
            )

        assert result["status"] == "deployed"
        assert result["validation_f1"] == 0.85

    def test_safety_gate_at_exact_threshold(self):
        """Deployment should proceed at exactly the threshold (>=, not >)."""
        with patch.object(self.manager, "_create_model"), \
             patch.object(self.manager, "_create_endpoint_config"), \
             patch.object(self.manager, "_update_endpoint"), \
             patch.object(self.manager, "_shift_traffic"):
            result = self.manager.deploy(
                endpoint_name="test-endpoint",
                model_artifact_s3="s3://bucket/model.tar.gz",
                validation_f1=0.80,  # Exactly at threshold
            )
        assert result["status"] == "deployed"

    def test_simulation_mode_on_missing_endpoint(self):
        """Should fall back to simulation when endpoint doesn't exist in AWS."""
        from botocore.exceptions import ClientError

        def raise_not_found(*args, **kwargs):
            raise ClientError(
                {"Error": {"Code": "ValidationException", "Message": "could not be found"}},
                "CreateModel",
            )

        with patch.object(self.manager, "_create_model", side_effect=raise_not_found):
            result = self.manager.deploy(
                endpoint_name="nonexistent-endpoint",
                model_artifact_s3="s3://bucket/model.tar.gz",
                validation_f1=0.88,
            )

        assert result["status"] == "deployed"
        assert result.get("simulated") is True


# ── Alert Publisher Tests ──────────────────────────────────────────────────────

class TestAlertPublisher:

    def setup_method(self):
        with patch("boto3.client"):
            from agent.tools.alert_publisher import AlertPublisher
            self.publisher = AlertPublisher()

    def test_publish_skips_sns_when_no_topic_arn(self):
        """SNS publish should be skipped gracefully when ARN is not configured."""
        import os
        original = os.environ.pop("SNS_TOPIC_ARN", None)
        try:
            result = self.publisher._publish_sns("test message", "Test Subject")
            assert result["status"] == "skipped"
            assert result["reason"] == "no_topic_arn"
        finally:
            if original:
                os.environ["SNS_TOPIC_ARN"] = original

    def test_publish_skips_slack_when_no_webhook(self):
        """Slack publish should be skipped gracefully when webhook is not configured."""
        import os
        original = os.environ.pop("SLACK_WEBHOOK_URL", None)
        try:
            result = self.publisher._publish_slack("test message", "warning", "test-endpoint")
            assert result["status"] == "skipped"
            assert result["reason"] == "no_webhook_url"
        finally:
            if original:
                os.environ["SLACK_WEBHOOK_URL"] = original

    def test_full_publish_returns_channel_statuses(self):
        """publish() should return status for both SNS and Slack channels."""
        result = self.publisher.publish(
            message="Drift detected on lab_glucose",
            severity="warning",
            endpoint_name="clinical-risk-model-v2",
        )
        assert "sns" in result
        assert "slack" in result
        assert result["severity"] == "warning"
        assert result["endpoint_name"] == "clinical-risk-model-v2"
        assert "timestamp" in result


# ── Integration: End-to-End Demo ───────────────────────────────────────────────

class TestDemoIntegration:
    """Verify the demo script runs without errors."""

    def test_simulate_drift_moderate(self, capsys):
        """Demo should run the moderate drift scenario without raising."""
        import sys
        sys.argv = ["simulate_drift.py", "--scenario", "moderate_drift"]
        try:
            from demo.simulate_drift import run_scenario
            run_scenario("moderate_drift")
            captured = capsys.readouterr()
            assert "Deployment complete" in captured.out or "deployed" in captured.out.lower()
        except SystemExit:
            pass  # argparse may call sys.exit on help

    def test_simulate_drift_no_drift(self, capsys):
        """Demo should show stable verdict for no_drift scenario."""
        from demo.simulate_drift import run_scenario
        run_scenario("no_drift")
        captured = capsys.readouterr()
        assert "stable" in captured.out.lower() or "No drift" in captured.out
