"""
CloudWatch Drift Metrics Publisher
====================================
Publishes custom drift metrics to CloudWatch so teams can build
dashboards and alarms on top of the agent's findings.

Namespace: ClinicalAIOps/DriftMonitoring
"""

import logging
import os
from datetime import datetime

import boto3

logger = logging.getLogger("DriftMetrics")

CW_NAMESPACE = "ClinicalAIOps/DriftMonitoring"


class DriftMetricsPublisher:
    """Publishes PSI and KS drift scores to Amazon CloudWatch."""

    def __init__(self):
        self.cw = boto3.client("cloudwatch")
        self.environment = os.environ.get("ENVIRONMENT", "prod")

    def publish_drift_report(self, report: dict) -> bool:
        """
        Publish all PSI scores from a drift report to CloudWatch.

        Args:
            report: Drift report dict from DriftDetector.run()

        Returns:
            True if published successfully
        """
        endpoint = report.get("endpoint_name", "unknown")
        psi_scores = report.get("psi_scores", {})
        timestamp = datetime.utcnow()

        metric_data = []

        # PSI per feature
        for feature, psi in psi_scores.items():
            metric_data.append({
                "MetricName": "PSIScore",
                "Dimensions": [
                    {"Name": "EndpointName", "Value": endpoint},
                    {"Name": "Feature", "Value": feature},
                    {"Name": "Environment", "Value": self.environment},
                ],
                "Timestamp": timestamp,
                "Value": float(psi),
                "Unit": "None",
            })

        # Overall drift verdict as a numeric flag
        verdict_score = {
            "stable": 0,
            "drift_detected_retrain": 1,
            "critical_drift_retrain_and_alert": 2,
            "skipped": -1,
        }.get(report.get("verdict", "stable"), 0)

        metric_data.append({
            "MetricName": "DriftSeverity",
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint},
                {"Name": "Environment", "Value": self.environment},
            ],
            "Timestamp": timestamp,
            "Value": float(verdict_score),
            "Unit": "None",
        })

        # Number of drifted features
        metric_data.append({
            "MetricName": "DriftedFeatureCount",
            "Dimensions": [
                {"Name": "EndpointName", "Value": endpoint},
                {"Name": "Environment", "Value": self.environment},
            ],
            "Timestamp": timestamp,
            "Value": float(len(report.get("drifted_features", []))),
            "Unit": "Count",
        })

        try:
            # CloudWatch accepts max 20 metrics per call
            for i in range(0, len(metric_data), 20):
                self.cw.put_metric_data(
                    Namespace=CW_NAMESPACE,
                    MetricData=metric_data[i:i + 20],
                )
            logger.info(f"Published {len(metric_data)} metrics to CloudWatch for {endpoint}")
            return True
        except Exception as e:
            logger.error(f"CloudWatch publish failed: {e}")
            return False

    def publish_deployment_event(
        self,
        endpoint_name: str,
        validation_f1: float,
        deployed: bool,
    ) -> bool:
        """Publish model deployment outcome to CloudWatch."""
        try:
            self.cw.put_metric_data(
                Namespace=CW_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": "DeploymentValidationF1",
                        "Dimensions": [
                            {"Name": "EndpointName", "Value": endpoint_name},
                            {"Name": "Environment", "Value": self.environment},
                        ],
                        "Timestamp": datetime.utcnow(),
                        "Value": validation_f1,
                        "Unit": "None",
                    },
                    {
                        "MetricName": "DeploymentSucceeded",
                        "Dimensions": [
                            {"Name": "EndpointName", "Value": endpoint_name},
                            {"Name": "Environment", "Value": self.environment},
                        ],
                        "Timestamp": datetime.utcnow(),
                        "Value": 1.0 if deployed else 0.0,
                        "Unit": "None",
                    },
                ],
            )
            return True
        except Exception as e:
            logger.error(f"CloudWatch deployment metric publish failed: {e}")
            return False
