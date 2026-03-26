"""
Alert Publisher
===============
Publishes alerts via Amazon SNS (for email/PagerDuty) and Slack webhooks.
Severity levels map to different notification channels.
"""

import json
import logging
import os
from datetime import datetime

import boto3
import urllib.request
import urllib.error

logger = logging.getLogger("AlertPublisher")

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

SEVERITY_EMOJI = {
    "info": "ℹ️",
    "warning": "⚠️",
    "critical": "🚨",
}


class AlertPublisher:
    """Publishes alerts to SNS and Slack for clinical AI ops events."""

    def __init__(self):
        self.sns = boto3.client("sns")

    def publish(self, message: str, severity: str, endpoint_name: str) -> dict:
        """
        Publish an alert to configured channels.

        Args:
            message: Alert body text
            severity: 'info', 'warning', or 'critical'
            endpoint_name: The affected clinical ML endpoint

        Returns:
            dict with delivery status per channel
        """
        emoji = SEVERITY_EMOJI.get(severity, "📋")
        subject = f"[{severity.upper()}] Clinical AI Ops — {endpoint_name}"
        full_message = (
            f"{emoji} {subject}\n\n"
            f"{message}\n\n"
            f"Endpoint: {endpoint_name}\n"
            f"Time: {datetime.utcnow().isoformat()}Z\n"
            f"Severity: {severity.upper()}"
        )

        results = {
            "severity": severity,
            "endpoint_name": endpoint_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sns": self._publish_sns(full_message, subject),
            "slack": self._publish_slack(full_message, severity, endpoint_name),
        }

        logger.info(f"Alert published: severity={severity}, sns={results['sns']['status']}, slack={results['slack']['status']}")
        return results

    # ── SNS ────────────────────────────────────────────────────────────────────

    def _publish_sns(self, message: str, subject: str) -> dict:
        if not SNS_TOPIC_ARN:
            logger.info("SNS_TOPIC_ARN not set — skipping SNS (local mode).")
            return {"status": "skipped", "reason": "no_topic_arn"}

        try:
            response = self.sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject=subject[:100],  # SNS subject limit
            )
            return {"status": "sent", "message_id": response["MessageId"]}
        except Exception as e:
            logger.error(f"SNS publish failed: {e}")
            return {"status": "error", "error": str(e)}

    # ── Slack ──────────────────────────────────────────────────────────────────

    def _publish_slack(self, message: str, severity: str, endpoint_name: str) -> dict:
        if not SLACK_WEBHOOK_URL:
            logger.info("SLACK_WEBHOOK_URL not set — skipping Slack (local mode).")
            return {"status": "skipped", "reason": "no_webhook_url"}

        color_map = {"info": "#36a64f", "warning": "#ff9900", "critical": "#ff0000"}
        payload = {
            "attachments": [
                {
                    "color": color_map.get(severity, "#cccccc"),
                    "title": f"Clinical AI Ops Alert — {endpoint_name}",
                    "text": message,
                    "footer": "Clinical-AI-Ops-Agent",
                    "ts": int(datetime.utcnow().timestamp()),
                }
            ]
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                SLACK_WEBHOOK_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return {"status": "sent", "http_status": resp.status}
        except urllib.error.URLError as e:
            logger.error(f"Slack webhook failed: {e}")
            return {"status": "error", "error": str(e)}
