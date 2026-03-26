"""
Agent State Store
=================
Persists agent run history and state to DynamoDB.
Enables the agent to reason about its own history across invocations
(e.g., "has this endpoint been retrained in the last 24 hours?").
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("StateStore")

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "clinical-ai-ops-agent-state")


class AgentStateStore:
    """DynamoDB-backed persistence for agent run history."""

    def __init__(self):
        self.dynamodb = boto3.resource("dynamodb")
        self._table = None

    @property
    def table(self):
        if self._table is None:
            try:
                self._table = self.dynamodb.Table(TABLE_NAME)
                self._table.load()  # Validates table exists
            except ClientError:
                logger.warning(
                    f"DynamoDB table '{TABLE_NAME}' not found. "
                    "State persistence disabled (local mode)."
                )
                self._table = None
        return self._table

    def save_run(self, run_id: str, endpoint_name: str, state: dict) -> bool:
        """
        Persist a completed agent run to DynamoDB.

        Args:
            run_id: Unique run identifier
            endpoint_name: The endpoint that was monitored
            state: Final agent state dict

        Returns:
            True if saved, False if DynamoDB is unavailable (local mode)
        """
        if self.table is None:
            logger.info(f"[LocalMode] Would save run {run_id} for {endpoint_name}")
            return False

        item = {
            "pk": f"ENDPOINT#{endpoint_name}",
            "sk": f"RUN#{run_id}",
            "run_id": run_id,
            "endpoint_name": endpoint_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ttl": int((datetime.utcnow() + timedelta(days=90)).timestamp()),
            "drift_verdict": state.get("drift_report", {}).get("verdict", "unknown"),
            "retrain_status": state.get("retrain_status", "unknown"),
            "deploy_status": state.get("deploy_status", "unknown"),
        }

        try:
            self.table.put_item(Item=item)
            logger.info(f"Saved run {run_id} to DynamoDB.")
            return True
        except Exception as e:
            logger.error(f"DynamoDB save failed: {e}")
            return False

    def get_last_retrain(self, endpoint_name: str) -> Optional[dict]:
        """
        Get the most recent successful retrain for an endpoint.
        Used by the agent to avoid retraining too frequently.

        Args:
            endpoint_name: The endpoint to query

        Returns:
            Most recent retrain record, or None
        """
        if self.table is None:
            return None

        try:
            response = self.table.query(
                KeyConditionExpression=(
                    boto3.dynamodb.conditions.Key("pk").eq(f"ENDPOINT#{endpoint_name}")
                ),
                FilterExpression=boto3.dynamodb.conditions.Attr("retrain_status").eq("succeeded"),
                ScanIndexForward=False,
                Limit=1,
            )
            items = response.get("Items", [])
            return items[0] if items else None
        except Exception as e:
            logger.warning(f"Could not query last retrain: {e}")
            return None

    def hours_since_last_retrain(self, endpoint_name: str) -> float:
        """Return hours since last successful retrain, or infinity if never."""
        last = self.get_last_retrain(endpoint_name)
        if last is None:
            return float("inf")

        last_time = datetime.fromisoformat(last["timestamp"].rstrip("Z"))
        delta = datetime.utcnow() - last_time
        return delta.total_seconds() / 3600
