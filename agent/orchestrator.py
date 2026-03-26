"""
Clinical AI Ops Agent — Orchestrator (Python 3.14 compatible)
==============================================================
Custom ReAct (Reason + Act) agent loop using the Anthropic SDK directly.

LangGraph/LangChain are NOT yet compatible with Python 3.14 as of March 2026.
This implementation replicates the same agentic behavior using:
  - anthropic SDK (pure Python, no Rust, works on 3.14)
  - Custom tool dispatch loop
  - Same 4-tool interface: drift check → retrain → deploy → alert

Author: Jahnavi Kachhia
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import anthropic

from agent.tools.drift_detector import DriftDetector
from agent.tools.retraining_trigger import RetrainingTrigger
from agent.tools.deployment_manager import DeploymentManager
from agent.tools.alert_publisher import AlertPublisher
from agent.memory.state_store import AgentStateStore

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("Agent")

# ── Tool Definitions for Anthropic tool_use API ────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "check_model_drift",
        "description": (
            "Check if the deployed model's input feature distribution has drifted "
            "from the training baseline using PSI and KS statistical tests. "
            "Returns per-feature PSI scores and an overall verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint_name": {
                    "type": "string",
                    "description": "SageMaker endpoint name to monitor",
                },
                "lookback_hours": {
                    "type": "integer",
                    "description": "Hours of recent prediction logs to analyze",
                    "default": 6,
                },
            },
            "required": ["endpoint_name"],
        },
    },
    {
        "name": "trigger_retraining",
        "description": (
            "Trigger a SageMaker retraining pipeline for the given endpoint. "
            "Uses the most recent labeled data from S3. "
            "Returns the pipeline execution ARN, status, and new model artifact URI."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint_name": {
                    "type": "string",
                    "description": "The endpoint whose model needs retraining",
                },
                "drift_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of features that triggered the retrain",
                },
            },
            "required": ["endpoint_name", "drift_features"],
        },
    },
    {
        "name": "deploy_new_model",
        "description": (
            "Deploy a newly trained model using blue/green traffic shifting. "
            "Validates F1 score meets the safety threshold (>= 0.80) before deploying. "
            "Returns deployment status and traffic shift details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "endpoint_name": {
                    "type": "string",
                    "description": "Target SageMaker endpoint",
                },
                "model_artifact_s3": {
                    "type": "string",
                    "description": "S3 URI of the new model artifact",
                },
                "validation_f1": {
                    "type": "number",
                    "description": "F1 score on the held-out safety validation set",
                },
            },
            "required": ["endpoint_name", "model_artifact_s3", "validation_f1"],
        },
    },
    {
        "name": "send_alert",
        "description": (
            "Send an alert via Amazon SNS and Slack webhook. "
            "Use severity='critical' for PSI > 0.25, 'warning' for normal drift, "
            "'info' for successful deployments."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Alert message body",
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "critical"],
                    "description": "Alert severity level",
                },
                "endpoint_name": {
                    "type": "string",
                    "description": "The affected endpoint name",
                },
            },
            "required": ["message", "severity", "endpoint_name"],
        },
    },
]

SYSTEM_PROMPT = """You are a Clinical AI Operations Agent responsible for maintaining
the health of deployed machine learning models in a healthcare environment.

Your job each run:
1. Call check_model_drift for the given endpoint
2. If drift is detected (PSI > 0.20 on any feature), call trigger_retraining
3. Once retraining completes successfully, call deploy_new_model
4. Send appropriate alerts at each significant event

Drift thresholds:
- PSI < 0.10  → No action, log as stable
- PSI 0.10–0.20 → Warning only, no retrain
- PSI > 0.20  → Trigger retraining + warning alert
- PSI > 0.25  → Trigger retraining + CRITICAL alert (page on-call)

Safety gate: only deploy if validation_f1 >= 0.80. If the new model fails, send a
critical alert and do NOT deploy. Patient safety takes priority over model freshness.

Always complete the full loop. Do not stop after drift detection — follow through
to retraining and deployment unless blocked by a safety gate failure."""


# ── Tool Executor ──────────────────────────────────────────────────────────────

class ToolExecutor:
    """Dispatches tool calls from the LLM to the actual tool implementations."""

    def __init__(self):
        self._drift    = DriftDetector()
        self._retrain  = RetrainingTrigger()
        self._deploy   = DeploymentManager()
        self._alert    = AlertPublisher()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a named tool and return its result as a JSON string."""
        logger.info(f"Executing tool: {tool_name}({list(tool_input.keys())})")

        if tool_name == "check_model_drift":
            result = self._drift.run(
                endpoint_name=tool_input["endpoint_name"],
                lookback_hours=tool_input.get("lookback_hours", 6),
            )
        elif tool_name == "trigger_retraining":
            result = self._retrain.run(
                endpoint_name=tool_input["endpoint_name"],
                drift_features=tool_input.get("drift_features", []),
            )
        elif tool_name == "deploy_new_model":
            result = self._deploy.deploy(
                endpoint_name=tool_input["endpoint_name"],
                model_artifact_s3=tool_input["model_artifact_s3"],
                validation_f1=float(tool_input["validation_f1"]),
            )
        elif tool_name == "send_alert":
            result = self._alert.publish(
                message=tool_input["message"],
                severity=tool_input["severity"],
                endpoint_name=tool_input["endpoint_name"],
            )
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result)


# ── ReAct Agent Loop ───────────────────────────────────────────────────────────

class ClinicalOpsAgent:
    """
    ReAct agent using the Anthropic SDK directly.
    Compatible with Python 3.14 — zero LangGraph/LangChain dependencies.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        self.client   = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self.model    = model
        self.executor = ToolExecutor()
        self.store    = AgentStateStore()

    def run(self, endpoint_name: str, run_id: str | None = None) -> dict:
        """
        Run a complete agent cycle for the given endpoint.

        Args:
            endpoint_name: SageMaker endpoint name to monitor
            run_id: Optional trace ID (auto-generated if not provided)

        Returns:
            Summary dict with drift verdict, retrain status, deploy status
        """
        run_id = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        logger.info(f"Starting agent run {run_id} for: {endpoint_name}")

        # Initial user message
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Run a complete health check for clinical ML endpoint '{endpoint_name}'.\n"
                    f"Run ID: {run_id}\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}Z\n\n"
                    "Follow the full loop: check drift → retrain if needed → "
                    "validate → deploy → alert."
                ),
            }
        ]

        # ── ReAct loop ────────────────────────────────────────────────────────
        MAX_ITERATIONS = 10  # safety cap
        for iteration in range(MAX_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # If no tool calls → agent is done
            if response.stop_reason == "end_turn":
                logger.info(f"Agent completed after {iteration + 1} iterations.")
                break

            # Process all tool calls in this response
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str = self.executor.execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            if not tool_results:
                break

            # Feed tool results back to the agent
            messages.append({"role": "user", "content": tool_results})

        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text
                break

        summary = {
            "run_id": run_id,
            "endpoint_name": endpoint_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "iterations": iteration + 1,
            "final_summary": final_text,
            "status": "completed",
        }

        # Persist to DynamoDB
        self.store.save_run(run_id=run_id, endpoint_name=endpoint_name, state=summary)
        logger.info(f"Run {run_id} complete.")
        return summary


# ── Entry Points ───────────────────────────────────────────────────────────────

def run_agent(endpoint_name: str, run_id: str | None = None) -> dict:
    """Convenience wrapper — creates and runs the agent."""
    agent = ClinicalOpsAgent()
    return agent.run(endpoint_name, run_id)


def lambda_handler(event: dict, context) -> dict:
    """AWS Lambda entry point. Triggered by EventBridge schedule."""
    endpoints = event.get(
        "endpoints",
        os.environ.get("MONITORED_ENDPOINTS", "clinical-risk-model-v2").split(","),
    )

    results = {}
    for endpoint in endpoints:
        try:
            result = run_agent(endpoint.strip())
            results[endpoint] = {"status": "success", "run_id": result.get("run_id")}
        except Exception as e:
            logger.error(f"Agent failed for {endpoint}: {e}")
            results[endpoint] = {"status": "error", "error": str(e)}

    return {"statusCode": 200, "body": json.dumps(results)}


if __name__ == "__main__":
    run_agent("clinical-risk-model-v2")
