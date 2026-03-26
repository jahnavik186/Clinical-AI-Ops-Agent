"""
simulate_drift.py
=================
Run the complete Clinical AI Ops Agent loop locally — no AWS account needed.
Simulates drift injection, retraining, and deployment with realistic output.

Usage:
    python demo/simulate_drift.py
    python demo/simulate_drift.py --scenario no_drift
    python demo/simulate_drift.py --scenario critical_drift
"""

import sys
import time
import json
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Demo")

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header():
    print(f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════════╗
║         🏥 Clinical AI Ops Agent — Local Demo                ║
║         Agentic ML Monitoring for Healthcare                 ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def print_section(title: str):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}▶ {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


def print_psi_table(psi_scores: dict, threshold: float = 0.20):
    print(f"\n{'Feature':<22} {'PSI Score':>10}  {'Status':>12}")
    print(f"{'─'*22} {'─'*10}  {'─'*12}")
    for feature, psi in sorted(psi_scores.items(), key=lambda x: -x[1]):
        if psi > threshold:
            status = f"{RED}⚠ DRIFT{RESET}"
            psi_str = f"{RED}{psi:.4f}{RESET}"
        elif psi > threshold * 0.5:
            status = f"{YELLOW}MONITOR{RESET}"
            psi_str = f"{YELLOW}{psi:.4f}{RESET}"
        else:
            status = f"{GREEN}OK{RESET}"
            psi_str = f"{GREEN}{psi:.4f}{RESET}"
        print(f"  {feature:<20} {psi_str:>10}   {status:>12}")


def run_scenario(scenario: str):
    """Run a complete agent cycle for the given scenario."""

    import numpy as np

    ENDPOINT = "clinical-risk-model-v2"

    print_header()
    print(f"Scenario   : {BOLD}{scenario}{RESET}")
    print(f"Endpoint   : {ENDPOINT}")
    print(f"Timestamp  : {datetime.utcnow().isoformat()}Z")
    print(f"Mode       : LOCAL SIMULATION (no AWS required)")

    # ── Step 1: Drift Detection ────────────────────────────────────────────────
    print_section("Step 1 — Drift Detection (PSI + KS Tests)")
    print(f"Loading baseline feature distribution from S3...")
    time.sleep(0.5)
    print(f"Loading recent predictions (last 6 hours)...")
    time.sleep(0.5)
    print(f"Running PSI and Kolmogorov-Smirnov tests...")
    time.sleep(1)

    if scenario == "no_drift":
        psi_scores = {
            "age": 0.031,
            "bmi": 0.018,
            "lab_glucose": 0.042,
            "lab_creatinine": 0.029,
            "vitals_spo2": 0.011,
            "vitals_heart_rate": 0.024,
        }
        drifted_features = []
        verdict = "stable"

    elif scenario == "critical_drift":
        psi_scores = {
            "age": 0.041,
            "bmi": 0.033,
            "lab_glucose": 0.312,
            "lab_creatinine": 0.281,
            "vitals_spo2": 0.019,
            "vitals_heart_rate": 0.058,
        }
        drifted_features = ["lab_glucose", "lab_creatinine"]
        verdict = "critical_drift_retrain_and_alert"

    else:  # moderate drift (default)
        psi_scores = {
            "age": 0.041,
            "bmi": 0.033,
            "lab_glucose": 0.312,
            "lab_creatinine": 0.088,
            "vitals_spo2": 0.019,
            "vitals_heart_rate": 0.058,
        }
        drifted_features = ["lab_glucose"]
        verdict = "drift_detected_retrain"

    print_psi_table(psi_scores)
    print(f"\nDrifted features : {drifted_features or 'None'}")
    print(f"Verdict          : {BOLD}{verdict}{RESET}")

    # ── Step 2: Agent Decision ─────────────────────────────────────────────────
    print_section("Step 2 — Agent Decision (LangGraph ReAct)")
    time.sleep(0.5)

    if verdict == "stable":
        print(f"\n{GREEN}✅ No drift detected. Model is stable.{RESET}")
        print("Agent decision: No action required. Logging health check to DynamoDB.")
        print_summary(verdict="stable", retrain=False, deployed=False)
        return

    if verdict == "critical_drift_retrain_and_alert":
        print(f"\n{RED}🚨 CRITICAL drift detected on: {drifted_features}{RESET}")
        print("Agent decision: Retrain immediately + Page on-call team via SNS.")
    else:
        print(f"\n{YELLOW}⚠️  Drift detected on: {drifted_features}{RESET}")
        print("Agent decision: Trigger retraining. No immediate page required.")

    # ── Step 3: Alert (if critical) ────────────────────────────────────────────
    if verdict == "critical_drift_retrain_and_alert":
        print_section("Step 3 — Alert: Paging On-Call Team")
        time.sleep(0.3)
        print(f"  📧 SNS  → clinical-ai-ops-alerts topic  [{YELLOW}SIMULATED{RESET}]")
        print(f"  💬 Slack → #clinical-ai-ops channel      [{YELLOW}SIMULATED{RESET}]")
        print(f"  Message : 'Critical drift on {drifted_features} for {ENDPOINT}'")
    else:
        print_section("Step 3 — Alert: Info Notification")
        time.sleep(0.3)
        print(f"  💬 Slack → #clinical-ai-ops channel      [{YELLOW}SIMULATED{RESET}]")
        print(f"  Message : 'Drift detected, retraining initiated'")

    # ── Step 4: Retraining ─────────────────────────────────────────────────────
    print_section("Step 4 — Retraining (SageMaker Pipeline)")
    print(f"Starting pipeline: clinical-risk-model-pipeline [{YELLOW}SIMULATED{RESET}]")
    print(f"Training data   : s3://clinical-ai-ops-data/training-data/{ENDPOINT}/latest/")
    print(f"Drift context   : {drifted_features} passed as pipeline parameters")
    print()

    steps = [
        ("DataPreprocessing",    12),
        ("FeatureEngineering",   8),
        ("ModelTraining",        45),
        ("ModelEvaluation",      6),
        ("ArtifactRegistration", 3),
    ]
    total = sum(s[1] for s in steps)
    elapsed = 0
    for step_name, duration in steps:
        elapsed += duration
        bar = "█" * int(elapsed / total * 30)
        pct = int(elapsed / total * 100)
        print(f"  {step_name:<25} {bar:<30} {pct}%")
        time.sleep(0.4)

    validation_f1 = 0.847
    print(f"\n{GREEN}✅ Pipeline succeeded.{RESET}")
    print(f"  Validation Macro F1 : {BOLD}{validation_f1:.4f}{RESET}")
    print(f"  Model artifact      : s3://clinical-ai-ops-data/models/{ENDPOINT}/...")
    print(f"  Elapsed (simulated) : 30 min 42 sec")

    # ── Step 5: Safety Gate + Deployment ──────────────────────────────────────
    print_section("Step 5 — Safety Gate + Blue/Green Deployment")
    threshold = 0.80
    print(f"Safety threshold : F1 >= {threshold}")
    print(f"New model F1     : {validation_f1:.4f}")

    if validation_f1 >= threshold:
        print(f"\n{GREEN}✅ Safety gate passed. Proceeding with deployment.{RESET}")
        time.sleep(0.3)
        for pct in [0, 10, 50, 100]:
            bar_g = "█" * int(pct / 10)
            bar_b = "█" * int((100 - pct) / 10)
            print(f"  Blue  [{bar_b:<10}] {100-pct:>3}%   Green [{bar_g:<10}] {pct:>3}%")
            time.sleep(0.5)
        print(f"\n{GREEN}✅ Deployment complete. Production traffic: 100% → new model.{RESET}")
        print(f"  Alert sent: 'New model deployed to {ENDPOINT}'")
        print_summary(verdict=verdict, retrain=True, deployed=True, f1=validation_f1)
    else:
        print(f"\n{RED}❌ Safety gate FAILED (F1={validation_f1:.4f} < {threshold}).{RESET}")
        print("Production endpoint unchanged. GitHub Issue filed automatically.")
        print_summary(verdict=verdict, retrain=True, deployed=False, f1=validation_f1)


def print_summary(verdict: str, retrain: bool, deployed: bool, f1: float = None):
    print(f"""
{BOLD}{'═'*60}
  AGENT RUN SUMMARY
{'═'*60}{RESET}
  Drift verdict   : {verdict}
  Retrain         : {"✅ Yes" if retrain else "⏭  No (stable)"}
  Deployed        : {"✅ Yes" if deployed else "❌ No"}
  Validation F1   : {f"{f1:.4f}" if f1 else "N/A"}
  State saved     : DynamoDB (clinical-ai-ops-agent-state)
{BOLD}{'═'*60}{RESET}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical AI Ops Agent — Local Demo")
    parser.add_argument(
        "--scenario",
        choices=["moderate_drift", "critical_drift", "no_drift"],
        default="moderate_drift",
        help="Drift scenario to simulate (default: moderate_drift)",
    )
    args = parser.parse_args()
    run_scenario(args.scenario)
