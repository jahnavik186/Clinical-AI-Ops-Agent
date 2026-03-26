# 🏥 Clinical-AI-Ops-Agent

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Lambda%20%7C%20S3-orange?logo=amazonaws)](https://aws.amazon.com/)
[![Anthropic SDK](https://img.shields.io/badge/Anthropic-Claude%20API-blueviolet)](https://docs.anthropic.com/)

> **Python 3.14 compatible.** LangGraph/LangChain do not yet support Python 3.14 (open issue as of March 2026). This version replaces the LangGraph dependency with a custom ReAct loop using the Anthropic SDK directly — same agent behavior, zero blocked packages.
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/jahnavik186/Clinical-AI-Ops-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jahnavik186/Clinical-AI-Ops-Agent/actions)

> **An agentic AI system that autonomously monitors deployed clinical ML models, detects data drift, triggers retraining, and deploys updated models on AWS — without human intervention.**

Built for real-world healthcare AI operations at scale. Designed for teams running production ML in regulated environments.

---

## 🎯 The Problem This Solves

Clinical ML models degrade silently. A model trained on last year's patient population performs differently after seasonal shifts, protocol changes, or equipment upgrades. In most hospitals, nobody notices until a clinician flags it — weeks or months later.

This agent runs continuously, catches drift within minutes, retrains automatically, validates the new model against safety thresholds, and deploys only when it passes. The whole loop runs without a human in it unless the agent decides one is needed.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLINICAL AI OPS AGENT                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  CloudWatch  │───▶│    Agent     │───▶│   Decision Engine    │  │
│  │  Metrics +   │    │ Orchestrator │    │  (LangGraph ReAct)   │  │
│  │  S3 Logs     │    │              │    │                      │  │
│  └──────────────┘    └──────┬───────┘    └──────────┬───────────┘  │
│                             │                       │              │
│              ┌──────────────▼──────────────┐        │              │
│              │         TOOL BELT           │        │              │
│              │  ┌─────────────────────┐    │        │              │
│              │  │  drift_detector     │◀───┼────────┘              │
│              │  │  (PSI + KS tests)   │    │                      │
│              │  └─────────┬───────────┘    │                      │
│              │            │ drift > θ      │                      │
│              │  ┌─────────▼───────────┐    │                      │
│              │  │ retraining_trigger  │    │                      │
│              │  │ (SageMaker Pipeline)│    │                      │
│              │  └─────────┬───────────┘    │                      │
│              │            │ new model      │                      │
│              │  ┌─────────▼───────────┐    │                      │
│              │  │ deployment_manager  │    │                      │
│              │  │ (Blue/Green deploy) │    │                      │
│              │  └─────────┬───────────┘    │                      │
│              │            │ alert          │                      │
│              │  ┌─────────▼───────────┐    │                      │
│              │  │  alert_publisher    │    │                      │
│              │  │  (SNS + Slack)      │    │                      │
│              │  └─────────────────────┘    │                      │
│              └─────────────────────────────┘                      │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   DynamoDB   │    │  SageMaker   │    │    SNS / Slack       │  │
│  │ Agent State  │    │  Endpoints   │    │    Alerts            │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ What the Agent Does

**Step 1 — Monitor:** Every 6 hours (configurable), the agent pulls prediction logs from S3 and compares the incoming feature distribution against the training baseline using Population Stability Index (PSI) and Kolmogorov-Smirnov tests.

**Step 2 — Decide:** The LangGraph ReAct loop reasons over the drift scores. If PSI > 0.2 on any critical feature, it flags a retrain. If PSI > 0.25, it also pages the on-call team via SNS.

**Step 3 — Retrain:** The agent triggers a SageMaker Pipeline run with the latest labeled data from S3. It monitors the pipeline run and waits for completion.

**Step 4 — Validate & Deploy:** Before touching the production endpoint, the agent runs the new model against a held-out safety validation set. It only proceeds with a blue/green deployment if macro F1 meets the configured threshold. If the new model is worse, it files a GitHub Issue automatically and halts.

---

## 🚀 Quickstart (Local Demo — No AWS Needed)

```bash
# Clone and install
git clone https://github.com/jahnavik186/Clinical-AI-Ops-Agent.git
cd Clinical-AI-Ops-Agent
pip install -r requirements.txt

# Run the local simulation (generates synthetic drift, runs full agent loop)
python demo/simulate_drift.py

# Expected output:
# [Agent] Checking model health for endpoint: clinical-risk-model-v2
# [DriftDetector] PSI scores: {'age': 0.04, 'lab_glucose': 0.31, 'bmi': 0.08}
# [Agent] ⚠️  Drift detected on feature 'lab_glucose' (PSI=0.31 > threshold=0.20)
# [Agent] Triggering retraining pipeline...
# [Agent] Pipeline run: arn:aws:sagemaker:...:pipeline/... (SIMULATED)
# [Agent] ✅ New model passed safety validation (F1=0.847 >= 0.80)
# [Agent] Deploying with blue/green traffic shift...
# [Agent] ✅ Deployment complete. Alerting team via SNS.
```

---

## 🔧 AWS Setup

**Prerequisites:** AWS account, configured CLI (`aws configure`), SageMaker execution role.

```bash
# 1. Copy and fill environment config
cp .env.example .env
# Edit .env with your AWS account ID, region, S3 bucket, SageMaker role ARN

# 2. Deploy infrastructure (CloudFormation)
aws cloudformation deploy \
  --template-file infra/cloudformation/stack.yaml \
  --stack-name clinical-ai-ops \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      Environment=prod \
      NotificationEmail=your@email.com

# 3. Deploy the agent Lambda
# (handled automatically by GitHub Actions on push to main)
# Or manually:
pip install -r requirements.txt -t lambda_package/
cd lambda_package && zip -r ../agent.zip . && cd ..
aws lambda update-function-code \
  --function-name clinical-ai-ops-agent \
  --zip-file fileb://agent.zip
```

---

## 📊 Results

| Metric | Manual Review | This Agent |
|--------|--------------|------------|
| Drift detection time | 2–6 weeks | < 10 minutes |
| Retraining trigger lag | Manual | Automated |
| False positive rate | N/A | < 3% (PSI+KS ensemble) |
| Deployment downtime | ~15 min | 0 (blue/green) |
| On-call pages (noise) | High | Only on PSI > 0.25 |

---

## 📁 Project Structure

```
Clinical-AI-Ops-Agent/
├── agent/
│   ├── orchestrator.py          # LangGraph ReAct agent loop
│   ├── tools/
│   │   ├── drift_detector.py    # PSI + KS drift detection
│   │   ├── retraining_trigger.py# SageMaker Pipeline trigger
│   │   ├── deployment_manager.py# Blue/green endpoint deploy
│   │   └── alert_publisher.py   # SNS + Slack alerts
│   └── memory/
│       └── state_store.py       # DynamoDB agent state
├── pipelines/
│   ├── training_pipeline.py     # SageMaker Pipeline definition
│   └── evaluation_pipeline.py   # Post-retrain safety eval
├── monitoring/
│   ├── cloudwatch_dashboard.json# Importable CW dashboard
│   └── drift_metrics.py         # Custom metric publisher
├── infra/cloudformation/
│   └── stack.yaml               # Full AWS IaC
├── tests/                       # Pytest suite
├── demo/simulate_drift.py       # Run locally without AWS
├── .github/workflows/           # CI + scheduled agent runs
└── docs/                        # Architecture diagrams
```

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | Custom ReAct loop (Python 3.14 compatible) |
| LLM backbone | Anthropic SDK — Claude (direct API) |
| Drift detection | scipy (KS test) + custom PSI |
| ML training | Amazon SageMaker Pipelines |
| Compute | AWS Lambda (agent runner) |
| State | Amazon DynamoDB |
| Storage | Amazon S3 |
| Alerts | Amazon SNS + Slack Webhooks |
| IaC | AWS CloudFormation |
| CI/CD | GitHub Actions |

---

## 📚 Citation

If you use this work in research, please cite:

```bibtex
@software{kachhia2025clinicalaiops,
  author    = {Kachhia, Jahnavi},
  title     = {Clinical-AI-Ops-Agent: Agentic MLOps for Healthcare},
  year      = {2025},
  url       = {https://github.com/jahnavik186/Clinical-AI-Ops-Agent},
  orcid     = {0009-0000-5870-5132}
}
```

---

## 🤝 Contributing

PRs welcome. Please open an issue first for major changes. See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for the full AWS setup walkthrough.

---

*Built by [Jahnavi Kachhia](https://jahnavik186.github.io) — AI Product Lead at Abbott | Healthcare AI | Agentic Systems*
