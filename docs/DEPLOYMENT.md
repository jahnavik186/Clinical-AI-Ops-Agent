# Deployment Guide — Clinical AI Ops Agent

Complete walkthrough for deploying to AWS from scratch.

---

## Prerequisites

- AWS account with admin or scoped IAM permissions
- AWS CLI v2 configured (`aws configure`)
- Python 3.11+
- Git

---

## Step 1 — Clone and Configure

```bash
git clone https://github.com/jahnavik186/Clinical-AI-Ops-Agent.git
cd Clinical-AI-Ops-Agent

# Install dependencies
pip install -r requirements.txt

# Copy and fill env config
cp .env.example .env
# Edit .env — fill in your AWS account ID, region, email
```

---

## Step 2 — Deploy AWS Infrastructure (CloudFormation)

This creates all required AWS resources: Lambda, DynamoDB, S3, SNS, EventBridge, and IAM roles.

```bash
aws cloudformation deploy \
  --template-file infra/cloudformation/stack.yaml \
  --stack-name clinical-ai-ops-prod \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      Environment=prod \
      NotificationEmail=your@email.com \
      MonitoredEndpoints=clinical-risk-model-v2 \
      AgentSchedule="rate(6 hours)"
```

Deployment takes ~3 minutes. Check progress in the AWS CloudFormation console.

After deployment, export the output values to your `.env`:

```bash
aws cloudformation describe-stacks \
  --stack-name clinical-ai-ops-prod \
  --query "Stacks[0].Outputs"
```

---

## Step 3 — Upload Baseline Feature Distributions to S3

The drift detector needs a baseline to compare against. Generate it from your training data:

```python
import pandas as pd
import boto3

# Load your training dataset
train_df = pd.read_csv("your_training_data.csv")

# Keep only feature columns (drop label, ID, etc.)
feature_cols = ["age", "bmi", "lab_glucose", "lab_creatinine", "vitals_spo2", "vitals_heart_rate"]
baseline_df = train_df[feature_cols]

# Upload as parquet to S3
baseline_df.to_parquet("/tmp/feature_baseline.parquet", index=False)

s3 = boto3.client("s3")
s3.upload_file(
    "/tmp/feature_baseline.parquet",
    "your-data-bucket",
    "baselines/clinical-risk-model-v2/feature_baseline.parquet"
)
print("Baseline uploaded.")
```

---

## Step 4 — Deploy the SageMaker Pipeline

```bash
export SAGEMAKER_ROLE_ARN="arn:aws:iam::YOUR_ACCOUNT:role/clinical-ai-ops-sagemaker-role-prod"
export DATA_BUCKET="clinical-ai-ops-data-YOUR_ACCOUNT-prod"

python pipelines/training_pipeline.py
# Output: ✅ Pipeline 'clinical-risk-model-pipeline-prod' deployed successfully.
```

---

## Step 5 — Package and Deploy the Lambda Agent

This is handled automatically by GitHub Actions on every push to `main`. For a manual deploy:

```bash
# Package dependencies + agent code
pip install -r requirements.txt -t lambda_package/
cp -r agent lambda_package/
cp -r pipelines lambda_package/
cp -r monitoring lambda_package/

cd lambda_package
zip -r ../agent.zip . -x "*.pyc" -x "__pycache__/*"
cd ..

# Deploy to Lambda
aws lambda update-function-code \
  --function-name clinical-ai-ops-agent-prod \
  --zip-file fileb://agent.zip

echo "✅ Lambda deployed"
```

---

## Step 6 — Configure GitHub Actions Secrets

For CI/CD to work, add these secrets to your GitHub repo (`Settings → Secrets → Actions`):

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | IAM user access key with Lambda + CloudFormation permissions |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |

---

## Step 7 — Verify End-to-End

Trigger a manual agent run and check the output:

```bash
aws lambda invoke \
  --function-name clinical-ai-ops-agent-prod \
  --payload '{"endpoints": ["clinical-risk-model-v2"], "triggered_by": "manual_test"}' \
  --cli-binary-format raw-in-base64-out \
  response.json

cat response.json
# Expected: {"statusCode": 200, "body": "{\"clinical-risk-model-v2\": {\"status\": \"success\", ...}}"}
```

Check DynamoDB for the run record:

```bash
aws dynamodb query \
  --table-name clinical-ai-ops-agent-state-prod \
  --key-condition-expression "pk = :pk" \
  --expression-attribute-values '{":pk": {"S": "ENDPOINT#clinical-risk-model-v2"}}' \
  --scan-index-forward false \
  --limit 1
```

---

## Monitoring & Observability

**CloudWatch Dashboard** — After deploying, import the dashboard:

```bash
aws cloudwatch put-dashboard \
  --dashboard-name ClinicalAIOps \
  --dashboard-body file://monitoring/cloudwatch_dashboard.json
```

**Key metrics to watch:**
- `ClinicalAIOps/DriftMonitoring` → `PSIScore` per feature per endpoint
- `ClinicalAIOps/DriftMonitoring` → `DriftSeverity` (0=stable, 1=drift, 2=critical)
- `AWS/Lambda` → `Errors` for the agent function

---

## Troubleshooting

**Agent Lambda times out:**
Increase `Timeout` in CloudFormation from 900s. Note SageMaker pipelines can take 30–60 min.

**PSI baseline not found:**
Check S3 path: `s3://{DATA_BUCKET}/baselines/{endpoint_name}/feature_baseline.parquet`. Run Step 3 above.

**SageMaker pipeline not found:**
Run `python pipelines/training_pipeline.py` to deploy it.

**Slack alerts not arriving:**
Verify `SLACK_WEBHOOK_URL` in Lambda environment variables. Test the webhook with curl:
```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test from Clinical AI Ops Agent"}' \
  YOUR_SLACK_WEBHOOK_URL
```
