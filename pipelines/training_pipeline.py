"""
SageMaker Training Pipeline
============================
Defines a SageMaker Pipeline for automated clinical model retraining.
Steps: Preprocessing → Training → Evaluation → Registration
Triggered automatically by the agent when drift is detected.
"""

import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics

PIPELINE_NAME = os.environ.get("SAGEMAKER_PIPELINE_NAME", "clinical-risk-model-pipeline")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "clinical-ai-ops-data")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def build_pipeline() -> Pipeline:
    """
    Build and return the SageMaker Pipeline for clinical model retraining.

    Pipeline steps:
    1. DataPreprocessing — clean + split train/val/test
    2. ModelTraining    — train scikit-learn classifier
    3. ModelEvaluation  — compute Macro F1 + safety metrics
    4. ModelRegistration — register in SageMaker Model Registry if F1 passes
    """
    session = PipelineSession()

    # ── Pipeline Parameters ────────────────────────────────────────────────────
    endpoint_name = ParameterString(name="EndpointName", default_value="clinical-risk-model-v2")
    training_data_uri = ParameterString(
        name="TrainingDataS3Uri",
        default_value=f"s3://{DATA_BUCKET}/training-data/clinical-risk-model-v2/latest/",
    )
    validation_data_uri = ParameterString(
        name="ValidationDataS3Uri",
        default_value=f"s3://{DATA_BUCKET}/validation-data/clinical-risk-model-v2/",
    )
    model_output_uri = ParameterString(
        name="ModelOutputS3Uri",
        default_value=f"s3://{DATA_BUCKET}/models/clinical-risk-model-v2/",
    )
    drift_features = ParameterString(name="DriftFeatures", default_value="[]")
    f1_threshold = ParameterFloat(name="F1Threshold", default_value=0.80)

    sklearn_version = "1.2-1"
    instance_type = "ml.m5.xlarge"

    processor = SKLearnProcessor(
        framework_version=sklearn_version,
        instance_type=instance_type,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=session,
    )

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    preprocessing_step = ProcessingStep(
        name="DataPreprocessing",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=training_data_uri,
                destination="/opt/ml/processing/input/train",
            ),
            ProcessingInput(
                source=validation_data_uri,
                destination="/opt/ml/processing/input/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/output/validation",
            ),
        ],
        code="pipelines/scripts/preprocess.py",
        job_arguments=["--drift-features", drift_features],
    )

    # ── Step 2: Training ───────────────────────────────────────────────────────
    estimator = SKLearn(
        entry_point="pipelines/scripts/train.py",
        framework_version=sklearn_version,
        instance_type=instance_type,
        instance_count=1,
        role=ROLE_ARN,
        output_path=model_output_uri,
        sagemaker_session=session,
        hyperparameters={
            "n-estimators": 200,
            "max-depth": 8,
            "class-weight": "balanced",
        },
    )

    training_step = TrainingStep(
        name="ModelTraining",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig
                .Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    # ── Step 3: Evaluation ─────────────────────────────────────────────────────
    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig
                .Outputs["validation"].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        code="pipelines/scripts/evaluate.py",
        job_arguments=["--f1-threshold", str(f1_threshold)],
    )

    # ── Step 4: Model Registration ─────────────────────────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"{evaluation_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri}/evaluation.json",
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="ArtifactRegistration",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=f"clinical-risk-model-group",
        model_metrics=model_metrics,
        approval_status="Approved",
    )

    # ── Assemble Pipeline ──────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            endpoint_name,
            training_data_uri,
            validation_data_uri,
            model_output_uri,
            drift_features,
            f1_threshold,
        ],
        steps=[
            preprocessing_step,
            training_step,
            evaluation_step,
            register_step,
        ],
        sagemaker_session=session,
    )

    return pipeline


def deploy_pipeline():
    """Create or update the pipeline in SageMaker."""
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f"✅ Pipeline '{PIPELINE_NAME}' deployed successfully.")
    return pipeline


if __name__ == "__main__":
    deploy_pipeline()
