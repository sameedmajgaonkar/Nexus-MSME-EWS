"""MLflow experiment tracking + model registry wrappers (plan.md §13.2).

Backend: local file store (file:./mlruns). MLflow 3.x places the filesystem
backend in maintenance mode behind MLFLOW_ALLOW_FILE_STORE=true; we opt in
because the demo is single-node with no database server (no Docker daemon —
see BUILD_CONTEXT constraint adaptations).

Stages vs aliases: MLflow >= 2.9 DEPRECATES registry stage transitions
(``transition_model_version_stage``) in favour of registered-model ALIASES.
This module therefore implements plan.md's "Staging -> Production" promotion
with the aliases 'staging' / 'production'.
"""

import math
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = (ROOT / "mlruns").as_uri()  # file:./mlruns as an absolute URI
EXPERIMENT = "nexus-msme-ews"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def log_model_run(
    name: str,
    params: dict | None = None,
    metrics: dict | None = None,
    artifact_path: str | Path | None = None,
    tags: dict | None = None,
) -> str:
    """Log one training/backfill run (params, metrics, optional artifact); return run_id.

    Non-finite / null metric values are skipped (e.g. the deliberate nulls in
    the historical phaseN_metrics.json files).
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            clean = {}
            for key, value in metrics.items():
                if value is None:
                    continue
                value = float(value)
                if math.isfinite(value):
                    clean[key] = value
            if clean:
                mlflow.log_metrics(clean)
        if tags:
            mlflow.set_tags(tags)
        if artifact_path is not None and Path(artifact_path).exists():
            mlflow.log_artifact(str(artifact_path))
        return run.info.run_id


def register_and_stage(run_id: str, model_name: str, stage: str) -> int:
    """Register the run as a new version of ``model_name`` and point the alias at it.

    ``stage`` is 'production' or 'staging' (case-insensitive). Implemented with
    registered-model ALIASES, not the deprecated stage API (MLflow >= 2.9).
    Returns the new model version number.
    """
    client = _client()
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        pass  # already registered — just add a version
    run = client.get_run(run_id)
    version = client.create_model_version(
        model_name, source=run.info.artifact_uri, run_id=run_id
    )
    client.set_registered_model_alias(model_name, stage.lower(), version.version)
    return int(version.version)


def get_production_metrics(model_name: str) -> dict | None:
    """Metrics dict of the run behind the 'production' alias, or None if absent."""
    client = _client()
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        return None
    return dict(client.get_run(version.run_id).data.metrics)
