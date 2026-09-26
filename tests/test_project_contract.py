"""Tests du kit. Le résultat étudiant est contrôlé par validate-submission."""
from pathlib import Path
import json
import pandas as pd
import pytest
import yaml
from maintenance_ml.contract import read_config, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_configuration():
    frame = pd.read_csv(ROOT / "data/raw/machines.csv")
    assert len(frame) == 5_000 and frame.machine_failure.sum() == 200
    assert not frame.isna().any().any()
    config = ROOT / "configs/train.yaml"
    read_config(config if config.exists() else ROOT / "configs/train.yaml.example")


def test_commented_threshold_is_not_a_configuration(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# decision_threshold\n")
    with pytest.raises(AssertionError):
        read_config(path)


def test_empty_submission_is_rejected(tmp_path):
    for file in ("models/model.joblib", "reports/metrics.json", "reports/confusion_matrix.json",
                 "reports/threshold_analysis.json", "reports/tp1_report.md",
                 "maintenance_ml/modeling/train.py"):
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    assert validate_submission(tmp_path)


def test_invalid_manifest_and_empty_model_are_rejected(tmp_path):
    import shutil
    shutil.copytree(ROOT / "data", tmp_path / "data")
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs/train.yaml.example", tmp_path / "configs/train.yaml")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/model.joblib").write_bytes(b"")
    (tmp_path / "models/run_manifest.json").write_text(json.dumps({"data_sha256": "wrong", "seed": None, "metrics": None}))
    assert validate_submission(tmp_path)
