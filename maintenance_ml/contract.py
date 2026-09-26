"""Contrat partagé par les tests et le contrôle final; aucune connexion réseau."""
from pathlib import Path
import hashlib
import json
import math
import re
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    c = yaml.safe_load(Path(path).read_text())
    assert isinstance(c, dict), "configuration YAML objet attendue"
    assert type(c["seed"]) is int, "seed entier"
    assert 0 < c["test_size"] < 1, "test_size dans ]0,1["
    assert 0 < c["decision_threshold"] < 1, "decision_threshold dans ]0,1["
    assert c["model"]["type"] == "logistic_regression"
    assert c["model"]["C"] > 0 and type(c["model"]["max_iter"]) is int and c["model"]["max_iter"] > 0
    assert c["analysis_thresholds"] == [0.01, 0.05, 0.10, 0.30, 0.50, 0.70]
    assert all(isinstance(c["business_costs"][k], (int, float)) and math.isfinite(c["business_costs"][k]) and c["business_costs"][k] > 0 for k in ("false_negative", "false_positive"))
    assert c["mlflow"]["tracking_uri"] and c["mlflow"]["experiment_name"]
    return c


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def matrix(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(zip(("TN", "FP", "FN", "TP"), map(int, (tn, fp, fn, tp))))


def assert_close(actual, expected):
    for key, value in expected.items():
        assert key in actual, f"champ absent: {key}"
        assert type(actual[key]) in (int, float) and math.isfinite(actual[key]), f"valeur invalide: {key}"
        assert math.isclose(actual[key], value, rel_tol=1e-8, abs_tol=1e-10), f"valeur incohérente: {key}"


def check_run(root, dataset):
    """Recalculer les preuves d'un run local ou d'un snapshot. Modèle local de confiance seulement."""
    root = Path(root)
    c = read_config(root / "configs/train.yaml")
    frame = pd.read_csv(dataset)
    X = frame.drop(columns=["machine_failure", "machine_id"])
    _, X_test, y_train, y_test = train_test_split(X, frame.machine_failure,
        test_size=c["test_size"], random_state=c["seed"], stratify=frame.machine_failure)
    model = joblib.load(root / "models/model.joblib")
    assert isinstance(model, Pipeline), "modèle = Pipeline sklearn attendu"
    assert list(model.feature_names_in_) == list(X.columns), "features incorrectes"
    estimator = model.steps[-1][1]
    assert estimator.C == c["model"]["C"] and estimator.max_iter == c["model"]["max_iter"]
    prob = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = prob >= c["decision_threshold"]
    def load(name):
        return json.loads((root / name).read_text())
    metrics = load("reports/metrics.json")
    assert_close(metrics, scores(y_test, pred))
    assert load("reports/confusion_matrix.json") == matrix(y_test, pred)
    rows = load("reports/threshold_analysis.json")
    assert isinstance(rows, list) and len(rows) == len(c["analysis_thresholds"])
    for threshold, row in zip(c["analysis_thresholds"], rows):
        predicted = prob >= threshold
        cm = matrix(y_test, predicted)
        expected = {"threshold": threshold, **scores(y_test, predicted), **cm,
            "cost": c["business_costs"]["false_negative"] * cm["FN"] + c["business_costs"]["false_positive"] * cm["FP"],
            "alert_rate": float(predicted.mean())}
        assert_close(row, expected)
    counts = load("reports/class_distribution.json")
    for name, y in (("train", y_train), ("test", y_test)):
        assert_close(counts[name], {"0": int((y == 0).sum()), "1": int((y == 1).sum()), "total": len(y), "failure_rate": float(y.mean())})
    m = load("models/run_manifest.json")
    assert re.fullmatch(r"[0-9a-f]{32}", m["run_id"]), "run_id invalide"
    assert re.fullmatch(r"[0-9a-f]{40,64}", m["git_commit"]), "commit Git absent"
    assert m["git_dirty"] is False, "code/configuration non committés"
    assert m["data_sha256"] == sha256(dataset), "hash des données incorrect"
    assert m["config_sha256"] == sha256(root / "configs/train.yaml")
    assert m["lock_sha256"] == sha256(root / "uv.lock")
    assert m["code_sha256"] == sha256(root / "maintenance_ml/modeling/train.py")
    assert m["model_sha256"] == sha256(root / "models/model.joblib")
    assert m["artifact"] == "models/model.joblib"
    assert m["seed"] == c["seed"] and m["test_size"] == c["test_size"]
    assert m["model"] == c["model"] and m["decision_threshold"] == c["decision_threshold"]
    assert_close(m["metrics"], metrics)
    assert isinstance(m["python_version"], str) and m["python_version"].startswith("3.13.")
    assert m["fit_rows"] == len(y_train), "fit_rows doit décrire le train"
    return m, prob


def validate_submission(root):
    root = Path(root)
    errors = []
    results = {}
    for label, folder in [("final", root)] + [(k, root / "reports/runs" / k) for k in ("C01", "A", "B", "C10")]:
        try:
            results[label] = check_run(folder, root / "data/raw/machines.csv")
            if label != "final":
                exported = json.loads((folder / "mlflow_run.json").read_text())
                m = results[label][0]
                assert exported["info"]["run_id"] == m["run_id"]
                assert exported["info"]["status"] == "FINISHED"
                assert_close(exported["data"]["metrics"], m["metrics"])
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    if len(results) == 5:
        try:
            a, b = results["A"], results["B"]
            for key in ("data_sha256", "config_sha256", "lock_sha256", "code_sha256", "git_commit", "seed", "test_size", "model", "metrics"):
                assert a[0][key] == b[0][key], f"A/B différents: {key}"
            assert np.allclose(a[1], b[1], rtol=0, atol=1e-12), "prédictions A/B différentes"
            assert len({results[k][0]["run_id"] for k in ("C01", "A", "B", "C10")}) == 4
            for label, C in (("C01", .1), ("A", 1.), ("B", 1.), ("C10", 10.)):
                assert results[label][0]["model"]["C"] == C, f"C incorrect pour {label}"
        except Exception as exc:
            errors.append(str(exc))
    try:
        report = (root / "reports/tp1_report.md").read_text()
        assert "À compléter" not in report and len(report.strip()) > 100, "rapport à compléter"
        figures = list((root / "reports/figures").glob("*"))
        assert any(f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.stat().st_size > 0 for f in figures), "captures absentes"
    except Exception as exc:
        errors.append(str(exc))
    return errors
