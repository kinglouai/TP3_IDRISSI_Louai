import mlflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import argparse
import yaml
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd 
import joblib
import json
import subprocess
import hashlib


ROOT = Path(__file__).resolve().parents[2]


def validate(config):
    df = pd.read_csv(ROOT / "data/raw/machines.csv")
    
    expected_columns = [
        "machine_id", "machine_type", "air_temperature_k", 
        "process_temperature_k", "rotational_speed_rpm", "torque_nm", 
        "tool_wear_min", "machine_failure"
    ]
    if list(df.columns) != expected_columns:
        raise ValueError("Colonnes invalides")
        
    if df.isnull().any().any():
        raise ValueError("Valeurs manquantes détectées")
        
    if not df["machine_id"].is_unique:
        raise ValueError("machine_id non unique")
        
    if not df["machine_type"].isin(["A", "B", "C"]).all():
        raise ValueError("machine_type invalide")
        
    if set(df["machine_failure"].unique()) != {0, 1}:
        raise ValueError("Classes cibles invalides")
        
    num_cols = ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    if not np.isfinite(df[num_cols].values).all():
        raise ValueError("Valeurs non finies")
        
    return df


def prepare(frame, config):
    X = frame.drop(columns=["machine_failure", "machine_id"])
    y = frame["machine_failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config["test_size"], 
        random_state=config["seed"], 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train(X_train, y_train, config):
    cat = ['machine_type']
    num = [c for c in X_train.columns if c not in cat]
    
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', StandardScaler(), num),
    ])
    
    model = LogisticRegression(
        C=config["model"]["C"], 
        max_iter=config["model"]["max_iter"]
    )
    
    pipe = Pipeline([('preprocess', prep), ('model', model)])
    pipe.fit(X_train, y_train)
    print("\n" + "="*30)
    print("Preuve de lecture YAML :")
    print(f"C utilisé = {config['model']['C']}")
    print("\nPreuve de découpage :")
    print(f"len(X_train) = {len(X_train)}")
    print(f"Lignes vues par StandardScaler = {pipe.named_steps['preprocess'].named_transformers_['num'].n_samples_seen_}")
    print("="*30 + "\n")
    return pipe


def evaluate(model, X_test, y_test, config):
    positive_class = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_class]
    
    threshold = config["decision_threshold"]
    predictions = (probabilities >= threshold).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0))
    }
    
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    confusion = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    
    thresholds_list = []
    costs = config["business_costs"]
    n_test = len(y_test)
    
    for t in config["analysis_thresholds"]:
        preds_t = (probabilities >= t).astype(int)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test, preds_t, labels=[0, 1]).ravel()
        t_cost = costs["false_negative"] * t_fn + costs["false_positive"] * t_fp
        t_alert_rate = (t_tp + t_fp) / n_test
        
        thresholds_list.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(y_test, preds_t)),
            "precision": float(precision_score(y_test, preds_t, zero_division=0)),
            "recall": float(recall_score(y_test, preds_t, zero_division=0)),
            "f1": float(f1_score(y_test, preds_t, zero_division=0)),
            "TN": int(t_tn), "FP": int(t_fp), "FN": int(t_fn), "TP": int(t_tp),
            "cost": float(t_cost), "alert_rate": float(t_alert_rate)
        })
        
    return metrics, confusion, thresholds_list



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
    frame = validate(config)
    print(f"DATA OK : {len(frame)} lignes")
    if args.check_data:
        return
    X_train, X_test, y_train, y_test = prepare(frame, config)
    model = train(X_train, y_train, config)
    metrics, confusion, thresholds = evaluate(model, X_test, y_test, config)
    
    class_distribution = {
        "train": {
            "0": int((y_train == 0).sum()), "1": int((y_train == 1).sum()),
            "total": len(y_train), "failure_rate": float((y_train == 1).mean())
        },
        "test": {
            "0": int((y_test == 0).sum()), "1": int((y_test == 1).sum()),
            "total": len(y_test), "failure_rate": float((y_test == 1).mean())
        }
    }
    
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/model.joblib")
    write_json(ROOT / "reports/metrics.json", metrics)
    write_json(ROOT / "reports/confusion_matrix.json", confusion)
    write_json(ROOT / "reports/threshold_analysis.json", thresholds)
    write_json(ROOT / "reports/class_distribution.json", class_distribution)

    import platform
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "run_id": "placeholder_until_mlflow",
        "data_sha256": sha(ROOT / "data/raw/machines.csv"),
        "config_sha256": sha(ROOT / "configs/train.yaml"),
        "code_sha256": sha(ROOT / "maintenance_ml/modeling/train.py"),
        "lock_sha256": sha(ROOT / "uv.lock"),
        "model_sha256": sha(ROOT / "models/model.joblib"),
        "git_commit": commit,
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "python_version": platform.python_version(),
        "seed": config["seed"], "test_size": config["test_size"],
        "model": config["model"], "decision_threshold": config["decision_threshold"],
        "metrics": metrics,
        "artifact": "models/model.joblib",
        "fit_rows": len(X_train)
    }

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run() as run:
        manifest["run_id"] = run.info.run_id
        write_json(ROOT / "models/run_manifest.json", manifest)
        
        mlflow.log_params({
            "seed": config["seed"], "test_size": config["test_size"],
            "model_type": config["model"]["type"], "C": config["model"]["C"],
            "max_iter": config["model"]["max_iter"], "decision_threshold": config["decision_threshold"]
        })
        mlflow.log_metrics(metrics)
        for relative in [
            "models/model.joblib", "models/run_manifest.json",
            "reports/metrics.json", "reports/confusion_matrix.json",
            "reports/threshold_analysis.json", "reports/class_distribution.json",
            "configs/train.yaml"
        ]:
            mlflow.log_artifact(str(ROOT / relative), artifact_path="evidence")

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
