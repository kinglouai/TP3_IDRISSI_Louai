"""Conserver les preuves et préparer une remise reproductible."""
from pathlib import Path
import argparse
import json
import re
import shutil
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maintenance_ml.contract import check_run, read_config, validate_submission
import numpy as np
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
FILES = ["configs/train.yaml", "uv.lock", "maintenance_ml/modeling/train.py",
    "models/model.joblib", "models/run_manifest.json", "reports/metrics.json",
    "reports/confusion_matrix.json", "reports/threshold_analysis.json", "reports/class_distribution.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["snapshot", "compare", "package", "clean"])
    parser.add_argument("name", nargs="?", default="")
    args = parser.parse_args()
    if args.action == "snapshot":
        assert args.name in ("C01", "A", "B", "C10"), "LABEL=C01, A, B ou C10 requis"
        dest = ROOT / "reports/runs" / args.name
        assert not dest.exists(), f"{dest} existe : conservez-le ou déplacez-le avant de recommencer"
        manifest, _ = check_run(ROOT, ROOT / "data/raw/machines.csv")
        c = read_config(ROOT / "configs/train.yaml")
        client = MlflowClient(tracking_uri=c["mlflow"]["tracking_uri"])
        run = client.get_run(manifest["run_id"])
        assert run.info.status == "FINISHED", "run non terminé"
        for file in FILES:
            target = dest / file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / file, target)
        (dest / "mlflow_run.json").write_text(json.dumps(run.to_dictionary(), indent=2))
        print(f"Snapshot {args.name}: {dest}")
    elif args.action == "compare":
        a, pa = check_run(ROOT / "reports/runs/A", ROOT / "data/raw/machines.csv")
        b, pb = check_run(ROOT / "reports/runs/B", ROOT / "data/raw/machines.csv")
        for key in ("data_sha256", "config_sha256", "lock_sha256", "code_sha256", "git_commit", "seed", "test_size", "model", "metrics"):
            assert a[key] == b[key], f"DIFFÉRENCE: {key}"
        assert a["run_id"] != b["run_id"], "deux runs distincts requis"
        assert np.allclose(pa, pb, rtol=0, atol=1e-12), "prédictions différentes"
        print("A/B : mêmes entrées, paramètres, métriques et probabilités (tolérance 1e-12).")
        print("Hash binaire du modèle identique :", a["model_sha256"] == b["model_sha256"])
        print("run_id distincts : normal. Ne pas exiger des manifestes identiques octet par octet.")
    elif args.action == "package":
        assert re.fullmatch(r"[A-Za-z0-9_-]+", args.name), "NAME=Nom_Prenom (sans espace ni accent) requis"
        errors = validate_submission(ROOT)
        assert not errors, "\n".join(errors)
        dest = ROOT.parent / f"MLOps_TP1_{args.name}.zip"
        assert not dest.exists(), f"{dest} existe déjà; renommez l'ancienne remise"
        excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "mlartifacts"}
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
            for f in sorted(ROOT.rglob("*")):
                relative = f.relative_to(ROOT)
                if f.is_file() and not excluded.intersection(relative.parts) and not f.name.startswith("mlflow.db") and f.suffix not in (".pyc", ".zip"):
                    archive.write(f, relative)
        print(dest)
    else:
        for file in FILES:
            if file.startswith(("models/", "reports/")):
                (ROOT / file).unlink(missing_ok=True)
        print("Sorties courantes nettoyées; snapshots, rapport et captures conservés.")


if __name__ == "__main__":
    main()
