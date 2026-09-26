"""Vérifie les preuves locales; ne remplace pas la lecture du code et du rapport."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maintenance_ml.contract import validate_submission

if __name__ == "__main__":
    errors = validate_submission(Path(__file__).resolve().parents[1])
    print("VALIDATION TP1: REJECT" if errors else "VALIDATION TP1: ACCEPT — contrôles automatiques réussis; revue pédagogique requise")
    for error in errors:
        print("-", error)
    sys.exit(bool(errors))
