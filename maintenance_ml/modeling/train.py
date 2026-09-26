from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import argparse
import yaml

ROOT = Path(__file__).resolve().parents[2]


def validate(config):
    """Lire data/raw/machines.csv, vérifier colonnes, valeurs et classes; retourner df."""
    raise NotImplementedError("Étape 2 : validate")


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
    
    return pipe


def evaluate(model, X_test, y_test, config):
    """Retourner metrics, confusion et liste d'analyses de seuils (voir contrat)."""
    raise NotImplementedError("Étape 4 : evaluate")


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
    # TODO : sauvegarder les sorties, créer le run MLflow et le manifeste.
    # Ordre et exemple d'instrumentation : docs/contrat.md.
    raise NotImplementedError("Étape 5 : sauvegarde et MLflow")


if __name__ == "__main__":
    main()
