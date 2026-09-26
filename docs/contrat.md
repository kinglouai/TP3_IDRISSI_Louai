# Contrat des fichiers et instrumentation

Le contrat vérifie la reproductibilité du parcours imposé. Il ne sert pas à
optimiser le modèle : la régression logistique est le support des principes MLOps
(configuration, provenance, exécution et preuves).

Les chemins sont relatifs à la racine du kit. Les JSON sont des objets sauf
`threshold_analysis.json`, qui est une liste. Les nombres doivent être finis,
les taux entre 0 et 1, les effectifs entiers. Les clés sont sensibles à la casse.
`json.dumps(payload, indent=2, allow_nan=False)` permet d'éviter les NaN.

## Fonctions à compléter

| Fonction | Entrée | Retour et obligation |
|---|---|---|
| `validate(config)` | configuration | DataFrame lu depuis `data/raw/machines.csv`; colonnes et valeurs contrôlées |
| `prepare(frame, config)` | données brutes | `X_train, X_test, y_train, y_test`; retirer `machine_id` et `machine_failure` de X |
| `train(X_train, y_train, config)` | train uniquement | Pipeline ajusté : ColumnTransformer puis LogisticRegression |
| `evaluate(model, X_test, y_test, config)` | modèle et test | dictionnaire metrics, dictionnaire confusion, liste thresholds |

Validation minimale : colonnes du CSV présentes, aucune valeur manquante,
`machine_id` unique, `machine_type` dans A/B/C, cible contenant les deux classes
0/1, capteurs numériques finis. Pour ce TP, lisez les noms des colonnes dans
le CSV et définissez ce schéma explicitement dans le code. Les noms de colonnes
sont un contrat de données; seed, C, seuils et coûts restent dans le YAML.
Les validations métier avancées et les datasets alternatifs appartiennent au TP2.
Le split aléatoire stratifié repose ici sur une hypothèse pédagogique : les
observations sont supposées indépendantes et issues de la même distribution. En contexte
industriel, un découpage temporel ou par machine peut être requis pour éviter une
évaluation trop optimiste.

Découpez avant tout fit :

```python
X = frame.drop(columns=["machine_failure", "machine_id"])
y = frame["machine_failure"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["test_size"],
    random_state=config["seed"], stratify=y,
)
```

Reprenez le ColumnTransformer du notebook. Appliquez `pipe.fit(X_train, y_train)`
et jamais `fit` sur le test. Pour prouver ce point, affichez `len(X_train)` et
`pipe.named_steps["preprocess"].named_transformers_["num"].n_samples_seen_`.
Les deux valent 3 750 dans la configuration fournie.

## Sorties de l'évaluation

`reports/metrics.json` : `accuracy`, `precision`, `recall`, `f1`, calculés au
`decision_threshold` configuré (0.50 pour les quatre expériences obligatoires).
Utilisez `zero_division=0` pour precision, recall et F1, et expliquez les zéros.
La valeur 0.50 est un seuil de référence pédagogique pour comparer les expériences;
elle ne constitue pas une recommandation opérationnelle.

```python
positive = list(model.classes_).index(1)
probabilities = model.predict_proba(X_test)[:, positive]
predictions = (probabilities >= config["decision_threshold"]).astype(int)
```

`reports/confusion_matrix.json` : `TN`, `FP`, `FN`, `TP` entiers.
Utilisez `confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()`.

`reports/threshold_analysis.json` : six objets, dans l'ordre de
`analysis_thresholds`. Chaque objet contient `threshold`, `accuracy`, `precision`,
`recall`, `f1`, `TN`, `FP`, `FN`, `TP`, `cost`, `alert_rate`.

- `cost = business_costs.false_negative * FN + business_costs.false_positive * FP`;
- `alert_rate = (TP + FP) / nombre_de_lignes_test`;
- recalculer les prédictions et métriques pour chaque seuil;
- conserver le modèle et le split : aucun réentraînement dans cette boucle.

Les valeurs `business_costs` sont des poids relatifs de décision propres à
l'exercice; elles ne sont ni des euros, ni des coûts opérationnels validés.

`reports/class_distribution.json` : objets `train` et `test`, chacun contenant
`"0"`, `"1"` (effectifs), `total` et `failure_rate` (proportion de classe 1).
Présentez les proportions en pourcentages dans le rapport.

## Modèle et manifeste

`models/model.joblib` contient le **Pipeline complet** sauvegardé par joblib,
pas seulement le classifieur. Le rechargement doit accepter les features brutes.

`models/run_manifest.json` contient obligatoirement :

| Clé | Valeur |
|---|---|
| `run_id` | ID du run MLflow actif |
| `data_sha256` | SHA-256 des octets de data/raw/machines.csv |
| `config_sha256` | SHA-256 des octets de configs/train.yaml |
| `code_sha256` | SHA-256 des octets de maintenance_ml/modeling/train.py |
| `lock_sha256` | SHA-256 des octets de uv.lock |
| `model_sha256` | SHA-256 des octets de models/model.joblib |
| `git_commit` | sortie de git rev-parse HEAD |
| `git_dirty` | false : code et configuration committés avant le run |
| `python_version` | platform.python_version() |
| `seed`, `test_size`, `model`, `decision_threshold` | valeurs exactes de la configuration |
| `metrics` | copie des quatre métriques de reports/metrics.json |
| `artifact` | chaîne `models/model.joblib` |
| `fit_rows` | nombre de lignes effectivement vues à l'ajustement |

Un timestamp ISO peut être ajouté; il varie normalement entre deux runs.
Le hash du code cible ici le seul fichier de pipeline; le commit identifie le
reste du projet. Si vous répartissez le pipeline en plusieurs modules, ajoutez
leurs hashes dans un champ supplémentaire et expliquez cette extension.

## Exemple MLflow à intégrer dans main

Ce fragment illustre l'ordre des opérations. `metrics`, `confusion`, `thresholds`
et `class_distribution` sont à construire; complétez tous les champs du manifeste.
Les chemins `ROOT` rendent les écritures indépendantes du dossier courant.

```python
import hashlib
import json
import platform
import subprocess
import joblib
import mlflow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


# À faire avant l'entraînement final : refuser un code/config non committé.
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
dirty = bool(subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
if dirty:
    raise RuntimeError("Committez le code et la configuration avant le run")

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
with mlflow.start_run() as run:
    # Exécuter prepare/train/evaluate ici ou avant, sans changer les entrées.
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/model.joblib")
    write_json(ROOT / "reports/metrics.json", metrics)
    write_json(ROOT / "reports/confusion_matrix.json", confusion)
    write_json(ROOT / "reports/threshold_analysis.json", thresholds)
    write_json(ROOT / "reports/class_distribution.json", class_distribution)
    mlflow.log_params({
        "seed": config["seed"], "test_size": config["test_size"],
        "model_type": config["model"]["type"], "C": config["model"]["C"],
        "max_iter": config["model"]["max_iter"],
        "decision_threshold": config["decision_threshold"],
    })
    mlflow.log_metrics(metrics)
    # Construire manifest selon le tableau, avec run.info.run_id.
    write_json(ROOT / "models/run_manifest.json", manifest)
    for relative in (
        "models/model.joblib", "models/run_manifest.json",
        "reports/metrics.json", "reports/confusion_matrix.json",
        "reports/threshold_analysis.json", "reports/class_distribution.json",
        "configs/train.yaml",
    ):
        mlflow.log_artifact(str(ROOT / relative), artifact_path="evidence")
```

Le modèle est journalisé comme fichier joblib dans `evidence/`; Model Registry
et déploiement ne sont pas demandés. Quitter le contexte termine le run.
Ensuite `make snapshot LABEL=A` sauvegarde les fichiers et les métadonnées du
serveur. Le snapshot nécessite un serveur accessible; sa lecture ultérieure
et le contrôle final fonctionnent hors ligne.

## Ce que les contrôles prouvent

`make validate-submission` recharge les modèles locaux, recalcule les métriques,
contrôle les hashes et les snapshots, compare A/B et détecte un rapport encore
marqué « À compléter ». Il ne certifie ni la qualité des explications, ni
l'authenticité des captures, ni l'absence de fuite : le code et la preuve du fit
restent à lire. N'utilisez joblib.load que sur vos propres modèles de confiance.
Aucun score minimal n'est imposé : un recall nul expliqué correctement est recevable.
