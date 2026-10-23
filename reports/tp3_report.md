# Rapport de TP3 — Servir et promouvoir un modèle

**Étudiant :** IDRISSI Louai  
**Filière :** 3A Génie Logiciel — ENSIAS  
**Module :** MLOps et Industrialisation des modèles IA (GL-M326)  
**Année académique :** 2026-2027  

---

Checklist :

- [x] import réel du TP2 `ACCEPT` ou `make bootstrap-reference` documenté ;
- [x] `make doctor`, tests verts et échec contrôlé ;
- [x] rapport JUnit `reports/test_report.xml` ;
- [x] contrat API et réponses HTTP ;
- [x] image, conteneur et smoke test Docker ;
- [x] CI verte et échec contrôlé ;
- [x] enregistrement, sélection, promotion et rollback ;
- [x] `contracts/openapi.json` et `reports/release_manifest.json` ;
- [x] `make validate-submission` ;
- [x] archive créée avec `make package ARCHIVE=MLOps_TP3_IDRISSI_Louai.zip`.

## 0. Continuité avec le TP2

| Élément TP2 | Valeur observée | Preuve | Réutilisé dans TP3 ? |
|---|---|---|---|
| `reports/quality_gate_decision.json` | `decision: ACCEPT`, 5 checks passés (`minimum_accuracy`, `minimum_recall`, `minimum_f1`, `baseline_recall`, `baseline_f1`) | `reports/quality_gate_decision.json` | Oui, condition préalable stricte pour autoriser l'enregistrement |
| `models/model.joblib` | Pipeline scikit-learn (OneHotEncoder, StandardScaler, LogisticRegression C=1.0) | `models/model.joblib` | Oui, servi en mode local et sérialisé dans MLflow |
| `reports/metrics.json` | Accuracy: 0.9616, Precision: 0.7895, Recall: 0.6522, F1: 0.7143, Latency: 0.0201 ms | `reports/metrics.json` | Oui, injecté dans MLflow run et comparé dans le manifest |
| `reports/run_manifest.json` | `run_id: tp3-reference-run`, `data_sha256: 5598644bc8721601c7855bea95fba27b41e2edfb4cf57f497e93b1d49cf08cfa` | `reports/run_manifest.json` | Oui, vérifie l'intégrité modèle-données |
| `data_sha256` | `5598644bc8721601c7855bea95fba27b41e2edfb4cf57f497e93b1d49cf08cfa` | SHA-256 de `data/raw/machines.csv` | Oui, garantit l'absence de drift dataset |

Nous avons validé les deux parcours d'entrée :
1. Le parcours nominal `make import-tp2 TP2_DIR=/chemin/vers/TP2` : il vérifie la présence de tous les fichiers obligatoires, la cohérence cryptographique du SHA-256 des données, l'évaluation de la policy du quality gate (`decision: ACCEPT`) et la complétude des métriques (notamment `latency_ms`). En cas de non-conformité, l'espace TP3 reste intact (principe de pré-validation atomique sans effets de bord).
2. Le parcours reproductible de référence `make bootstrap-reference` : utilisé pour produire un checkpoint initial local déterministe (`seed=42`) satisfaisant rigoureusement le quality gate avec `decision: ACCEPT` (`F1=0.7143`, `recall=0.6522`), documenté dans `reports/tp2_import.json`.

Un modèle ayant un statut `REJECT` ne peut en aucun cas être enregistré dans le MLflow Model Registry : la fonction `validate_registration_input()` inspecte `reports/quality_gate_decision.json` et lève une exception `RuntimeError("Model rejected by TP2 quality gate; registration cancelled.")` avant toute interaction avec le backend MLflow. Cela garantit qu'aucun modèle défaillant ou non conforme aux exigences industrielles n'entre dans le cycle de vie de release.

## 1. Contrat d'inférence

| Endpoint/cas | Commande ou payload | Code HTTP | Réponse observée | Preuve |
|---|---|---:|---|---|
| `/health` | `curl -i http://127.0.0.1:8000/health` | 200 | `{"status":"ok","model_loaded":true,"model_name":"predictive-maintenance","alias":"champion","version":"1","source":"registry","load_error":null}` | `reports/figures/03_endpoints_responses.svg` |
| `/model-info` | `curl -i http://127.0.0.1:8000/model-info` | 200 | `{"model_name":"predictive-maintenance","alias":"champion","version":"1","run_id":"52346f26358540e18bb7756182effc9b","data_sha256":"5598644b...","quality_gate":"ACCEPT","source":"registry"}` | `reports/figures/03_endpoints_responses.svg` |
| `/predict` valide | `curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"air_temperature_k":300,"process_temperature_k":310,"rotational_speed_rpm":1500,"torque_nm":40,"tool_wear_min":100,"machine_type":"A"}'` | 200 | `{"prediction":0,"probability":0.000045,"model_name":"predictive-maintenance","alias":"champion","version":"1"}` | `reports/figures/03_endpoints_responses.svg` |
| `/v1/predict` et `X-Request-ID` | Idem sur `/v1/predict` | 200 | Body identique + Header `X-Request-ID: 886bd745eccd4a90957149dcf2aaf1e9` | `reports/figures/03_endpoints_responses.svg` |
| champ manquant | Payload omettant `"torque_nm"` | 422 | `{"error":{"code":"validation_error","message":"Request validation failed","request_id":"...","details":[{"type":"missing","loc":["body","torque_nm"],"msg":"Field required"}]}}` | `reports/figures/05_http_422_error.svg` |
| champ inconnu | Payload incluant `"unauthorized_field": "malicious"` | 422 | `{"error":{"code":"validation_error","message":"Request validation failed","request_id":"...","details":[{"type":"extra_forbidden","loc":["body","unauthorized_field"],"msg":"Extra inputs are not permitted"}]}}` | `reports/figures/05_http_422_error.svg` |
| valeur invalide | Payload avec `"air_temperature_k": -50.0` | 422 | `{"error":{"code":"validation_error","message":"Request validation failed","request_id":"...","details":[{"type":"greater_than_equal","loc":["body","air_temperature_k"],"msg":"Input should be greater than or equal to 250"}]}}` | `reports/figures/05_http_422_error.svg` |
| `/metrics` | `curl -s http://127.0.0.1:8000/metrics` | 200 | Exposition Prometheus des compteurs de requêtes, latences HTTP et statut du modèle | `reports/figures/03_endpoints_responses.svg` |

Pydantic protège le pipeline d'inférence en validant rigoureusement chaque entrée avant que les données n'atteignent le modèle scikit-learn. Grâce au paramètre `extra="forbid"`, toute clé non documentée est immédiatement rejetée avec un code HTTP 422. De plus, les types stricts, les contraintes numériques (bornes physiques de température, vitesses et couples) et les énumérations (`machine_type` restreint à `A`, `B`, `C`) empêchent les erreurs d'exécution ou les empoisonnements silencieux de calcul. Si le modèle ne peut être résolu ou chargé, `/health` renvoie HTTP 503, protégeant l'orchestrateur.

La configuration du service est centralisée dans `configs/service.yaml` (source: `registry`, model_name: `predictive-maintenance`, alias: `champion`, tracking_uri: `http://127.0.0.1:5000`). Les variables d'environnement telles que `MODEL_SOURCE=local` permettent des surcharges propres lors du smoke test sans modifier le code.

Le contrat OpenAPI formel est généré via `make openapi` dans `contracts/openapi.json` et validé par `tests/test_openapi_contract.py`. La compatibilité entre les routes legacy (`/predict`) et versionnées (`/v1/predict`) est assurée afin de permettre une migration progressive des consommateurs sans interruption de service.

Le endpoint `/metrics` est une interface purement technique (M2M) au format OpenMetrics destinée au scraping par Prometheus. Il ne doit pas être consommé par une interface utilisateur métier ou un tableau de bord applicatif ; la visualisation MLOps sera prise en charge par Grafana dans le TP4.

## 2. Docker et exécution

| Élément | Commande | Sortie observée | Preuve |
|---|---|---|---|
| Image | `make docker-build` | Image `maintenance-ml:tp3` construite avec multi-stage, `uv export`, dépendances verrouillées et utilisateur non-root `app` | `reports/figures/06_docker_image.svg` |
| Conteneur | `make docker-run` | Conteneur démarré en mode lecture seule (`--read-only`), avec `--tmpfs /tmp`, sans capabilities (`--cap-drop=ALL`) et `--security-opt no-new-privileges` | `reports/figures/07_docker_container.svg` |
| Utilisateur effectif | `docker exec <id> id` | `uid=10001(app) gid=10001(app)` (aucun privilège root) | `reports/figures/07_docker_container.svg` |
| Healthcheck | `docker inspect --format='{{.State.Health.Status}}' <id>` | `healthy` après vérification périodique de `curl -f http://127.0.0.1:8000/health` | `reports/figures/07_docker_container.svg` |
| Smoke test | `make docker-smoke` | Exécution de `scripts/smoke_test.py` : validation des 4 endpoints et nettoyage garanti du conteneur (`Docker smoke test: ACCEPT`) | `reports/figures/07_docker_container.svg` |

Ce que garantissent les différents composants :
- **L'image Docker :** garantit la reproductibilité de l'environnement d'exécution (binaire Python, paquets `mlflow-skinny`, `skops`, `fastapi`, dépendances système minimales) sans embarquer de secrets ni de données d'entraînement volumineuses.
- **Le conteneur :** isole l'exécution au runtime avec des contraintes de sécurité strictes (système de fichiers en lecture seule, absence de capacités noyau, tmpfs temporaire pour les sockets).
- **L'artefact modèle :** encapsule le graphe de calcul scikit-learn et les paramètres appris sérialisés.
- **Le Registry :** assure la gouvernance, le versionnement immuable, la traçabilité des métriques et l'association dynamique des alias (`champion`, `challenger`).

Les données brutes d'entraînement (`machines.csv`) ne sont pas intégrées dans l'image de production : seul le modèle sérialisé et son runtime léger sont embarqués, respectant le principe de séparation des données et du code de serving.

## 3. CI

| Exécution | Commit | Étape | Résultat | Preuve |
|---|---|---|---|---|
| Job vert | `c3a4f81` | Pipeline GitHub Actions complet (`uv sync --locked` -> `ruff` -> `pytest` -> `bootstrap` -> `docker build` -> `smoke test`) | `PASS` (toutes les étapes validées, tests verts, rapport JUnit conforme) | `reports/figures/08_ci_green.svg` |
| Échec contrôlé | `e9f2b14` | Injection d'une non-conformité de typage ou de linting (`ruff check`) | `FAIL` (le pipeline s'arrête immédiatement avant la construction Docker) | `reports/figures/09_ci_failure.svg` |

Le workflow `.github/workflows/mlops-ci.yml` exécute la séquence stricte suivante :
1. `uv sync --locked` : vérification déterministe de l'arbre des dépendances ;
2. `ruff check .` : analyse statique et conformité du code Python ;
3. `pytest --junitxml=reports/test_report.xml` : 23 tests unitaires et d'intégration validant le contrat API et les règles de promotion ;
4. `scripts/bootstrap_reference.py` : vérification de la capacité à instancier un checkpoint de serving local ;
5. `docker build` et exécution en conteneur sécurisé ;
6. `scripts/smoke_test.py` : vérification HTTP de bout en bout.

Une pull request dont la CI est verte ne déclenche jamais automatiquement la promotion d'un modèle : la CI valide uniquement la conformité logicielle, la constructibilité de l'image et l'absence de régression fonctionnelle. La promotion en production est une décision métier et de release explicite, soumise à une politique multicritère auditée et tracée dans le Registry.

## 4. Enregistrement Registry

| Champ | Valeur de `reports/registration.json` |
|---|---|
| Modèle enregistré | `predictive-maintenance` |
| Version | `2` (et version `1` pour le checkpoint initial) |
| Run ID | `29e124a23bdb4dd8a231e91c161417de` |
| Hash des données | `5598644bc8721601c7855bea95fba27b41e2edfb4cf57f497e93b1d49cf08cfa` |
| Gate | `ACCEPT` |
| Signature et exemple d'entrée | Signature inférée avec 6 features (`machine_type`, `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`) et 5 lignes d'exemple issues du test set |

L'enregistrement dans le Registry exige des preuves cryptographiques et qualitatives obligatoires :
1. Présence du fichier `reports/quality_gate_decision.json` portant formellement `decision: ACCEPT` ;
2. Concordance exacte du hash SHA-256 du dataset entre `run_manifest.json` et le fichier physique `data/raw/machines.csv` ;
3. Cohérence numérique stricte entre les métriques de `reports/metrics.json` et celles déclarées dans le manifest ;
4. Borne de validité des métriques (valeurs finies entre 0 et 1, latence positive).
Si l'une de ces conditions échoue, `validate_registration_input()` interrompt le script et refuse la création de la version MLflow.

## 5. Sélection du challenger

Le processus de sélection inspecte l'ensemble des versions enregistrées dans le modèle `predictive-maintenance` selon un algorithme en trois temps :
1. **Éligibilité :** une version est éligible si elle n'est pas déjà le champion actuel, si son quality gate est `ACCEPT`, si son SHA-256 de données est valide, si `f1 >= 0.64`, `recall >= 0.60` et `latency_ms <= 50 ms`.
2. **Classement :** les candidats éligibles sont ordonnés selon le tuple `(f1, recall, -latency_ms, version)`.
3. **Désignation :** la meilleure version reçoit dynamiquement l'alias `challenger`.

| Version | Gate | F1 | Recall | Latence | Éligible | Rang |
|---|---|---:|---:|---:|---|---:|
| Version 1 | ACCEPT | 0.7143 | 0.6522 | 0.0201 ms | Non (champion en titre) | - |
| Version 2 | ACCEPT | 0.7101 | 0.6522 | 0.0145 ms | Oui | 1 (Challenger sélectionné) |

## 6. Promotion

| Élément | Valeur ou preuve |
|---|---|
| Champion avant décision | `1` |
| Challenger | `2` |
| Décision | `KEEP_CHAMPION` |
| Gain F1 | `-0.0042` (inférieur au seuil requis de `+0.01`) |
| Gain recall | `0.0000` (conforme au seuil `>= -0.01`) |
| Alias après décision | `champion` pointe sur `1`, `challenger` pointe sur `2` |
| JSON | `reports/promotion_decision.json` |

Un candidat éligible n'est pas automatiquement promu au rang de champion. Ici, la version 2 satisfait tous les critères d'éligibilité (F1=0.7101 > 0.64, recall=0.6522 > 0.60, gate=ACCEPT). Toutefois, lors de la comparaison avec le champion en titre (version 1, F1=0.7143), la politique de release exige un gain minimal strict de performance (`minimum_f1_gain: 0.01`). Constatant un différentiel de F1 de `-0.0042`, le script de promotion applique la règle de prudence industrielle et génère une décision auditable `KEEP_CHAMPION`. L'alias `champion` reste immuablement sur la version 1.

## 7. Rollback

| Moment | Alias `champion` | Version | Preuve |
|---|---|---|---|
| Avant rollback | `champion` | `2` (affecté temporairement pour test) | `reports/figures/12_rollback.svg` |
| Après rollback | `champion` | `1` | `reports/rollback_decision.json` |

La commande exécutée pour le retour arrière est :
```bash
make rollback ROLLBACK_VERSION=1
```
Le rollback ne détruit aucune version et ne supprime aucun artefact du Registry : il se contente de réassigner atomiquement le pointeur d'alias `champion` vers la version cible certifiée. Le fichier `reports/rollback_decision.json` atteste formellement que `previous_version: "2"` et `restored_version: "1"` sont distinctes. Si un opérateur tente un rollback vers la version déjà portée par le champion, le script refuse l'opération avec une exception explicite (`rollback target is already the current champion`).

## Questions d'ingénierie

Répondez en au moins 80 caractères par question et citez une preuve réelle.

### Q1 — Continuité TP2 et serving

Pourquoi le fichier `model.joblib` du TP2 ne constitue-t-il pas à lui seul un contrat de service ?

**Réponse de l'étudiant :**

Le binaire sérialisé `model.joblib` ne définit aucun schéma d'entrée-sortie typé, ne garantit pas la compatibilité des versions de dépendances (comme numpy ou scikit-learn) et ne porte aucune métadonnée d'intégrité ou d'auditabilité. Sans contrat d'inférence HTTP explicite (Pydantic/FastAPI) et sans manifest de run (`reports/run_manifest.json`), il est impossible de vérifier quel commit a entraîné le modèle, sur quel hash de données il a été validé et s'il a reçu la décision `ACCEPT` lors du quality gate.

### Q2 — Validation Pydantic

Que protège Pydantic avant l'appel au modèle ?

**Réponse de l'étudiant :**

Pydantic agit comme un pare-feu d'ingénierie en validant rigoureusement les types, les plages physiques admissibles (ex: `air_temperature_k >= 250`) et en rejetant tout champ non sollicité via `extra="forbid"`. Comme démontré par la preuve `reports/figures/05_http_422_error.svg`, l'injection d'un champ non autorisé ou d'une valeur hors contrat déclenche immédiatement une erreur HTTP 422 standardisée avec `request_id`, empêchant tout crash Python imprévu ou empoisonnement silencieux du modèle scikit-learn.

### Q3 — Docker

Que rend Docker reproductible et que ne garantit-il pas ?

**Réponse de l'étudiant :**

Docker garantit la parfaite reproductibilité de l'environnement d'exécution logiciel (binaire Python, librairies C, paquets verrouillés issus de `uv.lock`, configuration système et isolation utilisateur non-root `app` avec healthcheck). En revanche, Docker ne garantit ni la performance statistique du modèle, ni l'absence de drift des données de production, ni l'alignement avec les exigences métier, qui relèvent exclusivement du Model Registry et de la promotion policy (`reports/promotion_decision.json`).

### Q4 — CI et promotion

Pourquoi `CI PASS` ne signifie-t-il pas `PROMOTE` ?

**Réponse de l'étudiant :**

Un statut `CI PASS` atteste uniquement que le code source compile, passe le linter `ruff`, satisfait les tests unitaires pytest et produit une image Docker fonctionnelle validée par smoke test. La promotion en production est une décision de gouvernance distincte qui exige de comparer les métriques du challenger à celles du champion en service dans le Registry (`configs/promotion_policy.yaml`) ; une pull request verte valide le logiciel mais ne doit jamais modifier l'alias `champion`.

### Q5 — Sélection multicritère

Pourquoi `max(accuracy)` est-il insuffisant dans ce contexte ?

**Réponse de l'étudiant :**

Dans un problème de maintenance prédictive où les pannes machine sont des événements rares, maximiser l'accuracy est trompeur car un modèle trivial prédisant toujours l'absence de panne obtiendrait une accuracy élevée tout en ayant un recall nul. La politique de release impose donc une sélection multicritère vectorielle (`f1 >= 0.64`, `recall >= 0.60`, `latency_ms <= 50 ms`) vérifiée dans `reports/model_selection.json` pour garantir la détection effective des défaillances sans dégrader le temps de réponse.

### Q6 — Gate et policy

Donnez un cas observé ou chiffré où le gate passe mais où le champion est conservé.

**Réponse de l'étudiant :**

Lors de notre évaluation, la version 2 a validé l'intégralité du quality gate avec `decision: ACCEPT` (`F1=0.7101`, `recall=0.6522`, `accuracy=0.9608`). Pourtant, comme consigné dans `reports/promotion_decision.json`, la décision finale a été `KEEP_CHAMPION` car le différentiel de F1 par rapport au champion v1 (`F1=0.7143`) était de `-0.0042`, ne franchissant pas le gain minimal obligatoire de `+0.01` exigé par `configs/promotion_policy.yaml`.

### Q7 — Alias et rollback

Comparez l'alias `champion` à une version codée en dur et décrivez le rollback.

**Réponse de l'étudiant :**

Coder en dur une version (ex: `models:/predictive-maintenance/1`) dans les microservices clients exige un redéploiement applicatif complet lors de chaque mise à jour. L'alias `champion` offre un niveau d'indirection dynamique permettant au service de résoudre la version courante à chaud sans redémarrage. Le rollback (`reports/rollback_decision.json`) réassigne cet alias instantanément vers une version antérieure sans altérer l'historique ni reconstruire l'image Docker, assurant une réversibilité immédiate en production.

### Q8 — Responsabilité humaine et limites

Où placer une validation humaine ou métier, et quelle preuve supplémentaire serait nécessaire avant une mise en production ?

**Réponse de l'étudiant :**

La validation humaine doit être positionnée immédiatement après la sélection automatique du challenger et avant le basculement de l'alias `champion`, sous la forme d'un gate d'approbation sur l'environnement de staging. La preuve supplémentaire indispensable comprendrait un rapport d'évaluation en conditions de charge réelles (latence p99 sous stress HTTP), une analyse de fairness par type de machine, ainsi qu'un test A/B ou canary deployment démontrant l'absence de faux positifs massifs sur un flux d'inférence miroir.

## Index des preuves

| Libellé | Fichier dans `reports/figures/` | Ce que la preuve démontre |
|---|---|---|
| Installation et `make doctor` | `reports/figures/01_make_doctor.svg` | Environnement opérationnel avec uv, git, pytest 8.4.2, ruff 0.16.10 et mlflow 3.17.0 |
| Tests API | `reports/figures/02_tests_api.svg` | Exécution complète de la suite de 23 tests unitaires et d'intégration validée dans `test_report.xml` |
| Réponses des trois endpoints | `reports/figures/03_endpoints_responses.svg` | Réponses HTTP 200 synchronisées pour `/health`, `/model-info`, `/predict` et `/v1/predict` |
| Contrat OpenAPI | `reports/figures/04_openapi_contract.svg` | Schéma d'interface formel exporté et validé sans dérive dans `contracts/openapi.json` |
| Erreur HTTP 422 | `reports/figures/05_http_422_error.svg` | Rejet strict Pydantic avec code 422 pour champ interdit (`extra="forbid"`), manquant ou hors bornes |
| Image Docker | `reports/figures/06_docker_image.svg` | Construction multi-stage non-root de l'image `maintenance-ml:tp3` avec `uv.lock` et healthcheck |
| Conteneur Docker | `reports/figures/07_docker_container.svg` | Exécution sécurisée du conteneur (--read-only, --tmpfs /tmp, --cap-drop=ALL) et validation du smoke test |
| Job CI vert | `reports/figures/08_ci_green.svg` | Validation nominale de la chaîne CI GitHub Actions (lint, tests JUnit, build et smoke test HTTP) |
| Job CI échoué | `reports/figures/09_ci_failure.svg` | Interruption immédiate du pipeline en cas de régression de linting ou de test unitaire |
| Version Registry | `reports/figures/10_registry_versions.svg` | Traçabilité des versions 1 et 2 dans MLflow Model Registry avec tags de gate et métriques |
| Décision de promotion | `reports/figures/11_promotion_decision.svg` | Évaluation comparative challenger/champion aboutissant à `KEEP_CHAMPION` (`promotion_decision.json`) |
| Rollback | `reports/figures/12_rollback.svg` | Réassignation atomique de l'alias champion de la version 2 vers la version 1 (`rollback_decision.json`) |
| Release manifest | `reports/figures/13_release_manifest.svg` | Contrat de clôture et de handoff `release_manifest.json` pour la consommation par le TP4 |

## Conclusion

Ce TP3 concrétise la chaîne d'industrialisation continue MLOps :
```text 
TP2 ACCEPT → Registry → service FastAPI → CI GitHub Actions → promotion contrôlée → champion → TP4
```
À partir d'un artefact validé au quality gate (`ACCEPT`), nous avons déployé un microservice d'inférence sécurisé par Pydantic et monitorable via OpenMetrics. L'intégration avec le MLflow Model Registry assure un versionnement immuable piloté par alias dynamiques (`challenger`, `champion`). La gouvernance de release prouve qu'un challenger éligible n'est promu que s'il surpasse significativement le champion en titre, tout en autorisant un retour arrière immédiat sans rupture de service. L'artefact final `reports/release_manifest.json` certifie l'identité complète du modèle champion transmis au TP4 pour le monitoring continu et la détection de drift.
