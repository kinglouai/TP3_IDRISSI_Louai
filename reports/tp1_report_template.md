# Rapport TP1

Nom et prénom : À compléter
Environnement (OS, version Python, uv) : À compléter
Archive : MLOps_TP1_Nom_Prenom.zip

## 1. Installation et diagnostic du notebook

Capture `reports/figures/01-installation.png`, commande, observation et limite.

| Défaut réellement observé (au moins cinq) | Risque | Correction | Preuve |
|---|---|---|---|
| À compléter | | | |

Indiquez aussi deux bonnes pratiques présentes. Comparez les deux exécutions :
un score identique ne prouve pas que les splits sont identiques.

## 2. Pipeline et configuration

Dessinez données → validation → split → fit sur train → évaluation → MLflow.
Donnez les entrées/sorties, la commande de rejeu et le commit utilisé.
Collez la configuration réellement utilisée (ne recopiez pas un autre exemple).
Preuve de lecture du YAML et preuve du nombre de lignes vues par fit : À compléter

## 3. Classes, baseline et seuils

| Jeu | Classe 0 | Classe 1 | Total | Taux de panne |
|---|---:|---:|---:|---:|
| Train | | | | |
| Test | | | | |

Comparez à la baseline « toujours aucune panne ». Affichez la matrice TN/FP/FN/TP.

| Seuil | Precision | Recall | F1 | FN | FP | Coût | Taux d'alertes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | | | | | | | |
| 0.05 | | | | | | | |
| 0.10 | | | | | | | |
| 0.30 | | | | | | | |
| 0.50 | | | | | | | |
| 0.70 | | | | | | | |

Métrique principale justifiée : À compléter
Cette analyse sur test est exploratoire; elle ne valide pas un seuil de production.

## 4. Expériences et reproductibilité

| Label | C | Run ID | Accuracy | Precision | Recall | F1 | Commit |
|---|---:|---|---:|---:|---:|---:|---|
| C01 | 0.1 | | | | | | |
| A | 1.0 | | | | | | |
| B | 1.0 | | | | | | |
| C10 | 10.0 | | | | | | |

Résultat de `make compare`, invariants et différences normales : À compléter
Hash des données, de la configuration, du lockfile et du modèle : À compléter
Capture MLflow et emplacement des exports consultables hors ligne : À compléter

## 5. Questions Q1–Q6

Pour chaque réponse, indiquez : **constat**, **preuve** (commande, fichier ou
capture) et **limite**. La pertinence prime sur la longueur.

### Q1 — Rejeu contrôlé
Que fixe `seed: 42` ? Citez deux autres éléments nécessaires pour rejouer A,
leur emplacement dans le projet, puis une cause de divergence qui peut subsister.

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

### Q2 — Provenance vérifiable
Depuis un snapshot A, montrez le lien entre run MLflow, commit, YAML, données et
lockfile. Que prouve un `run_id` et que ne prouve-t-il pas hors de votre serveur ?

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

### Q3 — Absence de fuite
Décrivez le chemin d'une variable du CSV à `StandardScaler.fit`. Quelle preuve
montre que le fit a vu 3 750 lignes ? Quelle fuite resterait possible malgré elle ?

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

### Q4 — Décision sous déséquilibre
Comparez le modèle à « toujours 0 » avec accuracy, recall, F1 et matrice de
confusion. Quelle décision chaque approche prend-elle pour les pannes ?

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

### Q5 — Seuil, coût et validation
Choisissez deux seuils. Interprétez l'écart de FN, FP, coût et alertes. Pourquoi
ce résultat ne suffit-il pas à promouvoir un seuil en production ?

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

### Q6 — A/B et paramètre C
Comparez A/B, puis C01/C10. Que démontre l'identité A/B ? Que modifie `C` pendant
l'entraînement, et pourquoi ne faut-il pas le confondre avec `decision_threshold` ?

- Constat : À compléter
- Preuve : À compléter
- Limite : À compléter

## 6. Remise et difficultés

Résultat du contrôle automatique, captures réelles commentées, commandes de
rejeu, erreurs rencontrées et résolutions : À compléter

## Extension facultative

Si réalisée : problème, modification, preuve, bénéfice et limite.
