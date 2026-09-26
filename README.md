# TP1 — Projet de départ : expérience reproductible

Ce projet accompagne l'énoncé du TP1. Le ZIP contient directement la racine du
projet : toutes les commandes ci-dessous s'exécutent ici.

**Échéance de soumission : mercredi 30 septembre 2026 à 23 h 59. Aucun retard
n'est accepté.**

## Rôle du projet

La régression logistique sert de support à l'étude d'une expérience MLOps
traçable, configurable et réexécutable. Le TP évalue les preuves produites,
l'organisation du pipeline et l'interprétation des résultats; il ne demande pas
d'améliorer le score ni de rechercher un autre algorithme.

Le jeu de données est fourni pour ce laboratoire. Consultez `docs/donnees.md`
pour sa provenance, sa licence et ses limites d'interprétation.

## Environnement pris en charge

Linux, macOS avec outils de compilation/Make, ou Windows **dans WSL2 Ubuntu**.
PowerShell seul n'est pas pris en charge. Python de référence : 3.13, sélectionné
par `.python-version`. `uv.lock` verrouille les dépendances, pas l'interpréteur.
Internet est nécessaire à la première installation; le mode hors ligne suppose
que Python et toutes les dépendances sont déjà dans le cache.

```bash
uv python install 3.13
make install
make config
make report
make doctor
make test
make inspect-data
make notebook
```

Dans Jupyter, ouvrez `notebooks/0.01-starter.ipynb`. Vérifiez le kernel avec
`import sys; print(sys.executable)` : il doit pointer vers `.venv` du kit.
Le notebook travaille dans `notebooks/`; le programme travaille depuis la racine.

## Ce qui est fourni et ce qui reste à écrire

- Données et notebook fonctionnels; le preprocessing du notebook est déjà sans fuite.
- `maintenance_ml/modeling/train.py` : squelette, fonctions à compléter.
- `docs/contrat.md` : formats exacts des sorties et exemple MLflow.
- `tests/validate_submission.py` : contrôles automatiques du résultat.
- `scripts/evidence.py` : conservation des runs et création de l'archive.

`make test` vérifie l'installation et les contrôles du kit. Il ne valide pas
votre pipeline. `make train` échoue avec `NotImplementedError` tant que le
squelette n'est pas complété : cet échec initial est attendu.
`make inspect-data` résume le CSV sans appeler votre code; utilisez `make check-data`
une fois `validate` implémentée.

## Parcours

Suivez les étapes de l'énoncé : audit → configuration et Git → validation minimale
→ split et entraînement → évaluation → MLflow → quatre runs → remise.
Lancez `make mlflow-server` dans un second terminal **depuis la même racine**;
puis `make server-check` dans le premier. Les commandes utilisent `uv run`,
sans activation manuelle de l'environnement.

Exécutions obligatoires : `C01` (C=0.1), `A` (C=1), `B` (répétition de A),
`C10` (C=10). Après chacune : `make snapshot LABEL=C01` (adapter le label).
Ne changez que C entre les expériences; conservez seed, split, seuil et coûts.
Le fichier créé par `make config` commence à `C=1.0` : avant le premier
entraînement, mettez explicitement `model.C: 0.1`, committez ce changement et
lancez C01. `make compare` compare A et B. Les chemins et champs obligatoires sont définis
dans `docs/contrat.md` et dans le PDF. Les datasets externes sont réservés au TP2.

## Remise

Complétez uniquement `reports/tp1_report.md`, avec vos captures dans
`reports/figures/`. Répondez aux six questions Q1–Q6 du modèle, identiques au PDF.
Une extension est facultative, après le parcours obligatoire.

```bash
make validate-submission
make package NAME=Nom_Prenom
```

L'archive `MLOps_TP1_Nom_Prenom.zip` est écrite dans le dossier parent. Elle
contient code, données, configuration, modèle, rapports et snapshots MLflow;
elle exclut `.venv/`, `.git/`, caches, `mlflow.db` et `mlartifacts/`.
Un lecteur peut inspecter les snapshots hors ligne; rejouer crée de nouveaux
runs sur son propre serveur. Un identifiant de votre serveur local n'est pas
un lien consultable sur une autre machine.
