# Fiche de données — TP1

## Nature et provenance

- **Fichier utilisé :** `data/raw/machines.csv`.
- **Nature :** jeu de données pédagogique simulé; il ne contient pas de mesures
  industrielles réelles.
- **Création dans ce dépôt :** importé avec le kit TP1 le 20 septembre 2026.
- **Contenu fourni :** 5 000 observations, dont 200 pannes (4 %), et 8 colonnes.
  Les identifiants ont la forme `M-xxxxx`; la cible binaire est `machine_failure`.
- **Transformation pendant le TP :** aucune jointure, donnée externe ou
  transformation cachée n'est appliquée. Le CSV fourni est la référence.

## Licence et usage

Une licence de données distincte n'est pas déclarée dans ce dépôt. Le fichier est
fourni uniquement pour les activités pédagogiques du cours; ne le réutilisez pas
comme jeu de référence public ou comme base d'une décision industrielle.

## Limites scientifiques

- Les 5 000 observations ne représentent pas un historique réel de machines.
- Les 50 pannes du test rendent le recall moins instable que dans la première
  version du TP, mais restent insuffisantes pour valider une politique industrielle.
- Le split aléatoire stratifié est choisi pour illustrer un pipeline; il ne simule
  pas un déploiement chronologique ou sur de nouvelles machines.
- Les poids de coût du YAML sont des scénarios de discussion, sans unité monétaire.
