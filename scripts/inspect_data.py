"""Affiche une vue initiale du jeu de données, sans appeler le code étudiant."""

from collections import Counter
from csv import DictReader
from pathlib import Path


DATASET = Path("data/raw/machines.csv")


def main() -> None:
    with DATASET.open(encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []

    failures = Counter(row.get("machine_failure", "<absent>") for row in rows)
    missing = sum(
        1 for row in rows for value in row.values() if value is None or value.strip() == ""
    )

    print(f"Fichier : {DATASET}")
    print("Colonnes : " + ", ".join(columns))
    print(f"Lignes : {len(rows)}")
    print("machine_failure : " + ", ".join(f"{key}={value}" for key, value in sorted(failures.items())))
    print(f"Cellules vides : {missing}")
    print("Cette inspection est indicative : implémentez validate(config) pour le contrôle contractuel.")


if __name__ == "__main__":
    main()
