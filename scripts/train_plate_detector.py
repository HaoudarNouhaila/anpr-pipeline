"""Entrainement du detecteur de plaques.

C'est le seul modele du projet qui demande un entrainement : aucun jeu de
classes standard ne contient « plaque d'immatriculation », contrairement aux
classes de vehicules qui figurent deja dans COCO.

Le script n'embarque aucun telechargeur lie a un fournisseur particulier. Il
prend un dataset au format YOLO — un fichier `data.yaml` decrivant les chemins
et les classes — quelle qu'en soit la provenance. `docs/DATA.md` liste les
sources possibles et la maniere de les recuperer.

Chaque entrainement est journalise dans MLflow : parametres, metriques par
epoque et chemin des poids produits. Sans cela, comparer deux essais revient a
relire des dossiers `runs/` horodates.

Usage
-----
    python scripts/train_plate_detector.py --data data/datasets/plates/data.yaml
    python scripts/train_plate_detector.py --data ... --epochs 30 --imgsz 640
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# MLflow 3 refuse le backend fichier ("./mlruns"), passe en maintenance. Une base
# SQLite locale est la cible recommandee et ne demande aucun serveur. La variable
# doit etre posee avant l'import d'ultralytics, dont le callback MLflow
# s'initialise au chargement du module.
MLFLOW_DIR = PROJECT_ROOT / "data" / "mlflow"
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "plate_detector.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Chemin du data.yaml au format YOLO")
    parser.add_argument("--base", default="yolov8n.pt", help="Poids de depart")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille d'entree. 640 est le standard ; 416 divise par ~2,4 le "
             "cout d'entrainement sur processeur, au prix du rappel sur les "
             "plaques lointaines.",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10, help="Arret anticipe")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Processus de chargement des donnees. Ultralytics en met 0 par "
             "defaut sur Windows ; le chargement devient alors le goulot "
             "d'etranglement devant le calcul.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Garder les images decodees en memoire. Supprime le cout de "
             "decodage a chaque epoque, au prix de la RAM.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction du jeu d'entrainement a utiliser. Sur processeur, "
             "commencer a 0.1 permet de valider la chaine en quelques minutes.",
    )
    return parser.parse_args()


def log_to_mlflow(args, metrics: dict, weights: Path) -> None:
    """Journalise le run. L'echec de MLflow ne doit pas perdre l'entrainement."""
    try:
        import mlflow
    except ImportError:
        print("MLflow absent : journalisation ignoree.")
        return

    mlflow.set_experiment("anpr-plate-detector")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_weights": args.base,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "fraction": args.fraction,
                "dataset": args.data,
            }
        )
        mlflow.log_metrics(metrics)
        if weights.exists():
            mlflow.log_artifact(str(weights))
    print("Run journalise dans MLflow (mlflow ui pour le consulter).")


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print(f"Dataset introuvable : {data_path}")
        print("Voir docs/DATA.md pour obtenir un dataset au format YOLO.")
        return 1

    from ultralytics import YOLO

    print(f"Dataset    : {data_path}")
    print(f"Base       : {args.base}")
    print(f"Epoques    : {args.epochs} | taille {args.imgsz} | batch {args.batch}")
    print(f"Fraction   : {args.fraction:.0%} du jeu d'entrainement")
    print(f"Chargement : {args.workers} workers, cache={'oui' if args.cache else 'non'}")
    print("Entrainement sur processeur : comptez plusieurs dizaines de minutes.\n")

    model = YOLO(args.base)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        fraction=args.fraction,
        workers=args.workers,
        cache=args.cache,
        project=str(PROJECT_ROOT / "data" / "runs"),
        name="plate_detector",
        exist_ok=True,
        verbose=True,
    )

    # Recuperation des metriques de validation du meilleur modele.
    metrics = {}
    try:
        box = results.box
        metrics = {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
    except AttributeError:
        print("Metriques indisponibles dans l'objet de resultats.")

    best = Path(results.save_dir) / "weights" / "best.pt"
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    if best.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, output)
        print(f"\nPoids copies vers {output}")
    else:
        print(f"\nPoids introuvables : {best}")
        return 1

    if metrics:
        print("\n===== METRIQUES DE VALIDATION =====")
        for name, value in metrics.items():
            print(f"  {name:<12} {value:.4f}")

    if not args.no_mlflow:
        log_to_mlflow(args, metrics, output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
