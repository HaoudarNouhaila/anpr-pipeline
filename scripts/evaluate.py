"""Evaluation du pipeline.

Ce que ce script mesure, et ce qu'il ne mesure pas
-------------------------------------------------
Le dataset d'entrainement annote la **position** des plaques, pas leur
**texte**. On peut donc evaluer rigoureusement la detection, mais pas
l'exactitude de la lecture : il n'existe aucune verite terrain a laquelle
comparer les chaines produites.

Le script mesure donc trois choses reellement mesurables :

1. **Qualite de detection** — mAP, precision et rappel, via la validation
   d'Ultralytics sur le split de test.
2. **Apport de la cascade** — nombre de plaques detectees hors de tout
   vehicule quand on cherche dans l'image entiere. Ce sont, par construction,
   des faux positifs que la cascade elimine : c'est la justification chiffree
   du choix d'architecture.
3. **Taux de lecture** — proportion de plaques detectees qui produisent une
   chaine plausible, et proportion de ces chaines qui respectent un format de
   pays connu.

Le taux de lecture n'est pas un taux d'exactitude. Une plaque lue de travers
mais plausible compte comme lue. Mesurer l'exactitude demanderait un jeu de
donnees annote en texte ; c'est indique dans les limites du README.

Usage
-----
    python scripts/evaluate.py --data data/datasets/plates/data.yaml
    python scripts/evaluate.py --sample 100 --skip-detection
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/datasets/plates/data.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=80, help="Images pour les mesures pipeline")
    parser.add_argument("--skip-detection", action="store_true", help="Sauter la validation YOLO")
    parser.add_argument("--output", default="docs/evaluation.json")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_detection(data_yaml: Path, weights: Path, split: str) -> dict:
    """Metriques de detection standard, calculees par Ultralytics."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, verbose=False)

    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def evaluate_pipeline(images, pipeline, plate_detector) -> dict:
    """Compare la cascade a la detection directe, et mesure le taux de lecture.

    Le dataset est heterogene : il melange des scenes de rue et des gros plans
    synthetiques de plaques, sans vehicule autour. Melanger les deux fausserait
    toute conclusion — la cascade est par construction inevaluable sur une image
    ou aucun vehicule n'est visible, puisqu'il n'y a rien a recadrer.

    Les images sont donc classees en deux categories, et les metriques de
    cascade ne portent que sur les scenes. Le decompte des gros plans est
    rapporte separement, parce qu'il dit quelque chose d'utile : sur ce
    sous-ensemble, la detection directe est la seule approche possible.
    """
    import cv2

    stats = {
        "images_total": 0,
        "images_with_vehicles": 0,
        "images_plate_closeups": 0,
        "vehicles": 0,
        "plates_cascade": 0,
        "plates_direct_on_scenes": 0,
        "plates_direct_outside_vehicles": 0,
        "plates_direct_on_closeups": 0,
        "readings_plausible": 0,
        "readings_with_known_format": 0,
    }

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        stats["images_total"] += 1
        result = pipeline.process_frame(frame)
        direct = plate_detector.detect_in_frame(frame)

        if not result.records:
            # Aucun vehicule : gros plan de plaque, ou scene ou le detecteur a
            # echoue. Dans les deux cas la cascade n'a rien a evaluer.
            stats["images_plate_closeups"] += 1
            stats["plates_direct_on_closeups"] += len(direct)
            continue

        stats["images_with_vehicles"] += 1
        stats["vehicles"] += len(result.records)
        stats["plates_cascade"] += sum(1 for r in result.records if r.plate is not None)
        stats["readings_plausible"] += sum(1 for r in result.records if r.reading is not None)
        stats["readings_with_known_format"] += sum(
            1 for r in result.records if r.reading is not None and r.reading.is_valid_format
        )

        stats["plates_direct_on_scenes"] += len(direct)
        vehicle_boxes = [r.vehicle.box for r in result.records]
        stats["plates_direct_outside_vehicles"] += sum(
            1 for p in direct if not any(v.contains_centre_of(p.box) for v in vehicle_boxes)
        )

    detected = stats["plates_cascade"]
    stats["read_rate"] = round(stats["readings_plausible"] / detected, 4) if detected else 0.0
    stats["known_format_rate"] = (
        round(stats["readings_with_known_format"] / stats["readings_plausible"], 4)
        if stats["readings_plausible"] else 0.0
    )
    stats["direct_outside_vehicle_rate"] = (
        round(stats["plates_direct_outside_vehicles"] / stats["plates_direct_on_scenes"], 4)
        if stats["plates_direct_on_scenes"] else 0.0
    )

    return stats


def main() -> int:
    args = parse_args()

    data_yaml = resolve(args.data)
    if not data_yaml.exists():
        print(f"Dataset introuvable : {data_yaml}\nLancez scripts/download_dataset.py")
        return 1

    from anpr.config import load_config
    from anpr.detection import PlateDetector
    from anpr.pipeline import ANPRPipeline

    config = load_config()
    weights = Path(config.resolve(config.plate_weights))
    if not weights.exists():
        print(f"Poids introuvables : {weights}\nLancez scripts/train_plate_detector.py")
        return 1

    report: dict = {"weights": weights.name, "split": args.split}

    if not args.skip_detection:
        print("Validation du detecteur de plaques...")
        report["detection"] = evaluate_detection(data_yaml, weights, args.split)
        for name, value in report["detection"].items():
            print(f"  {name:<12} {value:.4f}")

    split_dir = data_yaml.parent / args.split / "images"
    if not split_dir.exists():
        print(f"Split introuvable : {split_dir}")
        return 1

    images = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(args.seed).shuffle(images)
    images = images[: args.sample]

    print(f"\nEvaluation du pipeline sur {len(images)} images...")
    pipeline = ANPRPipeline(config.to_pipeline_config())
    plate_detector = PlateDetector(config.resolve(config.plate_weights), config.plate_confidence)

    report["pipeline"] = evaluate_pipeline(images, pipeline, plate_detector)

    stats = report["pipeline"]

    print("\n===== COMPOSITION DE L'ECHANTILLON =====")
    print(f"  images analysees               {stats['images_total']}")
    print(f"  scenes avec vehicule           {stats['images_with_vehicles']}")
    print(f"  gros plans sans vehicule       {stats['images_plate_closeups']}"
          f"   (cascade inevaluable ; {stats['plates_direct_on_closeups']} plaques en direct)")

    print("\n===== CASCADE CONTRE DETECTION DIRECTE (scenes seulement) =====")
    print(f"  vehicules detectes             {stats['vehicles']}")
    print(f"  plaques via cascade            {stats['plates_cascade']}")
    print(f"  plaques via detection directe  {stats['plates_direct_on_scenes']}")
    print(f"  dont hors de tout vehicule     {stats['plates_direct_outside_vehicles']}"
          f"  ({stats['direct_outside_vehicle_rate']:.1%})")

    print("\n===== LECTURE =====")
    print(f"  taux de lecture                {stats['read_rate']:.1%}")
    print(f"  dont format pays reconnu       {stats['known_format_rate']:.1%}")
    print("\n  Rappel : ce taux mesure la production d'une chaine plausible,")
    print("  pas son exactitude — le dataset n'annote pas le texte des plaques.")

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nRapport ecrit dans {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
