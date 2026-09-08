"""Mesure du cout de chaque etage de la cascade.

La recherche prealable sur les projets ANPR publics montre que presque aucun ne
publie de chiffres de latence — y compris les plus aboutis. C'est pourtant la
premiere question d'un exploitant : combien de flux une machine peut-elle
traiter, et quel etage faut-il optimiser en premier.

Le script mesure separement chaque etage, et compare deux variantes qui font
souvent debat :

* **cascade contre detection directe** — chercher la plaque dans le crop du
  vehicule coute une passe de plus, mais sur une image plus petite. Le gain
  net n'est pas evident a priori.
* **une variante d'OCR contre plusieurs** — lire trois pretraitements
  ameliore le taux de lecture mais triple le cout de l'etage OCR.

Les mesures ecartent la premiere iteration : le premier appel a un modele
inclut son chargement et l'allocation des tampons, ce qui n'a rien a voir avec
le regime permanent.

Usage
-----
    python scripts/benchmark_latency.py --images data/input --runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WARMUP_RUNS = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/input", help="Dossier d'images de test")
    parser.add_argument("--runs", type=int, default=3, help="Repetitions par image")
    parser.add_argument("--limit", type=int, default=10, help="Nombre max d'images")
    parser.add_argument("--output", default="docs/benchmark.json")
    return parser.parse_args()


def load_images(folder: Path, limit: int):
    import cv2

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:limit]
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append((path.name, image))
    return images


def summarise(durations) -> dict:
    """Mediane et centile 95, plus parlants qu'une moyenne.

    Une moyenne est tiree par les valeurs extremes ; sur une mesure de latence
    ce sont precisement ces extremes qui interessent, mais separement.
    """
    if not durations:
        return {}
    ordered = sorted(durations)
    return {
        "median_ms": round(statistics.median(ordered), 2),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
        "samples": len(ordered),
    }


def measure(pipeline, images, runs: int) -> dict:
    """Chronometre le pipeline sur un jeu d'images."""
    per_stage: dict = {}
    totals = []

    for _ in range(WARMUP_RUNS):
        for _, image in images[:1]:
            pipeline.process_frame(image)

    for _ in range(runs):
        for _, image in images:
            started = time.perf_counter()
            result = pipeline.process_frame(image)
            totals.append((time.perf_counter() - started) * 1000)

            for stage, value in result.timings_ms.items():
                per_stage.setdefault(stage, []).append(value)

    return {
        "total": summarise(totals),
        "stages": {stage: summarise(values) for stage, values in per_stage.items()},
    }


def main() -> int:
    args = parse_args()

    folder = Path(args.images)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        print(f"Dossier introuvable : {folder}\nVoir docs/DATA.md.")
        return 1

    images = load_images(folder, args.limit)
    if not images:
        print(f"Aucune image lisible dans {folder}")
        return 1

    from anpr.config import load_config
    from anpr.pipeline import ANPRPipeline

    config = load_config()
    print(f"Images     : {len(images)}  |  repetitions : {args.runs}\n")

    report: dict = {"images": len(images), "runs": args.runs, "variants": {}}

    scenarios = [
        ("cascade_ocr_variants", {"read_plates": True, "ocr_variants": True}),
        ("cascade_ocr_single", {"read_plates": True, "ocr_variants": False}),
        ("detection_only", {"read_plates": False, "ocr_variants": False}),
    ]

    for name, overrides in scenarios:
        pipeline_config = config.to_pipeline_config()
        for key, value in overrides.items():
            setattr(pipeline_config, key, value)

        print(f"--- {name} ---")
        pipeline = ANPRPipeline(pipeline_config)
        measurements = measure(pipeline, images, args.runs)
        report["variants"][name] = measurements

        total = measurements["total"]
        print(f"  total   mediane {total['median_ms']:>8.2f} ms   p95 {total['p95_ms']:>8.2f} ms")
        for stage, values in measurements["stages"].items():
            print(f"  {stage:<20} mediane {values['median_ms']:>8.2f} ms")
        if total["median_ms"] > 0:
            print(f"  debit estime : {1000 / total['median_ms']:.2f} images/s\n")

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rapport ecrit dans {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
