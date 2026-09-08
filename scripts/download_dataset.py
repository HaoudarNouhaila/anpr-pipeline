"""Telechargement du dataset de plaques.

Le detecteur de plaques est le seul modele du projet qui demande un
entrainement, donc un jeu de donnees annote. Ce script recupere un dataset
public depuis Hugging Face, qui presente deux avantages sur les autres
sources : il ne demande aucune cle d'API — contrairement a Roboflow ou Kaggle,
ce qui rendrait le projet non reproductible sans compte — et il est deja au
format attendu par Ultralytics.

Le dataset n'est pas redistribue dans ce depot : il est exclu par `.gitignore`
et retelecharge par ce script. C'est la maniere correcte de traiter une donnee
sous licence CC BY 4.0 dont on n'est pas l'auteur.

Source : keremberke/license-plate-object-detection sur Hugging Face
         8823 images, licence CC BY 4.0, Augmented Startups via Roboflow

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output data/datasets/plates
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPO_ID = "keremberke/license-plate-object-detection"
SPLITS = {"train": "data/train.zip", "valid": "data/valid.zip", "test": "data/test.zip"}

DATA_YAML = """# Genere par scripts/download_dataset.py
# Source : {repo} (CC BY 4.0)
path: {root}
train: train/images
val: valid/images
test: test/images

names:
  0: license_plate
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/datasets/plates")
    parser.add_argument("--force", action="store_true", help="Retelecharger meme si present")
    return parser.parse_args()


def coco_to_yolo(annotations: dict, destination: Path) -> int:
    """Convertit des annotations COCO en fichiers d'etiquettes YOLO.

    Les deux formats decrivent la meme chose autrement :

    * COCO donne `[x, y, largeur, hauteur]` en pixels absolus, coin superieur
      gauche, dans un unique fichier JSON pour tout le split.
    * YOLO attend `classe cx cy largeur hauteur`, centre et normalise entre 0
      et 1, dans un fichier texte par image.

    La conversion doit donc translater l'origine vers le centre et diviser par
    les dimensions de l'image. Une image sans annotation recoit malgre tout un
    fichier vide : c'est ainsi qu'on declare un negatif a Ultralytics, et les
    negatifs sont utiles — ils apprennent au modele ou il n'y a pas de plaque.
    """
    images_dir = destination / "images"
    labels_dir = destination / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Ce dataset ne contient qu'une classe utile, mais l'export Roboflow ajoute
    # une super-categorie factice en position 0. On remappe donc les
    # identifiants COCO vers un indice YOLO contigu commencant a 0.
    categories = [c for c in annotations.get("categories", []) if c.get("supercategory") != "none"]
    category_to_index = {c["id"]: index for index, c in enumerate(categories)}

    boxes_by_image: dict = {}
    for annotation in annotations.get("annotations", []):
        boxes_by_image.setdefault(annotation["image_id"], []).append(annotation)

    written = 0
    for image in annotations.get("images", []):
        source = destination / image["file_name"]
        if not source.exists():
            continue

        width, height = image["width"], image["height"]
        lines = []

        for annotation in boxes_by_image.get(image["id"], []):
            index = category_to_index.get(annotation["category_id"])
            if index is None:
                continue

            x, y, box_width, box_height = annotation["bbox"]
            centre_x = (x + box_width / 2) / width
            centre_y = (y + box_height / 2) / height
            norm_width = box_width / width
            norm_height = box_height / height

            # Une boite degeneree ou hors cadre corromprait l'entrainement.
            if not (0 < norm_width <= 1 and 0 < norm_height <= 1):
                continue
            if not (0 <= centre_x <= 1 and 0 <= centre_y <= 1):
                continue

            lines.append(
                f"{index} {centre_x:.6f} {centre_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        stem = Path(image["file_name"]).stem
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        shutil.move(str(source), str(images_dir / image["file_name"]))
        written += 1

    return written


def extract_split(archive: Path, destination: Path) -> int:
    """Extrait une archive de split et la convertit au format YOLO."""
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)

    # Certains exports encapsulent le contenu dans un dossier intermediaire.
    if not (destination / "_annotations.coco.json").exists():
        for candidate in [p for p in destination.iterdir() if p.is_dir()]:
            if (candidate / "_annotations.coco.json").exists():
                for item in candidate.iterdir():
                    shutil.move(str(item), str(destination / item.name))
                candidate.rmdir()
                break

    annotations_file = destination / "_annotations.coco.json"
    if not annotations_file.exists():
        print(f"\n  Annotations introuvables dans {destination}", end="")
        return 0

    import json

    annotations = json.loads(annotations_file.read_text(encoding="utf-8"))
    count = coco_to_yolo(annotations, destination)
    annotations_file.unlink()

    return count


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub est requis :\n  pip install huggingface_hub")
        return 1

    args = parse_args()

    root = Path(args.output)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if root.exists() and not args.force:
        existing = root / "data.yaml"
        if existing.exists():
            print(f"Dataset deja present : {root}")
            print("Utilisez --force pour retelecharger.")
            return 0

    print(f"Source : {REPO_ID}  (CC BY 4.0)\nCible  : {root}\n")

    counts = {}
    for split, remote in SPLITS.items():
        print(f"  {split:<6} telechargement...", end=" ", flush=True)
        archive = Path(hf_hub_download(REPO_ID, remote, repo_type="dataset"))
        counts[split] = extract_split(archive, root / split)
        print(f"{counts[split]} images")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        DATA_YAML.format(repo=REPO_ID, root=root.as_posix()), encoding="utf-8"
    )

    total = sum(counts.values())
    print(f"\nTotal : {total} images")
    print(f"Configuration : {data_yaml}")
    print("\nEntrainement :")
    print(f"  python scripts/train_plate_detector.py --data {data_yaml.relative_to(PROJECT_ROOT).as_posix()}")

    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
