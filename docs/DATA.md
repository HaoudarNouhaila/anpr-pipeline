# Données et modèles

## Rien de tout cela n'est versionné

Ni les datasets, ni les poids, ni les images de test ne sont dans le dépôt. Trois raisons :

- **Licence** — le dataset d'entraînement est sous CC BY 4.0. On peut l'utiliser, à condition
  de créditer ses auteurs ; le redistribuer dans un dépôt tiers est une autre question, qu'il
  est plus simple d'éviter en le téléchargeant à la demande.
- **Taille** — 8 823 images pèsent environ 230 Mo, et les poids YOLO plusieurs dizaines de Mo.
  Un dépôt Git n'est pas fait pour ça.
- **Données personnelles** — une plaque d'immatriculation identifie indirectement une personne.
  Publier des images de voie publique dans un dépôt public engage une responsabilité inutile.

Tout se reconstruit avec deux commandes, documentées ci-dessous.

---

## Dataset de plaques

```bash
python scripts/download_dataset.py
```

| | |
|---|---|
| **Source** | [`keremberke/license-plate-object-detection`](https://huggingface.co/datasets/keremberke/license-plate-object-detection) sur Hugging Face |
| **Origine** | *Vehicle Registration Plates*, Augmented Startups, via Roboflow |
| **Licence** | **CC BY 4.0** — attribution obligatoire en cas d'utilisation |
| **Volume** | 8 823 images (6 176 entraînement · 1 765 validation · 882 test) |
| **Classes** | une seule : `license_plate` |

### Pourquoi Hugging Face et pas Roboflow

Roboflow et Kaggle exigent une clé d'API pour télécharger. Un projet dont l'installation
suppose de créer un compte et de coller un secret n'est pas reproductible. Hugging Face sert
ce dataset sans authentification.

### La conversion COCO vers YOLO

L'archive est annotée au format **COCO** : les images à plat, plus un unique
`_annotations.coco.json`. Ultralytics attend du **YOLO** : un fichier texte par image, dans un
dossier `labels/` parallèle à `images/`.

Les deux formats décrivent la même chose autrement :

| | COCO | YOLO |
|---|---|---|
| Boîte | `[x, y, largeur, hauteur]` | `classe cx cy largeur hauteur` |
| Origine | coin supérieur gauche | **centre** |
| Unité | pixels absolus | **normalisé entre 0 et 1** |
| Fichier | un JSON pour tout le split | un `.txt` par image |

`scripts/download_dataset.py` fait la conversion. Deux détails qui ne sont pas évidents :
l'export Roboflow ajoute une super-catégorie factice en position 0, qu'il faut écarter pour que
les indices de classe soient contigus ; et une image sans annotation reçoit malgré tout un
fichier vide, car c'est ainsi qu'on déclare un négatif à Ultralytics — et les négatifs
apprennent au modèle où il n'y a *pas* de plaque.

---

## Modèles

### Détecteur de véhicules — aucun entraînement

Les quatre classes utiles (`car`, `truck`, `bus`, `motorcycle`) figurent déjà dans COCO.
`yolov8n.pt` est téléchargé automatiquement par Ultralytics au premier lancement.

### Détecteur de plaques — entraînement obligatoire

Aucun jeu de classes standard ne contient « plaque d'immatriculation ». C'est le seul modèle
du projet à entraîner :

```bash
python scripts/train_plate_detector.py --data data/datasets/plates/data.yaml
```

Sur processeur, réduisez la charge :

```bash
python scripts/train_plate_detector.py \
    --data data/datasets/plates/data.yaml \
    --imgsz 416 --batch 16 --fraction 0.35 --epochs 20
```

`--imgsz 416` divise environ par 2,4 le coût d'une époque par rapport au 640 standard, au prix
du rappel sur les plaques lointaines. `--fraction` n'utilise qu'une part du jeu d'entraînement.

Les poids atterrissent dans `data/models/plate_detector.pt`, et le run est journalisé dans
MLflow (`mlflow ui` pour le consulter).

### EasyOCR

Les modèles de détection et de reconnaissance de texte (~100 Mo) sont téléchargés au premier
appel, dans `~/.EasyOCR/`.

---

## Images de test

Placez vos propres images dans `data/input/`. Le dossier est vide dans le dépôt, à dessein.

Pour un test rapide sans fournir d'images, le split de test du dataset fait l'affaire :

```bash
python scripts/benchmark_latency.py --images data/datasets/plates/test/images
```

---

## Attribution

En cas de réutilisation, créditez la source du dataset comme l'exige la licence CC BY 4.0 :

> *Vehicle Registration Plates* — Augmented Startups, distribué via Roboflow, licence CC BY 4.0.
