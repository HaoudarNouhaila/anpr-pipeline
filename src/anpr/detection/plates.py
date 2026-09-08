"""Detection des plaques a l'interieur de la region du vehicule.

Deuxieme etage de la cascade. Deux differences majeures avec l'etage vehicule :

1. **Le modele doit etre entraine.** Aucun jeu de classes standard ne contient
   « plaque d'immatriculation ». C'est le seul entrainement du projet, et
   `scripts/train_plate_detector.py` s'en charge.

2. **La recherche est restreinte au crop du vehicule.** L'interet mesure de
   cette restriction est l'ASSOCIATION : chaque plaque sort rattachee au
   vehicule qui la porte, sans etape de mise en correspondance, et le type de
   vehicule accompagne la plaque dans la sortie.

   L'argument souvent avance — la cascade eliminerait les faux positifs sur
   panneaux et affiches — est bien plus faible que annonce. Mesure par
   `scripts/evaluate.py` : la detection directe trouve les MEMES 17 plaques que
   la cascade sur les scenes du jeu de test, dont seulement 2 hors de tout
   vehicule, soit 11,8 %. C'est logique : cet argument vaut contre un detecteur
   generique, pas contre un modele entraine specifiquement sur des plaques, qui
   ne se declenche pas sur un panneau.

Les boites renvoyees sont ramenees dans le repere de l'image entiere avant de
quitter ce module, pour que le reste du pipeline n'ait jamais a se soucier de
quel crop provient quelle detection.
"""

from __future__ import annotations

from pathlib import Path

from ..types import BoundingBox, PlateDetection, VehicleDetection

# Une plaque europeenne mesure 520 x 110 mm, soit un rapport de 4,7. Vue de
# biais ce rapport diminue ; vue de face il ne le depasse guere. Les bornes
# ci-dessous ecartent les detections de forme manifestement impossible sans
# rejeter les prises de vue obliques.
MIN_ASPECT_RATIO = 1.5
MAX_ASPECT_RATIO = 8.0

# Marge ajoutee autour de la boite du vehicule avant le crop. Un detecteur
# cadre parfois le vehicule un peu court et coupe le pare-chocs, donc la plaque.
VEHICLE_CROP_MARGIN = 0.05


class PlateDetectorUnavailable(RuntimeError):
    """Levee quand les poids du detecteur de plaques sont introuvables."""


class PlateDetector:
    """Localise les plaques dans la region d'un vehicule.

    Parameters
    ----------
    weights:
        Poids d'un YOLO fine-tune sur des plaques.
    confidence:
        Seuil de confiance. Volontairement bas : une plaque lointaine occupe
        peu de pixels, et un faux positif est ensuite ecarte par le filtre de
        forme puis par le filtre de plausibilite de l'OCR.
    """

    def __init__(self, weights: str, confidence: float = 0.25):
        weights_path = Path(weights)
        if not weights_path.exists() and not str(weights).startswith("yolo"):
            raise PlateDetectorUnavailable(
                f"Poids du detecteur de plaques introuvables : {weights}\n"
                "Entrainez le modele avec :\n"
                "  python scripts/train_plate_detector.py\n"
                "Voir docs/DATA.md pour le telechargement du dataset."
            )

        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.confidence = confidence

    # ------------------------------------------------------------------
    def detect_in_vehicle(self, frame, vehicle: VehicleDetection) -> PlateDetection | None:
        """Cherche la plaque du vehicule donne. Renvoie la meilleure, ou None."""
        height, width = frame.shape[:2]
        region = vehicle.box.expand(VEHICLE_CROP_MARGIN).clip(width, height)
        crop = region.crop(frame)

        if crop.size == 0:
            return None

        candidates = self._detect_in_crop(crop)
        if not candidates:
            return None

        # Retour au repere de l'image entiere, une fois pour toutes.
        best = max(candidates, key=lambda d: d.confidence)
        return PlateDetection(
            box=best.box.shift(region.left, region.top),
            confidence=best.confidence,
        )

    def detect_in_frame(self, frame) -> list:
        """Cherche les plaques dans l'image entiere, sans etage vehicule.

        Sert de point de comparaison a la cascade dans `scripts/evaluate.py`.
        C'est aussi le seul mode utilisable sur une image ou aucun vehicule
        n'est visible — un gros plan de plaque, par exemple.
        """
        return self._detect_in_crop(frame)

    # ------------------------------------------------------------------
    def _detect_in_crop(self, image) -> list:
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]

        detections = []
        for box in result.boxes:
            bbox = BoundingBox.from_xyxy(box.xyxy.tolist()[0])
            if not self._has_plausible_shape(bbox):
                continue
            detections.append(
                PlateDetection(box=bbox, confidence=float(box.conf.tolist()[0]))
            )

        return detections

    @staticmethod
    def _has_plausible_shape(box: BoundingBox) -> bool:
        """Ecarte les detections dont la forme ne peut pas etre une plaque."""
        if box.width < 8 or box.height < 4:
            return False
        return MIN_ASPECT_RATIO <= box.aspect_ratio <= MAX_ASPECT_RATIO


def attach_plates(vehicles: list, plates: list) -> dict:
    """Rattache des plaques detectees globalement aux vehicules.

    Utilise uniquement en mode comparaison, quand les plaques ont ete cherchees
    dans l'image entiere. Le rattachement se fait par appartenance du centre de
    la plaque a la boite du vehicule.
    """
    assignments: dict = {}

    for index, vehicle in enumerate(vehicles):
        inside = [p for p in plates if vehicle.box.contains_centre_of(p.box)]
        if inside:
            assignments[index] = max(inside, key=lambda p: p.confidence)

    return assignments
