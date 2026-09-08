"""Types partages par tout le pipeline.

Le pipeline est une cascade : vehicule -> plaque dans la region du vehicule ->
lecture OCR. Chaque etage produit un type distinct plutot qu'un dictionnaire
anonyme, ce qui rend explicite ce que chaque etage ajoute et evite de faire
circuler des tuples dont on oublie l'ordre.

Convention de coordonnees : toutes les boites sont en pixels ABSOLUS dans le
repere de l'image d'origine. La detection de plaque travaille pourtant sur un
crop du vehicule ; c'est `BoundingBox.shift` qui ramene ses resultats dans le
repere global, une fois pour toutes, au moment ou ils sortent de l'etage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Classes COCO correspondant a des vehicules, avec le libelle rendu a l'usager.
VEHICLE_CLASSES = {
    "car": "voiture",
    "motorcycle": "moto",
    "bus": "bus",
    "truck": "camion",
}


@dataclass(frozen=True)
class BoundingBox:
    """Boite englobante en pixels absolus."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_xyxy(cls, values) -> "BoundingBox":
        left, top, right, bottom = values
        return cls(float(left), float(top), float(right), float(bottom))

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(self.left), int(self.top), int(self.right), int(self.bottom))

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        """Rapport largeur/hauteur. Une plaque europeenne avoisine 4,7."""
        return self.width / self.height if self.height else 0.0

    def shift(self, dx: float, dy: float) -> "BoundingBox":
        """Translate la boite — sert a passer du repere du crop a l'image entiere."""
        return BoundingBox(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    def expand(self, ratio: float) -> "BoundingBox":
        """Elargit la boite d'une fraction de sa taille.

        Un detecteur cadre souvent la plaque au plus juste ; l'OCR lit mieux
        avec quelques pixels de marge autour des caracteres.
        """
        dx = self.width * ratio / 2.0
        dy = self.height * ratio / 2.0
        return BoundingBox(self.left - dx, self.top - dy, self.right + dx, self.bottom + dy)

    def clip(self, width: float, height: float) -> "BoundingBox":
        """Ramene la boite dans les limites d'une image."""
        return BoundingBox(
            max(0.0, min(self.left, width)),
            max(0.0, min(self.top, height)),
            max(0.0, min(self.right, width)),
            max(0.0, min(self.bottom, height)),
        )

    def crop(self, image):
        """Extrait la region correspondante d'une image numpy."""
        height, width = image.shape[:2]
        left, top, right, bottom = self.clip(width, height).as_int()
        return image[top:bottom, left:right]

    def iou(self, other: "BoundingBox") -> float:
        """Intersection sur union, entre 0 et 1."""
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0

    def contains_centre_of(self, other: "BoundingBox") -> bool:
        """Vrai si le centre de `other` tombe dans cette boite.

        Sert a rattacher une plaque a un vehicule : un test de centre est plus
        tolerant qu'un test d'inclusion stricte, la boite de plaque debordant
        souvent legerement celle du vehicule.
        """
        x, y = other.centre
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(frozen=True)
class VehicleDetection:
    """Un vehicule detecte, avec son type."""

    box: BoundingBox
    vehicle_type: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def label(self) -> str:
        return VEHICLE_CLASSES.get(self.vehicle_type, self.vehicle_type)


@dataclass(frozen=True)
class PlateDetection:
    """Une plaque localisee, en coordonnees absolues de l'image."""

    box: BoundingBox
    confidence: float


@dataclass(frozen=True)
class PlateReading:
    """Le texte lu sur une plaque.

    `raw` conserve la sortie brute de l'OCR et `text` la version normalisee.
    Garder les deux permet de diagnostiquer si une erreur vient de la lecture
    ou de la normalisation, distinction impossible si l'on ecrase la sortie
    brute.
    """

    raw: str
    text: str
    confidence: float
    country_format: Optional[str] = None

    @property
    def is_valid_format(self) -> bool:
        return self.country_format is not None


@dataclass
class VehicleRecord:
    """Ce que le pipeline produit pour un vehicule : le resultat complet."""

    vehicle: VehicleDetection
    plate: Optional[PlateDetection] = None
    reading: Optional[PlateReading] = None
    plate_hash: Optional[str] = None

    def as_dict(self, include_plaintext: bool = True) -> dict:
        """Serialise en JSON.

        `include_plaintext=False` omet la plaque en clair et ne conserve que son
        empreinte : c'est le mode a utiliser pour tout stockage durable.
        """
        payload: dict = {
            "vehicle": {
                "type": self.vehicle.vehicle_type,
                "label": self.vehicle.label,
                "confidence": round(self.vehicle.confidence, 3),
                "box": [round(v, 1) for v in self.vehicle.box.as_xyxy()],
            },
            "track_id": self.vehicle.track_id,
            "plate": None,
        }

        if self.plate is not None:
            plate_payload: dict = {
                "box": [round(v, 1) for v in self.plate.box.as_xyxy()],
                "detection_confidence": round(self.plate.confidence, 3),
            }
            if self.reading is not None:
                if include_plaintext:
                    plate_payload["text"] = self.reading.text
                    plate_payload["raw"] = self.reading.raw
                plate_payload["ocr_confidence"] = round(self.reading.confidence, 3)
                plate_payload["format"] = self.reading.country_format
            if self.plate_hash is not None:
                plate_payload["hash"] = self.plate_hash
            payload["plate"] = plate_payload

        return payload


@dataclass
class FrameResult:
    """Resultat du pipeline sur une frame."""

    frame_index: int
    records: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)

    def as_dict(self, include_plaintext: bool = True) -> dict:
        return {
            "frame": self.frame_index,
            "vehicles_detected": len(self.records),
            "plates_read": sum(1 for r in self.records if r.reading is not None),
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "results": [r.as_dict(include_plaintext) for r in self.records],
        }
