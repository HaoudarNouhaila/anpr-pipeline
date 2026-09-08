"""Detection et classification des vehicules.

Premier etage de la cascade. Aucun entrainement n'est necessaire ici : les
quatre classes utiles — voiture, camion, bus, moto — figurent deja dans COCO,
sur lequel les modeles YOLO sont pre-entraines. C'est ce qui distingue cet
etage du suivant, ou la plaque n'existe dans aucun jeu de classes standard et
impose un fine-tuning.

Le suivi (ByteTrack) est active des l'etage vehicule, et non a l'etage plaque.
La raison est pratique : un vehicule est grand, contraste et present sur de
nombreuses frames, donc facile a suivre ; une plaque est minuscule et
disparait des que l'angle change. Suivre le vehicule et rattacher la plaque a
sa piste donne une identite bien plus stable.
"""

from __future__ import annotations

from ..types import VEHICLE_CLASSES, BoundingBox, VehicleDetection


class VehicleDetector:
    """Localise les vehicules et leur attribue un type.

    Parameters
    ----------
    weights:
        Poids YOLO pre-entraines sur COCO. Telecharges au premier usage.
    confidence:
        Seuil de confiance minimal.
    classes:
        Noms COCO a retenir. Par defaut les quatre classes de vehicules.
    """

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.35, classes=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence
        self.wanted = set(classes or VEHICLE_CLASSES)
        self.class_ids = {
            index for index, name in self.model.names.items() if name in self.wanted
        }

        if not self.class_ids:
            raise ValueError(
                f"Le modele {weights} ne connait aucune des classes {sorted(self.wanted)}. "
                "Utilisez un modele entraine sur COCO."
            )

    def detect(self, frame, track: bool = False) -> list:
        """Detecte les vehicules d'une frame.

        `track=True` maintient une identite entre frames, indispensable pour le
        vote de consensus sur video. En mode image isolee c'est inutile et
        couteux, d'ou le drapeau.
        """
        if track:
            result = self.model.track(
                frame, persist=True, classes=list(self.class_ids), conf=self.confidence, verbose=False
            )[0]
        else:
            result = self.model.predict(
                frame, classes=list(self.class_ids), conf=self.confidence, verbose=False
            )[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.tolist()[0])
            name = self.model.names[class_id]
            if name not in self.wanted:
                continue

            # ByteTrack n'attribue un identifiant qu'une fois la piste confirmee.
            track_id = int(box.id.tolist()[0]) if (track and box.id is not None) else None

            detections.append(
                VehicleDetection(
                    box=BoundingBox.from_xyxy(box.xyxy.tolist()[0]),
                    vehicle_type=name,
                    confidence=float(box.conf.tolist()[0]),
                    track_id=track_id,
                )
            )

        return detections
