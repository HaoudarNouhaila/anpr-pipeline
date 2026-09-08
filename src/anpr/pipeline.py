"""Orchestration de la cascade ANPR.

Ce module n'implemente aucune logique metier : il enchaine les etages et
mesure le temps de chacun. La geometrie est dans `types`, la normalisation dans
`ocr.normalise`, le vote dans `tracking.consensus`. Cette separation permet de
tester toute la logique sans jamais charger un modele.

Le chronometrage par etage n'est pas une commodite de developpement : il
alimente le benchmark de latence, et il repond a la question qu'un exploitant
pose en premier — ou passe le temps, et que gagnerait-on a optimiser tel etage.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .detection.plates import PlateDetector
from .detection.vehicles import VehicleDetector
from .privacy import hash_plate
from .tracking import ConsensusVoter
from .types import FrameResult, VehicleRecord

# Marge ajoutee autour de la plaque avant lecture : l'OCR segmente mal des
# caracteres colles au bord de l'image.
PLATE_CROP_MARGIN = 0.15


@dataclass
class PipelineConfig:
    """Parametres de la cascade."""

    vehicle_weights: str = "yolov8n.pt"
    plate_weights: str = "data/models/plate_detector.pt"
    vehicle_confidence: float = 0.35
    plate_confidence: float = 0.25
    countries: tuple = ("FR",)
    read_plates: bool = True
    hash_plates: bool = False
    hash_key: str | None = None
    ocr_variants: bool = True
    consensus_readings: int = 3


@dataclass
class Timer:
    """Accumulateur de durees par etage, en millisecondes."""

    stages: dict = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.stages[stage] = self.stages.get(stage, 0.0) + elapsed

    def reset(self) -> None:
        self.stages = {}


class ANPRPipeline:
    """Cascade complete : vehicule -> plaque -> lecture.

    Le lecteur OCR est instancie paresseusement. EasyOCR telecharge et charge
    ses modeles au premier appel, ce qui coute plusieurs secondes : un usage
    qui ne veut que compter des vehicules ne doit pas le payer.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        self.vehicle_detector = VehicleDetector(
            self.config.vehicle_weights, confidence=self.config.vehicle_confidence
        )
        self.plate_detector = PlateDetector(
            self.config.plate_weights, confidence=self.config.plate_confidence
        )
        self.voter = ConsensusVoter(min_readings=self.config.consensus_readings)

        self._reader = None

    @property
    def reader(self):
        """Instancie le lecteur OCR au premier besoin."""
        if self._reader is None:
            from .ocr.reader import PlateReader

            self._reader = PlateReader(
                countries=self.config.countries, use_variants=self.config.ocr_variants
            )
        return self._reader

    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_index: int = 0, track: bool = False) -> FrameResult:
        """Traite une frame et renvoie un resultat structure."""
        timer = Timer()
        records = []

        with timer.measure("vehicle_detection"):
            vehicles = self.vehicle_detector.detect(frame, track=track)

        for vehicle in vehicles:
            record = VehicleRecord(vehicle=vehicle)

            with timer.measure("plate_detection"):
                plate = self.plate_detector.detect_in_vehicle(frame, vehicle)

            if plate is not None:
                record.plate = plate

                if self.config.read_plates:
                    with timer.measure("ocr"):
                        reading = self._read_plate(frame, plate)

                    if reading is not None:
                        record.reading = reading

                        if vehicle.track_id is not None:
                            self.voter.add(vehicle.track_id, reading)

                        if self.config.hash_plates:
                            record.plate_hash = hash_plate(reading.text, self.config.hash_key)

            records.append(record)

        return FrameResult(frame_index=frame_index, records=records, timings_ms=timer.stages)

    def _read_plate(self, frame, plate):
        """Rogne la plaque avec un peu de marge et la lit."""
        height, width = frame.shape[:2]
        region = plate.box.expand(PLATE_CROP_MARGIN).clip(width, height)
        crop = region.crop(frame)
        return self.reader.read(crop)

    # ------------------------------------------------------------------
    def process_video_frames(self, frames, on_result=None) -> list:
        """Traite une sequence en maintenant le suivi et le vote de consensus.

        Chaque plaque n'est emise qu'une fois, apres accumulation d'assez de
        lectures concordantes — c'est ce que garantit le voteur.
        """
        results = []

        for index, frame in enumerate(frames):
            result = self.process_frame(frame, frame_index=index, track=True)
            results.append(result)

            if on_result is not None:
                on_result(result)

        return results

    def consensus_summary(self) -> list:
        """Plaques retenues par piste, apres vote sur toute la sequence."""
        return self.voter.summary()
