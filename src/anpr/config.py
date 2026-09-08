"""Configuration du pipeline.

Tout ce qui peut varier d'un deploiement a l'autre — chemins de modeles,
seuils, pays — est declare dans `configs/default.yaml` plutot que dans le code.
Le pays notamment n'est pas un detail : il determine les formats de plaques
acceptes, donc le resultat de la normalisation.

Les valeurs peuvent etre surchargees par des variables d'environnement
prefixees `ANPR_`, ce qui est la maniere habituelle de configurer un conteneur
sans reconstruire son image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

ENV_PREFIX = "ANPR_"


@dataclass
class AppConfig:
    """Configuration complete de l'application."""

    vehicle_weights: str = "yolov8n.pt"
    plate_weights: str = "data/models/plate_detector.pt"
    vehicle_confidence: float = 0.35
    plate_confidence: float = 0.25
    countries: tuple = ("FR",)
    read_plates: bool = True
    hash_plates: bool = False
    ocr_variants: bool = True
    consensus_readings: int = 3
    anonymise_faces: bool = False
    extras: dict = field(default_factory=dict)

    def to_pipeline_config(self):
        """Convertit en configuration de pipeline."""
        from .pipeline import PipelineConfig

        return PipelineConfig(
            vehicle_weights=self.resolve(self.vehicle_weights),
            plate_weights=self.resolve(self.plate_weights),
            vehicle_confidence=self.vehicle_confidence,
            plate_confidence=self.plate_confidence,
            countries=tuple(self.countries),
            read_plates=self.read_plates,
            hash_plates=self.hash_plates,
            ocr_variants=self.ocr_variants,
            consensus_readings=self.consensus_readings,
        )

    @staticmethod
    def resolve(value: str) -> str:
        """Resout un chemin relatif depuis la racine du projet.

        Les poids YOLO officiels (`yolov8n.pt`) ne sont pas des chemins : ils
        sont telecharges par Ultralytics et doivent rester tels quels.
        """
        if value.startswith("yolo") and "/" not in value and "\\" not in value:
            return value
        path = Path(value)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _coerce(value: str):
    """Convertit une valeur d'environnement vers son type probable."""
    lowered = value.lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return value


def load_config(path=None) -> AppConfig:
    """Charge la configuration, puis applique les surcharges d'environnement."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    raw: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    fields = {f for f in AppConfig.__dataclass_fields__ if f != "extras"}
    known = {k: v for k, v in raw.items() if k in fields}
    extras = {k: v for k, v in raw.items() if k not in fields}

    for name in fields:
        env_value = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
        if env_value is not None:
            known[name] = _coerce(env_value)

    if "countries" in known and isinstance(known["countries"], str):
        known["countries"] = (known["countries"],)
    if "countries" in known:
        known["countries"] = tuple(known["countries"])

    return AppConfig(**known, extras=extras)
