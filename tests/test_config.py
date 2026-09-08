"""Tests du chargement de configuration."""

import pytest

from anpr.config import AppConfig, _coerce, load_config


# --- Conversion des valeurs d'environnement ----------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True), ("TRUE", True), ("yes", True), ("1", True),
        ("false", False), ("no", False), ("0", False),
        ("42", 42), ("0.35", 0.35), ("FR", "FR"),
    ],
)
def test_environment_values_are_coerced_to_their_type(raw, expected):
    """Une variable d'environnement est toujours une chaine : il faut la typer."""
    assert _coerce(raw) == expected


def test_a_comma_separated_value_becomes_a_tuple():
    assert _coerce("FR,ES,IT") == ("FR", "ES", "IT")


def test_whitespace_around_items_is_stripped():
    assert _coerce("FR, ES , IT") == ("FR", "ES", "IT")


# --- Chargement --------------------------------------------------------------
def test_defaults_are_used_when_no_file_exists(tmp_path):
    config = load_config(tmp_path / "absent.yaml")
    assert config.vehicle_weights == "yolov8n.pt"
    assert config.countries == ("FR",)


def test_yaml_values_override_the_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "vehicle_confidence: 0.5\ncountries:\n  - ES\n  - IT\n", encoding="utf-8"
    )

    config = load_config(path)
    assert config.vehicle_confidence == 0.5
    assert config.countries == ("ES", "IT")


def test_environment_overrides_the_yaml(monkeypatch, tmp_path):
    """Ordre de priorite : environnement au-dessus du fichier.

    C'est ce qui permet de configurer un conteneur sans reconstruire son image.
    """
    path = tmp_path / "config.yaml"
    path.write_text("plate_confidence: 0.25\n", encoding="utf-8")

    monkeypatch.setenv("ANPR_PLATE_CONFIDENCE", "0.6")
    assert load_config(path).plate_confidence == 0.6


def test_a_single_country_string_becomes_a_tuple(monkeypatch, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("countries: MA\n", encoding="utf-8")
    assert load_config(path).countries == ("MA",)


def test_unknown_keys_are_kept_aside_not_dropped(tmp_path):
    """Une cle inconnue ne doit ni faire planter le chargement, ni disparaitre."""
    path = tmp_path / "config.yaml"
    path.write_text("vehicle_confidence: 0.4\nfuture_option: 123\n", encoding="utf-8")

    config = load_config(path)
    assert config.vehicle_confidence == 0.4
    assert config.extras["future_option"] == 123


# --- Resolution des chemins --------------------------------------------------
def test_official_yolo_weights_are_not_treated_as_paths():
    """`yolov8n.pt` est telecharge par Ultralytics : le transformer en chemin
    absolu du projet le rendrait introuvable."""
    assert AppConfig.resolve("yolov8n.pt") == "yolov8n.pt"


def test_a_relative_path_is_resolved_from_the_project_root():
    resolved = AppConfig.resolve("data/models/plate_detector.pt")
    assert resolved.endswith("plate_detector.pt")
    assert "anpr-pipeline" in resolved.replace("\\", "/")


def test_an_absolute_path_is_left_alone(tmp_path):
    absolute = str(tmp_path / "weights.pt")
    assert AppConfig.resolve(absolute) == absolute


def test_conversion_to_pipeline_config_resolves_paths():
    pipeline_config = AppConfig().to_pipeline_config()
    assert pipeline_config.vehicle_weights == "yolov8n.pt"
    assert pipeline_config.plate_weights.endswith("plate_detector.pt")
