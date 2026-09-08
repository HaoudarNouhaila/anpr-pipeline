"""Tests de l'API HTTP.

Le pipeline est remplace par un double : charger YOLO et EasyOCR prendrait
plusieurs secondes par test et ferait dependre la suite de la presence des
poids entraines. Ce qui est teste ici, ce sont les responsabilites de l'API —
decodage des envois, codes d'erreur, forme de la reponse, et surtout le mode
`store` qui doit retirer la plaque en clair.
"""

import io

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("httpx", reason="requis par le client de test de Starlette")

from fastapi.testclient import TestClient  # noqa: E402

from anpr.types import (  # noqa: E402
    BoundingBox,
    FrameResult,
    PlateDetection,
    PlateReading,
    VehicleDetection,
    VehicleRecord,
)


def sample_record():
    """Un vehicule avec plaque lue et empreinte, comme le pipeline en produit."""
    return VehicleRecord(
        vehicle=VehicleDetection(BoundingBox(10, 20, 210, 170), "car", 0.91, track_id=1),
        plate=PlateDetection(BoundingBox(80, 130, 160, 152), 0.78),
        reading=PlateReading(raw="A8123CD", text="AB123CD", confidence=0.64, country_format="FR-SIV"),
        plate_hash="deadbeefdeadbeef",
    )


class StubPipeline:
    """Double du pipeline : meme interface, aucun modele charge."""

    def __init__(self):
        self.voter = type("Voter", (), {"tracks": {1: None}})()
        self.calls = 0

    def process_frame(self, frame, frame_index=0, track=False):
        self.calls += 1
        return FrameResult(
            frame_index=frame_index,
            records=[sample_record()],
            timings_ms={"vehicle_detection": 12.0, "plate_detection": 5.0, "ocr": 40.0},
        )

    def process_video_frames(self, frames, on_result=None):
        return [self.process_frame(f, i) for i, f in enumerate(frames)]

    def consensus_summary(self):
        return [{"track_id": 1, "plate": "AB123CD", "format": "FR-SIV",
                 "confidence": 0.64, "readings": 4, "agreement": 1.0}]


@pytest.fixture
def client(monkeypatch):
    from anpr.api import main

    monkeypatch.setattr(main, "ANPRPipeline", lambda *a, **k: StubPipeline())
    with TestClient(main.app) as test_client:
        yield test_client


def encode_image(width=320, height=240) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


# --- Sante -------------------------------------------------------------------
def test_health_reports_the_active_configuration(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "FR" in body["countries"]


# --- Analyse d'image ---------------------------------------------------------
def test_image_analysis_returns_the_structured_payload(client):
    response = client.post(
        "/v1/analyse/image", files={"file": ("car.jpg", encode_image(), "image/jpeg")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["vehicles_detected"] == 1
    assert body["plates_read"] == 1

    plate = body["results"][0]["plate"]
    assert plate["text"] == "AB123CD"
    assert plate["raw"] == "A8123CD"  # la lecture brute est conservee
    assert plate["format"] == "FR-SIV"
    assert body["results"][0]["vehicle"]["label"] == "voiture"


def test_store_mode_removes_the_plaintext_plate(client):
    """Mode a utiliser pour tout stockage durable : seule l'empreinte sort."""
    response = client.post(
        "/v1/analyse/image?store=true",
        files={"file": ("car.jpg", encode_image(), "image/jpeg")},
    )
    body = response.json()
    plate = body["results"][0]["plate"]

    assert "text" not in plate
    assert "raw" not in plate
    assert plate["hash"] == "deadbeefdeadbeef"
    assert "AB123CD" not in response.text


def test_timings_are_reported_per_stage(client):
    body = client.post(
        "/v1/analyse/image", files={"file": ("car.jpg", encode_image(), "image/jpeg")}
    ).json()
    assert set(body["timings_ms"]) == {"vehicle_detection", "plate_detection", "ocr"}


# --- Erreurs -----------------------------------------------------------------
def test_a_non_image_payload_is_rejected(client):
    response = client.post(
        "/v1/analyse/image", files={"file": ("notes.txt", b"ceci n'est pas une image", "text/plain")}
    )
    assert response.status_code == 400
    assert "illisible" in response.json()["detail"].lower()


def test_an_empty_file_is_rejected(client):
    response = client.post(
        "/v1/analyse/image", files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400


def test_an_oversized_upload_is_rejected(client):
    from anpr.api.main import MAX_UPLOAD_BYTES

    payload = b"\xff" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/v1/analyse/image", files={"file": ("huge.jpg", io.BytesIO(payload), "image/jpeg")}
    )
    assert response.status_code == 413


def test_missing_file_is_a_validation_error(client):
    assert client.post("/v1/analyse/image").status_code == 422


def test_video_frame_limit_is_enforced(client):
    """Sans plafond, une longue video saturerait la memoire du serveur."""
    response = client.post(
        "/v1/analyse/video?max_frames=9999",
        files={"file": ("clip.mp4", b"\x00" * 100, "video/mp4")},
    )
    assert response.status_code == 422
