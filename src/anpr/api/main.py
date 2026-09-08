"""API HTTP du pipeline ANPR.

Trois endpoints : un controle de sante, une inference sur image, une inference
sur video courte. La reponse est le JSON structure decrit dans le cahier des
charges — pour chaque vehicule : son type, sa plaque, les scores de confiance.

Deux choix a signaler.

Le pipeline est charge **une seule fois** au demarrage, pas a chaque requete.
Charger YOLO et EasyOCR coute plusieurs secondes ; le faire par requete
rendrait l'API inutilisable. C'est le role du gestionnaire de cycle de vie.

Le mode `store` renvoie l'empreinte HMAC de la plaque au lieu du texte en
clair. Une API qui journalise des plaques constitue un fichier de donnees
personnelles ; laisser le client choisir explicitement ce qu'il recoit est
plus sain que de toujours tout renvoyer.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from ..config import load_config
from ..pipeline import ANPRPipeline

# Limite de taille des envois. Sans plafond, une video de plusieurs centaines
# de megaoctets saturerait la memoire du serveur.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_VIDEO_FRAMES = 300

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modeles au demarrage et les libere a l'arret."""
    config = load_config()
    state["pipeline"] = ANPRPipeline(config.to_pipeline_config())
    state["config"] = config
    yield
    state.clear()


app = FastAPI(
    title="ANPR Pipeline",
    version="0.1.0",
    description="Detection de vehicules, lecture de plaques et sortie JSON structuree.",
    lifespan=lifespan,
)


def _pipeline() -> ANPRPipeline:
    pipeline = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non initialise.")
    return pipeline


async def _read_upload(upload: UploadFile) -> bytes:
    payload = await upload.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux ({len(payload) // 1024 // 1024} Mo, "
                   f"maximum {MAX_UPLOAD_BYTES // 1024 // 1024} Mo).",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="Fichier vide.")
    return payload


def _decode_image(payload: bytes):
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image illisible ou format non supporte.")
    return image


# ----------------------------------------------------------------------
@app.get("/health", tags=["service"])
def health() -> dict:
    """Etat du service et configuration active."""
    config = state.get("config")
    return {
        "status": "ok" if state.get("pipeline") else "loading",
        "countries": list(config.countries) if config else [],
        "plate_weights": config.plate_weights if config else None,
    }


@app.post("/v1/analyse/image", tags=["inference"])
async def analyse_image(
    file: UploadFile = File(..., description="Image JPEG ou PNG"),
    store: bool = Query(
        False,
        description="Renvoyer l'empreinte HMAC de la plaque au lieu du texte en clair. "
                    "A utiliser pour tout stockage durable.",
    ),
) -> dict:
    """Analyse une image et renvoie les vehicules et plaques detectes."""
    payload = await _read_upload(file)
    image = _decode_image(payload)

    result = _pipeline().process_frame(image, frame_index=0, track=False)
    return result.as_dict(include_plaintext=not store)


@app.post("/v1/analyse/video", tags=["inference"])
async def analyse_video(
    file: UploadFile = File(..., description="Video MP4 ou AVI"),
    max_frames: int = Query(120, ge=1, le=MAX_VIDEO_FRAMES),
    store: bool = Query(False),
) -> dict:
    """Analyse une video courte et renvoie le consensus par vehicule suivi.

    La reponse ne liste pas chaque frame mais le verdict par piste : c'est ce
    qui interesse un exploitant, et cela evite de renvoyer la meme plaque
    autant de fois qu'elle est visible.
    """
    import tempfile
    from pathlib import Path

    payload = await _read_upload(file)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)

    try:
        capture = cv2.VideoCapture(str(temporary))
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Video illisible.")

        frames = []
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()

        if not frames:
            raise HTTPException(status_code=400, detail="Aucune frame lisible dans la video.")

        pipeline = _pipeline()
        results = pipeline.process_video_frames(frames)

        return {
            "frames_analysed": len(frames),
            "vehicles_tracked": len(pipeline.voter.tracks),
            "plates": pipeline.consensus_summary() if not store else [
                {k: v for k, v in row.items() if k != "plate"}
                for row in pipeline.consensus_summary()
            ],
            "timings_ms": {
                stage: round(sum(r.timings_ms.get(stage, 0) for r in results) / len(results), 2)
                for stage in ("vehicle_detection", "plate_detection", "ocr")
            },
        }
    finally:
        temporary.unlink(missing_ok=True)
