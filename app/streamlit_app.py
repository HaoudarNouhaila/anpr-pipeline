"""Interface de demonstration du pipeline ANPR.

Une page unique : on depose une image, le pipeline la traite, et l'on voit
cote a cote l'image annotee, les plaques lues et le JSON qui serait renvoye a
un systeme tiers. Les reglages sensibles — seuils, pays, anonymisation — sont
exposes dans la barre laterale pour que l'effet de chacun soit visible
immediatement.

Lancement
---------
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anpr.config import load_config  # noqa: E402
from anpr.pipeline import ANPRPipeline  # noqa: E402

VEHICLE_COLOUR = (60, 180, 255)
PLATE_COLOUR = (80, 220, 120)


@st.cache_resource(show_spinner="Chargement des modeles...")
def build_pipeline(plate_confidence: float, vehicle_confidence: float, countries: tuple, variants: bool):
    """Instancie le pipeline. Mis en cache : le chargement coute plusieurs secondes."""
    config = load_config()
    pipeline_config = config.to_pipeline_config()
    pipeline_config.plate_confidence = plate_confidence
    pipeline_config.vehicle_confidence = vehicle_confidence
    pipeline_config.countries = countries
    pipeline_config.ocr_variants = variants
    return ANPRPipeline(pipeline_config)


def annotate(image, result):
    """Dessine les vehicules, les plaques et le texte lu."""
    output = image.copy()

    for record in result.records:
        left, top, right, bottom = record.vehicle.box.as_int()
        cv2.rectangle(output, (left, top), (right, bottom), VEHICLE_COLOUR, 2)

        label = f"{record.vehicle.label} {record.vehicle.confidence:.0%}"
        cv2.putText(output, label, (left, max(20, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VEHICLE_COLOUR, 2, cv2.LINE_AA)

        if record.plate is None:
            continue

        pl, pt, pr, pb = record.plate.box.as_int()
        cv2.rectangle(output, (pl, pt), (pr, pb), PLATE_COLOUR, 2)

        if record.reading is not None:
            text = record.reading.text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (pl, pb + 2), (pl + tw + 8, pb + th + 12), PLATE_COLOUR, cv2.FILLED)
            cv2.putText(output, text, (pl + 4, pb + th + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    return output


def main() -> None:
    st.set_page_config(page_title="ANPR Pipeline", layout="wide")
    st.title("Reconnaissance de plaques d'immatriculation")
    st.caption(
        "Cascade : detection du vehicule et de son type, puis recherche de la plaque "
        "dans cette region seulement, puis lecture OCR."
    )

    with st.sidebar:
        st.header("Reglages")
        vehicle_confidence = st.slider("Confiance vehicule", 0.05, 0.9, 0.35, 0.05)
        plate_confidence = st.slider("Confiance plaque", 0.05, 0.9, 0.25, 0.05)
        countries = st.multiselect(
            "Formats de plaques acceptes",
            ["FR", "ES", "IT", "MA"],
            default=["FR"],
            help="Plusieurs pays partagent une meme longueur de plaque. Declarer "
                 "le pays evite qu'un format etranger l'emporte a la normalisation.",
        )
        variants = st.checkbox(
            "Lire plusieurs pretraitements", value=True,
            help="Plus fiable, mais multiplie le cout de l'etage OCR.",
        )
        anonymise = st.checkbox(
            "Flouter visages et plaques", value=False,
            help="Produit une version publiable de l'image.",
        )

    uploaded = st.file_uploader("Deposez une image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is None:
        st.info("Deposez une photo de vehicule pour lancer l'analyse.")
        return

    image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("Image illisible.")
        return

    try:
        pipeline = build_pipeline(plate_confidence, vehicle_confidence, tuple(countries or ["FR"]), variants)
    except Exception as error:  # noqa: BLE001 — l'usager doit voir la cause
        st.error(f"Chargement impossible : {error}")
        st.info("Le detecteur de plaques doit etre entraine : voir docs/DATA.md.")
        return

    with st.spinner("Analyse en cours..."):
        result = pipeline.process_frame(image)
        annotated = annotate(image, result)

        if anonymise:
            from anpr.privacy import FaceAnonymiser

            plate_boxes = [r.plate.box for r in result.records if r.plate is not None]
            annotated = FaceAnonymiser().anonymise(annotated, extra_boxes=plate_boxes)

    left, right = st.columns([3, 2])

    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    with right:
        read = [r for r in result.records if r.reading is not None]

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Vehicules", len(result.records))
        col_b.metric("Plaques lues", len(read))
        col_c.metric("Latence", f"{sum(result.timings_ms.values()):.0f} ms")

        if read:
            st.subheader("Plaques")
            for record in read:
                st.markdown(
                    f"**{record.reading.text}** — {record.vehicle.label}  \n"
                    f"format `{record.reading.country_format or 'inconnu'}` · "
                    f"OCR {record.reading.confidence:.0%} · "
                    f"lecture brute `{record.reading.raw}`"
                )
        elif result.records:
            st.warning("Vehicules detectes, mais aucune plaque lisible.")
        else:
            st.warning("Aucun vehicule detecte. Essayez d'abaisser le seuil de confiance.")

        st.subheader("Repartition du temps")
        for stage, value in result.timings_ms.items():
            st.text(f"{stage:<20} {value:>8.1f} ms")

    st.subheader("Sortie JSON")
    st.code(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()
