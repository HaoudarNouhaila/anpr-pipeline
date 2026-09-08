"""Lecture du texte d'une plaque.

EasyOCR est retenu plutot que Tesseract. Tesseract est concu pour du document
imprime : il segmente en lignes et en mots, suppose une police reguliere et un
fond uniforme. Une plaque est un cas contraire — texte tres court, police
stylisee, fond colore, prise de vue oblique. EasyOCR, base sur un reseau de
detection puis de reconnaissance, se comporte nettement mieux sur ce type
d'entree.

Le module lit plusieurs variantes de pretraitement de la meme plaque et retient
la plus sure, puis normalise le resultat selon le format du pays configure.
"""

from __future__ import annotations

from ..types import PlateReading
from . import preprocess
from .normalise import formats_for, is_plausible, normalise

# Liste blanche des caracteres possibles. La restreindre supprime d'un coup une
# grande partie des confusions : sans elle, l'OCR peut rendre des caracteres
# accentues ou de la ponctuation qu'aucune plaque ne porte.
ALLOWED_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class PlateReader:
    """Lit le texte d'une image de plaque.

    Parameters
    ----------
    countries:
        Codes pays dont les formats sont acceptes, par exemple `("FR",)`. Sans
        restriction, un format etranger de meme longueur peut l'emporter — voir
        `normalise.formats_for`.
    languages:
        Langues passees a EasyOCR. L'anglais suffit pour des caracteres latins.
    use_variants:
        Lire plusieurs pretraitements et garder le meilleur. Plus fiable, mais
        multiplie le cout par le nombre de variantes.
    """

    def __init__(self, countries=None, languages=None, use_variants: bool = True, gpu: bool = False):
        import easyocr

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
        self.formats = formats_for(*countries) if countries else None
        self.use_variants = use_variants

    # ------------------------------------------------------------------
    def read(self, plate_image) -> PlateReading | None:
        """Lit une plaque deja rognee. Renvoie None si rien de plausible."""
        if plate_image is None or plate_image.size == 0:
            return None

        images = (
            preprocess.variants(plate_image)
            if self.use_variants
            else [preprocess.prepare(plate_image)]
        )

        best_text, best_confidence = None, 0.0

        for image in images:
            text, confidence = self._read_single(image)
            if text and confidence > best_confidence:
                best_text, best_confidence = text, confidence

        if not best_text or not is_plausible(best_text):
            return None

        normalised, country_format = normalise(best_text, formats=self.formats)

        return PlateReading(
            raw=best_text,
            text=normalised,
            confidence=best_confidence,
            country_format=country_format,
        )

    # ------------------------------------------------------------------
    def _read_single(self, image) -> tuple[str, float]:
        """Lit une image preparee et concatene les fragments trouves.

        EasyOCR renvoie une boite par zone de texte. Une plaque en comporte
        souvent plusieurs — le pays a gauche, l'immatriculation, parfois le
        departement a droite. On les concatene dans l'ordre de lecture, de
        gauche a droite, et la confiance retenue est la plus faible du lot :
        une plaque n'est lue correctement que si tous ses fragments le sont.
        """
        results = self.reader.readtext(image, allowlist=ALLOWED_CHARACTERS, detail=1)

        if not results:
            return "", 0.0

        # Tri par abscisse du coin superieur gauche de chaque boite.
        results.sort(key=lambda item: item[0][0][0])

        fragments = [text for _, text, _ in results if text.strip()]
        confidences = [conf for _, text, conf in results if text.strip()]

        if not fragments:
            return "", 0.0

        return "".join(fragments), float(min(confidences))
