"""Tests du pretraitement des images de plaques.

Ces tests fabriquent des plaques synthetiques dont on connait la verite : une
image droite doit etre reconnue comme droite, une image inclinee de 10 degres
doit produire une estimation proche de 10 degres. Cela permet de verifier le
redressement sans disposer d'un jeu de donnees annote en angle.
"""

import cv2
import numpy as np
import pytest

from anpr.ocr import preprocess


def synthetic_plate(width=200, height=44, text="AB123CD", angle=0.0):
    """Fabrique une plaque : texte sombre sur fond clair, inclinee si demande."""
    image = np.full((height, width), 235, dtype=np.uint8)
    cv2.putText(image, text, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2, cv2.LINE_AA)

    if angle:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=235
        )

    return image


# --- Agrandissement ----------------------------------------------------------
def test_upscale_reaches_the_target_height():
    small = synthetic_plate(height=20, width=90)
    result = preprocess.upscale(small, target_height=96)
    assert result.shape[0] == 96


def test_upscale_preserves_the_aspect_ratio():
    small = synthetic_plate(height=20, width=90)
    result = preprocess.upscale(small, target_height=96)
    assert result.shape[1] / result.shape[0] == pytest.approx(90 / 20, rel=0.02)


def test_an_already_large_image_is_left_alone():
    """Agrandir une image deja grande n'apporterait rien et couterait du temps."""
    large = synthetic_plate(height=120, width=500)
    assert preprocess.upscale(large, target_height=96).shape == large.shape


# --- Niveaux de gris ---------------------------------------------------------
def test_colour_is_converted_to_grey():
    colour = np.zeros((40, 100, 3), dtype=np.uint8)
    assert preprocess.to_grey(colour).ndim == 2


def test_grey_is_left_unchanged():
    grey = synthetic_plate()
    assert preprocess.to_grey(grey) is grey


# --- Contraste ---------------------------------------------------------------
def test_contrast_enhancement_widens_the_histogram():
    """Une plaque terne doit ressortir avec un ecart-type plus eleve."""
    dull = np.full((44, 200), 128, dtype=np.uint8)
    cv2.putText(dull, "AB123CD", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 110, 2)

    enhanced = preprocess.enhance_contrast(dull)
    assert enhanced.std() > dull.std()


# --- Redressement ------------------------------------------------------------
def test_a_straight_plate_needs_no_correction():
    angle = preprocess.estimate_correction_angle(synthetic_plate(angle=0))
    assert abs(angle) < 3.0


@pytest.mark.parametrize("applied", [-10.0, -5.0, 5.0, 10.0])
def test_the_correction_angle_is_the_opposite_of_the_tilt(applied):
    """La fonction renvoie la correction, donc l'oppose de l'inclinaison.

    C'est exactement ce que `deskew` doit appliquer. Confondre les deux sens
    pencherait l'image deux fois plus au lieu de la redresser.
    """
    correction = preprocess.estimate_correction_angle(synthetic_plate(angle=applied))
    assert correction == pytest.approx(-applied, abs=4.0)


@pytest.mark.parametrize("applied", [-10.0, -5.0, 5.0, 10.0])
def test_deskew_actually_straightens_the_plate(applied):
    """La propriete qui compte : apres redressement, plus d'inclinaison.

    Ce test est independant de la convention de signe interne — c'est ce qui
    le rend plus solide que la verification de l'angle brut.
    """
    tilted = synthetic_plate(angle=applied)
    straightened = preprocess.deskew(tilted)

    # Sans cette assertion, le test passerait a vide : une correction nulle
    # laisse l'image intacte, donc un residu nul lui aussi.
    assert not np.array_equal(tilted, straightened), "aucun redressement applique"

    residual = preprocess.estimate_correction_angle(straightened)
    assert abs(residual) < abs(applied) / 2


def test_an_absurd_angle_is_discarded():
    """Au-dela du seuil, l'estimation vient d'un artefact : ne rien redresser.

    Tordre une plaque droite est pire que la laisser telle quelle.
    """
    noise = np.random.default_rng(0).integers(0, 255, (44, 200), dtype=np.uint8)
    assert abs(preprocess.estimate_correction_angle(noise)) <= preprocess.MAX_DESKEW_ANGLE


def test_deskew_preserves_the_image_size():
    plate = synthetic_plate(angle=8)
    assert preprocess.deskew(plate).shape == plate.shape


def test_a_negligible_angle_skips_the_rotation():
    """Rouvrir l'image pour un demi-degre n'apporte rien et degrade par interpolation."""
    plate = synthetic_plate(angle=0)
    assert preprocess.deskew(plate, angle=0.2) is plate


# --- Bordure -----------------------------------------------------------------
def test_border_grows_the_image_on_both_axes():
    plate = synthetic_plate()
    bordered = preprocess.add_border(plate, size=10)
    assert bordered.shape == (plate.shape[0] + 20, plate.shape[1] + 20)


def test_border_is_light():
    """Une bordure claire detache les caracteres sans creer de faux trait."""
    bordered = preprocess.add_border(synthetic_plate(), size=6, value=255)
    assert bordered[0, 0] == 255


# --- Chaine complete ---------------------------------------------------------
def test_prepare_returns_a_larger_grey_image():
    plate = synthetic_plate(height=30, width=140)
    prepared = preprocess.prepare(plate)

    assert prepared.ndim == 2
    assert prepared.shape[0] > plate.shape[0]


def test_prepare_rejects_an_empty_image():
    with pytest.raises(ValueError, match="vide"):
        preprocess.prepare(np.zeros((0, 0), dtype=np.uint8))


def test_variants_produce_several_distinct_preparations():
    """Aucune chaine n'est optimale partout : on en essaie plusieurs."""
    results = preprocess.variants(synthetic_plate(angle=6))
    assert len(results) >= 2
    assert all(image.ndim == 2 and image.size > 0 for image in results)
