"""Preparation de l'image d'une plaque avant lecture.

Une plaque extraite d'une scene de rue arrive dans un etat peu favorable a
l'OCR : quelques dizaines de pixels de haut, penchee, faiblement contrastee,
parfois surexposee par un reflet. Chaque etape ci-dessous corrige un de ces
defauts, et l'ordre compte.

1. **Agrandissement d'abord.** Les moteurs OCR sont entraines sur du texte
   d'une certaine taille ; en dessous de ~32 px de hauteur leurs performances
   s'effondrent. Agrandir avant tout autre traitement evite aussi d'amplifier
   les artefacts qu'un filtre aurait introduits.
2. **Niveaux de gris.** La couleur n'apporte rien a la lecture de caracteres
   et triple le cout des operations suivantes.
3. **Egalisation locale du contraste (CLAHE).** Une egalisation globale serait
   ruinee par un reflet lumineux sur une portion de la plaque ; CLAHE travaille
   par tuiles et reste efficace sur un eclairage inegal.
4. **Redressement.** L'inclinaison est estimee sur la disposition des pixels
   sombres — les caracteres — et non sur les bords de la plaque, souvent
   masques par le cadre du support.
"""

from __future__ import annotations

import cv2
import numpy as np

# Hauteur cible apres agrandissement. Au-dela, le gain de lecture ne compense
# plus le cout d'inference.
TARGET_HEIGHT = 96

# Au-dela de cet angle, l'estimation vient probablement d'un artefact et non de
# l'inclinaison reelle du texte : mieux vaut ne rien redresser que de tordre
# une plaque droite.
MAX_DESKEW_ANGLE = 30.0


def upscale(image, target_height: int = TARGET_HEIGHT):
    """Agrandit l'image jusqu'a la hauteur cible, en conservant le rapport.

    L'interpolation cubique est preferee a la lineaire : elle preserve mieux
    les transitions nettes entre un caractere et son fond, ce qui est
    exactement ce que l'OCR cherche.
    """
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or height >= target_height:
        return image

    scale = target_height / height
    return cv2.resize(
        image, (int(round(width * scale)), target_height), interpolation=cv2.INTER_CUBIC
    )


def to_grey(image):
    """Convertit en niveaux de gris si necessaire."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(grey, clip_limit: float = 2.5, tile: int = 8):
    """Egalise le contraste localement (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(grey)


def estimate_correction_angle(grey) -> float:
    """Estime l'angle de ROTATION A APPLIQUER pour redresser le texte, en degres.

    Attention au sens : la fonction ne renvoie pas l'inclinaison mesuree mais
    son oppose, c'est-a-dire la correction. Une plaque penchee de +10 degres
    donne -10, valeur directement utilisable par `deskew`. Nommer cette
    fonction `estimate_skew` inviterait a l'utiliser comme une mesure et
    produirait une erreur de signe silencieuse — l'image serait penchee deux
    fois plus au lieu d'etre redressee.

    On binarise pour isoler les caracteres, puis on cherche le plus petit
    rectangle oriente englobant l'ensemble des pixels sombres. Son angle est
    celui du texte.

    Le seuillage d'Otsu est retenu parce qu'il choisit son seuil a partir de
    l'histogramme de l'image : un seuil fixe echouerait sur une plaque sombre
    comme sur une plaque surexposee.
    """
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coordinates = cv2.findNonZero(binary)
    if coordinates is None or len(coordinates) < 20:
        return 0.0

    angle = cv2.minAreaRect(coordinates)[-1]

    # La convention d'angle de minAreaRect a CHANGE selon les versions d'OpenCV :
    # jusqu'a la 4.4 l'angle etait dans ]0, 90], depuis il est dans [-90, 0].
    # Un meme rectangle est donc decrit par des valeurs differentes selon la
    # version installee, et coder en dur l'une des deux conventions produit un
    # redressement qui ne fonctionne que dans un sens.
    #
    # La normalisation ci-dessous ramene l'angle dans [-45, 45] quelle que soit
    # la convention : un rectangle est invariant par rotation de 90 degres, donc
    # ajouter ou retrancher 90 ne change pas ce qu'il decrit.
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90

    return 0.0 if abs(angle) > MAX_DESKEW_ANGLE else float(angle)


def deskew(image, angle: float | None = None):
    """Redresse l'image.

    `angle` est la rotation a appliquer, pas l'inclinaison mesuree — voir
    `estimate_correction_angle`. Laisser a None pour qu'elle soit estimee.
    """
    grey = to_grey(image)
    angle = estimate_correction_angle(grey) if angle is None else angle

    if abs(angle) < 0.5:  # inutile de rouvrir l'image pour un demi-degre
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def add_border(image, size: int = 10, value: int = 255):
    """Ajoute une marge claire autour de l'image.

    Un moteur OCR segmente moins bien les caracteres qui touchent le bord :
    quelques pixels de marge suffisent a les detacher.
    """
    return cv2.copyMakeBorder(
        image, size, size, size, size, cv2.BORDER_CONSTANT, value=value
    )


def prepare(image, do_deskew: bool = True, do_border: bool = True):
    """Chaine complete de preparation d'une plaque.

    Renvoie une image en niveaux de gris prete pour l'OCR.
    """
    if image is None or image.size == 0:
        raise ValueError("Image de plaque vide.")

    working = upscale(image)
    grey = to_grey(working)
    grey = enhance_contrast(grey)

    if do_deskew:
        grey = deskew(grey)
    if do_border:
        grey = add_border(grey)

    return grey


def variants(image):
    """Genere plusieurs preparations de la meme plaque.

    Aucune chaine de pretraitement n'est optimale sur toutes les conditions :
    le redressement aide sur une prise oblique et peut nuire sur une plaque
    deja droite mais bruitee. Produire quelques variantes et retenir la lecture
    la plus sure coute quelques millisecondes et rattrape une partie des
    echecs. Le compromis est mesure dans le benchmark de latence.
    """
    return [
        prepare(image, do_deskew=True, do_border=True),
        prepare(image, do_deskew=False, do_border=True),
        upscale(to_grey(image)),
    ]
