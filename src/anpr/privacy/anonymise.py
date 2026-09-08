"""Anonymisation visuelle : floutage des visages et des plaques.

Publier ou archiver des images de voie publique fait apparaitre des visages et
des plaques appartenant a des tiers. Le pipeline sait donc produire, en
parallele de son analyse, une version anonymisee de l'image.

Le detecteur de visages retenu est le classifieur de Haar livre avec OpenCV. Ce
choix est deliberatif : il ne demande aucun telechargement de poids, tourne sur
processeur en quelques millisecondes et suffit largement pour un floutage, ou
un faux positif est sans consequence et un visage manque de face est rare. Un
detecteur profond serait plus precis mais alourdirait l'installation pour un
gain limite sur cet usage.
"""

from __future__ import annotations

import cv2

from ..types import BoundingBox

# Un noyau de flou proportionnel a la region garantit un rendu comparable que
# le visage occupe 20 ou 200 pixels. Une taille fixe laisserait les grands
# visages parfaitement reconnaissables.
BLUR_RATIO = 0.28


def _odd(value: int) -> int:
    """Le noyau gaussien d'OpenCV exige une taille impaire."""
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def blur_region(image, box: BoundingBox):
    """Floute une region rectangulaire de l'image, sur place."""
    height, width = image.shape[:2]
    left, top, right, bottom = box.clip(width, height).as_int()

    if right - left < 2 or bottom - top < 2:
        return image

    region = image[top:bottom, left:right]
    kernel = (_odd((right - left) * BLUR_RATIO), _odd((bottom - top) * BLUR_RATIO))
    image[top:bottom, left:right] = cv2.GaussianBlur(region, kernel, 0)
    return image


class FaceAnonymiser:
    """Detecte et floute les visages presents dans une image."""

    def __init__(self, scale_factor: float = 1.1, min_neighbours: int = 5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Classifieur de visages introuvable : {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours

    def detect(self, image) -> list:
        """Renvoie les boites des visages detectes."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            grey, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbours
        )
        return [BoundingBox(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h in faces]

    def anonymise(self, image, extra_boxes=None):
        """Floute les visages, et toute region supplementaire fournie.

        `extra_boxes` sert a flouter aussi les plaques : le pipeline les a deja
        localisees, il serait absurde de les rechercher.
        """
        output = image.copy()

        for box in self.detect(image):
            blur_region(output, box)

        for box in extra_boxes or []:
            blur_region(output, box)

        return output
