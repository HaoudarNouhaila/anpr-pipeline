"""Outils de conformite : pseudonymisation des plaques, anonymisation visuelle.

Les deux volets ont des dependances tres differentes. Le hachage n'utilise que
la bibliotheque standard ; le floutage exige OpenCV. Importer le second depuis
ce fichier obligerait un service qui ne fait que manipuler des empreintes — une
base de donnees, un worker de purge — a installer toute la pile de vision.

L'import du floutage est donc differe : `FaceAnonymiser` et `blur_region`
restent accessibles depuis `anpr.privacy`, mais OpenCV n'est charge qu'au
moment ou l'un des deux est reellement demande.
"""

from .hashing import MissingHashKey, generate_key, hash_plate, verify

__all__ = [
    "hash_plate",
    "verify",
    "generate_key",
    "MissingHashKey",
    "FaceAnonymiser",
    "blur_region",
]

_LAZY = {"FaceAnonymiser", "blur_region"}


def __getattr__(name):
    """Charge le module d'anonymisation visuelle a la premiere demande."""
    if name in _LAZY:
        from . import anonymise

        return getattr(anonymise, name)
    raise AttributeError(f"module {__name__!r} n'a pas d'attribut {name!r}")
