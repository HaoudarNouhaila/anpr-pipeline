"""Pseudonymisation des plaques lues.

Une plaque d'immatriculation est une donnee a caractere personnel : elle
identifie indirectement une personne physique. Un systeme qui journalise des
plaques en clair constitue donc un fichier de donnees personnelles, avec les
obligations qui vont avec.

Ce module permet de conserver une plaque sous forme d'empreinte plutot qu'en
clair. L'empreinte reste comparable — on peut verifier qu'un vehicule deja vu
repasse, ce qui suffit a la plupart des usages : controle d'acces, comptage de
vehicules uniques, detection de recidive.

Pourquoi HMAC et non un simple SHA-256
--------------------------------------
L'espace des plaques est minuscule : quelques centaines de millions de
combinaisons. Un SHA-256 nu se casse par force brute en quelques minutes — il
suffit de hacher toutes les plaques possibles et de comparer. Le HMAC introduit
une cle secrete propre au deploiement, sans laquelle cette attaque est
impossible. C'est la difference entre une donnee pseudonymisee et une donnee
qui n'est protegee qu'en apparence.

La cle est lue dans l'environnement et n'a pas de valeur par defaut : un secret
ecrit en dur dans le code n'est pas un secret.
"""

from __future__ import annotations

import hashlib
import hmac
import os

ENV_VAR = "ANPR_HASH_KEY"
DIGEST_LENGTH = 16  # 64 bits en hexadecimal, suffisant contre les collisions


class MissingHashKey(RuntimeError):
    """Levee quand la cle de hachage n'est pas configuree."""


def get_key(key: str | None = None) -> bytes:
    """Recupere la cle de hachage, depuis l'argument ou l'environnement."""
    resolved = key or os.environ.get(ENV_VAR)
    if not resolved:
        raise MissingHashKey(
            f"La variable {ENV_VAR} n'est pas definie.\n"
            "Copiez .env.example en .env et renseignez une cle aleatoire :\n"
            '  python -c "import secrets; print(secrets.token_hex(32))"'
        )
    return resolved.encode("utf-8")


def hash_plate(plate: str, key: str | None = None) -> str:
    """Renvoie l'empreinte HMAC-SHA256 tronquee d'une plaque.

    La plaque est mise en majuscules et debarrassee de ses espaces avant
    hachage, pour que deux ecritures de la meme plaque donnent la meme
    empreinte.
    """
    normalised = plate.upper().replace(" ", "").replace("-", "")
    digest = hmac.new(get_key(key), normalised.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:DIGEST_LENGTH]


def verify(plate: str, expected_hash: str, key: str | None = None) -> bool:
    """Verifie qu'une plaque correspond a une empreinte.

    La comparaison passe par `compare_digest` pour ne pas fuir d'information par
    le temps d'execution.
    """
    return hmac.compare_digest(hash_plate(plate, key), expected_hash)


def generate_key() -> str:
    """Genere une cle adaptee, a placer dans le fichier .env."""
    import secrets

    return secrets.token_hex(32)
