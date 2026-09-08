"""Normalisation du texte lu sur une plaque.

Un OCR generaliste confond systematiquement certains caracteres : 0 et O, 1 et
I, 5 et S, 8 et B. Les corriger sans contexte est impossible — remplacer tous
les 0 par des O casse les plaques dont ce chiffre est legitime.

L'idee de ce module est d'utiliser le **format attendu** comme contexte. Une
plaque francaise depuis 2009 s'ecrit `AA-123-AA` : deux lettres, trois chiffres,
deux lettres. Connaissant ce masque, la correction devient deterministe — a la
position 3 on attend un chiffre, donc un `O` lu la est forcement un `0` ; a la
position 1 on attend une lettre, donc un `0` est forcement un `O`.

La normalisation essaie chaque format connu, applique les corrections que son
masque impose, et retient celui qui valide en exigeant le moins de corrections.
Ce compte de corrections sert d'arbitre : entre deux formats qui valident,
celui qui respecte le mieux la lecture brute est le plus probable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Caracteres a retirer avant analyse : separateurs, espaces, et les elements
# decoratifs frequents sur les plaques europeennes (bande bleue, drapeau).
SEPARATORS = re.compile(r"[\s\-_.·•|/\\]+")
NON_ALNUM = re.compile(r"[^A-Z0-9]")

# Substitutions a appliquer quand le masque attend une LETTRE mais que l'OCR a
# rendu un chiffre.
DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}

# Substitutions inverses : le masque attend un CHIFFRE, l'OCR a rendu une lettre.
LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
}


@dataclass(frozen=True)
class PlateFormat:
    """Format de plaque d'un pays.

    Parameters
    ----------
    name:
        Identifiant du format, renvoye dans la sortie JSON.
    mask:
        Un caractere par position : `L` pour une lettre, `D` pour un chiffre.
    display:
        Gabarit d'affichage, les `#` etant remplaces par les caracteres dans
        l'ordre. Sert uniquement a la presentation.
    """

    name: str
    mask: str
    display: str

    @property
    def length(self) -> int:
        return len(self.mask)

    def format_for_display(self, text: str) -> str:
        """Reinsere les separateurs du pays dans une chaine normalisee."""
        characters = iter(text)
        return "".join(next(characters) if c == "#" else c for c in self.display)


# Formats reconnus. L'ordre n'a pas d'importance : le choix se fait au score.
PLATE_FORMATS = [
    # France, systeme SIV depuis 2009 : AA-123-AA
    PlateFormat(name="FR-SIV", mask="LLDDDLL", display="##-###-##"),
    # France, ancien systeme FNI : 1234-AB-56
    PlateFormat(name="FR-FNI", mask="DDDDLLDD", display="####-##-##"),
    # Espagne depuis 2000 : 1234-ABC
    PlateFormat(name="ES", mask="DDDDLLL", display="####-###"),
    # Maroc, partie latine : 12345-A-6 (la lettre arabe est translitteree)
    PlateFormat(name="MA", mask="DDDDDLD", display="#####-#-#"),
    PlateFormat(name="MA", mask="DDDDLD", display="####-#-#"),
    # Italie depuis 1994 : AB-123-CD
    PlateFormat(name="IT", mask="LLDDDLL", display="##-###-##"),
]

# Longueur acceptable pour une plaque non reconnue : en deca c'est du bruit,
# au-dela c'est que l'OCR a agrege deux zones de texte.
MIN_PLAUSIBLE_LENGTH = 4
MAX_PLAUSIBLE_LENGTH = 10


def formats_for(*country_codes) -> list:
    """Restreint les formats candidats a une liste de pays.

    Pourquoi c'est necessaire : plusieurs pays partagent une meme longueur de
    plaque, et l'arbitrage au nombre de corrections peut alors elire un format
    etranger. La lecture `081O5C1` valide en `08105C1` au format marocain avec
    une seule correction, contre quatre pour lire `OB105CI` au format francais.
    Le module a raison sur son propre critere, mais un parking lyonnais n'a
    aucune raison de considerer le format marocain.

    Un deploiement connait son pays : le lui faire declarer supprime cette
    ambiguite a la source, plutot que d'inventer une heuristique de
    departage.
    """
    wanted = {code.upper() for code in country_codes}
    selected = [f for f in PLATE_FORMATS if f.name.split("-")[0] in wanted]

    if not selected:
        known = sorted({f.name.split("-")[0] for f in PLATE_FORMATS})
        raise ValueError(
            f"Aucun format connu pour {sorted(wanted)}. Pays disponibles : {known}"
        )
    return selected


def clean(raw: str) -> str:
    """Met en majuscules et retire tout ce qui n'est pas alphanumerique."""
    return NON_ALNUM.sub("", SEPARATORS.sub("", raw.upper()))


def coerce_to_mask(text: str, mask: str) -> tuple[str, int]:
    """Force `text` a respecter `mask`, en comptant les corrections faites.

    Renvoie la chaine corrigee et le nombre de caracteres modifies. Un caractere
    impossible a corriger — une lettre sans equivalent chiffre a une position
    numerique — est laisse tel quel, ce qui fera echouer la validation ensuite.
    """
    if len(text) != len(mask):
        raise ValueError(f"Longueurs incompatibles : {len(text)} contre {len(mask)}")

    corrected = []
    corrections = 0

    for character, expected in zip(text, mask):
        if expected == "L":
            if character.isalpha():
                corrected.append(character)
            elif character in DIGIT_TO_LETTER:
                corrected.append(DIGIT_TO_LETTER[character])
                corrections += 1
            else:
                corrected.append(character)
        else:  # expected == "D"
            if character.isdigit():
                corrected.append(character)
            elif character in LETTER_TO_DIGIT:
                corrected.append(LETTER_TO_DIGIT[character])
                corrections += 1
            else:
                corrected.append(character)

    return "".join(corrected), corrections


def matches_mask(text: str, mask: str) -> bool:
    """Verifie qu'une chaine respecte exactement un masque."""
    if len(text) != len(mask):
        return False
    return all(
        (character.isalpha() if expected == "L" else character.isdigit())
        for character, expected in zip(text, mask)
    )


def normalise(raw: str, formats=None) -> tuple[str, str | None]:
    """Normalise une lecture brute.

    Renvoie le couple (texte normalise, nom du format reconnu). Le format vaut
    `None` quand aucun ne correspond : la chaine nettoyee est alors renvoyee
    telle quelle plutot que d'etre deformee de force vers un format arbitraire.
    """
    formats = PLATE_FORMATS if formats is None else formats
    cleaned = clean(raw)

    if not cleaned:
        return "", None

    best_text = cleaned
    best_format = None
    best_corrections = None

    for plate_format in formats:
        if len(cleaned) != plate_format.length:
            continue

        candidate, corrections = coerce_to_mask(cleaned, plate_format.mask)
        if not matches_mask(candidate, plate_format.mask):
            continue

        if best_corrections is None or corrections < best_corrections:
            best_text = candidate
            best_format = plate_format.name
            best_corrections = corrections

    return best_text, best_format


def is_plausible(text: str) -> bool:
    """Filtre grossier avant toute analyse de format.

    Une plaque contient des chiffres et fait entre 4 et 10 caracteres. Cela
    ecarte l'essentiel de ce qu'un OCR ramasse par erreur : marques de
    vehicules, autocollants, fragments de texte du decor.
    """
    cleaned = clean(text)
    if not MIN_PLAUSIBLE_LENGTH <= len(cleaned) <= MAX_PLAUSIBLE_LENGTH:
        return False
    return any(character.isdigit() for character in cleaned)
