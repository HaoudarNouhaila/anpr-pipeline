"""Tests de la normalisation des plaques.

Chaque test part d'une confusion d'OCR realiste et verifie que le masque du
format la corrige dans le bon sens. C'est le coeur du module : une correction
appliquee a l'envers produit une plaque fausse mais plausible, donc invisible
sans test.
"""

import pytest

from anpr.ocr.normalise import (
    PlateFormat,
    clean,
    formats_for,
    coerce_to_mask,
    is_plausible,
    matches_mask,
    normalise,
)


# --- Nettoyage ---------------------------------------------------------------
def test_clean_removes_separators_and_uppercases():
    assert clean("ab-123-cd") == "AB123CD"
    assert clean("AB 123 CD") == "AB123CD"
    assert clean("AB·123·CD") == "AB123CD"


def test_clean_removes_decorative_characters():
    """La bande bleue europeenne fait souvent lire un F ou un drapeau parasite."""
    assert clean("AB-123-CD*") == "AB123CD"
    assert clean("(AB-123-CD)") == "AB123CD"


def test_clean_on_empty_input():
    assert clean("") == ""
    assert clean("---") == ""


# --- Correction guidee par le masque -----------------------------------------
def test_letter_positions_convert_digits_to_letters():
    """Aux positions de lettres, 0 devient O et 1 devient I."""
    corrected, count = coerce_to_mask("0B123C0", "LLDDDLL")
    assert corrected == "OB123CO"
    assert count == 2


def test_digit_positions_convert_letters_to_digits():
    """Aux positions de chiffres, O devient 0 et I devient 1."""
    corrected, count = coerce_to_mask("ABO2ICD", "LLDDDLL")
    assert corrected == "AB021CD"
    assert count == 2


def test_the_same_character_is_corrected_in_opposite_directions():
    """C'est tout l'interet du masque : un 0 lu devient O ou reste 0 selon sa place.

    Sans contexte de format, ces deux corrections sont contradictoires et
    aucune regle globale ne peut les produire toutes les deux.
    """
    corrected, _ = coerce_to_mask("00000OO", "LLDDDLL")
    # Positions 0-1 : lettres attendues -> les 0 deviennent O.
    # Positions 2-4 : chiffres attendus -> les 0 restent 0.
    # Positions 5-6 : lettres attendues -> les O restent O.
    assert corrected == "OO000OO"


def test_correct_reading_needs_no_correction():
    corrected, count = coerce_to_mask("AB123CD", "LLDDDLL")
    assert corrected == "AB123CD"
    assert count == 0


def test_uncorrectable_character_is_left_alone():
    """Une lettre sans equivalent chiffre reste en place et fera echouer la validation."""
    corrected, count = coerce_to_mask("ABXY3CD", "LLDDDLL")
    assert "X" in corrected
    assert not matches_mask(corrected, "LLDDDLL")


def test_mask_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Longueurs incompatibles"):
        coerce_to_mask("AB12", "LLDDDLL")


# --- Reconnaissance de format ------------------------------------------------
def test_french_siv_plate_is_recognised():
    text, plate_format = normalise("AB-123-CD", formats=formats_for("FR"))
    assert text == "AB123CD"
    assert plate_format == "FR-SIV"


def test_ambiguous_formats_are_both_acceptable():
    """FR-SIV et IT partagent le masque LLDDDLL : les distinguer demanderait
    de connaitre les combinaisons de lettres reellement attribuees, ce qui
    depasse ce que le seul masque peut trancher. Le module ne pretend donc pas
    faire cette distinction."""
    _, plate_format = normalise("AB-123-CD")
    assert plate_format is not None


def test_spanish_plate_is_recognised():
    text, plate_format = normalise("1234 BCD")
    assert text == "1234BCD"
    assert plate_format == "ES"


def test_moroccan_plate_is_recognised():
    text, plate_format = normalise("12345-A-6")
    assert text == "12345A6"
    assert plate_format == "MA"


def test_ocr_confusions_are_fixed_end_to_end():
    """Une lecture entierement confondue doit ressortir juste.

    L'OCR a rendu 0 pour O, 8 pour B, O pour 0 et 1 pour I. Le masque LLDDDLL
    remet chacun a sa place, a condition d'avoir declare le pays.
    """
    text, plate_format = normalise("08-1O5-C1", formats=formats_for("FR"))
    assert text == "OB105CI"
    assert plate_format == "FR-SIV"


def test_country_scope_changes_the_verdict():
    """Meme lecture, deux pays, deux resultats — et les deux sont defendables.

    Sans restriction, le format marocain DDDDDLD gagne : il valide avec une
    seule correction contre quatre pour le format francais. Declarer le pays
    est donc ce qui rend le resultat previsible en production.
    """
    fr_text, fr_format = normalise("08-1O5-C1", formats=formats_for("FR"))
    ma_text, ma_format = normalise("08-1O5-C1", formats=formats_for("MA"))

    assert (fr_text, fr_format) == ("OB105CI", "FR-SIV")
    assert (ma_text, ma_format) == ("08105C1", "MA")


def test_unknown_country_is_rejected():
    with pytest.raises(ValueError, match="Aucun format connu"):
        formats_for("ZZ")


def test_unknown_format_returns_cleaned_text_unchanged():
    """Sans format reconnu, on ne deforme pas la lecture."""
    text, plate_format = normalise("XYZ99")
    assert text == "XYZ99"
    assert plate_format is None


def test_the_least_corrected_format_wins():
    """Entre deux formats valides, celui qui respecte le plus la lecture gagne."""
    fmt_letters = PlateFormat(name="LETTRES", mask="LLDD", display="####")
    fmt_digits = PlateFormat(name="CHIFFRES", mask="DDDD", display="####")

    # "AB12" respecte deja LLDD sans correction ; DDDD demanderait 2 corrections.
    _, chosen = normalise("AB12", formats=[fmt_digits, fmt_letters])
    assert chosen == "LETTRES"


def test_empty_input():
    assert normalise("") == ("", None)


# --- Filtre de plausibilite --------------------------------------------------
def test_plausible_plate():
    assert is_plausible("AB-123-CD") is True


def test_text_without_digits_is_implausible():
    """Un mot lu sur la carrosserie n'est pas une plaque."""
    assert is_plausible("RENAULT") is False


def test_too_short_and_too_long_are_implausible():
    assert is_plausible("A1") is False
    assert is_plausible("AB123CD456789") is False
