"""Tests de la pseudonymisation des plaques."""

import pytest

from anpr.privacy import MissingHashKey, generate_key, hash_plate, verify

KEY = "cle-de-test-uniquement"


def test_same_plate_gives_same_hash():
    assert hash_plate("AB-123-CD", KEY) == hash_plate("AB-123-CD", KEY)


def test_different_plates_give_different_hashes():
    assert hash_plate("AB-123-CD", KEY) != hash_plate("AB-123-CE", KEY)


def test_formatting_does_not_change_the_hash():
    """Deux ecritures de la meme plaque doivent se reconnaitre entre elles."""
    reference = hash_plate("AB-123-CD", KEY)
    assert hash_plate("ab 123 cd", KEY) == reference
    assert hash_plate("AB123CD", KEY) == reference


def test_a_different_key_gives_a_different_hash():
    """C'est tout l'interet du HMAC : sans la cle, l'empreinte est inexploitable."""
    assert hash_plate("AB-123-CD", KEY) != hash_plate("AB-123-CD", "autre-cle")


def test_verify_accepts_the_right_plate():
    assert verify("AB-123-CD", hash_plate("AB-123-CD", KEY), KEY) is True


def test_verify_rejects_a_wrong_plate():
    assert verify("XY-789-ZW", hash_plate("AB-123-CD", KEY), KEY) is False


def test_missing_key_raises_an_actionable_error(monkeypatch):
    monkeypatch.delenv("ANPR_HASH_KEY", raising=False)
    with pytest.raises(MissingHashKey, match=".env"):
        hash_plate("AB-123-CD")


def test_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANPR_HASH_KEY", KEY)
    assert hash_plate("AB-123-CD") == hash_plate("AB-123-CD", KEY)


def test_generated_keys_are_unique_and_long():
    first, second = generate_key(), generate_key()
    assert first != second
    assert len(first) == 64
