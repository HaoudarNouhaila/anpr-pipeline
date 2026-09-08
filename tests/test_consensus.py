"""Tests du vote de consensus sur les lectures successives d'une plaque."""

import pytest

from anpr.tracking import ConsensusVoter, TrackConsensus
from anpr.types import PlateReading


def reading(text, confidence, plate_format="FR-SIV"):
    return PlateReading(raw=text, text=text, confidence=confidence, country_format=plate_format)


@pytest.fixture
def voter():
    return ConsensusVoter(min_readings=3)


# --- Election du verdict -----------------------------------------------------
def test_unanimous_readings_give_that_verdict(voter):
    for _ in range(3):
        voter.add(1, reading("AB123CD", 0.9))

    verdict = voter.verdict(1)
    assert verdict.text == "AB123CD"
    assert verdict.confidence == pytest.approx(0.9)


def test_majority_wins_over_a_single_outlier(voter):
    voter.add(1, reading("AB123CD", 0.7))
    voter.add(1, reading("AB123CD", 0.7))
    voter.add(1, reading("AB123CO", 0.8))  # lecture isolee, pourtant plus sure

    assert voter.verdict(1).text == "AB123CD"


def test_concordance_beats_a_single_confident_reading(voter):
    """Trois lectures moyennes concordantes l'emportent sur une tres sure.

    C'est le comportement voulu : la concordance est un signal plus fiable que
    la confiance affichee par un OCR, qui est mal calibree sur du texte court.
    """
    for _ in range(3):
        voter.add(1, reading("AB123CD", 0.45))
    voter.add(1, reading("XY789ZW", 0.98))

    assert voter.verdict(1).text == "AB123CD"


def test_valid_format_outweighs_an_unformatted_reading(voter):
    """A confiance egale, une lecture conforme a un format connu gagne."""
    voter.add(1, reading("AB123CD", 0.6, plate_format="FR-SIV"))
    voter.add(1, reading("A8I23CD", 0.6, plate_format=None))

    assert voter.verdict(1).text == "AB123CD"


def test_confidence_is_averaged_not_summed(voter):
    """Le score de vote sert a classer, pas a etre renvoye comme confiance.

    Renvoyer la somme donnerait une valeur superieure a 1, denuee de sens.
    """
    voter.add(1, reading("AB123CD", 0.8))
    voter.add(1, reading("AB123CD", 0.6))

    verdict = voter.verdict(1)
    assert verdict.confidence == pytest.approx(0.7)
    assert verdict.confidence <= 1.0


def test_verdict_on_unknown_track_is_none(voter):
    assert voter.verdict(999) is None


# --- Emission unique par piste -----------------------------------------------
def test_track_is_not_ready_before_enough_readings(voter):
    voter.add(1, reading("AB123CD", 0.9))
    voter.add(1, reading("AB123CD", 0.9))

    assert voter.is_ready(1) is False
    assert voter.pending() == []


def test_track_becomes_ready_at_the_threshold(voter):
    for _ in range(3):
        voter.add(1, reading("AB123CD", 0.9))

    assert voter.is_ready(1) is True
    assert voter.pending() == [1]


def test_a_plate_is_emitted_only_once(voter):
    """Sans cela, la meme plaque serait signalee a chaque frame du passage."""
    for _ in range(5):
        voter.add(1, reading("AB123CD", 0.9))

    assert voter.take(1).text == "AB123CD"
    assert voter.is_ready(1) is False
    assert voter.pending() == []


def test_tracks_are_independent(voter):
    for _ in range(3):
        voter.add(1, reading("AB123CD", 0.9))
    voter.add(2, reading("XY789ZW", 0.9))

    assert voter.pending() == [1]
    assert voter.verdict(2).text == "XY789ZW"


# --- Mesure d'accord ---------------------------------------------------------
def test_agreement_is_one_when_unanimous():
    track = TrackConsensus(track_id=1)
    for _ in range(4):
        track.add(reading("AB123CD", 0.9))

    assert track.agreement() == pytest.approx(1.0)


def test_agreement_reflects_a_split_vote():
    track = TrackConsensus(track_id=1)
    track.add(reading("AB123CD", 0.9))
    track.add(reading("AB123CD", 0.9))
    track.add(reading("AB123CO", 0.9))
    track.add(reading("AB123C0", 0.9))

    assert track.agreement() == pytest.approx(0.5)


def test_summary_reports_every_track(voter):
    for _ in range(3):
        voter.add(1, reading("AB123CD", 0.9))
    for _ in range(2):
        voter.add(2, reading("XY789ZW", 0.8))

    rows = voter.summary()
    assert len(rows) == 2
    assert rows[0]["plate"] == "AB123CD"
    assert rows[0]["readings"] == 3
    assert rows[1]["readings"] == 2


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_readings"):
        ConsensusVoter(min_readings=0)
