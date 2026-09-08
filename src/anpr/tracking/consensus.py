"""Vote de consensus sur les lectures successives d'une meme plaque.

Sur une video, un vehicule reste visible plusieurs dizaines de frames et sa
plaque est lue autant de fois. Ces lectures divergent : l'angle change, la
plaque grossit puis sort du champ, un reflet passe. Retenir la derniere lecture
serait arbitraire, et les reemettre toutes noierait le consommateur sous des
doublons.

Ce module agrege les lectures par piste et elit une reponse. Le vote est
pondere plutot que majoritaire simple : une lecture sure vaut davantage qu'une
lecture hesitante, et une lecture conforme au format d'un pays vaut davantage
qu'une chaine qui ne correspond a rien. Trois lectures moyennes concordantes
peuvent ainsi l'emporter sur une lecture isolee tres sure — ce qui est le
comportement voulu, la concordance etant un signal plus fiable que la
confiance affichee par un OCR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..types import PlateReading

# Facteur applique a une lecture dont le format correspond a un pays connu.
# Une plaque qui respecte un format valide est nettement plus credible qu'une
# suite de caracteres arbitraire de meme confiance OCR.
VALID_FORMAT_BONUS = 1.6

# Nombre de lectures a accumuler avant de considerer un resultat comme stable.
DEFAULT_MIN_READINGS = 3


@dataclass
class TrackConsensus:
    """Lectures accumulees pour une piste, et le verdict qui en decoule."""

    track_id: int
    readings: list = field(default_factory=list)
    emitted: bool = False

    def add(self, reading: PlateReading) -> None:
        self.readings.append(reading)

    @property
    def count(self) -> int:
        return len(self.readings)

    def scores(self) -> dict:
        """Score cumule par texte candidat."""
        totals: dict = defaultdict(float)
        for reading in self.readings:
            weight = reading.confidence
            if reading.is_valid_format:
                weight *= VALID_FORMAT_BONUS
            totals[reading.text] += weight
        return dict(totals)

    def verdict(self) -> PlateReading | None:
        """Lecture retenue pour cette piste, ou None si aucune lecture.

        La confiance renvoyee est la moyenne des confiances des lectures ayant
        produit le texte gagnant — pas la somme des scores, qui n'aurait aucune
        signification en tant que probabilite et depasserait 1.
        """
        if not self.readings:
            return None

        totals = self.scores()
        winner = max(totals, key=totals.get)

        supporting = [r for r in self.readings if r.text == winner]
        mean_confidence = sum(r.confidence for r in supporting) / len(supporting)

        # On reprend le format de la premiere lecture gagnante qui en declare un.
        country_format = next(
            (r.country_format for r in supporting if r.country_format), None
        )
        raw = supporting[0].raw

        return PlateReading(
            raw=raw,
            text=winner,
            confidence=mean_confidence,
            country_format=country_format,
        )

    def agreement(self) -> float:
        """Proportion de lectures d'accord avec le verdict, entre 0 et 1.

        Cette mesure dit si le consensus est net ou fragile — utile pour decider
        d'emettre une alerte ou de demander une verification humaine.
        """
        if not self.readings:
            return 0.0
        winner = self.verdict().text
        return sum(1 for r in self.readings if r.text == winner) / len(self.readings)


class ConsensusVoter:
    """Accumule les lectures de toutes les pistes et decide quand emettre.

    Parameters
    ----------
    min_readings:
        Nombre de lectures a accumuler avant qu'une piste soit consideree
        comme stable et emise.
    """

    def __init__(self, min_readings: int = DEFAULT_MIN_READINGS):
        if min_readings < 1:
            raise ValueError(f"min_readings doit valoir au moins 1, recu {min_readings}")
        self.min_readings = min_readings
        self.tracks: dict = {}

    def add(self, track_id: int, reading: PlateReading) -> None:
        """Enregistre une lecture pour une piste."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackConsensus(track_id=track_id)
        self.tracks[track_id].add(reading)

    def is_ready(self, track_id: int) -> bool:
        """Vrai quand la piste a accumule assez de lectures et n'a pas ete emise."""
        track = self.tracks.get(track_id)
        if track is None:
            return False
        return track.count >= self.min_readings and not track.emitted

    def take(self, track_id: int) -> PlateReading | None:
        """Renvoie le verdict d'une piste et la marque comme emise.

        Le marquage est ce qui garantit qu'une meme plaque n'est pas signalee en
        boucle a chaque frame ou le vehicule reste visible.
        """
        track = self.tracks.get(track_id)
        if track is None:
            return None
        track.emitted = True
        return track.verdict()

    def verdict(self, track_id: int) -> PlateReading | None:
        """Verdict courant d'une piste, sans la marquer comme emise."""
        track = self.tracks.get(track_id)
        return track.verdict() if track else None

    def pending(self) -> list:
        """Pistes pretes a etre emises."""
        return [tid for tid in self.tracks if self.is_ready(tid)]

    def summary(self) -> list:
        """Etat de toutes les pistes, pour le rapport de fin de traitement."""
        rows = []
        for track_id, track in sorted(self.tracks.items()):
            verdict = track.verdict()
            if verdict is None:
                continue
            rows.append(
                {
                    "track_id": track_id,
                    "plate": verdict.text,
                    "format": verdict.country_format,
                    "confidence": round(verdict.confidence, 3),
                    "readings": track.count,
                    "agreement": round(track.agreement(), 3),
                }
            )
        return rows
