"""Tests des types geometriques du pipeline."""

import pytest

from anpr.types import BoundingBox, PlateReading, VehicleDetection, VehicleRecord


def test_dimensions_and_centre():
    box = BoundingBox(10, 20, 110, 70)
    assert box.width == 100
    assert box.height == 50
    assert box.centre == (60, 45)
    assert box.area == 5000


def test_aspect_ratio_of_a_european_plate():
    """Une plaque europeenne fait 520 x 110 mm, soit un rapport proche de 4,7."""
    assert BoundingBox(0, 0, 470, 100).aspect_ratio == pytest.approx(4.7)


def test_shift_moves_from_crop_to_image_coordinates():
    """L'etage plaque travaille sur un crop ; ses boites doivent revenir au global."""
    in_crop = BoundingBox(10, 5, 60, 20)
    in_image = in_crop.shift(300, 200)
    assert in_image.as_xyxy() == (310, 205, 360, 220)


def test_expand_adds_margin_around_the_plate():
    box = BoundingBox(100, 100, 200, 120).expand(0.2)
    assert box.left == pytest.approx(90)
    assert box.right == pytest.approx(210)
    assert box.top == pytest.approx(98)


def test_clip_keeps_the_box_inside_the_image():
    box = BoundingBox(-30, -20, 900, 700).clip(640, 480)
    assert box.as_xyxy() == (0, 0, 640, 480)


def test_iou_of_identical_boxes_is_one():
    box = BoundingBox(0, 0, 100, 100)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(50, 50, 60, 60)) == 0.0


def test_iou_of_half_overlapping_boxes():
    a = BoundingBox(0, 0, 100, 100)
    b = BoundingBox(50, 0, 150, 100)
    assert a.iou(b) == pytest.approx(1 / 3)


def test_plate_is_attached_to_the_vehicle_containing_its_centre():
    vehicle = BoundingBox(0, 0, 400, 300)
    plate = BoundingBox(150, 240, 250, 270)
    assert vehicle.contains_centre_of(plate) is True


def test_a_plate_overflowing_the_vehicle_box_stays_attached():
    """Le rattachement teste le centre, pas l'inclusion stricte.

    Un detecteur cadre rarement le vehicule au pixel pres : la boite de plaque
    deborde souvent un peu. Exiger l'inclusion complete perdrait ces plaques.
    """
    vehicle = BoundingBox(0, 0, 400, 300)
    overflowing = BoundingBox(340, 260, 420, 310)  # centre en (380, 285), dedans
    assert vehicle.contains_centre_of(overflowing) is True


def test_a_plate_belonging_to_another_vehicle_is_not_attached():
    vehicle = BoundingBox(0, 0, 400, 300)
    elsewhere = BoundingBox(600, 280, 700, 320)  # centre en (650, 300), dehors
    assert vehicle.contains_centre_of(elsewhere) is False


def test_json_output_can_omit_the_plaintext_plate():
    """Mode a utiliser pour tout stockage durable."""
    record = VehicleRecord(
        vehicle=VehicleDetection(BoundingBox(0, 0, 100, 100), "car", 0.9, track_id=3),
        reading=PlateReading(raw="AB123CD", text="AB123CD", confidence=0.8, country_format="FR-SIV"),
        plate_hash="deadbeefdeadbeef",
    )
    record.plate = None

    public = record.as_dict(include_plaintext=False)
    assert public["vehicle"]["label"] == "voiture"
    assert public["track_id"] == 3
    assert "AB123CD" not in str(public)
