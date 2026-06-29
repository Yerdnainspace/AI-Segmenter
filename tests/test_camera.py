import pytest

from ai_segmenter.camera import format_camera_choice, parse_camera_index


def test_parse_camera_index_from_parenthesized_choice():
    assert parse_camera_index("USB Capture (Kamera 3)") == 3


def test_parse_camera_index_from_plain_choice():
    assert parse_camera_index("Kamera 7") == 7


def test_parse_camera_index_rejects_invalid_choice():
    with pytest.raises(ValueError):
        parse_camera_index("Keine Live-Quelle gefunden")


def test_format_camera_choice_uses_detected_name():
    assert format_camera_choice(0, ["DeckLink Webcam"]) == "DeckLink Webcam (Kamera 0)"


def test_format_camera_choice_falls_back_to_index():
    assert format_camera_choice(4, []) == "Kamera 4"

