"""Tests for the face-display rings, the staircase, and answer parsing."""

import random

import pytest

from palmimo_vision_check import speech
from palmimo_vision_check.landolt import (
    FRAME_PX,
    QuickSession,
    eligible_levels,
    gap_mm,
    gap_px,
    render_gray8,
    to_rgb565,
)
from palmimo_vision_check.vision import distance_from_eyes


def test_levels_at_three_meters_span_low_to_above_one() -> None:
    levels = eligible_levels(3.0)
    assert levels[0] == 0.3
    assert levels[-1] == 1.5


def test_levels_shrink_with_a_shorter_distance() -> None:
    assert (eligible_levels(1.0)[0], eligible_levels(1.0)[-1]) == (0.1, 0.5)
    assert (eligible_levels(2.0)[0], eligible_levels(2.0)[-1]) == (0.2, 1.0)


def test_ring_gap_opens_on_the_named_side() -> None:
    gap = 10.0
    center = FRAME_PX // 2
    reach = round(2 * gap)
    for direction, (dx, dy) in {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}.items():
        frame = render_gray8(gap, direction)
        assert len(frame) == FRAME_PX * FRAME_PX
        assert frame[(center + dy * reach) * FRAME_PX + center + dx * reach] == 255
        assert frame[(center - dy * reach) * FRAME_PX + center - dx * reach] == 0


def test_rgb565_keeps_black_ink_on_a_yellow_background() -> None:
    assert to_rgb565(bytes([255, 0])) == (0xFEA0).to_bytes(2, "little") + b"\x00\x00"


def test_ring_too_large_for_the_frame_is_rejected() -> None:
    with pytest.raises(ValueError):
        render_gray8(40.0, "up")


def test_gap_follows_the_visual_angle() -> None:
    assert gap_mm(1.0, 5.0) == pytest.approx(1.45444, abs=1e-4)
    assert gap_mm(1.0, 3.0) == pytest.approx(0.6 * gap_mm(1.0, 5.0))
    with pytest.raises(ValueError):
        gap_mm(0, 3.0)


def test_gap_shrinks_as_acuity_rises() -> None:
    assert gap_px(1.0, 3.0) < gap_px(0.5, 3.0)


def test_quick_session_all_correct_reaches_the_ceiling() -> None:
    session = QuickSession((0.3, 0.5, 1.0), rng=random.Random(1))
    while not session.done:
        session.answer(session.direction)
    assert session.passed == 1.0
    assert session.at_ceiling
    assert len(session.history) == 3


def test_quick_session_one_miss_is_forgiven_by_two_of_three() -> None:
    session = QuickSession((0.3, 0.5), rng=random.Random(2))
    session.answer(None)
    session.answer(session.direction)
    session.answer(session.direction)
    assert session.passed == 0.3
    assert not session.done


def test_quick_session_two_misses_end_at_the_last_passed_level() -> None:
    session = QuickSession((0.3, 0.5), rng=random.Random(3))
    session.answer(session.direction)
    session.answer("unknown")
    session.answer(None)
    assert session.done
    assert session.passed == 0.3
    assert not session.at_ceiling


def test_quick_session_never_repeats_a_direction_back_to_back() -> None:
    session = QuickSession((0.3, 0.4, 0.5, 0.6, 0.7), rng=random.Random(4))
    previous = session.direction
    while not session.done:
        session.answer(session.direction)
        if not session.done:
            assert session.direction != previous
            previous = session.direction


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("みぎ", "right"),
        ("右です。", "right"),
        ("ひだり", "left"),
        ("上", "up"),
        ("した！", "down"),
        ("わかりません", "unknown"),
        ("見えない", "unknown"),
        ("右上", None),
        ("こんにちは", None),
        ("", None),
    ],
)
def test_parse_direction(text: str, expected: str | None) -> None:
    assert speech.parse_direction(text) == expected


def test_result_lines() -> None:
    assert speech.result_line(0.7, 0.3, False) == "あなたの視力は、0.7です。"
    assert speech.result_line(None, 0.3, False) == "あなたの視力は、0.3未満です。"
    assert speech.result_line(1.2, 0.3, True) == "あなたの視力は、1.2以上です。"
    assert "めがいい" in speech.comment_line(1.0).replace("目", "め")


def test_distance_from_eyes_matches_the_pinhole_model() -> None:
    near = distance_from_eyes(40.0, 1296, 53.5)
    assert distance_from_eyes(20.0, 1296, 53.5) == pytest.approx(2 * near)
    assert 1.5 < near < 2.5
