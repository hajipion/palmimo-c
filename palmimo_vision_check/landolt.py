"""Landolt-ring frames sized for the face display and a short staircase protocol."""

import functools
import math
import random
from dataclasses import dataclass, field

LEVELS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0)
"""Decimal acuity steps."""


def positive(value: float) -> float:
    """Reject nonfinite and nonpositive physical quantities."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Physical dimensions must be finite and positive")
    return value


def gap_mm(acuity: float, distance_m: float) -> float:
    """Project a gap subtending 1 / acuity arcminutes onto the display plane."""
    angle = math.radians(1 / positive(acuity) / 60)
    return 2 * positive(distance_m) * 1000 * math.tan(angle / 2)


FRAME_PX = 169
"""Largest square frame the face firmware accepts; it is shown at 2x on the panel."""

PANEL_DIAMETER_MM = 60.0
"""Active diameter of the round face panel in millimeters. Override with ``--panel-mm``."""

PANEL_DIAMETER_PX = 480
FRAME_SCALE = 2
MIN_GAP_PX = 2.0
"""Prototype gate in frame pixels (4 panel pixels). Not a JIS tolerance."""

CARDINALS = ("right", "down", "left", "up")
_ANGLES = {"right": 0.0, "down": 90.0, "left": 180.0, "up": 270.0}


def frame_pixel_mm(panel_diameter_mm: float = PANEL_DIAMETER_MM) -> float:
    """Return the physical size of one frame pixel on the panel."""
    return positive(panel_diameter_mm) / PANEL_DIAMETER_PX * FRAME_SCALE


def gap_px(acuity: float, distance_m: float, panel_diameter_mm: float = PANEL_DIAMETER_MM) -> float:
    """Return the gap width in frame pixels for an acuity level at a distance."""
    return gap_mm(acuity, distance_m) / frame_pixel_mm(panel_diameter_mm)


def eligible_levels(distance_m: float, panel_diameter_mm: float = PANEL_DIAMETER_MM) -> tuple[float, ...]:
    """Select the levels the frame can show: gap wide enough, ring with a white margin."""
    levels = []
    for acuity in LEVELS:
        gap = gap_px(acuity, distance_m, panel_diameter_mm)
        if gap >= MIN_GAP_PX and 2.8 * 5 * gap <= FRAME_PX:
            levels.append(acuity)
    return tuple(levels)


def render_gray8(gap: float, direction: str, size: int = FRAME_PX, samples: int = 3) -> bytes:
    """Rasterize a black ring on white as GRAY8, antialiased by supersampling.

    Args:
        gap: Gap and stroke width in frame pixels.
        direction: One of ``right``, ``down``, ``left``, ``up`` in screen coordinates.
        size: Frame edge in pixels.
        samples: Supersampling factor per axis.

    Returns:
        ``size * size`` bytes, row-major, 255 for white.
    """
    positive(gap)
    outer = 2.5 * gap
    inner = 1.5 * gap
    if outer > size / 2:
        raise ValueError("Target does not fit the frame")
    angle = math.radians(_ANGLES[direction])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    center = size / 2
    pixels = bytearray(b"\xff" * (size * size))
    low = max(0, int(center - outer) - 1)
    high = min(size, int(center + outer) + 2)
    total = samples * samples
    for y in range(low, high):
        for x in range(low, high):
            dark = 0
            for sy in range(samples):
                py = y + (sy + 0.5) / samples - center
                for sx in range(samples):
                    px = x + (sx + 0.5) / samples - center
                    radius = math.hypot(px, py)
                    if radius > outer or radius < inner:
                        continue
                    along = px * cos_a + py * sin_a
                    across = -px * sin_a + py * cos_a
                    if along > 0 and abs(across) <= gap / 2:
                        continue
                    dark += 1
            if dark:
                pixels[y * size + x] = 255 - round(255 * dark / total)
    return bytes(pixels)


BACKGROUND_RGB = (255, 214, 0)
"""Yellow behind the ring, close to the face's own idle color."""


@functools.cache
def _disc(size: int) -> bytes:
    """Return the coverage of a centered disc that fills the frame, 255 inside and 0 outside."""
    center = radius = size / 2
    mask = bytearray(size * size)
    for y in range(size):
        for x in range(size):
            edge = radius - math.hypot(x + 0.5 - center, y + 0.5 - center)
            mask[y * size + x] = round(255 * min(max(edge + 0.5, 0.0), 1.0))
    return bytes(mask)


def tint(gray: bytes, background: tuple[int, int, int] = BACKGROUND_RGB) -> list[tuple[int, int, int]]:
    """Map a square GRAY8 frame onto a round colored background.

    White becomes the background color inside the disc, ink stays black, and
    everything outside the disc is black so the face shows a circle, not a square.
    """
    red, green, blue = background
    table = [(red * v // 255, green * v // 255, blue * v // 255) for v in range(256)]
    disc = _disc(math.isqrt(len(gray)))
    return [table[v * m // 255] for v, m in zip(gray, disc, strict=True)]


def to_rgb565(gray: bytes, background: tuple[int, int, int] = BACKGROUND_RGB) -> bytes:
    """Encode a GRAY8 frame as little-endian RGB565 on a colored background."""
    out = bytearray()
    for red, green, blue in tint(gray, background):
        out += (((red >> 3) << 11) | ((green >> 2) << 5) | (blue >> 3)).to_bytes(2, "little")
    return bytes(out)


@dataclass
class QuickSession:
    """Short staircase: one correct answer advances, a miss needs two of three to pass.

    The result is the last passed level. Trials use the four cardinal directions.
    """

    levels: tuple[float, ...]
    rng: random.Random = field(default_factory=random.Random)
    index: int = 0
    passed: float | None = None
    done: bool = False
    history: list[dict[str, object]] = field(default_factory=list)
    _correct: int = 0
    _asked: int = 0
    _retry: bool = False
    _direction: str = ""

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("No levels can be shown at this distance")
        self._next_direction()

    def _next_direction(self) -> None:
        choices = [d for d in CARDINALS if d != self._direction]
        self._direction = self.rng.choice(choices)

    @property
    def acuity(self) -> float:
        """Return the level being shown."""
        return self.levels[self.index]

    @property
    def direction(self) -> str:
        """Return the direction of the current gap."""
        return self._direction

    def answer(self, direction: str | None) -> None:
        """Score one answer; ``None`` or ``unknown`` counts as a miss."""
        if self.done:
            raise ValueError("Session is finished")
        correct = direction == self._direction
        self.history.append({"acuity": self.acuity, "shown": self._direction, "answer": direction, "correct": correct})
        self._asked += 1
        self._correct += correct
        if not self._retry:
            if correct:
                self._advance()
            else:
                self._retry = True
                self._next_direction()
            return
        if self._correct >= 2:
            self._advance()
        elif self._asked - self._correct >= 2:
            self.done = True
        else:
            self._next_direction()

    def _advance(self) -> None:
        self.passed = self.acuity
        self._correct = self._asked = 0
        self._retry = False
        if self.index + 1 >= len(self.levels):
            self.done = True
            return
        self.index += 1
        self._next_direction()

    @property
    def at_ceiling(self) -> bool:
        """Return whether every shown level was passed."""
        return self.done and self.passed == self.levels[-1]
