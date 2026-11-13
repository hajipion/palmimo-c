"""Camera-based ranging and the hand-raise check. No motor access."""

import base64
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EYE_SPACING_M = 0.063
"""Assumed adult interpupillary distance. Individual spread is about 6 percent."""

_MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"

_VOLUNTEER_PROMPT = (
    "この画像に、片手または両手を頭より高く挙げている人はいますか。"
    "いる場合は、その人の上半身の服を「青いシャツ」「黒いパーカー」のように色と種類で短く答えてください。"
    'JSON だけを返してください: {"hand_raised": true か false, "clothing": "服の説明、いなければ空文字"}'
)


def distance_from_eyes(eye_px: float, frame_width: int, hfov_deg: float) -> float:
    """Estimate the camera-to-face distance from the eye spacing in pixels."""
    if eye_px <= 0:
        raise ValueError("Eye spacing must be positive")
    focal_px = frame_width / 2 / math.tan(math.radians(hfov_deg) / 2)
    return focal_px * EYE_SPACING_M / eye_px


class FaceRanger:
    """Estimate the distance to the largest face in a frame.

    Args:
        hfov_deg: Horizontal field of view of the frames passed to :meth:`distance`.
    """

    def __init__(self, hfov_deg: float) -> None:
        import cv2

        self._hfov_deg = hfov_deg
        self._detector = cv2.FaceDetectorYN.create(str(_MODEL), "", (320, 320), 0.6)

    def distance(self, frame: Any) -> float | None:
        """Return the distance in meters, or ``None`` when no face is found."""
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
        eye_px = math.hypot(float(face[4]) - float(face[6]), float(face[5]) - float(face[7]))
        if eye_px < 2:
            return None
        return distance_from_eyes(eye_px, width, self._hfov_deg)


def find_volunteer(client: Any, model: str, frame: Any) -> tuple[bool, str | None]:
    """Ask a vision model whether someone in the frame raises a hand.

    Args:
        client: An ``openai.OpenAI``-compatible client.
        model: Chat model that accepts image input.
        frame: BGR image from the head camera.

    Returns:
        Whether a hand is raised, and a short description of that person's clothing.
    """
    import cv2

    height, width = frame.shape[:2]
    scale = 768 / max(height, width)
    if scale < 1:
        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return False, None
    url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VOLUNTEER_PROMPT},
                    {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
                ],
            }
        ],
    )
    try:
        data = json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        logger.warning("Vision model returned non-JSON content")
        return False, None
    clothing = str(data.get("clothing") or "").strip()[:20]
    return bool(data.get("hand_raised")), clothing or None
