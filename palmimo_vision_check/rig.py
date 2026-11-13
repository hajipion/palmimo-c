"""The robot as the exam uses it: voice, face frames, camera, microphone, small motions.

Every part is optional. A missing part degrades to a log line so the flow can be
rehearsed without hardware.
"""

import base64
import logging
import os
import queue
import shutil
import subprocess
import sys
import time
import wave
from io import BytesIO
from typing import Any

from .landolt import FRAME_PX
from .speech import parse_direction

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1536
_STT_HINT = "上、下、右、左、わからない"


def _probe(label: str, build: Any) -> Any:
    """Build one hardware part, returning ``None`` with a log line when it is unavailable."""
    try:
        return build()
    except Exception as exc:
        print(f"[rig] {label} unavailable: {exc!r}", flush=True)
        return None


def _log_playback_devices() -> None:
    """Print the ALSA playback cards so a silent speaker can be diagnosed from the log."""
    if shutil.which("aplay") is None:
        return
    listing = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=False)
    for line in listing.stdout.splitlines():
        if line.startswith("card "):
            print(f"[rig] playback {line}", flush=True)


class Rig:
    """Owns the Palmimo facade and the peripherals the exam reads directly.

    Args:
        dry_run: Open no hardware at all.
        motion: Attach the servo bus.
        listen: Open the microphone.
        speaker_device: Output device name hint for the speaker.
    """

    def __init__(self, *, dry_run: bool, motion: bool, listen: bool, speaker_device: str) -> None:
        self.robot: Any = None
        self.camera: Any = None
        self.mic: Any = None
        self._driver: Any = None
        self._display: Any = None
        if dry_run:
            return
        from palmimo_sdk import FaceDisplay, HeadCamera, MicStream, Palmimo, Speaker, SpeakerConfig

        def speaker_part() -> Any:
            engines: list[tuple[str, Any]] = [("local", None)]
            if os.environ.get("OPENAI_API_KEY"):
                from palmimo_sdk import OpenAiEngine

                engines.append(("openai", OpenAiEngine()))
            hints = [speaker_device, None] if speaker_device else [None]
            last: Exception | None = None
            for name, engine in engines:
                for hint in hints:
                    try:
                        part = Speaker(SpeakerConfig(lang="ja", device_name_hint=hint), engine=engine)
                        part.open()
                    except Exception as exc:
                        print(f"[rig] voice={name} device={hint!r} failed: {exc!r}", flush=True)
                        last = exc
                        continue
                    print(f"[rig] speaking with voice={name} device={hint or 'ALSA default'}", flush=True)
                    return part
            raise RuntimeError(f"no usable speaker: {last!r}")

        def driver_part() -> Any:
            from palmimo_sdk import DynamixelDriver

            part = DynamixelDriver()
            part.connect()
            return part

        def display_part() -> Any:
            probe = FaceDisplay()
            probe.connect()
            probe.disconnect()
            return FaceDisplay()

        def camera_part() -> Any:
            part = HeadCamera()
            part.open()
            return part

        def mic_part() -> Any:
            try:
                part = MicStream()
                part.open()
            except Exception as exc:
                print(f"[rig] echo-cancelled mic unavailable ({exc!r}); using raw capture", flush=True)
                part = MicStream(processors=[])
                part.open()
            return part

        _log_playback_devices()
        speaker = _probe("speaker", speaker_part)
        self._driver = _probe("servo bus", driver_part) if motion else None
        self._display = _probe("face display", display_part)
        self.camera = _probe("camera", camera_part)
        self.mic = _probe("microphone", mic_part) if listen else None
        self.robot = Palmimo(driver=self._driver, display=self._display, speaker=speaker)
        self.robot.connect()

    def say(self, text: str) -> None:
        """Speak one line and wait until it has been spoken."""
        print(f"[palmimo] {text}", flush=True)
        if self.robot is None:
            if sys.platform == "darwin" and shutil.which("say"):
                subprocess.run(["say", text], check=False)
            return
        handle = self.robot.say(text)
        if handle is not None:
            handle.join(timeout=30)
            if handle.error is not None:
                print(f"[rig] speech failed: {handle.error!r}", flush=True)

    def expression(self, name: str) -> None:
        """Show one of the built-in face expressions."""
        if self.robot is not None and self._display is not None:
            try:
                self.robot.set_expression(name)
            except Exception as exc:
                print(f"[rig] expression failed: {exc!r}", flush=True)

    def _frame_command(self, line: str, expected: str) -> None:
        reply = self._display._command(line)
        for _ in range(8):
            if reply == expected:
                return
            if reply.startswith("ERR"):
                break
            raw = self._display._ser.readline()
            if not raw:
                break
            reply = raw.decode("ascii", "ignore").strip()
        raise RuntimeError(f"face display answered {reply!r} to {line[:24]!r}")

    def show_frame(self, pixels: bytes, size: int = FRAME_PX, fmt: str = "RGB565") -> bool:
        """Draw a GRAY8 or RGB565 frame on the face with the firmware's FRAME commands.

        Returns:
            Whether the display accepted the frame.
        """
        if self._display is None:
            return False
        try:
            self._frame_command(f"FRAME BEGIN {size} {size} {fmt}", "OK FRAME BEGIN")
            for offset in range(0, len(pixels), _CHUNK_BYTES):
                chunk = pixels[offset : offset + _CHUNK_BYTES]
                data = base64.b64encode(chunk).decode("ascii")
                self._frame_command(f"FRAME DATA {data}", f"ACK {offset + len(chunk)}")
            self._frame_command("FRAME END", "OK FRAME")
            return True
        except Exception as exc:
            print(f"[rig] frame failed: {exc!r}", flush=True)
            try:
                self._display._command("FRAME ABORT")
            except Exception:
                logger.exception("Frame abort failed")
            return False

    def idle_face(self) -> None:
        """Return the face to its neutral expression."""
        if self._display is not None:
            try:
                self._display.idle()
            except Exception as exc:
                print(f"[rig] idle failed: {exc!r}", flush=True)

    def grab(self) -> Any:
        """Return one camera frame, or ``None``."""
        if self.camera is None:
            return None
        try:
            ok, frame = self.camera.read()
        except Exception as exc:
            print(f"[rig] camera read failed: {exc!r}", flush=True)
            return None
        return frame if ok else None

    def move(self, motion: str, seconds: float) -> None:
        """Run one named motion for a bounded time, then glide back to neutral."""
        if self._driver is None or seconds <= 0:
            return
        try:
            self.robot.set_motion(motion)
            self.robot.run(seconds=min(seconds, 4.0))
        except Exception as exc:
            print(f"[rig] motion {motion} failed: {exc!r}", flush=True)
        finally:
            try:
                self.robot.stop()
                self.robot.run(seconds=1.0)
            except Exception as exc:
                print(f"[rig] stop failed: {exc!r}", flush=True)

    def close(self) -> None:
        """Restore the face, park the robot and release every device."""
        self.idle_face()
        for part in (self.mic, self.camera):
            if part is not None:
                try:
                    part.close()
                except Exception as exc:
                    print(f"[rig] close failed: {exc!r}", flush=True)
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as exc:
                print(f"[rig] disconnect failed: {exc!r}", flush=True)


def _wav(samples: Any, sample_rate: int = 16000) -> BytesIO:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.tobytes())
    buffer.seek(0)
    buffer.name = "answer.wav"
    return buffer


class Listener:
    """Wait for one spoken or manually entered answer.

    Args:
        mic: An open ``MicStream``, or ``None`` for manual answers only.
        client: An ``openai.OpenAI``-compatible client, or ``None``.
        stt_model: Transcription model name.
        manual: Queue of answers entered from the web page or the terminal.
    """

    def __init__(self, mic: Any, client: Any, stt_model: str, manual: "queue.Queue[str]") -> None:
        self._client = client
        self._stt_model = stt_model
        self._manual = manual
        self._sub: Any = None
        self._segmenter: Any = None
        if mic is not None and client is not None:
            try:
                from .vad import SileroVad, UtteranceSegmenter

                self._segmenter = UtteranceSegmenter(
                    vad=SileroVad.load(), silence_seconds=0.45, max_utterance_seconds=4.0
                )
                self._sub = mic.subscribe()
            except Exception as exc:
                print(f"[listen] voice answers unavailable: {exc!r}", flush=True)
                self._segmenter = None

    @property
    def voice(self) -> bool:
        """Return whether spoken answers are being taken."""
        return self._segmenter is not None and self._sub is not None

    def _transcribe(self, utterance: Any) -> str:
        result = self._client.audio.transcriptions.create(
            model=self._stt_model, file=_wav(utterance), language="ja", prompt=_STT_HINT
        )
        return str(result.text).strip()

    def wait_answer(self, timeout: float) -> str | None:
        """Return ``up``/``down``/``left``/``right``/``unknown``, or ``None`` on silence."""
        while not self._manual.empty():
            self._manual.get_nowait()
        if self.voice:
            while self._sub.get(timeout=0) is not None:
                pass
            self._segmenter.reset()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                answer = self._manual.get(timeout=0 if self.voice else 0.05)
            except queue.Empty:
                answer = None
            if answer in ("up", "down", "left", "right", "unknown"):
                print(f"[manual] {answer}", flush=True)
                return answer
            if not self.voice:
                continue
            chunk = self._sub.get(timeout=0.05)
            if chunk is None:
                continue
            utterance = self._segmenter.feed(chunk.reshape(-1))
            if utterance is None:
                continue
            try:
                text = self._transcribe(utterance)
            except Exception as exc:
                print(f"[listen] transcription failed: {exc!r}", flush=True)
                continue
            direction = parse_direction(text)
            print(f"[heard] {text!r} -> {direction}", flush=True)
            if direction is not None:
                return direction
            while self._sub.get(timeout=0) is not None:
                pass
            self._segmenter.reset()
        return None
