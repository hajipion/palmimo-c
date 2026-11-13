"""The exam itself: find the volunteer, settle the distance, show rings, announce the result."""

import argparse
import json
import os
import queue
import statistics
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import speech
from .landolt import FRAME_PX, PANEL_DIAMETER_MM, QuickSession, eligible_levels, gap_px, render_gray8, tint, to_rgb565
from .rig import Listener, Rig

_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>視力チェック</title>
<style>body{font-family:sans-serif;text-align:center;margin:16px}button{font-size:28px;margin:6px;padding:16px 24px}
#s{margin:16px;font-size:18px}</style>
<h1>視力チェック</h1><div id="s">…</div>
<div><img id="r" width="338" height="338" style="image-rendering:pixelated;border-radius:50%;border:40px solid #222"></div>
<div><button onclick="a('up')">↑ 上</button></div>
<div><button onclick="a('left')">← 左</button><button onclick="a('right')">右 →</button></div>
<div><button onclick="a('down')">↓ 下</button></div>
<div><button onclick="a('unknown')">わからない</button></div>
<div><button onclick="a('next')">次へ進める</button></div>
<script>
function a(d){fetch('/answer/'+d,{method:'POST'})}
document.onkeydown=e=>{const m={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right'};if(m[e.key])a(m[e.key])}
setInterval(async()=>{const r=await fetch('/state');const j=await r.json();document.getElementById('s').textContent=j.line;const i=document.getElementById('r');i.style.visibility=j.ring?'visible':'hidden';if(j.ring)i.src='/ring.bmp?'+j.ring},400)
</script>"""


class _State:
    """What the web page shows and what it sends back."""

    def __init__(self) -> None:
        self.line = ""
        self.frame: bytes | None = None
        self.ring = 0
        self.answers: queue.Queue[str] = queue.Queue()
        self.skip = threading.Event()


def _bmp(gray: bytes, size: int) -> bytes:
    """Encode a frame as a 24-bit BMP, tinted like the face, so a browser can preview it."""
    colors = tint(gray)
    pad = b"\0" * (-(size * 3) % 4)
    rows = b"".join(
        b"".join(bytes((b, g, r)) for r, g, b in colors[y * size : (y + 1) * size]) + pad
        for y in range(size - 1, -1, -1)
    )
    header = b"BM" + struct.pack("<IHHI", 54 + len(rows), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, size, size, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    return header + info + rows


def _serve(state: _State, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/state":
                body = {"line": state.line, "ring": state.ring if state.frame else 0}
                self._send(json.dumps(body).encode(), "application/json")
            elif self.path.startswith("/ring.bmp") and state.frame is not None:
                self._send(_bmp(state.frame, FRAME_PX), "image/bmp")
            else:
                self._send(_PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            word = self.path.rsplit("/", 1)[-1]
            if word == "next":
                state.skip.set()
            else:
                state.answers.put(word)
            self._send(b"{}", "application/json")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="page", daemon=True).start()


def _stdin_answers(state: _State) -> None:
    for line in sys.stdin:
        word = line.strip()
        if word == "next":
            state.skip.set()
        elif word:
            state.answers.put(word)


def _measure(rig: Rig, ranger: Any, seconds: float) -> float | None:
    """Return the median distance read over a short window, or ``None``."""
    readings = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = rig.grab()
        distance = ranger.distance(frame) if frame is not None else None
        if distance is not None:
            readings.append(distance)
        time.sleep(0.1)
    return statistics.median(readings) if readings else None


def _find_volunteer(rig: Rig, state: _State, client: Any, model: str, timeout: float) -> str | None:
    """Wait for a raised hand; return that person's clothing when the model names it."""
    from .vision import find_volunteer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not state.skip.is_set():
        frame = rig.grab()
        if frame is None or client is None:
            state.skip.wait(1.0)
            continue
        try:
            raised, clothing = find_volunteer(client, model, frame)
        except Exception as exc:
            print(f"[vision] {exc!r}", flush=True)
            state.skip.wait(2.0)
            continue
        print(f"[vision] hand_raised={raised} clothing={clothing}", flush=True)
        if raised:
            return clothing
    return None


def _settle_distance(rig: Rig, state: _State, ranger: Any, target: float, walk_seconds: float) -> float | None:
    """Coach the volunteer to the target distance, stepping forward a little if they are far."""
    steps = 0
    last = None
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not state.skip.is_set():
        distance = _measure(rig, ranger, 1.0)
        if distance is None:
            print("[range] no face", flush=True)
            continue
        last = distance
        print(f"[range] {distance:.2f} m", flush=True)
        state.line = f"距離 {distance:.1f} m"
        if abs(distance - target) <= 0.2:
            return distance
        if distance < target:
            rig.say("もう少し、さがってください。")
        elif walk_seconds > 0 and steps < 3 and distance - target < 0.6:
            rig.say("もう少し近づきますね。")
            rig.move("forward", walk_seconds)
            steps += 1
        else:
            rig.say("もう少し、近づいてください。")
        time.sleep(1.5)
    return last


def run(args: argparse.Namespace) -> None:
    """Run one exam from the greeting to the closing remark."""
    state = _State()
    _serve(state, args.port)
    if args.stdin:
        threading.Thread(target=_stdin_answers, args=(state,), daemon=True).start()

    client = None
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
    else:
        print("[exam] OPENAI_API_KEY is not set: no voice answers, no hand-raise check", flush=True)

    rig = Rig(
        dry_run=args.dry_run, motion=not args.no_motion, listen=not args.no_voice, speaker_device=args.speaker_device
    )
    try:
        listener = Listener(rig.mic, client, args.stt_model, state.answers)
        ranger = None
        if rig.camera is not None:
            from .vision import FaceRanger

            ranger = FaceRanger(args.camera_hfov_deg)
        print(f"[exam] voice={listener.voice} camera={rig.camera is not None} page=:{args.port}", flush=True)

        def say(text: str) -> None:
            state.line = text
            rig.say(text)

        preview = eligible_levels(args.distance_m, args.panel_mm)
        if preview:
            state.frame, state.ring = render_gray8(gap_px(preview[0], args.distance_m, args.panel_mm), "right"), 1
        rig.expression("smile")
        say(speech.INTRO_HAND)
        clothing = _find_volunteer(rig, state, client, args.vision_model, 3 if args.dry_run else 40)
        state.skip.clear()
        say(speech.you_are(clothing))

        distance = args.distance_m
        if ranger is not None:
            measured = _settle_distance(rig, state, ranger, args.distance_m, args.walk_seconds)
            state.skip.clear()
            if measured is not None:
                distance = measured
        say("ありがとうございます。")

        say(speech.START)
        say(speech.distance_line(distance))
        say(speech.ADJUSTING)
        levels = eligible_levels(args.distance_m, args.panel_mm)
        session = QuickSession(levels)
        print(f"[exam] sized for {args.distance_m:g} m, levels {levels}", flush=True)
        time.sleep(1.0)
        say(speech.ADJUSTED)

        while not session.done:
            gap = gap_px(session.acuity, args.distance_m, args.panel_mm)
            frame = render_gray8(gap, session.direction)
            state.frame, state.ring = frame, state.ring + 1
            shown = rig.show_frame(to_rgb565(frame))
            print(f"[trial] {session.acuity:g} {session.direction} gap={gap:.2f}px shown={shown}", flush=True)
            say(speech.ASK)
            answer = listener.wait_answer(120 if args.dry_run else args.answer_seconds)
            session.answer(answer)
            print(f"[trial] answer={answer} correct={session.history[-1]['correct']}", flush=True)

        state.frame = None
        rig.idle_face()
        rig.expression("happy")
        say(speech.THANKS)
        say(speech.result_line(session.passed, levels[0], session.at_ceiling))
        say(speech.comment_line(session.passed))
        rig.move("bow", 2.5)
        print(f"[exam] done: {json.dumps(session.history, ensure_ascii=False)}", flush=True)
    finally:
        rig.close()


def main() -> None:
    """Parse the launch settings and run one exam."""
    parser = argparse.ArgumentParser(description="Palmimo が検査員になる視力チェック（参考値。医療用ではありません）")
    parser.add_argument("--port", type=int, default=8010, help="予備の回答ページのポート")
    parser.add_argument("--distance-m", type=float, default=3.0, help="検査する距離 m（環の大きさはこの距離で決める）")
    parser.add_argument("--panel-mm", type=float, default=PANEL_DIAMETER_MM, help="顔画面の直径 mm")
    parser.add_argument("--camera-hfov-deg", type=float, default=53.5, help="カメラの水平画角（度）")
    parser.add_argument("--walk-seconds", type=float, default=0.8, help="近づくときに 1 回で歩く秒数。0 で歩かない")
    parser.add_argument("--answer-seconds", type=float, default=6.0, help="回答を待つ秒数")
    parser.add_argument("--stt-model", default="gpt-4o-mini-transcribe", help="文字起こしのモデル")
    parser.add_argument("--vision-model", default="gpt-5-mini", help="手を挙げた人を見つけるモデル")
    parser.add_argument("--speaker-device", default="ReSpeaker", help="スピーカーの機器名の一部")
    parser.add_argument("--no-motion", action="store_true", help="脚と首を動かさない")
    parser.add_argument("--no-voice", action="store_true", help="マイクを使わず、ページのボタンで答える")
    parser.add_argument("--dry-run", action="store_true", help="機器を一切開かずに流れだけ試す")
    parser.add_argument("--stdin", action="store_true", help="標準入力からも回答を受ける")
    parser.add_argument("--output", type=Path, default=None, help=argparse.SUPPRESS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
