"""Spoken lines and the parsing of spoken answers. No hardware access."""

import re

INTRO_HAND = "それでは検査をはじめます。検査をする人は手をあげてください。"
STEP_BACK = "ありがとうございます。もう少し離れますね。"
STEP_CLOSER = "ありがとうございます。もう少し近づきますね。"
START = "それでは、視力検査を開始します。"
ADJUSTING = "ランドルト環のサイズを調整中。"
ADJUSTED = "調整完了。それでは、いきます。"
ASK = "これは？"
THANKS = "おつかれさまでした。"

_UNKNOWN = ("わから", "わかり", "分かり", "分から", "わかん", "見えな", "みえな", "パス", "ぱす", "無理", "むり")
_WORDS = {
    "up": ("上", "うえ", "ウエ", "アップ", "あっぷ"),
    "down": ("下", "した", "シタ", "ダウン", "だうん"),
    "right": ("右", "みぎ", "ミギ", "ライト"),
    "left": ("左", "ひだり", "ヒダリ", "レフト"),
}


def parse_direction(text: str) -> str | None:
    """Extract one answer from a transcript.

    Returns:
        ``up``, ``down``, ``left`` or ``right``; ``unknown`` when the speaker gives up;
        ``None`` when the transcript names no direction or names several.
    """
    cleaned = re.sub(r"\s+", "", text)
    if any(word in cleaned for word in _UNKNOWN):
        return "unknown"
    found = [name for name, words in _WORDS.items() if any(word in cleaned for word in words)]
    return found[0] if len(found) == 1 else None


def meters(distance_m: float) -> str:
    """Render a distance the way the robot says it."""
    return f"{distance_m:.1f}"


def you_are(clothing: str | None) -> str:
    """Build the line that singles out the person who raised a hand."""
    who = f"{clothing}をきた、あなたですね。" if clothing else "あなたですね。"
    return who + "それでは、私と十分な距離をとってください。"


def distance_line(distance_m: float) -> str:
    """Build the line announcing the measured distance."""
    return f"対象との距離、{meters(distance_m)}メートル。"


def result_line(passed: float | None, lowest: float, at_ceiling: bool) -> str:
    """Build the line announcing the result."""
    if passed is None:
        return f"あなたの視力は、{lowest:g}未満です。"
    if at_ceiling:
        return f"あなたの視力は、{passed:g}以上です。"
    return f"あなたの視力は、{passed:g}です。"


def comment_line(passed: float | None) -> str:
    """Build the closing remark about the result."""
    if passed is None:
        return "今回の距離では、いちばん大きな輪も見えにくかったようです。もう少し近くで、また試してみてください。"
    if passed >= 1.5:
        return "ものすごく目がいいですね。遠くの看板まで読めそうです。"
    if passed >= 1.0:
        return "けっこう目がいいですね。その調子で、目を大事にしてください。"
    if passed >= 0.7:
        return "運転免許の基準は両目で0.7なので、合格ラインです。"
    return "運転免許の基準は両目で0.7です。眼鏡やコンタクトの度数を、一度確かめてみてもいいかもしれません。"
