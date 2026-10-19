"""The Whisper hallucination filter drops a segment that is nothing but a listed phrase.
Measured on large-v3-turbo: on noise / BGM it hallucinates 「ご視聴ありがとうございました」
with the same confidence as real speech (no_speech_prob 0.00, avg_logprob about -0.1),
so confidence cannot separate them. The list must therefore hold only boilerplate that
a streamer practically never says as a whole segment, not everyday phrases like
「ありがとうございました」 (thanking for a superchat) or 「おやすみなさい」."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr_engine import ASREngine  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    phrases = cfg["asr"]["whisper"]["hallucination_phrases"]
    e = ASREngine.__new__(ASREngine)
    e._hallucination_phrases = {ASREngine._normalize_text(p) for p in phrases if p}
    return e


@pytest.mark.parametrize("said", [
    "ありがとうございました",
    "ありがとうございました!",
    "ありがとうございました ありがとうございました",
    "おやすみなさい",
    "おやすみなさい〜",
])
def test_everyday_phrases_are_kept(engine, said):
    assert engine._is_hallucination(said) is False


@pytest.mark.parametrize("hallucinated", [
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました ご視聴ありがとうございました",
    "チャンネル登録お願いします",
    "字幕由Amara.org社区提供",
])
def test_known_boilerplate_is_still_dropped(engine, hallucinated):
    assert engine._is_hallucination(hallucinated) is True


def test_sentence_containing_a_phrase_is_kept(engine):
    """Error-path guard: only whole-segment matches are dropped."""
    assert engine._is_hallucination("今日もご視聴ありがとうございました、また明日ね") is False
