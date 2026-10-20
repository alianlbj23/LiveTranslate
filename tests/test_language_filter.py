"""main.py language filter with a forced ASR language (asr_language != "auto").

Regression 2026-10-20 (review of the Fun-ASR language fix): the label of an ASR result
covers the whole interim buffer, and short sentences (<= 8 alnum chars) were buffered in
_interim_pending without any language check, then prepended to the next sentence and
committed under its label. Korean misrecognitions such as 「위여.」「빨리 써.」 are short,
so they still reached the overlay and the translator. Each sentence, buffered fragment,
preview and final remainder is now judged on its own text (asr_language.script_language)."""
import logging
import queue
import re

import numpy as np

from test_live_preview import _Panel, _app
from test_funasr_nano_language import _engine

HANGUL = re.compile("[가-힯]")


class _Recorder:
    """Transcript stand-in that keeps the committed original lines."""

    def __init__(self):
        self.lines = []

    def write_original(self, msg_id, timestamp, text):
        self.lines.append(text)

    def write_translation(self, *a, **k):
        pass

    def finalize_no_translation(self, *a, **k):
        pass

    def close(self):
        pass


def _split(text, lang):
    return [s for s in re.split(r"(?<=[。.!?])\s*", text) if s]


def _filter_app(results, forced="ja"):
    """App whose interim ASR returns `results` one after another."""
    app = _app()
    app._panel = _Panel(forced)
    app._transcript = _Recorder()
    app._interim_active = True
    app._split_sentences = _split
    it = iter(results)
    app._run_asr = lambda audio, kind, **kw: (next(it), 10.0)
    return app


def _committed(app):
    return app._transcript.lines


def test_interim_korean_fragments_are_not_prepended_to_the_next_japanese_sentence():
    app = _filter_app([
        {"text": "위여. 빨리 써. 그래", "language": "ja"},
        {"text": "今日は本当にいい天気ですね。明日も", "language": "ja"},
    ])
    app._do_interim_asr()
    app._do_interim_asr()
    assert _committed(app) == ["今日は本当にいい天気ですね。"]
    assert not any(HANGUL.search(str(a)) for _fn, args, _kw in app._tl_executor.submitted for a in args)


def test_interim_final_does_not_flush_korean_fragments():
    app = _filter_app([{"text": "위여. 빨리 써. 그래", "language": "ja"}])
    app._do_interim_asr()
    app._run_asr = lambda audio, kind, **kw: (None, 10.0)
    app._process_interim_final(np.zeros(16000, dtype=np.float32))
    assert _committed(app) == []
    assert app._msg_id == 0


def test_japanese_fragment_survives_a_korean_final_remainder(caplog):
    """A verified Japanese fragment must not be dropped together with a foreign remainder."""
    app = _filter_app([{"text": "はい。본론으로 들어가자", "language": "ja"}])
    app._do_interim_asr()  # 「はい。」 is short: buffered
    app._run_asr = lambda audio, kind, **kw: ({"text": "본론으로 들어가자 친구들", "language": "ja"}, 10.0)
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._process_interim_final(np.zeros(16000, dtype=np.float32))
    assert _committed(app) == ["はい。"]
    assert "Language filter: expected 'ja' but got 'ko'" in caplog.text


def test_mixed_buffer_keeps_japanese_and_drops_the_korean_sentence(caplog):
    app = _filter_app([{"text": "本当にありがとうございます。위여 가자 빨리 가자 친구들。明日も", "language": "ja"}])
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._do_interim_asr()
    assert _committed(app) == ["本当にありがとうございます。"]
    assert "Language filter: expected 'ja' but got 'ko'" in caplog.text


def test_korean_tail_of_a_japanese_buffer_is_not_previewed():
    app = _filter_app([{"text": "本当にありがとうございます。위여 가자", "language": "ja"}])
    app._do_interim_asr()
    assert _committed(app) == ["本当にありがとうございます。"]
    assert not any(HANGUL.search(call[0]) for call in app._subwin.live_calls)
    assert app._live_executor.submitted == []


def test_engine_result_in_korean_is_dropped_end_to_end(caplog):
    """Fake Fun-ASR-MLT-Nano engine forced to ja -> main.py _process_segment."""
    eng = _engine(text="위여.")
    eng.set_language("ja")
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._run_asr = lambda audio, kind, **kw: (eng.transcribe(audio), 10.0)
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._process_segment(np.zeros(8000, dtype=np.float32))  # < 2 s: no noise filter
    assert app._msg_id == 0
    assert app._tl_executor.submitted == []
    assert "Language filter: expected 'ja' but got 'ko', discarding: 위여." in caplog.text
    eng._model.text = "本当にそう思う"
    app._process_segment(np.zeros(8000, dtype=np.float32))
    assert app._msg_id == 1
    assert _committed(app) == ["本当にそう思う"]


def test_dropped_line_clears_its_live_preview():
    app = _app(live_text="本当に行", seq=5)
    app._panel = _Panel("ja")
    app._process_segment_text("本当に行 위여 가자 빨리 가자 친구들 오늘", "ja")
    assert app._msg_id == 0
    assert app._subwin.cleared == 1
    assert app._live_last_text == ""


def test_dropped_line_keeps_an_unrelated_preview():
    app = _app(live_text="明日も晴れるといいな", seq=5)
    app._panel = _Panel("ja")
    app._process_segment_text("위여 가자 빨리 가자 친구들", "ja")
    assert app._subwin.cleared == 0
    assert app._live_last_text == "明日も晴れるといいな"


def test_a_streak_of_dropped_lines_notifies_once(caplog):
    app = _app()
    app._panel = _Panel("ja")
    notices = []
    app._lang_notice_callback = notices.append
    with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
        for _ in range(6):
            app._process_segment_text("위여 가자 빨리 가자 친구들", "ja")
    assert len(notices) == 1
    assert "ko" in notices[0] and "ja" in notices[0]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    app._process_segment_text("本当にそう思う", "ja")  # an accepted line ends the streak
    for _ in range(5):
        app._process_segment_text("위여 가자 빨리 가자 친구들", "ja")
    assert len(notices) == 2


def test_language_the_model_cannot_be_told_is_reported_in_the_main_log(caplog):
    app = _app()
    app._panel = _Panel("ko")
    notices = []
    app._lang_notice_callback = notices.append
    with caplog.at_level(logging.WARNING, logger="LiveTranslate"):
        app._note_asr_language_support("funasr", "funasr-nano-2512")
        app._note_asr_language_support("funasr", "funasr-nano-2512")  # same state: once
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "ko" in warnings[0].getMessage()
    assert len(notices) == 1 and "Fun-ASR-Nano" in notices[0]
    # the model got no hint, so its output cannot be judged: keep it visible
    assert app._heard_other_language("日本語です", "ko") is None
    app._note_asr_language_support("funasr", "funasr-mlt-nano-2512")
    assert app._asr_hint_missing is False
    assert app._heard_other_language("日本語です", "ko") == "ja"


def test_other_engines_never_report_a_missing_hint():
    app = _app()
    app._panel = _Panel("ko")
    app._note_asr_language_support("whisper", None)
    assert app._asr_hint_missing is False


def test_asr_loop_survives_a_processing_error(caplog):
    """An unexpected exception in segment handling must not end the ASR thread."""
    import threading

    app = _app()
    app._running = True
    app._asr_queue = queue.Queue()
    app._vad_lock = threading.Lock()
    app._last_interim_samples = 0
    app._last_interim_check_time = 0.0
    app._interim_active = True
    app._interim_pending = "はい"

    def boom(*a, **k):
        raise RuntimeError("boom")

    app._do_interim_asr = boom
    app._process_interim_final = boom
    with caplog.at_level(logging.ERROR, logger="LiveTranslate"):
        # one run per item: the loop puts a non-interim item back at the end of the
        # queue while draining duplicates, so it would land behind the None sentinel
        for item in (("interim", None), ("vad_flush", np.zeros(1600, dtype=np.float32))):
            app._asr_queue.put(item)
            app._asr_queue.put(None)
            app._asr_loop()  # returns at the None sentinel instead of dying on "boom"
    errors = [r for r in caplog.records if r.exc_info and "boom" in str(r.exc_info[1])]
    assert len(errors) == 2
    assert app._interim_active is False
    assert app._interim_pending == ""


def test_whole_segment_keeps_japanese_and_drops_the_korean_sentence(caplog):
    """Non-incremental path (_process_segment): a mixed segment is judged per sentence too.
    Kana + kanji (13) outnumber Hangul (11), so judging the whole text let Korean through."""
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._split_sentences = _split
    app._run_asr = lambda audio, kind, **kw: (
        {"text": "本当にありがとうございます。위여 가자 빨리 가자 친구들。", "language": "ja"}, 10.0)
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._process_segment(np.zeros(8000, dtype=np.float32))
    assert _committed(app) == ["本当にありがとうございます。"]
    assert "Language filter: expected 'ja' but got 'ko'" in caplog.text


def test_segment_text_with_several_sentences_is_judged_per_sentence():
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._split_sentences = _split
    app._process_segment_text("はい。본론으로 들어가자 친구들。", "ja")
    assert _committed(app) == ["はい。"]


def test_single_sentence_text_is_kept_unchanged():
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._split_sentences = _split
    app._process_segment_text("今日は いい天気。", "ja")
    assert _committed(app) == ["今日は いい天気。"]


def test_whole_text_with_another_engine_label_is_dropped_once(caplog):
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    app._split_sentences = _split
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._process_segment_text("你好。我们走吧。", "zh")
    assert _committed(app) == []
    assert caplog.text.count("Language filter") == 1


def test_missing_engine_label_is_judged_by_text():
    """SenseVoice answers "auto" without a language tag and the remote engine "unknown":
    no label, not another language."""
    app = _app()
    app._panel = _Panel("ja")
    for label in ("auto", "unknown", ""):
        assert app._heard_other_language("本当に", label) is None
        assert app._heard_other_language("위여 가자", label) == "ko"


def test_accepted_line_without_engine_label_uses_the_forced_language(caplog):
    app = _app()
    app._panel = _Panel("ja")
    app._transcript = _Recorder()
    with caplog.at_level(logging.INFO, logger="LiveTranslate"):
        app._process_segment_text("本当に", "auto")
    assert _committed(app) == ["本当に"]
    assert "ASR [ja]" in caplog.text
